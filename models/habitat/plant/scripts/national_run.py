"""Per-species MaxEnt maps, GPU stage: fit and render every species that the prepare-only stage has finished.

The CPU stage (`run_fit.py --prepare-only`) writes, per species, everything an occurrence-trained model needs
(cleaned, thinned presences; effort-weighted background in the calibration ecoregions; VIF-selected predictors) and
a summary with ``stage: prepared``. This worker claims prepared species (an atomic ``.claim`` directory, so any
number of workers on any GPUs can run side by side), fits them in GPU batches with the validated PyTorch
reimplementation of maxent.jar (beta by 5-fold CV over {2, 5, 10, 15, 20}, 5 replicates; docs/maxent_torch.md),
completes the summary exactly as a full run does (beta, cv_auc), then renders 240 m maps on every US grid the
calibration area reaches, the range card and benchmarks (`ranges.render`). Resumable at every step: a
species with a range card is done; a fitted but unrendered one is rendered; a stale claim is released at start-up.
It stops when nothing is left and the CPU stage has finished (``--until-done``).

usage: CUDA_VISIBLE_DEVICES=0 python3 scripts/national_run.py --dir <prepared>/occurrences [--batch 16] [--until-done]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config, maxent_torch  # noqa: E402
from ranges.pipeline import _write_summary  # noqa: E402
from ranges.render import RenderContext, render_species  # noqa: E402

R = config.data_root()


def state(wd: Path) -> str:
    label = wd.name
    if (wd / f"{label}.rangecard").exists():
        return "done"
    try:
        s = json.loads((wd / "summary.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return "absent"
    return s.get("stage", "fitted" if "beta" in s else "absent")


def claim(wd: Path) -> bool:
    try:
        (wd / ".claim").mkdir()
        return True
    except FileExistsError:
        return False


def release(wd: Path) -> None:
    try:
        (wd / ".claim").rmdir()
    except FileNotFoundError:
        pass


def prepare_running() -> bool:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return any(l.startswith("python3 scripts/run_fit.py") and "--prepare-only" in l for l in out.splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--until-done", action="store_true")
    ap.add_argument("--release-stale", action="store_true", help="remove claims left by killed workers (run once)")
    a = ap.parse_args()
    root = Path(a.dir)
    if a.release_stale:
        for c in root.glob("*/.claim"):
            c.rmdir()
    ctx = RenderContext(R)
    log = root.parent / f"national_run_{os.environ.get('CUDA_VISIBLE_DEVICES', 'x')}.log"
    while True:
        dirs = sorted(p for p in root.iterdir() if p.is_dir())
        fit_todo, render_todo = [], []
        for wd in dirs:
            st = state(wd)
            if st == "prepared":
                fit_todo.append(wd)
            elif st == "fitted":
                render_todo.append(wd)
        batch = []
        for wd in fit_todo:
            if len(batch) == a.batch:
                break
            if claim(wd):
                batch.append(wd)
        if batch:
            t = time.time()
            try:
                res = maxent_torch.fit_species_dirs([(wd / "maxent", wd.name) for wd in batch], device="cuda")
                for wd, (beta, cv) in zip(batch, res):
                    s = json.loads((wd / "summary.json").read_text())
                    s.update(beta=beta, cv_auc=cv, stage="fitted", engine="torch",
                             seconds_fit_batch=round(time.time() - t, 1), batch_size=len(batch))
                    _write_summary(wd, s)
                render_todo = batch + render_todo
            except Exception:
                with open(log, "a") as f:
                    f.write(f"FIT_FAIL batch {[w.name for w in batch]}\n{traceback.format_exc()}\n")
                for wd in batch:
                    release(wd)
                batch = []
            torch.cuda.empty_cache()
            print(f"fit {len(batch)} species in {time.time() - t:.1f} s", flush=True)
        rendered = 0
        for wd in render_todo:
            if wd not in batch and not claim(wd):
                continue
            try:
                if state(wd) == "fitted":
                    rec = render_species(wd, wd.name, ctx)
                    rendered += 1
                    with open(log, "a") as f:
                        f.write(f"DONE {wd.name} render {rec['render_seconds']}s\n")
            except Exception:
                with open(log, "a") as f:
                    f.write(f"RENDER_FAIL {wd.name}\n{traceback.format_exc()}\n")
            finally:
                release(wd)
                torch.cuda.empty_cache()             # wide-ranging species leave large cached blocks behind
            if rendered >= a.batch:                 # interleave: back to fitting after one batch of renders
                break
        if not batch and rendered == 0:
            if a.until_done and not prepare_running() and not any(state(w) in ("prepared", "fitted") for w in dirs):
                print("national run: nothing left", flush=True)
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
