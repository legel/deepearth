"""CPU stage at scale: every (species, presence mode) through the Daru (2024) pipeline up to fitted MaxEnt
replicates. ``--engine auto`` (default) fits with maxent_torch on a GPU when one is present and with maxent.jar
otherwise (no GPU or torch needed; rendering happens elsewhere from the outputs: lambdas, thresholds, summary).
``--prepare-only`` stops each species before the range hull and MaxEnt (mode ``occurrences``): cleaned records,
calibration ecoregions, presences and bias-weighted background with predictors (the joint model's inputs).
Resumable (a species with summary.json is skipped) and failure-logging, so it can run on interruptible machines;
``--shard i/n`` splits the species between machines.

Every input defaults to the configuration (data root, GBIF downloads, species table, dispersal table, bias grid,
output directory, name reassignment).

usage: run_fit.py [--prepare-only --group 8] [--modes daru,occurrences] [--workers 24] [--shard 0/1]
                  [--config configs/conus.json] [--root R --parquet D1,D2 --species-table T --sbm S --bias B --out O]
"""
import argparse
import ctypes
import gc
import json
import os
import sys
import threading
import time
import traceback
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config, modelling, names, pipeline  # noqa: E402

_RES = None
# named variants: presence_mode, background_mode
MODES = {"daru": ("hull", "uniform"),           # Daru (2024) as published (hull presences, uniform background)
         "hull": ("hull", "bias"),              # Daru's text (bias-weighted background)
         "occurrences": ("occurrences", "bias")}  # improved: thinned occurrences, bias-weighted background
COLS = {"gbifid": "gbifID", "species": "species", "decimallatitude": "decimalLatitude",
        "decimallongitude": "decimalLongitude", "coordinateuncertaintyinmeters": "coordinateUncertaintyInMeters",
        "countrycode": "countryCode", "basisofrecord": "basisOfRecord", "establishmentmeans": "establishmentMeans",
        "occurrencestatus": "occurrenceStatus", "year": "year", "month": "month", "day": "day"}


def finished(summary: Path, prepare_only: bool = False) -> bool:
    """A species is done once its summary parses (older runs could leave an empty file if interrupted) and, for a
    fitting run, records a fitted model (a prepare-only summary has no ``beta``)."""
    try:
        s = json.loads(summary.read_text())
    except (OSError, ValueError):
        return False
    return prepare_only or "beta" in s


def _init(root, sbm, bias):
    global _RES
    os.environ["OMP_NUM_THREADS"] = "1"
    # a worker whose main process has gone (stopped by its PID, or killed) would otherwise wait forever on the task
    # queue holding its memory: it exits as soon as it is re-parented
    parent = os.getppid()

    def _watch():
        while os.getppid() == parent:
            time.sleep(5)
        os._exit(1)
    threading.Thread(target=_watch, daemon=True).start()
    _RES = pipeline.Resources.load(Path(root), Path(sbm), Path(bias))


def _task(items, out, prepare_only=False):
    """(species, mode, records) items through the pipeline; the species' records are cleaned together in one R
    session first (pipeline.clean_species; same flags as one session per species)."""
    _release()
    uniq = {}
    for name, _, gbif in items:
        if not gbif.empty:
            uniq.setdefault(name, gbif)
    try:
        cleaned = dict(zip(uniq, pipeline.clean_species(list(uniq.items()), _RES)))
    except Exception:                                   # a failing record set must not fail the others
        cleaned = {}
    results = []
    for name, mode, gbif in items:
        wd = Path(out) / mode / name.replace(" ", "_")
        try:
            t = time.time()
            pm, bm = MODES[mode]
            s = pipeline.run_species(name, gbif, _RES, wd, presence_mode=pm, background_mode=bm,
                                     prepare_only=prepare_only, cleaned=cleaned.get(name))
            results.append({"ok": True, "species": name, "mode": mode, "seconds": round(time.time() - t, 1),
                            "beta": s.get("beta"), "n_presence": s["n_presence"]})
        except Exception as e:
            results.append({"ok": False, "species": name, "mode": mode, "error": repr(e),
                            "trace": traceback.format_exc()[-3000:]})
    del cleaned, uniq
    _release()
    return results


def _release():
    """Give freed memory back to the system: a long-lived worker otherwise keeps the peak of the largest record set
    it has seen (measured: 2-3 GB per worker growing to 8.5 GB after circumboreal species). Called at the start of a
    task too, when the previous task's arguments have been dropped."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--parquet", help="GBIF parquet directory, or several separated by commas")
    ap.add_argument("--species-table")
    ap.add_argument("--sbm")
    ap.add_argument("--bias")
    ap.add_argument("--out")
    ap.add_argument("--modes", default="daru,occurrences")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=300)
    ap.add_argument("--group", type=int, default=1,
                    help="species per worker task; their records are cleaned in one R session (saves R start-up)")
    ap.add_argument("--shard", default="0/1", help="i/n: fit only species whose crc32(name) % n == i (one shard per machine)")
    ap.add_argument("--engine", choices=("auto", "java", "torch"), default="auto",
                    help="MaxEnt fitting engine: the GPU reimplementation when a GPU is present, else maxent.jar "
                         "(auto, default; provenance 2026-10-04), or either explicitly")
    ap.add_argument("--gpu-slots", type=int, default=2,
                    help="with --engine torch: at most this many workers fit on the GPU at once")
    ap.add_argument("--reassign", default=None,
                    help="parquet (gbifid, species) of records counted toward another listed species (build_name_reassignment.py)")
    ap.add_argument("--prepare-only", action="store_true",
                    help="stop before the range hull and MaxEnt; mode occurrences only (pipeline.run_species)")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = config.load(a.config)
    ps = cfg["per_species"]
    a.root = a.root or str(cfg.root)
    a.parquet = a.parquet or ",".join(str(cfg.path(v) / "parquet") for v in cfg["sources"]["gbif_downloads"].values())
    a.species_table = a.species_table or str(cfg.path(cfg["species"]["table"]))
    a.sbm, a.bias = a.sbm or str(cfg.path(ps["sbm"])), a.bias or str(cfg.path(ps["bias"]))
    a.out = a.out or str(cfg.path(ps["out"]))
    reassign = cfg.path(cfg["species"].get("name_reassignment"))
    if a.reassign is None and reassign is not None and reassign.exists():
        a.reassign = str(reassign)
    if a.prepare_only:
        a.modes = "occurrences"
    if a.engine == "auto":
        a.engine = modelling.default_engine()
    print(f"MaxEnt engine: {a.engine}", flush=True)
    os.environ["MAXENT_ENGINE"] = a.engine               # inherited by the worker processes
    if a.engine == "torch":
        os.environ.setdefault("MAXENT_TORCH_GPU_SLOTS", str(a.gpu_slots))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(a.species_table).drop_duplicates("wcvp_accepted_name")
    si, sn = map(int, a.shard.split("/"))
    table = table[[zlib.crc32(n.encode()) % sn == si for n in table.wcvp_accepted_name]]
    # GBIF's 2026 occurrence files carry the new backbone (alphanumeric keys; accepted names that mostly equal
    # WCVP's), while the species-match API returned old numeric keys and sometimes older names (e.g. Photinia
    # arbutifolia for Heteromeles arbutifolia). Records are therefore taken under either name.
    parts = [str(f) for d in a.parquet.split(",") for f in sorted(Path(d).rglob("*")) if f.is_file() and f.stat().st_size > 0]
    dset = ds.dataset(parts, format="parquet")          # GBIF downloads include zero-byte part files
    # ...and the backbone files some species under other genera (Mahonia aquifolium for WCVP's Berberis
    # aquifolium), so every GBIF name in the files is also resolved through WCVP (Daru step 2a).
    resolved = names.gbif_names_by_accepted(pc.unique(dset.to_table(columns=["species"]).column("species")).to_pylist(),
                                            names.load_wcvp_names(Path(a.root) / "raw/wcvp"))
    names_of = {w: sorted({w, g} | set(resolved.get(w, []))) for w, g in zip(table.wcvp_accepted_name, table.gbif_name)}
    # Fallback for species with no records under any interpreted GBIF name: GBIF files some under a broader species
    # it lumps them into (Salix lasiandra under Salix lucida, Mentha canadensis under Mentha arvensis); their
    # records are then taken by the name they were originally identified under (verbatimScientificName), resolved
    # through WCVP. Applied only where the default finds nothing, so fitted species are unchanged.
    present = set(pc.unique(dset.to_table(columns=["species"]).column("species")).to_pylist())
    wcvp = names.load_wcvp_names(Path(a.root) / "raw/wcvp")
    verbatim_of = {w: names.names_of_accepted(w, wcvp) for w in table.wcvp_accepted_name if not set(names_of[w]) & present}
    if verbatim_of:
        print(f"{len(verbatim_of)} species taken by verbatim name: {sorted(verbatim_of)}", flush=True)
    modes = a.modes.split(",")
    todo = [(s, m) for s in table.wcvp_accepted_name for m in modes
            if not finished(out / m / s.replace(" ", "_") / "summary.json", a.prepare_only)]
    print(f"{len(todo)} (species, mode) tasks", flush=True)
    log = open(out / f"tasks_{a.shard.replace('/', '_')}.jsonl", "a")    # one log per shard (several machines share out/)
    moved = pd.read_parquet(a.reassign) if a.reassign else None
    moved_to = dict(zip(moved.gbifid, moved.species)) if moved is not None else {}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.root, a.sbm, a.bias)) as ex:
        pending = set()
        for i in range(0, len(todo), a.chunk):
            chunk = todo[i:i + a.chunk]
            wanted = sorted({n for s, _ in chunk for n in names_of[s]})
            occ = dset.to_table(columns=list(COLS), filter=ds.field("species").isin(wanted)).to_pandas().rename(columns=COLS)
            fallback = sorted({s for s, _ in chunk} & set(verbatim_of))
            if fallback:
                # one scan for every fallback species of the chunk (genus prefilter), then each species takes the
                # records whose verbatim binomial resolves to it
                expr = None
                for g in sorted({n.split(" × ")[0].split()[0] for w in fallback for n in verbatim_of[w]}):
                    e = pc.starts_with(pc.field("verbatimscientificname"), g)
                    expr = e if expr is None else expr | e
                v = dset.to_table(columns=list(COLS) + ["verbatimscientificname"], filter=expr).to_pandas()
                vb = v.verbatimscientificname.map(names.binomial)
                v = v.drop(columns="verbatimscientificname").rename(columns=COLS)
                occ = pd.concat([occ] + [v[vb.isin(verbatim_of[w])].assign(species=w) for w in fallback], ignore_index=True)
            if moved is not None:
                # records identified as another listed species leave; records identified as this one join
                ids = moved.gbifid[moved.species.isin({s for s, _ in chunk})].tolist()
                inc = dset.to_table(columns=list(COLS), filter=ds.field("gbifid").isin(ids)).to_pandas().rename(columns=COLS) if ids else occ.iloc[0:0]
                inc["species"] = inc.gbifID.map(moved_to)
                dest = occ.gbifID.map(moved_to)

            def records(s):
                base = occ[occ.species.isin(names_of[s])]
                if moved is None:
                    return base
                base = base[dest.loc[base.index].isna() | (dest.loc[base.index] == s)]
                return pd.concat([base, inc[inc.species == s]], ignore_index=True)
            for j in range(0, len(chunk), a.group):
                pending.add(ex.submit(_task, [(s, m, records(s)) for s, m in chunk[j:j + a.group]], str(out), a.prepare_only))
            # keep the pool full: load and submit the next chunk as soon as fewer tasks than workers remain, so a
            # few very slow species (e.g. circumboreal range hulls, ~1 h) do not idle the other cores
            while len(pending) >= a.workers:
                pending = _drain(pending, log, FIRST_COMPLETED)
        while pending:
            pending = _drain(pending, log, FIRST_COMPLETED)


def _drain(pending: set, log, how) -> set:
    done, pending = wait(pending, return_when=how)
    for r in (r for f in done for r in f.result()):
        log.write(json.dumps(r) + "\n"); log.flush()
        print(("DONE " if r["ok"] else "FAIL ") + f'{r["species"]} [{r["mode"]}] ' +
              (f'{r["seconds"]}s beta={r["beta"]} presences={r.get("n_presence")}' if r["ok"] else r["error"][:160]), flush=True)
    return pending


if __name__ == "__main__":
    main()
