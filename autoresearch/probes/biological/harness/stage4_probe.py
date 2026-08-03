"""Fixed stage 4 RULE-27 TEST: run the two-tree bilinear probe with the REAL dated pollinator tree
(--poll_dist realtree) vs the text-prior tree, multi-seed, plant-graph fixed. Report cross_tree_gain
(two_tree - one_tree) per seed and mean, and whether realtree clears the +0.008 floor.

Each seed = a fresh process (per the task: multi-seed per-process) so no state leaks between runs."""
import sys, json, argparse
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/deepearth/autoresearch/probes/biological")
from ensue_log import log_stage
from deepearth.autoresearch.probes.biological.harness import probe as P

def run(poll_dist, seed, steps=400):
    argv = ["--objective", "interaction", "--cache_dir", "/workspace/deepearth/autoresearch/data/deepcal",
            "--poll_dist", poll_dist, "--bidir_mask", "--seed", str(seed),
            "--steps", str(steps), "--device", "cuda:0"]
    return P.main(argv)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--dists", nargs="+", default=["text", "realtree"])
    ap.add_argument("--steps", type=int, default=400)
    a = ap.parse_args()
    results = {d: [] for d in a.dists}
    for d in a.dists:
        for s in a.seeds:
            r = run(d, s, a.steps)
            results[d].append({"seed": s, "cross": r["cross_tree_gain"], "two": r["two_tree_ap"],
                               "one": r["one_tree_ap"], "seed_ap": r["seed_ap"]})
            print(f"[{d} seed{s}] cross={r['cross_tree_gain']:+.4f} two={r['two_tree_ap']:.4f} "
                  f"one={r['one_tree_ap']:.4f} seed={r['seed_ap']:.4f}", flush=True)
    import numpy as np
    lines = []
    for d in a.dists:
        cr = np.array([x["cross"] for x in results[d]])
        lines.append(f"{d}: cross_tree_gain mean={cr.mean():+.4f} std={cr.std():.4f} "
                     f"min={cr.min():+.4f} max={cr.max():+.4f} per-seed={[round(x,4) for x in cr.tolist()]} "
                     f"(clears +0.008 floor: {'YES' if cr.mean()>0.008 and (cr>0).all() else 'no'})")
    verdict = " | ".join(lines)
    rt = np.array([x["cross"] for x in results.get("realtree", [])]) if "realtree" in results else np.array([])
    unlocked = bool(rt.size and rt.mean() > 0.008 and (rt > 0).all())
    msg = (f"STAGE4 RULE-27 ({len(a.seeds)} seeds x {a.steps} steps, per-process, plant-graph fixed, bidir_mask). "
           f"{verdict}. VERDICT: real dated pollinator tree {'UNLOCKS' if unlocked else 'does NOT clear'} "
           f"the +0.008 cross_tree_gain floor.")
    print(msg, flush=True)
    json.dump(results, open("/workspace/deepearth/autoresearch/logs/pollitree/stage4_results.json", "w"), indent=2)
    log_stage("rule27", msg, "Stage4: two-tree bilinear with real dated pollinator tree, multi-seed")

if __name__ == "__main__":
    main()
