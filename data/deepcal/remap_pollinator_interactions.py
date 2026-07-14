"""Reindex the plant->pollinator GloBI interaction labels (gbif_pollinator_dist.npz) from the OLD pollinator vocab
to the CLEANED one, via pollinator_vocab_final_map.json (old_idx -> final_idx | null=dropped). Interactions whose
pollinator was dropped (plant/fungus/family-junk/unresolvable) are removed; per-plant lists recompacted. The coo
arrays (loc_3km/300m, spacetime) carry pollinator indices in column 1 (plant, pollinator, ...) and are filtered +
remapped too. NON-DESTRUCTIVE: writes gbif_pollinator_dist_clean.npz. This is what B41/B51-55 score against once the
pollinator subsystem is enabled on the clean vocab.

    python data/deepcal/remap_pollinator_interactions.py
"""
import os, json
from pathlib import Path
import numpy as np

D = Path(os.environ.get("DEEPCAL_DATA_DIR", "/home/photon/4tb/deepcal_dogfood/data/deepcal"))
MAP = Path("/home/photon/ecological/sandbox/deepearth/data/deepcal/pollinator/pollinator_vocab_final_map.json")


def main():
    z = dict(np.load(D / "gbif_pollinator_dist.npz", allow_pickle=True))
    m = {int(k): v for k, v in json.load(open(MAP)).items()}                 # old_idx -> final_idx | None
    W = z["marg_poll_idx"].shape[1]                                          # top-K width (40)
    pidx, pfrq, npoll = z["marg_poll_idx"], z["marg_poll_frq"], z["marg_npoll"]
    n_plants = len(npoll)
    new_idx = np.full_like(pidx, -1); new_frq = np.zeros_like(pfrq); new_np = np.zeros_like(npoll)
    kept = dropped = 0
    for p in range(n_plants):
        row = []
        for k in range(min(int(npoll[p]), W)):
            fi = m.get(int(pidx[p, k]))
            if fi is not None:
                row.append((fi, pfrq[p, k])); kept += 1
            else:
                dropped += 1
        for j, (fi, fr) in enumerate(row[:W]):
            new_idx[p, j] = fi; new_frq[p, j] = fr
        new_np[p] = len(row)
    z["marg_poll_idx"] = new_idx; z["marg_poll_frq"] = new_frq; z["marg_npoll"] = new_np

    # NOTE: loc_*/spacetime_coo left UNCHANGED — their column semantics aren't referenced in data.py/evaluate.py
    # (likely builder intermediates, not model inputs); do not remap what we don't understand (a col-1=pollinator
    # assumption dropped 99.5%, clearly wrong). Clarify + remap-or-drop them before they're used as model inputs.

    np.savez(D.parent / "deepcal" / "gbif_pollinator_dist_clean.npz", **z)
    print(f"marg interactions: {kept} kept / {dropped} dropped | wrote gbif_pollinator_dist_clean.npz", flush=True)


if __name__ == "__main__":
    main()
