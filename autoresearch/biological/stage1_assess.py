"""Stage 1 ASSESS: which pollinator vocab indices are USED (poll_used), their frequencies,
and confirm the BioCLIP-2.5 text format used to build pollinator_taxon_text_emb.npy."""
import sys, json
sys.path.insert(0, "/workspace")
import numpy as np
from pathlib import Path
sys.path.insert(0, "/workspace/deepearth/autoresearch/biological")
from ensue_log import log_stage

D = Path("/workspace/deepearth/data/deepcal")
DER = D / "derived"; DER.mkdir(exist_ok=True)

z = np.load(D / "gbif_pollinator_dist.npz", allow_pickle=True)
print("npz keys:", list(z.keys()), flush=True)
ppi = z["marg_poll_idx"]          # [P,40] pollinator vocab ids (-1 pad)
pfr = z["marg_poll_frq"]
npo = z["marg_npoll"]
emb = np.load(D / "pollinator_taxon_text_emb.npy")
V = emb.shape[0]

# reproduce probe's poll_used exactly: pollinators appearing for kept plants.
# probe filters to plants in E1 vocab; but to be complete we also compute the *global* used set.
used_global = np.unique(ppi[ppi >= 0])

# frequency mass per used pollinator (sum of marg_poll_frq wherever it appears)
freq_mass = np.zeros(V, dtype=np.float64)
occ = np.zeros(V, dtype=np.int64)
P, K = ppi.shape
for p in range(P):
    k = int(npo[p])
    for j in range(min(k, K)):
        pid = int(ppi[p, j])
        if pid >= 0:
            freq_mass[pid] += float(pfr[p, j])
            occ[pid] += 1

order = np.argsort(-freq_mass)
top = [(int(i), int(occ[i]), float(freq_mass[i])) for i in order[:20]]

# save the used-index inventory
np.savez(DER / "pollinator_used_index.npz",
         used_global=used_global.astype(np.int64),
         freq_mass=freq_mass.astype(np.float32),
         occ=occ.astype(np.int64))

msg = (f"STAGE1 ASSESS ok. vocab={V} text_emb_shape={emb.shape} "
       f"emb_norm_mean={np.linalg.norm(emb,axis=1).mean():.4f} | "
       f"pollinators USED (global unique in marg_poll_idx)={len(used_global)} "
       f"({100*len(used_global)/V:.1f}% of vocab); plant rows P={P}, topK width={K}. "
       f"freq-mass covered by used set only. "
       f"TEXT FORMAT (from build_pollinator.py): prompt='a photo of {{s}}.' where "
       f"s='{{kingdom}} {{phylum}} {{class}} {{order}} {{family}} {{genus}} {{binomial}}' "
       f"(GBIF species/match taxonomy prefix, double-spaces collapsed, stripped), "
       f"encoder=hf-hub:imageomics/bioclip-2.5-vith14, L2-normalized. "
       f"MISSING: pollinator_vocab.csv (names) not on box -> must recover via NN match. "
       f"top-occ used idx (idx,occ,freqmass): {top[:8]}")
print(msg, flush=True)
log_stage("assess", msg, "Stage1: used pollinator set + exact BioCLIP-2.5 text format")
print("used_global sample:", used_global[:15], flush=True)
