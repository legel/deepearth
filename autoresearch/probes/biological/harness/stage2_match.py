"""Fixed stage 2b MATCH: embed candidate binomials with BioCLIP-2.5 (SAME text format as build_pollinator.py),
NN-match (cosine) each USED pollinator_taxon_text_emb row to its nearest candidate. Confident matches
(cosine > 0.98) recover the name. Writes derived/pollinator_names_recovered.csv (idx,name,cosine)."""
import sys, json, csv, time
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/deepearth/autoresearch/probes/biological")
from pathlib import Path
import numpy as np
import torch, open_clip, torch.nn.functional as F
from ensue_log import log_stage

D = Path("/workspace/deepearth/autoresearch/data/deepcal")
DER = D / "derived"
DEV = "cuda:0"

# ---- load used pollinator set + their embeddings ----
z = np.load(DER / "pollinator_used_index.npz")
used = z["used_global"].astype(np.int64)               # vocab indices actually used
emb_all = np.load(D / "pollinator_taxon_text_emb.npy").astype(np.float32)  # [V,1024] L2-normed
used_emb = torch.tensor(emb_all[used]).to(DEV)         # [U,1024]
U = len(used)

# ---- load candidates ----
_cf = DER / "pollinator_candidates.jsonl"      # prefer the full file if harvest finished; else the stable snapshot
if (DER / "pollinator_candidates_snap.jsonl").exists() and not (DER / "harvest_done.flag").exists():
    _cf = DER / "pollinator_candidates_snap.jsonl"
cands = [json.loads(l) for l in open(_cf)]
print(f"candidate source: {_cf.name}", flush=True)
# EXACT build format: "{k} {p} {c} {o} {f} {g} {name}" -> "a photo of {s}."
def fmt(s):
    base = f"{s['kingdom']} {s['phylum']} {s['class']} {s['order']} {s['family']} {s['genus']} {s['name']}"
    return " ".join(base.split()).strip()
strings = [fmt(c) for c in cands]
names = [c["name"] for c in cands]
print(f"loaded {U} used pollinators, {len(cands)} candidates", flush=True)

# ---- embed candidates with BioCLIP-2.5 ----
print("loading BioCLIP-2.5...", flush=True)
m, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2.5-vith14")
m = m.eval().to(DEV)
tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2.5-vith14")
t0 = time.time()
cand_emb = torch.empty((len(strings), used_emb.shape[1]), dtype=torch.float32, device=DEV)
with torch.no_grad():
    B = 256
    for i in range(0, len(strings), B):
        t = m.encode_text(tok([f"a photo of {s}." for s in strings[i:i+B]]).to(DEV))
        cand_emb[i:i+t.shape[0]] = F.normalize(t, dim=-1).float()
        if i % 5120 == 0:
            print(f"  embedded {i}/{len(strings)}  ({time.time()-t0:.0f}s)", flush=True)
print(f"candidate embedding done in {time.time()-t0:.0f}s", flush=True)

# ---- NN match: for each used pollinator, nearest candidate (cosine) ----
best_idx = np.empty(U, dtype=np.int64)
best_cos = np.empty(U, dtype=np.float32)
with torch.no_grad():
    B = 2048
    for i in range(0, U, B):
        sim = used_emb[i:i+B] @ cand_emb.t()          # [b, Ncand]
        mx, am = sim.max(dim=1)
        best_cos[i:i+mx.shape[0]] = mx.cpu().numpy()
        best_idx[i:i+am.shape[0]] = am.cpu().numpy()

# ---- report distribution + save confident matches ----
thr = 0.98
conf = best_cos > thr
qs = np.percentile(best_cos, [1, 5, 25, 50, 75, 95, 99])
rows = []
for u_local in range(U):
    rows.append((int(used[u_local]), names[best_idx[u_local]], float(best_cos[u_local])))
rows_sorted = sorted(rows, key=lambda r: -r[2])
with open(DER / "pollinator_names_recovered.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["idx", "name", "cosine"])
    for idx, nm, cos in sorted(rows, key=lambda r: r[0]):
        if cos > thr:
            w.writerow([idx, nm, f"{cos:.5f}"])
# also full (all matches, for tree fallback / inspection)
with open(DER / "pollinator_names_all.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["idx", "name", "cosine"])
    for idx, nm, cos in sorted(rows, key=lambda r: r[0]):
        w.writerow([idx, nm, f"{cos:.5f}"])

msg = (f"STAGE2b MATCH ok. {U} used pollinators matched to {len(cands)} GBIF candidates. "
       f"cosine pctiles [1,5,25,50,75,95,99]={np.round(qs,4).tolist()}. "
       f"confident (cos>{thr}): {int(conf.sum())}/{U} ({100*conf.mean():.1f}%); "
       f"cos>0.95: {int((best_cos>0.95).sum())} ({100*(best_cos>0.95).mean():.1f}%); "
       f"cos>0.99: {int((best_cos>0.99).sum())} ({100*(best_cos>0.99).mean():.1f}%). "
       f"wrote pollinator_names_recovered.csv ({int(conf.sum())} confident) + pollinator_names_all.csv (all {U}). "
       f"e.g. top: {[(r[1], round(r[2],4)) for r in rows_sorted[:4]]}")
print(msg, flush=True)
log_stage("namematch", msg, "Stage2b: BioCLIP-2.5 NN name recovery of used pollinators")
