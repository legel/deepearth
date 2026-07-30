"""Stage 3 TREE: build a DATED patristic distance matrix over the USED pollinators, aligned to the
`poll_used` ordering that probe.load_interactions produces. Strategy:
  (a) Open Tree of Life induced subtree over recovered names (rotl/OToL) gives TOPOLOGY;
  (b) since OToL branch lengths are not dated, we date via a TAXONOMIC-RANK ultrametric using the
      recovered GBIF taxonomy (order/family/genus depth) calibrated to coarse Myr rank-heights, which is
      the same 'hybrid genus-crown / dated-clade' philosophy as pollinator_dated_patristic.R. Where OToL
      places two taxa, we use their induced-tree topological depth; else we fall back to the taxonomic
      rank distance. Fill unresolved with the global taxonomic-rank fallback.
Output: derived/pollinator_distance_real.npy  [Nq, Nq] aligned to poll_used (the probe's pollinator order).
Report tree coverage (fraction placed by OToL topology vs rank-fallback)."""
import sys, json, csv, time, urllib.request
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/deepearth/autoresearch/biological")
from pathlib import Path
import numpy as np
from ensue_log import log_stage

D = Path("/workspace/deepearth/data/deepcal")
DER = D / "derived"

# ---- reconstruct poll_used EXACTLY as probe.load_interactions does (global variant: all in-vocab plants) ----
# probe restricts to plants in E1 vocab; we align by loading the same gidx. To keep Stage3 self-contained and
# EXACTLY consistent with the probe, we import the probe's loader.
import torch
from deepearth.autoresearch.biological import probe as P
E1, fam_id, tree, tip_row, gidx = P.load_species(str(D))
Pmap, Ptgt, poll_text, Pfrq = P.load_interactions(str(D), gidx, "cpu")
# recompute poll_used identically (probe does np.unique internally; reproduce to get the ordered ids)
z = np.load(D / "gbif_pollinator_dist.npz", allow_pickle=True)
ppi = z["marg_poll_idx"]; npo = z["marg_npoll"]
g2l = -np.ones(int(gidx.max()) + 1, dtype=np.int64); g2l[gidx] = np.arange(len(gidx))
plant_local = g2l[z["plant_idx"]]
keep_p = (plant_local >= 0) & (npo > 0)
rows = np.where(keep_p)[0]
poll_used = np.unique(ppi[rows][ppi[rows] >= 0])       # THE probe's pollinator order (ascending vocab idx)
Nq = len(poll_used)
print(f"probe pollinator order Nq={Nq} (poll_text rows={poll_text.shape[0]})", flush=True)
assert Nq == poll_text.shape[0], (Nq, poll_text.shape)

# ---- recovered names, indexed by vocab idx ----
name_by_idx = {}
tax_by_idx = {}
allrows = list(csv.DictReader(open(DER / "pollinator_names_all.csv")))
conf = {int(r["idx"]): r["name"] for r in allrows if float(r["cosine"]) > 0.98}
allname = {int(r["idx"]): r["name"] for r in allrows}
# taxonomy from the harvested candidates (by name); prefer the snapshot that the match used
_cf = DER / "pollinator_candidates_snap.jsonl"
if not _cf.exists():
    _cf = DER / "pollinator_candidates.jsonl"
cand_tax = {}
for l in open(_cf):
    c = json.loads(l); cand_tax[c["name"].lower()] = c

names_used = [allname.get(int(i), "") for i in poll_used]
conf_mask = np.array([float(1.0) if (int(i) in conf) else 0.0 for i in poll_used])
print(f"confident names for used set: {int(conf_mask.sum())}/{Nq}", flush=True)

# ---- Open Tree of Life induced subtree topology ----
OTOL_TNRS = "https://api.opentreeoflife.org/v3/tnrs/match_names"
OTOL_ITREE = "https://api.opentreeoflife.org/v3/tree_of_life/induced_subtree"

def _post(url, payload, want_err=False):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            try:
                eb = e.read().decode()
            except Exception:
                eb = ""
            if want_err:
                return {"_httperr": eb}
            last = eb or e; time.sleep(1.0)
        except Exception as e:
            last = e; time.sleep(1.0)
    print("  OToL err:", str(last)[:300], flush=True); return None


def induced_subtree_robust(ids):
    """OToL prunes unknown/broken ott ids; it reports them in the error body. Drop and retry (up to 6x)."""
    ids = list(ids)
    import re as _re
    for attempt in range(6):
        r = _post(OTOL_ITREE, {"ott_ids": ids})
        if r and "newick" in r:
            return r, ids
        # fetch the error body to learn which ids to drop
        er = _post(OTOL_ITREE, {"ott_ids": ids}, want_err=True)
        eb = er.get("_httperr", "") if er else ""
        bad = set()
        for key in ("ott_ids_not_found", "unknown", "node_ids_not_in_tree", "broken"):
            for m in _re.findall(r'"%s"\s*:\s*\[([^\]]*)\]' % key, eb):
                bad |= {int(x) for x in _re.findall(r'\d+', m)}
        # also generic: any ottNNN mentioned in error
        if not bad:
            bad = {int(x) for x in _re.findall(r'ott(\d+)', eb)}
        before = len(ids)
        ids = [i for i in ids if i not in bad]
        print(f"  induced_subtree attempt {attempt}: dropped {before-len(ids)} bad ids, {len(ids)} remain", flush=True)
        if len(ids) < 3 or before == len(ids):
            break
    return None, ids

# map confident names -> OToL ott ids (batch TNRS)
uniq_names = sorted({n for n in names_used if n})
ott_of = {}
CH = 250
for i in range(0, len(uniq_names), CH):
    chunk = uniq_names[i:i+CH]
    r = _post(OTOL_TNRS, {"names": chunk, "do_approximate_matching": False})
    if not r:
        continue
    for res in r.get("results", []):
        ms = res.get("matches", [])
        if ms:
            ott_of[res["name"]] = ms[0]["taxon"]["ott_id"]
    print(f"  TNRS {i+len(chunk)}/{len(uniq_names)} matched {len(ott_of)}", flush=True)

# build ott->list of used-local positions
pos_of_ott = {}
for local, nm in enumerate(names_used):
    oid = ott_of.get(nm)
    if oid is not None:
        pos_of_ott.setdefault(oid, []).append(local)
ott_ids = list(pos_of_ott.keys())
print(f"unique OToL ott ids to place: {len(ott_ids)}", flush=True)

# induced subtree -> newick; then topological patristic via dendropy
import dendropy
otol_dist = None
placed_local = set()
if len(ott_ids) >= 3:
    r, kept_ids = induced_subtree_robust(ott_ids)
    # keep only positions whose ott survived pruning
    if r and "newick" in r:
        kept_set = set(kept_ids)
        pos_of_ott = {o: p for o, p in pos_of_ott.items() if o in kept_set}
        nwk = r["newick"]
        open(DER / "pollinator_otol_induced.nwk", "w").write(nwk)
        # OToL labels tips like 'Genus_species_ottNNN' -> map ottNNN back
        tr = dendropy.Tree.get(data=nwk, schema="newick")
        # assign unit branch length (topology only -> we date via rank calibration below, but keep topo depth)
        for e in tr.edges():
            if e.length is None:
                e.length = 1.0
        pdm = tr.phylogenetic_distance_matrix()
        taxa = list(tr.taxon_namespace)
        import re
        def tip_ott(t):
            m = re.search(r"ott(\d+)", t.label.replace(" ", "_"))
            return int(m.group(1)) if m else None
        tip_ott_map = {t: tip_ott(t) for t in taxa}
        # topological patristic between every placed pair -> fill matrix
        otol_topo = {}
        for a in taxa:
            oa = tip_ott_map[a]
            if oa not in pos_of_ott: continue
            for b in taxa:
                ob = tip_ott_map[b]
                if ob not in pos_of_ott or ob == oa: continue
                otol_topo[(oa, ob)] = pdm.patristic_distance(a, b)
        for oid, locs in pos_of_ott.items():
            for l in locs:
                placed_local.add(l)
        print(f"OToL induced subtree: {len(taxa)} tips, placed {len(placed_local)} used-local pollinators", flush=True)
        otol_dist = otol_topo
    else:
        print("OToL induced_subtree returned no newick", flush=True)

# ---- taxonomic-rank fallback ultrametric (coarse Myr calibration) ----
# rank heights (half-distance to common ancestor, ~Myr): same species=0, same genus, same family, same order, else kingdom
RANKH = {"same": 0.0, "genus": 20.0, "family": 90.0, "order": 250.0, "class": 450.0, "far": 600.0}
def taxrank_dist(i, j):
    ci, cj = cand_tax.get(names_used[i].lower()), cand_tax.get(names_used[j].lower())
    if not ci or not cj:
        return 2 * RANKH["far"]
    if ci["genus"] and ci["genus"] == cj["genus"]:
        return 2 * RANKH["genus"]
    if ci["family"] and ci["family"] == cj["family"]:
        return 2 * RANKH["family"]
    if ci["order"] and ci["order"] == cj["order"]:
        return 2 * RANKH["order"]
    if ci["class"] and ci["class"] == cj["class"]:
        return 2 * RANKH["class"]
    return 2 * RANKH["far"]

# ---- assemble Nq x Nq distance aligned to poll_used ----
Dmat = np.zeros((Nq, Nq), dtype=np.float64)
# scale OToL topological depths to Myr-ish by matching median to the family-rank height, so the two sources are commensurate
otol_vals = np.array(list(otol_dist.values())) if otol_dist else np.array([])
otol_scale = (2 * RANKH["family"]) / (np.median(otol_vals) + 1e-9) if otol_vals.size else 1.0
n_otol = n_rank = 0
for i in range(Nq):
    oi = ott_of.get(names_used[i])
    for j in range(i + 1, Nq):
        oj = ott_of.get(names_used[j])
        d = None
        if otol_dist is not None and oi is not None and oj is not None and (oi, oj) in otol_dist:
            d = otol_dist[(oi, oj)] * otol_scale; n_otol += 1
        else:
            d = taxrank_dist(i, j); n_rank += 1
        Dmat[i, j] = Dmat[j, i] = d
np.fill_diagonal(Dmat, 0.0)
Dmat = 0.5 * (Dmat + Dmat.T)
np.save(DER / "pollinator_distance_real.npy", Dmat.astype(np.float32))

npairs = Nq * (Nq - 1) // 2
cov_otol = n_otol / max(npairs, 1)
placed_frac = len(placed_local) / Nq
msg = (f"STAGE3 TREE ok. Nq(used pollinators)={Nq}. names_confident={int(conf_mask.sum())} ({100*conf_mask.mean():.1f}%). "
       f"OToL TNRS matched {len(ott_of)}/{len(uniq_names)} unique names; induced-subtree placed "
       f"{len(placed_local)} used pollinators ({100*placed_frac:.1f}%). "
       f"pairwise distances: {n_otol} from OToL topology ({100*cov_otol:.1f}%), {n_rank} from taxonomic-rank fallback. "
       f"dist stats Myr-ish: min>0={Dmat[Dmat>0].min():.2f} med={np.median(Dmat[np.triu_indices(Nq,1)]):.1f} "
       f"max={Dmat.max():.1f}. wrote pollinator_distance_real.npy {Dmat.shape} aligned to poll_used.")
print(msg, flush=True)
log_stage("tree", msg, "Stage3: dated pollinator patristic (OToL topology + rank calibration)")
