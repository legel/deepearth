"""Embed pollinator observation photos into the plant-token space: DINOv2-large CLS (1024) + BioCLIP image (768),
mean-pooled over an observation's photos. Output gbif_pollinator_tokens.npz {obs_id, dino[N,1024], bio[N,768],
plant_taxon, pollinator_taxon, lat, lon, event_date}. GPU (shares with flowering). Resumable via the output."""
import os, csv, numpy as np, torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def load_meta():
    m = {}
    import pandas as pd
    d = pd.read_parquet(os.path.join(HERE, "globi_ca_inat.parquet")).drop_duplicates("obs_id")
    for _, r in d.iterrows():
        m[str(r["obs_id"])] = (r["plant_taxon"], r["pollinator_taxon"], float(r["lat"]), float(r["lon"]), str(r["event_date"]))
    return m

def main():
    from transformers import AutoImageProcessor, AutoModel
    import open_clip
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    dino = AutoModel.from_pretrained("facebook/dinov2-large").eval().to(DEV)
    bio, _, bpre = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    bio = bio.eval().to(DEV)
    meta = load_meta()
    man = {}
    with open(os.path.join(HERE, "photo_manifest.csv")) as f:
        for r in csv.reader(f):
            if len(r) >= 2 and r[1]: man[r[0]] = r[1].split("|")
    out_path = os.path.join(HERE, "..", "gbif_pollinator_tokens.npz")
    done = set()
    if os.path.exists(out_path):
        z = np.load(out_path, allow_pickle=True); done = set(z["obs_id"].tolist())
        acc = {k: list(z[k]) for k in z.files}
    else:
        acc = {k: [] for k in ("obs_id", "dino", "bio", "plant_taxon", "pollinator_taxon", "lat", "lon", "event_date")}
    todo = [o for o in man if o not in done and o in meta]
    print(f"{len(man)} obs w/ photos, {len(todo)} to embed on {DEV}", flush=True)
    n = 0
    for oid in todo:
        imgs = []
        for fp in man[oid][:6]:
            try: imgs.append(Image.open(fp).convert("RGB"))
            except Exception: pass
        if not imgs: continue
        with torch.no_grad():
            dpx = dproc(images=imgs, return_tensors="pt")["pixel_values"].to(DEV)
            de = dino(dpx).last_hidden_state[:, 0].float().mean(0).cpu().numpy()          # CLS, mean over photos
            bpx = torch.stack([bpre(im) for im in imgs]).to(DEV)
            be = bio.encode_image(bpx).float().mean(0).cpu().numpy()
        pt, po, la, lo, ed = meta[oid]
        for k, v in (("obs_id", oid), ("dino", de), ("bio", be), ("plant_taxon", pt),
                     ("pollinator_taxon", po), ("lat", la), ("lon", lo), ("event_date", ed)):
            acc[k].append(v)
        n += 1
        if n % 500 == 0:
            np.savez_compressed(out_path, **{k: np.array(v, object if k in ("obs_id","plant_taxon","pollinator_taxon","event_date") else np.float32) for k, v in acc.items()})
            print(f"  {n}/{len(todo)} embedded", flush=True)
    np.savez_compressed(out_path, **{k: np.array(v, object if k in ("obs_id","plant_taxon","pollinator_taxon","event_date") else np.float32) for k, v in acc.items()})
    print(f"DONE: {len(acc['obs_id'])} pollinator obs embedded -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
