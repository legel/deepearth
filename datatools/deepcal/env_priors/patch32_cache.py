"""Read DINOv3 patch32 caches by gbifID."""
import argparse
from pathlib import Path

import numpy as np


class Patch32Cache:
    def __init__(self, root, patch_dir="gbif_naip_dinov3_patch32_v1"):
        self.root = Path(root).expanduser()
        self.path = self.root / patch_dir
        self.manifest = np.load(self.path / "manifest.npz")
        self.ids = self.manifest["gbifID"].astype(np.int64)
        self.row = {int(g): i for i, g in enumerate(self.ids)}
        self.chunk_for_id = {}
        for chunk in sorted(self.path.glob("chunk*.npz")):
            z = np.load(chunk, allow_pickle=True)
            for g in z["gbifID"].astype(np.int64):
                self.chunk_for_id[int(g)] = chunk

    def has(self, gbif_id):
        return int(gbif_id) in self.chunk_for_id

    def get(self, gbif_id):
        gid = int(gbif_id)
        if gid not in self.row:
            raise KeyError(f"gbifID {gid} is absent from manifest")
        out = {
            "gbifID": gid,
            "manifest_row": self.row[gid],
            "lat": self.manifest["lat"][self.row[gid]],
            "lon": self.manifest["lon"][self.row[gid]],
            "elev_m": self.manifest["elev_m"][self.row[gid]],
            "event_day": self.manifest["event_day"][self.row[gid]],
            "obs_ord": self.manifest["obs_ord"][self.row[gid]],
            "has_naip": self.has(gid),
        }
        if not out["has_naip"]:
            return out
        z = np.load(self.chunk_for_id[gid], allow_pickle=True)
        j = int(np.flatnonzero(z["gbifID"].astype(np.int64) == gid)[0])
        out.update({
            "patch": z["patch"][j],
            "patch_lat": z["patch_lat"][j],
            "patch_lon": z["patch_lon"][j],
            "naip_year": z["naip_year"][j],
            "naip_scene": z["naip_scene"][j],
        })
        return out

    def earth4d_inputs(self, gbif_id):
        row = self.get(gbif_id)
        if not row["has_naip"]:
            return {
                "gbifID": row["gbifID"],
                "has_naip": False,
                "patch": None,
                "coords": None,
                "valid": None,
            }
        patch = row["patch"]
        lat = row["patch_lat"].astype(np.float32)
        lon = row["patch_lon"].astype(np.float32)
        elev = np.full(lat.shape, row["elev_m"], np.float32)
        day = np.full(lat.shape, row["event_day"], np.float32)
        coords = np.stack([lat, lon, elev, day], axis=-1)
        valid = np.isfinite(lat) & np.isfinite(lon)
        return {
            "gbifID": row["gbifID"],
            "has_naip": True,
            "patch": patch,
            "coords": coords,
            "valid": valid,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--patch-dir", default="gbif_naip_dinov3_patch32_v1")
    ap.add_argument("--gbif-id", type=int)
    args = ap.parse_args()

    cache = Patch32Cache(args.cache, args.patch_dir)
    gid = args.gbif_id or int(cache.ids[0])
    row = cache.get(gid)
    print(f"gbifID={gid} has_naip={row['has_naip']} lat={float(row['lat']):.6f} lon={float(row['lon']):.6f}")
    if row["has_naip"]:
        print(
            f"patch={row['patch'].shape} {row['patch'].dtype} "
            f"patch_lat={row['patch_lat'].shape} patch_lon={row['patch_lon'].shape}"
        )
        e4d = cache.earth4d_inputs(gid)
        print(
            f"earth4d patch={e4d['patch'].shape} coords={e4d['coords'].shape} "
            f"valid={int(e4d['valid'].sum())}/{e4d['valid'].size}"
        )


if __name__ == "__main__":
    main()
