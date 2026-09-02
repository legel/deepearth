"""Read DINOv3 patch32 caches by gbifID."""
import argparse
from pathlib import Path

import numpy as np

CHUNK_GLOB = "chunk[0-9]*.npz"


def _finite_scalar(value, fallback=0.0):
    value = np.float32(value)
    if np.isfinite(value):
        return value
    return np.float32(fallback)


def _finite_obs_elev(elev):
    elev = _finite_scalar(elev)
    if np.isfinite(elev):
        return np.float32(elev)
    return np.float32(0.0)


def _patch_elev_or_obs(chunk, row, obs_elev, shape):
    obs_elev = _finite_obs_elev(obs_elev)
    if "patch_elev" not in chunk:
        return np.full(shape, obs_elev, np.float32)
    elev = chunk["patch_elev"][row].astype(np.float32)
    if np.isfinite(elev).all():
        return elev
    return np.where(np.isfinite(elev), elev, np.float32(obs_elev))


class Patch32Cache:
    def __init__(self, root, patch_dir="gbif_naip_dinov3_patch32_v1"):
        self.root = Path(root).expanduser()
        self.path = self.root / patch_dir
        self.manifest = np.load(self.path / "manifest.npz")
        self.ids = self.manifest["gbifID"].astype(np.int64)
        self.row = {int(g): i for i, g in enumerate(self.ids)}
        self.chunk_row_for_id = {}
        chunks = sorted(self.path.glob(CHUNK_GLOB))
        index = self.path / "chunk_index.npz"
        indexed = set()
        if index.exists():
            z = np.load(index, allow_pickle=True)
            indexed = set(map(str, z["chunk"]))
            for gid, chunk, row in zip(z["gbifID"], z["chunk"], z["row"]):
                chunk_path = self.path / str(chunk)
                if chunk_path.exists():
                    self.chunk_row_for_id[int(gid)] = (chunk_path, int(row))
        missing = [chunk for chunk in chunks if chunk.name not in indexed]
        if missing:
            self.build_index(write=True, chunks=missing)
            return
        if self.chunk_row_for_id:
                return
        self.build_index(write=False)

    def build_index(self, write=True, chunks=None):
        append = chunks is not None
        if append:
            ids = list(self.chunk_row_for_id)
            chunk_names = [self.chunk_row_for_id[gid][0].name for gid in ids]
            rows = [self.chunk_row_for_id[gid][1] for gid in ids]
        else:
            self.chunk_row_for_id = {}
            ids, chunk_names, rows = [], [], []
        seen = set(ids)
        for chunk in sorted(chunks or self.path.glob(CHUNK_GLOB)):
            z = np.load(chunk, allow_pickle=True)
            for row, g in enumerate(z["gbifID"].astype(np.int64)):
                gid = int(g)
                if gid in seen:
                    raise ValueError(f"duplicate gbifID {gid} in {chunk}")
                seen.add(gid)
                self.chunk_row_for_id[gid] = (chunk, row)
                ids.append(gid)
                chunk_names.append(chunk.name)
                rows.append(row)
        if write:
            np.savez(
                self.path / "chunk_index.npz",
                gbifID=np.array(ids, np.int64),
                chunk=np.array(chunk_names, object),
                row=np.array(rows, np.int32),
            )

    def has(self, gbif_id):
        return int(gbif_id) in self.chunk_row_for_id

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
        chunk, j = self.chunk_row_for_id[gid]
        z = np.load(chunk, allow_pickle=True)
        out.update({
            "patch": z["patch"][j],
            "patch_lat": z["patch_lat"][j],
            "patch_lon": z["patch_lon"][j],
            "patch_elev": z["patch_elev"][j] if "patch_elev" in z else None,
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
        elev = row["patch_elev"]
        if elev is None:
            elev = np.full(lat.shape, _finite_obs_elev(row["elev_m"]), np.float32)
        else:
            elev = elev.astype(np.float32)
            elev = np.where(np.isfinite(elev), elev, _finite_obs_elev(row["elev_m"]))
        day = np.full(lat.shape, _finite_scalar(row["event_day"]), np.float32)
        coords = np.stack([lat, lon, elev, day], axis=-1)
        valid = np.isfinite(lat) & np.isfinite(lon)
        return {
            "gbifID": row["gbifID"],
            "has_naip": True,
            "patch": patch,
            "coords": coords,
            "valid": valid,
        }

    def get_many_earth4d(self, gbif_ids):
        gids = [int(g) for g in gbif_ids]
        n = len(gids)
        patch = np.zeros((n, 32, 32, 1024), np.float16)
        coords = np.zeros((n, 32, 32, 4), np.float32)
        valid = np.zeros(n, bool)
        by_chunk = {}
        for row, gid in enumerate(gids):
            item = self.chunk_row_for_id.get(gid)
            if item is None:
                continue
            chunk, j = item
            by_chunk.setdefault(chunk, []).append((row, j, gid))
        for chunk, rows in by_chunk.items():
            z = np.load(chunk, allow_pickle=True)
            for row, j, gid in rows:
                m = self.row[gid]
                patch[row] = z["patch"][j]
                lat = z["patch_lat"][j].astype(np.float32)
                lon = z["patch_lon"][j].astype(np.float32)
                coords[row, ..., 0] = lat
                coords[row, ..., 1] = lon
                coords[row, ..., 2] = _patch_elev_or_obs(
                    z, j, self.manifest["elev_m"][m], lat.shape
                )
                coords[row, ..., 3] = _finite_scalar(self.manifest["event_day"][m])
                valid[row] = np.isfinite(lat).all() and np.isfinite(lon).all()
        return patch, coords, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--patch-dir", default="gbif_naip_dinov3_patch32_v1")
    ap.add_argument("--gbif-id", type=int)
    ap.add_argument("--build-index", action="store_true")
    args = ap.parse_args()

    cache = Patch32Cache(args.cache, args.patch_dir)
    if args.build_index:
        cache.build_index(write=True)
        print(f"indexed {len(cache.chunk_row_for_id):,} patch rows")
        return
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
