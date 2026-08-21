"""Build state/observations.npz: one compact row per observation for the map + detail views.

Reproduces the default spatial holdout of core/data.py exactly
(0.5-degree cells, rng(0), 1/6 held out). Modality presence is a bitmask.

    python -m dashboard.observations
"""
import glob, json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "deepcal"
STATE = ROOT / "state"
MODS = ["daymet", "naip", "clay", "soil", "topo", "chm", "hydro", "flower", "species_dist"]


def _ids(pattern_or_file, key="gbifID"):
    ids = []
    for f in sorted(glob.glob(str(pattern_or_file))):
        ids.append(np.load(f, allow_pickle=True)[key])
    return np.concatenate(ids) if ids else np.array([], np.int64)


def build():
    gid, sp, lat, lon = [], [], [], []
    for f in sorted(glob.glob(str(DATA / "gbif_tokens" / "chunk*.npz"))):
        z = np.load(f, allow_pickle=True)
        gid.append(z["gbifID"]); sp.append(z["species_local"])
        lat.append(z["lat"]); lon.append(z["lon"])
    gid, sp = np.concatenate(gid), np.concatenate(sp)
    lat, lon = np.concatenate(lat), np.concatenate(lon)
    n = len(gid)

    ez = np.load(DATA / "gbif_elev.npz")
    elev_by = dict(zip(ez["gbifID"].tolist(), ez["elev"].tolist()))
    tz = np.load(DATA / "gbif_eventtime.npz")
    days_by = dict(zip(tz["gbifID"].tolist(), tz["days"].tolist()))
    elev = np.array([elev_by.get(g, 0.0) for g in gid.tolist()], np.float32)
    days = np.array([days_by.get(g, np.nan) for g in gid.tolist()], np.float32)

    cell = np.floor(lat / 0.5).astype(np.int64) * 10007 + np.floor(lon / 0.5).astype(np.int64)
    cells = np.unique(cell)
    np.random.default_rng(0).shuffle(cells)
    test = np.isin(cell, cells[: max(1, int(len(cells) / 6))])

    sets = {
        "daymet": _ids(DATA / "gbif_daymet_tokens" / "chunk*.npz"),
        "naip": _ids(DATA / "gbif_naip_tokens" / "chunk*.npz"),
        "clay": np.load(DATA / "gbif_clay_tokens.npz", allow_pickle=True)["gbifID"],
        "soil": np.load(DATA / "gbif_soil_tokens.npz", allow_pickle=True)["gbifID"],
        "topo": np.load(DATA / "gbif_topo_tokens.npz", allow_pickle=True)["gbifID"],
        "chm": np.load(DATA / "gbif_chm_tokens.npz", allow_pickle=True)["gbifID"],
        "hydro": np.load(DATA / "gbif_hydro_tokens.npz", allow_pickle=True)["gbifID"],
        "flower": np.load(DATA / "gbif_flower_all.npz", allow_pickle=True)["gbifID"],
        "species_dist": np.load(DATA / "gbif_species_dist.npz", allow_pickle=True)["gbifID"],
    }
    mods = np.zeros(n, np.uint16)
    for i, m in enumerate(MODS):
        mods |= np.isin(gid, sets[m]).astype(np.uint16) << i

    STATE.mkdir(exist_ok=True)
    np.savez_compressed(STATE / "observations.npz", gbifID=gid, sp=sp, lat=lat, lon=lon,
                        elev=elev, days=days, test=test, mods=mods)
    vocab = np.load(DATA / "gbif_vocab.npz", allow_pickle=True)
    counts = np.bincount(sp, minlength=len(vocab["binomial"]))
    (STATE / "species.json").write_text(json.dumps(
        {"binomial": vocab["binomial"].tolist(), "count": counts.tolist()}) + "\n")
    print(f"observations: {n} rows, {int(test.sum())} test ({test.mean():.1%}), "
          f"{len(vocab['binomial'])} species")


if __name__ == "__main__":
    build()
