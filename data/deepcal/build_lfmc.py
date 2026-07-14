"""B34 Live Fuel Moisture Content label from field measurements (fire/lfmc_data_conus.csv). Per-species MEDIAN LFMC
(%) over the California measurements, mapped onto the plant vocab by binomial.

Recompute, not reproduce: the previously-shipped gbif_lfmc.npz used an undocumented aggregation and carried an
unphysical outlier (Artemisia californica = 3581 %, a raw fresh-growth reading). Median + a physical clip [10,400]%
is the principled, auditable replacement — robust to those outliers. LFMC is a species-level fuel property; CA
measurements match the CA-plant vocab.

    python -m deepearth.data.deepcal.build_lfmc            # cache = data/deepcal (DEEPCAL_DATA_DIR override)
"""
import os
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(os.environ.get("DEEPCAL_DATA_DIR", Path(__file__).resolve().parent))
LO, HI = 10.0, 400.0                                       # physical LFMC % window (drop bad / fresh-growth readings)


def _norm(s):
    p = str(s).split(); return (p[0] + " " + p[1]).lower() if len(p) >= 2 else str(s).strip().lower()


def main():
    df = pd.read_csv(HERE / "fire/lfmc_data_conus.csv")
    df = df[(df["state_region"] == "California") & df["lfmc_value"].between(LO, HI)]
    med = df.assign(sp=df["species_collected"].map(_norm)).groupby("sp")["lfmc_value"].median()
    print(f"{len(df)} CA measurements -> {len(med)} species with a median LFMC", flush=True)

    vb = np.load(HERE / "gbif_vocab.npz", allow_pickle=True)["binomial"]
    lfmc = np.zeros(len(vb), np.float32); has = np.zeros(len(vb), bool)
    for i, b in enumerate(vb):
        v = med.get(_norm(b))
        if v is not None and np.isfinite(v):
            lfmc[i] = float(v); has[i] = True
    np.savez(HERE / "gbif_lfmc.npz", lfmc=lfmc, has_lfmc=has)
    lv = lfmc[has]
    print(f"gbif_lfmc.npz: {int(has.sum())}/{len(vb)} vocab species labeled | "
          f"LFMC {lv.min():.0f}..{lv.max():.0f}% (median {np.median(lv):.0f}%)", flush=True)


if __name__ == "__main__":
    main()
