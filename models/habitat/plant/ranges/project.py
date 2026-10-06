"""Render a species' MaxEnt replicates onto the CONUS 240 m grid (Daru 2024 steps 7–8).

Outputs, all restricted to the calibration area (Daru's realized-niche mask):
    suitability  median cloglog of the replicate models, quantized to uint8 (0 = outside, 1..255 = 0..1)
    binary_vote  Daru's binary map: each replicate thresholded on its own, then majority vote
    binary_p5    median suitability >= median of the replicates' 5th-percentile training-presence thresholds
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import rasterio
import torch

from .maxent import MaxentModel, default_device


@dataclass
class Replicate:
    model: MaxentModel
    threshold_ess: float      # "Equal training sensitivity and specificity" (maxentResults.csv)
    threshold_p5: float       # 5th percentile of the replicate's own training-presence cloglog


def load_replicates(final_dir: Path, label: str, variables: Sequence[str], device: str | None = None) -> list[Replicate]:
    device = default_device(device)
    res = pd.read_csv(final_dir / "maxentResults.csv")
    reps = []
    for lam in sorted(final_dir.glob(f"{label}_[0-9]*.lambdas")):
        i = int(lam.stem.rsplit("_", 1)[1])
        row = res[res.Species == f"{label}_{i}"].iloc[0]
        sp = pd.read_csv(final_dir / f"{label}_{i}_samplePredictions.csv")
        train = sp.loc[sp["Test or train"] == "train", "Cloglog prediction"].values
        reps.append(Replicate(MaxentModel.from_lambdas(lam, variables=variables, dtype=torch.float32, device=device),
                              float(row["Equal training sensitivity and specificity Cloglog threshold"]),
                              float(np.percentile(train, 5))))
    return reps


def compute_maps(reps: list[Replicate], variables: Sequence[str], ecoregion_ids: Sequence[int], stack,
                 rows_per_chunk: int = 256, device: str | None = None) -> tuple[dict, dict]:
    """The single code path that turns replicate models into the three uint8 maps (used by rendering and by
    range-card decoding, so both perform identical arithmetic on identical chunks)."""
    device = default_device(device)
    H, W = stack.shape
    names = ["suitability", "binary_vote", "binary_p5"]
    outs = {n: np.zeros((H, W), np.uint8) for n in names}
    ids = np.asarray(sorted(ecoregion_ids), dtype=np.uint16)
    bands = stack.band_index(variables)
    t_ess = torch.tensor([r.threshold_ess for r in reps], device=device)
    t_p5 = float(np.median([r.threshold_p5 for r in reps]))
    stats = {"cells_calibration": 0, "cells_vote": 0, "cells_p5": 0}
    win = stack.window(ids)
    if win is not None:
        r0, r1, c0, c1 = win
        ids_t = torch.from_numpy(ids.astype(np.int32)).to(device)
        acc = {n: torch.zeros((r1 - r0, c1 - c0), dtype=torch.uint8, device=device) for n in names}
        for a in range(r0, r1, rows_per_chunk):        # chunking (and so arithmetic) identical to the CPU path
            b = min(a + rows_per_chunk, r1)
            mask = torch.isin(torch.from_numpy(stack.eco[a:b, c0:c1].astype(np.int32)).to(device), ids_t)
            if not bool(mask.any()):
                continue
            x = torch.stack([torch.from_numpy(np.ascontiguousarray(stack.data[k, a:b, c0:c1])).to(device)
                             for k in bands], dim=-1)
            ok = mask & torch.isfinite(x).all(-1)
            if not bool(ok.any()):
                continue
            xt = x[ok]
            p = torch.stack([r.model.cloglog(xt) for r in reps], 0)
            med = p.median(0).values
            vote = (p >= t_ess[:, None]).float().mean(0) > 0.5
            p5 = med >= t_p5
            sl = (slice(a - r0, b - r0), slice(None))
            acc["suitability"][sl][ok] = (1 + torch.round(med * 254)).clamp(1, 255).to(torch.uint8)
            acc["binary_vote"][sl][ok] = 1 + vote.to(torch.uint8)
            acc["binary_p5"][sl][ok] = 1 + p5.to(torch.uint8)
            stats["cells_calibration"] += int(ok.sum().item())
            stats["cells_vote"] += int(vote.sum().item())
            stats["cells_p5"] += int(p5.sum().item())
        for n in names:
            outs[n][r0:r1, c0:c1] = acc[n].cpu().numpy()
        del acc
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()           # several renderers share each GPU
    stats["threshold_ess"] = [r.threshold_ess for r in reps]
    stats["threshold_p5_median"] = t_p5
    return outs, stats


def write_maps(outs: dict, stack, out_dir: Path, label: str) -> None:
    """Write the three uint8 maps as tiled ZSTD GeoTIFFs on the CONUS 240 m grid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prof = stack.profile.copy()
    prof.update(dtype="uint8", nodata=0, count=1, compress="zstd", predictor=2, tiled=True, blockxsize=512,
                blockysize=512, BIGTIFF="IF_SAFER")
    for n, a in outs.items():
        with rasterio.open(out_dir / f"{label}.{n}.tif", "w", **prof) as d:
            d.write(a, 1)
