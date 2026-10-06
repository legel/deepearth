"""Neighbourhood context layers for the CONUS 240 m grid (ledger L16): the NaN-aware mean of each WorldClim band
over a 29 x 29-cell window (~7 km), matching the 9 x 9 block of 30" cells that ``GlobalStack.focal`` averages at
training points, and the local deviation from it. One float32 memmap per variable in work/conus240/focal/
(``ConusStack(extra=...)``)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402

R = config.data_root() / "work/conus240"
K, STRIP = 29, 1024


def main():
    meta = json.loads((R / "conus240_stack.json").read_text())
    nb, H, W = meta["shape"]
    data = np.memmap(R / "conus240_stack.f32", dtype=np.float32, mode="r", shape=(nb, H, W))
    out_dir = R / "focal"
    out_dir.mkdir(exist_ok=True)
    h = K // 2
    for b, v in enumerate(meta["variables"]):
        f = out_dir / f"focal9_{v}.f32"
        if f.exists():
            continue
        out = np.memmap(f.with_suffix(".tmp"), dtype=np.float32, mode="w+", shape=(H, W))
        for r0 in range(0, H, STRIP):
            a, z = max(r0 - h, 0), min(r0 + STRIP + h, H)
            x = torch.from_numpy(np.ascontiguousarray(data[b, a:z])).cuda()
            m = torch.isfinite(x).float()
            x = torch.nan_to_num(x) * m
            s = F.avg_pool2d(x[None, None], K, stride=1, padding=h, count_include_pad=True)[0, 0]
            n = F.avg_pool2d(m[None, None], K, stride=1, padding=h, count_include_pad=True)[0, 0]
            mean = torch.where(n > 0, s / n.clamp_min(1e-12), torch.full_like(s, float("nan")))
            mean = torch.where(m > 0, mean, torch.full_like(mean, float("nan")))   # no value where the cell has none
            k = min(STRIP, H - r0)
            out[r0:r0 + k] = mean[r0 - a:r0 - a + k].cpu().numpy()
        out.flush()
        del out
        f.with_suffix(".tmp").rename(f)
        print("focal", v, flush=True)
    # local deviation from the neighbourhood mean (value - focal mean): the fine-scale part of each variable,
    # nearly uncorrelated with the mean, so both scales can pass VIF screening ("decomposed" predictors)
    for b, v in enumerate(meta["variables"]):
        f = out_dir / f"dev9_{v}.f32"
        if f.exists():
            continue
        fm = np.memmap(out_dir / f"focal9_{v}.f32", dtype=np.float32, mode="r", shape=(H, W))
        out = np.memmap(f.with_suffix(".tmp"), dtype=np.float32, mode="w+", shape=(H, W))
        for r0 in range(0, H, 2048):
            out[r0:r0 + 2048] = data[b, r0:r0 + 2048] - fm[r0:r0 + 2048]
        out.flush()
        del out
        f.with_suffix(".tmp").rename(f)
        print("dev", v, flush=True)


if __name__ == "__main__":
    main()
