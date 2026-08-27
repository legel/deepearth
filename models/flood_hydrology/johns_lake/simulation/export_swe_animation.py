#!/usr/bin/env python3
"""Export a compact, web-embeddable animation of the REAL solved water-depth field over the
Winter Garden house, for the benchmark artifact. Base (grayscale hillshade) + N uint8 depth
frames, base64-packed into simulation/outputs/swe_anim.json. Rain droplets are drawn client-side
(visual layer); the water field itself is the actual solver output."""
import os, sys, json, base64
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_swe_benchmark import build_dem, run_swe

DX = 0.10                 # 250x250 solve grid
SIZE = 80                 # exported frame resolution
NFRAMES = 48
SIM_SECONDS = 300.0       # 5 min of physics — long enough for visible flow/ponding
RAIN_MM_HR = 150.0        # heavier than mean so accumulation reads clearly

def main():
    import torch
    from matplotlib.colors import LightSource
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    Z, meta = build_dem(DX, verbose=True)
    print(f"    running {SIM_SECONDS:.0f}s @ {RAIN_MM_HR}mm/hr, capturing {NFRAMES} frames...")
    r = run_swe(Z, DX, SIM_SECONDS, dev, rain_mm_hr=RAIN_MM_HR,
                capture_n=NFRAMES, capture_size=SIZE)
    print(f"    steps={r['steps']:,}  h_max={r['h_max']*100:.1f}cm  "
          f"mass_resid={r['mass_resid_pct']:+.4f}%  frames={len(r['frames'])}")

    # base terrain: downsample Z to SIZE, shaded relief -> uint8 grayscale
    import torch.nn.functional as Fnn
    Zt = torch.as_tensor(Z[None, None], dtype=torch.float32)
    Zs = Fnn.adaptive_avg_pool2d(Zt, (SIZE, SIZE))[0, 0].numpy()
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(Zs, vert_exag=3, dx=DX, dy=DX)          # 0..1
    terr = (np.clip(hs, 0, 1) * 255).astype(np.uint8)

    # depth frames: shared vmax so accumulation grows visibly across the sequence
    stacked = np.stack(r["frames"]) if r["frames"] else np.zeros((1, SIZE, SIZE), np.float32)
    vmax = float(np.percentile(stacked[stacked > 1e-4], 99)) if (stacked > 1e-4).any() else 0.02
    vmax = max(vmax, 0.005)
    q = np.clip(stacked / vmax, 0, 1)
    q = (q * 255).astype(np.uint8)                            # (NFRAMES,SIZE,SIZE)

    payload = dict(
        size=SIZE, nframes=int(q.shape[0]), vmax_m=round(vmax, 4),
        dx=DX, sim_seconds=SIM_SECONDS, rain_mm_hr=RAIN_MM_HR,
        frame_times=[round(t, 1) for t in r["frame_times"]],
        h_max_cm=round(r["h_max"] * 100, 1),
        mass_resid_pct=round(r["mass_resid_pct"], 4),
        terrain_b64=base64.b64encode(terr.tobytes()).decode(),
        frames_b64=base64.b64encode(q.tobytes()).decode(),
    )
    outp = os.path.join(os.path.dirname(__file__), "outputs", "swe_anim.json")
    with open(outp, "w") as f:
        json.dump(payload, f)
    kb = os.path.getsize(outp) / 1024
    print(f"    wrote {outp}  ({kb:.0f} KB, vmax={vmax*100:.2f}cm)")

if __name__ == "__main__":
    main()
