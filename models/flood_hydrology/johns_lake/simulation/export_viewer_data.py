#!/usr/bin/env python3
"""Export real terrain + solved water frames as a compact JSON for the interactive WebGL
digital-twin viewer. outputs/viewer_data.json — elevation grid, roof mask, and N depth frames,
all uint8+base64."""
import os, sys, json, base64
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_swe_benchmark import build_dem, run_swe

DX = 0.10
N = 112
VELN = 36           # coarser grid for velocity — flow DIRECTION doesn't need depth resolution,
                    # and keeping this small is what keeps the artifact payload small enough
                    # for the hosting sandbox (a full-N velocity export roughly doubled file
                    # size and made the artifact go blank after ~10s — a real, measured limit)
NFRAMES = 40
SIM_SECONDS = 180.0
# Real OBSERVED peak hourly rate, not an arbitrary demo number: GSDR station US_086638
# (31.77 km away — the nearest usable station; none exist within 10km per this project's own
# gsdr/outputs/gsdr_intensity_28p5216_W81p6570.csv), all-time 1-hr max = 143.5mm, from a real
# 1960 storm (already this project's own "Historical GSDR 1960-07-25" scenario in flood_sim.py).
# Cross-checked against NOAA Atlas 14's statistical IDF curve for this exact site
# (precipitation/data/atlas14_idf_28.5216_81.6570W.csv): 143.5mm/hr sits between the 200-yr
# (133mm/hr) and 500-yr (152mm/hr) 1-hr events — a genuine, severe, real extreme, not an
# arbitrary or implausible value.
RAIN_MM_HR = 143.5

def downs(a, n):
    import torch, torch.nn.functional as Fnn
    return Fnn.adaptive_avg_pool2d(torch.as_tensor(a[None, None], dtype=torch.float32),
                                   (n, n))[0, 0].numpy()

def main():
    import torch
    from scipy.ndimage import gaussian_filter
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    Z, meta = build_dem(DX, verbose=True)
    print(f"    solving {SIM_SECONDS:.0f}s @ {RAIN_MM_HR}mm/hr, {NFRAMES} frames...")
    r = run_swe(Z, DX, SIM_SECONDS, dev, rain_mm_hr=RAIN_MM_HR,
                capture_n=NFRAMES, capture_size=N)
    print(f"    steps={r['steps']:,}  h_max={r['h_max']*100:.1f}cm  resid={r['mass_resid_pct']:+.4f}%")

    Zs = gaussian_filter(downs(Z, N), 0.6).astype(np.float32)
    zmin, zmax = float(Zs.min()), float(Zs.max())
    z_u = np.clip((Zs - zmin) / (zmax - zmin) * 255, 0, 255).astype(np.uint8)
    roof = (Zs > (np.percentile(Zs, 55) + 1.8)).astype(np.uint8)

    frames = np.stack([downs(f, N) for f in r["frames"]]).astype(np.float32)
    vmax = max(float(np.percentile(frames[frames > 1e-4], 98)) if (frames > 1e-4).any() else 0.05, 0.02)
    fr_u = np.clip(frames / vmax * 255, 0, 255).astype(np.uint8)

    # real solved velocity (vx,vy m/s), already at capture_size==N so no re-downsampling —
    # this is what actually drives visible downslope flow tracers in the viewer, distinct
    # from the depth-color wash above. Quantized signed (offset-uint8, 128=zero) so it packs
    # the same way as everything else; scale set from the 95th percentile speed over cells
    # that are actually wet (dry-cell divisions by a clamped-tiny depth can spike unphysically
    # and would otherwise blow out the quantization range).
    vel_full = np.stack(r["vel_frames"]).astype(np.float32)         # (NFRAMES,2,N,N)
    vel = np.stack([[downs(vel_full[i, 0], VELN), downs(vel_full[i, 1], VELN)]
                    for i in range(NFRAMES)]).astype(np.float32)    # (NFRAMES,2,VELN,VELN)
    wet_small = np.stack([downs(fr, VELN) for fr in frames]) > (vmax * 0.15)
    speed = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)
    vmax_v = float(np.percentile(speed[wet_small], 95)) if wet_small.any() else float(np.percentile(speed, 95))
    vmax_v = max(vmax_v, 0.01)
    vel_off = np.clip(np.round(np.clip(vel, -vmax_v, vmax_v) / vmax_v * 127) + 128, 0, 255).astype(np.uint8)

    # prepend a genuine t=0 frame: all-zero depth, all-zero velocity (offset-uint8 128=zero).
    # The solver's first CAPTURED frame is already 4.5s into the storm (a real but nonzero
    # state) — showing that as the viewer's starting point mislabels it "t=0" and can leave
    # a few already-just-barely-wet cells at the very start. A real dry t=0 has no such cells.
    fr_u = np.concatenate([np.zeros((1, N, N), dtype=np.uint8), fr_u], axis=0)
    vel_off = np.concatenate([np.full((1, 2, VELN, VELN), 128, dtype=np.uint8), vel_off], axis=0)
    frame_times = [0.0] + [round(t, 1) for t in r["frame_times"]]

    payload = dict(
        n=N, veln=VELN, ext_m=2 * 12.5, dx=DX, z_min=round(zmin, 3), z_max=round(zmax, 3),
        nframes=NFRAMES + 1, vmax_m=round(vmax, 4), vmax_v_ms=round(vmax_v, 4), rain_mm_hr=RAIN_MM_HR,
        frame_times=frame_times,
        h_max_cm=round(r["h_max"] * 100, 1), mass_resid_pct=round(r["mass_resid_pct"], 4),
        z_b64=base64.b64encode(z_u.tobytes()).decode(),
        roof_b64=base64.b64encode(roof.tobytes()).decode(),
        frames_b64=base64.b64encode(fr_u.tobytes()).decode(),
        vx_b64=base64.b64encode(vel_off[:, 0].tobytes()).decode(),
        vy_b64=base64.b64encode(vel_off[:, 1].tobytes()).decode(),
    )
    outp = os.path.join(os.path.dirname(__file__), "outputs", "viewer_data.json")
    json.dump(payload, open(outp, "w"))
    print(f"    wrote {outp}  ({os.path.getsize(outp)/1024:.0f} KB)")

if __name__ == "__main__":
    main()
