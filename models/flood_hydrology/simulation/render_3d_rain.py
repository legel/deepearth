#!/usr/bin/env python3
"""Render a 3D digital-twin-style animation of the REAL solved rainfall over the Winter Garden
house: the LiDAR surface in perspective (roof raised above the sloping yard), rain droplets
falling onto it, and the actual solved water depth flowing downslope into the low ground.
Frames -> simulation/outputs/swe_rain_3d.gif."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_swe_benchmark import build_dem, run_swe

DX = 0.10
MESH = 120          # render-grid resolution
NFRAMES = 44
SIM_SECONDS = 180.0
RAIN_MM_HR = 143.5  # real observed all-time 1-hr max, GSDR station US_086638 (31.77km), 1960 —
                    # see export_viewer_data.py for full sourcing/cross-check
DEPTH_EXAG = 10.0   # make the thin water sheet visible on the surface
WET_SHOW = 0.012    # only paint cells wetter than this blue, so shedding slopes stay visible
OUT = os.path.join(os.path.dirname(__file__), "outputs", "swe_rain_3d.gif")


def downsample(a, n):
    import torch, torch.nn.functional as Fnn
    t = torch.as_tensor(a[None, None], dtype=torch.float32)
    return Fnn.adaptive_avg_pool2d(t, (n, n))[0, 0].numpy()


def main():
    import torch
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource, Normalize
    from PIL import Image

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    Z, meta = build_dem(DX, verbose=True)
    print(f"    solving {SIM_SECONDS:.0f}s @ {RAIN_MM_HR}mm/hr, {NFRAMES} frames...")
    r = run_swe(Z, DX, SIM_SECONDS, dev, rain_mm_hr=RAIN_MM_HR,
                capture_n=NFRAMES, capture_size=MESH)
    print(f"    steps={r['steps']:,}  h_max={r['h_max']*100:.1f}cm  resid={r['mass_resid_pct']:+.4f}%")

    from scipy.ndimage import gaussian_filter
    Zs = gaussian_filter(downsample(Z, MESH), 0.7)   # elevation grid (m), lightly smoothed
    frames = [downsample(f, MESH) for f in r["frames"]]

    n = MESH
    ext = 25.0                                     # metres across
    xs = np.linspace(0, ext, n); ys = np.linspace(0, ext, n)
    X, Y = np.meshgrid(xs, ys)

    # terrain shading: hillshade INTENSITY tinted earth-tan (dry ground must NOT read blue,
    # so that real water is the only blue in the scene). Raised roof cells tinted as a building.
    ls = LightSource(azdeg=315, altdeg=45)
    inten = ls.hillshade(Zs, vert_exag=2.5, dx=DX, dy=DX)[..., None]  # 0..1
    ground_col = np.array([0.62, 0.60, 0.47])       # olive-tan grass/soil
    roof_col   = np.array([0.55, 0.44, 0.40])       # muted terracotta-grey roof
    roof_mask = Zs > (np.percentile(Zs, 55) + 1.8)  # the raised structure
    base_col = np.where(roof_mask[..., None], roof_col, ground_col)
    terr_rgb = np.clip(inten * base_col * 1.35, 0, 1)
    terr_rgb = np.dstack([terr_rgb, np.ones((n, n))])

    vmax = max(float(np.percentile(np.stack(frames), 99)), 0.02)
    LIFT_CAP = 0.15                                  # cap water-column height so a deep pit
                                                     # doesn't spike; water reads as a sheet

    # rain droplets: fixed (x,y) columns, z falls, respawn at top
    rng = np.random.default_rng(1)
    RN = 260
    ztop = Zs.max() + 6.0
    rx = rng.uniform(0, ext, RN); ry = rng.uniform(0, ext, RN)
    rz = rng.uniform(Zs.min(), ztop, RN)
    # ground height under each droplet (nearest cell)
    def ground_at(px, py):
        ci = np.clip((px / ext * (n - 1)).astype(int), 0, n - 1)
        ri = np.clip((py / ext * (n - 1)).astype(int), 0, n - 1)
        return Zs[ri, ci]

    imgs = []
    for i, h in enumerate(frames):
        fig = plt.figure(figsize=(6.4, 5.2), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("none"); fig.patch.set_facecolor("#0c1418")

        # terrain surface
        ax.plot_surface(X, Y, Zs, rstride=1, cstride=1, facecolors=terr_rgb,
                        linewidth=0, antialiased=False, shade=False)

        # water sheet: only where wet, lifted slightly, translucent blue by depth
        wmask = h > WET_SHOW
        if wmask.any():
            Wz = Zs + np.clip(h, 0, LIFT_CAP) * DEPTH_EXAG + 0.05
            Wz = np.where(wmask, Wz, np.nan)
            q = np.clip(h / vmax, 0, 1)
            wc = np.zeros((n, n, 4))
            wc[..., 0] = (70 - 55 * q) / 255
            wc[..., 1] = (150 - 90 * q) / 255
            wc[..., 2] = (225 - 70 * q) / 255
            wc[..., 3] = np.where(wmask, np.clip(0.45 + 0.5 * q, 0, 0.92), 0)
            ax.plot_surface(X, Y, Wz, rstride=1, cstride=1, facecolors=wc,
                            linewidth=0, antialiased=False, shade=False)

        # rain: advance, respawn, draw as falling points
        rz2 = rz - (ztop - Zs.min()) * 0.14
        g = ground_at(rx, ry)
        landed = rz2 <= g + 0.15
        rz2[landed] = ztop; rx[landed] = rng.uniform(0, ext, landed.sum())
        ry[landed] = rng.uniform(0, ext, landed.sum())
        rx[:], ry[:], rz[:] = rx, ry, rz2
        vis = rz2 > (g + 0.1)
        ax.scatter(rx[vis], ry[vis], rz2[vis], s=5, c="#bfe2f5",
                   alpha=0.7, depthshade=False, marker="|")

        ax.set_box_aspect((1, 1, 0.55))
        ax.view_init(elev=26, azim=-58)
        ax.set_xlim(0, ext); ax.set_ylim(0, ext); ax.set_zlim(Zs.min(), ztop)
        ax.set_axis_off()
        ax.set_title(f"17801 Champagne Dr  ·  real rainfall physics  ·  t = {r['frame_times'][i]:.0f}s",
                     color="#cfe0e6", fontsize=10, y=0.96)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, hh = fig.canvas.get_width_height()
        imgs.append(Image.fromarray(buf.reshape(hh, w, 4)[..., :3].copy()))
        plt.close(fig)
        if i % 10 == 0:
            print(f"      frame {i+1}/{len(frames)}")

    imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=140, loop=0, optimize=True)
    print(f"    wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB, {len(imgs)} frames)")


if __name__ == "__main__":
    main()
