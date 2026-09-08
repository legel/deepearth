"""Regenerate every figure in README.md from the pipeline's own outputs.

The README must not carry orphaned PNGs. If a figure cannot be rebuilt by this script from
files the shipped code produces, it does not belong in the README.

    python3 docs/make_figures.py --site site3 --storm ian --cell-size 25
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frames as frames_io  # noqa: E402
import sites  # noqa: E402
import validate  # noqa: E402
from validate import CFS_TO_CMS  # noqa: E402

DOCS = Path(__file__).resolve().parent
INK, GRID, SIM, OBS, RAIN = "#dfe7ef", "#2a3644", "#4ea1ff", "#f0a742", "#7f9ec7"


def _style(ax, title=None):
    ax.set_facecolor("#0b0f14")
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)


def _figure(w, h):
    fig = plt.figure(figsize=(w, h), facecolor="#0b0f14", dpi=150)
    return fig


def fig_hydrograph(site, storm, cell_size_m) -> Path:
    """Simulated outflow against the observed gauge record, log axis, timing annotated."""
    csv = site.out_path(f"hydrograph_{storm.name}_{cell_size_m:g}m.csv")
    assert csv.exists(), f"{csv} missing; run the simulate stage"
    data = np.genfromtxt(csv, delimiter=",", names=True)
    t_obs, q_obs = validate.load_observed(site, storm)

    t_sim = data["time_h"]
    q_sim_cfs = data["outflow_total_cms"] / CFS_TO_CMS
    limb_sim = validate.rising_limb_50(t_sim, (q_sim_cfs - q_sim_cfs[0]).clip(0))
    limb_obs = validate.rising_limb_50(t_obs, (q_obs - site.gauge.baseflow_cfs).clip(0))

    fig = _figure(9, 4.6)
    ax = fig.add_axes([0.09, 0.30, 0.88, 0.62])
    ax.semilogy(t_sim, np.maximum(q_sim_cfs, 10), color=SIM, lw=1.4,
                label="simulated, domain outflow (46.6 km$^2$)")
    ax.semilogy(t_obs, np.maximum(q_obs, 10), color=OBS, lw=1.4,
                label=f"observed, USGS {site.gauge.site_no} (33.2 km$^2$)")
    for t, c in ((limb_sim, SIM), (limb_obs, OBS)):
        ax.axvline(t, color=c, ls=":", lw=1.0, alpha=0.8)
    ax.annotate(f"rising limb differs by {abs(limb_sim - limb_obs):.2f} h\n"
                f"(gauge resolves {validate.GAUGE_DT_H:.2f} h)",
                xy=(limb_obs, 2.2e3), xytext=(limb_obs + 6, 6e3), color=INK, fontsize=8,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    ax.set_ylabel("discharge  [cfs, log]", color=INK, fontsize=9)
    ax.set_xlim(0, max(t_sim.max(), 96))
    ax.set_ylim(10, 3e4)
    _style(ax, f"Hurricane Ian  -  {site.label}")
    ax.tick_params(labelbottom=False)  # shared with the rainfall panel below
    leg = ax.legend(loc="upper right", fontsize=8, facecolor="#111823", edgecolor=GRID)
    for text in leg.get_texts():
        text.set_color(INK)

    axr = fig.add_axes([0.09, 0.10, 0.88, 0.17], sharex=ax)
    axr.bar(t_sim, data["rain_mm_hr"], width=0.05, color=RAIN)
    axr.invert_yaxis()
    axr.set_ylabel("rain\n[mm/hr]", color=INK, fontsize=8)
    axr.set_xlabel("hours from 2022-09-28 00:00 UTC", color=INK, fontsize=9)
    _style(axr)

    out = DOCS / "hydrograph_ian.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def fig_flood_peak(site, storm, cell_size_m) -> Path:
    """Peak simulated depth over shaded terrain."""
    import rasterio
    from matplotlib.colors import LightSource
    from scipy.ndimage import gaussian_filter

    frames_path = site.out_path(f"frames_{storm.name}_{cell_size_m:g}m.bin")
    assert frames_path.exists(), f"{frames_path} missing; run the simulate stage"
    _, frames = frames_io.read(frames_path)
    peak = frames.max(axis=0)

    with rasterio.open(site.dem_conditioned) as src:
        dem = src.read(1, out_shape=peak.shape).astype(float)
        nodata = src.nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan
    gaps = ~np.isfinite(dem)
    # Smooth before shading. A raw gradient on a downsampled DEM over 25 m of relief is mostly
    # resampling speckle, which reads as noise rather than terrain.
    filled = np.where(gaps, np.nanmedian(dem), dem)
    shade = LightSource(azdeg=315, altdeg=45).hillshade(
        gaussian_filter(filled, 1.2), vert_exag=30, dx=cell_size_m, dy=cell_size_m)
    shade = np.ma.masked_where(gaps, shade)

    # Size the canvas to the raster so the axes box fills it exactly; imshow preserves aspect
    # and any mismatch shows up as dead space around the map.
    rows, cols = peak.shape
    fig_w, ax_frac_w, ax_frac_h = 6.4, 0.80, 0.88
    fig = _figure(fig_w, (fig_w * ax_frac_w * rows / cols) / ax_frac_h)
    ax = fig.add_axes([0.02, 0.03, ax_frac_w, ax_frac_h])
    ax.imshow(shade, cmap="Greys_r", vmin=0.15, vmax=1.05, interpolation="bilinear")
    im = ax.imshow(np.ma.masked_less(peak, 0.05), cmap="YlGnBu", vmin=0.05, vmax=1.5,
                   interpolation="nearest", alpha=0.92)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_title(f"Peak flood depth, Hurricane Ian  -  {cell_size_m:g} m grid",
                 color=INK, fontsize=10, loc="left", pad=8)

    cax = fig.add_axes([0.855, 0.23, 0.028, 0.48])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("depth [m]", color=INK, fontsize=9)
    cb.ax.tick_params(colors=INK, labelsize=8)
    cb.outline.set_edgecolor(GRID)

    out = DOCS / "flood_peak_ian.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def fig_flood_animation(site, storm, cell_size_m, stride: int = 2) -> Path:
    """The storm as an animated GIF: the network fills, then drains."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    times, frames = frames_io.read(site.out_path(f"frames_{storm.name}_{cell_size_m:g}m.bin"))
    keep = np.arange(0, len(times), stride)

    fig = _figure(5.4, 5.6)
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.88])
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(frames[0], cmap="YlGnBu", vmin=0.02, vmax=1.5, interpolation="nearest")
    label = ax.set_title("", color=INK, fontsize=10, loc="left", pad=8)

    def update(i):
        k = keep[i]
        im.set_data(np.ma.masked_less(frames[k], 0.02))
        label.set_text(f"Hurricane Ian   t = {times[k] / 60:5.1f} h")
        return im, label

    anim = FuncAnimation(fig, update, frames=len(keep), blit=False)
    out = DOCS / "flood_ian.gif"
    anim.save(out, writer=PillowWriter(fps=6), savefig_kwargs={"facecolor": "#0b0f14"})
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--site", default="site3", choices=sorted(sites.SITES))
    ap.add_argument("--storm", default="ian", choices=sorted(sites.STORMS))
    ap.add_argument("--cell-size", type=float, default=25.0)
    args = ap.parse_args()

    site, storm = sites.get_site(args.site), sites.get_storm(args.storm)
    for fn in (fig_hydrograph, fig_flood_peak, fig_flood_animation):
        path = fn(site, storm, args.cell_size)
        print(f"  {path.name:24s} {path.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
