"""Figure 1 of the FluxFill report: one real validation window (US-YK2, from 2019-07-15) with part of every
reconstructed modality hidden (left), the network (centre), the reconstruction on the same axes (right), test-time
site adaptation as an unrolled recurrence over the weights, the predictive distribution inside the outage, and the
table of all methods.

Inputs: the final FluxFill on this window under one mask (bench/dashboard/report/fig1_all.npz, report_fig1_data.py),
the raw samples of FluxFill and the established methods' fills of the benchmark's 30-day mask (samples/,
fig1_window_base/), the Sentinel-2 and Clay stores of the tower, its half-hourly table, and the test-tower CRPS of every
method (`vals["fig1_table"]`, from report_figures.py).

Time series are daily means on blocks aligned to the start of each row's hidden span, so every hidden block is exactly
hidden and every visible block exactly visible. Input and reconstruction of a row share their axes.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Patch, Polygon, Rectangle

R = Path("/home/photon/4tb/fluxfill")
REP = R / "bench" / "dashboard" / "report"
SITE, START, KEY, LAT, LON = "US-YK2", "2019-07-15", "US-YK2_2019-07-15", 61.2548, -163.259
MASK = "M2: 30 d, all tower channels"
C = {"flux": "#1d4466", "met": "#6b6f2e", "era": "#8a9a3a", "img": "#6e1f3f", "mem": "#1e2a35", "loc": "#4a5a7a",
     "adapt": "#8a5a00"}
INK, GREY, SHADE = "#1e2a35", "#6b6f76", "#f1efe6"
TOP_C = "#eb6834"                      # the best established method on this window
NAMES = {"climatology": "Climatology", "mds": "MDS", "irvin_rf": "RF ensemble", "liu_xgb": "XGBoost", "saits": "SAITS"}
LAB = {"NEE": "NEE", "LE": "LE", "H": "H", "FCH4": r"FCH$_4$", "TA": "TA", "SW_IN": r"SW$_{in}$", "VPD": "VPD",
       "SWC_1": "SWC", "ERA5_TA": "TA", "ERA5_SW_IN": r"SW$_{in}$", "ERA5_VPD": "VPD"}
MONO = {"family": "DejaVu Sans Mono", "size": 6.0, "weight": "bold"}

# columns (figure fractions): input plots, encoders, sum, network, decoders, output plots; symmetric about 0.5
L0, L1, R0, R1 = 0.12, 0.36, 0.64, 0.88
XE, XS, XT0, XT1, XD = 0.373, 0.413, 0.43, 0.57, 0.586
XC = 0.5
H_ROW, PITCH, G_GAP = 0.02, 0.0245, 0.032
ASPECT = 7.2 / 10.8                    # figure width / height


def blocks(y, a, L=48):
    """Means over blocks of L half-hours that start at a (mod L), along the last axis; block centres in days."""
    off = a % L
    n = (y.shape[-1] - off) // L
    z = y[..., off:off + n * L].reshape(*y.shape[:-1], n, L)
    with np.errstate(all="ignore"):
        m = np.nanmean(z, -1)
    return (off + (np.arange(n) + 0.5) * L) / 48, m


def tri(fig, x, y, w, h, color, text, right=True, size=7.0):
    """Outlined triangle: an encoder (base left, apex right) or a decoder (apex left, base right)."""
    pts = [(x, y - h / 2), (x, y + h / 2), (x + w, y)] if right else [(x + w, y - h / 2), (x + w, y + h / 2), (x, y)]
    fig.add_artist(Polygon(pts, closed=True, fill=False, ec=color, lw=1.0, transform=fig.transFigure))
    fig.text(x + (w * 0.36 if right else w * 0.64), y, text, color=color, ha="center", va="center", fontsize=size,
             fontstyle="italic", fontweight="bold")


def arrow(fig, p0, p1, color, lw=0.8, ls="-", rad=0.0, scale=5):
    fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=scale, color=color,
                                   lw=lw, ls=ls, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def circle(fig, x, y, r, **kw):
    """A round circle of radius r (in figure widths)."""
    fig.add_artist(Ellipse((x, y), 2 * r, 2 * r * ASPECT, transform=fig.transFigure, **kw))


def line(fig, xs, ys, color, lw=0.8):
    fig.add_artist(Line2D(xs, ys, color=color, lw=lw, transform=fig.transFigure, solid_capstyle="butt"))


def box(fig, x, y, w, h, fc, ec, lw=0.8, r=0.01):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.003,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                  transform=fig.transFigure))


def rgb_of(g, i, t):
    x = np.array(g["r10"][i], float)[[2, 1, 0]]
    x = np.clip((x - 1000) / 10000, 0, None) if t >= np.datetime64("2022-01-25") else x / 10000   # baseline offset
    return np.transpose(np.clip(x / 0.16, 0, 1) ** 0.8, (1, 2, 0))             # one stretch for every chip


def chips(f):
    """Clear Sentinel-2 acquisitions of the window (true colour, 10 m), at least 20 days apart, two of them inside the
    hidden imagery span; for each hidden one, the clear acquisition of the tower's other years whose layer-normalised
    Clay embedding is nearest (cosine) to the reconstructed embedding."""
    g = zarr.open(str(R / "sentinel2" / "sites" / f"{SITE}.zarr"), mode="r")
    c = zarr.open(str(R / "clay" / "sites" / f"{SITE}.zarr"), mode="r")
    t = np.array(g["time_us"][:]).astype("datetime64[us]")
    t0 = np.datetime64(START)
    clear = np.asarray(c["meta"]["clear_frac"][:], float)
    day = ((t - t0) / np.timedelta64(1, "D"))
    inwin = np.flatnonzero((day >= 0) & (day < 180) & (clear > 0.6))
    a, b = f["img_span"]
    pick = []
    for target in [a + 2, b - 4, 90, 112, 130]:
        cand = [i for i in inwin if all(abs(day[i] - day[j]) > 19.8 for j in pick) and
                ((a <= day[i] < b) == (a <= target < b))]
        if cand:
            pick.append(min(cand, key=lambda i: abs(day[i] - target)))
    pick = sorted(pick, key=lambda i: day[i])
    years = pd.DatetimeIndex(t).year
    other = np.flatnonzero(~years.isin([2019, 2020]) & (clear > 0.9))
    emb = np.asarray(c["cls"].get_orthogonal_selection((other, slice(None))), float)
    emb = (emb - emb.mean(1, keepdims=True)) / np.sqrt(emb.var(1, keepdims=True) + 1e-5)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    out = []
    for i in pick:
        hidden = a <= day[i] < b
        rec = None
        if hidden:
            mu = f["img_mu"][int(np.argmin(np.abs(f["img_days"] - day[i])))]
            j = other[int(np.argmax(emb @ (mu / np.linalg.norm(mu))))]
            rec = rgb_of(g, j, t[j])
        out.append((float(day[i]), rgb_of(g, i, t[i]), hidden, rec))
    return out


def memory_band():
    """FCH4 of the tower's other years at each day of the window's season: daily mean and standard deviation."""
    t = pd.read_parquet(R / "towers" / "hh" / f"{SITE}.parquet", columns=["time_lst", "FCH4"])
    t["time_lst"] = pd.to_datetime(t["time_lst"])
    t = t[~t.time_lst.dt.year.isin([2019, 2020])].dropna()
    doy0 = pd.Timestamp(START).dayofyear
    t["d"] = (t.time_lst.dt.dayofyear - doy0) % 365
    t = t[t.d < 180]
    g = t.groupby("d").FCH4
    return g.mean().reindex(range(180)).to_numpy(), g.std().reindex(range(180)).to_numpy()


def top_method(wc, key):
    """The established method with the lowest CRPS on this window's 30-day mask for one variable."""
    sc = {m: wc["scores"][m].get(MASK, {}).get(key, {}).get("crps") for m in wc["scores"]}
    sc = {m: v for m, v in sc.items() if v is not None}
    return min(sc, key=sc.get) if sc else None


def pred(w, method, key):
    seg = w["predictions"].get(method, {}).get(MASK, {}).get(key)
    if not seg:
        return None
    s = seg[0]
    out = {k: np.full(8640, np.nan) for k in ("mean", "q025", "q975")}
    for k in out:
        out[k][s["a"]:s["a"] + len(s[k])] = s[k]
    return out


def grid_icon(fig, x, y, w, h, nv, nd, hl_row=None, hl_col=None, color=INK, img_row=False):
    """A token grid (variables x days); one row (attention over days) or one column (over variables) highlighted."""
    n = nv + int(img_row)
    cw, ch = w / nd, h / n
    for i in range(n):
        for j in range(nd):
            on = (hl_row is not None and i == hl_row) or (hl_col is not None and j == hl_col)
            fc = color if on else ("#efe6ea" if img_row and i == nv else "#e6e9ee")
            fig.add_artist(Rectangle((x + j * cw, y + h - (i + 1) * ch), cw * 0.82, ch * 0.76, fc=fc, ec="none",
                                     transform=fig.transFigure))


def frame(ax):
    ax.set_xlim(0, 180)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d5d8dd"); ax.spines["bottom"].set_linewidth(0.5)
    ax.patch.set_alpha(0)


def figure1(out: Path, vals: dict):
    f = np.load(REP / "fig1_all.npz")
    keys = [str(k) for k in f["keys"]]
    Y = np.where(f["measured"], f["values"], np.nan)
    M, S = f["mask"], f["samples"]
    wc = json.load(open(REP / "fig1_window_base" / f"{KEY}.json"))
    sm = np.load(REP / "samples" / f"{KEY}_fluxfill_M2-30d.npz")
    skeys = [str(k) for k in sm["keys"]]
    fig = plt.figure(figsize=(7.2, 10.8))
    top = 0.982
    for x, h in [((L0 + L1) / 2, "MASKED INPUT"), (XC, "DEEPEARTH"), ((R0 + R1) / 2, "RECONSTRUCTION")]:
        fig.text(x, top, h, ha="center", va="center", **MONO, color=INK)
    for lab, x in (("a", 0.02), ("b", XT0 - 0.012), ("c", R0 - 0.012)):
        fig.text(x, top, lab, fontsize=10, fontweight="bold", va="center")
    y = top - 0.03
    centers, spans = {}, {}

    def title(kind, text, y):
        fig.text(L0, y + 0.004, text, color=C[kind], **MONO)

    def series_row(kind, k, y0, recon=True):
        i = keys.index(k)
        hid = M[i]
        a = int(np.flatnonzero(hid)[0]) if hid.any() else 0
        b = int(np.flatnonzero(hid)[-1]) + 1 if hid.any() else 0
        xb, truth = blocks(Y[i], a)
        _, vis = blocks(np.where(hid, np.nan, Y[i]), a)
        _, sd = blocks(np.where(hid[None], S[:, i], np.nan), a)               # daily means of each sample
        with np.errstate(all="ignore"):
            lo, hi, mu = np.nanquantile(sd, 0.025, 0), np.nanquantile(sd, 0.975, 0), np.nanmean(sd, 0)
        allv = np.concatenate([truth[np.isfinite(truth)], lo[np.isfinite(lo)], hi[np.isfinite(hi)]])
        pad = 0.08 * (allv.max() - allv.min() + 1e-9)
        ylim = (allv.min() - pad, allv.max() + pad)
        ax = fig.add_axes([L0, y0, L1 - L0, H_ROW]); frame(ax); ax.set_ylim(ylim)
        ax.plot(xb, vis, color=C[kind], lw=0.6)
        if hid.any():
            ax.axvspan(a / 48, b / 48, color="black", lw=0)
            ax.text((a + b) / 96, 0.5, "?", color="white", ha="center", va="center", fontsize=5.5, fontweight="bold",
                    transform=ax.get_xaxis_transform())
        fig.text(L0 - 0.008, y0 + H_ROW / 2, LAB[k], ha="right", va="center", fontsize=5.4, color=C[kind])
        if not recon:
            return
        ax = fig.add_axes([R0, y0, R1 - R0, H_ROW]); frame(ax); ax.set_ylim(ylim)
        if hid.any():
            ax.axvspan(a / 48, b / 48, color=SHADE, lw=0, zorder=0)
            ax.fill_between(xb, lo, hi, color=C[kind], alpha=0.22, lw=0)
            ax.plot(xb, mu, color=C[kind], lw=0.9)
            inside = (xb > a / 48) & (xb < b / 48)
            ax.plot(xb[inside], truth[inside], ".", color="black", ms=1.5, mew=0)
        ax.plot(xb, vis, color=C[kind], lw=0.6)
        fig.text(R1 + 0.008, y0 + H_ROW / 2, LAB[k], ha="left", va="center", fontsize=5.4, color=C[kind])
        spans[kind] = (a, b)

    groups = [("flux", "TOWER FLUXES", ["NEE", "LE", "H", "FCH4"]),
              ("met", "TOWER METEOROLOGY", ["TA", "SW_IN", "VPD", "SWC_1"])]
    for kind, text, ks in groups:
        title(kind, text, y)
        ys = []
        for j, k in enumerate(ks):
            y0 = y - 0.008 - (j + 1) * PITCH + (PITCH - H_ROW)
            series_row(kind, k, y0); ys.append(y0 + H_ROW / 2)
        centers[kind] = (np.mean(ys), ys[0], ys[-1])
        y = y - 0.008 - len(ks) * PITCH - G_GAP + 0.008

    # site memory: three channels of every tower token (input only)
    title("mem", "SITE MEMORY, OTHER YEARS", y)
    mu_m, sd_m = memory_band()
    y0 = y - 0.008 - 0.022
    ax = fig.add_axes([L0, y0, L1 - L0, 0.022]); frame(ax)
    dd = np.arange(180) + 0.5
    ax.fill_between(dd, mu_m - sd_m, mu_m + sd_m, color=C["mem"], alpha=0.14, lw=0); ax.plot(dd, mu_m, color=C["mem"], lw=0.6)
    fig.text(L0 - 0.008, y0 + 0.011, LAB["FCH4"], ha="right", va="center", fontsize=5.4, color=C["mem"])
    centers["mem"] = (y0 + 0.011,)
    # legend of the reconstruction, in the free space of this row
    hand = [Line2D([], [], color=C["flux"], lw=0.9, label="mean"), Patch(color=C["flux"], alpha=0.22, lw=0, label="95% interval"),
            Line2D([], [], color="black", lw=0, marker=".", ms=3, label="measured, hidden")]
    fig.legend(handles=hand, loc="center", bbox_to_anchor=((R0 + R1) / 2, y0 + 0.011), ncol=3, fontsize=5.2, frameon=False,
               handlelength=1.4, columnspacing=1.2, handletextpad=0.5)
    y = y0 - G_GAP + 0.008

    title("era", "REANALYSIS (ERA5)", y)
    ys = []
    for j, k in enumerate(["ERA5_TA", "ERA5_SW_IN", "ERA5_VPD"]):
        y0 = y - 0.008 - (j + 1) * PITCH + (PITCH - H_ROW)
        series_row("era", k, y0); ys.append(y0 + H_ROW / 2)
    centers["era"] = (np.mean(ys), ys[0], ys[-1])
    y = y - 0.008 - 3 * PITCH - G_GAP + 0.008

    # Sentinel-2 chips and Clay embeddings
    title("img", "SENTINEL-2 300 m AND CLAY EMBEDDINGS", y)
    a_i, b_i = f["img_span"]
    cw = 0.026; chh = cw * ASPECT
    yc = y - 0.008 - chh
    for side in (0, 1):
        x0 = (L0, R0)[side]
        for d, im, hidden, rec in chips(f):
            ax = fig.add_axes([x0 + (L1 - L0) * d / 180 - cw / 2, yc, cw, chh])
            ax.set_xticks([]); ax.set_yticks([])
            if side == 0 and hidden:
                ax.set_facecolor("black")
                ax.text(0.5, 0.5, "?", color="white", ha="center", va="center", fontsize=6, fontweight="bold", transform=ax.transAxes)
            else:
                ax.imshow(rec if (side and hidden) else im, interpolation="nearest")
            for s in ax.spines.values():
                s.set_color(C["img"] if not (side and hidden) else "black"); s.set_linewidth(0.5 if not (side and hidden) else 1.0)
            if side and hidden:
                for s in ax.spines.values():
                    s.set_linestyle((0, (1.5, 1)))
    ys_ = yc - 0.006 - 0.016
    ln, mu = f["img_ln"], f["img_mu"]
    dims = np.argsort(ln.var(0))[::-1][:48]
    u = np.linalg.svd(ln[:, dims] - ln[:, dims].mean(0), full_matrices=False)[2][0]
    dims = dims[np.argsort(u)]                                # similar dimensions adjacent
    days = list(f["img_days"]) + [180]
    for side in (0, 1):
        z = np.full((48, 180), np.nan)
        for j, d in enumerate(f["img_days"]):
            hidden = f["img_hidden"][j]
            if side == 0 and hidden:
                continue
            v = (mu if (side and hidden) else ln)[j, dims]
            z[:, d:min(days[j + 1], d + 5)] = v[:, None]         # held until the next acquisition, at most 5 days
        x0 = (L0, R0)[side]
        ax = fig.add_axes([x0, ys_, L1 - L0, 0.016]); frame(ax)
        if side:
            ax.axvspan(a_i, b_i, color=SHADE, lw=0, zorder=0)
        ax.imshow(z, aspect="auto", cmap="PuOr_r", vmin=-2.5, vmax=2.5, extent=[0, 180, 0, 1], interpolation="nearest", zorder=1)
        ax.set_ylim(0, 1)
        if side:
            ax.add_patch(Rectangle((a_i, 0), b_i - a_i, 1, fc="none", ec="black", lw=0.7, zorder=2))
        if side == 0:
            ax.axvspan(a_i, b_i, color="black", lw=0, zorder=2)
            ax.text((a_i + b_i) / 2, 0.5, "?", color="white", ha="center", va="center", fontsize=5.5, fontweight="bold", zorder=3)
        lx, ha = (L0 - 0.008, "right") if side == 0 else (R1 + 0.008, "left")
        fig.text(lx, ys_ + 0.008, "Clay", ha=ha, va="center", fontsize=5.4, color=C["img"])
        fig.text(lx, yc + chh / 2, "S2", ha=ha, va="center", fontsize=5.4, color=C["img"])
    centers["img"] = ((yc + chh / 2 + ys_ + 0.008) / 2, yc + chh / 2, ys_ + 0.008)
    y = ys_ - G_GAP + 0.008

    # location: SINR encoder of the tower's coordinates
    title("loc", f"LOCATION, {LAT:.2f}° N, {-LON:.2f}° W", y)
    land = json.load(open(Path(__file__).parent / "static_data" / "ne_110m_land.geojson"))
    mh = 0.03; mw = mh / ASPECT * 360 / 145
    y0 = y - 0.008 - mh
    ax = fig.add_axes([L0, y0, mw, mh])
    for feat in land["features"]:
        g = feat["geometry"]
        for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            xy = np.array(poly[0]); ax.fill(xy[:, 0], xy[:, 1], color="#dde2ea", lw=0)
    ax.plot(LON, LAT, "o", ms=3, color=C["loc"]); ax.set_xlim(-180, 180); ax.set_ylim(-60, 85); ax.axis("off")
    centers["loc"] = (y0 + mh / 2,)
    bottom = y0

    # ---------------------------------------------------------------- b: the network
    for k in ("flux", "met", "era", "img"):
        yk = centers[k][0]
        arrow(fig, (L1 + 0.004, yk), (XE, yk), C[k])
        tri(fig, XE, yk, 0.022, 0.03, C[k], "e")
        arrow(fig, (XE + 0.022, yk), (XS - 0.0065, yk), C[k])
        circle(fig, XS, yk, 0.0065, fill=False, ec=INK, lw=0.7)
        fig.text(XS, yk, "+", ha="center", va="center", fontsize=6, color=INK)
        arrow(fig, (XS + 0.0065, yk), (XT0, yk), INK)
    for k in ("flux", "met", "era", "img"):                 # brackets from each group's rows to its encoder
        if len(centers[k]) == 3:
            line(fig, [L1 + 0.004] * 2, [centers[k][1], centers[k][2]], C[k], lw=0.6)
    xm = L1 + 0.009                                          # site memory joins the tower tokens before their encoders
    ym = centers["mem"][0]
    line(fig, [L1 + 0.004, xm], [ym, ym], C["mem"], lw=0.7)
    line(fig, [xm, xm], [ym, centers["flux"][0]], C["mem"], lw=0.7)
    for k in ("flux", "met"):
        circle(fig, xm, centers[k][0], 0.0018, fc=C["mem"], ec="none")
    yl = centers["loc"][0]
    tri(fig, XE, yl, 0.022, 0.03, C["loc"], "s", size=6.5)
    arrow(fig, (L0 + mw + 0.012, yl), (XE, yl), C["loc"])
    line(fig, [XE + 0.022, XS], [yl, yl], C["loc"])
    line(fig, [XS, XS], [yl, centers["img"][0] - 0.0065 * ASPECT], C["loc"])
    for k1, k2 in (("img", "era"), ("era", "met"), ("met", "flux")):
        line(fig, [XS, XS], [centers[k1][0] + 0.0065 * ASPECT, centers[k2][0] - 0.0065 * ASPECT], C["loc"])
    fig.text(XE - 0.008, yl + 0.004, "SINR", fontsize=5.2, color=C["loc"], ha="right", va="bottom")

    y_hi, y_lo = top - 0.022, bottom
    box(fig, XT0, y_lo, XT1 - XT0, y_hi - y_lo, "white", INK, lw=1.1, r=0.016)
    gw, gh = 0.112, 0.085
    gx = XC - gw / 2
    sp = y_hi - y_lo
    gy1 = y_hi - 0.11 * sp - gh
    gy2 = gy1 - 0.34 * sp + gh - gh
    gy3 = gy2 - 0.21 * sp
    xl, yt = XT1 - 0.0075, y_hi - 0.022                       # the block repeats: MLP back to the top, inside the box
    fig.text(XC, gy1 + gh + 0.007, "time attention", fontsize=5.8, ha="center", fontweight="bold", color=INK)
    grid_icon(fig, gx, gy1, gw, gh, 8, 16, hl_row=2, color=C["flux"])
    arrow(fig, (XC, gy1 - 0.007), (XC, gy2 + gh + 0.02), INK, lw=0.7)
    fig.text(XC, gy2 + gh + 0.007, "variable attention", fontsize=5.8, ha="center", fontweight="bold", color=INK)
    grid_icon(fig, gx, gy2, gw, gh, 8, 16, hl_col=7, color=C["met"], img_row=True)
    arrow(fig, (XC, gy2 - 0.007), (XC, gy3 + 0.016), INK, lw=0.7)
    box(fig, XC - 0.03, gy3, 0.06, 0.014, "#f2f2f2", INK, lw=0.6, r=0.004)
    fig.text(XC, gy3 + 0.007, "MLP", fontsize=5.8, ha="center", va="center", color=INK)
    line(fig, [XC + 0.033, xl], [gy3 + 0.007] * 2, INK, lw=0.7)
    line(fig, [xl, xl], [gy3 + 0.007, yt], INK, lw=0.7)
    line(fig, [xl, XC], [yt, yt], INK, lw=0.7)
    arrow(fig, (XC, yt), (XC, gy1 + gh + 0.02), INK, lw=0.7)
    fig.text(XC + 0.018, (gy2 + gy3 + 0.016) / 2, "×10", fontsize=7, ha="left", va="center", fontweight="bold", color=INK)
    fig.text(XC, y_lo + 0.03, "28.1 M parameters", fontsize=5.2, ha="center", color=GREY)
    fig.text(XC, y_lo + 0.012, r"$\theta$", fontsize=8, ha="center", color=C["adapt"])
    # heads
    for k in ("flux", "met", "era", "img"):
        yk = centers[k][0]
        arrow(fig, (XT1, yk), (XD, yk), INK)
        tri(fig, XD, yk, 0.022, 0.03, C[k], "d", right=False)
        arrow(fig, (XD + 0.022, yk), (R0 - 0.004, yk), C[k])
        if len(centers[k]) == 3:
            line(fig, [R0 - 0.004] * 2, [centers[k][1], centers[k][2]], C[k], lw=0.6)

    # ---------------------------------------------------------------- site adaptation: unrolled recurrence over the weights
    yb1 = bottom - 0.03
    yb0 = yb1 - 0.072
    box(fig, L0, yb0, R1 - L0, yb1 - yb0, "#fbf8f1", C["adapt"], lw=0.8, r=0.008)
    fig.text(L0 + 0.01, yb0 + 0.011, "SITE ADAPTATION, 300 STEPS PER WINDOW", color=C["adapt"], **MONO, va="center")
    fig.text(R1 - 0.01, yb0 + 0.011, "each step: a new mask over visible values and other years", fontsize=5.2, ha="right",
             va="center", color=GREY)
    cells = [r"$\theta^\star$", r"$\theta_1$", r"$\theta_2$", r"$\cdots$", r"$\theta_{299}$", r"$\theta_w$"]
    xs_c = np.linspace(L0 + 0.06, R1 - 0.06, len(cells))
    yc_ = yb0 + 0.041
    rc = 0.016
    for i, lab in enumerate(cells):
        if lab != r"$\cdots$":
            circle(fig, xs_c[i], yc_, rc, fc="white", ec=C["adapt"], lw=0.9)
        fig.text(xs_c[i], yc_, lab, fontsize=6.2, ha="center", va="center", color=C["adapt"])
        if i < len(cells) - 1:
            x0, x1 = xs_c[i] + rc + 0.003, xs_c[i + 1] - rc - 0.003
            arrow(fig, (x0, yc_), (x1, yc_), C["adapt"], lw=0.9)
            if r"$\cdots$" not in (lab, cells[i + 1]):
                fig.text((x0 + x1) / 2, yc_ + 0.004, r"$-\eta\nabla_\theta\mathcal{L}_w$", fontsize=5.4, ha="center",
                         va="bottom", color=C["adapt"])
    yr = (y_lo + yb1) / 2                                    # weights in and out of the network, on one routing line
    xa, xb_ = XC - 0.035, XC + 0.035
    line(fig, [xa, xa], [y_lo, yr], C["adapt"], lw=0.9)
    line(fig, [xs_c[0], xa], [yr, yr], C["adapt"], lw=0.9)
    arrow(fig, (xs_c[0], yr), (xs_c[0], yc_ + rc * ASPECT), C["adapt"], lw=0.9)
    line(fig, [xs_c[-1], xs_c[-1]], [yc_ + rc * ASPECT, yr], C["adapt"], lw=1.2)
    line(fig, [xb_, xs_c[-1]], [yr, yr], C["adapt"], lw=1.2)
    arrow(fig, (xb_, yr), (xb_, y_lo), C["adapt"], lw=1.2)

    # ---------------------------------------------------------------- d: inside the outage, half-hourly
    i_n, i_c = keys.index("NEE"), keys.index("FCH4")
    a, b = spans["flux"]
    yn, ycf = np.isfinite(Y[i_n]), np.isfinite(Y[i_c])
    best = max(range(0, (b - a) // 48 - 2), key=lambda d: min(yn[a + d * 48:a + (d + 3) * 48].sum(), ycf[a + d * 48:a + (d + 3) * 48].sum()))
    z0 = a + best * 48; z1 = z0 + 3 * 48
    tt = np.arange(8640) / 48
    yd1 = yb0 - 0.05
    yd0 = yd1 - 0.085
    pw = (R1 - L0 - 2 * 0.05) / 3
    for i, k in enumerate(("NEE", "FCH4")):
        ax = fig.add_axes([L0 + i * (pw + 0.05), yd0, pw, yd1 - yd0])
        s = sm["samples"][:, skeys.index(k)]
        qs = np.nanquantile(s, [0.025, 0.975], axis=0)
        ax.fill_between(tt[z0:z1], qs[0, z0:z1], qs[1, z0:z1], color=C["flux"], alpha=0.18, lw=0)
        tm = top_method(wc, k)
        ax.plot(tt[z0:z1], pred(wc, tm, k)["mean"][z0:z1], color=TOP_C, lw=0.8)
        ax.plot(tt[z0:z1], np.nanmean(s, 0)[z0:z1], color=C["flux"], lw=1.1)
        ax.plot(tt[z0:z1], Y[keys.index(k)][z0:z1], ".", color="black", ms=2, mew=0)
        unit = r"µmol m$^{-2}$ s$^{-1}$" if k == "NEE" else r"nmol m$^{-2}$ s$^{-1}$"
        ax.set_ylabel(f"{LAB[k]} ({unit})", fontsize=6); ax.tick_params(labelsize=5.6, length=2)
        ax.set_xlabel("day of window", fontsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    k = "FCH4"
    y = Y[i_c]; s = sm["samples"][:, skeys.index(k)]
    cand = [t for t in range(z0, z1) if np.isfinite(y[t])]
    t_star = cand[len(cand) // 2]
    ax = fig.add_axes([L0 + 2 * (pw + 0.05), yd0, pw, yd1 - yd0])
    x = s[:, t_star]
    lo, hi = np.percentile(x, [0.5, 99.5]); sp = hi - lo
    grid = np.linspace(lo - 0.4 * sp, hi + 0.4 * sp, 300)
    bw = 1.06 * x.std() * len(x) ** -0.2
    dens = np.exp(-0.5 * ((grid[:, None] - x[None]) / bw) ** 2).sum(1) / (len(x) * bw * np.sqrt(2 * np.pi))
    ax.fill_between(grid, dens, color=C["flux"], alpha=0.18, lw=0); ax.plot(grid, dens, color=C["flux"], lw=1.0)
    ax.axvline(y[t_star], color="black", lw=0.9)
    tm = top_method(wc, k); q = pred(wc, tm, k)
    yy = -dens.max() * 0.12
    ax.plot([q["q025"][t_star], q["q975"][t_star]], [yy, yy], color=TOP_C, lw=1.0)
    ax.plot(q["mean"][t_star], yy, "o", color=TOP_C, ms=2.5)
    ax.set_yticks([]); ax.spines[["left", "top", "right"]].set_visible(False); ax.tick_params(labelsize=5.6, length=2)
    ax.set_xlabel(r"FCH$_4$ at one hidden half-hour", fontsize=6)
    names = sorted({NAMES[top_method(wc, v)] for v in ("NEE", "FCH4")})
    hand = [Line2D([], [], color=C["flux"], lw=1.1, label="FluxFill mean"),
            Patch(color=C["flux"], alpha=0.18, lw=0, label="FluxFill 95% interval"),
            Line2D([], [], color=TOP_C, lw=0.8, label=" / ".join(names)),
            Line2D([], [], color="black", lw=0, marker=".", ms=3, label="measured")]
    fig.legend(handles=hand, loc="lower left", bbox_to_anchor=(L0 - 0.006, yd1 + 0.004), ncol=4, fontsize=5.2, frameon=False,
               handlelength=1.4, columnspacing=1.2, handletextpad=0.5)
    vals["fig1_t_star_day"] = f"{t_star / 48:.1f}"
    vals["fig1_top_d"] = " and ".join(names)

    # ---------------------------------------------------------------- e: every method, test towers
    tab = vals.get("fig1_table", {})
    y0 = yd0 - 0.052
    cols = (("NEE", 0.58), ("LE", 0.68), ("H", 0.78), ("FCH4", 0.88))
    for lab, x, ha in [("Method", L0, "left"), ("Model", 0.25, "left")] + [(LAB[v], x, "right") for v, x in cols]:
        fig.text(x, y0, lab, fontsize=5.6, fontweight="bold", ha=ha, color=INK)
    fig.text(0.715, y0 + 0.012, "test CRPS", fontsize=5.4, ha="center", color=GREY)
    line(fig, [0.55, 0.88], [y0 + 0.009] * 2, "#c5c9cf", lw=0.5)
    line(fig, [L0, R1], [y0 - 0.004] * 2, "#999999", lw=0.5)
    meth = [("climatology", "Climatology", "other years, same season"),
            ("mds", "MDS", "look-up of similar meteorology"),
            ("irvin_rf", "RF ensemble", "random forests per tower"),
            ("liu_xgb", "XGBoost", "boosted trees per tower"),
            ("vekuri_de", "Deep ensemble", "neural networks per tower"),
            ("saits", "SAITS", "self-attention imputer"),
            ("csdi", "CSDI", "score-based diffusion"),
            ("chronos2", "Chronos-2", "pretrained forecaster"),
            ("fluxfill", "FluxFill", "DeepEarth + site adaptation")]
    best = {v: min((tab[m][v] for m in tab if v in tab[m]), default=None) for v in ("NEE", "LE", "H", "FCH4")}
    for i, (m, name, model) in enumerate(meth):
        yy = y0 - 0.0125 - i * 0.0105
        if m == "fluxfill":
            fig.add_artist(Rectangle((L0 - 0.004, yy - 0.003), R1 - L0 + 0.008, 0.0098, fc=SHADE, ec="none", transform=fig.transFigure))
        fig.text(L0, yy, name, fontsize=5.4, fontweight="bold" if m == "fluxfill" else "normal", color=INK)
        fig.text(0.25, yy, model, fontsize=5.2, color=INK)
        if m not in tab:
            fig.text(0.88, yy, "pending", fontsize=5.0, ha="right", color="#9aa0a6")
            continue
        for v, x in cols:
            val = tab[m].get(v)
            if val is None:
                continue
            fig.text(x, yy, f"{val:.3f}" if val < 10 else f"{val:.2f}", fontsize=5.2, ha="right", color=INK,
                     fontweight="bold" if np.isclose(val, best[v]) else "normal")
    for lab, yl_ in (("d", yd1 + 0.012), ("e", y0)):
        fig.text(0.02, yl_, lab, fontsize=10, fontweight="bold", va="center")
    with plt.rc_context({"svg.fonttype": "path"}):            # glyphs as paths: mathtext renders in every browser
        fig.savefig(out / "fig0.svg", facecolor="white", bbox_inches="tight", pad_inches=0.05, dpi=300)
        (out / "preview").mkdir(exist_ok=True)
        fig.savefig(out / "preview" / "fig0.png", facecolor="white", dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    vals["fig1_acquisitions"] = int(len(f["img_days"]))
    vals["fig1_img_hidden"] = int(f["img_hidden"].sum())
    vals["fig1_img_cos"] = f"{float(np.mean(json.load(open(REP / 'fig1_all.json'))['img_cos'])):.2f}"
    days, hid = f["img_days"], f["img_hidden"]            # the same score for the nearest visible acquisition in time
    near = [float(ln[h] @ ln[v] / np.linalg.norm(ln[h]) / np.linalg.norm(ln[v]))
            for h in np.flatnonzero(hid) for v in [np.flatnonzero(~hid)[np.argmin(np.abs(days[~hid] - days[h]))]]]
    vals["fig1_img_cos_near"] = f"{np.mean(near):.2f}"
