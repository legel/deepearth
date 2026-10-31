"""Figure 1 of the FluxFill report: one real validation window (US-YK2, from 2019-07-15) with its 30-day station
outage (mask family M2), the network, the reconstruction with its predictive distribution, the established methods,
and the eddy-covariance equations that define the fluxes.

Inputs: the window export of the final FluxFill (bench/dashboard/report/fig1_window), its raw samples for the 30-day
mask (bench/dashboard/report/samples), the established methods' fills of the same window and mask
(bench/dashboard/windows), the Sentinel-2 and Clay stores of the tower, and its half-hourly table.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

R = Path("/home/photon/4tb/fluxfill")
SITE, START, KEY = "US-YK2", "2019-07-15", "US-YK2_2019-07-15"
MASK = "M2: 30 d, all tower channels"
C = {"flux": "#1d4466", "met": "#6b6f2e", "era": "#8a9a3a", "img": "#6e1f3f", "mem": "#1e2a35"}
COMP = {"climatology": ("#9a9a9a", "Climatology"), "mds": ("#2a78d6", "MDS"), "irvin_rf": ("#1baf7a", "RF ensemble"),
        "liu_xgb": ("#eb6834", "XGBoost"), "saits": ("#c2416b", "SAITS")}
LAB = {"NEE": "NEE", "LE": "LE", "H": "H", "FCH4": r"FCH$_4$", "TA": "TA", "SW_IN": r"SW$_{in}$", "VPD": "VPD",
       "SWC_1": "SWC", "ERA5_TA": "TA", "ERA5_SW_IN": r"SW$_{in}$", "ERA5_VPD": "VPD"}
MONO = {"family": "DejaVu Sans Mono", "size": 6.3, "weight": "bold"}


def series(w, key):
    keys = [v["key"] for v in w["variables"]]
    return np.array([np.nan if x is None else x for x in w["variables"][keys.index(key)]["values"]], float)


def pred(w, method, key):
    seg = w["predictions"].get(method, {}).get(MASK, {}).get(key)
    if not seg:
        return None
    s = seg[0]
    out = {k: np.full(8640, np.nan) for k in ("mean", "q025", "q10", "q90", "q975")}
    for k in out:
        out[k][s["a"]:s["a"] + len(s[k])] = s[k]
    return out


def daily(y):
    with np.errstate(all="ignore"):
        return np.nanmean(y.reshape(-1, 48), 1)


def mini(fig, rect, color, y, a, b, label, hide=True, show_y=False):
    ax = fig.add_axes(rect)
    d = daily(y)
    ax.plot(np.arange(180) + 0.5, d, color=color, lw=0.7)
    if hide:
        lo, hi = np.nanmin(d), np.nanmax(d)
        pad = 0.08 * (hi - lo if hi > lo else 1)
        ax.add_patch(Rectangle((a / 48, lo - pad), (b - a) / 48, hi - lo + 2 * pad, color="black", zorder=4, lw=0))
        ax.text((a + b) / 96, (lo + hi) / 2, "?", color="white", ha="center", va="center", fontsize=7, fontweight="bold", zorder=5)
    ax.set_xlim(0, 180)
    ax.axis("off")
    ax.text(-3, np.nanmean(d) if np.isfinite(np.nanmean(d)) else 0, label, ha="right", va="center", fontsize=5.6, color=color)
    return ax


def tri(fig, x, y, w, h, color, text, right=True):
    """Outlined triangle: an encoder (base left, apex right) or a decoder (apex left, base right)."""
    pts = [(x, y - h / 2), (x, y + h / 2), (x + w, y)] if right else [(x + w, y - h / 2), (x + w, y + h / 2), (x, y)]
    fig.add_artist(Polygon(pts, closed=True, fill=False, ec=color, lw=1.1, transform=fig.transFigure))
    fig.text(x + (w * 0.36 if right else w * 0.62), y, text, color=color, ha="center", va="center", fontsize=7.5,
             fontstyle="italic", fontweight="bold", transform=fig.transFigure)


def arrow(fig, p0, p1, color, ls="-", lw=0.9, cs="arc3"):
    fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=6, color=color,
                                   lw=lw, ls=ls, connectionstyle=cs))


def rlabel(fig, x, y, text, color):
    fig.text(x, y, text, color=color, fontsize=5.4, ha="center", va="bottom", family="DejaVu Sans", transform=fig.transFigure)


def chips():
    """Clear Sentinel-2 acquisitions inside the window (true colour, 10 m) and the Clay class tokens of all of them."""
    g = zarr.open(str(R / "sentinel2" / "sites" / f"{SITE}.zarr"), mode="r")
    t = np.array(g["time_us"][:]).astype("datetime64[us]")
    t0, t1 = np.datetime64(START), np.datetime64(START) + np.timedelta64(180, "D")
    sel = np.flatnonzero((t >= t0) & (t < t1))
    scl = np.array(g["scl"].get_orthogonal_selection((sel, slice(None), slice(None))))
    clear = np.isin(scl, [4, 5, 6, 11]).mean((1, 2))
    days = ((t[sel] - t0) / np.timedelta64(1, "D")).astype(float)
    pick = []
    for target in np.linspace(5, 150, 6):
        cand = [i for i in range(len(sel)) if clear[i] > 0.7 and all(abs(days[i] - days[j]) > 12 for j in pick)]
        if cand:
            pick.append(min(cand, key=lambda i: abs(days[i] - target)))
    pick = sorted(pick, key=lambda i: days[i])
    rgb = []
    for i in pick:
        x = np.array(g["r10"][sel[i]], float)[[2, 1, 0]]
        x = np.clip((x - 1000) / 10000, 0, None) if t[sel[i]] >= np.datetime64("2022-01-25") else x / 10000
        x = np.clip(x / max(np.percentile(x, 98), 1e-3), 0, 1) ** 0.8
        rgb.append(np.transpose(x, (1, 2, 0)))
    c = zarr.open(str(R / "clay" / "sites" / f"{SITE}.zarr"), mode="r")
    tc = np.array(c["time_us"][:]).astype("datetime64[us]")
    cs = np.flatnonzero((tc >= t0) & (tc < t1))
    cls = np.array(c["cls"].get_orthogonal_selection((cs, slice(None))), float)
    cdays = ((tc[cs] - t0) / np.timedelta64(1, "D")).astype(float)
    return [days[i] for i in pick], rgb, cdays, cls


def memory_band():
    """FCH4 of the tower's other years at each day of the window's season: daily mean and standard deviation."""
    t = pd.read_parquet(R / "towers" / "hh" / f"{SITE}.parquet", columns=["time_lst", "FCH4"])
    t["time_lst"] = pd.to_datetime(t["time_lst"])
    t = t[~t.time_lst.dt.year.isin([2019, 2020])].dropna()
    doy0 = pd.Timestamp(START).dayofyear
    t["d"] = (t.time_lst.dt.dayofyear - doy0) % 365
    t = t[t.d < 180]
    g = t.groupby("d").FCH4
    return g.mean().reindex(range(180)).to_numpy(), g.std().reindex(range(180)).to_numpy(), sorted(set(t.time_lst.dt.year))


def figure1(out: Path, vals: dict):
    w = json.load(open(R / "bench" / "dashboard" / "report" / "fig1_window" / f"{KEY}.json"))
    wc = json.load(open(R / "bench" / "dashboard" / "report" / "fig1_window_base" / f"{KEY}.json"))   # established methods, same masks
    sm = np.load(R / "bench" / "dashboard" / "report" / "samples" / f"{KEY}_fluxfill_M2-30d.npz")
    skeys = [str(k) for k in sm["keys"]]
    a, b = w["masks"][MASK]["NEE"][0]
    fig = plt.figure(figsize=(7.2, 8.4))
    heads = [(0.17, "MASKED WINDOW"), (0.345, "TOKEN\nENCODERS"), (0.455, "FACTORISED\nTRANSFORMER"), (0.565, "HEADS"),
             (0.80, "RECONSTRUCTION")]
    for x, h in heads:
        fig.text(x, 0.972, h, ha="center", va="center", **MONO, color="#1e2a35")

    # ---- rows of the window (a) and their reconstruction (c)
    rows = [("flux", "I  TOWER FLUXES", ["NEE", "LE", "H", "FCH4"], True),
            ("met", "II  TOWER METEOROLOGY", ["TA", "SW_IN", "VPD", "SWC_1"], True),
            ("era", "III  REANALYSIS (ERA5)", ["ERA5_TA", "ERA5_SW_IN", "ERA5_VPD"], False)]
    centers = {}
    h_ax, gap = 0.023, 0.005
    top = 0.945
    for kind, title, keys, hidden in rows:
        fig.text(0.075, top + 0.006, title, color=C[kind], **MONO)
        ys = []
        for j, k in enumerate(keys):
            y0 = top - (j + 1) * (h_ax + gap)
            ys.append(y0 + h_ax / 2)
            mini(fig, [0.105, y0, 0.215, h_ax], C[kind], series(w, k), a, b, LAB[k], hide=hidden)
            if hidden:                                                              # reconstruction column
                ax = fig.add_axes([0.66, y0, 0.315, h_ax])
                y = series(w, k)
                d = daily(y)
                ax.plot(np.arange(180) + 0.5, np.where((np.arange(180) >= a / 48) & (np.arange(180) < b / 48), np.nan, d),
                        color=C[kind], lw=0.7)
                p = pred(w, "fluxfill", k)
                if p is not None:
                    dd = np.arange(180) + 0.5
                    ax.fill_between(dd, daily(p["q025"]), daily(p["q975"]), color=C[kind], alpha=0.18, lw=0)
                    if kind == "flux":
                        for m, (cc, _) in COMP.items():
                            q = pred(wc, m, k)
                            if q is not None:
                                ax.plot(dd, daily(q["mean"]), color=cc, lw=0.45, alpha=0.75)
                    ax.plot(dd, daily(p["mean"]), color=C[kind], lw=1.1)
                    with np.errstate(all="ignore"):
                        ax.plot(dd, np.where((dd >= a / 48) & (dd < b / 48), d, np.nan), color="#999999", lw=0.6, ls=(0, (1.2, 0.8)))
                ax.set_xlim(max(0, a / 48 - 20), min(180, b / 48 + 20)); ax.axis("off")
                ax.axvspan(a / 48, b / 48, color="#f3f1e7", lw=0, zorder=-1)
                if np.isfinite(d).any():
                    ax.text(max(0, a / 48 - 20) - 1, np.nanmean(d), LAB[k], ha="right", va="center", fontsize=5.6, color=C[kind])
        centers[kind] = np.mean(ys)
        top = top - len(keys) * (h_ax + gap) - 0.028

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hand = [Line2D([], [], color=C["flux"], lw=1.1, label="FluxFill mean"), Patch(color=C["flux"], alpha=0.18, label="FluxFill 95% interval"),
            Line2D([], [], color="#999999", lw=0.6, ls=(0, (1.2, 0.8)), label="Measured, hidden")] + \
           [Line2D([], [], color=cc, lw=0.6, label=nm) for cc, nm in COMP.values()]
    fig.legend(handles=hand, loc="center left", bbox_to_anchor=(0.66, centers["era"]), fontsize=5.4, frameon=False, ncol=2,
               handlelength=1.6, title="c: 30-day block ± 20 days", title_fontsize=5.6)

    # Sentinel-2 chips and Clay embeddings (IV)
    fig.text(0.075, top + 0.006, "IV  SENTINEL-2 (300 m)  AND  CLAY EMBEDDINGS", color=C["img"], **MONO)
    cd, rgb, cdays, cls = chips()
    for d, im in zip(cd, rgb):
        ax = fig.add_axes([0.105 + 0.215 * d / 180 - 0.016, top - 0.042, 0.032, 0.036])
        ax.imshow(im, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(C["img"]); s.set_linewidth(0.6)
    ax = fig.add_axes([0.105, top - 0.066, 0.215, 0.019])
    order = np.argsort(cdays)
    z = cls[order][:, :96].T
    z = (z - z.mean(1, keepdims=True)) / (z.std(1, keepdims=True) + 1e-6)
    ax.imshow(z, aspect="auto", cmap="RdPu", extent=[cdays.min(), cdays.max(), 0, 1], vmin=-2, vmax=2, interpolation="nearest")
    ax.set_xlim(0, 180); ax.axis("off")
    fig.text(0.1, top - 0.057, "CLS", ha="right", va="center", fontsize=5.2, color=C["img"])
    centers["img"] = top - 0.035
    fig.text(0.66, centers["img"], f"{len(cdays)} acquisitions in the window, {len(cd)} clear ones shown;\n"
             "an embedding is reconstructed when imagery is hidden (M4, M5)", fontsize=5.4, color=C["img"], va="center")

    # Site memory (V)
    top = top - 0.09
    mu, sd, yrs = memory_band()
    fig.text(0.075, top + 0.006, f"V  SITE MEMORY: OTHER YEARS ({yrs[0]}-{yrs[-1]})", color=C["mem"], **MONO)
    ax = fig.add_axes([0.105, top - 0.042, 0.215, 0.036])
    dd = np.arange(180) + 0.5
    ax.fill_between(dd, mu - sd, mu + sd, color=C["mem"], alpha=0.15, lw=0); ax.plot(dd, mu, color=C["mem"], lw=0.8)
    ax.set_xlim(0, 180); ax.axis("off")
    ax.text(-3, np.nanmean(mu), r"FCH$_4$", ha="right", va="center", fontsize=5.6, color=C["mem"])
    centers["mem"] = top - 0.024

    # ---- network (b): encoders, factorised transformer, heads
    for k in ("flux", "met", "era", "img"):
        y = centers[k]
        arrow(fig, (0.325, y), (0.338, y), C[k])
        tri(fig, 0.34, y, 0.035, 0.05, C[k], "e")
        rlabel(fig, 0.392, y + 0.003, r"$\mathbb{R}^{320}$", C[k])
        arrow(fig, (0.377, y), (0.41, y), C[k])
    # memory: three channels (mean, s.d., count) of every tower token, added in the tower encoders
    xm = 0.331
    fig.add_artist(matplotlib.lines.Line2D([0.325, xm], [centers["mem"], centers["mem"]], color=C["mem"], lw=0.7, ls=(0, (2, 1.5)), transform=fig.transFigure))
    fig.add_artist(matplotlib.lines.Line2D([xm, xm], [centers["mem"], centers["flux"]], color=C["mem"], lw=0.7, ls=(0, (2, 1.5)), transform=fig.transFigure))
    for k in ("flux", "met"):
        arrow(fig, (xm, centers[k] - 0.012), (0.344, centers[k] - 0.012), C["mem"], lw=0.7)
    fig.text(0.105, centers["mem"] - 0.03, r"mean, s.d. and count $(\mu, \sigma, n)$ at each half-hour and season enter every tower token",
             fontsize=5.0, color=C["mem"], ha="left", va="top")
    y_lo, y_hi = centers["mem"] - 0.01, 0.935
    tb = FancyBboxPatch((0.412, y_lo), 0.087, y_hi - y_lo, boxstyle="round,pad=0.004,rounding_size=0.03", fc="white",
                        ec="#1e2a35", lw=1.2, transform=fig.transFigure)
    fig.add_artist(tb)
    for i in range(10):
        y = y_hi - 0.022 - i * (y_hi - y_lo - 0.035) / 10
        fig.add_artist(Rectangle((0.42, y - 0.016), 0.033, 0.03, fc="#eef0e6", ec="#1e2a35", lw=0.5, transform=fig.transFigure))
        fig.add_artist(Rectangle((0.458, y - 0.016), 0.033, 0.03, fc="#e8edf3", ec="#1e2a35", lw=0.5, transform=fig.transFigure))
        if i == 0:
            fig.text(0.4365, y - 0.001, "time", fontsize=4.8, ha="center", va="center")
            fig.text(0.4745, y - 0.001, "var.", fontsize=4.8, ha="center", va="center")
    fig.text(0.4555, y_lo - 0.006, "10 blocks, width 320", fontsize=4.9, ha="center", va="top")
    for k in ("flux", "met", "img"):
        y = centers[k]
        arrow(fig, (0.503, y), (0.53, y), "#1e2a35")
        tri(fig, 0.532, y, 0.036, 0.05, C[k], "d", right=False)
        arrow(fig, (0.57, y), (0.6, y), C[k])
    rlabel(fig, 0.59, centers["flux"] + 0.006, r"$p_\theta(x\,|\,\cdot)$", C["flux"])
    rlabel(fig, 0.59, centers["met"] + 0.006, r"$p_\theta(x\,|\,\cdot)$", C["met"])
    rlabel(fig, 0.59, centers["img"] + 0.006, r"$\mathcal{N}(\mu,\sigma^2)$", C["img"])
    fig.text(0.551, centers["flux"] - 0.034, "Student-t mixture\nper half-hour", fontsize=4.8, ha="center", va="top", color=C["flux"])
    arrow(fig, (0.82, 0.958), (0.47, 0.951), "#6e7355", ls=(0, (3, 2)), lw=0.8, cs="arc3,rad=0.12")
    fig.text(0.645, 0.993, "site adaptation: 300 steps on the visible values and the other years, then reset",
             fontsize=5.2, color="#6e7355", ha="center")

    # ---- (c) inside the gap, half-hourly, with the predictive distribution
    # Zoom: the three days inside the gap with the most measured NEE and FCH4 (the smaller of the two counts).
    yn, yc = np.isfinite(series(w, "NEE")), np.isfinite(series(w, "FCH4"))
    best = max(range(0, (b - a) // 48 - 2), key=lambda d: min(yn[a + d * 48:a + (d + 3) * 48].sum(), yc[a + d * 48:a + (d + 3) * 48].sum()))
    z0 = a + best * 48; z1 = z0 + 3 * 48
    tt = np.arange(8640) / 48
    yb = 0.205
    panels = [("NEE", [0.075, yb, 0.27, 0.15], C["flux"]), ("FCH4", [0.395, yb, 0.27, 0.15], C["flux"])]
    for k, rect, col in panels:
        ax = fig.add_axes(rect)
        y = series(w, k)
        s = sm["samples"][:, skeys.index(k)]
        qs = np.nanquantile(s, [0.025, 0.1, 0.5, 0.9, 0.975], axis=0)
        ax.fill_between(tt[z0:z1], qs[0, z0:z1], qs[4, z0:z1], color=col, alpha=0.13, lw=0, label="FluxFill 95%")
        ax.fill_between(tt[z0:z1], qs[1, z0:z1], qs[3, z0:z1], color=col, alpha=0.22, lw=0, label="FluxFill 80%")
        for m, (cc, nm) in COMP.items():
            q = pred(wc, m, k)
            if q is not None:
                ax.plot(tt[z0:z1], q["mean"][z0:z1], color=cc, lw=0.5, alpha=0.8, label=nm)
        ax.plot(tt[z0:z1], np.nanmean(s, 0)[z0:z1], color=col, lw=1.2, label="FluxFill mean")
        ax.plot(tt[z0:z1], y[z0:z1], ".", color="black", ms=1.8, label="Measured (hidden)")
        unit = r"µmol m$^{-2}$ s$^{-1}$" if k == "NEE" else r"nmol m$^{-2}$ s$^{-1}$"
        ax.set_ylabel(f"{LAB[k]} ({unit})", fontsize=6.5); ax.tick_params(labelsize=6)
        ax.set_xlabel("Day of window", fontsize=6.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if k == "NEE":
            h_, l_ = ax.get_legend_handles_labels()
            fig.legend(h_, l_, loc="lower left", bbox_to_anchor=(0.07, yb + 0.152), ncol=9, fontsize=5.3, frameon=False,
                       handlelength=1.3, columnspacing=0.9)
    # predictive distribution of one half-hour of FCH4 inside the gap: the hidden value with the median sample spread
    k = "FCH4"
    y = series(w, k); s = sm["samples"][:, skeys.index(k)]
    cand = [t for t in range(z0, z1) if np.isfinite(y[t])]
    t_star = cand[len(cand) // 2]
    ax = fig.add_axes([0.73, yb, 0.245, 0.15])
    x = s[:, t_star]
    lo, hi = np.percentile(x, [0.5, 99.5]); span = hi - lo
    grid = np.linspace(lo - 0.3 * span, hi + 0.3 * span, 300)
    bw = 1.06 * x.std() * len(x) ** -0.2
    dens = np.exp(-0.5 * ((grid[:, None] - x[None]) / bw) ** 2).sum(1) / (len(x) * bw * np.sqrt(2 * np.pi))
    ax.fill_between(grid, dens, color=C["flux"], alpha=0.25, lw=0); ax.plot(grid, dens, color=C["flux"], lw=1.0)
    ax.axvline(y[t_star], color="black", lw=0.9)
    ytop = dens.max()
    for i, (m, (cc, nm)) in enumerate(COMP.items()):
        q = pred(wc, m, k)
        if q is None or not np.isfinite(q["mean"][t_star]):
            continue
        yy = -ytop * (0.12 + 0.1 * i)
        ax.plot([q["q025"][t_star], q["q975"][t_star]], [yy, yy], color=cc, lw=0.7, alpha=0.85)
        ax.plot(q["mean"][t_star], yy, "o", color=cc, ms=2)
        ax.text(-0.02, yy, nm, fontsize=4.8, color=cc, va="center", ha="right", transform=ax.get_yaxis_transform())
    ax.set_yticks([]); ax.spines[["left", "top", "right"]].set_visible(False); ax.tick_params(labelsize=6)
    ax.set_xlabel(r"FCH$_4$ (nmol m$^{-2}$ s$^{-1}$), one hidden half-hour", fontsize=6.5)
    ax.text(y[t_star], ytop * 1.03, "measured", fontsize=5.2, ha="center")
    vals["fig1_t_star_day"] = f"{t_star / 48:.1f}"

    # ---- (d) eddy covariance: the fluxes from the raw 10-20 Hz signals
    fig.add_artist(FancyBboxPatch((0.075, 0.008), 0.9, 0.118, boxstyle="round,pad=0.004,rounding_size=0.01", fc="#f7f7f4",
                                  ec="#d8d8d0", lw=0.6, transform=fig.transFigure))
    fig.text(0.09, 0.113, "EDDY COVARIANCE: HALF-HOURLY FLUXES FROM 10-20 Hz WIND AND GAS SIGNALS", **MONO, color="#1e2a35")
    eqs = [r"$x' = x - \overline{x}\quad(\overline{x}$: 30-min mean$)$",
           r"$F_c = \overline{\rho}_d\,\overline{w' s_c'}$",
           r"$\mathrm{NEE} = F_c + \int_0^{z_m} \frac{\partial \overline{c}}{\partial t}\,dz$",
           r"$\mathrm{LE} = \lambda\,\overline{\rho}_d\,\overline{w' s_v'}$",
           r"$H = \overline{\rho}_a\, c_p\, \overline{w' T_s'}$",
           r"$F_{\mathrm{CH_4}} = \overline{\rho}_d\,\overline{w' s_m'}$"]
    for i, e in enumerate(eqs):
        fig.text(0.09 + (i % 3) * 0.3, 0.088 - (i // 3) * 0.03, e, fontsize=7.6, color="#1e2a35", va="center")
    fig.text(0.09, 0.013, r"$w$: vertical wind (sonic anemometer); $s_c, s_v, s_m$: dry mole fractions of CO$_2$, H$_2$O and CH$_4$ "
             r"(gas analyser); $T_s$: sonic temperature;" "\n" r"$\overline{\rho}_d, \overline{\rho}_a$: dry-air and air density; "
             r"$\lambda$: latent heat of vaporisation; $c_p$: specific heat of air; $c$: CO$_2$ concentration; $z_m$: measurement height.",
             fontsize=5.3, color="#555555", linespacing=1.5, va="bottom")
    for lab, (x, y) in {"a": (0.012, 0.985), "b": (0.3, 0.985), "c": (0.62, 0.985), "d": (0.012, yb + 0.175), "e": (0.012, 0.12)}.items():
        fig.text(x, y, lab, fontsize=10, fontweight="bold", transform=fig.transFigure)
    with plt.rc_context({"svg.fonttype": "path"}):            # glyphs as paths: mathtext primes render in every browser
        fig.savefig(out / "fig0.svg", facecolor="white")
    plt.close(fig)
    vals["fig1_acquisitions"] = len(cdays)
