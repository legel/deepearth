"""Figure 1 of the FluxFill report: one real validation window (US-YK2, from 2019-07-15) with its 30-day station
outage (mask family M2), the network, test-time site adaptation drawn as an unrolled recurrence over the weights,
the reconstruction, the predictive distribution against the best established method on this window, and the table
of all methods.

Inputs: the window export of the final FluxFill (bench/dashboard/report/fig1_window), its raw samples for the 30-day
mask (bench/dashboard/report/samples), the established methods' fills of the same window and masks
(bench/dashboard/report/fig1_window_base), the Sentinel-2 and Clay stores of the tower, its half-hourly table, and
the test-tower CRPS of every method (`vals["fig1_table"]`, from report_figures.py).

Time series are daily means on blocks aligned to the start of the hidden span, so every hidden block is exactly
hidden and every visible block exactly visible.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Polygon, Rectangle

R = Path("/home/photon/4tb/fluxfill")
SITE, START, KEY, LAT, LON = "US-YK2", "2019-07-15", "US-YK2_2019-07-15", 61.2548, -163.259
MASK = "M2: 30 d, all tower channels"
C = {"flux": "#1d4466", "met": "#6b6f2e", "era": "#8a9a3a", "img": "#6e1f3f", "mem": "#1e2a35", "loc": "#4a5a7a",
     "adapt": "#8a5a00"}
INK = "#1e2a35"
TOP_C = "#eb6834"                      # the best established method on this window
NAMES = {"climatology": "Climatology", "mds": "MDS", "irvin_rf": "RF ensemble", "liu_xgb": "XGBoost", "saits": "SAITS"}
LAB = {"NEE": "NEE", "LE": "LE", "H": "H", "FCH4": r"FCH$_4$", "TA": "TA", "SW_IN": r"SW$_{in}$", "VPD": "VPD",
       "SWC_1": "SWC", "ERA5_TA": "TA", "ERA5_SW_IN": r"SW$_{in}$", "ERA5_VPD": "VPD"}
MONO = {"family": "DejaVu Sans Mono", "size": 6.2, "weight": "bold"}
B6 = 12                                # half-hours per block of the drawn series


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


def blocks(y, a, L=B6):
    """Means over blocks of L half-hours that start at a (mod L); returns block centres in days and the means."""
    off = a % L
    n = (len(y) - off) // L
    z = y[off:off + n * L].reshape(n, L)
    with np.errstate(all="ignore"):
        m = np.nanmean(z, 1)
    return (off + (np.arange(n) + 0.5) * L) / 48, m


def tri(fig, x, y, w, h, color, text, right=True, size=7.5):
    """Outlined triangle: an encoder (base left, apex right) or a decoder (apex left, base right)."""
    pts = [(x, y - h / 2), (x, y + h / 2), (x + w, y)] if right else [(x + w, y - h / 2), (x + w, y + h / 2), (x, y)]
    fig.add_artist(Polygon(pts, closed=True, fill=False, ec=color, lw=1.1, transform=fig.transFigure))
    fig.text(x + (w * 0.36 if right else w * 0.64), y, text, color=color, ha="center", va="center", fontsize=size,
             fontstyle="italic", fontweight="bold")


def arrow(fig, p0, p1, color, lw=0.9, ls="-", rad=0.0, scale=6):
    fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=scale, color=color,
                                   lw=lw, ls=ls, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def line(fig, xs, ys, color, lw=0.8, ls="-"):
    fig.add_artist(Line2D(xs, ys, color=color, lw=lw, ls=ls, transform=fig.transFigure))


def box(fig, x, y, w, h, fc, ec, lw=0.8, r=0.01):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.003,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                  transform=fig.transFigure))


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


def top_method(wc, key):
    """The established method with the lowest CRPS on this window's 30-day mask for one variable."""
    sc = {m: wc["scores"][m].get(MASK, {}).get(key, {}).get("crps") for m in wc["scores"]}
    sc = {m: v for m, v in sc.items() if v is not None}
    return min(sc, key=sc.get) if sc else None


def grid_icon(fig, x, y, w, h, nv, nd, hl_row=None, hl_col=None, color="#1d4466", img_row=False):
    """A token grid (variables x days); one row (time attention) or one column (variable attention) highlighted."""
    n = nv + int(img_row)
    cw, ch = w / nd, h / n
    for i in range(n):
        for j in range(nd):
            on = (hl_row is not None and i == hl_row) or (hl_col is not None and j == hl_col)
            fc = color if on else ("#efe6ea" if img_row and i == nv else "#e9ecf1")
            fig.add_artist(Rectangle((x + j * cw, y + h - (i + 1) * ch), cw * 0.84, ch * 0.78, fc=fc, ec="none",
                                     transform=fig.transFigure))


def figure1(out: Path, vals: dict):
    w = json.load(open(R / "bench" / "dashboard" / "report" / "fig1_window" / f"{KEY}.json"))
    wc = json.load(open(R / "bench" / "dashboard" / "report" / "fig1_window_base" / f"{KEY}.json"))
    sm = np.load(R / "bench" / "dashboard" / "report" / "samples" / f"{KEY}_fluxfill_M2-30d.npz")
    skeys = [str(k) for k in sm["keys"]]
    a, b = w["masks"][MASK]["NEE"][0]
    fig = plt.figure(figsize=(7.2, 10.4))
    XT0, XT1 = 0.425, 0.585                                   # the transformer, centred on the figure
    XC = (XT0 + XT1) / 2
    for x, h in [(0.19, "MASKED WINDOW"), (0.355, "ENCODERS"), (XC, "DEEPEARTH"), (0.637, "HEADS"),
                 (0.845, "RECONSTRUCTION")]:
        fig.text(x, 0.977, h, ha="center", va="center", **MONO, color=INK)

    # ---------------------------------------------------------------- a: the window, c: its reconstruction
    rows = [("flux", "I  TOWER FLUXES", ["NEE", "LE", "H", "FCH4"], True),
            ("met", "II  TOWER METEOROLOGY", ["TA", "SW_IN", "VPD", "SWC_1"], True),
            ("era", "III  REANALYSIS (ERA5)", ["ERA5_TA", "ERA5_SW_IN", "ERA5_VPD"], False)]
    centers, tops, h_ax, gap, top = {}, {}, 0.0175, 0.004, 0.952
    z_lo, z_hi = max(0, a / 48 - 15), min(180, b / 48 + 15)
    xs = None
    for kind, title, keys, hidden in rows:
        fig.text(0.07, top + 0.004, title, color=C[kind], **MONO)
        tops[kind] = top
        ys = []
        for j, k in enumerate(keys):
            y0 = top - (j + 1) * (h_ax + gap)
            ys.append(y0 + h_ax / 2)
            y = series(w, k)
            yv = y.copy()
            if hidden:
                yv[a:b] = np.nan
            ax = fig.add_axes([0.1, y0, 0.215, h_ax])
            xb, mb = blocks(yv, a, 48)                                  # daily blocks aligned to the hidden span
            ax.plot(xb, mb, color=C[kind], lw=0.6)
            ax.set_xlim(0, 180); ax.axis("off")
            if hidden:
                ax.axvspan(a / 48, b / 48, color="black", lw=0)
                ax.text((a + b) / 96, 0.5, "?", color="white", ha="center", va="center", fontsize=6, fontweight="bold",
                        transform=ax.get_xaxis_transform())
            ax.text(-0.015, 0.5, LAB[k], ha="right", va="center", fontsize=5.4, color=C[kind], transform=ax.transAxes)
            if hidden:                                                              # c: reconstruction, zoomed
                ax = fig.add_axes([0.725, y0, 0.2, h_ax])
                ax.axvspan(a / 48, b / 48, color="#f3f1e7", lw=0, zorder=0)
                xo, mo = blocks(yv, a, 48)
                ax.plot(xo, mo, color="#666666", lw=0.6)
                _, mt = blocks(np.where((np.arange(8640) >= a) & (np.arange(8640) < b), y, np.nan), a, 48)
                ax.plot(xo, mt, color="black", lw=0, marker=".", ms=1.4)
                p = pred(w, "fluxfill", k)
                if p is not None:
                    xs, lo = blocks(p["q025"], a, 48); _, hi = blocks(p["q975"], a, 48); _, mu = blocks(p["mean"], a, 48)
                    ax.fill_between(xs, lo, hi, color=C[kind], alpha=0.2, lw=0)
                    ax.plot(xs, mu, color=C[kind], lw=1.0)
                tm = top_method(wc, k) if kind == "flux" else None
                if tm:
                    _, mt2 = blocks(pred(wc, tm, k)["mean"], a, 48)
                    ax.plot(xo, mt2, color=TOP_C, lw=0.6)
                    ax.text(1.01, 0.5, NAMES[tm], fontsize=4.7, color=TOP_C, va="center", transform=ax.transAxes)
                ax.set_xlim(z_lo, z_hi); ax.axis("off")
                ax.text(-0.01, 0.5, LAB[k], ha="right", va="center", fontsize=5.4, color=C[kind], transform=ax.transAxes)
        centers[kind] = np.mean(ys)
        top = top - len(keys) * (h_ax + gap) - 0.022
    hand = [Line2D([], [], color=C["flux"], lw=1.0, label="FluxFill mean"), Patch(color=C["flux"], alpha=0.2, label="FluxFill 95% interval"),
            Line2D([], [], color="black", lw=0, marker=".", ms=3, label="Measured, hidden"),
            Line2D([], [], color="#666666", lw=0.6, label="Measured, visible"),
            Line2D([], [], color=TOP_C, lw=0.6, label="Best established method\non this window (named)")]
    fig.legend(handles=hand, loc="upper left", bbox_to_anchor=(0.715, tops["era"] + 0.006), fontsize=5.0, frameon=False,
               handlelength=1.6, title="Hidden block (shaded) ± 15 days, daily means", title_fontsize=5.2, alignment="left")

    # IV Sentinel-2 and Clay
    fig.text(0.07, top + 0.004, "IV  SENTINEL-2 (300 m) AND CLAY", color=C["img"], **MONO)
    cd, rgb, cdays, cls = chips()
    for d, im in zip(cd, rgb):
        ax = fig.add_axes([0.1 + 0.215 * d / 180 - 0.014, top - 0.032, 0.028, 0.028])
        ax.imshow(im, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(C["img"]); s.set_linewidth(0.6)
    ax = fig.add_axes([0.1, top - 0.052, 0.215, 0.016])
    z = cls[np.argsort(cdays)][:, :96].T
    z = (z - z.mean(1, keepdims=True)) / (z.std(1, keepdims=True) + 1e-6)
    ax.imshow(z, aspect="auto", cmap="RdPu", extent=[cdays.min(), cdays.max(), 0, 1], vmin=-2, vmax=2, interpolation="nearest")
    ax.set_xlim(0, 180); ax.axis("off")
    fig.text(0.097, top - 0.044, "Clay", ha="right", va="center", fontsize=5.2, color=C["img"])
    centers["img"] = top - 0.027
    fig.text(0.725, centers["img"], f"{len(cdays)} acquisitions in the window, {len(cd)} clear ones shown. An\n"
             "embedding is reconstructed when imagery is hidden (M4, M5).", fontsize=5.0, color=C["img"], va="center")
    top = top - 0.08

    # V site memory
    mu, sd, yrs = memory_band()
    fig.text(0.07, top + 0.004, f"V  SITE MEMORY (OTHER YEARS, {yrs[0]}-{yrs[-1]})", color=C["mem"], **MONO)
    ax = fig.add_axes([0.1, top - 0.03, 0.215, 0.026])
    dd = np.arange(180) + 0.5
    ax.fill_between(dd, mu - sd, mu + sd, color=C["mem"], alpha=0.15, lw=0); ax.plot(dd, mu, color=C["mem"], lw=0.7)
    ax.set_xlim(0, 180); ax.axis("off")
    ax.text(-0.015, 0.5, r"FCH$_4$", ha="right", va="center", fontsize=5.4, color=C["mem"], transform=ax.transAxes)
    centers["mem"] = top - 0.017
    fig.text(0.1, top - 0.034, r"$(\mu,\sigma,n)$ at each half-hour and season: three channels of every tower token",
             fontsize=4.8, color=C["mem"], va="top")
    top = top - 0.058

    # VI location: SINR encoder of the tower's coordinates
    fig.text(0.07, top + 0.004, "VI  LOCATION", color=C["loc"], **MONO)
    land = json.load(open(Path(__file__).parent / "static_data" / "ne_110m_land.geojson"))
    ax = fig.add_axes([0.1, top - 0.045, 0.09, 0.042])
    for feat in land["features"]:
        g = feat["geometry"]
        for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            xy = np.array(poly[0]); ax.fill(xy[:, 0], xy[:, 1], color="#dde2ea", lw=0)
    ax.plot(LON, LAT, "o", ms=3, color=C["loc"]); ax.set_xlim(-180, 180); ax.set_ylim(-60, 85); ax.axis("off")
    fig.text(0.195, top - 0.024, f"{LAT:.2f}° N, {-LON:.2f}° W\n" r"$[\sin\phi,\cos\phi,\sin\lambda,\cos\lambda]$",
             fontsize=4.9, color=C["loc"], va="center")
    centers["loc"] = top - 0.024

    # ---------------------------------------------------------------- b: the network
    xe, xs_ = 0.338, 0.402
    for k in ("flux", "met", "era", "img"):
        y = centers[k]
        arrow(fig, (0.322, y), (xe, y), C[k])
        tri(fig, xe, y, 0.03, 0.04, C[k], "e")
        arrow(fig, (xe + 0.03, y), (xs_ - 0.007, y), C[k])
        fig.add_artist(Circle((xs_, y), 0.0068, fill=False, ec=INK, lw=0.8, transform=fig.transFigure))
        fig.text(xs_, y, "+", ha="center", va="center", fontsize=6.5, color=INK)
        arrow(fig, (xs_ + 0.0068, y), (XT0 - 0.004, y), INK)
    line(fig, [0.322, 0.329], [centers["mem"]] * 2, C["mem"])
    line(fig, [0.329, 0.329], [centers["mem"], centers["flux"] - 0.011], C["mem"])
    for k in ("flux", "met"):
        arrow(fig, (0.329, centers[k] - 0.011), (xe + 0.005, centers[k] - 0.011), C["mem"], lw=0.7)
    y = centers["loc"]
    arrow(fig, (0.31, y), (xe, y), C["loc"])
    tri(fig, xe, y, 0.03, 0.036, C["loc"], "s", size=6.5)
    fig.text(xe + 0.012, y + 0.021, "SINR", fontsize=4.8, color=C["loc"], ha="center")
    line(fig, [xe + 0.03, xs_], [y, y], C["loc"])
    line(fig, [xs_, xs_], [y, centers["flux"] - 0.0068], C["loc"], lw=0.8)
    fig.text(0.1, y - 0.027, "SINR location embedding, added to every token",
             fontsize=4.5, color=C["loc"], va="top", ha="left")
    # DeepEarth transformer: factorised attention
    y_lo, y_hi = centers["loc"] - 0.03, tops["flux"] + 0.012
    box(fig, XT0, y_lo, XT1 - XT0, y_hi - y_lo, "white", INK, lw=1.2, r=0.02)
    fig.text(XC, y_hi - 0.012, "factorised attention", fontsize=5.0, ha="center", va="center", color="#555555")
    gw, gh = 0.11, 0.075
    gx = XC - gw / 2
    gy1 = y_hi - 0.035 - gh
    fig.text(XC, gy1 + gh + 0.005, "time attention", fontsize=5.6, ha="center", fontweight="bold")
    grid_icon(fig, gx, gy1, gw, gh, 8, 16, hl_row=2, color=C["flux"])
    fig.text(XC, gy1 - 0.004, "each variable over its 180 days\n(rotary day positions)", fontsize=4.6, ha="center", va="top")
    gy2 = gy1 - 0.058 - gh
    arrow(fig, (XC, gy1 - 0.025), (XC, gy2 + gh + 0.014), INK, lw=0.7)
    fig.text(XC, gy2 + gh + 0.005, "variable attention", fontsize=5.6, ha="center", fontweight="bold")
    grid_icon(fig, gx, gy2, gw, gh, 8, 16, hl_col=7, color=C["met"], img_row=True)
    fig.text(XC, gy2 - 0.004, "each day across its variables\nand its image token", fontsize=4.6, ha="center", va="top")
    gy3 = gy2 - 0.052
    arrow(fig, (XC, gy2 - 0.025), (XC, gy3 + 0.016), INK, lw=0.7)
    box(fig, XC - 0.035, gy3, 0.07, 0.014, "#f2f2f2", INK, lw=0.6, r=0.004)
    fig.text(XC, gy3 + 0.007, "MLP", fontsize=5.4, ha="center", va="center")
    arrow(fig, (XC + 0.037, gy3 + 0.007), (XC + 0.058, gy1 + gh * 0.6), INK, lw=0.7, rad=0.32)
    fig.text(XC + 0.068, (gy1 + gy3) / 2 + 0.02, "×10", fontsize=7, ha="center", va="center", fontweight="bold")
    fig.text(XC, gy3 - 0.01, "residual connections, layer norm;\nwidth 320, 8 heads, 28.1 M parameters",
             fontsize=4.5, ha="center", va="top")
    fig.text(XC, y_lo + 0.012, r"weights $\theta$", fontsize=5.6, ha="center", color=C["adapt"], fontweight="bold")
    # heads
    for k, lab in (("flux", r"$p_\theta(x\,|\,\mathbf{h})$"), ("met", r"$p_\theta(x\,|\,\mathbf{h})$"),
                   ("img", r"$\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\sigma}^2)$")):
        y = centers[k]
        arrow(fig, (XT1 + 0.004, y), (0.618, y), INK)
        tri(fig, 0.62, y, 0.032, 0.04, C[k], "d", right=False)
        arrow(fig, (0.652, y), (0.69, y), C[k])
        fig.text(0.671, y + 0.005, lab, fontsize=5.0, color=C[k], ha="center", va="bottom")
    fig.text(0.637, centers["flux"] - 0.026, "two-component\nStudent-t per\nhalf-hour", fontsize=4.5, ha="center", va="top", color=C["flux"])

    # ---------------------------------------------------------------- site adaptation: an unrolled recurrence over the weights
    ya = y_lo - 0.105
    box(fig, 0.07, ya - 0.012, 0.905, 0.078, "#fbf8f1", C["adapt"], lw=0.9, r=0.012)
    fig.text(0.08, ya + 0.054, "TEST-TIME SITE ADAPTATION: 300 RECURRENT WEIGHT UPDATES PER WINDOW", color=C["adapt"], **MONO)
    cells = [r"$\theta^\star$", r"$\theta_1$", r"$\theta_2$", r"$\cdots$", r"$\theta_{299}$", r"$\theta_w$"]
    xs_c = np.linspace(XC - 0.33, XC + 0.33, len(cells))
    yc = ya + 0.024
    for i, lab in enumerate(cells):
        if lab != r"$\cdots$":
            fig.add_artist(Circle((xs_c[i], yc), 0.014, fc="white", ec=C["adapt"], lw=1.0, transform=fig.transFigure))
        fig.text(xs_c[i], yc, lab, fontsize=6.4, ha="center", va="center", color=C["adapt"])
        if i < len(cells) - 1:
            x0, x1 = xs_c[i] + 0.016, xs_c[i + 1] - 0.016
            arrow(fig, (x0, yc), (x1, yc), C["adapt"], lw=1.0)
            if r"$\cdots$" not in (lab, cells[i + 1]):
                xm = (x0 + x1) / 2
                fig.text(xm, yc + 0.004, r"$-\eta\,\nabla_\theta\mathcal{L}_w$", fontsize=5.0, ha="center", va="bottom", color=C["adapt"])
                arrow(fig, (xm, yc - 0.022), (xm, yc - 0.002), "#999999", lw=0.6)
                fig.text(xm, yc - 0.023, r"fresh mask $\mathbf{M}'$ on $\mathbf{X}^{\mathrm{vis}}_w\cup\mathcal{C}_w$", fontsize=4.3,
                         ha="center", va="top", color="#555555")
    fig.text(xs_c[0], yc - 0.019, "trained weights", fontsize=4.6, ha="center", va="top", color=C["adapt"])
    fig.text(xs_c[-1], yc - 0.019, "weights for window w", fontsize=4.6, ha="center", va="top", color=C["adapt"])
    # links to the transformer: theta* copied in, theta_w used to fill every mask of w, then restored
    arrow(fig, (XC - 0.02, y_lo - 0.002), (XC - 0.02, ya + 0.064), C["adapt"], lw=1.0)
    arrow(fig, (XC + 0.02, ya + 0.064), (XC + 0.02, y_lo - 0.002), C["adapt"], lw=1.4)
    fig.text(XC - 0.025, (y_lo + ya + 0.064) / 2, r"copy $\theta^\star$", fontsize=4.8, ha="right", va="center", color=C["adapt"])
    fig.text(XC + 0.025, (y_lo + ya + 0.064) / 2, r"fill window $w$ with $\theta_w$; restore $\theta^\star$ after", fontsize=4.8,
             ha="left", va="center", color=C["adapt"])
    fig.text(0.965, ya + 0.054, r"AdamW, lr $3\times10^{-4}$, batch 2, FCH$_4$ loss weight 4; hidden values never enter $\mathcal{L}_w$",
             fontsize=4.6, ha="right", va="bottom", color="#555555")

    # ---------------------------------------------------------------- d: inside the gap, half-hourly
    yn, yc_ = np.isfinite(series(w, "NEE")), np.isfinite(series(w, "FCH4"))
    best = max(range(0, (b - a) // 48 - 2), key=lambda d: min(yn[a + d * 48:a + (d + 3) * 48].sum(), yc_[a + d * 48:a + (d + 3) * 48].sum()))
    z0 = a + best * 48; z1 = z0 + 3 * 48
    tt = np.arange(8640) / 48
    yb = ya - 0.135
    for i, k in enumerate(("NEE", "FCH4")):
        ax = fig.add_axes([0.075 + i * 0.31, yb, 0.25, 0.085])
        y = series(w, k)
        s = sm["samples"][:, skeys.index(k)]
        qs = np.nanquantile(s, [0.025, 0.1, 0.9, 0.975], axis=0)
        ax.fill_between(tt[z0:z1], qs[0, z0:z1], qs[3, z0:z1], color=C["flux"], alpha=0.13, lw=0, label="FluxFill 95%")
        ax.fill_between(tt[z0:z1], qs[1, z0:z1], qs[2, z0:z1], color=C["flux"], alpha=0.22, lw=0, label="FluxFill 80%")
        tm = top_method(wc, k)
        q = pred(wc, tm, k)
        ax.plot(tt[z0:z1], q["mean"][z0:z1], color=TOP_C, lw=0.8, label=NAMES[tm])
        ax.plot(tt[z0:z1], np.nanmean(s, 0)[z0:z1], color=C["flux"], lw=1.2, label="FluxFill mean")
        ax.plot(tt[z0:z1], y[z0:z1], ".", color="black", ms=2, label="Measured")
        unit = r"µmol m$^{-2}$ s$^{-1}$" if k == "NEE" else r"nmol m$^{-2}$ s$^{-1}$"
        ax.set_ylabel(f"{LAB[k]} ({unit})", fontsize=6.2); ax.tick_params(labelsize=5.8)
        ax.set_xlabel("Day of window", fontsize=6.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=4.9, frameon=False, handlelength=1.3, columnspacing=0.8)
    k = "FCH4"
    y = series(w, k); s = sm["samples"][:, skeys.index(k)]
    cand = [t for t in range(z0, z1) if np.isfinite(y[t])]
    t_star = cand[len(cand) // 2]
    ax = fig.add_axes([0.72, yb, 0.255, 0.085])
    x = s[:, t_star]
    lo, hi = np.percentile(x, [0.5, 99.5]); span = hi - lo
    grid = np.linspace(lo - 0.4 * span, hi + 0.4 * span, 300)
    bw = 1.06 * x.std() * len(x) ** -0.2
    dens = np.exp(-0.5 * ((grid[:, None] - x[None]) / bw) ** 2).sum(1) / (len(x) * bw * np.sqrt(2 * np.pi))
    ax.fill_between(grid, dens, color=C["flux"], alpha=0.25, lw=0); ax.plot(grid, dens, color=C["flux"], lw=1.0, label="FluxFill (64 samples)")
    ax.axvline(y[t_star], color="black", lw=0.9, label="Measured")
    tm = top_method(wc, k); q = pred(wc, tm, k)
    yy = -dens.max() * 0.15
    ax.plot([q["q025"][t_star], q["q975"][t_star]], [yy, yy], color=TOP_C, lw=1.0, label=f"{NAMES[tm]} 95% interval")
    ax.plot(q["mean"][t_star], yy, "o", color=TOP_C, ms=2.5)
    ax.set_yticks([]); ax.spines[["left", "top", "right"]].set_visible(False); ax.tick_params(labelsize=5.8)
    ax.set_xlabel(r"FCH$_4$ (nmol m$^{-2}$ s$^{-1}$), one hidden half-hour", fontsize=6.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=4.9, frameon=False, handlelength=1.3)
    vals["fig1_t_star_day"] = f"{t_star / 48:.1f}"

    # ---------------------------------------------------------------- e: every method, test towers
    tab = vals.get("fig1_table", {})
    y0 = yb - 0.058
    fig.text(0.075, y0 + 0.014, "CRPS at the test towers (all mask families; cells that every scored method fills; lower is better)",
             fontsize=5.2, color="#555555")
    heads = [("Method", 0.075, "left"), ("Model", 0.18, "left"), ("Uncertainty", 0.43, "left"), ("NEE", 0.64, "center"),
             ("LE", 0.73, "center"), ("H", 0.82, "center"), (r"FCH$_4$", 0.91, "center")]
    for lab, x, ha in heads:
        fig.text(x, y0, lab, fontsize=5.6, fontweight="bold", ha=ha)
    line(fig, [0.075, 0.975], [y0 - 0.004] * 2, "#999999", lw=0.5)
    meth = [("climatology", "Climatology", "tower's other years, ±15 days", "empirical spread"),
            ("mds", "MDS", "look-up of similar meteorology", "spread of look-up values"),
            ("irvin_rf", "RF ensemble", "random forests per tower", "ensemble, Laplace"),
            ("liu_xgb", "XGBoost", "boosted trees per tower", "quantile models"),
            ("vekuri_de", "Deep ensemble", "neural networks per tower", "ensemble, Laplace"),
            ("saits", "SAITS", "self-attention imputer, all towers", "dropout samples"),
            ("csdi", "CSDI", "score-based diffusion, all towers", "diffusion samples"),
            ("chronos2", "Chronos-2", "pretrained forecaster, zero-shot", "quantiles"),
            ("fluxfill", "FluxFill", "DeepEarth network + site adaptation", "Student-t mixture")]
    best = {v: min((tab[m][v] for m in tab if v in tab[m]), default=None) for v in ("NEE", "LE", "H", "FCH4")}
    for i, (m, name, model, unc) in enumerate(meth):
        yy = y0 - 0.014 - i * 0.0115
        if m == "fluxfill":
            fig.add_artist(Rectangle((0.07, yy - 0.0035), 0.905, 0.011, fc="#f3f1e7", ec="none", transform=fig.transFigure))
        fig.text(0.075, yy, name, fontsize=5.5, fontweight="bold" if m == "fluxfill" else "normal")
        fig.text(0.18, yy, model, fontsize=5.3)
        fig.text(0.43, yy, unc, fontsize=5.3)
        if m not in tab:
            fig.text(0.775, yy, "not yet scored on every test window", fontsize=5.0, ha="center", color="#999999")
            continue
        for v, x in zip(("NEE", "LE", "H", "FCH4"), (0.64, 0.73, 0.82, 0.91)):
            val = tab[m].get(v)
            if val is None:
                continue
            fig.text(x, yy, f"{val:.3f}" if val < 10 else f"{val:.2f}", fontsize=5.3, ha="center",
                     fontweight="bold" if np.isclose(val, best[v]) else "normal")
    for lab, (x, y) in {"a": (0.012, 0.985), "b": (0.33, 0.985), "c": (0.7, 0.985), "d": (0.012, yb + 0.115),
                        "e": (0.012, y0 + 0.014)}.items():
        fig.text(x, y, lab, fontsize=10, fontweight="bold")
    with plt.rc_context({"svg.fonttype": "path"}):            # glyphs as paths: mathtext renders in every browser
        fig.savefig(out / "fig0.svg", facecolor="white")
    plt.close(fig)
    vals["fig1_acquisitions"] = len(cdays)
