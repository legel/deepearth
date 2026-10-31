"""Figures, tables and numbers of the FluxFill report, computed from the benchmark score files.

Run in the FluxFill environment (pandas, matplotlib):
    python report_figures.py --out /home/photon/4tb/fluxfill/bench/dashboard/report
Writes fig1.svg ... fig7.svg, tables.json (HTML tables) and values.json (numbers quoted in the text). build.py copies
the figures to img/report/ and fills the report template with the tables and values.

Every comparison between methods uses matched cells: a cell is one (tower, window, mask family, variable), and a
cell enters only if every compared method scored at least one hidden value in it (n > 0).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = Path("/home/photon/4tb/fluxfill")
BENCH = R / "bench"
FLUXES = ["NEE", "LE", "H", "FCH4"]
UNITS = {"NEE": "µmol m⁻² s⁻¹", "LE": "W m⁻²", "H": "W m⁻²", "FCH4": "nmol m⁻² s⁻¹"}
KEY = ["site", "window", "family", "variable"]
NAMES = {"climatology": "Climatology", "mds": "MDS", "irvin_rf": "RF ensemble", "liu_xgb": "XGBoost",
         "vekuri_de": "Deep ensemble", "saits": "SAITS", "csdi": "CSDI", "chronos2": "Chronos-2", "fluxfill": "FluxFill"}
COLORS = {"climatology": "#9a9a9a", "mds": "#2a78d6", "irvin_rf": "#1baf7a", "liu_xgb": "#eb6834",
          "vekuri_de": "#8a5cc2", "saits": "#c2416b", "csdi": "#6b8e23", "chronos2": "#b08900", "fluxfill": "#000000"}
PUNITS = {"NEE": r"µmol m$^{-2}$ s$^{-1}$", "LE": r"W m$^{-2}$", "H": r"W m$^{-2}$", "FCH4": r"nmol m$^{-2}$ s$^{-1}$"}
GAPS = [("M2-2h", "2 h"), ("M2-6h", "6 h"), ("M2-1d", "1 d"), ("M2-7d", "7 d"), ("M2-30d", "30 d"),
        ("M2-90d", "90 d"), ("M2-180d", "180 d")]

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "legend.frameon": False, "svg.fonttype": "none", "figure.dpi": 150})


def cmp(ours: float, base: float, digits: int) -> str:
    """'15.1% lower' or '2.0% higher': FluxFill's CRPS relative to a reference."""
    d = 100 * (ours / base - 1)
    return f"{abs(d):.{digits}f}% {'lower' if d < 0 else 'higher'}"


# ---------------------------------------------------------------- scores
def load(paths, method=None):
    d = pd.concat([pd.read_parquet(p) for p in paths if Path(p).exists()], ignore_index=True)
    if method:
        d["method"] = method
    return d.drop_duplicates(KEY + ["kind", "gap_class"], keep="last")


def matched(frames: dict, kind="values") -> dict:
    """Restrict every method's rows of one kind to the cells every method fills."""
    sel = {m: f[(f.kind == kind) & (f.n > 0)] for m, f in frames.items()}
    keys = None
    for f in sel.values():
        k = set(map(tuple, f[KEY].drop_duplicates().to_numpy()))
        keys = k if keys is None else keys & k
    idx = pd.MultiIndex.from_tuples(sorted(keys), names=KEY)
    return {m: f[f.set_index(KEY).index.isin(idx)] for m, f in sel.items()}


def agg(f: pd.DataFrame) -> dict:
    """Pooled scores over rows (each row = one cell and gap class, weighted by its hidden values)."""
    n = f.n.sum()
    if n == 0:
        return {}
    w = f.n / n
    out = {"n": int(n), "cells": int(f[KEY].drop_duplicates().shape[0]), "crps": float((f.crps * w).sum()),
           "mae": float((f.mae * w).sum()), "rmse": float(np.sqrt((f.rmse ** 2 * w).sum())),
           "bias": float((f.bias * w).sum())}
    for c in ("cov50", "cov80", "cov95"):
        out[c] = float((f[c] * w).sum())
    if {"sum_y", "sum_y2", "sum_abs_y"} <= set(f.columns):
        sst = f.sum_y2.sum() - f.sum_y.sum() ** 2 / n
        out["r2"] = float(1 - (f.rmse ** 2 * f.n).sum() / sst) if sst > 0 else float("nan")
        out["wape"] = float(100 * (f.mae * f.n).sum() / f.sum_abs_y.sum())
    return out


def test_frames():
    T, T0 = BENCH / "test_r02" / "parts", BENCH / "test" / "parts"
    fr = {"climatology": load([T0 / "climatology.parquet", T / "climatology_new.parquet"], "climatology"),
          "mds": load([T0 / "mds.parquet", T / "mds_new.parquet"], "mds"),
          "irvin_rf": load([T / "irvin_rf_merged.parquet"], "irvin_rf"),
          "liu_xgb": load([T / "liu_xgb_merged.parquet"], "liu_xgb"),
          "saits": load([T / "saits_merged.parquet"], "saits"),
          "chronos2": load([T / "chronos2_merged.parquet"], "chronos2")}
    fin = [T / "fluxfill_final_a.parquet", T / "fluxfill_final_b.parquet"]
    final = all(p.exists() for p in fin) and load(fin)[["site", "window"]].drop_duplicates().shape[0] >= 514
    fr["fluxfill"] = load(fin if final else [T / "fluxfill_v3.parquet"], "fluxfill")
    return {m: f[f.variable.isin(FLUXES)] for m, f in fr.items()}, final


# ---------------------------------------------------------------- figures
def save(fig, out: Path, name: str):
    fig.savefig(out / name, bbox_inches="tight", facecolor="white", dpi=300)     # raster parts of the SVG at 300 dpi
    plt.close(fig)


def fig1(out: Path, vals: dict):
    """a, towers by split; b, the length of real gaps in the tower records; c, the mask families on one window."""
    sites = json.load(open(R / "dashboard_build" / "site" / "data" / "sites.json"))["sites"]
    land = json.load(open(Path(__file__).parent / "static_data" / "ne_110m_land.geojson"))
    bm = json.load(open(R / "dashboard_build" / "site" / "data" / "benchmark.json"))
    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_axes([0.0, 0.47, 1.0, 0.5])
    for feat in land["features"]:
        g = feat["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            xy = np.array(poly[0])
            ax.fill(xy[:, 0], xy[:, 1], color="#ececec", lw=0)
    style = {"train": ("#9a9a9a", 7, "Training"), "validation": ("#1baf7a", 14, "Validation"), "test": ("#eb6834", 14, "Test")}
    spl = bm["splits"]["frozen"]
    for sp, (c, s, lab) in style.items():
        pts = [x for x in sites if x.get("split") == sp]
        ax.scatter([x["lon"] for x in pts], [x["lat"] for x in pts], s=s, c=c, lw=0, label=f"{lab}: {spl[sp]['towers']} towers, {spl[sp]['methane']} with FCH4", zorder=3)
        ch4 = [x for x in pts if x["ch4"]]
        ax.scatter([x["lon"] for x in ch4], [x["lat"] for x in ch4], s=s + 14, facecolors="none", edgecolors="black", lw=0.5, zorder=4)
    ax.scatter([], [], s=20, facecolors="none", edgecolors="black", lw=0.5, label="Measures methane (FCH4)")
    ax.set_xlim(-170, 180); ax.set_ylim(-58, 80); ax.set_aspect(1.0); ax.axis("off")
    ax.legend(loc="lower left", ncol=1, handletextpad=0.3, borderaxespad=0.2)
    ax.text(-170, 78, "a", fontweight="bold", fontsize=10)

    ax = fig.add_axes([0.07, 0.06, 0.38, 0.33])
    o = bm["outages"]
    labels = ["0.5 h", "1 h", "2 h", "4 h", "8 h", "1 d", "3 d", "7 d", "30 d", "90 d", "180 d", "1 y"]
    share = 100 * np.array(o["halfhours"]) / np.sum(o["halfhours"])
    ax.bar(range(len(share)), share, color="#555555", width=0.8)
    ax.set_xticks(range(len(share))); ax.set_xticklabels([f"≥{x}" for x in labels], rotation=60, ha="right")
    ax.set_ylabel("Missing half-hours (%)"); ax.set_xlabel("Length of the gap")
    long = share[labels.index("7 d"):].sum()
    vals["gap_share_ge7d"] = f"{long:.0f}"
    vals["gap_share_ge30d"] = f"{share[labels.index('30 d'):].sum():.0f}"
    vals["outage_n"] = f"{o['n']:,}"; vals["outage_tower_years"] = f"{o['tower_years']:,}"
    vals["gap_count_lt1d"] = f"{100 * sum(o['count'][:5]) / sum(o['count']):.0f}"   # bins below 48 half-hours
    tot = sum(o["halfhours"])                                                       # half-hours inside station outages
    vals["outage_halfhours_m"] = f"{tot / 1e6:.1f}"
    vals["outage_years"] = f"{tot / 17520:,.0f}"
    vals["outage_share"] = f"{100 * tot / (17520 * o['tower_years']):.0f}"
    ax.text(-0.18, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=10)

    ax = fig.add_axes([0.55, 0.06, 0.43, 0.33])
    w = json.load(open(BENCH / "dashboard" / "windows" / "US-Ha1_2017-10-13.json"))
    rows = [v["key"] for v in w["variables"]]
    obs = np.array([[x is not None for x in v["values"]] for v in w["variables"]], float)
    show = [("M2: 7 d, all tower channels", "M2, 7 d"), ("M2: 90 d, all tower channels", "M2, 90 d"),
            ("M4: reanalysis rows removed, plus 30 d of all tower channels", "M4")]
    show = [(k, s) for k, s in show if k in w["masks"]]
    T = obs.shape[1]
    img = np.where(obs > 0, 1.0, 0.0)
    k, lab = show[min(1, len(show) - 1)]
    for var, segs in w["masks"][k].items():
        if var in rows:
            for a, b in segs:
                img[rows.index(var), a:b] = np.where(img[rows.index(var), a:b] > 0, 2.0, img[rows.index(var), a:b])
    cmap = matplotlib.colors.ListedColormap(["#ffffff", "#c9c9c9", "#eb6834"])
    ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest", extent=[0, T / 48, len(rows), 0])
    ax.set_yticks(np.arange(len(rows)) + 0.5); ax.set_yticklabels(rows, fontsize=5.2)
    ax.set_xlabel("Day of the 180-day window (US-Ha1, from 2017-10-13)")
    ax.tick_params(axis="y", length=0)
    hand = [matplotlib.patches.Patch(color="#c9c9c9", label="Measured, visible"),
            matplotlib.patches.Patch(color="#eb6834", label=f"Measured, hidden ({lab} mask)"),
            matplotlib.patches.Patch(facecolor="white", edgecolor="#999999", lw=0.4, label="Not measured")]
    ax.legend(handles=hand, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handlelength=1)
    ax.text(-0.2, 1.04, "c", transform=ax.transAxes, fontweight="bold", fontsize=10)
    save(fig, out, "fig1.svg")


def fig3(out: Path, fr: dict, vals: dict):
    """CRPS by gap length for each flux, test towers; cells matched within each gap length over the methods scored
    at that length (climatology and MDS were not scored on the 90- and 180-day blocks of release 0.2)."""
    per = {}
    for g, _ in GAPS:
        cnt = {m: f[(f.family == g) & (f.kind == "values") & (f.n > 0)][KEY].drop_duplicates().shape[0] for m, f in fr.items()}
        have = {m: fr[m][fr[m].family == g] for m in fr if cnt[m] >= 0.5 * max(cnt.values())}
        per[g] = matched(have)
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.3))
    for ax, v in zip(axs, FLUXES):
        for m in fr:
            y = [agg(per[g][m][per[g][m].variable == v]).get("crps", np.nan) if m in per[g] else np.nan for g, _ in GAPS]
            ax.plot(range(len(GAPS)), y, marker="o", ms=2.6, lw=1.8 if m == "fluxfill" else 1.0, color=COLORS[m],
                    label=NAMES[m], zorder=5 if m == "fluxfill" else 3)
        ax.set_xticks(range(len(GAPS))); ax.set_xticklabels([s for _, s in GAPS], rotation=45)
        ax.set_title(f"{v} ({PUNITS[v]})")
        ax.set_ylim(bottom=0)
    axs[0].set_ylabel("CRPS (lower is better)")
    axs[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.supxlabel("Length of the hidden block (M2, every tower channel hidden)", fontsize=8, y=-0.08)
    for i, ax in enumerate(axs):
        ax.text(-0.25 if i == 0 else -0.12, 1.08, "abcd"[i], transform=ax.transAxes, fontweight="bold", fontsize=10)
    save(fig, out, "fig3.svg")
    # The text quotes FluxFill against the best other method at the shortest and longest gaps.
    for v in FLUXES:
        for g in ("M2-1d", "M2-180d"):
            sc = {m: agg(f[f.variable == v]).get("crps", np.nan) for m, f in per[g].items()}
            base = min((m for m in sc if m != "fluxfill"), key=lambda m: sc[m])
            vals[f"gap_{v}_{g[3:]}"] = cmp(sc["fluxfill"], sc[base], 1)
            vals[f"gap_{v}_{g[3:]}_base"] = NAMES[base]


FLUX_C = {"NEE": "#2f7d32", "LE": "#2a6fb0", "H": "#c8553d", "FCH4": "#7b4fa0"}
FLUX_LAB = {"NEE": "NEE", "LE": "LE", "H": "H", "FCH4": r"FCH$_4$"}


def fig4(out: Path, vals: dict):
    """Ten test windows, one hidden station outage each (report_fig4_data.py), the four fluxes per row: half-hourly for
    gaps up to 7 days, daily means of each sample beyond. The x-axis of each row covers the gap and 40% of its length on
    both sides (at least one day)."""
    d = BENCH / "dashboard" / "report" / "fig4"
    sel = pd.read_csv(d / "selection.csv")
    sel["days"] = sel["mask"].str.extract(r"(\d+)d")[0].astype(int)
    sel = sel[[(d / f"{r.site}_{r.start}_{r.mask}.npz").exists() for r in sel.itertuples()]]
    sel = sel.sort_values(["days", "site"]).reset_index(drop=True)
    n = len(sel)
    fig, axs = plt.subplots(n, 4, figsize=(7.2, 0.72 * n + 0.45), squeeze=False)
    fig.subplots_adjust(left=0.135, right=0.995, top=1 - 0.2 / (0.72 * n + 0.45), bottom=0.25 / (0.72 * n + 0.45), hspace=0.16, wspace=0.2)
    hits, tot = {v: 0 for v in FLUXES}, {v: 0 for v in FLUXES}
    for r, row in enumerate(sel.itertuples()):
        z = np.load(d / f"{row.site}_{row.start}_{row.mask}.npz")
        keys = [str(k) for k in z["keys"]]
        a, b = (int(x) for x in z["span"])
        ctx = max(48, int(0.4 * (b - a)))
        lo_t, hi_t = max(0, a - ctx) / 48, min(z["y"].shape[1], b + ctx) / 48
        for c, v in enumerate(FLUXES):
            ax = axs[r, c]
            if c == 0:
                ax.text(-0.3, 0.5, f"{row.site}\n{row.igbp}, {row.days} d", transform=ax.transAxes, ha="right", va="center", fontsize=6)
            if r == 0:
                ax.set_title(f"{FLUX_LAB[v]} ({PUNITS[v]})", fontsize=6.5, color=FLUX_C[v], pad=3)
            i = keys.index(v) if v in keys else None
            if i is None or not np.isfinite(z["y"][i, a:b]).any():
                ax.axis("off")
                continue
            y, (mu, lo, hi) = z["y"][i], z["hh"][:, i]
            inside = np.isfinite(y[a:b])
            hits[v] += int(((y[a:b] >= lo[a:b]) & (y[a:b] <= hi[a:b]))[inside].sum()); tot[v] += int(inside.sum())
            if row.days <= 7:
                t, truth, m_, l_, h_ = np.arange(len(y)) / 48 + 1 / 96, y, mu, lo, hi
            else:
                t = (int(z["off"]) + (np.arange(z["yd"].shape[1]) + 0.5) * 48) / 48
                truth, (m_, l_, h_) = z["yd"][i], z["dd"][:, i]
            hid = (t > a / 48) & (t < b / 48)
            show = (t >= lo_t) & (t <= hi_t)
            ax.axvspan(a / 48, b / 48, color="#f1efe6", lw=0, zorder=0)
            ax.plot(t, np.where(hid, np.nan, truth), color="#8a8f96", lw=0.6)
            ax.fill_between(t[hid], l_[hid], h_[hid], color=FLUX_C[v], alpha=0.2, lw=0)
            ax.plot(t[hid], m_[hid], color=FLUX_C[v], lw=0.9)
            ax.plot(t[hid], truth[hid], ".", color="black", ms=1.5 if row.days <= 7 else 2.2, mew=0)
            all_v = np.concatenate([truth[show], l_[hid], h_[hid]])
            all_v = all_v[np.isfinite(all_v)]
            pad = 0.06 * (all_v.max() - all_v.min() + 1e-9)
            ax.set_xlim(lo_t, hi_t); ax.set_ylim(all_v.min() - pad, all_v.max() + pad)
            ax.set_xticks([]); ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.tick_params(labelsize=5, length=1.5, pad=1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#c5c9cf"); ax.spines[["left", "bottom"]].set_linewidth(0.5)
    hand = [Line2D([], [], color="#8a8f96", lw=0.6, label="measured, visible"),
            Line2D([], [], color="black", lw=0, marker=".", ms=3, label="measured, hidden"),
            Line2D([], [], color="#555555", lw=0.9, label="FluxFill mean"),
            Patch(color="#555555", alpha=0.2, lw=0, label="FluxFill 95% interval"),
            Patch(color="#f1efe6", lw=0, label="hidden span")]
    fig.legend(handles=hand, loc="lower center", ncol=5, fontsize=6, frameon=False, handlelength=1.4, columnspacing=1.4,
               bbox_to_anchor=(0.55, 0))
    for v in FLUXES:
        if tot[v]:
            vals[f"ex_{v.lower()}_cov95"] = f"{100 * hits[v] / tot[v]:.0f}"
            vals[f"ex_{v.lower()}_n"] = f"{tot[v]:,}"
    vals["ex_windows"] = n
    save(fig, out, "fig4.svg")
    fig.savefig(out / "preview" / "fig4.png", dpi=200)


def fig5(out: Path, fr: dict, vals: dict):
    """Calibration on test towers: coverage of central intervals for values, and of 30-day and window sums."""
    mt = matched(fr)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axs[0]
    ax.plot([0.4, 1], [0.4, 1], color="#bbbbbb", lw=0.8, ls="--")
    for i, (m, f) in enumerate(mt.items()):
        s = agg(f[f.variable.isin(FLUXES)])
        ax.plot([0.5, 0.8, 0.95], [s["cov50"], s["cov80"], s["cov95"]], marker="o", ms=3, lw=1.6 if m == "fluxfill" else 0.9,
                color=COLORS[m], label=NAMES[m])
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage, half-hourly values")
    ax.set_xticks([0.5, 0.8, 0.95]); ax.legend(loc="upper left", fontsize=6)
    for ax, kind, title in ((axs[1], "sum-30-day", "30-day sums"), (axs[2], "sum-window", "Window sums")):
        ms = matched(fr, kind)
        names = list(ms)
        cov = [agg(ms[m][ms[m].variable.isin(FLUXES)]).get("cov95", np.nan) for m in names]
        ax.barh(range(len(names)), cov, color=[COLORS[m] for m in names], height=0.7)
        ax.axvline(0.95, color="#555555", lw=0.8, ls="--")
        ax.set_yticks(range(len(names))); ax.set_yticklabels([NAMES[m] for m in names])
        ax.set_xlim(0, 1.0); ax.set_xlabel("Empirical 95% coverage"); ax.set_title(title)
        for i, c in enumerate(cov):
            ax.text(min(c, 0.99) + 0.01 if c < 0.86 else c - 0.01, i, f"{c:.2f}", va="center", ha="left" if c < 0.86 else "right",
                    fontsize=6.5, color="black" if c < 0.86 else "white")
        vals[f"cov95_{kind}"] = {NAMES[m]: f"{c:.2f}" for m, c in zip(names, cov)}
        oth = [(c, NAMES[m]) for m, c in zip(names, cov) if m != "fluxfill"]
        vals[f"cov95_{kind}_min"], vals[f"cov95_{kind}_min_name"] = f"{min(oth)[0]:.2f}", min(oth)[1]
        vals[f"cov95_{kind}_max"], vals[f"cov95_{kind}_max_name"] = f"{max(oth)[0]:.2f}", max(oth)[1]
        vals[f"cov95_{kind}_all_max"] = f"{max(cov):.2f}"
    for i, ax in enumerate(axs):
        ax.text(-0.3 if i == 0 else -0.45, 1.05, "abc"[i], transform=ax.transAxes, fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig5.svg")


def fig7(out: Path, vals: dict):
    """Validation CRPS during training for the model sizes and lengths that were trained in full."""
    runs = [("v2-final", "Width 192, depth 6, 24,000 steps", "#a8a8a8"),
            ("model-d384-l10", "Width 384, depth 10 (stopped at 10,000 steps)", "#2a78d6"),
            ("model-d320-l10", "Width 320, depth 10, 24,000 steps", "#555555"),
            ("v4-d320-48k", "Width 320, depth 10, 48,000 steps", "#000000")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for name, lab, c in runs:
        p = R / "model" / name / "log.jsonl"
        if not p.exists():
            continue
        ev = [json.loads(x) for x in p.read_text().splitlines() if '"eval"' in x]
        if not ev:
            continue
        s = [e["step"] for e in ev]
        axs[0].plot(s, [e["val_crps"] for e in ev], color=c, lw=1.2, marker="o", ms=2, label=lab)
        axs[1].plot(s, [e.get("val_crps_flux", np.nan) for e in ev], color=c, lw=1.2, marker="o", ms=2, label=lab)
        vals[f"train_{name}"] = {"steps": s[-1], "val_crps": f"{ev[-1]['val_crps']:.4f}", "best": f"{min(e['val_crps'] for e in ev):.4f}",
                                 "flux": f"{ev[-1].get('val_crps_flux', float('nan')):.4f}"}
    ev = [json.loads(x) for x in (R / "model" / "v4-d320-48k" / "log.jsonl").read_text().splitlines() if '"s_per_step"' in x]
    sps = float(np.median([e["s_per_step"] for e in ev]))
    vals["train_s_per_step"] = f"{sps:.2f}"; vals["train_hours_48k"] = f"{48000 * sps / 3600:.1f}"
    axs[0].set_ylabel("Validation CRPS, all variables (standardised)"); axs[1].set_ylabel("Validation CRPS, NEE, LE, H, FCH4")
    for ax in axs:
        ax.set_xlabel("Training step"); ax.set_xlim(left=0)
    axs[0].legend(loc="upper right", fontsize=6)
    for i, ax in enumerate(axs):
        ax.text(-0.24, 1.06, "ab"[i], transform=ax.transAxes, fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig7.svg")


def fig8(out: Path, fr: dict, vals: dict):
    """CRPS skill of FluxFill against the best other method, by IGBP vegetation class, test towers, matched cells."""
    sites = {x["id"]: x for x in json.load(open(R / "dashboard_build" / "site" / "data" / "sites.json"))["sites"]}
    mt = matched(fr)
    groups = {"ENF": "Evergreen needleleaf", "DBF": "Deciduous broadleaf", "MF": "Mixed forest", "EBF": "Evergreen broadleaf",
              "GRA": "Grassland", "CRO": "Cropland", "WET": "Wetland", "SAV": "Savanna", "WSA": "Woody savanna",
              "OSH": "Open shrubland", "CSH": "Closed shrubland"}
    for m in mt:
        mt[m] = mt[m].assign(igbp=mt[m].site.map(lambda s: sites.get(s, {}).get("igbp", "")))
    rows = []
    for g, name in groups.items():
        ff = mt["fluxfill"][mt["fluxfill"].igbp == g]
        n_sites = ff.site.nunique()
        if n_sites < 3:
            continue
        skill = {}
        for v in ("NEE", "LE", "H"):
            sc = {m: agg(f[(f.igbp == g) & (f.variable == v)]).get("crps", np.nan) for m, f in mt.items()}
            base = min(sc[m] for m in sc if m != "fluxfill")
            skill[v] = 100 * (1 - sc["fluxfill"] / base)
        rows.append((name, n_sites, skill))
    rows.sort(key=lambda r: -r[2]["NEE"])
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    x = np.arange(len(rows)); wd = 0.26
    for j, (v, c) in enumerate((("NEE", "#000000"), ("LE", "#2a78d6"), ("H", "#eb6834"))):
        ax.bar(x + (j - 1) * wd, [r[2][v] for r in rows], width=wd * 0.92, color=c, label=v)
    ax.axhline(0, color="black", lw=0.6)
    code = {v: k for k, v in groups.items()}
    ax.set_xticks(x); ax.set_xticklabels([f"{code[r[0]]}\n{r[1]} towers" for r in rows], fontsize=6.5)
    ax.set_ylabel("CRPS skill against the best\nother method (%)"); ax.legend(loc="upper right", ncol=3)
    save(fig, out, "fig8.svg")
    vals["igbp_skill"] = {r[0]: {k: f"{v:.0f}" for k, v in r[2].items()} | {"towers": r[1]} for r in rows}


# ---------------------------------------------------------------- tables
def fmt(v, metric):
    if v is None or not np.isfinite(v):
        return "–"
    if metric in ("cov95", "r2"):
        return f"{v:.3f}"
    if metric == "wape":
        return f"{v:.1f}%"
    a = abs(v)
    return f"{v:.3f}" if a < 10 else f"{v:.2f}" if a < 100 else f"{v:.1f}"


def table_test(fr: dict, vals: dict, final: bool, prefix: str = "test") -> str:
    mt = {m: f[f.gap_class == "all"] for m, f in matched(fr).items()}      # one row per cell; gap classes repeat its values
    metrics = [("crps", "CRPS", min), ("cov95", "95% cov.", lambda xs: min(xs, key=lambda x: abs(x - 0.95))),
               ("mae", "MAE", min), ("rmse", "RMSE", min), ("bias", "Bias", lambda xs: min(xs, key=abs)),
               ("r2", "R²", max), ("wape", "WAPE", min)]
    h = ['<div class="table-wrap"><table class="rep-table"><thead><tr><th>Flux</th><th>Method</th>']
    h += [f'<th class="num">{lab}</th>' for _, lab, _ in metrics] + ["</tr></thead><tbody>"]
    for v in FLUXES:
        rows = {m: agg(f[f.variable == v]) for m, f in mt.items()}
        ref = mt["fluxfill"][mt["fluxfill"].variable == v]          # the same hidden values: older parts carry no truth sums
        sst, sabs = ref.sum_y2.sum() - ref.sum_y.sum() ** 2 / ref.n.sum(), ref.sum_abs_y.sum()
        for m, f in mt.items():
            f = f[f.variable == v]
            rows[m]["r2"] = float(1 - (f.rmse ** 2 * f.n).sum() / sst)
            rows[m]["wape"] = float(100 * (f.mae * f.n).sum() / sabs)
        best = {k: pick([rows[m][k] for m in rows]) for k, _, pick in metrics}
        base = min((m for m in rows if m != "fluxfill"), key=lambda m: rows[m]["crps"])
        vals[f"{prefix}_{v}"] = {"cmp": cmp(rows["fluxfill"]["crps"], rows[base]["crps"], 1),
                             "base": NAMES[base], "cells": f"{rows['fluxfill']['cells']:,}", "n": f"{rows['fluxfill']['n']:,}",
                             "ff_cov95": f"{rows['fluxfill']['cov95']:.3f}", "ff_r2": f"{rows['fluxfill']['r2']:.2f}",
                             "base_r2": f"{rows[base]['r2']:.2f}"}
        for i, m in enumerate(rows):
            r = rows[m]
            cells = "".join(f'<td class="num">{"<b>" if np.isclose(r[k], best[k]) else ""}{fmt(r[k], k)}{"</b>" if np.isclose(r[k], best[k]) else ""}</td>' for k, _, _ in metrics)
            first = f'<th scope="row" rowspan="{len(rows)}">{v}<span class="sub">{UNITS[v]}; {r["cells"]:,} cells, {r["n"]:,} values</span></th>' if i == 0 else ""
            h.append(f'<tr class="{"ours" if m == "fluxfill" else ""}">{first}<td>{NAMES[m]}</td>{cells}</tr>')
    h.append("</tbody></table></div>")
    return "".join(h)


def table_methods(timing: dict) -> str:
    rows = [("climatology", "Mean diurnal cycle of the tower's other years within ±15 days", "Tower history", "Empirical spread", "–"),
            ("mds", "Marginal distribution sampling, look-up of similar meteorology [[Reichstein2005]]", "Tower meteorology", "Spread of look-up values", "–"),
            ("irvin_rf", "Random forest ensemble per tower [[Irvin2021]]", "Tower and reanalysis drivers", "Ensemble spread", "Per window"),
            ("liu_xgb", "Gradient-boosted trees per tower [[Liu2025]]", "Tower and reanalysis drivers", "Quantile models", "Per window"),
            ("vekuri_de", "Deep ensemble of neural networks per tower [[Vekuri2025]]", "Tower and reanalysis drivers", "Ensemble and noise", "Per window"),
            ("saits", "Self-attention imputation [[Du2023]]", "All tower and reanalysis rows", "Dropout samples", "Training towers"),
            ("csdi", "Conditional score-based diffusion [[Tashiro2021]]", "All tower and reanalysis rows", "Diffusion samples", "Training towers"),
            ("chronos2", "Pretrained time-series foundation model [[Ansari2025]]", "Each series with covariates", "Quantiles", "Pretrained"),
            ("fluxfill", "DeepEarth network (this work)", "Tower, reanalysis, Sentinel-2 (Clay), site memory", "Samples", "Training towers")]
    h = ['<div class="table-wrap"><table class="rep-table"><thead><tr><th>Method</th><th>Model</th><th>Inputs</th>'
         '<th>Uncertainty</th><th>Fitted on</th><th class="num">Seconds per mask</th></tr></thead><tbody>']
    for m, model, inp, unc, fit in rows:
        t = timing.get(m)
        ts = "–" if t is None else (f"{t:.2f}" if t < 10 else f"{t:.0f}")
        h.append(f'<tr class="{"ours" if m == "fluxfill" else ""}"><td>{NAMES[m]}</td><td>{model}</td><td>{inp}</td><td>{unc}</td><td>{fit}</td><td class="num">{ts}</td></tr>')
    h.append("</tbody></table></div>")
    return "".join(h)


def timing_values() -> dict:
    out = {}
    d = BENCH / "timing"
    for f in d.glob("*.parquet"):
        x = pd.read_parquet(f)
        if "predict_s" not in x:
            continue
        per = x.drop_duplicates(["site", "window", "family"])["predict_s"]
        out[f.stem] = float(per.mean())
    return out


def methane_v2(vals: dict):
    """Split v2 (18 methane validation towers, 102 methane windows): the network retrained on split v2 against
    climatology, MDS, the RF ensemble and XGBoost on matched M2 cells."""
    import glob
    V = BENCH / "val"
    fr = {"climatology": load([V / "fch4v2_parts" / "climatology.parquet"]), "mds": load([V / "fch4v2_parts" / "mds.parquet"]),
          "irvin_rf": load(sorted(glob.glob(str(V / "fch4v2_win" / "irvin_rf" / "*.parquet")))),
          "liu_xgb": load(sorted(glob.glob(str(V / "fch4v2_win" / "liu_xgb" / "*.parquet")))),
          "fluxfill": load([V / "fch4v2_parts" / "fluxfill-s2-a0.parquet"])}
    fr = {m: f[f.variable.isin(FLUXES) & f.family.str.startswith("M2")] for m, f in fr.items()}
    mt = matched(fr)
    vals["v2_windows"] = mt["fluxfill"][["site", "window"]].drop_duplicates().shape[0]
    for v in FLUXES:
        sc = {m: agg(f[f.variable == v]) for m, f in mt.items()}
        base = min((m for m in sc if m != "fluxfill"), key=lambda m: sc[m]["crps"])
        vals[f"v2_{v}"] = {"cmp": cmp(sc["fluxfill"]["crps"], sc[base]["crps"], 1), "base": NAMES[base],
                           "a0_cmp": cmp(sc["fluxfill"]["crps"], sc[base]["crps"], 1), "cov95": f"{sc['fluxfill']['cov95']:.3f}",
                           "base_cov95": f"{sc[base]['cov95']:.3f}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BENCH / "dashboard" / "report"))
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)
    vals, tables = {}, {}
    fr, final = test_frames()
    vals["test_final"] = final
    vals["test_model"] = "the network trained for 48,000 steps" if final else "the network trained for 24,000 steps"
    mt = matched(fr)                                 # Figure 1e: CRPS of every method scored on all test windows
    vals["fig1_table"] = {m: {v: agg(f[f.variable == v])["crps"] for v in FLUXES} for m, f in mt.items()}
    from report_fig1 import KEY, figure1            # Figure 1 of the report (fig0.svg), once its exports exist
    if (out / "fig1_window_base" / f"{KEY}.json").exists():
        figure1(out, vals)
    else:
        vals["fig1_acquisitions"] = "–"
    fig1(out, vals)
    fig3(out, fr, vals)
    fig4(out, vals)
    fig5(out, fr, vals)
    fig7(out, vals)
    fig8(out, fr, vals)
    methane_v2(vals)
    tables["table_test"] = table_test(fr, vals, final)
    VA = BENCH / "val" / "parts_all"
    vf = {m: load([VA / f"{f}.parquet"], m) for m, f in (("climatology", "climatology"), ("mds", "mds"), ("irvin_rf", "irvin_rf"),
          ("vekuri_de", "vekuri_de"), ("liu_xgb", "liu_xgb"), ("saits", "saits"), ("fluxfill", "fluxfill_latest"))}
    vf = {m: f[f.variable.isin(FLUXES)] for m, f in vf.items()}
    tables["table_val"] = table_test(vf, vals, True, prefix="val")
    timing = timing_values()
    vals["timing"] = {k: f"{v:.2f}" for k, v in timing.items()}
    tables["table_methods"] = table_methods({k.replace("fluxfill-a0", "fluxfill_a0"): v for k, v in timing.items()})
    (out / "tables.json").write_text(json.dumps(tables))
    (out / "values.json").write_text(json.dumps(vals, indent=1, default=str))
    print(json.dumps(vals, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
