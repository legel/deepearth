"""National evaluation of the occurrence-trained range maps against Daru (2024) and independent plots.

Scans every rendered species of the configured product directories (``per_species.products``) and reports:
  * coverage: species rendered per US grid (CONUS, Alaska, Hawaii);
  * against Daru's published maps (species in daru_ref_all, Dryad SDM_set1): agreement (κ, Jaccard, suitability
    Spearman), share of each map inside the species' WCVP native regions, and — where BLM AIM or FIA have ≥ 20
    presence plots — AUC and TSS of ours vs Daru's at the same plots (from render.json);
  * VegBank (all US plots with public coordinates): AUC of every species with ≥ 20 presence plots, decoded from its
    range card at the plots (codec.decode_cells), and Daru's published map at the same plots where it exists.
Writes work/national/national_eval.json and work/national/national_eval_species.csv.

usage: national_eval.py [--config configs/conus.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import codec, config, validate  # noqa: E402
from ranges.predictors import ConusStack  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    cfg = config.load(ap.parse_args().config)
    R = cfg.root
    DIRS = {Path(d).parent.name + "/" + Path(d).name: cfg.path(d) for d in cfg["per_species"]["products"]}
    rows = []
    for run, root in DIRS.items():
        if not root.exists():
            continue
        for wd in root.iterdir():
            card = wd / f"{wd.name}.rangecard"
            if not card.exists() or not (wd / "render.json").exists():
                continue
            r = json.loads((wd / "render.json").read_text())
            st = r.get("render", {})
            rec = {"run": run, "label": wd.name, "species": wd.name.replace("_", " "),
                   "conus": st.get("cells_vote", 0) > 0, "alaska": st.get("cells_vote_alaska", 0) > 0,
                   "hawaii": st.get("cells_vote_hawaii", 0) > 0}
            if "vs_daru" in r:
                v = r["vs_daru"]
                rec.update(kappa=v.get("kappa"), jaccard=v.get("jaccard"), spearman=v.get("spearman_suitability"),
                           inside_native_ours=r["wcvp"]["ours"]["share_predicted_inside_native_l3"],
                           inside_native_daru=r["wcvp"]["daru"]["share_predicted_inside_native_l3"])
            for src in ("aim", "fia"):
                if src in r:
                    rec[f"{src}_auc_ours"] = r[src]["ours_240m_all_plots"].get("AUC")
                    rec[f"{src}_auc_daru"] = r[src]["daru_10min_all_plots"].get("AUC")
                    rec[f"{src}_tss_ours_p5"] = r[src]["ours_240m_p5_all_plots"]["TSS"]
                    rec[f"{src}_tss_daru"] = r[src]["daru_10min_all_plots"]["TSS"]
            rows.append(rec)
    df = pd.DataFrame(rows).drop_duplicates("species", keep="last")
    print(f"rendered species: {len(df):,} ({df.run.value_counts().to_dict()})", flush=True)

    # VegBank: decode each card at the plots
    truth = validate.PlotTruth(vegbank_dir=R / "raw/validation/vegbank", wcvp_dir=R / "raw/wcvp")
    locs = truth.vb_locs.reset_index(drop=True)
    stack = ConusStack(R / "work/conus240")
    t = stack.profile["transform"]
    x, y = Transformer.from_crs(4326, 5070, always_xy=True).transform(locs.lon.values, locs.lat.values)
    rows_, cols_ = np.floor((t.f - y) / -t.e).astype(np.int64), np.floor((x - t.c) / t.a).astype(np.int64)
    daru = {p.stem for p in (R / "daru_ref_all/rasters").glob("*.tif")}
    vb = {}
    for _, rec in df.iterrows():
        tr = truth.vegbank(rec.species)
        if tr is None or tr.present.sum() < 20:
            continue
        yv = tr.present.values
        card = DIRS[rec.run] / rec.label / f"{rec.label}.rangecard"
        v = codec.decode_cells(card, stack, rows_, cols_)
        s = np.where(v["suitability"] > 0, (v["suitability"].astype(float) - 1) / 254, 0.0)
        out = {"vb_auc_ours": validate._scores(yv, v["binary_p5"] == 2, s).get("AUC"),
               "vb_tss_ours_p5": validate._scores(yv, v["binary_p5"] == 2, None)["TSS"], "vb_presences": int(yv.sum())}
        if rec.species in daru:
            dr = validate._sample(R / f"daru_ref_all/raw_rasters/{rec.species}.tif", locs.lon.values, locs.lat.values)
            db = validate._sample(R / f"daru_ref_all/rasters/{rec.species}.tif", locs.lon.values, locs.lat.values)
            out.update(vb_auc_daru=validate._scores(yv, np.nan_to_num(db) == 1, np.nan_to_num(dr)).get("AUC"),
                       vb_tss_daru=validate._scores(yv, np.nan_to_num(db) == 1, None)["TSS"])
        vb[rec.species] = out
    df = df.merge(pd.DataFrame.from_dict(vb, orient="index"), left_on="species", right_index=True, how="left")
    df.to_csv(R / "work/national/national_eval_species.csv", index=False)

    def med(c, d=df):
        x = d[c].dropna()
        return round(float(x.median()), 4) if len(x) else None

    summ = {"species_rendered": int(len(df)), "conus": int(df.conus.sum()), "alaska": int(df.alaska.sum()),
            "hawaii": int(df.hawaii.sum()), "vegbank_species": int(df.vb_auc_ours.notna().sum()),
            "vegbank_auc_ours": med("vb_auc_ours"), "vegbank_tss_ours_p5": med("vb_tss_ours_p5")}
    for src in ("vb", "aim", "fia"):
        pair = df.dropna(subset=[f"{src}_auc_ours", f"{src}_auc_daru"]) if f"{src}_auc_daru" in df else df.iloc[0:0]
        if len(pair):
            summ[f"{src}_vs_daru"] = {"species": int(len(pair)), "auc_ours": med(f"{src}_auc_ours", pair),
                                       "auc_daru": med(f"{src}_auc_daru", pair),
                                       "better_share": round(float((pair[f"{src}_auc_ours"] > pair[f"{src}_auc_daru"]).mean()), 3)}
    if "kappa" in df:
        k = df.dropna(subset=["kappa"])
        summ["daru_agreement"] = {"species": int(len(k)), "kappa": med("kappa", k), "spearman": med("spearman", k),
                                  "inside_native_ours": med("inside_native_ours", k),
                                  "inside_native_daru": med("inside_native_daru", k)}
    (R / "work/national/national_eval.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
