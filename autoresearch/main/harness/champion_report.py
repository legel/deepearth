"""champion_report.py — the STANDARD way to save champion benchmark scores and format a git-commit report.

Every time a benchmark run becomes the new champion, run this. It (1) reads the run's scores, (2) diffs them
against the previous champion record (autoresearch/main/records/champion_scores.json), (3) prints a ready-to-paste commit
message — a headline with the net score BEFORE -> AFTER, then an enumerated per-benchmark BEFORE -> AFTER list —
and (4) with --save, promotes the run to the new champion record (the "before" for the next upgrade).

This makes every champion commit read consistently (see science.md rule 30). Example headline:
    "Latent Clade Attention doubled in size (harmonic 0.0204 -> 0.0231, arith 0.446 -> 0.461)"

Usage:
    # after a benchmark run whose stdout/eval was captured to run.log:
    python -m deepearth.autoresearch.main.harness.champion_report --log run.log --desc "widen d_model 256->384" --save
    # prints the commit message and promotes run.log's scores to the champion record.
    # omit --save to preview the report without changing the record (e.g. a candidate that did not win).
"""
import re
import json
import argparse
from typing import Optional
from pathlib import Path

# The champion record lives in main/records/, NOT beside this file. `Path(__file__).with_name(...)`
# resolved to harness/champion_scores.json — a path that has never existed — so
# `old` was always None, every report printed "BASELINE" instead of BEFORE -> AFTER, and --save wrote a
# second record the loop then ignored. That is rule 30's entire mechanism, silently off.
# Anchored by NAME so a directory move cannot re-break it.
RECORD = next(p for p in Path(__file__).resolve().parents
              if p.name == "autoresearch") / "main" / "records" / "champion_scores.json"

try:                                                        # canonical order so EVERY benchmark is listed (inactive ones marked, never silently missing)
    from deepearth.autoresearch.main.harness.evaluate import (
        BENCHMARKS as _CANON, BENCHMARK_PROTOCOL, capability_suite,
    )
except Exception:
    _CANON = []
    BENCHMARK_PROTOCOL = "unknown"

    def capability_suite(raw):
        return tuple(sorted(k for k in raw if not k.endswith("_gain")
                            and k != "B55_pollinator_phylo_transfer_recall"))

# One-line description per benchmark (given-set -> target, metric), so a reader grasps whole-system performance at a
# glance. "env" = the location's full environment vector U; "phylo graph gain" = ablation delta from species-graph refinement.
DESC = {
    "B1_species_from_env_top10": "env -> species (SDM), top-10 acc",
    "B2_species_from_photo_top1": "env + ground photo -> species (flagship), top-1",
    "B3_species_from_photo_top5": "env + ground photo -> species, top-5",
    "B4_species_from_photo_only_top1": "photo only -> species, top-1",
    "B5_species_from_spacetime_top10": "bare space-time -> species, top-10",
    "B6_family_from_env": "env -> family (niche determinism), acc",
    "B7_family_from_phylo": "phylo embedding -> family, acc",
    "B8_family_from_spacetime": "bare space-time -> family, acc",
    "B9_phylo_from_photo_cos": "env + photo -> phylo/evolutionary vector, cosine",
    "B10_traits_from_photo_env_f1": "env + photo -> traits, macro-F1",
    "B11_traits_from_photo_f1": "photo only -> traits, macro-F1",
    "B12_traits_leave_one_out_f1": "all-but-trait -> trait, macro-F1",
    "B13_imagine_vision_cos": "non-vision -> ground-vision (DINO), cosine",
    "B14_vision_leave_one_out_cos": "all-but-vision -> ground-vision, cosine",
    "B15_vision_from_aerial_cos": "aerial (NAIP) -> ground-vision, cosine",
    "B16_infer_clay_cos": "U-minus-clay -> clay (Sentinel-2), cosine",
    "B17_infer_soil_cos": "U-minus-soil -> soil (SSURGO), cosine",
    "B18_infer_climate_cos": "U-minus-climate -> climate (Daymet), cosine",
    "B19_infer_aerial_cos": "U-minus-NAIP -> aerial (NAIP), cosine",
    "B20_community_from_env_recall": "env -> local community set, recall@10",
    "B21_community_from_species_recall": "focal species -> co-occurring set, recall@10",
    "B22_companions_recall": "species + env -> companions, recall@10",
    "B23_species_calibration_mrr": "env -> species posterior, mean reciprocal rank",
    "B24_geo_information_gain": "species info gained from location (B2 - B1)",
    "B25_forecast_climate_cos": "future climate (temporal holdout), cosine",
    "B31_forecast_vision_cos": "future ground-vision/appearance (temporal holdout), cosine",
    "B26_flowering_auc": "env/imagined-vision -> flowering, ROC-AUC",
    "B27_flowering_fidelity": "flowering agreement: imagined vs real vision",
    "B28_flowering_peak_month_mrr": "true peak-flowering month via 12-month sweep, MRR",
    "B29_species_dist_30m_skill": "per-cell 30 m SDM skill = 1 - KL(true||pred)/KL(true||uniform)",
    "B30_seasonality_trait_f1": "seasonality trait, macro-F1",
    "B38_water_soil_regime_f1": "water + soil-drainage regime, macro-F1",
    "B39_species_dist_3km_skill": "per-cell 3 km SDM skill",
    "B40_species_dist_300m_skill": "per-cell 300 m SDM skill",
    "B49_form_trait_f1": "growth-form trait, macro-F1",
    "B37_imagine_vision_bio_cos": "non-vision -> BioCLIP-2 ground-vision, anomaly cosine",
    "B45_vision_bio_leave_one_out_cos": "all-but-vision_bio -> BioCLIP-2 ground-vision, anomaly cosine",
    "B32_plant_type_trait_f1": "plant-type/habit trait, macro-F1",
    "B33_growth_rate_trait_f1": "growth-rate trait, macro-F1",
    "B35_sun_trait_f1": "sun/light-requirement trait, macro-F1",
    "B36_ease_of_care_trait_f1": "ease-of-care/niche-breadth trait, macro-F1",
    "B34_lfmc_from_env": "peak fire-season live fuel moisture from env",
    "B42_mycorrhiza_from_env": "mycorrhizal type (AM/EcM/ErM/OM/NM) from env, macro-F1",
    "B41_pollinator_from_species_recall": "plant identity + env -> local pollinators (GloBI), recall@10",
    "B43_infer_hydro_cos": "U-minus-hydro -> drainage/wind, cosine",
    "B44_infer_topo_cos": "U-minus-topo -> 3DEP microtopography, anomaly cosine",
    "B46_infer_chm_cos": "U-minus-chm -> NAIP-CHM canopy height/structure, anomaly cosine",
    "B47_infer_naip_ir_cos": "U-minus-both-aerial -> NAIP-IR aerial, anomaly cosine",
    "B50_pollinator_from_spacetime_recall": "bare location -> pollinators, recall@10",
    "B51_pollinator_from_env_recall": "env only -> pollinators, recall@10",
    "B48_pollinator_from_photo_only_recall": "ground photo only -> pollinators, recall@10",
    "B52_pollinator_from_photo_recall": "env + ground photo -> pollinators, recall@10",
    "B53_pollinator_calibration_mrr": "pollinator posterior calibration, MRR",
    "B54_pollinator_dist_kl": "predicted vs true pollinator frequency, exp(-KL)",
    "B55_pollinator_phylo_transfer_recall": "plant's pollinators from relatives' pollinators, recall@10",
    "B56_family_phylo_graph_gain": "family-from-phylo acc gained from species-graph refinement",
    "B57_flowering_phylo_graph_gain": "flowering-AUC gained from species-graph refinement",
    "B58_lfmc_phylo_graph_gain": "LFMC correlation gained from species-graph refinement",
    "B59_pollinator_phylo_graph_gain": "pollinator recall gained from species-graph refinement",
    "B60_community_phylo_graph_gain": "env->community recall gained from species-graph refinement",
    "B61_trait_phylo_graph_gain": "trait macro-F1 gained from species-graph refinement",
    "B62_mycorrhiza_phylo_graph_gain": "mycorrhiza macro-F1 gained from species-graph refinement",
    "B63_myco_from_species_f1": "mycorrhiza imputation given species identity, macro-F1",
    "B64_family_phylo_masked_imputation": "seed-masked species -> family from relatives, NN accuracy",
    "B65_myco_phylo_masked_imputation_f1": "seed-masked species -> mycorrhiza from relatives, macro-F1",
    "B66_community_phylo_masked_recall": "seed-masked species -> community from relatives, recall@10",
    "B67_pollinator_phylo_masked_recall": "seed-masked plant -> own pollinators from relatives, recall@10",
}


def parse_run(log_path: str) -> dict:
    """Extract {Bxx_name: score}, harmonic (net_score) and arithmetic mean from a train/eval run log."""
    txt = Path(log_path).read_text()
    protocol_match = re.search(r"^BENCHMARK PROTOCOL:\s*(\S+)", txt, re.M)
    protocol = protocol_match.group(1) if protocol_match else None
    seed_match = re.search(r"^training_seed:\s*(-?\d+)", txt, re.M)
    seed = int(seed_match.group(1)) if seed_match else None
    steps_match = re.search(r"^trained\s+(\d+)\s+steps\b", txt, re.M)
    vram_match = re.search(r"^peak_vram_mb:\s+([0-9.]+)", txt, re.M)
    receipt_match = re.search(r"^RUN RECEIPT:\s*(\{.*\})$", txt, re.M)
    receipt = json.loads(receipt_match.group(1)) if receipt_match else None
    meta = {"training_seed": seed,
            "steps": int(steps_match.group(1)) if steps_match else None,
            "peak_vram_mb": float(vram_match.group(1)) if vram_match else None,
            "receipt": receipt}
    scores = {}
    # score may be followed by trailing text on the diagnostic lines, e.g. "B24_geo_information_gain 0.593 (net
    # contrib 0.997)" -- match the score after the name, not requiring end-of-line, so B24/B56-B62 are captured.
    for m in re.finditer(r"^\s*(B\d+_\w+)\s+(-?[0-9.]+)(?:\s|$)", txt, re.M):
        scores[m.group(1)] = float(m.group(2))                 # last occurrence wins (final eval)
    try:                                                       # RECOMPUTE the net from scores with the live logic, so
        from deepearth.autoresearch.main.harness.evaluate import net_score, arithmetic_net   # every champion record is comparable
        return {"benchmark_protocol": protocol, **meta, "scores": scores,
                "harmonic": float(net_score(scores)), "arithmetic": float(arithmetic_net(scores))}
    except Exception:                                          # fallback: parse whatever the log printed
        h = re.search(r"net_score:\s+([0-9.]+)", txt)
        a = re.search(r"arithmetic mean:\s+([0-9.]+)", txt)
        return {"benchmark_protocol": protocol, **meta, "scores": scores, "harmonic": float(h.group(1)) if h else None,
                "arithmetic": float(a.group(1)) if a else None}


def _n(x):                                                     # benchmark sort key: B<number>
    return int(re.match(r"B(\d+)", x).group(1))


def _f(v):
    return f"{v:.3f}" if v is not None else "  -  "


def format_commit(new: dict, old: Optional[dict], desc: str, config: str = "") -> str:
    ns = new["scores"]
    os_ = (old or {}).get("scores", {})
    oh, nh = (old or {}).get("harmonic"), new["harmonic"]
    oa, na = (old or {}).get("arithmetic"), new["arithmetic"]
    if old is None:                                            # first record -> a baseline report (no "before")
        head = f"{desc} (BASELINE: harmonic {_f(nh)}, arith {_f(na)})"
    else:
        head = f"{desc} (harmonic {_f(oh)} -> {_f(nh)}, arith {_f(oa)} -> {_f(na)})"
    lines = [head, ""]
    if config:
        lines += [config, ""]
    lines.append("All benchmarks (before -> after):" if old is not None else "All benchmarks (baseline):")
    allb = sorted(set(_CANON) | set(ns) | set(os_), key=_n)   # EVERY declared benchmark + anything seen; none missing
    for i, name in enumerate(allb, 1):
        after = ns.get(name)
        before = os_.get(name)
        desc = DESC.get(name, "")                             # rule 30: each line describes what the benchmark measures
        if after is None:                                     # not produced by THIS run's holdout (e.g. B25/B31 forecast need a temporal run)
            row = (f"{name}: {_f(before)} (carried; its holdout not re-run here)" if before is not None
                   else f"{name}: inactive (needs its holdout: e.g. B25/B31 = temporal)")
        elif old is None:
            row = f"{name}: {_f(after)}"
        else:
            d = f"{after - before:+.3f}" if before is not None else "  new  "
            flag = "" if before is None else ("  ^" if after > before + 1e-9 else ("  v" if after < before - 1e-9 else "  ="))
            row = f"{name}: {_f(before)} -> {_f(after)} ({d}){flag}"
        lines.append(f"{i:>2}. {row}" + (f"  -- {desc}" if desc else ""))
    if old is not None:
        # A comparison is valid only over the same human-capability suite. Optional mechanism
        # diagnostics remain reportable evidence but cannot create or invalidate a score change.
        old_suite, new_suite = set(capability_suite(os_)), set(capability_suite(ns))
        added, missing = sorted(new_suite - old_suite), sorted(old_suite - new_suite)
        if added or missing:
            lines += ["", "*** SUITE CHANGED — these two runs did not score the same benchmarks, so the",
                      "    net comparison above is NOT valid. Re-run both sides the same way.",
                      f"    added:   {', '.join(added) or '(none)'}",
                      f"    missing: {', '.join(missing) or '(none)'}"]
    if old is not None:                                        # regression guard summary (science.md: NO metric regressing)
        reg = [f"{n} ({os_[n]:.3f}->{ns[n]:.3f})" for n in sorted(ns, key=_n)
               if n in os_ and ns[n] < os_[n] - 0.005]
        lines += ["", f"REGRESSIONS (>0.005): {', '.join(reg) if reg else 'none'}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="run log (train + eval stdout) to score")
    ap.add_argument("--desc", required=True, help="one-line result description for the commit headline")
    ap.add_argument("--config", default="", help="optional config/hardware note line")
    ap.add_argument("--save", action="store_true", help="promote this run to the champion record")
    a = ap.parse_args()
    new = parse_run(a.log)
    if not new["scores"]:
        raise SystemExit(f"no Bxx scores found in {a.log}")
    if new.get("benchmark_protocol") != BENCHMARK_PROTOCOL:
        raise SystemExit(f"run log benchmark protocol {new.get('benchmark_protocol')!r} does not match "
                         f"the live protocol {BENCHMARK_PROTOCOL!r}; re-run with the current evaluator")
    old = json.loads(RECORD.read_text()) if RECORD.exists() else None
    comparable_old = old if (old or {}).get("benchmark_protocol") == BENCHMARK_PROTOCOL else None
    if old is not None and comparable_old is None:
        print(f"[benchmark protocol changed: {(old or {}).get('benchmark_protocol')!r} -> "
              f"{BENCHMARK_PROTOCOL!r}; establishing a new baseline]\n")
    print(format_commit(new, comparable_old, a.desc, a.config))
    if a.save:
        import getpass, datetime
        hist = (old or {}).get("history", [])
        hist.append({"user": getpass.getuser(),
                     "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                     "label": a.desc, "config": a.config, "benchmark_protocol": BENCHMARK_PROTOCOL,
                     "harmonic": new["harmonic"],
                     "arithmetic": new["arithmetic"], "scores": new["scores"]})   # append every champion -> both users' records plot over time
        RECORD.write_text(json.dumps({"label": a.desc, "config": a.config,
                                      "benchmark_protocol": BENCHMARK_PROTOCOL, "harmonic": new["harmonic"],
                                      "arithmetic": new["arithmetic"], "scores": new["scores"],
                                      "history": hist}, indent=2))
        print(f"\n[champion_scores.json updated: {len(new['scores'])} benchmarks, "
              f"harmonic {new['harmonic']}, arith {new['arithmetic']}; history={len(hist)} records]")


if __name__ == "__main__":
    main()
