"""What an agent needs to know after it picks a capability to improve.

The loop's step ② is "pick one capability from scorecard.md, with intention". Everything after that
used to require reading a 1,500-line `main()` to answer three questions:

    which probe modes measure this capability?
    what flags select each one, and what do they REQUIRE to run?
    where do I edit to change the mechanism vs the data channel?

Those answers were only discoverable by grep, and getting them wrong is cheap-looking and expensive:
eight of nineteen modes silently require `--forecast` via a bare `assert` buried mid-function, and
`--phenology` shadows `--pheno_env`/`--pheno_taxon`/`--pheno_densefield` entirely because its branch
returns first. This module states all of it in one place.

A `records=False` mode is a legitimate diagnostic that can never set a record -- either because its
target is not on the scorecard, or because it evaluates on raw coordinate features with Earth4D absent
from the comparison.

The five dynamics/AR modes that used to appear here (BREADTH, PROPAGATOR-ARCH, FIRST-ARRIVAL,
ABUNDANCE, AR-ROLLOUT, CONTINUOUS-LEAD) were DELETED: ~1,300 lines of instrument/ that could never
move a scorecard capability, plus 17 flags for them. They are in git history if a real target ever
needs them.

Usage:
    python -m deepearth.autoresearch.spacetime.editable_files.harness.probe_registry --capability family_from_env
    python -m deepearth.autoresearch.spacetime.editable_files.harness.probe_registry --all
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

DATA = "DATA"
ARCH = "ARCHITECTURE"
BOTH = "DATA+ARCHITECTURE"


@dataclass(frozen=True)
class Mode:
    """One probe mode: what it measures, how to select it, and where to change it."""

    mode: str                      # the declared mode string; part of measurement identity
    flags: str                      # flags that select this mode
    capability: str = ""            # "" means diagnostic
    requires: Tuple[str, ...] = ()  # flags this mode asserts on, currently as bare asserts
    lever: str = BOTH               # which lever family this mode can actually move
    records: bool = True
    reason: str = ""                # why it cannot record, when records is False
    notes: str = ""

    @property
    def is_diagnostic(self) -> bool:
        return not self.records


MODES: Tuple[Mode, ...] = (
    # ---- family_from_env -------------------------------------------------------------------------
    Mode("ENV(<split>)", "--env [--env_channels {all,worldclim,alphaearth,wcsoil,...}] [--env_extra]",
         capability="family_from_env", lever=BOTH,
         notes="Primary is the FUSED Earth4D+ENV accuracy. Earth4D alone currently LOSES to RFF "
               "(0.0938 vs 0.1010), so the record is carried by the env channel -- label it."),
    Mode("ENV-DECODE(<split>)", "--env_decode [--env_aux_weight W]",
         capability="family_from_env", lever=ARCH,
         notes="Trains the encoder end-to-end against an auxiliary env field. NOT reproducible "
               "run-to-run (gains move ~0.005), so a single-seed record here means nothing."),

    # ---- family_from_spacetime -------------------------------------------------------------------
    Mode("FORECAST(past->future)", "--forecast [--target family]",
         capability="family_from_spacetime", lever=ARCH,
         notes="The default coordinate path. --forecast_spatial switches to future+newplace."),
    Mode("FIELD-DECODE(<split>)", "--field_decode",
         capability="family_from_spacetime", lever=ARCH,
         notes="Trains the encoder end-to-end (rule 24). Also not reproducible run-to-run."),
    Mode("GNN(message-passing propagator)", "--gnn [--gnn_hops H] [--rec_k K]",
         capability="family_from_spacetime", requires=("--forecast",), lever=ARCH,
         notes="Declares the ENCODER gain (Earth4D GNN vs generic-PE GNN). propagator_gain, in "
               "extras, is propagation-vs-static on RAW features and gates nothing about Earth4D."),
    Mode("RECURRENCE(4D-LSTM rollout past->future)", "--recurrence [--rec_k K] [--rec_hidden H]",
         capability="family_from_spacetime", requires=("--forecast",), lever=ARCH,
         notes="science.md rule 2b. Currently negative vs RFF."),

    # ---- species_from_spacetime ------------------------------------------------------------------
    Mode("FORECAST(past->future)", "--forecast --target species",
         capability="species_from_spacetime", lever=ARCH,
         notes="Same tail as family; --target selects the capability and the metric name."),

    # ---- species_from_env ------------------------------------------------------------------------
    Mode("SDM-PRESENCE", "--sdm_presence [--cooccur_mech {env,space,both}]",
         capability="species_from_env", lever=DATA),
    Mode("SDM-HARD", "--sdm_hard [--sdm_channels ...] [--sdm_cell_deg D] [--sdm_holdout_mode ...]",
         capability="species_from_env", lever=DATA,
         notes="A DIFFERENT measurement from SDM-PRESENCE (0.3336 vs 0.6275 at 12 shards). "
               "Identity keeps them apart; never compare the two."),

    # ---- community_from_env ----------------------------------------------------------------------
    Mode("COOCCUR-ROUTING", "--cooccur [--cooccur_mech ...] [--cooccur_channels ...] [--cooccur_thresh N]",
         capability="community_from_env", lever=DATA),

    # ---- flowering_peak_month --------------------------------------------------------------------
    Mode("PHENOLOGY-FUTURE / -FUTURE-HELD / -HELD", "--phenology [--pheno_feats e4d,rff,raw] [--pheno_tol D]",
         capability="flowering_peak_month", requires=("--forecast",), lever=ARCH,
         notes="Record metric is Earth4D's best-head within-tolerance accuracy vs the generic PE's. "
               "WARNING: this branch returns FIRST, so --pheno_env/--pheno_taxon/--pheno_densefield "
               "are silently ignored whenever --phenology is passed."),

    # ---- calibration -----------------------------------------------------------------------------
    Mode("CALIBRATION", "--feature earth4d --ensemble N   (module: calib_probe, not probe)",
         capability="calibration", lever=ARCH,
         notes="Lives in calib_probe.py and reports conf_auroc (0.5 = useless). The live 0.591 "
               "record has NO fair baseline, so its bottleneck is undiagnosable. Not yet on the "
               "result contract."),

    # ---- diagnostics: not scorecard capabilities -------------------------------------------------
    Mode("ENV->NICHE-TRAIT(agg=<agg>)", "--env_trait [--env_agg {mean,medoid}] [--env_head {ridge,mlp}]",
         records=False, lever=DATA,
         reason="trait Spearman over species aggregates is not a scorecard capability"),
    Mode("ENV-CONSTRUCT(<construct><-<feature>)",
         "--env_construct [--construct {rarity,ease,ns_grank,crpr}] [--construct_feature ...]",
         records=False, lever=DATA,
         reason="a species-level construct, not a scorecard capability"),
    Mode("PHENO-ENV(mean-DOY)", "--pheno_env", requires=("--forecast", "not --phenology"),
         records=False, lever=DATA,
         reason="runs on RAW spatial features only; Earth4D is not in the comparison"),
    Mode("PHENO-DISTTARGET(<target>)", "--pheno_disttarget <target>",
         requires=("--forecast", "not --phenology"), records=False, lever=DATA,
         reason="runs on RAW spatial features only; Earth4D is not in the comparison",
         notes="This is the mode whose peak_week target once published 0.067 -> 0.683 by being "
               "scraped into flowering_peak_month's record."),
    Mode("PHENO-BY-TAXON(<col>)", "--pheno_taxon <col>", requires=("--forecast", "not --phenology"),
         records=False, lever=DATA,
         reason="runs on RAW spatial features only; Earth4D is not in the comparison"),
    Mode("PHENO-DENSEFIELD(mean-DOY, same-cell-EXCLUDED)", "--pheno_densefield",
         requires=("--forecast", "not --phenology"), records=False, lever=ARCH,
         reason="runs on RAW spatial features only; Earth4D is not in the comparison"),
)

# Where to make each kind of change. An agent that has picked a capability needs this more than it
# needs the file layout.
LEVER_SITES = {
    DATA: [
        "autoresearch/spacetime/editable_files/harness/probe.py: load_env / load_vision / load_env_species "
        "(which channel feeds the head)",
        "flags: --env_channels, --env_extra, --sdm_channels, --vision --vision_feats, --pheno_channel",
        "data prep: occurrence densification, channel fusion, per-entity aggregation",
    ],
    ARCH: [
        "encoders/spacetime/earth4d.py: __init__, forward, training objective (the encoder itself)",
        "autoresearch/spacetime/editable_files/lib/recurrence.py: run_recurrence, run_field_decode, propagators",
        "autoresearch/spacetime/editable_files/lib/gnn.py: message passing",
        "flags: --recurrence, --gnn, --forecast, --env_decode, --field_decode, --fourier, "
        "--spatial_siren, --time_harmonics",
    ],
}


def for_capability(capability: str) -> Tuple[Mode, ...]:
    """Every mode that can set a record for this capability."""
    return tuple(m for m in MODES if m.capability == capability and m.records)


def diagnostics() -> Tuple[Mode, ...]:
    return tuple(m for m in MODES if not m.records)


def capabilities() -> Tuple[str, ...]:
    seen = []
    for m in MODES:
        if m.capability and m.capability not in seen:
            seen.append(m.capability)
    return tuple(seen)


def describe(capability: str) -> str:
    modes = for_capability(capability)
    if not modes:
        known = "\n  ".join(capabilities())
        return (f"no recording mode measures {capability!r}.\n"
                f"capabilities with modes:\n  {known}")
    out = [f"=== {capability} — {len(modes)} mode(s) can set this record ==="]
    for m in modes:
        out.append(f"\n  mode     {m.mode}")
        out.append(f"  select   {m.flags}")
        if m.requires:
            out.append(f"  REQUIRES {' '.join(m.requires)}")
        out.append(f"  lever    {m.lever}")
        if m.notes:
            out.append(f"  note     {m.notes}")
    levers = {m.lever for m in modes}
    out.append("\n--- where to change things ---")
    for lever in (DATA, ARCH):
        if any(lever in l for l in levers):
            out.append(f"  {lever}:")
            out.extend(f"    - {site}" for site in LEVER_SITES[lever])
    out.append("\nA run is comparable to the record only if capability, mode, split, n_shards and "
               "protocol all match. Changing mode is a new measurement, not a better score.")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--capability", default="")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    if a.capability:
        print(describe(a.capability))
        return
    if a.all:
        for capability in capabilities():
            print(describe(capability))
            print()
        print("=== diagnostics — cannot set any record ===")
        for m in diagnostics():
            print(f"  {m.mode:52} {m.reason}")
        return
    print(f"capabilities: {', '.join(capabilities())}")
    print(f"{len(diagnostics())} diagnostic modes cannot record. Use --capability X or --all.")


if __name__ == "__main__":
    main()
