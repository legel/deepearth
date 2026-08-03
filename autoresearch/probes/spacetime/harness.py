"""The spacetime loop, in one file.

This is the harness: it declares what a run measures, runs the probe, decides whether the result may
become a record, writes the ledger, and publishes to the swarm. It was six files
(probe_contract, probe_emit, probe_registry, trace, plus two mode modules) which meant six places to
look for one question. A harness should be one file you can read start to finish.

    READ ──► PICK ──► DIAGNOSE ──► RUN ──► MEASURE ──► DECIDE ──► WRITE

Sections, in that order:

  1. CAPABILITIES   what may be optimized, and what is refused, with the reason
  2. CONTRACT       ProbeResult: identity, validation, fair-gain, rendering
  3. EMIT           declare() -- the ONE path by which a number becomes recordable
  4. REGISTRY       capability -> modes -> what each needs -> where to edit
  5. GATE           like-for-like comparison, re-baseline, atomic board commit
  6. PUBLISH        Ensue upsert
  7. CLI            the loop itself

What the harness enforces is measurement identity, not a menu of permitted edits: a run is comparable
to a record only when capability, mode, split, n_shards and protocol all match. Change anything the
hypothesis needs; a mismatch is recorded as a re-baseline or withheld, never as a win.

Editing THIS file changes what a number means. Do that as its own commit, with a test that fails before
and passes after -- never inside an experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# Paths are derived by NAME, not by counting parents. Every restructure this file has been through broke
# a parents[N]: REPO once resolved to "/" and the board once resolved outside its own loop, which made
# trace.py mint a record against an empty file. Named anchors cannot drift when a directory moves.
_HERE = Path(__file__).resolve()
AUTORESEARCH = next(p for p in _HERE.parents if p.name == "autoresearch")
REPO = AUTORESEARCH.parent                  # the deepearth package root
# The loop this harness JUDGES, named outright. It used to be `_HERE.parents[1]`, which was correct only
# while the harness lived inside the loop it scored. Moving it to autoresearch/scoring/ -- so that an
# experiment cannot edit its own judge -- made parents[1] resolve to autoresearch/, i.e. RECORDS pointed
# at autoresearch/records/records.json: a file that does not exist, which the harness would have created
# empty, found no prior record in, and reported "RECORD = YES (new best!)" against for every capability.
# Exactly the failure this comment block was already warning about. A judge that lives outside its
# subject must name its subject.
LOOP = AUTORESEARCH / "probes" / "spacetime"
assert (LOOP / "program").is_dir(), f"spacetime loop not found at {LOOP}"
assert REPO.name == "deepearth", f"expected the deepearth package root, resolved {REPO}"
sys.path.insert(0, str(REPO.parent))        # dir holding the deepearth package



# ============================================================================================================
# 1b. FAIR BASELINE — what every `vs RFF` gain is measured against
# ============================================================================================================
#
# The fair baseline — what every `vs RFF` gain is measured against.
#
# This is not a helper. It defines what "fair-gain" MEANS on this board, and therefore what every EARNING
# or ENCODER-LIMITED read is asserting. It lives apart from probe.py so it can be tested without CUDA,
# because a baseline nobody can test is how the previous one stayed broken.
#
# Folded in from lib/fair_baseline.py: 46 lines in their own module, imported by exactly one
# caller, is not a boundary — it is a file to keep in sync.
import numpy as np
import torch

# The control's width is now FIXED instead of tracking the encoder's.
#
# Both call sites used to pass `e4d.shape[1]`, so the baseline's width moved whenever any arm changed
# the encoder's output width -- and RFF accuracy is non-monotone in width. Measured: padding the encoder
# with columns of LITERAL ZEROS, adding no information at all, moved family_from_spacetime's share from
# 20.7% (dim 2592) to 27.2% (dim 3024) to 15.1% (dim 3744). Since `share = fair_gain / score` is the
# number that chooses DATA vs ARCHITECTURE, an unknown fraction of every EARNING / ENCODER-LIMITED read
# on this board was an artifact of output width.
#
# v4 set this to 2592, the width family_from_spacetime's record was set at, on the reasoning that ONE
# control everywhere makes gains comparable across rows. That reasoning was sound while encoder widths
# varied wildly per capability (144 to 20,663).
#
# Under v5 they no longer vary. The probe builds exactly what fusion.py:302 builds -- 36 spatial + 108
# tri-plane = 144 dims -- for EVERY capability, with the bolt-on bases off. So the two goals stop
# competing: 144 is both a fixed protocol constant AND matched to the encoder, and Earth4D vs RFF is
# finally a head-to-head at equal width.
#
# Leaving it at 2592 would have handed the control 18x the encoder's capacity and crushed every gain --
# the same class of error as the old dimension-MATCHED control, pointing the other way.
#
# If an experiment turns a bolt-on basis back on, the encoder gets wider and the control does NOT
# follow. That is correct: the control is a fixed reference, and an arm that buys width has to earn it.
FAIR_CONTROL_DIM = 144


def fair_rff(rn: np.ndarray, out_dim: int, train_mask=None, seed: int = 0,
             bandwidths=(1.0, 4.0, 16.0, 64.0, 256.0, 1024.0)):
    """A random-Fourier control that is actually FAIR.

    The old control was `rn @ N(0, 8)` where rn is (lat/90, lon/180): coordinates normalized to the
    GLOBE. On a regional corpus that is degenerate. California spans ~9.5 deg of latitude, i.e. ~0.05 of
    the normalized range, so at sigma=8 the projection varies ~0.04 CYCLES across the entire dataset --
    every sample gets nearly the same feature. Measured: it scored 0.008, BELOW the raw-coordinate
    baseline at 0.0166. A nonlinear control that loses to raw coordinates is not a control; it is a
    handicap, and every `vs RFF` gain computed against it was inflated by that handicap.

    Two fixes, both of which just extend to the baseline the courtesy the encoder already gets:

      1. Normalize to the TRAIN extent, exactly as the encoder's GeoAdaptiveRange does, so the control
         sees the data at its actual scale rather than as a speck of the globe. Fit on train rows only —
         using the full extent would leak the evaluation range into the control's features.
      2. Select the bandwidth. The encoder gets its hyperparameters chosen; a baseline pinned to one
         arbitrary sigma is not the strongest fair baseline, it is a straw man. Pick the sigma that
         maximizes the control's own held-out fit, and let the encoder beat THAT.

    Returns (features, chosen_sigma). Selection is on the same split the arms are scored on, which is
    generous to the baseline by design: an encoder gain that survives a baseline tuned on the evaluation
    split is not an artifact of the baseline being weak.
    """
    rn = np.asarray(rn, dtype=np.float32)
    fit = np.ones(len(rn), dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool)
    lo = rn[fit].min(0)
    span = np.maximum(rn[fit].max(0) - lo, 1e-6)
    scaled = ((rn - lo) / span * 2.0 - 1.0).astype(np.float32)      # train extent -> [-1, 1]
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, (scaled.shape[1], max(out_dim // 2, 1))).astype(np.float32)
    return scaled, base, tuple(bandwidths)


def _rff_features(scaled: np.ndarray, base: np.ndarray, sigma: float) -> torch.Tensor:
    proj = scaled @ (base * float(sigma))
    return torch.tensor(np.concatenate([np.sin(proj), np.cos(proj)], 1).astype(np.float32))

# ============================================================================================================
# 2. CONTRACT — what a probe must declare
# ============================================================================================================
#
# The contract itself now lives in `autoresearch/scoring/contract.py`, next to `definitions.py`, because
# it is not a spacetime fact -- ProbeResult, declare(), the record gate, the atomic board commit and the
# scorecard writer say what ANY probe must prove before a number becomes a record. The biological loop
# had none of that and was about to grow a second, weaker copy; one owner is the whole point.
#
# What stays here is what is genuinely spacetime: the fair RFF control, the mode registry, the capability
# list, the protocol history, and the loop bindings below that hand the shared machinery this loop's
# board path, protocol and capabilities.




# ============================================================================================================
# 3. EMIT — the one path by which a number becomes recordable
# ============================================================================================================

PHENO_RAW_REASON = (
    "this phenology direction runs on RAW spatial features only (Earth4D settled neutral here), "
    "so its numbers cannot speak to the encoder"
)


RAW_PE_REASON = (
    "this mode evaluates propagator architectures on RAW coordinate features only -- Earth4D is "
    "not in the comparison, so its numbers cannot speak to the encoder"
)




# ============================================================================================================
# 4. REGISTRY — capability -> modes -> where to edit
# ============================================================================================================
#
# What an agent needs to know after it picks a capability to improve.
# 
# The loop's step ② is "pick one capability from scorecard.md, with intention". Everything after that
# used to require reading a 1,500-line `main()` to answer three questions:
# 
#     which probe modes measure this capability?
#     what flags select each one, and what do they REQUIRE to run?
#     where do I edit to change the mechanism vs the data channel?
# 
# Those answers were only discoverable by grep, and getting them wrong is cheap-looking and expensive:
# eight of nineteen modes silently require `--forecast` via a bare `assert` buried mid-function, and
# `--phenology` shadows `--pheno_env`/`--pheno_taxon`/`--pheno_densefield` entirely because its branch
# returns first. This module states all of it in one place.
# 
# A `records=False` mode is a legitimate diagnostic that can never set a record -- either because its
# target is not on the scorecard, or because it evaluates on raw coordinate features with Earth4D absent
# from the comparison.
# 
# The five dynamics/AR modes that used to appear here (BREADTH, PROPAGATOR-ARCH, FIRST-ARRIVAL,
# ABUNDANCE, AR-ROLLOUT, CONTINUOUS-LEAD) were DELETED: ~1,300 lines of instrument/ that could never
# move a scorecard capability, plus 17 flags for them. They are in git history if a real target ever
# needs them.
# 
# Usage:
#     python -m deepearth.autoresearch.probes.spacetime.harness --capability family_from_env --list-modes
#     python -m deepearth.autoresearch.probes.spacetime.harness.harness.py --list-modes

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

    # ---- family_from_spacetime -------------------------------------------------------------------
    Mode("FORECAST(past->future)", "--forecast [--target family]",
         capability="family_from_spacetime", lever=ARCH,
         notes="The default coordinate path. --forecast_spatial switches to future+newplace."),
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
         notes="Record metric is Earth4D's best-head within-tolerance accuracy vs the generic PE's — "
               "NOT propagator_gain, which measures propagation vs static on raw features."),

)

# Where to make each kind of change. An agent that has picked a capability needs this more than it
# needs the file layout.
LEVER_SITES = {
    DATA: [
        "autoresearch/probes/spacetime/editable_files/lib/: scientific channel transforms and objectives",
        "new raw channels require a separate fixed-protocol maintenance change",
    ],
    ARCH: [
        "autoresearch/probes/spacetime/editable_files/earth4d.py: __init__, forward, training objective (the encoder itself)",
        "autoresearch/probes/spacetime/editable_files/lib/recurrence.py: run_recurrence, propagators",
        "CONFIG levers: recurrence, forecast, fourier, "
        "--time_harmonics",
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


def _list_modes(argv=None):
    ap = argparse.ArgumentParser()
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


# ============================================================================================================
# 5-7. GATE, PUBLISH, CLI — the loop
# ============================================================================================================

RECORDS = LOOP / "records" / "records.json"  # the machine record (fill scorecard by breaking these)
PROBE_MODULE = "deepearth.autoresearch.probes.spacetime.probe"

# The encoder-probeable capabilities (scorecard.md Layer 2). The objective must be one of these; the
# probe MODE and the architecture are the agent's choice. This list and scorecard.md Layer 2/3 are one
# contract -- change both together.
CAPABILITIES = [
    "species_from_env", "species_from_spacetime", "family_from_env", "family_from_spacetime",
    "community_from_env", "flowering_peak_month",
]

# Declared-and-refused, with the reason (scorecard.md Layer 3). These used to sit in CAPABILITIES with
# no PRIMARY_RE entry, so a run would fall through to the generic r"\bEarth4D\s+([\d.]+)" pattern and
# record whatever number matched first -- a legal --metric that measured nothing in particular. An
# explicit refusal is the honest version: the capability is real on the full-model board, it is simply
# not reachable through the encoder probe.
EXCLUDED_CAPABILITIES = {
    "calibration": "cannot resolve an encoder effect: the paired Earth4D-vs-RFF difference has a "
                   "per-seed spread of +/-0.055 against a 0.0118 barrier, so a single run says "
                   "nothing. Its record was also never a real measurement -- the probe imported a "
                   "module as a package, fell silently into a synthetic surrogate that ignores the "
                   "feature argument, and ran an unseeded encoder (0.5375/0.5682/0.6062 on three "
                   "identical commands). Retired, and its probe deleted with it",
    "family_from_vision": "borrowed frozen DINO/BioCLIP, and the stored record has no mode or shard "
                          "identity; it is not an Earth4D probe record",
    "lfmc_from_env": "non-encoder head: the capability lives in a downstream head",
    "mycorrhiza_from_env": "non-encoder head: the capability lives in a downstream head",
    "pollinator_from_env": "non-encoder head: the capability lives in a downstream head",
    "flowering_auc": "measured on the fusion model's flowering head, not the encoder",
    "flowering_fidelity": "measured on the fusion model's flowering head, not the encoder",
    "infer_clay": "env->env reconstruction runs through the field decoder, not the encoder probe",
    "infer_soil": "env->env reconstruction runs through the field decoder, not the encoder probe",
    "infer_climate": "env->env reconstruction runs through the field decoder, not the encoder probe",
    "infer_hydro": "env->env reconstruction runs through the field decoder, not the encoder probe",
}

# PROTOCOL VERSION. Bump this whenever a change alters what a run MEASURES rather than how well it does:
#   v5-encoder-only : 2026-07-31. The probe was not measuring Earth4D. At the v4 champion the head
#                     received 20,663 features and the hash grid was 36 of them (0.17%); CMAC tile
#                     coding was 89.2%, the RFF 9.9%, and drop_spatiotemporal had deleted the
#                     tri-planes. Every v4 and earlier `fair_gain vs RFF` therefore scored a tile coder
#                     with a hash-shaped residue attached, and no record on this board has ever been a
#                     measurement of the encoder. v5 builds exactly what fusion.py:302 builds -- 36
#                     spatial + 108 tri-plane = 144 dims -- and TRAINS it (the frozen-random protocol
#                     existed only because the trained backward was nondeterministic; the kernel fix
#                     lands that at bit-identical and 4.5% faster). Every row re-baselines. Expect the
#                     numbers to FALL: 0.0955 was the tile coder's.
# a leak fix, a split change, a target/normalization change. Records carry the protocol they were set under,
# and a run under a different protocol RE-BASELINES the capability instead of "beating" it -- mode and shard
# count both match across such a change, so neither of those gates catches it.
#   v1-prefix     : everything up to 2026-07-29. Leaked in three ways (train mask admitted future-at-seen-place
#                   and past-at-held-place rows; time normalization fitted its span on test dates; env/vision
#                   standardization fitted mu/sd over test rows) and normalized time so the held-out future
#                   landed where the hash grid saturates.
#   v2-leakfix    : strict spatiotemporal split, train-only time normalization with horizon headroom,
#                   train-only feature standardization, deterministic seeding.
#   v3-fairbaseline: the RFF control is no longer degenerate. It was `rn @ N(0, 8)` on GLOBE-normalized
#                   coords, which across a regional corpus varies ~0.04 cycles end to end -- it scored
#                   0.008, BELOW raw coordinates at 0.0166. Every `vs RFF` gain measured against it was
#                   inflated by that handicap, and every EARNING share on the board rested on it. The
#                   control now gets train-extent normalization (the same courtesy the encoder gets) and
#                   its bandwidth selected over a sweep. This changes what fair-gain MEANS, so v2 numbers
#                   are not comparable to v3 ones and must re-baseline rather than be beaten.
PROTOCOL = "v5-encoder-only"
# Only explicitly identified, audited protocols may be migrated automatically.
# Absence of a protocol is not evidence that a hand-restored or pre-gate record
# belongs to the known v1 measurement regime.
# Every protocol this board has ever run under, oldest first. APPEND on each bump; never reorder.
PROTOCOL_HISTORY = ("v1-prefix", "v2-leakfix", "v3-fairbaseline", "v4-fixedcontrol", "v5-encoder-only")
assert PROTOCOL in PROTOCOL_HISTORY, f"PROTOCOL {PROTOCOL!r} missing from PROTOCOL_HISTORY"

# A record under any SUPERSEDED protocol may be re-baselined by the current one -- that is what a
# protocol bump means. This was a hand-maintained allowlist frozen at {"v1-prefix", "v2-leakfix"}, so
# after the v3 bump nothing could migrate a v3 record: the v5 baseline run scored 0.0367 against a
# stored 0.0955 that measured a different object entirely, and was WITHHELD for "protocol migration
# mismatch" instead of re-baselining. The board could never have moved off v3. Derived from the history
# now, so bumping PROTOCOL is a one-line change that cannot half-apply.
REBASELINE_PROTOCOLS = frozenset(PROTOCOL_HISTORY[:PROTOCOL_HISTORY.index(PROTOCOL)])

# Fair-baseline preference: Earth4D must beat a TRAINED generic PE, not just raw coords.
# THE fair control. One entry, because there is one encoder question: does Earth4D beat a
# matched-width generic coordinate encoder on the same data, split and head?
#
# This was a 7-entry preference list, and the list is what let three different quantities share one
# column of the board. "GAIN" matched a gain over the CLASS PRIOR, so species_from_env published
# +0.4000 -- beating the base rate -- next to species_from_spacetime's +0.0608 against a real encoder,
# and the two were sorted against each other. "best-coord" matched the ENV CHANNEL's advantage over
# coordinates, which is how family_from_env read as an encoder gain of +0.0411 while Earth4D alone
# (0.0938) was losing to RFF (0.1010).
#
# A row that cannot produce "vs RFF" now yields fair_gain=None and is declared diagnostic at the call
# site, rather than silently scoring against whatever else it happened to report.
FAIR_ORDER = ["vs RFF"]


# ------------------------------------------------------------------------------------------------------------
# LOOP BINDINGS — the shared contract, told which board it is judging
# ------------------------------------------------------------------------------------------------------------
#
# Everything below is `scoring/contract.py` with this loop's board path, protocol and capability list
# bound in. The names and signatures are exactly what this file used to define itself, so every call
# site here and in probe.py is unchanged -- what changed is that there is now ONE implementation of the
# gate rather than one per loop, and `noise_barrier` in particular is no longer a second definition of
# the primitive `definitions.py` owns. The audit warned about that duplicate on every run.
from deepearth.autoresearch.scoring import contract                              # noqa: E402
from deepearth.autoresearch.scoring.contract import (                            # noqa: E402,F401
    CONTRACT_VERSION, DEADEND_CAP, ENCODER_SHARE_FLOOR, ContractError, Primary, ProbeResult,
    _evict_oldest_deadends, _next_seq, _same_mode, _same_probe, declare, noise_barrier,
)


def _set_result_sink(path, capability, protocol, args, config=None):
    """Arm the contract for a spacetime run — see contract._set_result_sink."""
    return contract._set_result_sink(path, capability, protocol, args, config, loop="spacetime")


def _read_records(path=None):
    return contract._read_records(path or RECORDS)


def _commit_records_if_unchanged(expected_raw, records, path=None):
    return contract._commit_records_if_unchanged(expected_raw, records, path or RECORDS)


def retire_record(capability, reason, path=None):
    return contract.retire_record(capability, reason, path or RECORDS)


def _record_gate(*a, **kw):
    return contract._record_gate(*a, protocol=PROTOCOL,
                                 rebaseline_protocols=REBASELINE_PROTOCOLS, **kw)


def _bottleneck(fair, primary) -> str:
    # floor=0.0: the spacetime head sees ONLY encoder features, so there is no score both arms inherit.
    return contract._bottleneck(fair, primary, floor=0.0, encoder="Earth4D",
                                input_lever="change the channel")


def _run(device: str, log_path: str, result_path: str, capability: str, seed=None) -> int:
    """Invoke the probe.

    There is ONE probe and it takes four arguments — capability, seed, device, result-json — because
    every lever that used to be a flag lives in its CONFIG block, and an experiment is a diff of that
    block. `--probe`/`--probe-module` existed only for lib/calib_probe, which declared its own flags and
    served the one capability now excluded; both are gone with it.
    """
    probe_argv = ["--device", device, "--result-json", result_path, "--capability", capability]
    if seed is not None:
        probe_argv += ["--seed", str(seed)]
    cmd = [sys.executable, "-m", PROBE_MODULE] + probe_argv
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO.parent) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print(f"[trace] $ {' '.join(cmd)}  (cwd={REPO})", flush=True)
    with open(log_path, "w") as lf:
        return subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              env=env, cwd=str(REPO)).returncode












MIN_CONFIRMATION_SEEDS = 2      # operator policy: one seed screens, two matched seeds confirm







# ---------------------------------------------------------------------------------------------------
# scorecard.txt — the git-visible view of the campaign
# ---------------------------------------------------------------------------------------------------
SCORECARD_TXT = LOOP / "program" / "scorecard.txt"


# The shared writers, bound to THIS loop's board, capabilities, protocol and reading key.
def _read_of(record: dict, gain, score) -> str:
    return contract._read_of(record, gain, score)


def write_scorecard(recs: dict, path: Optional[Path] = None) -> Path:
    return contract.write_scorecard(
        recs, path or SCORECARD_TXT, CAPABILITIES, PROTOCOL,
        title="EARTH4D SPACETIME PROBE — SCORECARD",
        legend=("read:  INPUT-LIMITED    loses to a matched-width RFF   -> DATA lever",
                "       ENCODER-LIMITED  wins but contributes <25%      -> ARCHITECTURE lever",
                "       EARNING          contributes >=25%              -> push the mechanism"),
        read_of=_read_of)


def _print_net_scorecard(recs: dict, current: str) -> None:
    contract._print_net_scorecard(recs, current, CAPABILITIES,
                                  "Earth4D encoder-probe records so far")


ENSUE_ENV_FILE = AUTORESEARCH / ".env"   # autoresearch/.env, gitignored


def _read_env_file(path: Path, key: str) -> str:
    """Read one KEY=value from a dotenv-style file. Never logs the value."""
    if not path.exists():
        return ""
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    return ""


def _ensue_token() -> str:
    """Resolve the Ensue token: environment first, then `autoresearch/.env`.

    The credential belongs to the autoresearch tree (gitignored there), not to a machine-specific
    absolute path. `/workspace/.env` is still honoured last so an existing box keeps working, but it is
    a fallback, not the home.
    """
    token = os.environ.get("ENSUE_API_TOKEN")
    if token:
        return token.strip()
    return (_read_env_file(ENSUE_ENV_FILE, "ENSUE_API_TOKEN")
            or _read_env_file(Path("/workspace/.env"), "ENSUE_API_TOKEN"))


def _code_provenance() -> dict:
    """Commit SHA + dirty flag for the tree this run measured.

    The evidence standard says a record from an unpushed commit is discovery-only, because nobody else
    can reproduce it. That rule was unenforceable: nothing recorded WHICH commit produced a number. It
    is not hypothetical -- a foreign agent's record on this board claims a `trained_rff` baseline that
    exists in no reachable tree, and a run of this loop was contaminated for an hour by an uncommitted
    edit to earth4d.py that nothing in the record would have revealed.
    """
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    # records/ is EXCLUDED: the harness writes the board and the scorecard on every run, so counting
    # them would make every result permanently "dirty" and the flag would mean nothing. Dirty here means
    # the CODE that produced the number is uncommitted.
    changed = [l for l in git("status", "--porcelain").splitlines()
               if l.strip() and "/records/" not in l and "/program/scorecard.txt" not in l]
    return {"commit": git("rev-parse", "HEAD")[:12],
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(changed)}


def post_ensue(trace: dict) -> None:
    tok = _ensue_token()
    if not tok:
        # A silent skip here means the swarm never learns this run happened, and the next agent pays to
        # rediscover the same dead-end. --ensue was explicitly requested, so a missing token is an error.
        sys.exit("[trace] --ensue was requested but no ENSUE_API_TOKEN is available (env or "
                 "/workspace/.env). The record was written locally; the swarm was NOT updated. "
                 "Export the token and re-publish rather than leaving the board stale.")
    o = trace["objective"]
    led = trace.get("ledger", {}) or {}
    hist = led.get("records", [])
    best = hist[-1] if hist else {"tag": trace["tag"], "score": o.get("record_value"), "gain": o.get("fair_st_gain")}
    dead = led.get("deadends", {})
    rec_str = " -> ".join(f"{r['tag']}:{r['score']}" for r in hist[-8:]) or "(none)"
    dead_str = "; ".join(f"{t}={d['score']}({(d.get('why') or '')[:34]})" for t, d in list(dead.items())[-12:]) or "(none)"
    # ONE upserted key per capability (LOOP-<program>-<capability> taxonomy): running best + record history +
    # this run's outcome + deduped dead-ends WITH their bottleneck reason. Win or dead-end, every run captured.
    # Evidence and provenance travel WITH the number. Without them the swarm cannot tell a two-seed
    # result from a single-seed one, nor reproduce the tree that produced it -- and both of those
    # ambiguities have already put a noise-mined record on this board.
    ev = trace.get("evidence", {})
    prov = trace.get("code", {})
    ev_str = (f"{ev.get('n_seeds', '?')} seeds"
              + (f" sd {ev['seed_std']:.6f}" if ev.get("seed_std") is not None else " (sd needs >=3)")
              + (" PROVISIONAL" if ev.get("provisional", True) else " CONFIRMED-ELIGIBLE"))
    prov_str = (f"{prov.get('branch', '?')}@{prov.get('commit', '?')}"
                + (" DIRTY-TREE" if prov.get("dirty") else ""))
    val = (f"LOOP-earth4d {trace['metric']}: BEST {best.get('score')} (gain {best.get('gain')}, {o.get('fair_baseline')}) "
           f"via '{best.get('tag')}'. runs={led.get('runs')}. record-history: {rec_str}. "
           f"THIS RUN '{trace['tag']}': primary={o['primary']} gain={o['fair_st_gain']} "
           f"decision={o.get('decision', 'legacy')} evidence={ev_str} code={prov_str} "
           f"bottleneck={trace['bottleneck']}. dead-ends-tried: {dead_str}.")
    # create_memory does NOT overwrite an existing key -- the API has a separate update_memory. This
    # loop upserts ONE key per capability, so every run after the first was a silent no-op: the server
    # returned 200, the harness printed "Ensue logged", and the stored value stayed at whatever the
    # first run wrote. Checked directly: the key still read "BEST 0.0474 ... runs=1" from 2026-07-29
    # while the local board had moved to 0.0787 over ~30 runs. The swarm was reading a stale board and
    # would have re-bought every dead-end published since.
    #
    # Update first, create only if the key does not exist yet.
    def _call(tool: str) -> tuple:
        # The two tools take DIFFERENT argument shapes: create_memory batches under "items", while
        # update_memory takes one flat object. Sending the batch form to update_memory fails, which is
        # why the earlier fallback still ended at create_memory's duplicate-key error.
        args = ({"items": [{"key_name": key, "value": val, "description": desc}]} if tool == "create_memory"
                else {"key_name": key, "value": val, "description": desc})
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": tool, "arguments": args}}
        req = urllib.request.Request("https://api.ensue-network.ai/", data=json.dumps(payload).encode(),
                                     headers={"Authorization": f"Bearer {tok}",
                                              "Content-Type": "application/json",
                                              "Accept": "application/json, text/event-stream"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read().decode()
            failed = '"failed":1' in body or '"error"' in body
            return (not failed), body
        except Exception as exc:
            return False, str(exc)

    key = f"LOOP-earth4d-{trace['metric']}"
    desc = (f"Earth4D encoder-probe loop {trace['metric']}: best {best.get('score')} "
            f"gain {best.get('gain')} over {led.get('runs')} runs")
    ok, body = _call("update_memory")
    if not ok:
        ok, body = _call("create_memory")
    if ok:
        print(f"[trace] Ensue upserted {key}", flush=True)
    else:
        # A silent failure here is how the swarm went stale for a whole session.
        sys.exit(f"[trace] ENSUE WRITE FAILED for {key}: {body[:300]}\n"
                 f"        The local board is correct but the swarm was NOT updated. Fix and re-publish.")



# ============================================================================================================
# DETERMINISM CHECK  (harness.py --determinism)
# ============================================================================================================
#
# The trained protocol was unusable for years because the backward is a storm of colliding float atomics:
# five seed-0 runs gave 0.1873/0.1925/0.1867/0.1872/0.1952 (sd 0.0038), as large as the whole across-seed
# spread. An irreproducible number cannot set a record, which is why every record before v5 was frozen-
# encoder. utils.cuh::atomicAddFixed replaces those atomics with order-independent int64 accumulation.
#
# This is the gate for that fix, and the regression test for it: exit 1 while the backward diverges,
# exit 0 once it is bit-identical. It also ATTRIBUTES the divergence per level -- concentrated in coarse
# levels (many points per cell, many colliding atomics) confirms atomicAdd; a flat profile would mean the
# cause is elsewhere and the kernel is the wrong target. Measured on 2x RTX PRO 6000: coarse levels ran
# 5-13x above a flat ~2.5e-7 fine-level plateau, and EARTH4D_DETERMINISTIC=1 made all four encoders
# bit-identical at 4.5% FASTER.



def _det_one_backward(enc: Earth4D, coords: torch.Tensor, seed: int):
    """One forward+backward at a fixed seed. Returns (output, per-encoder embedding grads)."""
    torch.manual_seed(seed)
    for p in enc.parameters():
        if p.grad is not None:
            p.grad = None
    out = enc(coords)
    # A fixed, seed-independent scalar objective: no randomness anywhere except the kernel itself.
    loss = (out * torch.linspace(1.0, 2.0, out.shape[1], device=out.device)).sum()
    loss.backward()
    grads = {name: p.grad.detach().clone()
             for name, p in enc.named_parameters()
             if p.grad is not None and name.endswith("embeddings")}
    return out.detach().clone(), grads


def _det_report(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    same = torch.equal(a, b)
    if same:
        print(f"  {name:<34} BIT-IDENTICAL")
        return True
    d = (a - b).abs()
    scale = a.abs().max().clamp_min(1e-30)
    n_diff = int((d > 0).sum())
    print(f"  {name:<34} DIVERGES   max|d|={d.max():.3e}  rel={d.max() / scale:.3e}  "
          f"elems={n_diff}/{a.numel()}")
    return False


def _determinism_check(device: str = "cuda:0", n: int = 200_000, repeats: int = 3,
                       levels: int = 18, log2_hashmap: int = 20) -> int:
    import torch
    from deepearth.autoresearch.probes.spacetime.editable_files.earth4d import Earth4D
    if not torch.cuda.is_available() and device.startswith("cuda"):
        print("[determinism] no CUDA device — this must run on the box.")
        return 2
    dev = torch.device(device)
    torch.manual_seed(0)
    coords = torch.stack([
        torch.rand(n, device=dev) * 9.5 + 32.5, torch.rand(n, device=dev) * 10.0 - 124.0,
        torch.rand(n, device=dev) * 2000.0, torch.rand(n, device=dev)], dim=1)
    enc = Earth4D(verbose=False, spatial_levels=levels, temporal_levels=levels,
                  spatial_log2_hashmap_size=log2_hashmap,
                  temporal_log2_hashmap_size=log2_hashmap).to(dev)
    print(f"[determinism] {n:,} coords · {levels} levels · 2^{log2_hashmap} hashmap · {repeats} repeats\n")
    outs, grads = [], []
    for _ in range(repeats):
        o, g = _det_one_backward(enc, coords, seed=0)
        outs.append(o); grads.append(g)
    print("1. FORWARD (gather — expected bit-identical)")
    for r in range(1, repeats):
        _det_report(f"output[0 vs {r}]", outs[0], outs[r])
    print("\n2. BACKWARD (scatter — atomicAdd lives here)")
    ok = True
    for name in grads[0]:
        for r in range(1, repeats):
            ok &= _det_report(f"{name}[0 vs {r}]", grads[0][name], grads[r][name])
    if ok:
        print("\nBackward is deterministic on this build.")
        return 0
    print("\n3. WHERE — per-level divergence (coarse-concentrated => atomicAdd confirmed)")
    named = {nm: m for nm, m in enc.named_modules() if hasattr(m, "offsets")}
    for name, g0 in grads[0].items():
        mod = named.get(name.rsplit(".", 1)[0])
        if mod is None:
            continue
        offs = mod.offsets.tolist()
        print(f"\n   {name.rsplit('.', 1)[0]}")
        for lvl in range(len(offs) - 1):
            lo, hi = offs[lvl], offs[lvl + 1]
            b0, b1 = g0[lo:hi], grads[1][name][lo:hi]
            rel = (b0 - b1).abs().max().item() / max(b0.abs().max().item(), 1e-30)
            print(f"     level {lvl:>2}  entries {hi-lo:>9,}  rel={rel:.3e}  " + "#" * min(int(rel * 2e4), 40))
    print("\nEXIT 1: backward is nondeterministic. Rerun with EARTH4D_DETERMINISTIC=1.")
    return 1


# ============================================================================================================
# PRIOR LEARNINGS  (harness.py --insights)
# ============================================================================================================
#
# 2,531 recorded runs and 123 dead-ends with their reasons sit in this board, and until now NOTHING read
# them back. Step (1) of the loop says "pull Ensue keys + records.json, skip logged dead-ends" and there
# was no command that did it, so every agent re-derived what the last one had already paid for.
#
# v5 voids the SCORES -- they measured a tile coder at 0.17% encoder content, against a fair-gain column
# that mixed three different quantities. It does NOT void the HYPOTHESES. "--recurrence -0.0180",
# "--gnn -0.0261", "extent_fit -0.0199", "tri-plane conjunction ~0", "17 capacity-sweep rows bought
# nothing" are all still information about which levers are worth a run, and re-buying them under the
# new regime is pure waste.
#
# So this prints the attempts and their reasons, with every pre-v5 number explicitly marked void so
# nobody compares against one. Ensue is queried too when a token is present, because the swarm's board
# carries attempts this checkout never ran.
def _ensue_fetch(key: str):
    """Read one Ensue key. Returns its value string, or None if unavailable."""
    tok = _ensue_token()
    if not tok:
        return None
    for tool, args in (("get_memory", {"key_name": key}),
                       ("search_memories", {"query": key})):
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": tool, "arguments": args}}
        req = urllib.request.Request("https://api.ensue-network.ai/", data=json.dumps(payload).encode(),
                                     headers={"Authorization": f"Bearer {tok}",
                                              "Content-Type": "application/json",
                                              "Accept": "application/json, text/event-stream"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                body = r.read().decode()
            if '"error"' not in body and key in body:
                return body
        except Exception:
            continue
    return None


def _insights(capability: str = "", ensue: bool = True) -> int:
    _raw, recs = _read_records()
    caps = [capability] if capability else sorted(recs)
    if capability and capability not in recs:
        print(f"[insights] no ledger for {capability!r}. known: {', '.join(sorted(recs))}")
        return 1

    print("=" * 108)
    print("PRIOR LEARNINGS — what has already been tried on this board")
    print("=" * 108)
    print("Every SCORE below predates protocol v5 and is VOID: it measured a feature vector that was")
    print("0.17% encoder, scored against a fair-gain column that mixed encoder / env-channel / class-prior")
    print("gains. Do not compare a v5 number to one. The HYPOTHESES and their reasons are still valid --")
    print("that is the point of reading this before you pick.\n")

    for cap in caps:
        v = recs.get(cap) or {}
        led = v.get("ledger") or {}
        dead = led.get("deadends", {})
        hist = led.get("records", [])
        print("-" * 108)
        print(f"{cap}   runs={led.get('runs', 0)}   dead-ends={len(dead)}   "
              f"pre-v5 record={v.get('score')} (VOID)   protocol={v.get('protocol')}")
        if hist:
            print("  record history (void): " + " -> ".join(f"{r.get('tag')}:{r.get('score')}" for r in hist[-6:]))
        if dead:
            print("  attempts and why they stopped:")
            for tag, x in sorted(dead.items(), key=lambda kv: -(kv[1].get("seq") or 0))[:24]:
                why = (x.get("why") or "").replace("\n", " ")
                print(f"    {tag[:44]:46} {str(x.get('score'))[:8]:9} {why[:52]}")
        if ensue:
            got = _ensue_fetch(f"LOOP-earth4d-{cap}")
            print(f"  ensue: {'fetched — swarm board may carry attempts this checkout never ran' if got else 'unavailable (no token, or key absent)'}")
    print("-" * 108)
    print("Before picking: skip anything above whose REASON still applies under v5. A lever that failed")
    print("because it was drowned in 18,432 tile-code dims is NOT settled -- it was never measured on the")
    print("encoder. A lever that failed on its own mechanics (extent_fit -0.0199, capacity sweeps) is.")
    return 0



# ============================================================================================================
# THE DATA LEVER  (harness.py --channels)
# ============================================================================================================
#
# program.md calls DATA co-equal with ARCHITECTURE, and until now an agent had no way to see it: the
# corpus is one flat bag shared with the biological loop and fusion, and nothing said which files THIS
# encoder can consume. Measured consequence -- five of six capabilities run on coordinates alone, two
# advertised levers (`env_extra`, `densify`) are not CONFIG keys at all, and two channels that exist on
# disk (topo, hydro) have no loader.
#
# probe.py declares CHANNELS; this derives their state from disk, so the table cannot claim a channel
# the corpus does not have.
def _channels(cap: str = "") -> int:
    import ast as _ast
    science_sources = tuple((LOOP / "editable_files").rglob("*.py"))
    src = "\n".join(path.read_text() for path in science_sources)

    def science_literal(name):
        """Find a uniquely declared science value without coupling to an internal filename."""
        found = []
        for path in science_sources:
            tree = _ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, (_ast.Assign, _ast.AnnAssign)):
                    targets = node.targets if isinstance(node, _ast.Assign) else [node.target]
                    if any(isinstance(target, _ast.Name) and target.id == name for target in targets):
                        found.append((path, _ast.literal_eval(node.value)))
        if len(found) != 1:
            raise RuntimeError(f"expected one editable declaration of {name}, found {[str(p) for p, _ in found]}")
        return found[0][1]

    CH = science_literal("CHANNELS")
    RP = science_literal("REPAIRED")
    CFG = science_literal("CONFIG")
    CAP = science_literal("CAPABILITY_CONFIG")
    cache = Path(CFG["cache_dir"])
    if not cache.is_absolute():
        cache = REPO.parent / cache if (REPO.parent / cache).exists() else AUTORESEARCH.parent / cache

    # a channel is LIVE if any capability preset (or CONFIG, for a bare switch) actually selects it
    def live_in(name, switch):
        for c, v in CAP.items():
            if cap and c != cap:
                continue
            m = {**CFG, **v}
            val = m.get(switch)
            if val is True or (isinstance(val, str) and (name in val or val == "all")):
                if switch != "env_channels" or m.get("env"):
                    return c
                if switch == "env_channels" and m.get("env"):
                    return c
        return ""

    print("=" * 100)
    print(f"DATA LEVER — channels this encoder can consume{f'  (capability: {cap})' if cap else ''}")
    print("=" * 100)
    print(f"corpus: {cache}\n")
    print(f"  {'channel':<14} {'state':<10} {'dims':>5}  {'switch':<14} files")
    n_unwired = n_missing = 0
    for name, (files, dims, switch, what) in CH.items():
        on_disk = all((cache / f).exists() for f in files)
        wired = f'"{name}"' in src or f"'{name}'" in src
        if not on_disk:
            state = "MISSING"; n_missing += 1
        elif live_in(name, switch):
            state = "LIVE"
        elif not wired:
            state = "UNWIRED"; n_unwired += 1
        else:
            state = "AVAILABLE"
        print(f"  {name:<14} {state:<10} {dims:>5}  {switch:<14} {', '.join(files)}")
        print(f"  {'':14} {what}")
    print()
    print("  LIVE       a capability preset selects it today")
    print("  AVAILABLE  on disk and a loader reads it -- flip the switch")
    print("  UNWIRED    on disk and NO loader reads it. A real lever nobody can pull.")
    print("  MISSING    declared but absent on disk")

    print("\n" + "-" * 100)
    print("REPAIRED CORPUS FILES — a run against an un-activated one measures the wrong data")
    stale = 0
    for live, rebuilt in RP.items():
        lp, rp = cache / live, cache / rebuilt
        if not rp.exists():
            print(f"  {live:<30} no rebuilt version on disk")
            continue
        if not lp.exists():
            print(f"  {live:<30} *** LIVE FILE MISSING, repair exists at {rebuilt}"); stale += 1
        elif lp.stat().st_size != rp.stat().st_size:
            print(f"  {live:<30} *** NOT ACTIVATED  live={lp.stat().st_size:,}  repaired={rp.stat().st_size:,}")
            stale += 1
        else:
            print(f"  {live:<30} activated")
    print("-" * 100)
    print(f"{n_unwired} unwired · {n_missing} missing · {stale} un-activated repair(s)")
    if stale:
        print("Activating one is `cp derived/<x>_rebuilt.<ext> <x>.<ext>` -- a champion-pipeline change,")
        print("so it is the OPERATOR's call, not an agent's. Until then those rows measure the old corpus.")
    return 1 if (n_missing or stale) else 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Earth4D legacy probe ledger — exact audited protocol migrations only"
    )
    ap.add_argument("--metric", required=True, help="objective capability (one of the scorecard capabilities)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=1,
                    help="run the probe this many times with matched seeds; >=3 makes the seed spread "
                         "measurable, so the noise barrier becomes 2 sigma instead of a fixed floor")
    ap.add_argument("--ensue", action="store_true")
    ap.add_argument("--log", default=None)
    ap.add_argument("--retire", default="", metavar="REASON",
                    help="OPERATOR ACTION, not an experiment: withdraw this capability's record "
                         "because it was never a valid measurement, giving the reason. The old score "
                         "moves into the ledger under `retired` and the row reopens. Use when a record "
                         "is provably wrong (surrogate data, unseeded encoder, leaked split) — never "
                         "because a run failed to beat it.")
    a = ap.parse_args()

    if a.metric in EXCLUDED_CAPABILITIES:
        sys.exit("[trace] --metric %r is excluded: %s\n"
                 "        See autoresearch/probes/spacetime/program/scorecard.md Layer 3."
                 % (a.metric, EXCLUDED_CAPABILITIES[a.metric]))
    if a.metric not in CAPABILITIES:
        sys.exit("[trace] --metric %r is not an encoder-probeable capability. one of:\n  %s"
                 % (a.metric, "\n  ".join(CAPABILITIES)))
    if a.retire:
        entry = retire_record(a.metric, a.retire)
        print(f"[trace] RETIRED {a.metric}: score {entry['score']} withdrawn — {a.retire}")
        print(f"[trace] the row is reopened; the next run sets a fresh baseline.")
        write_scorecard(_read_records()[1])
        return

    modes = for_capability(a.metric)
    if not modes:
        sys.exit(f"[trace] no recording probe mode measures {a.metric!r}. "
                 f"See harness.py --list-modes.")
    print(f"[trace] {a.metric}: {len(modes)} mode(s) can set this record — "
          + ", ".join(m.mode for m in modes), flush=True)
    records_snapshot, preflight_records = _read_records()

    tag = a.tag or f"e4d_{a.metric}"
    log_path = a.log or str(LOOP / "records" / "traces" / f"{tag}.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[trace] OBJECTIVE={a.metric}  tag={tag}", flush=True)
    # One run per seed. Seeds are matched across arms by construction: the probe seeds numpy and torch
    # from --seed, so seed k is the same initialization for every configuration compared here.
    seed_results, seed_values = [], []
    for k in range(max(1, a.seeds)):
        seed_log = log_path if a.seeds == 1 else str(Path(log_path).with_suffix(f".seed{k}.log"))
        result_path = str(Path(seed_log).with_suffix(".result.json"))
        rc = _run(a.device, seed_log, result_path, a.metric, seed=k)
        text = Path(seed_log).read_text(errors="ignore")
        if rc != 0:
            print(text[-1800:])
            sys.exit(f"[trace] probe FAILED on seed {k} (rc={rc}); see {seed_log}")
        try:
            seed_results.append(ProbeResult.read(result_path))
        except (ContractError, OSError) as exc:
            sys.exit(f"[trace] seed {k} emitted no usable result contract: {exc}")
        seed_values.append(seed_results[-1].primary.value)
        if a.seeds > 1:
            print(f"[trace] seed {k}: {seed_results[-1].primary.name} = {seed_values[-1]:.6f}", flush=True)

    log_path = str(Path(log_path).with_suffix(".seed0.log")) if a.seeds > 1 else log_path
    result_path = str(Path(log_path).with_suffix(".result.json"))
    seed_std = (statistics.pstdev(seed_values) if len(seed_values) > 2 else None)
    seed_mean = sum(seed_values) / len(seed_values)
    if a.seeds > 1:
        print(f"[trace] {a.seeds} seeds: mean {seed_mean:.6f}"
              + (f"  sd {seed_std:.6f}" if seed_std is not None else "  (sd needs >=3 seeds)"), flush=True)
    text = Path(log_path).read_text(errors="ignore")

    # The probe DECLARES what it measured. Nothing here parses stdout: a mode that does not emit a
    # contract cannot set a record, which is the point -- the old parser always produced *something*.
    result = seed_results[0]
    if result.diagnostic:
        sys.exit(f"[trace] {result.mode} is a DIAGNOSTIC and cannot set a record: "
                 f"{result.diagnostic_reason}\n        log preserved at {log_path}")
    if result.capability != a.metric:
        sys.exit(f"[trace] probe measured {result.capability!r} but --metric declared {a.metric!r}; "
                 f"refusing to record a different question's answer")

    known = {m.mode for m in modes}
    if result.mode not in known and not any(result.mode.startswith(k.split("(")[0]) for k in known):
        print(f"[trace] *** UNREGISTERED MODE {result.mode!r} for {a.metric}. Registered: "
              f"{sorted(known)}. Recording it, but add it to MODES in this file so the next agent can "
              f"find it.", flush=True)

    primary = seed_mean          # the mean across seeds — never the max of reruns
    fair, fair_base = result.fair_gain(FAIR_ORDER)
    bottleneck = _bottleneck(fair, primary)
    mode = result.mode
    shards = result.n_shards
    header = result.render()
    gains = dict(result.gains)
    metrics = [f"{k} = {v}" for k, v in sorted(result.baselines.items())]

    # RECORD tracking + full run LEDGER (taxonomy: never lose a run's result; publish win OR dead-end w/ reason) --
    recs = preflight_records
    key_val = primary if primary is not None else fair
    cur = recs.get(a.metric, {})
    prev = cur.get("score")
    # RECORD GATE. A record used to fire on any parsed number that beat the stored one -- no check that the run
    # measured the SAME THING. That is how --pheno_disttarget peak_week (a different target, and a leaked one)
    # took flowering_peak_month 0.067 -> 0.683 and published it. A capability's record may only be beaten by a
    # run in the SAME probe mode; a different mode is a different target and gets flagged for review instead.
    prev_mode, prev_shards = cur.get("mode"), cur.get("n_shards")
    prev_proto = cur.get("protocol")
    # An UNSTAMPED record (pre-gate, or hand-restored) is treated as unknown-mode and does NOT auto-pass:
    # that is exactly how the leaked peak-week run slipped through on its second attempt.
    is_record, rebaseline, beats, mode_ok, shards_ok = _record_gate(
        key_val,
        prev,
        prev_proto,
        mode,
        prev_mode,
        shards,
        prev_shards,
        cur.get("probe"),
        seed_std=seed_std,
        n_seeds=len(seed_values),
    )
    # A DIRTY TREE CANNOT SET A RECORD. With the flags gone, the experiment IS the CONFIG/earth4d.py
    # diff, so a record measured on uncommitted code has a configuration nobody can ever recover -- the
    # number survives and the thing that produced it does not. All three coordinate records carry
    # dirty=True, and that is the mechanism behind the family_from_spacetime noise-walk: seven accepted
    # single-seed steps from a second tree, invalidated, then immediately re-set at one dirty seed.
    # The run still measures, still publishes to Ensue, still writes its trace. It just cannot take the
    # record.
    _dirty = _code_provenance().get("dirty")
    if _dirty and is_record:
        print("[trace] *** RECORD WITHHELD: DIRTY TREE. The run measured uncommitted changes, so its "
              "configuration is not recoverable by anyone else. Commit, then re-run to claim it.",
              flush=True)
        is_record = rebaseline = False
    _barrier = noise_barrier(prev, seed_std, len(seed_values))
    if prev is not None and key_val is not None and key_val > prev and not beats:
        print(f"[trace] *** WITHIN NOISE: {key_val:.6f} beats {prev} by {key_val - prev:+.6f}, under the "
              f"barrier of {_barrier:.6f}"
              + (f" (2 sd of {len(seed_values)} seeds)" if seed_std is not None and len(seed_values) >= 3
                 else " (fixed floor; seed spread needs >=3)") +
              ".\n[trace]     Not a record. Seven steps of this size are how a record was walked "
              "0.1769 -> 0.1914 overnight.", flush=True)
    migration_withheld = prev is not None and prev_proto != PROTOCOL and not rebaseline
    if rebaseline and key_val is not None:
        print(f"[trace] *** RE-BASELINE: record was set under protocol {prev_proto!r}, this run is {PROTOCOL!r}.\n"
              f"[trace]     {prev} and {key_val} measure different things, so this is not a comparison —\n"
              f"[trace]     the capability's baseline is being RESET to {key_val}. Prior record archived in the ledger.",
              flush=True)
    elif prev is not None and prev_proto != PROTOCOL and prev_proto not in REBASELINE_PROTOCOLS:
        print(f"[trace] *** PROTOCOL MIGRATION WITHHELD: prior protocol {prev_proto!r} is not an "
              f"explicitly audited migration source.\n"
              f"[trace]     The protected record remains unchanged; migrate it deliberately after provenance review.",
              flush=True)
    elif prev is not None and prev_proto in REBASELINE_PROTOCOLS and prev_proto != PROTOCOL and not rebaseline:
        print(f"[trace] *** PROTOCOL MIGRATION WITHHELD: the old probe command, mode, and shard count "
              f"must match exactly.\n"
              f"[trace]     old={cur.get('probe')!r} mode={prev_mode!r} shards={prev_shards!r}\n"
              f"[trace]     new mode={mode!r} shards={shards!r}",
              flush=True)
    if beats and not (mode_ok and shards_ok):
        why = ("mode %r != record mode %r" % (mode, prev_mode) if not mode_ok
               else "n_shards %r != record n_shards %r" % (shards, prev_shards))
        print(f"[trace] *** RECORD WITHHELD: {why}.\n"
              f"[trace]     {key_val} vs {prev} is not a like-for-like comparison. Match the record's protocol,\n"
              f"[trace]     or verify the new one is sound and set the record deliberately.", flush=True)
    ledger = cur.get("ledger", {"runs": 0, "records": [], "deadends": {}})
    ledger["runs"] = ledger.get("runs", 0) + 1
    if is_record:
        cur = {"score": key_val, "primary": primary, "fair_st_gain": fair,
               "code": _code_provenance(),
               "n_seeds": len(seed_values), "seed_values": [float(v) for v in seed_values],
               "seed_std": (float(seed_std) if seed_std is not None else None),
               "provisional": len(seed_values) < MIN_CONFIRMATION_SEEDS,
               "fair_baseline": fair_base, "tag": tag, "mode": mode, "n_shards": shards,
               "protocol": PROTOCOL}
        ledger["records"] = (ledger.get("records", []) + [{"tag": tag, "score": key_val, "gain": fair,
                                                           "protocol": PROTOCOL,
                                                           "rebaseline_from": prev if rebaseline else None}])[-20:]
    elif migration_withheld and key_val is not None:
        ledger.setdefault("deadends", {})[tag] = {
            "score": key_val,
            "gain": fair,
            "why": (
                f"PROTOCOL MIGRATION WITHHELD (old protocol={prev_proto!r}, mode={prev_mode!r}, "
                f"n_shards={prev_shards!r}, probe={cur.get('probe')!r}; new mode={mode!r}, "
                f"n_shards={shards!r})"
            ),
        }
    elif beats and not (mode_ok and shards_ok):
        ledger.setdefault("deadends", {})[tag] = {
            "score": key_val, "gain": fair,
            "why": (f"RECORD WITHHELD (mode {mode!r} vs {prev_mode!r}, n_shards {shards!r} vs {prev_shards!r}) "
                    f"-- not like-for-like; needs a deliberate check before it can count")}
    elif key_val is not None:
        # dead-end: a lever below record — kept WITH its reason, deduped by tag (no noise-floor spam)
        ledger.setdefault("deadends", {})[tag] = {
            "score": key_val, "gain": fair, "why": bottleneck, "seq": _next_seq(ledger),
        }
        _evict_oldest_deadends(ledger)
    cur["ledger"] = ledger
    recs[a.metric] = cur
    if not _commit_records_if_unchanged(records_snapshot, recs):
        sys.exit(
            "[trace] WORKFLOW WITHHELD: records.json changed while the probe ran; "
            f"the probe log is preserved at {log_path}, but no record was written"
        )

    decision = (
        "rebaseline" if rebaseline
        else "record" if is_record
        else "migration_withheld" if migration_withheld
        else "no_record"
    )
    objective = {"primary": primary, "fair_st_gain": fair, "fair_baseline": fair_base,
                 "record": bool(is_record), "rebaseline": bool(rebaseline), "decision": decision,
                 "prev_record": prev, "record_value": key_val}
    code = _code_provenance()
    if code.get("dirty"):
        print("[trace] *** DIRTY TREE: this run measured uncommitted changes. The result is not "
              "reproducible by anyone else and is discovery-only. Commit before recording a claim.",
              flush=True)
    trace = {"metric": a.metric, "tag": tag,
             "code": code,
             "evidence": {"n_seeds": len(seed_values),
                          "seed_values": [float(v) for v in seed_values],
                          "seed_std": (float(seed_std) if seed_std is not None else None),
                          "provisional": len(seed_values) < MIN_CONFIRMATION_SEEDS},
             "objective": objective, "gains": gains, "header": header, "metrics": metrics,
             "bottleneck": bottleneck, "rc": rc, "ledger": ledger}

    # one-screen consistent summary ---------------------------------------------------------------------
    print("\n" + "=" * 76)
    print(f"OBJECTIVE  {a.metric}")
    print(header or "(no '=== SPACETIME' header parsed — check the log)")
    print("-" * 76)
    print(f"  primary(score) = {primary}   fair_st_gain = {fair} (vs {fair_base})   all_gains = {gains}")
    record_text = (
        "RE-BASELINE (not a comparable win)" if rebaseline
        else "YES (new best!)" if is_record
        else "WITHHELD (protocol migration mismatch)" if migration_withheld
        else "no"
    )
    print(f"  RECORD = {record_text}   prev_record = {prev}")
    print(f"  BOTTLENECK: {bottleneck}")
    print("  metrics:")
    for m in metrics:
        print("    " + m)
    print("=" * 76)
    out = Path(log_path).with_suffix(".trace.json")
    out.write_text(json.dumps(trace, indent=2))
    print(f"[trace] wrote {out}" + ("  |  RECORDS.json updated" if is_record else ""))

    _print_net_scorecard(recs, a.metric)   # show the whole board after every run
    print(f"[trace] scorecard -> {write_scorecard(recs)}", flush=True)

    if a.ensue:
        post_ensue(trace)


if __name__ == "__main__":
    # Two entry points, one file: `--list-modes` answers "what can move this capability, and where do I
    # edit?" without running anything; anything else runs the loop.
    if "--channels" in sys.argv:
        sys.argv.remove("--channels")
        _c = ""
        if "--metric" in sys.argv:
            _i = sys.argv.index("--metric"); _c = sys.argv[_i + 1]; del sys.argv[_i:_i + 2]
        raise SystemExit(_channels(_c))
    if "--insights" in sys.argv:
        sys.argv.remove("--insights")
        _cap = ""
        if "--metric" in sys.argv:
            _i = sys.argv.index("--metric"); _cap = sys.argv[_i + 1]; del sys.argv[_i:_i + 2]
        raise SystemExit(_insights(_cap, ensue="--no-ensue" not in sys.argv))
    if "--determinism" in sys.argv:
        sys.argv.remove("--determinism")
        raise SystemExit(_determinism_check())
    if "--list-modes" in sys.argv:
        sys.argv.remove("--list-modes")
        _list_modes()
    else:
        main()
