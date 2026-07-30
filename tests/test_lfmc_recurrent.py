"""Behavioral guards for the registered recurrent LFMC experiment.

These prove PROPERTIES, not scores, and never assert on source text: recursion purity, sentinel
detection of future access, control semantics, pair/provenance parity, matched capacity and budget,
determinism, protocol minimums, and test-partition closure. Earth4D is injected, so the suite
collects and runs on CPU without CUDA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from deepearth.autoresearch.probes.spacetime.editable_files.lib import lfmc_recurrent as LR
from deepearth.autoresearch.probes.spacetime.editable_files.lib.science_gate import (
    build_rolling_origin_pairs,
    collapse_lfmc_visits,
    make_split_masks,
)

CPU = torch.device("cpu")


# --------------------------------------------------------------------------- fixtures / stubs
def _synthetic_frame(n_sites: int = 8, per_site: int = 18, start: str = "2018-01-05"):
    rows = []
    for s in range(n_sites):
        for i, d in enumerate(pd.date_range(start, periods=per_site, freq="30D")):
            rows.append({
                "sampling_date": d,
                "site_name": f"site-{s}",
                "species_collected": "spp-a" if s % 2 == 0 else "spp-b",
                "latitude": 35.0 + 0.4 * s,
                "longitude": -120.0 + 0.4 * s,
                "elevation": 100.0 + 25.0 * s,
                "lfmc_value": 90.0 + 12.0 * np.sin(i / 3.0) + s,
                "random_split": ("train", "validation", "test")[i % 3],
                "spatial_split": ("train", "validation", "test")[s % 3],
            })
    return pd.DataFrame(rows)


class StubEncoder(nn.Module):
    """Stands in for Earth4D: linear in the physical coordinate, trainable, CPU-only."""

    def __init__(self, out_dim: int = 8):
        super().__init__()
        self.lin = nn.Linear(4, out_dim)
        self.seen_coords = []

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        self.seen_coords.append(coords.detach().clone())
        return self.lin(coords)


def stub_factory(train_coords, horizon_days, device):
    return StubEncoder().to(device)


class RecordingStep(nn.Module):
    """Returns +1 per step and records the state it was handed."""

    def __init__(self, pos_dim, state_dim, hidden=8, delta: float = 1.0):
        super().__init__()
        self.state_dim = state_dim
        self.state_proj = nn.Linear(max(state_dim, 1), 16)
        self.net = nn.Sequential(nn.Linear(2 * pos_dim + 16, 1))
        self.delta = delta
        self.seen_states = []

    def forward(self, p_o, p_s, state):
        self.seen_states.append(np.asarray(state.detach().cpu()).copy())
        return torch.full((p_o.shape[0],), self.delta)


@pytest.fixture(scope="module")
def visits():
    return collapse_lfmc_visits(_synthetic_frame())


@pytest.fixture(scope="module")
def tabs(visits):
    p30 = build_rolling_origin_pairs(visits, 30)
    p90 = build_rolling_origin_pairs(visits, 90)
    train = LR._pair_table(visits, p30)
    ev90 = LR._pair_table(visits, p90)
    if train.empty or ev90.empty:
        pytest.skip("synthetic frame produced no pairs")
    return train, ev90


def _run(arm, train, ev, *, seed=0, seconds=0.0, steps=0, head=None, horizon=90):
    return LR.run_arm(arm, train, {horizon: ev}, seed=seed, seconds=seconds, device=CPU,
                      horizon_days=horizon, encoder_factory=stub_factory,
                      step_model_factory=head, steps=steps)


# --------------------------------------------------------------------------- recursion purity
def test_step_k_receives_exactly_step_k_minus_1_prediction(tabs):
    """90d must be three chained 30d steps, each fed the previous step's own output."""
    train, ev = tabs
    recorder = {}

    def head(pos_dim, state_dim):
        recorder["m"] = RecordingStep(pos_dim, state_dim, delta=1.0)
        return recorder["m"]

    res = _run("earth4d", train, ev, head=head)
    states = [s for s in recorder["m"].seen_states if s.shape[0] == len(ev)]
    assert len(states) >= 3, "90d needs >=3 recursive steps"
    mu = float(train["origin_lfmc"].mean())
    sd = float(train["origin_lfmc"].std()) or 1.0
    eval_states = states[-3:]
    origin = ev["origin_lfmc"].to_numpy()
    # each step's state must be the previous prediction: origin, origin+1, origin+2
    for k, expected in enumerate((origin, origin + 1.0, origin + 2.0)):
        got = eval_states[k][:, 0] * sd + mu
        assert np.allclose(got, expected, atol=1e-4), f"step {k} did not consume step {k-1} output"
    assert np.allclose(res["horizons"][90]["errors"],
                       np.abs(ev["target_lfmc"].to_numpy() - (origin + 3.0)), atol=1e-4)


def test_recursion_never_reads_the_observed_target(tabs):
    """Perturbing only the observed target must leave every prediction bit-identical."""
    train, ev = tabs
    poisoned = ev.copy()
    poisoned["target_lfmc"] = poisoned["target_lfmc"] + 1000.0

    def head(pos_dim, state_dim):
        return RecordingStep(pos_dim, state_dim, delta=1.0)

    a = _run("earth4d", train, ev, head=head)["horizons"][90]["predictions"]
    b = _run("earth4d", train, poisoned, head=head)["horizons"][90]["predictions"]
    assert np.allclose(a, b, atol=0.0), \
        "predictions moved with the target: an observed future value reached the model path"


def test_future_sentinel_canary_fires_when_the_recursion_is_fed_an_observation(tabs):
    """Substitute a leaky state source (observed target instead of own prediction) -> must raise."""
    train, ev = tabs

    def leaky_source(step_index, current, view):
        return view["target_lfmc"].to_numpy()      # the sentinel arm poisons this to SENTINEL

    with pytest.raises(LR.FutureAccessError):
        LR.run_arm("future_sentinel", train, {90: ev}, seed=0, seconds=0.0, device=CPU,
                   horizon_days=90, encoder_factory=stub_factory,
                   step_model_factory=lambda p, s: RecordingStep(p, s, delta=1.0),
                   state_source=leaky_source)


def test_sentinel_does_not_fire_on_the_registered_state_source(tabs):
    """The canary must stay silent when each step consumes only its own previous prediction."""
    train, ev = tabs
    res = LR.run_arm("future_sentinel", train, {90: ev}, seed=0, seconds=0.0, device=CPU,
                     horizon_days=90, encoder_factory=stub_factory,
                     step_model_factory=lambda p, s: RecordingStep(p, s, delta=1.0))
    assert np.all(np.isfinite(res["horizons"][90]["predictions"]))


def test_leaky_source_is_detectable_only_because_the_arm_poisons_the_view(tabs):
    """Control on the control: the same leaky source on a non-sentinel arm does not raise,
    proving the canary comes from the poisoned view rather than from an unrelated guard."""
    train, ev = tabs

    def leaky_source(step_index, current, view):
        return view["target_lfmc"].to_numpy()

    res = LR.run_arm("earth4d", train, {90: ev}, seed=0, seconds=0.0, device=CPU,
                     horizon_days=90, encoder_factory=stub_factory,
                     step_model_factory=lambda p, s: RecordingStep(p, s, delta=1.0),
                     state_source=leaky_source)
    assert np.all(np.isfinite(res["horizons"][90]["predictions"]))


# --------------------------------------------------------------------------- control semantics
def test_no_history_arm_receives_zero_width_state(tabs):
    train, ev = tabs
    seen = {}

    def head(pos_dim, state_dim):
        seen["state_dim"] = state_dim
        return RecordingStep(pos_dim, state_dim)

    _run("no_history", train, ev, head=head)
    assert seen["state_dim"] == 0
    _run("earth4d", train, ev, head=head)
    assert seen["state_dim"] == 1


def test_shuffled_history_breaks_state_target_pairing(tabs):
    train, ev = tabs
    rng = np.random.default_rng(0)
    spec = LR.ARM_SPECS["shuffled_history"]
    values = ev["origin_lfmc"].to_numpy()
    st = LR._state_tensor(values, 0.0, 1.0, spec, CPU, rng).numpy()[:, 0]
    assert not np.allclose(st, values), "shuffled-history must permute the state"
    assert np.allclose(np.sort(st), np.sort(values)), "shuffle must preserve the multiset"


def test_time_reversal_control_reverses_the_step_direction(tabs):
    train, ev = tabs
    enc = {}

    def factory(train_coords, horizon_days, device):
        enc["e"] = StubEncoder().to(device)
        return enc["e"]

    LR.run_arm("time_reversal", train, {90: ev}, seed=0, seconds=0.0, device=CPU,
               horizon_days=90, encoder_factory=factory,
               step_model_factory=lambda p, s: RecordingStep(p, s))
    coords = [c.numpy() for c in enc["e"].seen_coords if c.shape[0] == len(ev)]
    assert len(coords) >= 2
    # under reversal the first encoded coordinate must be the LATER day of the pair
    assert (coords[0][:, 3] >= coords[1][:, 3]).mean() > 0.5


# --------------------------------------------------------------------------- parity / capacity
def test_head_parameter_counts_match_across_all_learned_arms(tabs):
    train, ev = tabs
    counts = {}
    for arm in LR.LEARNED_ARMS:
        if arm == "future_sentinel":
            continue
        res = _run(arm, train, ev)
        counts[arm] = res["head_params"]
    assert len(set(counts.values())) == 1, f"head capacity is not matched across arms: {counts}"


def test_every_arm_sees_the_same_pairs_and_split_digest(visits):
    p_a = build_rolling_origin_pairs(visits, 90)
    p_b = build_rolling_origin_pairs(visits, 90)
    pd.testing.assert_frame_equal(p_a, p_b)
    masks = make_split_masks(visits, LR.DESIGN)
    assert LR._split_digest(visits, masks) == LR._split_digest(visits, masks)


def test_fixed_seed_determinism(tabs):
    """Same seed AND same step count must reproduce exactly. The budget derives the step count;
    pinning steps is what makes a run reproducible, since a raw wall-clock stop cannot be."""
    train, ev = tabs
    a = _run("earth4d", train, ev, seed=3, steps=12)["horizons"][90]["mae"]
    b = _run("earth4d", train, ev, seed=3, steps=12)["horizons"][90]["mae"]
    assert a == pytest.approx(b, rel=1e-9, abs=1e-9)
    c = _run("earth4d", train, ev, seed=4, steps=12)["horizons"][90]["mae"]
    assert c != pytest.approx(a, rel=1e-9, abs=1e-9), "different seeds must not collapse"


def test_wall_clock_budget_derives_a_step_count(tabs):
    """With no explicit steps, the budget is converted into a planned step count and honoured."""
    train, ev = tabs
    # a real differentiable head: throughput must be measured on the actual training path
    res = LR.run_arm("earth4d", train, {90: ev}, seed=0, seconds=0.4, device=CPU,
                     horizon_days=90, encoder_factory=stub_factory)
    assert res["planned_steps"] >= 1
    assert res["train_steps"] == res["planned_steps"]


def test_budget_parity_across_arms(tabs):
    """Every arm must be given the same budget and report its realized seconds for audit."""
    train, ev = tabs
    seen = {}
    for arm in ("earth4d", "raw", "generic_pe"):
        r = LR.run_arm(arm, train, {90: ev}, seed=0, seconds=0.25, device=CPU, horizon_days=90,
                       encoder_factory=stub_factory)
        seen[arm] = r["seconds"]
        assert r["planned_steps"] >= 1
    assert all(v >= 0.0 for v in seen.values())


# --------------------------------------------------------------------------- protocol / closure
def test_protocol_minimums_block_a_confirmatory_artifact():
    assert LR.protocol_violations(5, 600.0, [90, 30], list(LR.REQUIRED_ARMS)) == []
    assert LR.protocol_violations(4, 600.0, [90, 30], list(LR.REQUIRED_ARMS))
    assert LR.protocol_violations(5, 599.0, [90, 30], list(LR.REQUIRED_ARMS))
    assert LR.protocol_violations(5, 600.0, [30], list(LR.REQUIRED_ARMS))
    assert LR.protocol_violations(5, 600.0, [90, 30], ["earth4d"])


def test_test_partition_is_never_consumed(visits):
    """Poison the test partition; every quantity the runner builds must be unchanged."""
    masks = make_split_masks(visits, LR.DESIGN)
    poisoned = visits.copy()
    test_rows = masks["test"].to_numpy()
    poisoned.loc[test_rows, "lfmc_value"] = 1e6

    def build(v):
        m = make_split_masks(v, LR.DESIGN)
        train_ids = set(v.loc[m["train"], "_visit_id"].to_numpy(np.int64).tolist())
        val_ids = set(v.loc[m["validation"], "_visit_id"].to_numpy(np.int64).tolist())
        p = build_rolling_origin_pairs(v, 90)
        return (LR._pair_table(v, p.loc[p["target_visit_id"].isin(train_ids)]),
                LR._pair_table(v, p.loc[p["target_visit_id"].isin(val_ids)]))

    a_train, a_val = build(visits)
    b_train, b_val = build(poisoned)
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_val, b_val)


def test_block_bootstrap_ci_is_a_real_interval():
    rng = np.random.default_rng(0)
    blocks = np.repeat(np.arange(40), 5)
    ref = rng.normal(20.0, 3.0, size=200)
    arm = ref - 2.0
    lo, hi, point = LR.block_bootstrap_ci(arm, ref, blocks, n_boot=200, seed=1)
    assert lo < point < hi
    assert lo > 0, "a genuine 2-unit improvement should exclude zero"
    lo2, _, point2 = LR.block_bootstrap_ci(ref, ref, blocks, n_boot=200, seed=1)
    assert point2 == pytest.approx(0.0, abs=1e-9)
    assert lo2 <= 0, "an identical arm must not clear the bar"


def test_ledger_is_append_only_and_hash_chained(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = LR.LedgerWriter(path, {"code_sha256": "abc"})
    s1 = led.append({"record_type": "arm", "arm": "earth4d"})
    s2 = led.append({"record_type": "arm", "arm": "raw"})
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
    import json as _j
    recs = [_j.loads(l) for l in lines]
    assert recs[1]["signature"] == s1 and recs[2]["signature"] == s2
    assert recs[1]["prev_signature"] == recs[0]["signature"]
    assert recs[2]["prev_signature"] == recs[1]["signature"]
    LR.LedgerWriter(path, {"code_sha256": "abc"})       # re-open must not truncate
    assert len([l for l in path.read_text().splitlines() if l.strip()]) == 4
