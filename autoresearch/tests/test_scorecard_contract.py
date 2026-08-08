"""The scorecard is a published contract, so its shape is tested rather than described.

`LOOP-deepearth-best` is read by things that are not this loop -- the front end, and the delivery
skill. That makes a malformed card worse than a missing one: a card with no benchmarks renders as "no
benchmarks were run" when the truth is "someone forgot to pass them", and there is no way to tell the
two apart after the fact. `scorecard()` therefore raises instead of defaulting, and these tests pin the
cases where raising is the only correct behaviour.

No torch, no GPU, no network: this runs anywhere the delivery skill runs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main" / "harness"))
from coordinator import SCHEMA, scorecard                          # noqa: E402


def _card(**over):
    benchmarks = over.pop("benchmarks", {"B08_species": 0.41, "B01_climate": 0.72,
                                         "B55_pollinator_phylo_transfer_recall": 0.04,
                                         "B56_family_phylo_graph_gain": 0.20})
    kw = dict(
        val_bpb=2.0356, macro=3.08,
        decomposition={"climate": 2.0004, "identity": 6.1, "clay": 5.2},
        revealed_dims={"climate": 17_762_000, "identity": 1, "clay": 1},
        benchmark_runs=[benchmarks, benchmarks],
        benchmark_protocol="v3-human-capability-gate",
        capability_suite=("B01_climate", "B08_species"),
        training_seeds=(1337, 1339), noise_floor=0.0167,
        params=24_000_000, steps=1000, config="screen.yaml", agent="test",
        commit="abc1234", branch="exp/x", hardware={"gpu": "RTX PRO 6000", "gpus": 2},
    )
    kw.update(over)
    return scorecard(**kw)


def test_the_front_end_can_rely_on_the_shape():
    c = _card()
    assert c["schema"] == SCHEMA
    assert set(c) == {"schema", "generated_at", "agent", "headline", "diagnostics", "model", "variables",
                      "benchmarks", "benchmark_runs", "evidence", "delivery", "previous"}
    assert set(c["headline"]) == {"harmonic", "arithmetic"}
    assert c["diagnostics"] == {"val_bpb": 2.0356, "macro": 3.08}
    assert c["evidence"]["benchmark_protocol"] == "v3-human-capability-gate"
    assert c["evidence"]["hardware"] == {"gpu": "RTX PRO 6000", "gpus": 2}
    assert c["model"]["params"] == 24_000_000 and c["model"]["params_m"] == 24.0
    assert json.loads(json.dumps(c)) == c, "must survive a JSON round trip"


def test_the_same_run_published_twice_is_identical():
    """Determinism is what lets a consumer diff two cards and see only what actually changed."""
    a, b = _card(), _card()
    a.pop("generated_at"), b.pop("generated_at")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_variables_carry_their_share_of_the_dimension_weighted_aggregate():
    """The share is why the gate needs a coverage rule: one variable can carry nearly all of it."""
    c = _card()
    shares = {v["name"]: v["share_pct"] for v in c["variables"]}
    assert shares["climate"] > 99.0, "climate dominates by revealed dims"
    assert c["variables"][0]["name"] == "climate", "sorted by share, descending"
    assert sum(shares.values()) == pytest.approx(100.0, abs=0.01)


def test_a_directional_variable_contributes_one_dimension():
    """clay is retrieval-scored against a frozen bank, so its 1024-d target is ONE revealed dim. The
    earlier native-width analysis put clay at 30.1% of the aggregate; it is ~0.076%."""
    c = _card()
    clay = next(v for v in c["variables"] if v["name"] == "clay")
    assert clay["revealed_dims"] == 1
    assert clay["share_pct"] < 0.01


@pytest.mark.parametrize("bad, why", [
    ({"benchmark_runs": [{"B01_climate": 0.5}]}, "a single-seed number is not a result"),
    ({"benchmarks": {}}, "an empty benchmark suite is not a result"),
    ({"decomposition": {}}, "val_bpb without its lens is a bare number"),
    ({"val_bpb": float("nan")}, "NaN is a crash, not a score"),
    ({"val_bpb": float("inf")}, "inf is a crash, not a score"),
    ({"val_bpb": 0.0}, "a loss of zero means the metric broke"),
    ({"val_bpb": None}, "None is a missing measurement"),
    ({"revealed_dims": {"climate": 17_762_000}}, "shares cannot be computed without every variable"),
])
def test_a_broken_run_fails_at_publish_time(bad, why):
    with pytest.raises(ValueError):
        _card(**bad)


def test_undelivered_is_stated_not_omitted():
    """Present-and-null means "not shipped"; absent would mean "nobody recorded whether it shipped"."""
    d = _card()["delivery"]
    assert d["pr"] is None and d["merged"] is False
    assert set(d) == {"pr", "pr_url", "base_commit", "merged", "delivered_at"}


def test_the_architecture_is_walked_from_the_live_model_not_declared():
    """A declared architecture is one more field that gets copied forward and is quietly wrong."""
    torch = pytest.importorskip("torch")
    from coordinator import architecture_graph

    net = torch.nn.Module()
    net.encoder = torch.nn.Sequential(torch.nn.Linear(4, 256), torch.nn.Linear(256, 128))
    net.head = torch.nn.Linear(128, 10)
    g = architecture_graph(net, max_depth=2)

    ids = {n["id"] for n in g["nodes"]}
    assert {"model", "encoder", "head", "encoder.0"} <= ids
    assert {"from": "model", "to": "encoder"} in g["edges"]
    assert g["edge_kind"] == "containment", "not dataflow -- fx cannot trace the CUDA hash encoders"

    root = next(n for n in g["nodes"] if n["id"] == "model")
    assert root["params"] == sum(p.numel() for p in net.parameters())
    assert root["params_pct"] == 100.0
    children = [n for n in g["nodes"] if n["depth"] == 1]
    assert sum(n["params_pct"] for n in children) == pytest.approx(100.0, abs=0.1), \
        "capacity must account for itself, so the graph shows where the parameters actually sit"
    assert architecture_graph(net, max_depth=2) == g, "must be deterministic"


def test_the_scorecard_key_cannot_be_clobbered_by_a_board_write():
    """BOARD.format(variable="best") IS the scorecard key, so publish_result(variable="best") wrote a
    board record over the front end's contract. It happened; the name is reserved now."""
    from coordinator import BEST, BOARD, Coordinator

    assert BOARD.format(variable="best") == BEST, "the collision this guards against"
    with pytest.raises(ValueError, match="scorecard"):
        Coordinator.publish_result(Coordinator.__new__(Coordinator), variable="best", description="x",
                                   val_bpb=1.0, decomposition={}, status="keep", config="y")


def test_retrieval_floors_separate_apparent_headroom_from_real():
    """phylo's bank holds 925 unique species across 4096 rows drawn with replacement, so a perfect
    predictor still pays ~2.11 nats. Reading its 9.64 bits as headroom overstates it by a third."""
    c = _card(retrieval_floors={"clay": 2.1132})
    clay = next(v for v in c["variables"] if v["name"] == "clay")
    assert clay["floor"] == pytest.approx(2.1132 / 0.6931, abs=0.01), "reported in bits, measured in nats"
    assert clay["headroom"] == pytest.approx(clay["bits_per_dim"] - clay["floor"], abs=1e-6)

    gaussian = next(v for v in c["variables"] if v["name"] == "climate")
    assert gaussian["floor"] is None, "no floor for a variable that is not retrieval-scored"
    assert gaussian["headroom"] is None, "absent, not zero -- unknown is not the same as none"


def test_only_human_capabilities_enter_the_two_headline_means():
    a = _card()
    b = _card(benchmarks={"B08_species": 0.41, "B01_climate": 0.72,
                          "B55_pollinator_phylo_transfer_recall": 0.99,
                          "B56_family_phylo_graph_gain": -0.99})
    assert a["headline"] == b["headline"], "quarantine and mechanism evidence cannot move the gate"
    roles = {row["name"]: row["role"] for row in b["benchmarks"]}
    assert roles["B55_pollinator_phylo_transfer_recall"] == "quarantined"
    assert roles["B56_family_phylo_graph_gain"] == "mechanism"


def test_headline_is_the_mean_of_per_seed_scores():
    fixed = {"B55_pollinator_phylo_transfer_recall": 0.04,
             "B56_family_phylo_graph_gain": 0.20}
    c = _card(benchmark_runs=[{"B08_species": 0.10, "B01_climate": 0.90, **fixed},
                              {"B08_species": 0.50, "B01_climate": 0.50, **fixed}])
    assert c["headline"]["harmonic"] == pytest.approx((0.18 + 0.50) / 2, abs=1e-6)
    assert c["headline"]["harmonic"] != pytest.approx(0.42, abs=1e-3), \
        "harmonic-of-mean is not the approved mean-of-seed-harmonics"
    assert c["evidence"]["floors"]["harmonic"] == pytest.approx(0.32, abs=1e-6)


def test_capability_suite_must_match_exactly_what_the_run_reports():
    with pytest.raises(ValueError, match="capability_suite"):
        _card(capability_suite=("B01_climate",))
    with pytest.raises(ValueError, match="quarantined measurements"):
        _card(benchmarks={"B08_species": 0.41, "B01_climate": 0.72})


def _publisher(incumbent, monkeypatch):
    from coordinator import Coordinator
    coord = Coordinator(agent_id="test", api_key="test")
    monkeypatch.setattr(coord, "best", lambda: incumbent)
    monkeypatch.setattr(coord, "put", lambda *args, **kwargs: True)
    return coord


def test_promotion_uses_harmonic_not_val_bpb(monkeypatch):
    old = _card(benchmarks={"B08_species": 0.40, "B01_climate": 0.70,
                            "B55_pollinator_phylo_transfer_recall": 0.04,
                            "B56_family_phylo_graph_gain": 0.20})
    new = _card(val_bpb=9.0, macro=9.0,
                benchmarks={"B08_species": 0.42, "B01_climate": 0.71,
                            "B55_pollinator_phylo_transfer_recall": 0.01,
                            "B56_family_phylo_graph_gain": -0.80})
    assert _publisher(old, monkeypatch).publish_best(new)


def test_arithmetic_is_the_breadth_guard(monkeypatch):
    old = _card(benchmarks={"B08_species": 0.10, "B01_climate": 0.90,
                            "B55_pollinator_phylo_transfer_recall": 0.04,
                            "B56_family_phylo_graph_gain": 0.20})
    new = _card(benchmarks={"B08_species": 0.12, "B01_climate": 0.85,
                            "B55_pollinator_phylo_transfer_recall": 0.04,
                            "B56_family_phylo_graph_gain": 0.20})
    assert new["headline"]["harmonic"] > old["headline"]["harmonic"]
    assert not _publisher(old, monkeypatch).publish_best(new)


def test_one_capability_regression_is_not_a_suite_wide_veto(monkeypatch):
    old = _card(benchmarks={"B08_species": 0.10, "B01_climate": 0.90,
                            "B55_pollinator_phylo_transfer_recall": 0.04,
                            "B56_family_phylo_graph_gain": 0.20})
    new = _card(benchmarks={"B08_species": 0.20, "B01_climate": 0.80,
                            "B55_pollinator_phylo_transfer_recall": 0.04,
                            "B56_family_phylo_graph_gain": 0.20})
    assert new["headline"]["harmonic"] > old["headline"]["harmonic"]
    assert new["headline"]["arithmetic"] == old["headline"]["arithmetic"]
    assert _publisher(old, monkeypatch).publish_best(new)


def test_harmonic_gain_must_beat_incumbent_seed_spread(monkeypatch):
    fixed = {"B55_pollinator_phylo_transfer_recall": 0.04,
             "B56_family_phylo_graph_gain": 0.20}
    old = _card(benchmark_runs=[{"B08_species": 0.10, "B01_climate": 0.90, **fixed},
                                {"B08_species": 0.50, "B01_climate": 0.50, **fixed}])
    new = _card(benchmarks={"B08_species": 0.50, "B01_climate": 0.50, **fixed})
    assert new["headline"]["harmonic"] > old["headline"]["harmonic"]
    assert not _publisher(old, monkeypatch).publish_best(new)


def test_protocol_change_freezes_comparison(monkeypatch):
    old = _card()
    new = _card(benchmark_protocol="v4-future")
    assert not _publisher(old, monkeypatch).publish_best(new)


def test_publisher_recomputes_the_headline_before_writing(monkeypatch):
    old, new = _card(), _card()
    new["headline"]["harmonic"] += 0.1
    with pytest.raises(ValueError, match="harmonic does not match"):
        _publisher(old, monkeypatch).publish_best(new)
