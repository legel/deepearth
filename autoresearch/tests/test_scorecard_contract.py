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
    kw = dict(
        val_bpb=2.0356, macro=3.08,
        decomposition={"climate": 2.0004, "identity": 6.1, "clay": 5.2},
        revealed_dims={"climate": 17_762_000, "identity": 1, "clay": 1},
        benchmarks={"B08_species": 0.41, "B01_climate": 0.72},
        harmonic=0.3187, arithmetic=0.5707, seeds=2, noise_floor=0.0167,
        params=24_000_000, steps=1000, config="screen.yaml", agent="test",
        commit="abc1234", branch="exp/x", hardware={"gpu": "RTX PRO 6000", "gpus": 2},
    )
    kw.update(over)
    return scorecard(**kw)


def test_the_front_end_can_rely_on_the_shape():
    c = _card()
    assert c["schema"] == SCHEMA
    assert set(c) == {"schema", "generated_at", "agent", "headline", "model", "variables",
                      "benchmarks", "evidence", "delivery", "previous"}
    assert set(c["headline"]) == {"val_bpb", "macro", "harmonic", "arithmetic"}
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
    ({"seeds": 1}, "a single-seed number is not a result"),
    ({"benchmarks": {}}, "rule 32 requires 100% of the suite"),
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
