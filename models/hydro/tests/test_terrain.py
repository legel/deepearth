"""Flowline topology, the one part of conditioning that runs without a DEM or richdem.

`_network_order` decides the sequence in which reaches are carved, and the carve is
cumulative -- each reach starts from the elevation the reach above it ended at. Order it wrong
and the burn imposes no consistent downstream gradient, which is the failure mode `terrain`
documents: the largest flow accumulation anywhere in 46.78 km2 was 0.99 km2 and drainage
fragmented into disconnected pockets.
"""

import pytest
from shapely.geometry import LineString

from terrain import MIN_GRADIENT, _network_order


def test_headwaters_come_before_the_reach_they_feed():
    """Two headwaters joining a trunk: the trunk must be carved last."""
    left = LineString([(0, 0), (10, 10)])
    right = LineString([(20, 0), (10, 10)])
    trunk = LineString([(10, 10), (10, 30)])
    order, ends = _network_order([left, right, trunk], tol=1.0)
    assert order.index(2) > order.index(0)
    assert order.index(2) > order.index(1)


def test_a_shared_endpoint_collapses_to_one_node():
    """Confluence detection is a snap to `tol`; without it every reach is its own headwater."""
    a = LineString([(0, 0), (10, 10)])
    b = LineString([(10.2, 10.1), (10, 30)])  # starts within tol of a's end
    _, ends = _network_order([a, b], tol=1.0)
    assert ends[0][1] == ends[1][0], "endpoints within tol must share a node id"


def test_endpoints_further_apart_than_tol_stay_separate():
    a = LineString([(0, 0), (10, 10)])
    b = LineString([(40, 40), (10, 30)])
    _, ends = _network_order([a, b], tol=1.0)
    assert ends[0][1] != ends[1][0]


def test_every_reach_appears_exactly_once():
    lines = [LineString([(0, 0), (10, 10)]), LineString([(10, 10), (20, 20)]),
             LineString([(50, 50), (60, 60)])]  # third is disconnected
    order, _ = _network_order(lines, tol=1.0)
    assert sorted(order) == [0, 1, 2]


def test_a_cycle_does_not_drop_reaches_or_hang():
    """Should not occur in 3DHP, but a dropped reach would be an un-carved channel."""
    a = LineString([(0, 0), (10, 0)])
    b = LineString([(10, 0), (10, 10)])
    c = LineString([(10, 10), (0, 0)])
    order, _ = _network_order([a, b, c], tol=1.0)
    assert sorted(order) == [0, 1, 2]


def test_single_reach_is_trivially_ordered():
    order, ends = _network_order([LineString([(0, 0), (10, 10)])], tol=1.0)
    assert order == [0]
    assert ends[0][0] != ends[0][1]


def test_min_gradient_is_a_floor_not_a_target():
    """1e-4 was used as the enforced slope and throttled real reaches 2.4-5x; site3 falls at
    1.90e-3. The constant must stay well below anything a real network measures."""
    assert MIN_GRADIENT == pytest.approx(1e-4)
    assert MIN_GRADIENT < 1.90e-3 / 10
