"""The class table and the drag rule."""

import pytest

from physics import CLASSES, GROUND_CLASS, KAPPA, M_S_PER_KNOT, AeroClass, drag_density


def test_all_46_ontology_classes_are_present():
    assert len(CLASSES) == 46
    assert GROUND_CLASS in CLASSES


def test_solid_rule_is_closed_or_nearly_impermeable():
    assert CLASSES["facade_masonry"].solid and CLASSES["tree_trunk"].solid
    assert CLASSES["bell_tower"].solid, "porosity 0.05 blocks flow"
    assert not CLASSES["tree_canopy"].solid and not CLASSES["balcony"].solid
    assert all(c.solid or c.poro > 0.1 for c in CLASSES.values())


def test_canopy_drag_scales_with_leaf_area_per_volume():
    tree = CLASSES["tree_canopy"]
    assert drag_density(tree, 10.0, 0.2) == pytest.approx(tree.cd * tree.lai / 10.0)
    assert drag_density(tree, 20.0, 0.2) == pytest.approx(drag_density(tree, 10.0, 0.2) / 2)


def test_open_structures_without_foliage_use_their_porosity():
    fence = CLASSES["fence_gate"]
    assert drag_density(fence, 1.5, 0.2) == pytest.approx(fence.cd * (1 - fence.poro) / 0.2)


def test_solids_and_flat_columns_carry_no_drag():
    assert drag_density(CLASSES["roof_tile"], 5.0, 0.2) == 0.0
    assert drag_density(AeroClass(0.1, 0.5, 0.5, 2.0, False), 0.0, 0.2) == 0.0


def test_constants():
    assert KAPPA == 0.4
    assert M_S_PER_KNOT == pytest.approx(1852 / 3600, rel=1e-5)
