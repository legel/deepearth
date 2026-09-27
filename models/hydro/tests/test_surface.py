"""Surface parameterisation: the class table, and the precedence rule that fixed the channel."""

import numpy as np
import pytest

import surface
from surface import CLASSES, CODES, SAM3_PROMPTS, _class_field, manning_spread


def test_every_class_has_a_code_and_the_codes_are_dense():
    assert set(CODES) == set(CLASSES)
    assert sorted(CODES.values()) == list(range(1, len(CLASSES) + 1))
    assert 0 not in CODES.values(), "0 is reserved for unlabelled"


@pytest.mark.parametrize("name,cls", sorted(CLASSES.items()))
def test_class_parameters_are_physical(name, cls):
    assert 0.005 < cls.manning_n < 0.30, f"{name} n={cls.manning_n}"
    assert 0.0 <= cls.impervious_fraction <= 1.0
    assert 0.0 <= cls.surface_storage_m < 0.02
    assert cls.vision_ks_mm_hr >= 0.0 and cls.vision_storage_m >= 0.0
    assert cls.basis, f"{name} has no stated basis for its roughness"


def test_sealed_classes_neither_infiltrate_nor_store_soil_water():
    for name in ("water", "building_roof", "road_paved"):
        assert CLASSES[name].impervious_fraction == 1.0
        assert CLASSES[name].vision_ks_mm_hr == 0.0
        assert CLASSES[name].vision_storage_m == 0.0


def test_roughness_ordering_is_physically_ordered():
    n = {k: c.manning_n for k, c in CLASSES.items()}
    assert n["road_paved"] < n["building_roof"] < n["bare_soil"] < n["grass_turf"]
    assert n["grass_turf"] < n["shrub_scrub"] < n["wetland_marsh"] < n["tree_canopy"]


def test_spread_is_the_documented_order_of_magnitude():
    """A scalar throws this away: the field spans 9.2x, the scalar spans 1x."""
    assert manning_spread() == pytest.approx(9.2, abs=0.1)


def test_wetland_has_no_prompts_because_a_sensor_cannot_see_a_water_table():
    assert "wetland_marsh" not in SAM3_PROMPTS
    assert set(SAM3_PROMPTS) == set(CLASSES) - {"wetland_marsh"}
    assert all(len(p) >= 1 for p in SAM3_PROMPTS.values())


def test_prompt_aggregation_maps_many_phrases_to_one_class():
    """SAM3 names one surface several things; the prompt list is the aggregation layer."""
    assert len(SAM3_PROMPTS["road_paved"]) > 1
    phrases = [p for v in SAM3_PROMPTS.values() for p in v]
    assert len(phrases) == len(set(phrases)), "a phrase must not claim two classes"


def test_unlabelled_cells_keep_the_scalar_fallback():
    """SAM3 left 17.4 % of the scene unlabelled, so the fallback is load-bearing."""
    codes = np.zeros((4, 4), dtype=np.int32)
    codes[0, 0] = CODES["road_paved"]
    n = _class_field(codes, "manning_n", 0.040)
    assert n[0, 0] == pytest.approx(CLASSES["road_paved"].manning_n)
    assert np.all(n[1:] == pytest.approx(0.040))


def test_channel_roughness_sits_between_bed_and_canopy():
    """The fix for nadir imagery putting forest roughness on a creek bed."""
    assert CLASSES["road_paved"].manning_n < surface.CHANNEL_MANNING_N
    assert surface.CHANNEL_MANNING_N < CLASSES["tree_canopy"].manning_n
    assert surface.CHANNEL_MANNING_N == pytest.approx(0.045)


def test_parameter_table_round_trips_every_class():
    table = surface.parameter_table()
    assert set(table) == set(CLASSES)
    assert table["tree_canopy"]["manning_n"] == pytest.approx(0.120)
    assert table["tree_canopy"]["vision_ks_mm_hr"] == pytest.approx(210.0)
