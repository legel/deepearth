from pathlib import Path

import yaml


def test_deepcal_promotion_uses_fixed_steps():
    config = yaml.safe_load((Path(__file__).parents[1] / "autoresearch" / "deepcal.yaml").read_text())
    assert config["training"]["steps"] == 8000
    assert "time_budget_s" not in config["training"]
