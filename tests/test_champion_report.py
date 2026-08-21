import json

import pytest

from deepearth.autoresearch.champion_report import combine_for_publication, parse_run


def _receipt(seed, harmonic, arithmetic, score):
    return {
        "protocol": "v2-held-species-pollinator-transfer",
        "seed": seed,
        "steps": 2291,
        "parameters": 25700000,
        "peak_vram_mb": 19000.0 + seed,
        "capability_suite": ["B1_species_from_env_top10"],
        "scores": {"B1_species_from_env_top10": score},
        "harmonic": harmonic,
        "arithmetic": arithmetic,
    }


def test_publication_requires_two_seeds():
    with pytest.raises(ValueError, match="exactly two"):
        combine_for_publication([_receipt(1337, 0.4, 0.5, 0.4)])


def test_two_full_precision_receipts_preserve_the_scorecard(tmp_path):
    first = _receipt(1337, 0.4, 0.5, 0.3)
    second = _receipt(1338, 0.6, 0.7, 0.5)
    log = tmp_path / "run.log"
    log.write_text("BENCHMARK_RECEIPT: " + json.dumps(first) + "\n")

    assert parse_run(log) == first
    combined = combine_for_publication([first, second])
    assert combined["seeds"] == [1337, 1338]
    assert combined["steps_completed"] == [2291, 2291]
    assert combined["parameters"] == 25700000
    assert combined["peak_vram_mb"] == 20338.0
    assert combined["harmonic"] == pytest.approx(0.5)
    assert combined["arithmetic"] == pytest.approx(0.6)
    assert combined["scores"] == {"B1_species_from_env_top10": pytest.approx(0.4)}
