import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "autoresearch_data",
    Path(__file__).parents[1] / "main" / "editable_files" / "lib" / "data.py",
)
data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(data)


def test_legacy_family_marker_is_not_a_class():
    assert data._canonical_family("Asteraceae") == "Asteraceae"
    assert data._canonical_family("Asteraceae*") == "Asteraceae"


@pytest.mark.parametrize("label", ["", "*", "Aster*aceae", "Asteraceae**"])
def test_malformed_family_markers_fail(label):
    with pytest.raises(ValueError):
        data._canonical_family(label)


def test_prepared_schema_invalidates_legacy_tag(monkeypatch):
    settings = {"adapter": "california", "cache_dir": "data/deepcal", "n_neighbors": 16}
    legacy = hashlib.md5(json.dumps(
        {k: settings.get(k) for k in data._PREPARED_FIELDS}, sort_keys=True, default=str
    ).encode()).hexdigest()[:10]
    before = data.prepared_tag(settings)
    assert before != legacy
    monkeypatch.setattr(data, "PREPARED_SCHEMA", data.PREPARED_SCHEMA + 1)
    assert data.prepared_tag(settings) != before
