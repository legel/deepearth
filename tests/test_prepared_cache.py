import importlib
import sys
import types

import pytest

from deepearth.autoresearch.data import prepared_cache_path


def test_prepared_cache_follows_the_data_directory(tmp_path):
    data = {"adapter": "california", "cache_dir": str(tmp_path), "n_neighbors": 16, "time_axis": True}
    path = prepared_cache_path(data)

    assert path.parent == tmp_path.resolve()
    assert path.name.startswith("prepared_")
    assert prepared_cache_path({**data, "n_neighbors": 24}) != path


def test_prepared_cache_key_includes_metadata_source(tmp_path):
    data = {"adapter": "california", "cache_dir": str(tmp_path), "meta_path": "a.csv"}

    assert prepared_cache_path(data) != prepared_cache_path({**data, "meta_path": "b.csv"})


def test_prepared_cache_key_tracks_optional_inputs(tmp_path):
    data = {"adapter": "california", "cache_dir": str(tmp_path)}
    before = prepared_cache_path(data)
    (tmp_path / "gbif_worldclim_tokens.npz").write_bytes(b"worldclim")
    after = prepared_cache_path(data)
    (tmp_path / after.name).write_bytes(b"prepared")

    assert after != before
    assert prepared_cache_path(data) == after


def test_training_requires_prepare_to_write_the_cache(tmp_path, monkeypatch):
    fusion = types.ModuleType("deepearth.core.fusion")
    fusion.DeepEarth = fusion.Variable = object
    monkeypatch.setitem(sys.modules, "deepearth.core.fusion", fusion)
    train = importlib.import_module("deepearth.autoresearch.train")
    monkeypatch.setattr(train.data_module, "build", lambda *args, **kwargs: pytest.fail("training built the cache"))
    config = {"training": {"seed": 0}, "data": {"adapter": "california", "cache_dir": str(tmp_path)}}

    with pytest.raises(FileNotFoundError, match="autoresearch.prepare"):
        train.train_and_evaluate(config, "cpu")
