"""The CONUS render stacks mark missing values as NaN. A placeholder (gdalwarp's float default -3.4e38, or integer
sentinels -32768 / -9999) passes the renderer's ``isfinite`` check and would be clamped into a fake prediction
(provenance 2026-10-04). Each stack is skipped when it has not been built."""
import json

import numpy as np
import pytest

from ranges.config import data_root

STACKS = [data_root() / "work/conus240/conus240_stack.f32", data_root() / "work/fine/conus240_fine.f32"]


@pytest.mark.parametrize("path", STACKS, ids=[p.stem for p in STACKS])
def test_no_placeholder_values(path):
    if not path.exists():
        pytest.skip(f"{path.name} not built (scripts/build_conus_stack.py, build_fine_stacks.py)")
    meta = json.loads(path.with_suffix(".json").read_text())
    m = np.memmap(path, dtype=np.float32, mode="r", shape=tuple(meta["shape"]))
    rows = np.random.default_rng(0).choice(m.shape[1], 300, replace=False)
    for k in range(m.shape[0]):
        v = m[k, np.sort(rows), :]
        assert not (np.isfinite(v) & ((np.abs(v) > 1e30) | (v == -32768) | (v == -9999))).any(), meta["variables"][k]
