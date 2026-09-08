"""The depth-frame binary. One format, four participants -- three here and a JS reader."""

import struct
from pathlib import Path

import numpy as np
import pytest

import frames


def test_round_trip_preserves_frames_and_times(tmp_path):
    data = np.random.default_rng(0).random((5, 7, 9)).astype(np.float32)
    times = [0.0, 30.0, 60.0, 90.0, 120.0]
    path = frames.write(tmp_path / "f.bin", data, times)
    got_times, got = frames.read(path)
    assert np.array_equal(got, data)
    assert np.array_equal(got_times, np.asarray(times, dtype=np.float32))


def test_header_matches_the_documented_layout(tmp_path):
    """The JS reader hardcodes these offsets; if they move, it breaks silently."""
    path = frames.write(tmp_path / "f.bin", np.zeros((3, 4, 5), np.float32), [0, 1, 2])
    raw = path.read_bytes()
    assert raw[:4] == b"SIML"
    assert frames.HEADER.unpack(raw[4:16]) == (3, 4, 5)
    assert len(raw) == 16 + 4 * 3 + 4 * 3 * 4 * 5


def test_byte_offsets_match_the_javascript_reader(tmp_path):
    """`flood.js` cannot import `frames.py`, so its constants are pinned from this side.

    The JS reads getUint32 at 4/8/12, Float32Array(buf, 16, n) for the times and
    Float32Array(buf, 16 + 4n, n*rows*cols) for the depths. If those move, the viewer renders a
    plausible-looking wrong animation rather than failing.
    """
    n, rows, cols = 4, 6, 5
    data = np.arange(n * rows * cols, dtype=np.float32).reshape(n, rows, cols)
    times = [0.0, 15.0, 30.0, 45.0]
    raw = frames.write(tmp_path / "f.bin", data, times).read_bytes()

    assert struct.unpack("<I", raw[4:8])[0] == n
    assert struct.unpack("<I", raw[8:12])[0] == rows
    assert struct.unpack("<I", raw[12:16])[0] == cols
    assert np.array_equal(np.frombuffer(raw, "<f4", count=n, offset=16), np.float32(times))
    depths = np.frombuffer(raw, "<f4", count=n * rows * cols, offset=16 + 4 * n)
    assert np.array_equal(depths.reshape(n, rows, cols), data)
    # The JS slices frame k as [k*rows*cols, (k+1)*rows*cols) -- row-major, frames outermost.
    assert np.array_equal(depths[2 * rows * cols:3 * rows * cols], data[2].ravel())


def test_the_javascript_reader_declares_the_same_layout():
    """The layout comment in flood.js is the contract; drift there is silent."""
    js = (Path(__file__).resolve().parents[1] / "viewer/static/js/flood.js").read_text()
    assert "getUint32(4, true)" in js and "getUint32(8, true)" in js
    assert "getUint32(12, true)" in js
    assert "new Float32Array(buf, 16, n)" in js
    assert "new Float32Array(buf, 16 + 4 * n, n * rows * cols)" in js
    assert "'SIML'" in js


def test_rejects_a_file_that_is_not_ours(tmp_path):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"NOPE" + b"\0" * 32)
    with pytest.raises(AssertionError, match="not a SIML file"):
        frames.read(bad)


def test_rejects_mismatched_times(tmp_path):
    with pytest.raises(AssertionError, match="times for"):
        frames.write(tmp_path / "f.bin", np.zeros((3, 2, 2), np.float32), [0.0])


def test_rejects_a_non_stack(tmp_path):
    with pytest.raises(AssertionError, match="rows, cols"):
        frames.write(tmp_path / "f.bin", np.zeros((4, 4), np.float32), [0.0])
