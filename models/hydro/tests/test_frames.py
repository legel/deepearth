"""The frame binaries. SIML has four participants, three here and a JS reader; SIMF is the
contract every solver writes to, laid out in `frames.py`."""

import json
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
    """`flood.js` cannot import `frames.py`, so its constants are pinned from this side."""
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


SIDECAR = frames.sidecar(t0="2026-01-15T00:00:00-08:00", timezone="America/Los_Angeles",
                         site="campanile", epsg=32610, anchor_utm=[565278.34, 4191891.74, 119.82],
                         fields={"depth": {"domain": [0, 0.5], "lut": "turbo"}},
                         provenance={"test": True})


def test_fields_round_trip_with_header_and_seconds(tmp_path):
    data = [np.random.default_rng(i).random((3, 4, 6)).astype(np.float32) for i in range(5)]
    times = [0.0, 30.5, 61.0, 91.5, 122.0]
    fields = (("depth", "m"), ("u", "m/s"), ("v", "m/s"))
    path = frames.write_fields(tmp_path / "f.bin", fields, times, data, cell_m=0.2,
                               origin=(-112.32, 112.32, 0.0), sidecar=SIDECAR)
    header, got_t, got = frames.read_fields(path)
    assert header.fields == fields and header.shape == (1, 4, 6) and header.n_frames == 5
    assert header.cell_m == 0.2 and header.origin == (-112.32, 112.32, 0.0)
    assert header.zf == pytest.approx((0.0, 0.2)) and header.sample == 0 and header.scale == 1.0
    assert header.version == 2
    assert got_t.dtype == np.float64 and np.array_equal(got_t, np.asarray(times))
    assert np.array_equal(got[:, :, 0], np.stack(data))
    assert json.loads((tmp_path / "f.bin.json").read_text())["fields"]["depth"]["lut"] == "turbo"


def test_header_matches_the_frames_contract_byte_for_byte(tmp_path):
    """The SIMF table in `frames.py`, offset by offset."""
    n, rows, cols, nf = 2, 3, 4, 2
    data = [np.arange(nf * rows * cols, dtype=np.float32).reshape(nf, rows, cols) * (k + 1)
            for k in range(n)]
    raw = frames.write_fields(tmp_path / "f.bin", (("depth", "m"), ("speed", "m/s")), [5.0, 10.0],
                              data, cell_m=1.5, origin=(10.0, 20.0, 0.5), sidecar=SIDECAR).read_bytes()
    assert raw[0:4] == b"SIMF"
    assert struct.unpack("<I", raw[4:8])[0] == 2
    assert struct.unpack("<I", raw[8:12])[0] == n
    assert struct.unpack("<I", raw[12:16])[0] == cols
    assert struct.unpack("<I", raw[16:20])[0] == rows
    assert struct.unpack("<I", raw[20:24])[0] == 1
    assert struct.unpack("<I", raw[24:28])[0] == nf
    assert struct.unpack("<I", raw[28:32])[0] == 0
    assert struct.unpack("<d", raw[32:40])[0] == 1.5
    assert struct.unpack("<ddd", raw[40:64]) == (10.0, 20.0, 0.5)
    assert struct.unpack("<d", raw[64:72])[0] == 1.0
    assert raw[72:104] == b"depth\0\0\0m\0\0\0\0\0\0\0speed\0\0\0m/s\0\0\0\0\0"
    assert np.frombuffer(raw, "<f4", count=2, offset=104).tolist() == [0.0, 1.5]
    record = 8 + 4 * nf * rows * cols
    assert len(raw) == 112 + n * record
    assert struct.unpack("<d", raw[112:120])[0] == 5.0
    assert struct.unpack("<d", raw[112 + record:120 + record])[0] == 10.0
    second = np.frombuffer(raw, "<f4", count=nf * rows * cols, offset=120 + record)
    assert np.array_equal(second.reshape(nf, rows, cols), data[1])
    sidecar = json.loads((tmp_path / "f.bin.json").read_text())
    assert set(sidecar) == frames.SIDECAR_KEYS


def test_writer_streams_and_patches_the_count(tmp_path):
    with frames.Writer(tmp_path / "f.bin", (2, 2), (("depth", "m"),), 1.0, (0.0, 0.0, 0.0), SIDECAR) as w:
        assert frames.Header.unpack((tmp_path / "f.bin").read_bytes()).n_frames == 0
        for t in (1.0, 2.0, 3.0):
            w.append(t, np.full((1, 2, 2), t, dtype=np.float32))
    header, times, data = frames.read_fields(tmp_path / "f.bin")
    assert header.n_frames == 3 and times.tolist() == [1.0, 2.0, 3.0] and data[2].max() == 3.0


def test_reader_counts_records_when_the_header_says_zero(tmp_path):
    w = frames.Writer(tmp_path / "f.bin", (2, 2), (("depth", "m"),), 1.0, (0.0, 0.0, 0.0), SIDECAR)
    w.append(0.0, np.ones((1, 2, 2), np.float32))
    w.append(9.0, np.ones((1, 2, 2), np.float32))
    w.fh.close()
    header, times, data = frames.read_fields(tmp_path / "f.bin")
    assert header.n_frames == 0 and times.tolist() == [0.0, 9.0] and data.shape == (2, 1, 1, 2, 2)


def test_writer_accepts_levels_and_faces_for_a_volume_field(tmp_path):
    with frames.Writer(tmp_path / "f.bin", (3, 2, 2), (("w", "m/s"),), 1.0, (0.0, 0.0, 0.0), SIDECAR,
                       zf=[0.0, 1.0, 3.0, 7.0]) as w:
        w.append(0.0, np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2))
    header, _, data = frames.read_fields(tmp_path / "f.bin")
    assert header.shape == (3, 2, 2) and header.zf == (0.0, 1.0, 3.0, 7.0)
    assert data.shape == (1, 1, 3, 2, 2) and data[0, 0, 2, 1, 1] == 11.0


def test_reader_accepts_version_one_without_faces(tmp_path):
    header = frames.Header(n_frames=1, shape=(1, 2, 2), fields=(("depth", "m"),), cell_m=0.5,
                           origin=(0.0, 0.0, 0.0), zf=(0.0, 0.5))
    raw = bytearray(header.pack()[:-8])
    raw[4:8] = struct.pack("<I", 1)
    raw += struct.pack("<d", 3.0) + np.full(4, 2.0, dtype="<f4").tobytes()
    (tmp_path / "v1.bin").write_bytes(bytes(raw))
    got, times, data = frames.read_fields(tmp_path / "v1.bin")
    assert got.version == 1 and got.zf == (0.0, 0.5) and times.tolist() == [3.0] and data.max() == 2.0


def test_float16_samples_are_read_back_as_float32(tmp_path):
    header = frames.Header(n_frames=1, shape=(1, 2, 2), fields=(("depth", "m"),), cell_m=1.0,
                           origin=(0.0, 0.0, 0.0), zf=(0.0, 1.0), sample=2)
    raw = header.pack() + struct.pack("<d", 0.0) + np.full(4, 0.25, dtype="<f2").tobytes()
    (tmp_path / "h.bin").write_bytes(raw)
    got, _, data = frames.read_fields(tmp_path / "h.bin")
    assert got.sample == 2 and data.dtype == np.float32 and data.max() == 0.25


def test_writer_refuses_an_incomplete_sidecar(tmp_path):
    with pytest.raises(AssertionError, match="sidecar lacks"):
        frames.Writer(tmp_path / "f.bin", (2, 2), (("depth", "m"),), 1.0, (0.0, 0.0, 0.0), {"t0": "x"})


def test_writer_rejects_a_frame_of_the_wrong_shape(tmp_path):
    with frames.Writer(tmp_path / "f.bin", (2, 2), (("depth", "m"),), 1.0, (0.0, 0.0, 0.0), SIDECAR) as w:
        with pytest.raises(ValueError):
            w.append(0.0, np.zeros((1, 3, 2), np.float32))


def test_fields_reader_rejects_a_depth_file(tmp_path):
    path = frames.write(tmp_path / "d.bin", np.zeros((1, 2, 2), np.float32), [0.0])
    with pytest.raises(AssertionError, match="not a SIMF file"):
        frames.read_fields(path)
