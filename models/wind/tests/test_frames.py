"""The frame binaries. SIML is byte-identical to models/hydro; SIMF shares its header."""

import struct
from pathlib import Path

import numpy as np
import pytest

import frames
from frames import Header, Writer, read_fields, write_fields


def test_2d_round_trip_preserves_frames_and_times(tmp_path):
    data = np.random.default_rng(0).random((5, 7, 9)).astype(np.float32)
    times = [0.0, 30.0, 60.0, 90.0, 120.0]
    got_times, got = frames.read(frames.write(tmp_path / "f.bin", data, times))
    assert np.array_equal(got, data)
    assert np.array_equal(got_times, np.asarray(times, dtype=np.float32))


def test_2d_header_matches_the_hydro_layout(tmp_path):
    raw = frames.write(tmp_path / "f.bin", np.zeros((3, 4, 5), np.float32), [0, 1, 2]).read_bytes()
    assert raw[:4] == b"SIML"
    assert frames.HEADER.unpack(raw[4:16]) == (3, 4, 5)
    assert len(raw) == 16 + 4 * 3 + 4 * 3 * 4 * 5


def _header(n=0, shape=(3, 4, 5), zf=(0.0, 1.0, 2.2, 3.6)) -> Header:
    return Header(n_frames=n, shape=shape, fields=frames.WIND_FIELDS, cell_m=0.5,
                  origin=(-10.0, 20.0, 3.5), zf=zf)


def test_fields_round_trip_preserves_everything(tmp_path):
    data = np.random.default_rng(1).random((3, 6, 3, 4, 5)).astype(np.float32)
    times = [0.0, 60.0, 120.0]
    path = write_fields(tmp_path / "v.bin", _header(), times, data)
    header, got_times, got = read_fields(path)
    assert np.array_equal(got, data)
    assert np.array_equal(got_times, np.float64(times))
    assert header.zf == pytest.approx(_header().zf, rel=1e-6)
    assert header == _header(n=3, zf=header.zf)
    header1, t1, one = read_fields(path, frame=1)
    assert np.array_equal(one[0], data[1]) and t1[0] == 60.0 and header1.n_frames == 3


def test_fields_header_offsets_are_the_shared_layout(tmp_path):
    """The header bytes are models/hydro's; the level faces follow the field table."""
    data = np.arange(2 * 6 * 3 * 4 * 5, dtype=np.float32).reshape(2, 6, 3, 4, 5)
    raw = write_fields(tmp_path / "v.bin", _header(), [0.0, 60.0], data).read_bytes()
    assert raw[:4] == b"SIMF"
    assert struct.unpack("<IIIIIII", raw[4:32]) == (2, 2, 5, 4, 3, 6, 0)
    assert struct.unpack("<ddddd", raw[32:72]) == (0.5, -10.0, 20.0, 3.5, 1.0)
    assert raw[72:88] == b"u\0\0\0\0\0\0\0m/s\0\0\0\0\0"
    assert raw[72 + 16 * 3:72 + 16 * 4] == b"vort_x\0\0" + b"1/s".ljust(8, b"\0")
    faces_at = 72 + 16 * 6
    assert np.array_equal(np.frombuffer(raw, "<f4", count=4, offset=faces_at), np.float32([0, 1, 2.2, 3.6]))
    record_at = faces_at + 16
    assert struct.unpack("<d", raw[record_at:record_at + 8])[0] == 0.0
    # Rows run south in the file: the first stored row of frame 0, field 0, level 0 is the solver's last.
    first_row = np.frombuffer(raw, "<f4", count=5, offset=record_at + 8)
    assert np.array_equal(first_row, data[0, 0, 0, -1])
    assert len(raw) == record_at + 2 * (8 + 4 * data[0].size)


def test_n_frames_is_patched_on_close(tmp_path):
    w = Writer(tmp_path / "v.bin", _header())
    for t in (0.0, 1.0, 2.0, 3.0):
        w.add(t, np.zeros((6, 3, 4, 5), np.float32))
    path = w.close()
    assert struct.unpack("<I", path.read_bytes()[8:12])[0] == 4
    assert read_fields(path)[0].n_frames == 4


def test_a_version_1_file_declares_uniform_levels(tmp_path):
    raw = bytearray(write_fields(tmp_path / "v.bin", _header(shape=(1, 4, 5), zf=(0.0, 0.5)),
                                 [0.0], np.zeros((1, 6, 1, 4, 5), np.float32)).read_bytes())
    raw[4:8] = struct.pack("<I", 1)
    without_faces = bytes(raw[:72 + 16 * 6]) + bytes(raw[72 + 16 * 6 + 8:])
    header = Header.unpack(without_faces)
    assert header.zf == (0.0, 0.5)


def test_compress_writes_a_gz_beside_the_product(tmp_path):
    import gzip

    data = np.zeros((2, 6, 3, 4, 5), np.float32)
    path = write_fields(tmp_path / "v.bin", _header(), [0.0, 1.0], data)
    sizes = frames.compress(path)
    gz = Path(str(path) + ".gz")
    assert gz.exists() and sizes == {"bytes": path.stat().st_size, "gz_bytes": gz.stat().st_size}
    assert gzip.decompress(gz.read_bytes()) == path.read_bytes()
    assert sizes["gz_bytes"] < sizes["bytes"]


def test_a_float16_view_product_halves_the_native_bytes(tmp_path):
    """The viewer copy carries u, v, w only, in float16; vorticity is recomputed from them."""
    native = _header()
    view = Header(n_frames=0, shape=native.shape, fields=frames.WIND_FIELDS[:3],
                  cell_m=native.cell_m, origin=native.origin, zf=native.zf, sample=2)
    values = np.random.default_rng(2).standard_normal((1, 3, 3, 4, 5)).astype(np.float32)
    path = write_fields(tmp_path / "view.bin", view, [0.0], values)
    header, _, got = read_fields(path)
    assert header.sample == 2 and header.fields == frames.WIND_FIELDS[:3]
    assert got == pytest.approx(values, abs=1e-2), "float16 keeps three significant digits"
    assert path.stat().st_size < write_fields(
        tmp_path / "n.bin", _header(shape=native.shape), [0.0],
        np.zeros((1, 6, 3, 4, 5), np.float32)).stat().st_size / 3


def test_wind_fields_are_velocity_then_vorticity_with_units():
    assert [f[0] for f in frames.WIND_FIELDS] == ["u", "v", "w", "vort_x", "vort_y", "vort_z"]
    assert {f[1] for f in frames.WIND_FIELDS[:3]} == {"m/s"}
    assert {f[1] for f in frames.WIND_FIELDS[3:]} == {"1/s"}


def test_rejects_wrong_shapes_and_foreign_files(tmp_path):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"NOPE" + b"\0" * 80)
    with pytest.raises(AssertionError, match="not a SIML file"):
        frames.read(bad)
    with pytest.raises(AssertionError, match="not a SIMF file"):
        read_fields(bad)
    with pytest.raises(AssertionError, match="faces for"):
        Header(0, (3, 4, 5), frames.WIND_FIELDS, 1.0, (0, 0, 0), (0.0, 1.0)).pack()
    with Writer(tmp_path / "v.bin", _header()) as w, pytest.raises(AssertionError, match="expected"):
        w.add(0.0, np.zeros((6, 3, 5, 4), np.float32))
