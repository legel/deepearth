"""UTC timestamps and resumable per-site stores."""

from datetime import datetime, timezone

import numpy as np

from bands import GROUPS
from store import META_FIELDS, SiteStore, from_us, to_us


def test_timestamps_utc_microseconds():
    assert to_us("1970-01-01T00:00:00Z") == 0
    us = to_us("2018-01-03T15:51:05.722000Z")
    assert from_us(us) == datetime(2018, 1, 3, 15, 51, 5, 722000, tzinfo=timezone.utc)
    assert to_us("2018-01-03T15:51:05.722000+00:00") == us
    assert to_us("2018-01-03T16:51:05.722000+01:00") == us
    assert to_us("2018-01-03T15:51:05Z") == us - 722000


def _chip(i):
    c = {g.name: np.full((len(g.bands), g.size, g.size) if g.name in ("r10", "r20", "r60") else (g.size, g.size),
                         i, dtype=g.dtype) for g in GROUPS}
    c["time_us"] = to_us("2020-01-01T00:00:00Z") + i
    for k, dt in META_FIELDS.items():
        c[k] = (f"k{i}" if k == "acq_key" else "x") if dt == "str" else (True if dt == "bool" else i)
    return c


def test_store_append_and_resume(tmp_path):
    p = tmp_path / "S.zarr"
    s = SiteStore(p, "S", 1.0, 2.0)
    s.append([_chip(i) for i in range(5)])
    assert len(s) == 5 and s.done_keys() == {f"k{i}" for i in range(5)}
    # Simulate a crash after the arrays were extended but before the commit counter moved.
    s.root["r10"].append(np.zeros((3, 4, 30, 30), np.uint16))
    s.root["meta"]["acq_key"].append(np.array(["z1", "z2", "z3"], dtype=object))
    s2 = SiteStore(p, "S", 1.0, 2.0)
    assert len(s2) == 5 and s2.root["r10"].shape[0] == 5 and s2.root["meta"]["acq_key"].shape[0] == 5
    assert s2.done_keys() == {f"k{i}" for i in range(5)}
    s2.append([_chip(i) for i in range(5, 8)])
    s3 = SiteStore(p, "S", 1.0, 2.0)
    assert len(s3) == 8
    assert int(s3.root["r10"][7, 0, 0, 0]) == 7 and s3.root["cld"].shape == (8, 15, 15)
    assert s3.root.attrs["site_id"] == "S"


def test_gather_site_resumes_only_missing(tmp_path, monkeypatch):
    import gather
    from stac import Scene

    scenes = [Scene(source="earth-search-c1", item_id=f"i{i}", datetime_utc=f"2020-01-0{i + 1}T00:00:00Z",
                    platform="sentinel-2a", mgrs_tile="T", epsg=32610, relative_orbit=1, datatake_id="",
                    acq_key=f"k{i}", processing_baseline="05.00", boa_add_offset=-1000, generation_time="",
                    sun_azimuth=0, sun_elevation=0, view_azimuth=0, view_incidence=0, scene_cloud_cover=0)
              for i in range(6)]
    calls, fail = [], {"k2", "k4"}

    def fake_read(s, lon, lat, token):
        calls.append(s.acq_key)
        if s.acq_key in fail:
            raise IOError("simulated")
        c = _chip(int(s.acq_key[1:]))
        c.update(acq_key=s.acq_key)
        return c

    monkeypatch.setattr(gather, "scenes_for", lambda lon, lat: (scenes, {"earth-search-c1": 6}))
    monkeypatch.setattr(gather, "read_scene", fake_read)
    for d in ("sites", "scenes", "failures", "status"):
        (tmp_path / d).mkdir()

    class Row:
        site_id, lat, lon = "S", 1.0, 2.0

    st = gather.gather_site(Row, tmp_path, threads=2, token=None, batch=2)
    assert st["chips"] == 4 and st["failed"] == 2 and not st["complete"]
    fail.clear()
    calls.clear()
    st = gather.gather_site(Row, tmp_path, threads=2, token=None, batch=2)
    assert sorted(calls) == ["k2", "k4"] and st["chips"] == 6 and st["complete"]
