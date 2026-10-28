"""Stage 1 and stage 2 on synthetic archives: only QC-0 values survive, and the merge takes one source per
variable-year in priority order."""

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

import build as B
import readers as R
import sources as S


def stamps(start: str, n: int, step: int = 30):
    t = pd.date_range(start, periods=n, freq=f"{step}min")
    return t.strftime("%Y%m%d%H%M").astype(np.int64), (t + pd.Timedelta(minutes=step)).strftime("%Y%m%d%H%M").astype(np.int64)


def oneflux_zip(path, site="XX-Tst", n=48 * 40):
    t0, t1 = stamps("2014-03-01", n)
    rng = np.random.default_rng(0)
    hour = (np.arange(n) % 48) / 2
    sw = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * 800
    nee = -0.02 * sw + 3 + rng.normal(0, 0.5, n)
    qc = np.where(np.arange(n) % 5 == 0, 2, 0)  # every fifth half-hour is an MDS fill
    df = pd.DataFrame({"TIMESTAMP_START": t0, "TIMESTAMP_END": t1, "NEE_VUT_REF": nee, "NEE_VUT_REF_QC": qc,
                       "SW_IN_F_MDS": sw, "SW_IN_F_MDS_QC": 0, "TA_F_MDS": 10 + rng.normal(0, 1, n),
                       "TA_F_MDS_QC": 0, "USTAR": 0.4, "PA": -9999})
    df.loc[101, "NEE_VUT_REF"] = 500.0  # physically impossible
    bif = pd.DataFrame([[site, 1, "GRP_LOCATION", "LOCATION_LAT", "38.0"], [site, 1, "GRP_LOCATION", "LOCATION_LONG", "-121.6"],
                        [site, 2, "GRP_UTC_OFFSET", "UTC_OFFSET", "-8"], [site, 3, "GRP_IGBP", "IGBP", "GRA"],
                        [site, 4, "GRP_FLUX_REF", "REF_VARIABLE", "NEE"], [site, 4, "GRP_FLUX_REF", "REF_VERSION", "VUT"],
                        [site, 4, "GRP_FLUX_REF", "REF_YEAR", "2014"], [site, 4, "GRP_FLUX_REF", "REF_USTAR_THRESHOLD", "0.2"],
                        [site, 4, "GRP_FLUX_REF", "REF_TIME_RESOLUTION", "HH"]],
                       columns=["SITE_ID", "GROUP_ID", "VARIABLE_GROUP", "VARIABLE", "DATAVALUE"])
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"AMF_{site}_FLUXNET_FLUXMET_HH_2014-2014_v1.3_r1.csv", df.to_csv(index=False))
        z.writestr(f"AMF_{site}_FLUXNET_BIF_2014-2014_v1.3_r1.csv", bif.to_csv(index=False))
    return df


def test_only_measured_values_survive_stage_one(tmp_path):
    src = oneflux_zip(tmp_path / "a.zip")
    rec = R.read_oneflux(tmp_path / "a.zip", B.start_minute())
    assert rec.site_id == "XX-Tst" and rec.step == 30 and rec.utc_offset_h == -8.0
    assert rec.meta["bif"]["ustar_vut"] == {2014: 0.2}
    assert "PA" in rec.series and not np.isfinite(rec.series["PA"].value).any()
    rec.meta.update(latlon=(38.0, -121.6), offset_h=-8.0)
    cleaned, counts, tests = B.apply_rules(rec, 38.0, -121.6, -8.0, "GRA", {}, set())
    v, flag = cleaned["NEE"]
    filled = src.NEE_VUT_REF_QC.values == 2
    assert np.isnan(v[filled]).all() and (flag[filled] & 1).all(), "an MDS fill never survives"
    assert np.isnan(v[101]) and flag[101] & 2, "the impossible value is removed by the range rule"
    c = next(c for c in counts if c["variable"] == "NEE")
    assert c["filled"] == filled.sum() and c["range"] == 1
    assert tests["clock_offset_min"]["all"] in (0.0, 30.0, -30.0)
    assert not tests["sign"]["NEE"]["flipped"]


def test_base_column_choice():
    cols = ["FCH4_F", "FCH4_1_2_1", "FCH4_1_1_1", "FCH4_PI_F_1_1_1", "TA_PI_1", "LE", "LE_SSITC_TEST"]
    assert R.base_column(cols, "FCH4") == "FCH4_1_1_1"
    assert R.base_column(cols, "LE") == "LE"
    assert R.base_column(cols, "TA") == "TA_PI_1"
    assert R.base_column(["FCH4_F"], "FCH4") is None, "a PI gap-fill is never read"


def interim(out, src, site, years, value, step=30):
    t = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31 23:30", freq=f"{step}min")
    t_lst = ((t.values - np.datetime64("1970-01-01T00:00")) // np.timedelta64(1, "m")).astype(np.int64)
    v = np.full(t.size, value, np.float32)
    d = pd.DataFrame({"t_lst": t_lst, "t_utc": t_lst + 480, "FCH4": v, "FCH4_flag": np.zeros(t.size, np.uint8)})
    (out / "interim" / src).mkdir(parents=True, exist_ok=True)
    d.to_parquet(out / "interim" / src / f"{site}.parquet")
    (out / "interim" / src / f"{site}.json").write_text(json.dumps(
        {"site_id": site, "source": src, "step": step, "offset_h": -8.0,
         "counts": [{"variable": "FCH4"}], "tests": {}}))


def test_merge_takes_one_source_per_variable_year(tmp_path):
    (tmp_path / "hh").mkdir()
    interim(tmp_path, "fluxnet_ch4", "XX-Tst", [2016, 2017], 10.0)
    interim(tmp_path, "amf_base", "XX-Tst", [2017, 2018], 20.0)
    r = B.merge_site("XX-Tst", tmp_path)
    df = pd.read_parquet(tmp_path / "hh" / "XX-Tst.parquet")
    y = df.time_lst.dt.year
    assert (df.FCH4[y == 2017] == 10.0).all(), "FLUXNET-CH4 outranks BASE for FCH4"
    assert (df.FCH4[y == 2018] == 20.0).all() and (df.FCH4_src[y == 2018] == S.SOURCE_CODES["amf_base"]).all()
    assert df.time_lst.iloc[0] == pd.Timestamp("2016-01-01 00:00") and str(df.time_utc.dt.tz) == "UTC"
    ov = r["overlaps"][0]
    assert (ov["a"], ov["b"], ov["n"]) == ("amf_base", "fluxnet_ch4", 17520)
    assert ov["mean_b_minus_a"] == pytest.approx(-10.0)
