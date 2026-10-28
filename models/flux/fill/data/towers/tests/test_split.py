"""The split keeps every test site at least 50 km from every training site, and balances strata."""

import numpy as np
import pandas as pd

import split as SP


def fake_sites(n=400, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(25, 60, n)
    lon = rng.uniform(-120, 30, n)
    lat[:30], lon[:30] = 50 + rng.normal(0, 0.1, 30), 10 + rng.normal(0, 0.1, 30)  # a dense cluster
    return pd.DataFrame({"site_id": [f"S{i:03d}" for i in range(n)], "lat": lat, "lon": lon, "included": True,
                         "igbp": rng.choice(["ENF", "DBF", "GRA", "CRO", "WET", "OSH"], n),
                         "koppen_main": rng.choice(["B", "C", "D"], n), "has_fch4": rng.random(n) < 0.15})


def test_no_test_site_within_the_block_distance_of_a_training_site():
    sp = SP.make_split(fake_sites())
    assert SP.check(sp) >= SP.BLOCK_KM
    cluster = sp[sp.site_id.isin([f"S{i:03d}" for i in range(30)])]
    assert cluster.split.isin(["test"]).all() or not cluster.split.isin(["test"]).any()


def test_fractions_and_strata_are_balanced():
    sp = SP.make_split(fake_sites())
    frac = (sp.split == "test").mean()
    assert 0.17 <= frac <= 0.21, frac
    val = (sp.split == "val").sum() / (sp.split != "test").sum()
    assert 0.07 <= val <= 0.11, val
    ch4 = sp[sp.has_fch4]
    assert 0.1 <= (ch4.split == "test").mean() <= 0.3


def test_the_split_is_reproducible():
    a, b = SP.make_split(fake_sites()), SP.make_split(fake_sites())
    assert a.equals(b)
