"""NEON plant presence and percent cover (DP1.10058.001) on the soil benchmark's plots: the plant community as a soil
indicator. Keeps the 1 m2 subplot cover table (div_1m2Data) for every bout at the benchmark's sites and writes
$DEEPEARTH_SOIL_DATA/neon/plant_cover.parquet (plotID, scientificName, percentCover, endDate)."""
import io, json, pathlib, re, time, urllib.request
import pandas as pd

from .config import DATA

W = DATA
API = "https://data.neonscience.org/api/v0"
TOK = pathlib.Path.home() / ".config/neon/token"


def get(url, raw=False):
    req = urllib.request.Request(url, headers={"X-API-Token": TOK.read_text().strip()} if TOK.exists() else {})
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                b = r.read()
                return b if raw else json.loads(b)
        except Exception:
            time.sleep(3 + 5 * k)
    raise RuntimeError(url)


def main():
    plots = set(pd.read_parquet(W / "benchmark/points_neon.parquet")["id"])
    sites = sorted({p.split("_")[0] for p in plots})
    prod = get(f"{API}/products/DP1.10058.001")["data"]
    urls = {s["siteCode"]: s["availableDataUrls"] for s in prod["siteCodes"]}
    out = []
    for site in sites:
        for url in urls.get(site, []):
            files = get(url + "?package=basic")["data"]["files"]
            for f in files:
                if re.search(r"\.div_1m2Data\.", f["name"]) and f["name"].endswith(".csv"):
                    d = pd.read_csv(io.BytesIO(get(f["url"], raw=True)), low_memory=False)
                    d = d[d["plotID"].isin(plots)]
                    if len(d):
                        out.append(d[["plotID", "subplotID", "endDate", "taxonID", "scientificName", "taxonRank",
                                      "percentCover", "nativeStatusCode"]])
            time.sleep(0.2)
        print(site, sum(len(x) for x in out), flush=True)
    df = pd.concat(out, ignore_index=True)
    df.astype({c: str for c in df.columns if df[c].dtype == object}).to_parquet(W / "neon/plant_cover.parquet", index=False)
    print(len(df), "records;", df["plotID"].nunique(), "plots;", df["taxonID"].nunique(), "taxa")


if __name__ == "__main__":
    main()
