"""NEON soil observations for the soil-product benchmark (independent of the NCSS/KSSL pedons the gridded products are
trained on): DP1.10086.001 "Soil physical and chemical properties, periodic". Per core: plot coordinates
(sls_soilCoreCollection), soil pH in water and CaCl2 (sls_soilpH), organic C (sls_soilChemistry), mineral/organic
horizon and sample depths. Output: $DEEPEARTH_SOIL_DATA/neon/{collection,ph,chem}.parquet."""
import json, os, pathlib, time, io, re, urllib.request
import pandas as pd

from .config import DATA

OUT = DATA / "neon"
API = "https://data.neonscience.org/api/v0"
TOK = pathlib.Path.home() / ".config/neon/token"
WANT = {"collection": "sls_soilCoreCollection", "ph": "sls_soilpH", "chem": "sls_soilChemistry"}


def get(url):
    req = urllib.request.Request(url, headers={"X-API-Token": TOK.read_text().strip()} if TOK.exists() else {})
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # transient API errors: back off
            time.sleep(2 + 4 * k)
    raise RuntimeError(url)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prod = json.loads(get(f"{API}/products/DP1.10086.001"))["data"]
    frames = {k: [] for k in WANT}
    done = OUT / "done.txt"
    seen = set(done.read_text().split()) if done.exists() else set()
    for s in prod["siteCodes"]:
        for url in s["availableDataUrls"]:
            key = url.rsplit("/", 2)[-2] + "_" + url.rsplit("/", 1)[-1]
            if key in seen:
                continue
            files = json.loads(get(url + "?package=basic"))["data"]["files"]
            for k, tab in WANT.items():
                for f in files:
                    if re.search(rf"\.{tab}\.", f["name"]) and f["name"].endswith(".csv"):
                        df = pd.read_csv(io.BytesIO(get(f["url"])), low_memory=False)
                        df.to_parquet(OUT / f"{k}_{key}.parquet", index=False)
            with open(done, "a") as fh:
                fh.write(key + "\n")
            time.sleep(0.2)
    for k in WANT:
        parts = sorted(OUT.glob(f"{k}_*.parquet"))
        if parts:
            df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
            df.astype({c: str for c in df.columns if df[c].dtype == object}).to_parquet(OUT / f"{k}.parquet", index=False)
            print(k, len(df))


if __name__ == "__main__":
    main()
