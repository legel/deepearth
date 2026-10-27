"""Download the methane sources and log each file's provenance. The Shuttle archives are fetched one at a time by
`build.py shuttle`, which processes and deletes each before the next.

    python fetch.py ch4   --raw /home/photon/4tb/fluxfill/raw/towers   # FLUXNET-CH4 V1.0, every CC-BY-4.0 site
    python fetch.py base  --raw ...                                     # AmeriFlux BASE-BADM, sites that measure FCH4
    python fetch.py icos  --raw ...                                     # ICOS ETC L2 Fluxes, every station

Every file is logged to <raw>/provenance.jsonl with its URL, version, download time and SHA-256.
"""

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List

import sources as S

START_YEAR = 2014


def base_fch4_sites(availability_csv: Path, start_year: int = START_YEAR) -> List[str]:
    """AmeriFlux sites whose BASE files hold a measured (not gap-filled) FCH4 in any year from `start_year`, from the
    amerifluxr BASE variable-availability summary (fraction of half-hours per year)."""
    sites = set()
    with open(availability_csv) as f:
        next(f)  # create_date line
        for row in csv.DictReader(f):
            if row["VARIABLE"] != "FCH4" or row["GAP_FILLED"].upper() == "TRUE":
                continue
            if any(float(row.get(f"Y{y}") or 0) > 0 for y in range(start_year, 2027) if f"Y{y}" in row):
                sites.add(row["SITE_ID"])
    return sorted(sites)


def intact(path: Path) -> bool:
    """True when the file is a readable zip archive with valid member checksums."""
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except zipfile.BadZipFile:
        return False


def version_of(name: str) -> str:
    m = re.search(r"_(\d+-\d+)\.zip$", name) or re.search(r"_(v[\d.]+_r\d+)\.zip$", name)
    return m.group(1) if m else ""


def fetch_amf(urls: Dict[str, str], out: Path, source: str, log: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    done = {r["file_name"] for r in S.read_provenance(log) if r["source"] == source}
    for site, url in sorted(urls.items()):
        name = url.split("/")[-1].split("?")[0]
        dest = out / name
        if name in done and dest.exists():
            continue
        if S.free_gb(out) < 30:
            raise SystemExit(f"under 30 GB free on {out}; stopping before {name}")
        t = S.utc_now()
        try:
            S.curl(url, dest)
        except Exception as e:  # logged and retried on the next run, which skips the files already logged
            print(site, name, "failed:", e, flush=True)
            dest.unlink(missing_ok=True)
            continue
        S.append_provenance(log, [S.record(dest, source, site, url.split("?")[0], version_of(name), True, t)])
        print(site, name, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["ch4", "base", "icos"])
    ap.add_argument("--raw", type=Path, required=True)
    a = ap.parse_args()
    log = a.raw / "provenance.jsonl"
    cat = a.raw / "catalog"
    if a.what == "ch4":
        av = S.amf_availability("FLUXNET", "FLUXNET-CH4")
        (cat / "fluxnet_ch4_availability.json").write_text(json.dumps(av))
        ids = [x["SITE_ID"] for x in av if x["publish_years"]]
        fetch_amf(S.amf_download_urls(ids, "FLUXNET-CH4"), a.raw / "fluxnet_ch4", S.FLUXNET_CH4, log)
    elif a.what == "base":
        ids = base_fch4_sites(cat / "AMF_BASE_VARIABLE-AVAILABILITY_LATEST.csv")
        av = {x["SITE_ID"] for x in S.amf_availability("AmeriFlux", "BASE-BADM") if x["publish_years"]}
        ids = [i for i in ids if i in av]
        urls = S.amf_download_urls(ids, "BASE-BADM", "FULLSET")
        (cat / "amf_base_urls.json").write_text(json.dumps(urls, indent=0))
        fetch_amf(urls, a.raw / "amf_base", S.AMF_BASE, log)
    else:
        out = a.raw / "icos_l2"
        jar = out / "cookies.txt"
        done = {r["file_name"] for r in S.read_provenance(log) if r["source"] == S.ICOS_L2}
        with open(cat / "icos_etcL2Fluxes.csv") as f:
            for row in csv.DictReader(f):
                oid = row["dobj"].rsplit("/", 1)[-1]
                dest = out / row["fileName"]
                if row["fileName"] in done:
                    continue
                t = S.utc_now()
                if dest.exists() and not intact(dest):
                    dest.unlink()  # a partial file from an interrupted download
                if not dest.exists():
                    try:
                        S.curl(S.ICOS_LICENCE.format(id=oid), dest, cookies=jar)
                    except Exception as e:
                        print(row["stationId"], "failed:", e, flush=True)
                        dest.unlink(missing_ok=True)
                        continue
                S.append_provenance(log, [S.record(dest, S.ICOS_L2, row["stationId"], row["dobj"], "L2", True, t)])
                print(row["stationId"], row["fileName"], flush=True)


if __name__ == "__main__":
    main()
