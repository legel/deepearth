"""Site page of every tower, from the networks' own site catalogues, for the Towers tab (static_data/site_links.json).

Order: the AmeriFlux site catalogue (api/v1/site_display/AmeriFlux; AmeriFlux site pages), then the ICOS Carbon
Portal ecosystem stations (SPARQL, class cpmeta:ES; ICOS station pages), then the FLUXNET site pages
(fluxnet.org/sites/siteinfo/<id>, kept only when the page has a site title). A site in none of them gets no link.

Usage: python site_links.py <sites.csv> [--out static_data/site_links.json]
"""

import argparse
import csv
import io
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
AMF_API = "https://amfcdn.lbl.gov/api/v1/site_display/AmeriFlux"
AMF_PAGE = "https://ameriflux.lbl.gov/sites/siteinfo/{}"
ICOS_SPARQL = "https://meta.icos-cp.eu/sparql"
ICOS_QUERY = ("prefix cpmeta: <http://meta.icos-cp.eu/ontologies/cpmeta/> "
              "select ?s ?id where { ?s a cpmeta:ES ; cpmeta:hasStationId ?id }")
FLUXNET_PAGE = "https://fluxnet.org/sites/siteinfo/{}"


def get(url: str, data: bytes = None, headers: dict = None) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "fluxfill-dashboard", **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fluxnet_has(site: str) -> bool:
    return b"page-title entry-title" in get(FLUXNET_PAGE.format(site))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sites", help="CSV with a site_id column")
    ap.add_argument("--out", default=str(HERE / "static_data" / "site_links.json"))
    a = ap.parse_args()
    sites = [r["site_id"] for r in csv.DictReader(open(a.sites))]
    amf = {s["SITE_ID"] for s in json.loads(get(AMF_API))}
    rows = csv.DictReader(io.StringIO(get(ICOS_SPARQL, urllib.parse.urlencode({"query": ICOS_QUERY}).encode(),
                                          {"Accept": "text/csv"}).decode()))
    icos = {r["id"]: r["s"].replace("http://", "https://") for r in rows}
    links = {}
    for s in sites:
        if s in amf:
            links[s] = {"url": AMF_PAGE.format(s), "source": "AmeriFlux"}
        elif s in icos:
            links[s] = {"url": icos[s], "source": "ICOS"}
    rest = [s for s in sites if s not in links]
    with ThreadPoolExecutor(4) as ex:
        for s, ok in zip(rest, ex.map(fluxnet_has, rest)):
            if ok:
                links[s] = {"url": FLUXNET_PAGE.format(s), "source": "FLUXNET"}
    doc = {"generated_utc": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()), "rule": __doc__.split("\n\n")[1], "links": links}
    Path(a.out).write_text(json.dumps(doc, indent=1))
    n = {k: sum(v["source"] == k for v in links.values()) for k in ("AmeriFlux", "ICOS", "FLUXNET")}
    print(f"{len(links)} of {len(sites)} sites linked {n} -> {a.out}")


if __name__ == "__main__":
    main()
