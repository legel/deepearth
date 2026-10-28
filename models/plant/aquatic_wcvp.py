"""The obligately aquatic species (deepearth_site/assets/aquatic_wcvp.json) from WCVP's life forms.

WCVP (Kew, World Checklist of Vascular Plants) records each accepted species' Raunkiaer life form in
lifeform_description. Its "hydro" forms (hydroannual, hydroperennial, hydrosubshrub, hydroshrub, hydrogeophyte,
epiphytic hydrophyte) are hydrophytes: plants whose perennating parts and leaves live in or on water (free-floating,
submerged, rooted floating-leaved). A helophyte is an emergent marsh plant (Typha) and is not one. A species is
aquatic when every alternative of its life form is a hydro form: "hydroannual or hydrosubshrub" is, "helophyte or
hydroperennial" (it may stand emergent) is not. The trait posteriors cannot tell them apart (Lemna, Typha and Cicuta are
all OBL herbs there), which is why the life form is read here.

WCVP codes some aquatic genera wholesale as helophytes: 67 of 72 Myriophyllum, so the submerged Myriophyllum
hippuroides leaked the gate (a coastal site, 2026-10-28). GIFT (Weigelt et al. 2020) Life_form_2 codes the same genus from
regional floras, 9 of its 10 species hydrophyte. A species WCVP leaves blank or calls a helophyte, and GIFT does not
code itself, takes its genus's form where GIFT codes at least GENUS_SHARE of at least GENUS_MIN species hydrophyte.
GIFT's own species values are not used: they disagree with WCVP on true aquatics (Potamogeton natans chamaephyte) and
on terrestrial ones (Hydrocotyle bonariensis, a dune pennywort, hydrophyte), and its genus vote keeps Hydrocotyle out.

Two checks against the model's own trait data take a WCVP coding out where it disagrees: the national catalog's
sourced habit (Anemopsis californica: WCVP hydroperennial, catalog perennial, an emergent), and a climber that is not
obligately wetland in the trait posteriors (Smilax: WCVP tuberous hydrogeophyte, greenbrier vines). Every exclusion
is listed in the output with its reason.

    python tools/aquatic_wcvp.py wcvp_names.csv out.json --posteriors posteriors.npz --catalog plant_catalog.json \
        --gift-species species.json.gz --gift-lifeform trait_2.3.2.json.gz --source-sha256 <zip sha256> --source-url <url>
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from collections import defaultdict

CLIMBING = 0.2      # a hydrophyte does not climb: true ones read P(vine) under 0.05, the Smilax 0.22 to 0.85
OBL_MIN = 0.5       # and a climber that is not even obligately wetland is no hydrophyte
GENUS_SHARE = 0.8   # GIFT genus vote: Myriophyllum 9 of 10 hydrophyte, Hydrocotyle 6 of 17, Isoetes 7 of 22
GENUS_MIN = 5       # species GIFT codes in the genus before its vote counts
GENUS_OVERRIDES = ("", "helophyte")     # the WCVP codings a genus vote may replace
CATALOG_SILENT = ("aquatic", "fern")    # "fern" names a lineage, not a habit: water clovers are ferns


def gift_lifeforms(species_gz, lifeform_gz) -> dict:
    """{species: GIFT Life_form_2 value} (trait 2.3.2; '/' joins alternatives)."""
    sp = {r["work_ID"]: r["work_species"] for r in json.load(gzip.open(species_gz))}
    return {sp[r["work_ID"]]: (r.get("trait_value") or "").strip() for r in json.load(gzip.open(lifeform_gz))
            if r["work_ID"] in sp}


def gift_hydrophyte(v: str) -> bool:
    parts = [p.strip() for p in (v or "").split("/") if p.strip()]
    return bool(parts) and all(p == "hydrophyte" for p in parts)


def genus_votes(gift: dict) -> dict:
    """{genus: (species GIFT codes, of them hydrophytes)}."""
    g = defaultdict(lambda: [0, 0])
    for n, v in gift.items():
        if v:
            g[n.split()[0]][0] += 1
            g[n.split()[0]][1] += gift_hydrophyte(v)
    return {k: tuple(v) for k, v in g.items()}


def by_genus(n: str, form: str, gift: dict, votes: dict) -> bool:
    """WCVP blank or helophyte, GIFT silent on the species, its genus a GIFT hydrophyte genus."""
    c, h = votes.get(n.split()[0], (0, 0))
    return form.strip() in GENUS_OVERRIDES and n not in gift and c >= GENUS_MIN and h / c >= GENUS_SHARE


def aquatic_form(lf: str) -> bool:
    parts = [p.strip().lower() for p in (lf or "").split(" or ") if p.strip()]
    return bool(parts) and all("hydro" in p for p in parts)


def catalog_habits(path) -> dict:
    """{species: habit} from the national catalog (data/recommender plant_catalog.json)."""
    C = json.load(open(path))
    items = C.get("plants") or C.get("species") or C
    items = list(items.values()) if isinstance(items, dict) else items
    return {e["species"]: e.get("habit") for e in items if e.get("species") and e.get("habit")}


def climbing_obl(path) -> dict:
    """{species: (climbing, P(OBL))} from trait posteriors v2: climbing is the larger of P(growth_form = vine) and
    P(form.vine = yes). Other growth forms are left out: a herb's shrub share is noise there (Myriophyllum spicatum,
    a WCVP hydrosubshrub, reads 0.21 shrub)."""
    import numpy as np
    z = np.load(path, allow_pickle=True)
    cat = [str(c) for c in z["cat_ids"]]
    lv = z["cat_levels"].item() if z["cat_levels"].shape == () else list(z["cat_levels"])
    lv = json.loads(lv) if isinstance(lv, str) else lv
    L = lambda k: [str(x) for x in (lv[k] if isinstance(lv, dict) else lv[cat.index(k)])]   # noqa: E731
    P = z["P"]
    gv = P[:, cat.index("growth_form"), L("growth_form").index("vine")]
    fv = P[:, cat.index("form.vine"), L("form.vine").index("yes")]
    obl = P[:, cat.index("wetland_indicator"), L("wetland_indicator").index("OBL")]
    c = np.maximum(gv, fv)
    return {str(n).replace("_", " "): (float(a), float(b)) for n, a, b in zip(z["species"], c, obl)}


def exclusion(n: str, form: str, hab: dict, clo: dict):
    """Why a WCVP hydrophyte is not taken as one, or None."""
    if n in hab and hab[n] not in CATALOG_SILENT:
        return f"catalog habit {hab[n]!r} (sourced) against WCVP {form!r}"
    c, o = clo.get(n, (0.0, 1.0))
    if c >= CLIMBING and o < OBL_MIN:
        return f"posterior climbing {c:.2f} and P(OBL) {o:.2f} against WCVP {form!r}"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names"); ap.add_argument("out")
    ap.add_argument("--posteriors", default=None); ap.add_argument("--catalog", default=None)
    ap.add_argument("--gift-species", default=None); ap.add_argument("--gift-lifeform", default=None)
    ap.add_argument("--source-sha256", default=None); ap.add_argument("--source-url", default=None)
    a = ap.parse_args(argv)
    accepted, forms, syn = {}, {}, []
    with open(a.names, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|"):
            if row.get("taxon_rank") != "Species":
                continue
            if row.get("taxon_status") == "Accepted":
                accepted[row["plant_name_id"]] = row["taxon_name"]
                forms[row["taxon_name"]] = row.get("lifeform_description") or ""
            elif row.get("accepted_plant_name_id"):
                syn.append((row["taxon_name"], row["accepted_plant_name_id"]))
    gift = gift_lifeforms(a.gift_species, a.gift_lifeform) if a.gift_species and a.gift_lifeform else {}
    votes = genus_votes(gift)
    genus = sorted(n for n, lf in forms.items() if not aquatic_form(lf) and by_genus(n, lf, gift, votes))
    aq0 = sorted({n for n, lf in forms.items() if aquatic_form(lf)} | set(genus))
    hab = catalog_habits(a.catalog) if a.catalog else {}
    clo = climbing_obl(a.posteriors) if a.posteriors else {}
    excluded = {n: why for n in aq0 if (why := exclusion(n, forms[n], hab, clo))}
    aq = [n for n in aq0 if n not in excluded]
    aqs = set(aq)
    # a synonym carries its accepted species' form, unless the name is itself an accepted species
    synonyms = sorted({s for s, acc in syn if accepted.get(acc) in aqs and s not in forms})
    doc = {"rule": "WCVP lifeform_description: every alternative a hydro form (Raunkiaer hydrophyte); helophytes and "
                   "mixed forms excluded", "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "source": {"name": "WCVP wcvp_names.csv", "url": a.source_url, "sha256": a.source_sha256},
           "checks": {"catalog": bool(hab), "posteriors": bool(clo), "climbing_at_least": CLIMBING,
                      "and_obl_under": OBL_MIN},
           "genus_vote": {"source": "GIFT Life_form_2 (trait 2.3.2)", "share_at_least": GENUS_SHARE,
                          "coded_at_least": GENUS_MIN, "replaces_wcvp": list(GENUS_OVERRIDES),
                          "genera": sorted({n.split()[0] for n in genus}), "species": genus},
           "forms": sorted({forms[n] for n in aq}), "accepted": aq, "synonyms": synonyms, "excluded": excluded}
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=0)
    print("CLIMBING", json.dumps({n: [round(clo[n][0], 2), round(clo[n][1], 2)] for n in aq0
                                  if n in clo and clo[n][0] >= 0.1}))
    print(json.dumps({"accepted_species": len(forms), "candidates": len(aq0), "by_genus": len(genus), "aquatic": len(aq),
                      "genera_by_vote": doc["genus_vote"]["genera"],
                      "synonyms": len(synonyms), "excluded": excluded, "forms": doc["forms"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
