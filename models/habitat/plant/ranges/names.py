"""Name reconciliation against the World Checklist of Vascular Plants (Daru 2024 step 2a).

A binomial can appear in WCVP several times with different statuses (accepted, synonym, illegitimate,
misapplied, ...). A misapplied row records that the name was wrongly used for *another* species, so its
accepted id points elsewhere and must never be followed. Resolution order for a name:
Accepted > Synonym > Orthographic > Illegitimate/Invalid > Unplaced/Artificial Hybrid; Misapplied is excluded.
Garden nothospecies are often written without the hybrid sign ("Calamagrostis acutiflora"), so a second
pass inserts "×" after the genus.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

STATUS_RANK = {"Accepted": 0, "Synonym": 1, "Orthographic": 2, "Illegitimate": 3, "Invalid": 3,
               "Unplaced": 4, "Artificial Hybrid": 4}


def load_wcvp_names(wcvp_dir: str | Path) -> pd.DataFrame:
    n = pd.read_csv(Path(wcvp_dir) / "wcvp_names.csv", sep="|", dtype=str,
                    usecols=["plant_name_id", "taxon_rank", "taxon_status", "family", "taxon_name",
                             "accepted_plant_name_id"])
    n = n[n.taxon_status.isin(STATUS_RANK)]
    n["status_rank"] = n.taxon_status.map(STATUS_RANK)
    n["accepted_id"] = n.accepted_plant_name_id.where(n.taxon_status != "Accepted", n.plant_name_id)
    return n.dropna(subset=["accepted_id"])


def clean_query(name: str) -> str:
    name = re.sub(r"['‘’\"].*$", "", name).strip()           # drop cultivar epithets
    name = re.sub(r"\s+[xX]\s+", " × ", name)
    return re.sub(r"\s+", " ", name)


def resolve(queries: pd.Series, names: pd.DataFrame) -> pd.DataFrame:
    """Return, per query, the matched WCVP row (best status) and its accepted name/id/family."""
    best = names.sort_values(["taxon_name", "status_rank"]).drop_duplicates("taxon_name").set_index("taxon_name")
    acc = names[names.taxon_status == "Accepted"].set_index("plant_name_id")
    rows = []
    for q in queries:
        c = clean_query(q)
        cands = [c]
        parts = c.split(" ")
        if len(parts) >= 2 and "×" not in c:
            cands.append(f"{parts[0]} × {' '.join(parts[1:])}")       # nothospecies written without ×
        hit = next((x for x in cands if x in best.index), None)
        if hit is None:
            rows.append({"query": q, "wcvp_status": None})
            continue
        r = best.loc[hit]
        a = acc.loc[r.accepted_id] if r.accepted_id in acc.index else None
        rows.append({"query": q, "wcvp_matched_name": hit, "wcvp_status": r.taxon_status, "wcvp_acc_id": r.accepted_id,
                     "wcvp_accepted_name": None if a is None else a.taxon_name,
                     "wcvp_accepted_rank": None if a is None else a.taxon_rank,
                     "wcvp_family": None if a is None else a.family})
    return pd.DataFrame(rows)


def gbif_names_by_accepted(gbif_names, names: pd.DataFrame) -> dict[str, list[str]]:
    """Daru's step 2a applied to the occurrence files: every distinct GBIF species name is resolved through
    WCVP, and records are grouped under the WCVP accepted name they resolve to. GBIF's backbone and WCVP
    disagree on some genera (GBIF 2026: Mahonia aquifolium, Morella californica; WCVP: Berberis aquifolium,
    Myrica californica), so selecting records by the WCVP name alone loses whole species."""
    q = pd.Series(sorted({n for n in gbif_names if isinstance(n, str) and n}))
    r = resolve(q, names).dropna(subset=["wcvp_accepted_name"])
    out: dict[str, list[str]] = {}
    for g, a in zip(r["query"], r["wcvp_accepted_name"]):
        out.setdefault(a, []).append(g)
    return out


def names_of_accepted(accepted: str, names: pd.DataFrame) -> list[str]:
    """Every WCVP name (accepted, synonyms, orthographic variants; misapplied names excluded by
    ``load_wcvp_names``) that resolves to the accepted name ``accepted``."""
    ids = set(names.loc[(names.taxon_name == accepted) & (names.taxon_status == "Accepted"), "plant_name_id"])
    return sorted(set(names.loc[names.accepted_id.isin(ids), "taxon_name"]) | {accepted})


def binomial(verbatim) -> str | None:
    """Genus + epithet (with the hybrid sign kept) from a verbatim scientific name with authors or ranks."""
    if not isinstance(verbatim, str):
        return None
    w = re.sub(r"\s+[×xX]\s+", " × ", " " + verbatim.strip()).split()
    if len(w) >= 3 and w[1] == "×":
        return " ".join(w[:3])
    return " ".join(w[:2]) if len(w) >= 2 else None


_RANKS = {"var.", "subsp.", "ssp.", "f.", "var", "subsp", "ssp"}


def canonical(verbatim) -> str | None:
    """Name without authors, keeping one infraspecific rank and epithet ("Rhus aromatica var. trilobata"), so an
    identification below species resolves to what WCVP accepts for it rather than to its parent binomial."""
    b = binomial(verbatim)
    if b is None:
        return None
    w = re.sub(r"\s+[×xX]\s+", " × ", " " + verbatim.strip()).split()
    rest = w[len(b.split()):]
    if len(rest) >= 2 and rest[0] in _RANKS and rest[1][:1].islower():
        r = {"ssp.": "subsp.", "ssp": "subsp.", "subsp": "subsp.", "var": "var."}.get(rest[0], rest[0])
        return f"{b} {r} {rest[1]}"
    return b
