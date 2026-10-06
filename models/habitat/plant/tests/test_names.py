"""WCVP resolution of GBIF names (Daru step 2a): synonyms map to their accepted name, misapplied names are never
followed, and unknown names are dropped."""
import pandas as pd

from ranges import names


def _wcvp():
    rows = [  # plant_name_id, taxon_status, taxon_name, accepted_plant_name_id
        ("1", "Accepted", "Berberis aquifolium", None),
        ("2", "Synonym", "Mahonia aquifolium", "1"),
        ("3", "Accepted", "Acer macrophyllum", None),
        ("4", "Accepted", "Acer palmatum", None),
        ("5", "Misapplied", "Acer palmatum", "3"),          # a misapplication must not move A. palmatum
        ("6", "Accepted", "Vaccinium corymbosum", None),
        ("7", "Synonym", "Vaccinium formosum", "6"),
    ]
    n = pd.DataFrame(rows, columns=["plant_name_id", "taxon_status", "taxon_name", "accepted_plant_name_id"])
    n["taxon_rank"], n["family"] = "Species", "F"
    n = n[n.taxon_status.isin(names.STATUS_RANK)].copy()
    n["status_rank"] = n.taxon_status.map(names.STATUS_RANK)
    n["accepted_id"] = n.accepted_plant_name_id.where(n.taxon_status != "Accepted", n.plant_name_id)
    return n.dropna(subset=["accepted_id"])


def test_gbif_names_grouped_by_wcvp_accepted_name():
    got = names.gbif_names_by_accepted(["Mahonia aquifolium", "Berberis aquifolium", "Acer palmatum",
                                        "Vaccinium formosum", "Vaccinium corymbosum", "Nonexistent plant", None], _wcvp())
    assert sorted(got["Berberis aquifolium"]) == ["Berberis aquifolium", "Mahonia aquifolium"]
    assert got["Acer palmatum"] == ["Acer palmatum"]
    assert "Acer macrophyllum" not in got
    assert sorted(got["Vaccinium corymbosum"]) == ["Vaccinium corymbosum", "Vaccinium formosum"]
    assert not any("Nonexistent plant" in v for v in got.values())
