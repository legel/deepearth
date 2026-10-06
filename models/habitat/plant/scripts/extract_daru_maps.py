#!/usr/bin/env python3
"""Daru (2024)'s published per-species maps for the species of the species table, the benchmark of every comparison:
from his Dryad archive (doi:10.5061/dryad.5x69p8d9w, DRYAD_DATA.zip, 15.2 GB, CC0; download it from Dryad by hand),
SDM_set1/rasters/<species>.tif (binary map) and SDM_set1/raw_rasters/<species>.tif (suitability), 0.1667 degree
(~18 km) cells, EPSG:4326, NaN outside his calibration area.

usage: extract_daru_maps.py DRYAD_DATA.zip [--config configs/conus.json]
Writes daru_ref_all/rasters/ and daru_ref_all/raw_rasters/ under the data root (as configured for the evaluation).
"""
import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = config.load(a.config)
    ev = cfg["evaluation"]
    wanted = set(pd.read_csv(cfg.path(cfg["species"]["table"])).wcvp_accepted_name)
    dirs = {"rasters": cfg.path(ev["daru_rasters"]), "raw_rasters": cfg.path(ev["daru_raw_rasters"])}
    n = 0
    with zipfile.ZipFile(a.archive) as z:
        for info in z.infolist():
            p = Path(info.filename)
            if (len(p.parts) == 4 and p.parts[:2] == ("DRYAD_DATA", "SDM_set1") and p.parts[2] in dirs
                    and p.suffix == ".tif" and p.stem in wanted):
                out = dirs[p.parts[2]] / p.name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(info))
                n += 1
    print(f"{n} rasters of {len(wanted)} species written")


if __name__ == "__main__":
    main()
