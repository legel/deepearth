"""Build the fine predictor stacks from Copernicus GLO-90 tiles (see ranges/fine.py):
1. training stack: North/Central America ~230 m int16 memmap (elevation + 4 terrain layers);
2. render stack: CONUS 240 m float32 band-sequential memmap with the 24 fine predictors
   (19 WorldClim variables, 7 temperature ones lapse-rate corrected; fine elevation; 4 terrain layers)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402
from ranges import fine  # noqa: E402
from ranges.predictors import VARIABLES  # noqa: E402

R = config.data_root()
DEM = R / "raw/dem_glo90"
OUT = R / "work/fine"
OUT.mkdir(parents=True, exist_ok=True)


def sh(cmd):
    subprocess.run(cmd, shell=True, check=True)


# 1. mosaic and average to the ~230 m North America grid ----------------------------------------------------
sh(f"ls {DEM}/*.tif > {OUT}/tiles.txt && gdalbuildvrt -q -input_file_list {OUT}/tiles.txt {OUT}/glo90.vrt")
na = OUT / "elev_7p5s_na.tif"
if not na.exists():
    sh(f"gdalwarp -q -overwrite -t_srs EPSG:4326 -te {fine.LON0} {fine.LAT1 - fine.NROW * fine.CELL} "
       f"{fine.LON0 + fine.NCOL * fine.CELL} {fine.LAT1} -ts {fine.NCOL} {fine.NROW} -r average -ot Float32 "
       f"-dstnodata -32768 -multi -wo NUM_THREADS=8 -co TILED=YES -co COMPRESS=ZSTD -co BIGTIFF=YES {OUT}/glo90.vrt {na}")
mm = np.memmap(OUT / "fine_7p5s_na.i16", dtype=np.int16, mode="w+", shape=(fine.NROW, fine.NCOL, len(fine.FINE_VARS)))
dy = fine.CELL * 111_320.0
with rasterio.open(na) as src:
    B, PAD = 1024, 8
    for r0 in range(0, fine.NROW, B):
        a, b = max(r0 - PAD, 0), min(r0 + B + PAD, fine.NROW)
        z = src.read(1, window=((a, b), (0, fine.NCOL))).astype(np.float64)
        nod = z == -32768
        z[nod] = 0.0                                   # ocean / no tile: sea level for derivatives, masked below
        lat = fine.LAT1 - (np.arange(a, b) + 0.5) * fine.CELL
        dx = fine.CELL * 111_320.0 * np.cos(np.radians(lat))
        t = fine.terrain(z, dx, dy)
        sl = slice(r0 - a, r0 - a + min(B, fine.NROW - r0))
        layers = {"fine_elev": z, **t}
        for k, n in enumerate(fine.FINE_VARS):
            v = np.round(layers[n][sl] / fine.SCALE[n])
            v[nod[sl]] = -32768
            mm[r0:r0 + v.shape[0], :, k] = np.clip(v, -32767, 32767).astype(np.int16)
        print("NA rows", r0, flush=True)
mm.flush()

# 2. CONUS 240 m render stack -------------------------------------------------------------------------------
C = R / "work/conus240"
meta = json.loads((C / "conus240_stack.json").read_text())
_, H, W = meta["shape"]
ce = OUT / "elev_conus240.tif"
if not ce.exists():
    sh(f"gdalwarp -q -overwrite -t_srs EPSG:5070 -tr 240 240 -te -2493045 177285 2342715 3310005 -r average "
       f"-ot Float32 -dstnodata -32768 -multi -wo NUM_THREADS=8 -co TILED=YES -co COMPRESS=ZSTD -co BIGTIFF=YES "
       f"{OUT}/glo90.vrt {ce}")
wc = np.memmap(C / "conus240_stack.f32", dtype=np.float32, mode="r", shape=tuple(meta["shape"]))
names = [n.replace("wc2.1_30s_", "fine_") if n in fine.TEMP_VARS else n for n in VARIABLES]
names = [("fine_elev" if n == "wc2.1_30s_elev" else n) for n in names] + fine.FINE_VARS[1:]
out = np.memmap(OUT / "conus240_fine.f32", dtype=np.float32, mode="w+", shape=(len(names), H, W))
zi = VARIABLES.index("wc2.1_30s_elev")
with rasterio.open(ce) as src:
    B, PAD = 1024, 8
    for r0 in range(0, H, B):
        a, b = max(r0 - PAD, 0), min(r0 + B + PAD, H)
        z = src.read(1, window=((a, b), (0, W))).astype(np.float64)
        nod = z == -32768
        z[nod] = np.nan
        t = fine.terrain(np.nan_to_num(z), 240.0, 240.0)
        sl = slice(r0 - a, r0 - a + min(B, H - r0))
        rows = slice(r0, r0 + (sl.stop - sl.start))
        zz = z[sl]
        dz = zz - wc[zi, rows].astype(np.float64)
        for k, v in enumerate(VARIABLES):
            x = wc[k, rows].astype(np.float64)
            if v in fine.TEMP_VARS:
                x = x + fine.LAPSE * dz
            elif v == "wc2.1_30s_elev":
                x = zz
            out[k, rows] = x.astype(np.float32)
        for j, n in enumerate(fine.FINE_VARS[1:]):
            x = t[n][sl].astype(np.float32)
            x[nod[sl]] = np.nan
            out[len(VARIABLES) + j, rows] = x
        print("CONUS rows", r0, flush=True)
out.flush()
(OUT / "conus240_fine.json").write_text(json.dumps({"variables": names, "shape": [len(names), H, W],
                                                    "transform": meta["transform"], "crs": meta["crs"]}))
print("done")
