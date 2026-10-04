"""The canopy's drag with height from the survey, the same canopy the solar model passes light through.

The first returns give each 2 m column its nadir optical depth and the share of it above each height (`F`, the survey's
wood and evergreens), and the live crown's share above each height (`F_leaf`, where the season's leaves go). The solar
model adds the leaves a leaf-off survey misses from MODIS LAI by day of year, with the fall lag; the optical depth above a
height is then

    tau(doy, h) = max(tau0 F(h) + G omega (dL(doy) - dL_survey) tau0 / mean(tau0) F_leaf(h), 0),   dL = max(LAI - floor, 0)

and the plant area above h is tau(doy, h) / (G omega). The drag density at height z is the plant area per volume there,
a(z) = -d PAI(>z) / dz, between the profile's heights, times the foliage drag coefficient: c_d a(z). The files are the sun
set's (`canopy_tau.json` and its profiles); a column without enough first returns keeps its class's uniform drag.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

CD_FOLIAGE = 0.2
"""Drag coefficient of foliage and branches (0.15 to 0.25 in forest canopies; Katul et al. 2004 use 0.2)."""


@dataclass
class Profile:
    """The survey's canopy on its own 2 m grid, row 0 at y0 (south), scene metres."""

    tau0: np.ndarray          # (ny, nx) nadir optical depth, NaN where too few first returns
    f: np.ndarray             # (nh, ny, nx) share of tau0 above each height
    f_leaf: np.ndarray        # (nh, ny, nx) share of the live crown above each height
    heights: np.ndarray       # (nh,) heights above the column's ground [m]
    x0: float
    y0: float
    cell: float
    leaf: Dict
    echo: Optional[np.ndarray] = None    # (ny, nx) share of raised first returns from pulses of several echoes
    green: Optional[np.ndarray] = None   # (ny, nx) NAIP greenness (G - R) / (G + R), NaN outside the photo
    flat: Optional[np.ndarray] = None    # (ny, nx) surface variation of the top over 3 m: ~0 on a plane


def load(root: Path) -> Profile:
    """The sun set's canopy files in `root`."""
    root = Path(root)
    meta = json.loads((root / "canopy_tau.json").read_text())
    ny, nx, p = meta["ny"], meta["nx"], meta["profile"]
    hs = np.asarray(p["heights_m"], float)
    read = lambda f, n: np.fromfile(root / f, "<f4").reshape(n, ny, nx).astype(np.float64)  # noqa: E731
    return Profile(tau0=read(meta["file"], 1)[0], f=read(p["file"], len(hs)), f_leaf=read(p["leaf_file"], len(hs)),
                   heights=hs, x0=float(meta["x0"]), y0=float(meta["y0"]), cell=float(meta["cell"]),
                   leaf=meta.get("leaf") or {}, **_evidence_files(root, meta, ny, nx))


def plant_area_above(p: Profile, doy: Optional[int]) -> np.ndarray:
    """(nh, ny, nx) plant area index above each profile height on day of year `doy` (None: the survey as flown), as the
    solar model's tau."""
    lf = p.leaf
    g_om = float(lf.get("G", 0.5)) * float(lf.get("omega", 0.8))
    tau = p.tau0[None] * p.f
    if lf.get("lai_doy") and doy is not None:
        floor = float(lf.get("floor", 0.0))
        d_l = max(float(lf["lai_doy"][int(doy) - 1]) - floor, 0.0) - max(float(lf.get("lai_survey", floor)) - floor, 0.0)
        tau = tau + g_om * d_l * p.tau0[None] / np.nanmean(p.tau0) * p.f_leaf
    return np.maximum(tau, 0.0) / g_om


def _evidence_files(root: Path, meta: Dict, ny: int, nx: int) -> Dict:
    """The sun set's echo and greenness grids named in the manifest's `vegetation` block, where it wrote them."""
    v = meta.get("vegetation") or {}
    out = {}
    for key, name in (("echo", v.get("echo_file")), ("green", v.get("green_file")), ("flat", v.get("flat_file"))):
        if name and (Path(root) / name).is_file():
            out[key] = np.fromfile(Path(root) / name, "<f4").reshape(ny, nx).astype(np.float64)
    return out


ECHO_PLANTS = 0.2
"""The echo share at and above which raised returns are plants: at California Memorial Stadium crowns 3 to 15 m tall
read 0.58 (p50; 91 % over 0.2) and roofs 0.07 (4 % over 0.2)."""
ECHO_STRUCTURE = 0.05
PLANAR = 0.01
"""Surface variation of the top over 3 m under which it is a plane (a roof, a deck, a glass front): LiDAR noise of a
few centimetres over a 3 m window gives about 0.002; a crown gives a tenth and more."""
NOT_GREEN = 0.03
"""NAIP greenness under which a planar or single-echo top is a structure (crowns read 0.085, lawn 0.10 and roofs 0.0 at
the stadium, p50)."""
"""An echo share under which raised returns are a structure (with a photo that does not say green)."""
ECHO_SURE = 0.5
"""An echo share this high is plants whatever the photo says (eucalyptus reads grey-green)."""
GREEN_STRUCTURE = -0.02
"""NAIP greenness under which a surface is paint, metal, concrete or tile (roofs p50 0.0, p10 -0.06; lawn 0.10,
crowns 0.085): with an echo share between ECHO_PLANTS and ECHO_SURE it decides for a structure."""
PLANTS, NONE, STRUCTURE = 1, 0, -1


def evidence(p: Profile, grid, origin) -> np.ndarray:
    """(ny, nx) int8 on the solver's columns: PLANTS where the survey's raised returns split into several echoes (and
    the photo, where there is one, does not say paint or concrete), STRUCTURE where the top is a plane or nearly every
    pulse returns one echo, and the photo does not say green; NONE where the evidence proves neither or too
    few raised returns were seen to judge (open ground, or a roof or a crown with no ground return near it)."""
    xs, ys = origin[0] + grid.xc, origin[1] + grid.yc
    ip = np.floor((xs - p.x0) / p.cell).astype(int)
    jp = np.floor((ys - p.y0) / p.cell).astype(int)
    ok = ((ip >= 0) & (ip < p.tau0.shape[1]))[None, :] & ((jp >= 0) & (jp < p.tau0.shape[0]))[:, None]
    J, I = np.clip(jp, 0, p.tau0.shape[0] - 1)[:, None], np.clip(ip, 0, p.tau0.shape[1] - 1)[None, :]
    out = np.zeros(ok.shape, np.int8)
    if p.echo is None:
        return out
    e = np.where(ok, p.echo[J, I], np.nan)
    g = np.where(ok, p.green[J, I], np.nan) if p.green is not None else np.full(ok.shape, np.nan)
    fl = np.where(ok, p.flat[J, I], np.nan) if p.flat is not None else np.full(ok.shape, np.nan)
    # a structure on two counts: its top a plane or nearly every pulse one echo (glass splits pulses, so echoes alone
    # cannot call a glass front), and not green where the photo is
    not_green = np.isnan(g) | (g < NOT_GREEN)
    structure = ((np.isfinite(e) & (e < ECHO_STRUCTURE)) | (np.isfinite(fl) & (fl < PLANAR))) & not_green
    plants = np.isfinite(e) & (e >= ECHO_PLANTS) & ~structure
    vetoed = plants & (e < ECHO_SURE) & np.isfinite(g) & (g < GREEN_STRUCTURE)
    out[plants & ~vetoed] = PLANTS
    out[structure] = STRUCTURE
    return out


VEGETATION_PAI = 0.5
"""Plant area index over a column's ground above which the survey holds a crown there (a crown carries 1 to 6; a deck,
a stand or a roof the classes read as paving carries next to none)."""


def vegetation_mask(p: Profile, grid, origin, pai_min: float = VEGETATION_PAI) -> np.ndarray:
    """(ny, nx) True on the solver's columns where the survey's own returns hold a crown (plant area index over the
    ground above `pai_min`, as flown): the evidence that a raised surface-class column is a tree, not a structure."""
    pai = plant_area_above(p, None)[0]
    xs, ys = origin[0] + grid.xc, origin[1] + grid.yc
    ip = np.floor((xs - p.x0) / p.cell).astype(int)
    jp = np.floor((ys - p.y0) / p.cell).astype(int)
    ok = ((ip >= 0) & (ip < pai.shape[1]))[None, :] & ((jp >= 0) & (jp < pai.shape[0]))[:, None]
    v = pai[np.clip(jp, 0, pai.shape[0] - 1)[:, None], np.clip(ip, 0, pai.shape[1] - 1)[None, :]]
    return ok & np.isfinite(v) & (np.nan_to_num(v) > pai_min)


def density(p: Profile, doy: Optional[int]) -> np.ndarray:
    """(nh, ny, nx) plant area per volume [1/m] in each band between profile heights (the top band as deep as the rest)."""
    pai = plant_area_above(p, doy)
    step = np.diff(p.heights, append=p.heights[-1] + (p.heights[-1] - p.heights[-2]))
    below = np.concatenate([pai[1:], np.zeros_like(pai[:1])])
    return np.clip(pai - below, 0.0, None) / step[:, None, None]


def apply(scene, p: Profile, doy: Optional[int], cd: float = CD_FOLIAGE) -> Dict[str, float]:
    """Replace the scene's canopy drag by c_d a(z) wherever the survey's profile covers a fluid column; returns a receipt."""
    g = scene.grid
    a = density(p, doy)
    xs, ys = scene.origin[0] + g.xc, scene.origin[1] + g.yc
    ip = np.clip(np.floor((xs - p.x0) / p.cell).astype(int), 0, a.shape[2] - 1)
    jp = np.clip(np.floor((ys - p.y0) / p.cell).astype(int), 0, a.shape[1] - 1)
    inside = (((xs - p.x0) >= 0) & ((xs - p.x0) < a.shape[2] * p.cell))[None, :] & \
             (((ys - p.y0) >= 0) & ((ys - p.y0) < a.shape[1] * p.cell))[:, None]
    cov = inside & np.isfinite(p.tau0[jp[:, None], ip[None, :]])                    # (ny, nx) columns the survey covers
    col = a[:, jp[:, None], ip[None, :]]                                           # (nh, ny, nx)
    terrain = scene.terrain if scene.terrain is not None else np.zeros((g.ny, g.nx))
    h = g.zc[:, None, None] - terrain[None]
    top = p.heights[-1] + (p.heights[-1] - p.heights[-2])
    band = np.clip(np.searchsorted(p.heights, h, side="right") - 1, 0, len(p.heights) - 1)
    val = np.take_along_axis(col, band, axis=0)                                    # (nz, ny, nx): each cell's band
    val = np.where((h >= 0) & (h < top), np.nan_to_num(val), 0.0)
    new = np.where(cov[None] & ~scene.solid, cd * val, scene.sink)
    plants = getattr(scene, "plants", None)
    if plants is not None:           # the survey's returns over turf, paving, cars and poles are not a crown's drag
        new = np.where((cov & ~plants)[None], 0.0, new)
    before = float(scene.sink[~scene.solid & cov[None]].sum())
    scene.sink[:] = new
    return {"columns_from_survey": int(cov.sum()), "columns": int(cov.size), "doy": doy, "cd": cd,
            "plant_area_index_median": round(float(np.nanmedian(plant_area_above(p, doy)[0][np.isfinite(p.tau0)])), 3),
            "drag_sum_ratio_new_over_uniform": round(float(new[~scene.solid & cov[None]].sum()) / max(before, 1e-12), 3)}
