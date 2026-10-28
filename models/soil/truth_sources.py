"""The ground truth SOIL MOISTURE is held to (soil.moisture_truth), fetched for a box and placed on a site's grid.

Two public, unauthenticated services, standard library only:
  USDA Soil Data Access: SSURGO map unit polygons under the box, clipped to it, with muaggatt's dominant drainage class
    (drclassdcd), annual minimum water table depth in cm (wtdepannmin) and hydric share in percent (hydclprs).
  USFWS National Wetlands Inventory: wetland polygons under the box with their Cowardin code (ATTRIBUTE).
Both answer in WGS 84 lon/lat. Placing them on the grid is `sims.site.to_enu`, the operation that places the order's own
polygon in the scene frame the solvers' grids are in (sims.site.from_scene), so the truth registers with the parcel
exactly as the solved domain does. Nothing is fitted (docs/COORDINATES.md).
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import net, ssurgo

NWI_URL = "https://fwspublicservices.wim.usgs.gov/wetlandsmapservice/rest/services/Wetlands/MapServer/0/query"
NWI_PAGE = 500
"""The layer's maxRecordCount (2026-10-27); a box with more wetlands is paged."""
NWI_PRECISION = 6
"""Decimal places of a degree NWI returns, about 0.1 m: Moffett's 37 polygons fell from 241 to 143 KB."""
SOURCES = {
    "ssurgo": "USDA-NRCS SSURGO via Soil Data Access (mupolygon, muaggatt: drclassdcd, wtdepannmin, hydclprs)",
    "nwi": "USFWS National Wetlands Inventory, Wetlands MapServer layer 0 (ATTRIBUTE, WETLAND_TYPE)"}

Bbox = Tuple[float, float, float, float]


# ------------------------------------------------------------------------------------------------ WKT

_TOKEN = re.compile(r"\s*([A-Za-z]+|\(|\)|,|[-+0-9.eE]+)")


def _tokens(text: str) -> List[str]:
    return [m.group(1) for m in _TOKEN.finditer(text)]


def parse_wkt(text: str) -> List[List[List[Tuple[float, float]]]]:
    """Polygons in a WKT geometry as [polygon][ring][(x, y)]: POLYGON, MULTIPOLYGON and the polygon members of a
    GEOMETRYCOLLECTION (an intersection with the box can leave lines and points on its edge, dropped here)."""
    tok, out = _tokens(text or ""), []
    pos = [0]

    def nxt() -> str:
        t = tok[pos[0]]
        pos[0] += 1
        return t

    def nested():
        # a bracketed list of coordinate tuples or of further lists
        assert nxt() == "("
        items, cur = [], []
        while True:
            t = nxt()
            if t == "(":
                pos[0] -= 1
                items.append(nested())
            elif t == ")":
                if cur:
                    items.append(tuple(cur))
                return items
            elif t == ",":
                if cur:
                    items.append(tuple(cur))
                cur = []
            else:
                cur.append(float(t))

    def geometry():
        kind = nxt().upper()
        if pos[0] < len(tok) and tok[pos[0]].upper() in ("Z", "M", "ZM"):
            pos[0] += 1
        if pos[0] < len(tok) and tok[pos[0]].upper() == "EMPTY":
            pos[0] += 1
            return
        if kind == "GEOMETRYCOLLECTION":
            assert nxt() == "("
            while True:
                geometry()
                if nxt() == ")":
                    return
        body = nested()
        if kind == "POLYGON":
            out.append([[p[:2] for p in ring] for ring in body])
        elif kind == "MULTIPOLYGON":
            out.extend([[p[:2] for p in ring] for ring in poly] for poly in body)

    while pos[0] < len(tok):
        geometry()
    return out


# ------------------------------------------------------------------------------------------------ SSURGO

def _box_wkt(bbox: Bbox) -> str:
    w, s, e, n = bbox
    return f"POLYGON(({w:.7f} {s:.7f}, {e:.7f} {s:.7f}, {e:.7f} {n:.7f}, {w:.7f} {n:.7f}, {w:.7f} {s:.7f}))"


def ssurgo_sql(bbox: Bbox) -> str:
    """Every map unit polygon under the box, clipped to it, with its muaggatt drainage fields. SDA refuses DECLARE, so
    the box is written twice."""
    b = f"geometry::STGeomFromText('{_box_wkt(bbox)}', 4326)"
    return (f"SELECT p.mukey, p.mupolygonkey, a.muname, a.drclassdcd, a.wtdepannmin, a.hydclprs, "
            f"p.mupolygongeo.STIntersection({b}).STAsText() AS wkt FROM mupolygon p "
            f"LEFT JOIN muaggatt a ON a.mukey = p.mukey WHERE p.mupolygongeo.STIntersects({b}) = 1")


def parse_sda(rows: Sequence[Dict]) -> List[Dict]:
    """SDA rows (ssurgo.query's dicts) as units: {mukey, mupolygonkey, muname, drclassdcd, wtdepannmin, hydclprs,
    polygons (lon/lat)}; a row whose clip is empty is dropped."""
    out = []
    for r in rows:
        polys = parse_wkt(r.get("wkt") or "")
        if not polys:
            continue
        out.append({"mukey": str(r["mukey"]), "mupolygonkey": str(r.get("mupolygonkey")), "muname": r.get("muname"),
                    "drclassdcd": r.get("drclassdcd"), "wtdepannmin": _num(r.get("wtdepannmin")),
                    "hydclprs": _num(r.get("hydclprs")), "polygons": polys})
    return out


def sda_table(raw: Dict) -> List[Dict]:
    """A saved SDA response body ({"Table": [cols, rows...]}) as the rows `ssurgo.query` returns."""
    t = raw.get("Table") or []
    return [dict(zip(t[0], row)) for row in t[1:]] if t else []


def ssurgo_units(bbox: Bbox, query: Callable[[str], List[Dict]] = ssurgo.query) -> List[Dict]:
    return parse_sda(query(ssurgo_sql(bbox)))


def _num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------ NWI

def nwi_params(bbox: Bbox, offset: int = 0) -> Dict[str, str]:
    return {"geometry": ",".join(f"{v:.7f}" for v in bbox), "geometryType": "esriGeometryEnvelope", "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects", "outFields": "ATTRIBUTE,WETLAND_TYPE", "returnGeometry": "true",
            "outSR": "4326", "geometryPrecision": str(NWI_PRECISION), "orderByFields": "OBJECTID",
            "resultOffset": str(offset), "resultRecordCount": str(NWI_PAGE), "f": "geojson"}


def _get_json(url: str, params: Dict[str, str]) -> Dict:
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), headers={"User-Agent": "ecological.dev"})
    doc = json.loads(net.urlopen(req, "USFWS National Wetlands Inventory").decode())
    if "error" in doc:                                   # ArcGIS reports a refused query with a 200
        raise RuntimeError(f"NWI refused the query: {json.dumps(doc['error'])[:300]}")
    return doc


def parse_nwi(doc: Dict) -> List[Dict]:
    """GeoJSON features as wetlands: {attribute, wetland_type, polygons (lon/lat)}."""
    out = []
    for f in doc.get("features") or []:
        g, p = f.get("geometry") or {}, f.get("properties") or {}
        polys = [g["coordinates"]] if g.get("type") == "Polygon" else g.get("coordinates") if g.get("type") == "MultiPolygon" else []
        if polys:
            out.append({"attribute": p.get("ATTRIBUTE"), "wetland_type": p.get("WETLAND_TYPE"),
                        "polygons": [[[tuple(pt[:2]) for pt in ring] for ring in poly] for poly in polys]})
    return out


def _more(doc: Dict) -> bool:
    return bool(doc.get("exceededTransferLimit") or (doc.get("properties") or {}).get("exceededTransferLimit"))


def nwi_wetlands(bbox: Bbox, get: Callable[[str, Dict], Dict] = _get_json, max_pages: int = 200) -> List[Dict]:
    out, offset = [], 0
    for _ in range(max_pages):
        doc = get(NWI_URL, nwi_params(bbox, offset))
        page = parse_nwi(doc)
        out += page
        n = len(doc.get("features") or [])
        if not _more(doc) or n == 0:
            return out
        offset += n
    raise RuntimeError(f"NWI: more than {max_pages} pages of {NWI_PAGE} under {bbox}")


# ------------------------------------------------------------------------------------------------ the site's grid

def frame_of(doc: Dict) -> Dict:
    """The scene@3 frame block from a scene manifest, a saved SiteSpec (sims.site.save) or the block itself."""
    f = doc.get("frame") if isinstance(doc.get("frame"), dict) else doc
    for k in ("anchor_ecef", "basis_east", "basis_north", "basis_up", "anchor_geodetic"):
        assert k in f, f"the frame has no {k}"
    return f


def to_local(frame: Dict, polygons: Sequence, h: Optional[float] = None) -> List[List[List[Tuple[float, float]]]]:
    """Lon/lat polygons in the frame's ENU metres, by `sims.site.to_enu` at the anchor's ellipsoidal height, as
    sims.site.from_scene places the order's polygon."""
    from sims.site import to_enu
    h = float(frame["anchor_geodetic"][2]) if h is None else h
    return [[[to_enu(frame, x, y, h)[:2] for x, y in ring] for ring in poly] for poly in polygons]


def grid_bbox(frame: Dict, shape: Sequence[int], transform: Dict, margin_m: float = 50.0) -> Bbox:
    """The lon/lat box over the grid's corners and a margin, each corner taken back through the frame's own basis
    (sims.plan.ecef_to_geodetic); it bounds the queries only, the placement is `to_local`."""
    from sims.plan import ecef_to_geodetic
    ny, nx = shape
    t = transform
    xs = (t["x0"] - margin_m, t["x0"] + nx * t["dx"] + margin_m)
    ys = sorted((t["y0"], t["y0"] + ny * t["dy"]))
    ys = (ys[0] - margin_m, ys[1] + margin_m)
    o, e, n = frame["anchor_ecef"], frame["basis_east"], frame["basis_north"]
    ll = [ecef_to_geodetic(*(o[i] + x * e[i] + y * n[i] for i in range(3)))[:2] for x in xs for y in ys]
    return (min(p[0] for p in ll), min(p[1] for p in ll), max(p[0] for p in ll), max(p[1] for p in ll))


def fetch(bbox: Bbox, query: Callable = ssurgo.query, get: Callable = _get_json) -> Dict:
    """Both sources for one box, each with the UTC time it was read."""
    stamp = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())  # noqa: E731
    units = ssurgo_units(bbox, query)
    t_ssurgo = stamp()
    wetlands = nwi_wetlands(bbox, get)
    return {"bbox": list(bbox), "units": units, "wetlands": wetlands,
            "fetched": {"ssurgo": t_ssurgo, "nwi": stamp()}, "sources": dict(SOURCES)}


def place(sources: Dict, frame: Dict) -> Dict:
    """`fetch`'s units and wetlands with their polygons in the frame's metres."""
    return dict(sources, units=[dict(u, polygons=to_local(frame, u["polygons"])) for u in sources["units"]],
                wetlands=[dict(w, polygons=to_local(frame, w["polygons"])) for w in sources["wetlands"]])
