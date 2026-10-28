"""SSURGO: the USDA's hand-surveyed soil map, read through Soil Data Access.

Standard library only, in keeping with the rest of this repo. SDA is public
domain and unauthenticated: no key, no account, no rate limit published.

WHAT THIS IS FOR. A site boundary goes in, the soil map units under it come
out, each with the share of the site it covers and a distilled profile. The
distillation matters: SSURGO is sixty-odd tables and about two hundred
interpretations per map unit, and a planting tool needs a few dozen fields.

THE CACHE IS THE WHOLE STRATEGY. There are 339,191 map units in the entire
country and they are stable between NRCS's annual refreshes, so a `mukey` we
have resolved once never needs resolving again. Measured at 789 bytes per
record gzipped, a complete national cache is 0.27 GB. Do not build an ETL over
the SSURGO schema; let traffic fill a cache instead.

TRAP, MEASURED 2026-08-31. Every tutorial reaches for
`SDA_Get_MupolygonWktWgs84_from_intersection_with_WktWgs84` to fetch polygon
geometry. It no longer exists on the service and returns `Invalid object name`
on both hostnames. Query the `mupolygon` table directly, as below.

TRAP. SDA answers errors with an OGC XML body and HTTP 200. Sniffing the
payload is the only reliable way to notice.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

from . import net

SDA_URL = 'https://sdmdataaccess.nrcs.usda.gov/tabular/post.rest'

# Bump when fetch_mapunit gains a field; older cached records are refetched.
SCHEMA = 3

# SSURGO ships ~200 interpretations per map unit, most of them about military
# vehicle trafficability and radioactive sequestration. These are the ones a
# landscape architect would act on.
# What a map-unit record keeps, table by table. `fetch_mapunit` and the offline
# index builder (`soil/build/build_records.py`) both read these, so a record
# from either source has the same fields.
MUAGG_COLS = ('muname', 'mukey', 'slopegradwta', 'brockdepmin', 'wtdepannmin',
              'flodfreqdcd', 'pondfreqprs', 'aws025wta', 'aws050wta', 'aws0100wta',
              'aws0150wta', 'drclassdcd', 'drclasswettest', 'hydgrpdcd', 'hydclprs')
COMP_COLS = ('cokey', 'compname', 'comppct_r', 'compkind', 'drainagecl', 'taxclname',
             'hydgrp', 'slope_r', 'runoff', 'nirrcapcl')
# Chemistry filled on 75-97% of surface horizons nationally (2026-09-25).
# P columns exist but are 0.02-0.2% filled, and N and K have no column.
HZ_COLS = ('cokey', 'hzname', 'hzdept_r', 'hzdepb_r', 'sandtotal_r', 'silttotal_r',
           'claytotal_r', 'om_r', 'ph1to1h2o_r', 'awc_r', 'ksat_r', 'dbthirdbar_r',
           'cec7_r', 'ecec_r', 'sumbases_r', 'ec_r', 'sar_r', 'caco3_r', 'gypsum_r')
INTERP_COLS = ('cokey', 'mrulename', 'interphr', 'interphrc')
# Ties in comppct_r are common (11 of 100 sampled units); SQL Server and SQLite
# break them differently, so both sources order by cokey after it.
COMP_ORDER = 'comppct_r DESC, CAST(cokey AS INT)'

INTERPRETATIONS = (
    'ENG - Lawn, Landscape, Golf Fairway',
    'URB/REC - Playgrounds',
    'ENG - Shallow Excavations',
    'FOR - Soil Compactibility Risk',
    'SOH - Soil Surface Sealing',
    'WMS - Irrigation, General',
)

_TIMEOUT = 120
MAINTENANCE = 'under daily maintenance'
"""SDA's page for its daily maintenance (12:30 to 12:45 AM Central), sent with a 200 status."""
MAINTENANCE_WAIT_S = 16 * 60          # the window's 15 min, and a minute over
MAINTENANCE_POLL_S = 60


# ------------------------------------------------------------- transport ----
def query(sql: str, tries: int = 2, sleep=time.sleep) -> list[dict]:
    """POST one T-SQL statement, return rows as dicts.

    The transport retries a 5xx, a 429, a timeout or a dropped connection about five times over
    about five minutes (`soil.net`) and then raises `soil.net.OutageError`: SDA is a government
    service with no SLA and goes down during shutdowns. An OGC XML body with a 200 status is
    SDA refusing the query, asked `tries` times in case the refusal was the server's own.
    SDA's daily maintenance page is waited out (MAINTENANCE_WAIT_S, asked again every
    MAINTENANCE_POLL_S) without spending a try: it is an outage, never the query's fault.
    """
    body = json.dumps({'query': sql, 'format': 'JSON+COLUMNNAME'}).encode()
    raw, attempt, waited = '', 0, 0
    while attempt < tries:
        req = urllib.request.Request(
            SDA_URL, data=body,
            headers={'Content-Type': 'application/json'})
        raw = net.urlopen(req, 'USDA Soil Data Access', timeout=_TIMEOUT).decode()
        # Errors arrive as OGC XML with a 200 status.
        if raw.lstrip().startswith('{'):
            table = json.loads(raw).get('Table')
            if not table:
                return []
            cols, rows = table[0], table[1:]
            return [dict(zip(cols, row)) for row in rows]
        if MAINTENANCE in raw and waited < MAINTENANCE_WAIT_S:
            # a site's storm asked during the window (05:35Z, 2026-10-24): the 200 page read as a refusal, the
            # attempt failed, and the attempt cap held the block as if the cause were its own
            sleep(MAINTENANCE_POLL_S)
            waited += MAINTENANCE_POLL_S
            continue
        attempt += 1
        if attempt < tries:
            sleep(2 * attempt)
    raise RuntimeError('SDA rejected the query: ' + raw[:400])


# -------------------------------------------------------------- geometry ----
def polygon_wkt(geom: dict) -> str:
    """GeoJSON Polygon to WKT. Outer ring only; SSURGO holes are not our problem."""
    ring = geom['coordinates'][0]
    return 'POLYGON((%s))' % ', '.join('%.8f %.8f' % (x, y) for x, y in ring)


def mukeys_for_site(geom: dict) -> list[dict]:
    """Map units under a site boundary, each with its share of the site area.

    Areas come back in square degrees, which is wrong as an absolute number and
    fine as a ratio over a parcel-sized polygon. Only the ratio is ever used;
    `geo.polygon_area_m2` is the right tool when a real area is wanted.
    """
    from .grid import default_grid
    g = default_grid()
    if g is not None:
        units = g.mukeys_for_site(geom)
        if units:
            return units

    wkt = polygon_wkt(geom)
    rows = query("""
        SELECT mukey,
               SUM(mupolygongeo.STIntersection(
                     geometry::STGeomFromText('%s', 4326)).STArea()) AS isect
        FROM mupolygon
        WHERE mupolygongeo.STIntersects(
                geometry::STGeomFromText('%s', 4326)) = 1
        GROUP BY mukey
    """ % (wkt, wkt))

    total = sum(float(r['isect']) for r in rows) or 1.0
    out = [{'mukey': r['mukey'],
            'site_fraction': round(float(r['isect']) / total, 4)}
           for r in rows]
    return sorted(out, key=lambda d: -d['site_fraction'])


def mukey_at_point(lon: float, lat: float) -> str | None:
    """The local grid when installed and the point is on it, else SDA."""
    from .grid import default_grid
    g = default_grid()
    if g is not None:
        mk = g.mukey_at(lon, lat)
        if mk:
            return mk
    return mukey_at_point_sda(lon, lat)


def mukey_at_point_sda(lon: float, lat: float) -> str | None:
    rows = query("""
        SELECT TOP 1 * FROM SDA_Get_Mukey_from_intersection_with_WktWgs84(
            'point(%.6f %.6f)')
    """ % (lon, lat))
    if not rows:
        return None
    return str(list(rows[0].values())[0])


# ------------------------------------------------------------ map  units ----
def fetch_mapunit(mukey: str) -> dict:
    """Everything worth keeping about one map unit, in four round trips."""
    agg = query("SELECT %s FROM muaggatt WHERE mukey = '%s'"
                % (', '.join(MUAGG_COLS), mukey))

    comps = query("SELECT %s FROM component WHERE mukey = '%s' ORDER BY %s"
                  % (', '.join(COMP_COLS), mukey, COMP_ORDER))
    if not comps:
        return {'mukey': mukey, 'schema': SCHEMA, 'fetched': _today(),
                'mapunit': agg[0] if agg else None,
                'components': []}

    cokeys = ','.join("'%s'" % c['cokey'] for c in comps)

    horizons = query("SELECT %s FROM chorizon WHERE cokey IN (%s) ORDER BY cokey, hzdept_r"
                     % (', '.join(HZ_COLS), cokeys))

    interps = query("""
        SELECT %s FROM cointerp
        WHERE cokey IN (%s) AND ruledepth = 0 AND mrulename IN (%s)
    """ % (', '.join(INTERP_COLS), cokeys, ','.join("'%s'" % i for i in INTERPRETATIONS)))

    by_hz: dict[str, list] = {}
    for h in horizons:
        by_hz.setdefault(h['cokey'], []).append(h)
    by_in: dict[str, list] = {}
    for i in interps:
        by_in.setdefault(i['cokey'], []).append(i)

    for c in comps:
        c['horizons'] = by_hz.get(c['cokey'], [])
        c['interpretations'] = by_in.get(c['cokey'], [])

    return {
        'mukey': mukey,
        'schema': SCHEMA,
        'fetched': _today(),
        'source': 'USDA-NRCS SSURGO via Soil Data Access',
        'mapunit': agg[0] if agg else None,
        'components': comps,
    }


def _today() -> str:
    return time.strftime('%Y-%m-%d')


# ----------------------------------------------------------------- cache ----
class FileCache:
    """One JSON file per map unit.

    Deliberately dumb. The real store is Mongo or Parquet in GCS and nothing
    above this call cares which, so the swap is a class with the same two
    methods rather than a migration.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def get(self, mukey: str) -> dict | None:
        p = os.path.join(self.path, '%s.json' % mukey)
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            return json.load(fh)

    def put(self, mukey: str, record: dict) -> None:
        p = os.path.join(self.path, '%s.json' % mukey)
        tmp = p + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(record, fh, separators=(',', ':'))
        os.replace(tmp, p)      # atomic, so a crash cannot leave half a record


_DEFAULT_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'soil_cache')


def get_mapunit(mukey: str, cache=None) -> tuple[dict, bool]:
    """Cache-first read. Returns (record, was_cache_hit)."""
    if cache is None:
        from .index import default_store
        files = FileCache(_DEFAULT_CACHE)
        cache = default_store(files) or files
    hit = cache.get(mukey)
    if hit is not None and hit.get('schema', 1) >= SCHEMA:
        return hit, True
    record = fetch_mapunit(mukey)
    cache.put(mukey, record)
    return record, False


# ----------------------------------------------------------- readability ----
def has_profile(record: dict) -> bool:
    """Does any component in this map unit carry horizon data?

    This is the question that decides whether we show SSURGO or fall back.
    Measured across ten metro cores: the median urban core answers no for 83%
    of its ground, because `Urban land` components carry no horizons at all.
    """
    return any(c.get('horizons') for c in record.get('components', []))


def dominant_component(record: dict) -> dict | None:
    """Highest-percentage component that actually has horizon data."""
    withdata = [c for c in record.get('components', []) if c.get('horizons')]
    if not withdata:
        return None
    return max(withdata, key=lambda c: int(c.get('comppct_r') or 0))
