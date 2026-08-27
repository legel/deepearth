import * as THREE from 'three';

/**
 * createOverlays(geoMeta) → Array<{ name, mesh, defaultOn }>
 *
 * Flat textured planes at Y ≈ 2-4 (just above terrain base), sharing
 * the same XZ extent as terrain.js so textures align pixel-perfect.
 *
 * PNG files exported by export_overlays.py are north-up (row 0 = north).
 * Three.js default flipY=true is correct: it maps UV(0,1)=north-west corner
 * to PNG row 0 (north edge).
 */
export async function createOverlays(geoMeta) {
  const { width_m, height_m } = geoMeta;
  const loader = new THREE.TextureLoader();

  function flatPlane(textureUrl, yOffset, opacity, transparent) {
    const tex = loader.load(textureUrl);
    tex.colorSpace = THREE.SRGBColorSpace;
    const mat = new THREE.MeshBasicMaterial({
      map: tex,
      transparent: transparent ?? true,
      opacity: opacity ?? 0.85,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const geo = new THREE.PlaneGeometry(width_m, height_m);
    geo.rotateX(-Math.PI / 2);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = yOffset;
    return mesh;
  }

  const overlays = [
    {
      name: 'SSURGO Soils',
      mesh: flatPlane('/data/ssurgo.png', 2, 0.75, true),
      defaultOn: true,
    },
    {
      name: 'Hydrography (3DHP)',
      mesh: flatPlane('/data/hydrography.png', 3, 0.85, true),
      defaultOn: true,
    },
    {
      name: 'FEMA Flood Zones',
      mesh: flatPlane('/data/floodplain.png', 4, 0.70, true),
      defaultOn: true,
    },
    {
      name: 'CFX Corridor Boundary',
      mesh: flatPlane('/data/boundary.png', 5, 0.90, true),
      defaultOn: true,
    },
    // ── Hydrological derivatives (dem/dem_hydro.py → dem/data/hydro/) ──────
    {
      name: 'HAND (Height Above Drainage)',
      mesh: flatPlane('/data/hydro_hand.png', 6, 0.80, true),
      defaultOn: true,
    },
    {
      name: 'Flow Accumulation',
      mesh: flatPlane('/data/hydro_flow_accum.png', 6, 0.75, true),
      defaultOn: false,
    },
    {
      name: 'Stream Network (D8)',
      mesh: flatPlane('/data/hydro_streams.png', 7, 0.90, true),
      defaultOn: true,
    },
    // ── Terrain derivatives (dem/dem_terrain.py → dem/data/terrain/) ──────
    {
      name: 'Slope',
      mesh: flatPlane('/data/terrain_slope.png', 6, 0.75, true),
      defaultOn: false,
    },
    {
      name: 'Hillshade',
      mesh: flatPlane('/data/terrain_hillshade.png', 6, 0.70, true),
      defaultOn: false,
    },
    {
      name: 'TPI (Topographic Position)',
      mesh: flatPlane('/data/terrain_tpi.png', 6, 0.75, true),
      defaultOn: false,
    },
    {
      name: 'Profile Curvature',
      mesh: flatPlane('/data/terrain_curvature.png', 6, 0.75, true),
      defaultOn: false,
    },
    // ── PlanetScope RGB true-colour imagery ──────────────────────────────
    { name: 'PS RGB: Ian 2022-09-30 (MAX)',         mesh: flatPlane('/data/ps_max1_rgb.png',  8, 0.95, true), defaultOn: true  },
    { name: 'PS RGB: 2025-05-29 (MAX)',              mesh: flatPlane('/data/ps_max2_rgb.png',  8, 0.95, true), defaultOn: false },
    { name: 'PS RGB: 2024-06-30 (MAX)',              mesh: flatPlane('/data/ps_max3_rgb.png',  8, 0.95, true), defaultOn: false },
    { name: 'PS RGB: 2025-06-17 (AVG wet)',          mesh: flatPlane('/data/ps_avg1_rgb.png',  8, 0.95, true), defaultOn: false },
    { name: 'PS RGB: 2023-09-17 (AVG wet)',          mesh: flatPlane('/data/ps_avg2_rgb.png',  8, 0.95, true), defaultOn: false },
    { name: 'PS RGB: 2025-12-04 (AVG dry)',          mesh: flatPlane('/data/ps_avg3_rgb.png',  8, 0.95, true), defaultOn: false },
    { name: 'PS RGB: 2023-12-23 (AVG dry)',          mesh: flatPlane('/data/ps_avg4_rgb.png',  8, 0.95, true), defaultOn: false },
    { name: 'PS RGB: 2021-03-17 (MIN 43d dry)',      mesh: flatPlane('/data/ps_min1_rgb.png',  8, 0.95, true), defaultOn: false },
    { name: 'PS RGB: 2024-12-09 (MIN 34d dry)',      mesh: flatPlane('/data/ps_min2_rgb.png',  8, 0.95, true), defaultOn: false },
    { name: 'PS RGB: 2022-04-19 (MIN 33d dry)',      mesh: flatPlane('/data/ps_min3_rgb.png',  8, 0.95, true), defaultOn: false },
    // ── PlanetScope NDVI (RdYlGn: red=bare/water, green=healthy vegetation) ─
    { name: 'PS NDVI: Ian 2022-09-30 (MAX)',         mesh: flatPlane('/data/ps_max1_ndvi.png', 8, 0.85, true), defaultOn: false },
    { name: 'PS NDVI: 2025-05-29 (MAX)',             mesh: flatPlane('/data/ps_max2_ndvi.png', 8, 0.85, true), defaultOn: false },
    { name: 'PS NDVI: 2024-06-30 (MAX)',             mesh: flatPlane('/data/ps_max3_ndvi.png', 8, 0.85, true), defaultOn: false },
    { name: 'PS NDVI: 2025-06-17 (AVG wet)',         mesh: flatPlane('/data/ps_avg1_ndvi.png', 8, 0.85, true), defaultOn: false },
    { name: 'PS NDVI: 2023-09-17 (AVG wet)',         mesh: flatPlane('/data/ps_avg2_ndvi.png', 8, 0.85, true), defaultOn: false },
    { name: 'PS NDVI: 2025-12-04 (AVG dry)',         mesh: flatPlane('/data/ps_avg3_ndvi.png', 8, 0.85, true), defaultOn: false },
    { name: 'PS NDVI: 2023-12-23 (AVG dry)',         mesh: flatPlane('/data/ps_avg4_ndvi.png', 8, 0.85, true), defaultOn: false },
    { name: 'PS NDVI: 2021-03-17 (MIN 43d dry)',     mesh: flatPlane('/data/ps_min1_ndvi.png', 8, 0.85, true), defaultOn: false },
    { name: 'PS NDVI: 2024-12-09 (MIN 34d dry)',     mesh: flatPlane('/data/ps_min2_ndvi.png', 8, 0.85, true), defaultOn: false },
    { name: 'PS NDVI: 2022-04-19 (MIN 33d dry)',     mesh: flatPlane('/data/ps_min3_ndvi.png', 8, 0.85, true), defaultOn: false },
    // ── PlanetScope water extent — all 10 scenes (compare by toggling) ───
    // MAX scenes — high-rain events (bright cyan)
    { name: 'PS: Ian Flood 2022-09-30 (MAX 25.9 ha)',   mesh: flatPlane('/data/ps_max1_water.png', 7, 0.85, true), defaultOn: true  },
    { name: 'PS: 2025-05-29 (MAX 3.8 ha)',               mesh: flatPlane('/data/ps_max2_water.png', 7, 0.80, true), defaultOn: false },
    { name: 'PS: 2024-06-30 (MAX 11.6 ha)',              mesh: flatPlane('/data/ps_max3_water.png', 7, 0.80, true), defaultOn: false },
    // AVG scenes — representative weeks
    { name: 'PS: 2025-06-17 (AVG wet 7.3 ha)',           mesh: flatPlane('/data/ps_avg1_water.png', 7, 0.75, true), defaultOn: false },
    { name: 'PS: 2023-09-17 (AVG wet 25.8 ha)',          mesh: flatPlane('/data/ps_avg2_water.png', 7, 0.75, true), defaultOn: false },
    { name: 'PS: 2025-12-04 (AVG dry 24.4 ha)',          mesh: flatPlane('/data/ps_avg3_water.png', 7, 0.75, true), defaultOn: false },
    { name: 'PS: 2023-12-23 (AVG dry 20.0 ha)',          mesh: flatPlane('/data/ps_avg4_water.png', 7, 0.75, true), defaultOn: false },
    // MIN scenes — dry streaks (muted green-blue)
    { name: 'PS: 2021-03-17 (MIN 43-day dry 17.0 ha)',   mesh: flatPlane('/data/ps_min1_water.png', 7, 0.70, true), defaultOn: false },
    { name: 'PS: 2024-12-09 (MIN 34-day dry 20.5 ha)',   mesh: flatPlane('/data/ps_min2_water.png', 7, 0.70, true), defaultOn: false },
    { name: 'PS: 2022-04-19 (MIN 33-day dry 15.5 ha)',   mesh: flatPlane('/data/ps_min3_water.png', 7, 0.70, true), defaultOn: false },
  ];

  overlays.forEach(o => { o.mesh.name = o.name; });
  return overlays;
}

/**
 * createDrapedOverlay(terrainGeometry, textureUrl, opacity) → THREE.Mesh
 *
 * Paints a texture directly onto the terrain's actual displaced surface
 * (same geometry instance as terrain.js's solid mesh) instead of a flat
 * floating plane. Same UVs as terrain.js / flatPlane() above, since both
 * are built from an undisplaced PlaneGeometry(width_m, height_m) before
 * vertex displacement — the texture lines up unmodified.
 */
export function createDrapedOverlay(terrainGeometry, textureUrl, opacity = 0.85) {
  const loader = new THREE.TextureLoader();
  const tex = loader.load(textureUrl);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.MeshBasicMaterial({
    map: tex,
    transparent: true,
    opacity,
    depthWrite: false,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: -4,
    polygonOffsetUnits: -4,
  });
  return new THREE.Mesh(terrainGeometry, mat);
}
