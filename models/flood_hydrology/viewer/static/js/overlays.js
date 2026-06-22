import * as THREE from 'three';

/**
 * createOverlays(geoMeta) → Array<{ name, mesh, defaultOn }>
 *
 * Flat textured planes at Y ≈ 2 (just above terrain base), sharing
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

  // Lake Mask removed — water surface in main.js now uses lake_mask.png texture
  // at the correct waterY elevation, which is the canonical lake-shape layer.
  const overlays = [
    {
      name: 'NAIP Aerial',
      mesh: flatPlane('/data/naip_rgb.png', 3, 0.90, false),
      defaultOn: false,
    },
    {
      name: 'SSURGO Soils',
      mesh: flatPlane('/data/ssurgo.png', 2, 0.75, true),
      defaultOn: false,
    },
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
 *
 * polygonOffset nudges the draped mesh slightly toward the camera in depth-
 * buffer space so it doesn't z-fight with the co-planar terrain solid mesh
 * (both share the exact same vertex positions).
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
