import * as THREE from 'three';

// Default vertical exaggeration. Live-adjustable via the panel slider (max 8x) as of
// 2026-07-28 — ported from the sibling cfx_sr417_corridor project's terrain.js, same
// updateExag()/createFixedExagGeometry() mechanism, adapted to Johns Lake's single dem.bin
// (no per-site demUrl parameter needed here, unlike CFX which serves multiple DEMs).
let VERT_EXAG = 8;

/**
 * createTerrain(geoMeta) → { mesh, solidMesh, wireMesh, geometry, exag, updateExag }
 *
 * Builds a 256×256 (or geoMeta rows×cols) PlaneGeometry displaced by DEM heights.
 * Scene coordinate frame (shared with voxelLayer and overlays):
 *   X = (col / cols) * width_m  - width_m/2   (west→east)
 *   Y = (elev - z_min) * VERT_EXAG             (elevation, up)
 *   Z = (row / rows) * height_m - height_m/2  (north→south, neg=far)
 *
 * `geometry` (the displaced, UV-mapped surface) is returned so other layers
 * (flood/infiltration draped textures) can paint directly onto the terrain's actual
 * elevation instead of floating as separate flat planes.
 */
async function loadHeights() {
  const buf = await fetch('/data/dem.bin').then(r => r.arrayBuffer());
  return new Float32Array(buf);
}

/**
 * createFixedExagGeometry(geoMeta, exag) → THREE.PlaneGeometry
 *
 * A displaced terrain geometry at a FIXED exaggeration, independent of the live Wireframe/
 * Surface slider above — same pattern as cfx_sr417_corridor's terrain.js. Used for the NAIP/
 * SSURGO draped overlays so they sit at a gentler fixed scale regardless of whatever the
 * wireframe is currently set to, since draping imagery onto a tall exaggerated surface reads
 * as more visually distorted than the wireframe/voxels alone.
 */
export async function createFixedExagGeometry(geoMeta, exag) {
  const { rows, cols, z_min, width_m, height_m } = geoMeta;
  const heights = await loadHeights();
  const geo = new THREE.PlaneGeometry(width_m, height_m, cols - 1, rows - 1);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    pos.setY(i, (heights[i] - z_min) * exag);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  return geo;
}

export async function createTerrain(geoMeta) {
  const { rows, cols, z_min, width_m, height_m } = geoMeta;

  const heights = await loadHeights();

  const geo = new THREE.PlaneGeometry(width_m, height_m, cols - 1, rows - 1);
  geo.rotateX(-Math.PI / 2);

  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    pos.setY(i, (heights[i] - z_min) * VERT_EXAG);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();

  // Semi-transparent ground so lake voxels are visible through it
  const solidMat = new THREE.MeshLambertMaterial({
    color: 0x122818,
    transparent: true,
    opacity: 0.72,
  });
  const solid = new THREE.Mesh(geo, solidMat);
  solid.name = 'Terrain Surface';

  // Wireframe — brighter so it reads on dark background
  let wireGeo = new THREE.WireframeGeometry(geo);
  const wireMat = new THREE.LineBasicMaterial({
    color: 0x3aaa60,
    transparent: true,
    opacity: 0.60,
  });
  const wire = new THREE.LineSegments(wireGeo, wireMat);
  wire.name = 'Terrain Wireframe';

  const group = new THREE.Group();
  group.name = 'terrain';
  group.add(solid);
  group.add(wire);

  // Recompute vertex heights at a new exaggeration factor — used by the viewer's live
  // vertical-exaggeration slider. WireframeGeometry doesn't update in place (it snapshots
  // edges at creation time), so it's disposed and rebuilt from the updated `geo`.
  function updateExag(newExag) {
    VERT_EXAG = newExag;
    for (let i = 0; i < pos.count; i++) {
      pos.setY(i, (heights[i] - z_min) * newExag);
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();

    wireGeo.dispose();
    wireGeo = new THREE.WireframeGeometry(geo);
    wire.geometry = wireGeo;
  }

  return { mesh: group, solidMesh: solid, wireMesh: wire, geometry: geo, exag: VERT_EXAG,
           updateExag };
}

export { VERT_EXAG };
