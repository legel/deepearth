import * as THREE from 'three';

// Default vertical exaggeration for the live terrain view.
// Kept low (2x) because SR417's real embankment relief (4-9m rise, confirmed directly from the
// uncorrected bare-earth 1m LiDAR DEM — not a data artifact) is a large fraction of this
// already-flat AOI's total natural relief (14.1m); at 8x the highway reads as an implausible
// hill. Adjustable live via the slider (max 8x).
//
// NOTE: this is the VIEW default only. Layers whose geometry is pre-baked on disk (LiDAR point
// clouds/meshes, flood layers, rain particles) use a fixed BAKED exaggeration of 8 and are
// rescaled at load by (currentExag / 8). Do not conflate the two constants.
let VERT_EXAG = 2;

/**
 * createTerrain(geoMeta) → { mesh, solidMesh, wireMesh, geometry, exag }
 *
 * Builds a 256×256 PlaneGeometry displaced by DEM heights.
 * Scene coordinate frame (shared with voxelLayer and overlays):
 *   X = (col / cols) * width_m  - width_m/2   (west→east)
 *   Y = (elev - z_min) * VERT_EXAG             (elevation, up)
 *   Z = (row / rows) * height_m - height_m/2  (north→south, neg=far)
 *
 * `geometry` (the displaced, UV-mapped surface) is returned so other layers
 * (overlays.js draped textures) can paint directly onto the terrain's actual
 * elevation instead of floating as separate flat planes — same UV convention
 * as the flat overlay planes (built from an undisplaced PlaneGeometry of the
 * same width_m/height_m), so existing NAIP/SSURGO PNGs line up unmodified.
 */
async function loadHeights(demUrl = '/data/dem.bin') {
  const buf = await fetch(demUrl).then(r => r.arrayBuffer());
  return new Float32Array(buf);
}

/**
 * createFixedExagGeometry(geoMeta, exag) → THREE.PlaneGeometry
 *
 * A displaced terrain geometry at a FIXED exaggeration, independent of the live Wireframe/
 * Surface slider above — same "own fixed local constant, not wired to the live slider"
 * pattern already used by rainParticles.js/floodLayer.js. Added so the NAIP/SSURGO draped
 * overlays (main.js) can sit at a gentler fixed 2x regardless of whatever the wireframe is
 * currently set to, since draping imagery onto a tall exaggerated surface reads as more
 * visually distorted than the wireframe alone does.
 */
export async function createFixedExagGeometry(geoMeta, exag, demUrl) {
  const { rows, cols, z_min, width_m, height_m } = geoMeta;
  const heights = await loadHeights(demUrl);
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

export async function createTerrain(geoMeta, demUrl) {
  const { rows, cols, z_min, width_m, height_m } = geoMeta;

  const heights = await loadHeights(demUrl);

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
