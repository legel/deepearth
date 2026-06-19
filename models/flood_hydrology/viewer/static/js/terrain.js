import * as THREE from 'three';

const VERT_EXAG = 8;  // vertical exaggeration factor

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
export async function createTerrain(geoMeta) {
  const { rows, cols, z_min, width_m, height_m } = geoMeta;

  const buf = await fetch('/data/dem.bin').then(r => r.arrayBuffer());
  const heights = new Float32Array(buf);

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
  const wireGeo = new THREE.WireframeGeometry(geo);
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

  return { mesh: group, solidMesh: solid, wireMesh: wire, geometry: geo, exag: VERT_EXAG };
}

export { VERT_EXAG };
