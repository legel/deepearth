import * as THREE from 'three';

/**
 * Scene frame, shared by every layer:
 *   X = (col / cols) * width_m  - width_m/2    west -> east
 *   Y = (elev - z_min) * exag                  up
 *   Z = (row / rows) * height_m - height_m/2   north -> south
 *
 * The geometry is returned so draped overlays paint onto real elevation rather than floating
 * on a separate plane. Exaggeration is applied here and nowhere else: layers that bake their
 * own scale drift out of alignment the moment the slider moves.
 */
export async function createTerrain(meta, exag = 2) {
  const { rows, cols, z_min, width_m, height_m } = meta;
  const heights = new Float32Array(await fetch('/data/dem.bin').then(r => r.arrayBuffer()));

  const geometry = new THREE.PlaneGeometry(width_m, height_m, cols - 1, rows - 1);
  geometry.rotateX(-Math.PI / 2);

  const surface = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
    color: 0x6b7a8a, roughness: 0.95, metalness: 0.0, flatShading: false,
  }));
  const wire = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
    color: 0x2f4356, wireframe: true, transparent: true, opacity: 0.55,
  }));

  function setExag(value) {
    const pos = geometry.attributes.position;
    for (let i = 0; i < pos.count; i++) pos.setY(i, (heights[i] - z_min) * value);
    pos.needsUpdate = true;
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
  }
  setExag(exag);

  return { geometry, surface, wire, setExag, heights };
}

/** A texture draped on the terrain surface itself, lifted clear of z-fighting. */
export function createDrape(geometry, url) {
  const texture = new THREE.TextureLoader().load(url);
  texture.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
    map: texture, transparent: true, depthWrite: false,
  }));
  mesh.renderOrder = 1;
  mesh.position.y = 0.6;
  return mesh;
}
