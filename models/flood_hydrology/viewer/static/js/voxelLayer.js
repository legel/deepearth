import * as THREE from 'three';
import { VERT_EXAG } from './terrain.js';

// 11-stop Viridis samples (t=0 deep purple → t=1 bright yellow)
const VIRIDIS = [
  [0.267, 0.005, 0.329],
  [0.283, 0.141, 0.458],
  [0.253, 0.265, 0.530],
  [0.207, 0.372, 0.553],
  [0.164, 0.471, 0.558],
  [0.128, 0.567, 0.551],
  [0.135, 0.659, 0.518],
  [0.267, 0.749, 0.441],
  [0.478, 0.821, 0.318],
  [0.741, 0.873, 0.150],
  [0.993, 0.906, 0.144],
];

function viridisRGB(t) {
  t = Math.max(0, Math.min(1, t));
  const raw = t * (VIRIDIS.length - 1);
  const lo = Math.floor(raw);
  const hi = Math.min(lo + 1, VIRIDIS.length - 1);
  const f = raw - lo;
  return VIRIDIS[lo].map((v, k) => v + f * (VIRIDIS[hi][k] - v));
}

/**
 * createVoxelLayer(geoMeta) → { mesh: THREE.InstancedMesh } | null
 *
 * Reads voxels.bin header to get cell size, z_res, water_surface.
 * Positions each voxel in the shared XYZ frame used by terrain.js.
 */
export async function createVoxelLayer(geoMeta) {
  const { width_m, height_m, z_min } = geoMeta;

  let buf;
  try {
    const resp = await fetch('/data/voxels.bin');
    if (!resp.ok) return null;
    buf = await resp.arrayBuffer();
  } catch {
    return null;
  }

  // Parse 32-byte header
  const view = new DataView(buf);
  const magic = String.fromCharCode(
    view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)
  );
  if (magic !== 'VOXL') { console.warn('voxels.bin: bad magic'); return null; }

  const nVoxels      = view.getUint32(4, true);
  const cellX        = view.getFloat32(8, true);
  const cellY        = view.getFloat32(12, true);
  const zRes         = view.getFloat32(16, true);
  const waterSurface = view.getFloat32(20, true);
  const demZMin      = view.getFloat32(24, true);
  const nZLayers     = view.getUint32(28, true);

  const HEADER = 32;
  const vox = new Uint16Array(buf, HEADER);  // (row, col, z_layer) triples

  const boxGeo = new THREE.BoxGeometry(1, 1, 1);
  // MeshBasicMaterial shows Viridis instance colors at full saturation regardless of lighting
  const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.92 });
  const mesh = new THREE.InstancedMesh(boxGeo, mat, nVoxels);
  mesh.name = 'voxels';

  const dummy = new THREE.Object3D();
  const color = new THREE.Color();
  const maxDepth = nZLayers * zRes;
  const instanceDepths = new Float32Array(nVoxels);

  for (let i = 0; i < nVoxels; i++) {
    const row    = vox[i * 3];
    const col    = vox[i * 3 + 1];
    const zLayer = vox[i * 3 + 2];

    // Centre of voxel in local frame
    const x = (col + 0.5) * cellX - width_m / 2;
    const z = (row + 0.5) * cellY - height_m / 2;
    const depth = (zLayer + 0.5) * zRes;
    const elev  = waterSurface - depth;
    const y     = (elev - demZMin) * VERT_EXAG;

    dummy.position.set(x, y, z);
    dummy.scale.set(cellX, zRes * VERT_EXAG, cellY);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);

    // Viridis reversed: surface (shallow) → bright yellow, deep → dark purple
    const [r, g, b] = viridisRGB(1.0 - depth / maxDepth);
    color.setRGB(r, g, b);
    mesh.setColorAt(i, color);

    instanceDepths[i] = depth;
  }

  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

  console.log(`Voxel layer: ${nVoxels.toLocaleString()} voxels, ` +
    `cell=${cellX.toFixed(2)}m, zRes=${zRes}m, waterSurface=${waterSurface.toFixed(2)}m`);

  return {
    mesh,
    getDepthAtInstance(id) { return instanceDepths[id]; },
  };
}
