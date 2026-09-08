import * as THREE from 'three';

/**
 * Animated flood depth, drawn as a texture on the terrain surface.
 *
 * SIML binary layout, written by cli.py and re-emitted by viewer/export.py:
 *   [0:4]   b'SIML'
 *   [4:16]  uint32 n_frames, rows, cols
 *   [16:..] float32 times_min[n_frames], then float32 depth[n_frames][rows][cols], metres
 */
const BUCKETS = [
  [0.02, 90, 200, 245], [0.05, 60, 150, 235], [0.15, 40, 100, 220],
  [0.40, 30, 65, 190], [Infinity, 40, 40, 150],
];

function paint(depth, rgba) {
  for (let i = 0; i < depth.length; i++) {
    const d = depth[i];
    const o = i * 4;
    if (d < 0.005) { rgba[o + 3] = 0; continue; }
    for (const [limit, r, g, b] of BUCKETS) {
      if (d < limit) {
        rgba[o] = r; rgba[o + 1] = g; rgba[o + 2] = b;
        rgba[o + 3] = Math.min(235, 90 + d * 420);
        break;
      }
    }
  }
}

export async function createFlood(geometry, url) {
  const buf = await fetch(url).then(r => r.arrayBuffer());
  const view = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
  if (magic !== 'SIML') throw new Error(`bad magic ${magic} in ${url}`);

  const n = view.getUint32(4, true);
  const rows = view.getUint32(8, true);
  const cols = view.getUint32(12, true);
  const times = new Float32Array(buf, 16, n);
  const depths = new Float32Array(buf, 16 + 4 * n, n * rows * cols);

  const rgba = new Uint8Array(rows * cols * 4);
  const texture = new THREE.DataTexture(rgba, cols, rows, THREE.RGBAFormat);
  texture.flipY = false;
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
    map: texture, transparent: true, depthWrite: false,
  }));
  mesh.renderOrder = 2;
  mesh.position.y = 1.2;

  function setFrame(i) {
    const k = Math.max(0, Math.min(n - 1, i | 0));
    rgba.fill(0);
    paint(depths.subarray(k * rows * cols, (k + 1) * rows * cols), rgba);
    texture.needsUpdate = true;
    return times[k] / 60.0;
  }
  setFrame(0);

  return { mesh, setFrame, frameCount: n, timesHours: Array.from(times, t => t / 60.0) };
}
