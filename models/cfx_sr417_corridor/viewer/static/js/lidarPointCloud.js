import * as THREE from 'three';

/**
 * createLidarPointCloud(url, opts) → { mesh, points }
 *
 * Loads a LiDAR point cloud exported by lidar/export_full_pointcloud.py. Three variants share
 * this same loader/binary format:
 *   /data/lidar_pointcloud.bin          ~4.16M pts, decimated, whole AOI (default layer)
 *   /data/lidar_pointcloud_full.bin     70,684,237 pts, EVERY point in the AOI, no decimation
 *   /data/lidar_pointcloud_5houses.bin  ~286k pts, a ~160m box around 5-6 houses, no decimation
 *                                       — checks whether raw point density can resolve
 *                                       individual building outlines.
 * All are colored by sampling the NAIP 2021 orthophoto at each point's location by default
 * (true color, cross-checkable against the NAIP layer directly), or by ASPRS classification
 * if exported with --color-by classification.
 *
 * Binary format:
 *   b'PCLD' + uint32 n_points + float32[n*3] positions (already in scene-space,
 *   VERT_EXAG/z_min/origin baked in — same convention as terrain.js) + uint8[n*3] colors
 */
export async function createLidarPointCloud(url = '/data/lidar_pointcloud.bin', opts = {}) {
  const buf = await fetch(url).then(r => r.arrayBuffer());
  const dv = new DataView(buf);

  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== 'PCLD') throw new Error(`${url}: bad magic "${magic}"`);
  const n = dv.getUint32(4, true);

  const posOffset = 8;
  const colOffset = posOffset + n * 3 * 4;
  const positions = new Float32Array(buf, posOffset, n * 3);
  const colorsU8  = new Uint8Array(buf, colOffset, n * 3);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colorsU8, 3, true));   // normalized

  const mat = new THREE.PointsMaterial({
    size: opts.size ?? 1.4,
    sizeAttenuation: true,
    vertexColors: true,
  });

  const points = new THREE.Points(geo, mat);
  points.name = opts.name ?? 'LiDAR Point Cloud';

  return { mesh: points, points, n };
}
