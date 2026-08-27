import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

/**
 * createDenseTestMesh(url) → { mesh }
 *
 * Loads the fused ground+roof surface built by lidar/droplet_flow_test.py — the DEM (modeled
 * bare earth) for the ground, plus a separate LiDAR-derived mesh per building roof, combined
 * into one OBJ. This is the actual 3D surface the droplet rainfall/runoff test flows across —
 * added specifically because the point cloud alone doesn't read as a surface (scattered dots,
 * nothing solid underneath the moving droplets).
 */
export async function createDenseTestMesh(url = '/data/dense_test_area_mesh.obj') {
  const loader = new OBJLoader();
  const mat = new THREE.MeshStandardMaterial({
    color: 0x4a7a5a,
    metalness: 0.05,
    roughness: 0.85,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.55,
  });

  const obj = await loader.loadAsync(url);
  obj.traverse(child => {
    if (child.isMesh) {
      child.material = mat;
      child.geometry.computeVertexNormals();
    }
  });
  obj.name = 'Dense Test Area Surface (DEM + roofs)';

  return { mesh: obj };
}
