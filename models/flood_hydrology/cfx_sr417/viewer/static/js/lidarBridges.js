import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

/**
 * createLidarBridges() → THREE.Group
 *
 * Loads the two raw-LiDAR 2.5D Delaunay TIN meshes built by
 * lidar/build_lidar_pointcloud.py — the actual point-cloud surface (including
 * bridge-deck returns) at the two places SR417 crosses a surface street
 * (Town Loop Boulevard, John Young Parkway/CR423). The bare-earth DEM used
 * for the rest of this viewer's terrain incorrectly drops ~7.5-8.4m to grade
 * level at these two spots (bridge-deck LiDAR returns get classified as
 * non-ground and stripped from a bare-earth DTM) — see
 * lidar/data/BRIDGE_VALIDATION.md for the full analysis.
 *
 * Mesh vertex coordinates are pre-baked into the viewer's own scene-space
 * convention (same X/Y/Z transform as terrain.js, VERT_EXAG included) by the
 * Python export step, so each loaded object is added with zero extra
 * position/scale transform.
 */
export async function createLidarBridges() {
  const loader = new OBJLoader();
  const mat = new THREE.MeshStandardMaterial({
    color: 0xff5533,
    metalness: 0.1,
    roughness: 0.6,
    side: THREE.DoubleSide,
  });

  const files = [
    { name: 'Town Loop Blvd bridge',   url: '/data/bridge_mesh_town_loop_blvd.obj' },
    { name: 'John Young Pkwy bridge',  url: '/data/bridge_mesh_john_young_pkwy.obj' },
  ];

  const group = new THREE.Group();
  group.name = 'lidar-bridge-crossings';

  await Promise.all(files.map(async ({ name, url }) => {
    const obj = await loader.loadAsync(url);
    obj.traverse(child => {
      if (child.isMesh) {
        child.material = mat;
        child.geometry.computeVertexNormals();
        child.name = name;
      }
    });
    group.add(obj);
  }));

  return { mesh: group };
}
