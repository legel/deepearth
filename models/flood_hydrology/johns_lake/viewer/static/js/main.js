import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createTerrain, createFixedExagGeometry } from './terrain.js';
import { createVoxelLayer } from './voxelLayer.js';
import { createOverlays, createDrapedOverlay } from './overlays.js';
import { setupLayerPanel } from './layerControls.js';
import { createRainSystem } from './rainParticles.js';
import { createFloodLayer } from './floodLayer.js';
import { setupSimulationControls } from './simulationControls.js';
import { createLidarPointCloud } from './lidarPointCloud.js';

const loadingEl  = document.getElementById('loading-overlay');
const loadingTxt = document.getElementById('loading-text');
const hoverInfo  = document.getElementById('hover-info');

function setStatus(msg) { loadingTxt.textContent = msg; }

async function init() {
  // ── Renderer ────────────────────────────────────────────────
  const canvas = document.getElementById('canvas');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = false;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  // ── Scene ───────────────────────────────────────────────────
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x080c14);
  scene.fog = new THREE.FogExp2(0x080c14, 0.00012);

  // ── Camera ──────────────────────────────────────────────────
  const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 1, 25000);

  // ── Controls ────────────────────────────────────────────────
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.screenSpacePanning = false;
  controls.minDistance = 50;
  controls.maxDistance = 10000;
  controls.maxPolarAngle = Math.PI / 2.05;

  // ── Lights ──────────────────────────────────────────────────
  scene.add(new THREE.AmbientLight(0x6080a0, 1.8));
  const sun = new THREE.DirectionalLight(0xfff0d8, 2.5);
  sun.position.set(600, 1200, 400);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0x4060c0, 0.6);
  fill.position.set(-400, 300, -600);
  scene.add(fill);

  // ── Load geo metadata ────────────────────────────────────────
  setStatus('Loading terrain data…');
  const geoMeta = await fetch('/data/geo_meta.json').then(r => r.json());
  const { width_m, height_m, z_min, z_max, water_surface,
          rows, cols,
          lake_x_center = 0, lake_z_center = 0 } = geoMeta;

  // ── Terrain ─────────────────────────────────────────────────
  setStatus('Building DEM wireframe…');
  const terrain = await createTerrain(geoMeta);
  scene.add(terrain.mesh);

  // ── Voxels ──────────────────────────────────────────────────
  setStatus('Loading lake voxels…');
  const voxelResult = await createVoxelLayer(geoMeta);
  if (voxelResult) scene.add(voxelResult.mesh);

  // ── Overlays ─────────────────────────────────────────────────
  setStatus('Loading overlays…');
  const overlays = await createOverlays(geoMeta);
  overlays.forEach(o => scene.add(o.mesh));

  // Draped variants — painted onto a FIXED 2x-exaggeration geometry (not the live
  // terrain.geometry) so they stay legible regardless of the wireframe/voxel exaggeration
  // slider below — same pattern as cfx_sr417_corridor's terrain.js (draping imagery onto a
  // tall exaggerated surface reads as more visually distorted than the wireframe alone).
  const drapedGeo2x = await createFixedExagGeometry(geoMeta, 2);
  const drapedNaip = createDrapedOverlay(drapedGeo2x, '/data/naip_rgb.png', 0.90);
  drapedNaip.name = 'NAIP Aerial (Draped)';
  drapedNaip.visible = false;
  scene.add(drapedNaip);

  const drapedSsurgo = createDrapedOverlay(drapedGeo2x, '/data/ssurgo.png', 0.75);
  drapedSsurgo.name = 'SSURGO Soils (Draped)';
  drapedSsurgo.visible = false;
  scene.add(drapedSsurgo);

  // ── Water surface plane (at lake level) ─────────────────────
  // Live vertical-exaggeration support (2026-07-28): waterY depends on the current
  // exaggeration, so it's tracked as `currentExag`/`waterY` (mutable) instead of the old
  // fixed VERT_EXAG constant, and recomputed by onExagChange() below on every slider move.
  let currentExag = terrain.exag;
  let waterY = (water_surface - z_min) * currentExag;
  // Water surface: full-extent plane textured with lake_mask.png so only the real lake
  // shape is visible (transparent pixels = land, blue pixels = lake).
  const waterSurfaceGeo = new THREE.PlaneGeometry(width_m, height_m);
  waterSurfaceGeo.rotateX(-Math.PI / 2);
  const waterTex = new THREE.TextureLoader().load('/data/lake_mask.png');
  waterTex.colorSpace = THREE.SRGBColorSpace;
  const waterMat = new THREE.MeshBasicMaterial({
    map: waterTex,
    transparent: true,
    opacity: 0.72,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const waterPlane = new THREE.Mesh(waterSurfaceGeo, waterMat);
  waterPlane.position.y = waterY;
  waterPlane.name = 'Water Surface';
  scene.add(waterPlane);

  // ── Rain particle system ─────────────────────────────────────
  const rain = createRainSystem(scene, geoMeta);

  // ── Flood depth animated layer ───────────────────────────────
  // Draped on terrain.geometry (the actual displaced DEM surface) instead of
  // a flat plane near z_min, so flooded areas paint onto the real elevation.
  const flood = createFloodLayer(scene, geoMeta, { terrainGeometry: terrain.geometry });

  // ── Infiltration animated layer (reuses floodLayer's depth-texture machinery) ──
  const infiltration = createFloodLayer(scene, geoMeta, {
    urlSuffix: 'infiltration',
    meshName: 'Infiltration',
    valueScale: 0.001,   // mm → m, to match depthToRGBA's metre buckets
    terrainGeometry: terrain.geometry,
  });

  // ── S2 ground truth overlay (static texture, toggled off by default) ──
  let s2Plane = null;
  try {
    const s2Tex = new THREE.TextureLoader().load('/data/s2_ground_truth_20240212.png');
    s2Tex.colorSpace = THREE.SRGBColorSpace;
    const s2Geo = new THREE.PlaneGeometry(width_m, height_m);
    s2Geo.rotateX(-Math.PI / 2);
    const s2Mat = new THREE.MeshBasicMaterial({
      map: s2Tex, transparent: true, opacity: 0.75,
      depthWrite: false, side: THREE.DoubleSide,
    });
    s2Plane = new THREE.Mesh(s2Geo, s2Mat);
    s2Plane.name = 'S2 Ground Truth 2024-02-12';
    s2Plane.position.y = waterY + 2;
    s2Plane.visible    = false;
    scene.add(s2Plane);
  } catch { /* S2 overlay optional */ }

  // ── Live vertical-exaggeration handler ────────────────────────
  // terrain.updateExag() rebuilds the wireframe/solid geometry; voxelResult.rescale() rescales
  // every InstancedMesh voxel's baked Y position/scale by the ratio (no live-updating geometry
  // to reuse there, unlike the terrain plane); the water/S2 planes get their Y position
  // recomputed directly since they're simple flat planes, not displaced geometry.
  function onExagChange(newExag, oldExag) {
    terrain.updateExag(newExag);
    if (voxelResult) voxelResult.rescale(newExag / oldExag);
    waterY = (water_surface - z_min) * newExag;
    waterPlane.position.y = waterY;
    if (s2Plane) s2Plane.position.y = waterY + 2;
    // rescalableMeshes is populated lazily (currently just the raw LiDAR point cloud, if
    // loaded) — defined further below, but only ever called here after user interaction, by
    // which point it's already initialized (closure over the same outer scope).
    rescalableMeshes.forEach(m => rescaleY(m, newExag / oldExag));
    currentExag = newExag;
  }

  // ── Layer controls ───────────────────────────────────────────
  const naipOverlay          = overlays.find(o => o.name === 'NAIP Aerial');
  const ssurgoOverlay        = overlays.find(o => o.name === 'SSURGO Soils');
  const hydrographyOverlay   = overlays.find(o => o.name === 'Hydrography (3DHP)');
  const femaOverlay          = overlays.find(o => o.name === 'FEMA Flood Zones');
  const roadsBuildingsOverlay = overlays.find(o => o.name === 'Roads & Buildings');

  setupLayerPanel({
    terrain: {
      solidMesh: terrain.solidMesh,
      wireMesh:  terrain.wireMesh,
      voxelMesh: voxelResult?.mesh,
      waterMesh: waterPlane,
      onExagChange,
    },
    hydrology: [
      ...(hydrographyOverlay ? [{ name: hydrographyOverlay.name, mesh: hydrographyOverlay.mesh,
                                   on: false, swatch: '#00aaff' }] : []),
    ],
    base: [
      // The one layer on at page load, matching the sibling site: aerial imagery draped on
      // the terrain, so the scene reads as a real place before any analysis layer is added.
      ...(naipOverlay ? [{ group: 'basemap',
                            name: 'Aerial Imagery (NAIP)', mesh: naipOverlay.mesh, on: true,
                            swatch: '#8a6a3a', drape: true, drapedMesh: drapedNaip,
                            drapedOn: true }] : []),
      ...(ssurgoOverlay ? [{ group: 'ground',
                              name: 'Soils (SSURGO)', mesh: ssurgoOverlay.mesh, on: false,
                              swatch: '#7a9a5a', drape: true, drapedMesh: drapedSsurgo,
                              drapedOn: false, legendUrl: '/data/ssurgo_legend.json' }] : []),
      ...(femaOverlay ? [{ group: 'regulatory',
                            name: 'FEMA Flood Zones', mesh: femaOverlay.mesh, on: false,
                            swatch: '#e02020' }] : []),
      ...(roadsBuildingsOverlay ? [{ group: 'ground',
                                      name: 'Roads & Buildings',
                                      mesh: roadsBuildingsOverlay.mesh, on: false,
                                      swatch: '#cd9b5f' }] : []),
    ],
    risk: [
      { name: 'Flood Depth',   mesh: flood.mesh,        on: false, swatch: '#2a60c0' },
      { name: 'Infiltration',  mesh: infiltration.mesh, on: false, swatch: '#5a8a3a' },
      ...(s2Plane ? [{ name: s2Plane.name, mesh: s2Plane, on: false, swatch: '#c04040' }] : []),
    ],
  });

  // ── Raw LiDAR point cloud (heavy, lazy-loaded on toggle) ──────────────────
  // Ported from cfx_sr417_corridor's wireLazyPointCloud pattern 2026-07-28 — the point cloud
  // isn't fetched until the checkbox is actually checked (multi-hundred-MB file), and rescales
  // in place to match whatever exaggeration is currently active (it's exported with
  // VERT_EXAG=8 baked into its positions, same convention as export_full_pointcloud.py /
  // terrain.js's own default). Not part of setupLayerPanel's declarative config since that
  // helper assumes the mesh already exists synchronously — this row is built manually and
  // inserted right after TOPOGRAPHY's last row (before the HYDROLOGY header) to match CFX's
  // own placement of the equivalent layer.
  function rescaleY(object3D, ratio) {
    object3D.traverse(child => {
      const geo = child.geometry;
      if (geo && geo.attributes && geo.attributes.position) {
        const pos = geo.attributes.position;
        for (let i = 0; i < pos.count; i++) pos.setY(i, pos.getY(i) * ratio);
        pos.needsUpdate = true;
        if (geo.computeVertexNormals) geo.computeVertexNormals();
      }
    });
  }
  const rescalableMeshes = [];

  const layerList = document.getElementById('layer-list');
  const hydroHeader = Array.from(layerList.querySelectorAll('.section-header'))
    .find(h => h.textContent === 'HYDROLOGY');

  const lidarRow = document.createElement('label');
  lidarRow.className = 'layer-row';
  const lidarCb = document.createElement('input');
  lidarCb.type = 'checkbox';
  lidarCb.id = 'lidar-full-cb';
  const lidarSw = document.createElement('div');
  lidarSw.className = 'layer-swatch';
  lidarSw.style.background = '#909090';
  const lidarLabel = document.createElement('span');
  lidarLabel.className = 'layer-label';
  lidarLabel.textContent = 'Raw LiDAR Point Cloud (full AOI, heavy)';
  lidarRow.append(lidarCb, lidarSw, lidarLabel);
  if (hydroHeader) layerList.insertBefore(lidarRow, hydroHeader);
  else layerList.appendChild(lidarRow);

  let loadedPointCloud = null;
  lidarCb.addEventListener('change', async () => {
    if (!lidarCb.checked) {
      if (loadedPointCloud) loadedPointCloud.mesh.visible = false;
      return;
    }
    if (loadedPointCloud) { loadedPointCloud.mesh.visible = true; return; }
    lidarCb.disabled = true;
    const prevText = lidarLabel.textContent;
    lidarLabel.textContent = prevText + ' — loading…';
    try {
      loadedPointCloud = await createLidarPointCloud('/data/lidar_pointcloud.bin',
        { name: 'LiDAR Point Cloud', size: 1.4 });
      rescaleY(loadedPointCloud.mesh, currentExag / 8);   // match whatever exag is active now
      scene.add(loadedPointCloud.mesh);
      rescalableMeshes.push(loadedPointCloud.mesh);
    } finally {
      lidarLabel.textContent = prevText;
      lidarCb.disabled = false;
    }
  });

  // ── Simulation controls (async; non-blocking — panel renders after scene is ready) ──
  setStatus('Loading simulation data…');
  let simController = null;
  setupSimulationControls({
    floodLayer: flood,
    infiltrationLayer: infiltration,
    rainParticles: rain,
    waterPlane,
    geoMeta,
    getExag: () => currentExag,
  }).then(ctrl => { simController = ctrl; }).catch(err => {
    console.warn('Simulation controls failed to load:', err);
  });

  // ── Camera: side-oblique view centred on lake so voxel bowl depth is visible ──
  // Low elevation angle (~25°) so the 3D depth of the voxel stack reads clearly. Uses the
  // initial waterY (not live-updated on exag change) — same as this project's existing
  // behavior; the camera doesn't re-center when the slider moves, only the geometry rescales.
  const dist = Math.max(width_m, height_m) * 0.55;
  camera.position.set(
    lake_x_center + dist * 0.25,
    waterY + dist * 0.45,   // ~25° above horizontal
    lake_z_center + dist * 0.90,
  );
  controls.target.set(lake_x_center, waterY - 30, lake_z_center);
  controls.update();

  // ── FWC bathymetry for depth-on-hover ────────────────────────
  // fwc_bed.bin: Float32, 256×256, same grid as dem.bin.
  // NaN outside Johns Lake — isFinite() is the lake test.
  let fwcBed = null;
  fetch('/data/fwc_bed.bin')
    .then(r => r.ok ? r.arrayBuffer() : Promise.reject())
    .then(buf => { fwcBed = new Float32Array(buf); })
    .catch(() => {});

  // ── Hover info via raycasting ─────────────────────────────────
  const raycaster = new THREE.Raycaster();
  const pointer   = new THREE.Vector2();
  const terrainMeshes = terrain.mesh.children.filter(c => c instanceof THREE.Mesh);

  window.addEventListener('mousemove', e => {
    pointer.x =  (e.clientX / window.innerWidth)  * 2 - 1;
    pointer.y = -(e.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);

    // Cast against terrain + voxels; voxels occlude terrain when viewed from the side
    const rayTargets = [...terrainMeshes];
    if (voxelResult) rayTargets.push(voxelResult.mesh);
    const hits = raycaster.intersectObjects(rayTargets);
    if (hits.length === 0) { hoverInfo.textContent = 'Hover for elevation'; return; }

    const hit = hits[0];

    // Voxel face hit — report depth directly from stored per-instance value
    if (hit.instanceId !== undefined && voxelResult) {
      const depth = voxelResult.getDepthAtInstance(hit.instanceId);
      hoverInfo.textContent = `Lake depth: ${depth.toFixed(2)} m (FWC survey)`;
      return;
    }

    // Terrain hit
    const pt   = hit.point;
    const elev = pt.y / currentExag + z_min;
    let info   = `Elev: ${elev.toFixed(1)} m NAVD88`;

    // Look up FWC lake depth at this XZ position
    if (fwcBed) {
      const ix = Math.round((pt.x + width_m  / 2) / width_m  * (cols - 1));
      const iy = Math.round((pt.z + height_m / 2) / height_m * (rows - 1));
      if (ix >= 0 && ix < cols && iy >= 0 && iy < rows) {
        const bedElev = fwcBed[iy * cols + ix];
        if (isFinite(bedElev)) {
          const depth = water_surface - bedElev;
          info += `  ·  Lake depth: ${depth.toFixed(2)} m (FWC survey)`;
        }
      }
    }
    hoverInfo.textContent = info;
  });

  // ── Resize ───────────────────────────────────────────────────
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // ── Hide loading overlay ──────────────────────────────────────
  loadingEl.classList.add('hidden');
  setTimeout(() => { loadingEl.style.display = 'none'; }, 600);

  // ── Render loop ───────────────────────────────────────────────
  const clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    const dt = clock.getDelta();
    controls.update();
    if (simController) simController.tick(dt);
    renderer.render(scene, camera);
  }
  animate();
}

init().catch(err => {
  console.error(err);
  loadingTxt.textContent = `Error: ${err.message}`;
  loadingTxt.style.color = '#e05050';
});
