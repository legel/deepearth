import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createTerrain, VERT_EXAG } from './terrain.js';
import { createVoxelLayer } from './voxelLayer.js';
import { createOverlays, createDrapedOverlay } from './overlays.js';
import { setupLayerControls } from './layerControls.js';
import { createRainSystem } from './rainParticles.js';
import { createFloodLayer } from './floodLayer.js';
import { setupSimulationControls } from './simulationControls.js';

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

  // Draped variants — same textures painted directly onto the terrain's
  // displaced surface (terrain.geometry) instead of floating as flat planes.
  const drapedNaip = createDrapedOverlay(terrain.geometry, '/data/naip_rgb.png', 0.90);
  drapedNaip.name = 'NAIP Aerial (Draped)';
  drapedNaip.visible = false;
  scene.add(drapedNaip);

  const drapedSsurgo = createDrapedOverlay(terrain.geometry, '/data/ssurgo.png', 0.75);
  drapedSsurgo.name = 'SSURGO Soils (Draped)';
  drapedSsurgo.visible = false;
  scene.add(drapedSsurgo);

  // ── Water surface plane (at lake level) ─────────────────────
  const waterY = (water_surface - z_min) * VERT_EXAG;
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

  // ── Layer controls ───────────────────────────────────────────
  // Grouped so each layer sits next to its draped variant (NAIP next to NAIP
  // Draped, SSURGO next to SSURGO Draped) instead of all flats then all draped.
  const naipOverlay   = overlays.find(o => o.name === 'NAIP Aerial');
  const ssurgoOverlay = overlays.find(o => o.name === 'SSURGO Soils');

  setupLayerControls([
    { name: 'Terrain Surface',        mesh: terrain.solidMesh, defaultOn: true,  swatch: '#1e5a3a' },
    { name: 'Terrain Wireframe',      mesh: terrain.wireMesh,  defaultOn: true,  swatch: '#3aaa60' },
    { name: 'Lake Voxels',            mesh: voxelResult?.mesh, defaultOn: true,  swatch: '#3a6abf' },
    { name: 'Water Surface',          mesh: waterPlane,        defaultOn: true,  swatch: '#1a5aaf' },
    ...(naipOverlay ? [{ name: naipOverlay.name, mesh: naipOverlay.mesh, defaultOn: naipOverlay.defaultOn }] : []),
    { name: 'NAIP Aerial (Draped)',   mesh: drapedNaip,        defaultOn: false, swatch: '#8a6a3a' },
    ...(ssurgoOverlay ? [{ name: ssurgoOverlay.name, mesh: ssurgoOverlay.mesh, defaultOn: ssurgoOverlay.defaultOn }] : []),
    { name: 'SSURGO Soils (Draped)',  mesh: drapedSsurgo,      defaultOn: false, swatch: '#7a9a5a' },
    { name: 'Flood Depth',            mesh: flood.mesh,        defaultOn: false, swatch: '#2a60c0' },
    { name: 'Infiltration',           mesh: infiltration.mesh, defaultOn: false, swatch: '#5a8a3a' },
    ...(s2Plane ? [{ name: s2Plane.name, mesh: s2Plane, defaultOn: false, swatch: '#c04040' }] : []),
  ]);

  // ── Simulation controls (async; non-blocking — panel renders after scene is ready) ──
  setStatus('Loading simulation data…');
  let simController = null;
  setupSimulationControls({
    floodLayer: flood,
    infiltrationLayer: infiltration,
    rainParticles: rain,
    waterPlane,
    geoMeta,
  }).then(ctrl => { simController = ctrl; }).catch(err => {
    console.warn('Simulation controls failed to load:', err);
  });

  // ── Camera: side-oblique view centred on lake so voxel bowl depth is visible ──
  // Low elevation angle (~25°) so the 3D depth of the voxel stack reads clearly.
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
    const elev = pt.y / VERT_EXAG + z_min;
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
