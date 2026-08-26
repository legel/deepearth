import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createTerrain, createFixedExagGeometry, VERT_EXAG } from './terrain.js';
import { createDrapedOverlay } from './overlays.js';
import { setupLayerPanel } from './layerControls.js';
import { createFloodLayer } from './floodLayer.js';
import { createRainSystem } from './rainParticles.js';
import { setupIanSimControls } from './ianSimControls.js';
import { createLidarBridges } from './lidarBridges.js';
import { createLidarPointCloud } from './lidarPointCloud.js';
import { createMeshShallowWater } from './meshShallowWater.js';

// Expose THREE utilities for layerControls.js (texture loading in PS slots)
window.__THREE_TEXTURE_LOADER__ = THREE.TextureLoader;
window.__THREE_SRGB__ = THREE.SRGBColorSpace;

const loadingEl  = document.getElementById('loading-overlay');
const loadingTxt = document.getElementById('loading-text');
const hoverInfo  = document.getElementById('hover-info');

function setStatus(msg) { loadingTxt.textContent = msg; }

// ── Flat overlay plane ────────────────────────────────────────────────────────
function flatPlane(textureUrl, yOffset, opacity, w, h) {
  const loader = new THREE.TextureLoader();
  const tex = loader.load(textureUrl);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.MeshBasicMaterial({
    map: tex, transparent: true, opacity: opacity ?? 0.85,
    depthWrite: false, side: THREE.DoubleSide,
  });
  const geo = new THREE.PlaneGeometry(w, h);
  geo.rotateX(-Math.PI / 2);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.y = yOffset;
  return mesh;
}

// ── Empty flat slot for PlanetScope on-demand texture ─────────────────────────
function psSlotMesh(w, h, yOffset) {
  const mat = new THREE.MeshBasicMaterial({
    transparent: true, opacity: 0.95, depthWrite: false, side: THREE.DoubleSide,
  });
  const geo = new THREE.PlaneGeometry(w, h);
  geo.rotateX(-Math.PI / 2);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.y = yOffset;
  mesh.visible = false;
  return mesh;
}

// ── Draped slot — on-demand texture mapped onto terrain surface ───────────────
function psDrapedSlot(terrainGeometry) {
  const mat = new THREE.MeshBasicMaterial({
    transparent: true, opacity: 0.95, depthWrite: false,
    side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
  });
  const mesh = new THREE.Mesh(terrainGeometry, mat);
  mesh.visible = false;
  return mesh;
}

async function init() {
  // ── Renderer ──────────────────────────────────────────────────────────────
  const canvas = document.getElementById('canvas');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = false;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  // ── Scene ─────────────────────────────────────────────────────────────────
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x080c14);
  scene.fog = new THREE.FogExp2(0x080c14, 0.00012);

  // ── Camera ────────────────────────────────────────────────────────────────
  const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 1, 25000);

  // ── Controls ──────────────────────────────────────────────────────────────
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.screenSpacePanning = false;
  controls.minDistance = 50;
  controls.maxDistance = 10000;
  controls.maxPolarAngle = Math.PI / 2.05;

  // ── Lights ────────────────────────────────────────────────────────────────
  scene.add(new THREE.AmbientLight(0x6080a0, 1.8));
  const sun = new THREE.DirectionalLight(0xfff0d8, 2.5);
  sun.position.set(600, 1200, 400);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0x4060c0, 0.6);
  fill.position.set(-400, 300, -600);
  scene.add(fill);

  // ── Geo metadata ──────────────────────────────────────────────────────────
  setStatus('Loading terrain data…');
  const geoMeta = await fetch('/data/geo_meta.json').then(r => r.json());
  const { width_m, height_m, z_min, rows, cols } = geoMeta;
  const W = width_m, H = height_m;

  // ── Terrain ───────────────────────────────────────────────────────────────
  setStatus('Building DEM terrain…');
  const terrain = await createTerrain(geoMeta);
  scene.add(terrain.mesh);

  // ── Raw LiDAR bridge-crossing meshes ──────────────────────────────────────
  // Corrects the bare-earth DEM's ~7.5-8.4m dip at the two SR417 grade-separated
  // crossings (see lidar/data/BRIDGE_VALIDATION.md) with the actual point-cloud
  // surface, bridge deck included.
  setStatus('Loading LiDAR bridge meshes…');
  const lidarBridges = await createLidarBridges();
  scene.add(lidarBridges.mesh);

  // ── Full raw LiDAR point cloud (whole AOI, not just the 2 bridges) ───────
  setStatus('Loading full LiDAR point cloud…');
  const lidarCloud = await createLidarPointCloud();
  scene.add(lidarCloud.mesh);

  // ── Overlays ──────────────────────────────────────────────────────────────
  setStatus('Loading overlays…');

  // BASE LAYERS  (renderOrder 0 — default)
  // NAIP/SSURGO draped meshes use their own FIXED 2x geometry (terrain.createFixedExagGeometry),
  // not terrain.geometry — draping imagery onto the wireframe's exaggeration (default 4x, up to
  // 8x) reads as more visually distorted than the wireframe alone, so these sit at a gentler
  // fixed scale independent of the live slider, same "own fixed constant" pattern already used
  // by rainParticles.js/floodLayer.js.
  const drapedGeo2x  = await createFixedExagGeometry(geoMeta, 2);
  const naip         = flatPlane('/data/naip_rgb.png',      1.5, 0.90, W, H);
  const naipDraped   = createDrapedOverlay(drapedGeo2x, '/data/naip_rgb.png', 0.90);
  const ssurgoFlat   = flatPlane('/data/ssurgo.png',        2, 0.75, W, H);
  const ssurgoDraped = createDrapedOverlay(drapedGeo2x, '/data/ssurgo.png', 0.75);
  const hydrography  = flatPlane('/data/hydrography.png',   3, 0.85, W, H);
  const fema         = flatPlane('/data/floodplain.png',    4, 0.70, W, H);
  const roadsBuildings = flatPlane('/data/roads_buildings.png', 3.5, 0.85, W, H);

  // PLANETSCOPE slots — renderOrder -1 (bottom layer, behind everything)
  // yOffset=1: sits just above terrain base (y=0), so wireframe/surface at
  // higher elevation always pass the depth test and render in front of PS.
  const psSlotA        = psSlotMesh(W, H, 1);
  const psSlotB        = psSlotMesh(W, H, 1.5);
  const psSlotA_draped = psDrapedSlot(terrain.geometry);
  const psSlotB_draped = psDrapedSlot(terrain.geometry);
  psSlotA.renderOrder        = -1;
  psSlotB.renderOrder        = -1;
  psSlotA_draped.renderOrder = -1;
  psSlotB_draped.renderOrder = -1;

  // CFX CORRIDOR BOUNDARY — renderOrder 2 (always on top of every layer)
  const boundary = flatPlane('/data/boundary.png', 9, 0.95, W, H);
  boundary.renderOrder = 2;

  // RISK ANALYSIS
  const femaHand = flatPlane('/data/fema_hand_risk.png', 4.5, 0.80, W, H);
  // Real ground-truth cross-check for the Ian simulation (analysis/fema_sim_extent_crossref.py,
  // 2026-07-27): amber/red = FEMA SFHA/floodway (AOI-clipped), cyan = simulated Ian flood with
  // no FEMA overlap, green = simulated AND FEMA agree. Sidesteps the watershed-area-mismatch
  // problem that makes the Shingle Creek gauge comparison invalid — both are spatial extents at
  // the SAME AOI. Real finding: only ~11% of the simulated extent falls inside a mapped SFHA
  // (~1% of the SFHA area itself) — see CLAUDE.md for the honest interpretation.
  const femaSimExtent = flatPlane('/data/fema_sim_extent_overlay.png', 4.7, 0.85, W, H);

  // HYDROLOGY
  const handMesh  = flatPlane('/data/hydro_hand.png',       6, 0.80, W, H);
  const flowAccum = flatPlane('/data/hydro_flow_accum.png', 6, 0.75, W, H);
  const streams   = flatPlane('/data/hydro_streams.png',    7, 0.90, W, H);

  // TOPOGRAPHY ANALYSIS
  const elevationMesh = flatPlane('/data/terrain_elevation.png', 6, 0.80, W, H);
  const slopeMesh = flatPlane('/data/terrain_slope.png',     6, 0.75, W, H);
  const hillshade = flatPlane('/data/terrain_hillshade.png', 6, 0.70, W, H);
  const tpi       = flatPlane('/data/terrain_tpi.png',       6, 0.75, W, H);
  const curvature = flatPlane('/data/terrain_curvature.png', 6, 0.75, W, H);
  const tri       = flatPlane('/data/terrain_tri.png',       6, 0.75, W, H);

  // Add all to scene
  [
    naip, naipDraped, ssurgoFlat, ssurgoDraped, hydrography, fema, roadsBuildings,
    psSlotA, psSlotB, psSlotA_draped, psSlotB_draped,
    boundary,
    femaHand, femaSimExtent,
    handMesh, flowAccum, streams,
    elevationMesh, slopeMesh, hillshade, tpi, curvature, tri,
  ].forEach(m => scene.add(m));

  // ── Animated flood layer (Ian) ────────────────────────────────────────────
  setStatus('Setting up Ian simulation…');
  const floodLayerObj = createFloodLayer(scene, geoMeta, {
    urlSuffix:       'frames',
    meshName:        'Ian Flood Animation',
    yOffset:         5.5,
    terrainGeometry: terrain.geometry,
  });

  // ── Rain particles ────────────────────────────────────────────────────────
  const rainSys = createRainSystem(scene, geoMeta);

  // ── Layer panel ───────────────────────────────────────────────────────────
  // Defaults (2026-07-27, per request): only NAIP Aerial Draped visible on load — Wireframe
  // (the green terrain grid) previously also defaulted on but was turned off per direct
  // request, since it visually competes with the Ian flood animation / other overlays when
  // running a simulation. Everything else, including PlanetScope Scene A's default scene (see
  // layerControls.js's _planetScopeSlot call), starts off too.
  setupLayerPanel({
    terrain: {
      solidMesh: terrain.solidMesh,
      wireMesh:  terrain.wireMesh,       // defaultOn: false (set in layerControls.js)
      analysisMeshes: {
        'Elevation':         elevationMesh,
        'Hillshade':         hillshade,
        'Slope':             slopeMesh,
        'TPI':               tpi,
        'Profile Curvature': curvature,
        'TRI':               tri,
      },
      lidarBridges: lidarBridges.mesh,
      lidarCloud:   lidarCloud.mesh,
    },
    hydrology: [
      { name: 'HAND',              mesh: handMesh,  on: false, swatch: '#2255cc' },
      { name: 'Stream Network',    mesh: streams,   on: false, swatch: '#00ccff' },
      { name: 'Flow Accumulation', mesh: flowAccum, on: false, swatch: '#004499' },
    ],
    base: [
      {
        // On + Draped by default (2026-07-20, per request) — the other page-load default is
        // Wireframe (set in layerControls.js); everything else in this panel defaults off.
        name: 'NAIP Aerial', mesh: naip, on: true, swatch: '#8a8a70',
        drape: true, drapedMesh: naipDraped, drapedOn: true,
      },
      {
        name: 'SSURGO Soils', mesh: ssurgoFlat, on: false, swatch: '#7a9a5a',
        drape: true, drapedMesh: ssurgoDraped, drapedOn: false,
        legendUrl: '/data/ssurgo_legend.json',
      },
      { name: 'Hydrography (3DHP)',    mesh: hydrography, on: false, swatch: '#00aaff' },
      { name: 'FEMA Flood Zones',      mesh: fema,        on: false, swatch: '#c04040' },
      { name: 'Roads & Buildings',     mesh: roadsBuildings, on: false, swatch: '#c99b5f' },
      { name: 'CFX Corridor Boundary', mesh: boundary,    on: false, swatch: '#ffc81e' },
    ],
    risk: [
      { name: 'FEMA × HAND Risk Map', mesh: femaHand,           on: false, swatch: '#e02020' },
      { name: 'Ian Sim vs. FEMA Extent', mesh: femaSimExtent,   on: false, swatch: '#30d090' },
      { name: 'Ian Flood Animation',  mesh: floodLayerObj.mesh, on: false, swatch: '#2060cc' },
    ],
    planetscope: {
      slotA:        psSlotA,
      slotB:        psSlotB,
      slotA_draped: psSlotA_draped,
      slotB_draped: psSlotB_draped,
      scenes:       [],
    },
  });

  // ── Ian simulation controls ───────────────────────────────────────────────
  setStatus('Loading simulation controls…');
  const ianSim = await setupIanSimControls({
    floodLayer:    floodLayerObj,
    rainParticles: rainSys,
    geoMeta,
    contextLayers: {
      wireframe: terrain.wireMesh,
    },
  });

  // ── Vertical exaggeration slider ──────────────────────────────────────────
  // Terrain recomputes its own vertex heights from stored raw elevation; the LiDAR bridge
  // meshes and point cloud have VERT_EXAG pre-baked into their exported coordinates (see
  // lidar/build_lidar_pointcloud.py / export_full_pointcloud.py), so they're rescaled in
  // place by the ratio of new/current exaggeration instead. Rain particles and the flood
  // depth-texture layer use their own fixed local exaggeration constants and are NOT
  // affected by this slider — a deliberate scope limit, not an oversight.
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

  // ── Rescalable-mesh registry ──────────────────────────────────────────────
  // Lazy-loaded LiDAR/droplet/shallow-water layers have VERT_EXAG pre-baked into their
  // exported coordinates, so moving the exaggeration slider AFTER one is already loaded
  // needs to rescale it in place too — this registry is what the slider handler (below)
  // iterates over. lidarBridges/lidarCloud load at page-init, so they register immediately;
  // every lazy-loaded layer below registers itself the moment it finishes loading.
  const rescalableMeshes = [lidarBridges.mesh, lidarCloud.mesh];

  // ── Heavy LiDAR layers — lazy-loaded on first toggle, not at page init ────
  // The full-res cloud is ~1GB; fetching it on every page load (even with the layer
  // hidden) would make every refresh slow during a live demo. Loaded once, on demand.
  function wireLazyPointCloud(checkboxId, url, name, size) {
    const cb = document.getElementById(checkboxId);
    if (!cb) return;
    let loaded = null;
    cb.addEventListener('change', async () => {
      if (!cb.checked) {
        if (loaded) loaded.mesh.visible = false;
        return;
      }
      if (loaded) { loaded.mesh.visible = true; return; }
      cb.disabled = true;
      const label = cb.parentElement.querySelector('.layer-label');
      const prevText = label.textContent;
      label.textContent = prevText + ' — loading…';
      try {
        loaded = await createLidarPointCloud(url, { name, size });
        rescaleY(loaded.mesh, currentExag / 8);   // match whatever exaggeration is active now
        scene.add(loaded.mesh);
        rescalableMeshes.push(loaded.mesh);
      } finally {
        label.textContent = prevText;
        cb.disabled = false;
      }
    });
  }
  wireLazyPointCloud('lidar-full-cb', '/data/lidar_pointcloud_full.bin',
                      'LiDAR Point Cloud (full, 70.9M)', 1.4);

  // ── Per-site fine-resolution water-flow layers ────────────────────────────
  // Two test sites (lidar/test_sites.py): site1 is a pure slope+roofs residential cluster;
  // site2 adds a real retention pond so water accumulation ("lake level rise") can be
  // checked too. Each site gets 2 lazy-loaded layers: a dense NAIP-true-color point cloud
  // (what the raw data actually looks like) and the shallow-water sim, which now ALSO
  // includes the physics-driven flow tracers internally (meshShallowWater.js loads
  // dropletFlow.js itself for that piece) — the older standalone fixed-step droplet layer
  // was removed from this panel 2026-07-21, fully superseded by the real-physics tracers.
  const SITE_POINT_CLOUD_URL = {
    site1: '/data/lidar_pointcloud_5houses.bin',
    site2: '/data/lidar_pointcloud_site2.bin',
  };

  // This is a real depth-field solver (simulation/mesh_shallow_water.py) with falling rain
  // that lands on the actual roof/ground surface height, plus physics-driven flow tracers
  // (real velocity field, not a fixed step) folded in as of 2026-07-21. Its own
  // setExag(newExag) — NOT the generic rescaleY() every other lazy layer uses — is what keeps
  // it correctly scaled: meshShallowWater.js rebuilds vertex Y from source arrays every time
  // the visible frame changes, so a one-shot mesh transform (what rescaleY does) gets
  // silently undone the moment playback advances to the next frame — confirmed 2026-07-21 as
  // the cause of the film "floating at a different exaggeration." setExag() is called once at
  // load (to match whatever the slider is already at — the default is no longer 8x) and again
  // on every subsequent slider move; it also rescales the tracer sub-layer and the falling
  // rain's landing height, both of which previously stayed pinned to the 8x-baked convention
  // regardless of the slider (a documented scope limit no longer necessary now that
  // meshShallowWater.js tracks exaggeration itself).
  function wireMeshSWESite(siteKey) {
    const cb = document.getElementById(`mesh-swe-${siteKey}-cb`);
    if (!cb) return { tick() {}, setExag() {} };
    const speedSlider = document.getElementById(`mesh-swe-speed-${siteKey}-slider`);
    const presetButtons = document.querySelectorAll(`.exag-preset-row[data-site="${siteKey}"] .rain-preset-btn`);
    const suffix = siteKey === 'site1' ? '' : `_${siteKey}`;
    let level = 'medium';   // matches the default-active button in index.html
    function urlsForLevel(lvl) {
      return {
        mesh: `/data/dense_test_area_mesh${suffix}.obj`,
        frames: `/data/swe_mesh_frames${suffix}_${lvl}.bin`,
        heightmap: `/data/swe_surface_heightmap${suffix}_${lvl}.bin`,
        summary: `/data/swe_mesh_summary${suffix}_${lvl}.json`,
        tracers: `/data/flow_tracer_paths${suffix}_${lvl}.bin`,
      };
    }
    let instance = null;

    async function loadInstance() {
      instance = await createMeshShallowWater(urlsForLevel(level));
      instance.setExag(currentExag);
      scene.add(instance.group);
      instance.setSpeed(parseFloat(speedSlider.value));
    }

    cb.addEventListener('change', async () => {
      if (!cb.checked) { if (instance) instance.group.visible = false; return; }
      if (instance) { instance.group.visible = true; return; }
      cb.disabled = true;
      const label = cb.parentElement.querySelector('.layer-label');
      const prevText = label.textContent;
      label.textContent = prevText + ' — loading…';
      try {
        await loadInstance();
      } finally {
        label.textContent = prevText;
        cb.disabled = false;
      }
    });
    speedSlider?.addEventListener('input', () => {
      document.getElementById(`mesh-swe-speed-${siteKey}-value`).textContent = parseFloat(speedSlider.value).toFixed(1) + '×';
      instance?.setSpeed(parseFloat(speedSlider.value));
    });

    // ── Pre-computed rain-intensity presets ───────────────────────────────────────────────
    // Low/Medium/High are real, physically distinct solver runs done ahead of time (see
    // simulation/mesh_shallow_water.py --peak-rain-mm-hr 40/100/180, one run per level per
    // site) — NOT a live re-simulation. Triggering the solver on click was tried and rejected:
    // it blocks the UI for ~4-13 min. Switching levels here just disposes the current instance
    // and fetches the already-computed files for the new level — a fetch of a precomputed
    // binary over localhost, not a physics run.
    presetButtons.forEach(btn => {
      btn.addEventListener('click', async () => {
        const newLevel = btn.dataset.level;
        if (newLevel === level && instance) return;
        presetButtons.forEach(b => b.classList.toggle('active', b === btn));
        level = newLevel;
        if (!instance) return;   // layer not loaded yet — next checkbox-on load uses `level`
        const wasVisible = instance.group.visible;
        const row = btn.closest('.exag-preset-row');
        row.querySelectorAll('button').forEach(b => b.disabled = true);
        try {
          instance.dispose();
          instance = null;
          await loadInstance();
          instance.group.visible = wasVisible;
        } finally {
          row.querySelectorAll('button').forEach(b => b.disabled = false);
        }
      });
    });

    return { tick: dt => instance?.tick(dt), setExag: newExag => instance?.setExag(newExag) };
  }

  // Dense test-area point clouds get a bigger point size — at ~1-10 pts/m² per rooftop,
  // small points read as scattered dots rather than a surface; a larger size helps
  // individual roofs read as a filled shape instead of sparse noise.
  wireLazyPointCloud('dense-cloud-site1-cb', SITE_POINT_CLOUD_URL.site1,
                      'LiDAR Point Cloud (Site 1, NAIP)', 3.5);
  wireLazyPointCloud('dense-cloud-site2-cb', SITE_POINT_CLOUD_URL.site2,
                      'LiDAR Point Cloud (Site 2, NAIP)', 3.5);
  const sweSite1 = wireMeshSWESite('site1');
  const sweSite2 = wireMeshSWESite('site2');

  // ── Test-site shortcut toggles for the Hydrography (3DHP) layer (2026-07-27) ─────────────
  // site1/site2 have no separate DEM-plane terrain the way the main AOI / site3 do (they only
  // have their own dense LiDAR mesh, which — unlike terrain.geometry — has no clean world-XY
  // UV mapping to drape a flat raster onto), so there's no way to build a genuinely separate,
  // tightly-cropped 3DHP overlay for each test site the way site3's page got one. Both test
  // sites sit fully inside the main AOI's own 2x2km box, so the SAME already-loaded
  // `hydrography` flatPlane (built above, HYDROLOGY section) already covers their locations —
  // this just adds a same-effect shortcut checkbox inside each Test Site section so it can be
  // toggled without scrolling up, and keeps all 3 checkboxes (this pair + the original HYDROLOGY
  // one) in sync with each other. Falls back to controlling only its own checkbox if the
  // original HYDROLOGY checkbox can't be found by label text (fails safe, doesn't break either
  // way).
  function findHydrographyMainCheckbox() {
    for (const row of document.querySelectorAll('.layer-row')) {
      const label = row.querySelector('.layer-label');
      if (label && label.textContent.trim() === 'Hydrography (3DHP)') {
        return row.querySelector('input[type=checkbox]');
      }
    }
    return null;
  }
  const hydroMainCb = findHydrographyMainCheckbox();
  const hydroShortcutCbs = [document.getElementById('hydro-site1-cb'), document.getElementById('hydro-site2-cb')]
    .filter(Boolean);
  const allHydroCbs = hydroMainCb ? [hydroMainCb, ...hydroShortcutCbs] : hydroShortcutCbs;
  allHydroCbs.forEach(cb => {
    cb.checked = hydrography.visible;
    cb.addEventListener('change', () => {
      hydrography.visible = cb.checked;
      allHydroCbs.forEach(other => { if (other !== cb) other.checked = cb.checked; });
    });
  });

  let currentExag = 2;   // must match terrain.js's own default VERT_EXAG and
                          // layerControls.js's _vertExagRow slider default — see terrain.js
                          // for why this was lowered from 8 (2026-07-20 default at page load).
  const exagSlider = document.getElementById('vert-exag-slider');
  const exagValue  = document.getElementById('vert-exag-value');
  if (exagSlider) {
    exagSlider.addEventListener('input', () => {
      const newExag = parseFloat(exagSlider.value);
      exagValue.textContent = newExag.toFixed(1) + '×';
      terrain.updateExag(newExag);
      const ratio = newExag / currentExag;
      rescalableMeshes.forEach(m => rescaleY(m, ratio));
      sweSite1.setExag(newExag);
      sweSite2.setExag(newExag);
      currentExag = newExag;
    });
  }

  // ── Camera ────────────────────────────────────────────────────────────────
  const dist = Math.max(W, H) * 0.65;
  camera.position.set(dist * 0.35, dist * 0.55, dist * 0.95);
  controls.target.set(0, 0, 0);
  controls.update();

  // ── Hover elevation + flood depth ─────────────────────────────────────────
  const raycaster = new THREE.Raycaster();
  const pointer   = new THREE.Vector2();
  const terrainMeshes = terrain.mesh.children.filter(c => c instanceof THREE.Mesh);

  window.addEventListener('mousemove', e => {
    pointer.x =  (e.clientX / window.innerWidth)  * 2 - 1;
    pointer.y = -(e.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(terrainMeshes);
    if (hits.length === 0) { hoverInfo.textContent = 'Hover for elevation'; return; }
    const elev = hits[0].point.y / VERT_EXAG + z_min;
    const uv   = hits[0].uv;
    let txt = `Elev: ${elev.toFixed(1)} m NAVD88`;
    if (uv && floodLayerObj.mesh.visible) {
      const depth = floodLayerObj.getDepthAt(uv.x, uv.y);
      if (depth > 0.005) txt += `  |  Depth: ${depth.toFixed(3)} m`;
    }
    hoverInfo.textContent = txt;
  });

  // ── Resize ────────────────────────────────────────────────────────────────
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // ── Done loading ──────────────────────────────────────────────────────────
  loadingEl.classList.add('hidden');
  setTimeout(() => { loadingEl.style.display = 'none'; }, 600);

  // ── Render loop ───────────────────────────────────────────────────────────
  const clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    const dt = clock.getDelta();
    controls.update();
    if (ianSim) ianSim.tick(dt);
    sweSite1.tick(dt);
    sweSite2.tick(dt);
    renderer.render(scene, camera);
  }
  animate();
}

init().catch(err => {
  console.error(err);
  loadingTxt.textContent = `Error: ${err.message}`;
  loadingTxt.style.color = '#e05050';
});
