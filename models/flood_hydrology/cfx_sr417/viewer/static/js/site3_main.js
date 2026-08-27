/**
 * site3_main.js — Gee Creek gauge-matched validation site (Site3) viewer scene
 *
 * A deliberately lean, standalone scene — site3 is 37km from the main AOI so it cannot share
 * that page's terrain/coordinate space; this reuses terrain.js/floodLayer.js/rainParticles.js
 * (all already parameterized enough to accept a second data source) rather than duplicating
 * the physics/rendering logic, but skips the many base/analysis/hydrology layers the main page
 * has, since site3's own data pipeline doesn't produce most of those yet.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createTerrain, createFixedExagGeometry } from './terrain.js';
import { initStaticSections } from '/shared/panelSections.js';
import { createDrapedOverlay } from './overlays.js';
import { createFloodLayer } from './floodLayer.js';
import { createRainSystem } from './rainParticles.js';
import { createLidarPointCloud } from './lidarPointCloud.js';
import { createMeshShallowWater } from './meshShallowWater.js';

const loadingEl  = document.getElementById('loading-overlay');
const loadingTxt = document.getElementById('loading-text');
const hoverInfo  = document.getElementById('hover-info');
function setStatus(msg) { loadingTxt.textContent = msg; }

async function init() {
  // Upgrade the panel's HTML-authored sections to the same collapsible behaviour
  // the main AOI page uses. Done first so the panel stays usable even if a layer
  // below fails to load.
  const nSections = initStaticSections(document);

  // Apply each layer checkbox's initial checked state to its mesh. Without this the markup's
  // `checked` attribute is cosmetic — the mesh stays at whatever visibility it was created
  // with until the user toggles it.
  function syncInitial(id, mesh) {
    const cb = document.getElementById(id);
    if (cb && mesh) mesh.visible = cb.checked;
  }

  if (nSections !== 6) console.warn(`site3 panel: expected 6 sections, wired ${nSections}`);

  const canvas = document.getElementById('canvas');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x080c14);
  scene.fog = new THREE.FogExp2(0x080c14, 0.00012);

  const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 1, 25000);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.screenSpacePanning = false;
  controls.minDistance = 50;
  controls.maxDistance = 10000;
  controls.maxPolarAngle = Math.PI / 2.05;

  scene.add(new THREE.AmbientLight(0x6080a0, 1.8));
  const sun = new THREE.DirectionalLight(0xfff0d8, 2.5);
  sun.position.set(600, 1200, 400);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0x4060c0, 0.6);
  fill.position.set(-400, 300, -600);
  scene.add(fill);

  setStatus('Loading site3 terrain data…');
  const geoMeta = await fetch('/data/geo_meta_site3.json').then(r => r.json());
  const { width_m, height_m } = geoMeta;

  const dist = Math.max(width_m, height_m) * 0.7;
  camera.position.set(dist * 0.5, dist * 0.55, dist * 0.5);
  controls.target.set(0, 0, 0);
  controls.update();

  // Live vertical-exaggeration slider (added per request — matches the main page's own
  // pattern). Terrain recomputes its own vertex heights via terrain.updateExag(); heavy
  // lazy-loaded layers below (point cloud, dense mesh) were baked at VERT_EXAG=8
  // (build_lidar_pointcloud.py's project-wide convention) and get rescaled in place by the
  // ratio of new/current exaggeration, same "own fixed baked convention, rescale to match" idea
  // main.js already uses for its LiDAR bridge meshes/point cloud.
  let currentExag = 2;
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

  setStatus('Building terrain…');
  const terrain = await createTerrain(geoMeta, '/data/dem_site3.bin');
  scene.add(terrain.mesh);
  terrain.wireMesh.visible = true;
  terrain.solidMesh.visible = false;

  setStatus('Setting up Ian flood layer…');
  const floodLayerObj = createFloodLayer(scene, geoMeta, {
    urlSuffix: 'frames',
    meshName: 'Ian Flood Animation (site3)',
    yOffset: 5.5,
    terrainGeometry: terrain.geometry,
  });
  const rainSys = createRainSystem(scene, geoMeta);

  // NAIP draped directly on the same terrain geometry the flood layer uses (no separate
  // fixed-exag geometry the way the main page's NAIP does — site3 has no exaggeration slider
  // yet, so there's no "gentler fixed scale vs. live slider" distinction to make here).
  const naipMesh = createDrapedOverlay(terrain.geometry, '/data/naip_site3.png', 0.90);
  naipMesh.visible = false;
  scene.add(naipMesh);

  // 3DHP hydrography (Gee/Howell/Soldier Creek flowlines + 167 waterbodies), same draped-
  // overlay mechanism as NAIP above — added 2026-07-27, fetch_3dhp_site3.py +
  // export_hydrography_overlay_site3.py. site3 had no mapped hydrography layer at all before
  // this (unlike the main AOI page, which has had one since the original project).
  const hydroMesh = createDrapedOverlay(terrain.geometry, '/data/hydrography_site3.png', 0.85);
  hydroMesh.visible = false;
  scene.add(hydroMesh);

  // SSURGO soils, FEMA flood zones, roads/buildings — same draped-overlay mechanism as NAIP/
  // hydrography above. All 3 real data sources already existed for site3 (SSURGO/roads/
  // buildings from the original site3 data pipeline; FEMA newly fetched via
  // fetch_fema_site3.py) but had no viewer overlay until now.
  const ssurgoMesh = createDrapedOverlay(terrain.geometry, '/data/ssurgo_site3.png', 0.80);
  ssurgoMesh.visible = false;
  scene.add(ssurgoMesh);

  const femaMesh = createDrapedOverlay(terrain.geometry, '/data/floodplain_site3.png', 0.75);
  femaMesh.visible = false;
  scene.add(femaMesh);

  const roadsBuildingsMesh = createDrapedOverlay(terrain.geometry, '/data/roads_buildings_site3.png', 0.85);
  roadsBuildingsMesh.visible = false;
  scene.add(roadsBuildingsMesh);

  // ── Layer checkboxes ──────────────────────────────────────────────────────
  // Honour the markup's initial checked state before wiring change handlers.
  syncInitial('site3-naip-cb', naipMesh);
  syncInitial('site3-wireframe-cb', terrain.wireMesh || terrain.mesh);

  document.getElementById('site3-wireframe-cb').addEventListener('change', e => {
    terrain.wireMesh.visible = e.target.checked;
  });
  document.getElementById('site3-naip-cb').addEventListener('change', e => {
    naipMesh.visible = e.target.checked;
  });
  document.getElementById('site3-hydro-cb').addEventListener('change', e => {
    hydroMesh.visible = e.target.checked;
  });
  document.getElementById('site3-ssurgo-cb').addEventListener('change', e => {
    ssurgoMesh.visible = e.target.checked;
  });
  document.getElementById('site3-fema-cb').addEventListener('change', e => {
    femaMesh.visible = e.target.checked;
  });
  document.getElementById('site3-roads-cb').addEventListener('change', e => {
    roadsBuildingsMesh.visible = e.target.checked;
  });

  // ── Vertical exaggeration slider ──────────────────────────────────────────
  const exagSlider = document.getElementById('site3-vert-exag-slider');
  const exagValue = document.getElementById('site3-vert-exag-value');
  const exagPresetRow = document.getElementById('site3-exag-preset-row');
  exagSlider.addEventListener('input', () => {
    const newExag = parseFloat(exagSlider.value);
    exagValue.textContent = newExag.toFixed(1) + '×';
    terrain.updateExag(newExag);
    const ratio = newExag / currentExag;
    rescalableMeshes.forEach(m => rescaleY(m, ratio));
    sweHouseInstance?.setExag(newExag);
    currentExag = newExag;
    exagPresetRow.querySelectorAll('.exag-preset-btn').forEach(btn => {
      btn.classList.toggle('active', parseFloat(btn.dataset.val) === newExag);
    });
  });
  exagPresetRow.querySelectorAll('.exag-preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      exagSlider.value = btn.dataset.val;
      exagSlider.dispatchEvent(new Event('input'));
    });
  });

  // ── Dense point cloud, raw LiDAR, NAIP colors (~8M pts) ──────────────────
  // The real equivalent of the main page's "Full-res cloud" / site1/site2's "Dense point
  // cloud, NAIP colors" layers (http://localhost:5051/, lidarPointCloud.js) — individual raw
  // LiDAR returns (ground + vegetation + buildings, not just the ground+roof triangulated
  // SURFACE the mesh layer below shows), colored by sampling NAIP per point. Built by
  // lidar/export_dense_pointcloud_site3.py, same PCLD binary format lidarPointCloud.js already
  // loads — no new rendering code needed, just point at the new file. Same VERT_EXAG=8-baked-
  // vs-this-page's-2x rescale as the dense mesh below.
  const SITE3_POINTCLOUD_BAKED_EXAG = 8;
  let pointCloudObj = null;
  const pointCloudCb = document.getElementById('site3-pointcloud-cb');
  const pointCloudStatus = document.getElementById('site3-pointcloud-status');

  pointCloudCb.addEventListener('change', async e => {
    if (e.target.checked) {
      if (pointCloudObj) { pointCloudObj.mesh.visible = true; return; }
      pointCloudStatus.style.display = 'block';
      pointCloudStatus.textContent = 'Loading ~8M-point cloud…';
      try {
        pointCloudObj = await createLidarPointCloud('/data/lidar_pointcloud_site3.bin', {
          name: 'Dense LiDAR point cloud (site3)', size: 2.0,
        });
        const scale = currentExag / SITE3_POINTCLOUD_BAKED_EXAG;
        const pos = pointCloudObj.mesh.geometry.attributes.position;
        for (let i = 0; i < pos.count; i++) pos.setY(i, pos.getY(i) * scale);
        pos.needsUpdate = true;
        scene.add(pointCloudObj.mesh);
        rescalableMeshes.push(pointCloudObj.mesh);
        pointCloudStatus.style.display = 'none';
      } catch (err) {
        pointCloudStatus.textContent = 'Failed to load: ' + err.message;
        console.error(err);
      }
    } else {
      if (pointCloudObj) pointCloudObj.mesh.visible = false;
    }
  });

  // ── 3D shallow-water sim: falling rain + physics-driven flow (1-house demo) ──────────
  // Reuses meshShallowWater.js completely unchanged, same real solver + tracer physics
  // site1/site2's own demo layers use (simulation/mesh_shallow_water.py, 100mm/hr peak,
  // 4min rain / 8min total) — just a new, much smaller crop within site3 (a single isolated
  // house, lidar/test_sites.py's "site3_1house"), since site3's own registered crops
  // (site3_crop/site3_crop_coarse) were built for GNN training-data volume, not a clean
  // single-building demo. Mass balance closed to -0.00% on this run (see swe_mesh_summary
  // json).
  let sweHouseInstance = null;
  const sweHouseCb = document.getElementById('site3-1house-swe-cb');
  const sweHouseStatus = document.getElementById('site3-1house-swe-status');
  const sweHouseSpeedSlider = document.getElementById('site3-1house-swe-speed-slider');
  const sweHouseSpeedValue = document.getElementById('site3-1house-swe-speed-value');

  sweHouseCb.addEventListener('change', async e => {
    if (e.target.checked) {
      if (sweHouseInstance) { sweHouseInstance.group.visible = true; return; }
      sweHouseStatus.style.display = 'block';
      sweHouseStatus.textContent = 'Loading…';
      try {
        sweHouseInstance = await createMeshShallowWater({
          mesh: '/data/dense_test_area_mesh_site3_1house.obj',
          frames: '/data/swe_mesh_frames_site3_1house.bin',
          heightmap: '/data/swe_surface_heightmap_site3_1house.bin',
          summary: '/data/swe_mesh_summary_site3_1house.json',
          tracers: '/data/flow_tracer_paths_site3_1house.bin',
        });
        sweHouseInstance.setExag(currentExag);
        sweHouseInstance.setSpeed(parseFloat(sweHouseSpeedSlider.value));
        scene.add(sweHouseInstance.group);
        sweHouseStatus.style.display = 'none';
      } catch (err) {
        sweHouseStatus.textContent = 'Failed to load: ' + err.message;
        console.error(err);
      }
    } else {
      if (sweHouseInstance) sweHouseInstance.group.visible = false;
    }
  });
  sweHouseSpeedSlider.addEventListener('input', () => {
    sweHouseSpeedValue.textContent = parseFloat(sweHouseSpeedSlider.value).toFixed(1) + '×';
    sweHouseInstance?.setSpeed(parseFloat(sweHouseSpeedSlider.value));
  });

  // ── Dense point cloud, raw LiDAR, NAIP colors (1-house crop only) ────────────────────
  // Direct follow-up to feedback on the shallow-water demo above: the full-site3 point cloud
  // (~30.5M points decimated across the whole 6x6km box) thins out to almost nothing at any
  // single ~120m spot, so the house wasn't actually visible up close. This is a SEPARATE,
  // dedicated export (lidar/export_dense_pointcloud_site3_1house.py) scoped to just this
  // house's own small bbox — real, un-decimated LiDAR returns (261,237 points, every class:
  // ground/vegetation/building), same PCLD format/renderer (lidarPointCloud.js) the full-site3
  // cloud already uses, just pointed at a different, much denser file. Baked at the same
  // VERT_EXAG=8 convention as every other heavy LiDAR layer on this page.
  const SITE3_1HOUSE_CLOUD_BAKED_EXAG = 8;
  let houseCloudObj = null;
  const houseCloudCb = document.getElementById('site3-1house-cloud-cb');
  const houseCloudStatus = document.getElementById('site3-1house-cloud-status');

  houseCloudCb.addEventListener('change', async e => {
    if (e.target.checked) {
      if (houseCloudObj) { houseCloudObj.mesh.visible = true; return; }
      houseCloudStatus.style.display = 'block';
      houseCloudStatus.textContent = 'Loading…';
      try {
        houseCloudObj = await createLidarPointCloud('/data/lidar_pointcloud_site3_1house.bin', {
          name: 'Dense LiDAR point cloud (site3, 1-house crop)', size: 2.5,
        });
        const scale = currentExag / SITE3_1HOUSE_CLOUD_BAKED_EXAG;
        const pos = houseCloudObj.mesh.geometry.attributes.position;
        for (let i = 0; i < pos.count; i++) pos.setY(i, pos.getY(i) * scale);
        pos.needsUpdate = true;
        scene.add(houseCloudObj.mesh);
        rescalableMeshes.push(houseCloudObj.mesh);
        houseCloudStatus.style.display = 'none';
      } catch (err) {
        houseCloudStatus.textContent = 'Failed to load: ' + err.message;
        console.error(err);
      }
    } else {
      if (houseCloudObj) houseCloudObj.mesh.visible = false;
    }
  });

  const animControls = document.getElementById('site3-anim-controls');
  const floodCb = document.getElementById('site3-flood-cb');
  const rainCb  = document.getElementById('site3-rain-cb');
  const playBtn = document.getElementById('site3-play-btn');
  const scrub   = document.getElementById('site3-scrub');
  const timeLabel = document.getElementById('site3-time-label');

  let hydro = null;   // simulation_ian_site3_hydrograph.json
  let playing = false;
  let currentFrame = 0;
  // Explicit fixed-interval stepping, per direct request 2026-07-27: each exported frame is
  // already 1 real-world hour apart (simulation/run_site3_ian.py --frame-interval 60), and
  // playback should advance exactly 1 frame every 0.5 real seconds (73 frames -> ~36.5s full
  // playback) — replaces the earlier fractional FRAME_ADVANCE_PER_SEC*dt accumulation, which
  // was correct but felt slow (~208s full playback) and, worse, spent a large first chunk of
  // that time showing genuinely-near-zero numbers (real rain/flooded-ha stay low for the first
  // ~20 of Ian's 72 hours — see the solver's own per-step log), reading as "stuck at 0".
  const MS_PER_FRAME = 500;

  async function ensureScenarioLoaded() {
    if (hydro) return;
    setStatus('Loading Ian animation frames…');
    await floodLayerObj.loadScenario('ian_site3');
    hydro = await fetch('/data/simulation_ian_site3_hydrograph.json').then(r => r.json());
    scrub.max = hydro.times_min.length - 1;
    animControls.style.display = 'block';
    setFrame(0);
    setStatus('');
    loadingEl.style.display = 'none';
  }

  function setFrame(idx) {
    currentFrame = Math.max(0, Math.min(idx, (hydro?.times_min.length ?? 1) - 1));
    if (!hydro) return;
    floodLayerObj.setFrame(currentFrame);
    scrub.value = currentFrame;
    const tHr = hydro.times_min[currentFrame] / 60;
    timeLabel.textContent = `t = ${tHr.toFixed(1)}h  (rain ${hydro.rain_mm_hr[currentFrame].toFixed(1)}mm/hr, ` +
      `${hydro.flooded_ha[currentFrame].toFixed(1)}ha flooded)`;
    updateChartCursor(hydro.times_min[currentFrame]);
  }

  floodCb.addEventListener('change', async e => {
    if (e.target.checked) {
      loadingEl.style.display = 'flex';
      await ensureScenarioLoaded();
      floodLayerObj.mesh.visible = true;
    } else {
      floodLayerObj.mesh.visible = false;
      animControls.style.display = 'none';
      playing = false;
      playBtn.textContent = 'Play';
    }
  });

  rainCb.addEventListener('change', e => {
    rainSys.points.visible = e.target.checked;
  });

  // Frame advancement runs on setInterval, NOT inside the requestAnimationFrame render loop —
  // With the heavy 8M-point cloud and the flood animation both on
  // at once, actual render FPS drops enough that rAF itself fires rarely, and gating the frame
  // advance on "time since last rAF tick" throttled WITH it (observed: 1 frame advanced in ~4
  // real seconds instead of the intended 2 frames/sec) — reported as "still very slow, numbers
  // stay 0." setInterval keeps its own independent clock, so frame advancement stays on
  // schedule regardless of how slow rendering itself gets; the visual update still only
  // actually paints at whatever rate the renderer manages, but the frame INDEX and displayed
  // numbers stay correctly on the intended 0.5s/frame schedule either way.
  let playInterval = null;

  function startPlaying() {
    playing = true;
    playBtn.textContent = 'Pause';
    playInterval = setInterval(() => {
      if (!hydro) return;
      const next = currentFrame + 1;
      if (next >= hydro.times_min.length - 1) {
        setFrame(hydro.times_min.length - 1);
        stopPlaying();
      } else {
        setFrame(next);
      }
    }, MS_PER_FRAME);
  }
  function stopPlaying() {
    playing = false;
    playBtn.textContent = 'Play';
    clearInterval(playInterval);
    playInterval = null;
  }

  playBtn.addEventListener('click', () => {
    if (playing) stopPlaying(); else startPlaying();
  });

  scrub.addEventListener('input', () => {
    stopPlaying();
    setFrame(parseInt(scrub.value, 10));
  });

  // ── Hydrograph comparison chart (dependency-free canvas 2D) ──────────────
  const chartCanvas = document.getElementById('hydro-chart');
  const ctx = chartCanvas.getContext('2d');
  let gaugeData = null;
  let cursorTimeMin = 0;

  function drawChart() {
    if (!gaugeData) return;
    const W = chartCanvas.width, H = chartCanvas.height;
    const padL = 42, padR = 10, padT = 10, padB = 22;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    ctx.clearRect(0, 0, W, H);

    const allT = gaugeData.real_time_min.concat(gaugeData.sim_time_min);
    const tMin = Math.min(...allT), tMax = Math.max(...allT);
    const cfsMax = Math.max(gaugeData.summary.real_peak_cfs, gaugeData.summary.sim_peak_cfs) * 1.08;
    const rainMax = Math.max(...gaugeData.sim_rain_mm_hr) * 3 || 1;

    const xOf = t => padL + (t - tMin) / (tMax - tMin) * plotW;
    const yOf = v => padT + plotH - (v / cfsMax) * plotH;
    const yRain = r => padT + (r / rainMax) * (plotH * 0.35);

    // axes
    ctx.strokeStyle = 'rgba(200,210,225,0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH); ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();
    ctx.fillStyle = '#9aa8ba';
    ctx.font = '11px sans-serif';
    ctx.fillText(`${Math.round(cfsMax)} cfs`, 2, padT + 8);
    ctx.fillText('0', 2, padT + plotH);

    // rain (inverted, top strip)
    ctx.strokeStyle = '#2060c0';
    ctx.beginPath();
    gaugeData.sim_time_min.forEach((t, i) => {
      const x = xOf(t), y = padT + yRain(gaugeData.sim_rain_mm_hr[i]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // real gauge
    ctx.strokeStyle = '#e07020';
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    gaugeData.real_time_min.forEach((t, i) => {
      const x = xOf(t), y = yOf(gaugeData.real_discharge_cfs[i]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // simulated
    ctx.strokeStyle = '#20a060';
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    gaugeData.sim_time_min.forEach((t, i) => {
      const x = xOf(t), y = yOf(gaugeData.sim_outflow_cfs[i]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // playback cursor
    if (tMax > tMin) {
      const cx = xOf(cursorTimeMin);
      ctx.strokeStyle = 'rgba(255,255,255,0.6)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, padT); ctx.lineTo(cx, padT + plotH);
      ctx.stroke();
    }
  }

  function updateChartCursor(tMin) {
    cursorTimeMin = tMin;
    drawChart();
  }

  fetch('/data/gauge_comparison_site3.json').then(r => r.json()).then(data => {
    gaugeData = data;
    drawChart();
    const s = data.summary;
    document.getElementById('site3-stats').innerHTML =
      `Real peak: <b>${s.real_peak_cfs.toFixed(0)} cfs</b> @ t=${(s.real_peak_time_min/60).toFixed(1)}h &nbsp;·&nbsp; ` +
      `Sim peak: <b>${s.sim_peak_cfs.toFixed(0)} cfs</b> @ t=${(s.sim_peak_time_min/60).toFixed(1)}h &nbsp;·&nbsp; ` +
      `Peak timing gap: <b>${Math.abs(s.sim_peak_time_min - s.real_peak_time_min)/60 |0}h</b>`;
  }).catch(() => {
    document.getElementById('site3-stats').textContent = 'gauge_comparison_site3.json not found — run viewer/preprocess/export_gauge_comparison_site3.py';
  });

  setStatus('');
  loadingEl.style.display = 'none';

  // ── Hover elevation readout ───────────────────────────────────────────────
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  renderer.domElement.addEventListener('mousemove', e => {
    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hit = raycaster.intersectObject(terrain.solidMesh.visible ? terrain.solidMesh : terrain.wireMesh, false);
    if (hit.length) {
      hoverInfo.textContent = `Elevation: ${(hit[0].point.y).toFixed(1)} (scene units)`;
    }
  });

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  let lastFrame = performance.now();
  function animate() {
    requestAnimationFrame(animate);
    const now = performance.now();
    const dt = (now - lastFrame) / 1000;
    lastFrame = now;

    if (rainSys.points.visible && hydro) {
      rainSys.update(hydro.rain_mm_hr[currentFrame] ?? 0, dt);
    }
    sweHouseInstance?.tick(dt);

    controls.update();
    renderer.render(scene, camera);
  }
  animate();
}

init().catch(err => {
  console.error(err);
  document.getElementById('loading-text').textContent = 'Error: ' + err.message;
});
