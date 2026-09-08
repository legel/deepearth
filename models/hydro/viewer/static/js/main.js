import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createTerrain, createDrape } from './terrain.js';
import { createFlood } from './flood.js';
import { createChart } from './chart.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f14);
scene.fog = new THREE.Fog(0x0b0f14, 6000, 20000);

const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 1, 60000);
const renderer = new THREE.WebGLRenderer({
  canvas: document.getElementById('scene'), antialias: true,
});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0x9fc4ff, 0x2a3038, 1.5));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(-1, 2, 1);
scene.add(sun);

const { meta, layers } = await fetch('/api/layers').then(r => r.json());

/* Default exaggeration is derived, not fixed. These domains are extraordinarily flat -- site3
   has 25 m of relief over 6.9 km, a 1:270 ratio -- so a constant 2x that suits a 2 km box
   renders a featureless plane here. Target relief at ~8 % of domain width, which reads as
   terrain without turning a road embankment into a mountain. */
const relief = Math.max(meta.z_max - meta.z_min, 0.1);
const defaultExag = Math.max(1, Math.min(40, Math.round(0.08 * meta.width_m / relief)));
const state = { exag: defaultExag, flood: null, chart: null, playing: false, frame: 0 };
document.getElementById('subtitle').textContent =
  `${(meta.width_m / 1000).toFixed(1)} x ${(meta.height_m / 1000).toFixed(1)} km` +
  (meta.gauge ? ` - USGS ${meta.gauge.site_no}` : '');

const terrain = await createTerrain(meta, state.exag);
scene.add(terrain.surface, terrain.wire);

const span = Math.max(meta.width_m, meta.height_m);
camera.position.set(span * 0.85, span * 0.60, span * 1.05);
controls.target.set(0, relief * state.exag * 0.4, 0);
controls.update();
document.getElementById('loading').remove();

/* Layer panel. A layer whose file is absent is shown disabled rather than hidden, so the page
   says what this site does not have instead of quietly differing from another site. */
const panel = document.getElementById('layers');
const meshes = {};

function addToggle(label, enabled, onToggle, checked = false) {
  const el = document.createElement('label');
  el.className = 'layer' + (enabled ? '' : ' disabled');
  el.innerHTML = `<input type="checkbox" ${checked ? 'checked' : ''} ${enabled ? '' : 'disabled'}>
                  <span>${label}</span>`;
  el.querySelector('input').addEventListener('change', e => onToggle(e.target.checked));
  panel.appendChild(el);
}

addToggle('Terrain wireframe', true, v => { terrain.wire.visible = v; }, true);
addToggle('Terrain surface', true, v => { terrain.surface.visible = v; }, true);

for (const layer of layers) {
  if (layer.kind === 'drape') {
    addToggle(layer.label, layer.available, async v => {
      if (!meshes[layer.id]) {
        meshes[layer.id] = createDrape(terrain.geometry, `/data/${layer.file}`);
        scene.add(meshes[layer.id]);
      }
      meshes[layer.id].visible = v;
    });
  } else if (layer.kind === 'flood') {
    addToggle(layer.label, layer.available, async v => {
      if (!state.flood) {
        state.flood = await createFlood(terrain.geometry, `/data/${layer.file}`);
        scene.add(state.flood.mesh);
        const slider = document.getElementById('frame');
        slider.max = state.flood.frameCount - 1;
        document.getElementById('playback-title').textContent =
          meta.storm ? meta.storm.label : 'Flood animation';
        document.getElementById('playback').hidden = false;
        slider.addEventListener('input', () => setFrame(+slider.value));
        await loadChart();
      }
      state.flood.mesh.visible = v;
    });
  }
}

function setFrame(i) {
  state.frame = i;
  const hours = state.flood.setFrame(i);
  document.getElementById('frame').value = i;
  document.getElementById('clock').textContent = `t = ${hours.toFixed(1)} h`;
  if (state.chart) {
    state.chart.draw(hours);
    const d = state.chart.data;
    const k = d.time_h.findIndex(t => t >= hours);
    if (k >= 0) {
      document.getElementById('readout').textContent =
        `${d.rain_mm_hr[k].toFixed(1)} mm/hr rain - ${d.flooded_ha[k].toFixed(0)} ha flooded ` +
        `- ${d.sim_cfs[k].toFixed(0)} cfs out`;
    }
  }
}

/* The caption states what the two series actually measure, from the payload's own numbers.
   It read "46.6 km2 / 33 km2" as literal HTML, which is true for site3 under Ian and silently
   wrong for any other site or storm the same page is asked to serve. */
function chartCaption() {
  const domain = meta.domain_km2 ? `${meta.domain_km2.toFixed(1)}\u00a0km\u00b2` : 'the domain';
  const gauge = meta.gauge
    ? `the gauge measures one channel draining ${meta.gauge.documented_area_km2}\u00a0km\u00b2, so`
    : 'there is no gauge here, so';
  return `Log axis. Simulated outflow crosses all four edges of ${domain}; ${gauge} read timing `
       + 'and shape, not relative height. The gauge samples at 15\u00a0min, so no timing '
       + 'difference below 0.25\u00a0h is measurable.';
}

async function loadChart() {
  if (!meta.storm) return;
  const res = await fetch(`/data/hydrograph_${meta.storm.name}.json`);
  if (!res.ok) return;
  const data = await res.json();
  const canvas = document.getElementById('chart');
  document.getElementById('chart-caption').textContent = chartCaption();
  document.getElementById('chart-section').hidden = false;
  state.chart = createChart(canvas, data);
  state.chart.data = data;
  state.chart.draw(0);
}

document.getElementById('play').addEventListener('click', e => {
  state.playing = !state.playing;
  e.target.textContent = state.playing ? 'Pause' : 'Play';
});

const exagInput = document.getElementById('exag');
exagInput.value = state.exag;
document.getElementById('exag-value').textContent = `${state.exag}x`;
exagInput.addEventListener('input', () => {
  state.exag = +exagInput.value;
  document.getElementById('exag-value').textContent = `${state.exag}x`;
  terrain.setExag(state.exag);
});

/* Frame advance is on its own timer, not the render loop. Tying it to requestAnimationFrame
   means the clock slows down exactly when the scene gets heavy, so the readout stops matching
   the animation. */
setInterval(() => {
  if (state.playing && state.flood) {
    setFrame((state.frame + 1) % state.flood.frameCount);
  }
}, 400);

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  if (state.chart) state.chart.draw(state.flood ? state.flood.timesHours[state.frame] : 0);
});

(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();
