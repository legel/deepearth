/**
 * simulationControls.js — Simulation panel, playback engine, lake volume HUD
 *
 * Injects into #simulation-panel (added to index.html).
 *
 * Wires together:
 *   - Scenario selector (from /api/scenarios)
 *   - ▶ / ⏸ / ⏮ playback buttons
 *   - Time scrubber (range input)
 *   - Lake volume HUD (updates per frame)
 *   - Rain intensity gauge
 *   - Scientific attribution line (per DeepEarth brand)
 *
 * Usage:
 *   const sim = await setupSimulationControls({
 *     floodLayer,       // floodLayer.js instance
 *     rainParticles,    // rainParticles.js instance
 *     waterPlane,       // THREE.Mesh for the static lake surface
 *     geoMeta,          // from geo_meta.json
 *   });
 *   // Call sim.tick(dt) in the THREE.js render loop.
 */

const FPS_PLAYBACK  = 5;    // frames per second in animation playback (fixed data cadence)
const FRAME_MS      = 1000 / FPS_PLAYBACK;
const WATER_SURFACE = 28.74;   // m NAVD88 (initial; overridden from geoMeta)
const SPEED_OPTIONS = [0.25, 0.5, 1, 2, 4];
const RAIN_PREVIEW_MAX_MMHR = 350;   // matches rainParticles.js MAX_INTENSITY

function fmtTime(minutes) {
  const h = Math.floor(minutes / 60);
  const m = Math.floor(minutes % 60);
  if (h > 0) return `${h}h ${m.toString().padStart(2, '0')}m`;
  return `${m}m`;
}

function fmtVol(m3) {
  if (m3 >= 1e6) return `${(m3 / 1e6).toFixed(3)} × 10⁶ m³`;
  return `${Math.round(m3).toLocaleString()} m³`;
}

export async function setupSimulationControls({ floodLayer, infiltrationLayer, rainParticles, waterPlane, geoMeta }) {
  const initialWSE = geoMeta.water_surface ?? WATER_SURFACE;
  const VERT_EXAG  = 8;
  const zMin       = geoMeta.z_min;

  const panel = document.getElementById('simulation-panel');
  if (!panel) { console.warn('No #simulation-panel element found.'); return null; }

  // ── Load available scenarios ─────────────────────────────────────────────
  let scenarios = [];
  try {
    scenarios = await fetch('/api/scenarios').then(r => r.json());
  } catch (e) {
    panel.innerHTML = '<p class="sim-error">Simulation data not available.<br>'
      + 'Run: python3 simulation/flood_sim.py --scenario all --save-frames</p>';
    return null;
  }

  if (!scenarios.length) {
    panel.innerHTML = '<p class="sim-error">No simulation scenarios found.<br>'
      + 'Run: python3 simulation/flood_sim.py --scenario all --save-frames</p>';
    return null;
  }

  // ── Build panel HTML ─────────────────────────────────────────────────────
  panel.innerHTML = `
    <div class="panel-title">Rainfall Simulation</div>
    <div class="panel-subtitle" id="sim-data-source">NOAA Atlas 14 · Central FL</div>

    <select id="sim-scenario-select" class="sim-dropdown">
      ${scenarios.map(s =>
        `<option value="${s.id}">${s.label}</option>`
      ).join('')}
    </select>

    <div class="sim-stats">
      <span id="sim-stat-rain"></span>
      <span id="sim-stat-flood"></span>
    </div>

    <div class="sim-btn-row">
      <button id="sim-btn-reset"  class="sim-btn" title="Reset">⏮</button>
      <button id="sim-btn-play"   class="sim-btn sim-btn-primary" title="Play">▶</button>
      <button id="sim-btn-pause"  class="sim-btn" title="Pause" style="display:none">⏸</button>
    </div>

    <div class="sim-scrubber-wrap">
      <input type="range" id="sim-scrubber" min="0" max="0" value="0" step="1">
      <div class="sim-time-labels">
        <span id="sim-t-current">0m</span>
        <span id="sim-t-total"></span>
      </div>
    </div>

    <div class="sim-speed-row">
      <span class="hud-label">Speed</span>
      <select id="sim-speed-select" class="sim-dropdown sim-speed-dropdown">
        ${SPEED_OPTIONS.map(s =>
          `<option value="${s}" ${s === 1 ? 'selected' : ''}>${s}×</option>`
        ).join('')}
      </select>
    </div>

    <div class="panel-divider"></div>

    <div class="rain-preview">
      <div class="hud-title">Rain Preview</div>
      <div class="rain-preview-row">
        <button id="sim-btn-rain-preview" class="sim-btn">🌧 Show Rain</button>
        <span id="rain-preview-value">${50} mm/hr</span>
      </div>
      <input type="range" id="rain-preview-slider" min="0" max="${RAIN_PREVIEW_MAX_MMHR}" value="50" step="5">
    </div>

    <div class="panel-divider"></div>

    <div class="lake-hud">
      <div class="hud-title">Johns Lake</div>
      <div class="hud-row">
        <span class="hud-label">Surface</span>
        <span id="hud-wse">${initialWSE.toFixed(2)} m NAVD88</span>
      </div>
      <div class="hud-row">
        <span class="hud-label">Rise</span>
        <span id="hud-rise">+0.000 m</span>
      </div>
      <div class="hud-row">
        <span class="hud-label">Rain now</span>
        <span id="hud-rain">0 mm/hr</span>
      </div>
      <div class="hud-row">
        <span class="hud-label">Flooded</span>
        <span id="hud-flooded">— ha</span>
      </div>
    </div>

    <div class="attribution">
      Physics · Bates et al. (2010) local inertia SWE ·
      Soil · USDA SSURGO Horton ·
      DEM · USGS 3DEP 3m ·
      Water extent · OmniWaterMask
    </div>
  `;

  // ── Element refs ─────────────────────────────────────────────────────────
  const selectEl   = document.getElementById('sim-scenario-select');
  const playBtn    = document.getElementById('sim-btn-play');
  const pauseBtn   = document.getElementById('sim-btn-pause');
  const resetBtn   = document.getElementById('sim-btn-reset');
  const scrubber   = document.getElementById('sim-scrubber');
  const tCurrent   = document.getElementById('sim-t-current');
  const tTotal     = document.getElementById('sim-t-total');
  const dataSource = document.getElementById('sim-data-source');
  const statRain   = document.getElementById('sim-stat-rain');
  const statFlood  = document.getElementById('sim-stat-flood');
  const hudWse     = document.getElementById('hud-wse');
  const hudRise    = document.getElementById('hud-rise');
  const hudRain    = document.getElementById('hud-rain');
  const hudFlooded = document.getElementById('hud-flooded');
  const speedSelect      = document.getElementById('sim-speed-select');
  const rainPreviewBtn   = document.getElementById('sim-btn-rain-preview');
  const rainPreviewSlider = document.getElementById('rain-preview-slider');
  const rainPreviewValue  = document.getElementById('rain-preview-value');

  // ── Playback state ───────────────────────────────────────────────────────
  let currentFrame  = 0;
  let isPlaying     = false;
  let accumulatedMs = 0;
  let hydrograph    = null;   // { times_min, rain_mm_hr, lake_rise_m, flooded_ha, ... }
  let nFrames       = 0;
  let speedMultiplier = 1;

  // ── Rain preview state (standalone, independent of scenario playback) ────
  let rainPreviewActive    = false;
  let rainPreviewIntensity = 50;   // mm/hr

  // ── Load a scenario ──────────────────────────────────────────────────────
  async function loadScenario(id) {
    const meta = scenarios.find(s => s.id === id);

    // Update attribution
    if (meta) {
      dataSource.textContent = `${meta.data_source} · ${meta.total_rain_mm} mm total`;
      statRain.textContent   = `${meta.total_rain_mm} mm`;
      statFlood.textContent  = `${meta.peak_flooded_ha} ha peak`;
    }

    // Load hydrograph JSON
    try {
      hydrograph = await fetch(`/data/simulation_${id}_hydrograph.json`).then(r => r.json());
    } catch (e) {
      console.error('Could not load hydrograph JSON:', e);
      hydrograph = null;
    }

    // Load frame data into flood layer
    try {
      const info = await floodLayer.loadScenario(id);
      nFrames    = info.nFrames;
      scrubber.max = String(nFrames - 1);
    } catch (e) {
      console.error('Could not load flood frames:', e);
      nFrames = 0;
    }

    // Load frame data into infiltration layer (optional — older exports may lack it)
    if (infiltrationLayer) {
      try {
        await infiltrationLayer.loadScenario(id);
      } catch (e) {
        console.warn('Could not load infiltration frames:', e);
      }
    }

    // Summary stats from scenario summary CSV (use hydrograph fallback)
    const totalMin = hydrograph ? hydrograph.times_min[hydrograph.times_min.length - 1] : 0;
    tTotal.textContent = fmtTime(totalMin);

    goToFrame(0);
    stopPlayback();
  }

  function goToFrame(idx) {
    if (idx < 0 || idx >= nFrames) return;
    currentFrame     = idx;
    scrubber.value   = String(idx);
    floodLayer.setFrame(idx);
    if (infiltrationLayer) infiltrationLayer.setFrame(idx);
    updateHUD(idx);
  }

  function updateHUD(idx) {
    if (!hydrograph || idx >= hydrograph.times_min.length) return;

    const tMin      = hydrograph.times_min[idx];
    const rainNow   = hydrograph.rain_mm_hr[idx] ?? 0;
    const rise      = hydrograph.lake_rise_m[idx] ?? 0;
    const flooded   = hydrograph.flooded_ha[idx]  ?? 0;
    const wse       = initialWSE + rise;

    tCurrent.textContent  = fmtTime(tMin);
    hudWse.textContent    = `${wse.toFixed(3)} m NAVD88`;
    hudRise.textContent   = `${rise >= 0 ? '+' : ''}${rise.toFixed(3)} m`;
    hudRain.textContent   = `${rainNow.toFixed(1)} mm/hr`;
    hudFlooded.textContent = flooded > 0 ? `${flooded.toFixed(1)} ha` : '—';
    hudRise.style.color   = rise > 0.01 ? '#6adaff' : '#4a7a9a';

    // Update rain particles
    rainParticles.update(rainNow, 0);

    // Update lake surface elevation (raise/lower the water plane)
    if (waterPlane) {
      waterPlane.position.y = (wse - zMin) * VERT_EXAG;
    }
  }

  function startPlayback() {
    if (!nFrames) return;
    setRainPreview(false);   // scenario playback drives rain instead
    isPlaying = true;
    accumulatedMs = 0;
    playBtn.style.display  = 'none';
    pauseBtn.style.display = '';
  }

  function stopPlayback() {
    isPlaying = false;
    playBtn.style.display  = '';
    pauseBtn.style.display = 'none';
    rainParticles.update(0, 0);
  }

  function resetPlayback() {
    stopPlayback();
    goToFrame(0);
    floodLayer.reset();
    if (infiltrationLayer) infiltrationLayer.reset();
    if (waterPlane) waterPlane.position.y = (initialWSE - zMin) * VERT_EXAG;
  }

  // ── Rain preview — standalone, independent of scenario playback ─────────
  function setRainPreview(active) {
    rainPreviewActive = active;
    rainPreviewBtn.classList.toggle('sim-btn-primary', active);
    rainPreviewBtn.textContent = active ? '🌧 Hide Rain' : '🌧 Show Rain';
    if (!active) rainParticles.update(0, 0);
  }

  // ── Event listeners ──────────────────────────────────────────────────────
  playBtn.addEventListener('click', () => {
    if (currentFrame >= nFrames - 1) goToFrame(0);
    startPlayback();
  });
  pauseBtn.addEventListener('click', stopPlayback);
  resetBtn.addEventListener('click', resetPlayback);

  scrubber.addEventListener('input', () => {
    stopPlayback();
    goToFrame(parseInt(scrubber.value, 10));
  });

  selectEl.addEventListener('change', async () => {
    await loadScenario(selectEl.value);
  });

  speedSelect.addEventListener('change', () => {
    speedMultiplier = parseFloat(speedSelect.value);
  });

  rainPreviewBtn.addEventListener('click', () => {
    if (!rainPreviewActive) stopPlayback();   // preview and scenario playback don't mix
    setRainPreview(!rainPreviewActive);
  });

  rainPreviewSlider.addEventListener('input', () => {
    rainPreviewIntensity = parseFloat(rainPreviewSlider.value);
    rainPreviewValue.textContent = `${rainPreviewIntensity} mm/hr`;
  });

  // ── tick() — called from main.js render loop ──────────────────────────────
  function tick(dt) {
    if (rainPreviewActive) {
      rainParticles.update(rainPreviewIntensity, dt);
      return;
    }

    if (!isPlaying || !nFrames) return;

    // Update rain particles every frame for smooth animation
    if (hydrograph && currentFrame < hydrograph.rain_mm_hr.length) {
      rainParticles.update(hydrograph.rain_mm_hr[currentFrame], dt);
    }

    const frameMs = FRAME_MS / speedMultiplier;
    accumulatedMs += dt * 1000;
    if (accumulatedMs >= frameMs) {
      accumulatedMs -= frameMs;
      const next = currentFrame + 1;
      if (next >= nFrames) {
        goToFrame(nFrames - 1);
        stopPlayback();
      } else {
        goToFrame(next);
      }
    }
  }

  // ── Initial load ─────────────────────────────────────────────────────────
  await loadScenario(scenarios[0].id);

  return { tick };
}
