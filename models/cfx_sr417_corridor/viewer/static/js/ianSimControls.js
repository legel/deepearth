/**
 * ianSimControls.js — Hurricane Ian playback panel
 *
 * Two scenarios switchable via infiltration toggle:
 *   ian_noinfil — no Horton infiltration (Pe = rain; water stays on surface) — DEFAULT
 *   ian         — SSURGO Horton AMC-III (fc=23 mm/hr; soil absorbs some rain; less flooding)
 */

const FPS_PLAYBACK   = 5;
const FRAME_MS       = 1000 / FPS_PLAYBACK;
const SPEED_OPTIONS  = [0.25, 0.5, 1, 2, 4, 8];
const MAX_RAIN_MMHR  = 350;
const IAN_BASE_UTC   = new Date('2022-09-28T00:00:00Z');

function minsToDate(minutes) {
  const d   = new Date(IAN_BASE_UTC.getTime() + minutes * 60 * 1000);
  const day = d.getUTCDate();
  const h   = String(d.getUTCHours()).padStart(2, '0');
  const m   = String(d.getUTCMinutes()).padStart(2, '0');
  return `Sep ${day} · ${h}:${m} UTC`;
}
function fmtMin(minutes) {
  const h = Math.floor(minutes / 60);
  const m = Math.floor(minutes % 60);
  return h > 0 ? `${h}h ${String(m).padStart(2,'0')}m` : `${m}m`;
}

export async function setupIanSimControls({ floodLayer, rainParticles, geoMeta, contextLayers = {} }) {
  const panel = document.getElementById('simulation-panel');
  if (!panel) { console.warn('No #simulation-panel'); return null; }

  // ── State ─────────────────────────────────────────────────────────────────
  let hydrograph    = null;
  let nFrames       = 0;
  let currentFrame  = 0;
  let isPlaying     = false;
  let accumMs       = 0;
  let speedMult     = 1;
  // Default: no-infiltration (water stays on surface — more dramatic, clearer to see)
  let useInfiltration = false;

  function scenarioId()       { return useInfiltration ? 'ian'         : 'ian_noinfil'; }
  function hydrographUrl()    { return useInfiltration
    ? '/data/simulation_ian_hydrograph.json'
    : '/data/simulation_ian_noinfil_hydrograph.json'; }

  // ── Build panel HTML ──────────────────────────────────────────────────────
  panel.innerHTML = `
    <div class="panel-title">Hurricane Ian Simulation</div>
    <div class="panel-subtitle">Sep 28–30, 2022 · ASOS MCO · 336 mm</div>

    <div class="sim-stats">
      <span id="sim-stat-peak" style="color:#6090c0">—</span>
    </div>

    <div class="sim-btn-row">
      <button id="sim-btn-reset" class="sim-btn" title="Reset to start">⏮</button>
      <button id="sim-btn-play"  class="sim-btn sim-btn-primary" title="Play">▶</button>
      <button id="sim-btn-pause" class="sim-btn" title="Pause" style="display:none">⏸</button>
    </div>

    <div class="sim-scrubber-wrap">
      <input type="range" id="sim-scrubber" min="0" max="143" value="0" step="1">
      <div class="sim-time-labels">
        <span id="sim-t-current">Sep 28 · 00:00 UTC</span>
        <span id="sim-t-total">72h</span>
      </div>
    </div>

    <div class="sim-speed-row">
      <span class="hud-label">Speed</span>
      <select id="sim-speed-select" class="sim-dropdown sim-speed-dropdown">
        ${SPEED_OPTIONS.map(s =>
          `<option value="${s}" ${s===1?'selected':''}>${s}×</option>`
        ).join('')}
      </select>
    </div>

    <div class="panel-divider"></div>

    <div class="infil-toggle-row">
      <label class="infil-label">
        <input type="checkbox" id="sim-infil-check">
        <span>Soil infiltration (Horton/SSURGO)</span>
      </label>
      <div class="infil-note" id="sim-infil-note">
        Off — all rain stays on surface (max runoff)
      </div>
    </div>

    <div class="panel-divider"></div>

    <div class="lake-hud">
      <div class="hud-title">AOI Status</div>
      <div class="hud-row">
        <span class="hud-label">Time (UTC)</span>
        <span id="hud-datetime" style="color:#7ab0d8">Sep 28 · 00:00</span>
      </div>
      <div class="hud-row">
        <span class="hud-label">Rain now</span>
        <span id="hud-rain">0.0 mm/hr</span>
      </div>
      <div class="hud-row">
        <span class="hud-label">Flooded</span>
        <span id="hud-flooded">— ha</span>
      </div>
      <div class="hud-row">
        <span class="hud-label">Max depth</span>
        <span id="hud-depth">— m</span>
      </div>
    </div>

    <div class="attribution">
      Physics · Bates et al. (2010) LISFLOOD-FP ·
      Soil · USDA SSURGO Horton AMC-III (fc=23 mm/hr) ·
      DEM · USGS 3DEP 1m LiDAR · Rain · ASOS MCO
    </div>
  `;

  // ── Element refs ──────────────────────────────────────────────────────────
  const playBtn      = document.getElementById('sim-btn-play');
  const pauseBtn     = document.getElementById('sim-btn-pause');
  const resetBtn     = document.getElementById('sim-btn-reset');
  const scrubber     = document.getElementById('sim-scrubber');
  const tCurrent     = document.getElementById('sim-t-current');
  const tTotal       = document.getElementById('sim-t-total');
  const statPeak     = document.getElementById('sim-stat-peak');
  const hudDatetime  = document.getElementById('hud-datetime');
  const hudRain      = document.getElementById('hud-rain');
  const hudFlooded   = document.getElementById('hud-flooded');
  const hudDepth     = document.getElementById('hud-depth');
  const speedSel     = document.getElementById('sim-speed-select');
  const infilCheck   = document.getElementById('sim-infil-check');
  const infilNote    = document.getElementById('sim-infil-note');

  // ── Load scenario ─────────────────────────────────────────────────────────
  async function loadScenario() {
    hydrograph = null;
    nFrames    = 0;

    try {
      hydrograph = await fetch(hydrographUrl()).then(r => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      });
    } catch (e) {
      panel.innerHTML = `<div class="sim-error">
        Ian simulation data not ready.<br>
        Run: <code>python3 simulation/flood_sim_ian.py --save-frames --no-infiltration</code><br>
        Then: <code>python3 viewer/preprocess/export_ian_simulation.py</code>
      </div>`;
      return false;
    }

    try {
      const info = await floodLayer.loadScenario(scenarioId());
      nFrames = info.nFrames;
      scrubber.max = String(nFrames - 1);
    } catch (e) {
      panel.innerHTML = `<div class="sim-error">
        Flood frame binary not found.<br>
        Run: <code>python3 simulation/flood_sim_ian.py --save-frames</code>
      </div>`;
      return false;
    }

    const totalMin = hydrograph.times_min[nFrames - 1] ?? (nFrames * 30);
    tTotal.textContent   = fmtMin(totalMin);
    statPeak.textContent = `${hydrograph.peak_flooded_ha?.toFixed(1) ?? '—'} ha peak`;
    return true;
  }

  // ── Context layers ───────────────────────────────────────────────────────
  // Used to force-enable Wireframe on Play (auto-restoring prior state on Reset) so the
  // simulation always had visible terrain context. Removed 2026-07-28 per direct request —
  // Wireframe now stays exactly as the user set it (via its own checkbox) through Play/Pause/
  // Reset, since it visually competes with the flood animation. Kept as a no-op function (not
  // deleted outright) so startPlay()/resetPlay() below don't need restructuring if a real
  // context-layer need comes up again later.
  function enableContextLayers(on) {}

  // ── Playback helpers ──────────────────────────────────────────────────────
  function goToFrame(idx) {
    if (!nFrames || idx < 0 || idx >= nFrames) return;
    currentFrame   = idx;
    scrubber.value = String(idx);
    floodLayer.setFrame(idx);

    if (!hydrograph) return;
    const tMin     = hydrograph.times_min[idx]  ?? idx * 30;
    const rainNow  = hydrograph.rain_mm_hr[idx] ?? 0;
    const flooded  = hydrograph.flooded_ha[idx]  ?? 0;
    const maxDepth = hydrograph.max_depth_m?.[idx] ?? 0;

    tCurrent.textContent    = minsToDate(tMin).replace(' · ', '\n');
    hudDatetime.textContent = minsToDate(tMin);
    hudRain.textContent     = rainNow > 0 ? `${rainNow.toFixed(1)} mm/hr` : '0 mm/hr';
    hudFlooded.textContent  = flooded > 0 ? `${flooded.toFixed(1)} ha` : '—';
    hudDepth.textContent    = maxDepth > 0.005 ? `${maxDepth.toFixed(3)} m` : '—';
    hudRain.style.color     = rainNow > 30 ? '#60c0ff' : rainNow > 5 ? '#90b8d8' : '#4a7a9a';

    rainParticles.update(rainNow, 0);
  }

  function startPlay() {
    if (!nFrames) return;
    isPlaying = true;
    accumMs   = 0;
    floodLayer.mesh.visible = true;
    enableContextLayers(true);
    playBtn.style.display  = 'none';
    pauseBtn.style.display = '';
  }

  function pausePlay() {
    isPlaying = false;
    rainParticles.update(0, 0);
    playBtn.style.display  = '';
    pauseBtn.style.display = 'none';
  }

  function resetPlay() {
    pausePlay();
    enableContextLayers(false);
    goToFrame(0);
    floodLayer.reset();
    floodLayer.mesh.visible = false;
  }

  // ── Infiltration toggle ───────────────────────────────────────────────────
  infilCheck.addEventListener('change', async () => {
    const wasPlaying = isPlaying;
    pausePlay();
    floodLayer.mesh.visible = false;

    useInfiltration = infilCheck.checked;
    infilNote.textContent = useInfiltration
      ? 'On — soil absorbs rain (Horton/SSURGO); less surface water'
      : 'Off — all rain stays on surface (max runoff)';
    statPeak.textContent = 'Loading…';

    const ok = await loadScenario();
    if (ok) {
      goToFrame(0);
      if (wasPlaying) startPlay();
    }
  });

  // ── Event listeners ───────────────────────────────────────────────────────
  playBtn.addEventListener('click', () => {
    if (currentFrame >= nFrames - 1) goToFrame(0);
    startPlay();
  });
  pauseBtn.addEventListener('click', pausePlay);
  resetBtn.addEventListener('click', resetPlay);
  scrubber.addEventListener('input', () => {
    pausePlay();
    goToFrame(parseInt(scrubber.value, 10));
  });
  speedSel.addEventListener('change', () => {
    speedMult = parseFloat(speedSel.value);
  });

  // ── tick() — called from render loop ─────────────────────────────────────
  function tick(dt) {
    if (!isPlaying || !nFrames) return;
    rainParticles.update(hydrograph?.rain_mm_hr[currentFrame] ?? 0, dt);
    const frameMs = FRAME_MS / speedMult;
    accumMs += dt * 1000;
    if (accumMs >= frameMs) {
      accumMs -= frameMs;
      const next = currentFrame + 1;
      if (next >= nFrames) { goToFrame(nFrames - 1); pausePlay(); }
      else goToFrame(next);
    }
  }

  // ── Initial load (no-infiltration default) ────────────────────────────────
  const ok = await loadScenario();
  if (ok) goToFrame(0);

  return { tick };
}
