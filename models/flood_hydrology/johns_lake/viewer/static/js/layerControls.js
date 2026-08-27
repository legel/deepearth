/**
 * layerControls.js — grouped layer panel with sections + vertical exaggeration slider.
 *
 * Ported from the sibling cfx_sr417_corridor project 2026-07-28 (trimmed to what Johns Lake
 * actually has — no PlanetScope, no multi-DEM analysis-layer dropdown), replacing the old flat
 * checkbox-list version. setupLayerPanel(config):
 *   config.terrain — { solidMesh, wireMesh, updateExag, onExagChange }
 *   config.hydrology — Array<{ name, mesh, on, swatch }>
 *   config.base      — Array<{ name, mesh, on, swatch, drape?, drapedMesh?, drapedOn?, legendUrl? }>
 *   config.risk      — Array<{ name, mesh, on, swatch }>
 */
import { createSection } from '/shared/panelSections.js';

export function setupLayerPanel(config) {
  const list = document.getElementById('layer-list');
  if (!list) return;

  // Same six-tier hierarchy as the sibling CFX site, built with the shared
  // /shared/panelSections.js so both viewers behave identically. Order is a data
  // hierarchy: what the place looks like -> what the ground is -> where water is ->
  // the regulatory answer -> our modelled answer -> the raw inputs.

  // ── 1. BASE MAP ────────────────────────────────────────────────────────────
  {
    const sec = createSection(list, 'Base Map', { open: true,
      note: 'Imagery and the elevation surface everything else drapes onto.' });
    for (const item of config.base.filter(i => i.group === 'basemap')) _row(sec, item);
    _toggleRow(sec, 'Terrain Surface',   config.terrain.solidMesh, false, '#1e5a3a');
    _toggleRow(sec, 'Terrain Wireframe', config.terrain.wireMesh,  false, '#3aaa60');
    _vertExagRow(sec, config.terrain.onExagChange);
  }

  // ── 2. GROUND & BUILT ──────────────────────────────────────────────────────
  {
    const sec = createSection(list, 'Ground & Built', { open: true,
      note: 'Drives infiltration: soil sets how fast water soaks in, roads and roofs '
          + 'set where it cannot.' });
    for (const item of config.base.filter(i => i.group === 'ground')) _row(sec, item);
  }

  // ── 3. WATER ───────────────────────────────────────────────────────────────
  {
    const sec = createSection(list, 'Water', { open: true,
      note: 'The lake itself plus mapped channels. Voxels show its modelled volume.' });
    _toggleRow(sec, 'Water Surface', config.terrain.waterMesh, true, '#1a5aaf');
    _toggleRow(sec, 'Lake Voxels',   config.terrain.voxelMesh, false, '#3a6abf');
    for (const item of config.hydrology) _row(sec, item);
  }

  // ── 4. FLOOD HAZARD (FEMA) ─────────────────────────────────────────────────
  {
    const sec = createSection(list, 'Flood Hazard (FEMA)', { open: true,
      note: "FEMA's published 1%-annual-chance floodplain — independent of this model." });
    for (const item of config.base.filter(i => i.group === 'regulatory')) _row(sec, item);
  }

  // ── 5. FLOOD SIMULATION ────────────────────────────────────────────────────
  {
    const sec = createSection(list, 'Flood Simulation', { open: true,
      note: "This project's solver output, plus satellite ground truth to compare against." });
    for (const item of config.risk) _row(sec, item);
  }

  // ── 6. SOURCE DATA ─────────────────────────────────────────────────────────
  {
    const sec = createSection(list, 'Source Data', { open: false,
      note: 'Raw inputs behind the layers above. Heavy, loaded on demand.' });
    const heavy = document.getElementById('heavy-lidar-control');
    if (heavy) { heavy.hidden = false; sec.appendChild(heavy); }
  }
}

/** Render one config item, choosing the flat/draped variant when it has one. */
function _row(parent, item) {
  if (item.drape) {
    _flatDrapedRow(parent, item.name, item.mesh, item.on, item.swatch,
                   item.drapedMesh, item.drapedOn, item.legendUrl);
  } else {
    _toggleRow(parent, item.name, item.mesh, item.on, item.swatch);
  }
}

// ── Private helpers ─────────────────────────────────────────────────────────

function _sectionHeader(list, title) {
  const div = document.createElement('div');
  div.className = 'section-header';
  div.textContent = title;
  list.appendChild(div);
}

function _toggleRow(list, label, mesh, defaultOn, swatch) {
  if (!mesh) return;
  mesh.visible = defaultOn ?? false;

  const row = document.createElement('label');
  row.className = 'layer-row';

  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = defaultOn ?? false;
  cb.addEventListener('change', () => { mesh.visible = cb.checked; });

  const sw = document.createElement('div');
  sw.className = 'layer-swatch';
  sw.style.background = swatch ?? '#4a6a8a';

  const span = document.createElement('span');
  span.className = 'layer-label';
  span.textContent = label;

  row.append(cb, sw, span);
  list.appendChild(row);
}

const EXAG_PRESETS = [1, 2, 4, 8];

function _vertExagRow(list, onExagChange) {
  const wrap = document.createElement('div');
  wrap.id = 'vert-exag-control';

  const label = document.createElement('label');
  label.htmlFor = 'vert-exag-slider';
  label.innerHTML = 'Vertical Exaggeration: <span id="vert-exag-value">8.0×</span>';

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.id = 'vert-exag-slider';
  slider.min = '1'; slider.max = '8'; slider.step = '0.5'; slider.value = '8';

  const presetRow = document.createElement('div');
  presetRow.className = 'exag-preset-row';

  function syncActive() {
    const cur = parseFloat(slider.value);
    presetRow.querySelectorAll('.exag-preset-btn').forEach(btn => {
      btn.classList.toggle('active', parseFloat(btn.dataset.val) === cur);
    });
  }

  let lastVal = parseFloat(slider.value);
  slider.addEventListener('input', () => {
    const newVal = parseFloat(slider.value);
    document.getElementById('vert-exag-value').textContent = newVal.toFixed(1) + '×';
    if (onExagChange) onExagChange(newVal, lastVal);
    lastVal = newVal;
    syncActive();
  });

  EXAG_PRESETS.forEach(v => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'exag-preset-btn';
    btn.dataset.val = String(v);
    btn.textContent = `${v}×`;
    btn.addEventListener('click', () => {
      slider.value = String(v);
      slider.dispatchEvent(new Event('input'));
      syncActive();
    });
    presetRow.appendChild(btn);
  });

  wrap.append(label, slider, presetRow);
  list.appendChild(wrap);
  syncActive();
}

let _flatDrapedRowCounter = 0;

function _flatDrapedRow(list, label, mesh, defaultOn, swatch, drapedMesh, drapedOn, legendUrl) {
  if (!mesh) return;
  const isOn = defaultOn ?? false;
  const startDraped = (drapedOn ?? false) && !!drapedMesh;

  const radioGroupName = `flat-draped-${_flatDrapedRowCounter++}`;

  const row = document.createElement('label');
  row.className = 'layer-row';

  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = isOn;

  const sw = document.createElement('div');
  sw.className = 'layer-swatch';
  sw.style.background = swatch ?? '#7a9a5a';

  const span = document.createElement('span');
  span.className = 'layer-label';
  span.textContent = label;

  const modeDiv = document.createElement('div');
  modeDiv.className = 'ssurgo-mode';

  function buildRadio(val, txt, checked) {
    const lbl = document.createElement('label');
    lbl.className = 'radio-label';
    const r = document.createElement('input');
    r.type = 'radio';
    r.name = radioGroupName;
    r.value = val;
    r.checked = checked;
    lbl.append(r, document.createTextNode(txt));
    return { lbl, r };
  }
  const { lbl: flatLbl, r: flatR } = buildRadio('flat',   'Flat',   !startDraped);
  const { lbl: drapLbl, r: drapR } = buildRadio('draped', 'Draped', startDraped);
  modeDiv.append(flatLbl, drapLbl);

  function applyMode() {
    if (!cb.checked) { mesh.visible = false; if (drapedMesh) drapedMesh.visible = false; return; }
    const draped = drapR.checked;
    mesh.visible        = !draped;
    if (drapedMesh) drapedMesh.visible = draped;
  }
  flatR.addEventListener('change', applyMode);
  drapR.addEventListener('change', applyMode);
  applyMode();

  let legendDiv = null;
  let legendCache = null;
  if (legendUrl) {
    legendDiv = document.createElement('div');
    legendDiv.className = 'ssurgo-legend';
    legendDiv.style.display = 'none';
  }

  cb.addEventListener('change', async () => {
    applyMode();
    if (cb.checked) {
      if (legendDiv) {
        legendDiv.style.display = 'block';
        if (!legendCache) {
          try { legendCache = await fetch(legendUrl).then(r => r.json()); }
          catch { return; }
          legendDiv.innerHTML = legendCache.map(e => {
            const [r, g, b] = e.rgba;
            return `<div class="legend-row">` +
              `<div class="legend-swatch" style="background:rgb(${r},${g},${b})"></div>` +
              `<span class="legend-label">${e.label}</span></div>`;
          }).join('');
        }
      }
    } else {
      mesh.visible = false;
      if (drapedMesh) drapedMesh.visible = false;
      if (legendDiv) legendDiv.style.display = 'none';
    }
  });

  row.append(cb, sw, span);
  list.appendChild(row);
  list.appendChild(modeDiv);
  if (legendDiv) list.appendChild(legendDiv);
}
