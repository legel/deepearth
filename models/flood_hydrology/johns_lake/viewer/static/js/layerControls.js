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
export function setupLayerPanel(config) {
  const list = document.getElementById('layer-list');
  if (!list) return;

  // ── TOPOGRAPHY ─────────────────────────────────────────────────────────────
  _sectionHeader(list, 'TOPOGRAPHY');
  _toggleRow(list, 'Surface',   config.terrain.solidMesh, false, '#1e5a3a');
  _toggleRow(list, 'Wireframe', config.terrain.wireMesh,  false, '#3aaa60');
  _vertExagRow(list, config.terrain.onExagChange);
  _toggleRow(list, 'Lake Voxels',   config.terrain.voxelMesh, true, '#3a6abf');
  _toggleRow(list, 'Water Surface', config.terrain.waterMesh, true, '#1a5aaf');

  // ── HYDROLOGY ──────────────────────────────────────────────────────────────
  _sectionHeader(list, 'HYDROLOGY');
  for (const item of config.hydrology) {
    _toggleRow(list, item.name, item.mesh, item.on, item.swatch);
  }

  // ── BASE LAYERS ────────────────────────────────────────────────────────────
  _sectionHeader(list, 'BASE LAYERS');
  for (const item of config.base) {
    if (item.drape) {
      _flatDrapedRow(list, item.name, item.mesh, item.on, item.swatch,
                      item.drapedMesh, item.drapedOn, item.legendUrl);
    } else {
      _toggleRow(list, item.name, item.mesh, item.on, item.swatch);
    }
  }

  // ── SIMULATION LAYERS ──────────────────────────────────────────────────────
  _sectionHeader(list, 'SIMULATION LAYERS');
  for (const item of config.risk) {
    _toggleRow(list, item.name, item.mesh, item.on, item.swatch);
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
