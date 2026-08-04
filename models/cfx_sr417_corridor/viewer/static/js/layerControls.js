/**
 * layerControls.js — grouped layer panel with sections, dropdowns, and
 * a two-slot PlanetScope scene selector.
 *
 * setupLayerPanel(config, scene)
 *   config.terrain   — { solidMesh, wireMesh, analysisMeshes{key→mesh},
 *                        lidarBridges, lidarCloud }
 *   config.hydrology — Array<{ name, mesh, on }>
 *   config.base      — Array<{ name, mesh, on, swatch, drape?, drapedMesh?, drapedOn?, legendUrl? }>
 *   config.risk      — Array<{ name, mesh, on, swatch }>
 *   config.planetscope — { scenes[], slotA_mesh, slotB_mesh }
 */
export function setupLayerPanel(config) {
  const list = document.getElementById('layer-list');
  if (!list) return;

  // ── TOPOGRAPHY ─────────────────────────────────────────────────────────────
  _sectionHeader(list, 'TOPOGRAPHY');

  _toggleRow(list, 'Surface',   config.terrain.solidMesh, false, '#1e5a3a');
  _toggleRow(list, 'Wireframe', config.terrain.wireMesh,  false, '#3aaa60');
  _vertExagRow(list);

  // Mutually-exclusive analysis layer dropdown
  _selectorRow(list, 'Analysis Layer', config.terrain.analysisMeshes, '', '#8866cc');

  // Raw LiDAR — the actual point-cloud/mesh data the DEM above is derived from
  _toggleRow(list, 'LiDAR Bridge Correction',    config.terrain.lidarBridges, false, '#ff5533');
  _toggleRow(list, 'LiDAR Point Cloud (full AOI)', config.terrain.lidarCloud, false, '#909090');

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

  // ── RISK ANALYSIS ──────────────────────────────────────────────────────────
  _sectionHeader(list, 'RISK ANALYSIS');
  for (const item of config.risk) {
    _toggleRow(list, item.name, item.mesh, item.on, item.swatch);
  }

  // ── PLANETSCOPE ────────────────────────────────────────────────────────────
  _sectionHeader(list, 'PLANETSCOPE');
  // Was defaulted to 'max1' (Hurricane Ian) — changed to off (2026-07-20, per request) so the
  // page loads with only NAIP Aerial Draped visible (Wireframe also defaulted on until
  // 2026-07-27, when it was turned off too per direct request).
  _planetScopeSlot(list, 'Scene A', config.planetscope.slotA, config.planetscope.slotA_draped, config.planetscope.scenes, '', 'rgb');
  _planetScopeSlot(list, 'Scene B', config.planetscope.slotB, config.planetscope.slotB_draped, config.planetscope.scenes, '', 'rgb');
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

// Moved here from a static block in index.html (2026-07-20) so it sits directly under
// Wireframe in TOPOGRAPHY, where it's actually relevant — it only affects the terrain-derived
// layers (Surface/Wireframe/draped overlays/LiDAR meshes), not the whole scene. Keeps the same
// element ids (vert-exag-slider/-value) since main.js's existing slider-drag event wiring reads
// them by id and isn't touched here — the preset buttons just set slider.value and dispatch a
// real 'input' event so that unchanged listener does the actual work (terrain.updateExag, etc).
const EXAG_PRESETS = [1, 2, 4, 8];

function _vertExagRow(list) {
  const wrap = document.createElement('div');
  wrap.id = 'vert-exag-control';

  const label = document.createElement('label');
  label.htmlFor = 'vert-exag-slider';
  label.innerHTML = 'Vertical Exaggeration: <span id="vert-exag-value">2.0×</span>';

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.id = 'vert-exag-slider';
  slider.min = '1'; slider.max = '8'; slider.step = '0.5'; slider.value = '2';

  const presetRow = document.createElement('div');
  presetRow.className = 'exag-preset-row';

  function syncActive() {
    const cur = parseFloat(slider.value);
    presetRow.querySelectorAll('.exag-preset-btn').forEach(btn => {
      btn.classList.toggle('active', parseFloat(btn.dataset.val) === cur);
    });
  }

  EXAG_PRESETS.forEach(v => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'exag-preset-btn';
    btn.dataset.val = String(v);
    btn.textContent = `${v}×`;
    btn.addEventListener('click', () => {
      slider.value = String(v);
      slider.dispatchEvent(new Event('input'));   // triggers main.js's existing handler
      syncActive();
    });
    presetRow.appendChild(btn);
  });
  slider.addEventListener('input', syncActive);

  wrap.append(label, slider, presetRow);
  list.appendChild(wrap);
  syncActive();
}

function _selectorRow(list, label, meshMap, defaultKey, swatch) {
  // meshMap: { key: mesh, ... }  — only one mesh visible at a time
  // Set all invisible initially except defaultKey
  Object.entries(meshMap).forEach(([k, m]) => { if (m) m.visible = k === defaultKey; });

  const wrap = document.createElement('div');
  wrap.className = 'selector-row';

  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = defaultKey !== '';
  cb.className = 'selector-cb';

  const sw = document.createElement('div');
  sw.className = 'layer-swatch';
  sw.style.background = swatch ?? '#4a6a8a';

  const span = document.createElement('span');
  span.className = 'layer-label selector-label';
  span.textContent = label;

  const sel = document.createElement('select');
  sel.className = 'layer-select';

  const noneOpt = document.createElement('option');
  noneOpt.value = '';
  noneOpt.textContent = '— none —';
  sel.appendChild(noneOpt);

  Object.entries(meshMap).forEach(([key, mesh]) => {
    if (!mesh) return;
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ');
    if (key === defaultKey) opt.selected = true;
    sel.appendChild(opt);
  });

  function applySelection() {
    const active = cb.checked ? sel.value : '';
    Object.entries(meshMap).forEach(([k, m]) => { if (m) m.visible = k === active; });
  }
  cb.addEventListener('change', applySelection);
  sel.addEventListener('change', () => {
    cb.checked = sel.value !== '';
    applySelection();
  });

  wrap.append(cb, sw, span);
  list.appendChild(wrap);
  list.appendChild(sel);
}

let _flatDrapedRowCounter = 0;   // each instance needs a unique radio `name` — a shared literal
                                   // (as this had when it was SSURGO-only) makes every Flat/Draped
                                   // widget on the page fight over the same radio group.

function _flatDrapedRow(list, label, mesh, defaultOn, swatch, drapedMesh, drapedOn, legendUrl) {
  if (!mesh) return;
  // isOn/startDraped are the SINGLE source of truth for the row's initial state — the radio's
  // checked attributes and mesh/drapedMesh.visible below all derive from these two, instead of
  // being set independently (that was a real bug: the radios always hardcoded Flat-checked
  // regardless of drapedOn, so a layer configured on+draped would show the Draped surface while
  // the UI still displayed "Flat" selected).
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

  // Flat/Draped radio
  const modeDiv = document.createElement('div');
  modeDiv.className = 'ssurgo-mode';   // shared styling class, not SSURGO-specific behavior

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
  applyMode();   // derive initial mesh/drapedMesh visibility from the radio state above,
                 // instead of setting it separately (that was the other half of the same bug)

  // Legend — optional; imagery layers (e.g. NAIP) don't have one, only classified data (SSURGO)
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

// ── PlanetScope scene selector ───────────────────────────────────────────────

const PS_SCENES = [
  { key: 'max1', label: 'Ian 2022-09-30 (MAX)',       category: 'MAX' },
  { key: 'max2', label: '2025-05-29 (MAX)',            category: 'MAX' },
  { key: 'max3', label: '2024-06-30 (MAX)',            category: 'MAX' },
  { key: 'avg1', label: '2025-06-17 (AVG wet)',        category: 'AVG' },
  { key: 'avg2', label: '2023-09-17 (AVG wet)',        category: 'AVG' },
  { key: 'avg3', label: '2025-12-04 (AVG dry)',        category: 'AVG' },
  { key: 'avg4', label: '2023-12-23 (AVG dry)',        category: 'AVG' },
  { key: 'min1', label: '2021-03-17 (MIN 43d dry)',    category: 'MIN' },
  { key: 'min2', label: '2024-12-09 (MIN 34d dry)',    category: 'MIN' },
  { key: 'min3', label: '2022-04-19 (MIN 33d dry)',    category: 'MIN' },
];

const PS_TYPES = ['rgb', 'ndvi', 'water'];

function _planetScopeSlot(list, slotLabel, flatMesh, drapedMesh, scenes, defaultScene, defaultType) {
  if (!flatMesh) return;

  const slotId    = slotLabel.replace(/\s+/g, '');
  const loader    = new (window.__THREE_TEXTURE_LOADER__);
  let currentScene = defaultScene;
  let currentType  = defaultType;
  let isDraped     = false;
  let isEnabled    = defaultScene !== '';

  // Apply a loaded texture to both flat and draped meshes
  function applyTexture(tex) {
    tex.colorSpace = window.__THREE_SRGB__;
    flatMesh.material.map = tex;
    flatMesh.material.needsUpdate = true;
    if (drapedMesh) {
      drapedMesh.material.map = tex;
      drapedMesh.material.needsUpdate = true;
    }
    updateVisibility();
  }

  function updateVisibility() {
    const show = isEnabled && !!currentScene;
    flatMesh.visible   = show && !isDraped;
    if (drapedMesh) drapedMesh.visible = show && isDraped;
  }

  function loadSceneTexture(sceneKey, typeKey) {
    if (!sceneKey) { isEnabled = false; updateVisibility(); return; }
    const url = `/data/ps_${sceneKey}_${typeKey}.png`;
    loader.load(url, tex => applyTexture(tex));
  }

  // ── Outer toggle checkbox ──────────────────────────────────────────────────
  const row = document.createElement('label');
  row.className = 'layer-row';

  const cb = document.createElement('input');
  cb.type    = 'checkbox';
  cb.checked = isEnabled;

  const sw = document.createElement('div');
  sw.className = 'layer-swatch';
  sw.style.background = defaultScene ? '#e8c060' : '#555';

  const span = document.createElement('span');
  span.className = 'layer-label';
  span.textContent = slotLabel;

  row.append(cb, sw, span);
  list.appendChild(row);

  // ── Date dropdown ──────────────────────────────────────────────────────────
  const dateSel = document.createElement('select');
  dateSel.className = 'layer-select ps-date-sel';

  const noneOpt = document.createElement('option');
  noneOpt.value = '';
  noneOpt.textContent = '— none —';
  dateSel.appendChild(noneOpt);

  ['MAX', 'AVG', 'MIN'].forEach(cat => {
    const grp = document.createElement('optgroup');
    grp.label = cat;
    PS_SCENES.filter(s => s.category === cat).forEach(s => {
      const opt = document.createElement('option');
      opt.value = s.key;
      opt.textContent = s.label;
      if (s.key === defaultScene) opt.selected = true;
      grp.appendChild(opt);
    });
    dateSel.appendChild(grp);
  });
  list.appendChild(dateSel);

  // ── Band type radio row ────────────────────────────────────────────────────
  const typeRow = document.createElement('div');
  typeRow.className = 'ps-type-row';

  PS_TYPES.forEach(t => {
    const lbl = document.createElement('label');
    lbl.className = 'radio-label';
    const r = document.createElement('input');
    r.type    = 'radio';
    r.name    = `ps-type-${slotId}`;
    r.value   = t;
    r.checked = t === defaultType;
    r.addEventListener('change', () => {
      currentType = t;
      if (isEnabled && currentScene) loadSceneTexture(currentScene, currentType);
    });
    lbl.append(r, document.createTextNode(t.toUpperCase()));
    typeRow.appendChild(lbl);
  });
  list.appendChild(typeRow);

  // ── Flat / Draped radio row ────────────────────────────────────────────────
  if (drapedMesh) {
    const modeRow = document.createElement('div');
    modeRow.className = 'ssurgo-mode ps-mode-row';

    ['flat', 'draped'].forEach(m => {
      const lbl = document.createElement('label');
      lbl.className = 'radio-label';
      const r = document.createElement('input');
      r.type    = 'radio';
      r.name    = `ps-mode-${slotId}`;
      r.value   = m;
      r.checked = m === 'flat';
      r.addEventListener('change', () => {
        isDraped = m === 'draped';
        updateVisibility();
      });
      lbl.append(r, document.createTextNode(m.charAt(0).toUpperCase() + m.slice(1)));
      modeRow.appendChild(lbl);
    });
    list.appendChild(modeRow);
  }

  // ── Event wiring ───────────────────────────────────────────────────────────
  dateSel.addEventListener('change', () => {
    currentScene = dateSel.value;
    isEnabled    = currentScene !== '';
    cb.checked   = isEnabled;
    sw.style.background = isEnabled ? '#e8c060' : '#555';
    if (isEnabled) loadSceneTexture(currentScene, currentType);
    else updateVisibility();
  });

  cb.addEventListener('change', () => {
    isEnabled = cb.checked;
    if (isEnabled && currentScene) loadSceneTexture(currentScene, currentType);
    else updateVisibility();
  });

  // ── Initial load ───────────────────────────────────────────────────────────
  updateVisibility();
  if (defaultScene) loadSceneTexture(defaultScene, defaultType);
}
