/**
 * layerControls.js — the digital-twin layer panel: collapsible sections ordered as a data
 * hierarchy, a mutually-exclusive terrain-analysis dropdown, and two satellite scene slots.
 *
 * setupLayerPanel(config)
 *   config.terrain     — { solidMesh, wireMesh, analysisMeshes{name→mesh},
 *                          lidarBridges, lidarCloud }
 *   config.base        — Array<LayerItem>, each tagged group:'basemap' | 'ground'
 *   config.hydrology   — Array<LayerItem>   water: mapped channels + DEM-derived drainage
 *   config.regulatory  — Array<LayerItem>   FEMA's published flood-hazard mapping
 *   config.simulation  — Array<LayerItem>   this project's own solver output
 *   config.planetscope — { scenes[], slotA, slotB, slotA_draped, slotB_draped }
 *
 * LayerItem = { name, mesh, on, swatch,
 *               drape?, drapedMesh?, drapedOn?, legendUrl?, group? }
 *
 * A layer belongs to exactly one section. Splitting one dataset across two sections (as FEMA
 * and the water layers previously were) makes related layers look unrelated.
 */
export function setupLayerPanel(config) {
  const list = document.getElementById('layer-list');
  if (!list) return;

  // Sections are ordered as a data hierarchy, not an arbitrary list:
  //
  //   1. BASE MAP        what the place looks like        (observed imagery + terrain)
  //   2. GROUND          what the surface is made of      (soil, built surface)
  //   3. WATER           where water is and where it goes (mapped + DEM-derived)
  //   4. FLOOD HAZARD    the regulatory answer            (FEMA)
  //   5. SIMULATION      our modelled answer              (solver output)
  //   6. SOURCE DATA     the raw inputs behind the above  (LiDAR, satellite scenes)
  //
  // Each tier depends only on the tiers above it, so reading top to bottom follows how the
  // twin is actually built. Everything is collapsible because the panel carries ~35 controls.

  // ── 1. BASE MAP ────────────────────────────────────────────────────────────
  {
    const s = _section(list, 'Base Map', { open: true,
      note: 'Imagery and the elevation surface everything else drapes onto.' });
    for (const item of config.base.filter(i => i.group === 'basemap')) _row(s, item);
    _toggleRow(s, 'Terrain Surface',   config.terrain.solidMesh, false, '#1e5a3a');
    _toggleRow(s, 'Terrain Wireframe', config.terrain.wireMesh,  false, '#3aaa60');
    _vertExagRow(s);
    _selectorRow(s, 'Terrain Analysis', config.terrain.analysisMeshes, '', '#8866cc');
  }

  // ── 2. GROUND & BUILT SURFACE ──────────────────────────────────────────────
  {
    const s = _section(list, 'Ground & Built', { open: true,
      note: 'Drives infiltration: soil sets how fast water soaks in, roads and roofs '
          + 'set where it cannot.' });
    for (const item of config.base.filter(i => i.group === 'ground')) _row(s, item);
  }

  // ── 3. WATER ───────────────────────────────────────────────────────────────
  {
    const s = _section(list, 'Water', { open: true,
      note: 'Mapped channels plus DEM-derived drainage. Low height-above-drainage '
          + 'floods first.' });
    for (const item of config.hydrology) _row(s, item);
  }

  // ── 4. FLOOD HAZARD (regulatory) ───────────────────────────────────────────
  {
    const s = _section(list, 'Flood Hazard (FEMA)', { open: true,
      note: 'FEMA\'s published 1%-annual-chance floodplain and floodway — independent '
          + 'of this model.' });
    for (const item of config.regulatory) _row(s, item);
  }

  // ── 5. FLOOD SIMULATION ────────────────────────────────────────────────────
  {
    const s = _section(list, 'Flood Simulation', { open: true,
      note: 'This project\'s solver output. Compare with the regulatory layer above; '
          + 'neither is ground truth.' });
    for (const item of config.simulation) _row(s, item);
    // The per-test-site sub-panels are authored in the template (they carry playback and
    // rain-intensity controls); they are relocated into this section so every modelled
    // layer lives in one place.
    const siteBlock = document.getElementById('site-sim-controls');
    if (siteBlock) { siteBlock.hidden = false; s.appendChild(siteBlock); }
  }

  // ── 6. SOURCE DATA (advanced) ──────────────────────────────────────────────
  {
    const s = _section(list, 'Source Data', { open: false,
      note: 'Raw inputs behind the layers above. Heavy, loaded on demand.' });
    _toggleRow(s, 'LiDAR Bridge Correction',      config.terrain.lidarBridges, false, '#ff5533');
    _toggleRow(s, 'LiDAR Point Cloud (full AOI)', config.terrain.lidarCloud,   false, '#909090');
    const heavy = document.getElementById('heavy-lidar-control');
    if (heavy) { heavy.hidden = false; s.appendChild(heavy); }
    _planetScopeSlot(s, 'Satellite Scene A', config.planetscope.slotA,
                     config.planetscope.slotA_draped, config.planetscope.scenes, '', 'rgb');
    _planetScopeSlot(s, 'Satellite Scene B', config.planetscope.slotB,
                     config.planetscope.slotB_draped, config.planetscope.scenes, '', 'rgb');
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

/**
 * Collapsible section. Returns the body element that rows should be appended to, so callers
 * add rows to the section rather than to the flat list.
 */
function _section(list, title, { open = true, note = '' } = {}) {
  const header = document.createElement('div');
  header.className = 'section-header collapsible' + (open ? ' open' : '');
  header.tabIndex = 0;
  header.setAttribute('role', 'button');
  header.setAttribute('aria-expanded', String(open));

  const chevron = document.createElement('span');
  chevron.className = 'section-chevron';
  chevron.textContent = '▼';
  chevron.setAttribute('aria-hidden', 'true');

  const label = document.createElement('span');
  label.textContent = title;

  const count = document.createElement('span');
  count.className = 'section-count';

  header.append(chevron, label, count);

  const body = document.createElement('div');
  body.className = 'section-body';
  body.hidden = !open;

  if (note) {
    const n = document.createElement('div');
    n.className = 'section-note';
    n.textContent = note;
    body.appendChild(n);
  }

  function toggle() {
    const nowOpen = body.hidden;
    body.hidden = !nowOpen;
    header.classList.toggle('open', nowOpen);
    header.setAttribute('aria-expanded', String(nowOpen));
  }
  header.addEventListener('click', toggle);
  header.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
  });

  list.append(header, body);

  // "2 / 5 on" tells you at a glance whether a collapsed section is doing anything.
  const refresh = () => {
    const boxes = body.querySelectorAll('input[type="checkbox"]');
    if (!boxes.length) { count.textContent = ''; return; }
    const on = [...boxes].filter(b => b.checked).length;
    count.textContent = on ? `${on}/${boxes.length} on` : `${boxes.length}`;
  };
  body.addEventListener('change', refresh);
  queueMicrotask(refresh);

  return body;
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
