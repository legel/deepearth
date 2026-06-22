/**
 * setupLayerControls(layers)
 *
 * layers: Array<{ name: string, mesh: THREE.Object3D|null, defaultOn: boolean, swatch?: string }>
 *
 * Injects a checkbox per layer into #layer-list and wires visibility toggling.
 * For the SSURGO layer, also fetches and shows a color legend on toggle-on.
 */
export function setupLayerControls(layers) {
  const list = document.getElementById('layer-list');
  if (!list) return;

  let ssurgoLegendCache = null;

  layers.forEach(({ name, mesh, defaultOn, swatch }) => {
    if (!mesh) return;

    mesh.visible = defaultOn ?? true;

    const row = document.createElement('label');
    row.className = 'layer-row';

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = defaultOn ?? true;

    const sw = document.createElement('div');
    sw.className = 'layer-swatch';
    sw.style.background = swatch ?? swatchFromName(name);

    const label = document.createElement('span');
    label.className = 'layer-label';
    label.textContent = name;

    row.appendChild(cb);
    row.appendChild(sw);
    row.appendChild(label);
    list.appendChild(row);

    const isSsurgo = name.toLowerCase().includes('ssurgo') || name.toLowerCase().includes('soil');

    if (isSsurgo) {
      const legendDiv = document.createElement('div');
      legendDiv.id = 'ssurgo-legend';
      legendDiv.className = 'ssurgo-legend';
      legendDiv.style.display = 'none';
      list.appendChild(legendDiv);

      cb.addEventListener('change', async () => {
        mesh.visible = cb.checked;
        if (cb.checked) {
          if (!ssurgoLegendCache) {
            try {
              ssurgoLegendCache = await fetch('/data/ssurgo_legend.json').then(r => r.json());
            } catch { return; }
            legendDiv.innerHTML = ssurgoLegendCache.map(entry => {
              const [r, g, b] = entry.rgba;
              return `<div class="legend-row">` +
                `<div class="legend-swatch" style="background:rgb(${r},${g},${b})"></div>` +
                `<span class="legend-label">${entry.label}</span></div>`;
            }).join('');
          }
          legendDiv.style.display = 'block';
        } else {
          legendDiv.style.display = 'none';
        }
      });
    } else {
      cb.addEventListener('change', () => { mesh.visible = cb.checked; });
    }
  });
}

function swatchFromName(name) {
  const n = name.toLowerCase();
  if (n.includes('terrain') || n.includes('dem')) return '#1e5a3a';
  if (n.includes('voxel') || n.includes('lake')) return '#4a7acf';
  if (n.includes('naip') || n.includes('aerial')) return '#8a6a3a';
  if (n.includes('ssurgo') || n.includes('soil')) return '#7a9a5a';
  return '#4a6a8a';
}
