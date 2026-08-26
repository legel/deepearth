/**
 * panelSections.js — the collapsible layer-panel section, shared by every viewer page.
 *
 * Two entry points, one implementation:
 *
 *   createSection(list, title, opts)  build a section from JS  (main AOI page, where the
 *                                     layer set is assembled programmatically)
 *   initStaticSections(root)          upgrade sections authored in HTML  (site pages, whose
 *                                     controls carry bespoke sliders and status lines)
 *
 * Both produce identical markup and behaviour, so the panel reads the same on every page
 * regardless of how its rows were built.
 *
 * Section order is a data hierarchy, and every page should follow it:
 *
 *   1. Base Map          what the place looks like        (imagery + elevation surface)
 *   2. Ground & Built    what the surface is made of      (soil, roads, buildings)
 *   3. Water             where water is and where it goes (mapped + DEM-derived)
 *   4. Flood Hazard      the regulatory answer            (FEMA)
 *   5. Flood Simulation  the modelled answer              (solver output)
 *   6. Source Data       the raw inputs behind the above  (LiDAR, satellite scenes)
 *
 * Each tier depends only on the tiers above it, so reading top to bottom follows how the
 * twin is actually built. A dataset belongs to exactly one section: splitting one across
 * two (as FEMA and the water layers once were) makes related layers look unrelated.
 */

/** Wire a header element to a body element: click / Enter / Space toggles, count auto-updates. */
function _bind(header, body, { open }) {
  header.classList.add('collapsible');
  header.classList.toggle('open', open);
  header.tabIndex = 0;
  header.setAttribute('role', 'button');
  header.setAttribute('aria-expanded', String(open));
  body.hidden = !open;

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

  // "2/5 on" so a collapsed section still says whether it is doing anything.
  let count = header.querySelector('.section-count');
  if (!count) {
    count = document.createElement('span');
    count.className = 'section-count';
    header.appendChild(count);
  }
  const refresh = () => {
    const boxes = body.querySelectorAll('input[type="checkbox"]');
    if (!boxes.length) { count.textContent = ''; return; }
    const on = [...boxes].filter(b => b.checked).length;
    count.textContent = on ? `${on}/${boxes.length} on` : `${boxes.length}`;
  };
  body.addEventListener('change', refresh);
  queueMicrotask(refresh);
  return toggle;
}

/** Prepend the rotating chevron, unless the header already has one. */
function _addChevron(header) {
  if (header.querySelector('.section-chevron')) return;
  const chevron = document.createElement('span');
  chevron.className = 'section-chevron';
  chevron.textContent = '▼';
  chevron.setAttribute('aria-hidden', 'true');
  header.prepend(chevron);
}

/**
 * Build a collapsible section and append it to `list`.
 * Returns the body element, so callers append their rows to the section rather than the list.
 */
export function createSection(list, title, { open = true, note = '' } = {}) {
  const header = document.createElement('div');
  header.className = 'section-header';

  const label = document.createElement('span');
  label.textContent = title;
  header.appendChild(label);
  _addChevron(header);

  const body = document.createElement('div');
  body.className = 'section-body';

  if (note) {
    const n = document.createElement('div');
    n.className = 'section-note';
    n.textContent = note;
    body.appendChild(n);
  }

  list.append(header, body);
  _bind(header, body, { open });
  return body;
}

/**
 * Upgrade sections authored directly in HTML.
 *
 * Expects, for each section, a `.section-header[data-section]` immediately followed by a
 * `.section-body`. Add `data-collapsed` to the header to start it closed, and
 * `data-note="…"` to render an explanatory line at the top of the body.
 *
 * Returns the number of sections upgraded, so a page can assert its panel was wired.
 */
export function initStaticSections(root = document) {
  const headers = root.querySelectorAll('.section-header[data-section]');
  let n = 0;
  for (const header of headers) {
    const body = header.nextElementSibling;
    if (!body || !body.classList.contains('section-body')) {
      console.warn('panelSections: header has no .section-body sibling', header.textContent);
      continue;
    }
    _addChevron(header);

    const note = header.dataset.note;
    if (note && !body.querySelector('.section-note')) {
      const n2 = document.createElement('div');
      n2.className = 'section-note';
      n2.textContent = note;
      body.prepend(n2);
    }

    _bind(header, body, { open: !header.hasAttribute('data-collapsed') });
    n++;
  }
  return n;
}
