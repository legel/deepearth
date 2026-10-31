// FluxFill dashboard. Static: every number comes from data/*.json written by dashboard/build.py.
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
const getJSON = (p) => fetch(p, { cache: "no-cache" }).then((r) => (r.ok ? r.json() : null)).catch(() => null);
async function getJSONgz(p) {
  if (typeof DecompressionStream !== "undefined") {
    try {
      const r = await fetch(p + ".gz", { cache: "no-cache" });
      if (r.ok) return await new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).json();
    } catch (e) { /* fall back to the plain file */ }
  }
  return getJSON(p);
}
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------- tabs ----------
const TABS = ["overview", "towers", "benchmark", "windows", "results", "methods", "report"];
function show(tab) {
  if (!TABS.includes(tab)) tab = "overview";
  for (const t of TABS) {
    $("#t-" + t).setAttribute("aria-selected", t === tab ? "true" : "false");
    $("#p-" + t).hidden = t !== tab;
  }
  if (tab === "towers" && map) setTimeout(() => map.invalidateSize(), 0);
  if (tab === "windows") drawWindow();
  if (tab === "results") { drawChart(); drawScaling(); }
}
document.addEventListener("click", (e) => {
  const a = e.target.closest("a[data-tab], a.cite");
  if (!a) return;
  const tab = a.dataset.tab || "report";
  const id = (a.getAttribute("href") || "").slice(1);
  e.preventDefault();
  history.replaceState(null, "", "#" + tab);
  show(tab);
  const el = id && document.getElementById(id);
  if (el && id !== tab) setTimeout(() => el.scrollIntoView({ block: "start" }), 0);
});
for (const t of TABS) $("#t-" + t).addEventListener("click", () => { history.replaceState(null, "", "#" + t); show(t); });
window.addEventListener("hashchange", () => show(location.hash.slice(1)));

// ---------- overview ----------
const OV_FLUX = ["NEE", "LE", "H", "FCH4"];
const OV_VIEWS = [["Validation sites", "Validation", "--series-3"], ["Test sites", "Test", "--series-2"]];
const METRIC_LABEL = { crps: "CRPS", cov95: "95% coverage", mae: "MAE", rmse: "RMSE", mse: "MSE", bias: "bias", r2: "R²", wape: "WAPE" };
function renderStatus(st) {
  if (!st) { $("#status-tiles").innerHTML = '<div class="tile"><div class="l">Status not available</div></div>'; return; }
  if (st.abstract) $("#ov-intro").textContent = st.abstract;
  const groups = [...new Set(st.tiles.map((t) => t.group || ""))];
  $("#status-tiles").innerHTML = groups.map((g) => { const ts = st.tiles.filter((t) => (t.group || "") === g); return `${g ? `<div class="tile-group">${esc(g)}</div>` : ""}<div class="grid cols-${ts.length % 3 === 0 ? 3 : 4}">${ts.map((t) =>
    `<div class="tile"><div class="v">${typeof t.value === "string" ? esc(t.value) : fmt(t.value)}</div><div class="l">${esc(t.label)}</div></div>`).join("")}</div>`; }).join("");
  $("#status-time").textContent = `Built ${st.built_utc} UTC. ${st.note || ""}`;
  $("#ov-updates").innerHTML = (st.updates || []).map((u) =>
    `<li><span class="ov-date">${esc(u.date)}</span><div><strong>${esc(u.title)}</strong> ${esc(u.text)}</div></li>`).join("");
  const res = st.results || [];
  if (!res.length) return;
  $("#ov-chart").innerHTML = ovChart(res);
  const cell = (f) => !f ? "<td>–</td>" : `<td class="num"><b>${f.pct > 0 ? "+" : "−"}${fmt(Math.abs(f.pct), 1)}%</b><span class="sub">${fmtMetric(f.ours_crps, "crps")} against ${fmtMetric(f.base_crps, "crps")} (${esc(f.base)})</span><span class="sub">best on ${f.best_metrics} of ${f.metrics} metrics</span><span class="sub">${fmtInt(f.n)} values</span></td>`;
  $("#ov-results").innerHTML = `<div class="table-wrap"><table class="ov-table"><thead><tr><th>Towers</th>${OV_FLUX.map((k) => `<th class="num">${k}</th>`).join("")}</tr></thead><tbody>
    ${res.map((r) => `<tr><th scope="row">${esc(r.view)}<span class="sub">${r.windows} windows, ${r.sites} towers</span></th>${OV_FLUX.map((k) => cell(r.fluxes.find((f) => f.flux === k))).join("")}</tr>`).join("")}
  </tbody></table></div>`;
  const lost = res.map((r) => ({ view: r.view, items: r.fluxes.flatMap((f) => (f.lost || []).map((l) => ({ flux: f.flux, ...l }))) })).filter((r) => r.items.length);
  $("#ov-open").innerHTML = !lost.length ? "" : `<h3>Metrics led by an established method</h3><p class="note">95% coverage is best at 0.95 and bias at 0.</p>${lost.map((r) =>
    `<p class="sub-title">${esc(r.view)}</p><ul class="ov-lost">${r.items.map((l) =>
    `<li><strong>${esc(l.flux)} ${esc(METRIC_LABEL[l.metric] || l.metric)}</strong>: FluxFill ${fmtMetric(l.ours, l.metric)}; ${esc(l.leader)} ${fmtMetric(l.lead, l.metric)}</li>`).join("")}</ul>`).join("")}`;
}

// CRPS reduction of FluxFill against the best established method, per flux and view (bars right of zero = lower CRPS).
function ovChart(res) {
  const rows = OV_FLUX.map((k) => ({ k, v: OV_VIEWS.map(([id, name, c]) => {
    const f = (res.find((r) => r.view === id) || { fluxes: [] }).fluxes.find((x) => x.flux === k);
    return f ? { name, c, red: -f.pct, f } : null;
  }) }));
  const all = rows.flatMap((r) => r.v.filter(Boolean).map((x) => x.red));
  const hi = Math.max(10, Math.ceil(Math.max(...all) / 10) * 10), lo = Math.min(0, Math.floor(Math.min(...all) / 10) * 10);
  const width = 720, L = 64, R = 70, bar = 16, gap = 3, grp = 18, top = 4, axis = 26;
  const H = top + rows.length * (2 * bar + gap + grp) - grp + axis;
  const X = (v) => L + (v - lo) / (hi - lo) * (width - L - R);
  let svg = `<svg viewBox="0 0 ${width} ${H}" role="img" aria-label="CRPS reduction of FluxFill against the best established method, per flux">`;
  const yAx = H - axis + 6;
  for (let t = lo; t <= hi; t += 10)
    svg += `<line x1="${X(t)}" x2="${X(t)}" y1="${top}" y2="${yAx - 4}" class="bm-grid"/><text x="${X(t)}" y="${yAx + 10}" text-anchor="middle" class="bm-t q">${t}%</text>`;
  rows.forEach((r, i) => {
    const y0 = top + i * (2 * bar + gap + grp);
    svg += `<text x="${L - 10}" y="${y0 + bar + 4}" text-anchor="end" class="bm-t ov-flux">${r.k}</text>`;
    r.v.forEach((x, j) => {
      if (!x) return;
      const y = y0 + j * (bar + gap), x0 = X(Math.min(0, x.red)), w = Math.abs(X(x.red) - X(0));
      const tip = `${r.k}, ${x.name} towers: FluxFill CRPS ${fmtMetric(x.f.ours_crps, "crps")} against ${fmtMetric(x.f.base_crps, "crps")} for ${x.f.base}; best on ${x.f.best_metrics} of ${x.f.metrics} metrics; ${fmtInt(x.f.n)} values`;
      svg += `<g class="ov-bar"><rect x="${L}" y="${y - 1}" width="${width - L}" height="${bar + 2}" fill="transparent"/><rect x="${x0}" y="${y}" width="${Math.max(w, 1)}" height="${bar}" rx="3" fill="var(${x.c})"/><title>${esc(tip)}</title></g>`;
      svg += `<text x="${X(Math.max(0, x.red)) + 6}" y="${y + bar * 0.75}" class="bm-t">${x.red >= 0 ? "−" : "+"}${fmt(Math.abs(x.red), 1)}%</text>`;
    });
  });
  svg += `<line x1="${X(0)}" x2="${X(0)}" y1="${top}" y2="${yAx - 4}" stroke="var(--ink)" stroke-width="1"/>`;
  return bmLegend(OV_VIEWS.map(([, name, c]) => [`${name} towers`, c])) + svg + "</svg>" +
    `<p class="note">CRPS reduction (%) of FluxFill against the established method with the lowest CRPS for each flux. Hover a bar for the scores.</p>`;
}

// ---------- towers ----------
let map = null, sites = [], layer = null;
const SPLIT_COLOR = { train: "--series-1", validation: "--series-3", test: "--series-2" };
function renderTowers() {
  const igbp = $("#f-igbp").value, net = $("#f-net").value, ch4 = $("#f-ch4").value;
  const rows = sites.filter((s) => (!igbp || s.igbp === igbp) && (!net || s.network === net) && (!ch4 || s.ch4));
  $("#tower-count").textContent = `${rows.length} of ${sites.length} towers shown.`;
  $("#tower-table tbody").innerHTML = rows.map((s) => `<tr><td>${s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener" title="${esc(s.source)} site page">${esc(s.id)}</a>` : esc(s.id)}</td><td>${s.name || ""}</td><td>${s.igbp || ""}</td><td>${s.koppen || ""}</td><td>${s.network || ""}</td><td>${s.years || ""}</td><td>${s.ch4 ? "yes" : ""}</td><td>${s.split || "not included"}</td></tr>`).join("");
  if (!map) return;
  if (layer) layer.remove();
  layer = L.layerGroup(rows.map((s) => L.circleMarker([s.lat, s.lon], {
    radius: 4.5, weight: 1.5, color: css("--ground"),
    fillColor: css(SPLIT_COLOR[s.split] || "--quiet"), fillOpacity: s.split ? 0.9 : 0.5,
  }).bindTooltip(`<b>${s.id}</b> ${s.name || ""}<br>${s.igbp || ""} · ${s.koppen || ""} · ${s.network || ""}<br>${s.years || ""}${s.ch4 ? " · FCH4" : ""}`)));
  layer.addTo(map);
}
function initTowers(data) {
  sites = (data && data.sites) || [];
  const uniq = (k) => [...new Set(sites.map((s) => s[k]).filter(Boolean))].sort();
  for (const v of uniq("igbp")) $("#f-igbp").insertAdjacentHTML("beforeend", `<option>${v}</option>`);
  for (const v of uniq("network")) $("#f-net").insertAdjacentHTML("beforeend", `<option>${v}</option>`);
  const frozen = sites.some((s) => s.split);
  $("#map-legend").innerHTML = frozen
    ? Object.entries(SPLIT_COLOR).map(([k, v]) => `<span><i style="background:${css(v)}"></i>${k}</span>`).join("") + `<span><i style="background:${css("--quiet")};opacity:.5"></i>not included (no usable 180-day window from 2014)</span>`
    : "<span>Site split not yet frozen; all towers shown as candidates.</span>";
  map = L.map("map", { worldCopyJump: false, scrollWheelZoom: false, zoomSnap: 0.25, minZoom: 1.5, maxZoom: 9, preferCanvas: true }).setView([28, 0], 1.75);
  map.getContainer().style.background = css("--ground");
  map.attributionControl.addAttribution("Land: Natural Earth (public domain)");
  map.createPane("land").style.zIndex = 250;
  getJSON("data/ne_110m_land.geojson").then((land) => {
    if (!land) return;
    L.geoJSON(land, { interactive: false, pane: "land",
      style: { color: css("--hair"), weight: 1, fillColor: css("--raised"), fillOpacity: 1 } }).addTo(map);
  });
  for (const id of ["#f-igbp", "#f-net", "#f-ch4"]) $(id).addEventListener("change", renderTowers);
  renderTowers();
}

// ---------- windows ----------
let windows = null, winCache = {};
function opt(sel, values, labels) {
  sel.innerHTML = values.map((v, i) => `<option value="${v}">${labels ? labels[i] : v}</option>`).join("");
}
const METHOD_NAMES = { climatology: "Climatology", mds: "MDS (REddyProc)", irvin_rf: "RF ensemble (Irvin 2021)",
  vekuri_de: "Deep ensemble (Vekuri 2025)", liu_xgb: "XGBoost (Liu 2025)", saits: "SAITS", csdi: "CSDI",
  chronos2: "Chronos-2 zero-shot", fluxfill: "FluxFill" };
const SPLIT_TEXT = {
  test: "Test tower: held out from training and from every model choice; each final FluxFill model is scored on the test towers once.",
  validation: "Validation tower: held out from training; used to choose between model versions." };
let WSPLIT = "validation";
function initWindows(idx) {
  windows = idx;
  if (!idx || !idx.windows || !idx.windows.length) {
    $("#w-body").innerHTML = '<div class="pending">No windows are exported yet. Windows appear here once the tower dataset build finishes and the first methods are scored.</div>';
    return;
  }
  const splitOf = (w) => w.split || "validation";
  const splits = new Set(idx.windows.map(splitOf));
  if (!splits.has(WSPLIT)) WSPLIT = [...splits][0];
  for (const b of $("#w-split").querySelectorAll("button")) {
    b.disabled = !splits.has(b.dataset.split);
    b.setAttribute("aria-checked", String(b.dataset.split === WSPLIT));
    b.addEventListener("click", () => { if (b.disabled) return; WSPLIT = b.dataset.split; for (const o of $("#w-split").querySelectorAll("button")) o.setAttribute("aria-checked", String(o === b)); pickSplit(); });
  }
  const pickSplit = () => {
    opt($("#w-site"), [...new Set(idx.windows.filter((w) => splitOf(w) === WSPLIT).map((w) => w.site))].sort());
    refresh();
  };
  const refresh = () => {
    const ws = idx.windows.filter((w) => w.site === $("#w-site").value && splitOf(w) === WSPLIT);
    opt($("#w-win"), ws.map((w) => w.file), ws.map((w) => `${w.start} to ${w.end}`));
    const w = ws[0];
    opt($("#w-mask"), w ? w.masks : []);
    const ms = w ? w.methods : [];
    opt($("#w-method"), ms, ms.map((m) => METHOD_NAMES[m] || m));
    const ff = ["fluxfill"].find((m) => ms.includes(m));
    if (ff) $("#w-method").value = ff;
    drawWindow();
  };
  $("#w-site").addEventListener("change", refresh);
  for (const id of ["#w-win", "#w-mask", "#w-method", "#w-zoom"]) $(id).addEventListener("change", drawWindow);
  pickSplit();
}

// Role of a row under the selected mask: an input the methods see, or hidden (and then filled and scored).
function rowRole(v, spans, sc) {
  if (!v.values.some((x) => x !== null)) return "not measured in this window";
  if (!spans || !spans.length) return "input: visible to every method";
  if (sc && sc.n) return "hidden: filled, then scored against the measurement";
  if (sc && sc.filled === 0) return "hidden: this method does not fill it";
  return "hidden: no measurement inside the hidden span";
}

async function drawWindow() {
  if (!windows || !windows.windows || !windows.windows.length || $("#p-windows").hidden) return;
  const file = $("#w-win").value;
  if (!file) return;
  const w = winCache[file] || (winCache[file] = await getJSONgz("data/windows/" + file));
  if (!w) { $("#w-body").innerHTML = '<div class="pending">This window could not be loaded.</div>'; return; }
  const mask = $("#w-mask").value, method = $("#w-method").value;
  const body = $("#w-body");
  body.innerHTML = "";
  const T = w.t.length;
  const ms = w.masks[mask] || {};
  // x-range: the whole window, or the union of masked spans with 25% padding on each side
  let x0 = 0, x1 = T - 1;
  if ($("#w-zoom").value === "mask") {
    const sp = Object.values(ms).flat();
    if (sp.length) { const a = Math.min(...sp.map((x) => x[0])), b = Math.max(...sp.map((x) => x[1])); const pad = Math.max(48, Math.round((b - a) * 0.25)); x0 = Math.max(0, a - pad); x1 = Math.min(T - 1, b + pad); }
  }
  const sc = ((w.scores || {})[method] || {})[mask] || {};
  const split = w.split || "validation";
  const nHid = w.variables.filter((v) => (ms[v.key] || []).length).length;
  const ctx = w.context_years === undefined ? "the tower's other years as site memory; the window's own years are never in the context" : w.context_years.length ? `the tower's other years (${w.context_years[0]} to ${w.context_years[w.context_years.length - 1]}, ${w.context_years.length} years) as site memory; the window's own years are never in the context` : "no other years of this tower";
  $("#w-info").innerHTML = `<p><span class="split-tag">${esc(split)}</span>${esc(SPLIT_TEXT[split] || "")}</p>
    <p><b>Mask.</b> ${esc(mask)}: ${nHid} of ${w.variables.length} rows hidden in part or in full; the rest are inputs.</p>
    <p><b>Context.</b> ${esc(METHOD_NAMES[method] || method)} reads the visible values of this window and ${ctx}. FluxFill's weights were trained on the training towers only.</p>
    ${w.note ? `<p class="note">${esc(w.note)}</p>` : ""}`;
  const rows = w.variables.filter((v) => sc[v.key]);
  body.insertAdjacentHTML("beforeend", rows.length ? `<div class="table-wrap"><table><thead><tr><th>Hidden variable</th><th class="num">CRPS</th><th class="num">MAE</th><th class="num">80% coverage</th><th class="num">95% coverage</th><th class="num">filled</th><th class="num">hidden measured values</th></tr></thead><tbody>${rows.map((v) => { const r = sc[v.key]; if (!r.n) return `<tr><td>${v.label} <span class="note">${v.units || ""}</span></td><td class="num" colspan="5">not filled by this method</td><td class="num">–</td></tr>`; return `<tr><td>${v.label} <span class="note">${v.units || ""}</span></td><td class="num">${fmt(r.crps, 3)}</td><td class="num">${fmt(r.mae, 3)}</td><td class="num">${fmt(100 * r.cov80, 1)}%</td><td class="num">${fmt(100 * r.cov95, 1)}%</td><td class="num">${fmt(100 * (r.filled ?? 1), 0)}%</td><td class="num">${fmt(r.n)}</td></tr>`; }).join("")}</tbody></table></div><p class="note">Scores on the hidden, measured values of this window only. CRPS and MAE in each variable's units.</p>` : "");
  let group = null;
  for (const v of w.variables) {
    const g = v.provenance === "reanalysis" ? "rean" : (["NEE", "LE", "H", "FCH4", "G"].includes(v.key) ? "flux" : "met");
    if (g !== group) {
      group = g;
      if (g === "rean") body.insertAdjacentHTML("beforeend", `<h3>Reanalysis drivers</h3><p class="note">ERA5, bias-corrected to the tower by ONEFlux. They are inputs unless the mask hides them. ONEFlux fits the correction (a slope and offset per variable) against this tower's measured meteorology over its whole record, so these rows carry a few site-level coefficients estimated partly from hidden periods; every method receives the same rows. Mask M4 removes them entirely; its Results row shows what they add.</p>`);
      else body.insertAdjacentHTML("beforeend", `<h3>${g === "flux" ? "Fluxes (tower)" : "Tower meteorology and soil"}</h3>`);
    }
    const m = ms[v.key];                                   // array of [start, stop) index pairs hidden
    const segs = ((w.predictions[method] || {})[mask] || {})[v.key]; // [{a, mean, q025, q10, q90, q975}]
    body.appendChild(panel(w, v, m, expand(segs, T), T, x0, x1, rowRole(v, m, sc[v.key])));
  }
}

function expand(segs, T) {
  if (!segs || !segs.length) return null;
  const out = {};
  for (const k of ["mean", "q025", "q10", "q90", "q975"]) out[k] = new Array(T).fill(null);
  for (const sg of segs) for (const k of Object.keys(out)) sg[k].forEach((x, j) => { out[k][sg.a + j] = x; });
  return out;
}

function niceTicks(lo, hi, n) {
  const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => span / s <= n) || 10 * mag;
  const out = []; for (let x = Math.ceil(lo / step) * step; x <= hi + 1e-9; x += step) out.push(+x.toFixed(10));
  return { ticks: out, step };
}

function panel(w, v, maskSpans, pred, T, x0 = 0, x1 = T - 1, role = "") {
  const W = 1080, H = 120, ml = 64, mr = 14, mt = 18, mb = 22;
  const iw = W - ml - mr, ih = H - mt - mb;
  const vals = v.values;
  const all = vals.slice(x0, x1 + 1).filter((x) => x !== null);
  if (pred) for (const k of ["q025", "q975"]) for (const x of pred[k].slice(x0, x1 + 1)) if (x !== null) all.push(x);
  let lo = Math.min(...all), hi = Math.max(...all);
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  if (hi === lo) { hi += 1; lo -= 1; }
  const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
  const X = (i) => ml + ((i - x0) / (x1 - x0)) * iw, Y = (y) => mt + (1 - (y - lo) / (hi - lo)) * ih;
  const hidden = new Uint8Array(T);
  for (const [a, b] of (maskSpans || [])) for (let i = a; i < b; i++) hidden[i] = 1;
  const path = (arr, keep) => { let d = "", on = false; for (let i = x0; i <= x1; i++) { const y = arr[i]; if (y === null || !keep(i)) { on = false; continue; } d += (on ? "L" : "M") + X(i).toFixed(1) + " " + Y(y).toFixed(1); on = true; } return d; };
  const band = (a, b) => { let d = "", seg = []; const flush = () => { if (seg.length > 1) { d += "M" + seg.map((i) => X(i).toFixed(1) + " " + Y(a[i]).toFixed(1)).join("L") + "L" + seg.slice().reverse().map((i) => X(i).toFixed(1) + " " + Y(b[i]).toFixed(1)).join("L") + "Z"; } seg = []; };
    for (let i = x0; i <= x1; i++) { if (a[i] === null || b[i] === null || !hidden[i]) { flush(); continue; } seg.push(i); } flush(); return d; };
  const ns = "http://www.w3.org/2000/svg";
  const wrap = document.createElement("div"); wrap.className = "viz";
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${v.label} over the window">`;
  for (const [a, b] of (maskSpans || [])) { const aa = Math.max(a, x0), bb = Math.min(b - 1, x1); if (bb >= aa) s += `<rect x="${X(aa)}" y="${mt}" width="${Math.max(1, X(bb) - X(aa))}" height="${ih}" fill="${css("--mask")}"/>`; }
  const nt = niceTicks(lo, hi, 4), dec = nt.step < 1 ? Math.min(3, Math.ceil(-Math.log10(nt.step))) : 0;
  s += '<g class="grid">' + nt.ticks.map((y) => `<line x1="${ml}" x2="${W - mr}" y1="${Y(y)}" y2="${Y(y)}"/>`).join("") + '</g><g class="axis">';
  s += nt.ticks.map((y) => `<text x="${ml - 6}" y="${Y(y) + 4}" text-anchor="end">${fmt(y, dec)}</text>`).join("");
  const spanDays = (x1 - x0) / 48, every = spanDays > 60 ? 30 : spanDays > 20 ? 7 : spanDays > 6 ? 2 : 1;
  for (let d = Math.ceil(x0 / 48 / every) * every; d * 48 <= x1; d += every) s += `<text x="${X(d * 48)}" y="${H - 6}" text-anchor="middle">${w.day_labels ? w.day_labels[d] || "" : "day " + d}</text>`;
  s += "</g>";
  if (pred) {
    s += `<path d="${band(pred.q025, pred.q975)}" fill="${css("--band-2")}"/>`;
    s += `<path d="${band(pred.q10, pred.q90)}" fill="${css("--band-1")}"/>`;
  }
  s += `<path d="${path(vals, (i) => hidden[i])}" fill="none" stroke="${css("--hidden")}" stroke-width="1"/>`;
  s += `<path d="${path(vals, (i) => !hidden[i])}" fill="none" stroke="${css("--ink")}" stroke-width="1"/>`;
  if (pred) s += `<path d="${path(pred.mean, (i) => hidden[i])}" fill="none" stroke="${css("--series-1")}" stroke-width="1.5"/>`;
  s += `<text class="lab-v" x="${ml}" y="12">${v.label}</text><text class="lab" x="${ml + 8 + v.label.length * 6.6}" y="12">${v.units || ""}</text><text class="lab" x="${W - mr}" y="12" text-anchor="end">${esc(role)}</text>`;
  s += `<line class="xh" x1="0" x2="0" y1="${mt}" y2="${mt + ih}" stroke="${css("--quiet")}" stroke-width="1" visibility="hidden"/>`;
  s += `<rect x="${ml}" y="${mt}" width="${iw}" height="${ih}" fill="transparent"/></svg><div class="tooltip"></div>`;
  wrap.innerHTML = s;
  const svg = wrap.querySelector("svg"), tip = wrap.querySelector(".tooltip"), xh = wrap.querySelector(".xh");
  svg.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect(), sx = (e.clientX - r.left) * (W / r.width);
    const i = Math.max(x0, Math.min(x1, Math.round(x0 + ((sx - ml) / iw) * (x1 - x0))));
    xh.setAttribute("x1", X(i)); xh.setAttribute("x2", X(i)); xh.setAttribute("visibility", "visible");
    let h = `<b>${w.t[i]}</b><br>${hidden[i] ? "hidden from the method" : "input (visible)"}<br>${hidden[i] ? "ground truth" : "measured"}: ${fmt(vals[i], 2)} ${v.units || ""}`;
    if (pred && hidden[i]) h += `<br>predicted: ${fmt(pred.mean[i], 2)}<br>80%: ${fmt(pred.q10[i], 2)} to ${fmt(pred.q90[i], 2)}<br>95%: ${fmt(pred.q025[i], 2)} to ${fmt(pred.q975[i], 2)}`;
    tip.innerHTML = h; tip.style.display = "block";
    const px = (X(i) / W) * r.width; tip.style.left = Math.min(px + 12, r.width - 200) + "px"; tip.style.top = "8px";
  });
  svg.addEventListener("mouseleave", () => { tip.style.display = "none"; xh.setAttribute("visibility", "hidden"); });
  return wrap;
}

// ---------- shared ----------
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function fmtMetric(v, metric) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  if (metric === "fill") return `${fmt(100 * v, 0)}%`;
  if (metric && (metric.startsWith("cov") || metric === "r2")) return fmt(v, 3);
  if (metric === "wape") return `${fmt(v, 1)}%`;
  const a = Math.abs(v);
  return fmt(v, a < 10 ? 3 : a < 100 ? 2 : a < 1000 ? 1 : 0);
}
const fmtInt = (v) => (v === null || v === undefined) ? "–" : Number(v).toLocaleString("en-US");
function fmtSI(v) {
  if (!Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e9) return fmt(v / 1e9, a >= 1e10 ? 0 : 1) + "B";
  if (a >= 1e6) return fmt(v / 1e6, a >= 1e7 ? 0 : 1) + "M";
  if (a >= 1e3) return fmt(v / 1e3, a >= 1e4 ? 0 : 1) + "K";
  return fmt(v, a < 10 ? 2 : 0);
}

// ---------- results ----------
let RES = null, SCALING = null, FLUX = "NEE", GAPMETRIC = "crps", SUMKIND = "sum-30-day", SUMMETRIC = "cov95", HIGHLIGHT = null;
const VALUE_METRICS = ["crps", "cov95", "cov80", "cov50", "mae", "rmse", "mse", "bias", "r2", "wape"];
const SUM_METRICS = ["crps", "cov95", "mae", "rmse", "bias", "r2", "wape"];
const SUM_LABEL = { "sum-daily": "daily sums", "sum-30-day": "30-day sums", "sum-window": "180-day window sums" };

function metricDef(id, units) {
  const m = RES.metrics[id];
  const better = { lower: "lower is better", higher: "higher is better, 1 is perfect", zero: "0 is unbiased" }[m.better] || `${m.nominal} is calibrated`;
  const u = m.units === "unit" ? units : m.units === "unit²" ? `(${units})²` : m.units === "%" ? "percent" : m.units === "share" ? "share of values" : "no unit";
  return `<b>${esc(m.name)}</b>: ${esc(m.long)}; ${esc(u)}; ${better}.`;
}

function renderResults(res) {
  RES = res;
  if (!res || !res.summary || !res.summary.length) {
    $("#r-findings").innerHTML = '<div class="pending">No method has been scored on FluxFill-Bench yet. Scores appear here as each baseline and the model are run on the frozen benchmark.</div>';
    for (const id of ["#r-tables-h", "#r-controls", "#metrics"]) $(id).hidden = true;
    return;
  }
  const sc = res.scope;
  const fams = sc.families.map((f) => f.replace(/^M2-(\d+)d$/, "$1 d")).join(", ");
  $("#r-scope").innerHTML = `<span class="pill">${esc(res.label)} sites</span> ${sc.windows} windows of 180 days at ${sc.sites} ${esc(sc.split)} sites (${sc.site_list.map(esc).join(", ")}); regime ${sc.regimes.join(", ")}; mask families ${fams}; ${sc.methods} methods on identical windows, masks and scoring code. ${sc.split === "test" ? "Test sites of the frozen split, at least 50 km from every training site, never used for training or model selection; masks verified against the frozen release manifest. Methods not yet scored on the test release are absent from these tables." : "These sites are held out from training and used for model selection."} Generated ${esc(res.generated_utc)} UTC.${res.note ? " " + esc(res.note) : ""} <a href="#metrics" data-tab="results">Metric definitions and formulas</a>.`;
  $("#r-findings").innerHTML = `<div class="findings"><h3>Findings on ${esc(res.label.toLowerCase())} sites</h3><ol>${res.findings.map((f) => `<li><b>${esc(f.head)}</b> ${esc(f.text)}<span class="data">Data: ${esc(f.data)}</span></li>`).join("")}</ol></div>`;
  $("#r-flux").innerHTML = res.fluxes.map((f) => `<button role="radio" aria-checked="${f.id === FLUX}" data-flux="${f.id}">${f.id}</button>`).join("");
  $("#r-flux").addEventListener("click", (e) => { const b = e.target.closest("button[data-flux]"); if (!b) return; FLUX = b.dataset.flux; for (const x of $("#r-flux").children) x.setAttribute("aria-checked", x.dataset.flux === FLUX); renderTables(); });
  renderGlossary();
  renderTables();
  drawChart();
}

function renderGlossary() {
  const role = { primary: "Primary score", calibration: "Calibration", point: "Point accuracy (secondary)", coverage: "Coverage of the method" };
  const ids = ["crps", "cov95", "cov80", "cov50", "mae", "mse", "rmse", "bias", "r2", "wape", "fill"];
  $("#r-glossary").innerHTML = `<p>Every score is computed on hidden measured values only (quality flag 0, never gap-filled values), pooled over the values of a table cell. Point scores use each method's predictive mean ŷ, the mean of its 64 samples. Each column compares the methods on the cells (window × gap × variable) that every compared method fills completely, so R² and WAPE always rest on complete cells. Units: NEE in µmol m⁻² s⁻¹ (negative is uptake), FCH4 in nmol m⁻² s⁻¹, LE and H in W m⁻²; sums in g C m⁻², mg CH₄ m⁻² and MJ m⁻².</p><div class="gloss">${ids.map((id) => { const m = RES.metrics[id]; return `<div class="g-item"><div class="g-role">${role[m.role] || ""}</div><div class="g-name">${esc(m.name)} <span class="note">${esc(m.long)}</span></div><code>${esc(m.formula)}</code><p>${esc(m.text)}</p></div>`; }).join("")}</div>`;
}

function headCell(c) { return `<th class="num" scope="col">${esc(c.label)}${c.n !== undefined ? `<span class="sub">n ${fmtInt(c.n)}</span>` : ""}</th>`; }

function rowsHTML(t, metricOf) {
  return t.rows.map((r) => `<tr class="${r.ours ? "ours" : ""}"><th scope="row">${esc(r.label)}</th>${r.cells.map((c, j) => {
    if (c.v === null || c.v === undefined) return `<td class="num muted">${esc(c.t || "–")}</td>`;
    const ties = t.best_all ? t.best_all[j] : [t.best[j]];
    const best = (ties || []).some((i) => i !== null && i !== undefined && t.rows[i] === r);
    return `<td class="num${best ? " best" : ""}"${best ? ' title="best in column (ties at the shown precision are all marked)"' : ""}>${fmtMetric(c.v, metricOf(j))}</td>`;
  }).join("")}</tr>`).join("");
}

function gapTable(t, units, extra) {
  return `<div class="tblock"><h4 class="headline">${esc(t.headline)}</h4><p class="sub-title">${esc(t.title)}${extra || ""}</p>
  <div class="table-wrap"><table class="results"><thead><tr><th scope="col">Method</th>${t.columns.map(headCell).join("")}</tr></thead><tbody>${rowsHTML(t, () => t.metric)}</tbody></table></div>
  <p class="defs">Columns: gap length of the M2 block${t.columns.some((c) => c.n !== undefined) ? "; n is the number of values (or block sums) each column rests on" : ""}. ${t.metric === "fill" ? metricDef("fill", "") : metricDef(t.metric, units)} Bold: best in the column. ${esc(t.note || "")}</p></div>`;
}

function renderTables() {
  const flux = RES.fluxes.find((f) => f.id === FLUX) || RES.fluxes[0];
  FLUX = flux.id;
  const s = RES.summary.find((t) => t.flux === FLUX);
  if (s) {
    const ms = s.metrics;
    const prob = ms.filter((m) => RES.metrics[m].role !== "point").length;
    $("#r-summary").innerHTML = `<div class="tblock"><h4 class="headline">${esc(s.headline)}</h4>
    <p class="sub-title">${esc(s.title)}: ${fmtInt(s.cells)} cells, ${fmtInt(s.n)} half-hourly values, ${s.windows} windows at ${s.sites} ${esc(RES.scope.split)} sites, gaps of ${s.families.map((f) => f.replace(/^M2-(\d+)d$/, "$1 d")).join(", ")}.</p>
    <div class="table-wrap"><table class="results"><thead><tr><th></th><th class="grp" colspan="${prob}">Probabilistic (primary)</th><th class="grp" colspan="${ms.length - prob}">Point, predictive mean (secondary)</th></tr>
    <tr><th scope="col">Method</th>${ms.map((m) => `<th class="num" scope="col">${esc(RES.metrics[m].name)}</th>`).join("")}</tr></thead><tbody>${rowsHTML(s, (j) => ms[j])}</tbody></table></div>
    <p class="defs">${ms.map((m) => metricDef(m, flux.units)).join(" ")} Bold: best in the column (lowest error, highest R², bias nearest 0, coverage nearest its nominal level); values that tie at the shown precision are all bold.</p></div>`;
  }
  const g = (RES.by_gap[FLUX] || {});
  const gopts = VALUE_METRICS.filter((m) => g[m]).map((m) => `<option value="${m}"${m === GAPMETRIC ? " selected" : ""}>${esc(RES.metrics[m].name)}</option>`).join("");
  $("#r-bygap").innerHTML = g[GAPMETRIC] ? `<div class="controls"><label>Metric, half-hourly values<select id="r-gm">${gopts}</select></label></div>${gapTable(g[GAPMETRIC], flux.units)}` : "";
  const sk = (RES.sums[FLUX] || {});
  if (sk[SUMKIND] && sk[SUMKIND][SUMMETRIC]) {
    const kopts = Object.keys(sk).map((k) => `<option value="${k}"${k === SUMKIND ? " selected" : ""}>${SUM_LABEL[k] || k}</option>`).join("");
    const mopts = SUM_METRICS.filter((m) => sk[SUMKIND][m]).map((m) => `<option value="${m}"${m === SUMMETRIC ? " selected" : ""}>${esc(RES.metrics[m].name)}</option>`).join("");
    $("#r-sums").innerHTML = `<div class="controls"><label>Sum<select id="r-sk">${kopts}</select></label><label>Metric, sums<select id="r-sm">${mopts}</select></label></div>${gapTable(sk[SUMKIND][SUMMETRIC], flux.sum_units)}`;
  } else $("#r-sums").innerHTML = "";
  $("#r-fill").innerHTML = RES.fill[FLUX] ? gapTable(RES.fill[FLUX], "") : "";
  const on = (id, f) => { const el = $(id); if (el) el.addEventListener("change", (e) => { f(e.target.value); renderTables(); }); };
  on("#r-gm", (v) => { GAPMETRIC = v; }); on("#r-sk", (v) => { SUMKIND = v; }); on("#r-sm", (v) => { SUMMETRIC = v; });
}

// The current FluxFill: the last "ours" method of the results file (results_dashboard.py lists earlier versions first).
function currentFluxFill() {
  const ours = RES.methods.filter((m) => m.ours);
  return ours.find((m) => m.id === "fluxfill") || ours.find((m) => m.id === "fluxfill-ref") || ours[ours.length - 1];
}

// Figure: CRPS relative to climatology by gap length, one panel per flux.
function drawChart() {
  const box = $("#r-chart");
  if (!RES || !RES.chart || !box || $("#p-results").hidden) return;
  const ch = RES.chart, F = ch.fluxes;
  const base = RES.methods.filter((m) => !m.ours && m.id !== "climatology");
  if (HIGHLIGHT && !base.some((m) => m.id === HIGHLIGHT)) HIGHLIGHT = null;   // not scored on this split
  if (!HIGHLIGHT && base.some((m) => m.id === "liu_xgb")) HIGHLIGHT = "liu_xgb";   // default: XGBoost (Liu 2025)
  if (!HIGHLIGHT) {   // else the baseline with the lowest mean CRPS relative to climatology over every panel and gap
    let best = null, bv = Infinity;
    for (const m of base) {
      const r = []; for (const f of Object.values(F)) { const c = f.series.find((s) => s.method === "climatology"), s = f.series.find((x) => x.method === m.id); s.crps.forEach((v, j) => { if (v !== null && c.crps[j]) r.push(v / c.crps[j]); }); }
      const mean = r.length ? r.reduce((a, b) => a + b, 0) / r.length : Infinity;
      if (r.length >= 10 && mean < bv) { bv = mean; best = m.id; }
    }
    HIGHLIGHT = best || (base[0] && base[0].id);
  }
  const hl = RES.methods.find((m) => m.id === HIGHLIGHT);
  const cur = currentFluxFill();
  const style = {};
  if (cur) style[cur.id] = { c: "--series-1", shape: "circle" };
  style[HIGHLIGHT] = { c: "--series-2", shape: "diamond" };
  box.innerHTML = `<div class="figure"><h3 class="headline">${esc(ch.headline)}</h3>
  <p class="sub-title"><b>Figure 1.</b> CRPS of half-hourly values relative to climatology, by gap length, on ${esc(RES.label.toLowerCase())} sites; 1 is climatology, below 1 is better. Each gap length compares the methods on the cells they all fill completely (n values under each label). Hover for every method's CRPS.</p>
  <div class="controls"><label>Compare with<select id="r-hl">${base.map((m) => `<option value="${m.id}"${m.id === HIGHLIGHT ? " selected" : ""}>${esc(m.label)}</option>`).join("")}</select></label></div>
  <div class="legend chart-legend"><span><svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="${css("--series-1")}" stroke-width="2"/><circle cx="11" cy="5" r="3.5" fill="${css("--series-1")}"/></svg>${esc(cur ? cur.label : "FluxFill")}</span><span><svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="${css("--series-2")}" stroke-width="2"/><path d="M11 0.5 L15.5 5 L11 9.5 L6.5 5 Z" fill="${css("--series-2")}"/></svg>${esc(hl ? hl.label : "")}</span><span><svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="${css("--hidden")}" stroke-width="1"/></svg>other methods and earlier FluxFill versions</span><span><svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="${css("--ink")}" stroke-width="1"/></svg>climatology = 1</span></div>
  <div class="panels">${Object.keys(F).map((k) => `<div class="viz panel" data-flux="${k}"></div>`).join("")}</div></div>`;
  $("#r-hl").addEventListener("change", (e) => { HIGHLIGHT = e.target.value; drawChart(); });
  for (const el of box.querySelectorAll(".panel")) panelChart(el, el.dataset.flux, F[el.dataset.flux], style);
}

function panelChart(el, flux, f, style) {
  const W = 480, H = 250, ml = 44, mr = 16, mt = 26, mb = 40, iw = W - ml - mr, ih = H - mt - mb;
  const clim = f.series.find((s) => s.method === "climatology");
  const J = f.families.length;
  const ratio = (s) => s.crps.map((v, j) => (v !== null && clim && clim.crps[j]) ? v / clim.crps[j] : null);
  const all = []; for (const s of f.series) for (const r of ratio(s)) if (r !== null) all.push(r);
  const emph = f.series.filter((s) => style[s.method]).flatMap(ratio).filter((r) => r !== null);
  let lo = Math.min(0.8, ...emph) * 0.97, hi = Math.min(Math.max(1.2, ...emph) * 1.03, Math.max(1.6, ...emph.map((x) => x * 1.03)));
  hi = Math.min(hi, 2.0); lo = Math.max(0, lo);
  const X = (j) => ml + (J === 1 ? iw / 2 : (j / (J - 1)) * iw), Y = (v) => mt + (1 - (Math.min(v, hi) - lo) / (hi - lo)) * ih;
  const nt = niceTicks(lo, hi, 4);
  const clip = `clip-${flux}`;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${flux}: CRPS relative to climatology by gap length"><defs><clipPath id="${clip}"><rect x="${ml - 8}" y="${mt}" width="${iw + 16}" height="${ih}"/></clipPath></defs>`;
  s += `<text class="lab-v" x="${ml}" y="14">${flux}</text><text class="lab" x="${ml + 8 + flux.length * 8}" y="14">CRPS / climatology CRPS</text>`;
  s += '<g class="grid">' + nt.ticks.map((y) => `<line x1="${ml}" x2="${W - mr}" y1="${Y(y)}" y2="${Y(y)}"/>`).join("") + "</g><g class=\"axis\">";
  s += nt.ticks.map((y) => `<text x="${ml - 6}" y="${Y(y) + 4}" text-anchor="end">${fmt(y, nt.step < 0.1 ? 2 : 1)}</text>`).join("");
  // Axis ticks use the family code (M1, M3, M4, M5) where the full label is long; the tooltip has the full label.
  const tick = (lab) => { const m = lab.match(/\((M\d)\)/); return m ? m[1] : /no reanalysis/.test(lab) ? "M4" : lab; };
  s += f.families.map((lab, j) => `<text x="${X(j)}" y="${H - mb + 16}" text-anchor="middle">${esc(tick(lab))}</text><text x="${X(j)}" y="${H - mb + 30}" text-anchor="middle" class="tiny">n ${fmtSI(f.n[j])}</text>`).join("") + "</g>";
  s += `<line x1="${ml}" x2="${W - mr}" y1="${Y(1)}" y2="${Y(1)}" stroke="${css("--ink")}" stroke-width="1"/>`;
  const Yu = (v) => mt + (1 - (v - lo) / (hi - lo)) * ih;   // unclamped: lines leave the plot and are clipped
  const line = (r) => { let d = "", on = false; r.forEach((v, j) => { if (v === null) { on = false; return; } d += (on ? "L" : "M") + X(j).toFixed(1) + " " + Yu(v).toFixed(1); on = true; }); return d; };
  for (const se of f.series) if (!style[se.method] && se.method !== "climatology") s += `<path d="${line(ratio(se))}" fill="none" stroke="${css("--hidden")}" stroke-width="1" stroke-opacity="0.55" clip-path="url(#${clip})"/>`;
  const order = [HIGHLIGHT, ...Object.keys(style).filter((k) => k !== HIGHLIGHT)];
  for (const id of order) {
    const se = f.series.find((x) => x.method === id); if (!se || !style[id]) continue;
    const col = css(style[id].c), r = ratio(se);
    s += `<path d="${line(r)}" fill="none" stroke="${col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" clip-path="url(#${clip})"/>`;
    r.forEach((v, j) => {
      if (v === null) return;
      const x = X(j), y = Y(v), off = v > hi;
      const ring = `stroke="${css("--ground")}" stroke-width="2"`;
      if (off) s += `<path d="M${x - 5} ${mt + 6} L${x} ${mt - 2} L${x + 5} ${mt + 6} Z" fill="${col}" ${ring}/>`;
      else if (style[id].shape === "square") s += `<rect x="${x - 4}" y="${y - 4}" width="8" height="8" fill="${col}" ${ring}/>`;
      else if (style[id].shape === "diamond") s += `<path d="M${x} ${y - 5.5} L${x + 5.5} ${y} L${x} ${y + 5.5} L${x - 5.5} ${y} Z" fill="${col}" ${ring}/>`;
      else s += `<circle cx="${x}" cy="${y}" r="4.5" fill="${col}" ${ring}/>`;
    });
  }
  s += `<line class="xh" x1="0" x2="0" y1="${mt}" y2="${mt + ih}" stroke="${css("--quiet")}" stroke-width="1" visibility="hidden"/>`;
  s += `<rect x="${ml - 20}" y="${mt}" width="${iw + 40}" height="${ih}" fill="transparent"/></svg><div class="tooltip"></div>`;
  el.innerHTML = s;
  const svg = el.querySelector("svg"), tip = el.querySelector(".tooltip"), xh = el.querySelector(".xh");
  const move = (cx) => {
    const r = svg.getBoundingClientRect(), sx = (cx - r.left) * (W / r.width);
    const j = Math.max(0, Math.min(J - 1, Math.round(J === 1 ? 0 : ((sx - ml) / iw) * (J - 1))));
    xh.setAttribute("x1", X(j)); xh.setAttribute("x2", X(j)); xh.setAttribute("visibility", "visible");
    const rows = f.series.map((se) => ({ se, v: se.crps[j] })).filter((x) => x.v !== null).sort((a, b) => a.v - b.v);
    tip.innerHTML = `<b>${flux}, ${f.families[j]} gap</b> <span class="note">n ${fmtInt(f.n[j])}</span><table class="tt">${rows.map(({ se, v }) => `<tr class="${style[se.method] ? "em" : ""}"><td>${esc(se.short)}</td><td class="num">${fmtMetric(v, "crps")}</td><td class="num">${clim && clim.crps[j] ? fmt(v / clim.crps[j], 2) : ""}</td></tr>`).join("")}</table><span class="note">CRPS in ${esc(f.units)}; ratio to climatology</span>`;
    tip.style.display = "block";
    const px = (X(j) / W) * r.width; tip.style.left = Math.max(0, Math.min(px + 14, r.width - 230)) + "px"; tip.style.top = "30px";
  };
  svg.addEventListener("mousemove", (e) => move(e.clientX));
  svg.addEventListener("touchstart", (e) => move(e.touches[0].clientX), { passive: true });
  svg.addEventListener("mouseleave", () => { tip.style.display = "none"; xh.setAttribute("visibility", "hidden"); });
}

// ---------- scaling ----------
function fitPower(xs, ys) {
  // L = a x^(-alpha) + c: for each c on a grid in [0, min L), ordinary least squares of ln(L - c) on ln x; keep the
  // fit with the smallest squared error in L. Needs 3 or more points.
  if (xs.length < 3) return null;
  const ymin = Math.min(...ys); let best = null;
  for (let k = 0; k < 400; k++) {
    const c = (k / 400) * ymin * 0.999;
    const lx = xs.map(Math.log), ly = ys.map((y) => Math.log(y - c));
    const mx = lx.reduce((a, b) => a + b) / lx.length, my = ly.reduce((a, b) => a + b) / ly.length;
    let sxy = 0, sxx = 0; lx.forEach((x, i) => { sxy += (x - mx) * (ly[i] - my); sxx += (x - mx) ** 2; });
    if (sxx === 0) return null;
    const slope = sxy / sxx, a = Math.exp(my - slope * mx), alpha = -slope;
    if (alpha <= 0) continue;
    const sse = xs.reduce((acc, x, i) => acc + (a * Math.pow(x, -alpha) + c - ys[i]) ** 2, 0);
    if (!best || sse < best.sse) best = { a, alpha, c, sse, n: xs.length };
  }
  return best;
}

function drawScaling() {
  const box = $("#r-scaling");
  if (!box || $("#p-results").hidden) return;
  const runs = Array.isArray(SCALING) ? SCALING : (SCALING && SCALING.runs) || [];
  const intro = '<p>Validation CRPS (standardised units, all variables and fluxes only) against model size and against the number of training sites, from <code>model/scaling/scaling.json</code>. A power law L = a·x<sup>−α</sup> + c is fitted to each series once it has 3 or more points. Points are runs trained in full on the whole training set; the size probes at matched steps are under Component study.</p>';
  if (!runs.length) { box.innerHTML = intro + '<div class="pending">Scaling runs are in progress. Points appear here as each run finishes; no point is plotted before it is measured.</div>'; return; }
  const axes = [{ id: "model", x: "n_params", lab: "parameters", title: "Validation CRPS against model size" }, { id: "data", x: "n_train_sites", lab: "training sites", title: "Validation CRPS against training sites" }];
  box.innerHTML = intro + `<div class="legend chart-legend"><span><i style="background:${css("--series-1")}"></i>all variables</span><span><i style="background:${css("--series-2")}"></i>fluxes only</span></div><div class="panels">${axes.map((a) => `<div class="viz panel" data-axis="${a.id}"></div>`).join("")}</div>`;
  for (const a of axes) {
    const el = box.querySelector(`[data-axis="${a.id}"]`);
    const pts = runs.filter((r) => r.axis === a.id && Number.isFinite(r[a.x]) && r[a.x] > 0);
    const series = [["val_crps", "--series-1", "all variables"], ["val_crps_flux", "--series-2", "fluxes"]].map(([k, c, lab]) => {
      const p = pts.filter((r) => Number.isFinite(r[k]) && r[k] > 0).sort((u, v) => u[a.x] - v[a.x]);
      return { k, c, lab, p, fit: fitPower(p.map((r) => r[a.x]), p.map((r) => r[k])) };
    });
    const head = (() => {
      const f = series[0].fit;
      if (!series[0].p.length) return `${a.title}: no run on this axis yet.`;
      if (!f) return `${a.title}: ${series[0].p.length} of the 3 points a power-law fit needs.`;
      return `Validation CRPS falls as ${a.lab}<sup>−${fmt(f.alpha, 2)}</sup> toward ${fmt(f.c, 3)} over ${f.n} runs.`;
    })();
    if (!pts.length) { el.innerHTML = `<h4 class="headline">${head}</h4><div class="pending">No ${a.id}-scaling run has finished yet.</div>`; continue; }
    const W = 480, H = 250, ml = 50, mr = 16, mt = 16, mb = 40, iw = W - ml - mr, ih = H - mt - mb;
    const xs = pts.map((r) => r[a.x]), ys = series.flatMap((s) => s.p.map((r) => r[s.k]));
    let x0 = Math.log10(Math.min(...xs)) - 0.15, x1 = Math.log10(Math.max(...xs)) + 0.15, y0 = Math.log10(Math.min(...ys)) - 0.05, y1 = Math.log10(Math.max(...ys)) + 0.05;
    if (x1 - x0 < 0.5) { x0 -= 0.25; x1 += 0.25; } if (y1 - y0 < 0.1) { y0 -= 0.05; y1 += 0.05; }
    const X = (v) => ml + ((Math.log10(v) - x0) / (x1 - x0)) * iw, Y = (v) => mt + (1 - (Math.log10(v) - y0) / (y1 - y0)) * ih;
    const decTicks = (a0, a1) => { const out = []; for (let e = Math.floor(a0); e <= Math.ceil(a1); e++) for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (Math.log10(v) >= a0 && Math.log10(v) <= a1) out.push(v); } return out; };
    let s = `<h4 class="headline">${head}</h4><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${a.title}"><g class="grid">`;
    const yt = decTicks(y0, y1).length >= 2 ? decTicks(y0, y1) : [10 ** y0, 10 ** ((y0 + y1) / 2), 10 ** y1];
    s += yt.map((v) => `<line x1="${ml}" x2="${W - mr}" y1="${Y(v)}" y2="${Y(v)}"/>`).join("") + '</g><g class="axis">';
    s += yt.map((v) => `<text x="${ml - 6}" y="${Y(v) + 4}" text-anchor="end">${fmt(v, 3)}</text>`).join("");
    s += decTicks(x0, x1).map((v) => `<text x="${X(v)}" y="${H - mb + 16}" text-anchor="middle">${fmtSI(v)}</text>`).join("");
    s += `<text x="${ml + iw / 2}" y="${H - 6}" text-anchor="middle">${a.lab} (log scale)</text></g>`;
    for (const se of series) {
      const col = css(se.c);
      if (se.fit) { let d = ""; for (let i = 0; i <= 60; i++) { const lv = x0 + (i / 60) * (x1 - x0), v = 10 ** lv, yv = se.fit.a * Math.pow(v, -se.fit.alpha) + se.fit.c; if (yv > 0) d += (d ? "L" : "M") + X(v).toFixed(1) + " " + Y(yv).toFixed(1); } s += `<path d="${d}" fill="none" stroke="${col}" stroke-width="1.5" stroke-opacity="0.7"/>`; }
      for (const r of se.p) s += `<circle class="pt" cx="${X(r[a.x])}" cy="${Y(r[se.k])}" r="4.5" fill="${col}" stroke="${css("--ground")}" stroke-width="2" data-run="${esc(r.run)}" data-k="${se.k}"><title>${esc(r.run)}: ${fmtInt(r[a.x])} ${a.lab}, CRPS ${fmt(r[se.k], 4)}${r.steps ? `, ${fmtInt(r.steps)} steps` : ""}${r.tokens_seen ? `, ${fmtSI(r.tokens_seen)} tokens` : ""}</title></circle>`;
    }
    s += "</svg>";
    const fits = series.filter((x) => x.fit).map((x) => `${x.lab}: L = ${fmt(x.fit.a, 3)}·x<sup>−${fmt(x.fit.alpha, 3)}</sup> + ${fmt(x.fit.c, 4)} (${x.fit.n} points)`);
    el.innerHTML = s + `<p class="defs">${fits.length ? "Fits, " + fits.join("; ") + "." : "A power-law fit needs 3 or more points."} Runs: ${pts.map((r) => esc(r.run)).join(", ")}.</p>`;
  }
}

// ---------- methods ----------
function renderMethods(m) {
  if (!m) { $("#m-cards").innerHTML = '<div class="pending">Method descriptions are not available in this build.</div>'; return; }
  const card = (b) => `<article class="card" id="m-${b.id}"><h4>${esc(b.name)} ${b.cite || ""}</h4>
    <dl><dt>Code</dt><dd>${esc(b.code)}</dd><dt>Version</dt><dd>${esc(b.version)}</dd><dt>Model</dt><dd>${esc(b.cls)}</dd>
    <dt>Settings</dt><dd><table class="kv">${b.hyper.map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join("")}</table></dd>
    <dt>Inputs</dt><dd>${esc(b.inputs)}</dd><dt>Trained on</dt><dd>${esc(b.training)}</dd><dt>Uncertainty</dt><dd>${esc(b.uncertainty)}</dd><dt>Limits</dt><dd>${esc(b.limits)}</dd></dl></article>`;
  $("#m-toc").innerHTML = [`<a href="#m-fluxfill-h" data-tab="methods">FluxFill</a>`, ...m.baselines.map((b) => `<a href="#m-${b.id}" data-tab="methods">${esc(b.name)}</a>`), `<a href="#m-exp-h" data-tab="methods">Experiments</a>`].join("");
  $("#m-fluxfill").innerHTML = (m.arch_html || "") + (m.fluxfill ? `<div class="cards one">${card(m.fluxfill)}</div>` : "");
  $("#m-cards").innerHTML = m.baselines.map(card).join("");
  const ex = m.experiments, runs = ex.runs.slice().reverse();
  const running = runs.filter((r) => r.status === "running").length;
  const scored = ex.scored ? ex.scored.run : null;
  const frozen = runs.filter((r) => r.split === "frozen" && r.best_crps !== null);
  const bestRun = frozen.length ? frozen.reduce((a, b) => (a.best_crps <= b.best_crps ? a : b)) : null;
  const head = bestRun ? `${runs.length} training runs; ${bestRun.run} has the lowest validation CRPS on the frozen split (${fmt(bestRun.best_crps, 4)} at step ${fmtInt(bestRun.best_step)})${running ? `; ${running} run${running > 1 ? "s are" : " is"} in progress` : ""}.` : `${runs.length} training runs.`;
  $("#m-exp").innerHTML = `<h4 class="headline">${esc(head)}</h4><p class="sub-title">Generated at build time from each run's config.json and log.jsonl, its checkpoint (parameter count) and model/NOTES.md (experiment id, purpose, outcome). Newest first.${scored ? ` The run scored on the Results tab is ${esc(scored)} at step ${fmtInt(ex.scored.step)}.` : ""}</p>
  <div class="table-wrap"><table class="exp"><thead><tr><th>Run</th><th>Exp.</th><th class="num">Parameters</th><th class="num">Width</th><th class="num">Depth</th><th class="num">Heads</th><th>Head type</th><th>Flags</th><th class="num">Training sites</th><th class="num">Steps</th><th class="num">Best val. CRPS</th><th>Status</th></tr></thead>
  <tbody>${runs.map((r) => `<tr class="main${r.run === scored ? " ours" : ""}"><th scope="row"><code>${esc(r.run)}</code></th><td>${esc(r.exp)}</td><td class="num">${fmtInt(r.n_params)}</td><td class="num">${r.d ?? "–"}</td><td class="num">${r.depth ?? "–"}</td><td class="num">${r.heads_n ?? "–"}</td><td>${esc(r.head)}</td><td>${esc(r.flags.join(", ") || "–")}</td><td class="num">${fmtInt(r.train_sites)}<span class="sub">${esc(r.split)}</span></td><td class="num">${fmtInt(r.steps)}<span class="sub">of ${fmtInt(r.target)}</span></td><td class="num">${r.best_crps === null ? `<span class="muted">${r.mode === "overfit" ? "training windows" : "–"}</span>` : fmt(r.best_crps, 4) + `<span class="sub">step ${fmtInt(r.best_step)}</span>`}</td><td><span class="status s-${r.status}">${esc(r.status)}</span></td></tr><tr class="detail${r.run === scored ? " ours" : ""}"><td colspan="12"><b>Purpose</b> ${esc(r.purpose || "not recorded in NOTES.md")} <b>Outcome</b> ${esc(r.outcome || (r.status === "running" ? "in progress" : "not yet recorded in NOTES.md"))}</td></tr>`).join("")}</tbody></table></div>
  <p class="defs"><b>Parameters</b>: trainable parameters of the run's checkpoint. <b>Width, depth, heads</b>: model width d, number of time-and-variable block pairs, attention heads. <b>Head type</b>: output distribution; "+ flow matching" means the joint head is trained on the same encoder. <b>Training sites</b>: sites the run trains on, and the split (provisional: a hash split used before the frozen split existed; development only). <b>Steps</b>: optimiser steps logged, of the planned length. <b>Best val. CRPS</b>: lowest mean CRPS over all hidden values of the run's validation windows, in standardised units (each variable divided by its training standard deviation); frozen-split runs share 60 windows at the 54 validation sites, provisional runs used other sites and are not comparable. <b>Status</b>: complete (reached its planned steps), running (log written in the last 20 minutes) or stopped.</p>
  <h4>Evaluations and diagnoses without a training run</h4><div class="table-wrap"><table><thead><tr><th>Exp.</th><th>Purpose</th><th>Outcome</th></tr></thead><tbody>${ex.evaluations.map((e) => `<tr><th scope="row">${esc(e.exp)}</th><td>${esc(e.purpose)}</td><td>${esc(e.outcome)}</td></tr>`).join("")}</tbody></table></div>`;
}

// ---------- report ----------
// Full-screen view of a report figure. The figures are vector graphics, so they stay sharp at any zoom: the first click
// fits the figure to the screen, a click on it zooms to 300% at the clicked point, Escape or the button closes.
function openFigure(img) {
  const view = $("#fig-view"), big = view.querySelector("img");
  big.src = img.src; big.alt = img.alt;
  view.classList.remove("zoomed");
  view.hidden = false;
  view.querySelector(".fig-close").focus();
}
function initFigureView() {
  const view = $("#fig-view"), big = view.querySelector("img"), box = view.querySelector(".fig-scroll");
  const close = () => { view.hidden = true; };
  view.querySelector(".fig-close").addEventListener("click", close);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !view.hidden) close(); });
  big.addEventListener("click", (e) => {
    const r = big.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    view.classList.toggle("zoomed");
    if (view.classList.contains("zoomed")) {
      box.scrollLeft = fx * box.scrollWidth - box.clientWidth / 2;
      box.scrollTop = fy * box.scrollHeight - box.clientHeight / 2;
    }
  });
}

function renderReport(rep) {
  if (!rep || !rep.html) { $("#rep-body").innerHTML = '<h2>Report</h2><div class="pending">The report draft is not available in this build.</div>'; return; }
  $("#rep-body").innerHTML = rep.html;
  const typeset = () => window.MathJax && MathJax.typesetPromise ? MathJax.typesetPromise([$("#rep-body")]) : setTimeout(typeset, 300);
  typeset();
  for (const img of document.querySelectorAll("#rep-body .rep-fig img")) {
    img.tabIndex = 0;
    img.title = "Open full screen";
    img.addEventListener("click", () => openFigure(img));
    img.addEventListener("keydown", (e) => { if (e.key === "Enter") openFigure(img); });
  }
  $("#rep-refs").innerHTML = rep.references ? `<section class="rep-sec"><div class="sec-label">07</div><h3 class="sec-title" id="references">References</h3><ol class="refs">${rep.references.map((r) => `<li id="ref-${r.n}">${r.html}</li>`).join("")}</ol></section>` : "";
}

// ---------- component study ----------
function renderComponents(d) {
  const box = $("#r-components");
  if (!box || !d || !d.rows.length) return;
  const f = (v) => (v === null || v === undefined ? "–" : Math.abs(v) < 0.05 ? "0.0%" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}%`);
  const names = ["crps", "NEE", "LE", "H", "FCH4", "cov95"];
  const heads = ["CRPS, all", "NEE", "LE", "H", "FCH4", "95% cov."];
  const row = (lab, seeds, r) => `<tr><th scope="row">${esc(lab)}</th><td class="num">${seeds}</td>${names.map((n) => `<td class="num">${f(r[n])}</td>`).join("")}<td>${esc(r.decision || "")}</td></tr>`;
  let h = `<p>One change at a time against a control of width 192 and depth 6, at ${d.step.toLocaleString()} steps, on 60 validation windows. Percentages are changes in validation CRPS (lower is better) and in 95% coverage, against the control of the same seed. ${d.done} of ${d.total} components are finished. A row counts only when it exceeds the noise row on a second seed; FCH4 on one evaluation moves more than the noise row shows, because the two control seeds happen to agree at this step. Rows named "Without" remove a component; the others add one.</p>${d.story ? `<p>${esc(d.story)}</p>` : ""}`;
  h += `<div class="table-wrap"><table class="results"><thead><tr><th scope="col">Component</th><th class="num">Seeds</th>${heads.map((x) => `<th class="num">${x}</th>`).join("")}<th scope="col">Decision</th></tr></thead><tbody>`;
  if (d.noise) h += row("Second control seed (noise)", "", d.noise);
  h += d.rows.map((r) => row(r.label, r.seeds, r)).join("") + "</tbody></table></div>";
  if (d.sizes.length) {
    h += `<h4 class="headline">Size probes at matched steps</h4><p class="note">All on the same 48,000-step schedule. Validation CRPS of all variables, then of the four fluxes.</p>`;
    h += `<div class="table-wrap"><table class="results"><thead><tr><th scope="col">Size</th><th class="num">Rate</th>${d.size_steps.map((s) => `<th class="num">${s.toLocaleString()} steps</th>`).join("")}</tr></thead><tbody>`;
    h += d.sizes.map((s) => `<tr><th scope="row">${esc(s.size)}, ${esc(s.shape)}</th><td class="num">${s.lr}</td>${d.size_steps.map((k) => { const p = s.points[k]; return `<td class="num">${p ? `${p.crps.toFixed(4)} / ${p.flux.toFixed(4)}` : "–"}</td>`; }).join("")}</tr>`).join("") + "</tbody></table></div>";
  }
  box.innerHTML = h;
}

// ---------- boot ----------
(async () => {
  const [st, s, idx, res, rep, meth, scal, rtest, bm, comp] = await Promise.all([getJSON("data/status.json"), getJSON("data/sites.json"), getJSON("data/windows/index.json"), getJSONgz("data/results.json"), getJSON("data/report.json"), getJSON("data/methods.json"), getJSON("data/scaling.json"), getJSONgz("data/results_test.json"), getJSON("data/benchmark.json"), getJSON("data/components.json")]);
  SCALING = scal;
  renderComponents(comp);
  renderStatus(st); initTowers(s); initWindows(idx); renderResults(res); renderMethods(meth); renderReport(rep); initFigureView(); renderBenchmark(bm);
  const splitBtns = [...document.querySelectorAll("#r-split button")];
  const testBtn = splitBtns.find((b) => b.dataset.split === "test");
  if (!rtest) testBtn.disabled = true;               // validation sites are the default view
  for (const btn of splitBtns) btn.addEventListener("click", () => {
    for (const x of splitBtns) x.setAttribute("aria-checked", String(x === btn));
    renderResults(btn.dataset.split === "test" ? rtest : res);
  });
  show(location.hash.slice(1) || "overview");
})();
