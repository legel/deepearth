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

// ---------- tabs and theme ----------
const TABS = ["overview", "towers", "benchmark", "windows", "results", "report"];
function show(tab) {
  if (!TABS.includes(tab)) tab = "overview";
  for (const t of TABS) {
    $("#t-" + t).setAttribute("aria-selected", t === tab ? "true" : "false");
    $("#p-" + t).hidden = t !== tab;
  }
  if (tab === "towers" && map) setTimeout(() => map.invalidateSize(), 0);
  if (tab === "windows") drawWindow();
}
for (const t of TABS) $("#t-" + t).addEventListener("click", () => { history.replaceState(null, "", "#" + t); show(t); });
window.addEventListener("hashchange", () => show(location.hash.slice(1)));
$("#theme").addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme");
  const dark = cur ? cur === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
  try { localStorage.setItem("fluxfill-theme", dark ? "light" : "dark"); } catch (e) { /* storage unavailable */ }
  drawWindow();
});
try { const th = localStorage.getItem("fluxfill-theme"); if (th) document.documentElement.setAttribute("data-theme", th); } catch (e) { /* none */ }

// ---------- overview ----------
function renderStatus(st) {
  if (!st) { $("#status-tiles").innerHTML = '<div class="tile"><div class="l">Status not available</div></div>'; return; }
  const tiles = st.tiles.map((t) => `<div class="tile"><div class="v">${fmt(t.value)}</div><div class="l">${t.label}</div></div>`);
  $("#status-tiles").innerHTML = tiles.join("");
  $("#status-time").textContent = `Built ${st.built_utc} UTC. ${st.note || ""}`;
}

// ---------- towers ----------
let map = null, sites = [], layer = null;
const SPLIT_COLOR = { train: "--series-1", validation: "--series-3", test: "--series-2" };
function renderTowers() {
  const igbp = $("#f-igbp").value, net = $("#f-net").value, ch4 = $("#f-ch4").value;
  const rows = sites.filter((s) => (!igbp || s.igbp === igbp) && (!net || s.network === net) && (!ch4 || s.ch4));
  $("#tower-count").textContent = `${rows.length} of ${sites.length} towers shown.`;
  $("#tower-table tbody").innerHTML = rows.map((s) => `<tr><td>${s.id}</td><td>${s.name || ""}</td><td>${s.igbp || ""}</td><td>${s.koppen || ""}</td><td>${s.network || ""}</td><td>${s.years || ""}</td><td>${s.ch4 ? "yes" : ""}</td><td>${s.split || "not included"}</td></tr>`).join("");
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
function initWindows(idx) {
  windows = idx;
  if (!idx || !idx.windows || !idx.windows.length) {
    $("#w-body").innerHTML = '<div class="pending">No windows are exported yet. Windows appear here once the tower dataset build finishes and the first methods are scored.</div>';
    return;
  }
  const sitesW = [...new Set(idx.windows.map((w) => w.site))].sort();
  opt($("#w-site"), sitesW);
  const refresh = () => {
    const ws = idx.windows.filter((w) => w.site === $("#w-site").value);
    opt($("#w-win"), ws.map((w) => w.file), ws.map((w) => `${w.start} to ${w.end}`));
    const w = ws[0];
    opt($("#w-mask"), w ? w.masks : []);
    opt($("#w-method"), w ? w.methods : []);
    drawWindow();
  };
  $("#w-site").addEventListener("change", refresh);
  for (const id of ["#w-win", "#w-mask", "#w-method", "#w-zoom"]) $(id).addEventListener("change", drawWindow);
  refresh();
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
  // x-range: the whole window, or the union of masked spans with 25% padding on each side
  let x0 = 0, x1 = T - 1;
  if ($("#w-zoom").value === "mask") {
    const sp = Object.values(w.masks[mask] || {}).flat();
    if (sp.length) { const a = Math.min(...sp.map((x) => x[0])), b = Math.max(...sp.map((x) => x[1])); const pad = Math.max(48, Math.round((b - a) * 0.25)); x0 = Math.max(0, a - pad); x1 = Math.min(T - 1, b + pad); }
  }
  if (w.note) body.insertAdjacentHTML("beforeend", `<p class="note">${w.note}</p>`);
  const sc = ((w.scores || {})[method] || {})[mask] || {};
  const rows = w.variables.filter((v) => sc[v.key]);
  body.insertAdjacentHTML("beforeend", rows.length ? `<div class="table-wrap"><table><thead><tr><th>Variable</th><th class="num">CRPS</th><th class="num">MAE</th><th class="num">80% coverage</th><th class="num">95% coverage</th><th class="num">filled</th><th class="num">masked values</th></tr></thead><tbody>${rows.map((v) => { const r = sc[v.key]; if (!r.n) return `<tr><td>${v.label} <span class="note">${v.units || ""}</span></td><td class="num" colspan="5">not filled by this method</td><td class="num">–</td></tr>`; return `<tr><td>${v.label} <span class="note">${v.units || ""}</span></td><td class="num">${fmt(r.crps, 3)}</td><td class="num">${fmt(r.mae, 3)}</td><td class="num">${fmt(100 * r.cov80, 1)}%</td><td class="num">${fmt(100 * r.cov95, 1)}%</td><td class="num">${fmt(100 * (r.filled ?? 1), 0)}%</td><td class="num">${fmt(r.n)}</td></tr>`; }).join("")}</tbody></table></div><p class="note">Scores on the masked, measured values of this window only. CRPS and MAE in each variable's units.</p>` : "");
  let group = null;
  for (const v of w.variables) {
    const g = v.provenance === "reanalysis" ? "Reanalysis drivers (ERA5, downscaled to the site by ONEFlux)" : (["NEE", "LE", "H", "FCH4", "G"].includes(v.key) ? "Fluxes" : "Tower meteorology and soil");
    if (g !== group) { group = g; body.insertAdjacentHTML("beforeend", `<h3>${g}</h3>`); }
    const m = (w.masks[mask] || {})[v.key];               // array of [start, stop) index pairs hidden
    const segs = ((w.predictions[method] || {})[mask] || {})[v.key]; // [{a, mean, q025, q10, q90, q975}]
    body.appendChild(panel(w, v, m, expand(segs, T), T, x0, x1));
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

function panel(w, v, maskSpans, pred, T, x0 = 0, x1 = T - 1) {
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
  s += `<text class="lab-v" x="${ml}" y="12">${v.label}</text><text class="lab" x="${ml + 8 + v.label.length * 6.6}" y="12">${v.units || ""}</text>`;
  s += `<line class="xh" x1="0" x2="0" y1="${mt}" y2="${mt + ih}" stroke="${css("--quiet")}" stroke-width="1" visibility="hidden"/>`;
  s += `<rect x="${ml}" y="${mt}" width="${iw}" height="${ih}" fill="transparent"/></svg><div class="tooltip"></div>`;
  wrap.innerHTML = s;
  const svg = wrap.querySelector("svg"), tip = wrap.querySelector(".tooltip"), xh = wrap.querySelector(".xh");
  svg.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect(), sx = (e.clientX - r.left) * (W / r.width);
    const i = Math.max(x0, Math.min(x1, Math.round(x0 + ((sx - ml) / iw) * (x1 - x0))));
    xh.setAttribute("x1", X(i)); xh.setAttribute("x2", X(i)); xh.setAttribute("visibility", "visible");
    let h = `<b>${w.t[i]}</b><br>${hidden[i] ? "masked" : "visible"}<br>measured: ${fmt(vals[i], 2)} ${v.units || ""}`;
    if (pred && hidden[i]) h += `<br>predicted: ${fmt(pred.mean[i], 2)}<br>80%: ${fmt(pred.q10[i], 2)} to ${fmt(pred.q90[i], 2)}<br>95%: ${fmt(pred.q025[i], 2)} to ${fmt(pred.q975[i], 2)}`;
    tip.innerHTML = h; tip.style.display = "block";
    const px = (X(i) / W) * r.width; tip.style.left = Math.min(px + 12, r.width - 200) + "px"; tip.style.top = "8px";
  });
  svg.addEventListener("mouseleave", () => { tip.style.display = "none"; xh.setAttribute("visibility", "hidden"); });
  return wrap;
}

// ---------- results and report ----------
function renderResults(res) {
  if (!res || !res.tables || !res.tables.length) {
    $("#r-body").innerHTML = '<div class="pending">No method has been scored on FluxFill-Bench yet. Scores appear here as each baseline and the model are run on the frozen benchmark.</div>';
    return;
  }
  $("#r-body").innerHTML = res.tables.map((t) => `<h3>${t.title}</h3><p class="note">${t.note || ""}</p><div class="table-wrap"><table><thead><tr>${t.columns.map((c, i) => `<th class="${i ? "num" : ""}">${c}</th>`).join("")}</tr></thead><tbody>${t.rows.map((r) => `<tr>${r.map((c, i) => `<td class="${i ? "num" : ""}">${typeof c === "number" ? fmt(c, 3) : (c ?? "–")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`).join("");
}
function renderReport(rep) {
  $("#rep-body").innerHTML = rep && rep.html ? rep.html : '<div class="pending">The scientific report is published here once FluxFill is evaluated on the full benchmark against every baseline, with ablations of every input.</div>';
}

// ---------- boot ----------
(async () => {
  const [st, s, idx, res, rep] = await Promise.all([getJSON("data/status.json"), getJSON("data/sites.json"), getJSON("data/windows/index.json"), getJSON("data/results.json"), getJSON("data/report.json")]);
  renderStatus(st); initTowers(s); initWindows(idx); renderResults(res); renderReport(rep);
  show(location.hash.slice(1) || "overview");
})();
