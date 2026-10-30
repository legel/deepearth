// Benchmark tab: what FluxFill-Bench is made of (data/benchmark.json, dashboard/benchmark_data.py).
const BM_SPLITS = [["train", "Train", "--series-1"], ["validation", "Validation", "--series-3"], ["test", "Test", "--series-2"]];
const bmEsc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const bmInt = (v) => Number(v).toLocaleString("en-US");
const bmSI = (v) => v >= 1e9 ? (v / 1e9).toFixed(1) + " G" : v >= 1e6 ? (v / 1e6).toFixed(1) + " M" : v >= 1e4 ? (v / 1e3).toFixed(0) + " k" : bmInt(v);
const bmPct = (v) => (100 * v).toFixed(v < 0.01 ? 2 : 1) + "%";

function bmLegend(items) {
  return `<div class="bm-legend">${items.map(([label, color, style]) =>
    `<span><i style="background:var(${color});${style || ""}"></i>${bmEsc(label)}</span>`).join("")}</div>`;
}

// Horizontal stacked bars: rows [{key, label?, values: {k: v}}], series [[k, label, color]].
function bmHStack(rows, series, { width = 720, bar = 16, gap = 8, label = 92, share = false, unit = "" } = {}) {
  const tot = rows.map((r) => series.reduce((a, [k]) => a + (r.values[k] || 0), 0));
  const max = share ? 1 : Math.max(...tot, 1);
  const W = width - label - 70, H = rows.length * (bar + gap);
  let svg = `<svg viewBox="0 0 ${width} ${H}" role="img">`;
  rows.forEach((r, i) => {
    const y = i * (bar + gap);
    svg += `<text x="${label - 8}" y="${y + bar * 0.72}" text-anchor="end" class="bm-t">${bmEsc(r.label || r.key)}</text>`;
    let x = label;
    series.forEach(([k, name, color]) => {
      const v = r.values[k] || 0;
      if (!v) return;
      const w = (share ? v / tot[i] : v / max) * W;
      svg += `<rect x="${x}" y="${y}" width="${Math.max(w - 1, 0.5)}" height="${bar}" fill="var(${color})"><title>${bmEsc(r.label || r.key)}, ${bmEsc(name)}: ${bmInt(v)}${unit} (${bmPct(v / tot[i])})</title></rect>`;
      x += w;
    });
    svg += `<text x="${x + 6}" y="${y + bar * 0.72}" class="bm-t q">${share ? "" : bmSI(tot[i]) + unit}</text>`;
  });
  return svg + "</svg>";
}

// Vertical columns, stacked by series.
function bmColumns(cols, series, { width = 720, height = 180, unit = "" } = {}) {
  const tot = cols.map((c) => series.reduce((a, [k]) => a + (c.values[k] || 0), 0));
  const max = Math.max(...tot, 1), L = 46, B = 22, W = width - L - 4, H = height - B - 6;
  const bw = W / cols.length;
  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img">`;
  for (const f of [0, 0.5, 1]) {
    const y = 6 + H * (1 - f);
    svg += `<line x1="${L}" x2="${width}" y1="${y}" y2="${y}" class="bm-grid"/><text x="${L - 6}" y="${y + 4}" text-anchor="end" class="bm-t q">${bmSI(max * f)}</text>`;
  }
  cols.forEach((c, i) => {
    let y = 6 + H;
    const x = L + i * bw + 1;
    series.forEach(([k, name, color]) => {
      const v = c.values[k] || 0;
      if (!v) return;
      const h = v / max * H;
      y -= h;
      svg += `<rect x="${x}" y="${y}" width="${Math.max(bw - 2, 1)}" height="${h}" fill="var(${color})"><title>${bmEsc(c.key)}, ${bmEsc(name)}: ${bmInt(v)}${unit}</title></rect>`;
    });
    if (cols.length <= 16 || i % Math.ceil(cols.length / 12) === 0)
      svg += `<text x="${x + bw / 2 - 1}" y="${height - 6}" text-anchor="middle" class="bm-t q">${bmEsc(c.key)}</text>`;
  });
  return svg + "</svg>";
}

const FAMILY_TEXT = { "M1": "M1 real-gap distribution", "M2-2h": "M2 2 hours", "M2-6h": "M2 6 hours", "M2-1d": "M2 1 day",
  "M2-7d": "M2 7 days", "M2-30d": "M2 30 days", "M2-90d": "M2 90 days", "M2-180d": "M2 180 days", "M3": "M3 real outage",
  "M4-no-reanalysis+M2-30d": "M4 no reanalysis + 30 d", "M5": "M5 mixed per-variable" };
const GAP_TEXT = { "1-1": "single half-hour", "2-8": "1 to 4 hours", "9-64": "4.5 hours to 1.3 days", "65-576": "1.4 to 12 days", "577-inf": "over 12 days" };

function renderBenchmark(bm) {
  const el = document.getElementById("bm-data");
  if (!el || !bm) return;
  const sp = bm.splits.frozen, t = bm.test, v = bm.validation || {};
  const towers = BM_SPLITS.reduce((a, [k]) => a + (sp[k] ? sp[k].towers : 0), 0);
  const fluxKept = bm.qc.filter((q) => ["NEE", "LE", "H", "FCH4"].includes(q.variable)).reduce((a, q) => a + q.final, 0);
  let h = `<div class="grid">
    <div class="tile"><div class="v">${bmInt(towers)}</div><div class="l">towers: ${BM_SPLITS.map(([k, n]) => `${sp[k].towers} ${n.toLowerCase()}`).join(", ")}</div></div>
    <div class="tile"><div class="v">${bmInt(t.n_windows)}</div><div class="l">test windows of 180 days at ${t.n_sites} test towers</div></div>
    <div class="tile"><div class="v">${bmInt(t.masks)}</div><div class="l">frozen test masks (release ${bmEsc(t.version)})</div></div>
    <div class="tile"><div class="v">${bmSI(t.hidden_total || 0)}</div><div class="l">hidden measured values scored on test</div></div>
    <div class="tile"><div class="v">${bmSI(fluxKept)}</div><div class="l">measured flux half-hours (NEE, LE, H, FCH4) after QC</div></div>
  </div>`;

  h += `<h3>Towers by ecosystem and split</h3>
  <p class="note">Every tower of the frozen split by IGBP class. Test towers lie at least 50 km from every training tower; whole spatial blocks go to one split.</p>
  ${bmLegend(BM_SPLITS.map(([, n, c]) => [n, c]))}
  <div class="viz">${bmHStack(bm.igbp.map((r) => ({ key: r.key, values: r })), BM_SPLITS)}</div>`;

  if (bm.igbp_methane_v2) {
    const s2 = bm.splits.v2;
    h += `<h3>Methane towers by split</h3>
    <p class="note">Towers that measure FCH4. The frozen split had ${sp.validation.methane} validation methane towers against ${sp.test.methane} test and ${sp.train.methane} training towers. Split v2 (test towers unchanged) moves whole training blocks to validation by a seeded draw toward the test mix: ${s2.validation.methane} validation, ${s2.train.methane} training, ${s2.test.methane} test.</p>
    ${bmLegend(BM_SPLITS.map(([, n, c]) => [n, c]))}
    <div class="bm-two"><div><p class="sub-title">Frozen split</p><div class="viz">${bmHStack(bm.igbp_methane.map((r) => ({ key: r.key, values: r })), BM_SPLITS, { width: 360, label: 48 })}</div></div>
    <div><p class="sub-title">Split v2</p><div class="viz">${bmHStack(bm.igbp_methane_v2.map((r) => ({ key: r.key, values: r })), BM_SPLITS, { width: 360, label: 48 })}</div></div></div>`;
  }

  const QC = [["final", "kept: measured, passed QC", "--series-1"], ["qc_removed", "removed by QC", "--series-2"],
              ["filled_at_source", "gap-filled at source (excluded)", "--quiet"], ["not_reported", "not reported", "--hair"]];
  h += `<h3>The measured-only rule, per variable</h3>
  <p class="note">All half-hours of the tower records on the benchmark grid. Only measured values that pass quality control enter, as inputs or as targets. Values the source network had already gap-filled are excluded. Hover a segment for the count; QC tests: quality flags, physical range, stuck sensor, spike, friction velocity (u*) and site-specific screens.</p>
  ${bmLegend(QC.map(([, n, c]) => [n, c]))}
  <div class="viz">${bmHStack(bm.qc.map((q) => ({ key: q.variable, values: q })), QC, { share: true })}</div>`;

  const cy = bm.coverage_years;
  h += `<h3>Measured flux half-hours per year</h3>
  <div class="controls"><label>Flux<select id="bm-flux">${Object.keys(cy).map((k) => `<option>${k}</option>`).join("")}</select></label></div>
  ${bmLegend(BM_SPLITS.map(([, n, c]) => [n, c]))}<div class="viz" id="bm-years"></div>`;

  if (t.families) {
    const maxH = Math.max(...t.families.map((f) => f.hidden));
    h += `<h3>Test masks: hidden values per family</h3>
    <p class="note">Measured values hidden and scored on the ${bmInt(t.n_windows)} test windows, per mask family and flux. Bars on a log scale.</p>
    <div class="table-wrap"><table class="bm-table"><thead><tr><th>Family</th><th class="num">Windows</th><th>Hidden values</th><th class="num">NEE</th><th class="num">LE</th><th class="num">H</th><th class="num">FCH4</th></tr></thead><tbody>
    ${t.families.map((f) => `<tr><th scope="row">${bmEsc(FAMILY_TEXT[f.family] || f.family)}</th><td class="num">${bmInt(f.windows)}</td>
      <td><span class="bm-bar" style="width:${(100 * Math.log10(f.hidden) / Math.log10(maxH)).toFixed(1)}%"></span>${bmSI(f.hidden)}</td>
      ${["NEE", "LE", "H", "FCH4"].map((k) => `<td class="num">${bmSI(f.flux[k])}</td>`).join("")}</tr>`).join("")}
    </tbody></table></div>`;
    const GC = Object.keys(GAP_TEXT).map((k, i) => [k, GAP_TEXT[k], "--series-1", `opacity:${(0.25 + 0.75 * i / 4).toFixed(2)}`]);
    h += `<h3>Gap lengths inside each family</h3>
    <p class="note">Share of hidden values by the length of the gap that contains them. Short-gap families (M1, M2 2 h and 6 h, M5) test interpolation; long blocks and real outages test the use of drivers, imagery and site memory.</p>
    ${bmLegend(GC.map(([, n, c, s]) => [n, c, s]))}
    <div class="viz">${bmGapStack(t.families, GC)}</div>`;
  }

  const o = bm.outages;
  const lab = (e) => e < 48 ? `${e / 2} h` : e < 1440 ? `${e / 48} d` : `${Math.round(e / 1440)} mo`;
  const ocols = o.edges.map((e, i) => ({ key: i < o.edges.length - 1 ? `${lab(e)}+` : `${lab(e)}+`, values: { h: o.halfhours[i] } }));
  h += `<h3>Real outages at training towers</h3>
  <p class="note">${bmInt(o.n)} outages (all tower channels missing together) in ${bmInt(o.tower_years)} training tower-years, by length; bars show the missing half-hours each length class accounts for. M3 transplants these outages into test windows.</p>
  <div class="viz">${bmColumns(ocols, [["h", "missing half-hours", "--series-1"]], { height: 170 })}</div>`;

  const seas = Object.entries(t.season).map(([k, n]) => ({ key: k, values: { n } }));
  const yrs = Object.entries(t.year).map(([k, n]) => ({ key: k, values: { n } }));
  h += `<h3>Test windows by start season and year</h3>
  <div class="bm-two"><div class="viz">${bmColumns(seas, [["n", "windows", "--series-2"]], { width: 300, height: 150 })}</div>
  <div class="viz">${bmColumns(yrs, [["n", "windows", "--series-2"]], { width: 420, height: 150 })}</div></div>
  <p class="note">Up to 4 windows per test tower, one per start season where the record allows: ${Object.entries(t.per_site).map(([k, n]) => `${n} towers with ${k}`).join(", ")}. Release checksums: split ${bmEsc(t.split_sha256)}, windows ${bmEsc(t.windows_sha256)}.</p>`;

  if (v.methane_v2) {
    const ps = Object.entries(v.methane_v2.per_site).map(([k, n]) => ({ key: k, values: { n } }));
    h += `<h3>Validation sets</h3>
    <p class="note">Model selection reads two validation sets: ${v.windows} windows at ${v.sites} validation towers (every flux), and ${v.methane_v2.windows} methane windows at the ${v.methane_v2.sites} split-v2 validation towers that measure FCH4 (every usable, non-overlapping 180-day window with FCH4 in at least 5% of its half-hours).</p>
    <div class="viz">${bmHStack(ps, [["n", "methane validation windows", "--series-3"]], { width: 720, bar: 12, gap: 5, label: 70 })}</div>`;
  }
  el.innerHTML = h;
  const drawYears = () => {
    const k = document.getElementById("bm-flux").value;
    document.getElementById("bm-years").innerHTML = bmColumns(cy[k].map((r) => ({ key: String(r.year), values: r })), BM_SPLITS, { unit: " half-hours" });
  };
  document.getElementById("bm-flux").addEventListener("change", drawYears);
  drawYears();
}

function bmGapStack(fams, GC) {
  const width = 720, label = 170, bar = 14, gap = 6, W = width - label - 10;
  let svg = `<svg viewBox="0 0 ${width} ${fams.length * (bar + gap)}" role="img">`;
  fams.forEach((f, i) => {
    const y = i * (bar + gap), tot = Object.values(f.gap_classes).reduce((a, b) => a + b, 0) || 1;
    svg += `<text x="${label - 8}" y="${y + bar * 0.75}" text-anchor="end" class="bm-t">${bmEsc(FAMILY_TEXT[f.family] || f.family)}</text>`;
    let x = label;
    GC.forEach(([k, name, color, style]) => {
      const w = (f.gap_classes[k] || 0) / tot * W;
      if (w <= 0) return;
      svg += `<rect x="${x}" y="${y}" width="${Math.max(w - 1, 0.5)}" height="${bar}" fill="var(${color})" style="${style}"><title>${bmEsc(FAMILY_TEXT[f.family] || f.family)}, gaps ${bmEsc(name)}: ${bmPct((f.gap_classes[k] || 0) / tot)} of hidden values</title></rect>`;
      x += w;
    });
  });
  return svg + "</svg>";
}
