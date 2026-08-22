/* DeepEarth dashboard — hash-routed one-page app over /api. */
const $ = (s, el = document) => el.querySelector(s);
const view = $("#view");
const state = { reg: null, graph: null, status: null };
const RANK = { critical: 0, serious: 1, warning: 2, unknown: 3, good: 4 };
const SYS_NAMES = { earth4d: "Earth4D", phylo: "Phylogenomic", fusion: "Fusion Core",
                    method: "Method", data: "Data" };
const GLYPH = { good: "✓", warning: "!", serious: "▲", critical: "✕", unknown: "·" };

const api = async p => { const r = await fetch("/api/" + p); return r.ok ? r.json() : null; };
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function load() {
  [state.reg, state.graph, state.status, state.verif, state.findings, state.recon, state.flow,
   state.callg, state.trace, state.triage] =
    await Promise.all([api("registry"), api("graph"), api("status"), api("verification"),
                       api("findings"), api("reconstructions"), api("flow"), api("callgraph"),
                       api("trace"), api("triage")]);
  const meta = await api("meta");
  if (meta?.head) $("#meta").innerHTML = esc(`${meta.head.sha} · ${meta.head.subject}`) +
    esc(meta.audited ? ` · audited ${meta.audited}` : " · not yet audited") +
    (meta.skew ? ` <span style="color:var(--serious)" title="state artifacts were built at different
      commits — run python -m dashboard.refresh">▲ state skew</span>` : "");
  route();
}

const empty = msg => `<div class="empty">${msg}</div>`;
const needReg = () => empty(`No registry. Run <code>python -m dashboard.registry</code>.`);

/* ---- helpers over the graph ---- */
const edgesFor = pred => (state.graph?.edges ?? []).filter(pred);
const ruleEdges = id => edgesFor(e => e.dst === "R" + id);
const benchEdges = id => edgesFor(e => e.dst === id);
const blockEdges = path => edgesFor(e => e.src.startsWith(path + ":"));
const verifOf = id => state.verif?.verifications?.find(v => v.id === id);
const ruleStatus = id => {
  const s = state.status?.rules?.find(r => r.id === id), v = verifOf(id);
  return v ? { ...s, status: v.status, verified: v.verdict } : s;   // adversarial verdict wins
};

function statusTile(href, status, name, headline, cls = "", verified = null) {
  const st = status ?? "unknown";
  return `<a class="tile ${cls}" href="${href}">
    <span class="s" style="color:var(--${st})"><span class="dot" style="background:var(--${st})"></span>${GLYPH[st]} ${st}</span>
    ${verified ? `<span class="s" style="float:right;color:var(--muted)" title="adversarially verified: ${verified}">✓✓</span>` : ""}
    <h3>${esc(name)}</h3><p>${esc(headline ?? "no audit yet")}</p></a>`;
}

/* ---- views ---- */
function vStatus() {
  if (!state.reg) return needReg();
  const sys = Object.entries(SYS_NAMES).map(([k, name]) => {
    const s = state.status?.systems?.[k];
    const worst = state.reg.rules.filter(r => r.system === k)                 // verified verdicts win
      .reduce((w, r) => Math.min(w, RANK[ruleStatus(r.id)?.status ?? "unknown"]), 4);
    const st = state.verif ? Object.keys(RANK)[worst] : s?.status;
    return statusTile("#/science/" + k, st, name, s?.headline, "sys");
  }).join("");
  const rules = [...state.reg.rules]
    .sort((a, b) => (RANK[ruleStatus(a.id)?.status ?? "unknown"] - RANK[ruleStatus(b.id)?.status ?? "unknown"]) || a.id - b.id)
    .map(r => statusTile("#/rule/" + r.id, ruleStatus(r.id)?.status, `R${r.id} · ${r.title}`,
                         ruleStatus(r.id)?.headline, "", ruleStatus(r.id)?.verified)).join("");
  const finds = (state.findings?.findings ?? []).map(f => `<div class="row">
    <span class="s" style="color:var(--${f.sev === "serious" ? "serious" : "warning"})">${GLYPH[f.sev] ?? "!"} ${f.id}</span>
    <span class="grow"><b>${esc(f.title)}</b> — <span style="color:var(--ink-2)">${esc(f.detail)}</span>
      ${(f.refs ?? []).map(r => `<code style="font-size:.76rem">${esc(r)}</code>`).join(" ")}</span></div>`).join("");
  let reachHtml = "";
  if (state.callg) {
    const model = state.callg.defs.filter(d =>
      /^(core|encoders)\//.test(d.path) || (/^autoresearch\//.test(d.path) && !/recipes/.test(d.path)));
    const isl = model.filter(d => d.reach === "island");
    const gat = model.filter(d => d.reach === "gated");
    const live = model.filter(d => d.reach === "live").length;
    reachHtml = `<h2>Champion-path reachability — static call graph under the current config</h2>
      <p class="sub">${model.length} defs in the model codebase: <b style="color:var(--good)">${live} live</b> ·
        <b style="color:var(--warning)">${gat.length} gated off</b> ·
        <b style="color:var(--critical)">${isl.length} never called</b>.
        Code that exists but does not run is capability, not implementation.</p>
      <div class="rows">${[...isl.map(d => ["island", d]), ...gat.map(d => ["gated", d])].map(([k, d]) => {
        const t = state.triage?.triage?.find(x => x.id === d.id);
        const tc = { "wire-in": "warning", delete: "critical", keep: "muted" }[t?.action] ?? "muted";
        return `<a class="row" href="#/file/${d.path}:${d.start}-${d.end}">
          <span class="s" style="color:var(--${k === "island" ? "critical" : "warning"})">${k === "island" ? "✕ island" : "◐ " + esc(d.gate ?? "gated")}</span>
          <code class="grow">${esc(d.id)}</code>
          ${t ? `<span class="s" style="color:var(--${tc})" title="${esc(t.reason)}">→ ${t.action}</span>` : ""}
          <span class="num">${esc(d.path)}:${d.start}–${d.end}</span></a>`;
      }).join("")}</div>`;
  }
  return `<h1>System status</h1>
    <p class="sub">Five systems, thirty-two principles. Every claim opens to its evidence.</p>
    <div class="tiles">${sys}</div><h2>Principles — worst first</h2><div class="tiles">${rules}</div>
    ${reachHtml}
    ${finds ? `<h2>Operational findings — from actually running the system</h2><div class="rows">${finds}</div>` : ""}`;
}

/* ---- code viewer: syntax-highlighted dark plates, everywhere code appears ---- */
const codeCache = {};
const LANG = { py: "python", js: "javascript", json: "json", yaml: "yaml", yml: "yaml",
  md: "markdown", sh: "bash", css: "css", html: "xml", cu: "cpp", cpp: "cpp", h: "cpp",
  R: "r", r: "r", cfg: "ini", toml: "ini", txt: "plaintext" };

function splitHl(html) {                                 // split highlighted html by line, carrying open spans
  const out = [], open = [];
  for (const raw of html.split("\n")) {
    const prefix = open.join("");
    for (const t of raw.match(/<span[^>]*>|<\/span>/g) ?? [])
      t === "</span>" ? open.pop() : open.push(t);
    out.push(prefix + raw + "</span>".repeat(open.length));
  }
  return out;
}

async function hlLines(path) {                           // highlighted lines, cached per file
  if (codeCache[path]) return codeCache[path];
  const text = await (await fetch("/api/code/" + path)).text();
  const lang = LANG[path.split(".").pop()];
  const html = (typeof hljs !== "undefined" && lang && hljs.getLanguage(lang))
    ? hljs.highlight(text, { language: lang, ignoreIllegals: true }).value
    : esc(text);
  return codeCache[path] = splitHl(html);
}

const codeLine = (n, html, cls = "", anno = "") =>
  `<div class="cl${cls}" id="L${n}">${anno}<span class="ln">${n}</span><span class="lc">${html || " "}</span></div>`;

async function snippet(ref) {
  const m = ref.match(/^(.+?):(\d+)-(\d+)$/);
  if (!m) return "";
  const [, p, s, e] = m;
  const lines = await hlLines(p);
  const cap = Math.min(+e, +s + 59);                     // cap 60 lines inline
  const body = lines.slice(+s - 1, cap).map((l, i) => codeLine(+s + i, l)).join("");
  return `<div class="snip codeplate">${body}${+e > cap
    ? `<div class="cl"><span class="ln">…</span><span class="lc">+${+e - cap} more — open the file</span></div>` : ""}</div>`;
}

const REACHORD = ["live", "gated", "data-pipeline", "tests", "recipes", "tooling", "island"];
function reachChip(ref) {                                // proof of integration from the call graph
  const m = ref.match(/^(.+?):(\d+)-(\d+)$/);
  if (!m || !state.callg || !m[1].endsWith(".py")) return "";
  const ds = state.callg.defs.filter(d => d.path === m[1] && d.start <= +m[3] && d.end >= +m[2]);
  if (!ds.length) return "";
  const best = ds.reduce((a, b) => REACHORD.indexOf(a.reach) <= REACHORD.indexOf(b.reach) ? a : b);
  return best.reach === "live" ? `<span class="s" style="color:var(--good)">● live</span>`
    : best.reach === "gated" ? `<span class="s" style="color:var(--warning)">◐ gated: ${esc(best.gate ?? "?")}</span>`
    : best.reach === "island" ? `<span class="s" style="color:var(--critical)">✕ never called</span>`
    : `<span class="s" style="color:var(--muted)">${esc(best.reach)}</span>`;
}

const mkRow = (ref, s, note) => `
    <div class="row erow" data-ref="${esc(ref)}">
      <span class="id">${s}</span>
      <code class="grow expand" title="show the code">▸ ${esc(ref)}</code>
      ${reachChip(ref)}
      <span class="num">${esc(note)}</span>
      <a class="pill" href="#/file/${ref}">open ↗</a>
    </div><div class="snipbox" hidden></div>`;

const edgeRows = edges => edges.length
  ? `<div class="rows">${edges.map(e => mkRow(e.src, "s" + e.s, e.note)).join("")}</div>` : "";

const ROLE = p => /\.(py|cu|cpp|h|js|R)$/.test(p) ? 0 : /\.(ya?ml|toml|cfg|ini)$/.test(p) ? 1 : 2;

function edgeSections(edges, extraImpl = "") {           // implementation ≠ configuration ≠ documentation
  const groups = [[], [], []];
  for (const e of edges) groups[ROLE(e.src.split(":")[0])].push(e);
  groups.forEach(g => g.sort((a, b) => b.s - a.s));
  if (!edges.length && !extraImpl) return empty("No connections yet — run the audit.");
  const names = ["Implementation", "Configuration", "Documentation & records"];
  return groups.map((g, i) => (g.length || (i === 0 && extraImpl))
    ? `<h3 class="esec">${names[i]}</h3>${i === 0 ? extraImpl : ""}${edgeRows(g)}` : "").join("");
}

function armRows() {
  document.querySelectorAll(".erow .expand").forEach(el => el.onclick = async () => {
    const row = el.closest(".erow"), box = row.nextElementSibling;
    box.hidden = !box.hidden;
    el.textContent = (box.hidden ? "▸ " : "▾ ") + row.dataset.ref;
    if (!box.hidden && !box.innerHTML) box.innerHTML = await snippet(row.dataset.ref);
  });
}

const refLink = t => /^[\w./-]+:\d+-\d+/.test(t)
  ? `<a href="#/file/${t.split(" ")[0]}"><code>${esc(t)}</code></a>` : `<code>${esc(t)}</code>`;

function vRule(id) {
  const r = state.reg?.rules.find(x => x.id === +id);
  if (!r) return needReg();
  const st = ruleStatus(r.id), vf = verifOf(r.id);
  const edges = ruleEdges(r.id);
  route.after = armRows;
  return `<h1>R${r.id} · ${esc(r.title)}</h1><p class="sub">${esc(r.summary)}</p>
    ${st ? `<div class="tiles">${statusTile("#", st.status, st.headline ?? "", st.next, "sys")}</div>` : ""}
    ${vf ? `<p class="sub" style="margin-top:.8rem">✓✓ <b>${vf.verdict}</b> by adversarial review — ${esc(vf.note)}
      ${(vf.key_evidence ?? []).map(refLink).join(" ")}</p>` : ""}
    ${r.id === 23 && state.recon?.marginal_fidelity ? r23Invariant() : ""}
    ${ruleBenches(r.id)}
    <h2>Connected code — ${edges.length} connections · click any row to read it</h2>
    ${edgeSections(edges)}`;
}

function r23Invariant() {                                 // measured, not asserted: fidelity at K=1 vs K=rounds
  const mf = state.recon.marginal_fidelity;
  const ks = Object.keys(Object.values(mf)[0]);
  const rows = Object.entries(mf).map(([v, d]) => {
    const delta = d[ks[ks.length - 1]] - d[ks[0]];
    return `<div class="row"><b class="grow">${esc(v)}</b>
      ${ks.map(k => `<span class="num">${k} ${d[k].toFixed(3)}</span>`).join("")}
      <span class="num" style="min-width:5em;text-align:right;color:var(--${delta >= -0.01 ? "good" : "serious"})">
        ${(delta >= 0 ? "+" : "") + delta.toFixed(3)}</span></div>`;
  }).join("");
  return `<h2>The rule's own invariant — measured on the real batch</h2>
    <p class="sub">Each variable hidden alone, decoded at K=1 vs K=${ks[ks.length - 1].slice(1)} rounds:
      pluralism is conserved iff the marginal does not degrade as coupling rises.</p>
    <div class="rows">${rows}</div>`;
}

function ruleBenches(id) {                                // benchmarks joined through shared implementing blocks
  const span = s => { const m = s.match(/:(\d+)-(\d+)$/); return m ? m[2] - m[1] : 0; };
  const all = ruleEdges(id).map(e => e.src)
    .filter(s => !s.startsWith("autoresearch/evaluate.py"));   // the evaluator touches every benchmark by definition
  const specific = all.filter(s => span(s) < 600);        // prefer precise blocks over module-spanning giants
  const blocks = new Set(specific.length ? specific : all);
  const bids = [...new Set((state.graph?.edges ?? [])
    .filter(e => e.dst[0] === "B" && blocks.has(e.src)).map(e => e.dst))];
  if (!bids.length) return "";
  const bs = bids.map(i => state.reg.benchmarks.find(b => b.id === i)).filter(Boolean)
    .sort((a, b) => (a.current_score ?? 2) - (b.current_score ?? 2));
  return `<h2>Benchmarks that exercise this principle — ${bs.length}</h2>
    <div style="margin-bottom:.4rem">${bs.map(b => `<a class="chip" href="#/bench/${b.id}"
      title="${esc(b.measures)}" style="border-color:${bandColor(b.current_score)}">${b.id}
      ${b.current_score?.toFixed(2) ?? "—"}</a>`).join(" ")}</div>`;
}

function vCode() {
  if (!state.reg) return needReg();
  const byDir = {};
  for (const f of state.reg.files.filter(f => f.kind === "text")) {
    const d = f.path.includes("/") ? f.path.split("/")[0] : "(root)";
    (byDir[d] ??= []).push(f);
  }
  return `<h1>Code</h1><p class="sub">${state.reg.counts.files} files · ${state.reg.counts.blocks} blocks.
      Bare rows have no connection to any principle or benchmark.</p>` +
    Object.entries(byDir).map(([d, fs]) => `<h2>${d}/</h2><div class="rows">` +
      fs.map(f => {
        const es = blockEdges(f.path);
        const rs = es.filter(e => e.dst[0] === "R").map(e => +e.dst.slice(1));
        const st = rs.length ? Object.keys(RANK)[rs.reduce((w, id) =>
          Math.min(w, RANK[ruleStatus(id)?.status ?? "unknown"]), 4)] : null;
        return `<a class="row" href="#/file/${f.path}">
          <span class="dot" style="background:${st ? `var(--${st})` : "var(--grid)"}"
            title="${st ? "worst linked principle: " + st : "no principle links"}"></span>
          <code class="grow">${f.path}</code>
          <span class="num">${f.lines} ln</span>
          <span class="num" style="${es.length ? "" : "color:var(--serious)"}">${es.length || "—"} edges</span></a>`;
      }).join("") + "</div>").join("");
}

function pillFor(dst, note = "") {                       // named, hoverable rule/benchmark pill
  if (dst.startsWith("R")) {
    const r = state.reg.rules.find(x => "R" + x.id === dst);
    return `<a class="pill science" href="#/rule/${dst.slice(1)}"
      title="${esc((r?.summary ?? "") + (note ? "\n↳ " + note : ""))}">${dst} · ${esc(r?.title ?? "")}</a>`;
  }
  const b = state.reg.benchmarks.find(x => x.id === dst);
  const label = (b?.measures ?? "").length > 38 ? b.measures.slice(0, 36) + "…" : b?.measures ?? "";
  return `<a class="pill bench" href="#/bench/${dst}"
    title="${esc((b?.measures ?? "") + (note ? "\n↳ " + note : ""))}">${dst} · ${esc(label)}</a>`;
}

async function vFile(spec) {
  const m = spec.match(/^(.+?):(\d+)-(\d+)$/);
  const [path, hs, he] = m ? [m[1], +m[2], +m[3]] : [spec, 0, 0];
  const probe = await fetch("/api/code/" + path);
  if (!probe.ok) return `<h1><code>${esc(path)}</code></h1>` +
    empty(`Not a repo file — nothing to show. (Torch builtins and external modules have no source here.)`);
  const hl = await hlLines(path);
  const blocks = state.reg.blocks.filter(b => b.path === path);
  const edges = blockEdges(path);
  const perLine = {};
  for (const b of blocks) {
    const es = edges.filter(e => e.src === b.id);
    if (es.length) perLine[b.start] = es;
  }
  const lines = hl.map((l, i) => {
    const n = i + 1;
    const anno = (perLine[n] ?? []).map(e => pillFor(e.dst, e.note)).join(" ");
    return codeLine(n, l, n >= hs && n <= he ? " hl" : "", anno ? `<div class="anno">${anno}</div>` : "");
  }).join("");
  route.after = () => {
    if (!hs) return;
    const go = () => document.getElementById("L" + hs)?.scrollIntoView({ block: "center" });
    go(); setTimeout(go, 150);                           // again after late layout
  };
  return `<h1><code>${path}</code>${hs ? ` <span class="pill">lines ${hs}–${he}</span>` : ""}</h1>
    <p class="sub">${blocks.length} blocks · ${edges.length} connections — pills name the principle
      or benchmark each block implements; hover for the why, click to traverse.</p>
    <div class="codeplate">${lines}</div>`;
}

function vScience(sys) {
  if (!state.reg) return needReg();
  const rules = state.reg.rules.filter(r => !sys || r.system === sys);
  return `<h1>Science${sys ? " · " + esc(SYS_NAMES[sys] ?? sys) : ""}</h1>
    <p class="sub">The 32 binding principles of science.md, with their implementations and proofs.</p>
    <div class="rows">` + rules.map(r => {
      const st = ruleStatus(r.id), n = ruleEdges(r.id).length;
      const sst = st?.status ?? "unknown";
      return `<a class="row" href="#/rule/${r.id}" style="box-shadow:inset 3px 0 0 var(--${sst})">
        <span class="id">R${r.id}</span>
        <span class="grow"><b>${esc(r.title)}</b> — <span style="color:var(--ink-2)">${esc(r.summary)}</span></span>
        <span class="s" style="color:var(--${sst})">${GLYPH[sst]} ${sst}${st?.verified ? " ✓✓" : ""}</span>
        <span class="num">${n} code</span></a>`;
    }).join("") + "</div>";
}

function vBench() {
  if (!state.reg) return needReg();
  return `<h1>Benchmarks</h1>
    <p class="sub">${state.reg.benchmarks.length} benchmarks. Harmonic mean is the north star; no metric may regress.</p>
    <div class="rows">` + state.reg.benchmarks.map(b => {
      const s = b.current_score, n = benchEdges(b.id).length;
      return `<a class="row" href="#/bench/${b.id}"><span class="id">${b.id}</span>
        <span class="grow">${esc(b.measures)} <span class="pill">${esc(b.family)}</span></span>
        <span class="num">${n} code</span>
        <span class="bartrack"><span class="bar" style="width:${(s ?? 0) * 100}%;background:${bandColor(s)}"></span></span>
        <span class="num" style="min-width:4.5em;text-align:right;color:${bandColor(s)}">${s == null ? "—" : s.toFixed(3)}</span></a>`;
    }).join("") + "</div>";
}

function vBenchOne(id) {
  const b = state.reg?.benchmarks.find(x => x.id === id);
  if (!b) return needReg();
  const edges = benchEdges(id);
  const evb = state.reg.blocks.find(x => x.path === "autoresearch/evaluate.py" && x.name === "evaluate_benchmarks");
  const scorer = evb ? `<div class="rows">${mkRow(evb.id, "⚖", "the frozen evaluator — computes and scores every benchmark")}</div>` : "";
  route.after = armRows;
  return `<h1>${b.id} · ${esc(b.measures)}</h1>
    <p class="sub">${esc(b.inputs)} → ${esc(b.target)} · family: ${esc(b.family)} ·
      score: <b style="color:${bandColor(b.current_score)}">${b.current_score?.toFixed(4) ?? "inactive"}</b></p>
    <h2>Connected code — click any row to read it</h2>${edgeSections(edges, scorer)}`;
}

function vData() {
  if (!state.reg) return needReg();
  route.after = () => initMap(location.hash.split("/")[2]);
  return `<h1>Data</h1>
    <p class="sub">Every observation the model trains on — pinned to earth at (x,y,z,t),
      colored by the spatial holdout. Click a point for its full record.</p>
    <div class="filterbar">
      <select id="fsplit"><option value="">train + test</option>
        <option value="train">train</option><option value="test">test</option></select>
      <input id="fsp" placeholder="search species — e.g. Quercus" autocomplete="off">
      <div class="spdrop" id="spdrop" hidden></div>
      <span class="count" id="obscount"></span>
    </div>
    <div class="maprow">
      <div id="map"></div>
      <div class="obspanel" id="obspanel"><p class="sub" style="margin:0">No observation selected.
        Points: <b style="color:var(--code)">train</b> · <b style="color:var(--data)">test</b> —
        the model never sees a test cell (0.5° blocks, 1/6 held out).</p></div>
    </div>
    ${modalityCensus()}`;
}

/* ---- modality census: the dataset as it actually exists on disk ---- */
const human = b => b > 2 ** 30 ? (b / 2 ** 30).toFixed(1) + " GB" : (b / 2 ** 20).toFixed(0) + " MB";
function modalityCensus() {
  const f = state.flow;
  if (!f) return `<h2>Modalities</h2>` + empty("Run <code>python -m dashboard.flow</code> for the real census.");
  const rows = f.modalities.map(m => {
    const main = Object.entries(m.keys).sort((a, b) => b[1][0].length - a[1][0].length)[0];
    const cov = m.coverage;
    return `<div class="row">
      <code class="grow" title="${esc(Object.entries(m.keys).map(([k, v]) => `${k} ${JSON.stringify(v[0])} ${v[1]}`).join("\n"))}">
        ${esc(m.name)}${m.files > 1 ? ` <span class="num">× ${m.files}</span>` : ""}
        ${m.stray ? `<span class="s" style="color:var(--serious)">▲ stray — loaded by nothing</span>` : ""}</code>
      <code class="num">${esc(main[0])} ${esc(JSON.stringify(main[1][0]))} ${esc(main[1][1])}</code>
      <span class="num">${human(m.bytes)}</span>
      <span class="bartrack" style="flex-basis:110px"><span class="bar" style="width:${(cov ?? 0) * 100}%"></span></span>
      <span class="num" style="min-width:4em;text-align:right">${cov == null ? "—" : (cov * 100).toFixed(1) + "%"}</span>
      ${m.loaders.slice(0, 1).map(l => `<a class="pill" href="#/file/${l.split(":")[0]}:${l.split(":")[1]}-${l.split(":")[1]}">${esc(l)}</a>`).join("")}
    </div>`;
  }).join("");
  return `<h2>Modalities — the dataset as it exists on disk (${f.modalities.length} artifacts,
      ${f.n_observations.toLocaleString()} observations)</h2>
    <p class="sub">Read from the cache file headers, joined on gbifID for coverage; each row links the
      exact loader line. Hover a name for every key, shape, and dtype.</p>
    <div class="rows">${rows}</div>`;
}

/* ---- Flow: the executed model — every nn.Module of a real forward pass on a real batch ---- */
const fmtS = s => (s && s[0] ? s[0].join("×") : "—");
const fmtP = p => p >= 1e6 ? (p / 1e6).toFixed(1) + "M" : p >= 1e3 ? (p / 1e3).toFixed(1) + "k" : String(p);

const srcLink = (e, inner) => e.file                      // torch builtins have no repo source
  ? `<a href="#/file/${e.file}:${e.line}-${e.line}">${inner}</a>`
  : `<span title="${esc(e.cls)} is a torch builtin — defined by PyTorch, not this repo">${inner}</span>`;

function bandHtml(b) {
  const comp = b.events.find(e => e.name === b.top);
  const params = b.events.reduce((s, e) => s + e.params, 0);
  const out = comp ?? b.events[b.events.length - 1];
  return `<div class="fband">
    <div class="fband-h">
      ${srcLink(out, `<b>${esc(b.top)}</b> <span class="num">${esc(out.cls)}${out.file ? "" : " · torch"}</span>`)}
      <span class="num">${fmtP(params)} params · out ${fmtS(out.out)}${out.sample
        ? ` · μ=${out.sample.mean} σ=${out.sample.std}` : ""}</span>
    </div>
    <div class="fsubs">${subGroups(b.events.filter(e => e.name !== b.top), b.top)}</div></div>`;
}

function subGroups(evts, top) {
  const subs = {};
  for (const e of evts) {
    const s = top ? e.name.slice(top.length + 1).split(".")[0] : e.name.split(".")[0];
    (subs[s] ??= []).push(e);
  }
  return Object.entries(subs).map(([s, es]) => {
    const head = es.find(e => e.name.endsWith("." + s) || e.name === s) ?? es[0];
    const src = es.find(e => e.file) ?? head;             // prefer a repo-sourced member for the link
    const body = `
      <div class="fsub-h">${esc(s)} <span>${esc(head.cls)}${head.file || src.file ? "" : " · torch"}</span></div>
      <div class="num">${fmtS(head.in)} → ${fmtS(head.out)}</div>
      <div class="num">${fmtP(es.reduce((x, e) => x + e.params, 0))} params · ${es.length} call${es.length > 1 ? "s" : ""}</div>`;
    const title = `${esc(head.name)} · ${esc(head.cls)} · in ${fmtS(head.in)} → out ${fmtS(head.out)}${head.sample
      ? `\nreal output values [${head.sample.first.join(", ")}…] μ=${head.sample.mean} σ=${head.sample.std}` : ""}`;
    return src.file
      ? `<a class="fsub" href="#/file/${src.file}:${src.line}-${src.line}" title="${title}">${body}</a>`
      : `<span class="fsub" title="${title}\n(torch builtin — no repo source)">${body}</span>`;
  }).join("");
}

function vFlow() {
  const t = state.trace;
  if (!t) return `<h1>Flow</h1>` + empty(`No execution trace yet. Run
    <code>python -m dashboard.trace data/deepcal/ckpt_&lt;tag&gt;.pt</code> — it hooks every nn.Module,
    pushes a real batch through the real forward pass, and records what executed.`);
  const inf = t.events.filter(e => e.phase === "inference");
  const infNames = new Set(inf.map(e => e.name));
  const trn = t.events.filter(e => e.phase === "training-loss" && !infNames.has(e.name));
  const bands = [], seen = {};
  for (const e of inf) {
    const top = e.name.split(".")[0];
    if (!(top in seen)) { seen[top] = bands.length; bands.push({ top, events: [] }); }
    bands[seen[top]].events.push(e);
  }
  const B = t.batch;
  const obsChips = B.gbifIDs.map((g, i) => `<a class="chip" href="#/data/${g}"
      title="(x,y,z) = ${B.coords[i].slice(0, 3).join(", ")}"><i>${esc(B.species[i])}</i></a>`).join(" ");
  const varRows = Object.entries(B.variables).map(([k, v]) => `<div class="row">
      <b class="grow">${esc(k)}</b><code class="num">${v.shape.join("×")} ${esc(v.dtype)}</code>
      ${v.observed != null ? `<span class="num" title="fraction of this batch where the variable is
        observed (given) rather than masked">given ${(v.observed * 100).toFixed(0)}%</span>` : ""}
      <code class="num" title="first values of the real tensor">[${(v.sample?.first ?? []).join(", ")} …]</code></div>`).join("");
  const outRows = B.outputs.top3.map((t3, i) => `<div class="row">
      <i class="grow">${esc(B.species[i])}</i><span class="num">→</span>
      ${t3.map(([n, p]) => `<span class="chip"
        style="${n === B.species[i] ? "border-color:var(--good)" : ""}">${esc(n)} ${(p * 100).toFixed(1)}%</span>`).join("")}
    </div>`).join("");
  return `<h1>Flow</h1>
    <p class="sub">The model as it actually executed: <b>${t.events.length} module calls</b> recorded by
      forward hooks on the ${(t.n_params / 1e6).toFixed(1)}M-parameter checkpoint
      <code>${esc(t.ckpt.split("/").pop())}</code>, over a batch of ${B.gbifIDs.length} real held-out
      observations. Every box is a real nn.Module — its code name, class, parameters, and the real
      tensor shapes and values it saw. Click any box to open its source.</p>
    <h2>The batch — ${B.gbifIDs.length} real observations</h2>
    <div style="margin-bottom:.6rem">${obsChips}</div>
    <div class="rows">${varRows}</div>
    <h2>Forward pass — execution order</h2>
    ${bands.map(bandHtml).join(`<div class="farrow">↓</div>`)}
    <div class="farrow">↓</div>
    <h2>Posterior — <code>${esc(B.outputs.target)}</code> masked, inferred from everything else</h2>
    <div class="rows">${outRows}</div>
    <p class="sub">training-step masked-reconstruction loss on this batch: <b>${B.outputs.loss}</b></p>
    ${trn.length ? `<h2>Training-path extras — modules that only run in the loss pass</h2>
      <div class="fband"><div class="fsubs">${subGroups(trn, "")}</div></div>` : ""}`;
}


/* compact climate strip: tmax/tmin lines (°C) + prcp bars (mm) over 180 days */
function climateChart(c) {
  const i = Object.fromEntries(c.cols.map((k, j) => [k, j]));
  const rows = c.rows, W = 296, H = 110, P = 6;
  const col = k => rows.map(r => r[i[k]]).map(v => v == null ? null : v);
  const tmax = col("tmax"), tmin = col("tmin"), prcp = col("prcp");
  const fin = vs => vs.filter(v => v != null);
  const t0 = Math.min(...fin(tmin)), t1 = Math.max(...fin(tmax)), p1 = Math.max(...fin(prcp), 1);
  const X = j => P + j / (rows.length - 1) * (W - 2 * P);
  const Yt = v => 8 + (t1 - v) / (t1 - t0 || 1) * (H - 40);
  const line = (vs, c) => `<polyline fill="none" stroke="${c}" stroke-width="1.6"
    points="${vs.map((v, j) => v == null ? "" : `${X(j).toFixed(1)},${Yt(v).toFixed(1)}`).join(" ")}"/>`;
  const bars = prcp.map((v, j) => v > 0 ? `<rect x="${(X(j) - .8).toFixed(1)}" width="1.6"
    y="${(H - 4 - v / p1 * 26).toFixed(1)}" height="${(v / p1 * 26).toFixed(1)}" fill="var(--bench)" opacity=".55"/>` : "").join("");
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%">${bars}
      ${line(tmax, "var(--data)")}${line(tmin, "var(--code)")}</svg>
    <div class="legend" style="margin-top:0;font-size:.75rem">
      <span><span class="dot" style="background:var(--data)"></span>tmax ${t1.toFixed(0)}°C</span>
      <span><span class="dot" style="background:var(--code)"></span>tmin ${t0.toFixed(0)}°C</span>
      <span><span class="dot" style="background:var(--bench)"></span>prcp ≤${p1.toFixed(0)}mm</span></div>`;
}

/* ---- map (Leaflet + ESRI World Imagery) ---- */
const mapState = { sp: null };
async function initMap(deepId) {
  if (typeof L === "undefined") { $("#map").textContent = "Leaflet unavailable (offline?)"; return; }
  const map = L.map("map", { renderer: L.canvas(), preferCanvas: true });
  L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    { attribution: "Esri World Imagery", maxZoom: 19 }).addTo(map);
  map.setView([37.3, -119.5], 6);
  let layer = null;
  async function draw() {
    const q = new URLSearchParams();
    if ($("#fsplit").value) q.set("split", $("#fsplit").value);
    if (mapState.sp != null) q.set("sp", mapState.sp);
    const o = await api("observations?" + q);
    if (!o) { $("#obscount").textContent = "no index — run dashboard.observations"; return; }
    $("#obscount").textContent = `${o.shown.toLocaleString()} of ${o.total.toLocaleString()} shown`;
    if (layer) layer.remove();
    const recIds = new Set(Object.keys(state.recon?.rows ?? {}));
    layer = L.layerGroup(o.id.map((id, i) => {
      const isRec = recIds.has(String(id));
      return L.circleMarker([o.lat[i], o.lon[i]], {
        radius: isRec ? 7 : 4.5, weight: isRec ? 2.2 : 1, color: "#fff", fillOpacity: .85,
        fillColor: o.test[i] ? "#eb6834" : "#3987e5",
      }).on("click", () => showObs(id));
    }), { pane: "markerPane" }).addTo(map);
  }
  async function showObs(id, pan = false) {
    const d = await api("observation/" + id);
    if (!d) return;
    if (pan) map.setView([d.lat, d.lon], 11);
    $("#obspanel").innerHTML = `<div class="latin">${esc(d.species)}</div>
      <span class="chip" style="border-color:var(--${d.split === "test" ? "data" : "code"})">${d.split}</span>
      <dl class="kv">
        <dt>x, y</dt><dd>${d.lat.toFixed(5)}, ${d.lon.toFixed(5)}</dd>
        <dt>z</dt><dd>${d.elev.toFixed(1)} m</dd>
        <dt>t</dt><dd>${d.date ?? "unknown"}</dd>
        <dt>gbifID</dt><dd><a href="${d.source}" target="_blank" style="color:var(--code)">${d.gbifID} ↗</a></dd>
      </dl>
      <div>${d.modalities.map(m => `<span class="chip">${m}</span>`).join("")}</div>
      <div id="obsphoto"></div>
      <div id="rawrec"><p class="sub" style="font-size:.8rem">loading raw record…</p></div>`;
    fetch(`https://api.gbif.org/v1/occurrence/${id}`).then(r => r.json()).then(g => {
      const img = (g.media ?? []).find(m => m.identifier);
      if (img) $("#obsphoto").innerHTML = `<img src="${img.identifier}" alt="${esc(d.species)}"
        style="width:100%;border-radius:4px;margin-top:.6rem" loading="lazy">
        <p class="sub" style="font-size:.74rem;margin:.15rem 0 0">${esc(img.rightsHolder ?? g.recordedBy ?? "")} ·
        the actual observation photo (DINOv2/BioCLIP source)</p>`;
    }).catch(() => {});
    const rec = state.recon?.rows?.[id];
    const recHtml = !rec ? "" : `<h4>Model reconstruction — each variable masked, predicted from the rest</h4>` +
      Object.entries(rec).map(([t, r]) => r.top ? `<div style="font-size:.82rem;margin:.3rem 0"><b>${esc(t)}</b>
          <span style="color:var(--${r.rank === 0 ? "good" : r.rank < 5 ? "warning" : "serious"})">rank ${r.rank}</span>
          ${r.top.map(([n, p], i) => `<div style="display:flex;gap:.5rem;align-items:center">
            <span class="bartrack" style="flex:0 0 90px"><span class="bar" style="width:${p * 100}%;${n === r.true ? "" : "opacity:.45"}"></span></span>
            <i style="${n === r.true ? "font-weight:650" : "color:var(--ink-2)"}">${esc(n)}</i>
            <span class="num">${(p * 100).toFixed(1)}%</span></div>`).join("")}
          <span style="color:var(--muted);font-size:.78rem">truth: <i>${esc(r.true)}</i></span></div>`
        : `<span class="chip" title="cosine to ground truth">${esc(t)} ${r.cos}</span>`).join("");
    const raw = await api(`observation/${id}/raw`);
    if (!raw) { $("#rawrec").outerHTML = recHtml; return; }
    const kv = obj => `<dl class="kv" style="font-size:.8rem">` +
      Object.entries(obj).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v ?? "—"}</dd>`).join("") + "</dl>";
    const t2 = (la, lo, z) => [z, Math.floor((lo + 180) / 360 * 2 ** z),
      Math.floor((1 - Math.log(Math.tan(la * Math.PI / 180) + 1 / Math.cos(la * Math.PI / 180)) / Math.PI) / 2 * 2 ** z)];
    const [az, ax, ay] = t2(d.lat, d.lon, 16), [sz, sx, sy] = t2(d.lat, d.lon, 13);
    $("#rawrec").innerHTML = recHtml +
      `<h4>Imagery at this coordinate</h4><div class="imgpair">
        <figure><img src="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${az}/${ay}/${ax}" loading="lazy">
          <figcaption>aerial · Esri World Imagery z16 (NAIP-derived in CA)</figcaption></figure>
        <figure><img src="https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/${sz}/${sy}/${sx}.jpg" loading="lazy">
          <figcaption>Sentinel-2 · EOX s2cloudless z13 (Clay's sensor)</figcaption></figure></div>` +
      (raw.climate ? `<h4>Daymet — 180 days to observation</h4>${climateChart(raw.climate)}` : "") +
      (raw.soil ? `<h4>SSURGO soil</h4>${kv(raw.soil)}` : "") +
      (raw.topo ? `<h4>3DEP terrain (1 m)</h4>${kv(raw.topo)}` : "") +
      (raw.chm ? `<h4>NAIP-CHM canopy structure</h4>${kv(raw.chm)}` : "") +
      (raw.hydro ? `<h4>Hydrology + wind (3DEP 2 m)</h4>${kv(raw.hydro)}` : "");
  }
  if (deepId) showObs(deepId, true);
  $("#fsplit").onchange = draw;
  const inp = $("#fsp"), drop = $("#spdrop");
  inp.oninput = async () => {
    mapState.sp = null;
    if (inp.value.length < 2) { drop.hidden = true; draw(); return; }
    const hits = await api("species?q=" + encodeURIComponent(inp.value));
    drop.innerHTML = (hits ?? []).map(h =>
      `<div data-sp="${h.sp}"><i>${esc(h.name)}</i> <span style="color:var(--muted)">${h.n}</span></div>`).join("");
    drop.hidden = !hits?.length;
    drop.querySelectorAll("div").forEach(el => el.onclick = () => {
      mapState.sp = +el.dataset.sp; inp.value = el.querySelector("i").textContent;
      drop.hidden = true; draw();
    });
  };
  draw();
}

/* ---- graph (tripartite: code | science | benchmarks) ---- */
function vGraph() {
  if (!state.reg) return needReg();
  if (!state.graph) return `<h1>Graph</h1>` + empty("No graph — run <code>python -m dashboard.audit</code>.");
  route.after = armGraph;
  return `<h1>Graph</h1>
    <p class="sub"><b>Click a node to pin it</b> — rules light their files on the left and their
      benchmarks on the right · <b>click the pinned node again to open it</b> · click empty space
      to release. Color carries state: rules and files wear audit status, benchmarks wear score
      bands.</p>
    <div id="gpreview" class="gpreview">click a node to explore its connections</div>
    <div id="graphbox" style="overflow-x:auto"></div>`;
}

const bandColor = s => s == null ? "var(--muted)"
  : s >= .7 ? "var(--good)" : s >= .35 ? "var(--warning)" : s > 0 ? "var(--serious)" : "var(--critical)";

function armGraph() {
  const files = state.reg.files.filter(f => f.kind === "text").map(f => f.path).sort();
  const rules = state.reg.rules, benches = state.reg.benchmarks;
  const fEdges = {}, blockDsts = {};
  for (const e of state.graph.edges) {
    const f = e.src.split(":")[0];
    (fEdges[f] ??= {})[e.dst] = Math.max(fEdges[f][e.dst] ?? 0, e.s);
    (blockDsts[e.src] ??= []).push([e.dst, e.s]);
  }
  const cross = {};                                      // direct rule->benchmark links via shared blocks
  for (const dsts of Object.values(blockDsts))
    for (const [r, rs] of dsts.filter(d => d[0][0] === "R"))
      for (const [b, bs] of dsts.filter(d => d[0][0] === "B"))
        cross[r + "|" + b] = Math.max(cross[r + "|" + b] ?? 0, Math.min(rs, bs));
  const STATUS_NAMES = Object.keys(RANK);
  const fileStatus = f => {
    const rs = Object.keys(fEdges[f] ?? {}).filter(d => d[0] === "R");
    return rs.length ? STATUS_NAMES[rs.reduce((w, d) =>
      Math.min(w, RANK[ruleStatus(+d.slice(1))?.status ?? "unknown"]), 4)] : null;
  };
  const RH = 15, P = 30, W = 1380, CX = [330, 610], BX = 950, XR = 712;
  const H = P * 2 + Math.max(files.length, rules.length * 2, benches.length) * RH + 20;
  const fy = i => P + 14 + i * RH;
  const ry = i => P + 14 + i * RH * 2 + (H - 2 * P - rules.length * RH * 2) / 2;
  const by = i => P + 14 + i * RH * (files.length / benches.length > 1 ? files.length / benches.length : 1);
  const fi = Object.fromEntries(files.map((f, i) => [f, i]));
  const ri = Object.fromEntries(rules.map((r, i) => ["R" + r.id, i]));
  const bi = Object.fromEntries(benches.map((b, i) => [b.id, i]));
  let paths = "", nodes = "";
  for (const [f, dsts] of Object.entries(fEdges)) {
    if (!(f in fi)) continue;
    for (const [d, s] of Object.entries(dsts)) {
      const [x1, y1] = [CX[0], fy(fi[f])];
      const [x2, y2, cls] = d.startsWith("R") ? [CX[1] - 150, ry(ri[d]), "science"]
        : [BX, by(bi[d] ?? 0), "bench"];
      if (d.startsWith("B") && !(d in bi)) continue;
      paths += `<path d="M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}"
        fill="none" stroke="var(--${cls === "science" ? "science" : "bench"})" stroke-width="${s * .7}"
        class="ge" data-f="${f}" data-d="${d}" opacity=".07"/>`;
    }
  }
  let xpaths = "";                                      // the right span: rule -> benchmark
  for (const [k, s] of Object.entries(cross)) {
    const [r, b] = k.split("|");
    if (!(r in ri) || !(b in bi)) continue;
    const [y1, y2] = [ry(ri[r]), by(bi[b])];
    xpaths += `<path d="M${XR},${y1} C${(XR + BX) / 2},${y1} ${(XR + BX) / 2},${y2} ${BX + 2},${y2}"
      fill="none" stroke="var(--bench)" stroke-width="${s * .6}" class="gc"
      data-r="${r}" data-b="${b}" opacity=".05"/>`;
  }
  const label = (x, y, key, txt, anchor, cls, href, tip = "", fill = "") =>
    `<text x="${x}" y="${y + 4}" text-anchor="${anchor}" font-size="10.5" class="gn ${cls}"
       data-key="${esc(key)}" data-href="${href}" style="cursor:pointer${fill ? `;fill:${fill}` : ""}">${
       tip ? `<title>${esc(tip)}</title>` : ""}${esc(txt)}</text>`;
  files.forEach((f, i) => {
    const st = fileStatus(f);
    nodes += label(CX[0] - 8, fy(i), f, f.length > 44 ? "…" + f.slice(-42) : f, "end", "gf",
      "#/file/" + f, "", st && st !== "good" ? `var(--${st})` : "");
  });
  rules.forEach((r, i) => {
    const st = ruleStatus(r.id)?.status ?? "unknown";
    const t = `R${r.id} ${r.title}`;
    nodes += `<circle cx="${CX[1] - 150 + 8}" cy="${ry(i)}" r="4.5" fill="var(--${st})"/>` +
      label(CX[1] - 150 + 18, ry(i), "R" + r.id, t.length > 36 ? t.slice(0, 34) + "…" : t,
        "start", "gr", "#/rule/" + r.id, r.summary, st !== "good" ? `var(--${st})` : "");
  });
  benches.forEach((b, i) => {
    const sc = b.current_score;
    nodes += `<circle cx="${BX + 8}" cy="${by(i)}" r="4" fill="${bandColor(sc)}"/>` +
      label(BX + 18, by(i), b.id,
        `${b.id} · ${b.measures.length > 40 ? b.measures.slice(0, 38) + "…" : b.measures}`,
        "start", "gb", "#/bench/" + b.id, `${b.measures} — score ${sc?.toFixed(3) ?? "inactive"}`,
        sc != null && sc < .35 ? bandColor(sc) : "");
  });
  $("#graphbox").innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="min-width:1250px;width:100%">
    <text x="${CX[0] - 8}" y="${P - 6}" text-anchor="end" font-size="12" font-weight="650" fill="var(--code)">CODE</text>
    <text x="${CX[1] - 142}" y="${P - 6}" font-size="12" font-weight="650" fill="var(--science)">SCIENCE</text>
    <text x="${BX + 6}" y="${P - 6}" font-size="12" font-weight="650" fill="var(--bench)">BENCHMARKS</text>
    ${paths}${xpaths}${nodes}</svg>`;
  const box = $("#graphbox"), prev = $("#gpreview"), svg = box.querySelector("svg");
  let locked = null, onEls = [];
  const eF = {}, eD = {}, xRi = {}, xBi = {};            // element indexes: O(hits) focus, no full scans
  box.querySelectorAll(".ge").forEach(p => {
    (eF[p.dataset.f] ??= []).push(p); (eD[p.dataset.d] ??= []).push(p);
  });
  box.querySelectorAll(".gc").forEach(p => {
    (xRi[p.dataset.r] ??= []).push(p); (xBi[p.dataset.b] ??= []).push(p);
  });
  const nodeEl = {};
  box.querySelectorAll(".gn").forEach(n => nodeEl[n.dataset.key] = n);
  function applyFocus(key) {
    for (const el of onEls) el.classList.remove("on", "focusnode");
    onEls = [];
    const hits = !key ? [] : /^R\d+$/.test(key) ? [...(eD[key] ?? []), ...(xRi[key] ?? [])]
      : /^B\d+$/.test(key) ? [...(eD[key] ?? []), ...(xBi[key] ?? [])]
      : eF[key] ?? [];
    svg.classList.toggle("dimmed", hits.length > 0);     // an unconnected node never blanks the fabric
    if (!key) return;
    const conn = new Set([key]);
    for (const p of hits) {
      p.classList.add("on"); onEls.push(p);
      conn.add(p.dataset.f ?? p.dataset.r); conn.add(p.dataset.d ?? p.dataset.b);
    }
    conn.delete(undefined);
    for (const k of conn) if (nodeEl[k]) { nodeEl[k].classList.add("on"); onEls.push(nodeEl[k]); }
    nodeEl[key]?.classList.add("focusnode");
  }
  function preview(key) {
    if (!key) {
      prev.innerHTML = "click a node to explore its connections";
      return;
    }
    let h;
    if (/^R\d+$/.test(key)) {
      const r = rules.find(x => "R" + x.id === key), st = ruleStatus(r.id);
      const nb = Object.keys(cross).filter(k => k.startsWith(key + "|")).length;
      h = `<b>${key} · ${esc(r.title)}</b>
           <span class="s" style="color:var(--${st?.status ?? "unknown"})">${GLYPH[st?.status ?? "unknown"]} ${st?.status ?? "unknown"}</span>
           — <span style="color:var(--ink-2)">${esc(r.summary)}</span>
           <span class="num">· ${ruleEdges(r.id).length} code links · ${nb} benchmarks</span>`;
    } else if (/^B\d+$/.test(key)) {
      const b = benches.find(x => x.id === key);
      h = `<b>${key} · ${esc(b.measures)}</b> <span class="pill">${esc(b.family)}</span>
           — <span style="color:var(--ink-2)">${esc(b.inputs)} → ${esc(b.target)}</span>
           <span class="s" style="color:${bandColor(b.current_score)}">● ${b.current_score?.toFixed(3) ?? "inactive"}</span>`;
    } else {
      const f = state.reg.files.find(x => x.path === key), st = fileStatus(key);
      h = `<b>${esc(key)}</b> <span class="num">· ${f?.lines ?? "?"} lines · ${blockEdges(key).length} connections</span>
           ${st ? `<span class="s" style="color:var(--${st})">${GLYPH[st]} worst linked rule: ${st}</span>` : ""}`;
    }
    prev.innerHTML = h + (locked === key
      ? ` <span class="pill" style="border-color:var(--code)">pinned — click it again to open</span>` : "");
  }
  box.onclick = e => {                                   // click-only: no hover state, nothing to glitch
    const n = e.target.closest(".gn");
    if (!n) { locked = null; applyFocus(null); preview(null); return; }
    if (locked === n.dataset.key) { location.hash = n.dataset.href; return; }
    locked = n.dataset.key;
    applyFocus(locked); preview(locked);
  };
}

/* ---- runs (live training, TensorBoard-style) ---- */
async function vRuns() {
  const runs = await api("runs");
  if (!runs?.length) return `<h1>Runs</h1>` + empty(
    `No runs yet. Launch one with <code>python -m dashboard.tracker --cache /path/to/deepcal</code> —
     train.py output passes through untouched and streams here live.`);
  const done = runs.filter(r => r.last?.t === "final" && r.last.scores?.benchmarks
    && Object.keys(r.last.scores.benchmarks).length);
  const opts = sel => done.map((r, i) =>
    `<option value="${esc(r.id)}" ${i === sel ? "selected" : ""}>${esc(r.id)}</option>`).join("");
  route.after = () => armCompare(done);
  return `<h1>Runs</h1><p class="sub">Every tracked experiment. Live runs update in place.</p>
    <div class="rows">` + runs.map(r => {
      const fin = r.last?.t === "final", s = fin ? r.last.scores : null;
      return `<a class="row" href="#/run/${r.id}">
        <span class="grow"><b>${esc(r.id)}</b></span>
        ${fin ? `<span class="num">H ${s.net_score?.toFixed(4) ?? "—"}</span>
                 <span class="num">A ${s.arithmetic?.toFixed(4) ?? "—"}</span>
                 <span class="num">${s.peak_vram_mb ? (s.peak_vram_mb / 1024).toFixed(1) + " GB" : ""}</span>`
              : `<span class="s" style="color:var(--good)">● live</span>`}</a>`;
    }).join("") + `</div>
    ${done.length >= 2 ? `<h2>Compare — every benchmark, A vs B</h2>
      <div class="filterbar">
        <select id="cmpA">${opts(done.length - 1)}</select><span class="num">vs</span>
        <select id="cmpB">${opts(0)}</select>
      </div><div id="cmpbox"></div>` : ""}`;
}

function armCompare(done) {
  const box = $("#cmpbox");
  if (!box) return;
  const byId = Object.fromEntries(done.map(r => [r.id, r.last.scores]));
  const draw = () => {
    const A = byId[$("#cmpA").value], B = byId[$("#cmpB").value];
    const keys = Object.keys(A.benchmarks).filter(k => k in B.benchmarks);
    const rows = keys.map(k => [k, A.benchmarks[k], B.benchmarks[k], B.benchmarks[k] - A.benchmarks[k]])
      .sort((a, b) => Math.abs(b[3]) - Math.abs(a[3]));
    box.innerHTML = `<div class="rows">
      <div class="row"><b class="grow">aggregate</b>
        <span class="num">H ${A.net_score?.toFixed(4)} → ${B.net_score?.toFixed(4)}</span>
        <span class="num" style="color:${bandColor(.5 + (B.net_score - A.net_score) * 5)}">
          ${(B.net_score - A.net_score >= 0 ? "+" : "") + (B.net_score - A.net_score).toFixed(4)}</span>
        <span class="num">A ${A.arithmetic?.toFixed(4)} → ${B.arithmetic?.toFixed(4)}</span></div>
      ${rows.map(([k, a, b, d]) => `<div class="row"><code class="grow">${esc(k)}</code>
        <span class="num">${a.toFixed(3)} → ${b.toFixed(3)}</span>
        <span class="num" style="min-width:5em;text-align:right;color:${d > 0.005 ? "var(--good)" : d < -0.005 ? "var(--critical)" : "var(--muted)"}">
          ${(d >= 0 ? "+" : "") + d.toFixed(3)}</span></div>`).join("")}</div>`;
  };
  $("#cmpA").onchange = draw;
  $("#cmpB").onchange = draw;
  draw();
}

function lossChart(steps) {
  if (steps.length < 2) return `<div class="empty">waiting for step events…</div>`;
  const W = 860, H = 260, P = 42;
  const xs = steps.map(d => d.step), ys = steps.map(d => d.loss);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const X = v => P + (v - x0) / (x1 - x0 || 1) * (W - 2 * P);
  const Yv = v => (H - P) - (v - y0) / (y1 - y0 || 1) * (H - 2 * P);
  const pts = steps.map(d => `${X(d.step).toFixed(1)},${Yv(d.loss).toFixed(1)}`).join(" ");
  const ticksY = [y0, (y0 + y1) / 2, y1], ticksX = [x0, Math.round((x0 + x1) / 2), x1];
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px" id="losschart"
       data-pts='${JSON.stringify(steps.map(d => [d.step, d.loss]))}'>
    ${ticksY.map(v => `<line x1="${P}" x2="${W - P}" y1="${Yv(v)}" y2="${Yv(v)}" stroke="var(--grid)"/>
      <text x="${P - 6}" y="${Yv(v) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${v.toFixed(2)}</text>`).join("")}
    ${ticksX.map(v => `<text x="${X(v)}" y="${H - P + 16}" text-anchor="middle" font-size="11" fill="var(--muted)">${v}</text>`).join("")}
    <line x1="${P}" x2="${W - P}" y1="${H - P}" y2="${H - P}" stroke="var(--line)"/>
    <polyline points="${pts}" fill="none" stroke="var(--seq)" stroke-width="2"/>
    <line id="xh" y1="${P}" y2="${H - P}" stroke="var(--line)" stroke-dasharray="3 3" visibility="hidden"/>
    <circle id="xhc" r="4" fill="var(--seq)" stroke="#fff" stroke-width="1.5" visibility="hidden"/>
    <text id="xht" font-size="11.5" fill="var(--ink)" visibility="hidden"></text></svg>`;
}

function armChart() {
  const svg = $("#losschart");
  if (!svg) return;
  const pts = JSON.parse(svg.dataset.pts), W = 860, P = 42;
  const x0 = pts[0][0], x1 = pts[pts.length - 1][0];
  svg.onmousemove = e => {
    const r = svg.getBoundingClientRect();
    const step = x0 + (e.clientX - r.left) / r.width * W > 0 ?
      x0 + ((e.clientX - r.left) / r.width * W - P) / (W - 2 * P) * (x1 - x0) : x0;
    const near = pts.reduce((a, b) => Math.abs(b[0] - step) < Math.abs(a[0] - step) ? b : a);
    const ys = pts.map(p => p[1]), yMin = Math.min(...ys), yMax = Math.max(...ys);
    const X = P + (near[0] - x0) / (x1 - x0 || 1) * (W - 2 * P);
    const Y = (260 - P) - (near[1] - yMin) / (yMax - yMin || 1) * (260 - 2 * P);
    $("#xh").setAttribute("x1", X); $("#xh").setAttribute("x2", X);
    $("#xhc").setAttribute("cx", X); $("#xhc").setAttribute("cy", Y);
    const t = $("#xht");
    t.setAttribute("x", Math.min(X + 8, W - 150)); t.setAttribute("y", P + 12);
    t.textContent = `step ${near[0]} · loss ${near[1].toFixed(3)}`;
    for (const id of ["xh", "xhc", "xht"]) $("#" + id).setAttribute("visibility", "visible");
  };
  svg.onmouseleave = () => { for (const id of ["xh", "xhc", "xht"]) $("#" + id)?.setAttribute("visibility", "hidden"); };
}

async function vRun(rid) {
  const r = await api(`runs/${rid}`);
  if (!r) return empty("run not found");
  const ev = r.events;
  const render = evs => {
    const steps = evs.filter(e => e.t === "step");
    const start = evs.find(e => e.t === "startup"), fin = evs.find(e => e.t === "final");
    const transfer = evs.filter(e => e.t === "transfer").at(-1);
    const bench = fin?.scores?.benchmarks ?? {};
    const champ = Object.fromEntries((state.reg?.benchmarks ?? []).map(b => [b.key, b.current_score]));
    const done = evs.find(e => e.t === "trained");
    const bsz = state.flow?.arch?.dims?.batch;
    const exposure = done && start && bsz
      ? ` · data exposure: ${done.steps.toLocaleString()} steps × ${bsz} = ${(done.steps * bsz).toLocaleString()}
         samples ≈ ${(done.steps * bsz / start.train).toFixed(1)}× the training split — the
         ${start.test.toLocaleString()} held-out rows are never sampled` : "";
    return `<h1>${esc(rid)} ${fin ? "" : '<span class="s" style="color:var(--good);font-size:1rem">● live</span>'}</h1>
      <p class="sub">${start ? `${start.observations.toLocaleString()} observations · ${start.params_m}M parameters ·
        ${start.train.toLocaleString()} train / ${start.test.toLocaleString()} test${exposure}` : "starting…"}</p>
      <h2>Loss</h2>${lossChart(steps)}
      ${transfer ? `<h2>Held-out transfer</h2><div class="rows">` +
        Object.entries(transfer.scores).map(([k, v]) => `<div class="row"><span class="grow">${esc(k)}</span>
          <span class="bartrack"><span class="bar" style="width:${v * 100}%"></span></span>
          <span class="num" style="min-width:4em;text-align:right">${v.toFixed(3)}</span></div>`).join("") + "</div>" : ""}
      ${fin ? `<h2>Final — H ${fin.scores.net_score?.toFixed(4) ?? "—"} · A ${fin.scores.arithmetic?.toFixed(4) ?? "—"} ·
          ${((fin.scores.peak_vram_mb ?? 0) / 1024).toFixed(1)} GB VRAM</h2>
        <div class="rows">` + Object.entries(bench).sort((a, b) => a[1] - b[1]).map(([k, v]) => {
          const d = champ[k] != null ? v - champ[k] : null;
          return `<div class="row"><code class="grow">${esc(k)}</code>
            <span class="num">${v.toFixed(3)}</span>
            <span class="num" style="min-width:5em;text-align:right;color:${d > 0 ? "var(--good)" : d < 0 ? "var(--critical)" : "var(--muted)"}">
              ${d == null ? "new" : (d >= 0 ? "+" : "") + d.toFixed(3)}</span></div>`;
        }).join("") + `</div><p class="sub" style="margin-top:.6rem">Delta vs committed champion.</p>` : ""}`;
  };
  route.after = () => {
    armChart();
    if (ev.some(e => e.t === "final")) return;
    let offset = r.offset, events = ev;
    window._poll = setInterval(async () => {
      const nxt = await api(`runs/${rid}?offset=${offset}`);
      if (!nxt?.events.length) return;
      offset = nxt.offset; events = events.concat(nxt.events);
      view.innerHTML = render(events); armChart();
      if (events.some(e => e.t === "final")) clearInterval(window._poll);
    }, 3000);
  };
  return render(ev);
}

/* ---- router ---- */
async function route() {
  const [_, p1, p2] = location.hash.split("/");
  document.querySelectorAll("nav a").forEach(a =>
    a.classList.toggle("active", a.hash === "#/" + (p1 || "status")));
  const r = { status: vStatus, graph: vGraph, flow: vFlow, code: vCode, science: () => vScience(p2),
              benchmarks: vBench, data: vData };
  route.after = null;
  clearInterval(window._poll);
  view.innerHTML = p1 === "file" ? await vFile(decodeURIComponent(location.hash.slice(7)))
    : p1 === "rule" ? vRule(p2)
    : p1 === "bench" ? vBenchOne(p2)
    : p1 === "runs" ? await vRuns()
    : p1 === "run" ? await vRun(p2)
    : (r[p1] ?? vStatus)();
  route.after?.();
  window.scrollTo(0, 0);
}
addEventListener("hashchange", route);
load();
