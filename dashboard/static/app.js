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
  [state.reg, state.graph, state.status] = await Promise.all([api("registry"), api("graph"), api("status")]);
  const meta = await api("meta");
  if (meta?.head) $("#meta").textContent = `${meta.head.sha} · ${meta.head.subject}` +
    (meta.audited ? ` · audited ${meta.audited}` : " · not yet audited");
  route();
}

const empty = msg => `<div class="empty">${msg}</div>`;
const needReg = () => empty(`No registry. Run <code>python -m dashboard.registry</code>.`);

/* ---- helpers over the graph ---- */
const edgesFor = pred => (state.graph?.edges ?? []).filter(pred);
const ruleEdges = id => edgesFor(e => e.dst === "R" + id);
const benchEdges = id => edgesFor(e => e.dst === id);
const blockEdges = path => edgesFor(e => e.src.startsWith(path + ":"));
const ruleStatus = id => state.status?.rules?.find(r => r.id === id);

function statusTile(href, status, name, headline, cls = "") {
  const st = status ?? "unknown";
  return `<a class="tile ${cls}" href="${href}">
    <span class="s" style="color:var(--${st})"><span class="dot" style="background:var(--${st})"></span>${GLYPH[st]} ${st}</span>
    <h3>${esc(name)}</h3><p>${esc(headline ?? "no audit yet")}</p></a>`;
}

/* ---- views ---- */
function vStatus() {
  if (!state.reg) return needReg();
  const sys = Object.entries(SYS_NAMES).map(([k, name]) => {
    const s = state.status?.systems?.[k];
    return statusTile("#/science/" + k, s?.status, name, s?.headline, "sys");
  }).join("");
  const rules = [...state.reg.rules]
    .sort((a, b) => (RANK[ruleStatus(a.id)?.status ?? "unknown"] - RANK[ruleStatus(b.id)?.status ?? "unknown"]) || a.id - b.id)
    .map(r => statusTile("#/rule/" + r.id, ruleStatus(r.id)?.status, `R${r.id} · ${r.title}`,
                         ruleStatus(r.id)?.headline)).join("");
  return `<h1>System status</h1>
    <p class="sub">Five systems, thirty-two principles. Every claim opens to its evidence.</p>
    <div class="tiles">${sys}</div><h2>Principles — worst first</h2><div class="tiles">${rules}</div>`;
}

function vRule(id) {
  const r = state.reg?.rules.find(x => x.id === +id);
  if (!r) return needReg();
  const st = ruleStatus(r.id);
  const edges = ruleEdges(r.id);
  return `<h1>R${r.id} · ${esc(r.title)}</h1><p class="sub">${esc(r.summary)}</p>
    ${st ? `<div class="tiles">${statusTile("#", st.status, st.headline ?? "", st.next, "sys")}</div>` : ""}
    <h2>Implementing code — ${edges.length} connections</h2>
    <div class="rows">${edges.map(e => `<a class="row" href="#/file/${e.src.split(":")[0]}">
      <span class="id">s${e.s}</span><code class="grow">${esc(e.src)}</code>
      <span class="num">${esc(e.note)}</span></a>`).join("") || empty("No connections yet — run the audit.")}</div>`;
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
        const n = blockEdges(f.path).length;
        return `<a class="row" href="#/file/${f.path}"><code class="grow">${f.path}</code>
          <span class="num">${f.lines} ln</span>
          <span class="num" style="${n ? "" : "color:var(--serious)"}">${n || "—"} edges</span></a>`;
      }).join("") + "</div>").join("");
}

async function vFile(path) {
  const text = await (await fetch("/api/code/" + path)).text();
  const blocks = state.reg.blocks.filter(b => b.path === path);
  const edges = blockEdges(path);
  const perLine = {};
  for (const b of blocks) {
    const es = edges.filter(e => e.src === b.id);
    if (es.length) perLine[b.start] = es;
  }
  const lines = text.split("\n").map((l, i) => {
    const anno = (perLine[i + 1] ?? []).map(e =>
      `<a class="pill ${e.dst.startsWith("R") ? "science" : "bench"}" href="#/${e.dst.startsWith("R") ? "rule/" + e.dst.slice(1) : "bench/" + e.dst}">${e.dst}</a>`).join(" ");
    return `<div class="cl">${anno ? `<div class="anno">${anno}</div>` : ""}<span class="ln">${i + 1}</span>${esc(l) || " "}</div>`;
  }).join("");
  return `<h1><code>${path}</code></h1>
    <p class="sub">${blocks.length} blocks · ${edges.length} connections</p>
    <style>.cl{white-space:pre;font:12.5px/1.5 ui-monospace,Menlo,monospace}
      .cl:hover{background:var(--surface)}.ln{color:var(--muted);display:inline-block;width:3.2em;text-align:right;margin-right:1.2em;user-select:none}
      .anno{padding-left:4.4em}</style>
    <div style="overflow-x:auto;border-top:1px solid var(--grid);padding-top:.6rem">${lines}</div>`;
}

function vScience(sys) {
  if (!state.reg) return needReg();
  const rules = state.reg.rules.filter(r => !sys || r.system === sys);
  return `<h1>Science${sys ? " · " + esc(SYS_NAMES[sys] ?? sys) : ""}</h1>
    <p class="sub">The 32 binding principles of science.md, with their implementations and proofs.</p>
    <div class="rows">` + rules.map(r => {
      const st = ruleStatus(r.id), n = ruleEdges(r.id).length;
      return `<a class="row" href="#/rule/${r.id}"><span class="id">R${r.id}</span>
        <span class="grow"><b>${esc(r.title)}</b> — <span style="color:var(--ink-2)">${esc(r.summary)}</span></span>
        <span class="s" style="color:var(--${st?.status ?? "unknown"})">${GLYPH[st?.status ?? "unknown"]}</span>
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
        <span class="bartrack"><span class="bar" style="width:${(s ?? 0) * 100}%"></span></span>
        <span class="num" style="min-width:4.5em;text-align:right">${s == null ? "—" : s.toFixed(3)}</span></a>`;
    }).join("") + "</div>";
}

function vBenchOne(id) {
  const b = state.reg?.benchmarks.find(x => x.id === id);
  if (!b) return needReg();
  const edges = benchEdges(id);
  return `<h1>${b.id} · ${esc(b.measures)}</h1>
    <p class="sub">${esc(b.inputs)} → ${esc(b.target)} · family: ${esc(b.family)} ·
      score: <b>${b.current_score?.toFixed(4) ?? "inactive"}</b></p>
    <h2>Implementing code</h2><div class="rows">${edges.map(e =>
      `<a class="row" href="#/file/${e.src.split(":")[0]}"><span class="id">s${e.s}</span>
       <code class="grow">${esc(e.src)}</code><span class="num">${esc(e.note)}</span></a>`).join("")
      || empty("No connections yet — run the audit.")}</div>`;
}

function vData() {
  if (!state.reg) return needReg();
  const ds = state.reg.data_schema, tk = state.reg.tokens;
  const mods = (ds?.modalities ?? []).map(m => `<div class="row"><span class="grow"><b>${esc(m.name)}</b>
      <span style="color:var(--ink-2)"> ${esc(m.origin ?? "")}</span></span>
      <code class="num">${esc(JSON.stringify(m.shape ?? m.dims ?? ""))}</code></div>`).join("");
  const toks = (tk?.tokens ?? []).map(t => `<div class="row"><span class="grow"><b>${esc(t.token_type)}</b>
      <span style="color:var(--ink-2)"> ${esc(t.composed_of ?? "")}</span></span>
      <span class="num">${esc(String(t.count_per_example ?? ""))}</span>
      <code class="num">d=${esc(String(t.dim ?? "?"))}</code></div>`).join("");
  return `<h1>Data</h1>
    <p class="sub">Every observation the model trains on — its sources, its token structure, its raw form.
      Map view and live runs land next.</p>
    <h2>Modalities</h2><div class="rows">${mods || empty("registry lacks data schema")}</div>
    <h2>Context window — one training example</h2>
    <p class="sub">${esc(tk?.context_window?.formula_round0 ?? "")}</p>
    <div class="rows">${toks}</div>`;
}

/* ---- router ---- */
async function route() {
  const [_, p1, p2] = location.hash.split("/");
  document.querySelectorAll("nav a").forEach(a =>
    a.classList.toggle("active", a.hash === "#/" + (p1 || "status")));
  const r = { status: vStatus, code: vCode, science: () => vScience(p2), benchmarks: vBench, data: vData };
  view.innerHTML = p1 === "file" ? await vFile(decodeURIComponent(location.hash.slice(7)))
    : p1 === "rule" ? vRule(p2)
    : p1 === "bench" ? vBenchOne(p2)
    : (r[p1] ?? vStatus)();
  window.scrollTo(0, 0);
}
addEventListener("hashchange", route);
load();
