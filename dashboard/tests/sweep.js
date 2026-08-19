/* Full behavioral sweep: every route, key interactions, console/page/network errors.
   Run: npm i puppeteer-core && node dashboard/tests/sweep.js  (server on :8321, Chrome installed)
   Prints ALL CLEAN or one line per problem. */
const puppeteer = require("puppeteer-core");
const BASE = "http://localhost:8321";

(async () => {
  const browser = await puppeteer.launch({ executablePath: "/usr/bin/google-chrome",
    headless: "new" });
  const page = await browser.newPage();
  await page.setViewport({ width: 1500, height: 1000 });
  const problems = [];
  page.on("pageerror", e => problems.push(`PAGEERR ${page.url()} :: ${String(e).slice(0, 160)}`));
  page.on("requestfailed", r => {
    if (!r.url().includes("gbif") && !r.url().includes("eox") && !r.url().includes("arcgis"))
      problems.push(`REQFAIL ${r.url().slice(0, 100)} ${r.failure()?.errorText}`);
  });
  page.on("response", r => {
    if (r.status() >= 400 && r.url().startsWith(BASE) && !r.url().includes("favicon"))
      problems.push(`HTTP${r.status()} ${r.url().slice(BASE.length)}`);
  });
  const go = async (hash, waitSel) => {
    await page.goto(BASE + "/#" + hash, { waitUntil: "networkidle2", timeout: 30000 });
    if (waitSel) await page.waitForSelector(waitSel, { timeout: 8000 })
      .catch(() => problems.push(`MISSING ${hash} :: ${waitSel}`));
    return page;
  };
  const check = async (name, fn) => {
    try {
      const ok = await fn();
      if (ok !== true) problems.push(`FAIL ${name} :: ${JSON.stringify(ok).slice(0, 140)}`);
    } catch (e) { problems.push(`THROW ${name} :: ${String(e).slice(0, 140)}`); }
  };

  await go("/status", ".tile");
  await check("status: 5 system tiles + 32 rules", () => page.evaluate(() =>
    document.querySelectorAll(".tile.sys").length === 5 &&
    document.querySelectorAll(".tile:not(.sys)").length >= 32 || {
      sys: document.querySelectorAll(".tile.sys").length,
      rules: document.querySelectorAll(".tile:not(.sys)").length }));
  await check("status: findings section", () => page.evaluate(() =>
    document.body.textContent.includes("Operational findings") || "no findings section"));
  await check("status: island triage chips", () => page.evaluate(() => {
    const t = document.body.textContent;
    return /→ (wire-in|delete|keep)/.test(t) || "no triage chips";   // dispositions present (mix varies as cleanup proceeds)
  }));

  await go("/graph", ".gn");
  await check("graph: pin R18 lights both spans", async () => await page.evaluate(() => {
    document.querySelector('.gn[data-key="R18"]').dispatchEvent(new MouseEvent("click", { bubbles: true }));
    const ge = document.querySelectorAll(".ge.on").length, gc = document.querySelectorAll(".gc.on").length;
    return (ge > 0 && gc > 0) || { fileEdges: ge, benchEdges: gc };
  }));
  await check("graph: release on background", async () => await page.evaluate(() => {
    document.querySelector("#graphbox svg").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    return !document.querySelector("svg.dimmed") || "still dimmed";
  }));

  await go("/flow", ".fband");
  await check("flow: bands + real values", () => page.evaluate(() => {
    const bands = document.querySelectorAll(".fband").length;
    const hasVals = /μ=/.test(document.body.textContent);
    const hasPosterior = document.body.textContent.includes("Posterior");
    return (bands > 10 && hasVals && hasPosterior) || { bands, hasVals, hasPosterior };
  }));
  await check("flow: repo boxes link, torch boxes don't", () => page.evaluate(() => {
    const a = document.querySelector("a.fsub");
    const linked = /#\/file\/.+:\d+-\d+$/.test(a?.getAttribute("href") ?? "");
    const noNulls = ![...document.querySelectorAll("a")].some(x => (x.getAttribute("href") ?? "").includes("null"));
    return (linked && noNulls) || { linked, noNulls };
  }));

  await go("/code", ".row");
  await check("code: rows have status dots", () => page.evaluate(() =>
    document.querySelectorAll(".row .dot").length > 30 || document.querySelectorAll(".row .dot").length));

  await go("/file/core/fusion.py:920-947", ".codeplate");
  await check("file: anchored scroll + highlight", async () => {
    await new Promise(r => setTimeout(r, 400));
    return await page.evaluate(() => {
      const hl = document.querySelectorAll(".cl.hl").length;
      const el = document.getElementById("L920");
      const r = el.getBoundingClientRect();
      const visible = r.top > 0 && r.top < window.innerHeight;
      return (hl === 28 && visible) || { hl, top: Math.round(r.top), visible };
    });
  });
  await check("file: syntax highlighted", () => page.evaluate(() =>
    document.querySelectorAll(".codeplate .hljs-keyword").length > 50 ||
    document.querySelectorAll(".codeplate .hljs-keyword").length));

  await go("/science", ".row");
  await go("/rule/24", ".erow");
  await check("rule24: expander shows real code", async () => {
    await page.evaluate(() => {
      [...document.querySelectorAll(".erow .expand")]
        .find(e => e.textContent.includes("fusion.py"))?.click();
    });
    await new Promise(r => setTimeout(r, 600));
    return await page.evaluate(() => {
      const box = [...document.querySelectorAll(".snipbox")].find(b => !b.hidden);
      return (box && box.textContent.includes("def ")) || "no visible snippet with code";
    });
  });
  await check("rule24: reach chips present", () => page.evaluate(() =>
    /● live|✕ never called|◐ gated/.test(document.body.textContent) || "no reach chips"));

  await go("/benchmarks", ".row");
  await go("/bench/B1", ".erow");
  await check("B1: evaluator row + sections", () => page.evaluate(() => {
    const t = document.body.textContent;
    return (t.includes("frozen evaluator") && t.includes("IMPLEMENTATION") !== null &&
            document.querySelectorAll(".esec").length >= 2) ||
      { esec: document.querySelectorAll(".esec").length };
  }));

  await go("/data/5237817917", "#obspanel .latin");
  await check("data: obs panel full record", async () => {
    await new Promise(r => setTimeout(r, 1500));
    return await page.evaluate(() => {
      const t = document.querySelector("#obspanel").textContent;
      const ok = ["Diplacus", "Model reconstruction", "Daymet", "SSURGO", "TWI", "rumple"]
        .filter(k => !t.includes(k));
      const imgs = document.querySelectorAll(".imgpair img").length;
      return (ok.length === 0 && imgs === 2) || { missing: ok, imgs };
    });
  });
  await check("data: census strays flagged", () => page.evaluate(() =>
    document.body.textContent.includes("stray — loaded by nothing") || "no stray flags"));

  await go("/runs", ".row");
  await check("runs: 3 runs listed", () => page.evaluate(() =>
    document.querySelectorAll(".row").length >= 3 || document.querySelectorAll(".row").length));
  const runId = await page.evaluate(() =>                 // a finished run (has H score), not a live one
    [...document.querySelectorAll(".row")].find(r => /H 0\./.test(r.textContent))
      ?.querySelector("b")?.textContent);
  await go("/run/" + runId, "svg");
  await check("run: loss chart + benchmark deltas", () => page.evaluate(() => {
    const pl = document.querySelectorAll("svg polyline").length;
    const deltas = document.body.textContent.match(/[+-]0\.\d{3}/g)?.length ?? 0;
    return (pl >= 1 && deltas > 20) || { polylines: pl, deltas };
  }));

  console.log(problems.length ? problems.join("\n") : "ALL CLEAN");
  await browser.close();
})();
