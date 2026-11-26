import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

import { createDefaultRecipe } from "../frontend/state/recipe_store.js";
import { sanitizeRecipe } from "../frontend/utils/sanitize.js";

const TESTS_DIR = path.dirname(fileURLToPath(import.meta.url));
const PLATFORM_DIR = path.resolve(TESTS_DIR, "..");

function walkJsFiles(dir, out = []) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) walkJsFiles(full, out);
    else if (ent.isFile() && ent.name.endsWith(".js")) out.push(full);
  }
  return out;
}

test("sw.js PWA shell manifest, cache policy & asset existence on disk", () => {
  const swSrc = fs.readFileSync(path.join(PLATFORM_DIR, "sw.js"), "utf8");
  assert.match(swSrc, /const CACHE_NAME = "geospec-wasm-v10"/);
  assert.match(swSrc, /event\.waitUntil\([\s\S]*?self\.skipWaiting\(\)/);
  assert.match(swSrc, /resp\.type\s*!==\s*"opaque"/);
  assert.match(swSrc, /ALLOWED_CDN_PREFIXES/);

  const match = swSrc.match(/const CORE_SHELL_ASSETS = \[([\s\S]*?)\];/);
  assert.ok(match, "CORE_SHELL_ASSETS array must exist in sw.js");
  const relAssets = [...match[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
  assert.ok(relAssets.length >= 20, "Expected >= 20 CORE_SHELL_ASSETS entries");

  for (const rel of relAssets) {
    const cleanRel = rel === "./" ? "index.html" : rel.replace(/^\.\//, "");
    const abs = path.join(PLATFORM_DIR, cleanRel);
    assert.ok(fs.existsSync(abs), `Missing sw.js CORE_SHELL_ASSETS file: ${cleanRel}`);
  }
});

test("index.html CSP, SRI integrity, Guard buttons, UX elements & 0 alert() in frontend", () => {
  const html = fs.readFileSync(path.join(PLATFORM_DIR, "index.html"), "utf8");

  const cspMatch = html.match(/http-equiv="Content-Security-Policy"\s+content="([^"]+)"/i);
  assert.ok(cspMatch, "CSP meta tag missing");
  const csp = cspMatch[1];
  assert.match(csp, /script-src[^;]*blob:/);
  assert.match(csp, /worker-src[^;]*blob:/);

  assert.match(html, /maplibre-gl@4\.7\.1\/dist\/maplibre-gl\.css[^>]*integrity="sha384-/);
  assert.match(html, /maplibre-gl@4\.7\.1\/dist\/maplibre-gl\.js[^>]*integrity="sha384-/);
  assert.match(html, /plotly-2\.35\.2\.min\.js[^>]*integrity="sha384-/);

  for (const id of ["g1Pill", "g2Pill", "g3Pill", "g4Pill", "g5Pill", "g6Pill"]) {
    assert.match(html, new RegExp(`<button[^>]*id="${id}"`), `Guard pill #${id} must be an interactive <button>`);
  }
  for (const id of ["toastStack", "workerCrashBanner", "advancedDrawer", "btnLangToggle", "btnTopSaveMenu"]) {
    assert.ok(html.includes(`id="${id}"`), `Required UX element #${id} missing from index.html`);
  }

  const jsFiles = walkJsFiles(path.join(PLATFORM_DIR, "frontend"));
  assert.ok(jsFiles.length >= 10);
  for (const f of jsFiles) {
    const src = fs.readFileSync(f, "utf8");
    assert.doesNotMatch(src, /\balert\(/, `Blocking alert() found in ${path.relative(PLATFORM_DIR, f)}`);
  }
});

test("pyodide_worker.js SRI SHA-384 fetch+Blob, error recovery & Round-4 RPC dispatch", () => {
  const wSrc = fs.readFileSync(path.join(PLATFORM_DIR, "worker", "pyodide_worker.js"), "utf8");
  assert.match(wSrc, /PYODIDE_JS_SRI_SHA384/);
  assert.match(wSrc, /crypto\.subtle\.digest\("SHA-384"/);
  assert.match(wSrc, /URL\.createObjectURL\(new Blob\(/);
  assert.match(wSrc, /initPromise = null/);
  assert.match(wSrc, /def _json_safe\(/);
  for (const rpc of ["remove_target", "reset_session", "export_latex", "export_geojson", "run_robustness_matrix"]) {
    assert.ok(wSrc.includes(`"${rpc}"`), `Worker RPC handler missing method '${rpc}'`);
  }
});

test("DagClient mock-Worker runtime: init, progress, stale CancelledError & restart recovery", async () => {
  const spawned = [];
  class MockWorker {
    constructor(url, opts) {
      this.url = String(url);
      this.opts = opts;
      this.terminated = false;
      this.messages = [];
      this.onmessage = null;
      this.onerror = null;
      spawned.push(this);
    }
    postMessage(msg) {
      this.messages.push(msg);
      if (msg.type === "init") {
        queueMicrotask(() => {
          this.onmessage?.({ data: { type: "boot_status", status: "ready", message: "Mock WASM ready" } });
          this.onmessage?.({ data: { id: msg.id, ok: true } });
        });
      }
    }
    terminate() {
      this.terminated = true;
    }
    emit(data) {
      this.onmessage?.({ data });
    }
  }

  const prevWorker = globalThis.Worker;
  globalThis.Worker = MockWorker;
  try {
    const modUrl = pathToFileURL(path.join(PLATFORM_DIR, "frontend", "state", "dag_client.js")).href;
    const { DagClient } = await import(`${modUrl}?t=${Date.now()}`);
    const statuses = [];
    const progressEvents = [];
    const client = new DagClient(
      "./worker/pyodide_worker.js",
      (s, m) => statuses.push({ s, m }),
      () => {},
      (ev) => progressEvents.push(ev),
    );
    await client.init();
    assert.equal(client.readyState, "ready");
    assert.equal(statuses.at(-1).s, "ready");

    // Emit progress event
    spawned[0].emit({ type: "progress", pct: 0.42, message: "MGWR step 2" });
    assert.equal(progressEvents.length, 1);
    assert.equal(progressEvents[0].pct, 0.42);

    // Out-of-order completion: run_id 1 must reject with CancelledError once run_id 2 was issued for same kind
    const p1 = client.call("run_stage_3", { run_id: 1 });
    const p2 = client.call("run_stage_3", { run_id: 2 });
    const m1 = spawned[0].messages.find((m) => m.payload?.run_id === 1);
    const m2 = spawned[0].messages.find((m) => m.payload?.run_id === 2);
    spawned[0].emit({ id: m1.id, ok: true, result: { run: 1 } });
    spawned[0].emit({ id: m2.id, ok: true, result: { run: 2 } });

    await assert.rejects(p1, (err) => err?.name === "CancelledError");
    assert.deepEqual(await p2, { run: 2 });

    // restart() terminates old worker, cancels pending calls with CancelledError, and recovers
    const inFlight = client.call("run_stage_4", { run_id: 3 });
    await client.restart();
    assert.equal(spawned[0].terminated, true);
    assert.equal(spawned.length, 2);
    await assert.rejects(inFlight, (err) => err?.name === "CancelledError");

    const pFresh = client.call("run_stage_4", { run_id: 4 });
    const mFresh = spawned[1].messages.find((m) => m.payload?.run_id === 4);
    spawned[1].emit({ id: mFresh.id, ok: true, result: { ok: true, recovered: true } });
    assert.deepEqual(await pFresh, { ok: true, recovered: true });
  } finally {
    if (prevWorker === undefined) delete globalThis.Worker;
    else globalThis.Worker = prevWorker;
  }
});

test("Cross-layer JS->Python contract: sanitizeRecipe(createDefaultRecipe()) executes GeoSpecSession E2E", () => {
  const recipe = sanitizeRecipe(createDefaultRecipe(), createDefaultRecipe());
  recipe.dataset_schema.row_id_col = "id";
  recipe.dataset_schema.lat_col = "lat";
  recipe.dataset_schema.lon_col = "lon";
  recipe.dataset_schema.y_spec = { mode: "column", num_col: "price", transform: "log", boxcox_lambda: null, log1p_shift: 1.0 };
  recipe.dataset_schema.cluster_spec = { mode: "column", col: "clust", radius_m: 150.0 };
  recipe.spatial_targets = [
    { id: "hub", name: "Hub", source: "osm", geometry_handling: "point", distance_modes: "both", route_profile: "walk", max_snap_distance_m: 2000.0 },
  ];
  recipe.models = [
    { id: "m1", name: "M1", active: true, covariates: [{ col: "area", type: "continuous", transform: "raw" }], spatial_terms: [], interactions: [] },
    { id: "m2", name: "M2", active: true, covariates: [{ col: "area", type: "continuous", transform: "raw" }], spatial_terms: [{ target_id: "hub", transform: "ln_d", metric: "eucl" }], interactions: [] },
  ];
  recipe.spatial_engine.active_model_id = "m2";
  recipe.spatial_engine.weights = [{ id: "knn5", type: "knn", k: 5, dedup_colocated: true }];
  recipe.spatial_engine.gwr = { mode: "gwr", focal_model_id: "m2", b_min_floor: 12, kappa_thresh: 500.0, mc_iters: 19 };

  const pyScript = `
import json, math, random, sys
from py_engine.engine import GeoSpecSession
recipe = json.loads(sys.argv[1])
rng = random.Random(123)
n = 60
lats = [47.500 + (i - 30) * 0.00012 + rng.uniform(-0.00004, 0.00004) for i in range(n)]
lons = [19.050 + (i % 10 - 5) * 0.00025 + rng.uniform(-0.00005, 0.00005) for i in range(n)]
areas = [40.0 + (i % 12) * 3.5 + rng.uniform(-1.5, 1.5) for i in range(n)]
prices = [math.exp(12.2 + 0.015 * a - 0.0004 * (abs(lt - 47.5) * 111000.0) + rng.gauss(0.0, 0.12)) for lt, a in zip(lats, areas)]
cols = {
    "id": list(range(n)),
    "lat": lats,
    "lon": lons,
    "price": prices,
    "area": areas,
    "clust": [f"G{i % 6}" for i in range(n)],
}
s = GeoSpecSession()
s.load_dataset(cols, "lat", "lon")
s.ingest_target_raw("hub", {"type": "FeatureCollection", "features": [{"type": "Feature", "id": "h1", "geometry": {"type": "Point", "coordinates": [19.050, 47.500]}}]}, "geojson")
s.ingest_street_graph("walk", {
    "elements": [{"type": "node", "id": i + 1, "lat": lats[i], "lon": lons[i]} for i in range(n)]
    + [{"type": "way", "id": 900, "nodes": list(range(1, n + 1)), "tags": {"highway": "residential"}}]
})
r1 = s.run_stage_1_distances(recipe, run_id=1)
r2 = s.run_stage_2_descriptives_and_composition(recipe, run_id=1)
r3 = s.run_stage_3_ols_workbench(recipe, run_id=1)
r4 = s.run_stage_4_spatial_econometrics(recipe, run_id=1)
r5 = s.run_stage_5_gwr(recipe, run_id=1)
rob = s.run_robustness_matrix(recipe, run_id=1)
print(json.dumps({"ok": bool(r1["ok"] and r2["ok"] and r3["ok"] and r4["ok"] and r5["ok"] and rob["ok"]), "rob_rows": len(rob["rows"])}))
`;

  const res = spawnSync("python", ["-c", pyScript, JSON.stringify(recipe)], {
    env: { ...process.env, PYTHONPATH: PLATFORM_DIR },
    encoding: "utf8",
  });
  assert.equal(res.status, 0, `Python stderr:\n${res.stderr}`);
  const parsed = JSON.parse(res.stdout.trim().split(/\r?\n/).at(-1));
  assert.equal(parsed.ok, true);
  assert.equal(parsed.rob_rows, 6);
});

test("Visual QA regressions: hidden banner, 4D est, i18n verdicts, toast/mobile CSS & Escape", async () => {
  const css = fs.readFileSync(path.join(PLATFORM_DIR, "frontend", "styles", "theme.css"), "utf8");
  const html = fs.readFileSync(path.join(PLATFORM_DIR, "index.html"), "utf8");
  const s4Src = fs.readFileSync(path.join(PLATFORM_DIR, "frontend", "components", "stage4_5_spatial_gwr.js"), "utf8");
  const appSrc = fs.readFileSync(path.join(PLATFORM_DIR, "frontend", "app.js"), "utf8");

  // 1. Crash banner [hidden] & global [hidden] must win over display:flex; toast-stack must not occupy top-right header
  assert.match(css, /\.worker-crash-banner\[hidden\][\s\S]*display:\s*none\s*!important/);
  assert.match(css, /\[hidden\]\s*\{\s*display:\s*none\s*!important;/);
  assert.doesNotMatch(css, /\.toast-stack\s*\{[^}]*top:\s*14px;\s*right:\s*14px;/);
  assert.match(css, /\.stage-rail\s*\{[^}]*overflow-x:\s*auto/);
  assert.match(css, /\.legend-ticks\s*\{[^}]*gap:\s*10px/);
  assert.match(css, /@media\s*\(max-width:\s*720px\)[\s\S]*\.console-strip\s*\{/);

  // 2. Dynamic wasmStatusText must not carry data-i18n="worker.booting"
  assert.doesNotMatch(html, /id="wasmStatusText"[^>]*data-i18n="worker\.booting"/);

  // 3. Stage 4D uses p.est (not broken p.beta) and 4C checks resid_moran_method
  assert.match(s4Src, /p\.est\s*\?\?\s*p\.beta/);
  assert.match(s4Src, /resid_moran_method/);

  // 4. Stage 3C post_selection_note & Guard #6 warnings translate completely in HU without mixed English phrases
  const i18nUrl = pathToFileURL(path.join(PLATFORM_DIR, "frontend", "utils", "i18n.js")).href;
  const { setLang, translateVerdict } = await import(`${i18nUrl}?qa=${Date.now()}`);
  setLang("hu");
  const noteHu = translateVerdict(
    "Profile-MLE cutoff c* maximizes likelihood over the grid; standard errors at c* treat c as fixed (post-selection inference caveat applies; inspect the 95% profile-likelihood CI)."
  );
  assert.doesNotMatch(noteHu, /over the grid|treat c as fixed|post-selection inference caveat/i);

  const g6Hu = translateVerdict("Guard #6 pre-screened 3 low-prevalence dummies before GWR: d1 (min cell 5 < 20)");
  assert.doesNotMatch(g6Hu, /pre-screened|low-prevalence|min cell/i);
  assert.match(g6Hu, /min\. cella 5 < 20/);

  setLang("en");
  assert.equal(translateVerdict("Starting Pyodide..."), "Starting Pyodide...");
  setLang("hu");
  assert.equal(translateVerdict("Starting Pyodide..."), "Pyodide indítása...");

  // 5. Escape handler & updateBootIndicator called on language switch + inspector refresh after Stages 3-5
  assert.match(appSrc, /e\.key\s*===\s*"Escape"/);
  assert.match(appSrc, /btnLangToggle[\s\S]*?updateBootIndicator/);
  assert.match(appSrc, /refreshSelectedInspector\(\)/);
});
