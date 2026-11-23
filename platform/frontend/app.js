/* GeoSpec Main Controller — Connects 5-Stage Workbench, 6-Guard Rail,
 * MapInspector, Overpass Client, and Pyodide WASM Worker.
 */
import { MapInspector } from "./components/map_inspector.js";
import { renderStage1 } from "./components/stage1_data_osm.js";
import { renderStage2 } from "./components/stage2_composition.js";
import { renderStage3 } from "./components/stage3_model_composer.js";
import { renderStage4, renderStage5 } from "./components/stage4_5_spatial_gwr.js";
import { fetchStreetGraphOSM, fetchTargetOSM } from "./gis/overpass_client.js";
import { DagClient } from "./state/dag_client.js";
import { createDefaultRecipe, inferColumnProfile } from "./state/recipe_store.js";

const state = {
  activeStage: 1,
  recipe: createDefaultRecipe(),
  fileName: "",
  headers: [],
  columns: null,
  nRows: 0,
  colProfile: {},
  datasetInfo: null,
  targetPreviews: {},     // targetId -> { preview_geojson, feature_index }
  rawGeojsonTargets: {},  // targetId -> GeoJSON
  stage1Res: null,
  stage2Res: null,
  stage3Res: null,
  stage4Res: null,
  stage5Res: null,
  activeSpecId: "spec_a",
  covType: "HC3",
  showPct: false,
  wasmStatus: "booting",
  wasmMessage: "WASM indítása...",
  logLine: "Várakozás saját CSV / GeoJSON adatbázis betöltésére.",
};

let debounceTimer = null;

function setLog(msg) {
  state.logLine = msg;
  const el = document.getElementById("consoleText");
  if (el) el.textContent = msg;
}

function updateBootIndicator(status, message) {
  state.wasmStatus = status;
  state.wasmMessage = message;
  const dot = document.getElementById("wasmDot");
  const lbl = document.getElementById("wasmStatusText");
  if (dot) {
    dot.className = "dot " + (status === "ready" ? "ready" : status === "error" ? "err" : "busy");
  }
  if (lbl) lbl.textContent = message;
}

const dag = new DagClient("./worker/pyodide_worker.js", updateBootIndicator, setLog);

const mapInspector = new MapInspector("mapContainer", "pointInspector", {
  onToggleOsmFeature: (targetId, uid) => actions.toggleOsmFeature(targetId, uid),
  onClickObservation: (obsIdx) => actions.inspectObservation(obsIdx),
});

const actions = {
  render() {
    // Stage tabs active/done states
    document.querySelectorAll(".stage-tab").forEach((btn) => {
      const s = Number(btn.getAttribute("data-stage"));
      btn.classList.toggle("active", s === state.activeStage);
      const isDone =
        (s === 1 && state.stage1Res?.ok) ||
        (s === 2 && state.stage2Res?.ok) ||
        (s === 3 && state.stage3Res?.ok) ||
        (s === 4 && state.stage4Res?.ok) ||
        (s === 5 && state.stage5Res?.ok);
      btn.classList.toggle("done", Boolean(isDone));
    });

    // Enable/disable Run All button
    const btnRunAll = document.getElementById("btnRunAll");
    if (btnRunAll) btnRunAll.disabled = !state.datasetInfo;

    const routeSelect = document.getElementById("selRouteTarget");
    if (routeSelect) {
      const tgts = state.recipe.spatial_targets || [];
      const prevVal = routeSelect.value;
      routeSelect.innerHTML = tgts.length
        ? tgts.map((t) => `<option value="${t.id}">${t.id}</option>`).join("")
        : `<option value="">— nincs célpont —</option>`;
      if (tgts.some((t) => t.id === prevVal)) routeSelect.value = prevVal;
    }

    const gwrVarSelect = document.getElementById("selGwrVarLayer");
    if (gwrVarSelect && state.stage5Res?.ok) {
      const cols = state.stage5Res.col_names || [];
      const cur = Number(gwrVarSelect.value) || 1;
      gwrVarSelect.innerHTML = cols.map((c, i) => `<option value="${i}" ${i === cur ? "selected" : ""}>β̂(${c})</option>`).join("");
      gwrVarSelect.style.display = document.getElementById("selMapLayer")?.value === "gwr_beta" ? "inline-block" : "none";
    } else if (gwrVarSelect) {
      gwrVarSelect.style.display = "none";
    }

    renderGuardStrip();

    const pane = document.getElementById("stageContent");
    const ctx = { state, actions };
    if (state.activeStage === 1) renderStage1(pane, ctx);
    else if (state.activeStage === 2) renderStage2(pane, ctx);
    else if (state.activeStage === 3) renderStage3(pane, ctx);
    else if (state.activeStage === 4) renderStage4(pane, ctx);
    else if (state.activeStage === 5) renderStage5(pane, ctx);
  },

  async onDatasetParsed(fileName, parsed, colProfile, guesses) {
    state.fileName = fileName;
    state.headers = parsed.headers;
    state.columns = parsed.columns;
    state.nRows = parsed.nRows;
    state.colProfile = colProfile;
    state.stage1Res = state.stage2Res = state.stage3Res = state.stage4Res = state.stage5Res = null;
    state.targetPreviews = {};
    mapInspector.clearTargets();

    const ds = state.recipe.dataset_schema;
    if (!ds.lat_col || !parsed.columns[ds.lat_col]) ds.lat_col = guesses.latGuess;
    if (!ds.lon_col || !parsed.columns[ds.lon_col]) ds.lon_col = guesses.lonGuess;
    if (!ds.row_id_col || !parsed.columns[ds.row_id_col]) ds.row_id_col = guesses.idGuess;
    if (!ds.y_spec.num_col || !parsed.columns[ds.y_spec.num_col]) ds.y_spec.num_col = guesses.yGuess;

    // Seed initial covariates for Spec A if empty
    const specA = state.recipe.models[0];
    if (specA && specA.covariates.length === 0) {
      const skip = new Set([ds.lat_col, ds.lon_col, ds.row_id_col, ds.y_spec.num_col, ds.y_spec.den_col].filter(Boolean));
      for (const h of parsed.headers) {
        if (skip.has(h)) continue;
        if (colProfile[h]?.isNumeric && specA.covariates.length < 3) {
          specA.covariates.push({ col: h, type: "numeric", transform: "raw" });
        }
      }
    }

    await this.syncDatasetToWorker();
  },

  async syncDatasetToWorker() {
    const ds = state.recipe.dataset_schema;
    if (!state.columns || !ds.lat_col || !ds.lon_col) {
      this.render();
      return;
    }
    try {
      setLog("Adatbázis betöltése és lokális metrikus vetület számítása (Krüger TM)...");
      const res = await dag.call("load_dataset", {
        columns: state.columns,
        lat_col: ds.lat_col,
        lon_col: ds.lon_col,
      });
      state.datasetInfo = res;
      mapInspector.setObservationsFromDataset(state.columns, ds.lat_col, ds.lon_col, ds.row_id_col, ds.y_spec, true);
      mapInspector.renderPointInspector(null, null);
      setLog(`Adatbázis kész: N=${res.n_rows} (geokódolt=${res.n_geocoded}, vetület=${res.projection.nearest_utm_epsg})`);
      this.render();
    } catch (err) {
      setLog(`Séma hiba: ${err.message}`);
      alert(`Séma hiba: ${err.message}`);
    }
  },

  async ingestSingleGeojsonTarget(targetId, geojsonObj) {
    if (!state.datasetInfo) return;
    const tgt = state.recipe.spatial_targets.find((t) => t.id === targetId);
    const prev = await dag.call("ingest_target_raw", {
      target_id: targetId,
      raw_payload: geojsonObj,
      source_kind: "geojson",
      excluded_osm_ids: tgt?.excluded_osm_ids || [],
    });
    state.targetPreviews[targetId] = prev;
    mapInspector.setTargetPreview(targetId, prev.preview_geojson);
  },

  async fetchAllTargetsOSM() {
    if (!state.datasetInfo) return alert("Előbb rögzítsd a dataset sémát!");
    const targets = state.recipe.spatial_targets;
    const osmTargets = targets.filter((t) => t.source === "osm_preset" || t.source === "osm_custom_tag");
    const needProfiles = new Set(
      targets.filter((t) => t.distance_modes === "network_dijkstra" || t.distance_modes === "both").map((t) => t.routing_profile || "walk")
    );

    try {
      for (const tgt of osmTargets) {
        setLog(`OSM Overpass lekérés: ${tgt.name}...`);
        const { data, fromCache } = await fetchTargetOSM(tgt, state.datasetInfo.bbox);
        const prev = await dag.call("ingest_target_raw", {
          target_id: tgt.id,
          raw_payload: data,
          source_kind: "osm",
          excluded_osm_ids: tgt.excluded_osm_ids || [],
        });
        state.targetPreviews[tgt.id] = prev;
        mapInspector.setTargetPreview(tgt.id, prev.preview_geojson);
        setLog(`OSM kész (${tgt.id}): ${prev.n_features} objektum (${fromCache ? "IndexedDB cache" : "Overpass élő"}).`);
      }

      for (const prof of needProfiles) {
        setLog(`Utcahálózat lekérése CSR Dijkstra útvonalhoz (${prof})...`);
        const { data, fromCache } = await fetchStreetGraphOSM(state.datasetInfo.bbox, 600, prof);
        const gRes = await dag.call("ingest_street_graph", { profile: prof, graph_json: data });
        setLog(`Utcahálózat (${prof}) kész: ${gRes.n_nodes} csúcs, ${gRes.n_edges} él (${fromCache ? "cache" : "Overpass"}).`);
      }
      this.render();
    } catch (err) {
      setLog(`OSM lekérési hiba: ${err.message}`);
      alert(`Overpass API hiba: ${err.message}`);
    }
  },

  toggleOsmFeature(targetId, uid) {
    const tgt = state.recipe.spatial_targets.find((t) => t.id === targetId);
    if (!tgt) return;
    tgt.excluded_osm_ids = tgt.excluded_osm_ids || [];
    const idx = tgt.excluded_osm_ids.indexOf(uid);
    if (idx >= 0) tgt.excluded_osm_ids.splice(idx, 1);
    else tgt.excluded_osm_ids.push(uid);

    const prev = state.targetPreviews[targetId];
    if (prev && prev.preview_geojson) {
      const exclSet = new Set(tgt.excluded_osm_ids);
      for (const f of prev.preview_geojson.features || []) {
        f.properties.excluded = exclSet.has(f.properties.uid);
      }
      mapInspector.setTargetPreview(targetId, prev.preview_geojson);
    }
    setLog(`OSM elem (${targetId}:${uid}) ${idx >= 0 ? "bekapcsolva" : "kizárva"} a távolságszámításból.`);
    this.render();
  },

  refreshMapTargets() {
    mapInspector.clearTargets();
    for (const t of state.recipe.spatial_targets) {
      if (state.targetPreviews[t.id]) {
        mapInspector.setTargetPreview(t.id, state.targetPreviews[t.id].preview_geojson);
      }
    }
  },

  async runStage1() {
    if (!state.datasetInfo) return;
    try {
      // Auto-fetch OSM if targets haven't been ingested yet
      const needsFetch = state.recipe.spatial_targets.some(
        (t) => (t.source === "osm_preset" || t.source === "osm_custom_tag") && !state.targetPreviews[t.id]
      );
      if (needsFetch) await this.fetchAllTargetsOSM();

      setLog("1. Fázis: Euklideszi és CSR Dijkstra távolságok számítása...");
      const res = await dag.call("run_stage_1", { recipe: state.recipe, run_id: dag.nextRunId() });
      state.stage1Res = res;
      setLog(`1. Fázis kész (${res.summary.length} célpont távolságmátrixai frissítve).`);
      this.render();
    } catch (err) {
      setLog(`Stage 1 hiba: ${err.message}`);
      alert(err.message);
    }
  },

  async runStage2() {
    if (!state.datasetInfo) return;
    try {
      setLog("2. Fázis: Leíró statisztika, Step 0..K & Gelbach (2016) dekompozíció...");
      const res = await dag.call("run_stage_2", { recipe: state.recipe, run_id: dag.nextRunId() });
      state.stage2Res = res;
      setLog("2. Fázis kész.");
      this.render();
    } catch (err) {
      setLog(`Stage 2 hiba: ${err.message}`);
      alert(err.message);
    }
  },

  async runStage3() {
    if (!state.datasetInfo) return;
    try {
      const t0 = performance.now();
      const res = await dag.call("run_stage_3", {
        recipe: state.recipe,
        cov_type: state.covType,
        run_id: dag.nextRunId(),
      });
      state.stage3Res = res;
      const dt = (performance.now() - t0).toFixed(1);
      setLog(`3. Fázis kész (${dt} ms, ${res.models.length} modell, ${state.covType} SE).`);
      mapInspector.restyleObservations({ stage3: state.stage3Res, stage4: state.stage4Res, stage5: state.stage5Res });
      this.render();
    } catch (err) {
      setLog(`Stage 3 hiba: ${err.message}`);
    }
  },

  runStage3Debounced() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => this.runStage3(), 90);
  },

  async generateM1M4Specs() {
    const active = state.recipe.models.find((m) => m.id === state.activeSpecId) || state.recipe.models[0];
    const covs = active ? active.covariates || [] : [];
    const tgts = state.recipe.spatial_targets || [];
    if (!tgts.length) return alert("Előbb vegyél fel legalább 1 térbeli célpontot az 1. fázisban!");
    const specs = await dag.call("generate_m1_m4", {
      covariates: covs,
      spatial_targets: tgts,
      default_rings: [150, 300, 500],
      default_cutoff_m: 300,
    });
    state.recipe.models = specs;
    state.activeSpecId = "m4_ad";
    state.recipe.spatial_engine.active_model_id = "m4_ad";
    state.recipe.spatial_engine.gwr.focal_model_id = "m4_ad";
    if (tgts.length) state.recipe.composition_chain.focal_term_ref = `${tgts[0].id}::andersson_day`;
    this.render();
    await this.runStage3();
  },

  async runStage4() {
    if (!state.datasetInfo) return;
    try {
      setLog("4. Fázis: Multi-W építés, Moran/LISA, SAR/SEM/SDM & 1000× Krinsky–Robb...");
      const res = await dag.call("run_stage_4", { recipe: state.recipe, run_id: dag.nextRunId() });
      state.stage4Res = res;
      setLog(`4. Fázis kész (${res.anselin?.verdict || ""}).`);
      mapInspector.restyleObservations({ stage3: state.stage3Res, stage4: state.stage4Res, stage5: state.stage5Res });
      this.render();
    } catch (err) {
      setLog(`Stage 4 hiba: ${err.message}`);
      alert(err.message);
    }
  },

  async runStage5() {
    if (!state.datasetInfo) return;
    try {
      setLog("5. Fázis: Szingularitás-védett Adaptív Bisquare GWR optimalizálása...");
      const res = await dag.call("run_stage_5", { recipe: state.recipe, run_id: dag.nextRunId() });
      state.stage5Res = res;
      if (res.ok) {
        setLog(`5. Fázis kész (b*=${res.optimal_bw}, GWR Adj. R²=${res.gwr_adj_r2}).`);
        mapInspector.restyleObservations({ stage3: state.stage3Res, stage4: state.stage4Res, stage5: state.stage5Res });
      } else {
        setLog(`GWR figyelmeztetés: ${res.reason}`);
      }
      this.render();
    } catch (err) {
      setLog(`Stage 5 hiba: ${err.message}`);
      alert(err.message);
    }
  },

  async runFullPipeline() {
    await this.runStage1();
    await this.runStage2();
    await this.runStage3();
    await this.runStage4();
    await this.runStage5();
    setLog("Teljes 5-fázisú térökonometriai pipeline sikeresen lefutott.");
  },

  async inspectObservation(obsIdx) {
    const tgtId = document.getElementById("selRouteTarget")?.value || state.recipe.spatial_targets?.[0]?.id;
    let route = null;
    if (tgtId) {
      try {
        route = await dag.call("get_single_route", { obs_idx: obsIdx, target_id: tgtId, recipe: state.recipe });
        mapInspector.showRoute(route);
      } catch {
        route = null;
      }
    }

    const feat = mapInspector.obsFeatures.find((f) => f.properties.obs_idx === obsIdx);
    let resid = null;
    if (state.stage3Res?.models?.length) {
      const m = state.stage3Res.models.find((x) => x.resid) || state.stage3Res.models[0];
      const pos = m?.valid_indices?.indexOf(obsIdx) ?? -1;
      if (pos >= 0) resid = m.resid[pos];
    }
    let lisa = null;
    if (state.stage4Res?.ok) {
      const pos = state.stage4Res.valid_indices.indexOf(obsIdx);
      if (pos >= 0) lisa = state.stage4Res.lisa_resid.quadrants[pos];
    }
    let gwr_beta = null, gwr_t = null;
    if (state.stage5Res?.ok) {
      const pos = state.stage5Res.valid_indices.indexOf(obsIdx);
      const ci = Math.min(Number(document.getElementById("selGwrVarLayer")?.value) || 1, state.stage5Res.col_names.length - 1);
      if (pos >= 0) {
        gwr_beta = state.stage5Res.betas_native[pos][ci];
        gwr_t = state.stage5Res.t_vals[pos][ci];
      }
    }

    mapInspector.restyleObservations({ stage3: state.stage3Res, stage4: state.stage4Res, stage5: state.stage5Res });
    mapInspector.renderPointInspector(obsIdx, {
      row_id: feat?.properties?.row_id ?? obsIdx,
      y_raw: feat?.properties?.y_raw ?? null,
      route,
      resid,
      lisa,
      gwr_beta,
      gwr_t,
    });
  },

  async copyLatexTable() {
    try {
      const bundle = await dag.call("export_bundle", { recipe: state.recipe });
      const tex = state.showPct ? bundle.latex_pct : bundle.latex_main;
      if (!tex) return alert("Előbb futtasd le a 3. fázist!");
      await navigator.clipboard.writeText(tex);
      setLog("Overleaf booktabs LaTeX táblázat a vágólapra másolva!");
    } catch (err) {
      alert(err.message);
    }
  },

  async exportBundle(kind = "zip") {
    if (!state.datasetInfo) return;
    try {
      setLog("Publikációs export csomag összeállítása...");
      const bundle = await dag.call("export_bundle", { recipe: state.recipe });
      if (kind === "geojson") {
        this.downloadText("geospec_enriched.geojson", JSON.stringify(bundle.geojson, null, 2), "application/geo+json");
      } else {
        const bin = Uint8Array.from(atob(bundle.zip_base64), (c) => c.charCodeAt(0));
        const blob = new Blob([bin], { type: "application/zip" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `geospec_publication_bundle_${Date.now()}.zip`;
        a.click();
        URL.revokeObjectURL(url);
      }
      setLog(`Export kész (${kind}).`);
    } catch (err) {
      alert(err.message);
    }
  },

  exportRecipe() {
    this.downloadText("experiment_recipe.json", JSON.stringify(state.recipe, null, 2), "application/json");
    setLog("experiment_recipe.json mentve (nyers adatok nélkül).");
  },

  importRecipe(loadedRecipe) {
    state.recipe = loadedRecipe;
    state.activeSpecId = loadedRecipe.models?.[0]?.id || "spec_a";
    setLog(`Recept betöltve: ${loadedRecipe.recipe_meta?.title || "experiment_recipe.json"}`);
    if (state.columns) {
      const { profile, guesses } = inferColumnProfile(state.headers, state.columns);
      state.colProfile = profile;
      const ds = state.recipe.dataset_schema;
      if (!state.columns[ds.lat_col]) ds.lat_col = guesses.latGuess;
      if (!state.columns[ds.lon_col]) ds.lon_col = guesses.lonGuess;
      this.syncDatasetToWorker();
    } else {
      this.render();
    }
  },

  downloadText(filename, content, mime = "text/plain") {
    const blob = new Blob([content], { type: mime + ";charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  },
};

function renderGuardStrip() {
  const s1 = state.stage1Res;
  const s2 = state.stage2Res;
  const s3 = state.stage3Res;
  const s4 = state.stage4Res;
  const s5 = state.stage5Res;

  const g1Warn = s3?.models?.some((m) => m.diagnostics?.status === "warn");
  const g2Count = s3?.models?.reduce((acc, m) => acc + (m.guard2_events?.length || 0), 0) || 0;
  const g3Atten = s2?.composition?.ok ? s2.composition.final_attenuation_pct : null;
  const g4Heck = s3?.heckman;
  const g5Warn = s4?.guard5_alerts?.length > 0;
  const g6Reg = s5?.ok ? s5.n_regularized : null;

  const setPill = (id, cls, text) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.className = `guard-pill ${cls}`;
    el.textContent = text;
  };

  setPill("g1Pill", s3 ? (g1Warn ? "warn" : "pass") : "", s3 ? (g1Warn ? "G1 ⚠ Konfound/VIF" : "G1 ✓ VIF/RESET") : "G1 Diagnosztika");
  setPill("g2Pill", s3 ? (g2Count > 0 ? "warn" : "pass") : "", s3 ? (g2Count > 0 ? `G2 ⚡ ${g2Count} összevonás` : "G2 ✓ Sűrűség ≥5") : "G2 Ritka cellák");
  setPill("g3Pill", g3Atten !== null ? (Math.abs(g3Atten) > 50 ? "warn" : "pass") : "", g3Atten !== null ? `G3 ${g3Atten}% atten.` : "G3 Gelbach/Step");
  setPill("g4Pill", g4Heck ? (g4Heck.activated && g4Heck.selection_bias_detected ? "warn" : "pass") : "", g4Heck?.activated ? `G4 λ p=${g4Heck.imr_p}` : "G4 Heckman/SE");
  setPill("g5Pill", s4?.ok ? (g5Warn ? "warn" : "pass") : "", s4?.ok ? (g5Warn ? "G5 ⚠ Co-loc W" : `G5 ✓ ${s4.anselin.recommended_model}`) : "G5 Térbeli W");
  setPill("g6Pill", s5?.ok ? (g6Reg > 0 ? "warn" : "pass") : "", s5?.ok ? (g6Reg > 0 ? `G6 🛡️ Ridge (${g6Reg})` : "G6 ✓ GWR Stabil") : "G6 GWR Ridge");
}

window.addEventListener("DOMContentLoaded", () => {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("./sw.js").catch(() => {});
  }

  mapInspector.init();
  mapInspector.renderPointInspector(null, null);
  dag.init().catch((err) => updateBootIndicator("error", err.message));

  document.querySelectorAll(".stage-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.activeStage = Number(btn.getAttribute("data-stage"));
      actions.render();
    });
  });

  document.getElementById("btnRunAll").addEventListener("click", () => actions.runFullPipeline());
  document.getElementById("btnTopRecipeExport").addEventListener("click", () => actions.exportRecipe());
  document.getElementById("btnTopZipExport").addEventListener("click", () => actions.exportBundle("zip"));
  document.getElementById("btnThemeToggle").addEventListener("click", () => {
    document.body.classList.toggle("dark-theme");
  });

  document.getElementById("selMapLayer").addEventListener("change", (e) => {
    mapInspector.restyleObservations({
      stage3: state.stage3Res,
      stage4: state.stage4Res,
      stage5: state.stage5Res,
      layerMode: e.target.value,
    });
    actions.render();
  });

  document.getElementById("selGwrVarLayer").addEventListener("change", (e) => {
    mapInspector.restyleObservations({
      stage3: state.stage3Res,
      stage4: state.stage4Res,
      stage5: state.stage5Res,
      gwrColIdx: Number(e.target.value),
    });
  });

  // Guard pill quick jumps
  document.getElementById("g1Pill").addEventListener("click", () => { state.activeStage = 3; actions.render(); });
  document.getElementById("g2Pill").addEventListener("click", () => { state.activeStage = 3; actions.render(); });
  document.getElementById("g3Pill").addEventListener("click", () => { state.activeStage = 2; actions.render(); });
  document.getElementById("g4Pill").addEventListener("click", () => { state.activeStage = 3; actions.render(); });
  document.getElementById("g5Pill").addEventListener("click", () => { state.activeStage = 4; actions.render(); });
  document.getElementById("g6Pill").addEventListener("click", () => { state.activeStage = 5; actions.render(); });

  actions.render();
});
