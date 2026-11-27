/* GeoSpec Main Controller — 5-Stage Workbench, 6-Guard Slide-Over Drawer, Non-Blocking Toasts,
 * HU|EN i18n, Crash Recovery, Plot<->Map Sync & Pyodide WASM Worker Integration
 */
import { MapInspector } from "./components/map_inspector.js";
import { renderStage1 } from "./components/stage1_data_osm.js";
import { renderStage2 } from "./components/stage2_composition.js";
import { renderStage3, patchStage3ResultsOnly } from "./components/stage3_model_composer.js";
import { renderStage4, renderStage5 } from "./components/stage4_5_spatial_gwr.js";
import { renderStage6 } from "./components/stage6_ml_shap.js";
import { buildOsmBundle, pruneExpiredOSM } from "./gis/osm_idb_cache.js";
import { fetchStreetGraphOSM, fetchTargetOSM, padBbox } from "./gis/overpass_client.js";
import { DagClient } from "./state/dag_client.js";
import {
  buildWorkspaceBundle,
  createDefaultRecipe,
  inferColumnProfile,
  inferGeojsonTargetDefaults,
  makeUniqueTargetId,
  readTextWithEncodingFallback,
  slugifyTitle,
} from "./state/recipe_store.js";
import { applyI18nDOM, getLang, initLang, setLang, t, translateVerdict } from "./utils/i18n.js";
import { escapeHtml, sanitizeRecipe } from "./utils/sanitize.js";

const state = {
  activeStage: 1,
  recipe: createDefaultRecipe(),
  fileName: "",
  recipeFileName: "",
  workspaceFileName: "",
  osmBundleFileName: "",
  headers: [],
  columns: null,
  nRows: 0,
  colProfile: {},
  datasetInfo: null,
  targetPreviews: {},
  rawOsmTargets: {},
  rawGeojsonTargets: {},
  rawStreetGraphs: {},
  osmFilterQuery: {},
  osmFetching: false,
  stage1Res: null,
  stage2Res: null,
  stage3Res: null,
  stage4Res: null,
  stage5Res: null,
  stage6Res: null,
  robustnessRes: null,
  staleStages: new Set(),
  stage5Busy: false,
  stage5ProgressPct: 0,
  stage5ProgressText: "",
  activeSpecId: "spec_a",
  covType: "HC3",
  showPct: false,
  stage3AdKey: "",
  stage3DiagModelId: "",
  stage4ImpactKind: "sar",
  stage4ParamKind: "sem",
  pipelineRunning: false,
  wasmStatus: "booting",
  wasmRawMessage: "Starting Pyodide...",
  wasmMessage: "Pyodide indítása...",
  lastInspectedRoute: null,
  logLine: "Várakozás saját CSV / GeoJSON adatbázis, recept vagy .geospec.json projekt betöltésére.",
};

let debounceTimer = null;
let osmFetchAbortCtrl = null;
let datasetSeq = 0;
let s1Seq = 0;
let s2Seq = 0;
let s3Seq = 0;
let s4Seq = 0;
let s5Seq = 0;
let s6Seq = 0;

function showToast(message, level = "info", durationMs = 5200) {
  const stack = document.getElementById("toastStack");
  if (!stack) return;
  const el = document.createElement("div");
  el.className = `toast toast-${level}`;
  el.innerHTML = `<span>${escapeHtml(String(message || ""))}</span><button type="button" class="toast-close" aria-label="${escapeHtml(t("toast.close"))}">✕</button>`;
  const close = () => el.remove();
  el.querySelector(".toast-close")?.addEventListener("click", close);
  stack.appendChild(el);
  if (durationMs > 0) setTimeout(close, durationMs);
}

function setLog(msg) {
  state.logLine = msg;
  const el = document.getElementById("consoleText");
  if (el) el.textContent = msg;
}

function updateBootIndicator(status, message) {
  state.wasmStatus = status;
  const fallbackRaw =
    status === "ready"
      ? "Pyodide NumPy+SciPy kernel ready"
      : status === "error"
        ? "WASM engine error"
        : "Starting Pyodide...";
  state.wasmRawMessage = message || fallbackRaw;
  const localizedMsg = translateVerdict(state.wasmRawMessage);
  state.wasmMessage = localizedMsg;
  const dot = document.getElementById("wasmDot");
  const lbl = document.getElementById("wasmStatusText");
  const crashBanner = document.getElementById("workerCrashBanner");
  const crashMsg = document.getElementById("workerCrashMsg");
  if (dot) dot.className = "dot " + (status === "ready" ? "ready" : status === "error" ? "err" : "busy");
  if (lbl) lbl.textContent = localizedMsg;
  if (crashBanner) {
    crashBanner.hidden = status !== "error";
    if (status === "error" && crashMsg && state.wasmRawMessage) {
      crashMsg.textContent = `${t("worker.crash_msg")} (${localizedMsg})`;
    }
  }
}

function handleWorkerProgress(msg) {
  if (msg.stage === "stage5") {
    state.stage5ProgressPct = Math.max(0, Math.min(100, Number(msg.pct) || 0));
    if (msg.message) state.stage5ProgressText = msg.message;
    const bar = document.getElementById("gwrProgressBar");
    const lbl = document.getElementById("gwrProgressLabel");
    const wrap = document.getElementById("gwrProgressWrap");
    if (wrap) wrap.style.display = "block";
    if (bar) bar.style.width = `${state.stage5ProgressPct}%`;
    if (lbl && msg.message) lbl.textContent = `${msg.message} (${Math.round(state.stage5ProgressPct)}%)`;
  }
}

const dag = new DagClient("./worker/pyodide_worker.js", updateBootIndicator, setLog, handleWorkerProgress);

const mapInspector = new MapInspector("mapContainer", "pointInspector", {
  onToggleOsmFeature: (targetId, uid) => actions.toggleOsmFeature(targetId, uid, true),
  onClickObservation: (obsIdx) => actions.inspectObservation(obsIdx),
});

function parseGwrColIdx(rawVal) {
  if (rawVal === undefined || rawVal === null || rawVal === "") return 1;
  const n = Number(rawVal);
  return Number.isFinite(n) && n >= 0 ? n : 1;
}

function restyleMap(opts = {}) {
  const layerMode = document.getElementById("selMapLayer")?.value;
  const gwrColIdx = parseGwrColIdx(document.getElementById("selGwrVarLayer")?.value);
  const gwrMaskMode = document.getElementById("selGwrMaskMode")?.value || "fdr";
  const routeTargetId = document.getElementById("selRouteTarget")?.value || "";
  mapInspector.restyleObservations({
    stage1: state.stage1Res,
    stage3: state.stage3Res,
    stage4: state.stage4Res,
    stage5: state.stage5Res,
    layerMode,
    gwrColIdx,
    gwrMaskMode,
    activeModelId: state.activeSpecId,
    routeTargetId,
  });
  if (!opts.skipInspectorRefresh) {
    actions.refreshSelectedInspector();
  }
}

function resetAnalysisResults() {
  state.stage1Res = null;
  state.stage2Res = null;
  state.stage3Res = null;
  state.stage4Res = null;
  state.stage5Res = null;
  state.stage6Res = null;
  state.robustnessRes = null;
  state.staleStages.clear();
  state.stage3AdKey = "";
  state.stage3DiagModelId = "";
  state.lastInspectedRoute = null;
}

function patchOsmListDom(targetId) {
  const tgt = state.recipe.spatial_targets.find((t) => t.id === targetId);
  const prev = state.targetPreviews[targetId];
  if (!tgt || !prev) return false;
  const pane = document.getElementById("stageContent");
  const row = pane?.querySelector(`[data-target-id="${CSS.escape(targetId)}"]`);
  if (!row) return false;

  const exclSet = new Set(tgt.excluded_osm_ids || []);
  const total = prev.feature_index?.length || 0;
  const badge = row.querySelector(".tgt-active-badge");
  if (badge) badge.textContent = `${total - exclSet.size}/${total} aktív elem`;

  row.querySelectorAll(".osm-feature-item[data-uid]").forEach((item) => {
    const uid = item.getAttribute("data-uid");
    const isExcl = exclSet.has(uid);
    item.classList.toggle("excluded", isExcl);
    item.setAttribute("aria-pressed", isExcl ? "true" : "false");
    const firstSpan = item.querySelector("span");
    if (firstSpan && firstSpan.firstChild && firstSpan.firstChild.nodeType === Node.TEXT_NODE) {
      firstSpan.firstChild.nodeValue = isExcl ? "☐ " : "☑ ";
    }
    const stateSpan = item.querySelector(".osm-item-state");
    if (stateSpan) stateSpan.textContent = isExcl ? "kizárva" : "aktív";
  });
  return true;
}

const actions = {
  setLog,
  showToast,

  markStageStale(fromStage) {
    const k = Number(fromStage);
    if (k === 1) [1, 2, 3, 4, 5, 6].forEach((s) => state.staleStages.add(s));
    else if (k === 2) state.staleStages.add(2);
    else if (k === 3) [2, 3, 4, 5, 6].forEach((s) => state.staleStages.add(s));
    else if (k === 4) state.staleStages.add(4);
    else if (k === 5) state.staleStages.add(5);
    else if (k === 6) state.staleStages.add(6);
    this.updateStageTabsDom();
  },

  invalidateFromStage(k) {
    const st = Number(k);
    if (st === 1) {
      resetAnalysisResults();
      [1, 2, 3, 4, 5, 6].forEach((s) => state.staleStages.add(s));
    } else if (st === 2) {
      state.stage2Res = null;
      state.robustnessRes = null;
      state.staleStages.add(2);
    } else if (st === 3) {
      state.stage2Res = null;
      state.stage3Res = null;
      state.stage4Res = null;
      state.stage5Res = null;
      state.stage6Res = null;
      state.robustnessRes = null;
      [2, 3, 4, 5, 6].forEach((s) => state.staleStages.add(s));
    } else if (st === 4) {
      state.stage4Res = null;
      state.staleStages.add(4);
    } else if (st === 5) {
      state.stage5Res = null;
      state.staleStages.add(5);
    } else if (st === 6) {
      state.stage6Res = null;
      state.staleStages.add(6);
    }
    renderGuardStrip();
    restyleMap();
    this.updateStageTabsDom();
  },

  updateStageTabsDom() {
    document.querySelectorAll(".stage-tab").forEach((btn) => {
      const s = Number(btn.getAttribute("data-stage"));
      btn.classList.toggle("active", s === state.activeStage);
      const isDone =
        (s === 1 && state.stage1Res?.ok) ||
        (s === 2 && state.stage2Res?.ok) ||
        (s === 3 && state.stage3Res?.ok) ||
        (s === 4 && state.stage4Res?.ok) ||
        (s === 5 && state.stage5Res?.ok) ||
        (s === 6 && state.stage6Res?.ok);
      btn.classList.toggle("done", Boolean(isDone));
      const staleEl = btn.querySelector(".stale-badge");
      if (staleEl) {
        staleEl.hidden = !state.staleStages.has(s);
      }
    });
  },

  render() {
    this.updateStageTabsDom();

    const btnRunAll = document.getElementById("btnRunAll");
    if (btnRunAll) btnRunAll.disabled = !state.datasetInfo || state.pipelineRunning;

    const routeSelect = document.getElementById("selRouteTarget");
    if (routeSelect) {
      const tgts = state.recipe.spatial_targets || [];
      const prevVal = routeSelect.value;
      routeSelect.innerHTML = tgts.length
        ? tgts.map((t) => `<option value="${escapeHtml(t.id)}">${escapeHtml(t.id)}</option>`).join("")
        : `<option value="">— nincs célpont —</option>`;
      if (tgts.some((t) => t.id === prevVal)) routeSelect.value = prevVal;
    }

    const layerMode = document.getElementById("selMapLayer")?.value;
    const gwrVarSelect = document.getElementById("selGwrVarLayer");
    const gwrMaskSelect = document.getElementById("selGwrMaskMode");
    if (gwrVarSelect && state.stage5Res?.ok) {
      const cols = state.stage5Res.col_names || [];
      const cur = Math.min(parseGwrColIdx(gwrVarSelect.value), Math.max(0, cols.length - 1));
      gwrVarSelect.innerHTML = cols.map((c, i) => `<option value="${i}" ${i === cur ? "selected" : ""}>β̂(${escapeHtml(c)})</option>`).join("");
      gwrVarSelect.style.display = layerMode === "gwr_beta" ? "inline-block" : "none";
      if (gwrMaskSelect) gwrMaskSelect.style.display = layerMode === "gwr_beta" ? "inline-block" : "none";
    } else {
      if (gwrVarSelect) gwrVarSelect.style.display = "none";
      if (gwrMaskSelect) gwrMaskSelect.style.display = "none";
    }

    renderGuardStrip();

    const pane = document.getElementById("stageContent");
    const ctx = { state, actions };
    if (state.activeStage === 1) renderStage1(pane, ctx);
    else if (state.activeStage === 2) renderStage2(pane, ctx);
    else if (state.activeStage === 3) renderStage3(pane, ctx);
    else if (state.activeStage === 4) renderStage4(pane, ctx);
    else if (state.activeStage === 5) renderStage5(pane, ctx);
    else if (state.activeStage === 6) renderStage6(pane, ctx);
  },

  focusObservation(obsIdx) {
    if (!mapInspector.focusObservation(obsIdx)) {
      showToast(`A megfigyelés (#${obsIdx}) nem található a térképen.`, "warn");
    }
  },

  async onDatasetParsed(fileName, parsed, colProfile, guesses, { skipSync = false, hasPendingRecipe = false } = {}) {
    datasetSeq++;
    s1Seq++; s2Seq++; s3Seq++; s4Seq++; s5Seq++; s6Seq++;
    const hadPrevDataset = Boolean(state.columns);
    state.fileName = fileName;
    state.workspaceFileName = "";
    state.headers = parsed.headers;
    state.columns = parsed.columns;
    state.nRows = parsed.nRows;
    state.colProfile = colProfile;
    resetAnalysisResults();
    state.targetPreviews = {};
    state.osmFilterQuery = {};
    if (hadPrevDataset) {
      state.rawOsmTargets = {};
      state.rawGeojsonTargets = {};
      state.rawStreetGraphs = {};
      state.osmBundleFileName = "";
    }
    mapInspector.clearObservations();
    mapInspector.clearTargets();
    mapInspector.clearRoute();
    mapInspector.setFetchedBbox(null);
    mapInspector.renderPointInspector(null, null);

    if (!hasPendingRecipe) {
      const ds = state.recipe.dataset_schema;
      if ((!ds.lat_col || !parsed.columns[ds.lat_col]) && guesses.latGuess) ds.lat_col = guesses.latGuess;
      if ((!ds.lon_col || !parsed.columns[ds.lon_col]) && guesses.lonGuess) ds.lon_col = guesses.lonGuess;
      if ((!ds.row_id_col || !parsed.columns[ds.row_id_col]) && guesses.idGuess) ds.row_id_col = guesses.idGuess;
      if ((!ds.y_spec.num_col || !parsed.columns[ds.y_spec.num_col]) && guesses.yGuess) ds.y_spec.num_col = guesses.yGuess;

      const specA = state.recipe.models[0];
      if (
        !state.recipeFileName &&
        !state.workspaceFileName &&
        state.recipe.models.length === 1 &&
        specA &&
        specA.id === "spec_a" &&
        specA.covariates.length === 0
      ) {
        const skip = new Set([ds.lat_col, ds.lon_col, ds.row_id_col, ds.y_spec.num_col, ds.y_spec.den_col].filter(Boolean));
        for (const h of parsed.headers) {
          if (skip.has(h)) continue;
          if (colProfile[h]?.isNumeric && specA.covariates.length < 3) {
            specA.covariates.push({ col: h, type: "numeric", transform: "raw" });
          }
        }
      }
    }

    if (!skipSync) {
      await this.syncDatasetToWorker();
    }
  },

  async syncDatasetToWorker() {
    const ds = state.recipe.dataset_schema;
    if (!state.columns || !ds.lat_col || !ds.lon_col || !state.columns[ds.lat_col] || !state.columns[ds.lon_col]) {
      state.datasetInfo = null;
      mapInspector.clearObservations();
      mapInspector.renderPointInspector(null, null);
      if (state.columns) {
        setLog("Add meg az érvényes lat/lon oszlopokat az 1B szekcióban, majd kattints a 'Séma Rögzítése' gombra.");
      }
      this.render();
      return;
    }
    const mySeq = ++datasetSeq;
    s1Seq++; s2Seq++; s3Seq++; s4Seq++; s5Seq++; s6Seq++;
    resetAnalysisResults();

    try {
      setLog("Adatbázis betöltése és lokális metrikus vetület számítása...");
      await dag.waitUntilReady();
      if (mySeq !== datasetSeq) return;

      const res = await dag.call("load_dataset", {
        columns: state.columns,
        lat_col: ds.lat_col,
        lon_col: ds.lon_col,
      });
      if (mySeq !== datasetSeq) return;

      state.datasetInfo = res;

      for (const tgt of state.recipe.spatial_targets || []) {
        if (mySeq !== datasetSeq) return;
        const rawGj = state.rawGeojsonTargets[tgt.id];
        const rawOsm = state.rawOsmTargets[tgt.id];
        const rawPayload = rawGj || rawOsm;
        if (rawPayload) {
          const srcKind = rawGj ? "geojson" : "osm";
          const prev = await dag.call("ingest_target_raw", {
            target_id: tgt.id,
            raw_payload: rawPayload,
            source_kind: srcKind,
            excluded_osm_ids: tgt.excluded_osm_ids || [],
          });
          if (mySeq !== datasetSeq) return;
          state.targetPreviews[tgt.id] = prev;
        }
      }
      for (const [prof, sg] of Object.entries(state.rawStreetGraphs || {})) {
        if (mySeq !== datasetSeq) return;
        await dag.call("ingest_street_graph", { profile: prof, graph_json: sg });
      }

      mapInspector.setObservationsFromDataset(state.columns, ds.lat_col, ds.lon_col, ds.row_id_col, ds.y_spec, true, res.bbox);
      mapInspector.setFetchedBbox(res.bbox ? padBbox(res.bbox, Math.max(800, ...(state.recipe.spatial_targets || []).map((t) => Number(t.bbox_buffer_m ?? 800) || 800))) : null);
      this.refreshMapTargets();
      restyleMap();
      mapInspector.renderPointInspector(null, null);
      setLog(`Adatbázis kész: N=${res.n_rows} (geokódolt=${res.n_geocoded}, vetület=${res.projection.nearest_utm_epsg} / ${res.projection.mode})`);
      this.render();
    } catch (err) {
      if (mySeq !== datasetSeq) return;
      state.datasetInfo = null;
      mapInspector.clearObservations();
      mapInspector.clearTargets();
      mapInspector.clearRoute();
      mapInspector.renderPointInspector(null, null);
      setLog(`Séma hiba: ${err.message}`);
      showToast(`Séma hiba: ${err.message}`, "error");
      this.render();
    }
  },

  async removeTarget(idx) {
    const tgt = state.recipe.spatial_targets[idx];
    if (!tgt) return;
    const tid = tgt.id;
    state.recipe.spatial_targets.splice(idx, 1);
    delete state.targetPreviews[tid];
    delete state.rawOsmTargets[tid];
    delete state.rawGeojsonTargets[tid];
    this.invalidateFromStage(1);
    this.refreshMapTargets();
    if (state.datasetInfo) {
      try {
        await dag.waitUntilReady();
        await dag.call("remove_target", { target_id: tid });
      } catch (err) {
        console.warn("remove_target RPC figyelmeztetés:", err);
      }
    }
    this.render();
  },

  async ingestSingleGeojsonTarget(targetId, geojsonObj) {
    const myDsSeq = datasetSeq;
    state.rawGeojsonTargets[targetId] = geojsonObj;
    delete state.rawOsmTargets[targetId];
    this.invalidateFromStage(1);
    if (!state.datasetInfo) return;
    await dag.waitUntilReady();
    if (myDsSeq !== datasetSeq) return;
    const tgt = state.recipe.spatial_targets.find((t) => t.id === targetId);
    const prev = await dag.call("ingest_target_raw", {
      target_id: targetId,
      raw_payload: geojsonObj,
      source_kind: "geojson",
      excluded_osm_ids: tgt?.excluded_osm_ids || [],
    });
    if (myDsSeq !== datasetSeq) return;
    state.targetPreviews[targetId] = prev;
    mapInspector.setTargetPreview(targetId, prev.preview_geojson);
  },

  cancelOsmFetch() {
    if (osmFetchAbortCtrl) {
      osmFetchAbortCtrl.abort();
      osmFetchAbortCtrl = null;
    }
    state.osmFetching = false;
    setLog("OSM lekérés megszakítva a felhasználó által.");
    showToast("OSM lekérés megszakítva.", "warn");
    this.render();
  },

  async fetchAllTargetsOSM(forceRefetch = false) {
    if (!state.datasetInfo) {
      showToast("Előbb rögzítsd a dataset sémát!", "warn");
      return false;
    }
    const myDsSeq = datasetSeq;
    osmFetchAbortCtrl = new AbortController();
    state.osmFetching = true;
    this.render();
    try {
      await dag.waitUntilReady();
      if (myDsSeq !== datasetSeq) return false;
      const targets = state.recipe.spatial_targets || [];
      const osmTargets = targets.filter((t) => t.source === "osm_preset" || t.source === "osm_custom_tag");
      const routingTargets = targets.filter((t) => t.source !== "csv_column" && (t.distance_modes === "network_dijkstra" || t.distance_modes === "both"));
      const needProfiles = new Set(routingTargets.map((t) => t.routing_profile || "walk"));
      const streetBufM = Math.max(800, ...targets.map((t) => Number(t.bbox_buffer_m ?? 800) || 800));
      mapInspector.setFetchedBbox(padBbox(state.datasetInfo.bbox, streetBufM));

      for (const tgt of osmTargets) {
        if (myDsSeq !== datasetSeq) return false;
        if (!forceRefetch && state.targetPreviews[tgt.id] && state.rawOsmTargets[tgt.id]) {
          continue;
        }
        setLog(`OSM lekérés: ${tgt.name}...`);
        const { data, fromCache } = await fetchTargetOSM(tgt, state.datasetInfo.bbox, { signal: osmFetchAbortCtrl.signal });
        if (myDsSeq !== datasetSeq) return false;
        state.rawOsmTargets[tgt.id] = data;
        const prev = await dag.call("ingest_target_raw", {
          target_id: tgt.id,
          raw_payload: data,
          source_kind: "osm",
          excluded_osm_ids: tgt.excluded_osm_ids || [],
        });
        if (myDsSeq !== datasetSeq) return false;
        state.targetPreviews[tgt.id] = prev;
        mapInspector.setTargetPreview(tgt.id, prev.preview_geojson);
        setLog(`OSM kész (${tgt.id}): ${prev.n_features} objektum (${fromCache ? "IndexedDB cache" : "Overpass/Postpass élő"}).`);
      }

      for (const prof of needProfiles) {
        if (myDsSeq !== datasetSeq) return false;
        if (!forceRefetch && state.rawStreetGraphs[prof]) continue;
        setLog(`Utcahálózat lekérése CSR Dijkstra útvonalhoz (${prof}, buffer=${streetBufM}m)...`);
        const { data, fromCache } = await fetchStreetGraphOSM(state.datasetInfo.bbox, streetBufM, prof, { signal: osmFetchAbortCtrl.signal });
        if (myDsSeq !== datasetSeq) return false;
        state.rawStreetGraphs[prof] = data;
        const gRes = await dag.call("ingest_street_graph", { profile: prof, graph_json: data });
        if (myDsSeq !== datasetSeq) return false;
        setLog(`Utcahálózat (${prof}) kész: ${gRes.n_nodes} csúcs, ${gRes.n_edges} él (${fromCache ? "cache" : "Overpass/Postpass"}).`);
      }
      return true;
    } catch (err) {
      if (err?.name === "AbortError" || myDsSeq !== datasetSeq) return false;
      setLog(`OSM lekérési hiba: ${err.message}`);
      showToast(`OSM API hiba: ${err.message}`, "error");
      return false;
    } finally {
      state.osmFetching = false;
      osmFetchAbortCtrl = null;
      this.render();
    }
  },

  restyleMap() {
    restyleMap();
  },

  toggleOsmFeature(targetId, uid, fromMap = false) {
    if (fromMap && state.activeStage !== 1) {
      setLog("OSM elemek ki/bekapcsolása a térképen csak az 1. fázisban érhető el.");
      return;
    }
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
    this.invalidateFromStage(1);
    mapInspector.clearRoute();
    mapInspector.renderPointInspector(null, null);
    setLog(`OSM elem (${targetId}:${uid}) ${idx >= 0 ? "bekapcsolva" : "kizárva"} (újratervezés szükséges).`);
    if (!patchOsmListDom(targetId)) {
      this.render();
    }
  },

  setTargetExclusions(targetId, uids, excludeBool) {
    const tgt = state.recipe.spatial_targets.find((t) => t.id === targetId);
    if (!tgt || !Array.isArray(uids)) return;
    const exclSet = new Set(tgt.excluded_osm_ids || []);
    for (const uid of uids) {
      if (excludeBool) exclSet.add(uid);
      else exclSet.delete(uid);
    }
    tgt.excluded_osm_ids = Array.from(exclSet);

    const prev = state.targetPreviews[targetId];
    if (prev && prev.preview_geojson) {
      for (const f of prev.preview_geojson.features || []) {
        f.properties.excluded = exclSet.has(f.properties.uid);
      }
      mapInspector.setTargetPreview(targetId, prev.preview_geojson);
    }
    this.invalidateFromStage(1);
    mapInspector.clearRoute();
    mapInspector.renderPointInspector(null, null);
    setLog(`OSM tömeges szűrés (${targetId}): ${uids.length} elem ${excludeBool ? "kizárva" : "aktívvá téve"}.`);
    if (!patchOsmListDom(targetId)) {
      this.render();
    }
  },

  refreshMapTargets() {
    mapInspector.clearTargets();
    for (const t of state.recipe.spatial_targets) {
      if (state.targetPreviews[t.id]) {
        mapInspector.setTargetPreview(t.id, state.targetPreviews[t.id].preview_geojson);
      }
    }
  },

  invalidateTargetSnapshot(targetId) {
    delete state.targetPreviews[targetId];
    delete state.rawOsmTargets[targetId];
    this.invalidateFromStage(1);
    mapInspector.clearRoute();
    mapInspector.renderPointInspector(null, null);
    this.refreshMapTargets();
    this.render();
  },

  async exportOsmBundle() {
    const hasTargets = Object.keys(state.rawOsmTargets).length || Object.keys(state.rawGeojsonTargets).length || Object.keys(state.rawStreetGraphs).length;
    if (!hasTargets) {
      showToast("Nincs exportálható OSM/GeoJSON célpont vagy utcahálózat.", "warn");
      return;
    }
    const bundle = buildOsmBundle({
      bbox: state.datasetInfo?.bbox || null,
      targets: state.recipe.spatial_targets || [],
      rawOsmTargets: state.rawOsmTargets,
      rawGeojsonTargets: state.rawGeojsonTargets,
      rawStreetGraphs: state.rawStreetGraphs,
    });
    const slug = slugifyTitle(state.recipe?.recipe_meta?.title, "geospec_snapshot");
    this.downloadText(`${slug}.osm_bundle.json`, JSON.stringify(bundle), "application/json");
    setLog(`Offline OSM & hálózati pillanatfelvétel (${slug}.osm_bundle.json) elmentve.`);
    showToast("OSM csomag exportálva.", "pass");
  },

  async importOsmBundle(parsedBundle, { fileName = "", skipIngest = false, skipRender = false } = {}) {
    if (!parsedBundle) return;
    if (fileName) state.osmBundleFileName = fileName;

    for (const [tid, entry] of Object.entries(parsedBundle.targets || {})) {
      const payload = entry?.payload || entry?.raw;
      if (!payload) continue;
      const cfg = entry.target_cfg || {};
      const srcKind = entry.source_kind === "geojson" || cfg.source === "geojson" ? "geojson" : "osm";
      if (srcKind === "geojson") {
        state.rawGeojsonTargets[tid] = payload;
        delete state.rawOsmTargets[tid];
      } else {
        state.rawOsmTargets[tid] = payload;
        delete state.rawGeojsonTargets[tid];
      }
      if (!state.recipe.spatial_targets.some((t) => t.id === tid)) {
        state.recipe.spatial_targets.push({
          id: tid,
          name: cfg.name || entry.name || tid,
          source: cfg.source || srcKind,
          preset_id: cfg.preset_id,
          osm_filter: cfg.osm_filter,
          element_types: cfg.element_types,
          excluded_osm_ids: Array.isArray(cfg.excluded_osm_ids) ? [...cfg.excluded_osm_ids] : [],
          geometry_handling: cfg.geometry_handling || entry.geometry_handling || "point",
          inside_is_zero: Boolean(cfg.inside_is_zero ?? entry.inside_is_zero),
          distance_modes: cfg.distance_modes || "both",
          routing_profile: cfg.routing_profile || "walk",
          max_snap_distance_m: Number(cfg.max_snap_distance_m) || 300,
          bbox_buffer_m: Number(cfg.bbox_buffer_m) || 800,
        });
      }
    }

    for (const [prof, sg] of Object.entries(parsedBundle.street_graphs || {})) {
      if (!sg) continue;
      state.rawStreetGraphs[prof] = sg;
    }

    if (state.datasetInfo && !skipIngest) {
      await dag.waitUntilReady();
      for (const [tid, entry] of Object.entries(parsedBundle.targets || {})) {
        const payload = entry?.payload || entry?.raw;
        if (!payload) continue;
        const cfg = entry.target_cfg || {};
        const srcKind = entry.source_kind === "geojson" || cfg.source === "geojson" ? "geojson" : "osm";
        const tgt = state.recipe.spatial_targets.find((t) => t.id === tid);
        const prev = await dag.call("ingest_target_raw", {
          target_id: tid,
          raw_payload: payload,
          source_kind: srcKind,
          excluded_osm_ids: tgt?.excluded_osm_ids || [],
        });
        state.targetPreviews[tid] = prev;
      }
      for (const [prof, sg] of Object.entries(parsedBundle.street_graphs || {})) {
        if (!sg) continue;
        await dag.call("ingest_street_graph", { profile: prof, graph_json: sg });
      }
    }

    resetAnalysisResults();
    this.refreshMapTargets();
    setLog(`Offline OSM pillanatfelvétel betöltve (${Object.keys(parsedBundle.targets || {}).length} célpont, ${Object.keys(parsedBundle.street_graphs || {}).length} úthálózat).`);
    if (!skipRender) this.render();
  },

  async runStage1() {
    if (!state.datasetInfo) return false;
    const mySeq = ++s1Seq;
    try {
      await dag.waitUntilReady();
      if (mySeq !== s1Seq) return false;

      const needsFetch = state.recipe.spatial_targets.some(
        (t) => (t.source === "osm_preset" || t.source === "osm_custom_tag") && !state.targetPreviews[t.id]
      );
      if (needsFetch) {
        const okFetch = await this.fetchAllTargetsOSM();
        if (!okFetch || mySeq !== s1Seq) return false;
      }

      const needProfiles = Array.from(
        new Set(
          state.recipe.spatial_targets
            .filter((t) => t.source !== "csv_column" && (t.distance_modes === "network_dijkstra" || t.distance_modes === "both"))
            .map((t) => t.routing_profile || "walk")
        )
      );
      const missingProfiles = needProfiles.filter((p) => !state.rawStreetGraphs[p]);
      if (missingProfiles.length) {
        const streetBufM = Math.max(800, ...state.recipe.spatial_targets.map((t) => Number(t.bbox_buffer_m ?? 800) || 800));
        osmFetchAbortCtrl = new AbortController();
        state.osmFetching = true;
        try {
          for (const prof of missingProfiles) {
            setLog(`Utcahálózat lekérése (${prof}, buffer=${streetBufM}m)...`);
            const { data } = await fetchStreetGraphOSM(state.datasetInfo.bbox, streetBufM, prof, { signal: osmFetchAbortCtrl.signal });
            if (mySeq !== s1Seq) return false;
            state.rawStreetGraphs[prof] = data;
            await dag.call("ingest_street_graph", { profile: prof, graph_json: data });
          }
        } finally {
          state.osmFetching = false;
          osmFetchAbortCtrl = null;
        }
      }

      setLog("1. Fázis: Euklideszi és CSR Dijkstra távolságok számítása...");
      const res = await dag.call("run_stage_1", { recipe: state.recipe, run_id: dag.nextRunId() });
      if (mySeq !== s1Seq) return false;
      state.stage1Res = res;
      state.stage2Res = null;
      state.stage3Res = null;
      state.stage4Res = null;
      state.stage5Res = null;
      state.robustnessRes = null;
      state.staleStages.delete(1);
      [2, 3, 4, 5].forEach((s) => state.staleStages.add(s));
      renderGuardStrip();
      restyleMap();
      setLog(`1. Fázis kész (${res.summary.length} célpont távolságmátrixai frissítve).`);
      this.render();
      return true;
    } catch (err) {
      if (err.name === "CancelledError" || err.name === "AbortError" || mySeq !== s1Seq) return false;
      setLog(`Stage 1 hiba: ${err.message}`);
      showToast(`Stage 1 hiba: ${err.message}`, "error");
      return false;
    }
  },

  async runStage2() {
    if (!state.datasetInfo) return false;
    const mySeq = ++s2Seq;
    try {
      await dag.waitUntilReady();
      if (mySeq !== s2Seq) return false;
      setLog("2. Fázis: Minta-vízesés, leíró statisztika, Step 0..K & Gelbach (2016)...");
      const res = await dag.call("run_stage_2", { recipe: state.recipe, run_id: dag.nextRunId() });
      if (mySeq !== s2Seq) return false;
      // Preserve downstream stage3Res..stage5Res (Stage 2 is descriptive/composition)
      state.stage2Res = res;
      state.staleStages.delete(2);
      renderGuardStrip();
      restyleMap();
      setLog("2. Fázis kész.");
      this.render();
      return true;
    } catch (err) {
      if (err.name === "CancelledError" || mySeq !== s2Seq) return false;
      setLog(`Stage 2 hiba: ${err.message}`);
      showToast(`Stage 2 hiba: ${err.message}`, "error");
      return false;
    }
  },

  async runStage3({ patchResultsOnly = false } = {}) {
    if (!state.datasetInfo) return false;
    const mySeq = ++s3Seq;
    try {
      await dag.waitUntilReady();
      if (mySeq !== s3Seq) return false;
      const t0 = performance.now();
      state.recipe.default_cov_type = state.covType;
      const res = await dag.call("run_stage_3", {
        recipe: state.recipe,
        cov_type: state.covType,
        run_id: dag.nextRunId(),
      });
      if (mySeq !== s3Seq) return false;
      state.stage3Res = res;
      state.stage4Res = null;
      state.stage5Res = null;
      state.stage6Res = null;
      state.staleStages.delete(3);
      if (!patchResultsOnly) {
        [4, 5, 6].forEach((s) => state.staleStages.add(s));
      }
      if (!res.ok) {
        setLog(`3. Fázis figyelmeztetés: ${res.reason || "Ismeretlen hiba"}`);
        if (!patchResultsOnly) showToast(res.reason || "3. Fázis nem futtatható.", "warn");
        renderGuardStrip();
        restyleMap();
        this.render();
        return false;
      }
      const dt = (performance.now() - t0).toFixed(1);
      setLog(`3. Fázis kész (${dt} ms, ${(res.models || []).length} modell, ${state.covType} SE).`);
      renderGuardStrip();
      restyleMap();
      if (patchResultsOnly && state.activeStage === 3) {
        patchStage3ResultsOnly(document.getElementById("stageContent"), { state, actions });
      } else {
        this.render();
      }
      return true;
    } catch (err) {
      if (err.name === "CancelledError" || mySeq !== s3Seq) return false;
      setLog(`Stage 3 hiba: ${err.message}`);
      if (!patchResultsOnly) showToast(`Stage 3 hiba: ${err.message}`, "error");
      return false;
    }
  },

  runStage3Debounced() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => this.runStage3({ patchResultsOnly: true }), 90);
  },

  async runRobustnessMatrix() {
    if (!state.datasetInfo) return;
    try {
      await dag.waitUntilReady();
      setLog("3×2 Funkcionális Forma × SE Robusztussági Mátrix számítása...");
      const res = await dag.call("run_robustness_matrix", { recipe: state.recipe, run_id: dag.nextRunId() });
      state.robustnessRes = res;
      if (res.ok) {
        setLog(`3×2 Robusztussági Mátrix kész (N=${res.n_obs}).`);
        showToast(`3×2 Robusztussági Mátrix kész (N=${res.n_obs})`, "pass");
      } else {
        showToast(res.reason || "Mátrix hiba", "warn");
      }
      if (state.activeStage === 3) {
        patchStage3ResultsOnly(document.getElementById("stageContent"), { state, actions });
      }
    } catch (err) {
      showToast(`Robusztussági mátrix hiba: ${err.message}`, "error");
    }
  },

  async generateM1M4Specs() {
    const active = state.recipe.models.find((m) => m.id === state.activeSpecId) || state.recipe.models[0];
    const covs = active ? active.covariates || [] : [];
    const tgts = state.recipe.spatial_targets || [];
    if (!tgts.length) {
      showToast("Előbb vegyél fel legalább 1 térbeli célpontot az 1. fázisban!", "warn");
      return;
    }
    await dag.waitUntilReady();
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
    this.invalidateFromStage(3);
    this.render();
    await this.runStage3();
  },

  async runStage4() {
    if (!state.datasetInfo) return false;
    const mySeq = ++s4Seq;
    try {
      await dag.waitUntilReady();
      if (mySeq !== s4Seq) return false;
      setLog("4. Fázis: Multi-W építés, Moran/LISA, SAR/SEM/SDM & Krinsky–Robb...");
      const res = await dag.call("run_stage_4", { recipe: state.recipe, run_id: dag.nextRunId() });
      if (mySeq !== s4Seq) return false;
      state.stage4Res = res;
      state.staleStages.delete(4);
      if (res.ok) {
        setLog(`4. Fázis kész (${translateVerdict(res.anselin?.verdict || "")}).`);
      } else {
        setLog(`Stage 4 figyelmeztetés: ${res.reason || "Nem futtatható"}`);
        showToast(res.reason || "4. Fázis nem futtatható.", "warn");
      }
      renderGuardStrip();
      this.render();
      restyleMap();
      return Boolean(res.ok);
    } catch (err) {
      if (err.name === "CancelledError" || mySeq !== s4Seq) return false;
      setLog(`Stage 4 hiba: ${err.message}`);
      showToast(`Stage 4 hiba: ${err.message}`, "error");
      return false;
    }
  },

  async runStage5() {
    if (!state.datasetInfo) return false;
    const mySeq = ++s5Seq;
    state.stage5Busy = true;
    state.stage5ProgressPct = 5;
    state.stage5ProgressText = "Adaptív GWR/MGWR optimalizálás indítása...";
    if (state.activeStage === 5) this.render();
    try {
      await dag.waitUntilReady();
      if (mySeq !== s5Seq) return false;
      setLog("5. Fázis: Szingularitás-védett Adaptív Bisquare GWR/MGWR...");
      const res = await dag.call("run_stage_5", { recipe: state.recipe, run_id: dag.nextRunId() });
      if (mySeq !== s5Seq) return false;
      state.stage5Res = res;
      state.staleStages.delete(5);
      state.stage5ProgressPct = 100;
      if (res.ok) {
        setLog(`5. Fázis kész (b*=${res.optimal_bw}, ${(res.mode || "gwr").toUpperCase()} Adj. R²=${res.gwr_adj_r2}).`);
      } else {
        setLog(`GWR figyelmeztetés: ${res.reason}`);
        showToast(res.reason || "GWR nem futtatható.", "warn");
      }
      this.render();
      restyleMap();
      return Boolean(res.ok);
    } catch (err) {
      if (err.name === "CancelledError" || mySeq !== s5Seq) return false;
      setLog(`Stage 5 hiba: ${err.message}`);
      showToast(`Stage 5 hiba: ${err.message}`, "error");
      return false;
    } finally {
      state.stage5Busy = false;
      const wrap = document.getElementById("gwrProgressWrap");
      if (wrap) wrap.style.display = "none";
    }
  },

  async runStage6() {
    if (!state.datasetInfo) return false;
    const mySeq = ++s6Seq;
    try {
      await dag.waitUntilReady();
      if (mySeq !== s6Seq) return false;
      setLog("6. Fázis: Pure-NumPy GBRT & Egzakt Lundberg TreeSHAP...");
      const res = await dag.call("run_stage_6", { recipe: state.recipe, run_id: dag.nextRunId() });
      if (mySeq !== s6Seq) return false;
      state.stage6Res = res;
      state.staleStages.delete(6);
      if (res.ok) {
        const g = res.grouped_importance || {};
        setLog(`6. Fázis kész (Seed=${res.random_seed}, Fizikai=${g.physical_pct}%, Vasút=${g.rail_pct}%, Δ_SHAP=${Number(res.local_accuracy_max_err).toExponential(1)}).`);
      } else {
        setLog(`Stage 6 figyelmeztetés: ${res.reason || "Nem futtatható"}`);
        showToast(res.reason || "6. Fázis nem futtatható.", "warn");
      }
      this.render();
      return Boolean(res.ok);
    } catch (err) {
      if (err.name === "CancelledError" || mySeq !== s6Seq) return false;
      setLog(`Stage 6 hiba: ${err.message}`);
      showToast(`Stage 6 hiba: ${err.message}`, "error");
      return false;
    }
  },

  async runFullPipeline() {
    if (!state.datasetInfo || state.pipelineRunning) return;
    state.pipelineRunning = true;
    const btnRunAll = document.getElementById("btnRunAll");
    if (btnRunAll) btnRunAll.disabled = true;
    try {
      setLog("WASM motor ellenőrzése és 6-fázisú pipeline indítása...");
      await dag.waitUntilReady();
      if (!(await this.runStage1())) return;
      if (!(await this.runStage2())) return;
      if (!(await this.runStage3())) return;
      if (!(await this.runStage4())) return;
      if (!(await this.runStage5())) return;
      if (!(await this.runStage6())) return;
      setLog("Teljes 6-fázisú térökonometriai & TreeSHAP pipeline befejezve.");
      showToast("Teljes 6-fázisú pipeline sikeresen lefutott.", "pass");
    } catch (err) {
      setLog(`Pipeline megszakadt: ${err.message}`);
      showToast(`Pipeline megszakadt: ${err.message}`, "error");
    } finally {
      state.pipelineRunning = false;
      if (btnRunAll) btnRunAll.disabled = !state.datasetInfo;
    }
  },

  refreshSelectedInspector() {
    const obsIdx = mapInspector.selectedObsIdx;
    if (obsIdx === null || obsIdx === undefined) {
      mapInspector.renderPointInspector(null, null);
      return;
    }
    const tgtId = document.getElementById("selRouteTarget")?.value || state.recipe.spatial_targets?.[0]?.id || "";
    const route =
      state.lastInspectedRoute && state.lastInspectedRoute.obsIdx === obsIdx && state.lastInspectedRoute.tgtId === tgtId
        ? state.lastInspectedRoute.route
        : null;
    const feat = mapInspector.obsByIdx.get(Number(obsIdx)) || mapInspector.obsFeatures.find((f) => f.properties.obs_idx === obsIdx);

    let resid = null;
    let modelLabel = "";
    if (state.stage3Res?.models?.length) {
      const m =
        state.stage3Res.models.find((x) => x.model_id === state.activeSpecId && x.resid) ||
        state.stage3Res.models.find((x) => x.resid) ||
        state.stage3Res.models[0];
      modelLabel = m?.model_name || m?.model_id || "";
      const pos = m?.valid_indices?.indexOf(obsIdx) ?? -1;
      if (pos >= 0) resid = m.resid[pos];
    }
    let lisa = null;
    if (state.stage4Res?.ok) {
      const pos = state.stage4Res.valid_indices.indexOf(obsIdx);
      if (pos >= 0) lisa = state.stage4Res.lisa_resid.quadrants[pos];
    }
    let gwr_beta = null;
    let gwr_t = null;
    let gwr_r2 = null;
    let gwr_kappa = null;
    if (state.stage5Res?.ok) {
      const pos = state.stage5Res.valid_indices.indexOf(obsIdx);
      const ci = Math.min(parseGwrColIdx(document.getElementById("selGwrVarLayer")?.value), state.stage5Res.col_names.length - 1);
      if (pos >= 0) {
        gwr_beta = state.stage5Res.betas_native[pos][ci];
        gwr_t = state.stage5Res.t_vals[pos][ci];
        gwr_r2 = state.stage5Res.local_r2?.[pos] ?? null;
        gwr_kappa = state.stage5Res.local_kappas?.[pos] ?? null;
      }
    }

    const covSummary = [];
    if (state.columns) {
      const activeM = state.recipe.models.find((x) => x.id === state.activeSpecId) || state.recipe.models[0];
      for (const c of (activeM?.covariates || []).slice(0, 4)) {
        if (state.columns[c.col]) covSummary.push({ key: c.col, val: state.columns[c.col][obsIdx] });
      }
    }

    mapInspector.renderPointInspector(obsIdx, {
      row_id: feat?.properties?.row_id ?? obsIdx,
      y_raw: feat?.properties?.y_raw ?? null,
      route,
      resid,
      modelLabel,
      lisa,
      gwr_beta,
      gwr_t,
      gwr_r2,
      gwr_kappa,
      covSummary,
    });
  },

  async inspectObservation(obsIdx) {
    const tgtId = document.getElementById("selRouteTarget")?.value || state.recipe.spatial_targets?.[0]?.id || "";
    let route = null;
    if (tgtId) {
      try {
        await dag.waitUntilReady();
        route = await dag.call("get_single_route", { obs_idx: obsIdx, target_id: tgtId, recipe: state.recipe });
        mapInspector.showRoute(route);
      } catch {
        route = null;
      }
    } else {
      mapInspector.clearRoute();
    }

    state.lastInspectedRoute = { obsIdx, tgtId, route };
    restyleMap({ skipInspectorRefresh: true });
    this.refreshSelectedInspector();
  },

  async copyLatexTable(kindOverride = null) {
    try {
      await dag.waitUntilReady();
      const kind = kindOverride || (state.showPct ? "ols_pct" : "ols_main");
      const out = await dag.call("export_latex", { recipe: state.recipe, kind });
      const tex = out?.latex || "";
      if (!tex || tex.startsWith("% No ")) {
        showToast("Előbb futtasd le a megfelelő elemzési fázist!", "warn");
        return;
      }
      let copied = false;
      if (navigator.clipboard?.writeText) {
        try {
          await navigator.clipboard.writeText(tex);
          copied = true;
        } catch (_) {}
      }
      if (!copied) {
        this.downloadText(`geospec_${kind}.tex`, tex, "text/plain");
        showToast(`Vágólap korlátozva — fájlként letöltve: geospec_${kind}.tex`, "info");
      } else {
        showToast(t("toast.copied_latex"), "pass");
      }
      setLog(`LaTeX (${kind}) exportálva.`);
    } catch (err) {
      showToast(`LaTeX hiba: ${err.message}`, "error");
    }
  },

  async exportGeojsonOnly() {
    if (!state.datasetInfo) {
      showToast("Előbb tölts be egy érvényes adatbázist a GeoJSON exporthoz!", "warn");
      return;
    }
    try {
      await dag.waitUntilReady();
      const out = await dag.call("export_geojson", { recipe: state.recipe });
      const slug = slugifyTitle(state.recipe?.recipe_meta?.title, "geospec");
      this.downloadText(`${slug}_enriched.geojson`, JSON.stringify(out.geojson, null, 2), "application/geo+json");
      setLog(`QGIS GeoJSON export kész (${slug}_enriched.geojson).`);
      showToast("QGIS GeoJSON exportálva.", "pass");
    } catch (err) {
      showToast(`GeoJSON export hiba: ${err.message}`, "error");
    }
  },

  async _collectPlotlySvgs() {
    const svgs = {};
    if (!window.Plotly) return svgs;
    for (const el of document.querySelectorAll(".plot-box")) {
      if (!el.id || !el.data) continue;
      try {
        const uri = await window.Plotly.toImage(el, { format: "svg", width: 900, height: 420 });
        if (typeof uri === "string" && uri.includes("<svg")) {
          const idx = uri.indexOf("<svg");
          svgs[el.id] = decodeURIComponent(uri.slice(idx));
        }
      } catch (_) {}
    }
    return svgs;
  },

  async exportBundle(kind = "zip") {
    if (kind === "geojson") return this.exportGeojsonOnly();
    if (!state.datasetInfo) {
      showToast("Előbb tölts be egy érvényes adatbázist a .ZIP exporthoz!", "warn");
      return;
    }
    try {
      await dag.waitUntilReady();
      setLog("Publikációs export csomag összeállítása (.ZIP + QML + SVG)...");
      const hasSnap =
        Object.keys(state.rawOsmTargets).length > 0 ||
        Object.keys(state.rawGeojsonTargets).length > 0 ||
        Object.keys(state.rawStreetGraphs).length > 0;
      const osm_bundle = hasSnap
        ? buildOsmBundle({
            bbox: state.datasetInfo.bbox || null,
            targets: state.recipe.spatial_targets || [],
            rawOsmTargets: state.rawOsmTargets,
            rawGeojsonTargets: state.rawGeojsonTargets,
            rawStreetGraphs: state.rawStreetGraphs,
          })
        : null;
      const svg_figures = await this._collectPlotlySvgs();

      const bundle = await dag.call("export_bundle", { recipe: state.recipe, osm_bundle, svg_figures });
      const slug = slugifyTitle(state.recipe?.recipe_meta?.title, "geospec");
      const bin = Uint8Array.from(atob(bundle.zip_base64), (c) => c.charCodeAt(0));
      const blob = new Blob([bin], { type: "application/zip" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${slug}_publication_bundle_${Date.now()}.zip`;
      a.click();
      URL.revokeObjectURL(url);
      setLog(`Publikációs .ZIP csomag elkészült (${Object.keys(svg_figures).length} SVG ábra).`);
      showToast("Publikációs .ZIP csomag letöltve.", "pass");
    } catch (err) {
      showToast(`Export hiba: ${err.message}`, "error");
    }
  },

  exportRecipe() {
    state.recipe.default_cov_type = state.covType;
    const slug = slugifyTitle(state.recipe?.recipe_meta?.title, "experiment");
    const fileName = `${slug}_recipe.json`;
    this.downloadText(fileName, JSON.stringify(state.recipe, null, 2), "application/json");
    setLog(`${fileName} mentve (nyers adatok nélkül).`);
    showToast(`${fileName} mentve.`, "pass");
  },

  exportWorkspace() {
    state.recipe.default_cov_type = state.covType;
    const hasSnap =
      Object.keys(state.rawOsmTargets).length > 0 ||
      Object.keys(state.rawGeojsonTargets).length > 0 ||
      Object.keys(state.rawStreetGraphs).length > 0;
    const osmBundle = hasSnap
      ? buildOsmBundle({
          bbox: state.datasetInfo?.bbox || null,
          targets: state.recipe.spatial_targets || [],
          rawOsmTargets: state.rawOsmTargets,
          rawGeojsonTargets: state.rawGeojsonTargets,
          rawStreetGraphs: state.rawStreetGraphs,
        })
      : null;

    const ws = buildWorkspaceBundle({
      recipe: state.recipe,
      fileName: state.fileName,
      headers: state.headers,
      columns: state.columns,
      nRows: state.nRows,
      osmBundle,
      activeSpecId: state.activeSpecId,
      covType: state.covType,
      showPct: state.showPct,
    });
    const slug = slugifyTitle(state.recipe?.recipe_meta?.title, "geospec_study");
    const fileName = `${slug}.geospec.json`;
    this.downloadText(fileName, JSON.stringify(ws), "application/json");
    setLog(`Teljes projekt mentve: ${fileName}`);
    showToast(`${fileName} projekt mentve.`, "pass");
  },

  async importRecipe(loadedRecipe, { fileName = "", skipSync = false, preserveSnapshots = true } = {}) {
    s1Seq++; s2Seq++; s3Seq++; s4Seq++; s5Seq++;
    const clean = sanitizeRecipe(loadedRecipe, createDefaultRecipe());
    state.recipe = clean;
    if (fileName) state.recipeFileName = fileName;
    state.workspaceFileName = "";
    const modelIds = new Set((clean.models || []).map((m) => m.id));
    const preferredSpec = clean.spatial_engine?.active_model_id || clean.spatial_engine?.gwr?.focal_model_id;
    state.activeSpecId = preferredSpec && modelIds.has(preferredSpec) ? preferredSpec : clean.models?.[0]?.id || "spec_a";
    const clMode = clean.dataset_schema?.cluster_spec?.mode;
    state.covType = clean.default_cov_type || (clMode && clMode !== "none" ? "cluster" : "HC3");

    resetAnalysisResults();
    state.targetPreviews = {};
    state.osmFilterQuery = {};
    mapInspector.clearTargets();
    mapInspector.clearRoute();

    if (!preserveSnapshots) {
      state.rawOsmTargets = {};
      state.rawGeojsonTargets = {};
      state.rawStreetGraphs = {};
      state.osmBundleFileName = "";
    } else {
      const validIds = new Set((clean.spatial_targets || []).map((t) => t.id));
      for (const tid of Object.keys(state.rawOsmTargets)) {
        if (!validIds.has(tid)) delete state.rawOsmTargets[tid];
      }
      for (const tid of Object.keys(state.rawGeojsonTargets)) {
        if (!validIds.has(tid)) delete state.rawGeojsonTargets[tid];
      }
    }

    if (loadedRecipe && typeof loadedRecipe.osm_bundle === "object" && loadedRecipe.osm_bundle !== null) {
      try {
        const parsedEmbeddedBundle = loadedRecipe.osm_bundle;
        await this.importOsmBundle(parsedEmbeddedBundle, {
          fileName: fileName || state.osmBundleFileName,
          skipIngest: true,
          skipRender: true,
        });
      } catch (err) {
        console.warn("Beágyazott osm_bundle figyelmen kívül hagyva:", err);
      }
    }

    if (state.columns) {
      const { profile, guesses } = inferColumnProfile(state.headers, state.columns);
      state.colProfile = profile;
      const ds = state.recipe.dataset_schema;
      if ((!ds.lat_col || !state.columns[ds.lat_col]) && guesses.latGuess) ds.lat_col = guesses.latGuess;
      if ((!ds.lon_col || !state.columns[ds.lon_col]) && guesses.lonGuess) ds.lon_col = guesses.lonGuess;
      if ((!ds.row_id_col || !state.columns[ds.row_id_col]) && guesses.idGuess) ds.row_id_col = guesses.idGuess;
      if ((!ds.y_spec.num_col || !state.columns[ds.y_spec.num_col]) && guesses.yGuess) ds.y_spec.num_col = guesses.yGuess;
    }

    setLog(`Recept betöltve: ${clean.recipe_meta?.title || fileName || "recept"}`);
    if (skipSync) return;

    if (state.columns) {
      await this.syncDatasetToWorker();
    } else {
      if (state.activeStage > 1) state.activeStage = 1;
      this.render();
    }
  },

  async importWorkspace(parsedWorkspace, { fileName = "", skipSync = false } = {}) {
    datasetSeq++;
    s1Seq++; s2Seq++; s3Seq++; s4Seq++; s5Seq++;

    resetAnalysisResults();
    state.targetPreviews = {};
    state.rawOsmTargets = {};
    state.rawGeojsonTargets = {};
    state.rawStreetGraphs = {};
    state.osmFilterQuery = {};
    state.osmBundleFileName = "";
    mapInspector.clearObservations();
    mapInspector.clearTargets();
    mapInspector.clearRoute();
    mapInspector.setFetchedBbox(null);
    mapInspector.renderPointInspector(null, null);

    state.recipe = parsedWorkspace.recipe;
    state.workspaceFileName = fileName || "workspace.geospec.json";
    state.recipeFileName = fileName || "workspace.geospec.json";
    state.activeSpecId = parsedWorkspace.ui_state.activeSpecId;
    state.covType = parsedWorkspace.ui_state.covType;
    state.showPct = parsedWorkspace.ui_state.showPct;

    if (parsedWorkspace.dataset) {
      state.fileName = parsedWorkspace.dataset.fileName;
      state.headers = parsedWorkspace.dataset.headers;
      state.columns = parsedWorkspace.dataset.columns;
      state.nRows = parsedWorkspace.dataset.nRows;
      const { profile } = inferColumnProfile(state.headers, state.columns);
      state.colProfile = profile;
    } else {
      state.fileName = "";
      state.headers = [];
      state.columns = null;
      state.nRows = 0;
      state.colProfile = {};
      state.datasetInfo = null;
      if (state.wasmStatus === "ready") {
        await dag.call("reset_session", {}).catch(() => {});
      }
    }

    if (parsedWorkspace.osm_bundle) {
      await this.importOsmBundle(parsedWorkspace.osm_bundle, {
        fileName: fileName || "workspace_osm_bundle",
        skipIngest: true,
        skipRender: true,
      });
    }

    setLog(`Projekt betöltve: ${state.recipe.recipe_meta?.title || fileName || ".geospec.json"}`);
    if (skipSync) return;

    if (state.columns) {
      await this.syncDatasetToWorker();
    } else {
      if (state.activeStage > 1) state.activeStage = 1;
      this.render();
    }
  },

  async handleIncomingFiles(fileList) {
    const files = Array.from(fileList || []).filter(Boolean);
    if (!files.length) return;
    try {
      const entries = [];
      for (const f of files) {
        const text = await readTextWithEncodingFallback(f);
        entries.push({ fileName: f.name, text });
      }
      // Atomic parsing & validation before any state/worker mutation
      const { prepareIncomingBatch } = await import("./state/recipe_store.js");
      const plan = prepareIncomingBatch(entries, { hasDataset: Boolean(state.columns) });
      const hasPendingRecipe = plan.some((item) => item.kind === "recipe");
      const needsFullWorkerSync =
        !state.datasetInfo ||
        plan.some((item) => ["workspace", "dataset_csv", "dataset_geojson", "recipe"].includes(item.kind));
      const summaryMsgs = [];

      for (const item of plan) {
        if (item.kind === "workspace") {
          await this.importWorkspace(item.parsed, { fileName: item.fileName, skipSync: true });
          summaryMsgs.push(`Projekt: ${item.fileName}`);
        } else if (item.kind === "dataset_csv" || item.kind === "dataset_geojson") {
          const parsed = item.parsed;
          const { profile, guesses } = inferColumnProfile(parsed.headers, parsed.columns);
          await this.onDatasetParsed(item.fileName, parsed, profile, guesses, {
            skipSync: true,
            hasPendingRecipe,
          });
          summaryMsgs.push(`Adat: ${item.fileName} (N=${parsed.nRows})`);
        } else if (item.kind === "recipe") {
          await this.importRecipe(item.parsed, {
            fileName: item.fileName,
            skipSync: true,
            preserveSnapshots: true,
          });
          summaryMsgs.push(`Recept: ${item.parsed.recipe_meta?.title || item.fileName}`);
        } else if (item.kind === "osm_bundle") {
          await this.importOsmBundle(item.parsed, {
            fileName: item.fileName,
            skipIngest: needsFullWorkerSync,
            skipRender: true,
          });
          summaryMsgs.push(`OSM csomag: ${item.fileName}`);
        } else if (item.kind === "target_geojson") {
          const tid = makeUniqueTargetId(
            item.fileName.replace(/\.[^.]+$/, ""),
            state.recipe.spatial_targets
          );
          const { geometry_handling, inside_is_zero } = inferGeojsonTargetDefaults(item.data);
          state.recipe.spatial_targets.push({
            id: tid,
            name: `GeoJSON: ${item.fileName}`,
            source: "geojson",
            excluded_osm_ids: [],
            geometry_handling,
            inside_is_zero,
            distance_modes: "euclidean",
            routing_profile: "walk",
            max_snap_distance_m: 300,
            bbox_buffer_m: 800,
          });
          if (needsFullWorkerSync) {
            state.rawGeojsonTargets[tid] = item.data;
          } else {
            try {
              await this.ingestSingleGeojsonTarget(tid, item.data);
            } catch (ingestErr) {
              const rmIdx = state.recipe.spatial_targets.findIndex((t) => t.id === tid);
              if (rmIdx >= 0) state.recipe.spatial_targets.splice(rmIdx, 1);
              delete state.rawGeojsonTargets[tid];
              throw ingestErr;
            }
          }
          summaryMsgs.push(`GeoJSON célpont (${tid}): ${item.fileName}`);
        }
      }

      if (state.columns && needsFullWorkerSync) {
        await this.syncDatasetToWorker();
      } else {
        if (!state.columns && state.activeStage > 1) state.activeStage = 1;
        this.render();
      }
      if (summaryMsgs.length) {
        setLog(`Betöltve (${summaryMsgs.join(" | ")})`);
        showToast(`Betöltve: ${summaryMsgs.join(", ")}`, "pass");
      }
    } catch (err) {
      setLog(`Betöltési hiba: ${err.message}`);
      showToast(`Fájl betöltési hiba: ${err.message}`, "error");
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
  const en = getLang() === "en";
  const s2 = state.stage2Res;
  const s3 = state.stage3Res;
  const s4 = state.stage4Res;
  const s5 = state.stage5Res;
  const tau = state.recipe.guards_cfg?.sparse_min_cell ?? 5;

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

  setPill("g1Pill", s3 ? (g1Warn ? "warn" : "pass") : "", s3 ? (g1Warn ? (en ? "G1 ⚠ Confound/VIF" : "G1 ⚠ Konfound/VIF") : "G1 ✓ VIF/RESET") : (en ? "G1 Diagnostics" : "G1 Diagnosztika"));
  setPill("g2Pill", s3 ? (g2Count > 0 ? "warn" : "pass") : "", s3 ? (g2Count > 0 ? (en ? `G2 ⚡ ${g2Count} merges (τ=${tau})` : `G2 ⚡ ${g2Count} összevonás (τ=${tau})`) : (en ? `G2 ✓ Cell ≥${tau}` : `G2 ✓ Sűrűség ≥${tau}`)) : (en ? `G2 Sparse (τ=${tau})` : `G2 Ritka (τ=${tau})`));
  setPill("g3Pill", g3Atten !== null ? (Math.abs(g3Atten) > (state.recipe.guards_cfg?.attenuation_warn_pct ?? 50) ? "warn" : "pass") : "", g3Atten !== null ? `G3 ${g3Atten}% atten.` : "G3 Gelbach/Step");
  setPill("g4Pill", g4Heck ? (g4Heck.activated && g4Heck.selection_bias_detected ? "warn" : "pass") : "", g4Heck?.activated ? `G4 λ p=${g4Heck.imr_p}` : "G4 Heckman/SE");
  setPill("g5Pill", s4?.ok ? (g5Warn ? "warn" : "pass") : "", s4?.ok ? (g5Warn ? "G5 ⚠ Co-loc W" : `G5 ✓ ${s4.anselin?.recommended_model || "OK"}`) : (en ? "G5 Spatial W" : "G5 Térbeli W"));
  setPill("g6Pill", s5?.ok ? (g6Reg > 0 ? "warn" : "pass") : "", s5?.ok ? (g6Reg > 0 ? `G6 🛡️ Ridge (${g6Reg})` : (en ? "G6 ✓ GWR Stable" : "G6 ✓ GWR Stabil")) : "G6 GWR Ridge");
}

function renderAdvancedDrawerBody() {
  const body = document.getElementById("advancedDrawerBody");
  if (!body) return;
  const r = state.recipe;
  const grd = r.guards_cfg || {};
  const ds = r.dataset_schema || {};
  const ySpec = ds.y_spec || {};
  const sel = ds.selection_spec || {};
  const sp = r.spatial_engine || {};
  const gwr = sp.gwr || {};
  const headers = state.headers || [];

  body.innerHTML = `
    <div style="font-weight:700;margin-bottom:6px;">Guard #1–#4 &amp; Mintazárolási Küszöbök</div>
    <div class="grid-2" style="margin-bottom:10px;">
      <div class="field-row">
        <label for="drvSparseTau">G2 Ritka cella min. N (τ)</label>
        <input type="number" id="drvSparseTau" value="${Number(grd.sparse_min_cell ?? 5)}" min="0" max="200" />
      </div>
      <div class="field-row">
        <label for="drvLevWarn">G1 HC3 Leverage riasztás</label>
        <input type="number" id="drvLevWarn" value="${Number(grd.hc3_leverage_warn ?? 0.85)}" min="0.1" max="0.999" step="0.05" />
      </div>
      <div class="field-row">
        <label for="drvAttenPct">G3 Gelbach attenuáció riasztás (%)</label>
        <input type="number" id="drvAttenPct" value="${Number(grd.attenuation_warn_pct ?? 50)}" min="5" max="100" />
      </div>
      <div class="field-row">
        <label style="display:flex;align-items:center;gap:6px;padding-top:16px;">
          <input type="checkbox" id="drvCommonLock" ${grd.common_sample_lock !== false ? "checked" : ""} /> Közös minta zárolása (N<sub>common</sub>)
        </label>
      </div>
    </div>

    <div style="font-weight:700;margin-bottom:6px;">Függő Változó (Box–Cox / log1p) &amp; Heckman (G4)</div>
    <div class="grid-2" style="margin-bottom:10px;">
      <div class="field-row">
        <label for="drvBcLambda">Box–Cox fix λ (üres = Auto MLE)</label>
        <input type="number" id="drvBcLambda" step="0.05" min="-5" max="5" value="${ySpec.boxcox_lambda != null ? ySpec.boxcox_lambda : ""}" placeholder="Auto MLE" />
      </div>
      <div class="field-row">
        <label for="drvLog1pShift">log1p shift paraméter</label>
        <input type="number" id="drvLog1pShift" min="0" step="1" value="${Number(ySpec.log1p_shift ?? 1.0)}" />
      </div>
      <div class="field-row">
        <label style="display:flex;align-items:center;gap:6px;">
          <input type="checkbox" id="drvHeckEnabled" ${sel.enabled !== false ? "checked" : ""} /> Heckman (1979) aktív
        </label>
      </div>
      <div class="field-row">
        <label for="drvHeckExclCol">Kizárási jelzőoszlop</label>
        <select id="drvHeckExclCol">
          <option value="">— automatikus (hiányzó koordináták) —</option>
          ${headers.map((h) => `<option value="${escapeHtml(h)}" ${sel.excluded_flag_col === h ? "selected" : ""}>${escapeHtml(h)}</option>`).join("")}
        </select>
      </div>
      <div class="field-row">
        <label for="drvHeckMinExcl">Min. kizárt minta (Heckman)</label>
        <input type="number" id="drvHeckMinExcl" value="${Number(grd.heckman_min_excluded ?? 10)}" min="1" max="500" />
      </div>
      <div class="field-row">
        <label for="drvHeckMinAnalytic">Min. analitikus N (Heckman)</label>
        <input type="number" id="drvHeckMinAnalytic" value="${Number(grd.heckman_min_analytic ?? 30)}" min="10" max="1000" />
      </div>
    </div>

    <div style="font-weight:700;margin-bottom:6px;">Térökonometria (G5) &amp; GWR / MGWR (G6)</div>
    <div class="grid-2">
      <div class="field-row">
        <label for="drvKrDraws">Krinsky–Robb szimulációk (draws)</label>
        <input type="number" id="drvKrDraws" value="${Number(sp.krinsky_robb_draws ?? 1000)}" min="100" max="5000" step="100" />
      </div>
      <div class="field-row">
        <label for="drvGwrFloor">G6 Minimum sávszélesség (b<sub>min</sub>)</label>
        <input type="number" id="drvGwrFloor" value="${Number(gwr.b_min_floor ?? 15)}" min="10" max="500" />
      </div>
      <div class="field-row">
        <label for="drvGwrPrev">G6 Dummy prevalencia küszöb (0–0.5)</label>
        <input type="number" id="drvGwrPrev" value="${Number(gwr.prevalence_min_pct ?? 0.10)}" min="0" max="0.5" step="0.05" />
      </div>
      <div class="field-row">
        <label for="drvGwrKappa">G6 Szelektív Ridge κ</label>
        <input type="number" id="drvGwrKappa" value="${Number(gwr.kappa_thresh ?? 1000)}" min="1" max="100000" step="50" />
      </div>
    </div>
  `;
}

function toggleAdvancedDrawer(openBool) {
  const drawer = document.getElementById("advancedDrawer");
  if (!drawer) return;
  const nextOpen = openBool === undefined ? drawer.hidden : Boolean(openBool);
  if (nextOpen) {
    renderAdvancedDrawerBody();
    drawer.hidden = false;
  } else {
    drawer.hidden = true;
  }
  document.querySelectorAll(".guard-pill").forEach((b) => b.setAttribute("aria-expanded", nextOpen ? "true" : "false"));
}

function applyTheme(isDark) {
  document.body.classList.toggle("dark-theme", Boolean(isDark));
  try {
    localStorage.setItem("geospec_theme", isDark ? "dark" : "light");
  } catch {}
  mapInspector.setTheme(Boolean(isDark));
}

const ALLOWED_STUDIES = new Set(["kobanya", "leopoldstadt"]);

async function autoLoadStudyFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const study = String(params.get("study") || "").trim().toLowerCase();
  if (!ALLOWED_STUDIES.has(study)) return;
  const profileRaw = String(params.get("profile") || "best").trim().toLowerCase();
  const recipeSuffix = profileRaw === "replication" ? "paper_replication_recipe.json" : "recipe.json";
  const csvName = `${study}_dataset.csv`;
  const recipeName = `${study}_${recipeSuffix}`;
  try {
    setLog(`Kutatási minta automatikus betöltése (${study})...`);
    const [csvRes, recRes] = await Promise.all([
      fetch(`../recipes/${csvName}`, { cache: "no-store" }),
      fetch(`../recipes/${recipeName}`, { cache: "no-store" }),
    ]);
    if (!csvRes.ok || !recRes.ok) {
      throw new Error(`HTTP hiba (CSV: ${csvRes.status}, Recept: ${recRes.status})`);
    }
    const [csvText, recipeText] = await Promise.all([csvRes.text(), recRes.text()]);
    const csvFile = new File([csvText], csvName, { type: "text/csv" });
    const recipeFile = new File([recipeText], recipeName, { type: "application/json" });
    await actions.handleIncomingFiles([csvFile, recipeFile]);
    const title = state.recipe?.recipe_meta?.title || study;
    showToast(`${title} betöltve — indítsd a „▶ Teljes Pipeline Futtatása (1–6)” gombbal.`, "info", 6500);
  } catch (err) {
    setLog(`Automatikus minta-betöltési hiba (${study}): ${err.message}`);
    showToast(`Automatikus minta-betöltés sikertelen: ${err.message}`, "error");
  }
}

window.addEventListener("DOMContentLoaded", () => {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("./sw.js").catch(() => {});
  }

  initLang();
  applyI18nDOM();

  let initialDark = false;
  try {
    initialDark = localStorage.getItem("geospec_theme") === "dark";
  } catch {
    initialDark = false;
  }
  document.body.classList.toggle("dark-theme", initialDark);

  pruneExpiredOSM().catch(() => {});

  mapInspector.init();
  mapInspector.setTheme(initialDark);
  mapInspector.renderPointInspector(null, null);
  dag
    .init()
    .then(() => autoLoadStudyFromUrl())
    .catch((err) => updateBootIndicator("error", err.message));

  document.getElementById("btnRestartWorker")?.addEventListener("click", async () => {
    try {
      await dag.restart();
      showToast("Pyodide WASM kernel sikeresen újraindítva.", "pass");
      if (state.columns) await actions.syncDatasetToWorker();
    } catch (err) {
      showToast(`WASM újraindítás sikertelen: ${err.message}`, "error");
    }
  });

  document.querySelectorAll(".stage-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.activeStage = Number(btn.getAttribute("data-stage"));
      actions.render();
    });
  });

  document.getElementById("btnRunAll")?.addEventListener("click", () => actions.runFullPipeline());

  const fileTopLoadAny = document.getElementById("fileTopLoadAny");
  document.getElementById("btnTopLoadAny")?.addEventListener("click", () => fileTopLoadAny?.click());
  fileTopLoadAny?.addEventListener("change", (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = "";
    if (files.length) actions.handleIncomingFiles(files);
  });

  // Save Dropdown Menu
  const btnSaveMenu = document.getElementById("btnTopSaveMenu");
  const saveMenu = document.getElementById("saveDropdownMenu");
  btnSaveMenu?.addEventListener("click", (e) => {
    e.stopPropagation();
    const opening = saveMenu.hidden;
    saveMenu.hidden = !opening;
    btnSaveMenu.setAttribute("aria-expanded", opening ? "true" : "false");
  });
  document.addEventListener("click", (e) => {
    if (saveMenu && !saveMenu.hidden && !e.target.closest(".dropdown-wrap")) {
      saveMenu.hidden = true;
      btnSaveMenu?.setAttribute("aria-expanded", "false");
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (saveMenu && !saveMenu.hidden) {
        saveMenu.hidden = true;
        btnSaveMenu?.setAttribute("aria-expanded", "false");
        btnSaveMenu?.focus();
      }
      const drawer = document.getElementById("advancedDrawer");
      if (drawer && !drawer.hidden) {
        toggleAdvancedDrawer(false);
      }
    }
  });
  document.getElementById("menuSaveRecipe")?.addEventListener("click", () => { saveMenu.hidden = true; btnSaveMenu?.setAttribute("aria-expanded", "false"); actions.exportRecipe(); });
  document.getElementById("menuSaveProject")?.addEventListener("click", () => { saveMenu.hidden = true; btnSaveMenu?.setAttribute("aria-expanded", "false"); actions.exportWorkspace(); });
  document.getElementById("menuSaveOsmBundle")?.addEventListener("click", () => { saveMenu.hidden = true; btnSaveMenu?.setAttribute("aria-expanded", "false"); actions.exportOsmBundle(); });
  document.getElementById("menuSaveLatexMain")?.addEventListener("click", () => { saveMenu.hidden = true; btnSaveMenu?.setAttribute("aria-expanded", "false"); actions.copyLatexTable(); });
  document.getElementById("menuSaveGeojson")?.addEventListener("click", () => { saveMenu.hidden = true; btnSaveMenu?.setAttribute("aria-expanded", "false"); actions.exportGeojsonOnly(); });
  document.getElementById("menuSaveZip")?.addEventListener("click", () => { saveMenu.hidden = true; btnSaveMenu?.setAttribute("aria-expanded", "false"); actions.exportBundle("zip"); });

  // HU | EN Language toggle
  document.getElementById("btnLangToggle")?.addEventListener("click", () => {
    const next = getLang() === "hu" ? "en" : "hu";
    setLang(next);
    applyI18nDOM();
    updateBootIndicator(state.wasmStatus, state.wasmRawMessage);
    renderGuardStrip();
    actions.render();
    restyleMap();
    const drawer = document.getElementById("advancedDrawer");
    if (drawer && !drawer.hidden) renderAdvancedDrawerBody();
    showToast(next === "en" ? "Language switched to English" : "Nyelv átváltva magyarra", "info");
  });

  document.getElementById("btnThemeToggle")?.addEventListener("click", () => {
    const nextDark = !document.body.classList.contains("dark-theme");
    applyTheme(nextDark);
    if (state.activeStage >= 2 && state.activeStage <= 5) {
      actions.render();
    }
  });

  // Guard & Advanced Drawer
  document.querySelectorAll(".guard-pill").forEach((pill) => {
    pill.addEventListener("click", () => toggleAdvancedDrawer(true));
  });
  document.getElementById("btnCloseAdvancedDrawer")?.addEventListener("click", () => toggleAdvancedDrawer(false));

  document.getElementById("btnPresetBestPractice")?.addEventListener("click", () => {
    state.recipe.guards_cfg.sparse_min_cell = 5;
    state.recipe.guards_cfg.common_sample_lock = true;
    state.recipe.spatial_engine.se_mode = "qml";
    state.recipe.spatial_engine.lisa_fdr = true;
    state.recipe.spatial_engine.gwr.prevalence_min_pct = 0.10;
    state.recipe.spatial_engine.gwr.b_min_floor = 15;
    state.recipe.spatial_engine.gwr.kappa_thresh = 1000;
    if (state.recipe.spatial_engine.weights?.[0]?.type === "knn") {
      state.recipe.spatial_engine.weights[0].dedup_colocated = true;
    }
    renderAdvancedDrawerBody();
    actions.invalidateFromStage(1);
    showToast("Best-Practice profil aktiválva (QML SE, τ=5, BH-FDR, parcel dedup).", "pass");
  });

  document.getElementById("btnPresetReplication")?.addEventListener("click", () => {
    state.recipe.guards_cfg.sparse_min_cell = 0;
    state.recipe.guards_cfg.common_sample_lock = true;
    state.recipe.default_cov_type = "HC3";
    state.covType = "HC3";
    state.recipe.spatial_engine.se_mode = "ml";
    state.recipe.spatial_engine.lisa_fdr = false;
    state.recipe.spatial_engine.gwr.prevalence_min_pct = 0;
    if (state.recipe.spatial_engine.weights?.[0]?.type === "knn") {
      state.recipe.spatial_engine.weights[0].dedup_colocated = false;
    }
    renderAdvancedDrawerBody();
    actions.invalidateFromStage(1);
    showToast("Replication profil aktiválva (klasszikus HC3/ML replikációs alapok).", "info");
  });

  document.getElementById("btnApplyDrawerConfig")?.addEventListener("click", () => {
    const grd = state.recipe.guards_cfg;
    const ds = state.recipe.dataset_schema;
    const sp = state.recipe.spatial_engine;
    const gwr = sp.gwr;
    grd.sparse_min_cell = Math.max(0, Math.round(Number(document.getElementById("drvSparseTau")?.value ?? 5)));
    grd.hc3_leverage_warn = Math.min(0.999, Math.max(0.05, Number(document.getElementById("drvLevWarn")?.value || 0.85)));
    grd.attenuation_warn_pct = Math.max(5, Number(document.getElementById("drvAttenPct")?.value || 50));
    grd.common_sample_lock = Boolean(document.getElementById("drvCommonLock")?.checked);
    const bcRaw = document.getElementById("drvBcLambda")?.value;
    ds.y_spec.boxcox_lambda = bcRaw === "" || bcRaw == null ? null : Number(bcRaw);
    ds.y_spec.log1p_shift = Math.max(0, Number(document.getElementById("drvLog1pShift")?.value || 1.0));
    ds.selection_spec.enabled = Boolean(document.getElementById("drvHeckEnabled")?.checked);
    ds.selection_spec.excluded_flag_col = document.getElementById("drvHeckExclCol")?.value || "";
    grd.heckman_min_excluded = Math.max(1, Math.round(Number(document.getElementById("drvHeckMinExcl")?.value || 10)));
    grd.heckman_min_analytic = Math.max(10, Math.round(Number(document.getElementById("drvHeckMinAnalytic")?.value || 30)));
    sp.krinsky_robb_draws = Math.max(100, Math.round(Number(document.getElementById("drvKrDraws")?.value || 1000)));
    gwr.b_min_floor = Math.max(10, Math.round(Number(document.getElementById("drvGwrFloor")?.value || 15)));
    gwr.prevalence_min_pct = Math.max(0, Math.min(0.5, Number(document.getElementById("drvGwrPrev")?.value ?? 0.10)));
    gwr.kappa_thresh = Math.max(1, Number(document.getElementById("drvGwrKappa")?.value || 1000));
    actions.invalidateFromStage(2);
    actions.invalidateFromStage(3);
    toggleAdvancedDrawer(false);
    actions.render();
    showToast("Guard és módszertani beállítások alkalmazva.", "pass");
  });

  window.addEventListener("dragover", (e) => {
    if (Array.from(e.dataTransfer?.types || []).includes("Files")) {
      e.preventDefault();
    }
  });
  window.addEventListener("drop", (e) => {
    if (e.dataTransfer?.files?.length) {
      e.preventDefault();
      actions.handleIncomingFiles(e.dataTransfer.files);
    }
  });

  document.getElementById("selMapLayer")?.addEventListener("change", () => {
    actions.render();
    restyleMap();
  });

  document.getElementById("selGwrVarLayer")?.addEventListener("change", () => {
    restyleMap();
  });

  document.getElementById("selGwrMaskMode")?.addEventListener("change", () => {
    restyleMap();
  });

  document.getElementById("selRouteTarget")?.addEventListener("change", () => {
    restyleMap();
    if (mapInspector.selectedObsIdx !== null && mapInspector.selectedObsIdx !== undefined) {
      actions.inspectObservation(mapInspector.selectedObsIdx);
    }
  });

  actions.render();
});
