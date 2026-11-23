/* GeoSpec Main Controller — Connects 5-Stage Workbench, 6-Guard Rail,
 * MapInspector, Overpass/Postpass Client, Offline Bundle & Pyodide WASM Worker.
 */
import { MapInspector } from "./components/map_inspector.js";
import { renderStage1 } from "./components/stage1_data_osm.js";
import { renderStage2 } from "./components/stage2_composition.js";
import { patchStage3ResultsOnly, renderStage3 } from "./components/stage3_model_composer.js";
import { renderStage4, renderStage5 } from "./components/stage4_5_spatial_gwr.js";
import { buildOsmBundle, parseOsmBundle } from "./gis/osm_idb_cache.js";
import { fetchStreetGraphOSM, fetchTargetOSM } from "./gis/overpass_client.js";
import { DagClient } from "./state/dag_client.js";
import {
  buildWorkspaceBundle,
  createDefaultRecipe,
  inferColumnProfile,
  inferGeojsonTargetDefaults,
  makeUniqueTargetId,
  parseCSVText,
  parseGeoJSONDataset,
  parseWorkspaceBundle,
  prepareIncomingBatch,
  readTextWithEncodingFallback,
  slugifyTitle,
} from "./state/recipe_store.js";
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
  targetPreviews: {},     // targetId -> { preview_geojson, feature_index }
  rawOsmTargets: {},      // targetId -> raw Overpass/converted payload
  rawGeojsonTargets: {},  // targetId -> GeoJSON FeatureCollection
  rawStreetGraphs: {},    // profile -> raw street graph payload
  osmFilterQuery: {},     // targetId -> filter string
  stage1Res: null,
  stage2Res: null,
  stage3Res: null,
  stage4Res: null,
  stage5Res: null,
  activeSpecId: "spec_a",
  covType: "HC3",
  showPct: false,
  stage3AdKey: "",
  stage3DiagModelId: "",
  stage4ImpactKind: "sar",
  pipelineRunning: false,
  wasmStatus: "booting",
  wasmMessage: "WASM indítása...",
  logLine: "Várakozás saját CSV / GeoJSON adatbázis, recept vagy .geospec.json projekt betöltésére.",
};

let debounceTimer = null;
let datasetSeq = 0;
let s1Seq = 0;
let s2Seq = 0;
let s3Seq = 0;
let s4Seq = 0;
let s5Seq = 0;

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

function parseGwrColIdx(rawVal) {
  if (rawVal === undefined || rawVal === null || rawVal === "") return 1;
  const n = Number(rawVal);
  return Number.isFinite(n) && n >= 0 ? n : 1;
}

function restyleMap() {
  const layerMode = document.getElementById("selMapLayer")?.value;
  const gwrColIdx = parseGwrColIdx(document.getElementById("selGwrVarLayer")?.value);
  const gwrMaskMode = document.getElementById("selGwrMaskMode")?.value || "fdr";
  mapInspector.restyleObservations({
    stage3: state.stage3Res,
    stage4: state.stage4Res,
    stage5: state.stage5Res,
    layerMode,
    gwrColIdx,
    gwrMaskMode,
    activeModelId: state.activeSpecId,
  });
}

function resetAnalysisResults() {
  state.stage1Res = null;
  state.stage2Res = null;
  state.stage3Res = null;
  state.stage4Res = null;
  state.stage5Res = null;
  state.stage3AdKey = "";
  state.stage3DiagModelId = "";
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

  render() {
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
  },

  async onDatasetParsed(fileName, parsed, colProfile, guesses, { skipSync = false, hasPendingRecipe = false } = {}) {
    datasetSeq++;
    s1Seq++; s2Seq++; s3Seq++; s4Seq++; s5Seq++;
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
    mapInspector.clearTargets();
    mapInspector.clearRoute();

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
      if (state.columns) {
        setLog("Add meg az érvényes szélességi (lat) és hosszúsági (lon) oszlopokat az 1B szekcióban, majd kattints a 'Séma Rögzítése' gombra.");
      }
      this.render();
      return;
    }
    const mySeq = ++datasetSeq;
    s1Seq++; s2Seq++; s3Seq++; s4Seq++; s5Seq++;
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

      // Re-ingest cached GeoJSON/OSM targets and street graphs if still attached to current workspace
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

      mapInspector.setObservationsFromDataset(state.columns, ds.lat_col, ds.lon_col, ds.row_id_col, ds.y_spec, true);
      this.refreshMapTargets();
      restyleMap();
      mapInspector.renderPointInspector(null, null);
      setLog(`Adatbázis kész: N=${res.n_rows} (geokódolt=${res.n_geocoded}, vetület=${res.projection.nearest_utm_epsg} / ${res.projection.mode})`);
      this.render();
    } catch (err) {
      if (mySeq !== datasetSeq) return;
      state.datasetInfo = null;
      setLog(`Séma hiba: ${err.message}`);
      alert(`Séma hiba: ${err.message}`);
      this.render();
    }
  },

  async ingestSingleGeojsonTarget(targetId, geojsonObj) {
    const myDsSeq = datasetSeq;
    state.rawGeojsonTargets[targetId] = geojsonObj;
    delete state.rawOsmTargets[targetId];
    state.stage1Res = null;
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

  async fetchAllTargetsOSM() {
    if (!state.datasetInfo) return alert("Előbb rögzítsd a dataset sémát!");
    const myDsSeq = datasetSeq;
    await dag.waitUntilReady();
    if (myDsSeq !== datasetSeq) return false;
    const targets = state.recipe.spatial_targets || [];
    const osmTargets = targets.filter((t) => t.source === "osm_preset" || t.source === "osm_custom_tag");
    const routingTargets = targets.filter((t) => t.distance_modes === "network_dijkstra" || t.distance_modes === "both");
    const needProfiles = new Set(routingTargets.map((t) => t.routing_profile || "walk"));
    const streetBufM = Math.max(800, ...targets.map((t) => Number(t.bbox_buffer_m ?? 800) || 800));

    try {
      for (const tgt of osmTargets) {
        if (myDsSeq !== datasetSeq) return false;
        setLog(`OSM lekérés: ${tgt.name}...`);
        const { data, fromCache } = await fetchTargetOSM(tgt, state.datasetInfo.bbox);
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
        if (state.rawStreetGraphs[prof] && !osmTargets.length) continue;
        setLog(`Utcahálózat lekérése CSR Dijkstra útvonalhoz (${prof}, buffer=${streetBufM}m)...`);
        const { data, fromCache } = await fetchStreetGraphOSM(state.datasetInfo.bbox, streetBufM, prof);
        if (myDsSeq !== datasetSeq) return false;
        state.rawStreetGraphs[prof] = data;
        const gRes = await dag.call("ingest_street_graph", { profile: prof, graph_json: data });
        if (myDsSeq !== datasetSeq) return false;
        setLog(`Utcahálózat (${prof}) kész: ${gRes.n_nodes} csúcs, ${gRes.n_edges} él (${fromCache ? "cache" : "Overpass/Postpass"}).`);
      }
      this.render();
      return true;
    } catch (err) {
      if (myDsSeq !== datasetSeq) return false;
      setLog(`OSM lekérési hiba: ${err.message}`);
      alert(`OSM API hiba: ${err.message}`);
      return false;
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
    resetAnalysisResults();
    renderGuardStrip();
    setLog(`OSM elem (${targetId}:${uid}) ${idx >= 0 ? "bekapcsolva" : "kizárva"} a távolságszámításból (újra-számítás szükséges).`);
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
    resetAnalysisResults();
    renderGuardStrip();
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
    resetAnalysisResults();
    this.refreshMapTargets();
    renderGuardStrip();
    this.render();
  },

  async exportOsmBundle() {
    const hasTargets = Object.keys(state.rawOsmTargets).length || Object.keys(state.rawGeojsonTargets).length || Object.keys(state.rawStreetGraphs).length;
    if (!hasTargets) {
      return alert("Nincs exportálható OSM/GeoJSON célpont vagy utcahálózat — előbb töltsd le a célpontokat az 1. fázisban!");
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
    setLog(
      `Offline OSM pillanatfelvétel betöltve (${Object.keys(parsedBundle.targets || {}).length} célpont, ${Object.keys(parsedBundle.street_graphs || {}).length} úthálózat).`
    );
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
            .filter((t) => t.distance_modes === "network_dijkstra" || t.distance_modes === "both")
            .map((t) => t.routing_profile || "walk")
        )
      );
      const missingProfiles = needProfiles.filter((p) => !state.rawStreetGraphs[p]);
      if (missingProfiles.length) {
        const streetBufM = Math.max(800, ...state.recipe.spatial_targets.map((t) => Number(t.bbox_buffer_m ?? 800) || 800));
        for (const prof of missingProfiles) {
          setLog(`Utcahálózat lekérése (${prof}, buffer=${streetBufM}m)...`);
          const { data } = await fetchStreetGraphOSM(state.datasetInfo.bbox, streetBufM, prof);
          if (mySeq !== s1Seq) return false;
          state.rawStreetGraphs[prof] = data;
          await dag.call("ingest_street_graph", { profile: prof, graph_json: data });
        }
      }

      setLog("1. Fázis: Euklideszi és CSR Dijkstra távolságok számítása...");
      const res = await dag.call("run_stage_1", { recipe: state.recipe, run_id: dag.nextRunId() });
      if (mySeq !== s1Seq) return false;
      state.stage1Res = res;
      setLog(`1. Fázis kész (${res.summary.length} célpont távolságmátrixai frissítve).`);
      this.render();
      return true;
    } catch (err) {
      if (err.name === "CancelledError" || mySeq !== s1Seq) return false;
      setLog(`Stage 1 hiba: ${err.message}`);
      alert(err.message);
      return false;
    }
  },

  async runStage2() {
    if (!state.datasetInfo) return false;
    const mySeq = ++s2Seq;
    try {
      await dag.waitUntilReady();
      if (mySeq !== s2Seq) return false;
      setLog("2. Fázis: Minta-vízesés, leíró statisztika, Step 0..K & Gelbach (2016) dekompozíció...");
      const res = await dag.call("run_stage_2", { recipe: state.recipe, run_id: dag.nextRunId() });
      if (mySeq !== s2Seq) return false;
      state.stage2Res = res;
      setLog("2. Fázis kész.");
      this.render();
      return true;
    } catch (err) {
      if (err.name === "CancelledError" || mySeq !== s2Seq) return false;
      setLog(`Stage 2 hiba: ${err.message}`);
      alert(err.message);
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
      const res = await dag.call("run_stage_3", {
        recipe: state.recipe,
        cov_type: state.covType,
        run_id: dag.nextRunId(),
      });
      if (mySeq !== s3Seq) return false;
      state.stage3Res = res;
      if (!res.ok) {
        setLog(`3. Fázis figyelmeztetés: ${res.reason || "Ismeretlen hiba"}`);
        if (!patchResultsOnly) alert(res.reason || "3. Fázis nem futtatható.");
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
      if (!patchResultsOnly) alert(err.message);
      return false;
    }
  },

  runStage3Debounced() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => this.runStage3({ patchResultsOnly: true }), 90);
  },

  async generateM1M4Specs() {
    const active = state.recipe.models.find((m) => m.id === state.activeSpecId) || state.recipe.models[0];
    const covs = active ? active.covariates || [] : [];
    const tgts = state.recipe.spatial_targets || [];
    if (!tgts.length) return alert("Előbb vegyél fel legalább 1 térbeli célpontot az 1. fázisban!");
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
      if (res.ok) {
        setLog(`4. Fázis kész (${res.anselin?.verdict || ""}).`);
        restyleMap();
      } else {
        setLog(`Stage 4 figyelmeztetés: ${res.reason || "Nem futtatható"}`);
      }
      this.render();
      return Boolean(res.ok);
    } catch (err) {
      if (err.name === "CancelledError" || mySeq !== s4Seq) return false;
      setLog(`Stage 4 hiba: ${err.message}`);
      alert(err.message);
      return false;
    }
  },

  async runStage5() {
    if (!state.datasetInfo) return false;
    const mySeq = ++s5Seq;
    try {
      await dag.waitUntilReady();
      if (mySeq !== s5Seq) return false;
      setLog("5. Fázis: Szingularitás-védett Adaptív Bisquare GWR optimalizálása...");
      const res = await dag.call("run_stage_5", { recipe: state.recipe, run_id: dag.nextRunId() });
      if (mySeq !== s5Seq) return false;
      state.stage5Res = res;
      if (res.ok) {
        setLog(`5. Fázis kész (b*=${res.optimal_bw}, GWR Adj. R²=${res.gwr_adj_r2}).`);
        restyleMap();
      } else {
        setLog(`GWR figyelmeztetés: ${res.reason}`);
      }
      this.render();
      return Boolean(res.ok);
    } catch (err) {
      if (err.name === "CancelledError" || mySeq !== s5Seq) return false;
      setLog(`Stage 5 hiba: ${err.message}`);
      alert(err.message);
      return false;
    }
  },

  async runFullPipeline() {
    if (!state.datasetInfo || state.pipelineRunning) return;
    state.pipelineRunning = true;
    const btnRunAll = document.getElementById("btnRunAll");
    if (btnRunAll) btnRunAll.disabled = true;
    try {
      setLog("WASM motor ellenőrzése és 5-fázisú pipeline indítása...");
      await dag.waitUntilReady();
      if (!(await this.runStage1())) return;
      if (!(await this.runStage2())) return;
      if (!(await this.runStage3())) return;
      if (!(await this.runStage4())) return;
      if (!(await this.runStage5())) return;
      setLog("Teljes 5-fázisú térökonometriai pipeline befejezve.");
    } catch (err) {
      setLog(`Pipeline megszakadt: ${err.message}`);
    } finally {
      state.pipelineRunning = false;
      if (btnRunAll) btnRunAll.disabled = !state.datasetInfo;
    }
  },

  async inspectObservation(obsIdx) {
    const tgtId = document.getElementById("selRouteTarget")?.value || state.recipe.spatial_targets?.[0]?.id;
    let route = null;
    if (tgtId) {
      try {
        await dag.waitUntilReady();
        route = await dag.call("get_single_route", { obs_idx: obsIdx, target_id: tgtId, recipe: state.recipe });
        mapInspector.showRoute(route);
      } catch {
        route = null;
      }
    }

    const feat = mapInspector.obsFeatures.find((f) => f.properties.obs_idx === obsIdx);
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
    if (state.stage5Res?.ok) {
      const pos = state.stage5Res.valid_indices.indexOf(obsIdx);
      const ci = Math.min(parseGwrColIdx(document.getElementById("selGwrVarLayer")?.value), state.stage5Res.col_names.length - 1);
      if (pos >= 0) {
        gwr_beta = state.stage5Res.betas_native[pos][ci];
        gwr_t = state.stage5Res.t_vals[pos][ci];
      }
    }

    restyleMap();
    mapInspector.renderPointInspector(obsIdx, {
      row_id: feat?.properties?.row_id ?? obsIdx,
      y_raw: feat?.properties?.y_raw ?? null,
      route,
      resid,
      modelLabel,
      lisa,
      gwr_beta,
      gwr_t,
    });
  },

  async copyLatexTable() {
    try {
      await dag.waitUntilReady();
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
    if (!state.datasetInfo) {
      alert("Előbb tölts be egy érvényes adatbázist a .ZIP / GeoJSON exporthoz! (A recept és a projekt már adatbázis nélkül is menthető.)");
      return;
    }
    try {
      await dag.waitUntilReady();
      setLog("Publikációs export csomag összeállítása...");
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

      const bundle = await dag.call("export_bundle", { recipe: state.recipe, osm_bundle });
      const slug = slugifyTitle(state.recipe?.recipe_meta?.title, "geospec");
      if (kind === "geojson") {
        this.downloadText(`${slug}_enriched.geojson`, JSON.stringify(bundle.geojson, null, 2), "application/geo+json");
      } else {
        const bin = Uint8Array.from(atob(bundle.zip_base64), (c) => c.charCodeAt(0));
        const blob = new Blob([bin], { type: "application/zip" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${slug}_publication_bundle_${Date.now()}.zip`;
        a.click();
        URL.revokeObjectURL(url);
      }
      setLog(`Export kész (${kind}).`);
    } catch (err) {
      alert(err.message);
    }
  },

  exportRecipe() {
    const slug = slugifyTitle(state.recipe?.recipe_meta?.title, "experiment");
    const fileName = `${slug}_recipe.json`;
    this.downloadText(fileName, JSON.stringify(state.recipe, null, 2), "application/json");
    setLog(`${fileName} mentve (nyers adatok nélkül).`);
  },

  exportWorkspace() {
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
    this.downloadText(fileName, JSON.stringify(ws, null, 2), "application/json");
    setLog(`Teljes projekt mentve: ${fileName}`);
  },

  async importRecipe(loadedRecipe, { fileName = "", skipSync = false, preserveSnapshots = true } = {}) {
    s1Seq++; s2Seq++; s3Seq++; s4Seq++; s5Seq++;
    const clean = sanitizeRecipe(loadedRecipe, createDefaultRecipe());
    state.recipe = clean;
    if (fileName) state.recipeFileName = fileName;
    state.workspaceFileName = "";
    const modelIds = new Set((clean.models || []).map((m) => m.id));
    const preferredSpec = clean.spatial_engine?.active_model_id || clean.spatial_engine?.gwr?.focal_model_id;
    state.activeSpecId =
      preferredSpec && modelIds.has(preferredSpec)
        ? preferredSpec
        : clean.models?.[0]?.id || "spec_a";

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
        const parsedEmbeddedBundle = parseOsmBundle(loadedRecipe.osm_bundle);
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

  async importWorkspace(rawWorkspace, { fileName = "", skipSync = false } = {}) {
    const parsed = parseWorkspaceBundle(rawWorkspace);
    datasetSeq++;
    s1Seq++; s2Seq++; s3Seq++; s4Seq++; s5Seq++;

    resetAnalysisResults();
    state.targetPreviews = {};
    state.rawOsmTargets = {};
    state.rawGeojsonTargets = {};
    state.rawStreetGraphs = {};
    state.osmFilterQuery = {};
    state.osmBundleFileName = "";
    mapInspector.clearTargets();
    mapInspector.clearRoute();

    state.recipe = parsed.recipe;
    state.workspaceFileName = fileName || "workspace.geospec.json";
    state.recipeFileName = fileName || "workspace.geospec.json";
    state.activeSpecId = parsed.ui_state.activeSpecId;
    state.covType = parsed.ui_state.covType;
    state.showPct = parsed.ui_state.showPct;

    if (parsed.dataset) {
      state.fileName = parsed.dataset.fileName;
      state.headers = parsed.dataset.headers;
      state.columns = parsed.dataset.columns;
      state.nRows = parsed.dataset.nRows;
      const { profile } = inferColumnProfile(state.headers, state.columns);
      state.colProfile = profile;
    } else {
      state.fileName = "";
      state.headers = [];
      state.columns = null;
      state.nRows = 0;
      state.colProfile = {};
      state.datasetInfo = null;
      mapInspector.clearObservations();
    }

    if (parsed.osm_bundle) {
      await this.importOsmBundle(parsed.osm_bundle, {
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
      const plan = prepareIncomingBatch(entries, { hasDataset: Boolean(state.columns) });
      const hasPendingRecipe = plan.some((item) => item.kind === "recipe");
      const needsFullWorkerSync =
        !state.datasetInfo ||
        plan.some((item) => ["workspace", "dataset_csv", "dataset_geojson", "recipe"].includes(item.kind));
      const summaryMsgs = [];

      for (const item of plan) {
        if (item.kind === "workspace") {
          await this.importWorkspace(item.data, { fileName: item.fileName, skipSync: true });
          summaryMsgs.push(`Projekt: ${item.fileName}`);
        } else if (item.kind === "dataset_csv" || item.kind === "dataset_geojson") {
          const parsed =
            item.kind === "dataset_csv"
              ? parseCSVText(item.text)
              : parseGeoJSONDataset(item.data);
          const { profile, guesses } = inferColumnProfile(parsed.headers, parsed.columns);
          await this.onDatasetParsed(item.fileName, parsed, profile, guesses, {
            skipSync: true,
            hasPendingRecipe,
          });
          summaryMsgs.push(`Adat: ${item.fileName} (N=${parsed.nRows})`);
        } else if (item.kind === "recipe") {
          await this.importRecipe(item.data, {
            fileName: item.fileName,
            skipSync: true,
            preserveSnapshots: true,
          });
          summaryMsgs.push(`Recept: ${item.data.recipe_meta?.title || item.fileName}`);
        } else if (item.kind === "osm_bundle") {
          const parsedBundle = parseOsmBundle(item.data);
          await this.importOsmBundle(parsedBundle, {
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
        if (state.columns && !state.datasetInfo) {
          if (!state.logLine.startsWith("Séma hiba:")) {
            setLog(`Betöltve (${summaryMsgs.join(" | ")}) — jelöld ki az érvényes lat/lon oszlopokat az 1B szekcióban!`);
          }
        } else {
          setLog(`Betöltve (${summaryMsgs.join(" | ")})`);
        }
      }
    } catch (err) {
      setLog(`Betöltési hiba: ${err.message}`);
      alert(`Fájl betöltési hiba: ${err.message}`);
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
  setPill("g3Pill", g3Atten !== null ? (Math.abs(g3Atten) > (state.recipe.guards_cfg?.attenuation_warn_pct ?? 50) ? "warn" : "pass") : "", g3Atten !== null ? `G3 ${g3Atten}% atten.` : "G3 Gelbach/Step");
  setPill("g4Pill", g4Heck ? (g4Heck.activated && g4Heck.selection_bias_detected ? "warn" : "pass") : "", g4Heck?.activated ? `G4 λ p=${g4Heck.imr_p}` : "G4 Heckman/SE");
  setPill("g5Pill", s4?.ok ? (g5Warn ? "warn" : "pass") : "", s4?.ok ? (g5Warn ? "G5 ⚠ Co-loc W" : `G5 ✓ ${s4.anselin?.recommended_model || "OK"}`) : "G5 Térbeli W");
  setPill("g6Pill", s5?.ok ? (g6Reg > 0 ? "warn" : "pass") : "", s5?.ok ? (g6Reg > 0 ? `G6 🛡️ Ridge (${g6Reg})` : "G6 ✓ GWR Stabil") : "G6 GWR Ridge");
}

function applyTheme(isDark) {
  document.body.classList.toggle("dark-theme", Boolean(isDark));
  try {
    localStorage.setItem("geospec_theme", isDark ? "dark" : "light");
  } catch {
    // ignore storage restriction
  }
  mapInspector.setTheme(Boolean(isDark));
}

window.addEventListener("DOMContentLoaded", () => {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("./sw.js").catch(() => {});
  }

  let initialDark = false;
  try {
    initialDark = localStorage.getItem("geospec_theme") === "dark";
  } catch {
    initialDark = false;
  }
  document.body.classList.toggle("dark-theme", initialDark);

  mapInspector.init();
  mapInspector.setTheme(initialDark);
  mapInspector.renderPointInspector(null, null);
  dag.init().catch((err) => updateBootIndicator("error", err.message));

  document.querySelectorAll(".stage-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.activeStage = Number(btn.getAttribute("data-stage"));
      actions.render();
    });
  });

  document.getElementById("btnRunAll").addEventListener("click", () => actions.runFullPipeline());

  const fileTopLoadAny = document.getElementById("fileTopLoadAny");
  document.getElementById("btnTopLoadAny")?.addEventListener("click", () => fileTopLoadAny?.click());
  fileTopLoadAny?.addEventListener("change", (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = "";
    if (files.length) actions.handleIncomingFiles(files);
  });

  const fileTopRecipeImport = document.getElementById("fileTopRecipeImport");
  document.getElementById("btnTopRecipeImport")?.addEventListener("click", () => fileTopRecipeImport?.click());
  fileTopRecipeImport?.addEventListener("change", (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = "";
    if (files.length) actions.handleIncomingFiles(files);
  });

  document.getElementById("btnTopRecipeExport")?.addEventListener("click", () => actions.exportRecipe());
  document.getElementById("btnTopProjectExport")?.addEventListener("click", () => actions.exportWorkspace());
  document.getElementById("btnTopZipExport")?.addEventListener("click", () => actions.exportBundle("zip"));
  document.getElementById("btnThemeToggle").addEventListener("click", () => {
    const nextDark = !document.body.classList.contains("dark-theme");
    applyTheme(nextDark);
    if (state.activeStage >= 2 && state.activeStage <= 5) {
      actions.render();
    }
  });

  // Prevent browser from navigating away if files are dropped anywhere in the workbench
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

  document.getElementById("selMapLayer").addEventListener("change", () => {
    actions.render();
    restyleMap();
  });

  document.getElementById("selGwrVarLayer").addEventListener("change", () => {
    restyleMap();
  });

  document.getElementById("selGwrMaskMode")?.addEventListener("change", () => {
    restyleMap();
  });

  document.getElementById("g1Pill").addEventListener("click", () => { state.activeStage = 3; actions.render(); });
  document.getElementById("g2Pill").addEventListener("click", () => { state.activeStage = 3; actions.render(); });
  document.getElementById("g3Pill").addEventListener("click", () => { state.activeStage = 2; actions.render(); });
  document.getElementById("g4Pill").addEventListener("click", () => { state.activeStage = 3; actions.render(); });
  document.getElementById("g5Pill").addEventListener("click", () => { state.activeStage = 4; actions.render(); });
  document.getElementById("g6Pill").addEventListener("click", () => { state.activeStage = 5; actions.render(); });

  actions.render();
});
