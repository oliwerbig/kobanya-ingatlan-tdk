/* Stage 1: Universal File Intake (CSV / GeoJSON / Recipe / OSM Bundle / Workspace),
 * Schema Mapping (with pre-CSV recipe preview/edit), Parametric OSM & Feature Filter
 */
import { clearAllCachedOSM, parseOsmBundle } from "../gis/osm_idb_cache.js";
import { OSM_TARGET_PRESETS, parseTagShorthand } from "../gis/overpass_client.js";
import {
  generateEmptyCSVTemplate,
  inferGeojsonTargetDefaults,
  makeUniqueTargetId,
  readTextWithEncodingFallback,
  validateGeojsonObject,
} from "../state/recipe_store.js";
import { escapeHtml, parseJsonText } from "../utils/sanitize.js";

function renderFeatureListHtml(t, previewInfo, filterQuery) {
  const exclSet = new Set(t.excluded_osm_ids || []);
  const featList = previewInfo?.feature_index || [];
  const q = (filterQuery || "").trim().toLowerCase();
  const matched = q
    ? featList.filter(
        (f) => String(f.name || "").toLowerCase().includes(q) || String(f.uid || "").toLowerCase().includes(q)
      )
    : featList;
  const shown = matched.slice(0, 350);

  if (!shown.length) {
    return `<div class="card-subtitle" style="padding:6px;">Nincs a szűrőnek megfelelő elem (${featList.length} elemből).</div>`;
  }

  return (
    shown
      .map((f) => {
        const isExcl = exclSet.has(f.uid);
        return `<div class="osm-feature-item ${isExcl ? "excluded" : ""}" data-uid="${escapeHtml(f.uid)}">
          <span>${isExcl ? "☐" : "☑"} <strong>${escapeHtml(f.name)}</strong> <span class="mono card-subtitle">(${escapeHtml(f.uid)}, ${escapeHtml(f.geom_type)})</span></span>
          <span class="card-subtitle osm-item-state">${isExcl ? "kizárva" : "aktív"}</span>
        </div>`;
      })
      .join("") +
    (matched.length > 350
      ? `<div class="card-subtitle" style="padding:4px;">… és további ${matched.length - 350} találat (szűkítsd a keresést vagy kattints a térképen).</div>`
      : "")
  );
}

function collectRecipeReferencedCols(recipe) {
  const ds = recipe?.dataset_schema || {};
  const targets = recipe?.spatial_targets || [];
  const models = recipe?.models || [];
  return Array.from(
    new Set(
      [
        ds.lat_col,
        ds.lon_col,
        ds.row_id_col,
        ds.y_spec?.num_col,
        ds.y_spec?.den_col,
        ds.cluster_spec?.col,
        ...(ds.spatial_proxy_cols || []),
        ...(ds.selection_spec?.exclusion_restriction_cols || []),
        ...(ds.selection_spec?.first_stage_covariates || []),
        ...targets.map((t) => t.csv_col),
        ...models.flatMap((m) => (m.covariates || []).map((c) => c.col)),
      ].filter(Boolean)
    )
  );
}

export function renderStage1(container, ctx) {
  const { state, actions } = ctx;
  const ds = state.recipe.dataset_schema;
  const headers = state.headers || [];
  const profile = state.colProfile || {};
  const recipeCols = collectRecipeReferencedCols(state.recipe);

  const numCols = headers.filter((h) => profile[h]?.isNumeric);
  const targets = state.recipe.spatial_targets || [];
  const models = state.recipe.models || [];
  const hasSnapshotTargets = targets.some((t) => state.rawOsmTargets[t.id] || state.rawGeojsonTargets[t.id]);
  const snapshotTargetIds = new Set([
    ...Object.keys(state.rawOsmTargets || {}),
    ...Object.keys(state.rawGeojsonTargets || {}),
  ]);
  const snapshotGraphCount = Object.keys(state.rawStreetGraphs || {}).length;
  const hasLoadedRecipe = Boolean(state.recipeFileName || state.workspaceFileName || targets.length > 0 || models.length > 1);

  const allOpts = (sel) => {
    const pool = headers.length > 0 ? [...headers] : [...recipeCols];
    if (sel && !pool.includes(sel)) {
      pool.unshift(sel);
    }
    const opts = pool
      .map((h) => {
        const missingNote =
          headers.length > 0 && !headers.includes(h)
            ? " (receptből — nincs a CSV-ben)"
            : headers.length === 0
              ? " (receptből)"
              : "";
        return `<option value="${escapeHtml(h)}" ${h === sel ? "selected" : ""}>${escapeHtml(h)}${missingNote}</option>`;
      })
      .join("");
    return `<option value="">— válassz —</option>` + opts;
  };

  const multiOpts = (selectedArr) => {
    const selectedSet = new Set(selectedArr || []);
    const base = headers.length > 0 ? [...numCols] : [...recipeCols];
    for (const s of selectedSet) {
      if (s && !base.includes(s)) base.unshift(s);
    }
    return base
      .map((h) => {
        const note =
          headers.length > 0 && !headers.includes(h)
            ? " (receptből)"
            : headers.length === 0
              ? " (receptből)"
              : "";
        return `<option value="${escapeHtml(h)}" ${selectedSet.has(h) ? "selected" : ""}>${escapeHtml(h)}${note}</option>`;
      })
      .join("");
  };

  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">1A. Saját Adatbázis, Recept &amp; Projekt Betöltése</h3>
          <div class="card-subtitle">100%-ban kliensoldali: tölts be CSV/GeoJSON adatot, <code>_recipe.json</code> receptet, <code>.osm_bundle.json</code> csomagot vagy <code>.geospec.json</code> projektet.</div>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <button class="btn btn-sm" id="btnDlCsvTemplate">📄 CSV Sablon</button>
          <button class="btn btn-sm" id="btnImportRecipeS1" title="Recept (_recipe.json) vagy projekt betöltése">📥 Recept (.json)</button>
          <input type="file" id="fileRecipeS1" accept=".json" multiple style="display:none" />
          <button class="btn btn-sm" id="btnExportRecipeS1" title="Aktuális recept (_recipe.json) mentése">💾 Recept Mentés</button>
          <button class="btn btn-sm" id="btnExportProjectS1" title="Teljes munkamenet mentése (&lt;slug&gt;.geospec.json)">💾 Projekt Mentés</button>
        </div>
      </div>

      <div class="dropzone" id="csvDropzone" tabindex="0" role="button" aria-label="Fájlok kiválasztása vagy áthúzása">
        <div style="font-weight:600;margin-bottom:4px;">📂 Húzd ide vagy válaszd ki a fájlokat (CSV, GeoJSON, Recept, OSM Bundle, .geospec.json)</div>
        <div class="card-subtitle">Egyszerre több fájl is kijelölhető (pl. adat + recept + GeoJSON célpontok) | Elválasztók: <code>,</code> <code>;</code> <code>Tab</code> | UTF-8 &amp; Windows-1250</div>
        <input type="file" id="csvFileInput" accept=".csv,.tsv,.txt,.geojson,.json" multiple style="display:none" />
      </div>

      <div class="workspace-status-grid">
        <div class="workspace-status-box ${headers.length > 0 ? "loaded" : ""}" id="datasetStatusCard">
          ${
            headers.length > 0
              ? `<div><strong>📊 Adatbázis:</strong> ${escapeHtml(state.fileName)}</div>
                 <div class="meta-chips">
                   <span class="diag-chip pass">${state.nRows} sor</span>
                   <span class="diag-chip pass">${headers.length} oszlop</span>
                   ${
                     state.datasetInfo
                       ? `<span class="diag-chip pass">Geokódolt: ${state.datasetInfo.n_geocoded}</span>
                          <span class="diag-chip">Heckman pool: ${state.datasetInfo.n_non_geocoded}</span>
                          <span class="diag-chip">${escapeHtml(state.datasetInfo.projection.nearest_utm_epsg)} (${escapeHtml(state.datasetInfo.projection.mode)})</span>`
                       : `<span class="diag-chip warn">Koordináta-oszlop kijelölése szükséges (1B)</span>`
                   }
                 </div>`
              : `<div><strong>📊 Adatbázis:</strong> Nincs még betöltve</div>
                 <div class="card-subtitle" style="margin-top:4px;">Húzz ide egy <code>.csv</code> / <code>.tsv</code> / Pont <code>.geojson</code> fájlt, vagy nyiss meg egy mentett <code>.geospec.json</code> projektet.</div>`
          }
        </div>

        <div class="workspace-status-box ${hasLoadedRecipe ? "loaded" : ""}" id="recipeStatusCard">
          <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
            <strong>🧪 Recept:</strong>
            <input type="text" id="txtRecipeTitle" value="${escapeHtml(state.recipe.recipe_meta?.title || "")}" placeholder="Vizsgálat címe..." style="flex:1;min-width:160px;padding:3px 6px;font-size:11.5px;font-weight:600;" />
          </div>
          <div class="card-subtitle" style="margin-top:4px;">
            Forrás: <span class="mono">${escapeHtml(state.workspaceFileName || state.recipeFileName || "Alapértelmezett sablon")}</span>
            ${
              state.osmBundleFileName &&
              state.osmBundleFileName !== state.workspaceFileName &&
              state.osmBundleFileName !== state.recipeFileName
                ? ` | OSM csomag: <span class="mono">${escapeHtml(state.osmBundleFileName)}</span>`
                : ""
            }
          </div>
          <div class="meta-chips">
            <span class="diag-chip ${targets.length ? "pass" : ""}">Célpontok: ${targets.length}</span>
            <span class="diag-chip ${models.length > 1 ? "pass" : ""}">Modellek: ${models.length} (${escapeHtml(models.map((m) => m.id).join(", "))})</span>
            <span class="diag-chip ${snapshotTargetIds.size || snapshotGraphCount ? "pass" : ""}">Snapshot: ${snapshotTargetIds.size} réteg, ${snapshotGraphCount} úthálózat</span>
          </div>
        </div>
      </div>

      ${
        headers.length > 0 && state.columns && state.nRows > 0
          ? `<div style="margin-top:10px;">
               <div class="card-subtitle" style="margin-bottom:4px;"><strong>Sorelőnézet</strong> (első ${Math.min(5, state.nRows)} sor a betöltött ${state.nRows} sorból — kódolás és tizedesjel ellenőrzéséhez):</div>
               <div class="table-wrap" style="max-height:170px;overflow:auto;">
                 <table class="academic" style="font-size:11px;">
                   <thead><tr>${headers.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead>
                   <tbody>
                     ${Array.from({ length: Math.min(5, state.nRows) }, (_, rIdx) => `<tr>${headers.map((h) => `<td class="mono">${escapeHtml(state.columns[h]?.[rIdx] ?? "")}</td>`).join("")}</tr>`).join("")}
                   </tbody>
                 </table>
               </div>
             </div>`
          : ""
      }

      ${
        (state.datasetInfo?.warnings || []).length > 0
          ? state.datasetInfo.warnings
              .map((w) => `<div class="alert-box" style="margin-top:8px;margin-bottom:0;">⚠️ ${escapeHtml(w)}</div>`)
              .join("")
          : ""
      }
    </div>

    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">1B. Oszlopszerepek &amp; Függő Változó (y)</h3>
          <div class="card-subtitle">${
            headers.length > 0
              ? "Koordináták, célváltozó-transzformáció és opcionális klaszter / szelekciós / proxy szerepkörök"
              : "Recept oszlopszerepek előnézete és szerkesztése (CSV betöltése előtt is módosítható)"
          }</div>
        </div>
        <button class="btn btn-primary btn-sm" id="btnApplySchema">${headers.length > 0 ? "✓ Séma Rögzítése" : "✓ Recept Séma Mentése"}</button>
      </div>

      <div class="grid-3">
        <div class="field-row">
          <label>Szélesség (lat)</label>
          <select id="mapLatCol">${allOpts(ds.lat_col)}</select>
        </div>
        <div class="field-row">
          <label>Hosszúság (lon)</label>
          <select id="mapLonCol">${allOpts(ds.lon_col)}</select>
        </div>
        <div class="field-row">
          <label>Sorazonosító (row_id)</label>
          <select id="mapIdCol">${allOpts(ds.row_id_col)}</select>
        </div>
      </div>

      <div class="grid-3" style="margin-top:8px;">
        <div class="field-row">
          <label>Függő változó mód</label>
          <select id="mapYMode">
            <option value="column" ${ds.y_spec.mode === "column" ? "selected" : ""}>Önálló oszlop (y)</option>
            <option value="ratio" ${ds.y_spec.mode === "ratio" ? "selected" : ""}>Hányados (számláló / nevező)</option>
          </select>
        </div>
        <div class="field-row">
          <label>${ds.y_spec.mode === "ratio" ? "Számláló (pl. ár)" : "Célváltozó (y)"}</label>
          <select id="mapYNum">${allOpts(ds.y_spec.num_col)}</select>
        </div>
        <div class="field-row" style="${ds.y_spec.mode === "ratio" ? "" : "opacity:0.45;"}">
          <label>Nevező (pl. alapterület)</label>
          <select id="mapYDen" ${ds.y_spec.mode === "ratio" ? "" : "disabled"}>${allOpts(ds.y_spec.den_col)}</select>
        </div>
      </div>

      <div class="grid-3" style="margin-top:8px;">
        <div class="field-row">
          <label>y Transzformáció</label>
          <select id="mapYTrans">
            <option value="log" ${ds.y_spec.transform === "log" ? "selected" : ""}>ln(y) (Fél-logaritmikus)</option>
            <option value="raw" ${ds.y_spec.transform === "raw" ? "selected" : ""}>y (Lineáris nyers)</option>
            <option value="log1p" ${ds.y_spec.transform === "log1p" ? "selected" : ""}>ln(y + shift)</option>
            <option value="boxcox" ${ds.y_spec.transform === "boxcox" ? "selected" : ""}>Box–Cox MLE (λ becsléssel)</option>
          </select>
        </div>
        <div class="field-row">
          <label>Klaszter-oszlop (CR1 SE-hez, opcionális)</label>
          <select id="mapClusterCol">${allOpts(ds.cluster_spec.col)}</select>
        </div>
        <div class="field-row">
          <label>Koordináta-klaszter sugár (m)</label>
          <input type="number" id="mapClusterRadius" value="${escapeHtml(ds.cluster_spec.radius_m || 15)}" min="1" max="500" />
        </div>
      </div>

      <div class="grid-2" style="margin-top:8px;">
        <div class="field-row">
          <label>Térbeli Proxy / CBD oszlopok (Guard #1, több is jelölhető)</label>
          <select id="mapProxyCols" multiple size="2" style="height:46px;">
            ${multiOpts(ds.spatial_proxy_cols)}
          </select>
        </div>
        <div class="field-row">
          <label>Heckman kizárási restrikció Z<sub>excl</sub> (Guard #4, opcionális)</label>
          <select id="mapHeckExcl" multiple size="2" style="height:46px;">
            ${multiOpts(ds.selection_spec.exclusion_restriction_cols)}
          </select>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">1C. Parametrikus Térbeli Célpontok (Bármilyen OSM Entitás / GeoJSON)</h3>
          <div class="card-subtitle">Adj hozzá célpontokat, kérd le az OSM objektumokat (vagy tölts be offline <code>.osm_bundle.json</code> csomagot)</div>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <button class="btn btn-sm" id="btnFetchOSM" ${state.datasetInfo && targets.some((t) => t.source === "osm_preset" || t.source === "osm_custom_tag") ? "" : "disabled"}>🌐 OSM Lekérése</button>
          <button class="btn btn-sm" id="btnImportOsmBundle" title="Korábban mentett offline .osm_bundle.json betöltése">📦 Bundle Betöltés</button>
          <input type="file" id="fileOsmBundle" accept=".json" style="display:none" />
          <button class="btn btn-sm" id="btnExportOsmBundle" ${hasSnapshotTargets || snapshotGraphCount ? "" : "disabled"} title="OSM / GeoJSON rétegek és utcahálózat mentése offline reprodukálhatósághoz">💾 .osm_bundle</button>
          <button class="btn btn-sm" id="btnClearOsmCache" title="IndexedDB OSM cache ürítése">🗑 Cache</button>
          <button class="btn btn-primary btn-sm" id="btnComputeDistances" ${state.datasetInfo && targets.length ? "" : "disabled"}>📐 Távolságok Számítása</button>
        </div>
      </div>

      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;align-items:center;">
        <select id="selOsmPreset" style="min-width:210px;">
          ${OSM_TARGET_PRESETS.map((p) => `<option value="${escapeHtml(p.preset_id)}">${escapeHtml(p.label)}</option>`).join("")}
        </select>
        <button class="btn btn-sm" id="btnAddOsmPreset">+ Sablon Célpont</button>
        <input type="text" id="txtCustomOsmTag" placeholder="Egyedi OSM tag (pl. amenity=school;leisure=park)" style="flex:1;min-width:210px;" />
        <button class="btn btn-sm" id="btnAddCustomTag">+ Egyedi Tag</button>
        <button class="btn btn-sm" id="btnAddGeojsonTarget">+ Saját GeoJSON</button>
        <input type="file" id="fileGeojsonTarget" accept=".geojson,.json" multiple style="display:none" />
        <button class="btn btn-sm" id="btnAddCsvDistTarget" ${numCols.length ? "" : 'disabled title="Előbb tölts be numerikus oszlopokat tartalmazó CSV adatbázist"'}>+ Meglévő CSV Oszlop</button>
      </div>

      <div id="targetListArea">
        ${
          targets.length === 0
            ? `<div class="card-subtitle">Még nincs felvett térbeli célpont. Válassz fentről egy OSM sablont, tölts fel GeoJSON-t, vagy tölts be receptet!</div>`
            : targets
                .map((t, idx) =>
                  renderTargetRow(
                    t,
                    idx,
                    numCols,
                    state.targetPreviews[t.id],
                    state.osmFilterQuery?.[t.id] || "",
                    Boolean(state.rawOsmTargets[t.id] || state.rawGeojsonTargets[t.id])
                  )
                )
                .join("")
        }
      </div>
    </div>

    ${
      state.stage1Res
        ? `<div class="card">
             <div class="card-header">
               <h3 class="card-title">1D. Térbeli Távolság &amp; Hálózati Kanyargósság Összesítő</h3>
               <span class="card-subtitle">A becslések készen állnak a 2–5. fázisokhoz</span>
             </div>
             ${(state.stage1Res.warnings || []).map((w) => `<div class="alert-box">⚠️ ${escapeHtml(w)}</div>`).join("")}
             <div class="table-wrap">
               <table class="academic">
                 <thead><tr>
                   <th>Célpont</th><th class="num">Aktív elem</th><th class="num">Kizárva</th>
                   <th class="num">Medián Eukl. (m)</th><th class="num">Medián Hálózati (m)</th>
                   <th class="num">Medián Δd (m)</th><th class="num">Max Δd (m)</th>
                 </tr></thead>
                 <tbody>
                   ${state.stage1Res.summary
                     .map(
                       (r) => `<tr>
                         <td><strong>${escapeHtml(r.name)}</strong> <span class="card-subtitle mono">(${escapeHtml(r.target_id)})</span></td>
                         <td class="num">${escapeHtml(r.n_active_features)}</td>
                         <td class="num">${escapeHtml(r.n_excluded_features)}</td>
                         <td class="num">${r.median_eucl_m ?? "—"}</td>
                         <td class="num">${r.median_net_m ?? "—"}</td>
                         <td class="num">${r.median_detour_m ?? "—"}</td>
                         <td class="num">${r.max_detour_m ?? "—"}</td>
                       </tr>`
                     )
                     .join("")}
                 </tbody>
               </table>
             </div>
           </div>`
        : ""
    }
  `;

  // Bind Dropzone / Multi-file intake / Template / Recipe / Project actions
  const dropzone = container.querySelector("#csvDropzone");
  const fileInput = container.querySelector("#csvFileInput");
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", (e) => {
    if (!dropzone.contains(e.relatedTarget)) dropzone.classList.remove("dragover");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer?.files?.length) actions.handleIncomingFiles(e.dataTransfer.files);
  });
  fileInput.addEventListener("change", (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = "";
    if (files.length) actions.handleIncomingFiles(files);
  });

  container.querySelector("#btnDlCsvTemplate").addEventListener("click", () => {
    actions.downloadText("geospec_template.csv", generateEmptyCSVTemplate(), "text/csv");
  });
  container.querySelector("#btnImportRecipeS1").addEventListener("click", () => {
    container.querySelector("#fileRecipeS1").click();
  });
  container.querySelector("#fileRecipeS1").addEventListener("change", async (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = "";
    if (files.length) await actions.handleIncomingFiles(files);
  });
  container.querySelector("#btnExportRecipeS1")?.addEventListener("click", () => actions.exportRecipe());
  container.querySelector("#btnExportProjectS1")?.addEventListener("click", () => actions.exportWorkspace());

  container.querySelector("#txtRecipeTitle")?.addEventListener("input", (e) => {
    if (!state.recipe.recipe_meta) state.recipe.recipe_meta = {};
    state.recipe.recipe_meta.title = e.target.value;
  });

  // Bind 1B Schema controls (live updates into ds + explicit Apply/Sync)
  const syncSchemaFromDom = () => {
    ds.lat_col = container.querySelector("#mapLatCol").value;
    ds.lon_col = container.querySelector("#mapLonCol").value;
    ds.row_id_col = container.querySelector("#mapIdCol").value;
    ds.y_spec.mode = container.querySelector("#mapYMode").value;
    ds.y_spec.num_col = container.querySelector("#mapYNum").value;
    ds.y_spec.den_col = container.querySelector("#mapYDen").value;
    ds.y_spec.transform = container.querySelector("#mapYTrans").value;
    ds.cluster_spec.col = container.querySelector("#mapClusterCol").value;
    ds.cluster_spec.radius_m = Number(container.querySelector("#mapClusterRadius").value) || 15;
    ds.spatial_proxy_cols = Array.from(container.querySelector("#mapProxyCols").selectedOptions).map((o) => o.value);
    ds.selection_spec.exclusion_restriction_cols = Array.from(container.querySelector("#mapHeckExcl").selectedOptions).map((o) => o.value);
  };

  container.querySelector("#mapYMode").addEventListener("change", (e) => {
    syncSchemaFromDom();
    ds.y_spec.mode = e.target.value;
    actions.render();
  });

  ["#mapLatCol", "#mapLonCol", "#mapIdCol", "#mapYNum", "#mapYDen", "#mapYTrans", "#mapClusterCol", "#mapClusterRadius", "#mapProxyCols", "#mapHeckExcl"].forEach(
    (sel) => {
      container.querySelector(sel)?.addEventListener("change", syncSchemaFromDom);
    }
  );
  container.querySelector("#mapClusterRadius")?.addEventListener("input", syncSchemaFromDom);

  container.querySelector("#btnApplySchema").addEventListener("click", () => {
    syncSchemaFromDom();
    if (state.columns) {
      actions.syncDatasetToWorker();
    } else {
      actions.setLog("Recept séma frissítve (CSV betöltésekor automatikusan aktiválódik).");
      actions.render();
    }
  });

  // Bind 1C Target Addition controls
  container.querySelector("#btnAddOsmPreset").addEventListener("click", () => {
    const pid = container.querySelector("#selOsmPreset").value;
    const preset = OSM_TARGET_PRESETS.find((p) => p.preset_id === pid);
    if (!preset) return;
    const uniqueId = makeUniqueTargetId(preset.preset_id, targets);
    targets.push({
      id: uniqueId,
      name: preset.label,
      source: "osm_preset",
      preset_id: preset.preset_id,
      osm_filter: preset.osm_filter,
      element_types: preset.element_types,
      excluded_osm_ids: [],
      geometry_handling: preset.geometry_handling,
      inside_is_zero: preset.inside_is_zero,
      distance_modes: preset.preset_id === "transit_stops" ? "both" : "euclidean",
      routing_profile: "walk",
      max_snap_distance_m: 300,
      bbox_buffer_m: 800,
    });
    actions.render();
  });

  container.querySelector("#btnAddCustomTag").addEventListener("click", () => {
    const rawTag = container.querySelector("#txtCustomOsmTag").value.trim();
    if (!rawTag) return;
    const parsed = parseTagShorthand(rawTag);
    if (!parsed.length) return alert("Adj meg érvényes key=value formátumot (pl. leisure=park)!");
    const firstKey = Object.keys(parsed[0])[0];
    const slug = `${firstKey}_${parsed[0][firstKey][0]}`.replace(/[^a-z0-9_]/gi, "_").toLowerCase();
    targets.push({
      id: makeUniqueTargetId(slug, targets),
      name: `OSM: ${rawTag}`,
      source: "osm_custom_tag",
      osm_filter: parsed,
      element_types: ["node", "way", "relation"],
      excluded_osm_ids: [],
      geometry_handling: "polygon_boundary",
      inside_is_zero: true,
      distance_modes: "euclidean",
      routing_profile: "walk",
      max_snap_distance_m: 300,
      bbox_buffer_m: 800,
    });
    actions.render();
  });

  container.querySelector("#btnAddGeojsonTarget").addEventListener("click", () => {
    container.querySelector("#fileGeojsonTarget").click();
  });
  container.querySelector("#fileGeojsonTarget").addEventListener("change", async (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = "";
    if (!files.length) return;
    let addedAny = false;
    try {
      for (const f of files) {
        const gj = parseJsonText(await readTextWithEncodingFallback(f), "GeoJSON");
        validateGeojsonObject(gj, f.name);
        const tid = makeUniqueTargetId(f.name.replace(/\.[^.]+$/, ""), targets);
        const { geometry_handling, inside_is_zero } = inferGeojsonTargetDefaults(gj);
        targets.push({
          id: tid,
          name: `GeoJSON: ${f.name}`,
          source: "geojson",
          excluded_osm_ids: [],
          geometry_handling,
          inside_is_zero,
          distance_modes: "euclidean",
          routing_profile: "walk",
          max_snap_distance_m: 300,
        });
        state.rawGeojsonTargets[tid] = gj;
        try {
          await actions.ingestSingleGeojsonTarget(tid, gj);
        } catch (ingestErr) {
          const rmIdx = targets.findIndex((t) => t.id === tid);
          if (rmIdx >= 0) targets.splice(rmIdx, 1);
          delete state.rawGeojsonTargets[tid];
          throw ingestErr;
        }
        addedAny = true;
      }
      actions.render();
    } catch (err) {
      if (addedAny) actions.render();
      alert(`GeoJSON célpont hiba: ${err.message}`);
    }
  });

  container.querySelector("#btnAddCsvDistTarget").addEventListener("click", () => {
    if (!numCols.length) return;
    const tid = makeUniqueTargetId("csv_dist", targets);
    targets.push({
      id: tid,
      name: `CSV oszlop (${numCols[0]})`,
      source: "csv_column",
      csv_col: numCols[0],
      excluded_osm_ids: [],
      geometry_handling: "point",
      distance_modes: "euclidean",
    });
    actions.render();
  });

  // Bundle export / import & Cache clear
  container.querySelector("#btnExportOsmBundle")?.addEventListener("click", () => actions.exportOsmBundle());
  container.querySelector("#btnImportOsmBundle")?.addEventListener("click", () => {
    container.querySelector("#fileOsmBundle").click();
  });
  container.querySelector("#fileOsmBundle")?.addEventListener("change", async (e) => {
    const f = e.target.files[0];
    e.target.value = "";
    if (!f) return;
    try {
      const parsed = parseOsmBundle(parseJsonText(await readTextWithEncodingFallback(f), "OSM bundle"));
      await actions.importOsmBundle(parsed, { fileName: f.name });
    } catch (err) {
      alert(`OSM bundle betöltési hiba: ${err.message}`);
    }
  });
  container.querySelector("#btnClearOsmCache")?.addEventListener("click", async () => {
    await clearAllCachedOSM();
    actions.setLog("IndexedDB OSM cache kiürítve.");
  });

  // Bind per-target edits & non-scrolling feature search/toggle
  container.querySelectorAll("[data-target-idx]").forEach((row) => {
    const idx = Number(row.getAttribute("data-target-idx"));
    const tgt = targets[idx];
    if (!tgt) return;

    row.querySelector(".tgt-geom")?.addEventListener("change", (e) => {
      const prevGeom = tgt.geometry_handling;
      const nextGeom = e.target.value;
      tgt.geometry_handling = nextGeom;
      const isOsm = tgt.source === "osm_preset" || tgt.source === "osm_custom_tag";
      if (isOsm && state.rawOsmTargets[tgt.id] && (prevGeom === "point" || nextGeom === "point") && prevGeom !== nextGeom) {
        actions.invalidateTargetSnapshot(tgt.id);
        actions.setLog(`Geometria mód váltás (${tgt.id}: ${prevGeom} → ${nextGeom}): OSM lekérés frissítésre jelölve.`);
      }
    });
    row.querySelector(".tgt-mode")?.addEventListener("change", (e) => {
      tgt.distance_modes = e.target.value;
    });
    row.querySelector(".tgt-profile")?.addEventListener("change", (e) => {
      tgt.routing_profile = e.target.value;
    });
    row.querySelector(".tgt-inside")?.addEventListener("change", (e) => {
      tgt.inside_is_zero = e.target.checked;
    });
    row.querySelector(".tgt-csv-col")?.addEventListener("change", (e) => {
      tgt.csv_col = e.target.value;
      tgt.name = `CSV oszlop (${e.target.value})`;
    });
    row.querySelector(".tgt-remove")?.addEventListener("click", () => {
      targets.splice(idx, 1);
      delete state.targetPreviews[tgt.id];
      delete state.rawOsmTargets[tgt.id];
      delete state.rawGeojsonTargets[tgt.id];
      actions.refreshMapTargets();
      actions.render();
    });

    const listBox = row.querySelector(".osm-feature-list");
    const bindListClicks = () => {
      if (!listBox) return;
      listBox.querySelectorAll(".osm-feature-item").forEach((el) => {
        el.addEventListener("click", () => {
          const uid = el.getAttribute("data-uid");
          actions.toggleOsmFeature(tgt.id, uid);
        });
      });
    };
    bindListClicks();

    row.querySelector(".osm-filter-input")?.addEventListener("input", (e) => {
      state.osmFilterQuery = state.osmFilterQuery || {};
      state.osmFilterQuery[tgt.id] = e.target.value;
      if (listBox) {
        listBox.innerHTML = renderFeatureListHtml(tgt, state.targetPreviews[tgt.id], e.target.value);
        bindListClicks();
      }
    });

    row.querySelector(".btn-osm-exclude-vis")?.addEventListener("click", () => {
      const q = (state.osmFilterQuery?.[tgt.id] || "").trim().toLowerCase();
      const featList = state.targetPreviews[tgt.id]?.feature_index || [];
      const vis = q
        ? featList.filter((f) => String(f.name || "").toLowerCase().includes(q) || String(f.uid || "").toLowerCase().includes(q))
        : featList;
      actions.setTargetExclusions(tgt.id, vis.map((f) => f.uid), true);
    });

    row.querySelector(".btn-osm-include-vis")?.addEventListener("click", () => {
      const q = (state.osmFilterQuery?.[tgt.id] || "").trim().toLowerCase();
      const featList = state.targetPreviews[tgt.id]?.feature_index || [];
      const vis = q
        ? featList.filter((f) => String(f.name || "").toLowerCase().includes(q) || String(f.uid || "").toLowerCase().includes(q))
        : featList;
      actions.setTargetExclusions(tgt.id, vis.map((f) => f.uid), false);
    });
  });

  // Bind Overpass Fetch & Distance Execution
  container.querySelector("#btnFetchOSM")?.addEventListener("click", () => actions.fetchAllTargetsOSM());
  container.querySelector("#btnComputeDistances")?.addEventListener("click", () => actions.runStage1());
}

function renderTargetRow(t, idx, numCols, previewInfo, filterQuery, hasRawSnapshot = false) {
  const exclSet = new Set(t.excluded_osm_ids || []);
  const featList = previewInfo?.feature_index || [];
  const csvColOptions = (() => {
    const base = numCols.length > 0 ? [...numCols] : t.csv_col ? [t.csv_col] : [];
    if (t.csv_col && !base.includes(t.csv_col)) {
      base.unshift(t.csv_col);
    }
    if (!base.length) return `<option value="">— nincs numerikus oszlop —</option>`;
    return base
      .map((c) => {
        const note =
          numCols.length > 0 && !numCols.includes(c)
            ? " (receptből — nincs a CSV-ben)"
            : numCols.length === 0
              ? " (receptből)"
              : "";
        return `<option value="${escapeHtml(c)}" ${c === t.csv_col ? "selected" : ""}>${escapeHtml(c)}${note}</option>`;
      })
      .join("");
  })();

  return `
    <div style="border:1px solid var(--border);border-radius:6px;padding:8px 10px;margin-bottom:8px;background:var(--surface-alt);" data-target-idx="${idx}" data-target-id="${escapeHtml(t.id)}">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
        <div>
          <strong>${escapeHtml(t.name)}</strong> <span class="mono card-subtitle">[${escapeHtml(t.id)}]</span>
          ${
            previewInfo
              ? `<span class="status-pill tgt-active-badge" style="margin-left:6px;">${featList.length - exclSet.size}/${featList.length} aktív elem</span>`
              : hasRawSnapshot
                ? `<span class="status-pill" style="margin-left:6px;">📦 Előtöltve (CSV után aktiválódik)</span>`
                : ""
          }
        </div>
        <button class="btn btn-sm tgt-remove" style="color:var(--fail);">✕ Eltávolítás</button>
      </div>
      ${
        t.source === "csv_column"
          ? `<div class="field-row"><label>Meglévő távolságoszlop (méter)</label>
             <select class="tgt-csv-col">${csvColOptions}</select></div>`
          : `<div class="grid-3">
               <div class="field-row">
                 <label>Geometria kezelése</label>
                 <select class="tgt-geom">
                   <option value="point" ${t.geometry_handling === "point" ? "selected" : ""}>Pont / Centroid</option>
                   <option value="line_segment" ${t.geometry_handling === "line_segment" ? "selected" : ""}>Vonalszakasz (egzakt merőleges)</option>
                   <option value="polygon_boundary" ${t.geometry_handling === "polygon_boundary" ? "selected" : ""}>Poligon határ</option>
                   <option value="polygon_centroid" ${t.geometry_handling === "polygon_centroid" ? "selected" : ""}>Poligon súlypont</option>
                 </select>
               </div>
               <div class="field-row">
                 <label>Távolságmód</label>
                 <select class="tgt-mode">
                   <option value="euclidean" ${t.distance_modes === "euclidean" ? "selected" : ""}>Euklideszi (légvonal)</option>
                   <option value="network_dijkstra" ${t.distance_modes === "network_dijkstra" ? "selected" : ""}>Hálózati Dijkstra</option>
                   <option value="both" ${t.distance_modes === "both" ? "selected" : ""}>Mindkettő + Δd Kanyargósság</option>
                 </select>
               </div>
               <div class="field-row">
                 <label>Hálózati profil &amp; Belső zóna</label>
                 <div style="display:flex;gap:8px;align-items:center;">
                   <select class="tgt-profile" style="flex:1;">
                     <option value="walk" ${t.routing_profile === "walk" ? "selected" : ""}>Gyalogos</option>
                     <option value="bike" ${t.routing_profile === "bike" ? "selected" : ""}>Kerékpáros</option>
                     <option value="drive" ${t.routing_profile === "drive" ? "selected" : ""}>Autós</option>
                   </select>
                   <label style="display:inline-flex;align-items:center;gap:4px;white-space:nowrap;">
                     <input type="checkbox" class="tgt-inside" ${t.inside_is_zero ? "checked" : ""} /> d=0 belül
                   </label>
                 </div>
               </div>
             </div>`
      }
      ${
        featList.length > 0
          ? `<div style="margin-top:8px;">
               <div class="osm-search-bar">
                 <input type="text" class="osm-filter-input" value="${escapeHtml(filterQuery)}" placeholder="🔍 Szűrés név / UID alapján (${featList.length} elem)..." style="flex:1;min-width:160px;" />
                 <button class="btn btn-sm btn-osm-include-vis" type="button">☑ Láthatók aktívvá</button>
                 <button class="btn btn-sm btn-osm-exclude-vis" type="button">☐ Láthatók kizárása</button>
               </div>
               <div class="osm-feature-list">
                 ${renderFeatureListHtml(t, previewInfo, filterQuery)}
               </div>
             </div>`
          : ""
      }
    </div>
  `;
}
