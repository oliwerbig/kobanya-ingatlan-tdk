/* Stage 1: BYO CSV/GeoJSON Upload, 1-Click Sandbox, Schema Mapping, Parametric OSM/Bundle & Non-Scrolling Feature Filter */
import { clearAllCachedOSM, parseOsmBundle } from "../gis/osm_idb_cache.js";
import { OSM_TARGET_PRESETS, parseTagShorthand } from "../gis/overpass_client.js";
import {
  generateEmptyCSVTemplate,
  generateSampleStudy,
  inferColumnProfile,
  parseCSVText,
  parseGeoJSONDataset,
} from "../state/recipe_store.js";
import { escapeHtml } from "../utils/sanitize.js";

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

export function renderStage1(container, ctx) {
  const { state, actions } = ctx;
  const ds = state.recipe.dataset_schema;
  const headers = state.headers || [];
  const profile = state.colProfile || {};

  const numCols = headers.filter((h) => profile[h]?.isNumeric);
  const allOpts = (sel) =>
    `<option value="">— válassz —</option>` +
    headers.map((h) => `<option value="${escapeHtml(h)}" ${h === sel ? "selected" : ""}>${escapeHtml(h)}</option>`).join("");

  const targets = state.recipe.spatial_targets || [];
  const hasSnapshotTargets = targets.some((t) => state.rawOsmTargets[t.id] || state.rawGeojsonTargets[t.id]);

  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">1A. Saját Adatbázis Betöltése (CSV / GeoJSON)</h3>
          <div class="card-subtitle">100%-ban kliensoldali: a nyers adatok soha nem hagyják el a böngésződet (UTF-8 &amp; Windows-1250 támogatással).</div>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <button class="btn btn-sm" id="btnLoadSampleStudy" title="Szintetikus városi minta-adatbázis, GeoJSON célpontok és utcahálózat azonnali kipróbáláshoz">🎯 Minta Sandbox</button>
          <button class="btn btn-sm" id="btnDlCsvTemplate">📄 CSV Sablon</button>
          <button class="btn btn-sm" id="btnImportRecipeS1">📥 Recept (.json)</button>
          <input type="file" id="fileRecipeS1" accept=".json" style="display:none" />
        </div>
      </div>

      <div class="dropzone" id="csvDropzone">
        <div style="font-weight:600;margin-bottom:4px;">📂 Húzd ide vagy válaszd ki a CSV / GeoJSON adatfájlt</div>
        <div class="card-subtitle">Támogatott elválasztók: <code>,</code> <code>;</code> <code>Tab</code> | idézett többsoros mezők | magyar/nemzetközi számformátum | hiányzó koordináták megőrzése Heckman-szelekcióhoz</div>
        <input type="file" id="csvFileInput" accept=".csv,.tsv,.txt,.geojson,.json" style="display:none" />
      </div>

      ${
        headers.length > 0
          ? `<div class="alert-box info" style="margin-top:10px;margin-bottom:0;">
               Betöltve: <strong>${escapeHtml(state.fileName)}</strong> — <span class="mono">${state.nRows} sor</span>, <span class="mono">${headers.length} oszlop</span>
               ${state.datasetInfo ? ` | Geokódolt: <strong class="mono">${state.datasetInfo.n_geocoded}</strong> | Nem geokódolt (Heckman pool): <strong class="mono">${state.datasetInfo.n_non_geocoded}</strong> | Vetület: <span class="mono">${escapeHtml(state.datasetInfo.projection.nearest_utm_epsg)} (${escapeHtml(state.datasetInfo.projection.mode)})</span>` : ""}
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

    ${
      headers.length > 0
        ? `
      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">1B. Oszlopszerepek &amp; Függő Változó (y)</h3>
            <div class="card-subtitle">Koordináták, célváltozó-transzformáció és opcionális klaszter / szelekciós / proxy szerepkörök</div>
          </div>
          <button class="btn btn-primary btn-sm" id="btnApplySchema">✓ Séma Rögzítése</button>
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
              ${numCols.map((h) => `<option value="${escapeHtml(h)}" ${(ds.spatial_proxy_cols || []).includes(h) ? "selected" : ""}>${escapeHtml(h)}</option>`).join("")}
            </select>
          </div>
          <div class="field-row">
            <label>Heckman kizárási restrikció Z<sub>excl</sub> (Guard #4, opcionális)</label>
            <select id="mapHeckExcl" multiple size="2" style="height:46px;">
              ${numCols.map((h) => `<option value="${escapeHtml(h)}" ${(ds.selection_spec.exclusion_restriction_cols || []).includes(h) ? "selected" : ""}>${escapeHtml(h)}</option>`).join("")}
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
            <button class="btn btn-sm" id="btnFetchOSM" ${targets.some((t) => t.source === "osm_preset" || t.source === "osm_custom_tag") ? "" : "disabled"}>🌐 OSM Lekérése</button>
            <button class="btn btn-sm" id="btnImportOsmBundle" title="Korábban mentett offline .osm_bundle.json betöltése">📦 Bundle Betöltés</button>
            <input type="file" id="fileOsmBundle" accept=".json" style="display:none" />
            <button class="btn btn-sm" id="btnExportOsmBundle" ${hasSnapshotTargets || Object.keys(state.rawStreetGraphs).length ? "" : "disabled"} title="OSM / GeoJSON rétegek és utcahálózat mentése offline reprodukálhatósághoz">💾 .osm_bundle</button>
            <button class="btn btn-sm" id="btnClearOsmCache" title="IndexedDB OSM cache ürítése">🗑 Cache</button>
            <button class="btn btn-primary btn-sm" id="btnComputeDistances" ${targets.length ? "" : "disabled"}>📐 Távolságok Számítása</button>
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
          <input type="file" id="fileGeojsonTarget" accept=".geojson,.json" style="display:none" />
          <button class="btn btn-sm" id="btnAddCsvDistTarget">+ Meglévő CSV Oszlop</button>
        </div>

        <div id="targetListArea">
          ${
            targets.length === 0
              ? `<div class="card-subtitle">Még nincs felvett térbeli célpont. Válassz fentről egy OSM sablont, vagy adj meg tetszőleges <code>key=value</code> címkét!</div>`
              : targets.map((t, idx) => renderTargetRow(t, idx, numCols, state.targetPreviews[t.id], state.osmFilterQuery?.[t.id] || "")).join("")
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
    `
        : ""
    }
  `;

  // Bind File / Sandbox / Template / Recipe events
  const dropzone = container.querySelector("#csvDropzone");
  const fileInput = container.querySelector("#csvFileInput");
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files[0]) handleDataFile(e.dataTransfer.files[0], actions);
  });
  fileInput.addEventListener("change", (e) => {
    const f = e.target.files[0];
    e.target.value = "";
    if (f) handleDataFile(f, actions);
  });

  container.querySelector("#btnLoadSampleStudy").addEventListener("click", async () => {
    const sample = generateSampleStudy();
    await actions.loadSampleStudy(sample);
  });

  container.querySelector("#btnDlCsvTemplate").addEventListener("click", () => {
    actions.downloadText("geospec_template.csv", generateEmptyCSVTemplate(), "text/csv");
  });
  container.querySelector("#btnImportRecipeS1").addEventListener("click", () => {
    container.querySelector("#fileRecipeS1").click();
  });
  container.querySelector("#fileRecipeS1").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    e.target.value = "";
    if (!f) return;
    try {
      const text = await readTextWithEncodingFallback(f);
      actions.importRecipe(JSON.parse(text));
    } catch (err) {
      alert(`Recept betöltési hiba: ${err.message}`);
    }
  });

  if (!headers.length) return;

  // Bind Schema controls
  container.querySelector("#mapYMode").addEventListener("change", (e) => {
    ds.y_spec.mode = e.target.value;
    actions.render();
  });
  container.querySelector("#btnApplySchema").addEventListener("click", () => {
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
    actions.syncDatasetToWorker();
  });

  // Bind Target Addition controls
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
    const f = e.target.files[0];
    e.target.value = "";
    if (!f) return;
    try {
      const gj = JSON.parse(await readTextWithEncodingFallback(f));
      const tid = makeUniqueTargetId(f.name.replace(/\.[^.]+$/, "").replace(/[^a-z0-9_]/gi, "_").toLowerCase(), targets);
      targets.push({
        id: tid,
        name: `GeoJSON: ${f.name}`,
        source: "geojson",
        excluded_osm_ids: [],
        geometry_handling: "polygon_boundary",
        inside_is_zero: true,
        distance_modes: "euclidean",
        routing_profile: "walk",
        max_snap_distance_m: 300,
      });
      state.rawGeojsonTargets[tid] = gj;
      await actions.ingestSingleGeojsonTarget(tid, gj);
      actions.render();
    } catch (err) {
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
      const parsed = parseOsmBundle(JSON.parse(await readTextWithEncodingFallback(f)));
      await actions.importOsmBundle(parsed);
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

function makeUniqueTargetId(base, targets) {
  const clean = base.replace(/[^a-z0-9_]/gi, "_").toLowerCase() || "target";
  const existing = new Set(targets.map((t) => t.id));
  if (!existing.has(clean)) return clean;
  let k = 2;
  while (existing.has(`${clean}_${k}`)) k++;
  return `${clean}_${k}`;
}

function renderTargetRow(t, idx, numCols, previewInfo, filterQuery) {
  const exclSet = new Set(t.excluded_osm_ids || []);
  const featList = previewInfo?.feature_index || [];
  return `
    <div style="border:1px solid var(--border);border-radius:6px;padding:8px 10px;margin-bottom:8px;background:var(--surface-alt);" data-target-idx="${idx}" data-target-id="${escapeHtml(t.id)}">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
        <div>
          <strong>${escapeHtml(t.name)}</strong> <span class="mono card-subtitle">[${escapeHtml(t.id)}]</span>
          ${previewInfo ? `<span class="status-pill tgt-active-badge" style="margin-left:6px;">${featList.length - exclSet.size}/${featList.length} aktív elem</span>` : ""}
        </div>
        <button class="btn btn-sm tgt-remove" style="color:var(--fail);">✕ Eltávolítás</button>
      </div>
      ${
        t.source === "csv_column"
          ? `<div class="field-row"><label>Meglévő távolságoszlop (méter)</label>
             <select class="tgt-csv-col">${numCols.map((c) => `<option value="${escapeHtml(c)}" ${c === t.csv_col ? "selected" : ""}>${escapeHtml(c)}</option>`).join("")}</select></div>`
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

async function readTextWithEncodingFallback(file) {
  const buf = await file.arrayBuffer();
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(buf);
  } catch {
    try {
      return new TextDecoder("windows-1250").decode(buf);
    } catch {
      return new TextDecoder("iso-8859-2").decode(buf);
    }
  }
}

async function handleDataFile(file, actions) {
  try {
    const text = await readTextWithEncodingFallback(file);
    let parsed;
    if (file.name.toLowerCase().endsWith(".geojson") || file.name.toLowerCase().endsWith(".json")) {
      parsed = parseGeoJSONDataset(text);
    } else {
      parsed = parseCSVText(text);
    }
    const { profile, guesses } = inferColumnProfile(parsed.headers, parsed.columns);
    await actions.onDatasetParsed(file.name, parsed, profile, guesses);
  } catch (err) {
    alert(`Fájl betöltési hiba: ${err.message}`);
  }
}
