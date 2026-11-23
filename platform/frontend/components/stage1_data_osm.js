/* Stage 1: BYO CSV/GeoJSON Upload, Schema Role Mapping, Parametric OSM Targets & Interactive Feature Filter */
import {
  OSM_TARGET_PRESETS,
  fetchStreetGraphOSM,
  fetchTargetOSM,
  parseTagShorthand,
} from "../gis/overpass_client.js";
import {
  generateEmptyCSVTemplate,
  inferColumnProfile,
  parseCSVText,
  parseGeoJSONDataset,
} from "../state/recipe_store.js";

export function renderStage1(container, ctx) {
  const { state, actions } = ctx;
  const ds = state.recipe.dataset_schema;
  const headers = state.headers || [];
  const profile = state.colProfile || {};

  const numCols = headers.filter((h) => profile[h]?.isNumeric);
  const allOpts = (sel) =>
    `<option value="">— válassz —</option>` +
    headers.map((h) => `<option value="${h}" ${h === sel ? "selected" : ""}>${h}</option>`).join("");

  const targets = state.recipe.spatial_targets || [];

  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">1A. Saját Adatbázis Betöltése (CSV / GeoJSON)</h3>
          <div class="card-subtitle">100%-ban kliensoldali: a nyers adatok soha nem hagyják el a böngésződet.</div>
        </div>
        <div style="display:flex;gap:6px;">
          <button class="btn btn-sm" id="btnDlCsvTemplate">📄 Üres CSV Sablon</button>
          <button class="btn btn-sm" id="btnImportRecipeS1">📥 Recept (.json)</button>
          <input type="file" id="fileRecipeS1" accept=".json" style="display:none" />
        </div>
      </div>

      <div class="dropzone" id="csvDropzone">
        <div style="font-weight:600;margin-bottom:4px;">📂 Húzd ide vagy válaszd ki a CSV / GeoJSON adatfájlt</div>
        <div class="card-subtitle">Támogatott elválasztók: <code>,</code> <code>;</code> <code>Tab</code> | magyar/nemzetközi tizedesformátum | hiányzó koordináták megőrzése Heckman-szelekcióhoz</div>
        <input type="file" id="csvFileInput" accept=".csv,.tsv,.txt,.geojson,.json" style="display:none" />
      </div>

      ${
        headers.length > 0
          ? `<div class="alert-box info" style="margin-top:10px;margin-bottom:0;">
               Betöltve: <strong>${state.fileName}</strong> — <span class="mono">${state.nRows} sor</span>, <span class="mono">${headers.length} oszlop</span>
               ${state.datasetInfo ? ` | Geokódolt: <strong class="mono">${state.datasetInfo.n_geocoded}</strong> | Nem geokódolt (Heckman pool): <strong class="mono">${state.datasetInfo.n_non_geocoded}</strong> | Vetület: <span class="mono">${state.datasetInfo.projection.nearest_utm_epsg} (${state.datasetInfo.projection.mode})</span>` : ""}
             </div>`
          : ""
      }
    </div>

    ${
      headers.length > 0
        ? `
      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">1B. Oszlopszerepek & Függő Változó (y)</h3>
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
            <input type="number" id="mapClusterRadius" value="${ds.cluster_spec.radius_m || 15}" min="1" max="500" />
          </div>
        </div>

        <div class="grid-2" style="margin-top:8px;">
          <div class="field-row">
            <label>Térbeli Proxy / CBD oszlopok (Guard #1, több is jelölhető)</label>
            <select id="mapProxyCols" multiple size="2" style="height:46px;">
              ${numCols.map((h) => `<option value="${h}" ${(ds.spatial_proxy_cols || []).includes(h) ? "selected" : ""}>${h}</option>`).join("")}
            </select>
          </div>
          <div class="field-row">
            <label>Heckman kizárási restrikció Z<sub>excl</sub> (Guard #4, opcionális)</label>
            <select id="mapHeckExcl" multiple size="2" style="height:46px;">
              ${numCols.map((h) => `<option value="${h}" ${(ds.selection_spec.exclusion_restriction_cols || []).includes(h) ? "selected" : ""}>${h}</option>`).join("")}
            </select>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">1C. Parametrikus Térbeli Célpontok (Bármilyen OSM Entitás / GeoJSON)</h3>
            <div class="card-subtitle">Adj hozzá célpontokat, kérd le az OSM objektumokat, majd kapcsold ki/be az elemeket a térképen mérés előtt</div>
          </div>
          <div style="display:flex;gap:6px;">
            <button class="btn btn-sm" id="btnFetchOSM" ${targets.some((t) => t.source !== "csv_column") ? "" : "disabled"}>🌐 OSM / Rétegek Lekérése</button>
            <button class="btn btn-primary btn-sm" id="btnComputeDistances" ${targets.length ? "" : "disabled"}>📐 Távolságok Számítása</button>
          </div>
        </div>

        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;align-items:center;">
          <select id="selOsmPreset" style="min-width:210px;">
            ${OSM_TARGET_PRESETS.map((p) => `<option value="${p.preset_id}">${p.label}</option>`).join("")}
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
              ? `<div class="card-subtitle">Még nincs felvett térbeli célpont. Válassz fentről egy OSM sablont vagy adj meg tetszőleges <code>key=value</code> címkét!</div>`
              : targets.map((t, idx) => renderTargetRow(t, idx, numCols, state.targetPreviews[t.id])).join("")
          }
        </div>
      </div>

      ${
        state.stage1Res
          ? `<div class="card">
               <div class="card-header">
                 <h3 class="card-title">1D. Térbeli Távolság & Hálózati Kanyargósság Összesítő</h3>
                 <span class="card-subtitle">A becslések készen állnak a 2–5. fázisokhoz</span>
               </div>
               ${(state.stage1Res.warnings || []).map((w) => `<div class="alert-box">⚠️ ${w}</div>`).join("")}
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
                           <td><strong>${r.name}</strong> <span class="card-subtitle mono">(${r.target_id})</span></td>
                           <td class="num">${r.n_active_features}</td>
                           <td class="num">${r.n_excluded_features}</td>
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

  // Bind File / Template / Recipe events
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
    if (e.target.files[0]) handleDataFile(e.target.files[0], actions);
  });

  container.querySelector("#btnDlCsvTemplate").addEventListener("click", () => {
    actions.downloadText("geospec_empty_template.csv", generateEmptyCSVTemplate(), "text/csv");
  });
  container.querySelector("#btnImportRecipeS1").addEventListener("click", () => {
    container.querySelector("#fileRecipeS1").click();
  });
  container.querySelector("#fileRecipeS1").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    actions.importRecipe(JSON.parse(await f.text()));
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
      distance_modes: "both",
      routing_profile: "walk",
      max_snap_distance_m: 300,
      bbox_buffer_m: 1500,
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
      distance_modes: "both",
      routing_profile: "walk",
      max_snap_distance_m: 300,
      bbox_buffer_m: 1500,
    });
    actions.render();
  });

  container.querySelector("#btnAddGeojsonTarget").addEventListener("click", () => {
    container.querySelector("#fileGeojsonTarget").click();
  });
  container.querySelector("#fileGeojsonTarget").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const gj = JSON.parse(await f.text());
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

  // Bind per-target edits & feature exclusions
  container.querySelectorAll("[data-target-idx]").forEach((row) => {
    const idx = Number(row.getAttribute("data-target-idx"));
    const tgt = targets[idx];
    if (!tgt) return;

    row.querySelector(".tgt-geom")?.addEventListener("change", (e) => {
      tgt.geometry_handling = e.target.value;
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
      actions.refreshMapTargets();
      actions.render();
    });
    row.querySelectorAll(".osm-feature-item").forEach((el) => {
      el.addEventListener("click", () => {
        const uid = el.getAttribute("data-uid");
        actions.toggleOsmFeature(tgt.id, uid);
      });
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

function renderTargetRow(t, idx, numCols, previewInfo) {
  const exclSet = new Set(t.excluded_osm_ids || []);
  const featList = previewInfo?.feature_index || [];
  return `
    <div style="border:1px solid var(--border);border-radius:6px;padding:8px 10px;margin-bottom:8px;background:var(--surface-alt);" data-target-idx="${idx}">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
        <div>
          <strong>${t.name}</strong> <span class="mono card-subtitle">[${t.id}]</span>
          ${previewInfo ? `<span class="status-pill" style="margin-left:6px;">${featList.length - exclSet.size}/${featList.length} aktív elem</span>` : ""}
        </div>
        <button class="btn btn-sm tgt-remove" style="color:var(--fail);">✕ Eltávolítás</button>
      </div>
      ${
        t.source === "csv_column"
          ? `<div class="field-row"><label>Meglévő távolságoszlop (méter)</label>
             <select class="tgt-csv-col">${numCols.map((c) => `<option value="${c}" ${c === t.csv_col ? "selected" : ""}>${c}</option>`).join("")}</select></div>`
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
                 <label>Hálózati profil & Belső zóna</label>
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
               <div class="card-subtitle" style="margin-bottom:4px;">Lekérdezett OSM / GeoJSON elemek (kattints itt vagy a térképen az egyedi kizáráshoz mérés előtt):</div>
               <div class="osm-feature-list">
                 ${featList
                   .slice(0, 120)
                   .map((f) => {
                     const isExcl = exclSet.has(f.uid);
                     return `<div class="osm-feature-item ${isExcl ? "excluded" : ""}" data-uid="${f.uid}">
                       <span>${isExcl ? "☐" : "☑"} <strong>${f.name}</strong> <span class="mono card-subtitle">(${f.uid}, ${f.geom_type})</span></span>
                       <span class="card-subtitle">${isExcl ? "kizárva" : "aktív"}</span>
                     </div>`;
                   })
                   .join("")}
                 ${featList.length > 120 ? `<div class="card-subtitle" style="padding:4px;">… és további ${featList.length - 120} elem (kattinthatsz közvetlenül a térképen is).</div>` : ""}
               </div>
             </div>`
          : ""
      }
    </div>
  `;
}

async function handleDataFile(file, actions) {
  const text = await file.text();
  let parsed;
  if (file.name.toLowerCase().endsWith(".geojson") || file.name.toLowerCase().endsWith(".json")) {
    parsed = parseGeoJSONDataset(text);
  } else {
    parsed = parseCSVText(text);
  }
  const { profile, guesses } = inferColumnProfile(parsed.headers, parsed.columns);
  actions.onDatasetParsed(file.name, parsed, profile, guesses);
}
