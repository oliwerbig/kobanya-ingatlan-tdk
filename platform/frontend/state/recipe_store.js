/* ExperimentRecipe Store, RFC-4180 CSV/GeoJSON Parser, Workspace Bundle & Multi-File Intake Classifier */
import { parseOsmBundle } from "../gis/osm_idb_cache.js";
import { parseJsonText, parseNumericScalar, sanitizeRecipe, stripBom } from "../utils/sanitize.js";

export function createDefaultRecipe() {
  return {
    schema_version: "1.0.0",
    recipe_meta: {
      title: "GeoSpec Spatial Study",
      created_utc: new Date().toISOString(),
    },
    dataset_schema: {
      row_id_col: "",
      lat_col: "",
      lon_col: "",
      y_spec: {
        mode: "column",
        num_col: "",
        den_col: "",
        transform: "log",
        boxcox_lambda: null,
        log1p_shift: 1.0,
      },
      cluster_spec: {
        mode: "coordinate_radius",
        col: "",
        regex: "",
        radius_m: 15.0,
      },
      selection_spec: {
        enabled: true,
        excluded_flag_col: "",
        first_stage_covariates: [],
        exclusion_restriction_cols: [],
      },
      spatial_proxy_cols: [],
    },
    spatial_targets: [],
    models: [
      {
        id: "spec_a",
        name: "Spec A — Saját Modell",
        active: true,
        covariates: [],
        spatial_terms: [],
        interactions: [],
      },
    ],
    composition_chain: {
      enabled: true,
      focal_term_ref: "",
      blocks: [],
      compute_gelbach_ovb: true,
    },
    spatial_engine: {
      active_model_id: "spec_a",
      weights: [
        { id: "knn5", type: "knn", k: 5, dedup_colocated: true },
        { id: "idw_auto", type: "idw", power: 1.0, cutoff_mode: "auto_connectivity", cutoff_m: null },
      ],
      models: ["SAR", "SEM", "SDM"],
      krinsky_robb_draws: 1000,
      lisa_fdr: true,
      gwr: {
        enabled: true,
        focal_model_id: "spec_a",
        kernel: "adaptive_bisquare",
        jitter_m: 1.0,
        b_min_floor: 15,
        kappa_thresh: 1000.0,
        prevalence_min_pct: 0.10,
      },
    },
    guards_cfg: {
      common_sample_lock: true,
      sparse_min_cell: 5,
      hc3_leverage_warn: 0.85,
      attenuation_warn_pct: 50.0,
      heckman_min_excluded: 10,
      heckman_min_analytic: 30,
    },
  };
}

export function detectDelimiter(firstLine) {
  const counts = { ",": 0, ";": 0, "\t": 0 };
  let inQuotes = false;
  for (let i = 0; i < firstLine.length; i++) {
    const ch = firstLine[i];
    if (ch === '"') inQuotes = !inQuotes;
    else if (!inQuotes && counts[ch] !== undefined) counts[ch]++;
  }
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
}

export function parseCSVText(text) {
  const cleanText = stripBom(text).replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  // Find first non-empty, non-comment line for delimiter detection
  let firstNonEmpty = "";
  for (const rawLine of cleanText.split("\n")) {
    const trimmed = rawLine.trim();
    if (trimmed && !trimmed.startsWith("#")) {
      firstNonEmpty = rawLine;
      break;
    }
  }
  if (!firstNonEmpty) {
    throw new Error("A CSV fájl üres.");
  }
  const delim = detectDelimiter(firstNonEmpty);

  // Full RFC-4180 state-machine parser (supports embedded newlines and escaped quotes inside quoted fields)
  const rows = [];
  let row = [];
  let cur = "";
  let inQ = false;
  let atLineStart = true;
  let skippingComment = false;

  for (let i = 0; i < cleanText.length; i++) {
    const ch = cleanText[i];
    if (skippingComment) {
      if (ch === "\n") {
        skippingComment = false;
        atLineStart = true;
      }
      continue;
    }
    if (atLineStart && !inQ && ch === "#") {
      skippingComment = true;
      continue;
    }
    if (ch === '"') {
      atLineStart = false;
      if (inQ && cleanText[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        inQ = !inQ;
      }
    } else if (ch === delim && !inQ) {
      atLineStart = false;
      row.push(cur.trim());
      cur = "";
    } else if (ch === "\n" && !inQ) {
      row.push(cur.trim());
      cur = "";
      if (row.some((cell) => cell !== "")) {
        rows.push(row);
      }
      row = [];
      atLineStart = true;
    } else {
      if (ch !== " " && ch !== "\t") atLineStart = false;
      cur += ch;
    }
  }
  if (cur.length > 0 || row.length > 0) {
    row.push(cur.trim());
    if (row.some((cell) => cell !== "")) rows.push(row);
  }

  if (rows.length < 2) {
    throw new Error("A CSV fájlnak legalább fejlécet és 1 adatsort kell tartalmaznia.");
  }

  // Deduplicate empty or repeated headers safely
  const seenHeaders = new Map();
  const headers = rows[0].map((h, idx) => {
    const base = h || `col_${idx}`;
    const cnt = seenHeaders.get(base) || 0;
    seenHeaders.set(base, cnt + 1);
    return cnt === 0 ? base : `${base}_${cnt + 1}`;
  });

  const columns = {};
  const setColProp = (obj, key, val) =>
    Object.defineProperty(obj, key, { value: val, enumerable: true, writable: true, configurable: true });
  headers.forEach((h) => {
    setColProp(columns, h, []);
  });

  for (let r = 1; r < rows.length; r++) {
    const vals = rows[r];
    for (let c = 0; c < headers.length; c++) {
      columns[headers[c]].push(vals[c] !== undefined ? vals[c] : "");
    }
  }

  return { headers, columns, nRows: rows.length - 1 };
}

export function parseGeoJSONDataset(textOrObj) {
  const gj = typeof textOrObj === "string" ? parseJsonText(textOrObj, "GeoJSON") : textOrObj;
  if (!gj || typeof gj !== "object") {
    throw new Error("A GeoJSON fájl nem érvényes objektum.");
  }
  const feats = gj.type === "FeatureCollection" ? gj.features || [] : gj.type === "Feature" ? [gj] : [];
  if (!feats.length) throw new Error("A GeoJSON fájl nem tartalmaz Feature elemeket.");

  const propKeys = new Set();
  for (const f of feats) {
    Object.keys(f?.properties || {}).forEach((k) => {
      if (k !== "lat" && k !== "lon") propKeys.add(k);
    });
  }
  const columns = { lat: [], lon: [] };
  propKeys.forEach((k) => {
    Object.defineProperty(columns, k, { value: [], enumerable: true, writable: true, configurable: true });
  });

  for (const f of feats) {
    let lon = null;
    let lat = null;
    if (f?.geometry && f.geometry.type === "Point" && Array.isArray(f.geometry.coordinates)) {
      lon = f.geometry.coordinates[0];
      lat = f.geometry.coordinates[1];
    }
    columns.lat.push(lat === null || lat === undefined ? "" : lat);
    columns.lon.push(lon === null || lon === undefined ? "" : lon);
    const props = f?.properties || {};
    propKeys.forEach((k) => {
      columns[k].push(props[k] !== undefined && props[k] !== null ? props[k] : "");
    });
  }

  return { headers: Object.keys(columns), columns, nRows: feats.length };
}

export function inferColumnProfile(headers, columns) {
  const profile = {};
  const n = columns[headers[0]] ? columns[headers[0]].length : 0;

  for (const h of headers) {
    const vals = columns[h];
    let numCount = 0;
    let nonEmpty = 0;
    const uniq = new Set();
    for (let i = 0; i < vals.length; i++) {
      const s = String(vals[i] ?? "").trim();
      if (!s || ["na", "nan", "null", "none", "n/a", "-"].includes(s.toLowerCase())) continue;
      nonEmpty++;
      uniq.add(s);
      if (Number.isFinite(parseNumericScalar(s))) numCount++;
    }
    const isNumeric = nonEmpty > 0 && numCount / nonEmpty >= 0.85;
    profile[h] = {
      name: h,
      nonEmpty,
      missing: n - nonEmpty,
      uniqueCount: uniq.size,
      isNumeric,
      sampleLevels: Array.from(uniq).slice(0, 12),
    };
  }

  const findMatch = (patterns) =>
    headers.find((h) => patterns.some((rx) => rx.test(h.toLowerCase()))) || "";

  const latGuess = findMatch([/^lat$/, /wgs84_lat/, /latitude/, /y_coord/, /szelesseg/]);
  const lonGuess = findMatch([/^lon$/, /^lng$/, /wgs84_lon/, /^wgs84_long$/, /longitude/, /x_coord/, /hosszusag/]);
  const idGuess = findMatch([/^id$/, /row_id/, /listing_id/, /obj_id/]);
  const yGuess =
    findMatch([/fajlagos/, /price_m2/, /unit_price/, /^price$/, /^ar$/, /rent/, /^y$/]) ||
    headers.find((h) => h !== latGuess && h !== lonGuess && h !== idGuess && profile[h].isNumeric) ||
    "";

  return { profile, guesses: { latGuess, lonGuess, idGuess, yGuess } };
}

export function generateEmptyCSVTemplate() {
  return [
    "# GeoSpec minta CSV sablon (a '#' kezdetű sorokat a betöltő automatikusan figyelmen kívül hagyja)",
    "id,lat,lon,price,area_m2,condition,rooms,street_name,dist_cbd_m",
    "1,47.00120,15.00210,125000,54.5,good,2,Main St 12,1850",
    "2,47.00340,15.00480,168000,72.0,renovated,3,Park Ave 4,1420",
  ].join("\n") + "\n";
}

export async function readTextWithEncodingFallback(file) {
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

export function slugifyTitle(title, fallback = "geospec_study") {
  const ascii = String(title ?? "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return ascii || fallback;
}

export function makeUniqueTargetId(base, targets = []) {
  const clean =
    String(base ?? "")
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/[^a-z0-9_]/gi, "_")
      .replace(/^_+|_+$/g, "")
      .toLowerCase() || "target";
  const existing = new Set((targets || []).map((t) => t.id));
  if (!existing.has(clean)) return clean;
  let k = 2;
  while (existing.has(`${clean}_${k}`)) k++;
  return `${clean}_${k}`;
}

export function inferGeojsonTargetDefaults(geojsonObj) {
  const feats =
    geojsonObj?.type === "FeatureCollection"
      ? geojsonObj.features || []
      : geojsonObj?.type === "Feature"
        ? [geojsonObj]
        : [];
  let hasPoly = false;
  let hasLine = false;
  for (const f of feats) {
    const gt = f?.geometry?.type;
    if (gt === "Polygon" || gt === "MultiPolygon") {
      hasPoly = true;
      break;
    }
    if (gt === "LineString" || gt === "MultiLineString") {
      hasLine = true;
    }
  }
  if (hasPoly) return { geometry_handling: "polygon_boundary", inside_is_zero: true };
  if (hasLine) return { geometry_handling: "line_segment", inside_is_zero: false };
  return { geometry_handling: "point", inside_is_zero: false };
}

export function buildWorkspaceBundle({
  recipe,
  fileName = "",
  headers = [],
  columns = null,
  nRows = 0,
  osmBundle = null,
  activeSpecId = "",
  covType = "HC3",
  showPct = false,
}) {
  return {
    kind: "geospec_workspace",
    schema_version: "1.0.0",
    saved_utc: new Date().toISOString(),
    recipe,
    dataset: columns
      ? {
          fileName: fileName || "dataset.csv",
          headers: Array.isArray(headers) ? [...headers] : Object.keys(columns),
          columns,
          nRows: Number(nRows) || (headers.length && columns[headers[0]] ? columns[headers[0]].length : 0),
        }
      : null,
    osm_bundle: osmBundle || null,
    ui_state: {
      activeSpecId: activeSpecId || recipe?.spatial_engine?.active_model_id || recipe?.models?.[0]?.id || "spec_a",
      covType: covType || "HC3",
      showPct: Boolean(showPct),
    },
  };
}

export function parseWorkspaceBundle(rawObj) {
  if (!rawObj || typeof rawObj !== "object" || Array.isArray(rawObj) || rawObj.kind !== "geospec_workspace") {
    throw new Error("Érvénytelen .geospec.json munkamenet formátum.");
  }
  const recipe = sanitizeRecipe(rawObj.recipe && typeof rawObj.recipe === "object" ? rawObj.recipe : {}, createDefaultRecipe());

  let dataset = null;
  if (rawObj.dataset !== null && rawObj.dataset !== undefined) {
    const dsObj = rawObj.dataset;
    if (!dsObj || typeof dsObj !== "object" || !dsObj.columns || typeof dsObj.columns !== "object" || Array.isArray(dsObj.columns)) {
      throw new Error("A .geospec.json fájl adatbázis-szekciója sérült.");
    }
    const rawCols = dsObj.columns;
    const safeCols = {};
    const candidateHeaders = Array.isArray(dsObj.headers) && dsObj.headers.length ? dsObj.headers.map(String) : Object.keys(rawCols);
    const headers = [];
    for (const h of candidateHeaders) {
      if (h === "__proto__" || h === "constructor" || h === "prototype") continue;
      if (!Object.prototype.hasOwnProperty.call(rawCols, h) || !Array.isArray(rawCols[h])) continue;
      Object.defineProperty(safeCols, h, {
        value: [...rawCols[h]],
        enumerable: true,
        writable: true,
        configurable: true,
      });
      headers.push(h);
    }
    if (!headers.length) {
      throw new Error("A .geospec.json fájl adatbázisa nem tartalmaz érvényes oszlopokat.");
    }
    const nRows = safeCols[headers[0]].length;
    dataset = {
      fileName: typeof dsObj.fileName === "string" && dsObj.fileName ? dsObj.fileName : "workspace_dataset.csv",
      headers,
      columns: safeCols,
      nRows,
    };
  }

  let osm_bundle = null;
  if (rawObj.osm_bundle !== null && rawObj.osm_bundle !== undefined) {
    osm_bundle = parseOsmBundle(rawObj.osm_bundle);
  }

  const uiIn = rawObj.ui_state && typeof rawObj.ui_state === "object" ? rawObj.ui_state : {};
  const modelIds = new Set((recipe.models || []).map((m) => m.id));
  const fallbackSpec =
    (recipe.spatial_engine?.active_model_id && modelIds.has(recipe.spatial_engine.active_model_id)
      ? recipe.spatial_engine.active_model_id
      : recipe.models?.[0]?.id) || "spec_a";
  const activeSpecId = typeof uiIn.activeSpecId === "string" && modelIds.has(uiIn.activeSpecId) ? uiIn.activeSpecId : fallbackSpec;
  const covType = ["HC3", "cluster", "OLS"].includes(uiIn.covType) ? uiIn.covType : "HC3";
  const showPct = Boolean(uiIn.showPct);

  return {
    kind: "geospec_workspace",
    recipe,
    dataset,
    osm_bundle,
    ui_state: { activeSpecId, covType, showPct },
  };
}

function isPlainObject(v) {
  return Boolean(v) && typeof v === "object" && !Array.isArray(v);
}

function isOsmBundleObject(obj) {
  if (!isPlainObject(obj)) return false;
  if (obj.kind === "geospec_osm_bundle" || obj.schema === "geospec.osm_bundle.v1") return true;
  if (isPlainObject(obj.targets) && isPlainObject(obj.street_graphs) && !obj.dataset_schema) return true;
  if (
    isPlainObject(obj.targets) &&
    !obj.dataset_schema &&
    !obj.models &&
    !obj.spatial_targets &&
    !obj.recipe_meta &&
    !obj.schema_version &&
    obj.type !== "FeatureCollection" &&
    obj.type !== "Feature"
  ) {
    return true;
  }
  return false;
}

function isRecipeObject(obj) {
  if (!isPlainObject(obj)) return false;
  if (obj.type === "FeatureCollection" || obj.type === "Feature") return false;
  return Boolean(obj.dataset_schema || obj.models || obj.spatial_targets || obj.recipe_meta || obj.schema_version);
}

export function validateGeojsonObject(gj, fileName = "GeoJSON") {
  if (!isPlainObject(gj)) {
    throw new Error(`Érvénytelen GeoJSON (${fileName}): JSON objektum szükséges.`);
  }
  if (gj.type !== "FeatureCollection" && gj.type !== "Feature") {
    throw new Error(`Érvénytelen GeoJSON (${fileName}): FeatureCollection vagy Feature típus szükséges.`);
  }
  const feats = gj.type === "FeatureCollection" ? (Array.isArray(gj.features) ? gj.features : []) : [gj];
  if (!feats.length) {
    throw new Error(`A GeoJSON fájl (${fileName}) nem tartalmaz Feature elemeket.`);
  }
  const hasValidGeom = feats.some(
    (f) =>
      f &&
      typeof f === "object" &&
      f.geometry &&
      typeof f.geometry === "object" &&
      typeof f.geometry.type === "string" &&
      (Array.isArray(f.geometry.coordinates) || Array.isArray(f.geometry.geometries))
  );
  if (!hasValidGeom) {
    throw new Error(`A GeoJSON fájl (${fileName}) nem tartalmaz érvényes geometriát.`);
  }
  return feats;
}

export function classifyIncomingJson(parsed, fileName = "", { hasDataset = false, batchHasDataset = false } = {}) {
  if (!isPlainObject(parsed)) {
    throw new Error(`Érvénytelen JSON objektum (${fileName || "fájl"}): gyökérobjektum szükséges.`);
  }
  if (parsed.kind === "geospec_workspace") {
    return { kind: "workspace", fileName, data: parsed };
  }
  if (isOsmBundleObject(parsed)) {
    return { kind: "osm_bundle", fileName, data: parsed };
  }
  if (parsed.type === "FeatureCollection" || parsed.type === "Feature") {
    const feats = validateGeojsonObject(parsed, fileName || "GeoJSON");
    const hasNonPoint = feats.some((f) => f?.geometry && f.geometry.type !== "Point");
    if (hasDataset || batchHasDataset || hasNonPoint) {
      return { kind: "target_geojson", fileName, data: parsed };
    }
    return { kind: "dataset_geojson", fileName, data: parsed };
  }
  if (isRecipeObject(parsed)) {
    return { kind: "recipe", fileName, data: parsed };
  }
  throw new Error(
    `Ismeretlen JSON fájlformátum (${fileName || "fájl"}): nem GeoSpec recept, nem projekt (.geospec.json), nem OSM csomag és nem GeoJSON.`
  );
}

const BATCH_PRIORITY = {
  workspace: 0,
  dataset_csv: 1,
  dataset_geojson: 1,
  recipe: 2,
  osm_bundle: 3,
  target_geojson: 4,
};

export function prepareIncomingBatch(entries, { hasDataset = false } = {}) {
  const list = Array.isArray(entries) ? entries : [];
  const prelim = [];
  let batchHasDataset = false;

  for (let idx = 0; idx < list.length; idx++) {
    const fileName = String(list[idx]?.fileName || "");
    const text = stripBom(list[idx]?.text ?? "");
    const lower = fileName.toLowerCase();

    if (lower.endsWith(".csv") || lower.endsWith(".tsv") || lower.endsWith(".txt")) {
      batchHasDataset = true;
      prelim.push({ idx, kind: "dataset_csv", fileName, text });
      continue;
    }

    let parsed;
    try {
      parsed = parseJsonText(text, fileName || "JSON");
    } catch (err) {
      if (lower.endsWith(".json") || lower.endsWith(".geojson")) {
        throw new Error(`Érvénytelen JSON fájl (${fileName}): ${err.message}`);
      }
      throw new Error(`Nem támogatott fájlformátum (${fileName}): csak .csv, .tsv, .txt, .json vagy .geojson tölthető be.`);
    }

    if (!isPlainObject(parsed)) {
      throw new Error(`Érvénytelen JSON objektum (${fileName}): gyökérobjektum szükséges.`);
    }

    if (parsed.kind === "geospec_workspace") {
      if (parsed.dataset && typeof parsed.dataset === "object" && parsed.dataset.columns) {
        batchHasDataset = true;
      }
      prelim.push({ idx, kind: "workspace", fileName, data: parsed });
    } else if (parsed.type === "FeatureCollection" || parsed.type === "Feature") {
      prelim.push({ idx, kind: "geojson_pending", fileName, data: parsed });
    } else {
      const classified = classifyIncomingJson(parsed, fileName, { hasDataset, batchHasDataset });
      prelim.push({ idx, ...classified });
    }
  }

  let datasetClaimed = Boolean(hasDataset || batchHasDataset);
  const resolved = prelim.map((item) => {
    if (item.kind !== "geojson_pending") return item;
    const classified = classifyIncomingJson(item.data, item.fileName, {
      hasDataset,
      batchHasDataset: datasetClaimed,
    });
    if (classified.kind === "dataset_geojson") {
      datasetClaimed = true;
    }
    return { idx: item.idx, ...classified };
  });

  resolved.sort((a, b) => {
    const pa = BATCH_PRIORITY[a.kind] ?? 99;
    const pb = BATCH_PRIORITY[b.kind] ?? 99;
    return pa !== pb ? pa - pb : a.idx - b.idx;
  });

  return resolved;
}
