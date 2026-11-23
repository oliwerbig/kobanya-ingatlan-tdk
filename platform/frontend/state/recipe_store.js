/* ExperimentRecipe Store, RFC-4180 CSV/GeoJSON Parser, Empty Template & Generic Sandbox Study */
import { parseNumericScalar } from "../utils/sanitize.js";

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
        mode: "auto_or_column",
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
        { id: "knn5", type: "knn", k: 5, dedup_colocated: false },
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
  const cleanText = String(text ?? "").replace(/^\uFEFF/, "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
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
  headers.forEach((h) => {
    columns[h] = [];
  });

  for (let r = 1; r < rows.length; r++) {
    const vals = rows[r];
    for (let c = 0; c < headers.length; c++) {
      columns[headers[c]].push(vals[c] !== undefined ? vals[c] : "");
    }
  }

  return { headers, columns, nRows: rows.length - 1 };
}

export function parseGeoJSONDataset(text) {
  const gj = JSON.parse(text);
  const feats = gj.type === "FeatureCollection" ? gj.features || [] : gj.type === "Feature" ? [gj] : [];
  if (!feats.length) throw new Error("A GeoJSON fájl nem tartalmaz Feature elemeket.");

  const propKeys = new Set();
  for (const f of feats) {
    Object.keys(f.properties || {}).forEach((k) => propKeys.add(k));
  }
  const columns = { lat: [], lon: [] };
  propKeys.forEach((k) => {
    columns[k] = [];
  });

  for (const f of feats) {
    let lon = null;
    let lat = null;
    if (f.geometry && f.geometry.type === "Point" && Array.isArray(f.geometry.coordinates)) {
      lon = f.geometry.coordinates[0];
      lat = f.geometry.coordinates[1];
    }
    columns.lat.push(lat === null ? "" : lat);
    columns.lon.push(lon === null ? "" : lon);
    const props = f.properties || {};
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

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randNorm(rng) {
  const u1 = Math.max(1e-12, rng());
  const u2 = rng();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

/**
 * Domain-agnostic offline Sandbox Study generator (~160 obs, synthetic grid street graph,
 * park polygon & transit corridor GeoJSON, + Heckman non-geocoded rows).
 */
export function generateSampleStudy() {
  const rng = mulberry32(202611);
  const lat0 = 47.0;
  const lon0 = 15.0;
  const mPerDegLat = 111132.95;
  const mPerDegLon = 111132.95 * Math.cos((lat0 * Math.PI) / 180);
  const toLatLon = (xM, yM) => [lon0 + xM / mPerDegLon, lat0 + yM / mPerDegLat];

  // 1. Synthetic 9x9 street grid covering [-800m, +800m] x [-800m, +800m] (200m blocks)
  const gridVals = [-800, -600, -400, -200, 0, 200, 400, 600, 800];
  const graphElements = [];
  let nid = 1;
  const nodeGrid = {};
  for (let iy = 0; iy < gridVals.length; iy++) {
    for (let ix = 0; ix < gridVals.length; ix++) {
      const [lo, la] = toLatLon(gridVals[ix], gridVals[iy]);
      nodeGrid[`${ix}_${iy}`] = nid;
      graphElements.push({ type: "node", id: nid, lat: Number(la.toFixed(6)), lon: Number(lo.toFixed(6)) });
      nid++;
    }
  }
  let wid = 1000;
  for (let iy = 0; iy < gridVals.length; iy++) {
    for (let ix = 0; ix < gridVals.length; ix++) {
      if (ix + 1 < gridVals.length) {
        graphElements.push({
          type: "way",
          id: wid++,
          nodes: [nodeGrid[`${ix}_${iy}`], nodeGrid[`${ix + 1}_${iy}`]],
          tags: { highway: "residential" },
        });
      }
      if (iy + 1 < gridVals.length) {
        graphElements.push({
          type: "way",
          id: wid++,
          nodes: [nodeGrid[`${ix}_${iy}`], nodeGrid[`${ix}_${iy + 1}`]],
          tags: { highway: "residential" },
        });
      }
    }
  }
  const streetGraph = { version: 0.6, generator: "GeoSpec-Sandbox", elements: graphElements };

  // 2. Central Park Polygon [-140..140] x [-140..140] + Transit Line & Stops
  const parkRing = [
    toLatLon(-140, -140),
    toLatLon(140, -140),
    toLatLon(140, 140),
    toLatLon(-140, 140),
    toLatLon(-140, -140),
  ];
  const parkGeojson = {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        id: "central_park",
        properties: { name: "Central Urban Park", leisure: "park" },
        geometry: { type: "Polygon", coordinates: [parkRing] },
      },
    ],
  };

  const transitGeojson = {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        id: "station_west",
        properties: { name: "West Transit Hub", railway: "station" },
        geometry: { type: "Point", coordinates: toLatLon(-400, 0) },
      },
      {
        type: "Feature",
        id: "station_east",
        properties: { name: "East Transit Hub", railway: "station" },
        geometry: { type: "Point", coordinates: toLatLon(400, 0) },
      },
    ],
  };

  // 3. Synthetic observations (145 geocoded + 15 non-geocoded for Heckman selection check)
  const columns = {
    row_id: [],
    lat: [],
    lon: [],
    price_eur: [],
    area_m2: [],
    rooms: [],
    condition: [],
    renovated: [],
    dist_cbd_m: [],
    listing_quality: [],
  };

  const nTotal = 160;
  for (let i = 0; i < nTotal; i++) {
    const isGeocoded = i < 145;
    const xM = (rng() * 1400) - 700;
    const yM = (rng() * 1400) - 700;
    const [lo, la] = toLatLon(xM, yM);

    const area = Math.max(24, Math.min(130, 58 + 0.015 * xM + randNorm(rng) * 14));
    const rooms = Math.max(1, Math.min(5, Math.round(area / 24 + randNorm(rng) * 0.4)));
    const renov = rng() < (xM > 0 ? 0.52 : 0.28) ? 1 : 0;
    const cond = renov ? "renovated" : rng() < 0.6 ? "good" : "fair";
    const condAdj = cond === "renovated" ? 0.14 : cond === "good" ? 0.0 : -0.11;

    // Distance to Park boundary (inside = 0)
    const dxP = Math.max(0, Math.abs(xM) - 140);
    const dyP = Math.max(0, Math.abs(yM) - 140);
    const dPark = Math.hypot(dxP, dyP);

    // Distance to nearest station
    const dSt = Math.min(Math.hypot(xM + 400, yM), Math.hypot(xM - 400, yM));
    const dCbd = Math.hypot(xM - 950, yM - 650);
    const qual = Number((0.4 + 0.6 * (isGeocoded ? 1 : 0) + randNorm(rng) * 0.4).toFixed(3));

    // Spatially varying park amenity + transit proximity + mild gradient
    const adPark = Math.max(0, Math.log(400 / Math.max(1, dPark)));
    const adTransit = Math.max(0, Math.log(500 / Math.max(1, dSt)));
    const parkSlope = 0.05 + 0.04 * ((xM + 700) / 1400);
    const lnUnitPrice =
      7.45 +
      parkSlope * adPark +
      0.06 * adTransit +
      condAdj +
      0.002 * area -
      0.00012 * dCbd +
      randNorm(rng) * 0.075;
    const totalPrice = Math.round(Math.exp(lnUnitPrice) * area);

    columns.row_id.push(i + 1);
    columns.lat.push(isGeocoded ? la.toFixed(6) : "");
    columns.lon.push(isGeocoded ? lo.toFixed(6) : "");
    columns.price_eur.push(totalPrice);
    columns.area_m2.push(Number(area.toFixed(1)));
    columns.rooms.push(rooms);
    columns.condition.push(cond);
    columns.renovated.push(renov);
    columns.dist_cbd_m.push(Math.round(dCbd));
    columns.listing_quality.push(qual);
  }

  const recipe = createDefaultRecipe();
  recipe.recipe_meta.title = "GeoSpec Sandbox — Synthetic Urban Study";
  recipe.dataset_schema = {
    row_id_col: "row_id",
    lat_col: "lat",
    lon_col: "lon",
    y_spec: { mode: "ratio", num_col: "price_eur", den_col: "area_m2", transform: "log", boxcox_lambda: null, log1p_shift: 1.0 },
    cluster_spec: { mode: "auto_or_column", col: "", regex: "", radius_m: 35.0 },
    selection_spec: {
      enabled: true,
      excluded_flag_col: "",
      first_stage_covariates: ["area_m2", "rooms"],
      exclusion_restriction_cols: ["listing_quality"],
    },
    spatial_proxy_cols: ["dist_cbd_m"],
  };
  recipe.spatial_targets = [
    {
      id: "park",
      name: "Central Park (GeoJSON)",
      source: "geojson",
      excluded_osm_ids: [],
      geometry_handling: "polygon_boundary",
      inside_is_zero: true,
      distance_modes: "both",
      routing_profile: "walk",
      max_snap_distance_m: 300,
    },
    {
      id: "transit",
      name: "Transit Hubs (GeoJSON)",
      source: "geojson",
      excluded_osm_ids: [],
      geometry_handling: "point",
      inside_is_zero: false,
      distance_modes: "both",
      routing_profile: "walk",
      max_snap_distance_m: 300,
    },
  ];
  const baseCovs = [
    { col: "area_m2", type: "numeric", transform: "raw" },
    { col: "rooms", type: "numeric", transform: "raw" },
    { col: "condition", type: "categorical", reference_level: "good" },
  ];
  recipe.models = [
    { id: "m1_base", name: "M1: Fizikai alapmodell", active: true, covariates: baseCovs.map((c) => ({ ...c })), spatial_terms: [], interactions: [] },
    {
      id: "m2_rings",
      name: "M2: Távolsági sávok",
      active: true,
      covariates: baseCovs.map((c) => ({ ...c })),
      spatial_terms: [
        { target_id: "park", metric: "eucl", transform: "rings", breakpoints_m: [150, 300, 500] },
        { target_id: "transit", metric: "net", transform: "rings", breakpoints_m: [200, 400, 650] },
      ],
      interactions: [],
    },
    {
      id: "m3_log",
      name: "M3: Folytonos ln(d)",
      active: true,
      covariates: baseCovs.map((c) => ({ ...c })),
      spatial_terms: [
        { target_id: "park", metric: "eucl", transform: "ln_d", floor_m: 1.0 },
        { target_id: "transit", metric: "net", transform: "ln_d", floor_m: 1.0 },
      ],
      interactions: [],
    },
    {
      id: "m4_ad",
      name: "M4: Andersson–Day",
      active: true,
      covariates: baseCovs.map((c) => ({ ...c })),
      spatial_terms: [
        { target_id: "park", metric: "eucl", transform: "andersson_day", floor_m: 1.0, cutoff_m: 400, cutoff_range_m: [150, 650, 25] },
        { target_id: "transit", metric: "net", transform: "andersson_day", floor_m: 1.0, cutoff_m: 500, cutoff_range_m: [200, 750, 25] },
      ],
      interactions: [],
    },
  ];
  recipe.composition_chain = {
    enabled: true,
    focal_term_ref: "park::andersson_day",
    blocks: [
      { name: "Méret & Szobák", terms: ["area_m2", "rooms"] },
      { name: "Műszaki állapot", terms: ["condition"] },
    ],
    compute_gelbach_ovb: true,
  };
  recipe.spatial_engine.active_model_id = "m4_ad";
  recipe.spatial_engine.gwr.focal_model_id = "m4_ad";

  const headers = Object.keys(columns);
  const { profile, guesses } = inferColumnProfile(headers, columns);

  return {
    fileName: "sandbox_urban_study.csv",
    parsed: { headers, columns, nRows: nTotal },
    profile,
    guesses,
    recipe,
    geojsonTargets: {
      park: parkGeojson,
      transit: transitGeojson,
    },
    streetGraphs: {
      walk: streetGraph,
    },
  };
}
