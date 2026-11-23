/* ExperimentRecipe Store, RFC-4180 CSV/GeoJSON Parser & Empty Template Builder */

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
  const cleanText = text.replace(/^\uFEFF/, "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const lines = cleanText.split("\n").filter((l) => l.trim().length > 0);
  if (lines.length < 2) throw new Error("A CSV fájlnak legalább fejlécet és 1 adatsort kell tartalmaznia.");

  const delim = detectDelimiter(lines[0]);
  const parseLine = (line) => {
    const row = [];
    let cur = "";
    let inQ = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (inQ && line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQ = !inQ;
        }
      } else if (ch === delim && !inQ) {
        row.push(cur.trim());
        cur = "";
      } else {
        cur += ch;
      }
    }
    row.push(cur.trim());
    return row;
  };

  const headers = parseLine(lines[0]).map((h, idx) => h || `col_${idx}`);
  const columns = {};
  headers.forEach((h) => {
    columns[h] = [];
  });

  for (let r = 1; r < lines.length; r++) {
    const vals = parseLine(lines[r]);
    for (let c = 0; c < headers.length; c++) {
      columns[headers[c]].push(vals[c] !== undefined ? vals[c] : "");
    }
  }

  return { headers, columns, nRows: lines.length - 1 };
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
      const norm = s.replace(/\u00a0/g, "").replace(/\s/g, "").replace(",", ".");
      if (!Number.isNaN(Number(norm)) && Number.isFinite(Number(norm))) numCount++;
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
  const lonGuess = findMatch([/^lon$/, /^lng$/, /wgs84_lon/, /longitude/, /x_coord/, /hosszusag/]);
  const idGuess = findMatch([/^id$/, /row_id/, /listing_id/, /obj_id/]);
  const yGuess =
    findMatch([/fajlagos/, /price_m2/, /unit_price/, /^price$/, /^ar$/, /rent/, /^y$/]) ||
    headers.find((h) => h !== latGuess && h !== lonGuess && h !== idGuess && profile[h].isNumeric) ||
    "";

  return { profile, guesses: { latGuess, lonGuess, idGuess, yGuess } };
}

export function generateEmptyCSVTemplate() {
  return [
    "id,lat,lon,price,area_m2,condition,rooms,street_name,dist_cbd_m",
    "# Példa fejléc (töltsd fel saját megfigyeléseiddel; a '#' kezdetű megjegyzéssorok elhagyhatók feltöltés előtt)",
  ][0] + "\n";
}
