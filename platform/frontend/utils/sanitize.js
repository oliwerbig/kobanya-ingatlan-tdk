/* Shared HTML Sanitization, Numeric Normalization, Recipe Guard & Plotly Theme Helpers */

export function escapeHtml(val) {
  if (val === null || val === undefined) return "";
  return String(val)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function normalizeNumericString(raw) {
  const s = String(raw ?? "").trim().replace(/\u00a0/g, "").replace(/\s+/g, "");
  if (!s) return "";
  const hasComma = s.includes(",");
  const hasDot = s.includes(".");
  if (hasComma && hasDot) {
    if (s.lastIndexOf(".") > s.lastIndexOf(",")) {
      return s.replace(/,/g, "");
    }
    return s.replace(/\./g, "").replace(",", ".");
  }
  if ((s.match(/,/g) || []).length > 1) return s.replace(/,/g, "");
  if ((s.match(/\./g) || []).length > 1) return s.replace(/\./g, "");
  if (hasComma) return s.replace(",", ".");
  return s;
}

export function parseNumericScalar(val) {
  if (val === null || val === undefined || typeof val === "boolean") return NaN;
  if (typeof val === "number") return Number.isFinite(val) ? val : NaN;
  const s = String(val).trim();
  if (!s || ["na", "nan", "null", "none", "n/a", "-"].includes(s.toLowerCase())) return NaN;
  const norm = normalizeNumericString(s);
  if (!norm) return NaN;
  const num = Number(norm);
  return Number.isFinite(num) ? num : NaN;
}

export function stripBom(text) {
  return String(text ?? "").replace(/^\uFEFF/, "");
}

export function parseJsonText(text, label = "JSON") {
  const clean = stripBom(text).trim();
  if (!clean) {
    throw new Error(`A ${label} fájl üres.`);
  }
  return JSON.parse(clean);
}

export function stripUnsafeKeys(obj) {
  if (obj === null || typeof obj !== "object") return obj;
  if (Array.isArray(obj)) return obj.map((v) => stripUnsafeKeys(v));
  const out = {};
  for (const [k, v] of Object.entries(obj)) {
    if (k === "__proto__" || k === "constructor" || k === "prototype") continue;
    out[k] = stripUnsafeKeys(v);
  }
  return out;
}

export function sanitizeRecipe(loaded, defaults) {
  if (!loaded || typeof loaded !== "object" || Array.isArray(loaded)) {
    throw new Error("Érvénytelen recept formátum (JSON objektum szükséges).");
  }
  const src = stripUnsafeKeys(loaded);
  const cleanStr = (v, fallback = "") => (typeof v === "string" ? v : v == null ? fallback : String(v));
  const cleanStrArr = (arr) => (Array.isArray(arr) ? arr.map((x) => cleanStr(x)).filter(Boolean) : []);
  const safeObj = (v) => (v && typeof v === "object" && !Array.isArray(v) ? v : {});
  const clampNum = (v, lo, hi, fallback) => {
    if (v === null || v === undefined || v === "") return fallback;
    const n = Number(v);
    if (!Number.isFinite(n)) return fallback;
    return Math.min(hi, Math.max(lo, n));
  };

  const dsIn = safeObj(src.dataset_schema);
  const yIn = safeObj(dsIn.y_spec);
  const clIn = safeObj(dsIn.cluster_spec);
  const selIn = safeObj(dsIn.selection_spec);
  const spIn = safeObj(src.spatial_engine);
  const gwrIn = safeObj(spIn.gwr);
  const ccIn = safeObj(src.composition_chain);
  const grdIn = safeObj(src.guards_cfg);

  const dDs = defaults.dataset_schema;
  const dSp = defaults.spatial_engine;

  const models = Array.isArray(src.models) && src.models.length
    ? src.models.map((m, idx) => {
        const causalIn = m?.causal_spec && typeof m.causal_spec === "object" && !Array.isArray(m.causal_spec) ? m.causal_spec : null;
        return {
          id: cleanStr(m?.id, `spec_${idx + 1}`),
          name: cleanStr(m?.name, `Spec ${idx + 1}`),
          active: m?.active !== false,
          covariates: Array.isArray(m?.covariates) ? m.covariates.filter((c) => c && typeof c === "object").map((c) => ({ ...c })) : [],
          spatial_terms: Array.isArray(m?.spatial_terms)
            ? m.spatial_terms
                .filter((st) => st && typeof st === "object")
                .map((st) => ({
                  ...st,
                  ...(Array.isArray(st.breakpoints_m)
                    ? { breakpoints_m: st.breakpoints_m.map((b) => Number(b)).filter((b) => Number.isFinite(b) && b > 0) }
                    : {}),
                }))
            : [],
          interactions: Array.isArray(m?.interactions) ? m.interactions.filter((it) => it && typeof it === "object").map((it) => ({ ...it })) : [],
          ...(causalIn
            ? {
                causal_spec: {
                  type: ["none", "spatial_did", "boundary_rdd"].includes(causalIn.type) ? causalIn.type : "none",
                  target_id: cleanStr(causalIn.target_id, ""),
                  time_col: cleanStr(causalIn.time_col, ""),
                  post_value: cleanStr(causalIn.post_value, ""),
                  treat_cutoff_m: clampNum(causalIn.treat_cutoff_m, 1, 100000, 300),
                  bandwidth_m: clampNum(causalIn.bandwidth_m, 1, 100000, 600),
                  signed_dist_col: cleanStr(causalIn.signed_dist_col, ""),
                },
              }
            : {}),
        };
      })
    : defaults.models;

  const spatialTargets = Array.isArray(src.spatial_targets)
    ? src.spatial_targets
        .filter((t) => t && typeof t === "object" && !Array.isArray(t))
        .map((t, idx) => ({
          ...t,
          id: cleanStr(t.id, `target_${idx + 1}`),
          name: cleanStr(t.name, cleanStr(t.id, `Target ${idx + 1}`)),
          max_snap_distance_m: clampNum(t.max_snap_distance_m, 10, 50000, 300),
          bbox_buffer_m: clampNum(t.bbox_buffer_m, 100, 50000, 800),
          excluded_osm_ids: cleanStrArr(t.excluded_osm_ids),
        }))
    : [];

  const bcRaw = yIn.boxcox_lambda;
  const bcClean = bcRaw === null || bcRaw === undefined || bcRaw === ""
    ? null
    : (Number.isFinite(Number(bcRaw)) ? Math.min(5, Math.max(-5, Number(bcRaw))) : null);

  return {
    schema_version: cleanStr(src.schema_version, defaults.schema_version),
    ...( ["HC3", "cluster", "OLS"].includes(src.default_cov_type) ? { default_cov_type: src.default_cov_type } : {} ),
    recipe_meta: {
      title: cleanStr(src.recipe_meta?.title, defaults.recipe_meta.title),
      created_utc: cleanStr(src.recipe_meta?.created_utc, new Date().toISOString()),
    },
    dataset_schema: {
      row_id_col: cleanStr(dsIn.row_id_col, dDs.row_id_col),
      lat_col: cleanStr(dsIn.lat_col, dDs.lat_col),
      lon_col: cleanStr(dsIn.lon_col, dDs.lon_col),
      y_spec: {
        ...dDs.y_spec,
        ...yIn,
        mode: ["column", "ratio"].includes(yIn.mode) ? yIn.mode : dDs.y_spec.mode,
        transform: ["log", "raw", "log1p", "boxcox"].includes(yIn.transform) ? yIn.transform : dDs.y_spec.transform,
        boxcox_lambda: bcClean,
        log1p_shift: clampNum(yIn.log1p_shift, 0, 1e12, dDs.y_spec.log1p_shift ?? 1.0),
      },
      cluster_spec: {
        ...dDs.cluster_spec,
        ...clIn,
        mode: ["coordinate_radius", "column", "auto_or_column", "none"].includes(clIn.mode) ? clIn.mode : dDs.cluster_spec.mode,
        col: cleanStr(clIn.col, dDs.cluster_spec.col || ""),
        radius_m: clampNum(clIn.radius_m, 1, 100000, dDs.cluster_spec.radius_m),
      },
      selection_spec: {
        ...dDs.selection_spec,
        ...selIn,
        first_stage_covariates: cleanStrArr(selIn.first_stage_covariates),
        exclusion_restriction_cols: cleanStrArr(selIn.exclusion_restriction_cols),
      },
      spatial_proxy_cols: cleanStrArr(dsIn.spatial_proxy_cols),
    },
    spatial_targets: spatialTargets,
    models,
    composition_chain: {
      ...defaults.composition_chain,
      ...ccIn,
      blocks: Array.isArray(ccIn.blocks) ? ccIn.blocks.map((b) => ({ name: cleanStr(b?.name), terms: cleanStrArr(b?.terms) })) : [],
    },
    spatial_engine: {
      ...dSp,
      ...spIn,
      se_mode: ["qml", "ml"].includes(spIn.se_mode) ? spIn.se_mode : (dSp.se_mode || "qml"),
      krinsky_robb_draws: Math.round(clampNum(spIn.krinsky_robb_draws, 100, 10000, dSp.krinsky_robb_draws ?? 1000)),
      weights: Array.isArray(spIn.weights) && spIn.weights.length
        ? spIn.weights
            .filter((w) => w && typeof w === "object")
            .map((w, idx) => ({
              ...w,
              id: cleanStr(w.id, `w_${idx + 1}`),
              type: ["knn", "idw", "network_idw"].includes(w.type) ? w.type : "knn",
              k: Math.round(clampNum(w.k, 1, 200, 5)),
              power: clampNum(w.power, 0.25, 5, 1.0),
              cutoff_m: clampNum(w.cutoff_m, 50, 100000, 1500),
            }))
        : dSp.weights,
      gwr: {
        ...dSp.gwr,
        ...gwrIn,
        mode: ["gwr", "mgwr"].includes(gwrIn.mode) ? gwrIn.mode : (dSp.gwr.mode || "gwr"),
        kappa_thresh: clampNum(gwrIn.kappa_thresh, 1, 10000, dSp.gwr.kappa_thresh),
        b_min_floor: Math.round(clampNum(gwrIn.b_min_floor, 10, 5000, dSp.gwr.b_min_floor ?? 30)),
        prevalence_min_pct: clampNum(gwrIn.prevalence_min_pct, 0, 50, dSp.gwr.prevalence_min_pct ?? 15.0),
      },
    },
    guards_cfg: {
      ...defaults.guards_cfg,
      ...grdIn,
      sparse_min_cell: Math.round(clampNum(grdIn.sparse_min_cell, 1, 200, defaults.guards_cfg.sparse_min_cell)),
      hc3_leverage_warn: clampNum(grdIn.hc3_leverage_warn, 0.05, 0.999, defaults.guards_cfg.hc3_leverage_warn),
      heckman_min_excluded: Math.round(clampNum(grdIn.heckman_min_excluded, 1, 5000, defaults.guards_cfg.heckman_min_excluded)),
      heckman_min_analytic: Math.round(clampNum(grdIn.heckman_min_analytic, 10, 5000, defaults.guards_cfg.heckman_min_analytic)),
    },
  };
}

export function getPlotlyThemeLayout(overrides = {}) {
  const isDark = typeof document !== "undefined" && document.body?.classList?.contains("dark-theme");
  const textCol = isDark ? "#e2e8f0" : "#0f172a";
  const gridCol = isDark ? "#263244" : "#e2e8f0";
  const zeroCol = isDark ? "#475569" : "#94a3b8";

  const base = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: textCol, family: "Inter, sans-serif", size: 11 },
    xaxis: {
      gridcolor: gridCol,
      zerolinecolor: zeroCol,
    },
    yaxis: {
      gridcolor: gridCol,
      zerolinecolor: zeroCol,
    },
  };
  const out = {
    ...base,
    ...overrides,
    font: { ...base.font, ...(overrides.font || {}) },
    xaxis: { ...base.xaxis, ...(overrides.xaxis || {}) },
    yaxis: { ...base.yaxis, ...(overrides.yaxis || {}) },
  };
  if (overrides.yaxis2) {
    out.yaxis2 = {
      gridcolor: "rgba(0,0,0,0)",
      zerolinecolor: zeroCol,
      ...overrides.yaxis2,
    };
  }
  return out;
}
