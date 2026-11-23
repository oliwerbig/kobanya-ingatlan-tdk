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

export function sanitizeRecipe(loaded, defaults) {
  if (!loaded || typeof loaded !== "object" || Array.isArray(loaded)) {
    throw new Error("Érvénytelen recept formátum (JSON objektum szükséges).");
  }
  const cleanStr = (v, fallback = "") => (typeof v === "string" ? v : v == null ? fallback : String(v));
  const cleanStrArr = (arr) => (Array.isArray(arr) ? arr.map((x) => cleanStr(x)).filter(Boolean) : []);
  const safeObj = (v) => (v && typeof v === "object" && !Array.isArray(v) ? v : {});

  const dsIn = safeObj(loaded.dataset_schema);
  const yIn = safeObj(dsIn.y_spec);
  const clIn = safeObj(dsIn.cluster_spec);
  const selIn = safeObj(dsIn.selection_spec);
  const spIn = safeObj(loaded.spatial_engine);
  const gwrIn = safeObj(spIn.gwr);
  const ccIn = safeObj(loaded.composition_chain);
  const grdIn = safeObj(loaded.guards_cfg);

  const dDs = defaults.dataset_schema;
  const dSp = defaults.spatial_engine;

  const models = Array.isArray(loaded.models) && loaded.models.length
    ? loaded.models.map((m, idx) => ({
        id: cleanStr(m?.id, `spec_${idx + 1}`),
        name: cleanStr(m?.name, `Spec ${idx + 1}`),
        active: m?.active !== false,
        covariates: Array.isArray(m?.covariates) ? m.covariates.filter((c) => c && typeof c === "object").map((c) => ({ ...c })) : [],
        spatial_terms: Array.isArray(m?.spatial_terms) ? m.spatial_terms.filter((st) => st && typeof st === "object").map((st) => ({ ...st })) : [],
        interactions: Array.isArray(m?.interactions) ? m.interactions.filter((it) => it && typeof it === "object").map((it) => ({ ...it })) : [],
      }))
    : defaults.models;

  const spatialTargets = Array.isArray(loaded.spatial_targets)
    ? loaded.spatial_targets
        .filter((t) => t && typeof t === "object" && !Array.isArray(t))
        .map((t, idx) => ({
          ...t,
          id: cleanStr(t.id, `target_${idx + 1}`),
          name: cleanStr(t.name, cleanStr(t.id, `Target ${idx + 1}`)),
          excluded_osm_ids: cleanStrArr(t.excluded_osm_ids),
        }))
    : [];

  return {
    schema_version: cleanStr(loaded.schema_version, defaults.schema_version),
    recipe_meta: {
      title: cleanStr(loaded.recipe_meta?.title, defaults.recipe_meta.title),
      created_utc: cleanStr(loaded.recipe_meta?.created_utc, new Date().toISOString()),
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
      },
      cluster_spec: {
        ...dDs.cluster_spec,
        ...clIn,
        radius_m: Number.isFinite(Number(clIn.radius_m)) ? Number(clIn.radius_m) : dDs.cluster_spec.radius_m,
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
      weights: Array.isArray(spIn.weights) && spIn.weights.length ? spIn.weights.map((w) => ({ ...w })) : dSp.weights,
      gwr: {
        ...dSp.gwr,
        ...gwrIn,
        kappa_thresh: Number.isFinite(Number(gwrIn.kappa_thresh)) ? Number(gwrIn.kappa_thresh) : dSp.gwr.kappa_thresh,
      },
    },
    guards_cfg: {
      ...defaults.guards_cfg,
      ...grdIn,
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
      ...(overrides.xaxis || {}),
    },
    yaxis: {
      gridcolor: gridCol,
      zerolinecolor: zeroCol,
      ...(overrides.yaxis || {}),
    },
  };
  if (overrides.yaxis2) {
    base.yaxis2 = {
      gridcolor: "rgba(0,0,0,0)",
      zerolinecolor: zeroCol,
      ...overrides.yaxis2,
    };
  }
  return { ...overrides, ...base };
}
