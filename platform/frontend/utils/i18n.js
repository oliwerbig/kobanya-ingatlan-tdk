/* Full HU/EN i18n dictionary, DOM updater & backend verdict translator */

export const SUPPORTED_LANGS = ["hu", "en"];

const DICT = {
  hu: {
    "app.brand_sub": "Parametrikus Térökonometriai Munkaállomás",
    "nav.run_all": "▶ Teljes Pipeline Futtatása (1–5)",
    "nav.load": "📥 Betöltés",
    "nav.save": "💾 Mentés ▾",
    "nav.save_recipe": "💾 Recept (_recipe.json)",
    "nav.save_project": "💾 Teljes Projekt (.geospec.json)",
    "nav.save_osm": "🌐 Offline OSM Csomag (.osm_bundle.json)",
    "nav.save_geojson": "🗺️ QGIS Gazdagított GeoJSON",
    "nav.save_latex": "📋 Overleaf LaTeX Másolása",
    "nav.save_zip": "📦 Publikációs .ZIP Csomag",
    "nav.lang_btn": "HU | EN",
    "stage.1": "1. Adat & OSM Célpontok",
    "stage.2": "2. Leíró & Gelbach",
    "stage.3": "3. Modell-Műhely & Diagnosztika",
    "stage.4": "4. Térökonometria (SAR/SEM/SDM)",
    "stage.5": "5. Adaptív GWR/MGWR & Export",
    "tab.stale": "Elavult",
    "worker.ready": "Pyodide NumPy+SciPy kernel kész",
    "worker.booting": "Pyodide indítása...",
    "worker.error": "WASM motor hiba",
    "worker.crash_msg": "A Pyodide WASM számítási motor leállt vagy hibára futott.",
    "worker.restart": "↻ WASM Újraindítása",
    "map.layer_lbl": "Térképi réteg:",
    "map.layer_y": "Megfigyelt célváltozó (y)",
    "map.layer_resid": "OLS Reziduumok (aktív modell)",
    "map.layer_lisa": "LISA Klaszterek (HH/LL/HL/LH)",
    "map.layer_gwr_beta": "GWR Lokális β̂ᵢ (halványított nem-szign.)",
    "map.layer_gwr_r2": "GWR Lokális R²",
    "map.layer_snap": "Hálózati illesztés (Snap / OSM BBox)",
    "map.gwr_mask_fdr": "BY-FDR (p_adj < 0.05)",
    "map.gwr_mask_nom": "Nominális (|t| ≥ t_crit)",
    "map.route_tgt": "Útvonal célpont:",
    "footer.tech": "Pure NumPy + SciPy WASM | 100% kliensoldali (nincs feltöltés)",
    "drawer.title": "Módszertani Őrszemek & Haladó Beállítások (G1–G6)",
    "preset.best": "Best-Practice (QML + Helyi GWR szűrés)",
    "preset.replication": "Replication (HC3 + Tanulmány replikáció)",
    "drawer.apply": "✓ Beállítások alkalmazása",
    "toast.close": "Bezárás",
    "toast.copied_latex": "Overleaf LaTeX táblázat a vágólapra másolva!",
  },
  en: {
    "app.brand_sub": "Parametric Spatial Econometric Workbench",
    "nav.run_all": "▶ Run Full Pipeline (1–5)",
    "nav.load": "📥 Load",
    "nav.save": "💾 Save ▾",
    "nav.save_recipe": "💾 Recipe (_recipe.json)",
    "nav.save_project": "💾 Full Workspace (.geospec.json)",
    "nav.save_osm": "🌐 Offline OSM Bundle (.osm_bundle.json)",
    "nav.save_geojson": "🗺️ QGIS Enriched GeoJSON",
    "nav.save_latex": "📋 Copy Overleaf LaTeX",
    "nav.save_zip": "📦 Publication Bundle (.zip)",
    "nav.lang_btn": "EN | HU",
    "stage.1": "1. Data & OSM Targets",
    "stage.2": "2. Descriptives & Gelbach",
    "stage.3": "3. Model Composer & Diagnostics",
    "stage.4": "4. Spatial Econometrics (SAR/SEM/SDM)",
    "stage.5": "5. Adaptive GWR/MGWR & Export",
    "tab.stale": "Stale",
    "worker.ready": "Pyodide NumPy+SciPy kernel ready",
    "worker.booting": "Starting Pyodide...",
    "worker.error": "WASM engine error",
    "worker.crash_msg": "Pyodide WASM compute worker stopped or crashed.",
    "worker.restart": "↻ Restart WASM",
    "map.layer_lbl": "Map layer:",
    "map.layer_y": "Observed Target (y)",
    "map.layer_resid": "OLS Residuals (active model)",
    "map.layer_lisa": "LISA Clusters (HH/LL/HL/LH)",
    "map.layer_gwr_beta": "GWR Local β̂ᵢ (faded non-sig.)",
    "map.layer_gwr_r2": "GWR Local R²",
    "map.layer_snap": "Network Snapping (Snap / OSM BBox)",
    "map.gwr_mask_fdr": "BY-FDR (p_adj < 0.05)",
    "map.gwr_mask_nom": "Nominal (|t| ≥ t_crit)",
    "map.route_tgt": "Route target:",
    "footer.tech": "Pure NumPy + SciPy WASM | 100% Client-Side (Zero-Upload)",
    "drawer.title": "Methodological Guards & Advanced Parameters (G1–G6)",
    "preset.best": "Best-Practice (QML + Local GWR Exclusion)",
    "preset.replication": "Replication (HC3 + Paper Preset)",
    "drawer.apply": "✓ Apply Settings",
    "toast.close": "Close",
    "toast.copied_latex": "Overleaf LaTeX table copied to clipboard!",
  },
};

const VERDICT_MAP_HU = [
  ["Loading WebAssembly Python runtime...", "WebAssembly Python futtatókörnyezet betöltése..."],
  ["Loading NumPy & SciPy kernels...", "NumPy és SciPy kernelek betöltése..."],
  ["Mounting GeoSpec py_engine...", "GeoSpec py_engine modulok betöltése..."],
  ["Pyodide NumPy+SciPy kernel ready", "Pyodide NumPy+SciPy kernel kész"],
  ["Classic OLS (No spatial dependence)", "Klasszikus OLS (nincs szignifikáns térbeli függőség)"],
  ["SDM (Neither SAR nor SEM restriction accepted)", "SDM javasolt (sem SAR, sem SEM redukció nem elfogadható)"],
  ["SAR accepted (H0: θ = 0 not rejected)", "SAR elfogadva (H0: θ = 0 nem elvethető)"],
  ["SEM accepted (H0: θ + ρβ = 0 not rejected)", "SEM elfogadva (H0: θ + ρβ = 0 nem elvethető)"],
  ["Either SAR/SEM acceptable", "SAR vagy SEM redukció is elfogadható"],
  ["SDM underdetermined", "SDM alulhatározott (kevés megfigyelés K_sdm-hez)"],
  ["Complete or minimal exclusion sample", "Teljes vagy minimális kizárási minta"],
  ["Profile-MLE cutoff c* maximizes likelihood", "Profil-MLE c* küszöb maximalizálja a log-likelihoodot"],
  ["Guard #2", "Guard #2 (Ritka cella összevonás)"],
  ["Guard #6", "Guard #6 (GWR helyi stabilizálás)"],
];

let currentLang = "hu";

export function getLang() {
  return currentLang;
}

export function setLang(lang) {
  currentLang = lang === "en" ? "en" : "hu";
  try {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem("geospec_lang", currentLang);
    }
  } catch (_) {}
  if (typeof document !== "undefined" && document.documentElement) {
    document.documentElement.lang = currentLang;
  }
  return currentLang;
}

export function initLang() {
  try {
    if (typeof localStorage !== "undefined") {
      const saved = localStorage.getItem("geospec_lang");
      if (saved === "en" || saved === "hu") {
        currentLang = saved;
      }
    }
  } catch (_) {}
  return setLang(currentLang);
}

export function t(key, params = null) {
  const table = DICT[currentLang] || DICT.hu;
  let s = table[key] ?? DICT.hu[key] ?? key;
  if (params && typeof params === "object") {
    for (const [k, v] of Object.entries(params)) {
      s = s.replaceAll(`{${k}}`, String(v ?? ""));
    }
  }
  return s;
}

export function translateVerdict(text) {
  const s = String(text ?? "");
  if (currentLang !== "hu") return s;
  for (const [enPrefix, huPrefix] of VERDICT_MAP_HU) {
    if (s === enPrefix) return huPrefix;
    if (s.startsWith(enPrefix)) return huPrefix + s.slice(enPrefix.length);
  }
  return s;
}

export function applyI18nDOM(root = typeof document !== "undefined" ? document : null) {
  if (!root) return;
  for (const el of root.querySelectorAll("[data-i18n]")) {
    const key = el.getAttribute("data-i18n");
    if (key) el.textContent = t(key);
  }
  for (const el of root.querySelectorAll("[data-i18n-title]")) {
    const key = el.getAttribute("data-i18n-title");
    if (key) el.title = t(key);
  }
}
