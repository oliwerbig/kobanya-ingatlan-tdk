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
  ["Starting Pyodide...", "Pyodide indítása..."],
  ["Loading WebAssembly Python runtime...", "WebAssembly Python futtatókörnyezet betöltése..."],
  ["Loading NumPy & SciPy kernels...", "NumPy és SciPy kernelek betöltése..."],
  ["Mounting GeoSpec py_engine...", "GeoSpec py_engine modulok betöltése..."],
  ["Pyodide NumPy+SciPy kernel ready", "Pyodide NumPy+SciPy kernel kész"],
  ["WASM engine error", "WASM motor hiba"],
  ["Classic OLS (No spatial dependence)", "Klasszikus OLS (nincs szignifikáns térbeli függőség)"],
  ["SAR (Spatial Lag)", "SAR (Térbeli késleltetés)"],
  ["SEM (Spatial Error)", "SEM (Térbeli hiba)"],
  ["SAR (Robust LM-Lag dominant)", "SAR (Robusztus LM-Lag domináns)"],
  ["SEM (Robust LM-Error dominant)", "SEM (Robusztus LM-Error domináns)"],
  ["SDM (Neither SAR nor SEM restriction accepted)", "SDM javasolt (sem SAR, sem SEM redukció nem elfogadható)"],
  ["SAR accepted (H0: θ = 0 not rejected)", "SAR elfogadva (H0: θ = 0 nem elvethető)"],
  ["SEM accepted (H0: θ + ρβ = 0 not rejected)", "SEM elfogadva (H0: θ + ρβ = 0 nem elvethető)"],
  ["Either SAR/SEM acceptable", "SAR vagy SEM redukció is elfogadható"],
  ["SDM underdetermined", "SDM alulhatározott (kevés megfigyelés K_sdm-hez)"],
  ["Complete or minimal exclusion sample", "Teljes vagy minimális kizárási minta"],
  [
    "Profile-MLE cutoff c* maximizes likelihood over the grid; standard errors at c* treat c as fixed (post-selection inference caveat applies; inspect the 95% profile-likelihood CI).",
    "A Profil-MLE c* küszöb maximalizálja a log-likelihoodot a rácson; a c* pontbeli standard hibák fixként kezelik a c küszöböt (poszt-szelekciós fenntartás: vizsgáld a 95%-os profil-likelihood CI-t).",
  ],
];

const VERDICT_REGEX_HU = [
  [
    /^Guard #6 pre-screened (\d+) low-prevalence dummies before GWR:\s*(.+)$/,
    (_, cnt, rest) => {
      const cleaned = rest.replace(/\(min cell (\d+) < (\d+)\)/g, "(min. cella $1 < $2)");
      return `Guard #6: ${cnt} alacsony gyakoriságú dummy változó kiszűrve a GWR előtt: ${cleaned}`;
    },
  ],
  [
    /^Guard #6 locally excluded invariant dummy variables at (\d+)\/(\d+) locations.*$/,
    (_, nLoc, nTot) =>
      `Guard #6: lokálisan invariáns dummy változók automatikusan kizárva ${nLoc}/${nTot} helyszínen (torzításmentes helyi becslés, Ridge-büntetés nélkül).`,
  ],
  [
    /^Guard #6 Selective Ridge applied at (\d+)\/(\d+) \(([^)]+)\) locations due to continuous local collinearity \(κ > ([^)]+)\)\.?$/,
    (_, nLoc, nTot, pct, kappa) =>
      `Guard #6 Szelektív Ridge alkalmazva ${nLoc}/${nTot} (${pct}) helyszínen folytonos lokális kollinearitás miatt (κ > ${kappa}).`,
  ],
  [
    /^Complete or minimal exclusion sample \(N_excluded=(\d+) < (\d+)\)\.?$/,
    (_, nEx, thr) => `Teljes vagy minimális kizárási minta (kizárt N=${nEx} < ${thr}).`,
  ],
  [
    /^Small cluster count \(G=(\d+) < 30\);.*$/,
    (_, g) =>
      `Kis klaszterszám (G=${g} < 30); a klaszterezett SE alulbecsülheti a szóródást (használd a Wild Cluster Bootstrap p-értékeket).`,
  ],
  [
    /^SDM underdetermined \(N=(\d+) <= K_sdm\+2=(\d+)\); rely on Anselin LM \(([^)]+)\)\.?$/,
    (_, n, kPlus, rec) => `SDM alulhatározott (N=${n} ≤ K_sdm+2=${kPlus}); az Anselin LM (${rec}) irányadó.`,
  ],
  [
    /^SDM underdetermined for N=(\d+) and K_sdm=(\d+);.*$/,
    (_, n, k) => `SDM alulhatározott (N=${n}, K_sdm=${k}); az SDM becslés és spillover hatásfelbontás kihagyva.`,
  ],
  [
    /^Guard #2 pooled sparse levels (.+) \(N=(\d+)\) of '(.+)' into '(.+)'\.?$/,
    (_, lvls, n, col, other) => `Guard #2: a '${col}' változó ritka szintjei (${lvls}, N=${n}) összevonva: '${other}'.`,
  ],
  [
    /^Guard #2 merged tiny levels (.+) \(sum N=(\d+) < (\d+)\) of '(.+)' into reference '(.+)'\.?$/,
    (_, lvls, n, tau, col, ref) => `Guard #2: a '${col}' változó ritka szintjei (${lvls}, N=${n} < ${tau}) a '${ref}' referenciába olvasztva.`,
  ],
  [
    /^Guard #2 dropped sparse binary dummy '(.+)' \(N1=(\d+), N0=(\d+) < (\d+)\)\.?$/,
    (_, col, n1, n0, tau) => `Guard #2: a ritka '${col}' bináris dummy elhagyva (N1=${n1}, N0=${n0} < ${tau}).`,
  ],
  [
    /^Guard #2 sparse reference fallback on '(.+)': requested reference '(.+)' \(N=(\d+) < (\d+)\) replaced by modal level '(.+)' \(N=(\d+)\)\.?$/,
    (_, col, req, nReq, tau, mod, nMod) =>
      `Guard #2: a '${col}' ritka '${req}' referenciája (N=${nReq} < ${tau}) helyett a modális '${mod}' (N=${nMod}) lett a referencia.`,
  ],
  [
    /^Co-located coordinates: ([\d.]+)% of (\S+) neighbor links.*$/,
    (_, pct, wid) =>
      `Co-located koordináták: a '${wid}' szomszédsági élek ${pct}%-a d_ij ≤ 0.5 m (épületen belüli klaszter). Ellenőrizd a stabilitást az IDW mátrixszal.`,
  ],
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
  for (const [rx, fn] of VERDICT_REGEX_HU) {
    const m = s.match(rx);
    if (m) return fn(...m);
  }
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
