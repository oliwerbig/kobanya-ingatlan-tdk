/* Stage 6: Pure-NumPy Deterministic GBRT, Building-Cluster GroupKFold CV & Exact Lundberg TreeSHAP */

import { escapeHtml, getPlotlyThemeLayout } from "../utils/sanitize.js";

const GROUP_LABELS_HU = {
  physical: "Fizikai épületminőség",
  rail: "Vasúti infrastruktúra",
  road: "Közúti infrastruktúra",
};

const GROUP_COLORS = {
  physical: "#1d4ed8",
  rail: "#d95338",
  road: "#64748b",
};

export function renderStage6(container, ctx) {
  const { state, actions } = ctx;
  if (!state.datasetInfo) {
    container.innerHTML = `<div class="card"><div class="card-subtitle">Előbb tölts fel adatot az 1. fázisban.</div></div>`;
    return;
  }

  if (!state.recipe.ml_shap) {
    state.recipe.ml_shap = {
      enabled: true,
      focal_model_id: state.recipe.spatial_engine?.gwr?.focal_model_id || state.recipe.models?.[0]?.id || "spec_a",
      n_estimators: 150,
      max_depth: 3,
      learning_rate: 0.05,
      subsample: 0.8,
      min_samples_leaf: 5,
      l2_reg: 1.0,
      cv_folds: 5,
      random_seed: 42,
    };
  }
  const mlCfg = state.recipe.ml_shap;
  const activeModels = (state.recipe.models || []).filter((m) => m.active);
  if (activeModels.length && !activeModels.some((m) => m.id === mlCfg.focal_model_id)) {
    mlCfg.focal_model_id =
      state.recipe.spatial_engine?.gwr?.focal_model_id ||
      state.recipe.spatial_engine?.active_model_id ||
      activeModels[activeModels.length - 1].id;
  }

  const s6 = state.stage6Res;
  const grp = s6?.grouped_importance || {};

  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">6A. Nemlineáris GBRT &amp; Egzakt TreeSHAP (Pure-NumPy WASM Kernel)</h3>
          <div class="card-subtitle">Determinisztikus faegyüttes (fix seed) · Épületklaszteres GroupKFold CV · Egzakt Lundberg TreeSHAP</div>
        </div>
        <button type="button" class="btn btn-primary btn-sm" id="btnRunStage6">▶ Stage 6 ML &amp; TreeSHAP Futtatása</button>
      </div>

      <div class="grid-3">
        <div class="field-row">
          <label for="selMlModel">Fókusz specifikáció (változómátrix)</label>
          <select id="selMlModel">
            ${activeModels.map((m) => `<option value="${escapeHtml(m.id)}" ${m.id === mlCfg.focal_model_id ? "selected" : ""}>${escapeHtml(m.name)}</option>`).join("")}
          </select>
        </div>
        <div class="field-row">
          <label for="inpMlSeed">Determinisztikus Seed &amp; Fák (N<sub>trees</sub> / Mélység)</label>
          <div style="display:flex;gap:6px;">
            <input type="number" id="inpMlSeed" aria-label="ML random seed" value="${Number(mlCfg.random_seed ?? 42)}" min="0" max="999999" style="width:78px;" title="random_seed" />
            <input type="number" id="inpMlTrees" aria-label="GBRT fák száma" value="${Number(mlCfg.n_estimators ?? 150)}" min="10" max="500" step="10" style="width:74px;" title="n_estimators" />
            <input type="number" id="inpMlDepth" aria-label="Maximális famélység" value="${Number(mlCfg.max_depth ?? 3)}" min="1" max="6" style="width:58px;" title="max_depth" />
          </div>
        </div>
        <div class="field-row">
          <label for="inpMlLr">Tanulási ráta (η) &amp; GroupKFold K</label>
          <div style="display:flex;gap:6px;">
            <input type="number" id="inpMlLr" aria-label="Learning rate" value="${Number(mlCfg.learning_rate ?? 0.05)}" min="0.005" max="0.5" step="0.01" style="flex:1;" title="learning_rate" />
            <input type="number" id="inpMlFolds" aria-label="CV foldok száma" value="${Number(mlCfg.cv_folds ?? 5)}" min="2" max="10" style="width:64px;" title="cv_folds" />
          </div>
        </div>
      </div>
    </div>

    ${
      s6 && s6.ok
        ? `
      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">6B. Épületklaszteres GroupKFold CV Benchmark (OLS vs. Ridge vs. GBRT)</h3>
            <div class="card-subtitle">
              N = ${s6.n_obs} (${s6.n_clusters} független épületklaszter, ${s6.cv_folds}-fold szivárgásmentes CV) · Seed = ${s6.random_seed}
            </div>
          </div>
          <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
            <span class="diag-chip pass" title="Lundberg TreeSHAP Local Accuracy (base + Σφ - ŷ)">Δ<sub>SHAP</sub> = ${Number(s6.local_accuracy_max_err).toExponential(2)}</span>
            <button type="button" class="btn btn-sm" id="btnCopyStage6Latex">📋 Stage 6 LaTeX</button>
          </div>
        </div>
        <div class="table-wrap">
          <table class="academic">
            <thead>
              <tr>
                <th>Algoritmus</th>
                <th class="num">GroupKFold CV R²</th>
                <th class="num">CV RMSE (ln)</th>
                <th class="num">CV MAE</th>
                <th class="num">CV MAPE (%)</th>
                <th class="num">In-sample R²</th>
              </tr>
            </thead>
            <tbody>
              ${(s6.benchmark || [])
                .map(
                  (b) => `<tr>
                    <td><strong>${escapeHtml(b.model)}</strong></td>
                    <td class="num">${b.cv_r2}</td>
                    <td class="num">${b.cv_rmse}</td>
                    <td class="num">${b.cv_mae}</td>
                    <td class="num">${b.cv_mape_pct != null ? `${b.cv_mape_pct}%` : "—"}</td>
                    <td class="num">${b.in_sample_r2}</td>
                  </tr>`
                )
                .join("")}
            </tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">6C. Egzakt TreeSHAP Globális Változó- és Csoport-Fontosság</h3>
            <div class="card-subtitle">
              Fizikai minőség: <strong>${grp.physical_pct}%</strong> | Vasút összesen: <strong>${grp.rail_pct}%</strong> (Vágány: ${grp.rail_track_pct}%, Állomás: ${grp.rail_station_pct}%) | Főút: <strong>${grp.road_pct}%</strong>
            </div>
          </div>
        </div>
        <div id="plotShapImportance" class="plot-box"></div>
        <div class="table-wrap" style="margin-top:8px;">
          <table class="academic">
            <thead>
              <tr>
                <th class="num">#</th>
                <th>Változó</th>
                <th>Hatáscsoport</th>
                <th class="num">Átl. |SHAP|</th>
                <th class="num">Relatív súly (%)</th>
                <th class="num">Irány (r<sub>x,φ</sub>)</th>
              </tr>
            </thead>
            <tbody>
              ${(s6.feature_importance || [])
                .map(
                  (r) => `<tr>
                    <td class="num">${r.rank}</td>
                    <td><strong>${escapeHtml(r.feature)}</strong></td>
                    <td><span class="diag-chip">${escapeHtml(GROUP_LABELS_HU[r.group] || r.group)}</span></td>
                    <td class="num mono">${r.mean_abs_shap.toFixed(4)}</td>
                    <td class="num"><strong>${r.rel_pct.toFixed(2)}%</strong></td>
                    <td class="num mono">${r.corr_x_shap >= 0 ? "+" : ""}${r.corr_x_shap.toFixed(3)}</td>
                  </tr>`
                )
                .join("")}
            </tbody>
          </table>
        </div>
      </div>
    `
        : s6 && s6.reason
          ? `<div class="alert-box">⚠️ ${escapeHtml(s6.reason)}</div>`
          : ""
    }
  `;

  container.querySelector("#btnRunStage6")?.addEventListener("click", () => actions.runStage6());
  container.querySelector("#selMlModel")?.addEventListener("change", (e) => {
    mlCfg.focal_model_id = e.target.value;
    actions.invalidateFromStage?.(6);
  });
  container.querySelector("#inpMlSeed")?.addEventListener("change", (e) => {
    mlCfg.random_seed = Math.max(0, Math.round(Number(e.target.value) || 42));
    actions.invalidateFromStage?.(6);
  });
  container.querySelector("#inpMlTrees")?.addEventListener("change", (e) => {
    mlCfg.n_estimators = Math.max(10, Math.min(500, Math.round(Number(e.target.value) || 150)));
    actions.invalidateFromStage?.(6);
  });
  container.querySelector("#inpMlDepth")?.addEventListener("change", (e) => {
    mlCfg.max_depth = Math.max(1, Math.min(6, Math.round(Number(e.target.value) || 3)));
    actions.invalidateFromStage?.(6);
  });
  container.querySelector("#inpMlLr")?.addEventListener("change", (e) => {
    mlCfg.learning_rate = Math.max(0.005, Math.min(0.5, Number(e.target.value) || 0.05));
    actions.invalidateFromStage?.(6);
  });
  container.querySelector("#inpMlFolds")?.addEventListener("change", (e) => {
    mlCfg.cv_folds = Math.max(2, Math.min(10, Math.round(Number(e.target.value) || 5)));
    actions.invalidateFromStage?.(6);
  });
  container.querySelector("#btnCopyStage6Latex")?.addEventListener("click", () => actions.copyLatexTable("stage6"));

  if (s6?.ok && Array.isArray(s6.feature_importance) && window.Plotly) {
    const ordered = [...s6.feature_importance].slice(0, 12).reverse();
    window.Plotly.newPlot(
      "plotShapImportance",
      [
        {
          y: ordered.map((r) => r.feature),
          x: ordered.map((r) => r.rel_pct),
          type: "bar",
          orientation: "h",
          text: ordered.map((r) => `${r.rel_pct.toFixed(1)}%`),
          textposition: "outside",
          marker: { color: ordered.map((r) => GROUP_COLORS[r.group] || "#1d4ed8") },
        },
      ],
      getPlotlyThemeLayout({
        title: { text: `Egzakt TreeSHAP Relatív Ármeghatározó Súlyok (${escapeHtml(s6.focal_model_id || "m3_log")}, N=${s6.n_obs})`, font: { size: 12 } },
        xaxis: { title: "Relatív |SHAP| arány (%)", range: [0, Math.max(40, Math.ceil((ordered[ordered.length - 1]?.rel_pct || 30) * 1.18))] },
        margin: { t: 34, r: 42, b: 40, l: 165 },
        height: 310,
      }),
      { responsive: true, displayModeBar: false }
    );
  }
}
