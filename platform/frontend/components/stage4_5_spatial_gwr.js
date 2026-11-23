/* Stages 4 & 5: Spatial Econometrics (Multi-W, Moran/LISA, Anselin/Elhorst, SAR/SEM/SDM, Krinsky-Robb)
 * + Singularity-Guarded Adaptive Bisquare GWR & Publication Export Studio
 */

import { escapeHtml, getPlotlyThemeLayout } from "../utils/sanitize.js";

export function renderStage4(container, ctx) {
  const { state, actions } = ctx;
  if (!state.datasetInfo) {
    container.innerHTML = `<div class="card"><div class="card-subtitle">Előbb tölts fel adatot az 1. fázisban.</div></div>`;
    return;
  }

  const spCfg = state.recipe.spatial_engine;
  if (!Array.isArray(spCfg.weights)) spCfg.weights = [];
  if (!spCfg.weights[0]) spCfg.weights[0] = { id: "knn5", type: "knn", k: 5, dedup_colocated: false };
  const w0 = spCfg.weights[0];
  if (!spCfg.weights[1]) spCfg.weights[1] = { id: "idw_auto", type: "idw", power: 1, cutoff_mode: "auto_connectivity", cutoff_m: null };
  const w1 = spCfg.weights[1];
  const w1IsFixed = w1.cutoff_mode === "fixed_m" || w1.cutoff_mode === "fixed";

  const activeModels = state.recipe.models.filter((m) => m.active);
  const s4 = state.stage4Res;

  const hasSarImp = Array.isArray(s4?.impacts_sar) && s4.impacts_sar.length > 0;
  const hasSdmImp = Array.isArray(s4?.impacts_sdm) && s4.impacts_sdm.length > 0;
  if (state.stage4ImpactKind !== "sdm" && state.stage4ImpactKind !== "sar") {
    state.stage4ImpactKind = "sar";
  }
  if (state.stage4ImpactKind === "sdm" && !hasSdmImp) state.stage4ImpactKind = "sar";
  const activeImpacts = state.stage4ImpactKind === "sdm" && hasSdmImp ? s4.impacts_sdm : (s4?.impacts_sar || []);

  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">4A. Térökonometria — Súlymátrixok, SAR / SEM / SDM &amp; Spillover</h3>
          <div class="card-subtitle">Ord/SuperLU Maximum Likelihood, Anselin LM, Elhorst Common-Factor &amp; Krinsky–Robb</div>
        </div>
        <button class="btn btn-primary btn-sm" id="btnRunStage4">▶ Térökonometria Futtatása</button>
      </div>

      <div class="grid-3">
        <div class="field-row">
          <label>Célmodell specifikáció</label>
          <select id="selSpModel">
            ${activeModels.map((m) => `<option value="${escapeHtml(m.id)}" ${m.id === spCfg.active_model_id ? "selected" : ""}>${escapeHtml(m.name)}</option>`).join("")}
          </select>
        </div>
        <div class="field-row">
          <label>Elsődleges k-NN szomszédszám</label>
          <input type="number" id="inpKnnK" value="${Number(w0.k) || 5}" min="2" max="100" />
        </div>
        <div class="field-row">
          <label>Társasházi &amp; FDR beállítások</label>
          <div style="display:flex;gap:10px;align-items:center;padding-top:4px;">
            <label style="font-size:11px;display:inline-flex;align-items:center;gap:4px;">
              <input type="checkbox" id="chkDedupKnn" ${w0.dedup_colocated ? "checked" : ""} /> Parcel dedup
            </label>
            <label style="font-size:11px;display:inline-flex;align-items:center;gap:4px;">
              <input type="checkbox" id="chkLisaFdr" ${spCfg.lisa_fdr ? "checked" : ""} /> LISA FDR
            </label>
          </div>
        </div>
      </div>

      <div class="grid-3" style="margin-top:8px;padding-top:8px;border-top:1px dashed var(--border);">
        <div class="field-row">
          <label>Másodlagos IDW kitevő (1/d<sup>p</sup>)</label>
          <select id="selIdwPower">
            <option value="1" ${Number(w1.power || 1) === 1 ? "selected" : ""}>p = 1 (1/d)</option>
            <option value="2" ${Number(w1.power) === 2 ? "selected" : ""}>p = 2 (1/d²)</option>
          </select>
        </div>
        <div class="field-row">
          <label>IDW küszöbsugár mód</label>
          <select id="selIdwCutoffMode">
            <option value="auto_connectivity" ${!w1IsFixed ? "selected" : ""}>Auto-összefüggőség (P99 védett)</option>
            <option value="fixed_m" ${w1IsFixed ? "selected" : ""}>Fix sugár (méter)</option>
          </select>
        </div>
        <div class="field-row">
          <label>IDW fix sugár (m)</label>
          <input type="number" id="inpIdwCutoffM" value="${Number(w1.cutoff_m) || 500}" min="50" max="50000" step="50" ${w1IsFixed ? "" : "disabled"} />
        </div>
      </div>
    </div>

    ${
      s4 && s4.ok
        ? `
      ${(s4.guard5_alerts || []).map((a) => `<div class="alert-box">⚠️ ${escapeHtml(a)}</div>`).join("")}

      <div class="card">
        <div class="card-header">
          <h3 class="card-title">4B. Térbeli Autokorreláció, Anselin LM-Fa &amp; Elhorst Teszt</h3>
          <span class="status-pill mono">Moran I(resid) = ${s4.moran_resid?.moran_i} (p=${s4.moran_resid?.p})</span>
        </div>
        <div class="grid-2" style="font-size:12px;">
          <div style="padding:8px;border:1px solid var(--border);border-radius:6px;background:var(--surface-alt);">
            <div><strong>Anselin (1988) LM Döntés:</strong> ${escapeHtml(s4.anselin?.verdict)}</div>
            <div class="mono" style="font-size:11px;margin-top:4px;">
              LM-Lag=${s4.anselin?.lm_lag} (p=${s4.anselin?.lm_lag_p}) | RLM-Lag=${s4.anselin?.rlm_lag} (p=${s4.anselin?.rlm_lag_p})<br/>
              LM-Err=${s4.anselin?.lm_error} (p=${s4.anselin?.lm_error_p}) | RLM-Err=${s4.anselin?.rlm_error} (p=${s4.anselin?.rlm_error_p})
            </div>
          </div>
          <div style="padding:8px;border:1px solid var(--border);border-radius:6px;background:var(--surface-alt);">
            <div><strong>Elhorst Common-Factor (SDM):</strong> ${escapeHtml(s4.elhorst?.verdict)}</div>
            <div class="mono" style="font-size:11px;margin-top:4px;">
              LR(H₀: θ=0 vs SAR)=${s4.elhorst?.lr_vs_sar} (p=${s4.elhorst?.p_vs_sar})<br/>
              LR(H₀: θ+ρβ=0 vs SEM)=${s4.elhorst?.lr_vs_sem} (p=${s4.elhorst?.p_vs_sem})<br/>
              LISA Klaszterek: HH=${s4.lisa_resid?.counts?.HH ?? 0} | LL=${s4.lisa_resid?.counts?.LL ?? 0} | HL=${s4.lisa_resid?.counts?.HL ?? 0} | LH=${s4.lisa_resid?.counts?.LH ?? 0}
            </div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header"><h3 class="card-title">4C. OLS vs. SAR vs. SEM vs. SDM &amp; Súlymátrix Robusztusság</h3></div>
        <div class="table-wrap" style="margin-bottom:10px;">
          <table class="academic">
            <thead><tr><th>Modell</th><th class="num">ρ / λ</th><th class="num">SE (p)</th><th class="num">Pseudo R²</th><th class="num">LL</th><th class="num">AIC</th><th class="num">Resid. Moran I (p)</th></tr></thead>
            <tbody>
              ${(s4.model_comparison || [])
                .map(
                  (r) => `<tr>
                    <td><strong>${escapeHtml(r.model)}</strong></td>
                    <td class="num">${r.spatial_param ?? "—"}</td>
                    <td class="num">${r.spatial_se !== null ? `${r.spatial_se} (p=${r.spatial_p})` : "—"}</td>
                    <td class="num">${r.r2_or_pseudo}</td>
                    <td class="num">${r.llf}</td><td class="num">${r.aic}</td>
                    <td class="num">${r.resid_moran_i} (${r.resid_moran_p})</td>
                  </tr>`
                )
                .join("")}
            </tbody>
          </table>
        </div>
        <div class="table-wrap">
          <table class="academic">
            <thead><tr><th>W Súlymátrix</th><th class="num">Átl. szomszéd (Sziget)</th><th class="num">Co-located %</th><th class="num">SAR ρ (p)</th><th class="num">SAR AIC</th><th class="num">SEM λ (p)</th></tr></thead>
            <tbody>
              ${(s4.weights_robustness || [])
                .map(
                  (w) => `<tr>
                    <td><strong>${escapeHtml(w.w_id)}</strong> (${escapeHtml(w.w_type)})</td>
                    <td class="num">${w.mean_neighbors} (${w.islands ?? 0})</td>
                    <td class="num">${w.colocated_pairs_pct}%</td>
                    <td class="num">${w.sar_rho} (${w.sar_rho_p})</td>
                    <td class="num">${w.sar_aic}</td>
                    <td class="num">${w.sem_lambda} (${w.sem_lambda_p})</td>
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
            <h3 class="card-title">4D. LeSage &amp; Pace (2009) Spillover Hatásfelbontás (Krinsky–Robb 95% CI)</h3>
            <div class="card-subtitle">Közvetlen, szomszédsági spillover és teljes hatás becslése (${state.stage4ImpactKind === "sdm" ? "SDM" : "SAR"} modell)</div>
          </div>
          <div class="spec-tabs" style="margin:0;">
            <button class="spec-tab sp-impact-tab ${state.stage4ImpactKind === "sar" ? "active" : ""}" data-impact-kind="sar" ${hasSarImp ? "" : "disabled"}>SAR Hatások</button>
            <button class="spec-tab sp-impact-tab ${state.stage4ImpactKind === "sdm" ? "active" : ""}" data-impact-kind="sdm" ${hasSdmImp ? "" : "disabled"}>SDM Hatások</button>
          </div>
        </div>
        <div id="plotImpacts" class="plot-box"></div>
        <div class="table-wrap" style="margin-top:8px;">
          <table class="academic">
            <thead><tr><th>Változó</th><th class="num">Közvetlen (Direct)</th><th class="num">Közvetett (Spillover)</th><th class="num">Teljes (Total)</th><th class="num">Spillover %</th></tr></thead>
            <tbody>
              ${activeImpacts
                .map(
                  (r) => `<tr>
                    <td><strong>${escapeHtml(r.variable)}</strong></td>
                    <td class="num">
                      <div>${r.direct.est}<span class="sig-stars">${escapeHtml(r.direct.sig)}</span> (${r.direct.se})</div>
                      <div class="card-subtitle mono" style="font-size:10px;">[${r.direct.ci_lo}, ${r.direct.ci_hi}]</div>
                    </td>
                    <td class="num">
                      <div>${r.indirect.est}<span class="sig-stars">${escapeHtml(r.indirect.sig)}</span> (${r.indirect.se})</div>
                      <div class="card-subtitle mono" style="font-size:10px;">[${r.indirect.ci_lo}, ${r.indirect.ci_hi}]</div>
                    </td>
                    <td class="num">
                      <div>${r.total.est}<span class="sig-stars">${escapeHtml(r.total.sig)}</span> (${r.total.se})</div>
                      <div class="card-subtitle mono" style="font-size:10px;">[${r.total.ci_lo}, ${r.total.ci_hi}]</div>
                    </td>
                    <td class="num">${r.spillover_share_pct}%</td>
                  </tr>`
                )
                .join("")}
            </tbody>
          </table>
        </div>
      </div>
    `
        : s4 && s4.reason
          ? `<div class="alert-box">⚠️ ${escapeHtml(s4.reason)}</div>`
          : ""
    }
  `;

  container.querySelector("#btnRunStage4").addEventListener("click", () => actions.runStage4());
  container.querySelector("#selSpModel").addEventListener("change", (e) => {
    spCfg.active_model_id = e.target.value;
  });
  container.querySelector("#inpKnnK").addEventListener("change", (e) => {
    w0.k = Math.max(2, Math.min(100, Number(e.target.value) || 5));
  });
  container.querySelector("#chkDedupKnn").addEventListener("change", (e) => {
    w0.dedup_colocated = e.target.checked;
  });
  container.querySelector("#chkLisaFdr").addEventListener("change", (e) => {
    spCfg.lisa_fdr = e.target.checked;
  });

  container.querySelector("#selIdwPower").addEventListener("change", (e) => {
    w1.power = Number(e.target.value) === 2 ? 2 : 1;
  });
  container.querySelector("#selIdwCutoffMode").addEventListener("change", (e) => {
    w1.cutoff_mode = e.target.value;
    const inp = container.querySelector("#inpIdwCutoffM");
    if (inp) {
      inp.disabled = w1.cutoff_mode !== "fixed_m";
      if (w1.cutoff_mode === "fixed_m" && !w1.cutoff_m) w1.cutoff_m = Number(inp.value) || 500;
    }
  });
  container.querySelector("#inpIdwCutoffM").addEventListener("change", (e) => {
    w1.cutoff_m = Math.max(50, Number(e.target.value) || 500);
  });

  container.querySelectorAll(".sp-impact-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.stage4ImpactKind = btn.getAttribute("data-impact-kind");
      actions.render();
    });
  });

  if (s4?.ok && activeImpacts.length && window.Plotly) {
    const labelKind = state.stage4ImpactKind === "sdm" ? "SDM" : "SAR";
    window.Plotly.newPlot(
      "plotImpacts",
      [
        {
          x: activeImpacts.map((r) => r.variable),
          y: activeImpacts.map((r) => r.direct.est),
          error_y: { type: "data", array: activeImpacts.map((r) => 1.96 * r.direct.se), visible: true },
          name: "Direct",
          type: "bar",
          marker: { color: "#1d4ed8" },
        },
        {
          x: activeImpacts.map((r) => r.variable),
          y: activeImpacts.map((r) => r.indirect.est),
          error_y: { type: "data", array: activeImpacts.map((r) => 1.96 * r.indirect.se), visible: true },
          name: "Indirect (Spillover)",
          type: "bar",
          marker: { color: "#0d9488" },
        },
      ],
      getPlotlyThemeLayout({
        barmode: "group",
        title: { text: `LeSage & Pace (${labelKind}) Közvetlen vs. Szomszédsági Spillover Hatás (±95% CI)`, font: { size: 12 } },
        margin: { t: 34, r: 16, b: 60, l: 48 },
        height: 250,
      }),
      { responsive: true, displayModeBar: false }
    );
  }
}

export function renderStage5(container, ctx) {
  const { state, actions } = ctx;
  if (!state.datasetInfo) {
    container.innerHTML = `<div class="card"><div class="card-subtitle">Előbb tölts fel adatot az 1. fázisban.</div></div>`;
    return;
  }

  const gwrCfg = state.recipe.spatial_engine.gwr;
  const activeModels = state.recipe.models.filter((m) => m.active);
  const s5 = state.stage5Res;

  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">5A. Szingularitás-Védett Adaptív Bisquare GWR (Guard #6)</h3>
          <div class="card-subtitle">Mikro-jitter + Szelektív Kondíciószám-Ridge (κ &gt; ${gwrCfg.kappa_thresh}) + Natív skálájú β̂<sub>i</sub></div>
        </div>
        <button class="btn btn-primary btn-sm" id="btnRunStage5">▶ GWR Futtatása</button>
      </div>

      <div class="grid-3">
        <div class="field-row">
          <label>GWR Modell</label>
          <select id="selGwrModel">
            ${activeModels.map((m) => `<option value="${escapeHtml(m.id)}" ${m.id === gwrCfg.focal_model_id ? "selected" : ""}>${escapeHtml(m.name)}</option>`).join("")}
          </select>
        </div>
        <div class="field-row">
          <label>Mikro-jitter (m)</label>
          <input type="number" id="inpGwrJitter" value="${Number(gwrCfg.jitter_m ?? 1)}" min="0" max="20" step="0.5" />
        </div>
        <div class="field-row">
          <label>Ridge κ küszöb</label>
          <input type="number" id="inpGwrKappa" value="${Number(gwrCfg.kappa_thresh) || 1000}" min="50" max="100000" step="100" />
        </div>
      </div>
    </div>

    ${
      s5 && s5.ok
        ? `
      ${(s5.warnings || []).map((w) => `<div class="alert-box">🛡️ ${escapeHtml(w)}</div>`).join("")}
      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">5B. Globális OLS vs. Lokális GWR &amp; Nem-Stacionaritás</h3>
            <div class="card-subtitle">Optimális b* = ${s5.optimal_bw} (${s5.bw_pct_sample}%) | OLS Adj. R² = ${s5.global_ols_adj_r2} → GWR Adj. R² = ${s5.gwr_adj_r2} | tr(S) = ${s5.tr_s} | ENP df = ${s5.eff_df ?? "—"} (t<sub>crit</sub>=${s5.t_crit_nom ?? "1.96"}, dSF=${s5.t_crit_sdf ?? "—"})</div>
          </div>
        </div>
        <div id="plotGwrBw" class="plot-box"></div>
        <div class="table-wrap" style="margin-top:8px;">
          <table class="academic">
            <thead><tr><th>Változó (Natív egység)</th><th class="num">OLS β̂</th><th class="num">Min</th><th class="num">Q1</th><th class="num">Medián</th><th class="num">Q3</th><th class="num">Max</th><th class="num">Lok. szig. % (Nom / BY-FDR / dSF)</th><th>FBC (2002) IQR &gt; 2·SE<sub>OLS</sub></th></tr></thead>
            <tbody>
              ${(s5.param_summaries || [])
                .map(
                  (r) => `<tr>
                    <td><strong>${escapeHtml(r.name)}</strong></td>
                    <td class="num">${r.ols_beta}</td>
                    <td class="num">${r.min}</td><td class="num">${r.q1}</td><td class="num">${r.median}</td><td class="num">${r.q3}</td><td class="num">${r.max}</td>
                    <td class="num mono">${r.pct_sig_95}% / ${r.pct_sig_fdr ?? r.pct_sig_95}% / ${r.pct_sig_sdf ?? "—"}%</td>
                    <td><span class="diag-chip ${r.non_stationary ? "warn" : "pass"}" title="${escapeHtml(r.stationarity_verdict || "")}">${r.non_stationary ? "Nem-stacionárius" : "Stacionárius"}</span></td>
                  </tr>`
                )
                .join("")}
            </tbody>
          </table>
        </div>
      </div>
    `
        : s5 && s5.reason
          ? `<div class="alert-box">⚠️ ${escapeHtml(s5.reason)}</div>`
          : ""
    }

    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">5C. Publikációs Export Stúdió &amp; Hordozható Módszertani Recept</h3>
          <div class="card-subtitle">Overleaf LaTeX, QGIS GeoJSON, Offline OSM pillanatfelvétel, CSV/JSON/ZIP és adatmentes <code>experiment_recipe.json</code></div>
        </div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <button class="btn btn-primary" id="btnExportZip">📦 Teljes Tudományos Csomag (.ZIP)</button>
        <button class="btn" id="btnExportWorkspaceS5">💾 Teljes Projekt (.geospec.json)</button>
        <button class="btn" id="btnExportLatex">📋 Overleaf Booktabs LaTeX (.tex)</button>
        <button class="btn" id="btnExportGeojson">🗺️ QGIS Gazdagított GeoJSON</button>
        <button class="btn" id="btnExportOsmBundleS5">💾 Offline OSM Pillanatfelvétel (.osm_bundle.json)</button>
        <button class="btn" id="btnExportRecipe">💾 Recept Mentése (_recipe.json)</button>
      </div>
    </div>
  `;

  container.querySelector("#btnRunStage5").addEventListener("click", () => actions.runStage5());
  container.querySelector("#selGwrModel").addEventListener("change", (e) => {
    gwrCfg.focal_model_id = e.target.value;
  });
  container.querySelector("#inpGwrJitter").addEventListener("change", (e) => {
    gwrCfg.jitter_m = Math.max(0, Number(e.target.value) || 0);
  });
  container.querySelector("#inpGwrKappa").addEventListener("change", (e) => {
    gwrCfg.kappa_thresh = Math.max(50, Number(e.target.value) || 1000);
  });

  container.querySelector("#btnExportZip").addEventListener("click", () => actions.exportBundle("zip"));
  container.querySelector("#btnExportWorkspaceS5").addEventListener("click", () => actions.exportWorkspace());
  container.querySelector("#btnExportLatex").addEventListener("click", () => actions.copyLatexTable());
  container.querySelector("#btnExportGeojson").addEventListener("click", () => actions.exportBundle("geojson"));
  container.querySelector("#btnExportOsmBundleS5").addEventListener("click", () => actions.exportOsmBundle());
  container.querySelector("#btnExportRecipe").addEventListener("click", () => actions.exportRecipe());

  if (s5?.ok && s5.bw_curve?.length && window.Plotly) {
    window.Plotly.newPlot(
      "plotGwrBw",
      [
        {
          x: s5.bw_curve.map((p) => p.bw),
          y: s5.bw_curve.map((p) => p.aicc),
          type: "scatter",
          mode: "lines+markers",
          line: { color: "#0f766e", width: 2 },
        },
      ],
      getPlotlyThemeLayout({
        title: { text: `Adaptív Bisquare Sávszélesség Optimalizáció (b* = ${s5.optimal_bw})`, font: { size: 12 } },
        xaxis: { title: "Szomszédszám (b)" },
        yaxis: { title: "AICc" },
        margin: { t: 34, r: 16, b: 40, l: 52 },
        height: 230,
      }),
      { responsive: true, displayModeBar: false }
    );
  }
}
