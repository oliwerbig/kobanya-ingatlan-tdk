/* Stages 4 & 5: Spatial Econometrics (Multi-W, Moran/LISA, Anselin/Elhorst, SAR/SEM/SDM, Krinsky-Robb)
 * + Singularity-Guarded Adaptive Bisquare GWR & Publication Export Studio
 */

export function renderStage4(container, ctx) {
  const { state, actions } = ctx;
  if (!state.datasetInfo) {
    container.innerHTML = `<div class="card"><div class="card-subtitle">Előbb tölts fel adatot az 1. fázisban.</div></div>`;
    return;
  }

  const spCfg = state.recipe.spatial_engine;
  const activeModels = state.recipe.models.filter((m) => m.active);
  const w0 = spCfg.weights[0] || { id: "knn5", type: "knn", k: 5, dedup_colocated: false };
  const s4 = state.stage4Res;

  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">4A. Térökonometria — Súlymátrixok, SAR / SEM / SDM & Spillover</h3>
          <div class="card-subtitle">Ord/SuperLU Maximum Likelihood, Anselin LM, Elhorst Common-Factor & 1000× Krinsky–Robb</div>
        </div>
        <button class="btn btn-primary btn-sm" id="btnRunStage4">▶ Térökonometria Futtatása</button>
      </div>

      <div class="grid-3">
        <div class="field-row">
          <label>Célmodell specifikáció</label>
          <select id="selSpModel">
            ${activeModels.map((m) => `<option value="${m.id}" ${m.id === spCfg.active_model_id ? "selected" : ""}>${m.name}</option>`).join("")}
          </select>
        </div>
        <div class="field-row">
          <label>Elsődleges k-NN szomszédszám</label>
          <input type="number" id="inpKnnK" value="${w0.k || 5}" min="2" max="50" />
        </div>
        <div class="field-row">
          <label>Társasházi & FDR beállítások</label>
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
    </div>

    ${
      s4 && s4.ok
        ? `
      ${(s4.guard5_alerts || []).map((a) => `<div class="alert-box">⚠️ ${a}</div>`).join("")}

      <div class="card">
        <div class="card-header">
          <h3 class="card-title">4B. Térbeli Autokorreláció, Anselin LM-Fa & Elhorst Teszt</h3>
          <span class="status-pill mono">Moran I(resid) = ${s4.moran_resid.moran_i} (p=${s4.moran_resid.p})</span>
        </div>
        <div class="grid-2" style="font-size:12px;">
          <div style="padding:8px;border:1px solid var(--border);border-radius:6px;background:var(--surface-alt);">
            <div><strong>Anselin (1988) LM Döntés:</strong> ${s4.anselin.verdict}</div>
            <div class="mono" style="font-size:11px;margin-top:4px;">
              LM-Lag=${s4.anselin.lm_lag} (p=${s4.anselin.lm_lag_p}) | RLM-Lag=${s4.anselin.rlm_lag} (p=${s4.anselin.rlm_lag_p})<br/>
              LM-Err=${s4.anselin.lm_error} (p=${s4.anselin.lm_error_p}) | RLM-Err=${s4.anselin.rlm_error} (p=${s4.anselin.rlm_error_p})
            </div>
          </div>
          <div style="padding:8px;border:1px solid var(--border);border-radius:6px;background:var(--surface-alt);">
            <div><strong>Elhorst Common-Factor (SDM):</strong> ${s4.elhorst.verdict}</div>
            <div class="mono" style="font-size:11px;margin-top:4px;">
              LR(H₀: θ=0 vs SAR)=${s4.elhorst.lr_vs_sar} (p=${s4.elhorst.p_vs_sar})<br/>
              LR(H₀: θ+ρβ=0 vs SEM)=${s4.elhorst.lr_vs_sem} (p=${s4.elhorst.p_vs_sem})<br/>
              LISA Klaszterek: HH=${s4.lisa_resid.counts.HH} | LL=${s4.lisa_resid.counts.LL} | HL=${s4.lisa_resid.counts.HL} | LH=${s4.lisa_resid.counts.LH}
            </div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header"><h3 class="card-title">4C. OLS vs. SAR vs. SEM vs. SDM & Súlymátrix Robusztusság</h3></div>
        <div class="table-wrap" style="margin-bottom:10px;">
          <table class="academic">
            <thead><tr><th>Modell</th><th class="num">ρ / λ</th><th class="num">SE (p)</th><th class="num">Pseudo R²</th><th class="num">LL</th><th class="num">AIC</th><th class="num">Resid. Moran I (p)</th></tr></thead>
            <tbody>
              ${s4.model_comparison
                .map(
                  (r) => `<tr>
                    <td><strong>${r.model}</strong></td>
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
            <thead><tr><th>W Súlymátrix</th><th class="num">Átl. szomszéd</th><th class="num">Co-located %</th><th class="num">SAR ρ (p)</th><th class="num">SAR AIC</th><th class="num">SEM λ (p)</th></tr></thead>
            <tbody>
              ${s4.weights_robustness
                .map(
                  (w) => `<tr>
                    <td><strong>${w.w_id}</strong> (${w.w_type})</td>
                    <td class="num">${w.mean_neighbors}</td>
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
        <div class="card-header"><h3 class="card-title">4D. LeSage & Pace (2009) Spillover Hatásfelbontás (Krinsky–Robb SE)</h3></div>
        <div id="plotImpacts" class="plot-box"></div>
        <div class="table-wrap" style="margin-top:8px;">
          <table class="academic">
            <thead><tr><th>Változó</th><th class="num">Közvetlen (Direct)</th><th class="num">Közvetett (Spillover)</th><th class="num">Teljes (Total)</th><th class="num">Spillover %</th></tr></thead>
            <tbody>
              ${s4.impacts_sar
                .map(
                  (r) => `<tr>
                    <td><strong>${r.variable}</strong></td>
                    <td class="num">${r.direct.est}<span class="sig-stars">${r.direct.sig}</span> (${r.direct.se})</td>
                    <td class="num">${r.indirect.est}<span class="sig-stars">${r.indirect.sig}</span> (${r.indirect.se})</td>
                    <td class="num">${r.total.est}<span class="sig-stars">${r.total.sig}</span> (${r.total.se})</td>
                    <td class="num">${r.spillover_share_pct}%</td>
                  </tr>`
                )
                .join("")}
            </tbody>
          </table>
        </div>
      </div>
    `
        : ""
    }
  `;

  container.querySelector("#btnRunStage4").addEventListener("click", () => actions.runStage4());
  container.querySelector("#selSpModel").addEventListener("change", (e) => {
    spCfg.active_model_id = e.target.value;
  });
  container.querySelector("#inpKnnK").addEventListener("change", (e) => {
    w0.k = Math.max(2, Number(e.target.value) || 5);
  });
  container.querySelector("#chkDedupKnn").addEventListener("change", (e) => {
    w0.dedup_colocated = e.target.checked;
  });
  container.querySelector("#chkLisaFdr").addEventListener("change", (e) => {
    spCfg.lisa_fdr = e.target.checked;
  });

  if (s4?.ok && s4.impacts_sar?.length && window.Plotly) {
    const imps = s4.impacts_sar;
    window.Plotly.newPlot(
      "plotImpacts",
      [
        { x: imps.map((r) => r.variable), y: imps.map((r) => r.direct.est), name: "Direct", type: "bar", marker: { color: "#1d4ed8" } },
        { x: imps.map((r) => r.variable), y: imps.map((r) => r.indirect.est), name: "Indirect (Spillover)", type: "bar", marker: { color: "#0d9488" } },
      ],
      {
        barmode: "relative",
        title: { text: "LeSage & Pace Közvetlen vs. Szomszédsági Spillover Hatás", font: { size: 12 } },
        margin: { t: 34, r: 16, b: 60, l: 48 },
        height: 240,
      },
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
          <div class="card-subtitle">1.0 m jitter + Szelektív Kondíciószám-Ridge (κ &gt; ${gwrCfg.kappa_thresh}) + Natív skálájú β̂<sub>i</sub></div>
        </div>
        <button class="btn btn-primary btn-sm" id="btnRunStage5">▶ GWR Futtatása</button>
      </div>

      <div class="grid-3">
        <div class="field-row">
          <label>GWR Modell</label>
          <select id="selGwrModel">
            ${activeModels.map((m) => `<option value="${m.id}" ${m.id === gwrCfg.focal_model_id ? "selected" : ""}>${m.name}</option>`).join("")}
          </select>
        </div>
        <div class="field-row">
          <label>Mikro-jitter (m)</label>
          <input type="number" id="inpGwrJitter" value="${gwrCfg.jitter_m}" min="0" max="20" step="0.5" />
        </div>
        <div class="field-row">
          <label>Ridge κ küszöb</label>
          <input type="number" id="inpGwrKappa" value="${gwrCfg.kappa_thresh}" min="50" max="100000" step="100" />
        </div>
      </div>
    </div>

    ${
      s5 && s5.ok
        ? `
      ${(s5.warnings || []).map((w) => `<div class="alert-box">🛡️ ${w}</div>`).join("")}
      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">5B. Globális OLS vs. Lokális GWR & Nem-Stacionaritás</h3>
            <div class="card-subtitle">Optimális sávszélesség b* = ${s5.optimal_bw} (${s5.bw_pct_sample}%) | OLS Adj. R² = ${s5.global_ols_adj_r2} → GWR Adj. R² = ${s5.gwr_adj_r2} | tr(S) = ${s5.tr_s}</div>
          </div>
        </div>
        <div id="plotGwrBw" class="plot-box"></div>
        <div class="table-wrap" style="margin-top:8px;">
          <table class="academic">
            <thead><tr><th>Változó (Natív egység)</th><th class="num">OLS β̂</th><th class="num">Min</th><th class="num">Q1</th><th class="num">Medián</th><th class="num">Q3</th><th class="num">Max</th><th class="num">|t|≥1.96</th><th>Leung/IQR Státusz</th></tr></thead>
            <tbody>
              ${s5.param_summaries
                .map(
                  (r) => `<tr>
                    <td><strong>${r.name}</strong></td>
                    <td class="num">${r.ols_beta}</td>
                    <td class="num">${r.min}</td><td class="num">${r.q1}</td><td class="num">${r.median}</td><td class="num">${r.q3}</td><td class="num">${r.max}</td>
                    <td class="num">${r.pct_sig_95}%</td>
                    <td><span class="diag-chip ${r.non_stationary ? "warn" : "pass"}">${r.non_stationary ? "Nem-stacionárius" : "Stacionárius"}</span></td>
                  </tr>`
                )
                .join("")}
            </tbody>
          </table>
        </div>
      </div>
    `
        : ""
    }

    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">5C. Publikációs Export Stúdió & Hordozható Módszertani Recept</h3>
          <div class="card-subtitle">Overleaf LaTeX, QGIS GeoJSON, CSV/JSON/ZIP és adatmentes <code>experiment_recipe.json</code></div>
        </div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <button class="btn btn-primary" id="btnExportZip">📦 Teljes Tudományos Csomag (.ZIP)</button>
        <button class="btn" id="btnExportLatex">📋 Overleaf Booktabs LaTeX (.tex)</button>
        <button class="btn" id="btnExportGeojson">🗺️ QGIS Gazdagított GeoJSON</button>
        <button class="btn" id="btnExportRecipe">🧪 experiment_recipe.json Mentése</button>
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
  container.querySelector("#btnExportLatex").addEventListener("click", () => actions.copyLatexTable());
  container.querySelector("#btnExportGeojson").addEventListener("click", () => actions.exportBundle("geojson"));
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
      {
        title: { text: `Adaptív Bisquare Sávszélesség Optimalizáció (b* = ${s5.optimal_bw})`, font: { size: 12 } },
        xaxis: { title: "Szomszédszám (b)" },
        yaxis: { title: "AICc" },
        margin: { t: 34, r: 16, b: 40, l: 52 },
        height: 230,
      },
      { responsive: true, displayModeBar: false }
    );
  }
}
