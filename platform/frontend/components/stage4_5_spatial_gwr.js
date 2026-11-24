/* Stages 4 & 5: Multi-W Spatial Econometrics (incl. W_net, QML vs ML SE, Anselin-Kelejian 1997,
 * SAR/SEM/SDM Parameter Tables, Krinsky-Robb Spillover, Dedicated LaTeX Copy) & Adaptive GWR / MGWR
 */

import { translateVerdict } from "../utils/i18n.js";
import { escapeHtml, getPlotlyThemeLayout } from "../utils/sanitize.js";

export function renderStage4(container, ctx) {
  const { state, actions } = ctx;
  if (!state.datasetInfo) {
    container.innerHTML = `<div class="card"><div class="card-subtitle">Előbb tölts fel adatot az 1. fázisban.</div></div>`;
    return;
  }

  const spCfg = state.recipe.spatial_engine;
  if (!Array.isArray(spCfg.weights) || !spCfg.weights.length) {
    spCfg.weights = [
      { id: "knn5", type: "knn", k: 5, dedup_colocated: true },
      { id: "idw_auto", type: "idw", power: 1, cutoff_mode: "auto_connectivity", cutoff_m: null },
    ];
  }
  if (!spCfg.se_mode) spCfg.se_mode = "qml";

  const activeModels = state.recipe.models.filter((m) => m.active);
  if (activeModels.length && !activeModels.some((m) => m.id === spCfg.active_model_id)) {
    spCfg.active_model_id = activeModels[activeModels.length - 1].id;
  }
  const s4 = state.stage4Res;

  const hasSarImp = Array.isArray(s4?.impacts_sar) && s4.impacts_sar.length > 0;
  const hasSdmImp = Array.isArray(s4?.impacts_sdm) && s4.impacts_sdm.length > 0;
  if (state.stage4ImpactKind !== "sdm" && state.stage4ImpactKind !== "sar") state.stage4ImpactKind = "sar";
  if (state.stage4ImpactKind === "sdm" && !hasSdmImp) state.stage4ImpactKind = "sar";
  const activeImpacts = state.stage4ImpactKind === "sdm" && hasSdmImp ? s4.impacts_sdm : (s4?.impacts_sar || []);

  if (!["sar", "sem", "sdm"].includes(state.stage4ParamKind)) state.stage4ParamKind = "sem";

  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">4A. Térökonometria — Multi-W Mátrixok, QML/ML, SAR / SEM / SDM &amp; Spillover</h3>
          <div class="card-subtitle">Ord/SuperLU ML, Kelejian–Prucha QML Szendvics SE, Anselin–Kelejian (1997) &amp; Hálózati W<sub>net</sub></div>
        </div>
        <button type="button" class="btn btn-primary btn-sm" id="btnRunStage4">▶ Térökonometria Futtatása</button>
      </div>

      <div class="grid-3">
        <div class="field-row">
          <label for="selSpModel">Célmodell specifikáció</label>
          <select id="selSpModel">
            ${activeModels.map((m) => `<option value="${escapeHtml(m.id)}" ${m.id === spCfg.active_model_id ? "selected" : ""}>${escapeHtml(m.name)}</option>`).join("")}
          </select>
        </div>
        <div class="field-row">
          <label for="selSpSeMode">Térbeli SE mód (QML vs. Klasszikus ML)</label>
          <select id="selSpSeMode">
            <option value="qml" ${spCfg.se_mode === "qml" ? "selected" : ""}>QML Szendvics (heteroszkedaszticitás-álló)</option>
            <option value="ml" ${spCfg.se_mode === "ml" ? "selected" : ""}>Klasszikus ML (homoszkedasztikus)</option>
          </select>
        </div>
        <div class="field-row">
          <label>LISA többszörös tesztelés</label>
          <div style="display:flex;gap:10px;align-items:center;padding-top:6px;">
            <label style="font-size:11px;display:inline-flex;align-items:center;gap:4px;">
              <input type="checkbox" id="chkLisaFdr" ${spCfg.lisa_fdr ? "checked" : ""} /> BH-FDR (p<sub>adj</sub> &lt; 0.05)
            </label>
          </div>
        </div>
      </div>

      <div style="margin-top:10px;padding-top:8px;border-top:1px dashed var(--border);">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;flex-wrap:wrap;gap:6px;">
          <strong style="font-size:11.5px;">Konfigurált Súlymátrixok (W<sub>1</sub> = elsődleges a modellekhez, több W összehasonlítása)</strong>
          <div style="display:flex;gap:6px;flex-wrap:wrap;">
            <button type="button" class="btn btn-sm" id="btnAddWKnn">+ k-NN Mátrix</button>
            <button type="button" class="btn btn-sm" id="btnAddWIdw">+ Eukl. IDW Mátrix</button>
            <button type="button" class="btn btn-sm" id="btnAddWNet">+ Hálózati W<sub>net</sub></button>
          </div>
        </div>
        <div id="wMatrixList">
          ${spCfg.weights.map((w, idx) => renderWeightRow(w, idx)).join("")}
        </div>
      </div>
    </div>

    ${
      s4 && s4.ok
        ? `
      ${(s4.guard5_alerts || []).map((a) => `<div class="alert-box">⚠️ ${escapeHtml(translateVerdict(a))}</div>`).join("")}

      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">4B. Térbeli Autokorreláció, Anselin LM-Fa &amp; Elhorst Teszt</h3>
            <div class="card-subtitle">Primer W: <code>${escapeHtml(s4.primary_w_id)}</code> | SE: <strong>${escapeHtml((s4.se_mode || "qml").toUpperCase())}</strong> | N = ${s4.n_obs}</div>
          </div>
          <span class="status-pill mono">Moran I(resid) = ${s4.moran_resid?.moran_i} (p=${s4.moran_resid?.p})</span>
        </div>
        <div class="grid-2" style="font-size:12px;">
          <div style="padding:8px;border:1px solid var(--border);border-radius:6px;background:var(--surface-alt);">
            <div><strong>Anselin (1988) LM Döntés:</strong> ${escapeHtml(translateVerdict(s4.anselin?.verdict))}</div>
            <div class="mono" style="font-size:11px;margin-top:4px;">
              LM-Lag=${s4.anselin?.lm_lag} (p=${s4.anselin?.lm_lag_p}) | RLM-Lag=${s4.anselin?.rlm_lag} (p=${s4.anselin?.rlm_lag_p})<br/>
              LM-Err=${s4.anselin?.lm_error} (p=${s4.anselin?.lm_error_p}) | RLM-Err=${s4.anselin?.rlm_error} (p=${s4.anselin?.rlm_error_p})
            </div>
          </div>
          <div style="padding:8px;border:1px solid var(--border);border-radius:6px;background:var(--surface-alt);">
            <div><strong>Elhorst Common-Factor (SDM):</strong> ${escapeHtml(translateVerdict(s4.elhorst?.verdict))}</div>
            <div class="mono" style="font-size:11px;margin-top:4px;">
              LR(H₀: θ=0 vs SAR)=${s4.elhorst?.lr_vs_sar ?? "—"} (p=${s4.elhorst?.p_vs_sar ?? "—"})<br/>
              LR(H₀: θ+ρβ=0 vs SEM)=${s4.elhorst?.lr_vs_sem ?? "—"} (p=${s4.elhorst?.p_vs_sem ?? "—"})<br/>
              LISA: HH=${s4.lisa_resid?.counts?.HH ?? 0} | LL=${s4.lisa_resid?.counts?.LL ?? 0} | HL=${s4.lisa_resid?.counts?.HL ?? 0} | LH=${s4.lisa_resid?.counts?.LH ?? 0}
            </div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">4C. OLS vs. SAR vs. SEM vs. SDM &amp; Súlymátrix Robusztusság</h3>
            <div class="card-subtitle">SAR és SDM reziduális diagnosztikája egzakt Anselin–Kelejian (1997) statisztikával</div>
          </div>
          <div style="display:flex;gap:6px;flex-wrap:wrap;">
            <button type="button" class="btn btn-sm" id="btnCopyStage4Latex">📋 Összevető LaTeX</button>
            <button type="button" class="btn btn-sm" id="btnCopyStage4ParamsLatex">📋 Kovariánsok LaTeX</button>
          </div>
        </div>
        <div class="table-wrap" style="margin-bottom:10px;">
          <table class="academic">
            <thead><tr><th>Modell</th><th class="num">ρ / λ</th><th class="num">SE (p)</th><th class="num">Pseudo R²</th><th class="num">LL</th><th class="num">AIC</th><th class="num">Resid. Diagnosztika (Moran / AK 1997)</th></tr></thead>
            <tbody>
              ${(s4.model_comparison || [])
                .map(
                  (r) => `<tr>
                    <td><strong>${escapeHtml(r.model)}</strong></td>
                    <td class="num">${r.spatial_param ?? "—"}</td>
                    <td class="num">${r.spatial_se !== null ? `${r.spatial_se} (p=${r.spatial_p})` : "—"}</td>
                    <td class="num">${r.r2_or_pseudo ?? "—"}</td>
                    <td class="num">${r.llf ?? "—"}</td><td class="num">${r.aic ?? "—"}</td>
                    <td class="num">${r.resid_test_type === "anselin_kelejian_1997" ? `AK=${r.resid_moran_i}` : `I=${r.resid_moran_i ?? "—"}`} ${r.resid_moran_p != null ? `(p=${r.resid_moran_p})` : ""}</td>
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
                    <td class="num">${w.empty_graph ? "üres gráf" : `${w.sar_rho} (${w.sar_rho_p})`}</td>
                    <td class="num">${w.sar_aic ?? "—"}</td>
                    <td class="num">${w.empty_graph ? "üres gráf" : `${w.sem_lambda} (${w.sem_lambda_p})`}</td>
                  </tr>`
                )
                .join("")}
            </tbody>
          </table>
        </div>
      </div>

      ${renderSpatialParamsCard(s4, state.stage4ParamKind)}

      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">4E. LeSage &amp; Pace (2009) Spillover Hatásfelbontás (Krinsky–Robb 95% CI)</h3>
            <div class="card-subtitle">Közvetlen, szomszédsági spillover és teljes hatás becslése (${state.stage4ImpactKind === "sdm" ? "SDM" : "SAR"} modell)</div>
          </div>
          <div class="spec-tabs" style="margin:0;">
            <button type="button" class="spec-tab sp-impact-tab ${state.stage4ImpactKind === "sar" ? "active" : ""}" data-impact-kind="sar" ${hasSarImp ? "" : "disabled"}>SAR Hatások</button>
            <button type="button" class="spec-tab sp-impact-tab ${state.stage4ImpactKind === "sdm" ? "active" : ""}" data-impact-kind="sdm" ${hasSdmImp ? "" : "disabled"}>SDM Hatások</button>
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
    actions.invalidateFromStage?.(4);
  });
  container.querySelector("#selSpSeMode").addEventListener("change", (e) => {
    spCfg.se_mode = e.target.value;
    actions.invalidateFromStage?.(4);
  });
  container.querySelector("#chkLisaFdr").addEventListener("change", (e) => {
    spCfg.lisa_fdr = e.target.checked;
    actions.invalidateFromStage?.(4);
  });

  container.querySelector("#btnAddWKnn")?.addEventListener("click", () => {
    spCfg.weights.push({ id: `knn_${Date.now().toString(36).slice(-3)}`, type: "knn", k: 8, dedup_colocated: true });
    actions.invalidateFromStage?.(4);
    actions.render();
  });
  container.querySelector("#btnAddWIdw")?.addEventListener("click", () => {
    spCfg.weights.push({ id: `idw_${Date.now().toString(36).slice(-3)}`, type: "idw", power: 1.0, cutoff_mode: "auto_connectivity", cutoff_m: 1000 });
    actions.invalidateFromStage?.(4);
    actions.render();
  });
  container.querySelector("#btnAddWNet")?.addEventListener("click", () => {
    spCfg.weights.push({ id: `wnet_${Date.now().toString(36).slice(-3)}`, type: "network_idw", profile: "walk", power: 1.0, cutoff_m: 1500, max_snap_distance_m: 300 });
    actions.invalidateFromStage?.(4);
    actions.render();
  });

  container.querySelectorAll("[data-w-idx]").forEach((row) => {
    const idx = Number(row.getAttribute("data-w-idx"));
    const w = spCfg.weights[idx];
    if (!w) return;
    row.querySelector(".w-id-inp")?.addEventListener("change", (e) => {
      w.id = e.target.value.trim() || w.id;
      actions.invalidateFromStage?.(4);
    });
    row.querySelector(".w-k-inp")?.addEventListener("change", (e) => {
      w.k = Math.max(2, Math.min(200, Number(e.target.value) || 5));
      actions.invalidateFromStage?.(4);
    });
    row.querySelector(".w-dedup-chk")?.addEventListener("change", (e) => {
      w.dedup_colocated = e.target.checked;
      actions.invalidateFromStage?.(4);
    });
    row.querySelector(".w-pow-sel")?.addEventListener("change", (e) => {
      w.power = Number(e.target.value) || 1;
      actions.invalidateFromStage?.(4);
    });
    row.querySelector(".w-cutmode-sel")?.addEventListener("change", (e) => {
      w.cutoff_mode = e.target.value;
      actions.invalidateFromStage?.(4);
      actions.render();
    });
    row.querySelector(".w-cutm-inp")?.addEventListener("change", (e) => {
      w.cutoff_m = Math.max(50, Number(e.target.value) || 1000);
      actions.invalidateFromStage?.(4);
    });
    row.querySelector(".w-prof-sel")?.addEventListener("change", (e) => {
      w.profile = e.target.value;
      actions.invalidateFromStage?.(4);
    });
    row.querySelector(".w-del-btn")?.addEventListener("click", () => {
      if (spCfg.weights.length <= 1) return actions.showToast?.("Legalább 1 térbeli súlymátrix kötelező.", "warn");
      spCfg.weights.splice(idx, 1);
      actions.invalidateFromStage?.(4);
      actions.render();
    });
  });

  container.querySelector("#btnCopyStage4Latex")?.addEventListener("click", () => actions.copyLatexTable("stage4"));
  container.querySelector("#btnCopyStage4ParamsLatex")?.addEventListener("click", () => actions.copyLatexTable("stage4_params"));

  container.querySelectorAll(".sp-param-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.stage4ParamKind = btn.getAttribute("data-param-kind");
      actions.render();
    });
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
          error_y: {
            type: "data",
            symmetric: false,
            array: activeImpacts.map((r) => Math.max(0, (r.direct.ci_hi ?? r.direct.est) - r.direct.est)),
            arrayminus: activeImpacts.map((r) => Math.max(0, r.direct.est - (r.direct.ci_lo ?? r.direct.est))),
            visible: true,
          },
          name: "Direct",
          type: "bar",
          marker: { color: "#1d4ed8" },
        },
        {
          x: activeImpacts.map((r) => r.variable),
          y: activeImpacts.map((r) => r.indirect.est),
          error_y: {
            type: "data",
            symmetric: false,
            array: activeImpacts.map((r) => Math.max(0, (r.indirect.ci_hi ?? r.indirect.est) - r.indirect.est)),
            arrayminus: activeImpacts.map((r) => Math.max(0, r.indirect.est - (r.indirect.ci_lo ?? r.indirect.est))),
            visible: true,
          },
          name: "Indirect (Spillover)",
          type: "bar",
          marker: { color: "#0d9488" },
        },
      ],
      getPlotlyThemeLayout({
        barmode: "group",
        title: { text: `LeSage & Pace (${labelKind}) Közvetlen vs. Szomszédsági Spillover Hatás (95% CI)`, font: { size: 12 } },
        margin: { t: 34, r: 16, b: 60, l: 48 },
        height: 250,
      }),
      { responsive: true, displayModeBar: false }
    );
  }
}

function renderWeightRow(w, idx) {
  const isFixed = w.cutoff_mode === "fixed_m" || w.cutoff_mode === "fixed";
  return `
    <div data-w-idx="${idx}" style="display:flex;gap:8px;align-items:center;padding:6px 8px;margin-bottom:6px;border:1px solid var(--border);border-radius:6px;background:var(--surface-alt);flex-wrap:wrap;font-size:11px;">
      <strong>${idx === 0 ? "W₁" : `W${idx + 1}`}:</strong>
      <input type="text" class="w-id-inp mono" value="${escapeHtml(w.id)}" aria-label="Súlymátrix azonosító" style="width:92px;" />
      <span class="diag-chip">${escapeHtml(w.type)}</span>
      ${
        w.type === "knn"
          ? `<span>k=</span><input type="number" class="w-k-inp" value="${Number(w.k) || 5}" min="2" max="200" style="width:58px;" />
             <label style="display:inline-flex;align-items:center;gap:4px;"><input type="checkbox" class="w-dedup-chk" ${w.dedup_colocated ? "checked" : ""} /> Társasház dedup</label>`
          : w.type === "network_idw"
            ? `<select class="w-prof-sel" aria-label="Hálózati profil"><option value="walk" ${w.profile !== "bike" && w.profile !== "drive" ? "selected" : ""}>Gyalogos</option><option value="bike" ${w.profile === "bike" ? "selected" : ""}>Kerékpár</option><option value="drive" ${w.profile === "drive" ? "selected" : ""}>Autós</option></select>
               <select class="w-pow-sel" aria-label="IDW kitevő"><option value="1" ${Number(w.power || 1) === 1 ? "selected" : ""}>1/d</option><option value="2" ${Number(w.power) === 2 ? "selected" : ""}>1/d²</option></select>
               <span>cutoff=</span><input type="number" class="w-cutm-inp" value="${Number(w.cutoff_m) || 1500}" min="50" max="50000" step="50" style="width:74px;" /><span>m</span>`
            : `<select class="w-pow-sel" aria-label="IDW kitevő"><option value="1" ${Number(w.power || 1) === 1 ? "selected" : ""}>1/d</option><option value="2" ${Number(w.power) === 2 ? "selected" : ""}>1/d²</option></select>
               <select class="w-cutmode-sel" aria-label="IDW küszöbsugár mód"><option value="auto_connectivity" ${!isFixed ? "selected" : ""}>Auto (P99)</option><option value="fixed_m" ${isFixed ? "selected" : ""}>Fix (m)</option></select>
               <input type="number" class="w-cutm-inp" value="${Number(w.cutoff_m) || 500}" min="50" max="50000" step="50" style="width:74px;" ${isFixed ? "" : "disabled"} /><span>m</span>`
      }
      <button type="button" class="btn btn-sm w-del-btn" style="margin-left:auto;color:var(--fail);" aria-label="Súlymátrix törlése">✕</button>
    </div>
  `;
}

function renderSpatialParamsCard(s4, paramKind) {
  const map = { sem: s4.sem_params || [], sar: s4.sar_params || [], sdm: s4.sdm_params || [] };
  const rows = map[paramKind] || [];
  if (!map.sem.length && !map.sar.length) return "";
  return `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">4D. Teljes Térbeli Paraméter-Tábla (${escapeHtml(paramKind.toUpperCase())} — ${(s4.se_mode || "qml").toUpperCase()} SE)</h3>
          <div class="card-subtitle">Együtthatók közvetlenül a SAR, SEM és SDM modellekből</div>
        </div>
        <div class="spec-tabs" style="margin:0;">
          <button type="button" class="spec-tab sp-param-tab ${paramKind === "sem" ? "active" : ""}" data-param-kind="sem">SEM</button>
          <button type="button" class="spec-tab sp-param-tab ${paramKind === "sar" ? "active" : ""}" data-param-kind="sar">SAR</button>
          <button type="button" class="spec-tab sp-param-tab ${paramKind === "sdm" ? "active" : ""}" data-param-kind="sdm" ${map.sdm.length ? "" : "disabled"}>SDM</button>
        </div>
      </div>
      <div class="table-wrap" style="max-height:240px;">
        <table class="academic">
          <thead><tr><th>Változó</th><th class="num">β̂ / ρ / λ</th><th class="num">SE</th><th class="num">z</th><th class="num">p</th></tr></thead>
          <tbody>
            ${
              rows.length
                ? rows
                    .map(
                      (p) => `<tr>
                        <td><strong>${escapeHtml(p.name)}</strong></td>
                        <td class="num">${p.beta}<span class="sig-stars">${escapeHtml(p.sig)}</span></td>
                        <td class="num">${p.se}</td>
                        <td class="num">${p.z}</td>
                        <td class="num">${p.p}</td>
                      </tr>`
                    )
                    .join("")
                : `<tr><td colspan="5" class="muted">Ez a modell nem becsült paraméterekkel (pl. SDM alulhatározott).</td></tr>`
            }
          </tbody>
        </table>
      </div>
    </div>
  `;
}

export function renderStage5(container, ctx) {
  const { state, actions } = ctx;
  if (!state.datasetInfo) {
    container.innerHTML = `<div class="card"><div class="card-subtitle">Előbb tölts fel adatot az 1. fázisban.</div></div>`;
    return;
  }

  const gwrCfg = state.recipe.spatial_engine.gwr;
  if (!gwrCfg.mode) gwrCfg.mode = "gwr";
  const activeModels = state.recipe.models.filter((m) => m.active);
  if (activeModels.length && !activeModels.some((m) => m.id === gwrCfg.focal_model_id)) {
    gwrCfg.focal_model_id = state.recipe.spatial_engine.active_model_id || activeModels[activeModels.length - 1].id;
  }
  const s5 = state.stage5Res;

  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">5A. Szingularitás-Védett Adaptív Bisquare GWR / MGWR (Guard #6)</h3>
          <div class="card-subtitle">Lokális invariáns dummy kizárás + Szelektív Ridge (κ &gt; ${escapeHtml(gwrCfg.kappa_thresh)}) + MC nem-stacionaritás teszt</div>
        </div>
        <button type="button" class="btn btn-primary btn-sm" id="btnRunStage5">▶ ${gwrCfg.mode === "mgwr" ? "MGWR" : "GWR"} Futtatása</button>
      </div>

      <div class="grid-3">
        <div class="field-row">
          <label for="selGwrMode">Módszer (GWR vs. MGWR Backfitting)</label>
          <select id="selGwrMode">
            <option value="gwr" ${gwrCfg.mode !== "mgwr" ? "selected" : ""}>Standard Adaptív GWR (egységes b*)</option>
            <option value="mgwr" ${gwrCfg.mode === "mgwr" ? "selected" : ""}>MGWR (változónkénti sávszélesség b*<sub>j</sub>)</option>
          </select>
        </div>
        <div class="field-row">
          <label for="selGwrModel">GWR Fókusz Modell</label>
          <select id="selGwrModel">
            ${activeModels.map((m) => `<option value="${escapeHtml(m.id)}" ${m.id === gwrCfg.focal_model_id ? "selected" : ""}>${escapeHtml(m.name)}</option>`).join("")}
          </select>
        </div>
        <div class="field-row">
          <label for="inpGwrKappa">Ridge κ küszöb &amp; Mikro-jitter (m)</label>
          <div style="display:flex;gap:6px;">
            <input type="number" id="inpGwrKappa" aria-label="GWR Ridge kappa küszöb" value="${Number(gwrCfg.kappa_thresh) || 1000}" min="50" max="100000" step="100" style="flex:1;" />
            <input type="number" id="inpGwrJitter" aria-label="GWR mikro jitter" value="${Number(gwrCfg.jitter_m ?? 1)}" min="0" max="20" step="0.5" style="width:72px;" />
          </div>
        </div>
      </div>

      <div id="gwrProgressWrap" style="margin-top:8px;${state.stage5Busy ? "" : "display:none;"}">
        <div class="card-subtitle" id="gwrProgressLabel">${escapeHtml(state.stage5ProgressText || "GWR optimalizálása...")}</div>
        <div class="prog-bar-wrap"><div class="prog-bar-fill" id="gwrProgressBar" style="width:${Number(state.stage5ProgressPct || 0)}%;"></div></div>
      </div>
    </div>

    ${
      s5 && s5.ok
        ? `
      ${(s5.warnings || []).map((w) => `<div class="alert-box">🛡️ ${escapeHtml(translateVerdict(w))}</div>`).join("")}
      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">5B. Globális OLS vs. Lokális ${(s5.mode || "gwr").toUpperCase()} &amp; Nem-Stacionaritás</h3>
            <div class="card-subtitle">b* = ${s5.optimal_bw} (${s5.bw_pct_sample}%) | OLS Adj. R² = ${s5.global_ols_adj_r2} → GWR Adj. R² = ${s5.gwr_adj_r2} | tr(S) = ${s5.tr_s} | ENP = ${s5.eff_df ?? "—"} | Helyi dummy kizárás: ${s5.n_local_dropped_locs ?? 0} pont</div>
          </div>
          <button type="button" class="btn btn-sm" id="btnCopyStage5Latex">📋 GWR LaTeX Másolása</button>
        </div>
        <div id="plotGwrBw" class="plot-box"></div>
        <div class="table-wrap" style="margin-top:8px;">
          <table class="academic">
            <thead><tr><th>Változó</th><th class="num">OLS β̂</th><th class="num">Min</th><th class="num">Q1</th><th class="num">Medián</th><th class="num">Q3</th><th class="num">Max</th><th class="num">Lok. szig. % (Nom / FDR / dSF)</th><th class="num">MC p</th><th>FBC (2002)</th></tr></thead>
            <tbody>
              ${(s5.param_summaries || [])
                .map(
                  (r) => `<tr>
                    <td><strong>${escapeHtml(r.name)}</strong>${r.mgwr_bw ? ` <span class="card-subtitle mono">[b=${r.mgwr_bw}]</span>` : ""}</td>
                    <td class="num">${r.ols_beta}</td>
                    <td class="num">${r.min}</td><td class="num">${r.q1}</td><td class="num">${r.median}</td><td class="num">${r.q3}</td><td class="num">${r.max}</td>
                    <td class="num mono">${r.pct_sig_95}% / ${r.pct_sig_fdr ?? r.pct_sig_95}% / ${r.pct_sig_sdf ?? "—"}%</td>
                    <td class="num mono">${r.mc_nonstat_p ?? "—"}</td>
                    <td><span class="diag-chip ${r.non_stationary ? "warn" : "pass"}" title="${escapeHtml(r.stationarity_verdict || "")}">${r.non_stationary ? "Nem-stac." : "Stac."}</span></td>
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
          <h3 class="card-title">5C. Publikációs Export Stúdió (ZIP + QML + SVG + LaTeX + GeoJSON)</h3>
          <div class="card-subtitle">Teljes archívum QGIS <code>.qml</code> stílusokkal, Plotly <code>.svg</code> vektorábrákkal és LaTeX táblákkal</div>
        </div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <button type="button" class="btn btn-primary" id="btnExportZip">📦 Teljes Tudományos Csomag (.ZIP)</button>
        <button type="button" class="btn" id="btnExportLatex">📋 OLS LaTeX (.tex)</button>
        <button type="button" class="btn" id="btnExportGeojson">🗺️ QGIS Gazdagított GeoJSON</button>
        <button type="button" class="btn" id="btnExportWorkspaceS5">💾 Teljes Projekt (.geospec.json)</button>
        <button type="button" class="btn" id="btnExportOsmBundleS5">💾 Offline OSM (.osm_bundle.json)</button>
        <button type="button" class="btn" id="btnExportRecipe">💾 Recept (_recipe.json)</button>
      </div>
    </div>
  `;

  container.querySelector("#btnRunStage5").addEventListener("click", () => actions.runStage5());
  container.querySelector("#selGwrMode").addEventListener("change", (e) => {
    gwrCfg.mode = e.target.value;
    actions.invalidateFromStage?.(5);
    actions.render();
  });
  container.querySelector("#selGwrModel").addEventListener("change", (e) => {
    gwrCfg.focal_model_id = e.target.value;
    actions.invalidateFromStage?.(5);
  });
  container.querySelector("#inpGwrJitter").addEventListener("change", (e) => {
    gwrCfg.jitter_m = Math.max(0, Number(e.target.value) || 0);
    actions.invalidateFromStage?.(5);
  });
  container.querySelector("#inpGwrKappa").addEventListener("change", (e) => {
    gwrCfg.kappa_thresh = Math.max(50, Number(e.target.value) || 1000);
    actions.invalidateFromStage?.(5);
  });

  container.querySelector("#btnCopyStage5Latex")?.addEventListener("click", () => actions.copyLatexTable("stage5"));
  container.querySelector("#btnExportZip").addEventListener("click", () => actions.exportBundle("zip"));
  container.querySelector("#btnExportWorkspaceS5").addEventListener("click", () => actions.exportWorkspace());
  container.querySelector("#btnExportLatex").addEventListener("click", () => actions.copyLatexTable());
  container.querySelector("#btnExportGeojson").addEventListener("click", () => actions.exportGeojsonOnly());
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
