/* Stage 2: Sample Waterfall, Missingness, Descriptive Statistics, Normality, Box-Cox & Stepwise + Gelbach (2016) OVB Decomposition */

import { escapeHtml, getPlotlyThemeLayout } from "../utils/sanitize.js";

export function renderStage2(container, ctx) {
  const { state, actions } = ctx;
  if (!state.datasetInfo) {
    container.innerHTML = `<div class="card"><div class="card-subtitle">Előbb tölts fel és rögzíts egy adatbázist az 1. fázisban.</div></div>`;
    return;
  }

  const chain = state.recipe.composition_chain;
  const activeModels = state.recipe.models.filter((m) => m.active);
  const refModel = activeModels.find((m) => m.id === state.recipe.spatial_engine.active_model_id) || activeModels[activeModels.length - 1];
  const spatialTerms = refModel ? refModel.spatial_terms || [] : [];
  const covCols = refModel ? (refModel.covariates || []).map((c) => c.col) : [];
  const s2 = state.stage2Res;

  const wf = s2?.sample_waterfall || null;
  const pvm = Array.isArray(s2?.per_variable_missing) ? s2.per_variable_missing : [];

  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">2A. Összetételi Hatás Lépcső &amp; Gelbach (2016) Dekompozíció (Guard #3)</h3>
          <div class="card-subtitle">Ellenőrzi, hogy a térbeli hatáskülönbséget a fizikai kontrollok összetételi hatása okozza-e</div>
        </div>
        <button class="btn btn-primary btn-sm" id="btnRunStage2">▶ 2. Fázis Számítása</button>
      </div>

      <div class="grid-2" style="margin-bottom:8px;">
        <div class="field-row">
          <label>Fókusz térbeli változó (attenuáció vizsgálata)</label>
          <select id="selFocalTerm">
            ${
              spatialTerms.length
                ? spatialTerms
                    .map((st) => {
                      const ref = `${st.target_id}::${st.transform}`;
                      return `<option value="${escapeHtml(ref)}" ${chain.focal_term_ref === ref ? "selected" : ""}>${escapeHtml(st.target_id)} (${escapeHtml(st.transform)})</option>`;
                    })
                    .join("")
                : `<option value="">— Adj meg térbeli változót a 3. fázisban —</option>`
            }
          </select>
        </div>
        <div class="field-row">
          <label>Kontroll-blokkok kezelése (Step 0 → Step K)</label>
          <div style="display:flex;gap:6px;">
            <button class="btn btn-sm" id="btnAutoBlocks">⚡ Blokkok automatikus létrehozása kovariánsokból</button>
            <button class="btn btn-sm" id="btnAddBlock">+ Üres blokk</button>
          </div>
        </div>
      </div>

      ${
        !spatialTerms.length
          ? `<div class="alert-box info" style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px;">
               <span>ℹ️ A Gelbach (2016) összetételi dekompozíció a <strong>3. fázisban</strong> definiált térbeli távolsági változóra épül. Jelenleg nincs aktív térbeli tag a referenciamodellben (a leíró statisztika és a minta-vízesés ettől függetlenül lefut).</span>
               <button class="btn btn-sm" id="btnGoStage3FromS2">→ 3. Fázis: Térbeli változó megadása</button>
             </div>`
          : ""
      }

      <div id="blocksArea">
        ${(chain.blocks || [])
          .map(
            (blk, bidx) => `
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;padding:6px 8px;border:1px solid var(--border);border-radius:6px;background:var(--surface-alt);" data-blk-idx="${bidx}">
            <input type="text" class="blk-name" value="${escapeHtml(blk.name)}" style="width:150px;" />
            <select class="blk-terms" multiple size="2" style="flex:1;height:42px;">
              ${covCols.map((c) => `<option value="${escapeHtml(c)}" ${(blk.terms || []).includes(c) ? "selected" : ""}>${escapeHtml(c)}</option>`).join("")}
            </select>
            <button class="btn btn-sm blk-del">✕</button>
          </div>`
          )
          .join("")}
      </div>
    </div>

    ${
      s2 && s2.ok
        ? `
      ${
        wf
          ? `<div class="card">
               <div class="card-header">
                 <div>
                   <h3 class="card-title">2B. Minta-vízesés &amp; Hiányzó Adatok (Közös Analitikus Minta)</h3>
                   <div class="card-subtitle">Nyers soroktól (N = ${wf.n_total}) a lezárt közös mintáig (N = ${wf.n_common_analytic})</div>
                 </div>
               </div>
               <div class="grid-2">
                 <div class="table-wrap">
                   <table class="academic">
                     <thead><tr><th>Szűrési lépcső</th><th class="num">Megmaradó N</th><th class="num">Kiesett sorok</th></tr></thead>
                     <tbody>
                       <tr><td>Nyers betöltött tábla</td><td class="num">${wf.n_total}</td><td class="num">—</td></tr>
                       <tr><td>Érvényes WGS84 koordináta (lat/lon)</td><td class="num">${wf.n_geocoded}</td><td class="num">-${wf.dropped_non_geocoded ?? 0}</td></tr>
                       <tr><td>Kizárási flag / szelekciós szűrő után</td><td class="num">${wf.n_after_excl_flag}</td><td class="num">-${wf.dropped_excl_flag ?? 0}</td></tr>
                       <tr><td>Érvényes függő változó (Y &gt; 0 ha log/ratio)</td><td class="num">${wf.n_after_y_valid}</td><td class="num">-${wf.dropped_y_missing_or_nonpos ?? 0}</td></tr>
                       <tr><td><strong>Közös analitikus minta (minden X &amp; távolság teljes)</strong></td><td class="num"><strong>${wf.n_common_analytic}</strong></td><td class="num"><strong>-${wf.dropped_covs_or_dist ?? 0}</strong></td></tr>
                     </tbody>
                   </table>
                 </div>
                 <div class="table-wrap" style="max-height:220px;">
                   <table class="academic">
                     <thead><tr><th>Változó / Oszlop</th><th class="num">Hiányzó (N)</th><th class="num">Arány (%)</th></tr></thead>
                     <tbody>
                       ${
                         pvm.length
                           ? pvm
                               .map(
                                 (m) => `<tr>
                                   <td>${escapeHtml(m.variable)}</td>
                                   <td class="num">${m.n_missing}</td>
                                   <td class="num ${Number(m.pct_missing) > 10 ? "text-warn" : ""}">${m.pct_missing}%</td>
                                 </tr>`
                               )
                               .join("")
                           : `<tr><td colspan="3" class="muted">Nincs regisztrált kovariáns vagy hiányzó adat.</td></tr>`
                       }
                     </tbody>
                   </table>
                 </div>
               </div>
             </div>`
          : ""
      }

      ${
        s2.descriptives?.boxcox_info
          ? `<div class="alert-box info">
               <strong>Box–Cox Profile-MLE (Uncond.):</strong> λ̂ = <span class="mono">${escapeHtml(s2.descriptives.boxcox_info.lambda_mle)}</span>
               (alkalmazva: <span class="mono">${escapeHtml(s2.descriptives.boxcox_info.lambda_used)}</span>) |
               LR teszt λ=0 (ln) ellen: p = <span class="mono">${escapeHtml(s2.descriptives.boxcox_info.p_vs_log)}</span> |
               λ=1 (lineáris) ellen: p = <span class="mono">${escapeHtml(s2.descriptives.boxcox_info.p_vs_linear)}</span>
             </div>`
          : ""
      }

      ${
        s2.composition?.ok
          ? `<div class="card">
               <div class="card-header">
                 <div>
                   <h3 class="card-title">Lépcsőzetes Összetételi Hatás &amp; Gelbach OVB (${escapeHtml(s2.composition.focal_term)})</h3>
                   <div class="card-subtitle">Közös N = ${s2.composition.n_obs} | Nyers β₀ = ${s2.composition.beta_uncontrolled} → Kontrollált β = ${s2.composition.beta_controlled} (Attenuáció: ${s2.composition.final_attenuation_pct}%)</div>
                 </div>
               </div>
               ${
                 Math.abs(s2.composition.final_attenuation_pct) > state.recipe.guards_cfg.attenuation_warn_pct
                   ? `<div class="alert-box">⚠️ <strong>Guard #3 Riasztás:</strong> A kontrollok a nyers térbeli hatás ${escapeHtml(s2.composition.final_attenuation_pct)}%-át elnyelték (erős összetételi hatás${s2.composition.sign_reversed ? " + előjelváltás" : ""}).</div>`
                   : ""
               }
               <div class="grid-2">
                 <div id="plotStepwise" class="plot-box"></div>
                 <div id="plotGelbach" class="plot-box"></div>
               </div>
               <div class="table-wrap" style="margin-top:8px;">
                 <table class="academic">
                   <thead><tr><th>Lépcső</th><th>Hozzáadott blokk</th><th class="num">β̂</th><th class="num">HC3 SE</th><th class="num">p-érték</th><th class="num">Adj. R²</th><th class="num">Attenuáció (%)</th></tr></thead>
                   <tbody>
                     ${(s2.composition.steps || [])
                       .map(
                         (st) => `<tr>
                           <td>${escapeHtml(st.step)}</td><td>${escapeHtml(st.added_block)}</td>
                           <td class="num">${st.beta}<span class="sig-stars">${escapeHtml(st.sig)}</span></td>
                           <td class="num">${st.se}</td><td class="num">${st.p}</td>
                           <td class="num">${st.adj_r2}</td><td class="num">${st.attenuation_pct}%</td>
                         </tr>`
                       )
                       .join("")}
                   </tbody>
                 </table>
               </div>
               ${
                 s2.composition.gelbach?.length
                   ? `<div class="table-wrap" style="margin-top:8px;">
                        <table class="academic">
                          <thead><tr><th>Gelbach (2016) Blokk</th><th class="num">OVB δ̂</th><th class="num">SE(δ̂)</th><th class="num">t-érték</th><th class="num">p</th><th class="num">Részesedés (%)</th></tr></thead>
                          <tbody>
                            ${s2.composition.gelbach
                              .map(
                                (g) => `<tr>
                                  <td>${escapeHtml(g.block)}</td>
                                  <td class="num">${g.delta}<span class="sig-stars">${escapeHtml(g.sig)}</span></td>
                                  <td class="num">${g.se}</td><td class="num">${g.t}</td><td class="num">${g.p}</td>
                                  <td class="num">${g.share_of_gap_pct}%</td>
                                </tr>`
                              )
                              .join("")}
                          </tbody>
                        </table>
                      </div>`
                   : ""
               }
             </div>`
          : s2.composition?.reason || s2.composition?.error
            ? `<div class="alert-box" style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;">
                 <span>⚠️ Összetételi dekompozíció nem futott le: ${escapeHtml(s2.composition.reason || s2.composition.error)}</span>
                 ${!spatialTerms.length ? `<button class="btn btn-sm" id="btnGoStage3FromComp">→ 3. Fázis</button>` : ""}
               </div>`
            : ""
      }

      ${
        s2.descriptives?.ok
          ? `<div class="card">
               <div class="card-header"><h3 class="card-title">2C. Leíró Statisztika &amp; Normalitásvizsgálat (N = ${s2.descriptives.n})</h3></div>
               <div class="table-wrap" style="margin-bottom:10px;">
                 <table class="academic">
                   <thead><tr><th>Változó</th><th class="num">N</th><th class="num">Átlag</th><th class="num">Szórás</th><th class="num">Min</th><th class="num">Q1</th><th class="num">Medián</th><th class="num">Q3</th><th class="num">Max</th></tr></thead>
                   <tbody>
                     ${(s2.descriptives.descriptives || [])
                       .map(
                         (r) => `<tr>
                           <td><strong>${escapeHtml(r.variable)}</strong></td><td class="num">${r.n}</td>
                           <td class="num">${r.mean}</td><td class="num">${r.sd}</td>
                           <td class="num">${r.min}</td><td class="num">${r.q1}</td><td class="num">${r.median}</td><td class="num">${r.q3}</td><td class="num">${r.max}</td>
                         </tr>`
                       )
                       .join("")}
                   </tbody>
                 </table>
               </div>
               <div class="table-wrap">
                 <table class="academic">
                   <thead><tr><th>Eloszlás</th><th class="num">Ferdeség</th><th class="num">Többlet-csúcsosság</th><th class="num">D'Agostino K² (p)</th><th class="num">Jarque–Bera (p)</th><th class="num">Shapiro W (p)</th></tr></thead>
                   <tbody>
                     ${(s2.descriptives.normality || [])
                       .map(
                         (r) => `<tr>
                           <td><strong>${escapeHtml(r.variable)}</strong></td>
                           <td class="num">${r.skewness}</td><td class="num">${r.excess_kurtosis}</td>
                           <td class="num">${r.k2_stat ?? "—"} (${r.k2_p ?? "—"})</td>
                           <td class="num">${r.jb_stat ?? "—"} (${r.jb_p ?? "—"})</td>
                           <td class="num">${r.shapiro_w ?? "—"} (${r.shapiro_p ?? "—"})</td>
                         </tr>`
                       )
                       .join("")}
                   </tbody>
                 </table>
               </div>
             </div>`
          : ""
      }
    `
        : ""
    }
  `;

  container.querySelector("#btnRunStage2").addEventListener("click", () => actions.runStage2());
  const goS3 = () => {
    ctx.state.activeStage = 3;
    actions.render();
  };
  container.querySelector("#btnGoStage3FromS2")?.addEventListener("click", goS3);
  container.querySelector("#btnGoStage3FromComp")?.addEventListener("click", goS3);
  container.querySelector("#selFocalTerm")?.addEventListener("change", (e) => {
    chain.focal_term_ref = e.target.value;
  });
  container.querySelector("#btnAutoBlocks").addEventListener("click", () => {
    chain.blocks = covCols.map((c) => ({ name: c, terms: [c] }));
    if (spatialTerms.length && !chain.focal_term_ref) {
      chain.focal_term_ref = `${spatialTerms[0].target_id}::${spatialTerms[0].transform}`;
    }
    actions.render();
  });
  container.querySelector("#btnAddBlock").addEventListener("click", () => {
    chain.blocks.push({ name: `Blokk ${chain.blocks.length + 1}`, terms: [] });
    actions.render();
  });

  container.querySelectorAll("[data-blk-idx]").forEach((el) => {
    const idx = Number(el.getAttribute("data-blk-idx"));
    el.querySelector(".blk-name").addEventListener("change", (e) => {
      chain.blocks[idx].name = e.target.value;
    });
    el.querySelector(".blk-terms").addEventListener("change", (e) => {
      chain.blocks[idx].terms = Array.from(e.target.selectedOptions).map((o) => o.value);
    });
    el.querySelector(".blk-del").addEventListener("click", () => {
      chain.blocks.splice(idx, 1);
      actions.render();
    });
  });

  if (s2?.composition?.ok && window.Plotly) {
    const steps = s2.composition.steps || [];
    window.Plotly.newPlot(
      "plotStepwise",
      [
        {
          x: steps.map((s) => s.step),
          y: steps.map((s) => s.beta),
          error_y: { type: "data", array: steps.map((s) => 1.96 * s.se), visible: true },
          type: "scatter",
          mode: "lines+markers",
          line: { color: "#0f766e", width: 2.5 },
          name: "β̂ (±95% CI)",
        },
      ],
      getPlotlyThemeLayout({
        title: { text: "Step 0 → Step K Attenuáció", font: { size: 12 } },
        margin: { t: 34, r: 14, b: 60, l: 48 },
        height: 250,
        shapes: [{ type: "line", x0: 0, x1: 1, xref: "paper", y0: 0, y1: 0, line: { dash: "dot", color: "#94a3b8" } }],
      }),
      { responsive: true, displayModeBar: false }
    );

    const gel = s2.composition.gelbach || [];
    if (gel.length) {
      window.Plotly.newPlot(
        "plotGelbach",
        [
          {
            y: gel.map((g) => g.block),
            x: gel.map((g) => g.delta),
            error_x: { type: "data", array: gel.map((g) => 1.96 * g.se), visible: true },
            type: "bar",
            orientation: "h",
            marker: { color: gel.map((g) => (g.delta >= 0 ? "#0d9488" : "#d97706")) },
          },
        ],
        getPlotlyThemeLayout({
          title: { text: "Gelbach (2016) Sorrendfüggetlen OVB (δ̂)", font: { size: 12 } },
          margin: { t: 34, r: 16, b: 40, l: 96 },
          height: 250,
        }),
        { responsive: true, displayModeBar: false }
      );
    }
  }
}
