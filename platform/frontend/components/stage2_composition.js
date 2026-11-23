/* Stage 2: Descriptive Statistics, Normality, Box-Cox & Stepwise + Gelbach (2016) OVB Decomposition */

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

  container.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">2A. Összetételi Hatás Lépcső & Gelbach (2016) Dekompozíció (Guard #3)</h3>
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
                      return `<option value="${ref}" ${chain.focal_term_ref === ref ? "selected" : ""}>${st.target_id} (${st.transform})</option>`;
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

      <div id="blocksArea">
        ${(chain.blocks || [])
          .map(
            (blk, bidx) => `
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;padding:6px 8px;border:1px solid var(--border);border-radius:6px;background:var(--surface-alt);" data-blk-idx="${bidx}">
            <input type="text" class="blk-name" value="${blk.name}" style="width:150px;" />
            <select class="blk-terms" multiple size="2" style="flex:1;height:42px;">
              ${covCols.map((c) => `<option value="${c}" ${(blk.terms || []).includes(c) ? "selected" : ""}>${c}</option>`).join("")}
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
        s2.descriptives?.boxcox_info
          ? `<div class="alert-box info">
               <strong>Box–Cox Profile-MLE:</strong> λ̂ = <span class="mono">${s2.descriptives.boxcox_info.lambda_mle}</span>
               (alkalmazva: <span class="mono">${s2.descriptives.boxcox_info.lambda_used}</span>) |
               LR teszt λ=0 (ln) ellen: p = <span class="mono">${s2.descriptives.boxcox_info.p_vs_log}</span> |
               λ=1 (lineáris) ellen: p = <span class="mono">${s2.descriptives.boxcox_info.p_vs_linear}</span>
             </div>`
          : ""
      }

      ${
        s2.composition?.ok
          ? `<div class="card">
               <div class="card-header">
                 <div>
                   <h3 class="card-title">Lépcsőzetes Összetételi Hatás & Gelbach OVB (${s2.composition.focal_term})</h3>
                   <div class="card-subtitle">Közös N = ${s2.composition.n_obs} | Nyers β₀ = ${s2.composition.beta_uncontrolled} → Kontrollált β = ${s2.composition.beta_controlled} (Attenuáció: ${s2.composition.final_attenuation_pct}%)</div>
                 </div>
               </div>
               ${
                 Math.abs(s2.composition.final_attenuation_pct) > state.recipe.guards_cfg.attenuation_warn_pct
                   ? `<div class="alert-box">⚠️ <strong>Guard #3 Riasztás:</strong> A kontrollok a nyers térbeli hatás ${s2.composition.final_attenuation_pct}%-át elnyelték (erős összetételi hatás${s2.composition.sign_reversed ? " + előjelváltás" : ""}).</div>`
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
                     ${s2.composition.steps
                       .map(
                         (st) => `<tr>
                           <td>${st.step}</td><td>${st.added_block}</td>
                           <td class="num">${st.beta}<span class="sig-stars">${st.sig}</span></td>
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
                                  <td>${g.block}</td>
                                  <td class="num">${g.delta}<span class="sig-stars">${g.sig}</span></td>
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
          : ""
      }

      ${
        s2.descriptives?.ok
          ? `<div class="card">
               <div class="card-header"><h3 class="card-title">2B. Leíró Statisztika & Normalitásvizsgálat (N = ${s2.descriptives.n})</h3></div>
               <div class="table-wrap" style="margin-bottom:10px;">
                 <table class="academic">
                   <thead><tr><th>Változó</th><th class="num">N</th><th class="num">Átlag</th><th class="num">Szórás</th><th class="num">Min</th><th class="num">Q1</th><th class="num">Medián</th><th class="num">Q3</th><th class="num">Max</th></tr></thead>
                   <tbody>
                     ${s2.descriptives.descriptives
                       .map(
                         (r) => `<tr>
                           <td><strong>${r.variable}</strong></td><td class="num">${r.n}</td>
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
                     ${s2.descriptives.normality
                       .map(
                         (r) => `<tr>
                           <td><strong>${r.variable}</strong></td>
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

  // Render Plotly Stepwise & Gelbach charts if results exist
  if (s2?.composition?.ok && window.Plotly) {
    const steps = s2.composition.steps;
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
      {
        title: { text: "Step 0 → Step K Attenuáció", font: { size: 12 } },
        margin: { t: 34, r: 14, b: 60, l: 48 },
        height: 250,
        shapes: [{ type: "line", x0: 0, x1: 1, xref: "paper", y0: 0, y1: 0, line: { dash: "dot", color: "#94a3b8" } }],
      },
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
        {
          title: { text: "Gelbach (2016) Sorrendfüggetlen OVB (δ̂)", font: { size: 12 } },
          margin: { t: 34, r: 16, b: 40, l: 96 },
          height: 250,
        },
        { responsive: true, displayModeBar: false }
      );
    }
  }
}
