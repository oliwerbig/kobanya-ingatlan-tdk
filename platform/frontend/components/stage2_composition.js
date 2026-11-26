/* Stage 2: Sample Waterfall, Missingness, Dynamic y-Transform/Distribution, Descriptives & Gelbach (2016) OVB */

import { translateVerdict } from "../utils/i18n.js";
import { escapeHtml, getPlotlyThemeLayout, parseNumericScalar } from "../utils/sanitize.js";

export function renderStage2(container, ctx) {
  const { state, actions } = ctx;
  if (!state.datasetInfo) {
    container.innerHTML = `<div class="card"><div class="card-subtitle">Előbb tölts fel és rögzíts egy adatbázist az 1. fázisban.</div></div>`;
    return;
  }

  const ds = state.recipe.dataset_schema;
  const ySpec = ds.y_spec;
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
          <h3 class="card-title">2A. Függő Változó Skála &amp; Gelbach (2016) Összetételi Lépcső (Guard #3)</h3>
          <div class="card-subtitle">Box–Cox / ln(y) transzformáció és kontroll-blokkok attenuációs vizsgálata</div>
        </div>
        <button type="button" class="btn btn-primary btn-sm" id="btnRunStage2">▶ 2. Fázis Számítása</button>
      </div>

      <div class="grid-3" style="margin-bottom:8px;">
        <div class="field-row">
          <label for="s2YTrans">Függő változó (y) transzformáció</label>
          <select id="s2YTrans">
            <option value="log" ${ySpec.transform === "log" ? "selected" : ""}>ln(y) (Fél-logaritmikus)</option>
            <option value="boxcox" ${ySpec.transform === "boxcox" ? "selected" : ""}>Box–Cox (λ)</option>
            <option value="raw" ${ySpec.transform === "raw" ? "selected" : ""}>y (Nyers lineáris)</option>
            <option value="log1p" ${ySpec.transform === "log1p" ? "selected" : ""}>ln(y + shift)</option>
          </select>
        </div>
        <div class="field-row" style="${ySpec.transform === "boxcox" ? "" : "opacity:0.45;"}">
          <label for="s2BcMode">Box–Cox λ mód</label>
          <select id="s2BcMode" ${ySpec.transform === "boxcox" ? "" : "disabled"}>
            <option value="auto" ${ySpec.boxcox_lambda == null ? "selected" : ""}>Automatikus Profile-MLE</option>
            <option value="fixed" ${ySpec.boxcox_lambda != null ? "selected" : ""}>Fix λ érték</option>
          </select>
        </div>
        <div class="field-row" style="${ySpec.transform === "boxcox" && ySpec.boxcox_lambda != null ? "" : "opacity:0.45;"}">
          <label for="s2BcLambda">Fix Box–Cox λ ([-5, 5])</label>
          <input type="number" id="s2BcLambda" step="0.05" min="-5" max="5" value="${ySpec.boxcox_lambda != null ? escapeHtml(ySpec.boxcox_lambda) : 0.5}" ${ySpec.transform === "boxcox" && ySpec.boxcox_lambda != null ? "" : "disabled"} />
        </div>
      </div>

      <div class="grid-2" style="margin-bottom:8px;">
        <div class="field-row">
          <label for="selFocalTerm">Fókusz térbeli változó (attenuáció vizsgálata)</label>
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
            <button type="button" class="btn btn-sm" id="btnAutoBlocks">⚡ Blokkok automatikus létrehozása</button>
            <button type="button" class="btn btn-sm" id="btnAddBlock">+ Üres blokk</button>
          </div>
        </div>
      </div>

      ${
        !spatialTerms.length
          ? `<div class="alert-box info" style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px;">
               <span>ℹ️ A Gelbach (2016) összetételi dekompozíció a <strong>3. fázisban</strong> definiált térbeli változóra épül (a leíró statisztika és minta-vízesés ettől függetlenül lefut).</span>
               <button type="button" class="btn btn-sm" id="btnGoStage3FromS2">→ 3. Fázis: Térbeli változó</button>
             </div>`
          : ""
      }

      <div id="blocksArea">
        ${(chain.blocks || [])
          .map(
            (blk, bidx) => `
          <div style="display:flex;gap:8px;align-items:flex-start;margin-bottom:6px;padding:6px 8px;border:1px solid var(--border);border-radius:6px;background:var(--surface-alt);" data-blk-idx="${bidx}">
            <input type="text" class="blk-name" value="${escapeHtml(blk.name)}" aria-label="Blokk neve" style="width:150px;" />
            <div class="checkbox-pills-wrap" style="flex:1;">
              ${
                covCols.length
                  ? covCols
                      .map((c) => {
                        const checked = (blk.terms || []).includes(c);
                        return `<label class="checkbox-pill ${checked ? "checked" : ""}">
                          <input type="checkbox" class="blk-term-chk" value="${escapeHtml(c)}" ${checked ? "checked" : ""} />
                          <span>${escapeHtml(c)}</span>
                        </label>`;
                      })
                      .join("")
                  : `<span class="card-subtitle">Adj meg kovariánsokat a 3. fázisban.</span>`
              }
            </div>
            <button type="button" class="btn btn-sm blk-del" aria-label="Blokk törlése">✕</button>
          </div>`
          )
          .join("")}
      </div>
    </div>

    ${
      s2 && s2.ok
        ? `
      ${(s2.warnings || []).map((w) => `<div class="alert-box">⚠️ ${escapeHtml(w)}</div>`).join("")}
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
                       <tr><td>Érvényes WGS84 koordináta</td><td class="num">${wf.n_geocoded}</td><td class="num">-${wf.dropped_non_geocoded ?? 0}</td></tr>
                       <tr><td>Kizárási flag / szelekciós szűrő után</td><td class="num">${wf.n_after_excl_flag}</td><td class="num">-${wf.dropped_excl_flag ?? 0}</td></tr>
                       <tr><td>Érvényes függő változó</td><td class="num">${wf.n_after_y_valid}</td><td class="num">-${wf.dropped_y_missing_or_nonpos ?? 0}</td></tr>
                       <tr><td><strong>Közös analitikus minta</strong></td><td class="num"><strong>${wf.n_common_analytic}</strong></td><td class="num"><strong>-${wf.dropped_covs_or_dist ?? 0}</strong></td></tr>
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
                           : `<tr><td colspan="3" class="muted">Nincs regisztrált hiányzó érték.</td></tr>`
                       }
                     </tbody>
                   </table>
                 </div>
               </div>
             </div>`
          : ""
      }

      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">2B-2. Célváltozó (y) Eloszlás &amp; Box–Cox Profile-MLE</h3>
            <div class="card-subtitle">Aktív transzformáció összevetése nyers vs. transzformált skálán</div>
          </div>
        </div>
        ${
          s2.descriptives?.boxcox_info
            ? `<div class="alert-box info" style="margin-bottom:8px;">
                 <strong>Box–Cox Profile-MLE (Feltétel nélküli):</strong> λ̂ = <span class="mono">${escapeHtml(s2.descriptives.boxcox_info.lambda_mle)}</span>
                 (alkalmazva: <span class="mono">${escapeHtml(s2.descriptives.boxcox_info.lambda_used)}</span>) |
                 LR teszt λ=0 (ln) ellen: p = <span class="mono">${escapeHtml(s2.descriptives.boxcox_info.p_vs_log)}</span> |
                 λ=1 (lineáris) ellen: p = <span class="mono">${escapeHtml(s2.descriptives.boxcox_info.p_vs_linear)}</span>
               </div>`
            : ""
        }
        <div class="grid-2">
          <div id="plotYRawDist" class="plot-box"></div>
          <div id="plotYTransDist" class="plot-box"></div>
        </div>
      </div>

      ${
        s2.composition?.ok
          ? `<div class="card">
               <div class="card-header">
                 <div>
                   <h3 class="card-title">Lépcsőzetes Összetételi Hatás &amp; Gelbach OVB (${escapeHtml(s2.composition.focal_term)})</h3>
                   <div class="card-subtitle">Közös N = ${s2.composition.n_obs} | Nyers β₀ = ${s2.composition.beta_uncontrolled} → Kontrollált β = ${s2.composition.beta_controlled} (Attenuáció: ${s2.composition.final_attenuation_pct}%) | SE: ${escapeHtml(s2.composition.cov_type_used || "HC3")}</div>
                 </div>
               </div>
               ${
                 Math.abs(s2.composition.final_attenuation_pct) > state.recipe.guards_cfg.attenuation_warn_pct
                   ? `<div class="alert-box">⚠️ <strong>Guard #3 Riasztás:</strong> A kontrollok a nyers térbeli hatás ${escapeHtml(s2.composition.final_attenuation_pct)}%-át elnyelték${s2.composition.sign_reversed ? " (+ előjelváltás)" : ""}.</div>`
                   : ""
               }
               <div class="grid-2">
                 <div id="plotStepwise" class="plot-box"></div>
                 <div id="plotGelbach" class="plot-box"></div>
               </div>
               <div class="table-wrap" style="margin-top:8px;">
                 <table class="academic">
                   <thead><tr><th>Lépcső</th><th>Hozzáadott blokk</th><th class="num">β̂</th><th class="num">SE</th><th class="num">p-érték</th><th class="num">Adj. R²</th><th class="num">Attenuáció (%)</th></tr></thead>
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
                 <span>⚠️ Összetételi dekompozíció nem futott le: ${escapeHtml(translateVerdict(s2.composition.reason || s2.composition.error))}</span>
                 ${!spatialTerms.length ? `<button type="button" class="btn btn-sm" id="btnGoStage3FromComp">→ 3. Fázis</button>` : ""}
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

  container.querySelector("#s2YTrans")?.addEventListener("change", (e) => {
    ySpec.transform = e.target.value;
    actions.invalidateFromStage?.(2);
    actions.render();
  });
  container.querySelector("#s2BcMode")?.addEventListener("change", (e) => {
    if (e.target.value === "auto") {
      ySpec.boxcox_lambda = null;
    } else {
      ySpec.boxcox_lambda = s2?.descriptives?.boxcox_info?.lambda_mle ?? 0.5;
    }
    actions.invalidateFromStage?.(2);
    actions.render();
  });
  container.querySelector("#s2BcLambda")?.addEventListener("change", (e) => {
    const v = Number(e.target.value);
    ySpec.boxcox_lambda = Number.isFinite(v) ? Math.min(5, Math.max(-5, v)) : 0.5;
    actions.invalidateFromStage?.(2);
  });

  container.querySelector("#selFocalTerm")?.addEventListener("change", (e) => {
    chain.focal_term_ref = e.target.value;
    actions.invalidateFromStage?.(2);
  });
  container.querySelector("#btnAutoBlocks").addEventListener("click", () => {
    chain.blocks = covCols.map((c) => ({ name: c, terms: [c] }));
    if (spatialTerms.length && !chain.focal_term_ref) {
      chain.focal_term_ref = `${spatialTerms[0].target_id}::${spatialTerms[0].transform}`;
    }
    actions.invalidateFromStage?.(2);
    actions.render();
  });
  container.querySelector("#btnAddBlock").addEventListener("click", () => {
    chain.blocks.push({ name: `Blokk ${chain.blocks.length + 1}`, terms: [] });
    actions.invalidateFromStage?.(2);
    actions.render();
  });

  container.querySelectorAll("[data-blk-idx]").forEach((el) => {
    const idx = Number(el.getAttribute("data-blk-idx"));
    el.querySelector(".blk-name").addEventListener("change", (e) => {
      chain.blocks[idx].name = e.target.value;
      actions.invalidateFromStage?.(2);
    });
    el.querySelectorAll(".blk-term-chk").forEach((chk) => {
      chk.addEventListener("change", () => {
        chk.closest(".checkbox-pill")?.classList.toggle("checked", chk.checked);
        chain.blocks[idx].terms = Array.from(el.querySelectorAll(".blk-term-chk:checked")).map((c) => c.value);
        actions.invalidateFromStage?.(2);
      });
    });
    el.querySelector(".blk-del").addEventListener("click", () => {
      chain.blocks.splice(idx, 1);
      actions.invalidateFromStage?.(2);
      actions.render();
    });
  });

  if (s2?.ok && window.Plotly && state.columns) {
    // Render Raw y vs Transformed y histograms
    const numCol = ySpec.num_col;
    const denCol = ySpec.den_col;
    const rawArr = state.columns[numCol] || [];
    const denArr = ySpec.mode === "ratio" && denCol ? state.columns[denCol] || [] : null;
    const yValsRaw = [];
    for (let i = 0; i < rawArr.length; i++) {
      let v = parseNumericScalar(rawArr[i]);
      if (denArr && Number.isFinite(v)) {
        const dv = parseNumericScalar(denArr[i]);
        v = dv > 0 ? v / dv : NaN;
      }
      if (Number.isFinite(v)) yValsRaw.push(v);
    }
    const posRaw = yValsRaw.filter((v) => v > 0);
    const geomMeanY =
      Number(s2.descriptives?.boxcox_info?.scale_geom_mean) ||
      (posRaw.length ? Math.exp(posRaw.reduce((acc, v) => acc + Math.log(v), 0) / posRaw.length) : 1);
    const yValsTrans = yValsRaw
      .filter((v) => ySpec.transform === "raw" || (ySpec.transform === "log1p" ? v + (ySpec.log1p_shift ?? 1) > 0 : v > 0))
      .map((v) => {
        if (ySpec.transform === "log") return Math.log(v);
        if (ySpec.transform === "log1p") return Math.log(v + (ySpec.log1p_shift ?? 1));
        if (ySpec.transform === "boxcox") {
          const lam = s2.descriptives?.boxcox_info?.lambda_used ?? ySpec.boxcox_lambda ?? 0;
          const vs = v / (geomMeanY > 0 ? geomMeanY : 1);
          return Math.abs(lam) < 1e-8 ? Math.log(vs) : (Math.pow(vs, lam) - 1) / lam;
        }
        return v;
      });

    if (yValsRaw.length && container.querySelector("#plotYRawDist")) {
      window.Plotly.newPlot(
        "plotYRawDist",
        [{ x: yValsRaw, type: "histogram", marker: { color: "#1d4ed8", opacity: 0.78 }, nbinsx: 28 }],
        getPlotlyThemeLayout({ title: { text: "Nyers y eloszlása", font: { size: 12 } }, margin: { t: 32, r: 12, b: 38, l: 42 }, height: 220 }),
        { responsive: true, displayModeBar: false }
      );
    }
    if (yValsTrans.length && container.querySelector("#plotYTransDist")) {
      window.Plotly.newPlot(
        "plotYTransDist",
        [{ x: yValsTrans, type: "histogram", marker: { color: "#0f766e", opacity: 0.8 }, nbinsx: 28 }],
        getPlotlyThemeLayout({ title: { text: `Transzformált y (${escapeHtml(ySpec.transform)})`, font: { size: 12 } }, margin: { t: 32, r: 12, b: 38, l: 42 }, height: 220 }),
        { responsive: true, displayModeBar: false }
      );
    }
  }

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
