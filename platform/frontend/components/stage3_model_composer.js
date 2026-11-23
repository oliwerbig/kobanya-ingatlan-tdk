/* Stage 3: Visual Pill Formula Composer, Cloneable Spec Tabs, M1-M4 Starter,
 * Live Andersson-Day Cutoff Slider + Multi-Target Profile-MLE Curve,
 * Residual/QQ Diagnostics, Per-Variable VIF & Heckman 2-Step
 */

import { escapeHtml, getPlotlyThemeLayout } from "../utils/sanitize.js";

function normalQuantile(p) {
  // Peter J. Acklam inverse standard normal CDF approximation
  if (p <= 0) return -6;
  if (p >= 1) return 6;
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const plow = 0.02425;
  const phigh = 1 - plow;
  if (p < plow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p <= phigh) {
    const q = p - 0.5;
    const r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  const q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

export function renderStage3(container, ctx) {
  const { state, actions } = ctx;
  if (!state.datasetInfo) {
    container.innerHTML = `<div class="card"><div class="card-subtitle">Előbb tölts fel és rögzíts egy adatbázist az 1. fázisban.</div></div>`;
    return;
  }

  const models = state.recipe.models;
  if (!state.activeSpecId || !models.find((m) => m.id === state.activeSpecId)) {
    state.activeSpecId = models[0]?.id;
  }
  const activeSpec = models.find((m) => m.id === state.activeSpecId);
  const headers = state.headers || [];
  const ds = state.recipe.dataset_schema;
  const colProfile = state.colProfile || {};
  const excludedCols = new Set([ds.lat_col, ds.lon_col, ds.row_id_col, ds.y_spec.num_col, ds.y_spec.den_col].filter(Boolean));
  const availCovCols = headers.filter((h) => !excludedCols.has(h));
  const targets = state.recipe.spatial_targets || [];
  const s3 = state.stage3Res;

  container.innerHTML = `
    <div id="stage3ComposerCard" class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">3A. Vizuális Modell-Összeállító (Formula Composer)</h3>
          <div class="card-subtitle">Szerkeszthető kovariánsok, térbeli transzformációk és klónozható specifikációk közös N<sub>common</sub> mintán</div>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <button class="btn btn-sm" id="btnGenM1M4" title="M1 Fizikai, M2 Sávos, M3 Log, M4 Andersson–Day generálása a jelenlegi kovariánsokból">⚡ M1–M4 Alapsorozat</button>
          <button class="btn btn-primary btn-sm" id="btnRunStage3">▶ Becslés Futtatása</button>
        </div>
      </div>

      <div class="spec-tabs">
        ${models
          .map(
            (m) => `
          <button class="spec-tab ${m.id === state.activeSpecId ? "active" : ""}" data-spec-id="${escapeHtml(m.id)}">
            ${escapeHtml(m.name)}
          </button>`
          )
          .join("")}
        <button class="btn btn-sm" id="btnCloneSpec">+ Klónozás</button>
        <button class="btn btn-sm" id="btnNewSpec">+ Új Spec</button>
      </div>

      ${
        activeSpec
          ? `
        <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap;">
          <input type="text" id="txtSpecName" value="${escapeHtml(activeSpec.name)}" style="width:220px;font-weight:600;" />
          <label style="display:inline-flex;align-items:center;gap:4px;font-size:11px;">
            <input type="checkbox" id="chkSpecActive" ${activeSpec.active ? "checked" : ""} /> Aktív az összehasonlításban
          </label>
          ${models.length > 1 ? `<button class="btn btn-sm" id="btnDelSpec" style="color:var(--fail);">✕ Spec törlése</button>` : ""}
        </div>

        <div class="formula-box" style="margin-bottom:10px;">
          <span class="pill y-pill">${ds.y_spec.transform === "log" ? "ln(y)" : ds.y_spec.transform === "boxcox" ? "BoxCox(y)" : "y"}</span>
          <span class="mono" style="font-weight:700;">~</span>
          ${(activeSpec.covariates || [])
            .map(
              (c, cidx) => `
            <span class="pill">
              ${
                c.type === "categorical"
                  ? `${escapeHtml(c.col)} [ref:${escapeHtml(c.reference_level || "auto")}]`
                  : `${escapeHtml(c.transform && c.transform !== "raw" ? `${c.transform}(${c.col})` : c.col)}`
              }
              <button class="remove-x" data-rm-cov="${cidx}">×</button>
            </span>`
            )
            .join("")}
          ${(activeSpec.spatial_terms || [])
            .map(
              (st, sidx) => `
            <span class="pill spatial-pill">
              <span class="sp-pill-text" data-sp-pill-idx="${sidx}">${escapeHtml(formatSpatialPill(st))}</span>
              <button class="remove-x" data-rm-sp="${sidx}">×</button>
            </span>`
            )
            .join("")}
          ${(activeSpec.interactions || [])
            .map(
              (it, iidx) => `
            <span class="pill inter-pill">
              ${escapeHtml(it.left)} × ${escapeHtml(it.right)}
              <button class="remove-x" data-rm-int="${iidx}">×</button>
            </span>`
            )
            .join("")}
        </div>

        <div class="grid-3" style="margin-bottom:8px;">
          <div class="field-row">
            <label>Kovariáns oszlop hozzáadása</label>
            <div style="display:flex;gap:4px;">
              <select id="selAddCov" style="flex:1;">
                ${availCovCols.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)} (${colProfile[c]?.isNumeric ? "num" : "kat"})</option>`).join("")}
              </select>
              <select id="selCovTrans" style="width:92px;">
                <option value="raw">nyers</option>
                <option value="log">ln(x)</option>
                <option value="centered">közép</option>
                <option value="zscore">z-score</option>
                <option value="poly">x + x²</option>
                <option value="categorical">kategória</option>
              </select>
              <button class="btn btn-sm" id="btnAddCov">+</button>
            </div>
          </div>

          <div class="field-row">
            <label>Térbeli transzformáció hozzáadása</label>
            <div style="display:flex;gap:4px;">
              <select id="selAddSpTgt" style="flex:1;">
                ${
                  targets.length
                    ? targets.map((t) => `<option value="${escapeHtml(t.id)}">${escapeHtml(t.id)}</option>`).join("")
                    : `<option value="">— nincs célpont —</option>`
                }
              </select>
              <select id="selAddSpTrans" style="width:125px;">
                <option value="andersson_day">Andersson–Day</option>
                <option value="rings">Távolsági sávok</option>
                <option value="ln_d">ln(d) folytonos</option>
                <option value="linear">lineáris d/100m</option>
                <option value="exp_decay">Exp-decay (h½)</option>
                <option value="circuity_delta">Δd Kanyargósság</option>
                <option value="circuity_ratio">τ Kanyargósság</option>
              </select>
              <button class="btn btn-sm" id="btnAddSp" ${targets.length ? "" : "disabled"}>+</button>
            </div>
          </div>

          <div class="field-row">
            <label>Interakció (x₁ × x₂)</label>
            <div style="display:flex;gap:4px;">
              <input type="text" id="txtIntLeft" placeholder="x₁ (pl. area)" style="width:45%;" />
              <input type="text" id="txtIntRight" placeholder="x₂ (pl. rail::andersson_day)" style="width:45%;" />
              <button class="btn btn-sm" id="btnAddInt">+</button>
            </div>
          </div>
        </div>

        ${renderSpatialTermInspectors(activeSpec)}
      `
          : ""
      }
    </div>

    <div id="stage3ResultsArea">
      ${renderStage3ResultsHTML(state, s3)}
    </div>
  `;

  container.querySelectorAll(".spec-tab").forEach((b) => {
    b.addEventListener("click", () => {
      state.activeSpecId = b.getAttribute("data-spec-id");
      actions.render();
    });
  });

  container.querySelector("#btnGenM1M4").addEventListener("click", () => {
    const hasCustom =
      models.length > 4 ||
      models.some((m) => !["m1_base", "m2_rings", "m3_log", "m4_ad"].includes(m.id)) ||
      models.some((m) => (m.covariates?.length || 0) > 0 || (m.spatial_terms?.length || 0) > 0);
    if (hasCustom && !window.confirm("Ez felülírja a jelenlegi modell-specifikációkat az M1–M4 alapsorozattal. Folytatod?")) {
      return;
    }
    actions.generateM1M4Specs();
  });
  container.querySelector("#btnRunStage3").addEventListener("click", () => actions.runStage3());

  container.querySelector("#btnCloneSpec").addEventListener("click", () => {
    if (!activeSpec) return;
    const clone = JSON.parse(JSON.stringify(activeSpec));
    clone.id = `spec_${Date.now().toString(36).slice(-4)}`;
    clone.name = `${activeSpec.name} (Klón)`;
    models.push(clone);
    state.activeSpecId = clone.id;
    actions.render();
  });
  container.querySelector("#btnNewSpec").addEventListener("click", () => {
    const id = `spec_${Date.now().toString(36).slice(-4)}`;
    models.push({ id, name: `Spec ${models.length + 1}`, active: true, covariates: [], spatial_terms: [], interactions: [] });
    state.activeSpecId = id;
    actions.render();
  });

  if (activeSpec) {
    container.querySelector("#txtSpecName")?.addEventListener("change", (e) => {
      activeSpec.name = e.target.value;
      const tabBtn = container.querySelector(`.spec-tab[data-spec-id="${CSS.escape(activeSpec.id)}"]`);
      if (tabBtn) tabBtn.textContent = e.target.value;
    });
    container.querySelector("#chkSpecActive")?.addEventListener("change", (e) => {
      activeSpec.active = e.target.checked;
    });
    container.querySelector("#btnDelSpec")?.addEventListener("click", () => {
      const idx = models.findIndex((m) => m.id === activeSpec.id);
      if (idx >= 0) models.splice(idx, 1);
      state.activeSpecId = models[0]?.id;
      actions.render();
    });

    container.querySelectorAll("[data-rm-cov]").forEach((btn) =>
      btn.addEventListener("click", () => {
        activeSpec.covariates.splice(Number(btn.getAttribute("data-rm-cov")), 1);
        actions.render();
      })
    );
    container.querySelectorAll("[data-rm-sp]").forEach((btn) =>
      btn.addEventListener("click", () => {
        activeSpec.spatial_terms.splice(Number(btn.getAttribute("data-rm-sp")), 1);
        actions.render();
      })
    );
    container.querySelectorAll("[data-rm-int]").forEach((btn) =>
      btn.addEventListener("click", () => {
        activeSpec.interactions.splice(Number(btn.getAttribute("data-rm-int")), 1);
        actions.render();
      })
    );

    container.querySelector("#btnAddCov")?.addEventListener("click", () => {
      const col = container.querySelector("#selAddCov").value;
      const tr = container.querySelector("#selCovTrans").value;
      if (!col) return;
      if (tr === "categorical") {
        activeSpec.covariates.push({ col, type: "categorical", reference_level: colProfile[col]?.sampleLevels?.[0] || "" });
      } else {
        activeSpec.covariates.push({ col, type: "numeric", transform: tr });
      }
      actions.render();
    });

    container.querySelector("#btnAddSp")?.addEventListener("click", () => {
      const tid = container.querySelector("#selAddSpTgt").value;
      const tr = container.querySelector("#selAddSpTrans").value;
      const tgtObj = targets.find((t) => t.id === tid);
      const metric =
        tr === "circuity_delta" || tr === "circuity_ratio"
          ? tr
          : tgtObj?.distance_modes === "network_dijkstra"
          ? "net"
          : "eucl";
      activeSpec.spatial_terms.push({
        target_id: tid,
        metric,
        transform: tr,
        floor_m: 1.0,
        cutoff_m: 300.0,
        cutoff_range_m: [100.0, 600.0, 25.0],
        breakpoints_m: [150.0, 300.0, 500.0],
        half_life_m: 250.0,
        scale: 100.0,
      });
      actions.render();
    });

    container.querySelector("#btnAddInt")?.addEventListener("click", () => {
      const l = container.querySelector("#txtIntLeft").value.trim();
      const r = container.querySelector("#txtIntRight").value.trim();
      if (!l || !r) return;
      activeSpec.interactions.push({ left: l, right: r, center_continuous: true });
      actions.render();
    });

    // Live AD Cutoff slider (<15 ms DAG incremental OLS update without breaking slider drag)
    container.querySelectorAll("[data-sp-edit-idx]").forEach((box) => {
      const sidx = Number(box.getAttribute("data-sp-edit-idx"));
      const st = activeSpec.spatial_terms[sidx];
      box.querySelector(".sp-metric-sel")?.addEventListener("change", (e) => {
        st.metric = e.target.value;
        const pillTxt = container.querySelector(`.sp-pill-text[data-sp-pill-idx="${sidx}"]`);
        if (pillTxt) pillTxt.textContent = formatSpatialPill(st);
      });
      const slider = box.querySelector(".sp-ad-slider");
      const valLbl = box.querySelector(".sp-ad-val");
      if (slider && valLbl) {
        slider.addEventListener("input", (e) => {
          const cv = Number(e.target.value);
          st.cutoff_m = cv;
          valLbl.textContent = `${cv} m`;
          const pillTxt = container.querySelector(`.sp-pill-text[data-sp-pill-idx="${sidx}"]`);
          if (pillTxt) pillTxt.textContent = formatSpatialPill(st);
          actions.runStage3Debounced();
        });
      }
      box.querySelector(".sp-rings-inp")?.addEventListener("change", (e) => {
        const arr = e.target.value
          .split(",")
          .map((v) => Number(v.trim()))
          .filter((v) => Number.isFinite(v) && v > 0);
        if (arr.length) {
          st.breakpoints_m = arr;
          const pillTxt = container.querySelector(`.sp-pill-text[data-sp-pill-idx="${sidx}"]`);
          if (pillTxt) pillTxt.textContent = formatSpatialPill(st);
        }
      });
      box.querySelector(".sp-hl-inp")?.addEventListener("change", (e) => {
        st.half_life_m = Math.max(1, Number(e.target.value) || 250);
        const pillTxt = container.querySelector(`.sp-pill-text[data-sp-pill-idx="${sidx}"]`);
        if (pillTxt) pillTxt.textContent = formatSpatialPill(st);
      });
    });
  }

  bindStage3ResultsEventsAndPlots(container, state, actions);
}

export function patchStage3ResultsOnly(container, ctx) {
  const { state, actions } = ctx;
  const resultsArea = container.querySelector("#stage3ResultsArea");
  if (!resultsArea) {
    renderStage3(container, ctx);
    return;
  }
  resultsArea.innerHTML = renderStage3ResultsHTML(state, state.stage3Res);
  bindStage3ResultsEventsAndPlots(container, state, actions);
}

function renderStage3ResultsHTML(state, s3) {
  if (!s3) return "";
  if (!s3.ok) {
    return `<div class="alert-box">⚠️ ${escapeHtml(s3.reason || "3. Fázis nem futtatható.")}</div>`;
  }
  const valid = (s3.models || []).filter((m) => m.params);
  const failedModels = (s3.models || []).filter((m) => m.error);
  if (!state.stage3DiagModelId || !valid.find((m) => m.model_id === state.stage3DiagModelId)) {
    state.stage3DiagModelId =
      valid.find((m) => m.model_id === state.activeSpecId)?.model_id || valid[0]?.model_id || "";
  }
  const diagModel = valid.find((m) => m.model_id === state.stage3DiagModelId) || valid[0];

  return `
    ${
      s3.conditional_boxcox
        ? `<div class="alert-box info">
             <strong>Feltételes Box–Cox (${escapeHtml(s3.conditional_boxcox.model_id)} | X adott):</strong>
             λ̂ = <span class="mono">${escapeHtml(s3.conditional_boxcox.lambda_mle)}</span> |
             LR teszt ln(y) ellen: p = <span class="mono">${escapeHtml(s3.conditional_boxcox.p_vs_log)}</span> |
             lineáris y ellen: p = <span class="mono">${escapeHtml(s3.conditional_boxcox.p_vs_linear)}</span>
           </div>`
        : ""
    }

    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">3B. Összehasonlító Regressziós Mátrix &amp; Diagnosztika</h3>
          <div class="card-subtitle">SE típus, Halvorsen–Palmquist / Kennedy %Δ kapcsoló és Overleaf booktabs LaTeX export</div>
        </div>
        <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
          <select id="selCovType">
            <option value="HC3" ${state.covType === "HC3" ? "selected" : ""}>HC3 Robusztus SE</option>
            <option value="cluster" ${state.covType === "cluster" ? "selected" : ""}>CR1 Klaszterezett SE</option>
            <option value="OLS" ${state.covType === "OLS" ? "selected" : ""}>Klasszikus OLS SE</option>
          </select>
          <label style="font-size:11px;display:inline-flex;align-items:center;gap:4px;">
            <input type="checkbox" id="chkShowPct" ${state.showPct ? "checked" : ""} /> Kennedy/HP %Δ
          </label>
          <button class="btn btn-sm" id="btnCopyLatex">📋 LaTeX Másolása</button>
        </div>
      </div>

      ${failedModels.map((fm) => `<div class="alert-box">⚠️ <strong>${escapeHtml(fm.model_name || fm.model_id)}:</strong> ${escapeHtml(fm.error)}</div>`).join("")}
      ${renderComparisonTable(s3.models || [], state.showPct)}
    </div>

    ${renderDiagnosticsDeepCard(valid, diagModel)}
    ${renderADProfileCard(s3.ad_profiles, state)}
    ${renderHeckmanCard(s3.heckman)}
  `;
}

function bindStage3ResultsEventsAndPlots(container, state, actions) {
  const s3 = state.stage3Res;
  const resultsArea = container.querySelector("#stage3ResultsArea") || container;

  resultsArea.querySelector("#selCovType")?.addEventListener("change", (e) => {
    state.covType = e.target.value;
    actions.runStage3();
  });
  resultsArea.querySelector("#chkShowPct")?.addEventListener("change", (e) => {
    state.showPct = e.target.checked;
    patchStage3ResultsOnly(container, { state, actions });
  });
  resultsArea.querySelector("#btnCopyLatex")?.addEventListener("click", () => actions.copyLatexTable());

  resultsArea.querySelector("#selDiagModel")?.addEventListener("change", (e) => {
    state.stage3DiagModelId = e.target.value;
    patchStage3ResultsOnly(container, { state, actions });
  });

  resultsArea.querySelectorAll(".ad-prof-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.stage3AdKey = btn.getAttribute("data-ad-key");
      patchStage3ResultsOnly(container, { state, actions });
    });
  });

  if (!s3?.ok || !window.Plotly) return;

  // Render AD Profile Curve Plotly chart for active tab
  if (s3.ad_profiles) {
    const okKeys = Object.keys(s3.ad_profiles).filter((k) => s3.ad_profiles[k]?.ok);
    const activeKey = okKeys.includes(state.stage3AdKey) ? state.stage3AdKey : okKeys[0];
    if (activeKey && resultsArea.querySelector("#plotADProfile")) {
      state.stage3AdKey = activeKey;
      const prof = s3.ad_profiles[activeKey];
      const pts = (prof.curve || []).filter((p) => p.aicc !== null);
      window.Plotly.newPlot(
        "plotADProfile",
        [
          {
            x: pts.map((p) => p.c_m),
            y: pts.map((p) => p.aicc),
            type: "scatter",
            mode: "lines+markers",
            name: "AICc(c)",
            line: { color: "#1d4ed8", width: 2 },
          },
          {
            x: pts.map((p) => p.c_m),
            y: pts.map((p) => p.beta),
            yaxis: "y2",
            type: "scatter",
            mode: "lines+markers",
            name: "β̂(c)",
            line: { color: "#0f766e", width: 2, dash: "dot" },
          },
        ],
        getPlotlyThemeLayout({
          title: { text: `Andersson–Day Profil-MLE (${escapeHtml(activeKey)}) | c* = ${prof.c_star_m} m [95% CI: ${prof.c_ci_95?.[0]}–${prof.c_ci_95?.[1]} m]`, font: { size: 12 } },
          xaxis: { title: "Cenzorálási küszöb c (m)" },
          yaxis: { title: "AICc (kisebb jobb)" },
          yaxis2: { title: "β̂(c)", overlaying: "y", side: "right" },
          margin: { t: 36, r: 52, b: 42, l: 52 },
          height: 250,
          legend: { orientation: "h", y: -0.22 },
        }),
        { responsive: true, displayModeBar: false }
      );
    }
  }

  // Render Residual vs Fitted & Normal Q-Q plots for selected model
  const valid = (s3.models || []).filter((m) => m.params && Array.isArray(m.resid) && Array.isArray(m.fitted));
  const diagModel = valid.find((m) => m.model_id === state.stage3DiagModelId) || valid[0];
  if (diagModel && diagModel.resid.length > 2) {
    const fitted = diagModel.fitted;
    const resid = diagModel.resid;
    const cooks = diagModel.cooks_d || [];
    const nObs = resid.length;
    const cookThresh = 4 / Math.max(1, nObs);

    if (resultsArea.querySelector("#plotResidFitted")) {
      window.Plotly.newPlot(
        "plotResidFitted",
        [
          {
            x: fitted,
            y: resid,
            mode: "markers",
            type: "scatter",
            marker: {
              size: cooks.map((c) => (c > cookThresh ? 8 : 5)),
              color: cooks.map((c) => (c > cookThresh ? "#dc2626" : "#0f766e")),
              opacity: 0.72,
            },
            text: cooks.map((c, idx) => `Obs #${idx} | Cook D: ${Number(c).toFixed(4)}`),
            name: "Reziduális",
          },
        ],
        getPlotlyThemeLayout({
          title: { text: `Reziduális vs. Becsült érték (${escapeHtml(diagModel.model_id)})`, font: { size: 12 } },
          xaxis: { title: "ŷ (illesztett)" },
          yaxis: { title: "ε̂ (reziduális)" },
          margin: { t: 34, r: 16, b: 44, l: 50 },
          height: 240,
          shapes: [{ type: "line", x0: Math.min(...fitted), x1: Math.max(...fitted), y0: 0, y1: 0, line: { dash: "dot", color: "#94a3b8" } }],
        }),
        { responsive: true, displayModeBar: false }
      );
    }

    if (resultsArea.querySelector("#plotResidQQ")) {
      const meanR = resid.reduce((a, b) => a + b, 0) / nObs;
      const sdR = Math.sqrt(resid.reduce((a, b) => a + (b - meanR) ** 2, 0) / Math.max(1, nObs - 1)) || 1;
      const stdSorted = resid.map((r) => (r - meanR) / sdR).sort((a, b) => a - b);
      const theoQ = stdSorted.map((_, i) => normalQuantile((i + 0.5) / nObs));
      const minQ = Math.min(theoQ[0], stdSorted[0]);
      const maxQ = Math.max(theoQ[nObs - 1], stdSorted[nObs - 1]);

      window.Plotly.newPlot(
        "plotResidQQ",
        [
          {
            x: theoQ,
            y: stdSorted,
            mode: "markers",
            type: "scatter",
            marker: { size: 5, color: "#1d4ed8", opacity: 0.72 },
            name: "Std. reziduális",
          },
          {
            x: [minQ, maxQ],
            y: [minQ, maxQ],
            mode: "lines",
            type: "scatter",
            line: { color: "#dc2626", dash: "dot", width: 1.5 },
            name: "N(0,1)",
          },
        ],
        getPlotlyThemeLayout({
          title: { text: `Normál Q–Q Ábra (${escapeHtml(diagModel.model_id)})`, font: { size: 12 } },
          xaxis: { title: "Elméleti N(0,1) kvantilis" },
          yaxis: { title: "Standardizált reziduális" },
          margin: { t: 34, r: 16, b: 44, l: 50 },
          height: 240,
          showlegend: false,
        }),
        { responsive: true, displayModeBar: false }
      );
    }
  }
}

function formatSpatialPill(st) {
  if (st.transform === "andersson_day") return `AD(${st.target_id}, ${st.metric}, c=${st.cutoff_m}m)`;
  if (st.transform === "rings") return `Sávok(${st.target_id}, [${(st.breakpoints_m || []).join(",")}])`;
  if (st.transform === "exp_decay") return `Exp(${st.target_id}, h½=${st.half_life_m}m)`;
  return `${st.transform}(${st.target_id}, ${st.metric})`;
}

function renderSpatialTermInspectors(spec) {
  const terms = spec.spatial_terms || [];
  if (!terms.length) return "";
  return terms
    .map((st, idx) => {
      const isCircuity = st.transform === "circuity_delta" || st.transform === "circuity_ratio";
      return `
    <div style="display:flex;gap:10px;align-items:center;padding:6px 8px;border:1px solid var(--border);border-radius:6px;background:var(--surface-alt);margin-bottom:6px;flex-wrap:wrap;" data-sp-edit-idx="${idx}">
      <strong style="font-size:11px;">${escapeHtml(st.target_id)} (${escapeHtml(st.transform)})</strong>
      <select class="sp-metric-sel" style="font-size:11px;" ${isCircuity ? 'disabled title="Kanyargósság transzformációnál a metrika rögzített (eucl és net szükséges)"' : ""}>
        <option value="eucl" ${!isCircuity && st.metric === "eucl" ? "selected" : ""}>Euklideszi</option>
        <option value="net" ${!isCircuity && st.metric === "net" ? "selected" : ""}>Hálózati Dijkstra</option>
        <option value="circuity_delta" ${st.transform === "circuity_delta" || st.metric === "circuity_delta" ? "selected" : ""}>Δd (m)</option>
        <option value="circuity_ratio" ${st.transform === "circuity_ratio" || st.metric === "circuity_ratio" ? "selected" : ""}>τ arány</option>
      </select>
      ${
        st.transform === "andersson_day"
          ? (() => {
              const curC = Number(st.cutoff_m) || 300;
              const rMin = Math.min(50, Number(st.cutoff_range_m?.[0]) || 50, curC);
              const rMax = Math.max(1000, Number(st.cutoff_range_m?.[1]) || 1000, curC);
              let rStep = Math.max(1, Number(st.cutoff_range_m?.[2]) || 25);
              if (Math.abs(((curC - rMin) % rStep)) > 1e-6) {
                rStep = Math.abs((curC - rMin) % 25) < 1e-6 ? 25 : Math.abs((curC - rMin) % 5) < 1e-6 ? 5 : 1;
              }
              return `<div style="display:flex;align-items:center;gap:8px;flex:1;min-width:220px;">
               <span style="font-size:11px;">Küszöb c:</span>
               <input type="range" class="sp-ad-slider" min="${rMin}" max="${rMax}" step="${rStep}" value="${curC}" style="flex:1;" />
               <span class="mono sp-ad-val" style="font-weight:600;min-width:48px;">${curC} m</span>
             </div>`;
            })()
          : st.transform === "rings"
          ? `<div style="display:flex;align-items:center;gap:6px;">
               <span style="font-size:11px;">Sávhatárok (m):</span>
               <input type="text" class="sp-rings-inp mono" value="${escapeHtml((st.breakpoints_m || [150, 300, 500]).join(", "))}" style="width:160px;" />
             </div>`
          : st.transform === "exp_decay"
          ? `<div style="display:flex;align-items:center;gap:6px;">
               <span style="font-size:11px;">Felezési távolság h½ (m):</span>
               <input type="number" class="sp-hl-inp mono" value="${Number(st.half_life_m) || 250}" style="width:80px;" />
             </div>`
          : ""
      }
    </div>`;
    })
    .join("");
}

function renderComparisonTable(models, showPct) {
  const valid = models.filter((m) => m.params);
  if (!valid.length) return `<div class="alert-box">Nincs sikeresen becsült modell.</div>`;

  const varOrder = [];
  const seen = new Set();
  for (const m of valid) {
    for (const p of m.params) {
      if (!seen.has(p.name)) {
        seen.add(p.name);
        varOrder.push(p.name);
      }
    }
  }
  const maps = valid.map((m) => Object.fromEntries(m.params.map((p) => [p.name, p])));

  const guardNotes = valid.flatMap((m) => (m.footnotes || []).map((f) => `[${m.model_id}] ${f}`));
  const proxyAlerts = valid.flatMap((m) => (m.diagnostics?.proxy_alerts || []).map((a) => `[${m.model_id}] ${a.message}`));
  const covWarnings = valid.flatMap((m) => (m.diagnostics?.cov_warnings || []).map((w) => `[${m.model_id}] ${w}`));
  const mixedScale = new Set(valid.map((m) => m.y_transform)).size > 1;

  return `
    ${proxyAlerts.map((a) => `<div class="alert-box">⚠️ ${escapeHtml(a)}</div>`).join("")}
    ${covWarnings.map((w) => `<div class="alert-box">⚠️ ${escapeHtml(w)}</div>`).join("")}
    ${guardNotes.map((f) => `<div class="alert-box info">🛡️ ${escapeHtml(f)}</div>`).join("")}
    ${
      mixedScale
        ? `<div class="alert-box info">ℹ️ Eltérő függőváltozó-transzformációk mellett a nyers AIC nem összehasonlítható; lásd a Jakobi-korrigált Y-skálás AIC sort.</div>`
        : ""
    }
    <div class="table-wrap">
      <table class="academic">
        <thead>
          <tr>
            <th>Változó</th>
            ${valid.map((m) => `<th class="num">${escapeHtml(m.model_name)}<div class="card-subtitle" style="font-weight:400;font-size:10px;">SE: ${escapeHtml(m.cov_type_used || "—")} (df=${m.df_infer ?? "—"})</div></th>`).join("")}
          </tr>
        </thead>
        <tbody>
          ${varOrder
            .map(
              (vn) => `
            <tr>
              <td><strong>${escapeHtml(vn)}</strong></td>
              ${maps
                .map((mp) => {
                  const p = mp[vn];
                  if (!p) return `<td class="num card-subtitle">—</td>`;
                  const pctBadge =
                    showPct && p.kennedy_pct !== null
                      ? `<div class="card-subtitle mono" style="font-size:10px;">Kennedy: ${p.kennedy_pct >= 0 ? "+" : ""}${p.kennedy_pct}% (HP: ${p.hp_pct >= 0 ? "+" : ""}${p.hp_pct}%)</div>`
                      : "";
                  return `<td class="num">
                    <div>${p.beta}<span class="sig-stars">${escapeHtml(p.sig)}</span></div>
                    <div class="card-subtitle" style="font-size:10px;">(${p.se})</div>
                    ${pctBadge}
                  </td>`;
                })
                .join("")}
            </tr>`
            )
            .join("")}
          <tr style="border-top:2px solid var(--border-strong);background:var(--surface-alt);">
            <td><strong>N / Adj. R² / AIC</strong></td>
            ${valid.map((m) => `<td class="num">${m.n_obs} | ${m.adj_r2} | ${m.aic}</td>`).join("")}
          </tr>
          ${
            valid.some((m) => m.aic_y_scale !== null && m.aic_y_scale !== undefined)
              ? `<tr>
                   <td><strong>Jakobi Y-skála AIC</strong></td>
                   ${valid.map((m) => `<td class="num">${m.aic_y_scale ?? "—"}</td>`).join("")}
                 </tr>`
              : ""
          }
          <tr>
            <td><strong>5-szörös Diagnosztika</strong></td>
            ${valid
              .map((m) => {
                const d = m.diagnostics || {};
                const vifDisp = d.collinear_cols?.length || (d.max_vif ?? 0) >= 1e6 ? "∞ (Koll.)" : d.max_vif;
                return `<td class="num">
                  <div class="diag-bar" style="justify-content:flex-end;">
                    <span class="diag-chip ${d.max_vif < 5 ? "pass" : d.max_vif < 10 ? "warn" : "fail"}">VIF:${escapeHtml(vifDisp)}</span>
                    <span class="diag-chip ${d.bp_p >= 0.05 ? "pass" : "warn"}">BP:${d.bp_p ?? "—"}</span>
                    <span class="diag-chip ${d.reset_p === null || d.reset_p >= 0.05 ? "pass" : "fail"}">RESET:${d.reset_p ?? "—"}</span>
                    <span class="diag-chip ${d.jb_p >= 0.05 ? "pass" : "warn"}">JB:${d.jb_p ?? "—"}</span>
                  </div>
                </td>`;
              })
              .join("")}
          </tr>
        </tbody>
      </table>
    </div>
  `;
}

function renderDiagnosticsDeepCard(valid, diagModel) {
  if (!diagModel) return "";
  const d = diagModel.diagnostics || {};
  const vifs = Array.isArray(d.vifs) ? d.vifs : [];
  const kappa = d.condition_number_kappa;
  const kappaCls = kappa < 30 ? "pass" : kappa < 100 ? "warn" : "fail";
  return `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">3B-2. Mélydiagnosztika: Per-Változó VIF, κ &amp; Reziduális Ábrák</h3>
          <div class="card-subtitle">Kondíciószám κ = <span class="diag-chip ${kappaCls}">${kappa ?? "—"}</span> | Max Leverage h<sub>ii</sub>: <span class="mono">${d.max_leverage ?? "—"}</span> | White p = <span class="mono">${d.white_p ?? "—"}</span> (${escapeHtml(d.white_mode || "—")})</div>
        </div>
        <select id="selDiagModel" style="font-size:11px;">
          ${valid.map((m) => `<option value="${escapeHtml(m.model_id)}" ${m.model_id === diagModel.model_id ? "selected" : ""}>${escapeHtml(m.model_name)}</option>`).join("")}
        </select>
      </div>
      <div class="grid-2" style="margin-bottom:10px;">
        <div id="plotResidFitted" class="plot-box"></div>
        <div id="plotResidQQ" class="plot-box"></div>
      </div>
      <div class="table-wrap" style="max-height:200px;">
        <table class="academic">
          <thead><tr><th>Magyarázó változó</th><th class="num">VIF</th><th>Állapot</th></tr></thead>
          <tbody>
            ${
              vifs.length
                ? vifs
                    .map((v) => {
                      const cls = v.collinear || v.vif >= 10 ? "fail" : v.vif >= 5 ? "warn" : "pass";
                      const lbl = v.collinear ? "Multikollineáris (R²≈1)" : v.vif >= 10 ? "Magas (≥10)" : v.vif >= 5 ? "Mérsékelt" : "OK";
                      return `<tr>
                        <td>${escapeHtml(v.name)}</td>
                        <td class="num">${v.collinear || v.vif >= 1e6 ? "∞" : v.vif}</td>
                        <td><span class="diag-chip ${cls}">${lbl}</span></td>
                      </tr>`;
                    })
                    .join("")
                : `<tr><td colspan="3" class="muted">Csak konstans szerepel a modellben.</td></tr>`
            }
          </tbody>
        </table>
      </div>
    </div>
  `;
}

function renderADProfileCard(adProfiles, state) {
  const allKeys = Object.keys(adProfiles || {});
  const okKeys = allKeys.filter((k) => adProfiles[k]?.ok);
  if (!allKeys.length) return "";
  const activeKey = okKeys.includes(state.stage3AdKey) ? state.stage3AdKey : okKeys[0] || allKeys[0];
  state.stage3AdKey = activeKey;
  const prof = adProfiles[activeKey];
  return `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">3C. Andersson–Day Küszöbprofil-Szkennelés (Profile-MLE)</h3>
          <div class="card-subtitle">${escapeHtml(prof?.post_selection_note || prof?.reason || prof?.error || "")}</div>
        </div>
      </div>
      ${
        allKeys.length > 1
          ? `<div class="spec-tabs" style="margin-bottom:8px;">
               ${allKeys
                 .map(
                   (k) => `<button class="spec-tab ad-prof-tab ${k === activeKey ? "active" : ""}" data-ad-key="${escapeHtml(k)}">${escapeHtml(k)}${adProfiles[k]?.ok ? ` (c*=${adProfiles[k].c_star_m}m)` : " (hiba)"}</button>`
                 )
                 .join("")}
             </div>`
          : ""
      }
      ${prof?.ok ? `<div id="plotADProfile" class="plot-box"></div>` : `<div class="alert-box">⚠️ ${escapeHtml(prof?.reason || prof?.error || "Profil-MLE sikertelen.")}</div>`}
    </div>
  `;
}

function renderHeckmanCard(heck) {
  if (!heck) return "";
  if (!heck.activated) {
    return `<div class="card"><div class="card-subtitle"><strong>Guard #4 Heckman 2-Lépcsős Szelekció:</strong> ${escapeHtml(heck.reason)}</div></div>`;
  }
  return `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">3D. Guard #4 — Heckman (1979) 2-Lépcsős Szelekciós Modell</h3>
          <div class="card-subtitle">Teljes pool: ${heck.n_total_pool} | Elemzett: ${heck.n_analytic} | Kizárt: ${heck.n_excluded} | λ<sub>IMR</sub> = ${heck.imr_beta} (SE=${heck.imr_se}, p=${heck.imr_p}, VIF=${heck.imr_vif ?? "—"})</div>
        </div>
      </div>
      ${(heck.warnings || []).map((w) => `<div class="alert-box">⚠️ ${escapeHtml(w)}</div>`).join("")}
      <div class="table-wrap">
        <table class="academic">
          <thead><tr><th>Változó</th><th class="num">Alap HC3 β̂ (SE)</th><th class="num">Heckman β̂ (Korrigált SE)</th><th class="num">Heckman p</th></tr></thead>
          <tbody>
            ${(heck.comparison || [])
              .map(
                (r) => `<tr>
                  <td><strong>${escapeHtml(r.label)}</strong></td>
                  <td class="num">${r.base_beta !== null ? `${r.base_beta} (${r.base_se_hc3})` : "—"}</td>
                  <td class="num">${r.heckman_beta} (${r.heckman_se})</td>
                  <td class="num">${r.heckman_p}</td>
                </tr>`
              )
              .join("")}
          </tbody>
        </table>
      </div>
    </div>
  `;
}
