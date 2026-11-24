/* Stage 3: Visual Pill Formula Composer (Clickable Covariates, Interaction Builder, Causal DiD/RDD),
 * Wild Cluster Bootstrap (G<30), 3x2 Robustness Matrix, Editable AD Scan + c* Apply, & Plot->Map Focus
 */

import { translateVerdict } from "../utils/i18n.js";
import { escapeHtml, getPlotlyThemeLayout } from "../utils/sanitize.js";

function normalQuantile(p) {
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

const NUM_TRANS_CYCLE = ["raw", "log", "centered", "zscore", "poly"];

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
  const numCols = availCovCols.filter((h) => colProfile[h]?.isNumeric);
  const targets = state.recipe.spatial_targets || [];
  const s3 = state.stage3Res;

  const yPillLabel =
    ds.y_spec.transform === "log"
      ? "ln(y)"
      : ds.y_spec.transform === "log1p"
        ? "ln(y+shift)"
        : ds.y_spec.transform === "boxcox"
          ? "BoxCox(y)"
          : "y";

  const activeTermChoices = [];
  if (activeSpec) {
    for (const c of activeSpec.covariates || []) {
      activeTermChoices.push(c.col);
    }
    for (const st of activeSpec.spatial_terms || []) {
      activeTermChoices.push(`${st.target_id}::${st.transform}`);
    }
  }
  for (const h of availCovCols) {
    if (!activeTermChoices.includes(h)) activeTermChoices.push(h);
  }

  const causal = activeSpec?.causal_spec || { type: "none", target_id: targets[0]?.id || "", time_col: "", post_value: "", treat_cutoff_m: 300, bandwidth_m: 600, signed_dist_col: "" };

  container.innerHTML = `
    <div id="stage3ComposerCard" class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">3A. Vizuális Modell-Összeállító (Formula Composer)</h3>
          <div class="card-subtitle">Kattints egy kovariáns kapszulára a transzformáció / referencia-szint módosításához (közös N<sub>common</sub> minta)</div>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <button type="button" class="btn btn-sm" id="btnGenM1M4" title="M1 Fizikai, M2 Sávos, M3 Log, M4 Andersson–Day generálása">⚡ M1–M4 Alapsorozat</button>
          <button type="button" class="btn btn-primary btn-sm" id="btnRunStage3">▶ Becslés Futtatása</button>
        </div>
      </div>

      <div class="spec-tabs">
        ${models
          .map(
            (m) => `
          <button type="button" class="spec-tab ${m.id === state.activeSpecId ? "active" : ""}" data-spec-id="${escapeHtml(m.id)}">
            ${escapeHtml(m.name)}
          </button>`
          )
          .join("")}
        <button type="button" class="btn btn-sm" id="btnCloneSpec">+ Klónozás</button>
        <button type="button" class="btn btn-sm" id="btnNewSpec">+ Új Spec</button>
      </div>

      ${
        activeSpec
          ? `
        <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap;">
          <input type="text" id="txtSpecName" aria-label="Specifikáció neve" value="${escapeHtml(activeSpec.name)}" style="width:220px;font-weight:600;" />
          <label style="display:inline-flex;align-items:center;gap:4px;font-size:11px;">
            <input type="checkbox" id="chkSpecActive" ${activeSpec.active ? "checked" : ""} /> Aktív az összehasonlításban
          </label>
          ${models.length > 1 ? `<button type="button" class="btn btn-sm" id="btnDelSpec" style="color:var(--fail);">✕ Spec törlése</button>` : ""}
        </div>

        <div class="formula-box" style="margin-bottom:10px;">
          <span class="pill y-pill">${yPillLabel}</span>
          <span class="mono" style="font-weight:700;">~</span>
          ${(activeSpec.covariates || [])
            .map((c, cidx) => {
              const titleTip = c.type === "categorical" ? "Kattints a referencia-szint cseréjéhez" : "Kattints a transzformáció léptetéséhez (nyers→ln→közép→zscore→poly)";
              return `<span class="pill clickable-pill cov-pill" data-cov-idx="${cidx}" title="${titleTip}" role="button" tabindex="0">
                ${
                  c.type === "categorical"
                    ? `${escapeHtml(c.col)} [ref:${escapeHtml(c.reference_level || "auto")}]`
                    : `${escapeHtml(c.transform && c.transform !== "raw" ? `${c.transform}(${c.col})` : c.col)}`
                }
                <button type="button" class="remove-x" data-rm-cov="${cidx}" aria-label="${escapeHtml(c.col)} eltávolítása">×</button>
              </span>`;
            })
            .join("")}
          ${(activeSpec.spatial_terms || [])
            .map(
              (st, sidx) => `
            <span class="pill spatial-pill">
              <span class="sp-pill-text" data-sp-pill-idx="${sidx}">${escapeHtml(formatSpatialPill(st))}</span>
              <button type="button" class="remove-x" data-rm-sp="${sidx}" aria-label="${escapeHtml(st.target_id)} eltávolítása">×</button>
            </span>`
            )
            .join("")}
          ${(activeSpec.interactions || [])
            .map(
              (it, iidx) => `
            <span class="pill inter-pill">
              ${escapeHtml(it.left)} × ${escapeHtml(it.right)}
              <button type="button" class="remove-x" data-rm-int="${iidx}" aria-label="Interakció eltávolítása">×</button>
            </span>`
            )
            .join("")}
        </div>

        <div class="grid-3" style="margin-bottom:8px;">
          <div class="field-row">
            <label for="selAddCov">Kovariáns hozzáadása</label>
            <div style="display:flex;gap:4px;">
              <select id="selAddCov" aria-label="Kovariáns oszlop kiválasztása" style="flex:1;">
                ${availCovCols.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)} (${colProfile[c]?.isNumeric ? "num" : "kat"})</option>`).join("")}
              </select>
              <select id="selCovTrans" aria-label="Kovariáns transzformáció" style="width:92px;">
                <option value="raw">nyers</option>
                <option value="log">ln(x)</option>
                <option value="centered">közép</option>
                <option value="zscore">z-score</option>
                <option value="poly">x + x²</option>
                <option value="categorical">kategória</option>
              </select>
              <button type="button" class="btn btn-sm" id="btnAddCov" aria-label="Kovariáns hozzáadása">+</button>
            </div>
          </div>

          <div class="field-row">
            <label for="selAddSpTgt">Térbeli transzformáció hozzáadása</label>
            <div style="display:flex;gap:4px;">
              <select id="selAddSpTgt" aria-label="Térbeli célpont" style="flex:1;">
                ${
                  targets.length
                    ? targets.map((t) => `<option value="${escapeHtml(t.id)}">${escapeHtml(t.id)}</option>`).join("")
                    : `<option value="">— nincs célpont —</option>`
                }
              </select>
              <select id="selAddSpTrans" aria-label="Térbeli transzformáció módja" style="width:125px;">
                <option value="andersson_day">Andersson–Day</option>
                <option value="rings">Távolsági sávok</option>
                <option value="ln_d">ln(d) folytonos</option>
                <option value="linear">lineáris d/100m</option>
                <option value="exp_decay">Exp-decay (h½)</option>
                <option value="circuity_delta">Δd Kanyargósság</option>
                <option value="circuity_ratio">τ Kanyargósság</option>
              </select>
              <button type="button" class="btn btn-sm" id="btnAddSp" ${targets.length ? "" : "disabled"} aria-label="Térbeli tag hozzáadása">+</button>
            </div>
          </div>

          <div class="field-row">
            <label for="selIntLeft">Interakció (x₁ × x₂)</label>
            <div style="display:flex;gap:4px;">
              <select id="selIntLeft" aria-label="Interakció bal tagja" style="flex:1;">
                <option value="">— x₁ —</option>
                ${activeTermChoices.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("")}
              </select>
              <select id="selIntRight" aria-label="Interakció jobb tagja" style="flex:1;">
                <option value="">— x₂ —</option>
                ${activeTermChoices.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("")}
              </select>
              <button type="button" class="btn btn-sm" id="btnAddInt" aria-label="Interakció hozzáadása">+</button>
            </div>
          </div>
        </div>

        ${renderSpatialTermInspectors(activeSpec)}

        <details style="margin-top:8px;padding:6px 8px;border:1px dashed var(--border);border-radius:6px;background:var(--surface-alt);">
          <summary style="cursor:pointer;font-size:11.5px;font-weight:600;">🧭 Opcionális Kauzális Térbeli Specifikáció (Spatial DiD / Boundary RDD) — aktuális: ${escapeHtml(causal.type)}</summary>
          <div class="grid-3" style="margin-top:8px;">
            <div class="field-row">
              <label for="selCausalType">Kauzális design</label>
              <select id="selCausalType">
                <option value="none" ${causal.type === "none" ? "selected" : ""}>Nincs (Hagyományos keresztmetszet)</option>
                <option value="spatial_did" ${causal.type === "spatial_did" ? "selected" : ""}>Spatial DiD (Treated × Post)</option>
                <option value="boundary_rdd" ${causal.type === "boundary_rdd" ? "selected" : ""}>Boundary RDD (Határvonal folytonosság)</option>
              </select>
            </div>
            <div class="field-row">
              <label for="selCausalTarget">Célpont (Treat távolság)</label>
              <select id="selCausalTarget">
                ${targets.map((t) => `<option value="${escapeHtml(t.id)}" ${t.id === causal.target_id ? "selected" : ""}>${escapeHtml(t.id)}</option>`).join("")}
              </select>
            </div>
            <div class="field-row">
              <label for="inpCausalCutoff">Kezelési küszöb / sávszél (m)</label>
              <input type="number" id="inpCausalCutoff" value="${Number(causal.type === "boundary_rdd" ? (causal.bandwidth_m ?? 600) : (causal.treat_cutoff_m ?? 300))}" min="10" max="50000" step="25" />
            </div>
          </div>
          ${
            causal.type === "spatial_did"
              ? `<div class="grid-2" style="margin-top:6px;">
                   <div class="field-row">
                     <label for="selCausalTimeCol">Idő / Periódus oszlop</label>
                     <select id="selCausalTimeCol">
                       <option value="">— válassz —</option>
                       ${availCovCols.map((c) => `<option value="${escapeHtml(c)}" ${c === causal.time_col ? "selected" : ""}>${escapeHtml(c)}</option>`).join("")}
                     </select>
                   </div>
                   <div class="field-row">
                     <label for="txtCausalPostVal">Post periódus értéke (pl. 2024)</label>
                     <input type="text" id="txtCausalPostVal" value="${escapeHtml(causal.post_value || "")}" placeholder="pl. 1 vagy 2024" />
                   </div>
                 </div>`
              : causal.type === "boundary_rdd"
                ? `<div class="grid-2" style="margin-top:6px;">
                     <div class="field-row">
                       <label for="selCausalSignedCol">Előjeles határtávolság oszlop (negatív=kontroll, pozitív=kezelt)</label>
                       <select id="selCausalSignedCol">
                         <option value="">— vagy használd a célpont távolságát —</option>
                         ${numCols.map((c) => `<option value="${escapeHtml(c)}" ${c === causal.signed_dist_col ? "selected" : ""}>${escapeHtml(c)}</option>`).join("")}
                       </select>
                     </div>
                   </div>`
                : ""
          }
        </details>
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
      actions.restyleMap?.();
      actions.render();
    });
  });

  container.querySelector("#btnGenM1M4").addEventListener("click", () => {
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
    actions.invalidateFromStage?.(3);
    actions.render();
  });
  container.querySelector("#btnNewSpec").addEventListener("click", () => {
    const id = `spec_${Date.now().toString(36).slice(-4)}`;
    models.push({ id, name: `Spec ${models.length + 1}`, active: true, covariates: [], spatial_terms: [], interactions: [] });
    state.activeSpecId = id;
    actions.invalidateFromStage?.(3);
    actions.render();
  });

  if (activeSpec) {
    container.querySelector("#txtSpecName")?.addEventListener("change", (e) => {
      activeSpec.name = e.target.value;
      const tabBtn = container.querySelector(`.spec-tab[data-spec-id="${CSS.escape(activeSpec.id)}"]`);
      if (tabBtn) tabBtn.textContent = e.target.value;
      actions.invalidateFromStage?.(3);
    });
    container.querySelector("#chkSpecActive")?.addEventListener("change", (e) => {
      activeSpec.active = e.target.checked;
      actions.invalidateFromStage?.(3);
    });
    container.querySelector("#btnDelSpec")?.addEventListener("click", () => {
      const idx = models.findIndex((m) => m.id === activeSpec.id);
      if (idx >= 0) models.splice(idx, 1);
      state.activeSpecId = models[0]?.id;
      const activeFallback = models.find((m) => m.active)?.id || models[0]?.id;
      if (state.recipe.spatial_engine.active_model_id === activeSpec.id) {
        state.recipe.spatial_engine.active_model_id = activeFallback;
      }
      if (state.recipe.spatial_engine.gwr?.focal_model_id === activeSpec.id) {
        state.recipe.spatial_engine.gwr.focal_model_id = activeFallback;
      }
      actions.invalidateFromStage?.(3);
      actions.render();
    });

    container.querySelectorAll(".cov-pill").forEach((pillEl) => {
      const cycleCov = (e) => {
        if (e.target.closest("button.remove-x")) return;
        const cidx = Number(pillEl.getAttribute("data-cov-idx"));
        const cov = activeSpec.covariates[cidx];
        if (!cov) return;
        if (cov.type === "categorical") {
          const levels = colProfile[cov.col]?.sampleLevels || [];
          if (levels.length) {
            const curPos = levels.indexOf(cov.reference_level);
            cov.reference_level = levels[(curPos + 1) % levels.length];
            actions.showToast?.(`${cov.col} referencia-szint: ${cov.reference_level}`, "info");
          }
        } else {
          const curT = cov.transform || "raw";
          const nextIdx = (NUM_TRANS_CYCLE.indexOf(curT) + 1) % NUM_TRANS_CYCLE.length;
          cov.transform = NUM_TRANS_CYCLE[nextIdx];
        }
        actions.invalidateFromStage?.(3);
        actions.render();
      };
      pillEl.addEventListener("click", cycleCov);
      pillEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          cycleCov(e);
        }
      });
    });

    container.querySelectorAll("[data-rm-cov]").forEach((btn) =>
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        activeSpec.covariates.splice(Number(btn.getAttribute("data-rm-cov")), 1);
        actions.invalidateFromStage?.(3);
        actions.render();
      })
    );
    container.querySelectorAll("[data-rm-sp]").forEach((btn) =>
      btn.addEventListener("click", () => {
        activeSpec.spatial_terms.splice(Number(btn.getAttribute("data-rm-sp")), 1);
        actions.invalidateFromStage?.(3);
        actions.render();
      })
    );
    container.querySelectorAll("[data-rm-int]").forEach((btn) =>
      btn.addEventListener("click", () => {
        activeSpec.interactions.splice(Number(btn.getAttribute("data-rm-int")), 1);
        actions.invalidateFromStage?.(3);
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
      actions.invalidateFromStage?.(3);
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
      actions.invalidateFromStage?.(3);
      actions.render();
    });

    container.querySelector("#btnAddInt")?.addEventListener("click", () => {
      const l = container.querySelector("#selIntLeft").value.trim();
      const r = container.querySelector("#selIntRight").value.trim();
      if (!l || !r) return;
      activeSpec.interactions.push({ left: l, right: r, center_continuous: true });
      actions.invalidateFromStage?.(3);
      actions.render();
    });

    const syncCausal = (reRender = false) => {
      const cType = container.querySelector("#selCausalType")?.value || "none";
      if (cType === "none") {
        delete activeSpec.causal_spec;
      } else {
        const tgtId = container.querySelector("#selCausalTarget")?.value || targets[0]?.id || "";
        const cutVal = Number(container.querySelector("#inpCausalCutoff")?.value) || (cType === "boundary_rdd" ? 600 : 300);
        activeSpec.causal_spec = {
          ...(activeSpec.causal_spec || {}),
          type: cType,
          target_id: tgtId,
          time_col: container.querySelector("#selCausalTimeCol")?.value || activeSpec.causal_spec?.time_col || "",
          post_value: container.querySelector("#txtCausalPostVal")?.value ?? activeSpec.causal_spec?.post_value ?? "",
          treat_cutoff_m: cType === "spatial_did" ? cutVal : (activeSpec.causal_spec?.treat_cutoff_m || 300),
          bandwidth_m: cType === "boundary_rdd" ? cutVal : (activeSpec.causal_spec?.bandwidth_m || 600),
          signed_dist_col: container.querySelector("#selCausalSignedCol")?.value || activeSpec.causal_spec?.signed_dist_col || "",
        };
      }
      actions.invalidateFromStage?.(3);
      if (reRender) actions.render();
    };
    container.querySelector("#selCausalType")?.addEventListener("change", () => syncCausal(true));
    ["#selCausalTarget", "#inpCausalCutoff", "#selCausalTimeCol", "#txtCausalPostVal", "#selCausalSignedCol"].forEach((s) => {
      container.querySelector(s)?.addEventListener("change", () => syncCausal(false));
    });

    container.querySelectorAll("[data-sp-edit-idx]").forEach((box) => {
      const sidx = Number(box.getAttribute("data-sp-edit-idx"));
      const st = activeSpec.spatial_terms[sidx];
      box.querySelector(".sp-metric-sel")?.addEventListener("change", (e) => {
        st.metric = e.target.value;
        const pillTxt = container.querySelector(`.sp-pill-text[data-sp-pill-idx="${sidx}"]`);
        if (pillTxt) pillTxt.textContent = formatSpatialPill(st);
        actions.invalidateFromStage?.(3);
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
          actions.invalidateFromStage?.(3);
        }
      });
      box.querySelector(".sp-hl-inp")?.addEventListener("change", (e) => {
        st.half_life_m = Math.max(1, Number(e.target.value) || 250);
        const pillTxt = container.querySelector(`.sp-pill-text[data-sp-pill-idx="${sidx}"]`);
        if (pillTxt) pillTxt.textContent = formatSpatialPill(st);
        actions.invalidateFromStage?.(3);
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
    ${(s3.warnings || []).map((w) => `<div class="alert-box">⚠️ ${escapeHtml(w)}</div>`).join("")}
    ${
      s3.conditional_boxcox
        ? `<div class="alert-box ${s3.conditional_boxcox.p_vs_log < 0.05 && valid.some((m) => m.y_transform === "log" || m.y_transform === "log1p") ? "" : "info"}">
             <strong>Feltételes Box–Cox (${escapeHtml(s3.conditional_boxcox.model_id)} | X adott):</strong>
             λ̂ = <span class="mono">${escapeHtml(s3.conditional_boxcox.lambda_mle)}</span> |
             LR teszt ln(y) ellen: p = <span class="mono">${escapeHtml(s3.conditional_boxcox.p_vs_log)}</span> |
             lineáris y ellen: p = <span class="mono">${escapeHtml(s3.conditional_boxcox.p_vs_linear)}</span>
             ${
               s3.conditional_boxcox.p_vs_log < 0.05 && valid.some((m) => m.y_transform === "log" || m.y_transform === "log1p")
                 ? `<div style="margin-top:4px;font-size:11px;">⚠️ A Box–Cox MLE elveti a tiszta ln(y) formát (p &lt; 0.05); futtasd alul a 3×2 robusztussági mátrixot.</div>`
                 : ""
             }
           </div>`
        : ""
    }

    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">3B. Összehasonlító Regressziós Mátrix &amp; Diagnosztika</h3>
          <div class="card-subtitle">SE típus (+ automatikus Wild Cluster Bootstrap G &lt; 30 esetén), %Δ és Overleaf LaTeX</div>
        </div>
        <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
          <select id="selCovType" aria-label="Standard hiba (SE) típus">
            <option value="HC3" ${state.covType === "HC3" ? "selected" : ""}>HC3 Robusztus SE</option>
            <option value="cluster" ${state.covType === "cluster" ? "selected" : ""}>CR1 Klaszterezett SE</option>
            <option value="OLS" ${state.covType === "OLS" ? "selected" : ""}>Klasszikus OLS SE</option>
          </select>
          <label style="font-size:11px;display:inline-flex;align-items:center;gap:4px;">
            <input type="checkbox" id="chkShowPct" ${state.showPct ? "checked" : ""} /> Kennedy/HP %Δ
          </label>
          <button type="button" class="btn btn-sm" id="btnRunRobustnessMatrix">🧪 3×2 Robusztussági Mátrix</button>
          <button type="button" class="btn btn-sm" id="btnCopyLatex">📋 LaTeX Másolása</button>
        </div>
      </div>

      ${failedModels.map((fm) => `<div class="alert-box">⚠️ <strong>${escapeHtml(fm.model_name || fm.model_id)}:</strong> ${escapeHtml(fm.error)}</div>`).join("")}
      ${renderComparisonTable(s3.models || [], state.showPct)}
      ${renderRobustnessMatrixHTML(state.robustnessRes)}
    </div>

    ${renderDiagnosticsDeepCard(valid, diagModel)}
    ${renderADProfileCard(s3.ad_profiles, state)}
    ${renderHeckmanCard(s3.heckman)}
  `;
}

function renderRobustnessMatrixHTML(rob) {
  if (!rob) return "";
  if (!rob.ok) {
    return `<div class="alert-box" style="margin-top:8px;">⚠️ Robusztussági mátrix: ${escapeHtml(rob.reason || "sikertelen")}</div>`;
  }
  return `
    <div style="margin-top:12px;padding-top:10px;border-top:1px dashed var(--border);" id="robustnessMatrixArea">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;flex-wrap:wrap;gap:6px;">
        <strong>🧪 Funkcionális Forma × SE Robusztussági Mátrix (${escapeHtml(rob.focal_term)}, N = ${rob.n_obs})</strong>
        <button type="button" class="btn btn-sm" id="btnCopyRobLatex">📋 Mátrix LaTeX Másolása</button>
      </div>
      <div class="table-wrap">
        <table class="academic">
          <thead><tr><th>y Skála</th><th>SE</th><th class="num">Fókusz β̂</th><th class="num">SE</th><th class="num">p</th><th class="num">R²</th><th class="num">Jakobi AIC</th><th class="num">Koenker p</th></tr></thead>
          <tbody>
            ${(rob.rows || [])
              .map(
                (r) => `<tr>
                  <td>${escapeHtml(r.y_label)}</td>
                  <td>${escapeHtml(r.cov_type_used)}</td>
                  <td class="num">${r.focal_beta}</td>
                  <td class="num">${r.focal_se}</td>
                  <td class="num">${r.focal_p}</td>
                  <td class="num">${r.r2}</td>
                  <td class="num">${r.aic_y_scale}</td>
                  <td class="num">${r.koenker_p ?? "—"}</td>
                </tr>`
              )
              .join("")}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

function bindStage3ResultsEventsAndPlots(container, state, actions) {
  const s3 = state.stage3Res;
  const resultsArea = container.querySelector("#stage3ResultsArea") || container;

  resultsArea.querySelector("#selCovType")?.addEventListener("change", (e) => {
    state.covType = e.target.value;
    state.recipe.default_cov_type = e.target.value;
    actions.runStage3();
  });
  resultsArea.querySelector("#chkShowPct")?.addEventListener("change", (e) => {
    state.showPct = e.target.checked;
    patchStage3ResultsOnly(container, { state, actions });
  });
  resultsArea.querySelector("#btnCopyLatex")?.addEventListener("click", () => actions.copyLatexTable());
  resultsArea.querySelector("#btnRunRobustnessMatrix")?.addEventListener("click", () => actions.runRobustnessMatrix());
  resultsArea.querySelector("#btnCopyRobLatex")?.addEventListener("click", () => actions.copyLatexTable("robustness"));

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

  resultsArea.querySelector("#btnApplyAdCstar")?.addEventListener("click", () => {
    const prof = s3?.ad_profiles?.[state.stage3AdKey];
    if (!prof?.ok) return;
    const parts = String(state.stage3AdKey).split("::");
    const mid = parts[0];
    const tid = parts[1];
    const sp = state.recipe.models.find((m) => m.id === mid);
    const term = sp?.spatial_terms?.find((st) => st.target_id === tid && st.transform === "andersson_day");
    if (term) {
      term.cutoff_m = prof.c_star_m;
      actions.showToast?.(`c* = ${prof.c_star_m} m alkalmazva (${mid}:${tid})`, "pass");
      actions.render();
      actions.runStage3();
    }
  });

  resultsArea.querySelector("#btnUpdateAdRange")?.addEventListener("click", () => {
    const parts = String(state.stage3AdKey || "").split("::");
    const mid = parts[0];
    const tid = parts[1];
    const sp = state.recipe.models.find((m) => m.id === mid);
    const term = sp?.spatial_terms?.find((st) => st.target_id === tid && st.transform === "andersson_day");
    const rMin = Number(resultsArea.querySelector("#inpAdMin")?.value) || 100;
    const rMax = Number(resultsArea.querySelector("#inpAdMax")?.value) || 600;
    const rStep = Math.max(5, Number(resultsArea.querySelector("#inpAdStep")?.value) || 25);
    if (term && rMax > rMin) {
      term.cutoff_range_m = [rMin, rMax, rStep];
      actions.runStage3();
    }
  });

  if (!s3?.ok || !window.Plotly) return;

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

  const valid = (s3.models || []).filter((m) => m.params && Array.isArray(m.resid) && Array.isArray(m.fitted));
  const diagModel = valid.find((m) => m.model_id === state.stage3DiagModelId) || valid[0];
  if (diagModel && diagModel.resid.length > 2) {
    const fitted = diagModel.fitted;
    const resid = diagModel.resid;
    const cooks = diagModel.cooks_d || [];
    const nObs = resid.length;
    const cookThresh = 4 / Math.max(1, nObs);

    const rfEl = resultsArea.querySelector("#plotResidFitted");
    if (rfEl) {
      window.Plotly.newPlot(
        rfEl,
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
            text: cooks.map((c, idx) => `Sor #${diagModel.valid_indices?.[idx] ?? idx} | Cook D: ${Number(c).toFixed(4)} (kattints a térképes ugráshoz)`),
            name: "Reziduális",
          },
        ],
        getPlotlyThemeLayout({
          title: { text: `Reziduális vs. Becsült érték (${escapeHtml(diagModel.model_id)}) — kattints a pontra`, font: { size: 12 } },
          xaxis: { title: "ŷ (illesztett)" },
          yaxis: { title: "ε̂ (reziduális)" },
          margin: { t: 34, r: 16, b: 44, l: 50 },
          height: 240,
          shapes: [{ type: "line", x0: Math.min(...fitted), x1: Math.max(...fitted), y0: 0, y1: 0, line: { dash: "dot", color: "#94a3b8" } }],
        }),
        { responsive: true, displayModeBar: false }
      );
      if (typeof rfEl.on === "function") {
        rfEl.on("plotly_click", (evtData) => {
          const ptIdx = evtData?.points?.[0]?.pointIndex;
          if (ptIdx !== undefined && Array.isArray(diagModel.valid_indices) && diagModel.valid_indices[ptIdx] !== undefined) {
            actions.focusObservation?.(diagModel.valid_indices[ptIdx]);
          }
        });
      }
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
      <select class="sp-metric-sel" aria-label="Távolsági metrika" style="font-size:11px;" ${isCircuity ? 'disabled title="Kanyargósság transzformációnál a metrika rögzített"' : ""}>
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
               <input type="range" class="sp-ad-slider" aria-label="Andersson–Day c küszöb" min="${rMin}" max="${rMax}" step="${rStep}" value="${curC}" style="flex:1;" />
               <span class="mono sp-ad-val" style="font-weight:600;min-width:48px;">${curC} m</span>
             </div>`;
            })()
          : st.transform === "rings"
            ? `<div style="display:flex;align-items:center;gap:6px;">
                 <span style="font-size:11px;">Sávhatárok (m):</span>
                 <input type="text" class="sp-rings-inp mono" aria-label="Sávhatárok" value="${escapeHtml((st.breakpoints_m || [150, 300, 500]).join(", "))}" style="width:160px;" />
               </div>`
            : st.transform === "exp_decay"
              ? `<div style="display:flex;align-items:center;gap:6px;">
                   <span style="font-size:11px;">Felezési távolság h½ (m):</span>
                   <input type="number" class="sp-hl-inp mono" aria-label="Felezési távolság" value="${Number(st.half_life_m) || 250}" style="width:80px;" />
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

  const guardNotes = valid.flatMap((m) => (m.footnotes || []).map((f) => `[${m.model_id}] ${translateVerdict(f)}`));
  const proxyAlerts = valid.flatMap((m) => (m.diagnostics?.proxy_alerts || []).map((a) => `[${m.model_id}] ${translateVerdict(a.message)}`));
  const covWarnings = valid.flatMap((m) => (m.diagnostics?.cov_warnings || []).map((w) => `[${m.model_id}] ${translateVerdict(w)}`));
  const mixedScale = new Set(valid.map((m) => m.y_transform)).size > 1;
  const anyWcb = valid.some((m) => m.wild_bootstrap?.triggered);

  return `
    ${proxyAlerts.map((a) => `<div class="alert-box">⚠️ ${escapeHtml(a)}</div>`).join("")}
    ${covWarnings.map((w) => `<div class="alert-box">⚠️ ${escapeHtml(w)}</div>`).join("")}
    ${guardNotes.map((f) => `<div class="alert-box info">🛡️ ${escapeHtml(f)}</div>`).join("")}
    ${
      anyWcb
        ? `<div class="alert-box info">🔔 <strong>Wild Cluster Bootstrap-t (G &lt; 30):</strong> A zárójelek alatt megjelenített <code>WCB p</code> értékek kis klaszterszám mellett is érvényes inferenciát adnak.</div>`
        : ""
    }
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
            ${valid
              .map((m) => {
                const cm = m.cluster_meta;
                const clInfo = m.cov_type_used === "cluster" && cm?.n_clusters ? `, ${cm.method}, G=${cm.n_clusters}` : "";
                return `<th class="num">${escapeHtml(m.model_name)}<div class="card-subtitle" style="font-weight:400;font-size:10px;">SE: ${escapeHtml(m.cov_type_used || "—")} (df=${m.df_infer ?? "—"}${escapeHtml(clInfo)})</div></th>`;
              })
              .join("")}
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
                  const wcbBadge = p.wcb_p !== undefined ? `<div class="card-subtitle mono" style="font-size:10px;">WCB p=${p.wcb_p}</div>` : "";
                  return `<td class="num">
                    <div>${p.beta}<span class="sig-stars">${escapeHtml(p.sig)}</span></div>
                    <div class="card-subtitle" style="font-size:10px;">(${p.se})</div>
                    ${wcbBadge}${pctBadge}
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
            <td><strong>Diagnosztika</strong></td>
            ${valid
              .map((m) => {
                const d = m.diagnostics || {};
                const vifDisp = d.collinear_cols?.length || (d.max_vif ?? 0) >= 1e6 ? "∞ (Koll.)" : d.max_vif;
                return `<td class="num">
                  <div class="diag-bar" style="justify-content:flex-end;">
                    <span class="diag-chip ${d.max_vif < 5 ? "pass" : d.max_vif < 10 ? "warn" : "fail"}">VIF:${escapeHtml(vifDisp)}</span>
                    <span class="diag-chip ${(d.koenker_p ?? d.bp_p ?? 1) >= 0.05 ? "pass" : "warn"}">Koenker:${d.koenker_p ?? d.bp_p ?? "—"}</span>
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
          <div class="card-subtitle">Kondíciószám κ = <span class="diag-chip ${kappaCls}">${kappa ?? "—"}</span> | Max Leverage h<sub>ii</sub>: <span class="mono">${d.max_leverage ?? "—"}</span> | Koenker BP p = <span class="mono">${d.koenker_p ?? "—"}</span> | White p = <span class="mono">${d.white_p ?? "—"}</span> (${escapeHtml(d.white_mode || "—")})</div>
        </div>
        <select id="selDiagModel" aria-label="Mélydiagnosztikai modell kiválasztása" style="font-size:11px;">
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
  const parts = String(activeKey).split("::");
  const spModel = state.recipe.models.find((m) => m.id === parts[0]);
  const spTerm = spModel?.spatial_terms?.find((st) => st.target_id === parts[1] && st.transform === "andersson_day");
  const curRng = spTerm?.cutoff_range_m || [100, 600, 25];

  return `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">3C. Andersson–Day Küszöbprofil-Szkennelés (Profile-MLE)</h3>
          <div class="card-subtitle">${escapeHtml(translateVerdict(prof?.post_selection_note || prof?.reason || prof?.error || ""))}</div>
        </div>
        ${prof?.ok ? `<button type="button" class="btn btn-primary btn-sm" id="btnApplyAdCstar">✓ c* = ${prof.c_star_m} m alkalmazása</button>` : ""}
      </div>
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap;font-size:11px;">
        <span>Tartomány (m): min</span>
        <input type="number" id="inpAdMin" value="${curRng[0] ?? 100}" style="width:68px;" min="20" max="5000" />
        <span>max</span>
        <input type="number" id="inpAdMax" value="${curRng[1] ?? 600}" style="width:68px;" min="50" max="10000" />
        <span>lépés</span>
        <input type="number" id="inpAdStep" value="${curRng[2] ?? 25}" style="width:60px;" min="5" max="500" />
        <button type="button" class="btn btn-sm" id="btnUpdateAdRange">↻ Szkennelés frissítése</button>
      </div>
      ${
        allKeys.length > 1
          ? `<div class="spec-tabs" style="margin-bottom:8px;">
               ${allKeys
                 .map(
                   (k) => `<button type="button" class="spec-tab ad-prof-tab ${k === activeKey ? "active" : ""}" data-ad-key="${escapeHtml(k)}">${escapeHtml(k)}${adProfiles[k]?.ok ? ` (c*=${adProfiles[k].c_star_m}m)` : " (hiba)"}</button>`
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
    return `<div class="card"><div class="card-subtitle"><strong>Guard #4 Heckman 2-Lépcsős Szelekció:</strong> ${escapeHtml(translateVerdict(heck.reason))}</div></div>`;
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
          <thead><tr><th>Változó</th><th class="num">Alap SE β̂</th><th class="num">Heckman β̂ (Korrigált SE)</th><th class="num">Heckman p</th></tr></thead>
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
