/* Stage 3: Visual Pill Formula Composer, Cloneable Spec Tabs, M1-M4 Starter,
 * Live Andersson-Day Cutoff Slider + Profile-MLE Curve, Diagnostics & Heckman 2-Step
 */

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
    <div class="card">
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
          <button class="spec-tab ${m.id === state.activeSpecId ? "active" : ""}" data-spec-id="${m.id}">
            ${m.name}
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
          <input type="text" id="txtSpecName" value="${activeSpec.name}" style="width:220px;font-weight:600;" />
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
              ${c.type === "categorical" ? `${c.col} [ref:${c.reference_level || "auto"}]` : `${c.transform && c.transform !== "raw" ? `${c.transform}(${c.col})` : c.col}`}
              <button class="remove-x" data-rm-cov="${cidx}">×</button>
            </span>`
            )
            .join("")}
          ${(activeSpec.spatial_terms || [])
            .map(
              (st, sidx) => `
            <span class="pill spatial-pill">
              ${formatSpatialPill(st)}
              <button class="remove-x" data-rm-sp="${sidx}">×</button>
            </span>`
            )
            .join("")}
          ${(activeSpec.interactions || [])
            .map(
              (it, iidx) => `
            <span class="pill inter-pill">
              ${it.left} × ${it.right}
              <button class="remove-x" data-rm-int="${iidx}">×</button>
            </span>`
            )
            .join("")}
        </div>

        <!-- Add Covariate Controls -->
        <div class="grid-3" style="margin-bottom:8px;">
          <div class="field-row">
            <label>Kovariáns oszlop hozzáadása</label>
            <div style="display:flex;gap:4px;">
              <select id="selAddCov" style="flex:1;">
                ${availCovCols.map((c) => `<option value="${c}">${c} (${colProfile[c]?.isNumeric ? "num" : "kat"})</option>`).join("")}
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
                    ? targets.map((t) => `<option value="${t.id}">${t.id}</option>`).join("")
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

        <!-- Inline Spatial Term Parameters & Live AD Cutoff Slider -->
        ${renderSpatialTermInspectors(activeSpec)}
      `
          : ""
      }
    </div>

    ${
      s3 && s3.ok
        ? `
      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">3B. Összehasonlító Regressziós Mátrix & Diagnosztika</h3>
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

        ${renderComparisonTable(s3.models, state.showPct)}
      </div>

      ${renderADProfileCard(s3.ad_profiles)}
      ${renderHeckmanCard(s3.heckman)}
    `
        : ""
    }
  `;

  // Bind Spec tabs
  container.querySelectorAll(".spec-tab").forEach((b) => {
    b.addEventListener("click", () => {
      state.activeSpecId = b.getAttribute("data-spec-id");
      actions.render();
    });
  });

  container.querySelector("#btnGenM1M4").addEventListener("click", () => actions.generateM1M4Specs());
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

    // Live AD Cutoff slider (<15 ms DAG incremental OLS update!)
    container.querySelectorAll("[data-sp-edit-idx]").forEach((box) => {
      const sidx = Number(box.getAttribute("data-sp-edit-idx"));
      const st = activeSpec.spatial_terms[sidx];
      box.querySelector(".sp-metric-sel")?.addEventListener("change", (e) => {
        st.metric = e.target.value;
      });
      const slider = box.querySelector(".sp-ad-slider");
      const valLbl = box.querySelector(".sp-ad-val");
      if (slider && valLbl) {
        slider.addEventListener("input", (e) => {
          const cv = Number(e.target.value);
          st.cutoff_m = cv;
          valLbl.textContent = `${cv} m`;
          actions.runStage3Debounced();
        });
      }
      box.querySelector(".sp-rings-inp")?.addEventListener("change", (e) => {
        const arr = e.target.value
          .split(",")
          .map((v) => Number(v.trim()))
          .filter((v) => Number.isFinite(v) && v > 0);
        if (arr.length) st.breakpoints_m = arr;
      });
      box.querySelector(".sp-hl-inp")?.addEventListener("change", (e) => {
        st.half_life_m = Math.max(1, Number(e.target.value) || 250);
      });
    });
  }

  container.querySelector("#selCovType")?.addEventListener("change", (e) => {
    state.covType = e.target.value;
    actions.runStage3();
  });
  container.querySelector("#chkShowPct")?.addEventListener("change", (e) => {
    state.showPct = e.target.checked;
    actions.render();
  });
  container.querySelector("#btnCopyLatex")?.addEventListener("click", () => actions.copyLatexTable());

  // Render AD Profile Curve Plotly chart
  if (s3 && s3.ad_profiles && window.Plotly) {
    const firstKey = Object.keys(s3.ad_profiles).find((k) => s3.ad_profiles[k]?.ok);
    if (firstKey) {
      const prof = s3.ad_profiles[firstKey];
      const pts = prof.curve.filter((p) => p.aicc !== null);
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
        {
          title: { text: `Andersson–Day Profil-MLE (${firstKey}) | c* = ${prof.c_star_m} m [95% CI: ${prof.c_ci_95[0]}–${prof.c_ci_95[1]} m]`, font: { size: 12 } },
          xaxis: { title: "Cenzorálási küszöb c (m)" },
          yaxis: { title: "AICc (kisebb jobb)" },
          yaxis2: { title: "β̂(c)", overlaying: "y", side: "right" },
          margin: { t: 36, r: 52, b: 42, l: 52 },
          height: 250,
          legend: { orientation: "h", y: -0.22 },
        },
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
    .map(
      (st, idx) => `
    <div style="display:flex;gap:10px;align-items:center;padding:6px 8px;border:1px solid var(--border);border-radius:6px;background:var(--surface-alt);margin-bottom:6px;flex-wrap:wrap;" data-sp-edit-idx="${idx}">
      <strong style="font-size:11px;">${st.target_id} (${st.transform})</strong>
      <select class="sp-metric-sel" style="font-size:11px;">
        <option value="eucl" ${st.metric === "eucl" ? "selected" : ""}>Euklideszi</option>
        <option value="net" ${st.metric === "net" ? "selected" : ""}>Hálózati Dijkstra</option>
        <option value="circuity_delta" ${st.metric === "circuity_delta" ? "selected" : ""}>Δd (m)</option>
        <option value="circuity_ratio" ${st.metric === "circuity_ratio" ? "selected" : ""}>τ arány</option>
      </select>
      ${
        st.transform === "andersson_day"
          ? `<div style="display:flex;align-items:center;gap:8px;flex:1;min-width:220px;">
               <span style="font-size:11px;">Küszöb c:</span>
               <input type="range" class="sp-ad-slider" min="50" max="1000" step="25" value="${st.cutoff_m || 300}" style="flex:1;" />
               <span class="mono sp-ad-val" style="font-weight:600;min-width:48px;">${st.cutoff_m || 300} m</span>
             </div>`
          : st.transform === "rings"
          ? `<div style="display:flex;align-items:center;gap:6px;">
               <span style="font-size:11px;">Sávhatárok (m):</span>
               <input type="text" class="sp-rings-inp mono" value="${(st.breakpoints_m || [150, 300, 500]).join(", ")}" style="width:160px;" />
             </div>`
          : st.transform === "exp_decay"
          ? `<div style="display:flex;align-items:center;gap:6px;">
               <span style="font-size:11px;">Felezési távolság h½ (m):</span>
               <input type="number" class="sp-hl-inp mono" value="${st.half_life_m || 250}" style="width:80px;" />
             </div>`
          : ""
      }
    </div>`
    )
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

  return `
    ${proxyAlerts.map((a) => `<div class="alert-box">⚠️ ${a}</div>`).join("")}
    ${guardNotes.map((f) => `<div class="alert-box info">🛡️ ${f}</div>`).join("")}
    <div class="table-wrap">
      <table class="academic">
        <thead>
          <tr>
            <th>Változó</th>
            ${valid.map((m) => `<th class="num">${m.model_name}</th>`).join("")}
          </tr>
        </thead>
        <tbody>
          ${varOrder
            .map(
              (vn) => `
            <tr>
              <td><strong>${vn}</strong></td>
              ${maps
                .map((mp) => {
                  const p = mp[vn];
                  if (!p) return `<td class="num card-subtitle">—</td>`;
                  const pctBadge =
                    showPct && p.kennedy_pct !== null
                      ? `<div class="card-subtitle mono" style="font-size:10px;">Kennedy: ${p.kennedy_pct >= 0 ? "+" : ""}${p.kennedy_pct}% (HP: ${p.hp_pct >= 0 ? "+" : ""}${p.hp_pct}%)</div>`
                      : "";
                  return `<td class="num">
                    <div>${p.beta}<span class="sig-stars">${p.sig}</span></div>
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
          <tr>
            <td><strong>5-szörös Diagnosztika</strong></td>
            ${valid
              .map((m) => {
                const d = m.diagnostics;
                return `<td class="num">
                  <div class="diag-bar" style="justify-content:flex-end;">
                    <span class="diag-chip ${d.max_vif < 5 ? "pass" : d.max_vif < 10 ? "warn" : "fail"}">VIF:${d.max_vif}</span>
                    <span class="diag-chip ${d.bp_p >= 0.05 ? "pass" : "warn"}">BP:${d.bp_p}</span>
                    <span class="diag-chip ${d.reset_p === null || d.reset_p >= 0.05 ? "pass" : "fail"}">RESET:${d.reset_p ?? "—"}</span>
                    <span class="diag-chip ${d.jb_p >= 0.05 ? "pass" : "warn"}">JB:${d.jb_p}</span>
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

function renderADProfileCard(adProfiles) {
  const keys = Object.keys(adProfiles || {}).filter((k) => adProfiles[k]?.ok);
  if (!keys.length) return "";
  const prof = adProfiles[keys[0]];
  return `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">3C. Andersson–Day Küszöbprofil-Szkennelés (Profile-MLE)</h3>
          <div class="card-subtitle">${prof.post_selection_note}</div>
        </div>
      </div>
      <div id="plotADProfile" class="plot-box"></div>
    </div>
  `;
}

function renderHeckmanCard(heck) {
  if (!heck) return "";
  if (!heck.activated) {
    return `<div class="card"><div class="card-subtitle"><strong>Guard #4 Heckman 2-Lépcsős Szelekció:</strong> ${heck.reason}</div></div>`;
  }
  return `
    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">3D. Guard #4 — Heckman (1979) 2-Lépcsős Szelekciós Modell</h3>
          <div class="card-subtitle">Teljes pool: ${heck.n_total_pool} | Elemzett: ${heck.n_analytic} | Kizárt: ${heck.n_excluded} | λ<sub>IMR</sub> = ${heck.imr_beta} (SE=${heck.imr_se}, p=${heck.imr_p})</div>
        </div>
      </div>
      ${(heck.warnings || []).map((w) => `<div class="alert-box">⚠️ ${w}</div>`).join("")}
      <div class="table-wrap">
        <table class="academic">
          <thead><tr><th>Változó</th><th class="num">Alap HC3 β̂ (SE)</th><th class="num">Heckman β̂ (Korrigált SE)</th><th class="num">Heckman p</th></tr></thead>
          <tbody>
            ${heck.comparison
              .map(
                (r) => `<tr>
                  <td><strong>${r.label}</strong></td>
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
