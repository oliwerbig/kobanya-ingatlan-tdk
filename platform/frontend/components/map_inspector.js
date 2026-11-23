/* Docked MapLibre GL Map & Clickable Point / OSM-Toggle Inspector */

const LISA_COLORS = {
  HH: "#dc2626",
  LL: "#2563eb",
  HL: "#f59e0b",
  LH: "#7c3aed",
  NS: "#94a3b8",
};

export class MapInspector {
  constructor(containerId, inspectorId, callbacks) {
    this.containerId = containerId;
    this.inspectorId = inspectorId;
    this.callbacks = callbacks || {};
    this.map = null;
    this.ready = false;
    this.layerMode = "y_raw";
    this.gwrColIdx = 1;
    this.targetGeojsonMap = new Map(); // targetId -> FeatureCollection
    this.obsFeatures = [];
    this.selectedObsIdx = null;
    this.lastRouteTargetId = null;
  }

  init() {
    if (this.map || !window.maplibregl) return;
    const osmRasterFallback = {
      version: 8,
      sources: {
        osm: {
          type: "raster",
          tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
          tileSize: 256,
          attribution: "&copy; OpenStreetMap contributors",
        },
      },
      layers: [{ id: "osm-base", type: "raster", source: "osm" }],
    };

    this.map = new window.maplibregl.Map({
      container: this.containerId,
      style: "https://tiles.openfreemap.org/styles/positron",
      center: [15.0, 48.0],
      zoom: 4,
    });
    this.map.on("error", (e) => {
      if (e?.error?.message && /style/i.test(e.error.message)) {
        this.map.setStyle(osmRasterFallback);
      }
    });
    this.map.addControl(new window.maplibregl.NavigationControl(), "top-right");
    this.map.addControl(new window.maplibregl.ScaleControl({ maxWidth: 120, unit: "metric" }), "bottom-left");

    this.map.on("load", () => {
      this.ready = true;
      // Target geometries source (supports interactive Stage 1 click-to-toggle!)
      this.map.addSource("targets-src", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      this.map.addLayer({
        id: "targets-fill",
        type: "fill",
        source: "targets-src",
        filter: ["in", ["geometry-type"], ["literal", ["Polygon", "MultiPolygon"]]],
        paint: {
          "fill-color": ["case", ["boolean", ["get", "excluded"], false], "#94a3b8", "#0d9488"],
          "fill-opacity": ["case", ["boolean", ["get", "excluded"], false], 0.08, 0.22],
        },
      });
      this.map.addLayer({
        id: "targets-line",
        type: "line",
        source: "targets-src",
        filter: ["in", ["geometry-type"], ["literal", ["LineString", "MultiLineString", "Polygon", "MultiPolygon"]]],
        paint: {
          "line-color": ["case", ["boolean", ["get", "excluded"], false], "#94a3b8", "#0f766e"],
          "line-width": ["case", ["boolean", ["get", "excluded"], false], 1.5, 2.5],
          "line-dasharray": [2, 1],
        },
      });
      this.map.addLayer({
        id: "targets-point",
        type: "circle",
        source: "targets-src",
        filter: ["in", ["geometry-type"], ["literal", ["Point", "MultiPoint"]]],
        paint: {
          "circle-radius": 6,
          "circle-color": ["case", ["boolean", ["get", "excluded"], false], "#94a3b8", "#0f766e"],
          "circle-stroke-width": 1.5,
          "circle-stroke-color": "#ffffff",
          "circle-opacity": ["case", ["boolean", ["get", "excluded"], false], 0.35, 0.92],
        },
      });

      // Route replay source
      this.map.addSource("route-src", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      this.map.addLayer({
        id: "route-eucl",
        type: "line",
        source: "route-src",
        filter: ["==", ["get", "kind"], "euclidean"],
        paint: { "line-color": "#64748b", "line-width": 2, "line-dasharray": [3, 2] },
      });
      this.map.addLayer({
        id: "route-net",
        type: "line",
        source: "route-src",
        filter: ["==", ["get", "kind"], "network_dijkstra"],
        paint: { "line-color": "#0d9488", "line-width": 3.5 },
      });

      // Observations source
      this.map.addSource("obs-src", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      this.map.addLayer({
        id: "obs-points",
        type: "circle",
        source: "obs-src",
        paint: {
          "circle-radius": ["case", ["==", ["get", "selected"], true], 8, 5],
          "circle-color": ["coalesce", ["get", "color"], "#1d4ed8"],
          "circle-opacity": ["coalesce", ["get", "opacity"], 0.85],
          "circle-stroke-width": ["case", ["==", ["get", "selected"], true], 2.5, 1],
          "circle-stroke-color": "#ffffff",
        },
      });

      // Click handlers
      const handleTargetClick = (e) => {
        if (!e.features || !e.features.length) return;
        const props = e.features[0].properties || {};
        if (props.target_id && props.uid && this.callbacks.onToggleOsmFeature) {
          this.callbacks.onToggleOsmFeature(props.target_id, props.uid);
        }
      };
      ["targets-fill", "targets-line", "targets-point"].forEach((lid) => {
        this.map.on("click", lid, handleTargetClick);
        this.map.on("mouseenter", lid, () => { this.map.getCanvas().style.cursor = "pointer"; });
        this.map.on("mouseleave", lid, () => { this.map.getCanvas().style.cursor = ""; });
      });

      this.map.on("click", "obs-points", (e) => {
        if (!e.features || !e.features.length) return;
        const idx = Number(e.features[0].properties.obs_idx);
        this.selectedObsIdx = idx;
        if (this.callbacks.onClickObservation) {
          this.callbacks.onClickObservation(idx);
        }
      });
      this.map.on("mouseenter", "obs-points", () => { this.map.getCanvas().style.cursor = "pointer"; });
      this.map.on("mouseleave", "obs-points", () => { this.map.getCanvas().style.cursor = ""; });

      this._flushSources();
    });
  }

  _colorScale(val, minV, maxV, diverging = false) {
    if (val === null || val === undefined || !Number.isFinite(val)) return "#94a3b8";
    const span = Math.max(1e-9, maxV - minV);
    if (diverging) {
      const lim = Math.max(Math.abs(minV), Math.abs(maxV), 1e-9);
      const t = Math.max(-1, Math.min(1, val / lim));
      if (t >= 0) {
        const g = Math.round(230 - 150 * t);
        return `rgb(220, ${g}, 60)`;
      }
      const r = Math.round(230 + 180 * t);
      return `rgb(${Math.max(30, r)}, 130, 235)`;
    }
    const u = Math.max(0, Math.min(1, (val - minV) / span));
    const r = Math.round(37 + 190 * u);
    const g = Math.round(99 + 50 * (1 - Math.abs(u - 0.5) * 2));
    const b = Math.round(235 - 180 * u);
    return `rgb(${r}, ${g}, ${b})`;
  }

  setTargetPreview(targetId, previewGeojson) {
    this.targetGeojsonMap.set(targetId, previewGeojson);
    this._flushTargets();
  }

  clearTargets() {
    this.targetGeojsonMap.clear();
    this._flushTargets();
  }

  _flushTargets() {
    if (!this.ready) return;
    const merged = [];
    for (const [tid, fc] of this.targetGeojsonMap.entries()) {
      for (const f of fc.features || []) {
        merged.push({
          ...f,
          properties: { ...(f.properties || {}), target_id: tid },
        });
      }
    }
    const src = this.map.getSource("targets-src");
    if (src) src.setData({ type: "FeatureCollection", features: merged });
  }

  setObservationsFromDataset(columns, latCol, lonCol, rowIdCol, ySpec, fitMap = true) {
    const lats = columns[latCol] || [];
    const lons = columns[lonCol] || [];
    const ids = rowIdCol && columns[rowIdCol] ? columns[rowIdCol] : null;
    const nums = ySpec.num_col && columns[ySpec.num_col] ? columns[ySpec.num_col] : null;
    const dens = ySpec.mode === "ratio" && ySpec.den_col && columns[ySpec.den_col] ? columns[ySpec.den_col] : null;

    const parseNum = (v) => {
      if (v === null || v === undefined || v === "") return NaN;
      const n = Number(String(v).replace(/\u00a0/g, "").replace(/\s/g, "").replace(",", "."));
      return Number.isFinite(n) ? n : NaN;
    };

    const feats = [];
    let minLon = 180, maxLon = -180, minLat = 90, maxLat = -90;
    for (let i = 0; i < lats.length; i++) {
      const la = parseNum(lats[i]);
      const lo = parseNum(lons[i]);
      if (!Number.isFinite(la) || !Number.isFinite(lo) || (Math.abs(la) < 1e-12 && Math.abs(lo) < 1e-12)) continue;
      let yv = nums ? parseNum(nums[i]) : NaN;
      if (dens && Number.isFinite(yv)) {
        const dv = parseNum(dens[i]);
        yv = dv > 0 ? yv / dv : NaN;
      }
      feats.push({
        type: "Feature",
        geometry: { type: "Point", coordinates: [lo, la] },
        properties: {
          obs_idx: i,
          row_id: ids ? ids[i] : i,
          y_raw: Number.isFinite(yv) ? yv : null,
          color: "#1d4ed8",
          opacity: 0.85,
          selected: false,
        },
      });
      minLon = Math.min(minLon, lo);
      maxLon = Math.max(maxLon, lo);
      minLat = Math.min(minLat, la);
      maxLat = Math.max(maxLat, la);
    }

    this.obsFeatures = feats;
    this.restyleObservations({});
    if (fitMap && this.ready && feats.length > 0) {
      this.map.fitBounds([[minLon, minLat], [maxLon, maxLat]], { padding: 36, maxZoom: 15, duration: 500 });
    }
  }

  restyleObservations({ stage3, stage4, stage5, layerMode, gwrColIdx }) {
    if (layerMode) this.layerMode = layerMode;
    if (gwrColIdx !== undefined) this.gwrColIdx = gwrColIdx;

    // Build lookup by original row index
    const olsMap = new Map();
    if (stage3 && stage3.models && stage3.models.length) {
      const m = stage3.models.find((x) => x.resid) || stage3.models[0];
      if (m && m.valid_indices && m.resid) {
        m.valid_indices.forEach((origI, pos) => {
          olsMap.set(origI, { resid: m.resid[pos], cooks: m.cooks_d ? m.cooks_d[pos] : 0 });
        });
      }
    }

    const lisaMap = new Map();
    if (stage4 && stage4.ok && stage4.valid_indices && stage4.lisa_resid) {
      stage4.valid_indices.forEach((origI, pos) => {
        lisaMap.set(origI, {
          quad_y: stage4.lisa_y.quadrants[pos],
          quad_r: stage4.lisa_resid.quadrants[pos],
        });
      });
    }

    const gwrMap = new Map();
    if (stage5 && stage5.ok && stage5.valid_indices) {
      stage5.valid_indices.forEach((origI, pos) => {
        const ci = Math.min(this.gwrColIdx, (stage5.betas_native[pos] || []).length - 1);
        gwrMap.set(origI, {
          beta: stage5.betas_native[pos][ci],
          t: stage5.t_vals[pos][ci],
          r2: stage5.local_r2[pos],
        });
      });
    }

    // Collect bounds for scaling
    const vals = [];
    for (const f of this.obsFeatures) {
      const idx = f.properties.obs_idx;
      if (this.layerMode === "y_raw" && f.properties.y_raw !== null) vals.push(f.properties.y_raw);
      else if (this.layerMode === "ols_resid" && olsMap.has(idx)) vals.push(olsMap.get(idx).resid);
      else if (this.layerMode === "gwr_beta" && gwrMap.has(idx)) vals.push(gwrMap.get(idx).beta);
      else if (this.layerMode === "gwr_r2" && gwrMap.has(idx)) vals.push(gwrMap.get(idx).r2);
    }
    const minV = vals.length ? Math.min(...vals) : 0;
    const maxV = vals.length ? Math.max(...vals) : 1;

    for (const f of this.obsFeatures) {
      const idx = f.properties.obs_idx;
      f.properties.selected = idx === this.selectedObsIdx;
      f.properties.opacity = 0.88;
      if (this.layerMode === "y_raw") {
        f.properties.color = this._colorScale(f.properties.y_raw, minV, maxV, false);
      } else if (this.layerMode === "ols_resid") {
        const o = olsMap.get(idx);
        f.properties.color = o ? this._colorScale(o.resid, minV, maxV, true) : "#cbd5e1";
      } else if (this.layerMode === "lisa") {
        const l = lisaMap.get(idx);
        f.properties.color = l ? LISA_COLORS[l.quad_r] || "#94a3b8" : "#cbd5e1";
        if (l && l.quad_r === "NS") f.properties.opacity = 0.45;
      } else if (this.layerMode === "gwr_beta") {
        const g = gwrMap.get(idx);
        f.properties.color = g ? this._colorScale(g.beta, minV, maxV, true) : "#cbd5e1";
        // Dim non-significant points (|t_i| < 1.96)
        if (g && Math.abs(g.t) < 1.96) f.properties.opacity = 0.30;
      } else if (this.layerMode === "gwr_r2") {
        const g = gwrMap.get(idx);
        f.properties.color = g ? this._colorScale(g.r2, minV, maxV, false) : "#cbd5e1";
      }
    }

    this._flushObs();
  }

  _flushObs() {
    if (!this.ready) return;
    const src = this.map.getSource("obs-src");
    if (src) src.setData({ type: "FeatureCollection", features: this.obsFeatures });
  }

  _flushSources() {
    this._flushTargets();
    this._flushObs();
  }

  showRoute(routeRes) {
    if (!this.ready) return;
    const feats = [];
    if (routeRes && routeRes.euclidean_geojson) feats.push(routeRes.euclidean_geojson);
    if (routeRes && routeRes.network_geojson) feats.push(routeRes.network_geojson);
    const src = this.map.getSource("route-src");
    if (src) src.setData({ type: "FeatureCollection", features: feats });
  }

  renderPointInspector(obsIdx, info) {
    const el = document.getElementById(this.inspectorId);
    if (!el) return;
    if (obsIdx === null || !info) {
      el.innerHTML = `<div class="card-subtitle">Kattints bármely megfigyelési pontra a térképen a pontos távolságok, Dijkstra sétaútvonal, reziduum, LISA-klaszter és GWR lokális együtthatók megtekintéséhez (Stage 1-ben az OSM objektumokra kattintva egyesével ki-/bekapcsolhatók az elemek).</div>`;
      return;
    }
    const route = info.route;
    const detourTxt =
      route && route.ok && route.dist_net_m !== null
        ? `<span class="mono">d<sub>eucl</sub>=${route.dist_eucl_m} m | d<sub>net</sub>=${route.dist_net_m} m | Δd=${route.detour_m} m</span>`
        : route && route.ok
        ? `<span class="mono">d<sub>eucl</sub>=${route.dist_eucl_m} m (Dijkstra útvonal nincs betöltve)</span>`
        : `<span class="card-subtitle">Nincs aktív célpont útvonal-visszajátszáshoz</span>`;

    el.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <strong>📍 Megfigyelés #${info.row_id} <span class="card-subtitle">(sor ${obsIdx})</span></strong>
        <span class="mono">y = ${info.y_raw !== null ? Number(info.y_raw).toLocaleString("hu-HU", { maximumFractionDigits: 2 }) : "—"}</span>
      </div>
      <div style="margin-bottom:6px;">${detourTxt}</div>
      <div class="grid-3" style="font-size:11px;">
        <div><strong>OLS reziduum:</strong> <span class="mono">${info.resid !== null ? info.resid : "—"}</span></div>
        <div><strong>LISA klaszter:</strong> <span class="mono">${info.lisa || "—"}</span></div>
        <div><strong>GWR lokális β̂ / t:</strong> <span class="mono">${info.gwr_beta !== null ? `${info.gwr_beta} (t=${info.gwr_t})` : "—"}</span></div>
      </div>
    `;
  }
}
