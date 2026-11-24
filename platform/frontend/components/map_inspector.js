/* Docked MapLibre GL Map, Dynamic Legend, Co-located Parcel Paginator, BBox/Snap Overlay & Bidirectional Plot Focus */
import { translateVerdict } from "../utils/i18n.js";
import { escapeHtml, parseNumericScalar } from "../utils/sanitize.js";

const LISA_COLORS = {
  HH: "#dc2626",
  LL: "#2563eb",
  HL: "#f59e0b",
  LH: "#7c3aed",
  NS: "#94a3b8",
};

const SNAP_COLORS = {
  0: "#16a34a", // SNAP_OK
  1: "#d97706", // SNAP_DISCONNECTED
  2: "#dc2626", // SNAP_EXCEEDED
  3: "#94a3b8", // SNAP_TARGET_EMPTY / fallback
  4: "#64748b", // SNAP_FETCH_FAILED
};

function makeFallbackStyle(isDark, useTiles) {
  if (!useTiles) {
    return {
      version: 8,
      sources: {},
      layers: [
        {
          id: "offline-bg",
          type: "background",
          paint: { "background-color": isDark ? "#0f172a" : "#f1f5f9" },
        },
      ],
    };
  }
  return {
    version: 8,
    sources: {
      osm: {
        type: "raster",
        tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
        attribution: "&copy; OpenStreetMap contributors",
      },
    },
    layers: [
      {
        id: "osm-base",
        type: "raster",
        source: "osm",
        paint: isDark
          ? { "raster-brightness-max": 0.42, "raster-saturation": -0.45, "raster-contrast": 0.15 }
          : {},
      },
    ],
  };
}

export class MapInspector {
  constructor(containerId, inspectorId, callbacks) {
    this.containerId = containerId;
    this.inspectorId = inspectorId;
    this.legendId = "mapLegend";
    this.callbacks = callbacks || {};
    this.map = null;
    this.ready = false;
    this.isDark = false;
    this.fallbackTier = 0;
    this.layerMode = "y_raw";
    this.gwrColIdx = 1;
    this.gwrMaskMode = "fdr";
    this.activeModelId = null;
    this.activeRouteTargetId = "";
    this.targetGeojsonMap = new Map();
    this.obsFeatures = [];
    this.obsByIdx = new Map();
    this.colocatedGroups = new Map();
    this.obsClusterKey = new Map();
    this.routeFeatures = [];
    this.fetchedBboxGeojson = null;
    this.selectedObsIdx = null;
    this._pendingBounds = null;
    this._lastRestyleArgs = {};
  }

  init() {
    if (this.map || !window.maplibregl) return;
    this.isDark = typeof document !== "undefined" && document.body?.classList?.contains("dark-theme");
    const online = typeof navigator === "undefined" ? true : navigator.onLine !== false;
    let initialStyle;
    if (!online) {
      this.fallbackTier = 2;
      initialStyle = makeFallbackStyle(this.isDark, false);
    } else {
      this.fallbackTier = 0;
      initialStyle = this.isDark
        ? "https://tiles.openfreemap.org/styles/dark"
        : "https://tiles.openfreemap.org/styles/positron";
    }

    this.map = new window.maplibregl.Map({
      container: this.containerId,
      style: initialStyle,
      center: [15.0, 48.0],
      zoom: 4,
    });

    const triggerFallback = () => {
      if (this.fallbackTier >= 2) return;
      this.fallbackTier += 1;
      const useTiles = this.fallbackTier === 1 && (typeof navigator === "undefined" || navigator.onLine !== false);
      if (!useTiles) this.fallbackTier = 2;
      this.map.setStyle(makeFallbackStyle(this.isDark, useTiles));
    };

    const bootWatchdog = setTimeout(() => {
      if (!this.ready && this.fallbackTier === 0) {
        triggerFallback();
      }
    }, 4500);

    this.map.on("error", (e) => {
      if (!this.ready) {
        triggerFallback();
      } else if (e?.error?.message && /style/i.test(e.error.message)) {
        triggerFallback();
      }
    });

    this.map.addControl(new window.maplibregl.NavigationControl(), "top-right");
    this.map.addControl(new window.maplibregl.ScaleControl({ maxWidth: 120, unit: "metric" }), "bottom-left");

    this.map.on("load", () => {
      clearTimeout(bootWatchdog);
      this.ready = true;
      this._ensureLayers();
      this._bindMapEvents();
      this._flushSources();
      if (this._pendingBounds) {
        this._applyFitBounds(this._pendingBounds);
        this._pendingBounds = null;
      }
    });

    this.map.on("style.load", () => {
      if (this.ready) {
        this._ensureLayers();
        this._flushSources();
      }
    });
  }

  _applyFitBounds(bounds) {
    if (!this.map || !this.ready || !bounds) return;
    try {
      this.map.fitBounds(bounds, { padding: 36, maxZoom: 15, duration: 500 });
    } catch (_) {}
  }

  setTheme(isDark) {
    this.isDark = Boolean(isDark);
    if (!this.map) return;
    if (this.fallbackTier === 0 && (typeof navigator === "undefined" || navigator.onLine !== false)) {
      this.map.setStyle(
        this.isDark
          ? "https://tiles.openfreemap.org/styles/dark"
          : "https://tiles.openfreemap.org/styles/positron"
      );
    } else {
      this.map.setStyle(makeFallbackStyle(this.isDark, this.fallbackTier === 1));
    }
  }

  _ensureLayers() {
    if (!this.map) return;
    if (!this.map.getSource("fetched-bbox-src")) {
      this.map.addSource("fetched-bbox-src", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
    }
    if (!this.map.getLayer("fetched-bbox-line")) {
      this.map.addLayer({
        id: "fetched-bbox-line",
        type: "line",
        source: "fetched-bbox-src",
        paint: { "line-color": "#0284c7", "line-width": 2, "line-dasharray": [4, 3], "line-opacity": 0.75 },
      });
    }

    if (!this.map.getSource("targets-src")) {
      this.map.addSource("targets-src", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
    }
    if (!this.map.getLayer("targets-fill")) {
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
    }
    if (!this.map.getLayer("targets-line")) {
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
    }
    if (!this.map.getLayer("targets-point")) {
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
    }

    if (!this.map.getSource("route-src")) {
      this.map.addSource("route-src", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
    }
    if (!this.map.getLayer("route-eucl")) {
      this.map.addLayer({
        id: "route-eucl",
        type: "line",
        source: "route-src",
        filter: ["==", ["get", "kind"], "euclidean"],
        paint: { "line-color": "#64748b", "line-width": 2, "line-dasharray": [3, 2] },
      });
    }
    if (!this.map.getLayer("route-net")) {
      this.map.addLayer({
        id: "route-net",
        type: "line",
        source: "route-src",
        filter: ["==", ["get", "kind"], "network_dijkstra"],
        paint: { "line-color": "#0d9488", "line-width": 3.5 },
      });
    }

    if (!this.map.getSource("obs-src")) {
      this.map.addSource("obs-src", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
    }
    if (!this.map.getLayer("obs-points")) {
      this.map.addLayer({
        id: "obs-points",
        type: "circle",
        source: "obs-src",
        paint: {
          "circle-radius": ["case", ["==", ["get", "selected"], true], 9, 5],
          "circle-color": ["coalesce", ["get", "color"], "#1d4ed8"],
          "circle-opacity": ["coalesce", ["get", "opacity"], 0.85],
          "circle-stroke-width": ["case", ["==", ["get", "selected"], true], 2.5, 1],
          "circle-stroke-color": "#ffffff",
        },
      });
    }
  }

  _bindMapEvents() {
    if (this._eventsBound || !this.map) return;
    this._eventsBound = true;

    let _lastTargetClickTs = 0;
    let _lastTargetClickUid = "";
    const handleTargetClick = (e) => {
      if (!e.features || !e.features.length) return;
      if (e.point) {
        const obsHits = this.map.queryRenderedFeatures(e.point, { layers: ["obs-points"] });
        if (obsHits && obsHits.length > 0) return;
      }
      const props = e.features[0].properties || {};
      if (props.target_id && props.uid && this.callbacks.onToggleOsmFeature) {
        const now = Date.now();
        const key = `${props.target_id}::${props.uid}`;
        if (key === _lastTargetClickUid && now - _lastTargetClickTs < 120) return;
        _lastTargetClickTs = now;
        _lastTargetClickUid = key;
        this.callbacks.onToggleOsmFeature(props.target_id, props.uid);
      }
    };
    ["targets-fill", "targets-line", "targets-point"].forEach((lid) => {
      this.map.on("click", lid, handleTargetClick);
      this.map.on("mouseenter", lid, () => {
        this.map.getCanvas().style.cursor = "pointer";
      });
      this.map.on("mouseleave", lid, () => {
        this.map.getCanvas().style.cursor = "";
      });
    });

    this.map.on("click", "obs-points", (e) => {
      if (!e.features || !e.features.length) return;
      const idx = Number(e.features[0].properties.obs_idx);
      this.selectedObsIdx = idx;
      if (this.callbacks.onClickObservation) {
        this.callbacks.onClickObservation(idx);
      }
    });
    this.map.on("mouseenter", "obs-points", () => {
      this.map.getCanvas().style.cursor = "pointer";
    });
    this.map.on("mouseleave", "obs-points", () => {
      this.map.getCanvas().style.cursor = "";
    });
  }

  _colorScale(val, minV, maxV, diverging = false) {
    if (val === null || val === undefined || !Number.isFinite(val)) return "#94a3b8";
    const span = Math.max(1e-9, maxV - minV);
    if (diverging) {
      const lim = Math.max(Math.abs(minV), Math.abs(maxV), 1e-9);
      const t = Math.max(-1, Math.min(1, val / lim));
      if (t >= 0) {
        const r = Math.round(230 - 10 * t);
        const g = Math.round(230 - 150 * t);
        const b = Math.round(230 - 170 * t);
        return `rgb(${r}, ${g}, ${b})`;
      }
      const u = -t;
      const r = Math.round(230 - 180 * u);
      const g = Math.round(230 - 100 * u);
      const b = Math.round(230 + 5 * u);
      return `rgb(${Math.max(30, r)}, ${g}, ${b})`;
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

  setFetchedBbox(bbox) {
    if (!bbox || !Number.isFinite(bbox.south) || !Number.isFinite(bbox.north) || !Number.isFinite(bbox.west) || !Number.isFinite(bbox.east)) {
      this.fetchedBboxGeojson = null;
    } else {
      const { south: s, west: w, north: n, east: e } = bbox;
      const boxes = w > e ? [[w, s, 180, n], [-180, s, e, n]] : [[w, s, e, n]];
      this.fetchedBboxGeojson = {
        type: "FeatureCollection",
        features: boxes.map(([bxW, bxS, bxE, bxN]) => ({
          type: "Feature",
          geometry: {
            type: "Polygon",
            coordinates: [[[bxW, bxS], [bxE, bxS], [bxE, bxN], [bxW, bxN], [bxW, bxS]]],
          },
          properties: { kind: "fetched_bbox" },
        })),
      };
    }
    this._flushBbox();
  }

  clearRoute() {
    this.selectedObsIdx = null;
    this.routeFeatures = [];
    this._flushRoute();
  }

  clearObservations() {
    this.obsFeatures = [];
    this.obsByIdx.clear();
    this.colocatedGroups.clear();
    this.obsClusterKey.clear();
    this.selectedObsIdx = null;
    this.routeFeatures = [];
    this._pendingBounds = null;
    this._flushObs();
    this._flushRoute();
    this._renderLegend({});
  }

  _flushBbox() {
    if (!this.ready) return;
    const src = this.map.getSource("fetched-bbox-src");
    if (src) {
      src.setData(this.fetchedBboxGeojson || { type: "FeatureCollection", features: [] });
    }
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

  setObservationsFromDataset(columns, latCol, lonCol, rowIdCol, ySpec, fitMap = true, bbox = null) {
    const lats = columns[latCol] || [];
    const lons = columns[lonCol] || [];
    const ids = rowIdCol && columns[rowIdCol] ? columns[rowIdCol] : null;
    const nums = ySpec.num_col && columns[ySpec.num_col] ? columns[ySpec.num_col] : null;
    const dens = ySpec.mode === "ratio" && ySpec.den_col && columns[ySpec.den_col] ? columns[ySpec.den_col] : null;

    const feats = [];
    this.obsByIdx.clear();
    this.colocatedGroups.clear();
    this.obsClusterKey.clear();
    let minLon = 180, maxLon = -180, minLat = 90, maxLat = -90;
    const validLons = [];
    for (let i = 0; i < lats.length; i++) {
      const la = parseNumericScalar(lats[i]);
      const lo = parseNumericScalar(lons[i]);
      if (!Number.isFinite(la) || !Number.isFinite(lo) || la < -90 || la > 90 || lo < -180 || lo > 180 || (Math.abs(la) < 1e-12 && Math.abs(lo) < 1e-12)) continue;
      let yv = nums ? parseNumericScalar(nums[i]) : NaN;
      if (dens && Number.isFinite(yv)) {
        const dv = parseNumericScalar(dens[i]);
        yv = dv > 0 ? yv / dv : NaN;
      }
      const feat = {
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
      };
      feats.push(feat);
      this.obsByIdx.set(i, feat);
      const ck = `${la.toFixed(6)},${lo.toFixed(6)}`;
      this.obsClusterKey.set(i, ck);
      if (!this.colocatedGroups.has(ck)) this.colocatedGroups.set(ck, []);
      this.colocatedGroups.get(ck).push(i);
      validLons.push(lo);
      minLon = Math.min(minLon, lo);
      maxLon = Math.max(maxLon, lo);
      minLat = Math.min(minLat, la);
      maxLat = Math.max(maxLat, la);
    }

    this.selectedObsIdx = null;
    this.routeFeatures = [];
    this._flushRoute();
    this.obsFeatures = feats;
    this.restyleObservations({});

    if (fitMap && feats.length > 0) {
      let bounds = [[minLon, minLat], [maxLon, maxLat]];
      if (bbox && Number.isFinite(bbox.west) && Number.isFinite(bbox.east) && bbox.west > bbox.east) {
        // Antimeridian dataset: compute unwrapped positive span from circular lon0
        let sx = 0, sy = 0;
        for (const lo of validLons) {
          const rad = (lo * Math.PI) / 180;
          sx += Math.sin(rad);
          sy += Math.cos(rad);
        }
        const lon0 = (Math.atan2(sx, sy) * 180) / Math.PI;
        let dMin = Infinity, dMax = -Infinity;
        for (const lo of validLons) {
          const d = ((((lo - lon0 + 180) % 360) + 360) % 360) - 180;
          if (d < dMin) dMin = d;
          if (d > dMax) dMax = d;
        }
        bounds = [[lon0 + dMin, minLat], [lon0 + dMax, maxLat]];
      }
      if (this.ready) {
        this._applyFitBounds(bounds);
      } else {
        this._pendingBounds = bounds;
      }
    }
  }

  focusObservation(obsIdx) {
    const feat = this.obsByIdx.get(Number(obsIdx));
    if (!feat) return false;
    this.selectedObsIdx = Number(obsIdx);
    const [lon, lat] = feat.geometry.coordinates;
    if (this.map && this.ready) {
      this.map.easeTo({
        center: [lon, lat],
        zoom: Math.max(this.map.getZoom(), 15),
        duration: 400,
      });
    }
    if (this.callbacks.onClickObservation) {
      this.callbacks.onClickObservation(this.selectedObsIdx);
    }
    return true;
  }

  getColocatedNeighbors(obsIdx) {
    const ck = this.obsClusterKey.get(Number(obsIdx));
    if (!ck) return [Number(obsIdx)];
    return this.colocatedGroups.get(ck) || [Number(obsIdx)];
  }

  restyleObservations({ stage1, stage3, stage4, stage5, layerMode, gwrColIdx, gwrMaskMode, activeModelId, routeTargetId }) {
    if (layerMode) this.layerMode = layerMode;
    if (gwrColIdx !== undefined) this.gwrColIdx = gwrColIdx;
    if (gwrMaskMode) this.gwrMaskMode = gwrMaskMode;
    if (activeModelId !== undefined) this.activeModelId = activeModelId;
    if (routeTargetId !== undefined) this.activeRouteTargetId = routeTargetId;
    this._lastRestyleArgs = { stage1, stage3, stage4, stage5 };

    let activeOlsModel = null;
    const olsMap = new Map();
    if (stage3 && stage3.models && stage3.models.length) {
      activeOlsModel =
        (this.activeModelId && stage3.models.find((x) => x.model_id === this.activeModelId && x.resid)) ||
        stage3.models.find((x) => x.resid) ||
        stage3.models[0];
      if (activeOlsModel && activeOlsModel.valid_indices && activeOlsModel.resid) {
        activeOlsModel.valid_indices.forEach((origI, pos) => {
          olsMap.set(origI, { resid: activeOlsModel.resid[pos], cooks: activeOlsModel.cooks_d ? activeOlsModel.cooks_d[pos] : 0 });
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
    let gwrVarName = "";
    if (stage5 && stage5.ok && stage5.valid_indices) {
      const ci = Math.min(this.gwrColIdx, (stage5.col_names || []).length - 1);
      gwrVarName = stage5.col_names?.[ci] || "";
      stage5.valid_indices.forEach((origI, pos) => {
        const fdrSig = stage5.fdr_sig_mask?.[pos]?.[ci];
        gwrMap.set(origI, {
          beta: stage5.betas_native[pos][ci],
          t: stage5.t_vals[pos][ci],
          r2: stage5.local_r2[pos],
          fdrSig: fdrSig !== undefined ? Boolean(fdrSig) : Math.abs(stage5.t_vals[pos][ci]) >= (stage5.t_crit_nom || 1.96),
        });
      });
    }

    const snapByTarget = stage1?.snap_status_by_target || {};
    const snapArr =
      (this.activeRouteTargetId && snapByTarget[this.activeRouteTargetId]) ||
      Object.values(snapByTarget)[0] ||
      null;

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
    const tCritNom = stage5?.t_crit_nom || 1.96;
    const snapCounts = { 0: 0, 1: 0, 2: 0, 3: 0 };

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
        const isSig = g ? (this.gwrMaskMode === "fdr" ? g.fdrSig : Math.abs(g.t) >= tCritNom) : true;
        if (g && !isSig) f.properties.opacity = 0.28;
      } else if (this.layerMode === "gwr_r2") {
        const g = gwrMap.get(idx);
        f.properties.color = g ? this._colorScale(g.r2, minV, maxV, false) : "#cbd5e1";
      } else if (this.layerMode === "snap_status") {
        const st = snapArr && idx < snapArr.length ? Number(snapArr[idx]) : 3;
        f.properties.color = SNAP_COLORS[st] || "#94a3b8";
        const bucket = st === 0 || st === 1 || st === 2 ? st : 3;
        snapCounts[bucket] = (snapCounts[bucket] || 0) + 1;
      }
    }

    this._flushObs();
    this._renderLegend({ minV, maxV, activeOlsModel, stage1, stage4, stage5, gwrVarName, snapCounts });
  }

  _renderLegend({ minV, maxV, activeOlsModel, stage1, stage4, stage5, gwrVarName, snapCounts }) {
    const el = document.getElementById(this.legendId);
    if (!el) return;
    if (!this.obsFeatures.length) {
      el.style.display = "none";
      return;
    }
    el.style.display = "block";
    const fmt = (v) => (Number.isFinite(v) ? (Math.abs(v) >= 1000 ? v.toLocaleString("hu-HU", { maximumFractionDigits: 0 }) : v.toFixed(3)) : "—");

    if (this.layerMode === "snap_status") {
      el.innerHTML = `
        <div><strong>Hálózati illesztés (Snap &amp; BBox)</strong></div>
        ${
          stage1?.ok
            ? `<div class="legend-swatches">
                 <span class="legend-swatch"><span class="legend-dot" style="background:${SNAP_COLORS[0]};"></span>OK (${snapCounts?.[0] ?? 0})</span>
                 <span class="legend-swatch"><span class="legend-dot" style="background:${SNAP_COLORS[2]};"></span>Snap &gt; limit (${snapCounts?.[2] ?? 0})</span>
                 <span class="legend-swatch"><span class="legend-dot" style="background:${SNAP_COLORS[1]};"></span>Elszigetelt (${snapCounts?.[1] ?? 0})</span>
               </div>`
            : `<div class="card-subtitle">Futtasd az 1. fázist a hálózati snap diagnosztikához.</div>`
        }
      `;
      return;
    }

    if (this.layerMode === "lisa") {
      const cnt = stage4?.ok ? stage4.lisa_resid?.counts || {} : null;
      const items = [
        ["HH", "Magas–Magas", LISA_COLORS.HH],
        ["LL", "Alacsony–Alacsony", LISA_COLORS.LL],
        ["HL", "Magas–Alacsony", LISA_COLORS.HL],
        ["LH", "Alacsony–Magas", LISA_COLORS.LH],
        ["NS", "Nem szign.", LISA_COLORS.NS],
      ];
      el.innerHTML = `
        <div><strong>LISA Reziduális Klaszterek</strong>${stage4?.ok ? (stage4.lisa_resid?.fdr_applied ? " (BH-FDR)" : " (p&lt;0.05)") : ""}</div>
        ${
          cnt
            ? `<div class="legend-swatches">${items
                .map(
                  ([k, , col]) =>
                    `<span class="legend-swatch"><span class="legend-dot" style="background:${col};"></span>${k} (${cnt[k] ?? 0})</span>`
                )
                .join("")}</div>`
            : `<div class="card-subtitle">Futtasd a 4. fázist a LISA klaszterekhez.</div>`
        }
      `;
      return;
    }

    if (this.layerMode === "ols_resid") {
      if (!activeOlsModel) {
        el.innerHTML = `<div><strong>OLS Reziduumok</strong></div><div class="card-subtitle">Futtasd a 3. fázist.</div>`;
        return;
      }
      const lim = Math.max(Math.abs(minV), Math.abs(maxV), 1e-6);
      el.innerHTML = `
        <div><strong>Reziduum (${escapeHtml(activeOlsModel.model_name || activeOlsModel.model_id)})</strong></div>
        <div class="legend-ramp" style="background:linear-gradient(90deg, rgb(50,130,235), rgb(230,230,230), rgb(220,80,60));"></div>
        <div class="legend-ticks"><span>${fmt(-lim)}</span><span>0</span><span>+${fmt(lim)}</span></div>
      `;
      return;
    }

    if (this.layerMode === "gwr_beta") {
      if (!stage5?.ok) {
        el.innerHTML = `<div><strong>GWR Lokális β̂ᵢ</strong></div><div class="card-subtitle">Futtasd az 5. fázist.</div>`;
        return;
      }
      const lim = Math.max(Math.abs(minV), Math.abs(maxV), 1e-6);
      const maskNote = this.gwrMaskMode === "fdr" ? "Halvány: BY-FDR p_adj ≥ 0.05" : `Halvány: |t| < ${stage5.t_crit_nom || 1.96}`;
      el.innerHTML = `
        <div><strong>GWR β̂(${escapeHtml(gwrVarName)})</strong></div>
        <div class="legend-ramp" style="background:linear-gradient(90deg, rgb(50,130,235), rgb(230,230,230), rgb(220,80,60));"></div>
        <div class="legend-ticks"><span>${fmt(-lim)}</span><span>0</span><span>+${fmt(lim)}</span></div>
        <div class="card-subtitle" style="margin-top:2px;">${escapeHtml(maskNote)}</div>
      `;
      return;
    }

    const title = this.layerMode === "gwr_r2" ? "GWR Lokális R²" : "Megfigyelt célváltozó (y)";
    el.innerHTML = `
      <div><strong>${title}</strong></div>
      <div class="legend-ramp" style="background:linear-gradient(90deg, rgb(37,99,235), rgb(132,149,145), rgb(227,99,55));"></div>
      <div class="legend-ticks"><span>${fmt(minV)}</span><span>${fmt(0.5 * (minV + maxV))}</span><span>${fmt(maxV)}</span></div>
    `;
  }

  _flushObs() {
    if (!this.ready) return;
    const src = this.map.getSource("obs-src");
    if (src) src.setData({ type: "FeatureCollection", features: this.obsFeatures });
  }

  _flushRoute() {
    if (!this.ready) return;
    const src = this.map.getSource("route-src");
    if (src) src.setData({ type: "FeatureCollection", features: this.routeFeatures });
  }

  _flushSources() {
    this._flushBbox();
    this._flushTargets();
    this._flushObs();
    this._flushRoute();
  }

  showRoute(routeRes) {
    const feats = [];
    if (routeRes && routeRes.euclidean_geojson) feats.push(routeRes.euclidean_geojson);
    if (routeRes && routeRes.network_geojson) feats.push(routeRes.network_geojson);
    this.routeFeatures = feats;
    this._flushRoute();
  }

  renderPointInspector(obsIdx, info) {
    const el = document.getElementById(this.inspectorId);
    if (!el) return;
    if (obsIdx === null || !info) {
      el.innerHTML = `<div class="card-subtitle">Kattints bármely megfigyelési pontra a térképen (vagy Stage 3-ban a reziduális ábrán) a pontos távolságok, Dijkstra sétaútvonal, co-located csoport, reziduum, LISA és GWR adatokhoz.</div>`;
      return;
    }

    const route = info.route;
    let detourTxt = `<span class="card-subtitle">Nincs aktív célpont útvonal-visszajátszáshoz</span>`;
    if (route && route.ok && route.inside_zero) {
      detourTxt = `<span class="mono">Belső zóna (d<sub>eucl</sub>=0 m | d<sub>net</sub>=0 m | 0 perc)</span>`;
    } else if (route && route.ok && route.dist_net_m !== null) {
      const timeStr = route.travel_time_min != null ? ` | ~${escapeHtml(route.travel_time_min)} perc` : "";
      detourTxt = `<span class="mono">d<sub>eucl</sub>=${escapeHtml(route.dist_eucl_m)} m | d<sub>net</sub>=${escapeHtml(route.dist_net_m)} m | Δd=${escapeHtml(route.detour_m)} m${timeStr}</span>`;
    } else if (route && route.ok) {
      const reasonMap = {
        SNAP_EXCEEDED: "távolabb a hálózattól mint max_snap_m",
        SNAP_DISCONNECTED: "szigetkomponens (nem elérhető célpont)",
        SNAP_NO_GRAPH: "Dijkstra utcahálózat nincs betöltve",
        SNAP_TARGET_EMPTY: "minden célpont elem kizárva",
      };
      const reasonLbl = reasonMap[route.snap_reason] || translateVerdict(route.snap_reason || "Dijkstra nem elérhető");
      detourTxt = `<span class="mono">d<sub>eucl</sub>=${escapeHtml(route.dist_eucl_m)} m (${escapeHtml(reasonLbl)})</span>`;
    }

    const group = this.getColocatedNeighbors(obsIdx);
    const posInGroup = Math.max(0, group.indexOf(Number(obsIdx)));
    const clusterBar =
      group.length > 1
        ? `<div style="display:flex;align-items:center;justify-content:space-between;background:var(--surface-alt);border:1px solid var(--border);border-radius:6px;padding:4px 8px;margin-bottom:6px;font-size:11px;">
             <span>🏢 Épületben / azonos ponton <strong>${group.length}</strong> megfigyelés — <strong>[${posInGroup + 1}/${group.length}]</strong></span>
             <span style="display:flex;gap:4px;">
               <button type="button" class="btn btn-sm" id="btnInspPrevCluster" aria-label="Előző co-located megfigyelés">◀</button>
               <button type="button" class="btn btn-sm" id="btnInspNextCluster" aria-label="Következő co-located megfigyelés">▶</button>
             </span>
           </div>`
        : "";

    const covLine =
      info.covSummary && info.covSummary.length
        ? `<div class="card-subtitle mono" style="margin-top:4px;font-size:10.5px;">${info.covSummary.map((kv) => `${escapeHtml(kv.key)}=${escapeHtml(kv.val)}`).join(" | ")}</div>`
        : "";

    el.innerHTML = `
      ${clusterBar}
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
        <strong>📍 Megfigyelés #${escapeHtml(info.row_id)} <span class="card-subtitle">(sor ${escapeHtml(obsIdx)})</span></strong>
        <span class="mono">y = ${info.y_raw !== null && Number.isFinite(Number(info.y_raw)) ? Number(info.y_raw).toLocaleString("hu-HU", { maximumFractionDigits: 2 }) : "—"}</span>
      </div>
      <div style="margin-bottom:5px;">${detourTxt}</div>
      <div class="grid-3" style="font-size:11px;">
        <div><strong>OLS reziduum${info.modelLabel ? ` (${escapeHtml(info.modelLabel)})` : ""}:</strong> <span class="mono">${info.resid !== null ? escapeHtml(info.resid) : "—"}</span></div>
        <div><strong>LISA klaszter:</strong> <span class="mono">${info.lisa ? escapeHtml(info.lisa) : "—"}</span></div>
        <div><strong>GWR β̂/t/R²:</strong> <span class="mono">${info.gwr_beta !== null ? `${escapeHtml(info.gwr_beta)} (t=${escapeHtml(info.gwr_t)}, R²=${escapeHtml(info.gwr_r2 ?? "—")}, κ=${escapeHtml(info.gwr_kappa ?? "—")})` : "—"}</span></div>
      </div>
      ${covLine}
    `;

    if (group.length > 1 && this.callbacks.onClickObservation) {
      el.querySelector("#btnInspPrevCluster")?.addEventListener("click", () => {
        const nextIdx = group[(posInGroup - 1 + group.length) % group.length];
        this.focusObservation(nextIdx);
      });
      el.querySelector("#btnInspNextCluster")?.addEventListener("click", () => {
        const nextIdx = group[(posInGroup + 1) % group.length];
        this.focusObservation(nextIdx);
      });
    }
  }
}
