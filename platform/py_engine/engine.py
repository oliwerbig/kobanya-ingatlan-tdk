"""
GeoSpec Unified Session & Pipeline Orchestrator
=============================================
Connects DAGCache, Projection, Parametric OSM/GeoJSON GIS, Design Matrix,
OLS/Gelbach/Heckman Kernel, Spatial Econometrics Kernel, GWR Kernel, and Exporters
into a clean JSON-RPC session callable from Pyodide Web Worker or native Python.
"""

from __future__ import annotations

import datetime
import hashlib
import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .dag_cache import DAGCache, hash_payload
from .design_matrix import (
    _boxcox_profile_mle,
    build_dependent_variable,
    build_model_design,
    compute_common_sample_mask,
    generate_m1_m4_specs,
    parse_binary_flag,
    parse_numeric_array,
)
from .exporters import (
    build_qgis_geojson,
    build_zip_bundle,
    dict_rows_to_csv,
    generate_gwr_summary_latex,
    generate_multi_model_latex,
    generate_qgis_qml_gwr,
    generate_qgis_qml_lisa,
    generate_robustness_matrix_latex,
    generate_spatial_comparison_latex,
    generate_spatial_params_latex,
)
from .gwr_kernel import fit_adaptive_gwr
from .ols_kernel import (
    build_cluster_ids,
    compute_descriptive_and_normality,
    compute_semi_log_effects,
    fit_ols_core,
    run_guard1_diagnostics,
    run_heckman_two_step,
    run_stepwise_and_gelbach,
    run_wild_cluster_bootstrap,
    scan_andersson_day_profile,
)
from .projection import LocalMetricProjection, diagnose_coordinate_issues


def _array_sha256(arr: np.ndarray) -> str:
    """Fast collision-free SHA-256 hash of any NumPy array's exact values."""
    arr_c = np.ascontiguousarray(arr, dtype=np.float64)
    return hashlib.sha256(arr_c.tobytes()).hexdigest()
from .spatial_gis import (
    build_street_graph,
    compute_pairwise_network_distances,
    compute_single_route,
    compute_target_distances,
    features_to_preview_geojson,
    parse_geojson_features,
    parse_osm_elements,
)
from .spatial_kernel import run_spatial_pipeline


def _sig_stars(p: float) -> str:
    if not math.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


class GeoSpecSession:
    """Stateful session managing incremental DAG computation for 1 study area."""

    def __init__(self) -> None:
        self.cache = DAGCache()
        self.columns: Dict[str, List[Any]] = {}
        self.n_rows: int = 0
        self.lat: np.ndarray = np.array([], dtype=np.float64)
        self.lon: np.ndarray = np.array([], dtype=np.float64)
        self.valid_coord_mask: np.ndarray = np.array([], dtype=bool)
        self.proj: Optional[LocalMetricProjection] = None
        self.obs_xy: np.ndarray = np.empty((0, 2), dtype=np.float64)

        self.target_features: Dict[str, List[Dict[str, Any]]] = {}
        self.street_graphs: Dict[str, Dict[str, Any]] = {}
        self.dist_results: Dict[str, Dict[str, Any]] = {}
        self.last_outputs: Dict[str, Any] = {}
        self._stage_hashes: Dict[int, str] = {}

    def _invalidate_outputs_from(self, stage: int) -> None:
        """Clear last_outputs for this stage and all downstream outputs (keeping DAGCache intact for upstream hits)."""
        if stage == 1:
            for k in ("stage1", "stage2", "stage3", "stage4", "stage5"):
                self.last_outputs.pop(k, None)
        elif stage == 2:
            self.last_outputs.pop("stage2", None)
        elif stage == 3:
            for k in ("stage2", "stage3", "stage4", "stage5"):
                self.last_outputs.pop(k, None)
        elif stage == 4:
            self.last_outputs.pop("stage4", None)
        elif stage == 5:
            self.last_outputs.pop("stage5", None)

    def _invalidate_from_stage(self, stage: int, clear_cache: bool = True) -> None:
        """Canonical DAG invalidation helper across last_outputs, _stage_hashes, and DAGCache layers."""
        self._invalidate_outputs_from(stage)
        if not clear_cache:
            return
        if stage == 1:
            for s in (1, 2, 3, 4, 5):
                self._stage_hashes.pop(s, None)
            self.cache.invalidate_from("ols")
        elif stage == 3:
            for s in (2, 3, 4, 5):
                self._stage_hashes.pop(s, None)
            self.cache.invalidate_from("ols")
        elif stage == 4:
            self._stage_hashes.pop(4, None)
            self.cache._store.get("spatial", {}).clear()
        elif stage == 5:
            self._stage_hashes.pop(5, None)
            self.cache._store.get("gwr", {}).clear()

    def reset_session(self) -> Dict[str, Any]:
        """Reset all dataset, GIS targets, graphs, caches, and stage outputs cleanly."""
        self.cache.clear_all()
        self.columns = {}
        self.n_rows = 0
        self.lat = np.array([], dtype=np.float64)
        self.lon = np.array([], dtype=np.float64)
        self.valid_coord_mask = np.array([], dtype=bool)
        self.proj = None
        self.obs_xy = np.empty((0, 2), dtype=np.float64)
        self.target_features.clear()
        self.street_graphs.clear()
        self.dist_results.clear()
        self.last_outputs.clear()
        self._stage_hashes.clear()
        return {"ok": True}

    def remove_target(self, target_id: str) -> Dict[str, Any]:
        """Remove a spatial target completely and invalidate all downstream outputs/caches."""
        self.target_features.pop(target_id, None)
        self.dist_results.pop(target_id, None)
        self.cache.invalidate_from("dist")
        self._invalidate_from_stage(1, clear_cache=True)
        return {"ok": True, "removed_target_id": target_id}

    def load_dataset(
        self,
        columns: Dict[str, List[Any]],
        lat_col: str,
        lon_col: str,
    ) -> Dict[str, Any]:
        """Load tabular dataset columns and initialize universal metric projection."""
        self.reset_session()

        if not columns or lat_col not in columns or lon_col not in columns:
            raise ValueError(f"Coordinate columns '{lat_col}' and '{lon_col}' are required.")

        self.columns = columns
        self.n_rows = len(columns[lat_col])
        self.lat = parse_numeric_array(columns[lat_col])
        self.lon = parse_numeric_array(columns[lon_col])
        coord_diag = diagnose_coordinate_issues(self.lat, self.lon)
        self.valid_coord_mask = (
            np.isfinite(self.lat)
            & np.isfinite(self.lon)
            & (self.lat >= -90.0)
            & (self.lat <= 90.0)
            & (self.lon >= -180.0)
            & (self.lon <= 180.0)
            & ~((np.abs(self.lat) < 1e-12) & (np.abs(self.lon) < 1e-12))
        )

        n_geo = int(np.sum(self.valid_coord_mask))
        if n_geo < 5:
            self.reset_session()
            hint_txt = (" " + " ".join(coord_diag["hints"])) if coord_diag["hints"] else ""
            raise ValueError(f"At least 5 valid geographic coordinates required (found {n_geo}).{hint_txt}")

        self.proj = LocalMetricProjection.from_points(
            self.lat[self.valid_coord_mask],
            self.lon[self.valid_coord_mask],
        )
        self.obs_xy = np.full((self.n_rows, 2), np.nan, dtype=np.float64)
        xv, yv = self.proj.forward(self.lat[self.valid_coord_mask], self.lon[self.valid_coord_mask])
        self.obs_xy[self.valid_coord_mask, 0] = xv
        self.obs_xy[self.valid_coord_mask, 1] = yv

        # Antimeridian-safe circular bounding box for Postpass/Overpass
        lat_v = self.lat[self.valid_coord_mask]
        lon_v = self.lon[self.valid_coord_mask]
        lon0 = float(self.proj.lon0_deg)
        dlon = ((lon_v - lon0 + 180.0) % 360.0) - 180.0
        west_wrap = ((lon0 + float(np.min(dlon)) + 180.0) % 360.0) - 180.0
        east_wrap = ((lon0 + float(np.max(dlon)) + 180.0) % 360.0) - 180.0
        bbox = {
            "south": float(np.min(lat_v)),
            "west": float(west_wrap),
            "north": float(np.max(lat_v)),
            "east": float(east_wrap),
        }

        proj_meta = self.proj.metadata()
        all_warnings = list(coord_diag["hints"]) + list(proj_meta.get("warnings", []))

        return {
            "ok": True,
            "n_rows": self.n_rows,
            "n_geocoded": n_geo,
            "n_non_geocoded": self.n_rows - n_geo,
            "projection": proj_meta,
            "coord_diagnostics": coord_diag,
            "warnings": all_warnings,
            "bbox": bbox,
            "columns": list(columns.keys()),
        }

    def ingest_target_raw(
        self,
        target_id: str,
        raw_payload: Dict[str, Any],
        source_kind: str = "osm",
        excluded_osm_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Parse Overpass JSON or GeoJSON for a target and return interactive map preview GeoJSON."""
        if source_kind == "geojson":
            feats = parse_geojson_features(raw_payload, prefix=target_id)
        else:
            feats = parse_osm_elements(raw_payload)
        self.target_features[target_id] = feats
        self.cache.invalidate_from("dist")
        self._invalidate_from_stage(1, clear_cache=True)
        preview = features_to_preview_geojson(feats, excluded_ids=excluded_osm_ids)
        return {
            "ok": True,
            "target_id": target_id,
            "n_features": len(feats),
            "preview_geojson": preview,
            "feature_index": [
                {"uid": f["uid"], "name": f["name"], "geom_type": f["geom_type"]}
                for f in feats
            ],
        }

    def ingest_street_graph(self, profile: str, overpass_graph_json: Dict[str, Any]) -> Dict[str, Any]:
        """Parse street network JSON once into low-RAM CSR Dijkstra structure."""
        if self.proj is None:
            raise ValueError("Load dataset before ingesting street graph.")
        graph = build_street_graph(overpass_graph_json, self.proj, routing_profile=profile)
        self.street_graphs[profile] = graph
        self.cache.invalidate_from("dist")
        self._invalidate_from_stage(1, clear_cache=True)
        return {
            "ok": not graph.get("empty", True),
            "profile": profile,
            "n_nodes": graph.get("n_nodes", 0),
            "n_edges": graph.get("n_edges", 0),
        }

    def _hydrate_csv_column_target(self, tgt: Dict[str, Any]) -> Dict[str, Any]:
        """Build a standardized distance dict for a pre-computed CSV column target."""
        col_n = tgt.get("csv_col")
        arr = parse_numeric_array(self.columns[col_n]) if col_n in self.columns else np.full(self.n_rows, np.nan)
        return {
            "dist_eucl": arr,
            "dist_net": np.full(self.n_rows, np.nan),
            "travel_time_min": np.full(self.n_rows, np.nan),
            "circuity_delta": np.full(self.n_rows, np.nan),
            "circuity_ratio": np.full(self.n_rows, np.nan),
            "snap_status": np.zeros(self.n_rows, dtype=np.int32),
            "n_active_features": 1,
            "n_excluded_features": 0,
            "warnings": [],
        }

    def run_stage_1_distances(self, recipe: Dict[str, Any], run_id: int = 1) -> Dict[str, Any]:
        """Stage 1: Compute Euclidean/Dijkstra distances & circuity for all configured spatial targets."""
        self._invalidate_outputs_from(1)
        if self.proj is None:
            raise ValueError("No dataset loaded.")
        self.cache.start_run(run_id)
        targets = recipe.get("spatial_targets", [])
        active_ids = {t.get("id") for t in targets if t.get("id")}
        self.dist_results = {k: v for k, v in self.dist_results.items() if k in active_ids}
        self.target_features = {k: v for k, v in self.target_features.items() if k in active_ids}

        summary = []
        warnings: List[str] = []

        for tgt in targets:
            self.cache.check_cancelled(run_id)
            tid = tgt["id"]
            if tgt.get("source") == "csv_column":
                d_res = self._hydrate_csv_column_target(tgt)
            else:
                feats = self.target_features.get(tid, [])
                prof = tgt.get("routing_profile", "walk")
                graph = self.street_graphs.get(prof) if tgt.get("distance_modes") in ("network_dijkstra", "both") else None
                feats_sig = hash_payload([
                    (
                        f.get("uid"),
                        f.get("geom_type"),
                        f.get("rings", []),
                        f.get("inner_rings", []),
                    )
                    for f in feats
                ])
                d_key = hash_payload({
                    "tid": tid,
                    "n_feats": len(feats),
                    "feats_sig": feats_sig,
                    "excl": sorted(tgt.get("excluded_osm_ids", []) or []),
                    "geom": tgt.get("geometry_handling"),
                    "inside_is_zero": tgt.get("inside_is_zero", True),
                    "mode": tgt.get("distance_modes"),
                    "prof": prof,
                    "snap_m": tgt.get("max_snap_distance_m", 300.0),
                    "graph_n": graph.get("n_nodes", 0) if graph else 0,
                })
                d_res = self.cache.get_or_compute(
                    "dist",
                    d_key,
                    lambda f=feats, t=tgt, g=graph: compute_target_distances(
                        self.obs_xy, self.valid_coord_mask, f, t, self.proj, g  # type: ignore[arg-type]
                    ),
                    run_id=run_id,
                )

            self.dist_results[tid] = d_res
            warnings.extend(d_res.get("warnings", []))
            eucl_v = d_res["dist_eucl"][self.valid_coord_mask & np.isfinite(d_res["dist_eucl"])]
            net_v = d_res["dist_net"][self.valid_coord_mask & np.isfinite(d_res["dist_net"])]
            tt_arr = d_res.get("travel_time_min")
            time_v = (
                tt_arr[self.valid_coord_mask & np.isfinite(tt_arr)]
                if tt_arr is not None
                else np.empty(0, dtype=np.float64)
            )
            det_v = d_res["circuity_delta"][self.valid_coord_mask & np.isfinite(d_res["circuity_delta"])]
            st_arr = np.asarray(d_res.get("snap_status", np.zeros(self.n_rows, dtype=np.int32)))[self.valid_coord_mask]
            has_net = tgt.get("distance_modes") in ("network_dijkstra", "both")
            summary.append({
                "target_id": tid,
                "name": tgt.get("name", tid),
                "n_active_features": d_res["n_active_features"],
                "n_excluded_features": d_res["n_excluded_features"],
                "median_eucl_m": round(float(np.median(eucl_v)), 1) if len(eucl_v) else None,
                "median_net_m": round(float(np.median(net_v)), 1) if len(net_v) else None,
                "median_time_min": round(float(np.median(time_v)), 2) if len(time_v) else None,
                "median_detour_m": round(float(np.median(det_v)), 1) if len(det_v) else None,
                "max_detour_m": round(float(np.max(det_v)), 1) if len(det_v) else None,
                "n_snap_ok": int(np.sum(st_arr == 0)) if has_net else None,
                "n_snap_disconnected": int(np.sum(st_arr == 1)) if has_net else None,
                "n_snap_exceeded": int(np.sum(st_arr == 2)) if has_net else None,
            })

        stage1_hash = hash_payload({
            tid: _array_sha256(np.where(np.isfinite(d["dist_eucl"]), d["dist_eucl"], -1.0))
            + ":" + _array_sha256(np.where(np.isfinite(d["dist_net"]), d["dist_net"], -1.0))
            for tid, d in sorted(self.dist_results.items())
        })
        if self._stage_hashes.get(1) != stage1_hash:
            self.cache.invalidate_from("ols")
            for s in (2, 3, 4, 5):
                self._stage_hashes.pop(s, None)
        self._stage_hashes[1] = stage1_hash

        snap_status_by_target = {
            tid: [int(x) for x in np.asarray(d_res.get("snap_status", []), dtype=np.int32)]
            for tid, d_res in self.dist_results.items()
            if d_res.get("snap_status") is not None
        }
        self.last_outputs["stage1"] = {"summary": summary, "warnings": warnings}
        return {
            "ok": True,
            "summary": summary,
            "warnings": warnings,
            "snap_status_by_target": snap_status_by_target,
        }

    def get_single_point_route(
        self,
        obs_idx: int,
        target_id: str,
        recipe: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute on-demand Dijkstra route replay for clicked observation on the map."""
        if self.proj is None or obs_idx < 0 or obs_idx >= self.n_rows or not self.valid_coord_mask[obs_idx]:
            return {"ok": False, "reason": "invalid_observation", "snap_reason": "INVALID_OBSERVATION"}
        tgt = next((t for t in recipe.get("spatial_targets", []) if t["id"] == target_id), None)
        if tgt is None:
            return {"ok": False, "reason": "unknown_target", "snap_reason": "UNKNOWN_TARGET"}
        feats = self.target_features.get(target_id, [])
        graph = self.street_graphs.get(tgt.get("routing_profile", "walk"))
        return compute_single_route(
            (float(self.obs_xy[obs_idx, 0]), float(self.obs_xy[obs_idx, 1])),
            feats,
            tgt,
            self.proj,
            graph,
        )

    def _prepare_y_and_masks(self, recipe: Dict[str, Any]) -> Dict[str, Any]:
        for tgt in recipe.get("spatial_targets", []):
            tid = tgt.get("id")
            if tid and tgt.get("source") == "csv_column" and tid not in self.dist_results:
                self.dist_results[tid] = self._hydrate_csv_column_target(tgt)
        ds = recipe.get("dataset_schema", {})
        sel_cfg = ds.get("selection_spec", {})
        excl_col = sel_cfg.get("excluded_flag_col")
        if excl_col and excl_col in self.columns:
            explicit_excl = parse_binary_flag(self.columns[excl_col]) >= 0.5
        else:
            explicit_excl = np.zeros(self.n_rows, dtype=bool)
        analytic_base = self.valid_coord_mask & ~explicit_excl

        active_specs = [m for m in recipe.get("models", []) if m.get("active", True)]
        missing_stage1_targets = sorted({
            st.get("target_id")
            for m in active_specs
            for st in m.get("spatial_terms", [])
            if st.get("target_id") and st.get("target_id") not in self.dist_results
        })

        y_spec = ds.get("y_spec", {})
        x_cond_mat: Optional[np.ndarray] = None
        cond_mask: Optional[np.ndarray] = None
        if y_spec.get("transform") == "boxcox" and active_specs:
            guards_cfg = recipe.get("guards_cfg", {})
            tau = int(guards_cfg.get("sparse_min_cell", 5))
            sp_mid = recipe.get("spatial_engine", {}).get("active_model_id")
            ref_spec = next((m for m in active_specs if m.get("id") == sp_mid), active_specs[-1])
            raw_y_spec = {**y_spec, "transform": "raw"}
            y_probe = build_dependent_variable(self.columns, raw_y_spec)
            y_probe["valid_mask"] = y_probe["valid_mask"] & (y_probe["y_raw"] > 0.0)
            common_probe = (
                compute_common_sample_mask(self.columns, self.dist_results, active_specs, y_probe, analytic_base, tau)
                if guards_cfg.get("common_sample_lock", True)
                else None
            )
            ref_design = build_model_design(
                self.columns, self.dist_results, ref_spec, y_probe, analytic_base, tau, common_probe
            )
            if ref_design["n_obs"] >= max(10, ref_design["k_params"] + 5) and ref_design["k_params"] >= 1:
                x_cond_mat = np.zeros((self.n_rows, ref_design["k_params"]), dtype=np.float64)
                x_cond_mat[ref_design["valid_mask"]] = ref_design["X"]
                cond_mask = ref_design["valid_mask"]

        y_info = build_dependent_variable(self.columns, y_spec, X_cond=x_cond_mat, cond_mask=cond_mask)
        return {
            "y_info": y_info,
            "analytic_base": analytic_base,
            "explicit_excl": explicit_excl,
            "missing_stage1_targets": missing_stage1_targets,
        }

    def run_stage_2_descriptives_and_composition(self, recipe: Dict[str, Any], run_id: int = 1) -> Dict[str, Any]:
        """Stage 2: Descriptive stats, sample-drop waterfall, normality, and Step 0..K + Gelbach OVB decomposition."""
        self._invalidate_outputs_from(2)
        if self.proj is None:
            raise ValueError("No dataset loaded.")
        self.cache.start_run(run_id)
        prep = self._prepare_y_and_masks(recipe)
        y_info = prep["y_info"]
        analytic_base = prep["analytic_base"]
        ds = recipe.get("dataset_schema", {})
        guards_cfg = recipe.get("guards_cfg", {})
        tau = int(guards_cfg.get("sparse_min_cell", 5))
        cov_type = str(recipe.get("default_cov_type", "HC3"))
        cluster_spec = ds.get("cluster_spec", {})

        active_specs = [m for m in recipe.get("models", []) if m.get("active", True)]
        if active_specs and guards_cfg.get("common_sample_lock", True):
            common_mask = compute_common_sample_mask(
                self.columns, self.dist_results, active_specs, y_info, analytic_base, tau
            )
        else:
            common_mask = analytic_base & y_info["valid_mask"]

        cov_union = []
        seen_cov = set()
        for m in active_specs:
            for c in m.get("covariates", []):
                if c["col"] not in seen_cov:
                    seen_cov.add(c["col"])
                    cov_union.append(c)

        # Sample-drop waterfall & per-variable missingness audit on geocoded non-excluded rows
        y_valid_base = analytic_base & y_info["valid_mask"]
        n_y_base = int(np.sum(y_valid_base))
        sample_waterfall = {
            "n_total": int(self.n_rows),
            "n_geocoded": int(np.sum(self.valid_coord_mask)),
            "dropped_non_geocoded": int(self.n_rows - np.sum(self.valid_coord_mask)),
            "n_after_excl_flag": int(np.sum(analytic_base)),
            "dropped_excl_flag": int(np.sum(self.valid_coord_mask) - np.sum(analytic_base)),
            "n_after_y_valid": n_y_base,
            "dropped_y_missing_or_nonpos": int(np.sum(analytic_base) - n_y_base),
            "n_common_analytic": int(np.sum(common_mask)),
            "dropped_covs_or_dist": int(n_y_base - np.sum(common_mask)),
        }
        per_variable_missing: List[Dict[str, Any]] = []
        if n_y_base > 0:
            for c in cov_union:
                cname = c.get("col", "")
                if cname not in self.columns:
                    per_variable_missing.append({"variable": cname, "n_missing": n_y_base, "pct_missing": 100.0})
                    continue
                if c.get("type", "numeric") == "categorical":
                    raw_c = self.columns[cname]
                    miss_cnt = sum(
                        1 for idx in np.where(y_valid_base)[0]
                        if raw_c[idx] is None or str(raw_c[idx]).strip() == ""
                    )
                else:
                    arr_c = parse_numeric_array(self.columns[cname])[y_valid_base]
                    miss_cnt = int(np.sum(~np.isfinite(arr_c)))
                per_variable_missing.append({
                    "variable": cname,
                    "n_missing": int(miss_cnt),
                    "pct_missing": round(miss_cnt / n_y_base * 100.0, 2),
                })
            for tid, tres in self.dist_results.items():
                eucl_miss = int(np.sum(~np.isfinite(tres["dist_eucl"][y_valid_base])))
                if eucl_miss > 0:
                    per_variable_missing.append({
                        "variable": f"dist_eucl_{tid}",
                        "n_missing": eucl_miss,
                        "pct_missing": round(eucl_miss / n_y_base * 100.0, 2),
                    })
                if np.any(np.isfinite(tres["dist_net"])):
                    net_miss = int(np.sum(~np.isfinite(tres["dist_net"][y_valid_base])))
                    if net_miss > 0:
                        per_variable_missing.append({
                            "variable": f"dist_net_{tid}",
                            "n_missing": net_miss,
                            "pct_missing": round(net_miss / n_y_base * 100.0, 2),
                        })

        desc = compute_descriptive_and_normality(self.columns, self.dist_results, y_info, common_mask, cov_union)

        chain_cfg = recipe.get("composition_chain", {})
        stepwise = {"ok": False, "reason": "Disabled or no active model."}
        if chain_cfg.get("enabled", True) and active_specs:
            sp_mid = recipe.get("spatial_engine", {}).get("active_model_id")
            ref_spec = next((m for m in active_specs if m["id"] == sp_mid), active_specs[-1])
            stepwise = run_stepwise_and_gelbach(
                self.columns,
                self.dist_results,
                ref_spec,
                chain_cfg,
                y_info,
                analytic_base,
                tau,
                common_mask,
                cov_type=cov_type,
                cluster_spec=cluster_spec,
                coords_xy=self.obs_xy,
            )

        warnings: List[str] = []
        if prep.get("missing_stage1_targets"):
            warnings.append(
                f"Hiányzó Stage 1 távolságeredmények a recept célpontjaihoz: {', '.join(prep['missing_stage1_targets'])}. Futtasd előbb Stage 1-et."
            )

        res = {
            "ok": True,
            "descriptives": desc,
            "composition": stepwise,
            "sample_waterfall": sample_waterfall,
            "per_variable_missing": per_variable_missing,
            "warnings": warnings,
        }
        self.last_outputs["stage2"] = res
        return res

    def run_stage_3_ols_workbench(
        self,
        recipe: Dict[str, Any],
        cov_type: str = "HC3",
        run_id: int = 1,
    ) -> Dict[str, Any]:
        """
        Stage 3: Estimate all active composed ModelSpecs (with Guard #1, #2, #4 + WCB + Causal DiD/RDD),
        plus Andersson-Day Profile-MLE cutoff curve when enabled.
        """
        self._invalidate_outputs_from(3)
        if self.proj is None:
            raise ValueError("No dataset loaded.")
        self.cache.start_run(run_id)
        prep = self._prepare_y_and_masks(recipe)
        y_info = prep["y_info"]
        analytic_base = prep["analytic_base"]
        ds = recipe.get("dataset_schema", {})
        y_spec = ds.get("y_spec", {})
        cluster_spec = ds.get("cluster_spec", {})
        guards_cfg = recipe.get("guards_cfg", {})
        tau = int(guards_cfg.get("sparse_min_cell", 5))
        lev_warn = float(guards_cfg.get("hc3_leverage_warn", 0.85))
        active_specs = [m for m in recipe.get("models", []) if m.get("active", True)]

        if not active_specs:
            return {"ok": False, "reason": "No active model specifications."}

        common_mask = None
        if guards_cfg.get("common_sample_lock", True):
            common_mask = compute_common_sample_mask(
                self.columns, self.dist_results, active_specs, y_info, analytic_base, tau
            )

        y_sig = _array_sha256(np.where(np.isfinite(y_info["y"]), y_info["y"], -999999.0)) + ":" + str(y_info.get("transform", "raw"))
        mask_arr = common_mask if common_mask is not None else analytic_base
        mask_sig = _array_sha256(mask_arr.astype(np.float64))
        dist_sig = hash_payload({
            tid: _array_sha256(np.where(np.isfinite(d["dist_eucl"]), d["dist_eucl"], -1.0))
            + ":" + _array_sha256(np.where(np.isfinite(d["dist_net"]), d["dist_net"], -1.0))
            for tid, d in sorted(self.dist_results.items())
        })
        is_boxcox_mle = y_info.get("transform") == "boxcox" and y_spec.get("boxcox_lambda") is None

        stage3_hash = hash_payload({"y": y_sig, "mask": mask_sig, "dist": dist_sig, "specs": active_specs, "cov": cov_type, "cluster": cluster_spec})
        if self._stage_hashes.get(3) != stage3_hash:
            self.cache.invalidate_from("spatial")
            for s in (4, 5):
                self._stage_hashes.pop(s, None)

        # Build spatial proxy vectors for Guard #1 confounding alert
        proxy_cols_raw: Dict[str, np.ndarray] = {}
        for pcol in ds.get("spatial_proxy_cols", []) or []:
            if pcol in self.columns:
                proxy_cols_raw[pcol] = parse_numeric_array(self.columns[pcol])

        models_out: List[Dict[str, Any]] = []
        ad_profiles: Dict[str, Any] = {}
        heckman_out: Optional[Dict[str, Any]] = None

        for m_spec in active_specs:
            self.cache.check_cancelled(run_id)
            m_key = hash_payload({
                "kind": "ols_model",
                "spec": m_spec,
                "cov_type": cov_type,
                "cluster": cluster_spec,
                "proxies": ds.get("spatial_proxy_cols", []) or [],
                "y": y_sig,
                "mask": mask_sig,
                "dist": dist_sig,
                "tau": tau,
                "lev": lev_warn,
                "bc_mle": is_boxcox_mle,
            })

            def _compute_single_model(spec_cur: Dict[str, Any] = m_spec) -> Dict[str, Any]:
                design = build_model_design(
                    self.columns, self.dist_results, spec_cur, y_info, analytic_base, tau, common_mask
                )
                if design["n_obs"] <= design["k_params"] + 2:
                    return {
                        "model_id": spec_cur["id"],
                        "model_name": spec_cur.get("name", spec_cur["id"]),
                        "error": f"Insufficient observations (N={design['n_obs']}) for K={design['k_params']}.",
                    }

                cluster_ids = None
                cluster_meta = {"method": "none"}
                if cov_type == "cluster":
                    cluster_ids, cluster_meta = build_cluster_ids(
                        self.columns, self.obs_xy, design["valid_indices"], cluster_spec
                    )

                ols = fit_ols_core(
                    design["X"],
                    design["y"],
                    cov_type=cov_type,
                    cluster_ids=cluster_ids,
                    leverage_warn=lev_warn,
                )

                wcb_res: Optional[Dict[str, Any]] = None
                if cov_type == "cluster" and cluster_ids is not None and 2 <= cluster_meta.get("n_clusters", 0) < 30:
                    wcb_res = run_wild_cluster_bootstrap(design["X"], design["y"], cluster_ids, n_boot=999, seed=42)

                proxy_sub = {k: v[design["valid_mask"]] for k, v in proxy_cols_raw.items()}
                focal_dist = None
                if recipe.get("spatial_targets"):
                    ftgt = recipe["spatial_targets"][0]["id"]
                    if ftgt in self.dist_results:
                        focal_dist = (ftgt, self.dist_results[ftgt]["dist_eucl"][design["valid_mask"]])

                diag = run_guard1_diagnostics(design["X"], design["y"], ols, design["col_meta"], proxy_sub, focal_dist)

                params_rows = []
                wcb_pvals = wcb_res.get("p_values", []) if (wcb_res and wcb_res.get("triggered")) else []
                for j, meta in enumerate(design["col_meta"]):
                    b_j = float(ols["beta"][j])
                    se_j = float(ols["se"][j])
                    p_j = float(ols["p"][j])
                    pcts = compute_semi_log_effects(b_j, se_j, meta["is_binary"], design["is_log_scale"])
                    row_p: Dict[str, Any] = {
                        "name": meta["name"],
                        "label": meta["label"],
                        "kind": meta["kind"],
                        "is_binary": meta["is_binary"],
                        "beta": round(b_j, 4),
                        "se": round(se_j, 4),
                        "t": round(float(ols["t"][j]), 3),
                        "p": round(p_j, 5),
                        "sig": _sig_stars(p_j),
                        "ci_lo": round(float(ols["ci_lo"][j]), 4),
                        "ci_hi": round(float(ols["ci_hi"][j]), 4),
                        "hp_pct": pcts["hp_pct"],
                        "kennedy_pct": pcts["kennedy_pct"],
                    }
                    if j < len(wcb_pvals):
                        row_p["wcb_p"] = wcb_pvals[j]
                    params_rows.append(row_p)

                llf_y_scale = float(ols["llf"]) + float(design.get("jacobian_log_sum", 0.0))
                k_aic = int(ols.get("rank", design["k_params"])) + 1 + (1 if is_boxcox_mle else 0)
                aic_y_scale = 2.0 * k_aic - 2.0 * llf_y_scale

                return {
                    "model_id": spec_cur["id"],
                    "model_name": spec_cur.get("name", spec_cur["id"]),
                    "n_obs": design["n_obs"],
                    "k_params": design["k_params"],
                    "rank": int(ols.get("rank", design["k_params"])),
                    "is_log_scale": design["is_log_scale"],
                    "y_label": design["y_label"],
                    "y_transform": design.get("y_transform", "raw"),
                    "r2": round(float(ols["r2"]), 4),
                    "adj_r2": round(float(ols["adj_r2"]), 4),
                    "aic": round(float(ols["aic"]), 2),
                    "aicc": round(float(ols["aicc"]), 2),
                    "bic": round(float(ols["bic"]), 2),
                    "llf": round(float(ols["llf"]), 2),
                    "aic_y_scale": round(aic_y_scale, 2),
                    "llf_y_scale": round(llf_y_scale, 2),
                    "f_stat": round(float(ols["f_stat"]), 2) if math.isfinite(ols["f_stat"]) else None,
                    "f_pval": round(float(ols["f_pval"]), 5) if math.isfinite(ols["f_pval"]) else None,
                    "cov_type_used": ols["cov_type_used"],
                    "df_infer": ols.get("df_infer", ols["df_resid"]),
                    "cluster_meta": cluster_meta,
                    "wild_bootstrap": wcb_res,
                    "params": params_rows,
                    "diagnostics": diag,
                    "guard2_events": design["guard2_events"],
                    "footnotes": design["footnotes"],
                    "fitted": np.round(ols["fitted"], 5).tolist(),
                    "resid": np.round(ols["resid"], 5).tolist(),
                    "cooks_d": np.round(ols["cooks_d"], 5).tolist(),
                    "valid_indices": design["valid_indices"].tolist(),
                }

            model_entry = self.cache.get_or_compute("ols", m_key, _compute_single_model, run_id=run_id)
            models_out.append(model_entry)
            if "error" in model_entry:
                continue

            # Scan AD cutoff profile for any AD spatial term in this spec (cached independent of slider cutoff_m)
            sp_terms = m_spec.get("spatial_terms", [])
            ad_tids = [s.get("target_id") for s in sp_terms if s.get("transform") == "andersson_day"]
            for st_idx, st in enumerate(sp_terms):
                if st.get("transform") == "andersson_day":
                    rng_cfg = st.get("cutoff_range_m") or [100.0, 600.0, 25.0]
                    c_min_v = float(rng_cfg[0]) if len(rng_cfg) >= 1 else 100.0
                    c_max_v = float(rng_cfg[1]) if len(rng_cfg) >= 2 else max(c_min_v + 50.0, 600.0)
                    step_v = max(1.0, float(rng_cfg[2])) if len(rng_cfg) >= 3 else 25.0
                    if c_max_v < c_min_v:
                        c_max_v = c_min_v + step_v
                    rng_norm = [c_min_v, c_max_v, step_v]
                    spec_no_cutoff = {
                        **m_spec,
                        "spatial_terms": [
                            {k: v for k, v in s.items() if k != "cutoff_m"} if i == st_idx else s
                            for i, s in enumerate(sp_terms)
                        ],
                    }
                    ad_key = hash_payload({
                        "kind": "ad_profile",
                        "spec": spec_no_cutoff,
                        "st_idx": st_idx,
                        "rng": rng_norm,
                        "cov_type": cov_type,
                        "cluster": cluster_spec,
                        "y": y_sig,
                        "mask": mask_sig,
                        "dist": dist_sig,
                        "tau": tau,
                    })
                    prof_res = self.cache.get_or_compute(
                        "ols",
                        ad_key,
                        lambda sp=m_spec, idx=st_idx, c0=c_min_v, c1=c_max_v, stp=step_v: scan_andersson_day_profile(
                            self.columns,
                            self.dist_results,
                            sp,
                            y_info,
                            analytic_base,
                            target_term_idx=idx,
                            c_min=c0,
                            c_max=c1,
                            step=stp,
                            sparse_min_cell=tau,
                            override_valid_mask=common_mask,
                            cov_type=cov_type,
                            cluster_spec=cluster_spec,
                            coords_xy=self.obs_xy,
                        ),
                        run_id=run_id,
                    )
                    tid = st.get("target_id")
                    pkey = f"{m_spec['id']}::{tid}::{st.get('metric', 'eucl')}" if ad_tids.count(tid) > 1 else f"{m_spec['id']}::{tid}"
                    ad_profiles[pkey] = prof_res

        # Guard #4 Heckman 2-Step Selection Check & Cached Conditional Box-Cox on reference active model
        sel_cfg = ds.get("selection_spec", {})
        sp_mid = recipe.get("spatial_engine", {}).get("active_model_id")
        ref_spec = next((m for m in active_specs if m["id"] == sp_mid), active_specs[-1])
        ref_design = build_model_design(
            self.columns, self.dist_results, ref_spec, y_info, analytic_base, tau, common_mask
        )

        cond_boxcox: Optional[Dict[str, Any]] = None
        if ref_design["n_obs"] > ref_design["k_params"] + 5 and np.all(ref_design["y_raw"] > 0.0):
            cond_bc_key = hash_payload({
                "kind": "cond_boxcox",
                "mid": ref_spec["id"],
                "x_sha": _array_sha256(ref_design["X"]),
                "y_raw_sha": _array_sha256(ref_design["y_raw"]),
            })

            def _compute_cond_bc() -> Optional[Dict[str, Any]]:
                try:
                    lam_c, ll_c, p_log_c, p_lin_c = _boxcox_profile_mle(ref_design["y_raw"], ref_design["X"])
                    return {
                        "mode": "conditional_on_X",
                        "model_id": ref_spec["id"],
                        "lambda_mle": round(lam_c, 4),
                        "ll_mle": round(ll_c, 3),
                        "p_vs_log": round(p_log_c, 5),
                        "p_vs_linear": round(p_lin_c, 5),
                    }
                except Exception:
                    return None

            cond_boxcox = self.cache.get_or_compute("ols", cond_bc_key, _compute_cond_bc, run_id=run_id)

        sel_indicator = analytic_base.astype(np.float64)
        n_excl = int(np.sum(sel_indicator == 0.0))
        if (
            sel_cfg.get("enabled", True)
            and n_excl >= int(guards_cfg.get("heckman_min_excluded", 10))
            and ref_design["n_obs"] >= int(guards_cfg.get("heckman_min_analytic", 30))
        ):
            cov_cols = [c["col"] for c in ref_spec.get("covariates", [])]
            fs_covs = sel_cfg.get("first_stage_covariates") or cov_cols
            excl_cols = sel_cfg.get("exclusion_restriction_cols") or []
            heck_key = hash_payload({
                "kind": "heckman",
                "fs_covs": fs_covs,
                "excl": excl_cols,
                "ref_cols": [m["name"] for m in ref_design["col_meta"]],
                "y": y_sig,
                "mask": _array_sha256(ref_design["valid_mask"].astype(np.float64)),
                "excl_mask": _array_sha256(sel_indicator),
                "x_sha": _array_sha256(ref_design["X"]),
            })
            heckman_out = self.cache.get_or_compute(
                "ols",
                heck_key,
                lambda: run_heckman_two_step(
                    self.columns,
                    sel_indicator,
                    ref_design["valid_indices"],
                    ref_design["X"],
                    ref_design["y"],
                    ref_design["col_meta"],
                    fs_covs,
                    excl_cols,
                ),
                run_id=run_id,
            )
        else:
            heckman_out = {
                "activated": False,
                "reason": f"Complete or minimal exclusion sample (N_excluded={n_excl} < {guards_cfg.get('heckman_min_excluded', 10)}).",
            }

        self._stage_hashes[3] = stage3_hash
        warnings: List[str] = []
        if prep.get("missing_stage1_targets"):
            warnings.append(
                f"Hiányzó Stage 1 távolságeredmények: {', '.join(prep['missing_stage1_targets'])}. Futtasd előbb Stage 1-et."
            )

        res = {
            "ok": True,
            "common_sample_locked": bool(guards_cfg.get("common_sample_lock", True)),
            "models": models_out,
            "ad_profiles": ad_profiles,
            "heckman": heckman_out,
            "conditional_boxcox": cond_boxcox,
            "warnings": warnings,
        }
        self.last_outputs["stage3"] = res
        return res

    def run_stage_4_spatial_econometrics(self, recipe: Dict[str, Any], run_id: int = 1) -> Dict[str, Any]:
        """Stage 4: Multi-W Moran/LISA, Anselin/Elhorst, ML/QML SAR/SEM/SDM, W_net & Krinsky-Robb."""
        self._invalidate_outputs_from(4)
        if self.proj is None:
            raise ValueError("No dataset loaded.")
        self.cache.start_run(run_id)
        prep = self._prepare_y_and_masks(recipe)
        y_info = prep["y_info"]
        analytic_base = prep["analytic_base"]
        guards_cfg = recipe.get("guards_cfg", {})
        tau = int(guards_cfg.get("sparse_min_cell", 5))
        sp_cfg = recipe.get("spatial_engine", {})
        se_mode = str(sp_cfg.get("se_mode", "qml")).lower()
        active_specs = [m for m in recipe.get("models", []) if m.get("active", True)]
        if not active_specs:
            return {"ok": False, "reason": "No active model specification."}

        target_mid = sp_cfg.get("active_model_id")
        ref_spec = next((m for m in active_specs if m["id"] == target_mid), active_specs[-1])
        common_mask = None
        if guards_cfg.get("common_sample_lock", True):
            common_mask = compute_common_sample_mask(
                self.columns, self.dist_results, active_specs, y_info, analytic_base, tau
            )

        design = build_model_design(
            self.columns, self.dist_results, ref_spec, y_info, analytic_base, tau, common_mask
        )
        if design["n_obs"] <= design["k_params"] + 4:
            return {"ok": False, "reason": f"Too few observations (N={design['n_obs']}) for spatial ML estimation."}

        coords_v = self.obs_xy[design["valid_indices"]]
        w_configs = sp_cfg.get("weights") or [
            {"id": "knn5", "type": "knn", "k": 5, "dedup_colocated": False},
            {"id": "idw_auto", "type": "idw", "power": 1.0, "cutoff_mode": "auto_connectivity"},
        ]

        # Build street-network pairwise distances W_net for any network_idw matrices
        net_dist_mats: Dict[str, Any] = {}
        net_sigs: Dict[str, Any] = {}
        valid_sub = np.ones(len(coords_v), dtype=bool)
        for w_cfg in w_configs:
            if w_cfg.get("type") == "network_idw":
                prof = w_cfg.get("profile", "walk")
                sg_obj = self.street_graphs.get(prof) or next(
                    (g_v for g_v in self.street_graphs.values() if not g_v.get("empty", True)),
                    None,
                )
                wid = str(w_cfg.get("id", "net_idw"))
                net_sigs[wid] = (sg_obj.get("n_nodes", 0), sg_obj.get("n_edges", 0)) if sg_obj else None
                if sg_obj and not sg_obj.get("empty", True):
                    nd_m = compute_pairwise_network_distances(
                        coords_v,
                        valid_sub,
                        sg_obj,
                        cutoff_m=float(w_cfg.get("cutoff_m", 1500.0)),
                        max_snap_m=float(w_cfg.get("max_snap_distance_m", 300.0)),
                    )
                    if nd_m is not None:
                        net_dist_mats[wid] = nd_m

        col_names = [m["name"] for m in design["col_meta"]]
        sp_key = hash_payload({
            "mid": ref_spec["id"],
            "n": design["n_obs"],
            "k": design["k_params"],
            "cols": col_names,
            "w": w_configs,
            "net_sigs": net_sigs,
            "se_mode": se_mode,
            "kr": int(sp_cfg.get("krinsky_robb_draws", 1000)),
            "fdr": bool(sp_cfg.get("lisa_fdr", True)),
            "x_sha": _array_sha256(design["X"]),
            "y_sha": _array_sha256(design["y"]),
            "xy_sha": _array_sha256(coords_v),
        })

        cached_res = self.cache.get_or_compute(
            "spatial",
            sp_key,
            lambda: run_spatial_pipeline(
                design["X"],
                design["y"],
                col_names,
                coords_v,
                w_configs,
                krinsky_robb_draws=int(sp_cfg.get("krinsky_robb_draws", 1000)),
                lisa_fdr=bool(sp_cfg.get("lisa_fdr", True)),
                se_mode=se_mode,
                net_dist_mats=net_dist_mats or None,
            ),
            run_id=run_id,
        )
        res = dict(cached_res)
        res["model_id"] = ref_spec["id"]
        res["valid_indices"] = design["valid_indices"].tolist()
        self._stage_hashes[4] = sp_key
        self.last_outputs["stage4"] = res
        return res

    def run_stage_5_gwr(
        self,
        recipe: Dict[str, Any],
        run_id: int = 1,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """Stage 5: Cached Singularity-Guarded Adaptive Bisquare GWR / MGWR with native-scale local estimates."""
        self._invalidate_outputs_from(5)
        if self.proj is None:
            raise ValueError("No dataset loaded.")
        self.cache.start_run(run_id)
        prep = self._prepare_y_and_masks(recipe)
        y_info = prep["y_info"]
        analytic_base = prep["analytic_base"]
        guards_cfg = recipe.get("guards_cfg", {})
        tau = int(guards_cfg.get("sparse_min_cell", 5))
        sp_cfg = recipe.get("spatial_engine", {})
        gwr_cfg = sp_cfg.get("gwr", {})
        active_specs = [m for m in recipe.get("models", []) if m.get("active", True)]
        if not active_specs:
            return {"ok": False, "reason": "No active model specification."}

        focal_mid = gwr_cfg.get("focal_model_id") or sp_cfg.get("active_model_id")
        ref_spec = next((m for m in active_specs if m["id"] == focal_mid), active_specs[-1])
        common_mask = None
        if guards_cfg.get("common_sample_lock", True):
            common_mask = compute_common_sample_mask(
                self.columns, self.dist_results, active_specs, y_info, analytic_base, tau
            )

        design = build_model_design(
            self.columns, self.dist_results, ref_spec, y_info, analytic_base, tau, common_mask
        )
        if design["n_obs"] <= design["k_params"] + 4:
            return {"ok": False, "reason": f"Too few observations (N={design['n_obs']}) for Stage 5 GWR/MGWR."}
        coords_v = self.obs_xy[design["valid_indices"]]

        gwr_key = hash_payload({
            "kind": "gwr_fit",
            "mid": ref_spec["id"],
            "n": design["n_obs"],
            "k": design["k_params"],
            "cols": [m["name"] for m in design["col_meta"]],
            "gwr_cfg": gwr_cfg,
            "x_sha": _array_sha256(design["X"]),
            "y_sha": _array_sha256(design["y"]),
            "xy_sha": _array_sha256(coords_v),
        })
        cached_res = self.cache.get_or_compute(
            "gwr",
            gwr_key,
            lambda: fit_adaptive_gwr(
                design["X"],
                design["y"],
                design["col_meta"],
                coords_v,
                gwr_cfg=gwr_cfg,
                progress_cb=progress_cb,
            ),
            run_id=run_id,
        )
        res = dict(cached_res)
        if res.get("ok"):
            res["model_id"] = ref_spec["id"]
            res["valid_indices"] = design["valid_indices"].tolist()
            if progress_cb is not None:
                try:
                    progress_cb(100.0, "GWR/MGWR ready")
                except Exception:
                    pass
        self._stage_hashes[5] = gwr_key
        self.last_outputs["stage5"] = res
        return res

    def run_robustness_matrix(self, recipe: Dict[str, Any], run_id: int = 1) -> Dict[str, Any]:
        """Run 3 y-transforms (log, boxcox, raw) x 2 SE types (HC3, cluster) on identical N."""
        if self.proj is None:
            raise ValueError("No dataset loaded.")
        self.cache.start_run(run_id)
        for tgt in recipe.get("spatial_targets", []):
            tid = tgt.get("id")
            if tid and tgt.get("source") == "csv_column" and tid not in self.dist_results:
                self.dist_results[tid] = self._hydrate_csv_column_target(tgt)

        ds = recipe.get("dataset_schema", {})
        sel_cfg = ds.get("selection_spec", {})
        excl_col = sel_cfg.get("excluded_flag_col")
        if excl_col and excl_col in self.columns:
            explicit_excl = parse_binary_flag(self.columns[excl_col]) >= 0.5
        else:
            explicit_excl = np.zeros(self.n_rows, dtype=bool)
        analytic_base = self.valid_coord_mask & ~explicit_excl

        active_specs = [m for m in recipe.get("models", []) if m.get("active", True)]
        if not active_specs:
            return {"ok": False, "reason": "No active model specification."}

        guards_cfg = recipe.get("guards_cfg", {})
        tau = int(guards_cfg.get("sparse_min_cell", 5))
        lev_warn = float(guards_cfg.get("hc3_leverage_warn", 0.85))
        sp_mid = recipe.get("spatial_engine", {}).get("active_model_id")
        ref_spec = next((m for m in active_specs if m["id"] == sp_mid), active_specs[-1])

        base_y_spec = ds.get("y_spec", {})
        raw_pos_y = build_dependent_variable(self.columns, {**base_y_spec, "transform": "raw"})
        raw_pos_y["valid_mask"] = raw_pos_y["valid_mask"] & (raw_pos_y["y_raw"] > 0.0)
        if not np.any(raw_pos_y["valid_mask"]):
            return {"ok": False, "reason": "Nincs pozitív y érték a 3×2 robusztussági mátrixhoz."}

        if guards_cfg.get("common_sample_lock", True):
            common_mask = compute_common_sample_mask(
                self.columns, self.dist_results, active_specs, raw_pos_y, analytic_base, tau
            )
        else:
            ref_probe = build_model_design(self.columns, self.dist_results, ref_spec, raw_pos_y, analytic_base, tau)
            common_mask = ref_probe["valid_mask"]

        probe_design = build_model_design(
            self.columns, self.dist_results, ref_spec, raw_pos_y, analytic_base, tau, common_mask
        )
        if probe_design["n_obs"] <= probe_design["k_params"] + 2:
            return {"ok": False, "reason": f"Kevés megfigyelés (N={probe_design['n_obs']}) a mátrixhoz."}

        x_cond_mat = np.zeros((self.n_rows, probe_design["k_params"]), dtype=np.float64)
        x_cond_mat[probe_design["valid_mask"]] = probe_design["X"]
        cond_mask = probe_design["valid_mask"]

        focal_idx = next((i for i, m in enumerate(probe_design["col_meta"]) if m.get("kind") == "spatial"), None)
        if focal_idx is None:
            focal_idx = 1 if len(probe_design["col_meta"]) > 1 else 0
        focal_term = probe_design["col_meta"][focal_idx]["name"]

        cluster_spec = ds.get("cluster_spec", {})
        cluster_ids, _ = build_cluster_ids(self.columns, self.obs_xy, probe_design["valid_indices"], cluster_spec)

        rows: List[Dict[str, Any]] = []
        for y_t in ("log", "boxcox", "raw"):
            spec_t = {**base_y_spec, "transform": y_t}
            if y_t == "boxcox":
                spec_t["boxcox_lambda"] = None
            y_inf = build_dependent_variable(self.columns, spec_t, X_cond=x_cond_mat, cond_mask=cond_mask)
            design = build_model_design(
                self.columns, self.dist_results, ref_spec, y_inf, analytic_base, tau, common_mask
            )
            is_bc = y_t == "boxcox"
            for se_t in ("HC3", "cluster"):
                ols = fit_ols_core(
                    design["X"],
                    design["y"],
                    cov_type=se_t,
                    cluster_ids=cluster_ids if se_t == "cluster" else None,
                    leverage_warn=lev_warn,
                )
                diag = run_guard1_diagnostics(design["X"], design["y"], ols, design["col_meta"])
                llf_y = float(ols["llf"]) + float(design.get("jacobian_log_sum", 0.0))
                k_aic = int(ols.get("rank", design["k_params"])) + 1 + (1 if is_bc else 0)
                aic_y = 2.0 * k_aic - 2.0 * llf_y
                f_i = focal_idx if focal_idx < len(ols["beta"]) else 0
                rows.append({
                    "y_transform": y_t,
                    "y_label": y_inf["label"],
                    "cov_type_used": ols["cov_type_used"],
                    "focal_beta": round(float(ols["beta"][f_i]), 4),
                    "focal_se": round(float(ols["se"][f_i]), 4),
                    "focal_p": round(float(ols["p"][f_i]), 5),
                    "r2": round(float(ols["r2"]), 4),
                    "aic_y_scale": round(aic_y, 2),
                    "koenker_p": diag.get("koenker_p"),
                    "n_obs": design["n_obs"],
                })

        rob_res: Dict[str, Any] = {
            "ok": True,
            "model_id": ref_spec["id"],
            "focal_term": focal_term,
            "n_obs": probe_design["n_obs"],
            "rows": rows,
        }
        rob_res["latex"] = generate_robustness_matrix_latex(rob_res)
        self.last_outputs["robustness"] = rob_res
        return rob_res

    def generate_m1_m4(
        self,
        covariates: List[Dict[str, Any]],
        spatial_targets: List[Dict[str, Any]],
        default_rings: Optional[List[float]] = None,
        default_cutoff_m: float = 300.0,
    ) -> List[Dict[str, Any]]:
        """Expose 1-click editable M1-M4 generator."""
        return generate_m1_m4_specs(covariates, spatial_targets, default_rings, default_cutoff_m)

    def export_latex_only(self, recipe: Dict[str, Any], kind: str = "ols_main") -> Dict[str, Any]:
        """Lightweight RPC returning only the requested LaTeX table string without building GeoJSON/ZIP."""
        s3 = self.last_outputs.get("stage3", {})
        s4 = self.last_outputs.get("stage4", {})
        s5 = self.last_outputs.get("stage5", {})
        rob = self.last_outputs.get("robustness", {})
        ols_models = [m for m in s3.get("models", []) if "params" in m]

        if kind in ("ols_main", "ols_pct"):
            prep = self._prepare_y_and_masks(recipe)
            y_info = prep["y_info"]
            tex = generate_multi_model_latex(
                ols_models,
                show_pct_for_dummies=(kind == "ols_pct"),
                conditional_boxcox=s3.get("conditional_boxcox"),
                boxcox_info=y_info.get("boxcox_info"),
            ) if ols_models else "% No OLS models available."
        elif kind == "stage4":
            tex = generate_spatial_comparison_latex(s4)
        elif kind == "stage4_params":
            tex = generate_spatial_params_latex(s4)
        elif kind == "stage5":
            tex = generate_gwr_summary_latex(s5)
        elif kind == "robustness":
            tex = generate_robustness_matrix_latex(rob)
        else:
            raise ValueError(f"Unknown LaTeX table kind: {kind}")

        return {"ok": True, "kind": kind, "latex": tex}

    def export_geojson_only(self, recipe: Dict[str, Any]) -> Dict[str, Any]:
        """Lightweight RPC returning enriched QGIS GeoJSON without building CSV/LaTeX/ZIP."""
        prep = self._prepare_y_and_masks(recipe)
        y_info = prep["y_info"]
        ds = recipe.get("dataset_schema", {})
        id_col = ds.get("row_id_col")
        row_ids = self.columns[id_col] if (id_col and id_col in self.columns) else list(range(self.n_rows))

        s3 = self.last_outputs.get("stage3", {})
        s4 = self.last_outputs.get("stage4", {})
        s5 = self.last_outputs.get("stage5", {})
        ols_models = [m for m in s3.get("models", []) if "params" in m]
        sp_mid = recipe.get("spatial_engine", {}).get("active_model_id")
        active_ols = next((m for m in ols_models if m["model_id"] == sp_mid), ols_models[-1] if ols_models else None)
        analytic_indices = active_ols["valid_indices"] if active_ols else None

        geojson = build_qgis_geojson(
            [None if not math.isfinite(v) else float(v) for v in self.lat],
            [None if not math.isfinite(v) else float(v) for v in self.lon],
            row_ids,
            [None if not math.isfinite(v) else float(v) for v in y_info["y_raw"]],
            [None if not math.isfinite(v) else float(v) for v in y_info["y"]],
            self.dist_results,
            active_ols=active_ols,
            spatial_res=s4 if s4.get("ok") else None,
            gwr_res=s5 if s5.get("ok") else None,
            analytic_indices=analytic_indices,
        )
        return {"ok": True, "geojson": geojson}

    def export_publication_bundle(
        self,
        recipe: Dict[str, Any],
        osm_bundle: Optional[Dict[str, Any]] = None,
        svg_figures: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Generate LaTeX, CSV, QGIS GeoJSON, QML styles, SVG figures, Reproducibility Manifest, and ZIP bytes."""
        prep = self._prepare_y_and_masks(recipe)
        y_info = prep["y_info"]
        ds = recipe.get("dataset_schema", {})
        id_col = ds.get("row_id_col")
        row_ids = self.columns[id_col] if (id_col and id_col in self.columns) else list(range(self.n_rows))

        s3 = self.last_outputs.get("stage3", {})
        s4 = self.last_outputs.get("stage4", {})
        s5 = self.last_outputs.get("stage5", {})
        s2 = self.last_outputs.get("stage2", {})
        rob = self.last_outputs.get("robustness", {})

        ols_models = [m for m in s3.get("models", []) if "params" in m]
        sp_mid = recipe.get("spatial_engine", {}).get("active_model_id")
        active_ols = next((m for m in ols_models if m["model_id"] == sp_mid), ols_models[-1] if ols_models else None)

        analytic_indices = active_ols["valid_indices"] if active_ols else None
        geojson = build_qgis_geojson(
            [None if not math.isfinite(v) else float(v) for v in self.lat],
            [None if not math.isfinite(v) else float(v) for v in self.lon],
            row_ids,
            [None if not math.isfinite(v) else float(v) for v in y_info["y_raw"]],
            [None if not math.isfinite(v) else float(v) for v in y_info["y"]],
            self.dist_results,
            active_ols=active_ols,
            spatial_res=s4 if s4.get("ok") else None,
            gwr_res=s5 if s5.get("ok") else None,
            analytic_indices=analytic_indices,
        )

        tables: Dict[str, List[Dict[str, Any]]] = {}
        if s2.get("descriptives", {}).get("ok"):
            tables["descriptives"] = s2["descriptives"].get("descriptives", [])
            tables["normality"] = s2["descriptives"].get("normality", [])
        if s2.get("composition", {}).get("ok"):
            tables["stepwise_composition"] = s2["composition"].get("steps", [])
            tables["gelbach_ovb"] = s2["composition"].get("gelbach", [])
        if ols_models:
            reg_rows = []
            for m in ols_models:
                for p in m["params"]:
                    reg_rows.append({"model_id": m["model_id"], "model_name": m["model_name"], **p})
            tables["ols_coefficients"] = reg_rows
            tables["ols_diagnostics"] = [
                {
                    "model_id": m["model_id"],
                    "n_obs": m["n_obs"],
                    "r2": m["r2"],
                    "adj_r2": m["adj_r2"],
                    "aic": m["aic"],
                    "aic_y_scale": m.get("aic_y_scale"),
                    **{k: v for k, v in m["diagnostics"].items() if isinstance(v, (int, float, str, type(None)))},
                }
                for m in ols_models
            ]
        if s3.get("heckman", {}).get("activated"):
            tables["heckman_selection"] = s3["heckman"].get("comparison", [])
        if s4.get("ok"):
            tables["spatial_model_comparison"] = s4.get("model_comparison", [])
            tables["weights_robustness"] = s4.get("weights_robustness", [])
            sp_coef_rows: List[Dict[str, Any]] = []
            for mtype, pkey in [("SAR", "sar_params"), ("SEM", "sem_params"), ("SDM", "sdm_params")]:
                for pr in s4.get(pkey, []):
                    sp_coef_rows.append({"model": mtype, **pr})
            if sp_coef_rows:
                tables["spatial_coefficients"] = sp_coef_rows
            for imp_key, tbl_name in [("impacts_sar", "lesage_pace_sar"), ("impacts_sdm", "lesage_pace_sdm")]:
                if s4.get(imp_key):
                    tables[tbl_name] = [
                        {
                            "variable": r["variable"],
                            "direct": r["direct"]["est"], "direct_se": r["direct"]["se"], "direct_p": r["direct"]["p"],
                            "indirect": r["indirect"]["est"], "indirect_se": r["indirect"]["se"], "indirect_p": r["indirect"]["p"],
                            "total": r["total"]["est"], "total_se": r["total"]["se"], "total_p": r["total"]["p"],
                            "spillover_pct": r["spillover_share_pct"],
                        }
                        for r in s4.get(imp_key, [])
                    ]
        if rob.get("ok") and rob.get("rows"):
            tables["robustness_matrix"] = rob.get("rows", [])
        if s5.get("ok"):
            tables["gwr_parameter_summary"] = s5.get("param_summaries", [])

        latex_tables = {}
        if ols_models:
            latex_tables["regression_comparison"] = generate_multi_model_latex(
                ols_models,
                conditional_boxcox=s3.get("conditional_boxcox"),
                boxcox_info=y_info.get("boxcox_info"),
            )
            latex_tables["regression_comparison_pct"] = generate_multi_model_latex(
                ols_models,
                show_pct_for_dummies=True,
                conditional_boxcox=s3.get("conditional_boxcox"),
                boxcox_info=y_info.get("boxcox_info"),
            )
        if s4.get("ok"):
            latex_tables["spatial_models_comparison"] = generate_spatial_comparison_latex(s4)
            latex_tables["spatial_params"] = generate_spatial_params_latex(s4)
        if rob.get("ok"):
            latex_tables["robustness_matrix"] = rob.get("latex") or generate_robustness_matrix_latex(rob)
        if s5.get("ok"):
            latex_tables["gwr_summary"] = generate_gwr_summary_latex(s5)

        qml_styles: Dict[str, str] = {}
        if s4.get("ok"):
            qml_styles["style_lisa"] = generate_qgis_qml_lisa("lisa_resid_quad")
        if s5.get("ok"):
            qml_styles["style_gwr_r2"] = generate_qgis_qml_gwr("gwr_local_r2")
            gwr_cols = s5.get("col_names") or []
            focal_gwr = next((c for c in gwr_cols if "dist" in c.lower()), gwr_cols[1] if len(gwr_cols) > 1 else None)
            if focal_gwr:
                safe_c = "".join(ch if ch.isalnum() else "_" for ch in str(focal_gwr)).strip("_") or "col"
                qml_styles["style_gwr_beta"] = generate_qgis_qml_gwr(f"gwr_beta_{safe_c}")

        manifest = {
            "tool": "GeoSpec Parametric Spatial Econometric Workbench",
            "version": "1.2.0",
            "exported_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "recipe_hash": hash_payload(recipe),
            "stage_hashes": dict(self._stage_hashes),
            "n_rows_total": self.n_rows,
            "n_geocoded": int(np.sum(self.valid_coord_mask)),
            "projection": self.proj.metadata() if self.proj else {},
            "stage1_summary": self.last_outputs.get("stage1", {}).get("summary", []),
            "boxcox_info": y_info.get("boxcox_info"),
            "conditional_boxcox": s3.get("conditional_boxcox"),
            "missing_stage1_targets": prep.get("missing_stage1_targets", []),
        }

        if osm_bundle is None and self.target_features:
            tgt_cfg_map = {t.get("id"): dict(t) for t in recipe.get("spatial_targets", []) if t.get("id")}
            osm_bundle = {
                "schema": "geospec.osm_bundle.v1",
                "exported_utc": manifest["exported_utc"],
                "targets": {
                    tid: {
                        "target_cfg": tgt_cfg_map.get(tid, {"id": tid, "name": tid, "source": "geojson"}),
                        "source_kind": "geojson",
                        "payload": features_to_preview_geojson(feats),
                    }
                    for tid, feats in self.target_features.items()
                },
            }

        zip_bytes = build_zip_bundle(
            tables,
            latex_tables,
            geojson,
            recipe,
            manifest,
            osm_bundle=osm_bundle,
            qml_styles=qml_styles or None,
            svg_figures=svg_figures,
        )
        return {
            "ok": True,
            "latex_main": latex_tables.get("regression_comparison", ""),
            "latex_pct": latex_tables.get("regression_comparison_pct", ""),
            "geojson": geojson,
            "manifest": manifest,
            "zip_bytes": zip_bytes,
        }
