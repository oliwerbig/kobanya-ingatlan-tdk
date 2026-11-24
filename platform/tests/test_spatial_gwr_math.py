"""Unit tests for Spatial Econometrics (W, Moran, LISA, Anselin, ML SAR/SEM/SDM, Krinsky-Robb) & Guard #6 GWR."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_engine.gwr_kernel import fit_adaptive_gwr
from py_engine.spatial_kernel import (
    _fit_ml_sar_core,
    _log_det_i_minus_rho_w,
    build_spatial_weight_matrix,
    compute_lesage_pace_impacts,
    run_spatial_pipeline,
)


def test_spatial_ml_sar_sdm_and_krinsky_robb_impacts() -> None:
    rng = np.random.default_rng(2026)
    n = 150
    coords_xy = rng.uniform(0.0, 2000.0, size=(n, 2))
    # Simulate co-located condo pairs on first 20 rows
    coords_xy[:10] = coords_xy[0]

    w_knn5 = build_spatial_weight_matrix(coords_xy, {"id": "knn5", "type": "knn", "k": 5, "dedup_colocated": False})
    w_idw = build_spatial_weight_matrix(coords_xy, {"id": "idw_auto", "type": "idw", "power": 1.0, "cutoff_mode": "auto_connectivity"})
    assert w_knn5["colocated_pairs_pct"] > 0.0
    assert len(w_idw["islands"]) == 0
    assert w_idw["is_symmetric_base"] is True

    # Verify Ord eigenvalue log-det matches exact dense log-det
    W_dense = w_idw["W"].toarray()
    ld_ord = _log_det_i_minus_rho_w(0.35, w_idw)
    _, ld_exact = np.linalg.slogdet(np.eye(n) - 0.35 * W_dense)
    assert abs(ld_ord - float(ld_exact)) < 1e-6

    # Generate true SAR data: y = (I - 0.45 W)^{-1} (2 + 0.5 x1 + eps)
    x1 = rng.normal(0.0, 1.0, size=n)
    X = np.column_stack([np.ones(n), x1])
    eps = rng.normal(0.0, 0.25, size=n)
    y = np.linalg.solve(np.eye(n) - 0.45 * w_knn5["W"].toarray(), 2.0 + 0.5 * x1 + eps)

    res = run_spatial_pipeline(
        X, y, ["const", "x1"], coords_xy,
        [
            {"id": "knn5", "type": "knn", "k": 5, "dedup_colocated": False},
            {"id": "idw_auto", "type": "idw", "power": 1.0, "cutoff_mode": "auto_connectivity"},
        ],
        krinsky_robb_draws=400,
    )
    assert res["ok"] is True
    sar_row = next(r for r in res["model_comparison"] if r["model"].startswith("SAR"))
    # Estimated rho close to 0.45 and significant
    assert 0.25 < sar_row["spatial_param"] < 0.65
    assert sar_row["spatial_p"] < 0.01

    # Direct + Indirect == Total exact identity
    imp = res["impacts_sar"][0]
    assert abs((imp["direct"]["est"] + imp["indirect"]["est"]) - imp["total"]["est"]) < 1e-3
    assert imp["spillover_share_pct"] > 15.0
    assert imp["direct"]["se"] > 0.0 and imp["indirect"]["se"] > 0.0

    # Verify Tier 1 (eigvals) vs Tier 2 (splu) exact equivalence
    sar_t1 = _fit_ml_sar_core(X, y, w_idw)
    w_idw_no_eig = {**w_idw, "eigvals": None}
    sar_t2 = _fit_ml_sar_core(X, y, w_idw_no_eig)
    assert abs(sar_t1["rho"] - sar_t2["rho"]) < 1e-4
    assert abs(sar_t1["llf"] - sar_t2["llf"]) < 1e-3

    imp_t2 = compute_lesage_pace_impacts(sar_t2, w_idw_no_eig, ["const", "x1"], krinsky_robb_draws=200)
    assert abs(imp_t2[0]["total"]["est"] - (imp_t2[0]["direct"]["est"] + imp_t2[0]["indirect"]["est"])) < 1e-3


def test_guard6_singularity_guarded_gwr_native_recovery() -> None:
    rng = np.random.default_rng(314)
    n = 160
    coords_xy = rng.uniform(0.0, 2000.0, size=(n, 2))
    # Force a big condo cluster of 30 points at (200, 200) all sharing identical dummy value=1!
    coords_xy[:30] = np.array([200.0, 200.0])
    condo_dummy = np.zeros(n)
    condo_dummy[:30] = 1.0
    condo_dummy[30:55] = 1.0  # total 55/160 > 10% prevalence so kept in GWR and triggers local collinearity!

    x_cont = rng.normal(50.0, 10.0, size=n)
    # Spatially varying slope beta_1(x) ranging from -0.02 in west to +0.06 in east
    beta_true = -0.02 + 0.08 * (coords_xy[:, 0] / 2000.0)
    y = 10.0 + beta_true * x_cont + 0.3 * condo_dummy + rng.normal(0.0, 0.15, size=n)

    X = np.column_stack([np.ones(n), x_cont, condo_dummy])
    col_meta = [
        {"name": "const", "label": "Intercept", "is_binary": False},
        {"name": "x_cont", "label": "x_cont", "is_binary": False},
        {"name": "condo_dummy", "label": "condo_dummy", "is_binary": True},
    ]

    gwr = fit_adaptive_gwr(
        X, y, col_meta, coords_xy,
        gwr_cfg={"jitter_m": 1.0, "b_min_floor": 15, "kappa_thresh": 500.0, "prevalence_min_pct": 0.10},
    )
    assert gwr["ok"] is True
    assert gwr["gwr_r2"] > gwr["global_ols_r2"]
    assert (gwr["n_regularized"] + gwr.get("n_local_dropped_locs", 0)) > 0  # Selective Ridge or local drop caught identical-dummy cluster!

    betas_x = np.array([row[1] for row in gwr["betas_native"]])
    assert np.all(np.isfinite(betas_x))
    # Native-scale recovered slopes correlate strongly with planted spatially varying slope!
    assert np.corrcoef(betas_x, beta_true)[0, 1] > 0.75
    # Non-stationarity test detects true spatial slope gradient
    x_summary = next(s for s in gwr["param_summaries"] if s["name"] == "x_cont")
    assert x_summary["non_stationary"] is True


def test_geospec_session_end_to_end_stages_1_to_5_and_export() -> None:
    import io
    import zipfile
    from py_engine.engine import GeoSpecSession
    from py_engine.projection import LocalMetricProjection

    rng = np.random.default_rng(99)
    n = 90
    proj = LocalMetricProjection(47.49, 19.06, 0.05)
    xy = rng.uniform(-600.0, 600.0, size=(n, 2))
    lats, lons = proj.inverse(xy[:, 0], xy[:, 1])

    # Park polygon at center (-80..80, -80..80)
    p_xy = np.array([[-80, -80], [80, -80], [80, 80], [-80, 80], [-80, -80]], dtype=float)
    p_lats, p_lons = proj.inverse(p_xy[:, 0], p_xy[:, 1])
    park_gj = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "id": "central_park",
            "properties": {"name": "Central Park"},
            "geometry": {"type": "Polygon", "coordinates": [[[float(lo), float(la)] for lo, la in zip(p_lons, p_lats)]]},
        }],
    }

    area = rng.uniform(35.0, 95.0, size=n)
    d_approx = np.maximum(1.0, np.hypot(xy[:, 0], xy[:, 1]) - 60.0)
    price = np.exp(10.0 + 0.008 * area + 0.15 * np.maximum(0.0, np.log(350.0 / d_approx)) + rng.normal(0, 0.1, size=n)) * area

    columns = {
        "id": list(range(1, n + 1)),
        "lat": [float(v) for v in lats],
        "lon": [float(v) for v in lons],
        "price": [float(v) for v in price],
        "area_m2": [float(v) for v in area],
        "cond": ["good" if i % 2 == 0 else "renovated" for i in range(n)],
    }

    sess = GeoSpecSession()
    load_res = sess.load_dataset(columns, "lat", "lon")
    assert load_res["n_geocoded"] == n

    prev = sess.ingest_target_raw("park", park_gj, source_kind="geojson")
    assert prev["n_features"] == 1

    recipe = {
        "schema_version": "1.0.0",
        "recipe_meta": {"title": "E2E Test"},
        "dataset_schema": {
            "row_id_col": "id",
            "lat_col": "lat",
            "lon_col": "lon",
            "y_spec": {"mode": "ratio", "num_col": "price", "den_col": "area_m2", "transform": "log"},
            "cluster_spec": {"mode": "auto_or_column", "radius_m": 25.0},
            "selection_spec": {"enabled": False},
            "spatial_proxy_cols": [],
        },
        "spatial_targets": [{
            "id": "park",
            "name": "Parks",
            "source": "geojson",
            "excluded_osm_ids": [],
            "geometry_handling": "polygon_boundary",
            "inside_is_zero": True,
            "distance_modes": "euclidean",
        }],
        "models": sess.generate_m1_m4(
            [{"col": "area_m2", "type": "numeric", "transform": "raw"}, {"col": "cond", "type": "categorical", "reference_level": "good"}],
            [{"id": "park", "distance_modes": "euclidean"}],
            default_cutoff_m=350.0,
        ),
        "composition_chain": {
            "enabled": True,
            "focal_term_ref": "park::andersson_day",
            "blocks": [{"name": "Area", "terms": ["area_m2"]}, {"name": "Condition", "terms": ["cond"]}],
            "compute_gelbach_ovb": True,
        },
        "spatial_engine": {
            "active_model_id": "m4_ad",
            "weights": [{"id": "knn5", "type": "knn", "k": 5}, {"id": "idw_auto", "type": "idw", "power": 1.0, "cutoff_mode": "auto_connectivity"}],
            "krinsky_robb_draws": 200,
            "lisa_fdr": True,
            "gwr": {"enabled": True, "focal_model_id": "m4_ad", "jitter_m": 1.0, "b_min_floor": 15, "kappa_thresh": 1000.0},
        },
        "guards_cfg": {"common_sample_lock": True, "sparse_min_cell": 5},
    }

    s1 = sess.run_stage_1_distances(recipe, run_id=1)
    assert s1["ok"] is True
    s2 = sess.run_stage_2_descriptives_and_composition(recipe, run_id=2)
    assert s2["ok"] is True and s2["composition"]["ok"] is True
    s3 = sess.run_stage_3_ols_workbench(recipe, cov_type="HC3", run_id=3)
    assert s3["ok"] is True and len(s3["models"]) == 4
    s4 = sess.run_stage_4_spatial_econometrics(recipe, run_id=4)
    assert s4["ok"] is True and len(s4["model_comparison"]) == 4
    s5 = sess.run_stage_5_gwr(recipe, run_id=5)
    assert s5["ok"] is True and s5["gwr_r2"] > 0.0

    bundle = sess.export_publication_bundle(recipe)
    assert bundle["ok"] is True and "\\toprule" in bundle["latex_main"]
    assert len(bundle["geojson"]["features"]) == n
    with zipfile.ZipFile(io.BytesIO(bundle["zip_bytes"]), "r") as zf:
        names = zf.namelist()
        assert "experiment_recipe.json" in names
        assert "reproducibility_manifest.json" in names
        assert "spatial/enriched_observations.geojson" in names
        assert "latex/regression_comparison.tex" in names
