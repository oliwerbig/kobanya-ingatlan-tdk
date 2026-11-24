"""Comprehensive P0/P1/P2 Regression Tests for GeoSpec Pyodide Engine."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PLATFORM_DIR = Path(__file__).resolve().parents[1]
if str(PLATFORM_DIR) not in sys.path:
    sys.path.insert(0, str(PLATFORM_DIR))

from py_engine.design_matrix import _evaluate_spatial_term, build_dependent_variable, build_model_design
from py_engine.engine import GeoSpecSession
from py_engine.exporters import dict_rows_to_csv, generate_multi_model_latex
from py_engine.gwr_kernel import fit_adaptive_gwr
from py_engine.ols_kernel import fit_ols_core, run_guard1_diagnostics, run_heckman_two_step
from py_engine.projection import LocalMetricProjection, diagnose_coordinate_issues
from py_engine.spatial_gis import (
    _points_in_ring,
    build_street_graph,
    compute_single_route,
    compute_target_distances,
    features_to_preview_geojson,
    parse_geojson_features,
    parse_osm_elements,
)
from py_engine.spatial_kernel import (
    _fit_ml_sar_core,
    build_spatial_weight_matrix,
    compute_lesage_pace_impacts,
    compute_lisa,
)


def test_gis_duplicate_csr_edge_min_dedup_and_single_route():
    """Parallel duplicate OSM ways and multi-exit super-source edges must take min weight, never sum."""
    proj = LocalMetricProjection(47.5, 19.0)
    # 3 collinear nodes along lat=47.5 from lon=19.0 to lon=19.005 (~376m)
    nodes = [
        {"type": "node", "id": 1, "lat": 47.5, "lon": 19.0},
        {"type": "node", "id": 2, "lat": 47.5, "lon": 19.0025},
        {"type": "node", "id": 3, "lat": 47.5, "lon": 19.0050},
    ]
    # Include DUPLICATE ways covering the exact same segments (e.g. overlapping footpath/cycleway relations)
    overpass_graph = {
        "elements": nodes
        + [
            {"type": "way", "id": 100, "nodes": [1, 2, 3], "tags": {"highway": "residential"}},
            {"type": "way", "id": 101, "nodes": [1, 2, 3], "tags": {"highway": "footway"}},
            {"type": "way", "id": 102, "nodes": [1, 2, 3], "tags": {"highway": "path"}},
        ]
    }
    graph = build_street_graph(overpass_graph, proj, "walk")
    x1_arr, _ = proj.forward(47.5, 19.0)
    x3_arr, _ = proj.forward(47.5, 19.0050)
    x1, x3 = float(x1_arr), float(x3_arr)
    expected_len = abs(x3 - x1)

    g_fwd = graph["g_fwd"]
    # Shortest edge weight 0->1->2 must equal segment lengths (not 3x!)
    assert g_fwd[0, 1] + g_fwd[1, 2] == pytest.approx(expected_len, rel=1e-4)

    # Target sits near node 3 AND also create an exit node near node 2 with huge off-street snap vs small snap at node 3
    obs_xy = np.array([[x1, 0.0]])
    lat_n2, lon_n2 = 47.5, 19.0025
    x2_arr, _ = proj.forward(lat_n2, lon_n2)
    x2 = float(x2_arr)
    t_lat_far, t_lon_far = proj.inverse(x1, 220.0)
    t_lat_near, t_lon_near = proj.inverse(x2, 5.0)
    target_gj = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "exit_far", "properties": {"name": "Far Exit"}, "geometry": {"type": "Point", "coordinates": [float(t_lon_far), float(t_lat_far)]}},
            {"type": "Feature", "id": "exit_near", "properties": {"name": "Near Exit"}, "geometry": {"type": "Point", "coordinates": [float(t_lon_near), float(t_lat_near)]}},
        ],
    }
    feats = parse_geojson_features(target_gj)
    t_cfg = {
        "id": "tgt",
        "geometry_handling": "point",
        "inside_is_zero": False,
        "distance_modes": "both",
        "max_snap_distance_m": 300.0,
    }
    dists = compute_target_distances(obs_xy, np.array([True]), feats, t_cfg, proj, graph)
    route = compute_single_route((x1, 0.0), feats, t_cfg, proj, graph)

    expected_best = abs(x2 - x1) + 5.0
    assert dists["dist_net"][0] == pytest.approx(expected_best, abs=1.0)
    assert route["dist_net_m"] == pytest.approx(expected_best, abs=1.0)
    # Polyline must end at the exit coordinate (Near Exit), not stop at road node
    end_lon, end_lat = route["network_geojson"]["geometry"]["coordinates"][-1]
    ex_arr, ey_arr = proj.forward(end_lat, end_lon)
    assert math.hypot(float(ex_arr) - x2, float(ey_arr) - 5.0) < 0.5


def test_gis_multipolygon_island_in_hole_inside_is_zero_and_chunking():
    """Island-in-hole MultiPolygon geometry must apply even-odd winding identically to Eucl/Net/Circuity and preview holes."""
    proj = LocalMetricProjection(47.5, 19.0)
    # Build coordinates in meters for exact clarity:
    # Outer ring: [-500, 500]^2, Hole: [-300, 300]^2, Nested Island inside hole: [-100, 100]^2
    def rect_ll(xmin, ymin, xmax, ymax):
        pts = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
        return [[float(lon), float(lat)] for x, y in pts for lat, lon in [proj.inverse(x, y)]]

    outer_ll = rect_ll(-500, -500, 500, 500)
    hole_ll = rect_ll(-300, -300, 300, 300)
    island_ll = rect_ll(-100, -100, 100, 100)

    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "park_mp",
                "properties": {"name": "Donut With Island"},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[outer_ll, hole_ll], [island_ll]],
                },
            }
        ],
    }
    feats = parse_geojson_features(gj)
    preview = features_to_preview_geojson(feats)
    geom_prev = preview["features"][0]["geometry"]
    assert geom_prev["type"] == "MultiPolygon"
    assert len(geom_prev["coordinates"][0]) == 2  # outer + hole preserved!

    # Obs 0: inside outer ring (outside hole): (400, 0) -> inside_is_zero == True
    # Obs 1: inside donut hole (outside island): (200, 0) -> outside -> eucl == 100m
    # Obs 2: inside nested island: (0, 0) -> inside_is_zero == True
    obs_xy = np.array([[400.0, 0.0], [200.0, 0.0], [0.0, 0.0]])
    n0_lat, n0_lon = proj.inverse(-600, 0)
    n1_lat, n1_lon = proj.inverse(600, 0)
    sg = build_street_graph(
        {
            "elements": [
                {"type": "node", "id": 1, "lat": float(n0_lat), "lon": float(n0_lon)},
                {"type": "node", "id": 2, "lat": float(n1_lat), "lon": float(n1_lon)},
                {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"highway": "footway"}},
            ]
        },
        proj,
        "walk",
    )
    t_cfg = {
        "id": "park",
        "geometry_handling": "polygon_boundary",
        "inside_is_zero": True,
        "distance_modes": "both",
        "max_snap_distance_m": 600.0,
    }
    dists = compute_target_distances(obs_xy, np.array([True, True, True]), feats, t_cfg, proj, sg)
    assert dists["dist_eucl"][0] == 0.0 and dists["dist_net"][0] == 0.0
    assert dists["circuity_delta"][0] == 0.0 and dists["circuity_ratio"][0] == 1.0
    assert dists["dist_eucl"][1] == pytest.approx(100.0, abs=1.5)
    assert dists["dist_net"][1] > 0.0
    assert dists["dist_eucl"][2] == 0.0 and dists["dist_net"][2] == 0.0
    assert dists["circuity_delta"][2] == 0.0 and dists["circuity_ratio"][2] == 1.0

    # Also test chunked _points_in_ring with >512 pts and >2048 ring edges
    angles = np.linspace(0, 2 * np.pi, 2200, endpoint=True)
    big_ring = np.column_stack([100.0 * np.cos(angles), 100.0 * np.sin(angles)])
    pts_test = np.vstack([np.zeros((300, 2)), np.full((300, 2), 150.0)])
    mask = _points_in_ring(pts_test, big_ring, pt_chunk=512, edge_chunk=2048)
    assert np.all(mask[:300]) and not np.any(mask[300:])


def test_projection_oblique_aeqd_and_coordinate_diagnostics():
    """Wide longitudinal span uses Oblique AEQD with sub-millimeter inverse roundtrip; CRS/swap issues flagged."""
    proj = LocalMetricProjection(lat0_deg=47.5, lon0_deg=19.0, lon_span_deg=12.0)
    assert proj.mode == "aeqd"
    x, y = proj.forward(49.2, 24.5)
    lat_rt, lon_rt = proj.inverse(x, y)
    assert float(lat_rt) == pytest.approx(49.2, abs=1e-7)
    assert float(lon_rt) == pytest.approx(24.5, abs=1e-7)

    # Projected metric CRS mislabeled as lat/lon
    diag_meters = diagnose_coordinate_issues(np.array([5260000.0, 5261000.0]), np.array([385000.0, 386000.0]))
    assert diag_meters["looks_projected_crs"] is True
    assert any("WGS84" in w or "EPSG" in w for w in diag_meters["hints"])

    # Swapped coordinates where lon > 90
    diag_swap = diagnose_coordinate_issues(np.array([105.0, 105.1]), np.array([47.5, 47.6]))
    assert len(diag_swap["hints"]) >= 1


def test_ols_collinear_vif_and_crve_df_warning():
    """Collinear predictors must NOT clamp VIF to 1.0; CRVE must use df = G - 1 and warn on small G."""
    rng = np.random.default_rng(7)
    n = 80
    x1 = rng.normal(0, 1, n)
    x2 = 2.0 * x1  # exact collinearity!
    y = 3.0 + 1.5 * x1 + rng.normal(0, 0.5, n)
    X = np.column_stack([np.ones(n), x1, x2])
    col_meta = [
        {"name": "const", "label": "Intercept", "kind": "intercept", "is_binary": False},
        {"name": "x1", "label": "x1", "kind": "numeric", "is_binary": False},
        {"name": "x2", "label": "x2", "kind": "numeric", "is_binary": False},
    ]

    clusters = np.array([f"c{i % 5}" for i in range(n)])
    ols = fit_ols_core(X[:, :2], y, cov_type="cluster", cluster_ids=clusters)
    assert ols["df_infer"] == 4  # G - 1, not n - k = 78
    assert any("over-reject" in w for w in ols["cov_warnings"])

    ols_col = fit_ols_core(X, y, cov_type="HC3")
    diag = run_guard1_diagnostics(X, y, ols_col, col_meta)
    assert diag["status"] == "warn"
    assert diag["max_vif"] >= 1e5
    assert set(["x1", "x2"]).issubset(set(diag["collinear_cols"]))


def test_lisa_conditional_permutation_fdr_small_cluster():
    """With n_perm=1999 and exact conditional permutation, BH-FDR detects <10% hotspot cluster."""
    rng = np.random.default_rng(11)
    gx, gy = np.meshgrid(np.linspace(0, 900, 10), np.linspace(0, 900, 10))
    coords = np.column_stack([gx.ravel(), gy.ravel()])  # N = 100
    vals = rng.normal(0.0, 0.15, 100)
    # Put 6 neighboring hot observations (6% < 10% share!)
    hot_idx = [0, 1, 2, 10, 11, 12]
    vals[hot_idx] = 4.5

    w_obj = build_spatial_weight_matrix(coords, {"id": "knn4", "type": "knn", "k": 4})
    lisa = compute_lisa(vals, w_obj, n_perm=1999, use_fdr=True, alpha=0.05, seed=42)
    assert lisa["counts"]["HH"] >= 4


def test_sar_sdm_tier2_and_islands_impacts_match_dense():
    """Krinsky-Robb with islands > 0 and Tier-2 splu interpolation matches exact dense (I - rho W)^{-1} impacts."""
    rng = np.random.default_rng(19)
    n = 65
    coords = rng.uniform(0, 500, (n, 2))
    # Force an isolated point at index 0 -> islands == 1 under fixed cutoff 200m
    coords[0] = [50000.0, 50000.0]
    w_obj = build_spatial_weight_matrix(coords, {"id": "idw_200", "type": "idw", "power": 1, "cutoff_mode": "fixed_m", "cutoff_m": 200.0})
    assert len(w_obj["islands"]) >= 1

    x = rng.normal(0, 1, n)
    W_dense = w_obj["W"].toarray()
    rho_true = 0.35
    A_inv = np.linalg.inv(np.eye(n) - rho_true * W_dense)
    X = np.column_stack([np.ones(n), x])
    y = A_inv @ (2.0 + 1.5 * x + rng.normal(0, 0.3, n))

    # SDM impacts with islands > 0: check deterministic Total = mean(rowSums(A_inv @ (b I + th W)))
    n_x = 1
    X_sdm = np.column_stack([X, W_dense @ x])
    sdm_fit = _fit_ml_sar_core(X_sdm, y, w_obj)
    imp = compute_lesage_pace_impacts(sdm_fit, w_obj, ["const", "x"], is_sdm=True, n_x_no_int=1, krinsky_robb_draws=100)
    rho_hat = sdm_fit["rho"]
    b_hat = sdm_fit["beta"][1]
    th_hat = sdm_fit["beta"][2]
    S = np.linalg.inv(np.eye(n) - rho_hat * W_dense) @ (b_hat * np.eye(n) + th_hat * W_dense)
    wrong_no_island_total = (b_hat + th_hat) / (1.0 - rho_hat)
    assert abs(imp[0]["total"]["est"] - float(np.mean(S.sum(axis=1)))) < 5e-4
    assert abs(imp[0]["total"]["est"] - wrong_no_island_total) > 5e-3
    assert imp[0]["direct"]["est"] == pytest.approx(float(np.trace(S) / n), abs=5e-3)

    # Also test N > 1500 Tier-2 splu path stays finite and accurate without Taylor blowup
    n2 = 1520
    coords2 = rng.uniform(0, 1000, (n2, 2))
    coords2[0] = [50000.0, 50000.0]
    w2 = build_spatial_weight_matrix(coords2, {"id": "idw_auto", "type": "idw", "power": 1, "cutoff_mode": "auto_connectivity"})
    x2 = rng.normal(0, 1, n2)
    X2 = np.column_stack([np.ones(n2), x2])
    y2 = 1.0 + 0.3 * (w2["W"] @ (2.0 * x2)) + 2.0 * x2 + rng.normal(0, 0.3, n2)
    sar2 = _fit_ml_sar_core(X2, y2, w2)
    imp2 = compute_lesage_pace_impacts(sar2, w2, ["const", "x"], is_sdm=False, n_x_no_int=1, krinsky_robb_draws=60)
    assert np.isfinite(imp2[0]["total"]["se"]) and imp2[0]["total"]["est"] > imp2[0]["direct"]["est"] * 0.9


def test_gwr_native_intercept_se_no_15pct_floor_symmetric_bw_fdr():
    """Verify GWR native intercept SE includes Cov(b0,b1), no 0.15*n floor, symmetric bw_curve, FBC label."""
    rng = np.random.default_rng(23)
    n = 260
    coords = rng.uniform(0, 1000, (n, 2))
    # Strongly non-centered predictor (mean ~ 100) so native intercept SE strongly depends on Cov(beta_0, beta_1)
    x = rng.normal(100.0, 5.0, n)
    beta1_loc = 0.5 + 0.4 * np.sin(coords[:, 0] / 150.0)
    y = 20.0 + beta1_loc * x + rng.normal(0, 0.8, n)
    X = np.column_stack([np.ones(n), x])
    col_meta = [
        {"name": "const", "label": "Intercept", "kind": "intercept"},
        {"name": "x", "label": "x", "kind": "numeric"},
    ]

    gwr = fit_adaptive_gwr(X, y, col_meta, coords, {"jitter_m": 0.0, "kappa_thresh": 1e6, "b_min_floor": 10})
    assert gwr["ok"] is True
    # Old 0.15 * 260 = 39 floor is gone!
    assert gwr["b_min_floor"] < 0.15 * n
    # Stationarity label is FBC (2002), NOT Leung
    for s in gwr["param_summaries"]:
        assert "FBC" in s["stationarity_verdict"] and "Leung" not in s["stationarity_verdict"]
        assert "pct_sig_fdr" in s and "pct_sig_sdf" in s

    # Verify native intercept SE matches exact unstandardized bisquare WLS covariance at i=0
    bw = gwr["optimal_bw"]
    dists = np.linalg.norm(coords - coords[0], axis=1)
    d_bw = np.sort(dists)[bw]  # bw_radii = dists_sorted[:, bw] matches gwr_kernel.py
    u = dists / (d_bw + 1e-12)
    w = np.where(u < 1.0, (1.0 - u ** 2) ** 2, 0.0)
    W_sqrt = np.sqrt(w)[:, None]
    Xw = X * W_sqrt
    C0 = np.linalg.pinv(Xw.T @ Xw) @ (X * w[:, None]).T  # (p, n) projection row at i=0
    # gwr sigma^2 = RSS / eff_df (where sigma_hat matches RSS / (n - tr(S)) or RSS / eff_df)
    se0_expected = math.sqrt(float((C0[0] ** 2).sum()) * float(np.sum(np.array(gwr["resid_native"]) ** 2) / gwr["eff_df"]))
    assert gwr["se_native"][0][0] == pytest.approx(se0_expected, rel=1e-4)

    # Optimal bw in bw_curve has minimal raw aicc
    best_curve_pt = min(gwr["bw_curve"], key=lambda p: p["aicc"])
    assert abs(best_curve_pt["bw"] - gwr["optimal_bw"]) <= 3


def test_design_rings_merge_circuity_heckman_stage4_spkey_and_exporters():
    """Cover rings ref_count==0 dummy trap, circuity term metric parity, multi-target AD, Heckman edge cases, sp_key sensitivity, and LaTeX/CSV safety."""
    # 1. Rings ref_count == 0 -> merged safely without rank deficiency
    n = 40
    dist_dict = {
        "tgt": {
            "eucl": np.concatenate([np.full(20, 50.0), np.full(20, 220.0)]),
            "net": np.concatenate([np.full(20, 80.0), np.full(20, 310.0)]),
            "circuity_delta": np.concatenate([np.full(20, 30.0), np.full(20, 90.0)]),
            "circuity_ratio": np.concatenate([np.full(20, 1.6), np.full(20, 1.409)]),
        }
    }
    cols = {"y": [100.0 + i for i in range(n)], "x": [float(i) for i in range(n)]}
    y_info = build_dependent_variable(cols, {"mode": "level", "num_col": "y", "transform": "raw"})
    model_rings = {
        "id": "m_r",
        "name": "Rings Test",
        "covariates": [{"col": "x", "type": "numeric", "transform": "raw"}],
        "spatial_terms": [{"target_id": "tgt", "metric": "eucl", "transform": "rings", "breakpoints_m": [150.0, 500.0]}],
        "interactions": [],
    }
    dm = build_model_design(cols, dist_dict, model_rings, y_info, np.ones(n, dtype=bool), sparse_min_cell=5)
    assert any(ev["type"] == "ring_empty_dropped" for ev in dm["guard2_events"])
    assert np.linalg.matrix_rank(dm["X"]) == dm["X"].shape[1]

    # 2. Circuity term evaluated with metric='eucl' still computes net-eucl / net/eucl
    c_cols, _, _ = _evaluate_spatial_term(
        dist_dict, cols, {"target_id": "tgt", "metric": "eucl", "transform": "circuity_delta"}, n
    )
    assert np.allclose(c_cols[0]["values"][:20], 30.0)

    # 3. Heckman non-numeric col warning + degenerate selection + IMR collinearity check
    bad_cols = {"y": [10.0] * 20, "x_num": [float(i) for i in range(20)], "bad_str": ["abc"] * 20}
    sel_all_one = np.ones(20, dtype=bool)
    hk_empty = run_heckman_two_step(bad_cols, sel_all_one, np.arange(20), np.ones((20, 2)), np.ones(20), [], ["x_num"], ["bad_str"])
    assert hk_empty["activated"] is False

    # Degenerate missing/non-numeric exclusion with nonzero excluded rows
    sel_half = np.array([True] * 35 + [False] * 15)
    bad_cols2 = {
        "x_num": np.linspace(1, 50, 50).tolist(),
        "bad_str": ["foo"] * 50,
    }
    X_s2 = np.column_stack([np.ones(35), np.linspace(1, 35, 35)])
    y_s2 = np.linspace(2, 36, 35)
    hk_nonnum = run_heckman_two_step(
        bad_cols2, sel_half, np.arange(35), X_s2, y_s2,
        [{"name": "const", "label": "const"}, {"name": "x_num", "label": "x_num"}],
        ["x_num"], ["bad_str"],
    )
    assert hk_nonnum["activated"] is True
    assert hk_nonnum["imr_vif"] is not None
    assert any("non-numeric" in w.lower() for w in hk_nonnum["warnings"])

    # 4. Multi-target AD profile & Stage 4 sp_key sensitivity beyond row 10 and on y mutation
    rng = np.random.default_rng(55)
    m = 50
    lat = 47.5 + rng.uniform(-0.005, 0.005, m)
    lon = 19.0 + rng.uniform(-0.008, 0.008, m)
    area = rng.uniform(30, 80, m)
    y_base = 100.0 + 2.0 * area + rng.normal(0, 2.0, m)

    def make_sess(y_vec):
        s = GeoSpecSession()
        s.load_dataset({"id": list(range(m)), "lat": lat.tolist(), "lon": lon.tolist(), "price": y_vec.tolist(), "area": area.tolist()}, "lat", "lon")
        s.ingest_target_raw("t1", {"type": "FeatureCollection", "features": [{"type": "Feature", "id": "p1", "geometry": {"type": "Point", "coordinates": [19.0, 47.5]}}]}, source_kind="geojson")
        s.ingest_target_raw("t2", {"type": "FeatureCollection", "features": [{"type": "Feature", "id": "p2", "geometry": {"type": "Point", "coordinates": [19.005, 47.503]}}]}, source_kind="geojson")
        rec = {
            "dataset_schema": {"row_id_col": "id", "lat_col": "lat", "lon_col": "lon", "y_spec": {"mode": "level", "num_col": "price", "transform": "raw"}, "cluster_spec": {"mode": "none"}, "selection_spec": {"enabled": False}},
            "spatial_targets": [
                {"id": "t1", "name": "T1", "source": "geojson", "geometry_handling": "point", "inside_is_zero": False, "distance_modes": "euclidean"},
                {"id": "t2", "name": "T2", "source": "geojson", "geometry_handling": "point", "inside_is_zero": False, "distance_modes": "euclidean"},
            ],
            "models": [
                {
                    "id": "m_dual",
                    "name": "Dual AD",
                    "active": True,
                    "covariates": [{"col": "area", "type": "numeric", "transform": "raw"}],
                    "spatial_terms": [
                        {"target_id": "t1", "metric": "eucl", "transform": "andersson_day", "cutoff_m": 250, "cutoff_range_m": [100, 400, 50]},
                        {"target_id": "t2", "metric": "eucl", "transform": "andersson_day", "cutoff_m": 300, "cutoff_range_m": [100, 400, 50]},
                    ],
                    "interactions": [],
                }
            ],
            "spatial_engine": {"active_model_id": "m_dual", "weights": [{"id": "knn5", "type": "knn", "k": 5}], "krinsky_robb_draws": 50, "lisa_fdr": True},
        }
        return s, rec

    sess, rec = make_sess(y_base)
    sess.run_stage_1_distances(rec, run_id=1)
    s3 = sess.run_stage_3_ols_workbench(rec, run_id=1)
    assert "m_dual::t1" in s3["ad_profiles"] and "m_dual::t2" in s3["ad_profiles"]
    assert s3["ad_profiles"]["m_dual::t1"]["ok"] and s3["ad_profiles"]["m_dual::t2"]["ok"]

    s4_a = sess.run_stage_4_spatial_econometrics(rec, run_id=1)
    # Mutate ONLY y[35] (row > 10!) in-place via new load_dataset -> Stage 4 MUST recompute!
    y_mut = y_base.copy()
    y_mut[35] += 500.0
    sess.load_dataset({"id": list(range(m)), "lat": lat.tolist(), "lon": lon.tolist(), "price": y_mut.tolist(), "area": area.tolist()}, "lat", "lon")
    sess.ingest_target_raw("t1", {"type": "FeatureCollection", "features": [{"type": "Feature", "id": "p1", "geometry": {"type": "Point", "coordinates": [19.0, 47.5]}}]}, source_kind="geojson")
    sess.ingest_target_raw("t2", {"type": "FeatureCollection", "features": [{"type": "Feature", "id": "p2", "geometry": {"type": "Point", "coordinates": [19.005, 47.503]}}]}, source_kind="geojson")
    sess.run_stage_1_distances(rec, run_id=2)
    s4_b = sess.run_stage_4_spatial_econometrics(rec, run_id=2)
    assert s4_a["moran_y"]["moran_i"] != s4_b["moran_y"]["moran_i"]

    # 5. Exporters LaTeX escaping & CSV formula neutralization
    tex = generate_multi_model_latex(
        [
            {
                "model_id": "m2_rings",
                "model_name": "Model_A & B",
                "n_obs": 50,
                "k_params": 2,
                "adj_r2": 0.5,
                "aic": 10.0,
                "llf": -3.0,
                "is_log_scale": False,
                "cov_type_used": "HC3_cluster&1",
                "params": [{"name": "price_m2 & tax", "label": "Price_m2 & Tax", "is_binary": False, "beta": 1.2, "se": 0.3, "sig": "***", "hp_pct": None, "kennedy_pct": None}],
                "footnotes": ["Footnote with under_score & percent %"],
            }
        ],
        label="tab:bad_label&1",
    )
    assert r"m2\_rings" in tex
    assert r"HC3\_cluster\&1" in tex
    assert r"price\_m2 \& tax" in tex
    assert r"under\_score \& percent \%" in tex

    csv_txt = dict_rows_to_csv([{"formula_eq": "=CMD()", "plus": "+1", "dash_cmd": "-2+3", "at": "@SUM", "lead_sp": "  =CMD()", "neg_num": -12.5, "tab": "\t=1"}])
    lines = csv_txt.strip().splitlines()
    assert "'=CMD()" in lines[1]
    assert "'+1" in lines[1]
    assert "'-2+3" in lines[1]
    assert "'@SUM" in lines[1]
    assert "'  =CMD()" in lines[1]
    assert "'\t=1" in lines[1]
    assert ",-12.5," in lines[1]  # legitimate numeric float scalar is NOT altered


def test_p2_knn_dedup_and_idw_outlier_guard():
    """Verify knn dedup_colocated without O(N^2) k=n and idw auto_connectivity isolated outlier guard."""
    rng = np.random.default_rng(99)
    n = 100
    coords = rng.uniform(0, 300, (n, 2))
    # Inject colocated group at indices 0..17 (>= k + 12 = 16 to exercise still_pending) + 1 extreme 50km outlier at index 99
    coords[:18] = [100.0, 100.0]
    coords[99] = [50000.0, 50000.0]

    w_knn_raw = build_spatial_weight_matrix(coords[:99], {"id": "knn4r", "type": "knn", "k": 4, "dedup_colocated": False})
    w_knn = build_spatial_weight_matrix(coords[:99], {"id": "knn4d", "type": "knn", "k": 4, "dedup_colocated": True})
    assert w_knn_raw["colocated_pairs_pct"] > 0.0
    assert w_knn["colocated_pairs_pct"] == 0.0
    assert w_knn["W"].shape == (99, 99)

    w_idw = build_spatial_weight_matrix(coords, {"id": "idw_auto", "type": "idw", "power": 1, "cutoff_mode": "auto_connectivity"})
    assert w_idw["W"].shape == (n, n)
    assert w_idw["meta"].get("outlier_guard_triggered") is True
    assert len(w_idw["islands"]) >= 1

    # Verify both cutoff_mode='fixed' and UI's cutoff_mode='fixed_m' actually honor cutoff_m
    w_fixed_m = build_spatial_weight_matrix(coords, {"id": "idw_fix", "type": "idw", "power": 1, "cutoff_mode": "fixed_m", "cutoff_m": 175.0})
    assert w_fixed_m["meta"]["cutoff_m"] == 175.0
    assert len(w_fixed_m["islands"]) >= 1


def test_export_fallback_osm_bundle_and_closed_preview_rings():
    """Python-exported spatial/osm_bundle.json must match frontend import schema and close polygon rings."""
    rng = np.random.default_rng(101)
    m = 30
    lat = 47.5 + rng.uniform(-0.003, 0.003, m)
    lon = 19.0 + rng.uniform(-0.004, 0.004, m)
    sess = GeoSpecSession()
    sess.load_dataset({"id": list(range(m)), "lat": lat.tolist(), "lon": lon.tolist(), "y": np.linspace(100, 200, m).tolist(), "x": np.linspace(10, 40, m).tolist()}, "lat", "lon")
    # Deliberately unclosed outer ring in GeoJSON polygon
    sess.ingest_target_raw(
        "park",
        {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "id": "p1", "geometry": {"type": "Polygon", "coordinates": [[[19.0, 47.5], [19.002, 47.5], [19.002, 47.502], [19.0, 47.502]]]}}],
        },
        source_kind="geojson",
    )
    prev = features_to_preview_geojson(sess.target_features["park"])
    ring0 = prev["features"][0]["geometry"]["coordinates"][0]
    assert ring0[0] == ring0[-1]

    rec = {
        "dataset_schema": {"row_id_col": "id", "lat_col": "lat", "lon_col": "lon", "y_spec": {"mode": "column", "num_col": "y", "transform": "raw"}, "cluster_spec": {"mode": "none"}, "selection_spec": {"enabled": False}},
        "spatial_targets": [{"id": "park", "name": "Park", "source": "geojson", "geometry_handling": "polygon_boundary", "inside_is_zero": True, "distance_modes": "euclidean"}],
        "models": [{"id": "m1", "name": "M1", "active": True, "covariates": [{"col": "x", "type": "numeric", "transform": "raw"}], "spatial_terms": [{"target_id": "park", "metric": "eucl", "transform": "andersson_day", "cutoff_m": 250, "cutoff_range_m": [100, 400, 0]}], "interactions": []}],
        "spatial_engine": {"active_model_id": "m1"},
    }
    sess.run_stage_1_distances(rec, run_id=1)
    s3 = sess.run_stage_3_ols_workbench(rec, run_id=1)
    assert s3["ok"] is True
    assert s3["ad_profiles"]["m1::park"]["ok"] is True

    import io, json, zipfile
    b = sess.export_publication_bundle(rec)
    with zipfile.ZipFile(io.BytesIO(b["zip_bytes"]), "r") as zf:
        osm_b = json.loads(zf.read("spatial/osm_bundle.json").decode("utf-8"))
    assert osm_b["targets"]["park"]["source_kind"] == "geojson"
    assert "payload" in osm_b["targets"]["park"]
    assert osm_b["targets"]["park"]["target_cfg"]["id"] == "park"


def test_round2_p0_regressions():
    """Verify Round-2 swarm audit P0 fixes:
    1. Closed polygon boundary distance (4th closing edge not missing).
    2. Conditional Box-Cox estimation in GeoSpecSession & exact normal constant in profile MLE.
    3. Stage 4 OLS vs SAR AIC +2.0 parity when rho ~ 0 and SDM small-N fallback guard.
    4. Single-pass _latex_escape without double-escaping backslash braces.
    5. build_qgis_geojson multi-stage valid_indices alignment and auto-hydrated csv_column targets.
    """
    from py_engine.exporters import _latex_escape, build_qgis_geojson
    from py_engine.spatial_gis import compute_target_distances, parse_geojson_features
    from py_engine.projection import LocalMetricProjection

    # 1. Single-pass LaTeX escape
    assert _latex_escape(r"a\b_c") == r"a\textbackslash{}b\_c"

    # 2. Unclosed 4-point GeoJSON square polygon boundary: point near the 4th closing edge (west edge x=19.0)
    proj = LocalMetricProjection(47.501, 19.001)
    unclosed_poly = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "id": "sq1",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[19.000, 47.500], [19.002, 47.500], [19.002, 47.502], [19.000, 47.502]]],
            },
        }],
    }
    feats = parse_geojson_features(unclosed_poly, "sq")
    # Point just west of the 4th edge (lon=18.9999, lat=47.501 -> ~7.5m from west edge, >110m from north/south edges)
    px, py = proj.forward(np.array([47.501]), np.array([18.9999]))
    obs_xy = np.column_stack([px, py])
    d_res = compute_target_distances(
        obs_xy,
        np.array([True]),
        feats,
        {"id": "sq", "geometry_handling": "polygon_boundary", "inside_is_zero": False, "distance_modes": "euclidean"},
        proj,
        None,
    )
    assert float(d_res["dist_eucl"][0]) < 20.0

    # 3. Auto-hydrated csv_column targets + Conditional Box-Cox + Stage 4 OLS/SAR AIC parity
    rng = np.random.default_rng(2026)
    n = 45
    lat = 47.49 + rng.uniform(-0.01, 0.01, n)
    lon = 19.14 + rng.uniform(-0.01, 0.01, n)
    x_val = rng.uniform(30.0, 100.0, n)
    d_csv = rng.uniform(50.0, 800.0, n)
    # True log-linear model so conditional Box-Cox lambda_mle is near 0
    y_val = np.exp(10.0 + 0.015 * x_val - 0.0004 * d_csv + rng.normal(0, 0.08, n))

    sess = GeoSpecSession()
    sess.load_dataset(
        {
            "id": list(range(n)),
            "lat": lat.tolist(),
            "lon": lon.tolist(),
            "price": y_val.tolist(),
            "area": x_val.tolist(),
            "dist_park_m": d_csv.tolist(),
        },
        "lat",
        "lon",
    )
    recipe = {
        "dataset_schema": {
            "row_id_col": "id",
            "lat_col": "lat",
            "lon_col": "lon",
            "y_spec": {"mode": "column", "num_col": "price", "transform": "boxcox", "boxcox_lambda": None},
            "cluster_spec": {"mode": "none"},
            "selection_spec": {"enabled": False},
        },
        "spatial_targets": [
            {"id": "park", "name": "Park CSV", "source": "csv_column", "csv_col": "dist_park_m", "distance_modes": "euclidean"}
        ],
        "models": [
            {
                "id": "m1",
                "name": "M1",
                "active": True,
                "covariates": [{"col": "area", "type": "numeric", "transform": "raw"}],
                "spatial_terms": [{"target_id": "park", "metric": "eucl", "transform": "linear"}],
                "interactions": [],
            }
        ],
        "spatial_engine": {
            "active_model_id": "m1",
            "weights": [{"id": "knn4", "type": "knn", "k": 4}],
        },
    }
    # Deliberately call Stage 3 WITHOUT calling Stage 1 first: csv_column must auto-hydrate and Box-Cox must be conditional_on_X
    s3 = sess.run_stage_3_ols_workbench(recipe, run_id=1)
    assert s3["ok"] is True
    assert "park" in sess.dist_results
    s2 = sess.run_stage_2_descriptives_and_composition(recipe, run_id=1)
    assert s2["descriptives"]["boxcox_info"]["mode"] == "conditional_on_X"

    s4 = sess.run_stage_4_spatial_econometrics(recipe, run_id=1)
    assert s4["ok"] is True
    comp = {row["model"]: row for row in s4["model_comparison"]}
    # OLS AIC in Stage 4 uses k = X.shape[1] + 1 (counting sigma^2), so at the same LLF, AIC_SAR - AIC_OLS <= 2.0
    ols_aic = comp["OLS (Baseline)"]["aic"]
    ols_llf = comp["OLS (Baseline)"]["llf"]
    sar_aic = comp["SAR (ML, knn4)"]["aic"]
    sar_llf = comp["SAR (ML, knn4)"]["llf"]
    assert abs((sar_aic + 2.0 * sar_llf) - (ols_aic + 2.0 * ols_llf) - 2.0) < 0.1

    # 4. build_qgis_geojson with mismatched valid_indices across active_ols and spatial_res
    gj = build_qgis_geojson(
        row_ids=[0, 1, 2],
        lat_arr=[47.5, 47.51, 47.52],
        lon_arr=[19.1, 19.11, 19.12],
        y_raw=[100.0, 200.0, 300.0],
        y_trans=[4.6, 5.3, 5.7],
        dist_dict={},
        active_ols={"valid_indices": [0, 2], "fitted": [4.5, 5.8], "resid": [0.1, -0.1], "cooks_d": [0.01, 0.02]},
        spatial_res={"ok": True, "valid_indices": [1, 2], "lisa_y": {"quadrants": ["HH", "LL"]}, "lisa_resid": {"quadrants": ["NS", "HL"]}},
        gwr_res=None,
        analytic_indices=[0, 2],
    )
    p0, p1, p2 = [f["properties"] for f in gj["features"]]
    assert p0["ols_fitted"] == 4.5 and "lisa_y_quad" not in p0
    assert "ols_fitted" not in p1 and p1["lisa_y_quad"] == "HH"
    assert p2["ols_fitted"] == 5.8 and p2["lisa_y_quad"] == "LL"


def test_round3_best_practice_and_regressions():
    """Verify Round-3 swarm audit fixes (ECON-01..03, ARCH-02/04, M3-04/05, and recipe split)."""
    import io
    import json
    import pathlib
    import zipfile
    from py_engine.dag_cache import DAGCache
    from py_engine.spatial_kernel import run_spatial_pipeline

    # 1. ECON-01: SDM small-N underdetermined fallback without crash or SAR masquerade
    rng = np.random.default_rng(77)
    n_small, k_small = 10, 6  # K_sdm = 6 + 5 = 11 => n_small <= K_sdm + 2
    X_s = np.column_stack([np.ones(n_small), rng.normal(size=(n_small, k_small - 1))])
    y_s = rng.normal(10.0, 1.0, size=n_small)
    coords_s = rng.uniform(0.0, 500.0, size=(n_small, 2))
    sp_small = run_spatial_pipeline(
        X_s, y_s, ["Intercept", "x1", "x2", "x3", "x4", "x5"], coords_s,
        [{"id": "knn3", "type": "knn", "k": 3}], krinsky_robb_draws=200, lisa_fdr=False,
    )
    assert sp_small["ok"] is True
    assert sp_small["sdm_underdetermined"] is True
    assert sp_small["impacts_sdm"] == []
    assert sp_small["elhorst"]["lr_vs_sar"] is None
    sdm_row = sp_small["model_comparison"][-1]
    assert sdm_row.get("underdetermined") is True
    assert sdm_row["aic"] is None
    assert any("SDM underdetermined" in msg for msg in sp_small["guard5_alerts"])

    # 2. ARCH-04: DAGCache bounded LRU eviction
    dc = DAGCache()
    for i in range(12):
        dc.set("dist", f"k_{i}", i)
    assert len(dc._store["dist"]) == DAGCache.HEAVY_MAX_ENTRIES
    assert dc.get("dist", "k_0") is None
    assert dc.get("dist", "k_11") == 11

    # 3. ECON-02, ECON-03, ARCH-02, M3-04, M3-05 via full GeoSpecSession
    n = 55
    lat = 47.48 + rng.uniform(-0.008, 0.008, n)
    lon = 19.14 + rng.uniform(-0.008, 0.008, n)
    area = rng.uniform(35.0, 95.0, n)
    hood = ["A" if i % 2 == 0 else "B" for i in range(n)]
    # Make y depend strongly on categorical + spatial proximity so unconditional vs full-design Box-Cox differ
    y = np.exp(10.2 + 0.012 * area + np.where(np.array(hood) == "A", 0.35, 0.0) + rng.normal(0, 0.1, n))

    sess = GeoSpecSession()
    sess.load_dataset(
        {"id": list(range(n)), "lat": lat.tolist(), "lon": lon.tolist(), "price": y.tolist(), "area": area.tolist(), "hood": hood},
        "lat", "lon",
    )
    tgt1 = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "id": "st1", "geometry": {"type": "Point", "coordinates": [19.14, 47.48]}}],
    }
    tgt2 = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "id": "st1", "geometry": {"type": "Point", "coordinates": [19.15, 47.49]}}],
    }
    rec = {
        "dataset_schema": {
            "row_id_col": "id", "lat_col": "lat", "lon_col": "lon",
            "y_spec": {"mode": "column", "num_col": "price", "transform": "boxcox", "boxcox_lambda": None},
            "cluster_spec": {"mode": "none"}, "selection_spec": {"enabled": False},
        },
        "spatial_targets": [{"id": "st", "name": "Station", "source": "geojson", "geometry_handling": "point", "distance_modes": "euclidean"}],
        "models": [{
            "id": "m1", "name": "M1", "active": True,
            "covariates": [{"col": "area", "type": "numeric", "transform": "raw"}, {"col": "hood", "type": "categorical"}],
            "spatial_terms": [{"target_id": "st", "metric": "eucl", "transform": "log"}],
            "interactions": [],
        }],
        "spatial_engine": {
            "active_model_id": "m1",
            "weights": [{"id": "knn5", "type": "knn", "k": 5, "dedup_colocated": True}],
            "krinsky_robb_draws": 150,
            "gwr": {"focal_model_id": "m1", "jitter_m": 1.0, "b_min_floor": 12, "kappa_thresh": 1000.0},
        },
    }

    # ARCH-02: Same feature count (1) & same uid ('st1'), moved coordinates -> must recompute distances
    sess.ingest_target_raw("st", tgt1, source_kind="geojson")
    s1_a = sess.run_stage_1_distances(rec, run_id=1)
    sess.ingest_target_raw("st", tgt2, source_kind="geojson")
    s1_b = sess.run_stage_1_distances(rec, run_id=2)
    assert s1_a["summary"][0]["median_eucl_m"] != s1_b["summary"][0]["median_eucl_m"]

    # ECON-02 & ECON-03: Full-design Box-Cox lambda matches Stage 3 conditional_boxcox & OLS AIC matches Stage 4
    s2 = sess.run_stage_2_descriptives_and_composition(rec, run_id=3)
    s3 = sess.run_stage_3_ols_workbench(rec, cov_type="HC3", run_id=4)
    s4 = sess.run_stage_4_spatial_econometrics(rec, run_id=5)
    s5 = sess.run_stage_5_gwr(rec, run_id=6)
    assert s2["ok"] and s3["ok"] and s4["ok"] and s5["ok"]
    assert s2["descriptives"]["boxcox_info"]["mode"] == "conditional_on_X"
    assert abs(s2["descriptives"]["boxcox_info"]["lambda_mle"] - s3["conditional_boxcox"]["lambda_mle"]) <= 1e-4
    assert s3["models"][0]["aic"] == s4["model_comparison"][0]["aic"]

    # M3-04 & M3-05: Publication bundle has Stage 4/5 LaTeX tables, Box-Cox note, and gwr_fdr_sig_*
    bundle = sess.export_publication_bundle(rec)
    with zipfile.ZipFile(io.BytesIO(bundle["zip_bytes"]), "r") as zf:
        names = set(zf.namelist())
        assert "latex/spatial_models_comparison.tex" in names
        assert "latex/gwr_summary.tex" in names
        tex_main = zf.read("latex/regression_comparison.tex").decode("utf-8")
        assert "Box--Cox" in tex_main
    props0 = bundle["geojson"]["features"][0]["properties"]
    assert any(k.startswith("gwr_fdr_sig_") for k in props0)
    assert any(k.startswith("gwr_se_") for k in props0)
    assert "gwr_kappa" in props0
    assert "lisa_resid_i" in props0 and "lisa_resid_p" in props0
    assert s4["moran_resid"]["moments_mode"] == "ols_mx_exact"

    # 3b. Exact M_X residual Moran's I vs dense matrix trace formula
    from py_engine.ols_kernel import fit_ols_core
    from py_engine.spatial_kernel import build_spatial_weight_matrix, compute_moran_i
    n_mx = 35
    x_mx = np.column_stack([np.ones(n_mx), np.linspace(-2.0, 2.0, n_mx), np.sin(np.linspace(0.0, 3.0, n_mx))])
    y_mx = 5.0 + 0.8 * x_mx[:, 1] + rng.normal(0.0, 0.5, n_mx)
    coords_mx = np.column_stack([np.linspace(0.0, 1000.0, n_mx), np.linspace(0.0, 400.0, n_mx)])
    w_mx = build_spatial_weight_matrix(coords_mx, {"id": "knn4", "type": "knn", "k": 4})
    ols_mx = fit_ols_core(x_mx, y_mx, cov_type="OLS")
    mor_mx = compute_moran_i(ols_mx["resid"], w_mx, X=x_mx)
    assert mor_mx["moments_mode"] == "ols_mx_exact"
    W_d = w_mx["W"].toarray()
    M_d = np.eye(n_mx) - x_mx @ np.linalg.pinv(x_mx)
    s0_d = float(np.sum(W_d))
    df_d = n_mx - x_mx.shape[1]
    A_d = 0.5 * (W_d + W_d.T)
    tr_mw_d = float(np.trace(M_d @ W_d))
    tr_mama_d = float(np.trace((M_d @ A_d) @ (M_d @ A_d)))
    ei_dense = (n_mx / (s0_d * df_d)) * tr_mw_d
    ei2_dense = ((n_mx / s0_d) ** 2) * (tr_mw_d ** 2 + 2.0 * tr_mama_d) / (df_d * (df_d + 2.0))
    se_dense = float(np.sqrt(ei2_dense - ei_dense ** 2))
    assert abs(mor_mx["expected_i"] - round(ei_dense, 4)) <= 1e-4
    assert abs(mor_mx["se"] - round(se_dense, 4)) <= 1e-4


def test_round4_all_fixes_and_roadmap():
    """Verify every Round-4 econometric, GIS, DAG, and exporter fix & roadmap feature."""
    import csv
    import io
    import json
    import zipfile
    from py_engine.design_matrix import _boxcox_profile_mle
    from py_engine.exporters import (
        _latex_escape,
        generate_robustness_matrix_latex,
        generate_spatial_params_latex,
    )
    from py_engine.spatial_gis import compute_pairwise_network_distances
    from py_engine.spatial_kernel import (
        compute_anselin_kelejian_test,
        fit_ml_sem,
        run_spatial_pipeline,
    )
    from py_engine.ols_kernel import run_wild_cluster_bootstrap

    rng = np.random.default_rng(202611)

    # 1. GWR: local zero-variance dummy auto-exclusion, native raw_aicc, MGWR backfitting, mc_nonstat_p, progress_cb
    n_gwr = 70
    gx = np.linspace(0.0, 1000.0, n_gwr)
    gy = rng.uniform(0.0, 200.0, n_gwr)
    coords_g = np.column_stack([gx, gy])
    x1_g = rng.normal(0.0, 1.0, n_gwr)
    # Binary dummy is 1 across half the study strip (gx < 450) -> invariant locally around left interior points
    dummy_g = (gx < 450.0).astype(np.float64)
    y_g = 5.0 + (1.2 + 0.001 * gx) * x1_g + 0.8 * dummy_g + rng.normal(0.0, 0.25, n_gwr)
    X_g = np.column_stack([np.ones(n_gwr), x1_g, dummy_g])
    meta_g = [
        {"name": "const", "label": "Intercept", "kind": "intercept", "is_binary": False},
        {"name": "x1", "label": "x1", "kind": "continuous", "is_binary": False},
        {"name": "dummy", "label": "dummy", "kind": "categorical", "is_binary": True},
    ]
    gwr_drop = fit_adaptive_gwr(X_g, y_g, meta_g, coords_g, gwr_cfg={"mode": "gwr", "b_min_floor": 10, "jitter_m": 0.5, "mc_iters": 19})
    assert gwr_drop["ok"] is True
    assert gwr_drop["n_local_dropped_locs"] > 0
    assert gwr_drop["gwr_aicc"] == pytest.approx(min(p["aicc"] for p in gwr_drop["bw_curve"]), rel=0.05)

    prog_events = []
    mgwr_res = fit_adaptive_gwr(
        X_g[:, :2],
        y_g,
        meta_g[:2],
        coords_g,
        gwr_cfg={"mode": "mgwr", "b_min_floor": 10, "jitter_m": 0.5, "mc_iters": 19},
        progress_cb=lambda frac, msg: prog_events.append((frac, msg)),
    )
    assert mgwr_res["ok"] is True and mgwr_res["mode"] == "mgwr"
    assert len(prog_events) >= 1
    assert all(0.0 <= s["mc_nonstat_p"] <= 1.0 for s in mgwr_res["param_summaries"])

    # 2. Design Matrix: scale-invariant Box-Cox, sparse reference fallback, Gram-Schmidt poly, DiD & RDD
    y_bc = np.exp(2.0 + 0.5 * x1_g + rng.normal(0.0, 0.15, n_gwr))
    X_bc = np.column_stack([np.ones(n_gwr), x1_g])
    lam1, _, p_log1, _ = _boxcox_profile_mle(y_bc, X_bc)
    lam2, _, p_log2, _ = _boxcox_profile_mle(y_bc * 1e9, X_bc)
    assert abs(lam1 - lam2) < 1e-3 and abs(p_log1 - p_log2) < 1e-3

    # Build synthetic dataset columns for Session + Design checks
    n_s = 80
    lat_s = 47.500 + np.linspace(-0.003, 0.003, n_s) + rng.normal(0.0, 0.0002, n_s)
    lon_s = 19.050 + np.linspace(-0.004, 0.004, n_s) + rng.normal(0.0, 0.0002, n_s)
    proj_s = LocalMetricProjection(float(np.mean(lat_s)), float(np.mean(lon_s)))
    sx_arr, sy_arr = proj_s.forward(lat_s, lon_s)
    obs_s = np.column_stack([sx_arr, sy_arr])
    x_cont_s = rng.normal(50.0, 12.0, n_s)
    cat_vals = ["Panel"] * 45 + ["Brick"] * 33 + ["RareCastle"] * 2
    post_vals = [0 if i < 40 else 1 for i in range(n_s)]
    clust_vals = [f"C{i % 8}" for i in range(n_s)]  # G = 8 < 30 -> Webb WCB!
    dist_fake = np.abs(sx_arr) + 50.0
    y_raw_s = np.exp(12.0 + 0.015 * x_cont_s - 0.0008 * dist_fake + rng.normal(0.0, 0.25, n_s))
    cols_s = {
        "id": list(range(n_s)),
        "lat": lat_s.tolist(),
        "lon": lon_s.tolist(),
        "price": y_raw_s.tolist(),
        "area": x_cont_s.tolist(),
        "btype": cat_vals,
        "post": post_vals,
        "clust": clust_vals,
        "ext_flag": [0] * (n_s - 10) + [1] * 10,
    }
    dist_map_s = {"hub": {"dist_eucl": dist_fake, "dist_net": dist_fake * 1.15}}
    y_inf_s = build_dependent_variable(cols_s, {"mode": "column", "num_col": "price", "transform": "log"})
    analytic_s = np.ones(n_s, dtype=bool)
    spec_fallback = {
        "id": "m_fb",
        "name": "Fallback & Poly & Causal",
        "covariates": [
            {"col": "area", "type": "continuous", "transform": "poly"},
            {"col": "btype", "type": "categorical", "reference_level": "RareCastle"},
        ],
        "spatial_terms": [],
        "interactions": [],
        "causal_spec": {"mode": "spatial_did", "treat_target_id": "hub", "treat_cutoff_m": 200.0, "post_col": "post"},
    }
    d_did = build_model_design(cols_s, dist_map_s, spec_fallback, y_inf_s, analytic_s, sparse_min_cell=5)
    assert any(ev["type"] == "sparse_reference_fallback" for ev in d_did["guard2_events"])
    names_did = [m["name"] for m in d_did["col_meta"]]
    assert any(nm.startswith("did_att(") for nm in names_did)
    idx_a = names_did.index("area")
    idx_a2 = names_did.index("area^2")
    a_c = d_did["X"][:, idx_a] - np.mean(d_did["X"][:, idx_a])
    assert abs(float(np.corrcoef(a_c, d_did["X"][:, idx_a2])[0, 1])) < 1e-10

    spec_rdd = {**spec_fallback, "causal_spec": {"mode": "boundary_rdd", "treat_target_id": "hub", "treat_cutoff_m": 200.0, "bandwidth_m": 150.0}}
    d_rdd = build_model_design(cols_s, dist_map_s, spec_rdd, y_inf_s, analytic_s, sparse_min_cell=5)
    assert any(nm.startswith("rdd_treat(") for nm in [m["name"] for m in d_rdd["col_meta"]])

    # 3. OLS Kernel: rank-deficient df_resid, Koenker studentized BP, Wild Cluster Bootstrap, Heckman categorical Z
    x_dup = np.column_stack([np.ones(n_gwr), x1_g, 2.0 * x1_g])
    ols_rd = fit_ols_core(x_dup, y_g, cov_type="HC3")
    assert ols_rd["rank"] == 2 and ols_rd["df_resid"] == n_gwr - 2

    y_het = 2.0 + 1.5 * x1_g + rng.normal(0.0, 0.05 + 1.5 * (x1_g - float(np.min(x1_g))))
    ols_h = fit_ols_core(np.column_stack([np.ones(n_gwr), x1_g]), y_het, cov_type="HC3")
    diag_h = run_guard1_diagnostics(np.column_stack([np.ones(n_gwr), x1_g]), y_het, ols_h, meta_g[:2])
    assert diag_h["koenker_lm"] > 0.0 and diag_h["koenker_p"] < 0.05 and diag_h["het_detected"] is True

    wcb_res = run_wild_cluster_bootstrap(d_did["X"][:, :3], d_did["y"], np.asarray(clust_vals), n_boot=99, seed=7)
    assert wcb_res["triggered"] is True and wcb_res["n_clusters"] == 8 and wcb_res["weight_dist"] == "webb"
    assert len(wcb_res["p_values"]) == 3 and all(0.0 < pv <= 1.0 for pv in wcb_res["p_values"])

    sel_ind = (np.asarray(cols_s["ext_flag"]) < 0.5).astype(np.float64)
    a_idx = np.where(sel_ind >= 0.5)[0]
    heck = run_heckman_two_step(
        cols_s, sel_ind, a_idx, np.column_stack([np.ones(len(a_idx)), x_cont_s[a_idx]]),
        np.log(y_raw_s[a_idx]), meta_g[:2], first_stage_covs=["btype", "area"], exclusion_cols=[],
    )
    assert "activated" in heck
    if heck["activated"]:
        assert isinstance(heck["imr_unreliable"], bool) and isinstance(heck["rho_clamped"], bool)

    # 4. GIS: drive vs bike contraflow/maxspeed, GeometryCollection, bridge abutment connectivity, segment-vs-route parity
    n_map = [
        {"type": "node", "id": 1, "lat": 47.500, "lon": 19.000},
        {"type": "node", "id": 2, "lat": 47.500, "lon": 19.002},
        {"type": "node", "id": 3, "lat": 47.500, "lon": 19.004},
        {"type": "node", "id": 10, "lat": 47.500, "lon": 19.002},   # co-located non-ground node at abutment!
        {"type": "node", "id": 11, "lat": 47.4995, "lon": 19.002},
    ]
    ways_map = [
        {"type": "way", "id": 10, "nodes": [1, 2, 3], "tags": {"highway": "residential", "oneway": "yes", "oneway:bicycle": "no", "maxspeed": "50"}},
        # Underpass layer=-1 crossing node 2 visually without connecting in street graph, EXCEPT via explicit abutment
        {"type": "way", "id": 11, "nodes": [10, 11], "tags": {"highway": "footway", "tunnel": "yes", "layer": "-1"}},
    ]
    g_drive = build_street_graph({"elements": n_map + ways_map}, proj_s, "drive")
    g_bike = build_street_graph({"elements": n_map + ways_map}, proj_s, "bike")
    assert g_drive["edge_speed_kmh"][0] >= 40.0
    # Bike can travel reverse 2 -> 1 because of oneway:bicycle=no, while drive cannot
    assert g_drive["g_fwd"][1, 0] == 0.0 and g_bike["g_fwd"][1, 0] > 0.0

    # Grade-separated way is disconnected unless bridged by an explicit top-level role='abutment' connector
    abut_ways = ways_map + [
        {"type": "way", "id": 12, "role": "abutment", "nodes": [2, 10], "tags": {"highway": "footway"}},
    ]
    g_no_abut = build_street_graph({"elements": n_map + ways_map}, proj_s, "walk")
    g_with_abut = build_street_graph({"elements": n_map + abut_ways}, proj_s, "walk")
    from scipy.sparse.csgraph import connected_components
    n_cc_iso, _ = connected_components(g_no_abut["g_fwd"], directed=False)
    n_cc_ab, _ = connected_components(g_with_abut["g_fwd"], directed=False)
    assert n_cc_iso == 2 and n_cc_ab == 1

    # GeometryCollection & unified segment-vs-route parity + travel_time_min + W_net
    gc_gj = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "id": "gc1",
            "properties": {"name": "GC Hub"},
            "geometry": {
                "type": "GeometryCollection",
                "geometries": [{"type": "Point", "coordinates": [19.004, 47.500]}],
            },
        }],
    }
    gc_feats = parse_geojson_features(gc_gj, "gc")
    assert len(gc_feats) == 1
    p_orig_x, p_orig_y = proj_s.forward(47.500, 19.001)  # midpoint on segment 1-2
    tgt_gc_cfg = {"id": "gc", "geometry_handling": "point", "inside_is_zero": False, "distance_modes": "both", "max_snap_distance_m": 250.0}
    b_res = compute_target_distances(np.array([[float(p_orig_x), float(p_orig_y)]]), np.array([True]), gc_feats, tgt_gc_cfg, proj_s, g_bike)
    s_rt = compute_single_route((float(p_orig_x), float(p_orig_y)), gc_feats, tgt_gc_cfg, proj_s, g_bike)
    assert round(float(b_res["dist_net"][0]), 2) == pytest.approx(round(float(s_rt["dist_net_m"]), 2), abs=1e-6)
    assert math.isfinite(float(b_res["travel_time_min"][0])) and s_rt["snap_reason"] == "OK" and s_rt["travel_time_min"] is not None

    pts_wn = np.column_stack([
        [float(proj_s.forward(47.500, 19.0005)[0]), float(proj_s.forward(47.500, 19.0020)[0]), float(proj_s.forward(47.500, 19.0035)[0])],
        [float(proj_s.forward(47.500, 19.0005)[1]), float(proj_s.forward(47.500, 19.0020)[1]), float(proj_s.forward(47.500, 19.0035)[1])],
    ])
    nd_mat = compute_pairwise_network_distances(pts_wn, np.ones(3, dtype=bool), g_bike, cutoff_m=600.0)
    assert nd_mat.nnz >= 4

    # 5. Spatial Kernel: QML SE, Anselin-Kelejian, SAR/SEM/SDM param tables, adaptive_bisquare, network_idw, fit_ml_sem
    sp_res = run_spatial_pipeline(
        d_did["X"][:, :3], d_did["y"], ["const", "area", "area^2"], obs_s,
        [{"id": "knn5", "type": "knn", "k": 5}, {"id": "ab15", "type": "adaptive_bisquare", "k": 15}],
        krinsky_robb_draws=60, se_mode="qml",
    )
    assert sp_res["ok"] is True and sp_res["se_mode"] == "qml"
    assert len(sp_res["sar_params"]) >= 2 and len(sp_res["sem_params"]) >= 2 and len(sp_res["sdm_params"]) >= 2
    assert any(r.get("resid_moran_method") == "anselin_kelejian_1997" for r in sp_res["model_comparison"])
    sem_ml = fit_ml_sem(d_did["X"][:, :3], d_did["y"], build_spatial_weight_matrix(obs_s, {"id": "ab15", "type": "adaptive_bisquare", "k": 15}), se_mode="ml")
    assert sem_ml["se_mode"] == "ml"

    # 6. Exporters & Engine DAG E2E: invalidation, remove_target, reset_session, robustness_matrix, lightweight exports, ZIP QML/SVG
    assert _latex_escape("a\nb<=c—d") == r"a b$\le$c---d"
    sp_tex = generate_spatial_params_latex(sp_res)
    assert "\\begin{tablenotes}" in sp_tex and "QML" in sp_tex

    sess = GeoSpecSession()
    sess.load_dataset(cols_s, "lat", "lon")
    assert sess.ingest_target_raw("hub", {"type": "FeatureCollection", "features": [{"type": "Feature", "id": "h1", "geometry": {"type": "Point", "coordinates": [19.05, 47.50]}}]}, "geojson")["n_features"] == 1
    e2e_osm = {
        "elements": [
            {"type": "node", "id": i + 1, "lat": float(lat_s[i]), "lon": float(lon_s[i])} for i in range(n_s)
        ] + [{"type": "way", "id": 999, "nodes": list(range(1, n_s + 1)), "tags": {"highway": "residential"}}]
    }
    sess.ingest_street_graph("walk", e2e_osm)
    recipe_s = {
        "title": "Round4 Test",
        "dataset_schema": {
            "row_id_col": "id",
            "lat_col": "lat",
            "lon_col": "lon",
            "y_spec": {"mode": "column", "num_col": "price", "transform": "log"},
            "cluster_spec": {"mode": "column", "col": "clust"},
        },
        "spatial_targets": [{
            "id": "hub", "name": "Hub", "source": "osm", "geometry_handling": "point",
            "distance_modes": "both", "route_profile": "walk", "max_snap_distance_m": 2000.0,
        }],
        "guards_cfg": {"sparse_min_cell": 5, "common_sample_lock": True, "auto_exclude_missing_stage1": True},
        "default_cov_type": "cluster",
        "models": [
            {"id": "m1", "name": "M1", "active": True, "covariates": [{"col": "area", "type": "continuous", "transform": "raw"}], "spatial_terms": [], "interactions": []},
            {"id": "m2", "name": "M2", "active": True, "covariates": [{"col": "area", "type": "continuous", "transform": "raw"}], "spatial_terms": [{"target_id": "hub", "transform": "ln_d", "metric": "eucl"}], "interactions": []},
        ],
        "spatial_engine": {
            "active_model_id": "m2",
            "se_mode": "qml",
            "krinsky_robb_draws": 60,
            "weights": [
                {"id": "knn5", "type": "knn", "k": 5, "dedup_colocated": False},
                {"id": "wnet", "type": "network_idw", "power": 1.0, "cutoff_mode": "fixed", "cutoff_m": 5000.0, "profile": "walk"},
            ],
            "gwr": {"mode": "gwr", "focal_model_id": "m2", "b_min_floor": 12, "mc_iters": 19},
        },
    }
    r1 = sess.run_stage_1_distances(recipe_s)
    assert r1["ok"] and "hub" in r1["snap_status_by_target"] and "median_time_min" in r1["summary"][0]
    r2 = sess.run_stage_2_descriptives_and_composition(recipe_s)
    assert r2["ok"]
    r3 = sess.run_stage_3_ols_workbench(recipe_s, cov_type="cluster")
    assert r3["ok"] and r3["models"][1]["wild_bootstrap"]["triggered"] is True
    r4 = sess.run_stage_4_spatial_econometrics(recipe_s)
    assert r4["ok"] and any(w["w_id"] == "wnet" for w in r4["weights_robustness"])
    gwr_prog_a, gwr_prog_b = [], []
    r5a = sess.run_stage_5_gwr(recipe_s, progress_cb=lambda f, m: gwr_prog_a.append(f))
    r5b = sess.run_stage_5_gwr(recipe_s, progress_cb=lambda f, m: gwr_prog_b.append(f))
    assert r5a["ok"] and r5b["ok"]
    assert len(gwr_prog_a) >= 5 and len(gwr_prog_b) <= 1 and len(sess.cache._store["gwr"]) == 1

    # Running Stage 2 preserves downstream Stage 3-5 outputs!
    sess.run_stage_2_descriptives_and_composition(recipe_s)
    assert "stage4" in sess.last_outputs and "stage5" in sess.last_outputs

    rob = sess.run_robustness_matrix(recipe_s)
    assert rob["ok"] and len(rob["rows"]) == 6
    assert "\\begin{tablenotes}" in generate_robustness_matrix_latex(rob)

    assert "\\begin{table}" in sess.export_latex_only(recipe_s, "robustness")["latex"]
    assert "\\begin{table}" in sess.export_latex_only(recipe_s, "stage4_params")["latex"]
    assert sess.export_geojson_only(recipe_s)["geojson"]["type"] == "FeatureCollection"

    zip_res = sess.export_publication_bundle(recipe_s, svg_figures={"test_svg": "<svg xmlns='http://www.w3.org/2000/svg'></svg>"})
    assert zip_res["ok"]
    with zipfile.ZipFile(io.BytesIO(zip_res["zip_bytes"]), "r") as zf:
        names = set(zf.namelist())
        assert "spatial/style_lisa.qml" in names
        assert "spatial/style_gwr_r2.qml" in names
        assert "figures/test_svg.svg" in names
        assert "latex/spatial_params.tex" in names
        assert "latex/robustness_matrix.tex" in names

    # Invalidation lifecycle: remove_target & reset_session
    sess.remove_target("hub")
    assert "stage3" not in sess.last_outputs and "hub" not in sess.dist_results
    sess.reset_session()
    assert sess.n_rows == 0 and sess.proj is None


_GI_DIR = Path(__file__).resolve().parents[2] / "geospec_input"


@pytest.mark.skipif(
    not (_GI_DIR / "kobanya_dataset.csv").exists() or not (_GI_DIR / "leopoldstadt_dataset.csv").exists(),
    reason="Local gitignored geospec_input city assets not present in environment",
)
def test_golden_cross_city_replication_and_best_practice():
    """Golden Cross-City Suite: Kőbánya & Leopoldstadt M2 paper parity, live GeoJSON parity, Best-Practice E2E."""
    import csv
    import io
    import json
    import zipfile

    def _load_csv_cols(csv_path: Path):
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            cols = {h: [] for h in (reader.fieldnames or [])}
            for row in reader:
                for h in cols:
                    cols[h].append(row[h])
            return cols

    kb_bp = json.loads((_GI_DIR / "kobanya_recipe.json").read_text(encoding="utf-8"))
    kb_rep = json.loads((_GI_DIR / "kobanya_paper_replication_recipe.json").read_text(encoding="utf-8"))
    assert kb_bp["guards_cfg"]["sparse_min_cell"] == 5
    assert kb_bp["dataset_schema"]["cluster_spec"]["mode"] == "coordinate_radius"
    assert kb_bp["spatial_engine"]["weights"][0]["dedup_colocated"] is True
    assert kb_rep["guards_cfg"]["sparse_min_cell"] == 0
    assert kb_rep["default_cov_type"] == "HC3"

    expected_m2 = {
        "kobanya": {"n": 269, "k": 20, "r2": 0.6667, "adj_r2": 0.6413},
        "leopoldstadt": {"n": 378, "k": 17, "r2": 0.7418, "adj_r2": 0.7304},
    }
    for city, exp in expected_m2.items():
        cols = _load_csv_cols(_GI_DIR / f"{city}_dataset.csv")
        rep = json.loads((_GI_DIR / f"{city}_paper_replication_recipe.json").read_text(encoding="utf-8"))
        sess = GeoSpecSession()
        sess.load_dataset(cols, "wgs84_lat", "wgs84_long")
        s1 = sess.run_stage_1_distances(rep)
        assert s1["ok"]
        s3 = sess.run_stage_3_ols_workbench(rep)
        assert s3["ok"]
        m2 = next(m for m in s3["models"] if m["model_id"] == "m2_rings")
        assert m2["n_obs"] == exp["n"]
        assert m2["k_params"] == exp["k"]
        assert m2["r2"] == pytest.approx(exp["r2"], abs=0.002)
        assert m2["adj_r2"] == pytest.approx(exp["adj_r2"], abs=0.002)
        assert m2["cov_type_used"] == "HC3"

        # Verify live unclipped GeoJSON Euclidean distance parity vs CSV columns (<= 0.5 m median)
        for gj_name, csv_col in [(f"{city}_railways.geojson", "dist_vagany_m"), (f"{city}_roads.geojson", "dist_fout_m")]:
            feats = parse_geojson_features(json.loads((_GI_DIR / gj_name).read_text(encoding="utf-8")), csv_col)
            vm = sess.valid_coord_mask
            res = compute_target_distances(
                sess.obs_xy, vm, feats, {"id": csv_col, "geometry_handling": "line", "distance_modes": "euclidean"}, sess.proj, None
            )
            csv_v = np.array([float(v) if str(v).strip() != "" else np.nan for v in cols[csv_col]])
            ok_m = vm & np.isfinite(csv_v)
            med_err = float(np.median(np.abs(res["dist_eucl"][ok_m] - csv_v[ok_m])))
            assert med_err <= 0.5

    # Full Best-Practice E2E on Kőbánya Stages 1-5 + ZIP bundle
    kb_cols = _load_csv_cols(_GI_DIR / "kobanya_dataset.csv")
    sess_bp = GeoSpecSession()
    sess_bp.load_dataset(kb_cols, "wgs84_lat", "wgs84_long")
    assert sess_bp.run_stage_1_distances(kb_bp)["ok"]
    assert sess_bp.run_stage_2_descriptives_and_composition(kb_bp)["ok"]
    assert sess_bp.run_stage_3_ols_workbench(kb_bp)["ok"]
    assert sess_bp.run_stage_4_spatial_econometrics(kb_bp)["ok"]
    assert sess_bp.run_stage_5_gwr(kb_bp)["ok"]
    z_bp = sess_bp.export_publication_bundle(kb_bp)
    assert z_bp["ok"]
    with zipfile.ZipFile(io.BytesIO(z_bp["zip_bytes"]), "r") as zf:
        assert "experiment_recipe.json" in zf.namelist()
        assert "spatial/style_gwr_r2.qml" in zf.namelist()


