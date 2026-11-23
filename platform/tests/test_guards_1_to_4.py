"""Unit tests for Design Matrix, Guard #1-4 (HC3 Leverage, Sparse Merge, Gelbach, Heckman, Cluster, AD MLE)."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_engine.design_matrix import (
    build_dependent_variable,
    build_model_design,
    generate_m1_m4_specs,
    parse_numeric_array,
)
from py_engine.exporters import generate_multi_model_latex
from py_engine.ols_kernel import (
    build_cluster_ids,
    compute_semi_log_effects,
    fit_ols_core,
    run_guard1_diagnostics,
    run_heckman_two_step,
    run_stepwise_and_gelbach,
    scan_andersson_day_profile,
)


def test_parse_numeric_and_boxcox() -> None:
    arr = parse_numeric_array(["1\xa0250,5", "980.25", "NA", "", 42])
    assert abs(arr[0] - 1250.5) < 1e-9
    assert abs(arr[1] - 980.25) < 1e-9
    assert np.isnan(arr[2]) and np.isnan(arr[3])
    assert abs(arr[4] - 42.0) < 1e-9

    rng = np.random.default_rng(7)
    y_pos = np.exp(10.0 + rng.normal(0.0, 0.25, size=200))
    y_info = build_dependent_variable(
        {"price": y_pos.tolist()},
        {"mode": "column", "num_col": "price", "transform": "boxcox"},
    )
    assert y_info["boxcox_info"] is not None
    # True lognormal => lambda_mle close to 0 and p_vs_log > 0.05
    assert abs(y_info["boxcox_info"]["lambda_mle"]) < 0.35
    assert y_info["boxcox_info"]["p_vs_log"] > 0.05


def test_guard1_leverage_proxy_and_kennedy() -> None:
    rng = np.random.default_rng(11)
    n = 60
    x1 = rng.normal(size=n)
    x2 = np.zeros(n)
    x2[0] = 1.0  # Single-observation spike -> hat_diag[0] == 1.0!
    X = np.column_stack([np.ones(n), x1, x2])
    y = 2.0 + 0.5 * x1 - 0.3 * x2 + rng.normal(0, 0.1, size=n)

    ols = fit_ols_core(X, y, cov_type="HC3", leverage_warn=0.85)
    assert ols["cov_type_used"] in ("HC2", "HC1")
    assert np.all(np.isfinite(ols["se"]))
    assert any("Leverage" in w for w in ols["cov_warnings"])

    # Kennedy vs Halvorsen-Palmquist check
    pct = compute_semi_log_effects(0.20, 0.10, is_binary=True, is_log_scale=True)
    assert pct["hp_pct"] > pct["kennedy_pct"]

    # Polycentric spatial proxy multicollinearity check:
    # Construct proxy strongly correlated with d_foc (r > 0.9) and orthogonal to y (r ~ 0)
    d_foc = rng.uniform(50, 800, size=n)
    yc = y - np.mean(y)
    d_orth = d_foc - (np.dot(d_foc - np.mean(d_foc), yc) / np.dot(yc, yc)) * yc
    proxy_cbd = d_orth + rng.normal(0, 5, size=n)
    diag = run_guard1_diagnostics(
        X, y, ols,
        [{"name": "const"}, {"name": "x1"}, {"name": "x2"}],
        spatial_proxy_cols={"dist_cbd": proxy_cbd},
        focal_dist=("rail", d_foc),
    )
    assert any(a["kind"] == "polycentric_multicollinearity" for a in diag["proxy_alerts"])


def test_guard2_sparse_ring_and_category_auto_merge() -> None:
    n = 80
    rng = np.random.default_rng(19)
    # Only 2 points in [0, 150m], 25 in (150, 300m], 25 in (300, 500m], 28 > 500m
    d_vals = np.concatenate([
        np.array([40.0, 90.0]),
        rng.uniform(160.0, 290.0, size=25),
        rng.uniform(310.0, 490.0, size=25),
        rng.uniform(520.0, 1000.0, size=28),
    ])
    cats = ["good"] * 45 + ["new"] * 32 + ["rare_a"] * 2 + ["rare_b"] * 1
    rare_bin = [1] * 2 + [0] * (n - 2)
    y_vals = np.exp(12.0 + rng.normal(0, 0.1, size=n))

    columns = {"price": y_vals.tolist(), "cond": cats, "rare_bin": rare_bin}
    dist_dict = {"rail": {"dist_eucl": d_vals}}
    y_info = build_dependent_variable(columns, {"mode": "column", "num_col": "price", "transform": "log"})

    spec = {
        "id": "m_test",
        "name": "Sparse Test",
        "covariates": [
            {"col": "cond", "type": "categorical", "reference_level": "good"},
            {"col": "rare_bin", "type": "numeric"},
        ],
        "spatial_terms": [
            {"target_id": "rail", "metric": "eucl", "transform": "rings", "breakpoints_m": [150.0, 300.0, 500.0]},
        ],
        "interactions": [
            {"left": "rare_bin", "right": "cond"},
        ],
    }
    design = build_model_design(columns, dist_dict, spec, y_info, np.ones(n, dtype=bool), sparse_min_cell=5)
    names = [m["name"] for m in design["col_meta"]]
    # Sparse [0, 150m] (N=2) merged into [0, 300m]; rare_bin (N1=2) dropped + interaction cascaded; rare_a/b merged into ref
    assert "ring_rail_0_300" in names
    assert "rare_bin" not in names
    assert len(design["guard2_events"]) >= 3
    assert len(design["footnotes"]) >= 2


def test_guard3_gelbach_identity_and_ad_profile_mle() -> None:
    rng = np.random.default_rng(101)
    n = 300
    d_rail = rng.uniform(25.0, 900.0, size=n)
    # Worse physical condition near railway creates composition attenuation
    cond_poor = (d_rail < 250.0).astype(float) * 0.6 + rng.binomial(1, 0.15, size=n) * 0.4
    cond_poor = (cond_poor > 0.5).astype(float)
    area = 60.0 + 0.03 * d_rail + rng.normal(0, 12.0, size=n)
    area = np.maximum(20.0, area)

    ad_true = np.maximum(0.0, np.log(300.0 / np.maximum(1.0, d_rail)))
    ln_y = 10.0 - 0.18 * ad_true - 0.25 * cond_poor + 0.005 * area + rng.normal(0, 0.06, size=n)
    y = np.exp(ln_y)

    columns = {"price": y.tolist(), "area": area.tolist(), "poor": cond_poor.tolist()}
    dist_dict = {"rail": {"dist_eucl": d_rail}}
    y_info = build_dependent_variable(columns, {"mode": "column", "num_col": "price", "transform": "log"})

    m4_specs = generate_m1_m4_specs(
        [{"col": "area", "type": "numeric"}, {"col": "poor", "type": "numeric"}],
        [{"id": "rail", "distance_modes": "euclidean"}],
        default_cutoff_m=300.0,
    )
    assert len(m4_specs) == 4
    m4 = m4_specs[3]

    # Gelbach exactness check: sum(delta_g) == beta_uncontrolled - beta_controlled
    chain_cfg = {
        "enabled": True,
        "focal_term_ref": "rail::andersson_day",
        "blocks": [{"name": "Size", "terms": ["area"]}, {"name": "Condition", "terms": ["poor"]}],
        "compute_gelbach_ovb": True,
    }
    res_g = run_stepwise_and_gelbach(columns, dist_dict, m4, chain_cfg, y_info, np.ones(n, dtype=bool))
    assert res_g["ok"] is True
    gap = res_g["beta_uncontrolled"] - res_g["beta_controlled"]
    assert abs(res_g["gelbach_sum_check"] - gap) < 1e-4

    # AD Profile-MLE scan recovers cutoff near 300 m
    prof = scan_andersson_day_profile(
        columns, dist_dict, m4, y_info, np.ones(n, dtype=bool),
        target_term_idx=0, c_min=150.0, c_max=500.0, step=25.0,
    )
    assert prof["ok"] is True
    assert prof["c_ci_95"][0] <= 300.0 <= prof["c_ci_95"][1]


def test_guard4_coordinate_cluster_and_heckman_and_latex() -> None:
    rng = np.random.default_rng(77)
    n_tot = 200
    area = rng.uniform(30, 100, size=n_tot)
    z_excl = rng.normal(0, 1, size=n_tot)
    # Selection depends on exclusion restriction z_excl
    s_prob = 1.0 / (1.0 + np.exp(-(0.8 + 1.1 * z_excl)))
    is_geo = (rng.uniform(size=n_tot) < s_prob).astype(float)
    # Guarantee enough excluded
    is_geo[:40] = 0.0
    is_geo[40:100] = 1.0

    # Cluster coordinates into ~40 buildings within 5 m
    b_xy = rng.uniform(0, 1000, size=(40, 2))
    coords_xy = b_xy[np.arange(n_tot) % 40] + rng.normal(0, 2.0, size=(n_tot, 2))

    y = 10.0 + 0.02 * area + rng.normal(0, 0.2, size=n_tot)
    columns = {"y": y.tolist(), "area": area.tolist(), "z_excl": z_excl.tolist()}
    analytic_idx = np.where(is_geo == 1.0)[0]

    # Coordinate union-find clustering
    c_ids, c_meta = build_cluster_ids(columns, coords_xy, analytic_idx, {"mode": "auto_or_column", "radius_m": 15.0})
    assert c_meta["method"] == "coordinate_radius"
    assert c_meta["n_clusters"] <= 45

    X2 = np.column_stack([np.ones(len(analytic_idx)), area[analytic_idx]])
    y2 = y[analytic_idx]
    ols_cl = fit_ols_core(X2, y2, cov_type="cluster", cluster_ids=c_ids)
    assert ols_cl["cov_type_used"] == "cluster"

    heck = run_heckman_two_step(
        columns,
        is_geo,
        analytic_idx,
        X2,
        y2,
        [{"name": "const", "label": "Intercept"}, {"name": "area", "label": "area"}],
        ["area"],
        ["z_excl"],
    )
    assert heck["activated"] is True
    assert heck["has_exclusion_restriction"] is True
    assert np.isfinite(heck["imr_se"]) and heck["imr_se"] > 0.0

    # Verify LaTeX exporter emits valid booktabs table
    tex = generate_multi_model_latex([
        {
            "model_id": "m1",
            "model_name": "Spec A",
            "n_obs": len(y2),
            "r2": round(ols_cl["r2"], 4),
            "adj_r2": round(ols_cl["adj_r2"], 4),
            "aic": round(ols_cl["aic"], 2),
            "cov_type_used": "cluster",
            "is_log_scale": True,
            "diagnostics": {"max_vif": 1.0, "reset_p": 0.45},
            "footnotes": ["Guard #2 test footnote."],
            "params": [{"name": "const", "beta": float(ols_cl["beta"][0]), "se": float(ols_cl["se"][0]), "p": float(ols_cl["p"][0]), "kennedy_pct": None}],
        }
    ])
    assert "\\toprule" in tex and "\\bottomrule" in tex and "Guard \\#2" in tex
