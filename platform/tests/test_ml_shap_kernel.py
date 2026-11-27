"""Unit and empirical parity tests for GeoSpec Stage 6 (Pure-NumPy GBRT & Exact TreeSHAP)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
_PLATFORM_DIR = _TESTS_DIR.parent
_COLAB_DIR = _PLATFORM_DIR.parent
if str(_PLATFORM_DIR) not in sys.path:
    sys.path.insert(0, str(_PLATFORM_DIR))

from py_engine.engine import GeoSpecSession  # noqa: E402
from py_engine.ml_shap_kernel import (  # noqa: E402
    build_group_kfold_splits,
    compute_exact_treeshap,
    fit_gbrt,
    run_ml_shap_pipeline,
)


def test_exact_treeshap_local_accuracy_and_determinism():
    rng = np.random.default_rng(123)
    n, k = 120, 5
    X = rng.normal(size=(n, k))
    y = 14.0 + 0.35 * X[:, 0] - 0.25 * (X[:, 1] > 0.2) + 0.15 * np.tanh(X[:, 2]) + rng.normal(0, 0.05, n)
    names = ["alapterulet", "d_felujitott", "vagany_dist", "allomas_dist", "fout_dist"]
    clusters = [f"bld_{i // 3}" for i in range(n)]

    m1 = fit_gbrt(X, y, n_estimators=60, max_depth=3, learning_rate=0.08, random_seed=42)
    m2 = fit_gbrt(X, y, n_estimators=60, max_depth=3, learning_rate=0.08, random_seed=42)
    b1, sv1, err1 = compute_exact_treeshap(m1, X)
    b2, sv2, err2 = compute_exact_treeshap(m2, X)

    assert err1 < 1e-10
    assert err2 < 1e-10
    assert b1 == b2
    assert np.array_equal(sv1, sv2)

    out = run_ml_shap_pipeline(X, y, names, cluster_ids=clusters, ml_cfg={"random_seed": 42, "n_estimators": 60})
    assert out["ok"]
    assert out["local_accuracy_max_err"] < 1e-10
    assert abs(sum(r["rel_pct"] for r in out["feature_importance"]) - 100.0) < 0.2


def test_group_kfold_zero_cluster_leakage():
    clusters = [f"c_{i // 4}" for i in range(80)]
    folds, n_cl = build_group_kfold_splits(clusters, len(clusters), n_splits=5)
    assert n_cl == 20
    assert len(folds) == 5
    seen = set()
    for test_idx in folds:
        test_clusters = {clusters[int(i)] for i in test_idx}
        assert seen.isdisjoint(test_clusters), "Cluster leaked across multiple CV folds!"
        seen.update(test_clusters)
    assert len(seen) == 20


def test_sealed_real_study_stage6_empirical_contrast():
    res_by_city = {}
    for city in ("kobanya", "leopoldstadt"):
        rec = json.loads((_COLAB_DIR / "geospec_input" / f"{city}_recipe.json").read_text(encoding="utf-8"))
        with (_COLAB_DIR / "geospec_input" / f"{city}_dataset.csv").open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        cols = {k: [r[k] for r in rows] for k in rows[0].keys()}
        sess = GeoSpecSession()
        sess.load_dataset(cols, rec["dataset_schema"]["lat_col"], rec["dataset_schema"]["lon_col"])
        sess.run_stage_1_distances(rec)
        s6 = sess.run_stage_6_ml_shap(rec)
        assert s6["ok"]
        assert s6["local_accuracy_max_err"] < 1e-10
        res_by_city[city] = s6

    kb_grp = res_by_city["kobanya"]["grouped_importance"]
    vi_grp = res_by_city["leopoldstadt"]["grouped_importance"]
    assert kb_grp["physical_pct"] > 75.0
    assert vi_grp["rail_track_pct"] > 3.0 * kb_grp["rail_track_pct"]
