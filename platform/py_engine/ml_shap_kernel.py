"""
GeoSpec Stage 6 Kernel — Pure-NumPy Deterministic GBRT & Exact Lundberg TreeSHAP
==================================================================================
Zero external ML dependencies (runs bit-for-bit identically in Pyodide WASM
NumPy+SciPy and native Python). Implements:
- Deterministic L2-regularized CART regression trees & gradient-boosted ensembles
- Exact path-dependent Lundberg TreeSHAP (Algorithm 2) with Local Accuracy checks
- Leakage-free building-cluster GroupKFold CV benchmark (OLS, Ridge-GCV, GBRT)
- Domain-grouped feature attribution (Physical Quality, Rail [Track/Station], Road)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RegressionTree:
    """Compact array representation of a deterministic binary CART regression tree."""

    feature: List[int]       # feature index >= 0 for split nodes, -1 for leaves
    threshold: List[float]   # split threshold <= go left
    left: List[int]          # left child node idx, -1 for leaves
    right: List[int]         # right child node idx, -1 for leaves
    value: List[float]       # weighted node prediction (leaf prediction at leaves)
    cover: List[float]       # training sample count reaching each node
    n_features: int


def _fit_regression_tree(
    X: np.ndarray,
    residuals: np.ndarray,
    sample_indices: np.ndarray,
    *,
    max_depth: int = 3,
    min_samples_leaf: int = 5,
    l2_reg: float = 1.0,
    lr_scale: float = 1.0,
) -> RegressionTree:
    """Fit a deterministic CART tree using exact greedy L2 gain splitting."""
    n_features = X.shape[1]
    feature_l: List[int] = []
    threshold_l: List[float] = []
    left_l: List[int] = []
    right_l: List[int] = []
    value_l: List[float] = []
    cover_l: List[float] = []

    def _build(idxs: np.ndarray, depth: int) -> int:
        node_id = len(feature_l)
        n_node = int(idxs.size)
        g_sum = float(np.sum(residuals[idxs]))
        leaf_val = float(lr_scale * g_sum / (n_node + l2_reg)) if n_node > 0 else 0.0
        mean_val = float(lr_scale * g_sum / n_node) if n_node > 0 else 0.0

        feature_l.append(-1)
        threshold_l.append(0.0)
        left_l.append(-1)
        right_l.append(-1)
        value_l.append(leaf_val)
        cover_l.append(float(n_node))

        if depth >= max_depth or n_node < 2 * min_samples_leaf:
            return node_id

        parent_score = (g_sum * g_sum) / (n_node + l2_reg)
        best_gain = 1e-12
        best_feat = -1
        best_thresh = 0.0
        best_left_mask: Optional[np.ndarray] = None

        r_sub = residuals[idxs]
        x_sub = X[idxs]

        for j in range(n_features):
            col = x_sub[:, j]
            order = np.argsort(col, kind="mergesort")
            col_s = col[order]
            r_s = r_sub[order]

            if col_s[-1] <= col_s[0] + 1e-12:
                continue

            # Candidate split positions where sorted feature strictly increases
            diff_mask = (col_s[1:] - col_s[:-1]) > 1e-12
            split_positions = np.nonzero(diff_mask)[0]
            if split_positions.size == 0:
                continue

            # Filter to respect min_samples_leaf on both sides
            valid_pos = split_positions[
                (split_positions + 1 >= min_samples_leaf)
                & (n_node - (split_positions + 1) >= min_samples_leaf)
            ]
            if valid_pos.size == 0:
                continue

            cum_g = np.cumsum(r_s)
            g_l = cum_g[valid_pos]
            n_l = (valid_pos + 1).astype(np.float64)
            g_r = g_sum - g_l
            n_r = float(n_node) - n_l
            gains = (g_l * g_l) / (n_l + l2_reg) + (g_r * g_r) / (n_r + l2_reg) - parent_score

            local_best_k = int(np.argmax(gains))
            g_best = float(gains[local_best_k])
            if g_best > best_gain + 1e-12:
                pos = int(valid_pos[local_best_k])
                best_gain = g_best
                best_feat = j
                best_thresh = 0.5 * float(col_s[pos] + col_s[pos + 1])
                best_left_mask = col <= best_thresh

        if best_feat < 0 or best_left_mask is None:
            return node_id

        left_idxs = idxs[best_left_mask]
        right_idxs = idxs[~best_left_mask]
        if left_idxs.size < min_samples_leaf or right_idxs.size < min_samples_leaf:
            return node_id

        left_id = _build(left_idxs, depth + 1)
        right_id = _build(right_idxs, depth + 1)

        feature_l[node_id] = best_feat
        threshold_l[node_id] = best_thresh
        left_l[node_id] = left_id
        right_l[node_id] = right_id
        # Internal node weighted value (by leaf cover weights) for TreeSHAP consistency
        c_l = cover_l[left_id]
        c_r = cover_l[right_id]
        c_tot = c_l + c_r
        value_l[node_id] = (c_l * value_l[left_id] + c_r * value_l[right_id]) / c_tot if c_tot > 0 else mean_val
        return node_id

    _build(sample_indices, 0)
    return RegressionTree(
        feature=feature_l,
        threshold=threshold_l,
        left=left_l,
        right=right_l,
        value=value_l,
        cover=cover_l,
        n_features=n_features,
    )


def _predict_tree(tree: RegressionTree, X: np.ndarray) -> np.ndarray:
    """Vectorized prediction of a single CART regression tree."""
    n = X.shape[0]
    out = np.empty(n, dtype=np.float64)
    feat = tree.feature
    thr = tree.threshold
    left = tree.left
    right = tree.right
    val = tree.value

    if feat[0] == -1:
        out.fill(val[0])
        return out

    def _assign(mask_idx: np.ndarray, node: int) -> None:
        if mask_idx.size == 0:
            return
        f = feat[node]
        if f == -1:
            out[mask_idx] = val[node]
            return
        go_left = X[mask_idx, f] <= thr[node]
        _assign(mask_idx[go_left], left[node])
        _assign(mask_idx[~go_left], right[node])

    _assign(np.arange(n, dtype=np.int64), 0)
    return out


@dataclass
class GBRTModel:
    """Fitted pure-NumPy Gradient Boosted Regression Tree ensemble."""

    init_value: float
    trees: List[RegressionTree]
    n_features: int
    hyperparams: Dict[str, Any]

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = np.full(X.shape[0], self.init_value, dtype=np.float64)
        for tree in self.trees:
            pred += _predict_tree(tree, X)
        return pred


def fit_gbrt(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 150,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    min_samples_leaf: int = 5,
    l2_reg: float = 1.0,
    random_seed: int = 42,
) -> GBRTModel:
    """Fit a deterministic pure-NumPy L2 Gradient Boosted Regression Tree ensemble."""
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n_obs, n_features = X_arr.shape
    init_value = float(np.mean(y_arr))
    pred = np.full(n_obs, init_value, dtype=np.float64)

    rng = np.random.Generator(np.random.PCG64(int(random_seed)))
    sub_n = n_obs if subsample >= 0.999 else max(min(n_obs, min_samples_leaf * 2), int(round(subsample * n_obs)))
    all_idx = np.arange(n_obs, dtype=np.int64)

    trees: List[RegressionTree] = []
    for _ in range(int(n_estimators)):
        resid = y_arr - pred
        if sub_n < n_obs:
            sample_idx = np.sort(rng.choice(all_idx, size=sub_n, replace=False))
        else:
            sample_idx = all_idx
        tree = _fit_regression_tree(
            X_arr,
            resid,
            sample_idx,
            max_depth=int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            l2_reg=float(l2_reg),
            lr_scale=float(learning_rate),
        )
        pred += _predict_tree(tree, X_arr)
        trees.append(tree)

    return GBRTModel(
        init_value=init_value,
        trees=trees,
        n_features=n_features,
        hyperparams={
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": float(subsample),
            "min_samples_leaf": int(min_samples_leaf),
            "l2_reg": float(l2_reg),
            "random_seed": int(random_seed),
        },
    )


def _tree_shap_single_row(tree: RegressionTree, x_row: np.ndarray, phi: np.ndarray) -> None:
    """
    Exact Path-Dependent Lundberg et al. (2020) TreeSHAP Algorithm 2 on 1 observation.
    Adds tree contributions in-place into 1D vector `phi` (length n_features).
    """
    feat = tree.feature
    thr = tree.threshold
    left = tree.left
    right = tree.right
    val = tree.value
    cover = tree.cover

    if feat[0] == -1:
        return

    # Separate feature metadata list [(feat, pz, po), ...] from subset-size weights [w_0, ..., w_{l-1}]
    def _extend(
        m_meta: List[Tuple[int, float, float]],
        m_w: List[float],
        pz: float,
        po: float,
        pi: int,
    ) -> Tuple[List[Tuple[int, float, float]], List[float]]:
        l_len = len(m_meta)
        new_meta = m_meta + [(pi, pz, po)]
        new_w = list(m_w) + [1.0 if l_len == 0 else 0.0]
        for i in range(l_len - 1, -1, -1):
            new_w[i + 1] += po * new_w[i] * (i + 1) / (l_len + 1)
            new_w[i] = pz * new_w[i] * (l_len - i) / (l_len + 1)
        return new_meta, new_w

    def _unwind_weights(
        m_w: List[float],
        pz: float,
        po: float,
        l_len: int,
    ) -> List[float]:
        # Reverses an EXTEND of (pz, po) when the resulting path had (l_len + 1) items (weights 0..l_len).
        unw_w = [0.0] * l_len
        n = m_w[l_len]
        if po > 0.0:
            for j in range(l_len - 1, -1, -1):
                t = m_w[j]
                wj = n * (l_len + 1) / ((j + 1) * po)
                unw_w[j] = wj
                n = t - wj * pz * (l_len - j) / (l_len + 1)
        else:
            for j in range(l_len - 1, -1, -1):
                unw_w[j] = (m_w[j] * (l_len + 1)) / (pz * (l_len - j))
        return unw_w

    def _unwind(
        m_meta: List[Tuple[int, float, float]],
        m_w: List[float],
        k_idx: int,
    ) -> Tuple[List[Tuple[int, float, float]], List[float]]:
        l_len = len(m_meta) - 1
        _, pz, po = m_meta[k_idx]
        unw_w = _unwind_weights(m_w, pz, po, l_len)
        unw_meta = [m_meta[i] for i in range(len(m_meta)) if i != k_idx]
        return unw_meta, unw_w

    def _recurse(
        node: int,
        m_meta: List[Tuple[int, float, float]],
        m_w: List[float],
        pz: float,
        po: float,
        pi: int,
    ) -> None:
        meta_m, w_m = _extend(m_meta, m_w, pz, po, pi)
        f = feat[node]
        if f == -1:
            v = val[node]
            l_len = len(meta_m) - 1
            for i in range(1, len(meta_m)):
                mi, zi, oi = meta_m[i]
                w_sum = sum(_unwind_weights(w_m, zi, oi, l_len))
                phi[mi] += w_sum * (oi - zi) * v
            return

        l_child = left[node]
        r_child = right[node]
        hot = l_child if x_row[f] <= thr[node] else r_child
        cold = r_child if hot == l_child else l_child
        c_tot = cover[node]
        r_hot = cover[hot] / c_tot if c_tot > 0 else 0.5
        r_cold = cover[cold] / c_tot if c_tot > 0 else 0.5

        iz = 1.0
        io = 1.0
        k_match = -1
        for i in range(1, len(meta_m)):
            if meta_m[i][0] == f:
                iz = meta_m[i][1]
                io = meta_m[i][2]
                k_match = i
                break
        if k_match >= 0:
            next_meta, next_w = _unwind(meta_m, w_m, k_match)
        else:
            next_meta, next_w = meta_m, w_m

        _recurse(hot, next_meta, next_w, iz * r_hot, io, f)
        _recurse(cold, next_meta, next_w, iz * r_cold, 0.0, f)

    _recurse(0, [], [], 1.0, 1.0, -1)


def compute_exact_treeshap(model: GBRTModel, X: np.ndarray) -> Tuple[float, np.ndarray, float]:
    """
    Compute exact path-dependent TreeSHAP values for an entire matrix X.
    Uses exact split-signature deduplication per tree (zero approximation:
    TreeSHAP depends on x solely via binary branch outcomes `x[f] <= thr`).
    Returns (expected_base_value, shap_matrix [N x K], local_accuracy_max_abs_err).
    """
    X_arr = np.asarray(X, dtype=np.float64)
    n_obs, n_features = X_arr.shape
    shap_mat = np.zeros((n_obs, n_features), dtype=np.float64)
    base_value = float(model.init_value)

    if n_obs == 0:
        return base_value, shap_mat, 0.0

    for tree in model.trees:
        base_value += float(tree.value[0])
        if tree.feature[0] == -1:
            continue
        internal_nodes = [u for u, f in enumerate(tree.feature) if f >= 0]
        feats_u = np.array([tree.feature[u] for u in internal_nodes], dtype=np.int64)
        thrs_u = np.array([tree.threshold[u] for u in internal_nodes], dtype=np.float64)
        sig_mat = X_arr[:, feats_u] <= thrs_u  # shape (N, n_internal)

        # Deduplicate rows sharing identical binary split decisions across this tree
        seen: Dict[bytes, np.ndarray] = {}
        for i in range(n_obs):
            sig_key = sig_mat[i].tobytes()
            phi_u = seen.get(sig_key)
            if phi_u is None:
                phi_u = np.zeros(n_features, dtype=np.float64)
                _tree_shap_single_row(tree, X_arr[i], phi_u)
                seen[sig_key] = phi_u
            shap_mat[i] += phi_u

    y_pred = model.predict(X_arr)
    reconstructed = base_value + np.sum(shap_mat, axis=1)
    max_abs_err = float(np.max(np.abs(reconstructed - y_pred)))
    return base_value, shap_mat, max_abs_err


def build_group_kfold_splits(
    cluster_ids: Optional[Sequence[Any]],
    n_obs: int,
    n_splits: int = 5,
) -> Tuple[List[np.ndarray], int]:
    """
    Deterministic balanced GroupKFold partition with zero building-cluster leakage.
    Clusters sorted descending by count (tie-broken by cluster id) are bin-packed into folds.
    Returns (list of test index arrays, effective n_clusters).
    """
    if cluster_ids is None or len(cluster_ids) != n_obs:
        ids = list(range(n_obs))
    else:
        ids = [str(c) for c in cluster_ids]

    groups: Dict[str, List[int]] = {}
    for idx, cid in enumerate(ids):
        groups.setdefault(cid, []).append(idx)

    n_clusters = len(groups)
    k = max(2, min(int(n_splits), n_clusters))
    sorted_groups = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    folds: List[List[int]] = [[] for _ in range(k)]
    fold_sizes = [0] * k
    for _cid, member_idxs in sorted_groups:
        target_f = min(range(k), key=lambda f: (fold_sizes[f], f))
        folds[target_f].extend(member_idxs)
        fold_sizes[target_f] += len(member_idxs)

    return [np.sort(np.asarray(f, dtype=np.int64)) for f in folds], n_clusters


def _fit_ridge_gcv(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    """Standardized closed-form Ridge regression with zero-leakage GCV alpha selection inside train fold."""
    mu_x = np.mean(X_tr, axis=0)
    sd_x = np.std(X_tr, axis=0)
    sd_x = np.where(sd_x > 1e-12, sd_x, 1.0)
    Z_tr = (X_tr - mu_x) / sd_x
    Z_te = (X_te - mu_x) / sd_x
    y_mean = float(np.mean(y_tr))
    yc = y_tr - y_mean

    n, p = Z_tr.shape
    gram = Z_tr.T @ Z_tr
    zty = Z_tr.T @ yc
    eigvals, _ = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, 0.0)

    best_gcv = math.inf
    best_beta = np.zeros(p, dtype=np.float64)
    for alpha in (0.05, 0.25, 1.0, 5.0, 20.0, 80.0):
        try:
            beta = np.linalg.solve(gram + alpha * np.eye(p), zty)
        except np.linalg.LinAlgError:
            beta, *_ = np.linalg.lstsq(gram + alpha * np.eye(p), zty, rcond=None)
        pred_tr = Z_tr @ beta
        mse = float(np.mean((yc - pred_tr) ** 2))
        df_eff = float(np.sum(eigvals / (eigvals + alpha)))
        denom = max(1e-6, 1.0 - df_eff / max(1, n))
        gcv = mse / (denom * denom)
        if gcv < best_gcv:
            best_gcv = gcv
            best_beta = beta

    return y_mean + Z_te @ best_beta


def _eval_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, is_log_scale: bool = True) -> Dict[str, Optional[float]]:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    rmse = math.sqrt(ss_res / max(1, len(y_true)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape_pct: Optional[float] = None
    if is_log_scale:
        yt_clamped = np.clip(y_true, -50.0, 50.0)
        yp_clamped = np.clip(y_pred, -50.0, 50.0)
        orig_y = np.exp(yt_clamped)
        orig_p = np.exp(yp_clamped)
        nz = np.abs(orig_y) > 1e-12
        if np.any(nz):
            mape_pct = float(np.mean(np.abs((orig_y[nz] - orig_p[nz]) / orig_y[nz])) * 100.0)
    return {
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape_pct": round(mape_pct, 2) if mape_pct is not None else None,
    }


def categorize_feature_group(feature_name: str) -> str:
    """Classify covariate name into physical vs. rail vs. road infrastructure groups."""
    low = str(feature_name).lower()
    if "vagany" in low or "rail" in low or "track" in low or "allomas" in low or "station" in low:
        return "rail"
    if "fout" in low or "road" in low:
        return "road"
    return "physical"


def run_ml_shap_pipeline(
    X_design: np.ndarray,
    y: np.ndarray,
    col_names: List[str],
    *,
    cluster_ids: Optional[Sequence[Any]] = None,
    valid_indices: Optional[np.ndarray] = None,
    is_log_scale: bool = True,
    ml_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute Stage 6 ML & Exact TreeSHAP pipeline on a model design matrix:
      1. Drop intercept column if present (`col_names[0] == "const"`).
      2. Run 5-fold building-cluster GroupKFold CV for OLS, Ridge-GCV, and GBRT.
      3. Fit full-sample deterministic GBRT (fixed random_seed) & compute Exact TreeSHAP.
      4. Verify Local Accuracy axiom and aggregate physical / rail / road shares.
    """
    cfg = ml_cfg or {}
    n_estimators = max(10, min(500, int(cfg.get("n_estimators", 150))))
    max_depth = max(1, min(6, int(cfg.get("max_depth", 3))))
    learning_rate = max(0.005, min(0.5, float(cfg.get("learning_rate", 0.05))))
    subsample = max(0.3, min(1.0, float(cfg.get("subsample", 0.8))))
    min_samples_leaf = max(2, min(50, int(cfg.get("min_samples_leaf", 5))))
    l2_reg = max(0.0, min(100.0, float(cfg.get("l2_reg", 1.0))))
    cv_folds = max(2, min(10, int(cfg.get("cv_folds", 5))))
    random_seed = int(cfg.get("random_seed", 42))

    X_arr = np.asarray(X_design, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    names = list(col_names)
    if names and names[0] == "const":
        X_feat = X_arr[:, 1:]
        feat_names = names[1:]
    else:
        X_feat = X_arr
        feat_names = names

    n_obs, n_features = X_feat.shape
    if n_obs < 15 or n_features < 1:
        return {"ok": False, "reason": f"Kevés megfigyelés (N={n_obs}) vagy változó (K={n_features}) a Stage 6 ML-hez."}

    folds, n_clusters = build_group_kfold_splits(cluster_ids, n_obs, n_splits=cv_folds)
    actual_folds = len(folds)

    oof_ols = np.zeros(n_obs, dtype=np.float64)
    oof_ridge = np.zeros(n_obs, dtype=np.float64)
    oof_gbrt = np.zeros(n_obs, dtype=np.float64)

    all_mask = np.ones(n_obs, dtype=bool)
    for test_idx in folds:
        tr_mask = all_mask.copy()
        tr_mask[test_idx] = False
        X_tr, y_tr = X_feat[tr_mask], y_arr[tr_mask]
        X_te = X_feat[test_idx]

        # 1. OLS
        X_tr_aug = np.column_stack([np.ones(X_tr.shape[0]), X_tr])
        X_te_aug = np.column_stack([np.ones(X_te.shape[0]), X_te])
        beta_ols, *_ = np.linalg.lstsq(X_tr_aug, y_tr, rcond=None)
        oof_ols[test_idx] = X_te_aug @ beta_ols

        # 2. Ridge-GCV
        oof_ridge[test_idx] = _fit_ridge_gcv(X_tr, y_tr, X_te)

        # 3. Pure-NumPy GBRT
        m_fold = fit_gbrt(
            X_tr,
            y_tr,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_samples_leaf=min_samples_leaf,
            l2_reg=l2_reg,
            random_seed=random_seed,
        )
        oof_gbrt[test_idx] = m_fold.predict(X_te)

    # Full-sample fits for in-sample R2 & Exact TreeSHAP
    X_aug = np.column_stack([np.ones(n_obs), X_feat])
    beta_full_ols, *_ = np.linalg.lstsq(X_aug, y_arr, rcond=None)
    in_ols = _eval_regression_metrics(y_arr, X_aug @ beta_full_ols, is_log_scale=is_log_scale)
    in_ridge = _eval_regression_metrics(y_arr, _fit_ridge_gcv(X_feat, y_arr, X_feat), is_log_scale=is_log_scale)

    full_gbrt = fit_gbrt(
        X_feat,
        y_arr,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        min_samples_leaf=min_samples_leaf,
        l2_reg=l2_reg,
        random_seed=random_seed,
    )
    full_pred = full_gbrt.predict(X_feat)
    in_gbrt = _eval_regression_metrics(y_arr, full_pred, is_log_scale=is_log_scale)

    base_value, shap_mat, local_acc_err = compute_exact_treeshap(full_gbrt, X_feat)

    # Feature importance & directional stats
    mean_abs = np.mean(np.abs(shap_mat), axis=0)
    total_abs = float(np.sum(mean_abs))
    feature_rows: List[Dict[str, Any]] = []
    for j, fname in enumerate(feat_names):
        ma = float(mean_abs[j])
        rel_pct = (ma / total_abs * 100.0) if total_abs > 1e-12 else 0.0
        sv_j = shap_mat[:, j]
        xv_j = X_feat[:, j]
        if np.std(sv_j) > 1e-12 and np.std(xv_j) > 1e-12:
            corr_val = float(np.corrcoef(xv_j, sv_j)[0, 1])
        else:
            corr_val = 0.0
        feature_rows.append({
            "feature": fname,
            "group": categorize_feature_group(fname),
            "mean_abs_shap": round(ma, 5),
            "rel_pct": round(rel_pct, 2),
            "mean_signed_shap": round(float(np.mean(sv_j)), 5),
            "corr_x_shap": round(corr_val, 3),
        })

    feature_rows.sort(key=lambda r: (-r["mean_abs_shap"], r["feature"]))
    for rk, row in enumerate(feature_rows, start=1):
        row["rank"] = rk

    # Exact group totals (computed from unrounded shares so sums reconcile cleanly)
    raw_pcts = [float(mean_abs[j]) / total_abs * 100.0 if total_abs > 1e-12 else 0.0 for j in range(n_features)]
    phys_pct = sum(p for fn, p in zip(feat_names, raw_pcts) if categorize_feature_group(fn) == "physical")
    rail_pct = sum(p for fn, p in zip(feat_names, raw_pcts) if categorize_feature_group(fn) == "rail")
    road_pct = sum(p for fn, p in zip(feat_names, raw_pcts) if categorize_feature_group(fn) == "road")
    rail_track_pct = sum(
        p for fn, p in zip(feat_names, raw_pcts)
        if ("vagany" in fn.lower() or "rail" in fn.lower() or "track" in fn.lower())
    )
    rail_station_pct = sum(
        p for fn, p in zip(feat_names, raw_pcts)
        if ("allomas" in fn.lower() or "station" in fn.lower())
    )

    cv_ols = _eval_regression_metrics(y_arr, oof_ols, is_log_scale=is_log_scale)
    cv_ridge = _eval_regression_metrics(y_arr, oof_ridge, is_log_scale=is_log_scale)
    cv_gbrt = _eval_regression_metrics(y_arr, oof_gbrt, is_log_scale=is_log_scale)

    benchmark = [
        {"model": "OLS Baseline", "cv_r2": cv_ols["r2"], "cv_rmse": cv_ols["rmse"], "cv_mae": cv_ols["mae"], "cv_mape_pct": cv_ols["mape_pct"], "in_sample_r2": in_ols["r2"]},
        {"model": "Ridge (GCV)", "cv_r2": cv_ridge["r2"], "cv_rmse": cv_ridge["rmse"], "cv_mae": cv_ridge["mae"], "cv_mape_pct": cv_ridge["mape_pct"], "in_sample_r2": in_ridge["r2"]},
        {"model": "GBRT (Pure-NumPy)", "cv_r2": cv_gbrt["r2"], "cv_rmse": cv_gbrt["rmse"], "cv_mae": cv_gbrt["mae"], "cv_mape_pct": cv_gbrt["mape_pct"], "in_sample_r2": in_gbrt["r2"]},
    ]
    benchmark_sorted = sorted(benchmark, key=lambda r: r["cv_rmse"])

    # Normalized X [0, 1] per feature for plotting beeswarm / inspection
    beeswarm: Dict[str, Any] = {}
    for j, fname in enumerate(feat_names):
        col = X_feat[:, j]
        c_min, c_max = float(np.min(col)), float(np.max(col))
        x_norm = ((col - c_min) / (c_max - c_min)).tolist() if c_max > c_min + 1e-12 else [0.5] * n_obs
        beeswarm[fname] = {
            "x_norm": [round(float(v), 4) for v in x_norm],
            "shap_vals": [round(float(v), 5) for v in shap_mat[:, j]],
        }

    return {
        "ok": True,
        "engine": "pure_numpy_gbrt_exact_treeshap",
        "n_obs": n_obs,
        "n_features": n_features,
        "n_clusters": n_clusters,
        "cv_folds": actual_folds,
        "random_seed": random_seed,
        "hyperparams": full_gbrt.hyperparams,
        "local_accuracy_max_err": float(local_acc_err),
        "base_value": round(base_value, 6),
        "benchmark": benchmark_sorted,
        "feature_importance": feature_rows,
        "grouped_importance": {
            "physical_pct": round(phys_pct, 2),
            "rail_pct": round(rail_pct, 2),
            "rail_track_pct": round(rail_track_pct, 2),
            "rail_station_pct": round(rail_station_pct, 2),
            "road_pct": round(road_pct, 2),
        },
        "grouped_rows": [
            {"category": "Fizikai épületminőség", "key": "physical", "rel_pct": round(phys_pct, 2)},
            {"category": "Vasúti infrastruktúra (összesen)", "key": "rail", "rel_pct": round(rail_pct, 2)},
            {"category": "  ebből: vágányközelség (zaj)", "key": "rail_track", "rel_pct": round(rail_track_pct, 2)},
            {"category": "  ebből: vasútállomás (elérés)", "key": "rail_station", "rel_pct": round(rail_station_pct, 2)},
            {"category": "Közúti infrastruktúra (főút)", "key": "road", "rel_pct": round(road_pct, 2)},
        ],
        "feature_names": feat_names,
        "beeswarm": beeswarm,
        "y_pred": [round(float(v), 5) for v in full_pred],
        "y_true": [round(float(v), 5) for v in y_arr],
        "valid_indices": valid_indices.tolist() if valid_indices is not None else list(range(n_obs)),
    }
