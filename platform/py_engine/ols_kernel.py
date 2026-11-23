"""
Parametric OLS, HC3/CR1 Cluster SE, Diagnostics, Gelbach OVB & Heckman 2-Step Kernel
=====================================================================================
Pure NumPy + SciPy implementation (zero statsmodels):
- QR/SVD OLS with high-leverage HC3 guard (h_ii > 0.85 -> safe HC2/HC1)
- Liang-Zeger CR1 cluster SE (column, regex, OR coordinate union-find r <= 15m)
- Kennedy (1981) & Halvorsen-Palmquist exact semi-log percentages
- Andersson-Day profile-MLE cutoff scanner with 95% profile-likelihood CI
- Guard #1: VIF, Belsley-Kuh-Welsch kappa, BP, White/Koenker, HC3 RESET, JB, Proxy Confound
- Guard #3: Dynamic Step 0..K + Gelbach (2016) order-independent OVB decomposition
- Guard #4: Exact Heckman (1979) 2-step Probit (log_ndtr) + corrected V_Heckman
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize, special, stats
from scipy.spatial import cKDTree

from .design_matrix import build_model_design, parse_numeric_array


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


def _safe_pinv_xtx(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve OLS via economy QR when full rank, falling back to SVD pseudoinverse.
    Returns (beta, xtx_inv, hat_diag) without allocating N x N matrices.
    """
    n, k = X.shape
    if n <= k:
        raise ValueError(f"Insufficient observations (N={n}) for K={k} parameters.")
    q, r = np.linalg.qr(X, mode="reduced")
    diag_r = np.abs(np.diag(r))
    if np.min(diag_r) > 1e-10 * np.max(diag_r):
        # Full rank fast path
        qty = q.T @ np.zeros(n)  # placeholder replaced below
        r_inv = np.linalg.inv(r)
        xtx_inv = r_inv @ r_inv.T
        hat_diag = np.sum(q * q, axis=1)
        return r_inv, xtx_inv, hat_diag
    # Rank-deficient SVD fallback
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    tol = 1e-10 * s[0] if len(s) > 0 else 1e-12
    keep = s > tol
    s_inv = np.where(keep, 1.0 / np.maximum(s, 1e-15), 0.0)
    xtx_inv = (vt.T * (s_inv ** 2)) @ vt
    hat_diag = np.sum((u[:, keep]) ** 2, axis=1) if np.any(keep) else np.zeros(n)
    return None, xtx_inv, hat_diag


def fit_ols_core(
    X: np.ndarray,
    y: np.ndarray,
    cov_type: str = "HC3",
    cluster_ids: Optional[np.ndarray] = None,
    leverage_warn: float = 0.85,
) -> Dict[str, Any]:
    """
    Fit OLS and compute classic, HC3 (with leverage safety), or CR1 cluster covariance.
    """
    n, k = X.shape
    if n <= k:
        raise ValueError(f"Underdetermined OLS: N={n} <= K={k}.")

    # Solve beta via lstsq (SVD-based, unconditionally stable)
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    fitted = X @ beta
    df_resid = max(1, n - k)
    rss = float(resid @ resid)
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 1e-18 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid if df_resid > 0 else 0.0
    sigma2 = rss / df_resid
    llf = -0.5 * n * (math.log(2.0 * math.pi) + math.log(max(rss / n, 1e-18)) + 1.0)
    aic = 2.0 * k - 2.0 * llf
    bic = k * math.log(n) - 2.0 * llf
    aicc = aic + (2.0 * k * (k + 1)) / max(1, n - k - 1)

    q, r = np.linalg.qr(X, mode="reduced")
    diag_r = np.abs(np.diag(r)) if k > 0 else np.array([1.0])
    if rank == k and np.min(diag_r) > 1e-10 * max(1e-12, float(np.max(diag_r))):
        r_inv = np.linalg.inv(r)
        xtx_inv = r_inv @ r_inv.T
        hat = np.sum(q * q, axis=1)
    else:
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        tol = 1e-10 * (s[0] if len(s) > 0 else 1.0)
        keep = s > tol
        s_inv = np.where(keep, 1.0 / np.maximum(s, 1e-15), 0.0)
        xtx_inv = (vt.T * (s_inv ** 2)) @ vt
        hat = np.sum(u[:, keep] ** 2, axis=1) if np.any(keep) else np.zeros(n)

    max_hat = float(np.max(hat)) if n > 0 else 0.0
    actual_cov = cov_type
    cov_warnings: List[str] = []
    df_infer = df_resid
    g_count: Optional[int] = None

    if cov_type == "cluster" and cluster_ids is None:
        cov_type = "HC3"
        cov_warnings.append("Cluster IDs unavailable; fell back to HC3 robust SEs.")

    if cov_type == "cluster" and cluster_ids is not None:
        uniq_g, inv_idx = np.unique(cluster_ids, return_inverse=True)
        g_count = len(uniq_g)
        if g_count <= 1:
            actual_cov = "HC3"
            cov_warnings.append("Cluster SE fallback to HC3: only 1 cluster found.")
        else:
            df_infer = max(1, g_count - 1)
            scores = X * resid[:, None]
            g_scores = np.zeros((g_count, k), dtype=np.float64)
            np.add.at(g_scores, inv_idx, scores)
            meat = g_scores.T @ g_scores
            cr1_scale = (g_count / (g_count - 1.0)) * ((n - 1.0) / df_resid)
            vcov = cr1_scale * (xtx_inv @ meat @ xtx_inv)
            if g_count < 30:
                cov_warnings.append(
                    f"Small cluster count (G={g_count} < 30); cluster-robust inference may over-reject (anti-conservative; interpret p-values cautiously)."
                )
    elif cov_type in ("HC3", "HC2", "HC1"):
        if cov_type == "HC3" and (max_hat > leverage_warn or np.any(1.0 - hat < 1e-6)):
            # Guard #1 high-leverage protection: switch to HC2 if 1 - h_ii is safe, else HC1
            if np.all(1.0 - hat > 1e-4):
                actual_cov = "HC2"
                w_hc = (resid ** 2) / (1.0 - hat)
                cov_warnings.append(
                    f"Guard #1 High-Leverage Alert (max h_ii={max_hat:.3f} > {leverage_warn}): switched HC3 -> HC2."
                )
            else:
                actual_cov = "HC1"
                w_hc = (n / df_resid) * (resid ** 2)
                cov_warnings.append(
                    f"Guard #1 Extreme Leverage (max h_ii={max_hat:.4f} -> 1): switched HC3 -> HC1."
                )
        elif cov_type == "HC2":
            w_hc = (resid ** 2) / np.maximum(1e-4, 1.0 - hat)
        elif cov_type == "HC1":
            w_hc = (n / df_resid) * (resid ** 2)
        else:
            w_hc = (resid ** 2) / ((1.0 - hat) ** 2)
        meat = (X * w_hc[:, None]).T @ X
        vcov = xtx_inv @ meat @ xtx_inv
    else:
        actual_cov = "OLS"
        vcov = sigma2 * xtx_inv

    se = np.sqrt(np.maximum(0.0, np.diag(vcov)))
    t_vals = np.divide(beta, se, out=np.zeros_like(beta), where=se > 1e-15)
    p_vals = 2.0 * stats.t.sf(np.abs(t_vals), df=df_infer)
    t_crit = float(stats.t.ppf(0.975, df=df_infer))
    ci_lo = beta - t_crit * se
    ci_hi = beta + t_crit * se

    # Omnibus F-test (excluding intercept col 0)
    if k > 1:
        if actual_cov == "cluster" and g_count is not None and (g_count - 1) < (k - 1):
            f_stat, f_pval = float("nan"), float("nan")
            cov_warnings.append(
                f"Cluster count (G={g_count}) <= K parameters ({k}); omnibus F-test omitted due to rank-deficient cluster covariance."
            )
        else:
            b_sub = beta[1:]
            v_sub = vcov[1:, 1:]
            try:
                if np.linalg.cond(v_sub) > 1e12:
                    raise np.linalg.LinAlgError("Ill-conditioned covariance submatrix.")
                wald = float(b_sub.T @ np.linalg.solve(v_sub, b_sub))
                f_stat = wald / (k - 1)
                f_pval = float(stats.f.sf(f_stat, k - 1, df_infer))
            except np.linalg.LinAlgError:
                f_stat, f_pval = float("nan"), float("nan")
    else:
        f_stat, f_pval = float("nan"), float("nan")

    # Cook's distance
    cooks_d = (resid ** 2 / max(1e-18, k * sigma2)) * (hat / np.maximum(1e-8, (1.0 - hat) ** 2))

    return {
        "beta": beta,
        "se": se,
        "t": t_vals,
        "p": p_vals,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "vcov": vcov,
        "xtx_inv": xtx_inv,
        "resid": resid,
        "fitted": fitted,
        "hat": hat,
        "max_hat": max_hat,
        "cooks_d": cooks_d,
        "n": n,
        "k": k,
        "df_resid": df_resid,
        "df_infer": df_infer,
        "rss": rss,
        "sigma2": sigma2,
        "r2": r2,
        "adj_r2": adj_r2,
        "llf": llf,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "f_stat": f_stat,
        "f_pval": f_pval,
        "cov_type_used": actual_cov,
        "cov_warnings": cov_warnings,
    }


def compute_semi_log_effects(beta: float, se: float, is_binary: bool, is_log_scale: bool) -> Dict[str, Optional[float]]:
    """Compute exact Halvorsen-Palmquist & Kennedy (1981) variance-corrected % effects for dummies."""
    if not is_log_scale or not is_binary:
        return {"hp_pct": None, "kennedy_pct": None}
    hp = (math.exp(beta) - 1.0) * 100.0
    kennedy = (math.exp(beta - 0.5 * (se ** 2)) - 1.0) * 100.0
    return {"hp_pct": round(hp, 3), "kennedy_pct": round(kennedy, 3)}


def build_cluster_ids(
    columns: Dict[str, List[Any]],
    coords_xy: np.ndarray,
    valid_indices: np.ndarray,
    cluster_spec: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Build universal cluster IDs for valid observations:
    1) Column (with optional regex street-number stripping)
    2) Coordinate-radius union-find (building clustering within radius_m)
    """
    mode = cluster_spec.get("mode", "auto_or_column")
    col = cluster_spec.get("col")
    regex = cluster_spec.get("regex")
    radius_m = float(cluster_spec.get("radius_m", 15.0))
    n = len(valid_indices)

    if mode in ("column", "auto_or_column") and col and col in columns:
        raw_vals = columns[col]
        labels = []
        for idx in valid_indices:
            s = str(raw_vals[idx]).strip() if raw_vals[idx] is not None else f"row_{idx}"
            if not s:
                s = f"row_{idx}"
            elif regex:
                s_sub = re.sub(regex, "", s).strip()
                s = s_sub if s_sub else s
            labels.append(s)
        arr = np.asarray(labels)
        g_count = len(np.unique(arr))
        if g_count < n or mode == "column":
            return arr, {"method": "column", "col": col, "n_clusters": g_count}

    # Coordinate-radius union-find fallback
    if coords_xy is not None and len(coords_xy) >= n:
        pts = coords_xy[valid_indices]
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

        tree = cKDTree(pts)
        for i, j in tree.query_pairs(r=radius_m):
            _union(int(i), int(j))
        roots = np.array([_find(i) for i in range(n)], dtype=np.int32)
        g_count = len(np.unique(roots))
        return roots, {"method": "coordinate_radius", "radius_m": radius_m, "n_clusters": g_count}

    return None, {"method": "none", "n_clusters": n}


def run_guard1_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    ols_res: Dict[str, Any],
    col_meta: List[Dict[str, Any]],
    spatial_proxy_cols: Optional[Dict[str, np.ndarray]] = None,
    focal_dist: Optional[Tuple[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Guard #1: 5-fold econometric diagnostics (VIF, BKW kappa, BP, White/Koenker,
    HC3 Ramsey RESET p=2,3, Jarque-Bera) + Spatial Confounding proxy correlation guard.
    """
    n, k = X.shape
    resid = ols_res["resid"]
    fitted = ols_res["fitted"]
    df_resid = ols_res["df_resid"]

    # 1. VIF per non-intercept column (using inv(corr) if well-conditioned, else auxiliary OLS R^2)
    vifs: List[Dict[str, Any]] = []
    collinear_cols: List[str] = []
    max_vif = 1.0
    X_no_int = X[:, 1:]
    p_no_int = X_no_int.shape[1]
    if p_no_int == 1:
        cname = col_meta[1]["name"]
        is_col = float(np.std(X_no_int[:, 0])) <= 1e-12
        vj = 1e6 if is_col else 1.0
        if is_col:
            collinear_cols.append(cname)
        max_vif = vj
        vifs.append({"name": cname, "vif": round(vj, 3), "collinear": is_col})
    elif p_no_int >= 2:
        Xc = X_no_int - np.mean(X_no_int, axis=0)
        stds = np.std(Xc, axis=0)
        used_inv = False
        if np.all(stds > 1e-12) and n > p_no_int + 1:
            Xs = Xc / stds
            corr = (Xs.T @ Xs) / n
            try:
                eigs = np.linalg.eigvalsh(corr)
                if float(np.min(eigs)) > 1e-8:
                    vif_diag = np.diag(np.linalg.inv(corr))
                    for j, val in enumerate(vif_diag):
                        vj = float(max(1.0, val))
                        is_col = vj >= 1e6
                        if is_col:
                            collinear_cols.append(col_meta[j + 1]["name"])
                        max_vif = max(max_vif, vj)
                        vifs.append({"name": col_meta[j + 1]["name"], "vif": round(vj, 3), "collinear": is_col})
                    used_inv = True
            except np.linalg.LinAlgError:
                used_inv = False

        if not used_inv:
            for j in range(p_no_int):
                cname = col_meta[j + 1]["name"]
                yj = Xc[:, j]
                tss_j = float(yj @ yj)
                if tss_j <= 1e-12:
                    vj = 1e6
                    is_col = True
                else:
                    other = np.delete(Xc, j, axis=1)
                    b_j, _, _, _ = np.linalg.lstsq(other, yj, rcond=None)
                    res_j = yj - other @ b_j
                    one_minus_r2 = float(res_j @ res_j) / tss_j
                    if one_minus_r2 <= 1e-6:
                        vj = 1e6
                        is_col = True
                    else:
                        vj = float(max(1.0, 1.0 / one_minus_r2))
                        is_col = False
                if is_col:
                    collinear_cols.append(cname)
                max_vif = max(max_vif, vj)
                vifs.append({"name": cname, "vif": round(vj, 3), "collinear": is_col})

    # 2. Belsley-Kuh-Welsch Condition Number kappa(X_scaled)
    col_norms = np.linalg.norm(X, axis=0)
    X_normed = X / np.where(col_norms > 1e-12, col_norms, 1.0)
    svals = np.linalg.svd(X_normed, compute_uv=False)
    kappa = float(svals[0] / max(1e-12, svals[-1])) if len(svals) > 0 else 1.0

    # 3. Breusch-Pagan & Koenker-Bassett LM test
    e2 = resid ** 2
    e2_norm = e2 / max(1e-18, float(np.mean(e2)))
    bp_fit = fit_ols_core(X, e2_norm, cov_type="OLS")
    tss_e2 = float(np.sum((e2_norm - np.mean(e2_norm)) ** 2))
    ess_e2 = max(0.0, tss_e2 - bp_fit["rss"])
    bp_lm = 0.5 * ess_e2
    bp_p = float(stats.chi2.sf(bp_lm, df=max(1, k - 1)))

    # Koenker studentized LM (n * R2_aux)
    e2_c = e2 - np.mean(e2)
    e2_fit = fit_ols_core(X, e2, cov_type="OLS")
    koenker_lm = float(n * e2_fit["r2"])
    koenker_p = float(stats.chi2.sf(koenker_lm, df=max(1, k - 1)))

    # 4. White test (with Koenker fallback if quadratic terms >= N/2)
    white_cols = [X[:, 0]]
    for j in range(1, k):
        for m in range(j, k):
            prod = X[:, j] * X[:, m]
            if np.ptp(prod) > 1e-12:
                white_cols.append(prod)
    for j in range(1, k):
        if np.ptp(X[:, j]) > 1e-12:
            white_cols.append(X[:, j])
    X_wh = np.column_stack(white_cols)
    # Remove duplicate columns via SVD rank check
    if X_wh.shape[1] < max(4, n // 2):
        wh_fit = fit_ols_core(X_wh, e2, cov_type="OLS")
        _, _, rank_wh, _ = np.linalg.lstsq(X_wh, e2, rcond=None)
        wh_df = max(1, int(rank_wh) - 1)
        white_lm = float(n * wh_fit["r2"])
        white_p = float(stats.chi2.sf(white_lm, df=wh_df))
        white_mode = "White"
    else:
        white_lm, white_p, white_mode = koenker_lm, koenker_p, "Koenker-Bassett (df-safe fallback)"

    # 5. Ramsey RESET (power=2,3) with HC3 covariance
    yhat_std = (fitted - np.mean(fitted)) / max(1e-12, float(np.std(fitted)))
    X_reset = np.column_stack([X, yhat_std ** 2, yhat_std ** 3])
    try:
        reset_fit = fit_ols_core(X_reset, y, cov_type="HC3")
        b_aug = reset_fit["beta"][k:]
        v_aug = reset_fit["vcov"][k:, k:]
        wald_r = float(b_aug.T @ np.linalg.solve(v_aug, b_aug))
        reset_f = wald_r / 2.0
        reset_p = float(stats.f.sf(reset_f, 2, max(1, n - k - 2)))
    except Exception:
        reset_f, reset_p = float("nan"), float("nan")

    # 6. Jarque-Bera Normality on Residuals
    skew = float(stats.skew(resid))
    kurt_excess = float(stats.kurtosis(resid, fisher=True))
    jb_stat = (n / 6.0) * (skew ** 2 + 0.25 * (kurt_excess ** 2))
    jb_p = float(stats.chi2.sf(jb_stat, df=2))

    # 7. Spatial Confounding Proxy Guard
    proxy_alerts: List[Dict[str, Any]] = []
    if spatial_proxy_cols and focal_dist is not None:
        focal_name, d_foc = focal_dist
        for pname, pvals in spatial_proxy_cols.items():
            m = np.isfinite(pvals) & np.isfinite(d_foc) & np.isfinite(y)
            if np.sum(m) >= 15 and np.ptp(pvals[m]) > 1e-12 and np.ptp(d_foc[m]) > 1e-12:
                r_conf = float(np.corrcoef(d_foc[m], pvals[m])[0, 1])
                r_out = float(np.corrcoef(y[m], pvals[m])[0, 1])
                if abs(r_out) < 0.10 and abs(r_conf) > 0.60:
                    proxy_alerts.append({
                        "proxy": pname,
                        "target": focal_name,
                        "r_conf": round(r_conf, 3),
                        "r_out": round(r_out, 3),
                        "kind": "polycentric_multicollinearity",
                        "message": (
                            f"Polycentric No-Signal Multicollinearity: '{pname}' correlates strongly with {focal_name} "
                            f"(r={r_conf:+.3f}) but negligibly with y (r={r_out:+.3f}). Including it risks distorting target estimates."
                        ),
                    })
                elif abs(r_out) > 0.25 and abs(r_conf) > 0.40:
                    proxy_alerts.append({
                        "proxy": pname,
                        "target": focal_name,
                        "r_conf": round(r_conf, 3),
                        "r_out": round(r_out, 3),
                        "kind": "omitted_spatial_gradient",
                        "message": (
                            f"Spatial Confounding Risk: '{pname}' correlates with both y (r={r_out:+.3f}) and {focal_name} "
                            f"(r={r_conf:+.3f}). Consider controlling for it or inspecting GWR local coefficients."
                        ),
                    })

    status = "pass"
    if max_vif >= 10.0 or len(collinear_cols) > 0 or kappa >= 30.0 or len(proxy_alerts) > 0 or (math.isfinite(reset_p) and reset_p < 0.05):
        status = "warn"

    return {
        "status": status,
        "max_vif": round(max_vif, 3),
        "vifs": vifs,
        "collinear_cols": collinear_cols,
        "condition_number_kappa": round(kappa, 2),
        "bp_lm": round(bp_lm, 3),
        "bp_p": round(bp_p, 5),
        "white_mode": white_mode,
        "white_lm": round(white_lm, 3),
        "white_p": round(white_p, 5),
        "reset_f": round(reset_f, 3) if math.isfinite(reset_f) else None,
        "reset_p": round(reset_p, 5) if math.isfinite(reset_p) else None,
        "jb_stat": round(jb_stat, 3),
        "jb_p": round(jb_p, 5),
        "max_leverage": round(ols_res["max_hat"], 4),
        "cov_warnings": ols_res["cov_warnings"],
        "proxy_alerts": proxy_alerts,
    }


def scan_andersson_day_profile(
    columns: Dict[str, List[Any]],
    dist_dict: Dict[str, Dict[str, np.ndarray]],
    model_spec: Dict[str, Any],
    y_info: Dict[str, Any],
    base_valid_mask: np.ndarray,
    target_term_idx: int = 0,
    c_min: float = 100.0,
    c_max: float = 600.0,
    step: float = 25.0,
    sparse_min_cell: int = 5,
    override_valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Scan Andersson-Day censoring cutoff c in [c_min, c_max] via profile-MLE,
    returning AICc(c), LL(c), beta(c), 95% CI, and 95% profile-likelihood CI for c*.
    """
    spatial_terms = model_spec.get("spatial_terms", [])
    if target_term_idx >= len(spatial_terms) or spatial_terms[target_term_idx].get("transform") != "andersson_day":
        return {"ok": False, "reason": "No Andersson-Day spatial term found at requested index."}

    want_tid = spatial_terms[target_term_idx].get("target_id")
    want_metric = spatial_terms[target_term_idx].get("metric", "eucl")

    spec_copy = {
        **model_spec,
        "spatial_terms": [dict(st) for st in spatial_terms],
    }
    cutoffs = np.arange(c_min, c_max + 0.5 * step, step, dtype=np.float64)
    curve: List[Dict[str, Any]] = []

    # Lock common sample across scan points where distance is valid
    base_design = build_model_design(columns, dist_dict, spec_copy, y_info, base_valid_mask, sparse_min_cell, override_valid_mask)
    scan_mask = base_design["valid_mask"]
    n_obs = base_design["n_obs"]
    if n_obs < 15:
        return {"ok": False, "reason": "Too few valid observations for threshold scan."}

    for c_val in cutoffs:
        spec_copy["spatial_terms"][target_term_idx]["cutoff_m"] = float(c_val)
        d = build_model_design(columns, dist_dict, spec_copy, y_info, base_valid_mask, sparse_min_cell, scan_mask)
        # Match exact target_id & metric of the scanned AD term (multi-target safe)
        ad_col_idx = None
        for idx_c, meta in enumerate(d["col_meta"]):
            m_info = meta.get("meta", {})
            if (
                meta.get("kind") == "spatial_ad"
                and m_info.get("target_id") == want_tid
                and m_info.get("metric") == want_metric
            ):
                ad_col_idx = idx_c
                break
        if ad_col_idx is None:
            curve.append({"c_m": float(c_val), "llf": None, "aicc": None, "beta": 0.0, "se": 0.0, "t": 0.0, "p": 1.0})
            continue

        ols_r = fit_ols_core(d["X"], d["y"], cov_type="HC3")
        b = float(ols_r["beta"][ad_col_idx])
        se = float(ols_r["se"][ad_col_idx])
        curve.append({
            "c_m": float(c_val),
            "llf": round(float(ols_r["llf"]), 4),
            "aicc": round(float(ols_r["aicc"]), 4),
            "aic": round(float(ols_r["aic"]), 4),
            "r2": round(float(ols_r["r2"]), 4),
            "beta": round(b, 4),
            "se": round(se, 4),
            "ci_lo": round(b - 1.96 * se, 4),
            "ci_hi": round(b + 1.96 * se, 4),
            "t": round(float(ols_r["t"][ad_col_idx]), 3),
            "p": round(float(ols_r["p"][ad_col_idx]), 5),
        })

    valid_pts = [pt for pt in curve if pt["llf"] is not None]
    if not valid_pts:
        return {"ok": False, "reason": "AD covariate had zero variance across all scanned cutoffs."}

    best_pt = max(valid_pts, key=lambda pt: pt["llf"])
    ll_star = best_pt["llf"]
    # 95% profile likelihood confidence set: 2 * (ll* - ll(c)) <= 3.841459
    ci_set = [pt["c_m"] for pt in valid_pts if 2.0 * (ll_star - pt["llf"]) <= 3.841459]

    return {
        "ok": True,
        "c_star_m": best_pt["c_m"],
        "ll_star": ll_star,
        "aicc_star": best_pt["aicc"],
        "c_ci_95": [min(ci_set), max(ci_set)] if ci_set else [best_pt["c_m"], best_pt["c_m"]],
        "n_obs": n_obs,
        "curve": curve,
        "post_selection_note": (
            "Profile-MLE cutoff c* maximizes likelihood over the grid; standard errors at c* treat c as fixed "
            "(post-selection inference caveat applies; inspect the 95% profile-likelihood CI)."
        ),
    }


def run_stepwise_and_gelbach(
    columns: Dict[str, List[Any]],
    dist_dict: Dict[str, Dict[str, np.ndarray]],
    full_model_spec: Dict[str, Any],
    chain_cfg: Dict[str, Any],
    y_info: Dict[str, Any],
    base_valid_mask: np.ndarray,
    sparse_min_cell: int = 5,
    common_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Guard #3: Dynamic Step 0 -> Step K composition chain + Gelbach (2016)
    exact order-independent OVB decomposition (delta_g = Gamma_g @ gamma_g).
    """
    blocks = chain_cfg.get("blocks", [])
    focal_ref = chain_cfg.get("focal_term_ref", "")
    cov_lookup = {c["col"]: c for c in full_model_spec.get("covariates", [])}

    # Build full design first to lock identical sample across Step 0..Step K
    full_design = build_model_design(columns, dist_dict, full_model_spec, y_info, base_valid_mask, sparse_min_cell, common_mask)
    locked_mask = full_design["valid_mask"]
    n_obs = full_design["n_obs"]
    if n_obs < 15:
        return {"ok": False, "reason": "Too few valid observations for Stepwise/Gelbach analysis."}

    # Identify focal column index in full_design
    focal_idx = None
    for idx_c, meta in enumerate(full_design["col_meta"]):
        if idx_c == 0:
            continue
        if focal_ref and (meta["ref"] == focal_ref or meta["name"] == focal_ref):
            focal_idx = idx_c
            break
    if focal_idx is None:
        # Fallback to first spatial term
        for idx_c, meta in enumerate(full_design["col_meta"]):
            if meta["kind"].startswith("spatial"):
                focal_idx = idx_c
                break
    if focal_idx is None:
        return {"ok": False, "reason": "No focal spatial term found for composition chain."}

    focal_label = full_design["col_meta"][focal_idx]["label"]
    X_full = full_design["X"]
    y = full_design["y"]

    # Map each block's terms to columns in X_full
    # Step 0 includes intercept + spatial terms (and focal term)
    spatial_indices = [i for i, m in enumerate(full_design["col_meta"]) if m["kind"].startswith("spatial")]
    if focal_idx not in spatial_indices:
        spatial_indices.append(focal_idx)
    x1_indices = [0] + sorted(set(spatial_indices))

    assigned_cols = set(x1_indices)
    block_col_indices: List[Tuple[str, List[int]]] = []
    for blk in blocks:
        bname = blk.get("name", "Control Block")
        b_terms = set(blk.get("terms", []))
        cols_in_blk = []
        for idx_c, meta in enumerate(full_design["col_meta"]):
            if idx_c in assigned_cols:
                continue
            if meta.get("base_col") in b_terms or meta["ref"] in b_terms or meta["name"] in b_terms:
                cols_in_blk.append(idx_c)
                assigned_cols.add(idx_c)
        if cols_in_blk:
            block_col_indices.append((bname, cols_in_blk))

    # Any remaining covariates from full_model_spec go into a final "Other Controls" block
    leftover = [i for i in range(X_full.shape[1]) if i not in assigned_cols]
    if leftover:
        block_col_indices.append(("Remaining Controls", leftover))

    # Step 0 -> Step K estimation
    steps_out: List[Dict[str, Any]] = []
    cur_indices = list(x1_indices)
    ols_step0 = fit_ols_core(X_full[:, cur_indices], y, cov_type="HC3")
    pos_focal_0 = cur_indices.index(focal_idx)
    beta_0 = float(ols_step0["beta"][pos_focal_0])
    se_0 = float(ols_step0["se"][pos_focal_0])
    p_0 = float(ols_step0["p"][pos_focal_0])

    steps_out.append({
        "step": "Step 0 (Uncontrolled)",
        "added_block": "—",
        "k_params": len(cur_indices),
        "n_obs": n_obs,
        "beta": round(beta_0, 4),
        "se": round(se_0, 4),
        "p": round(p_0, 5),
        "sig": _sig_stars(p_0),
        "r2": round(float(ols_step0["r2"]), 4),
        "adj_r2": round(float(ols_step0["adj_r2"]), 4),
        "attenuation_pct": 0.0,
    })

    for s_idx, (bname, b_cols) in enumerate(block_col_indices, start=1):
        cur_indices.extend(b_cols)
        ols_s = fit_ols_core(X_full[:, cur_indices], y, cov_type="HC3")
        pos_f = cur_indices.index(focal_idx)
        b_s = float(ols_s["beta"][pos_f])
        se_s = float(ols_s["se"][pos_f])
        p_s = float(ols_s["p"][pos_f])
        atten = (1.0 - b_s / beta_0) * 100.0 if abs(beta_0) > 1e-12 else 0.0
        steps_out.append({
            "step": f"Step {s_idx} (+ {bname})",
            "added_block": bname,
            "k_params": len(cur_indices),
            "n_obs": n_obs,
            "beta": round(b_s, 4),
            "se": round(se_s, 4),
            "p": round(p_s, 5),
            "sig": _sig_stars(p_s),
            "r2": round(float(ols_s["r2"]), 4),
            "adj_r2": round(float(ols_s["adj_r2"]), 4),
            "attenuation_pct": round(atten, 2),
        })

    ols_full = fit_ols_core(X_full, y, cov_type="HC3")
    beta_full = float(ols_full["beta"][focal_idx])
    total_diff = beta_0 - beta_full

    # Exact Gelbach (2016) order-independent decomposition:
    # For each block g: H_g = X_{2,g} @ gamma_g; regress H_g on X_1 -> delta_g = coef[pos_focal_0]
    gelbach_rows: List[Dict[str, Any]] = []
    if bool(chain_cfg.get("compute_gelbach_ovb", True)) and block_col_indices:
        X1 = X_full[:, x1_indices]
        gamma_full = ols_full["beta"]
        x1_x1_inv = ols_step0["xtx_inv"]
        # Projection selector row e_f^T (X_1' X_1)^{-1} X_1'
        p_vec = x1_x1_inv[pos_focal_0, :] @ X1.T  # shape (N,)
        # Cross-covariate influence correction for asymptotic SE of delta_g (Gelbach 2016 Eq. 12)
        e_full = ols_full["resid"]
        # M1 residual maker for X1
        for bname, b_cols in block_col_indices:
            H_g = X_full[:, b_cols] @ gamma_full[b_cols]
            delta_g = float(p_vec @ H_g)
            # Auxiliary residual + full residual influence approximation (HC-robust)
            aux_fit = fit_ols_core(X1, H_g, cov_type="HC3")
            u_g = aux_fit["resid"]
            # Full influence for delta_g = A_g * e_full + p_vec * u_g
            # where A_g selects gamma_g block via (X'X)^{-1}
            sel_g = np.zeros(X_full.shape[1], dtype=np.float64)
            # Compute linear combination c_g = sum_{j in b_cols} (p_vec @ X_full[:, j]) * e_j
            proj_x2g = p_vec @ X_full[:, b_cols]  # shape (len(b_cols),)
            c_full = np.zeros(X_full.shape[1], dtype=np.float64)
            c_full[b_cols] = proj_x2g
            # Influence vector psi_i = (c_full' (X'X)^{-1} x_i) * e_full_i + p_vec_i * u_g_i
            w_full = c_full @ ols_full["xtx_inv"] @ X_full.T
            psi_i = w_full * e_full + p_vec * u_g
            se_delta = float(np.sqrt(max(0.0, np.sum(psi_i ** 2))))
            t_delta = delta_g / se_delta if se_delta > 1e-15 else 0.0
            p_delta = float(2.0 * stats.t.sf(abs(t_delta), df=ols_full["df_resid"]))
            share_pct = (delta_g / total_diff * 100.0) if abs(total_diff) > 1e-12 else 0.0
            gelbach_rows.append({
                "block": bname,
                "delta": round(delta_g, 5),
                "se": round(se_delta, 5),
                "t": round(t_delta, 3),
                "p": round(p_delta, 5),
                "sig": _sig_stars(p_delta),
                "share_of_gap_pct": round(share_pct, 2),
            })

    final_atten = (1.0 - beta_full / beta_0) * 100.0 if abs(beta_0) > 1e-12 else 0.0
    sign_reversed = (beta_0 * beta_full) < 0.0 and abs(beta_0) > 1e-6

    return {
        "ok": True,
        "focal_term": focal_label,
        "n_obs": n_obs,
        "beta_uncontrolled": round(beta_0, 5),
        "beta_controlled": round(beta_full, 5),
        "total_ovb_gap": round(total_diff, 5),
        "final_attenuation_pct": round(final_atten, 2),
        "sign_reversed": sign_reversed,
        "steps": steps_out,
        "gelbach": gelbach_rows,
        "gelbach_sum_check": round(sum(r["delta"] for r in gelbach_rows), 5),
    }


def run_heckman_two_step(
    columns: Dict[str, List[Any]],
    selection_indicator: np.ndarray,
    analytic_indices: np.ndarray,
    X_stage2: np.ndarray,
    y_stage2: np.ndarray,
    col_meta_stage2: List[Dict[str, Any]],
    first_stage_covs: List[str],
    exclusion_cols: List[str],
) -> Dict[str, Any]:
    """
    Guard #4: Exact Heckman (1979) 2-Step Selection Model:
    - Stage 1 Probit using scipy.special.log_ndtr (underflow-proof) + L2 penalty fallback
    - Strict row_id index alignment for IMR (zero mean-imputation)
    - Exact Heckman (1979) corrected two-step asymptotic covariance V_Heckman
    """
    n_total = len(selection_indicator)
    n_analytic = len(analytic_indices)
    n_excluded = int(np.sum(selection_indicator == 0))

    # Build Stage 1 Probit matrix Z over rows with complete first-stage covariates
    z_names: List[str] = ["const"]
    z_cols: List[np.ndarray] = [np.ones(n_total, dtype=np.float64)]
    all_probit_vars = list(dict.fromkeys(list(first_stage_covs) + list(exclusion_cols)))
    skipped_non_numeric: List[str] = []

    for col in all_probit_vars:
        if col in columns:
            arr = parse_numeric_array(columns[col])
            n_fin = int(np.sum(np.isfinite(arr)))
            if n_fin > 20 and float(np.nanstd(arr)) > 1e-12:
                z_names.append(col)
                z_cols.append(arr)
            elif n_fin <= 20:
                skipped_non_numeric.append(col)

    if len(z_cols) < 2:
        return {
            "activated": False,
            "reason": "No valid non-constant numeric Stage-1 covariates available for Heckman Probit.",
            "skipped_non_numeric": skipped_non_numeric,
        }

    Z_raw = np.column_stack(z_cols)
    z_valid_mask = np.all(np.isfinite(Z_raw), axis=1)
    # Require all analytic rows to have valid Z; otherwise subset Stage 2 strictly by row_id
    s2_keep_mask = np.array([z_valid_mask[idx] for idx in analytic_indices], dtype=bool)
    if int(np.sum(s2_keep_mask)) < 25:
        return {"activated": False, "reason": "Insufficient non-missing Stage-1 covariates on analytic sample."}

    Z_fit = Z_raw[z_valid_mask]
    s_fit = selection_indicator[z_valid_mask]
    n_sel_fit = int(np.sum(s_fit >= 0.5))
    n_excl_fit = int(np.sum(s_fit < 0.5))
    if n_excl_fit < 5 or n_sel_fit < 15:
        return {
            "activated": False,
            "reason": f"Empty/degenerate selection pool after Stage-1 missingness filter (selected={n_sel_fit}, excluded={n_excl_fit}).",
        }
    fit_global_indices = np.where(z_valid_mask)[0]

    # Standardize non-constant columns of Z_fit for numerically stable Probit optimization
    z_means = np.mean(Z_fit[:, 1:], axis=0) if Z_fit.shape[1] > 1 else np.array([])
    z_stds = np.where(np.std(Z_fit[:, 1:], axis=0) > 1e-12, np.std(Z_fit[:, 1:], axis=0), 1.0) if Z_fit.shape[1] > 1 else np.array([])
    Z_std = Z_fit.copy()
    if Z_fit.shape[1] > 1:
        Z_std[:, 1:] = (Z_fit[:, 1:] - z_means) / z_stds

    def _probit_neg_ll(gamma: np.ndarray, l2_reg: float = 0.0) -> Tuple[float, np.ndarray]:
        eta = np.clip(Z_std @ gamma, -30.0, 30.0)
        log_phi1 = special.log_ndtr(eta)
        log_phi0 = special.log_ndtr(-eta)
        ll = float(np.sum(s_fit * log_phi1 + (1.0 - s_fit) * log_phi0)) - 0.5 * l2_reg * float(gamma[1:] @ gamma[1:])
        # Exact score: (s * phi/Phi - (1-s) * phi/(1-Phi)) * Z
        log_pdf = -0.5 * eta * eta - 0.5 * math.log(2.0 * math.pi)
        mills1 = np.exp(np.clip(log_pdf - log_phi1, -50.0, 50.0))
        mills0 = np.exp(np.clip(log_pdf - log_phi0, -50.0, 50.0))
        d_ll = s_fit * mills1 - (1.0 - s_fit) * mills0
        grad = Z_std.T @ d_ll
        if len(gamma) > 1 and l2_reg > 0.0:
            grad[1:] -= l2_reg * gamma[1:]
        return -ll, -grad

    p0 = float(np.clip(np.mean(s_fit), 0.01, 0.99))
    g_init = np.zeros(Z_std.shape[1], dtype=np.float64)
    g_init[0] = float(stats.norm.ppf(p0))

    res_p = optimize.minimize(
        lambda g: _probit_neg_ll(g, l2_reg=0.0),
        g_init,
        jac=True,
        method="BFGS",
        options={"maxiter": 200},
    )
    used_penalized = False
    if not res_p.success or np.max(np.abs(res_p.x)) > 15.0:
        res_p = optimize.minimize(
            lambda g: _probit_neg_ll(g, l2_reg=1e-3),
            g_init,
            jac=True,
            method="BFGS",
            options={"maxiter": 200},
        )
        used_penalized = True

    gamma_hat = res_p.x
    # Probit Fisher information matrix (exact Hessian at MLE)
    eta_hat = np.clip(Z_std @ gamma_hat, -30.0, 30.0)
    log_pdf_hat = -0.5 * eta_hat * eta_hat - 0.5 * math.log(2.0 * math.pi)
    log_phi1_h = special.log_ndtr(eta_hat)
    log_phi0_h = special.log_ndtr(-eta_hat)
    imr_all = np.exp(np.clip(log_pdf_hat - log_phi1_h, -50.0, 50.0))
    imr0_all = np.exp(np.clip(log_pdf_hat - log_phi0_h, -50.0, 50.0))
    # Probit weights w_i = imr1 * imr0
    w_probit = imr_all * imr0_all
    info_z = (Z_std * w_probit[:, None]).T @ Z_std + (1e-3 * np.eye(Z_std.shape[1]) if used_penalized else 1e-9 * np.eye(Z_std.shape[1]))
    var_gamma = np.linalg.pinv(info_z)

    # Strict row_id alignment back to Stage 2 rows
    row_to_fit_pos = {int(g_idx): pos for pos, g_idx in enumerate(fit_global_indices)}
    aligned_pos = [row_to_fit_pos[int(analytic_indices[i])] for i in range(n_analytic) if s2_keep_mask[i]]

    X2 = X_stage2[s2_keep_mask]
    y2 = y_stage2[s2_keep_mask]
    Z2 = Z_std[aligned_pos]
    eta2 = eta_hat[aligned_pos]
    imr2 = imr_all[aligned_pos]

    # Check IMR multicollinearity against Stage-2 regressors (Puhani 2000)
    imr_c = imr2 - float(np.mean(imr2))
    imr_tss = float(imr_c @ imr_c)
    if imr_tss <= 1e-12:
        return {"activated": False, "reason": "Inverse Mills Ratio has zero variance on analytic sample."}
    if X2.shape[1] >= 1:
        b_imr, _, _, _ = np.linalg.lstsq(X2, imr2, rcond=None)
        res_imr = imr2 - X2 @ b_imr
        one_minus_r2_imr = float(res_imr @ res_imr) / imr_tss
        imr_vif = 1e6 if one_minus_r2_imr <= 1e-6 else float(1.0 / one_minus_r2_imr)
    else:
        imr_vif = 1.0

    X_heck = np.column_stack([X2, imr2])
    n2, k2 = X_heck.shape
    if n2 <= k2 + 1:
        return {"activated": False, "reason": f"Insufficient Stage-2 degrees of freedom for Heckman (N={n2}, K={k2})."}
    beta_heck, _, _, _ = np.linalg.lstsq(X_heck, y2, rcond=None)
    e_heck = y2 - X_heck @ beta_heck
    theta_imr = float(beta_heck[-1])

    # Exact Heckman (1979) two-step corrected variance V_Heckman:
    # delta_i = lambda_i * (lambda_i + Z_i' gamma_hat) in (0, 1)
    delta_i = np.clip(imr2 * (imr2 + eta2), 0.0, 0.999)
    sigma2_hat = float(np.mean(e_heck ** 2) + (theta_imr ** 2) * float(np.mean(delta_i)))
    sigma2_hat = max(sigma2_hat, 1e-12)
    rho2 = min(0.99, (theta_imr ** 2) / sigma2_hat)

    xtx_h = X_heck.T @ X_heck
    try:
        xtx_inv_h = np.linalg.inv(xtx_h) if np.linalg.cond(xtx_h) < 1e12 else np.linalg.pinv(xtx_h)
    except np.linalg.LinAlgError:
        xtx_inv_h = np.linalg.pinv(xtx_h)
    # Correction matrix: I - rho^2 * diag(delta_i) + rho^2 * (X' D Z) V_gamma (Z' D X)
    xdx_adj = (X_heck * (1.0 - rho2 * delta_i)[:, None]).T @ X_heck
    xdz = (X_heck * delta_i[:, None]).T @ Z2
    q_corr = rho2 * (xdz @ var_gamma @ xdz.T)
    v_heckman = sigma2_hat * (xtx_inv_h @ (xdx_adj + q_corr) @ xtx_inv_h)
    se_heck = np.sqrt(np.maximum(0.0, np.diag(v_heckman)))
    t_heck = np.where(se_heck > 1e-15, beta_heck / se_heck, 0.0)
    p_heck = 2.0 * stats.t.sf(np.abs(t_heck), df=max(1, n2 - k2))

    # Compare against baseline HC3 OLS on same rows
    ols_base = fit_ols_core(X2, y2, cov_type="HC3")
    rows_comp = []
    for idx_c, meta in enumerate(col_meta_stage2):
        rows_comp.append({
            "name": meta["name"],
            "label": meta["label"],
            "base_beta": round(float(ols_base["beta"][idx_c]), 4),
            "base_se_hc3": round(float(ols_base["se"][idx_c]), 4),
            "base_p": round(float(ols_base["p"][idx_c]), 5),
            "heckman_beta": round(float(beta_heck[idx_c]), 4),
            "heckman_se": round(float(se_heck[idx_c]), 4),
            "heckman_p": round(float(p_heck[idx_c]), 5),
        })
    rows_comp.append({
        "name": "lambda_imr",
        "label": "Inverse Mills Ratio (λ)",
        "base_beta": None,
        "base_se_hc3": None,
        "base_p": None,
        "heckman_beta": round(theta_imr, 4),
        "heckman_se": round(float(se_heck[-1]), 4),
        "heckman_p": round(float(p_heck[-1]), 5),
    })

    has_excl = len([c for c in exclusion_cols if c in z_names]) > 0
    warnings = []
    if skipped_non_numeric:
        warnings.append(
            f"Skipped non-numeric or mostly missing Stage-1 variables: {', '.join(skipped_non_numeric)}."
        )
    if not has_excl:
        warnings.append("Heckman Probit has no explicit exclusion restriction (Z_excl); identified via Probit nonlinearity only.")
    if imr_vif >= 20.0:
        warnings.append(
            f"High IMR collinearity (VIF={imr_vif:.1f} >= 20): Stage-2 identification is weak without an exclusion restriction Z_excl."
        )
    if used_penalized:
        warnings.append("Stage-1 Probit encountered quasi-separation and used L2 stabilization (1e-3).")

    return {
        "activated": True,
        "n_total_pool": int(np.sum(z_valid_mask)),
        "n_analytic": n2,
        "n_excluded": int(np.sum(s_fit == 0)),
        "has_exclusion_restriction": has_excl,
        "imr_beta": round(theta_imr, 4),
        "imr_se": round(float(se_heck[-1]), 4),
        "imr_p": round(float(p_heck[-1]), 5),
        "imr_vif": round(imr_vif, 2),
        "selection_bias_detected": bool(float(p_heck[-1]) < 0.05),
        "rho_selection_est": round(math.sqrt(rho2) * (1.0 if theta_imr >= 0 else -1.0), 4),
        "sigma_est": round(math.sqrt(sigma2_hat), 4),
        "comparison": rows_comp,
        "warnings": warnings,
    }


def compute_descriptive_and_normality(
    columns: Dict[str, List[Any]],
    dist_dict: Dict[str, Dict[str, np.ndarray]],
    y_info: Dict[str, Any],
    valid_mask: np.ndarray,
    covariates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute Stage-2 descriptive statistics, normality tests (K2, Shapiro, JB), and band tests."""
    idx = np.where(valid_mask)[0]
    n = len(idx)
    if n < 5:
        return {"ok": False, "reason": "Insufficient valid observations."}

    def _stats_row(name: str, vals: np.ndarray) -> Dict[str, Any]:
        v = vals[np.isfinite(vals)]
        if len(v) == 0:
            return {"variable": name, "n": 0}
        return {
            "variable": name,
            "n": int(len(v)),
            "mean": round(float(np.mean(v)), 4),
            "sd": round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 4),
            "min": round(float(np.min(v)), 4),
            "q1": round(float(np.percentile(v, 25)), 4),
            "median": round(float(np.median(v)), 4),
            "q3": round(float(np.percentile(v, 75)), 4),
            "max": round(float(np.max(v)), 4),
        }

    desc_rows = [
        _stats_row(y_info["base_label"], y_info["y_raw"][valid_mask]),
        _stats_row(y_info["label"], y_info["y"][valid_mask]),
    ]
    for cov in covariates:
        col = cov.get("col", "")
        if cov.get("type", "numeric") != "categorical" and col in columns:
            arr = parse_numeric_array(columns[col])[valid_mask]
            desc_rows.append(_stats_row(col, arr))

    for tid, tres in dist_dict.items():
        if np.any(np.isfinite(tres["dist_eucl"][valid_mask])):
            desc_rows.append(_stats_row(f"dist_eucl_{tid} (m)", tres["dist_eucl"][valid_mask]))
        if np.any(np.isfinite(tres["dist_net"][valid_mask])):
            desc_rows.append(_stats_row(f"dist_net_{tid} (m)", tres["dist_net"][valid_mask]))
        if np.any(np.isfinite(tres["circuity_delta"][valid_mask])):
            desc_rows.append(_stats_row(f"detour_{tid} (m)", tres["circuity_delta"][valid_mask]))

    # Normality comparison on raw vs transformed y
    normality_rows = []
    for lbl, vec in [(y_info["base_label"], y_info["y_raw"][valid_mask]), (y_info["label"], y_info["y"][valid_mask])]:
        v = vec[np.isfinite(vec)]
        sk = float(stats.skew(v))
        ku = float(stats.kurtosis(v, fisher=True))
        k2_s, k2_p = stats.normaltest(v) if len(v) >= 8 else (float("nan"), float("nan"))
        jb_s, jb_p = stats.jarque_bera(v) if len(v) >= 5 else (float("nan"), float("nan"))
        sw_s, sw_p = stats.shapiro(v[:4500]) if len(v) >= 3 else (float("nan"), float("nan"))
        normality_rows.append({
            "variable": lbl,
            "n": int(len(v)),
            "skewness": round(sk, 4),
            "excess_kurtosis": round(ku, 4),
            "k2_stat": round(float(k2_s), 3) if math.isfinite(k2_s) else None,
            "k2_p": round(float(k2_p), 5) if math.isfinite(k2_p) else None,
            "jb_stat": round(float(jb_s), 3) if math.isfinite(jb_s) else None,
            "jb_p": round(float(jb_p), 5) if math.isfinite(jb_p) else None,
            "shapiro_w": round(float(sw_s), 4) if math.isfinite(sw_s) else None,
            "shapiro_p": round(float(sw_p), 5) if math.isfinite(sw_p) else None,
        })

    return {
        "ok": True,
        "n": n,
        "descriptives": desc_rows,
        "normality": normality_rows,
        "boxcox_info": y_info.get("boxcox_info"),
    }
