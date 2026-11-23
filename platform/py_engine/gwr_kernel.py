"""
Singularity-Guarded Batched Adaptive Bisquare GWR Kernel (Pure NumPy)
====================================================================
Guard #6 Implementation:
- Low-prevalence dummy pre-screening (N1 < max(20, 0.10*N))
- Deterministic coordinate micro-jitter (1.0 m) breaking co-located ties
- Pre-sorted cKDTree neighbor matrix (computed once)
- Batched (B=128) np.einsum local WLS with ZERO N x N hat-matrix allocation
- Selective Condition-Number Ridge (kappa_i > kappa_thresh, intercept unpenalized)
- Back-transformation from Z-score to native regression units
- Leung / IQR-vs-2*SE_OLS spatial non-stationarity test
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize, stats
from scipy.spatial import cKDTree

from .ols_kernel import fit_ols_core


def fit_adaptive_gwr(
    X: np.ndarray,
    y: np.ndarray,
    col_meta: List[Dict[str, Any]],
    coords_xy: np.ndarray,
    gwr_cfg: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Run singularity-guarded Adaptive Bisquare GWR with Golden-Section AICc bandwidth search.
    """
    cfg = gwr_cfg or {}
    jitter_m = float(cfg.get("jitter_m", 1.0))
    b_min_floor = int(cfg.get("b_min_floor", 15))
    kappa_thresh = float(cfg.get("kappa_thresh", 1000.0))
    prevalence_min_pct = float(cfg.get("prevalence_min_pct", 0.10))
    batch_size = int(cfg.get("batch_size", 128))

    n, k_orig = X.shape
    warnings: List[str] = []

    # 1. Pre-screen low-prevalence binary dummies (preventing local rank deficiency)
    keep_cols = [0]
    dropped_dummies: List[str] = []
    min_prev_count = max(20, int(math.ceil(prevalence_min_pct * n)))
    for j in range(1, k_orig):
        meta = col_meta[j]
        if meta.get("is_binary", False):
            n1 = int(np.sum(X[:, j] == 1.0))
            n0 = int(np.sum(X[:, j] == 0.0))
            if min(n1, n0) < min_prev_count:
                dropped_dummies.append(f"{meta['name']} (min cell {min(n1,n0)} < {min_prev_count})")
                continue
        keep_cols.append(j)

    if len(keep_cols) <= 1:
        # Ensure at least non-constant numeric covariates remain if all were dropped
        keep_cols = [0] + [j for j in range(1, k_orig) if np.ptp(X[:, j]) > 1e-12]
        dropped_dummies = []
    elif dropped_dummies:
        warnings.append(
            f"Guard #6 pre-screened {len(dropped_dummies)} low-prevalence dummies before GWR: " + ", ".join(dropped_dummies)
        )

    X_use = X[:, keep_cols]
    meta_use = [col_meta[j] for j in keep_cols]
    p = X_use.shape[1]

    if n <= p + 10:
        return {"ok": False, "reason": f"Insufficient observations (N={n}) for GWR with p={p} parameters."}

    # 2. Deterministic 1.0 m micro-jitter on co-located points
    rng = np.random.default_rng(42)
    coords_jit = coords_xy.astype(np.float64).copy()
    if jitter_m > 0.0:
        # Apply deterministic jitter around original coordinates
        angles = rng.uniform(0.0, 2.0 * math.pi, size=n)
        radii = rng.uniform(0.1 * jitter_m, jitter_m, size=n)
        coords_jit[:, 0] += radii * np.cos(angles)
        coords_jit[:, 1] += radii * np.sin(angles)

    # 3. Standardize X_non_int and y for well-conditioned local solves, storing exact scale factors
    y_mean = float(np.mean(y))
    y_std_val = max(1e-12, float(np.std(y, ddof=1)))
    y_z = (y - y_mean) / y_std_val

    x_means = np.zeros(p, dtype=np.float64)
    x_stds = np.ones(p, dtype=np.float64)
    X_z = np.ones((n, p), dtype=np.float64)
    for j in range(1, p):
        m_j = float(np.mean(X_use[:, j]))
        s_j = max(1e-12, float(np.std(X_use[:, j], ddof=1)))
        x_means[j] = m_j
        x_stds[j] = s_j
        X_z[:, j] = (X_use[:, j] - m_j) / s_j

    # Global OLS reference on native X_use, y
    ols_global = fit_ols_core(X_use, y, cov_type="HC3")

    # 4. Pre-sort all neighbors once via cKDTree
    dists_sorted, idx_sorted = cKDTree(coords_jit).query(coords_jit, k=n)

    def _eval_bandwidth(bw_int: int, return_full: bool = False) -> Dict[str, Any]:
        bw = int(max(p + 5, min(n - 1, bw_int)))
        nbr_idx = idx_sorted[:, :bw]  # (N, bw)
        nbr_dist = dists_sorted[:, :bw]  # (N, bw)
        bw_radii = np.maximum(1e-6, dists_sorted[:, bw])  # distance to (bw+1)-th neighbor (strictly > all bw)
        u = nbr_dist / bw_radii[:, None]
        w_loc = np.square(np.maximum(0.0, 1.0 - np.square(u)))  # Adaptive bisquare (N, bw)

        betas_z = np.zeros((n, p), dtype=np.float64)
        se_z = np.zeros((n, p), dtype=np.float64)
        s_diag = np.zeros(n, dtype=np.float64)
        s_row_sq = np.zeros(n, dtype=np.float64)
        local_r2 = np.zeros(n, dtype=np.float64)
        local_kappas = np.zeros(n, dtype=np.float64)
        n_regularized = 0

        # Penalty matrix leaving intercept unpenalized
        P_ridge = np.eye(p, dtype=np.float64)
        P_ridge[0, 0] = 0.0

        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            b_len = end - start
            b_idx = nbr_idx[start:end]  # (B, bw)
            b_w = w_loc[start:end]      # (B, bw)

            X_loc = X_z[b_idx]          # (B, bw, p)
            y_loc = y_z[b_idx]          # (B, bw)
            X_focal = X_z[start:end]    # (B, p)

            # Batched A_i = X_i^T W_i X_i and A2_i = X_i^T W_i^2 X_i via einsum (zero N x N S!)
            A_batch = np.einsum("bik,bi,bij->bkj", X_loc, b_w, X_loc)
            b_vec = np.einsum("bik,bi,bi->bk", X_loc, b_w, y_loc)
            w2 = b_w * b_w
            A2_batch = np.einsum("bik,bi,bij->bkj", X_loc, w2, X_loc)

            for bi in range(b_len):
                gi = start + bi
                A_i = A_batch[bi]
                # Check condition number of correlation-scaled local A_i
                d_scale = np.sqrt(np.maximum(1e-12, np.diag(A_i)))
                A_corr = A_i / np.outer(d_scale, d_scale)
                svals = np.linalg.svd(A_corr, compute_uv=False)
                kappa_i = float(svals[0] / max(1e-15, svals[-1]))
                local_kappas[gi] = kappa_i

                if kappa_i > kappa_thresh or not np.isfinite(kappa_i):
                    # Selective minimal Ridge leaving intercept unpenalized
                    lam_reg = max(1e-4, float(svals[0]) / kappa_thresh) * float(np.mean(np.diag(A_i)[1:]) if p > 1 else 1.0)
                    A_reg = A_i + lam_reg * P_ridge
                    n_regularized += 1
                else:
                    A_reg = A_i + 1e-10 * P_ridge

                A_inv = np.linalg.pinv(A_reg)
                beta_i = A_inv @ b_vec[bi]
                betas_z[gi] = beta_i

                xf = X_focal[bi]
                # Hat row r_i = xf^T A_inv X_loc^T W_loc -> S_ii = xf^T A_inv xf * w_focal(=1)
                c_i = A_inv @ xf
                s_diag[gi] = float(xf @ c_i)
                # ||r_i||_2^2 = c_i^T A2_i c_i
                s_row_sq[gi] = float(c_i @ A2_batch[bi] @ c_i)

                if return_full:
                    # Store variance factor diag(A_inv @ A2_i @ A_inv)
                    cov_fac = A_inv @ A2_batch[bi] @ A_inv
                    se_z[gi] = np.sqrt(np.maximum(0.0, np.diag(cov_fac)))
                    # Local weighted R^2
                    yhat_loc = X_loc[bi] @ beta_i
                    w_i = b_w[bi]
                    y_wmean = float(np.sum(w_i * y_loc[bi]) / max(1e-12, np.sum(w_i)))
                    tss_loc = float(np.sum(w_i * ((y_loc[bi] - y_wmean) ** 2)))
                    rss_loc = float(np.sum(w_i * ((y_loc[bi] - yhat_loc) ** 2)))
                    local_r2[gi] = max(0.0, 1.0 - rss_loc / max(1e-15, tss_loc))

        yhat_z = np.sum(X_z * betas_z, axis=1)
        resid_z = y_z - yhat_z
        rss_z = float(resid_z @ resid_z)
        tr_s = float(np.sum(s_diag))
        tr_sts = float(np.sum(s_row_sq))
        eff_df = max(2.0, n - 2.0 * tr_s + tr_sts)
        sigma2_z = rss_z / eff_df

        # Standard Fotheringham et al. GWR AICc
        denom_aicc = n - 2.0 - tr_s
        if denom_aicc <= 1.0:
            aicc = 1e12
        else:
            aicc = n * math.log(max(1e-18, rss_z / n)) + n * math.log(2.0 * math.pi) + n * ((n + tr_s) / denom_aicc)

        # Penalize bandwidths during search if >20% locations require Ridge regularization
        reg_share = n_regularized / n
        if not return_full and reg_share > 0.20:
            aicc += 1e6 * (reg_share - 0.20)

        out = {
            "bw": bw,
            "aicc": float(aicc),
            "tr_s": tr_s,
            "tr_sts": tr_sts,
            "eff_df": eff_df,
            "rss_z": rss_z,
            "sigma2_z": sigma2_z,
            "n_regularized": n_regularized,
            "pct_regularized": round(reg_share * 100.0, 2),
        }
        if return_full:
            se_z_scaled = se_z * math.sqrt(sigma2_z)
            # Back-transform Z-score estimates & SEs to native units!
            betas_native = np.zeros_like(betas_z)
            se_native = np.zeros_like(se_z_scaled)
            for j in range(1, p):
                scale_j = y_std_val / x_stds[j]
                betas_native[:, j] = betas_z[:, j] * scale_j
                se_native[:, j] = se_z_scaled[:, j] * scale_j
            # Intercept back-transformation: b0_nat = y_mean + y_std * b0_z - sum_{j>=1} b_j_nat * x_mean_j
            betas_native[:, 0] = y_mean + y_std_val * betas_z[:, 0] - np.sum(betas_native[:, 1:] * x_means[None, 1:], axis=1)
            se_native[:, 0] = y_std_val * se_z_scaled[:, 0]

            t_vals = np.divide(betas_native, se_native, out=np.zeros_like(betas_native), where=se_native > 1e-15)
            resid_native = y - np.sum(X_use * betas_native, axis=1)
            rss_native = float(resid_native @ resid_native)
            tss_native = float(np.sum((y - y_mean) ** 2))
            gwr_r2 = 1.0 - rss_native / max(1e-18, tss_native)
            gwr_adj_r2 = 1.0 - (1.0 - gwr_r2) * (n - 1.0) / max(1.0, n - tr_s - 1.0)
            sigma2_nat = rss_native / eff_df
            aic_gwr = n * math.log(max(1e-18, rss_native / n)) + n * math.log(2.0 * math.pi) + n + 2.0 * tr_s
            bic_gwr = n * math.log(max(1e-18, rss_native / n)) + n * math.log(2.0 * math.pi) + n + tr_s * math.log(n)

            out.update({
                "betas_native": betas_native,
                "se_native": se_native,
                "t_vals": t_vals,
                "local_r2": local_r2,
                "local_kappas": local_kappas,
                "resid_native": resid_native,
                "gwr_r2": gwr_r2,
                "gwr_adj_r2": gwr_adj_r2,
                "aic": aic_gwr,
                "bic": bic_gwr,
                "sigma_native": math.sqrt(sigma2_nat),
            })
        return out

    # 5. Bounded Golden-Section + Coarse Grid Bandwidth Search
    b_min = min(n - 2, max(p + b_min_floor, int(math.ceil(0.15 * n))))
    b_max = n - 1
    eval_history: Dict[int, float] = {}

    def _obj(bw_float: float) -> float:
        bw_i = int(round(bw_float))
        if bw_i not in eval_history:
            res_b = _eval_bandwidth(bw_i, return_full=False)
            eval_history[bw_i] = res_b["aicc"]
        return eval_history[bw_i]

    if b_max <= b_min + 2:
        opt_bw = b_min
        _obj(opt_bw)
    else:
        # Coarse scan first (6 points) to prevent local minima, then Golden-Section around best bracket
        grid_pts = sorted({int(round(v)) for v in np.linspace(b_min, b_max, 7)})
        for idx_g, g_bw in enumerate(grid_pts):
            _obj(g_bw)
            if progress_cb:
                progress_cb(0.1 + 0.4 * ((idx_g + 1) / len(grid_pts)), f"GWR bandwidth scan b={g_bw}...")

        best_grid_bw = min(grid_pts, key=lambda b: eval_history[b])
        g_pos = grid_pts.index(best_grid_bw)
        lo_br = grid_pts[max(0, g_pos - 1)]
        hi_br = grid_pts[min(len(grid_pts) - 1, g_pos + 1)]
        if hi_br > lo_br + 1:
            optimize.minimize_scalar(_obj, bounds=(lo_br, hi_br), method="bounded", options={"xatol": 1.0, "maxiter": 12})
        opt_bw = min(eval_history.keys(), key=lambda b: eval_history[b])

    if progress_cb:
        progress_cb(0.75, f"Fitting final GWR at optimal b={opt_bw}...")

    final_res = _eval_bandwidth(opt_bw, return_full=True)

    if final_res["n_regularized"] > 0:
        warnings.append(
            f"Guard #6 Selective Ridge applied at {final_res['n_regularized']}/{n} ({final_res['pct_regularized']}%) "
            f"locations with local κ > {kappa_thresh:g}."
        )

    # 6. Five-Number Summary & Spatial Non-Stationarity Test (IQR vs 2*SE_OLS)
    param_summaries: List[Dict[str, Any]] = []
    betas_nat = final_res["betas_native"]
    t_mat = final_res["t_vals"]
    for j, meta in enumerate(meta_use):
        vals = betas_nat[:, j]
        t_j = t_mat[:, j]
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = float(q3 - q1)
        ols_se_j = float(ols_global["se"][j])
        non_stat = iqr > 2.0 * ols_se_j
        pct_sig = float(np.mean(np.abs(t_j) >= 1.96) * 100.0)
        param_summaries.append({
            "name": meta["name"],
            "label": meta["label"],
            "ols_beta": round(float(ols_global["beta"][j]), 4),
            "ols_se": round(ols_se_j, 4),
            "min": round(float(np.min(vals)), 4),
            "q1": round(float(q1), 4),
            "median": round(float(med), 4),
            "q3": round(float(q3), 4),
            "max": round(float(np.max(vals)), 4),
            "iqr": round(iqr, 4),
            "pct_sig_95": round(pct_sig, 1),
            "non_stationary": non_stat,
            "stationarity_verdict": "Non-stationary (IQR > 2·SE_OLS)" if non_stat else "Stationary (IQR <= 2·SE_OLS)",
        })

    bw_curve = [{"bw": b, "aicc": round(val, 2)} for b, val in sorted(eval_history.items())]

    return {
        "ok": True,
        "n_obs": n,
        "p_params": p,
        "optimal_bw": int(opt_bw),
        "bw_pct_sample": round(opt_bw / n * 100.0, 1),
        "b_min_floor": int(b_min),
        "n_regularized": final_res["n_regularized"],
        "pct_regularized": final_res["pct_regularized"],
        "global_ols_r2": round(float(ols_global["r2"]), 4),
        "global_ols_adj_r2": round(float(ols_global["adj_r2"]), 4),
        "global_ols_aic": round(float(ols_global["aic"]), 2),
        "gwr_r2": round(float(final_res["gwr_r2"]), 4),
        "gwr_adj_r2": round(float(final_res["gwr_adj_r2"]), 4),
        "gwr_aic": round(float(final_res["aic"]), 2),
        "gwr_aicc": round(float(final_res["aicc"]), 2),
        "gwr_bic": round(float(final_res["bic"]), 2),
        "tr_s": round(float(final_res["tr_s"]), 2),
        "bw_curve": bw_curve,
        "param_summaries": param_summaries,
        "col_names": [m["name"] for m in meta_use],
        "betas_native": np.round(betas_nat, 5).tolist(),
        "se_native": np.round(final_res["se_native"], 5).tolist(),
        "t_vals": np.round(t_mat, 3).tolist(),
        "local_r2": np.round(final_res["local_r2"], 4).tolist(),
        "local_kappas": np.round(final_res["local_kappas"], 1).tolist(),
        "resid_native": np.round(final_res["resid_native"], 5).tolist(),
        "warnings": warnings,
    }
