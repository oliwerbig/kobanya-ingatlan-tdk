"""
Singularity-Guarded Batched Adaptive Bisquare GWR & MGWR Kernel (Pure NumPy)
=============================================================================
Guard #6 Implementation:
- Low-prevalence dummy pre-screening (N1 < max(20, 0.10*N))
- Deterministic coordinate micro-jitter (1.0 m) breaking co-located ties
- Pre-sorted cKDTree neighbor matrix (computed once)
- Batched (B=128) np.einsum local WLS with ZERO N x N hat-matrix allocation
- Local zero-variance dummy exclusion within window i + Selective Ridge (kappa_i > kappa_thresh)
- Optional MGWR (Fotheringham, Yang & Kang 2017) backfitting covariate-specific bandwidths
- Native-scale Hurvich-Simonoff-Tsai AICc & ML k+1 AIC/BIC parity
- Back-transformation from Z-score to native regression units
- Fotheringham, Brunsdon & Charlton (2002) IQR-vs-2*SE_OLS + MC spatial non-stationarity test
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
    Run singularity-guarded Adaptive Bisquare GWR (or MGWR backfitting) with Golden-Section AICc search.
    """
    cfg = gwr_cfg or {}
    gwr_mode = str(cfg.get("mode", "gwr")).lower()
    if gwr_mode not in ("gwr", "mgwr"):
        gwr_mode = "gwr"
    jitter_m = float(cfg.get("jitter_m", 1.0))
    b_min_floor = int(cfg.get("b_min_floor", 15))
    kappa_thresh = float(cfg.get("kappa_thresh", 1000.0))
    prevalence_min_pct = float(cfg.get("prevalence_min_pct", 0.10))
    batch_size = int(cfg.get("batch_size", 128))
    n_mc_nonstat = int(cfg.get("n_mc_nonstat", 29 if len(y) <= 800 else 0))

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

    # 4. Build cKDTree once; pre-sort all neighbors only for N <= 1200 (avoid O(N^2) allocation in WASM)
    nbr_tree = cKDTree(coords_jit)
    if n <= 1200:
        dists_sorted, idx_sorted = nbr_tree.query(coords_jit, k=n)
    else:
        dists_sorted, idx_sorted = None, None

    # Exact linear map from Z-score beta_z to native intercept: b0_nat = y_mean + a_int^T beta_z
    a_int = np.empty(p, dtype=np.float64)
    a_int[0] = y_std_val
    if p > 1:
        a_int[1:] = -y_std_val * (x_means[1:] / x_stds[1:])

    is_bin_col = [bool(m.get("is_binary", False)) for m in meta_use]

    def _eval_bandwidth(
        bw_int: int,
        return_full: bool = False,
        X_in: Optional[np.ndarray] = None,
        y_in: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        X_cur = X_z if X_in is None else X_in
        y_cur = y_z if y_in is None else y_in
        bw = int(max(p + 5, min(n - 1, bw_int)))
        betas_z_num = np.zeros((n, p), dtype=np.float64)
        dropped_mask = np.zeros((n, p), dtype=bool)
        is_regularized = np.zeros(n, dtype=bool)
        n_local_dropped = np.zeros(n, dtype=np.int32)
        se_z = np.zeros((n, p), dtype=np.float64)
        se_int_fac = np.zeros(n, dtype=np.float64)
        s_diag = np.zeros(n, dtype=np.float64)
        s_row_sq = np.zeros(n, dtype=np.float64)
        local_r2 = np.zeros(n, dtype=np.float64)
        local_kappas = np.ones(n, dtype=np.float64)
        n_regularized = 0

        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            b_len = end - start
            if idx_sorted is not None and dists_sorted is not None:
                b_idx = idx_sorted[start:end, :bw]
                b_dist = dists_sorted[start:end, :bw]
                bw_radii = np.maximum(1e-6, dists_sorted[start:end, bw])
            else:
                d_b, i_b = nbr_tree.query(coords_jit[start:end], k=bw + 1)
                b_idx = i_b[:, :bw]
                b_dist = d_b[:, :bw]
                bw_radii = np.maximum(1e-6, d_b[:, bw])

            u = b_dist / bw_radii[:, None]
            b_w = np.square(np.maximum(0.0, 1.0 - np.square(u)))  # Adaptive bisquare (B, bw)

            X_loc = X_cur[b_idx]        # (B, bw, p)
            y_loc = y_cur[b_idx]        # (B, bw)
            X_focal = X_cur[start:end]  # (B, p)

            # Batched A_i = X_i^T W_i X_i and A2_i = X_i^T W_i^2 X_i via einsum (zero N x N S!)
            A_batch = np.einsum("bik,bi,bij->bkj", X_loc, b_w, X_loc)
            b_vec = np.einsum("bik,bi,bi->bk", X_loc, b_w, y_loc)
            w2 = b_w * b_w
            A2_batch = np.einsum("bik,bi,bij->bkj", X_loc, w2, X_loc)
            w_sums = np.maximum(1e-12, np.sum(b_w, axis=1))  # (B,)
            wx_means = np.einsum("bi,bij->bj", b_w, X_loc) / w_sums[:, None]
            wx2_means = np.einsum("bi,bij->bj", b_w, X_loc * X_loc) / w_sums[:, None]
            w_vars = np.maximum(0.0, wx2_means - wx_means * wx_means)

            for bi in range(b_len):
                gi = start + bi
                w_i = b_w[bi]
                wv_i = w_vars[bi]

                # Exclude locally invariant columns (e.g., homogeneous dummies in condo clusters)
                active: List[int] = [0]
                for j in range(1, p):
                    if wv_i[j] < 1e-10:
                        dropped_mask[gi, j] = True
                        n_local_dropped[gi] += 1
                        continue
                    if is_bin_col[j]:
                        # Guard degenerate 1-neighbor boundary sliver
                        col_vals = X_loc[bi, :, j]
                        wm = float(wx_means[bi, j])
                        pos_w = float(np.sum(w_i[col_vals > wm]))
                        neg_w = float(np.sum(w_i[col_vals <= wm]))
                        if min(pos_w, neg_w) < 1e-4 * float(w_sums[bi]):
                            dropped_mask[gi, j] = True
                            n_local_dropped[gi] += 1
                            continue
                    active.append(j)

                act_idx = np.asarray(active, dtype=np.int32)
                p_act = len(act_idx)
                A_sub = A_batch[bi][np.ix_(act_idx, act_idx)]
                b_sub = b_vec[bi][act_idx]
                A2_sub = A2_batch[bi][np.ix_(act_idx, act_idx)]

                P_sub = np.eye(p_act, dtype=np.float64)
                P_sub[0, 0] = 0.0

                if p_act > 1:
                    d_scale = np.sqrt(np.maximum(1e-12, np.diag(A_sub)))
                    A_corr = A_sub / np.outer(d_scale, d_scale)
                    svals = np.linalg.svd(A_corr, compute_uv=False)
                    kappa_i = float(svals[0] / max(1e-15, svals[-1]))
                else:
                    svals = np.array([1.0], dtype=np.float64)
                    kappa_i = 1.0
                local_kappas[gi] = kappa_i

                if p_act > 1 and (kappa_i > kappa_thresh or not np.isfinite(kappa_i)):
                    lam_reg = max(1e-4, float(svals[0]) / kappa_thresh) * float(np.mean(np.diag(A_sub)[1:]))
                    A_reg = A_sub + lam_reg * P_sub
                    is_regularized[gi] = True
                    n_regularized += 1
                else:
                    A_reg = A_sub + 1e-10 * P_sub

                A_inv = np.linalg.pinv(A_reg)
                beta_act = A_inv @ b_sub
                betas_z_num[gi, act_idx] = beta_act

                xf_act = X_focal[bi, act_idx]
                c_i = A_inv @ xf_act
                s_diag[gi] = float(xf_act @ c_i)
                s_row_sq[gi] = float(c_i @ A2_sub @ c_i)

                if return_full:
                    cov_fac = A_inv @ A2_sub @ A_inv
                    se_z[gi, act_idx] = np.sqrt(np.maximum(0.0, np.diag(cov_fac)))
                    a_sub = a_int[act_idx]
                    se_int_fac[gi] = math.sqrt(max(0.0, float(a_sub @ cov_fac @ a_sub)))
                    yhat_loc = X_loc[bi, :, :][:, act_idx] @ beta_act
                    y_wmean = float(np.sum(w_i * y_loc[bi]) / w_sums[bi])
                    tss_loc = float(np.sum(w_i * ((y_loc[bi] - y_wmean) ** 2)))
                    rss_loc = float(np.sum(w_i * ((y_loc[bi] - yhat_loc) ** 2)))
                    local_r2[gi] = max(0.0, 1.0 - rss_loc / max(1e-15, tss_loc))

        yhat_z = np.sum(X_cur * betas_z_num, axis=1)
        resid_z = y_cur - yhat_z
        rss_z = float(resid_z @ resid_z)
        rss_native = rss_z * (y_std_val ** 2)
        sigma2_nat_ml = max(1e-18, rss_native / n)
        tr_s = float(np.sum(s_diag))
        tr_sts = float(np.sum(s_row_sq))
        eff_df = max(2.0, n - 2.0 * tr_s + tr_sts)
        sigma2_z = rss_z / eff_df

        # Fotheringham, Brunsdon & Charlton (2002) GWR AICc on NATIVE y scale
        denom_aicc = n - 2.0 - tr_s
        if denom_aicc <= 1.0 or not math.isfinite(rss_native):
            raw_aicc = 1e12
        else:
            raw_aicc = n * math.log(sigma2_nat_ml) + n * math.log(2.0 * math.pi) + n * ((n + tr_s) / denom_aicc)

        reg_share = n_regularized / n
        obj_score = float(raw_aicc)

        out: Dict[str, Any] = {
            "bw": bw,
            "aicc": float(raw_aicc),
            "obj_score": float(obj_score),
            "tr_s": tr_s,
            "tr_sts": tr_sts,
            "eff_df": eff_df,
            "rss_z": rss_z,
            "rss_native": rss_native,
            "sigma2_z": sigma2_z,
            "n_regularized": n_regularized,
            "pct_regularized": round(reg_share * 100.0, 2),
            "n_local_dropped": n_local_dropped,
            "n_local_dropped_locs": int(np.sum(n_local_dropped > 0)),
            "is_regularized": is_regularized,
            "dropped_mask": dropped_mask,
            "betas_z_num": betas_z_num,
        }
        if return_full:
            sqrt_s2z = math.sqrt(sigma2_z)
            se_z_scaled = se_z * sqrt_s2z
            # Back-transform Z-score estimates & SEs to native units!
            betas_native = np.zeros_like(betas_z_num)
            se_native = np.zeros_like(se_z_scaled)
            for j in range(1, p):
                scale_j = y_std_val / x_stds[j]
                betas_native[:, j] = betas_z_num[:, j] * scale_j
                se_native[:, j] = se_z_scaled[:, j] * scale_j
            # Locally dropped columns contributed 0 in X_z; their back-transform offset is also 0
            betas_native[:, 0] = y_mean + y_std_val * betas_z_num[:, 0] - np.sum(betas_native[:, 1:] * x_means[None, 1:], axis=1)
            se_native[:, 0] = se_int_fac * sqrt_s2z

            t_vals = np.divide(betas_native, se_native, out=np.zeros_like(betas_native), where=se_native > 1e-15)
            # Mark locally dropped cells cleanly as NaN so summaries/popups don't mistake 0 for an estimated coefficient
            if np.any(dropped_mask):
                betas_native = betas_native.copy()
                se_native = se_native.copy()
                t_vals = t_vals.copy()
                betas_native[dropped_mask] = np.nan
                se_native[dropped_mask] = np.nan
                t_vals[dropped_mask] = np.nan

            resid_native = resid_z * y_std_val
            tss_native = float(np.sum((y - y_mean) ** 2))
            gwr_r2 = 1.0 - rss_native / max(1e-18, tss_native)
            gwr_adj_r2 = 1.0 - (1.0 - gwr_r2) * (n - 1.0) / max(1.0, n - tr_s - 1.0)
            sigma2_nat = rss_native / eff_df
            # Unified k+1 ML AIC/BIC convention (+1 for sigma^2)
            aic_gwr = n * math.log(sigma2_nat_ml) + n * math.log(2.0 * math.pi) + n + 2.0 * (tr_s + 1.0)
            bic_gwr = n * math.log(sigma2_nat_ml) + n * math.log(2.0 * math.pi) + n + (tr_s + 1.0) * math.log(n)

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

    # 5. Bounded Golden-Section + Coarse Grid Bandwidth Search (no 0.15*N oversmoothing floor)
    b_min = min(n - 2, max(p + b_min_floor, 2 * p + 5))
    b_max = n - 1
    eval_history: Dict[int, Dict[str, Any]] = {}

    def _obj(bw_float: float) -> float:
        bw_i = int(round(bw_float))
        if bw_i not in eval_history:
            res_b = _eval_bandwidth(bw_i, return_full=False)
            eval_history[bw_i] = {
                "aicc": res_b["aicc"],
                "obj_score": res_b["obj_score"],
                "pct_regularized": res_b["pct_regularized"],
                "n_local_dropped_locs": res_b["n_local_dropped_locs"],
            }
        return eval_history[bw_i]["obj_score"]

    if b_max <= b_min + 2:
        opt_bw = b_min
        _obj(opt_bw)
    else:
        grid_pts = sorted({int(round(v)) for v in np.linspace(b_min, b_max, 7)})
        for idx_g, g_bw in enumerate(grid_pts):
            _obj(g_bw)
            if progress_cb:
                pct_base = 0.08 + (0.32 if gwr_mode == "mgwr" else 0.52) * ((idx_g + 1) / len(grid_pts))
                progress_cb(pct_base, f"GWR sávszélesség keresés b={g_bw}...")

        best_grid_bw = min(grid_pts, key=lambda b: eval_history[b]["obj_score"])
        g_pos = grid_pts.index(best_grid_bw)
        lo_br = grid_pts[max(0, g_pos - 1)]
        hi_br = grid_pts[min(len(grid_pts) - 1, g_pos + 1)]
        if hi_br > lo_br + 1:
            optimize.minimize_scalar(_obj, bounds=(lo_br, hi_br), method="bounded", options={"xatol": 1.0, "maxiter": 12})
        opt_bw = min(eval_history.keys(), key=lambda b: eval_history[b]["obj_score"])

    if progress_cb:
        progress_cb(0.62 if gwr_mode == "gwr" else 0.45, f"Illesztés optimális sávszélességgel b={opt_bw}...")

    final_res = _eval_bandwidth(opt_bw, return_full=True)

    # Optional MGWR backfitting (Fotheringham, Yang & Kang 2017)
    mgwr_bandwidths: Optional[Dict[str, int]] = None
    mgwr_converged: Optional[bool] = None
    mgwr_iters: Optional[int] = None

    if gwr_mode == "mgwr":
        betas_mgwr_z = final_res["betas_z_num"].copy()
        f_mat = X_z * betas_mgwr_z
        r_res = y_z - np.sum(f_mat, axis=1)
        bw_list = [int(opt_bw)] * p
        s_diag_cov = np.full((n, p), float(final_res["tr_s"]) / max(1, n * p), dtype=np.float64)
        var_fac_cov = np.zeros((n, p), dtype=np.float64)
        mgwr_dropped = np.zeros((n, p), dtype=bool)
        mgwr_local_dropped = np.zeros(n, dtype=np.int32)
        b_min_mgwr = min(n - 2, max(20, p + b_min_floor))
        max_bf_iters = int(cfg.get("mgwr_max_iter", 20))
        mgwr_converged = False
        mgwr_iters = 0

        def _fit_univariate_bw(
            j_col: int,
            target_y: np.ndarray,
            bw_int: int,
            tr_other: float = 0.0,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
            bw_u = int(max(b_min_mgwr, min(n - 1, bw_int)))
            x_col = X_z[:, j_col]
            beta_out = np.zeros(n, dtype=np.float64)
            s_out = np.zeros(n, dtype=np.float64)
            vfac_out = np.zeros(n, dtype=np.float64)
            drop_j = np.zeros(n, dtype=bool)
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                if idx_sorted is not None and dists_sorted is not None:
                    b_idx = idx_sorted[start:end, :bw_u]
                    b_dist = dists_sorted[start:end, :bw_u]
                    bw_radii = np.maximum(1e-6, dists_sorted[start:end, bw_u])
                else:
                    d_b, i_b = nbr_tree.query(coords_jit[start:end], k=bw_u + 1)
                    b_idx = i_b[:, :bw_u]
                    b_dist = d_b[:, :bw_u]
                    bw_radii = np.maximum(1e-6, d_b[:, bw_u])
                u = b_dist / bw_radii[:, None]
                b_w = np.square(np.maximum(0.0, 1.0 - np.square(u)))
                xj_loc = x_col[b_idx]
                yj_loc = target_y[b_idx]
                denom = np.sum(b_w * xj_loc * xj_loc, axis=1)
                num = np.sum(b_w * xj_loc * yj_loc, axis=1)
                w2_x2 = np.sum(b_w * b_w * xj_loc * xj_loc, axis=1)
                # Local variance guard (exclude locally invariant dummies cleanly, except intercept j_col == 0)
                if j_col > 0:
                    w_s = np.maximum(1e-12, np.sum(b_w, axis=1))
                    xm = np.sum(b_w * xj_loc, axis=1) / w_s
                    x2m = np.sum(b_w * xj_loc * xj_loc, axis=1) / w_s
                    inv_loc = (x2m - xm * xm) < 1e-10
                else:
                    inv_loc = np.zeros(end - start, dtype=bool)
                ok_loc = (denom > 1e-10) & (~inv_loc)
                b_sl = np.where(ok_loc, num / np.maximum(1e-12, denom), 0.0)
                xf = x_col[start:end]
                beta_out[start:end] = b_sl
                s_out[start:end] = np.where(ok_loc, (xf * xf) / np.maximum(1e-12, denom), 0.0)
                vfac_out[start:end] = np.where(ok_loc, w2_x2 / np.maximum(1e-18, denom * denom), 0.0)
                drop_j[start:end] = ~ok_loc

            res_u = target_y - x_col * beta_out
            rss_u = max(1e-18, float(res_u @ res_u) * (y_std_val ** 2))
            tr_tot = tr_other + float(np.sum(s_out))
            den_u = n - 2.0 - tr_tot
            if den_u <= 1.0:
                aicc_u = 1e12
            else:
                aicc_u = n * math.log(rss_u / n) + n * math.log(2.0 * math.pi) + n * ((n + tr_tot) / den_u)
            return beta_out, s_out, vfac_out, aicc_u, drop_j

        for bf_it in range(1, max_bf_iters + 1):
            mgwr_iters = bf_it
            f_prev = f_mat.copy()
            grid_u = sorted({int(round(v)) for v in np.linspace(b_min_mgwr, b_max, 6)})
            search_bw = bf_it <= max(8, max_bf_iters // 2)
            for j in range(p):
                y_part = r_res + f_mat[:, j]
                tr_other_j = float(np.sum(s_diag_cov) - np.sum(s_diag_cov[:, j]))
                if search_bw:
                    hist_u: Dict[int, float] = {}
                    for gu in grid_u:
                        _, _, _, ac, _ = _fit_univariate_bw(j, y_part, gu, tr_other=tr_other_j)
                        hist_u[gu] = ac
                    best_g = min(grid_u, key=lambda b: hist_u[b])
                    gp = grid_u.index(best_g)
                    lo_u = grid_u[max(0, gp - 1)]
                    hi_u = grid_u[min(len(grid_u) - 1, gp + 1)]
                    if hi_u > lo_u + 1:
                        def _obj_u(bu: float) -> float:
                            bi = int(round(bu))
                            if bi not in hist_u:
                                _, _, _, ac_u, _ = _fit_univariate_bw(j, y_part, bi, tr_other=tr_other_j)
                                hist_u[bi] = ac_u
                            return hist_u[bi]
                        optimize.minimize_scalar(_obj_u, bounds=(lo_u, hi_u), method="bounded", options={"xatol": 1.5, "maxiter": 8})
                    best_bw_j = min(hist_u.keys(), key=lambda b: hist_u[b])
                else:
                    best_bw_j = bw_list[j]
                b_new, s_new, vf_new, _, drp_j = _fit_univariate_bw(j, y_part, best_bw_j, tr_other=tr_other_j)
                f_new = X_z[:, j] * b_new
                r_res = y_part - f_new
                f_mat[:, j] = f_new
                betas_mgwr_z[:, j] = b_new
                s_diag_cov[:, j] = s_new
                var_fac_cov[:, j] = vf_new
                mgwr_dropped[:, j] = drp_j
                bw_list[j] = int(best_bw_j)

            denom_soc = max(1e-12, float(np.sum(f_prev * f_prev)))
            soc_f = math.sqrt(float(np.sum((f_mat - f_prev) ** 2)) / denom_soc)
            if progress_cb:
                progress_cb(0.45 + 0.45 * min(1.0, bf_it / max(4, min(max_bf_iters, 10))), f"MGWR backfitting iteráció {bf_it} (SOC-f={soc_f:.2e})...")
            if soc_f < 5e-4 and bf_it >= 3:
                mgwr_converged = True
                break

        mgwr_local_dropped = np.sum(mgwr_dropped, axis=1, dtype=np.int32)
        tr_s_mgwr = min(float(n - 3.0), float(np.sum(s_diag_cov)))
        eff_df_mgwr = max(2.0, float(n) - tr_s_mgwr)
        rss_z_m = float(r_res @ r_res)
        rss_nat_m = rss_z_m * (y_std_val ** 2)
        sigma2_z_m = rss_z_m / eff_df_mgwr
        sigma2_nat_ml_m = max(1e-18, rss_nat_m / n)

        se_z_m = np.sqrt(np.maximum(0.0, var_fac_cov * sigma2_z_m))
        betas_nat_m = np.zeros_like(betas_mgwr_z)
        se_nat_m = np.zeros_like(se_z_m)
        for j in range(1, p):
            scale_j = y_std_val / x_stds[j]
            betas_nat_m[:, j] = betas_mgwr_z[:, j] * scale_j
            se_nat_m[:, j] = se_z_m[:, j] * scale_j
        betas_nat_m[:, 0] = y_mean + y_std_val * betas_mgwr_z[:, 0] - np.sum(betas_nat_m[:, 1:] * x_means[None, 1:], axis=1)
        # Approximate univariate back-transformed intercept SE variance sum
        var_b0_m = (a_int[0] ** 2) * var_fac_cov[:, 0] * sigma2_z_m
        if p > 1:
            var_b0_m = var_b0_m + np.sum((a_int[None, 1:] ** 2) * var_fac_cov[:, 1:] * sigma2_z_m, axis=1)
        se_nat_m[:, 0] = np.sqrt(np.maximum(0.0, var_b0_m))
        t_vals_m = np.divide(betas_nat_m, se_nat_m, out=np.zeros_like(betas_nat_m), where=se_nat_m > 1e-15)
        if np.any(mgwr_dropped):
            betas_nat_m[mgwr_dropped] = np.nan
            se_nat_m[mgwr_dropped] = np.nan
            t_vals_m[mgwr_dropped] = np.nan

        resid_nat_m = r_res * y_std_val
        tss_nat = float(np.sum((y - y_mean) ** 2))
        r2_m = 1.0 - rss_nat_m / max(1e-18, tss_nat)
        adj_r2_m = 1.0 - (1.0 - r2_m) * (n - 1.0) / max(1.0, n - tr_s_mgwr - 1.0)
        denom_aicc_m = max(1.0, n - 2.0 - tr_s_mgwr)
        aicc_m = n * math.log(sigma2_nat_ml_m) + n * math.log(2.0 * math.pi) + n * ((n + tr_s_mgwr) / denom_aicc_m)
        aic_m = n * math.log(sigma2_nat_ml_m) + n * math.log(2.0 * math.pi) + n + 2.0 * (tr_s_mgwr + 1.0)
        bic_m = n * math.log(sigma2_nat_ml_m) + n * math.log(2.0 * math.pi) + n + (tr_s_mgwr + 1.0) * math.log(n)

        # Compute approximate local R^2 under initial bw_u for map display
        mgwr_bandwidths = {meta_use[j]["name"]: int(bw_list[j]) for j in range(p)}
        opt_bw = int(round(float(np.median(bw_list))))
        final_res.update({
            "tr_s": tr_s_mgwr,
            "eff_df": eff_df_mgwr,
            "aicc": aicc_m,
            "aic": aic_m,
            "bic": bic_m,
            "gwr_r2": r2_m,
            "gwr_adj_r2": adj_r2_m,
            "betas_native": betas_nat_m,
            "se_native": se_nat_m,
            "t_vals": t_vals_m,
            "resid_native": resid_nat_m,
            "dropped_mask": mgwr_dropped,
            "n_local_dropped": mgwr_local_dropped,
            "n_local_dropped_locs": int(np.sum(mgwr_local_dropped > 0)),
            "n_regularized": 0,
            "pct_regularized": 0.0,
            "is_regularized": np.zeros(n, dtype=bool),
        })

    if final_res.get("n_local_dropped_locs", 0) > 0:
        warnings.append(
            f"Guard #6 locally excluded invariant dummy variables at {final_res['n_local_dropped_locs']}/{n} "
            "locations (unbiased local fit without Ridge penalty)."
        )
    if final_res["n_regularized"] > 0:
        warnings.append(
            f"Guard #6 Selective Ridge applied at {final_res['n_regularized']}/{n} ({final_res['pct_regularized']}%) "
            f"locations due to continuous local collinearity (κ > {kappa_thresh:g})."
        )

    # Fast Monte Carlo spatial non-stationarity permutation test (at opt_bw)
    mc_p_vals: Optional[np.ndarray] = None
    if n_mc_nonstat > 0 and p > 1:
        betas_obs_z = final_res["betas_z_num"] if gwr_mode == "gwr" else ((X_z * 0.0) + final_res["betas_z_num"])
        obs_vars = np.nanvar(final_res["betas_native"], axis=0)
        ge_mc = np.zeros(p, dtype=np.int32)
        bw_mc = int(max(p + 5, min(n - 1, opt_bw)))
        rng_mc = np.random.default_rng(1042)
        scale_vec = np.ones(p, dtype=np.float64)
        for j in range(1, p):
            scale_vec[j] = y_std_val / x_stds[j]
        for mc_idx in range(n_mc_nonstat):
            perm = rng_mc.permutation(n)
            X_pm = X_z[perm]
            y_pm = y_z[perm]
            res_pm = _eval_bandwidth(bw_mc, return_full=False, X_in=X_pm, y_in=y_pm)
            b_nat_pm = res_pm["betas_z_num"].copy()
            b_nat_pm[res_pm["dropped_mask"]] = np.nan
            b_nat_pm[:, 1:] *= scale_vec[None, 1:]
            b_nat_pm[:, 0] = y_mean + y_std_val * res_pm["betas_z_num"][:, 0] - np.nansum(b_nat_pm[:, 1:] * x_means[None, 1:], axis=1)
            vars_pm = np.nanvar(b_nat_pm, axis=0)
            ge_mc += np.where(np.isfinite(vars_pm) & (vars_pm >= obs_vars), 1, 0)
        mc_p_vals = (ge_mc + 1.0) / (n_mc_nonstat + 1.0)

    if progress_cb:
        progress_cb(0.95, "Lokális statisztikák és BY-FDR szűrés számítása...")

    # 6. Five-Number Summary, Multiple-Testing Adjusted Local Significance (BY-FDR & da Silva-Fotheringham) & FBC IQR Test
    param_summaries: List[Dict[str, Any]] = []
    betas_nat = final_res["betas_native"]
    t_mat = final_res["t_vals"]
    eff_df = float(final_res["eff_df"])
    tr_s_final = float(final_res["tr_s"])
    p_mat = np.where(np.isfinite(t_mat), 2.0 * stats.t.sf(np.abs(t_mat), df=eff_df), 1.0)
    t_crit_nom = float(stats.t.ppf(0.975, df=eff_df))
    alpha_sdf = 0.05 / max(1.0, tr_s_final / max(1, p))
    t_crit_sdf = float(stats.t.ppf(1.0 - 0.5 * alpha_sdf, df=eff_df))
    c_harmonic = float(np.sum(1.0 / np.arange(1, n + 1, dtype=np.float64)))
    by_sig_mat = np.zeros((n, p), dtype=bool)
    reg_mask = final_res["is_regularized"]
    drp_mat = final_res["dropped_mask"]

    for j, meta in enumerate(meta_use):
        vals_raw = betas_nat[:, j]
        fin_mask = np.isfinite(vals_raw)
        vals = vals_raw[fin_mask]
        t_j = t_mat[:, j]
        p_j = p_mat[:, j]
        if len(vals) == 0:
            q1 = med = q3 = vmin = vmax = iqr = 0.0
        else:
            q1, med, q3 = [float(v) for v in np.percentile(vals, [25, 50, 75])]
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            iqr = float(q3 - q1)
        ols_se_j = float(ols_global["se"][j])
        non_stat = iqr > 2.0 * ols_se_j

        valid_test = (~drp_mat[:, j]) & (~reg_mask) & np.isfinite(t_j)
        pct_sig_nom = float(np.mean(valid_test & (np.abs(t_j) >= t_crit_nom)) * 100.0)
        pct_sig_sdf = float(np.mean(valid_test & (np.abs(t_j) >= t_crit_sdf)) * 100.0)

        # Benjamini-Yekutieli (2001) FDR across non-dropped, non-regularized finite local tests
        p_for_fdr = np.where(valid_test, p_j, 1.0)
        order = np.argsort(p_for_fdr)
        ranks = np.arange(1, n + 1, dtype=np.float64)
        by_thresh = (ranks / (n * c_harmonic)) * 0.05
        passed = np.where(p_for_fdr[order] <= by_thresh)[0]
        if len(passed) > 0:
            cutoff_p = float(p_for_fdr[order[np.max(passed)]])
            by_mask = valid_test & (p_for_fdr <= cutoff_p)
        else:
            by_mask = np.zeros(n, dtype=bool)
        by_sig_mat[:, j] = by_mask
        pct_sig_fdr = float(np.mean(by_mask) * 100.0)
        mc_p = round(float(mc_p_vals[j]), 4) if mc_p_vals is not None else None

        param_summaries.append({
            "name": meta["name"],
            "label": meta["label"],
            "ols_beta": round(float(ols_global["beta"][j]), 4),
            "ols_se": round(ols_se_j, 4),
            "min": round(vmin, 4),
            "q1": round(q1, 4),
            "median": round(med, 4),
            "q3": round(q3, 4),
            "max": round(vmax, 4),
            "iqr": round(iqr, 4),
            "n_local_active": int(np.sum(fin_mask)),
            "pct_sig_95": round(pct_sig_nom, 1),
            "pct_sig_fdr": round(pct_sig_fdr, 1),
            "pct_sig_sdf": round(pct_sig_sdf, 1),
            "non_stationary": non_stat,
            "mc_nonstat_p": mc_p,
            "mgwr_bw": mgwr_bandwidths.get(meta["name"]) if mgwr_bandwidths else int(opt_bw),
            "stationarity_verdict": "Non-stationary (FBC IQR > 2·SE_OLS)" if non_stat else "Stationary (FBC IQR <= 2·SE_OLS)",
        })

    bw_curve = [
        {
            "bw": b,
            "aicc": round(v["aicc"], 2),
            "obj_score": round(v["obj_score"], 2),
            "pct_regularized": v["pct_regularized"],
            "n_local_dropped_locs": v["n_local_dropped_locs"],
        }
        for b, v in sorted(eval_history.items())
    ]

    if progress_cb:
        progress_cb(1.0, "GWR/MGWR kész.")

    return {
        "ok": True,
        "mode": gwr_mode,
        "n_obs": n,
        "p_params": p,
        "optimal_bw": int(opt_bw),
        "mgwr_bandwidths": mgwr_bandwidths,
        "mgwr_converged": mgwr_converged,
        "mgwr_iters": mgwr_iters,
        "bw_pct_sample": round(opt_bw / n * 100.0, 1),
        "b_min_floor": int(b_min),
        "n_regularized": int(final_res["n_regularized"]),
        "pct_regularized": final_res["pct_regularized"],
        "n_local_dropped_locs": int(final_res.get("n_local_dropped_locs", 0)),
        "is_regularized": final_res["is_regularized"].tolist(),
        "n_local_dropped": final_res["n_local_dropped"].tolist(),
        "global_ols_r2": round(float(ols_global["r2"]), 4),
        "global_ols_adj_r2": round(float(ols_global["adj_r2"]), 4),
        "global_ols_aic": round(float(ols_global["aic"]), 2),
        "gwr_r2": round(float(final_res["gwr_r2"]), 4),
        "gwr_adj_r2": round(float(final_res["gwr_adj_r2"]), 4),
        "gwr_aic": round(float(final_res["aic"]), 2),
        "gwr_aicc": round(float(final_res["aicc"]), 2),
        "gwr_bic": round(float(final_res["bic"]), 2),
        "tr_s": round(tr_s_final, 2),
        "eff_df": round(eff_df, 2),
        "t_crit_nom": round(t_crit_nom, 3),
        "t_crit_sdf": round(t_crit_sdf, 3),
        "bw_curve": bw_curve,
        "param_summaries": param_summaries,
        "col_names": [m["name"] for m in meta_use],
        "betas_native": np.round(betas_nat, 5).tolist(),
        "se_native": np.round(final_res["se_native"], 5).tolist(),
        "t_vals": np.round(t_mat, 3).tolist(),
        "fdr_sig_mask": by_sig_mat.tolist(),
        "local_r2": np.round(final_res["local_r2"], 4).tolist(),
        "local_kappas": np.round(final_res["local_kappas"], 1).tolist(),
        "resid_native": np.round(final_res["resid_native"], 5).tolist(),
        "warnings": warnings,
    }
