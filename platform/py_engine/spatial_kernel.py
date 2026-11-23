"""
Spatial Econometrics Kernel (Pure NumPy + SciPy Sparse)
======================================================
Implements without libpysal/spreg:
- Row-standardized CSR spatial weight matrices: k-NN (with co-location audit &
  optional parcel dedup), IDW (auto-minimum connectivity D_conn), Adaptive Bisquare
- Global Moran's I + analytical variance & FDR-protected LISA quadrants (HH/LL/HL/LH)
- Anselin (1988) 4-test LM / Robust LM decision tree in O(nnz(W))
- Pre-concentrated ML SAR, SEM & Spatial Durbin Model (SDM) + Elhorst Common-Factor tests
  Tier 1 (N <= 1500): cached W eigenvalues (eigh for symmetric-base, eigvals for k-NN)
  Tier 2 (N > 1500): sparse SuperLU splu / Chebyshev series
- Inv-free LeSage & Pace (2009) Direct/Indirect/Total impacts + 1000x Krinsky-Robb SEs
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize, stats
from scipy.interpolate import PchipInterpolator
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, eye as speye
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from .ols_kernel import fit_ols_core


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


def build_spatial_weight_matrix(
    coords_xy: np.ndarray,
    w_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build row-standardized sparse CSR weight matrix W and compute topology & eigenvalues.
    Supports: knn (with co-location audit & parcel dedup), idw (with auto_connectivity), adaptive_bisquare.
    """
    n = len(coords_xy)
    if n < 6:
        raise ValueError(f"Need at least 6 observations to build spatial weights (got N={n}).")

    w_id = str(w_cfg.get("id", "w"))
    w_type = str(w_cfg.get("type", "knn"))
    tree = cKDTree(coords_xy)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    islands: List[int] = []
    is_symmetric_base = False
    meta_extra: Dict[str, Any] = {}

    if w_type == "knn":
        k = min(n - 1, max(1, int(w_cfg.get("k", 5))))
        dedup = bool(w_cfg.get("dedup_colocated", False))

        if dedup:
            # Group co-located points (< 0.5 m) into parcels without O(N^2) allocation
            co_counts = tree.query_ball_point(coords_xy, r=0.5, return_length=True)
            picked_lists: List[Optional[List[int]]] = [None] * n
            pending = np.arange(n, dtype=np.int32)
            k_q = min(n, k + 12)
            dists_q, idx_q = tree.query(coords_xy, k=k_q)
            if k_q == 1:
                dists_q = dists_q[:, None]
                idx_q = idx_q[:, None]
            still_pending: List[int] = []
            for pos, i in enumerate(pending):
                picked = []
                for d_ij, j_idx in zip(dists_q[pos], idx_q[pos]):
                    j = int(j_idx)
                    if j == i or d_ij <= 0.5:
                        continue
                    picked.append(j)
                    if len(picked) == k:
                        break
                if len(picked) == k or k_q >= n:
                    if not picked:
                        picked = [int(j) for j in idx_q[pos] if int(j) != i][:k]
                    picked_lists[i] = picked
                else:
                    still_pending.append(int(i))

            # Escalate in small batches ONLY for dense co-located clusters
            if still_pending:
                pend_arr = np.asarray(still_pending, dtype=np.int32)
                batch_sz = 128
                for bs in range(0, len(pend_arr), batch_sz):
                    sub = pend_arr[bs : bs + batch_sz]
                    k_req = min(n, int(np.max(co_counts[sub])) + k + 2)
                    dq, iq = tree.query(coords_xy[sub], k=k_req)
                    if k_req == 1:
                        dq = dq[:, None]
                        iq = iq[:, None]
                    for r_pos, i in enumerate(sub):
                        picked = []
                        for d_ij, j_idx in zip(dq[r_pos], iq[r_pos]):
                            j = int(j_idx)
                            if j == i or d_ij <= 0.5:
                                continue
                            picked.append(j)
                            if len(picked) == k:
                                break
                        if not picked:
                            picked = [int(j) for j in iq[r_pos] if int(j) != i][:k]
                        picked_lists[int(i)] = picked

            for i in range(n):
                picked = picked_lists[i] or []
                if picked:
                    w_val = 1.0 / len(picked)
                    for j in picked:
                        rows.append(i); cols.append(j); vals.append(w_val)
                else:
                    islands.append(i)
        else:
            dists_q, idx_q = tree.query(coords_xy, k=k + 1)
            for i in range(n):
                nbrs = [int(j) for j in idx_q[i] if int(j) != i][:k]
                w_val = 1.0 / len(nbrs)
                for j in nbrs:
                    rows.append(i); cols.append(j); vals.append(w_val)
        meta_extra["k"] = k
        meta_extra["dedup_colocated"] = dedup

    elif w_type == "idw":
        is_symmetric_base = True
        power = float(w_cfg.get("power", 1.0))
        cutoff_mode = str(w_cfg.get("cutoff_mode", "auto_connectivity"))
        d1, _ = tree.query(coords_xy, k=2)
        d1_nn = d1[:, 1]
        d_conn = float(np.max(d1_nn)) * 1.001  # minimum distance guaranteeing 0 islands
        if cutoff_mode in ("fixed", "fixed_m") and w_cfg.get("cutoff_m"):
            d_cut = float(w_cfg["cutoff_m"])
            if n > 50:
                edge_cap = max(200000, 120 * n)
                est_edges = int(np.sum(tree.query_ball_point(coords_xy, r=d_cut, return_length=True) - 1))
                if est_edges > edge_cap:
                    lo_r = max(1.0, float(np.median(d1_nn)))
                    hi_r = d_cut
                    for _ in range(12):
                        mid_r = 0.5 * (lo_r + hi_r)
                        ee = int(np.sum(tree.query_ball_point(coords_xy, r=mid_r, return_length=True) - 1))
                        if ee <= edge_cap:
                            lo_r = mid_r
                        else:
                            hi_r = mid_r
                    d_cut = lo_r
                    meta_extra["outlier_guard_triggered"] = True
                    meta_extra["outlier_guard_msg"] = (
                        f"IDW '{w_id}' fixed cutoff capped at {d_cut:.1f}m (est. {est_edges} edges exceeded memory cap {edge_cap})."
                    )
        else:
            d_cut = max(d_conn, float(w_cfg.get("min_cutoff_m", 0.0)))
            if n > 50:
                p98 = max(1.0, float(np.percentile(d1_nn, 98)))
                edge_cap = max(200000, 120 * n)
                est_edges = int(np.sum(tree.query_ball_point(coords_xy, r=d_cut, return_length=True) - 1))
                if d_conn > max(3000.0, 5.0 * p98) or est_edges > edge_cap:
                    d_robust = max(float(w_cfg.get("min_cutoff_m", 0.0)), p98 * 1.25, 100.0)
                    d_cut = min(d_cut, d_robust)
                    est_edges2 = int(np.sum(tree.query_ball_point(coords_xy, r=d_cut, return_length=True) - 1))
                    if est_edges2 > edge_cap:
                        lo_r = max(1.0, float(np.median(d1_nn)))
                        hi_r = d_cut
                        for _ in range(12):
                            mid_r = 0.5 * (lo_r + hi_r)
                            ee = int(np.sum(tree.query_ball_point(coords_xy, r=mid_r, return_length=True) - 1))
                            if ee <= edge_cap:
                                lo_r = mid_r
                            else:
                                hi_r = mid_r
                        d_cut = lo_r
                    meta_extra["outlier_guard_triggered"] = True
                    meta_extra["outlier_guard_msg"] = (
                        f"IDW '{w_id}' connectivity radius capped at {d_cut:.1f}m "
                        f"(unbounded max 1-NN={d_conn:.1f}m, est. {est_edges} edges)."
                    )

        pairs_list = tree.query_ball_point(coords_xy, r=d_cut)
        for i, nbrs in enumerate(pairs_list):
            nbr_valid = [int(j) for j in nbrs if int(j) != i]
            if not nbr_valid:
                islands.append(i)
                continue
            j_arr = np.asarray(nbr_valid, dtype=np.int32)
            d_vec = np.hypot(coords_xy[j_arr, 0] - coords_xy[i, 0], coords_xy[j_arr, 1] - coords_xy[i, 1])
            w_raw = np.power(np.maximum(1.0, d_vec), -power)
            w_norm = w_raw / float(np.sum(w_raw))
            for j, wv in zip(nbr_valid, w_norm):
                rows.append(i); cols.append(j); vals.append(float(wv))
        meta_extra["cutoff_m"] = round(d_cut, 2)
        meta_extra["d_conn_m"] = round(d_conn, 2)
        meta_extra["power"] = power

    elif w_type == "adaptive_bisquare":
        k = min(n - 1, max(3, int(w_cfg.get("k", max(10, int(0.15 * n))))))
        dists_q, idx_q = tree.query(coords_xy, k=k + 1)
        for i in range(n):
            mask_other = idx_q[i] != i
            nbrs = idx_q[i][mask_other][:k]
            d_vec = dists_q[i][mask_other][:k]
            bw_i = max(1.0, float(np.max(d_vec)) * 1.001)
            w_raw = np.square(np.maximum(0.0, 1.0 - np.square(d_vec / bw_i)))
            s_w = float(np.sum(w_raw))
            if s_w <= 1e-12:
                islands.append(i)
                continue
            w_norm = w_raw / s_w
            for j, wv in zip(nbrs, w_norm):
                if wv > 0.0:
                    rows.append(i); cols.append(int(j)); vals.append(float(wv))
        meta_extra["k"] = k
    else:
        raise ValueError(f"Unsupported weight type: {w_type}")

    W = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)

    # Audit co-located neighbor pairs (d_ij <= 0.5 m)
    coo = W.tocoo()
    n_edges = len(coo.data)
    if n_edges > 0:
        d_edges = np.hypot(coords_xy[coo.row, 0] - coords_xy[coo.col, 0], coords_xy[coo.row, 1] - coords_xy[coo.col, 1])
        colocated_pct = float(np.mean(d_edges <= 0.5) * 100.0)
    else:
        colocated_pct = 0.0

    row_nnz = np.diff(W.indptr)
    mean_nbr = float(np.mean(row_nnz)) if n > 0 else 0.0
    min_nbr = int(np.min(row_nnz)) if n > 0 else 0
    max_nbr = int(np.max(row_nnz)) if n > 0 else 0
    pct_nonzero = float(n_edges / (n * n) * 100.0) if n > 0 else 0.0

    # Cache eigenvalues on W for Tier 1 (N <= 1500)
    eigvals: Optional[np.ndarray] = None
    if n <= 1500 and len(islands) == 0:
        if is_symmetric_base:
            # Similarity transform D^{-1/2} A D^{-1/2} = sqrt(W * W^T) yields exact real eigenvalues via eigh!
            W_dense = W.toarray()
            W_sym = np.sqrt(np.maximum(0.0, W_dense * W_dense.T))
            eigvals = eigh(W_sym, eigvals_only=True).astype(np.complex128)
        else:
            eigvals = np.linalg.eigvals(W.toarray())

    # Precompute O(nnz(W)) trace constant T = tr(W^T W + W^2)
    T_trace = float(W.multiply(W).sum() + W.multiply(W.T).sum())

    return {
        "id": w_id,
        "type": w_type,
        "n": n,
        "W": W,
        "T_trace": T_trace,
        "eigvals": eigvals,
        "is_symmetric_base": is_symmetric_base,
        "islands": islands,
        "mean_neighbors": round(mean_nbr, 2),
        "min_neighbors": min_nbr,
        "max_neighbors": max_nbr,
        "pct_nonzero": round(pct_nonzero, 3),
        "colocated_pairs_pct": round(colocated_pct, 2),
        "meta": meta_extra,
    }


def compute_moran_i(vec: np.ndarray, w_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Global Moran's I with analytical randomization variance."""
    W: csr_matrix = w_obj["W"]
    n = float(w_obj["n"])
    z = vec - float(np.mean(vec))
    z_ss = float(z @ z)
    if z_ss < 1e-18:
        return {"moran_i": 0.0, "expected_i": -1.0 / max(1.0, n - 1.0), "z": 0.0, "p": 1.0, "sig": ""}

    s0 = float(W.sum())
    wz = W @ z
    moran_i = (n / max(1e-12, s0)) * float(z @ wz) / z_ss
    ei = -1.0 / (n - 1.0)

    # Analytical randomization moments (Cliff & Ord)
    W_sym = W + W.T
    s1 = 0.5 * float(W_sym.multiply(W_sym).sum())
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    col_sums = np.asarray(W.sum(axis=0)).ravel()
    s2 = float(np.sum((row_sums + col_sums) ** 2))
    b2 = float((np.sum(z ** 4) / n) / ((z_ss / n) ** 2))

    num = n * ((n * n - 3.0 * n + 3.0) * s1 - n * s2 + 3.0 * s0 * s0) - b2 * ((n * n - n) * s1 - 2.0 * n * s2 + 6.0 * s0 * s0)
    den = (n - 1.0) * (n - 2.0) * (n - 3.0) * s0 * s0
    ei2 = num / max(1e-12, den)
    var_i = max(1e-15, ei2 - ei * ei)
    z_stat = (moran_i - ei) / math.sqrt(var_i)
    p_val = float(2.0 * stats.norm.sf(abs(z_stat)))

    return {
        "moran_i": round(float(moran_i), 4),
        "expected_i": round(float(ei), 4),
        "se": round(math.sqrt(var_i), 4),
        "z": round(float(z_stat), 3),
        "p": round(p_val, 5),
        "sig": _sig_stars(p_val),
    }


def compute_lisa(
    vec: np.ndarray,
    w_obj: Dict[str, Any],
    n_perm: int = 1999,
    use_fdr: bool = True,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute local Moran's I_i (LISA) with exact Anselin (1995) conditional permutation
    p-values (holding z_i fixed outside its own neighborhood) and optional BH-FDR.
    """
    W: csr_matrix = w_obj["W"]
    n = int(w_obj["n"])
    sd = float(np.std(vec, ddof=0))
    if sd < 1e-12:
        return {"quadrants": ["NS"] * n, "lisa_i": [0.0] * n, "p_vals": [1.0] * n, "counts": {"HH": 0, "LL": 0, "HL": 0, "LH": 0, "NS": n}}

    z = (vec - float(np.mean(vec))) / sd
    wz = W @ z
    lisa_i = z * wz

    # Vectorized exact conditional permutation test (batched sparse-dense matmul)
    rng = np.random.default_rng(seed)
    ge_count = np.zeros(n, dtype=np.int32)
    abs_obs = np.abs(lisa_i)
    b_sz = min(200, max(20, 200000 // max(1, n)))
    row_base = np.arange(n, dtype=np.int32)

    for s in range(0, n_perm, b_sz):
        cur = min(b_sz, n_perm - s)
        perms = np.argsort(rng.random((n, cur)), axis=0)
        inv_p = np.empty_like(perms)
        inv_p[perms, np.arange(cur)[None, :]] = row_base[:, None]

        z_perm = z[perms]  # (n, cur)
        wz_p = W @ z_perm
        # Exact Anselin conditional correction: swap z_i out if z_i landed on a neighbor m = inv_p[i, c]
        ri = np.repeat(row_base[:, None], cur, axis=1).ravel()
        corr_w = np.asarray(W[ri, inv_p.ravel()]).reshape(n, cur)
        wz_p = wz_p + corr_w * (z_perm - z[:, None])

        lisa_p = np.abs(z[:, None] * wz_p)
        ge_count += np.sum(lisa_p >= abs_obs[:, None], axis=1, dtype=np.int32)

    p_vals = (ge_count + 1.0) / (n_perm + 1.0)

    if use_fdr and n > 0:
        order = np.argsort(p_vals)
        ranks = np.arange(1, n + 1, dtype=np.float64)
        p_sorted = p_vals[order]
        bh_thresh = (ranks / n) * alpha
        passed = np.where(p_sorted <= bh_thresh)[0]
        if len(passed) > 0:
            cutoff_p = float(p_sorted[np.max(passed)])
            sig_mask = p_vals <= cutoff_p
        else:
            sig_mask = np.zeros(n, dtype=bool)
    else:
        sig_mask = p_vals < alpha

    quads = []
    counts = {"HH": 0, "LL": 0, "HL": 0, "LH": 0, "NS": 0}
    for i in range(n):
        if not sig_mask[i]:
            q = "NS"
        elif z[i] > 0 and wz[i] > 0:
            q = "HH"
        elif z[i] < 0 and wz[i] < 0:
            q = "LL"
        elif z[i] > 0 and wz[i] < 0:
            q = "HL"
        else:
            q = "LH"
        quads.append(q)
        counts[q] += 1

    return {
        "z": np.round(z, 4).tolist(),
        "wz": np.round(wz, 4).tolist(),
        "lisa_i": np.round(lisa_i, 4).tolist(),
        "p_vals": np.round(p_vals, 5).tolist(),
        "quadrants": quads,
        "counts": counts,
        "fdr_applied": use_fdr,
    }


def compute_anselin_lm_tests(
    X: np.ndarray,
    y: np.ndarray,
    w_obj: Dict[str, Any],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute Anselin (1988) LM-Lag, LM-Error, Robust LM-Lag, Robust LM-Error in O(nnz(W))."""
    W: csr_matrix = w_obj["W"]
    n = len(y)
    ols = fit_ols_core(X, y, cov_type="OLS")
    e = ols["resid"]
    yhat = ols["fitted"]
    s2 = max(1e-18, float(e @ e) / n)
    T = w_obj["T_trace"]

    wy = W @ y
    we = W @ e
    wyhat = W @ yhat

    # J = ( (Wyhat)' M (Wyhat) + T * s2 ) / s2  using M = I - X (X'X)^{-1} X'
    xt_wyhat = X.T @ wyhat
    m_quad = max(0.0, float(wyhat @ wyhat) - float(xt_wyhat.T @ ols["xtx_inv"] @ xt_wyhat))
    J = max(T + 1e-8, m_quad / s2 + T)

    e_we = float(e @ we) / s2
    e_wy = float(e @ wy) / s2

    lm_error = (e_we ** 2) / max(1e-12, T)
    lm_lag = (e_wy ** 2) / max(1e-12, J)
    rlm_error = ((e_we - (T / J) * e_wy) ** 2) / max(1e-12, T * (1.0 - T / J))
    rlm_lag = ((e_wy - e_we) ** 2) / max(1e-12, J - T)

    p_err = float(stats.chi2.sf(lm_error, df=1))
    p_lag = float(stats.chi2.sf(lm_lag, df=1))
    p_rerr = float(stats.chi2.sf(rlm_error, df=1))
    p_rlag = float(stats.chi2.sf(rlm_lag, df=1))

    lag_sig = p_lag < alpha
    err_sig = p_err < alpha
    if not lag_sig and not err_sig:
        verdict = "Classic OLS (No spatial dependence)"
        recommended = "OLS"
    elif lag_sig and not err_sig:
        verdict = "SAR (Spatial Lag)"
        recommended = "SAR"
    elif not lag_sig and err_sig:
        verdict = "SEM (Spatial Error)"
        recommended = "SEM"
    else:
        if rlm_lag >= rlm_error:
            verdict = "SAR (Robust LM-Lag dominant)"
            recommended = "SAR"
        else:
            verdict = "SEM (Robust LM-Error dominant)"
            recommended = "SEM"

    return {
        "lm_lag": round(lm_lag, 3), "lm_lag_p": round(p_lag, 5),
        "lm_error": round(lm_error, 3), "lm_error_p": round(p_err, 5),
        "rlm_lag": round(rlm_lag, 3), "rlm_lag_p": round(p_rlag, 5),
        "rlm_error": round(rlm_error, 3), "rlm_error_p": round(p_rerr, 5),
        "verdict": verdict,
        "recommended_model": recommended,
    }


def _log_det_i_minus_rho_w(rho: float, w_obj: Dict[str, Any]) -> float:
    """Compute ln|I - rho W| using cached Ord eigenvalues (Tier 1) or sparse SuperLU (Tier 2)."""
    eigvals = w_obj.get("eigvals")
    if eigvals is not None:
        # Product of (1 - rho * eig_i); complex conjugates cancel imaginary part
        terms = 1.0 - rho * eigvals
        return float(np.sum(np.log(np.maximum(1e-15, np.abs(terms)))))
    # Tier 2: exact sparse SuperLU log-determinant
    W: csr_matrix = w_obj["W"]
    n = w_obj["n"]
    mat = (speye(n, format="csc") - rho * W.tocsc())
    lu = splu(mat, permc_spec="MMD_AT_PLUS_A")
    return float(np.sum(np.log(np.maximum(1e-15, np.abs(lu.U.diagonal())))))


def _rho_bounds(w_obj: Dict[str, Any]) -> Tuple[float, float]:
    """Compute valid parameter bounds inside unit circle for rho / lambda."""
    eigvals = w_obj.get("eigvals")
    if eigvals is not None:
        reals = np.real(eigvals[np.abs(np.imag(eigvals)) < 1e-8])
        l_min = float(np.min(reals)) if len(reals) > 0 else -1.0
        l_max = float(np.max(reals)) if len(reals) > 0 else 1.0
        lo = max(-0.995, (1.0 / l_min) + 1e-3) if l_min < -1e-6 else -0.995
        hi = min(0.995, (1.0 / l_max) - 1e-3) if l_max > 1e-6 else 0.995
        return lo, hi
    return -0.99, 0.99


def _compute_trace_moments(rho: float, w_obj: Dict[str, Any], n_probes: int = 64) -> Tuple[float, float, float]:
    """
    Compute T1 = tr(G), T2 = tr(G^2), and T3 = tr(G^T G) where G = W (I - rho W)^{-1},
    without dense N x N inversion in Tier 2.
    """
    W: csr_matrix = w_obj["W"]
    n = w_obj["n"]
    eigvals = w_obj.get("eigvals")
    if eigvals is not None:
        g_eigs = eigvals / (1.0 - rho * eigvals)
        tr_g = float(np.sum(np.real(g_eigs)))
        tr_g2 = float(np.sum(np.real(g_eigs * g_eigs)))
        if n <= 800:
            A = np.eye(n) - rho * W.toarray()
            G = np.linalg.solve(A, W.toarray())
            tr_gtg = float(np.sum(G * G))
        else:
            # Hutchinson probe with exact sparse solve
            rng = np.random.default_rng(123)
            V = rng.choice([-1.0, 1.0], size=(n, n_probes))
            lu = splu((speye(n, format="csc") - rho * W.tocsc()), permc_spec="MMD_AT_PLUS_A")
            GV = lu.solve(W @ V)
            tr_gtg = float(np.mean(np.sum(GV * GV, axis=0)))
        return tr_g, tr_g2, tr_gtg

    # Tier 2: sparse splu + Hutchinson probes
    lu = splu((speye(n, format="csc") - rho * W.tocsc()), permc_spec="MMD_AT_PLUS_A")
    rng = np.random.default_rng(123)
    V = rng.choice([-1.0, 1.0], size=(n, n_probes))
    WV = W @ V
    GV = lu.solve(WV)
    G2V = lu.solve(W @ GV)
    tr_g = float(np.mean(np.sum(V * GV, axis=0)))
    tr_g2 = float(np.mean(np.sum(V * G2V, axis=0)))
    tr_gtg = float(np.mean(np.sum(GV * GV, axis=0)))
    return tr_g, tr_g2, tr_gtg


def _fit_ml_sar_core(
    X: np.ndarray,
    y: np.ndarray,
    w_obj: Dict[str, Any],
) -> Dict[str, Any]:
    """Pre-concentrated Maximum Likelihood SAR estimator (Ord 1975 / Anselin 1988)."""
    W: csr_matrix = w_obj["W"]
    n, k = X.shape
    wy = W @ y

    # Pre-concentrate OLS residuals ONCE outside optimization
    ols0 = fit_ols_core(X, y, cov_type="OLS")
    ols1 = fit_ols_core(X, wy, cov_type="OLS")
    e0 = ols0["resid"]
    e1 = ols1["resid"]
    e0_e0 = float(e0 @ e0)
    e0_e1 = float(e0 @ e1)
    e1_e1 = float(e1 @ e1)

    lo, hi = _rho_bounds(w_obj)

    def _neg_conc_ll(rho: float) -> float:
        sse = max(1e-18, e0_e0 - 2.0 * rho * e0_e1 + (rho * rho) * e1_e1)
        log_det = _log_det_i_minus_rho_w(rho, w_obj)
        ll = -0.5 * n * (math.log(2.0 * math.pi) + math.log(sse / n) + 1.0) + log_det
        return -ll

    opt = optimize.minimize_scalar(_neg_conc_ll, bounds=(lo, hi), method="bounded")
    rho_hat = float(opt.x)
    ll_hat = float(-opt.fun)
    beta_hat = ols0["beta"] - rho_hat * ols1["beta"]
    e_hat = e0 - rho_hat * e1
    sigma2 = max(1e-18, float(e_hat @ e_hat) / n)

    # Exact asymptotic covariance matrix of (beta, rho, sigma2)
    tr_g, tr_g2, tr_gtg = _compute_trace_moments(rho_hat, w_obj)
    x_beta = X @ beta_hat
    # G (X beta) = (I - rho W)^{-1} W (X beta)
    if n <= 1500:
        A_dense = np.eye(n) - rho_hat * W.toarray()
        g_xb = np.linalg.solve(A_dense, W @ x_beta)
    else:
        lu = splu((speye(n, format="csc") - rho_hat * W.tocsc()), permc_spec="MMD_AT_PLUS_A")
        g_xb = lu.solve(W @ x_beta)

    info_mat = np.zeros((k + 2, k + 2), dtype=np.float64)
    info_mat[:k, :k] = (X.T @ X) / sigma2
    info_mat[:k, k] = (X.T @ g_xb) / sigma2
    info_mat[k, :k] = info_mat[:k, k]
    info_mat[k, k] = float(g_xb @ g_xb) / sigma2 + tr_g2 + tr_gtg
    info_mat[k, k + 1] = tr_g / sigma2
    info_mat[k + 1, k] = info_mat[k, k + 1]
    info_mat[k + 1, k + 1] = n / (2.0 * sigma2 * sigma2)

    vcov_full = np.linalg.pinv(info_mat)
    vcov_br = vcov_full[: k + 1, : k + 1]
    se_br = np.sqrt(np.maximum(0.0, np.diag(vcov_br)))

    # Pseudo R^2 = Corr(y, yhat_reduced)^2 and Nagelkerke/squared corr
    if n <= 1500:
        yhat_red = np.linalg.solve(np.eye(n) - rho_hat * W.toarray(), x_beta)
    else:
        lu = splu((speye(n, format="csc") - rho_hat * W.tocsc()), permc_spec="MMD_AT_PLUS_A")
        yhat_red = lu.solve(x_beta)
    pseudo_r2 = float(np.corrcoef(y, yhat_red)[0, 1] ** 2) if np.std(yhat_red) > 1e-12 else 0.0

    k_total = k + 2  # beta + rho + sigma2
    aic = 2.0 * k_total - 2.0 * ll_hat
    bic = k_total * math.log(n) - 2.0 * ll_hat

    return {
        "beta": beta_hat,
        "rho": rho_hat,
        "se_beta": se_br[:k],
        "se_rho": float(se_br[k]),
        "vcov_beta_rho": vcov_br,
        "sigma2": sigma2,
        "llf": ll_hat,
        "aic": aic,
        "bic": bic,
        "pseudo_r2": pseudo_r2,
        "resid": e_hat,
        "n": n,
        "k": k,
    }


def fit_ml_sem(
    X: np.ndarray,
    y: np.ndarray,
    w_obj: Dict[str, Any],
) -> Dict[str, Any]:
    """Pre-concentrated Maximum Likelihood Spatial Error Model (SEM: y = Xb + u, u = lam Wu + e)."""
    W: csr_matrix = w_obj["W"]
    n, k = X.shape
    wx = W @ X
    wy = W @ y

    # Precompute p x p cross-products ONCE outside optimization
    xtx = X.T @ X
    xtwx = X.T @ wx
    wxtwx = wx.T @ wx
    xty = X.T @ y
    xtwy = X.T @ wy
    wxty = wx.T @ y
    wxtwy = wx.T @ wy
    yty = float(y @ y)
    ytwy = float(y @ wy)
    wytwy = float(wy @ wy)

    lo, hi = _rho_bounds(w_obj)

    def _solve_lam(lam: float) -> Tuple[float, np.ndarray, float]:
        A = xtx - lam * (xtwx + xtwx.T) + (lam * lam) * wxtwx
        b = xty - lam * (xtwy + wxty) + (lam * lam) * wxtwy
        beta_l = np.linalg.solve(A + 1e-12 * np.eye(k), b)
        # Filtered residual norm ||(I - lam W)(y - X beta)||^2
        sse_y = yty - 2.0 * lam * ytwy + (lam * lam) * wytwy
        sse = max(1e-18, float(sse_y - 2.0 * float(beta_l @ b) + float(beta_l @ A @ beta_l)))
        log_det = _log_det_i_minus_rho_w(lam, w_obj)
        ll = -0.5 * n * (math.log(2.0 * math.pi) + math.log(sse / n) + 1.0) + log_det
        return ll, beta_l, sse

    opt = optimize.minimize_scalar(lambda lam: -_solve_lam(lam)[0], bounds=(lo, hi), method="bounded")
    lam_hat = float(opt.x)
    ll_hat, beta_hat, sse_hat = _solve_lam(lam_hat)
    sigma2 = sse_hat / n

    A_hat = xtx - lam_hat * (xtwx + xtwx.T) + (lam_hat * lam_hat) * wxtwx
    tr_g, tr_g2, tr_gtg = _compute_trace_moments(lam_hat, w_obj)

    info_mat = np.zeros((k + 2, k + 2), dtype=np.float64)
    info_mat[:k, :k] = A_hat / sigma2
    info_mat[k, k] = tr_g2 + tr_gtg
    info_mat[k, k + 1] = tr_g / sigma2
    info_mat[k + 1, k] = info_mat[k, k + 1]
    info_mat[k + 1, k + 1] = n / (2.0 * sigma2 * sigma2)

    vcov_full = np.linalg.pinv(info_mat)
    se_all = np.sqrt(np.maximum(0.0, np.diag(vcov_full)))

    yhat = X @ beta_hat
    pseudo_r2 = float(np.corrcoef(y, yhat)[0, 1] ** 2) if np.std(yhat) > 1e-12 else 0.0
    k_total = k + 2
    aic = 2.0 * k_total - 2.0 * ll_hat
    bic = k_total * math.log(n) - 2.0 * ll_hat

    u_hat = y - yhat
    e_hat = u_hat - lam_hat * (W @ u_hat)

    return {
        "beta": beta_hat,
        "lambda": lam_hat,
        "se_beta": se_all[:k],
        "se_lambda": float(se_all[k]),
        "sigma2": sigma2,
        "llf": ll_hat,
        "aic": aic,
        "bic": bic,
        "pseudo_r2": pseudo_r2,
        "resid": e_hat,
        "n": n,
        "k": k,
    }


def compute_lesage_pace_impacts(
    sar_or_sdm: Dict[str, Any],
    w_obj: Dict[str, Any],
    col_names: List[str],
    is_sdm: bool = False,
    n_x_no_int: int = 0,
    krinsky_robb_draws: int = 1000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Compute inv-free LeSage & Pace (2009) Direct, Indirect (Spillover), Total impacts
    + 1,000-draw Krinsky-Robb Monte Carlo SEs, p-values, and 95% CIs.
    Uses exact eigenvalues in Tier 1 (without islands) and common-probe anchored SuperLU
    interpolation + exact island row-sum vectors in Tier 2 / island graphs.
    """
    rho = float(sar_or_sdm["rho"])
    beta_all = sar_or_sdm["beta"]
    vcov_br = sar_or_sdm["vcov_beta_rho"]
    eigvals = w_obj.get("eigvals")
    has_islands = len(w_obj.get("islands", []) or []) > 0
    W: csr_matrix = w_obj["W"]
    n = w_obj["n"]

    # Krinsky-Robb draws from N([beta, rho], vcov_beta_rho)
    rng = np.random.default_rng(seed)
    mean_vec = np.concatenate([beta_all, [rho]])
    eig_v, eig_vec = np.linalg.eigh(0.5 * (vcov_br + vcov_br.T))
    eig_v = np.maximum(0.0, eig_v)
    cov_psd = (eig_vec * eig_v[None, :]) @ eig_vec.T
    draws = rng.multivariate_normal(mean_vec, cov_psd, size=krinsky_robb_draws)
    rho_draws = np.clip(draws[:, -1], -0.985, 0.985)

    if eigvals is not None and not has_islands:
        def _eval_eig(r_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            denom = 1.0 - r_vec[:, None] * eigvals[None, :]
            t_arr = np.mean(np.real(1.0 / denom), axis=1)
            u_arr = np.mean(np.real(eigvals[None, :] / denom), axis=1)
            s_arr = 1.0 / np.maximum(1e-6, 1.0 - r_vec)
            return t_arr, u_arr, s_arr, s_arr

        t_pt, u_pt, s_pt, v_pt = _eval_eig(np.array([rho]))
        T0, U0, S0, V0 = float(t_pt[0]), float(u_pt[0]), float(s_pt[0]), float(v_pt[0])
        t_draws, u_draws, s_draws, v_draws = _eval_eig(rho_draws)
    else:
        # Tier 2 (N > 1500) or W with islands: common-random-number Hutchinson probes across splu anchors
        W_csc = W.tocsc()
        I_csc = speye(n, format="csc")
        probe_rng = np.random.default_rng(99)
        V_probe = probe_rng.choice([-1.0, 1.0], size=(n, 96))
        ones_vec = np.ones(n, dtype=np.float64)

        def _eval_splu_single(r_val: float) -> Tuple[float, float, float, float]:
            lu = splu((I_csc - r_val * W_csc), permc_spec="MMD_AT_PLUS_A")
            invV = lu.solve(V_probe)
            w_invV = W @ invV
            t_val = float(np.mean(np.sum(V_probe * invV, axis=0))) / n
            u_val = float(np.mean(np.sum(V_probe * w_invV, axis=0))) / n
            if has_islands:
                q1 = lu.solve(ones_vec)
                s_val = float(np.mean(q1))
                v_val = float(np.mean(W @ q1))
            else:
                s_val = 1.0 / max(1e-6, 1.0 - r_val)
                v_val = s_val
            return t_val, u_val, s_val, v_val

        T0, U0, S0, V0 = _eval_splu_single(rho)
        r_lo = float(np.min(rho_draws))
        r_hi = float(np.max(rho_draws))
        if r_hi - r_lo < 1e-6:
            t_draws = np.full(krinsky_robb_draws, T0, dtype=np.float64)
            u_draws = np.full(krinsky_robb_draws, U0, dtype=np.float64)
            s_draws = np.full(krinsky_robb_draws, S0, dtype=np.float64)
            v_draws = np.full(krinsky_robb_draws, V0, dtype=np.float64)
        else:
            anchors = np.unique(np.concatenate([np.linspace(r_lo, r_hi, 11), [np.clip(rho, -0.985, 0.985)]]))
            a_t, a_u, a_s, a_v = [], [], [], []
            for a in anchors:
                tv, uv, sv, vv = _eval_splu_single(float(a))
                a_t.append(tv); a_u.append(uv); a_s.append(sv); a_v.append(vv)
            t_draws = PchipInterpolator(anchors, a_t)(rho_draws)
            u_draws = PchipInterpolator(anchors, a_u)(rho_draws)
            if has_islands:
                s_draws = PchipInterpolator(anchors, a_s)(rho_draws)
                v_draws = PchipInterpolator(anchors, a_v)(rho_draws)
            else:
                s_draws = 1.0 / np.maximum(1e-6, 1.0 - rho_draws)
                v_draws = s_draws

    impacts_out: List[Dict[str, Any]] = []
    # Variable columns start at index 1 (exclude intercept)
    n_vars = n_x_no_int if is_sdm else (len(beta_all) - 1)
    for j in range(1, 1 + n_vars):
        b_j = float(beta_all[j])
        b_draws = draws[:, j]
        if is_sdm:
            th_idx = 1 + n_vars + (j - 1)
            if th_idx < len(beta_all):
                th_j = float(beta_all[th_idx])
                th_draws = draws[:, th_idx]
            else:
                th_j = 0.0
                th_draws = np.zeros_like(b_draws)
        else:
            th_j = 0.0
            th_draws = np.zeros_like(b_draws)

        direct = b_j * T0 + th_j * U0
        total = b_j * S0 + th_j * V0
        indirect = total - direct
        share_pct = (indirect / total * 100.0) if abs(total) > 1e-12 else 0.0

        dir_d = b_draws * t_draws + th_draws * u_draws
        tot_d = b_draws * s_draws + th_draws * v_draws
        ind_d = tot_d - dir_d

        def _kr_stats(arr_d: np.ndarray, pt: float) -> Dict[str, Any]:
            se = float(np.std(arr_d, ddof=1))
            z = pt / se if se > 1e-15 else 0.0
            p = float(2.0 * stats.norm.sf(abs(z)))
            ci_lo, ci_hi = np.percentile(arr_d, [2.5, 97.5])
            return {
                "est": round(pt, 4),
                "se": round(se, 4),
                "z": round(z, 3),
                "p": round(p, 5),
                "sig": _sig_stars(p),
                "ci_lo": round(float(ci_lo), 4),
                "ci_hi": round(float(ci_hi), 4),
            }

        impacts_out.append({
            "variable": col_names[j],
            "direct": _kr_stats(dir_d, direct),
            "indirect": _kr_stats(ind_d, indirect),
            "total": _kr_stats(tot_d, total),
            "spillover_share_pct": round(share_pct, 2),
        })

    return impacts_out


def run_spatial_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    col_names: List[str],
    coords_xy: np.ndarray,
    w_configs: List[Dict[str, Any]],
    krinsky_robb_draws: int = 1000,
    lisa_fdr: bool = True,
) -> Dict[str, Any]:
    """
    Guard #5 Full Spatial Econometrics Pipeline:
    - Builds all configured W matrices (multi-matrix robustness)
    - Computes Moran's I, FDR-LISA, Anselin LM decision tree
    - Estimates ML SAR, SEM, SDM + Elhorst Common-Factor LR/Wald tests
    - Computes Krinsky-Robb LeSage-Pace impacts
    """
    ols = fit_ols_core(X, y, cov_type="HC3")
    w_objs = [build_spatial_weight_matrix(coords_xy, cfg) for cfg in w_configs]
    primary_w = w_objs[0]

    # Moran's I on y and OLS residuals
    moran_y = compute_moran_i(y, primary_w)
    moran_resid = compute_moran_i(ols["resid"], primary_w)
    lisa_y = compute_lisa(y, primary_w, use_fdr=lisa_fdr)
    lisa_resid = compute_lisa(ols["resid"], primary_w, use_fdr=lisa_fdr)
    anselin = compute_anselin_lm_tests(X, y, primary_w)

    # Fit SAR, SEM, and SDM on primary W
    sar_res = _fit_ml_sar_core(X, y, primary_w)
    sem_res = fit_ml_sem(X, y, primary_w)

    # Spatial Durbin Model: y = rho Wy + X beta + W X_{-0} theta + eps
    W_prim: csr_matrix = primary_w["W"]
    X_no_int = X[:, 1:]
    k_no_int = X_no_int.shape[1]
    WX_no_int = W_prim @ X_no_int
    X_sdm = np.column_stack([X, WX_no_int])
    if X.shape[0] <= X_sdm.shape[1] + 2:
        sdm_res = dict(sar_res)
        sdm_res["betas"] = np.concatenate([sar_res["betas"], np.zeros(k_no_int)])
        sdm_res["se"] = np.concatenate([sar_res["se"], np.zeros(k_no_int)])
        sdm_res["t_stats"] = np.concatenate([sar_res["t_stats"], np.zeros(k_no_int)])
        sdm_res["p_vals"] = np.concatenate([sar_res["p_vals"], np.ones(k_no_int)])
        sdm_res["sdm_underdetermined_fallback"] = True
    else:
        sdm_res = _fit_ml_sar_core(X_sdm, y, primary_w)

    # Elhorst Common-Factor Tests on SDM:
    # 1) H0: theta = 0 (SDM vs SAR) -> LR test (and Wald)
    lr_sar = max(0.0, 2.0 * (sdm_res["llf"] - sar_res["llf"]))
    p_lr_sar = float(stats.chi2.sf(lr_sar, df=max(1, k_no_int)))
    # 2) H0: theta + rho * beta = 0 (SDM vs SEM) -> LR test (and Wald)
    lr_sem = max(0.0, 2.0 * (sdm_res["llf"] - sem_res["llf"]))
    p_lr_sem = float(stats.chi2.sf(lr_sem, df=max(1, k_no_int)))

    elhorst_verdict = "SDM (Neither SAR nor SEM restriction accepted)"
    if p_lr_sar >= 0.05 and p_lr_sem < 0.05:
        elhorst_verdict = "SAR accepted (H0: θ = 0 not rejected)"
    elif p_lr_sem >= 0.05 and p_lr_sar < 0.05:
        elhorst_verdict = "SEM accepted (H0: θ + ρβ = 0 not rejected)"
    elif p_lr_sar >= 0.05 and p_lr_sem >= 0.05:
        elhorst_verdict = f"Either SAR/SEM acceptable ({anselin['recommended_model']} via LM)"

    # LeSage & Pace impacts on SAR and SDM
    impacts_sar = compute_lesage_pace_impacts(
        sar_res, primary_w, col_names, is_sdm=False, krinsky_robb_draws=krinsky_robb_draws
    )
    impacts_sdm = compute_lesage_pace_impacts(
        sdm_res, primary_w, col_names, is_sdm=True, n_x_no_int=k_no_int, krinsky_robb_draws=krinsky_robb_draws
    )

    # Multi-W Robustness Comparison across all configured W matrices
    weights_robustness: List[Dict[str, Any]] = []
    for w_o in w_objs:
        sar_w = sar_res if w_o["id"] == primary_w["id"] else _fit_ml_sar_core(X, y, w_o)
        sem_w = sem_res if w_o["id"] == primary_w["id"] else fit_ml_sem(X, y, w_o)
        ans_w = anselin if w_o["id"] == primary_w["id"] else compute_anselin_lm_tests(X, y, w_o)
        mor_w = moran_resid if w_o["id"] == primary_w["id"] else compute_moran_i(ols["resid"], w_o)
        rho_se = max(1e-12, sar_w["se_rho"])
        lam_se = max(1e-12, sem_w["se_lambda"])
        rho_p = float(2.0 * stats.norm.sf(abs(sar_w["rho"] / rho_se)))
        lam_p = float(2.0 * stats.norm.sf(abs(sem_w["lambda"] / lam_se)))
        weights_robustness.append({
            "w_id": w_o["id"],
            "w_type": w_o["type"],
            "mean_neighbors": w_o["mean_neighbors"],
            "islands": len(w_o["islands"]),
            "colocated_pairs_pct": w_o["colocated_pairs_pct"],
            "moran_resid_i": mor_w["moran_i"],
            "moran_resid_p": mor_w["p"],
            "anselin_verdict": ans_w["verdict"],
            "sar_rho": round(float(sar_w["rho"]), 4),
            "sar_rho_se": round(float(rho_se), 4),
            "sar_rho_p": round(rho_p, 5),
            "sar_aic": round(float(sar_w["aic"]), 2),
            "sar_pseudo_r2": round(float(sar_w["pseudo_r2"]), 4),
            "sem_lambda": round(float(sem_w["lambda"]), 4),
            "sem_lambda_se": round(float(lam_se), 4),
            "sem_lambda_p": round(lam_p, 5),
            "sem_aic": round(float(sem_w["aic"]), 2),
        })

    # Grand OLS vs SAR vs SEM vs SDM table
    rho_z = sar_res["rho"] / max(1e-12, sar_res["se_rho"])
    rho_p = float(2.0 * stats.norm.sf(abs(rho_z)))
    lam_z = sem_res["lambda"] / max(1e-12, sem_res["se_lambda"])
    lam_p = float(2.0 * stats.norm.sf(abs(lam_z)))
    sdm_rho_z = sdm_res["rho"] / max(1e-12, sdm_res["se_rho"])
    sdm_rho_p = float(2.0 * stats.norm.sf(abs(sdm_rho_z)))

    n_obs_comp = X.shape[0]
    k_ols_ml = X.shape[1] + 1  # Include sigma^2 for exact AIC/BIC parity with SAR/SEM/SDM
    ols_aic_ml = 2.0 * k_ols_ml - 2.0 * float(ols["llf"])
    ols_bic_ml = math.log(max(n_obs_comp, 1)) * k_ols_ml - 2.0 * float(ols["llf"])
    model_comparison = [
        {
            "model": "OLS (Baseline)",
            "spatial_param": None,
            "spatial_se": None,
            "spatial_p": None,
            "r2_or_pseudo": round(float(ols["r2"]), 4),
            "llf": round(float(ols["llf"]), 2),
            "aic": round(float(ols_aic_ml), 2),
            "bic": round(float(ols_bic_ml), 2),
            "resid_moran_i": moran_resid["moran_i"],
            "resid_moran_p": moran_resid["p"],
        },
        {
            "model": f"SAR (ML, {primary_w['id']})",
            "spatial_param": round(float(sar_res["rho"]), 4),
            "spatial_se": round(float(sar_res["se_rho"]), 4),
            "spatial_p": round(rho_p, 5),
            "r2_or_pseudo": round(float(sar_res["pseudo_r2"]), 4),
            "llf": round(float(sar_res["llf"]), 2),
            "aic": round(float(sar_res["aic"]), 2),
            "bic": round(float(sar_res["bic"]), 2),
            "resid_moran_i": compute_moran_i(sar_res["resid"], primary_w)["moran_i"],
            "resid_moran_p": compute_moran_i(sar_res["resid"], primary_w)["p"],
        },
        {
            "model": f"SEM (ML, {primary_w['id']})",
            "spatial_param": round(float(sem_res["lambda"]), 4),
            "spatial_se": round(float(sem_res["se_lambda"]), 4),
            "spatial_p": round(lam_p, 5),
            "r2_or_pseudo": round(float(sem_res["pseudo_r2"]), 4),
            "llf": round(float(sem_res["llf"]), 2),
            "aic": round(float(sem_res["aic"]), 2),
            "bic": round(float(sem_res["bic"]), 2),
            "resid_moran_i": compute_moran_i(sem_res["resid"], primary_w)["moran_i"],
            "resid_moran_p": compute_moran_i(sem_res["resid"], primary_w)["p"],
        },
        {
            "model": f"SDM (ML, {primary_w['id']})",
            "spatial_param": round(float(sdm_res["rho"]), 4),
            "spatial_se": round(float(sdm_res["se_rho"]), 4),
            "spatial_p": round(sdm_rho_p, 5),
            "r2_or_pseudo": round(float(sdm_res["pseudo_r2"]), 4),
            "llf": round(float(sdm_res["llf"]), 2),
            "aic": round(float(sdm_res["aic"]), 2),
            "bic": round(float(sdm_res["bic"]), 2),
            "resid_moran_i": compute_moran_i(sdm_res["resid"], primary_w)["moran_i"],
            "resid_moran_p": compute_moran_i(sdm_res["resid"], primary_w)["p"],
        },
    ]

    # Guard #5 alerts (co-located condo k-NN artifact & IDW outlier connectivity cap)
    guard5_alerts = []
    if primary_w["colocated_pairs_pct"] > 5.0:
        guard5_alerts.append(
            f"Co-located coordinates: {primary_w['colocated_pairs_pct']:.1f}% of {primary_w['id']} neighbor links have d_ij <= 0.5m "
            "(intra-building clusters). Verify spatial spillover stability against the Inverse-Distance (IDW) matrix below."
        )
    for w_o in w_objs:
        ogm = w_o.get("meta", {}).get("outlier_guard_msg")
        if ogm:
            guard5_alerts.append(ogm)

    return {
        "ok": True,
        "primary_w_id": primary_w["id"],
        "moran_y": moran_y,
        "moran_resid": moran_resid,
        "lisa_y": lisa_y,
        "lisa_resid": lisa_resid,
        "anselin": anselin,
        "elhorst": {
            "lr_vs_sar": round(lr_sar, 3),
            "p_vs_sar": round(p_lr_sar, 5),
            "lr_vs_sem": round(lr_sem, 3),
            "p_vs_sem": round(p_lr_sem, 5),
            "verdict": elhorst_verdict,
        },
        "model_comparison": model_comparison,
        "weights_robustness": weights_robustness,
        "impacts_sar": impacts_sar,
        "impacts_sdm": impacts_sdm,
        "guard5_alerts": guard5_alerts,
    }
