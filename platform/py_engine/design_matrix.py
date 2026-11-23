"""
Parametric Design Matrix, Transformations & Guard #2 Sparse Auto-Merger
======================================================================
Builds regression design matrices dynamically from user-defined ExperimentRecipe
specifications (zero hardcoded variable names):
- Numeric parser supporting comma/space decimals
- y transforms: raw, log, log1p, Jacobian-adjusted profile-MLE Box-Cox + LR test
- Covariate transforms: raw, centered, zscore, log, poly, categorical reference encoding
- Spatial transforms: ln_d, linear, andersson_day, rings, exp_decay, circuity
- Guard #2: Automatic sparse category & ring merger (N_cell < tau) + interaction cascade
- Common-sample lock across active specs + 1-click M1..M4 generator
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize, stats


def _normalize_numeric_str(s: str) -> str:
    """Normalize international/EU/HU thousand and decimal separators safely."""
    s = s.replace("\xa0", "").replace(" ", "")
    if "," in s and "." in s:
        if s.rfind(".") > s.rfind(","):
            return s.replace(",", "")
        return s.replace(".", "").replace(",", ".")
    if s.count(",") > 1:
        return s.replace(",", "")
    if s.count(".") > 1:
        return s.replace(".", "")
    if "," in s:
        return s.replace(",", ".")
    return s


def parse_numeric_array(raw_values: List[Any]) -> np.ndarray:
    """Parse arbitrary string/number list (including EU/HU commas, spaces, and thousands) to float64 array."""
    out = np.full(len(raw_values), np.nan, dtype=np.float64)
    for i, val in enumerate(raw_values):
        if val is None or isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            fval = float(val)
            if math.isfinite(fval):
                out[i] = fval
            continue
        s = str(val).strip()
        if not s or s.lower() in ("na", "nan", "null", "none", "n/a", "-"):
            continue
        s = _normalize_numeric_str(s)
        try:
            fval = float(s)
            if math.isfinite(fval):
                out[i] = fval
        except ValueError:
            continue
    return out


def parse_binary_flag(raw_values: List[Any]) -> np.ndarray:
    """Parse column values into binary 0/1 indicator (NaN -> 0)."""
    out = np.zeros(len(raw_values), dtype=np.float64)
    for i, val in enumerate(raw_values):
        if val is None:
            continue
        if isinstance(val, bool):
            out[i] = 1.0 if val else 0.0
            continue
        if isinstance(val, (int, float)):
            if math.isfinite(val) and val >= 1.0:
                out[i] = 1.0
            continue
        s = str(val).strip().lower()
        if s in ("1", "true", "yes", "igen", "ja", "y"):
            out[i] = 1.0
        else:
            try:
                if float(_normalize_numeric_str(s)) >= 1.0:
                    out[i] = 1.0
            except ValueError:
                pass
    return out


def _boxcox_transform(y: np.ndarray, lam: float) -> np.ndarray:
    """Compute Box-Cox transformation (y^lam - 1) / lam safely near lam=0."""
    if abs(lam) < 1e-6:
        return np.log(y)
    return (np.power(y, lam) - 1.0) / lam


def _boxcox_profile_mle(y: np.ndarray, X: Optional[np.ndarray] = None) -> Tuple[float, float, float, float]:
    """
    Estimate profile-MLE Box-Cox lambda on strictly positive y (conditional on X if provided,
    otherwise intercept-only unconditional Jacobian LL), returning (lambda_hat, ll_hat, p_vs_log, p_vs_linear).
    """
    n = len(y)
    sum_log_y = float(np.sum(np.log(y)))

    def _neg_ll(lam: float) -> float:
        yt = _boxcox_transform(y, lam)
        if X is not None and len(X) == n and X.shape[1] >= 1 and n > X.shape[1]:
            beta, _, _, _ = np.linalg.lstsq(X, yt, rcond=None)
            resid = yt - X @ beta
            var = float(np.mean(resid * resid))
        else:
            var = float(np.mean((yt - np.mean(yt)) ** 2))
        var = max(var, 1e-18)
        ll = -0.5 * n * math.log(var) + (lam - 1.0) * sum_log_y
        return -ll

    res = optimize.minimize_scalar(_neg_ll, bounds=(-2.0, 2.0), method="bounded")
    lam_hat = float(res.x)
    ll_hat = float(-res.fun)
    ll_log = float(-_neg_ll(0.0))
    ll_lin = float(-_neg_ll(1.0))

    lr_log = max(0.0, 2.0 * (ll_hat - ll_log))
    lr_lin = max(0.0, 2.0 * (ll_hat - ll_lin))
    p_vs_log = float(stats.chi2.sf(lr_log, df=1))
    p_vs_lin = float(stats.chi2.sf(lr_lin, df=1))
    return lam_hat, ll_hat, p_vs_log, p_vs_lin


def build_dependent_variable(
    columns: Dict[str, List[Any]],
    y_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct target vector y, raw y, validity mask, per-obs Jacobian log|g'(y)|, and Box-Cox diagnostics."""
    mode = y_spec.get("mode", "column")
    num_col = y_spec.get("num_col")
    den_col = y_spec.get("den_col")
    transform = y_spec.get("transform", "log")
    fixed_lam = y_spec.get("boxcox_lambda")
    shift = float(y_spec.get("log1p_shift", 1.0))

    if not num_col or num_col not in columns:
        raise ValueError(f"Dependent variable numerator column '{num_col}' not found.")

    num_arr = parse_numeric_array(columns[num_col])
    n = len(num_arr)
    if mode == "ratio":
        if not den_col or den_col not in columns:
            raise ValueError(f"Dependent variable denominator column '{den_col}' not found.")
        den_arr = parse_numeric_array(columns[den_col])
        valid_raw = np.isfinite(num_arr) & np.isfinite(den_arr) & (den_arr > 0.0)
        y_raw = np.full(n, np.nan, dtype=np.float64)
        y_raw[valid_raw] = num_arr[valid_raw] / den_arr[valid_raw]
        base_label = f"{num_col} / {den_col}"
    else:
        valid_raw = np.isfinite(num_arr)
        y_raw = num_arr.copy()
        base_label = str(num_col)

    y_trans = np.full(n, np.nan, dtype=np.float64)
    jacobian_log = np.zeros(n, dtype=np.float64)
    dropped_nonpos = 0
    boxcox_info: Optional[Dict[str, Any]] = None

    if transform == "raw":
        valid_y = valid_raw.copy()
        y_trans[valid_y] = y_raw[valid_y]
        label = base_label
    elif transform == "log":
        nonpos_mask = valid_raw & (y_raw <= 0.0)
        dropped_nonpos = int(np.sum(nonpos_mask))
        valid_y = valid_raw & (y_raw > 0.0)
        y_trans[valid_y] = np.log(y_raw[valid_y])
        jacobian_log[valid_y] = -np.log(y_raw[valid_y])
        label = f"ln({base_label})"
    elif transform == "log1p":
        nonpos_mask = valid_raw & ((y_raw + shift) <= 0.0)
        dropped_nonpos = int(np.sum(nonpos_mask))
        valid_y = valid_raw & ((y_raw + shift) > 0.0)
        y_trans[valid_y] = np.log(y_raw[valid_y] + shift)
        jacobian_log[valid_y] = -np.log(y_raw[valid_y] + shift)
        label = f"ln({base_label} + {shift:g})"
    elif transform == "boxcox":
        nonpos_mask = valid_raw & (y_raw <= 0.0)
        dropped_nonpos = int(np.sum(nonpos_mask))
        valid_y = valid_raw & (y_raw > 0.0)
        yv = y_raw[valid_y]
        if len(yv) < 5:
            raise ValueError("Insufficient positive observations for Box-Cox transformation.")
        lam_mle, ll_mle, p_log, p_lin = _boxcox_profile_mle(yv)
        lam_use = float(fixed_lam) if fixed_lam is not None else lam_mle
        y_trans[valid_y] = _boxcox_transform(yv, lam_use)
        jacobian_log[valid_y] = (lam_use - 1.0) * np.log(yv)
        label = f"BoxCox({base_label}, λ={lam_use:.3f})"
        boxcox_info = {
            "mode": "unconditional_intercept_only",
            "lambda_used": round(lam_use, 4),
            "lambda_mle": round(lam_mle, 4),
            "ll_mle": round(ll_mle, 3),
            "p_vs_log": round(p_log, 5),
            "p_vs_linear": round(p_lin, 5),
        }
    else:
        raise ValueError(f"Unsupported dependent transform: {transform}")

    return {
        "y": y_trans,
        "y_raw": y_raw,
        "valid_mask": valid_y,
        "jacobian_log": jacobian_log,
        "transform": transform,
        "is_log_scale": transform in ("log", "log1p"),
        "label": label,
        "base_label": base_label,
        "dropped_nonpos": dropped_nonpos,
        "boxcox_info": boxcox_info,
    }


def _evaluate_spatial_term(
    dist_dict: Dict[str, Dict[str, np.ndarray]],
    columns: Dict[str, List[Any]],
    term_spec: Dict[str, Any],
    n_rows: int,
    sparse_min_cell: int = 5,
    valid_row_mask: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    Build columns and Guard #2 audit entries for a single spatial_term spec.
    Supports OSM/GeoJSON target metrics AND pre-existing CSV distance columns.
    """
    target_id = term_spec.get("target_id", "")
    metric = term_spec.get("metric", "eucl")  # eucl | net | circuity_delta | circuity_ratio | csv_col
    transform = term_spec.get("transform", "andersson_day")
    floor_m = max(0.1, float(term_spec.get("floor_m", 1.0)))

    if metric == "csv_col":
        col_name = term_spec.get("csv_col") or target_id
        d_raw = parse_numeric_array(columns[col_name]) if col_name in columns else np.full(n_rows, np.nan)
    else:
        t_res = dist_dict.get(target_id, {})
        key_map = {
            "eucl": "dist_eucl",
            "net": "dist_net",
            "circuity_delta": "circuity_delta",
            "circuity_ratio": "circuity_ratio",
        }
        if transform in ("circuity_delta", "circuity_ratio"):
            arr_key = transform
        else:
            arr_key = key_map.get(metric, "dist_eucl")
        d_raw = np.asarray(t_res.get(arr_key, np.full(n_rows, np.nan)), dtype=np.float64)

    term_key_base = f"{target_id}::{metric}::{transform}"
    cols_out: List[Dict[str, Any]] = []
    guard2_events: List[Dict[str, Any]] = []
    footnotes: List[str] = []

    if transform == "andersson_day":
        c_m = max(floor_m + 1.0, float(term_spec.get("cutoff_m", 300.0)))
        vals = np.where(np.isfinite(d_raw), np.maximum(0.0, np.log(c_m / np.maximum(floor_m, d_raw))), np.nan)
        cols_out.append({
            "name": f"AD({target_id},{metric},c={c_m:g}m)",
            "ref": f"{target_id}::andersson_day",
            "label": f"AD {target_id} ({metric}, c={c_m:g}m)",
            "values": vals,
            "kind": "spatial_ad",
            "is_binary": False,
            "meta": {"target_id": target_id, "metric": metric, "cutoff_m": c_m, "floor_m": floor_m, "d_raw": d_raw},
        })
    elif transform == "ln_d":
        vals = np.where(np.isfinite(d_raw), np.log(np.maximum(floor_m, d_raw)), np.nan)
        cols_out.append({
            "name": f"ln({target_id}_{metric})",
            "ref": f"{target_id}::ln_d",
            "label": f"ln(dist {target_id}, {metric})",
            "values": vals,
            "kind": "spatial_continuous",
            "is_binary": False,
            "meta": {"target_id": target_id, "metric": metric, "floor_m": floor_m},
        })
    elif transform == "linear":
        scale = max(1e-6, float(term_spec.get("scale", 100.0)))
        vals = np.where(np.isfinite(d_raw), d_raw / scale, np.nan)
        suffix = f"/{scale:g}m" if scale != 1.0 else ""
        cols_out.append({
            "name": f"{target_id}_{metric}{suffix}",
            "ref": term_key_base,
            "label": f"{target_id} ({metric}{suffix})",
            "values": vals,
            "kind": "spatial_continuous",
            "is_binary": False,
            "meta": {"target_id": target_id, "metric": metric, "scale": scale},
        })
    elif transform == "exp_decay":
        half_life = max(1.0, float(term_spec.get("half_life_m", 250.0)))
        k_decay = math.log(2.0) / half_life
        vals = np.where(np.isfinite(d_raw), np.exp(-k_decay * np.maximum(0.0, d_raw)), np.nan)
        cols_out.append({
            "name": f"exp({target_id},h={half_life:g}m)",
            "ref": f"{target_id}::exp_decay",
            "label": f"Exp-decay {target_id} (h½={half_life:g}m)",
            "values": vals,
            "kind": "spatial_exp",
            "is_binary": False,
            "meta": {"target_id": target_id, "metric": metric, "half_life_m": half_life},
        })
    elif transform == "rings":
        raw_breaks = sorted({float(b) for b in term_spec.get("breakpoints_m", [150.0, 300.0, 500.0]) if float(b) > 0.0})
        if not raw_breaks:
            raw_breaks = [150.0, 300.0, 500.0]
        check_mask = valid_row_mask if valid_row_mask is not None else np.isfinite(d_raw)
        valid_mask = check_mask & np.isfinite(d_raw)

        # Build initial ordered rings: [0, b1], (b1, b2], ..., (b_{K-1}, b_K] vs reference > b_K
        bounds: List[Tuple[float, float]] = [(0.0, raw_breaks[0])]
        for idx in range(len(raw_breaks) - 1):
            bounds.append((raw_breaks[idx], raw_breaks[idx + 1]))

        def _ring_mask(lo: float, hi: float) -> np.ndarray:
            if lo == 0.0:
                return valid_mask & (d_raw >= 0.0) & (d_raw <= hi)
            return valid_mask & (d_raw > lo) & (d_raw <= hi)

        ring_items = [{"lo": lo, "hi": hi, "counts": int(np.sum(_ring_mask(lo, hi)))} for lo, hi in bounds]
        ref_lo = raw_breaks[-1]
        ref_count = int(np.sum(valid_mask & (d_raw > ref_lo)))

        # Drop empty rings (N=0) with audit event
        empty_rings = [r for r in ring_items if r["counts"] == 0]
        for r in empty_rings:
            guard2_events.append({
                "term": target_id,
                "type": "ring_empty_dropped",
                "detail": f"Empty ring [{r['lo']:g}, {r['hi']:g}]m (N=0) omitted.",
            })
        ring_items = [r for r in ring_items if r["counts"] > 0]

        # Guard #2: Enforce N_k >= sparse_min_cell on all rings AND reference zone (prevents dummy trap when ref_count==0!)
        changed = True
        while changed and len(ring_items) > 0:
            changed = False
            # 1) Merge any ring with N_k < sparse_min_cell
            for idx, r in enumerate(ring_items):
                if r["counts"] < sparse_min_cell:
                    if len(ring_items) == 1:
                        guard2_events.append({
                            "term": target_id,
                            "type": "ring_merged_to_ref",
                            "detail": f"Ring [{r['lo']:g}, {r['hi']:g}]m (N={r['counts']} < {sparse_min_cell}) merged into reference >{ref_lo:g}m.",
                        })
                        ref_count += r["counts"]
                        ring_items.pop(idx)
                        changed = True
                        break
                    if idx == 0:
                        nbr_idx = 1
                    elif idx == len(ring_items) - 1:
                        if ref_count <= ring_items[idx - 1]["counts"]:
                            guard2_events.append({
                                "term": target_id,
                                "type": "ring_merged_to_ref",
                                "detail": f"Outermost ring ({r['lo']:g}, {r['hi']:g}]m (N={r['counts']} < {sparse_min_cell}) merged into reference >{r['lo']:g}m.",
                            })
                            ref_lo = r["lo"]
                            ref_count += r["counts"]
                            ring_items.pop(idx)
                            changed = True
                            break
                        nbr_idx = idx - 1
                    else:
                        nbr_idx = idx - 1 if ring_items[idx - 1]["counts"] <= ring_items[idx + 1]["counts"] else idx + 1

                    lo_new = min(r["lo"], ring_items[nbr_idx]["lo"])
                    hi_new = max(r["hi"], ring_items[nbr_idx]["hi"])
                    cnt_new = r["counts"] + ring_items[nbr_idx]["counts"]
                    guard2_events.append({
                        "term": target_id,
                        "type": "ring_merged",
                        "detail": (
                            f"Guard #2 merged sparse ring [{r['lo']:g},{r['hi']:g}]m (N={r['counts']} < {sparse_min_cell}) "
                            f"with [{ring_items[nbr_idx]['lo']:g},{ring_items[nbr_idx]['hi']:g}]m -> [{lo_new:g},{hi_new:g}]m (N={cnt_new})."
                        ),
                    })
                    ring_items[nbr_idx] = {"lo": lo_new, "hi": hi_new, "counts": cnt_new}
                    ring_items.pop(idx)
                    changed = True
                    break

            if changed:
                continue

            # 2) Check reference zone: if ref_count < sparse_min_cell (e.g. ref_count == 0 -> dummy trap!), merge outermost ring into ref
            if ref_count < sparse_min_cell and len(ring_items) > 0:
                outer_r = ring_items.pop(-1)
                old_ref_count = ref_count
                ref_count += outer_r["counts"]
                ref_lo = outer_r["lo"]
                guard2_events.append({
                    "term": target_id,
                    "type": "ref_zone_merged",
                    "detail": (
                        f"Reference zone (N={old_ref_count} < {sparse_min_cell}) absorbed outermost ring "
                        f"({outer_r['lo']:g}, {outer_r['hi']:g}]m (N={outer_r['counts']}) -> new reference >{ref_lo:g}m (N={ref_count}) to prevent dummy trap."
                    ),
                })
                changed = True

        if guard2_events:
            footnotes.append(
                f"Guard #2 ({target_id} rings): " + "; ".join(e["detail"] for e in guard2_events)
            )

        for r in ring_items:
            m = _ring_mask(r["lo"], r["hi"])
            vals = np.where(np.isfinite(d_raw), m.astype(np.float64), np.nan)
            cname = f"ring_{target_id}_{r['lo']:g}_{r['hi']:g}"
            cols_out.append({
                "name": cname,
                "ref": cname,
                "label": f"{target_id} [{r['lo']:g}-{r['hi']:g}m]",
                "values": vals,
                "kind": "spatial_ring",
                "is_binary": True,
                "meta": {"target_id": target_id, "lo": r["lo"], "hi": r["hi"], "n_active": r["counts"]},
            })
    elif transform in ("circuity_delta", "circuity_ratio"):
        vals = d_raw.astype(np.float64)
        cols_out.append({
            "name": f"{transform}_{target_id}",
            "ref": f"{target_id}::{transform}",
            "label": f"{'Δd detour' if transform == 'circuity_delta' else 'Circuity τ'} ({target_id})",
            "values": vals,
            "kind": "spatial_circuity",
            "is_binary": False,
            "meta": {"target_id": target_id},
        })

    return cols_out, guard2_events, footnotes


def _evaluate_covariate(
    columns: Dict[str, List[Any]],
    cov_spec: Dict[str, Any],
    n_rows: int,
    sparse_min_cell: int = 5,
    valid_row_mask: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Build design columns for numeric, binary, or categorical covariates + Guard #2 merge."""
    col = cov_spec.get("col", "")
    if col not in columns:
        return [], [{"term": col, "type": "missing_column", "detail": f"Column '{col}' missing in dataset."}], []

    raw_vals = columns[col]
    cov_type = cov_spec.get("type", "numeric")
    transform = cov_spec.get("transform", "raw")
    cols_out: List[Dict[str, Any]] = []
    guard2_events: List[Dict[str, Any]] = []
    footnotes: List[str] = []

    check_mask = valid_row_mask if valid_row_mask is not None else np.ones(n_rows, dtype=bool)

    if cov_type == "categorical":
        str_vals = [str(v).strip() if (v is not None and str(v).strip() != "") else None for v in raw_vals]
        non_null_indices = [i for i in range(n_rows) if check_mask[i] and str_vals[i] is not None]
        if not non_null_indices:
            return [], [{"term": col, "type": "all_missing", "detail": f"Categorical '{col}' has 0 valid values."}], []

        counts: Dict[str, int] = {}
        for i in non_null_indices:
            counts[str_vals[i]] = counts.get(str_vals[i], 0) + 1  # type: ignore[index]

        ref_level = cov_spec.get("reference_level")
        if not ref_level or ref_level not in counts:
            ref_level = max(counts.keys(), key=lambda k: counts[k])

        # Guard #2: Pool sparse non-reference levels (< sparse_min_cell)
        sparse_levels = [lvl for lvl, cnt in counts.items() if lvl != ref_level and cnt < sparse_min_cell]
        kept_levels = [lvl for lvl, cnt in counts.items() if lvl != ref_level and cnt >= sparse_min_cell]

        remapped = list(str_vals)
        if sparse_levels:
            pooled_n = sum(counts[l] for l in sparse_levels)
            if pooled_n >= sparse_min_cell:
                other_lbl = f"Other (<{sparse_min_cell})"
                for i in range(n_rows):
                    if remapped[i] in sparse_levels:
                        remapped[i] = other_lbl
                kept_levels.append(other_lbl)
                msg = f"Guard #2 pooled sparse levels {sparse_levels} (N={pooled_n}) of '{col}' into '{other_lbl}'."
            else:
                for i in range(n_rows):
                    if remapped[i] in sparse_levels:
                        remapped[i] = ref_level
                msg = f"Guard #2 merged tiny levels {sparse_levels} (sum N={pooled_n} < {sparse_min_cell}) of '{col}' into reference '{ref_level}'."
            guard2_events.append({"term": col, "type": "categorical_pooled", "detail": msg})
            footnotes.append(msg)

        for lvl in kept_levels:
            arr = np.array([np.nan if remapped[i] is None else (1.0 if remapped[i] == lvl else 0.0) for i in range(n_rows)], dtype=np.float64)
            cols_out.append({
                "name": f"{col}[{lvl}]",
                "ref": f"{col}[{lvl}]",
                "base_col": col,
                "label": f"{col}: {lvl} (ref: {ref_level})",
                "values": arr,
                "kind": "dummy",
                "is_binary": True,
                "meta": {"reference": ref_level, "level": lvl},
            })
        return cols_out, guard2_events, footnotes

    # Numeric or binary covariate
    num_arr = parse_numeric_array(raw_vals)
    valid_vals = num_arr[check_mask & np.isfinite(num_arr)]
    if len(valid_vals) == 0:
        return [], [{"term": col, "type": "all_missing", "detail": f"Numeric '{col}' has 0 valid values."}], []

    uniq = np.unique(valid_vals)
    is_binary = len(uniq) == 2 and set(uniq.tolist()).issubset({0.0, 1.0})

    if is_binary:
        n1 = int(np.sum(valid_vals == 1.0))
        n0 = int(np.sum(valid_vals == 0.0))
        if n1 < sparse_min_cell or n0 < sparse_min_cell:
            msg = f"Guard #2 dropped sparse binary dummy '{col}' (N1={n1}, N0={n0} < {sparse_min_cell})."
            guard2_events.append({"term": col, "type": "sparse_binary_dropped", "detail": msg})
            footnotes.append(msg)
            return [], guard2_events, footnotes
        cols_out.append({
            "name": col,
            "ref": col,
            "base_col": col,
            "label": col,
            "values": num_arr,
            "kind": "dummy",
            "is_binary": True,
            "meta": {"n1": n1, "n0": n0},
        })
        return cols_out, guard2_events, footnotes

    if len(uniq) <= 1:
        guard2_events.append({"term": col, "type": "constant_dropped", "detail": f"Constant covariate '{col}' omitted."})
        return [], guard2_events, footnotes

    # Apply continuous transform
    if transform == "log":
        vals = np.where(np.isfinite(num_arr) & (num_arr > 0.0), np.log(np.maximum(1e-9, num_arr)), np.nan)
        lbl = f"ln({col})"
    elif transform == "centered":
        mu = float(np.mean(valid_vals))
        vals = num_arr - mu
        lbl = f"{col} (centered)"
    elif transform == "zscore":
        mu = float(np.mean(valid_vals))
        sd = max(1e-12, float(np.std(valid_vals, ddof=1)))
        vals = (num_arr - mu) / sd
        lbl = f"z({col})"
    elif transform == "poly":
        mu = float(np.mean(valid_vals))
        sd = max(1e-12, float(np.std(valid_vals, ddof=1)))
        c1 = (num_arr - mu) / sd
        cols_out.append({
            "name": col, "ref": col, "base_col": col, "label": col,
            "values": num_arr, "kind": "continuous", "is_binary": False, "meta": {},
        })
        cols_out.append({
            "name": f"{col}^2", "ref": f"{col}^2", "base_col": col, "label": f"{col}² (orth)",
            "values": c1 * c1, "kind": "continuous", "is_binary": False, "meta": {},
        })
        return cols_out, guard2_events, footnotes
    else:
        vals = num_arr
        lbl = col

    cols_out.append({
        "name": col if transform == "raw" else f"{transform}({col})",
        "ref": col,
        "base_col": col,
        "label": lbl,
        "values": vals,
        "kind": "continuous",
        "is_binary": False,
        "meta": {"transform": transform},
    })
    return cols_out, guard2_events, footnotes


def build_model_design(
    columns: Dict[str, List[Any]],
    dist_dict: Dict[str, Dict[str, np.ndarray]],
    model_spec: Dict[str, Any],
    y_info: Dict[str, Any],
    base_valid_mask: np.ndarray,
    sparse_min_cell: int = 5,
    override_valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Assemble complete design matrix X, y, column metadata, and Guard #2 audit trail
    for a single ModelSpec.
    """
    n_rows = len(y_info["y"])
    pre_mask = base_valid_mask & y_info["valid_mask"]
    if override_valid_mask is not None:
        pre_mask = pre_mask & override_valid_mask

    all_cols: List[Dict[str, Any]] = []
    guard2_events: List[Dict[str, Any]] = []
    footnotes: List[str] = []

    for cov in model_spec.get("covariates", []):
        c_list, g_ev, fnotes = _evaluate_covariate(columns, cov, n_rows, sparse_min_cell, pre_mask)
        all_cols.extend(c_list)
        guard2_events.extend(g_ev)
        footnotes.extend(fnotes)

    for st in model_spec.get("spatial_terms", []):
        c_list, g_ev, fnotes = _evaluate_spatial_term(dist_dict, columns, st, n_rows, sparse_min_cell, pre_mask)
        all_cols.extend(c_list)
        guard2_events.extend(g_ev)
        footnotes.extend(fnotes)

    # Index built terms by both exact name and ref for interaction resolution
    term_lookup: Dict[str, List[Dict[str, Any]]] = {}
    for c in all_cols:
        term_lookup.setdefault(c["name"], []).append(c)
        term_lookup.setdefault(c["ref"], []).append(c)
        if "base_col" in c:
            term_lookup.setdefault(c["base_col"], []).append(c)

    for inter in model_spec.get("interactions", []):
        left_key = inter.get("left", "")
        right_key = inter.get("right", "")
        center_cont = bool(inter.get("center_continuous", True))
        left_cands = term_lookup.get(left_key, [])
        right_cands = term_lookup.get(right_key, [])
        if not left_cands or not right_cands:
            msg = f"Guard #2 skipped interaction '{left_key} × {right_key}' because constituent term was dropped/absent."
            guard2_events.append({"term": f"{left_key}x{right_key}", "type": "interaction_cascade", "detail": msg})
            footnotes.append(msg)
            continue

        for lc in left_cands:
            for rc in right_cands:
                lv = lc["values"].copy()
                rv = rc["values"].copy()
                m_both = pre_mask & np.isfinite(lv) & np.isfinite(rv)
                if center_cont and not lc["is_binary"] and np.any(m_both):
                    lv = lv - float(np.mean(lv[m_both]))
                if center_cont and not rc["is_binary"] and np.any(m_both):
                    rv = rv - float(np.mean(rv[m_both]))
                prod = lv * rv
                # Verify binary interaction cell count if both binary
                if lc["is_binary"] and rc["is_binary"]:
                    n1 = int(np.sum(prod[m_both] == 1.0))
                    if n1 < sparse_min_cell:
                        msg = f"Guard #2 dropped sparse binary interaction '{lc['name']} × {rc['name']}' (N={n1} < {sparse_min_cell})."
                        guard2_events.append({"term": f"{lc['name']}x{rc['name']}", "type": "sparse_interaction_dropped", "detail": msg})
                        footnotes.append(msg)
                        continue
                all_cols.append({
                    "name": f"{lc['name']} × {rc['name']}",
                    "ref": f"{lc['ref']}*{rc['ref']}",
                    "label": f"{lc['label']} × {rc['label']}",
                    "values": prod,
                    "kind": "interaction",
                    "is_binary": bool(lc["is_binary"] and rc["is_binary"]),
                    "meta": {"left": lc["name"], "right": rc["name"]},
                })

    # Compute final valid mask across y and all active X columns
    valid_mask = pre_mask.copy()
    for c in all_cols:
        valid_mask &= np.isfinite(c["values"])

    # Drop any column that became constant on the final valid mask
    final_cols: List[Dict[str, Any]] = []
    for c in all_cols:
        vals_v = c["values"][valid_mask]
        if len(vals_v) > 0 and np.ptp(vals_v) > 1e-12:
            final_cols.append(c)
        else:
            guard2_events.append({"term": c["name"], "type": "zero_variance", "detail": f"Term '{c['name']}' has zero variance on valid sample and was omitted."})

    idx_valid = np.where(valid_mask)[0]
    y_vec = y_info["y"][valid_mask]
    n_obs = len(y_vec)
    X_cols = [np.ones(n_obs, dtype=np.float64)] + [c["values"][valid_mask] for c in final_cols]
    X_mat = np.column_stack(X_cols) if n_obs > 0 else np.empty((0, len(X_cols)), dtype=np.float64)

    col_meta = [
        {"name": "const", "ref": "const", "label": "Intercept", "kind": "intercept", "is_binary": False, "meta": {}}
    ] + [
        {k: v for k, v in c.items() if k != "values"} for c in final_cols
    ]

    jac_vec = y_info.get("jacobian_log")
    jac_sum = float(np.sum(jac_vec[valid_mask])) if jac_vec is not None else 0.0

    return {
        "model_id": model_spec.get("id", "spec"),
        "model_name": model_spec.get("name", "Specification"),
        "X": X_mat,
        "y": y_vec,
        "y_raw": y_info["y_raw"][valid_mask],
        "valid_mask": valid_mask,
        "valid_indices": idx_valid,
        "n_obs": n_obs,
        "k_params": X_mat.shape[1],
        "col_meta": col_meta,
        "guard2_events": guard2_events,
        "footnotes": footnotes,
        "is_log_scale": y_info["is_log_scale"],
        "y_label": y_info["label"],
        "y_transform": y_info.get("transform", "raw"),
        "jacobian_log_sum": jac_sum,
    }


def compute_common_sample_mask(
    columns: Dict[str, List[Any]],
    dist_dict: Dict[str, Dict[str, np.ndarray]],
    active_specs: List[Dict[str, Any]],
    y_info: Dict[str, Any],
    base_valid_mask: np.ndarray,
    sparse_min_cell: int = 5,
) -> np.ndarray:
    """Compute common valid observation mask across all active ModelSpecs (N_common lock)."""
    common = base_valid_mask & y_info["valid_mask"]
    for spec in active_specs:
        d = build_model_design(columns, dist_dict, spec, y_info, base_valid_mask, sparse_min_cell)
        common &= d["valid_mask"]
    return common


def generate_m1_m4_specs(
    covariates: List[Dict[str, Any]],
    spatial_targets: List[Dict[str, Any]],
    default_rings: Optional[List[float]] = None,
    default_cutoff_m: float = 300.0,
) -> List[Dict[str, Any]]:
    """
    1-Click M1-M4 Baseline Series Generator from user-selected covariates & spatial targets:
    - M1: Physical/Control Baseline
    - M2: Discrete Distance Rings
    - M3: Continuous Log-Distance
    - M4: Andersson-Day Censored Proximity
    """
    rings_bp = default_rings or [150.0, 300.0, 500.0]
    cov_copy = [dict(c) for c in covariates]

    m1 = {
        "id": "m1_base",
        "name": "M1: Physical Baseline",
        "active": True,
        "covariates": cov_copy,
        "spatial_terms": [],
        "interactions": [],
    }

    m2_terms = []
    m3_terms = []
    m4_terms = []

    for tgt in spatial_targets:
        tid = tgt.get("id", "")
        d_mode = tgt.get("distance_modes", "euclidean")
        metric = "net" if d_mode == "network_dijkstra" else "eucl"
        m2_terms.append({
            "target_id": tid,
            "metric": metric,
            "transform": "rings",
            "breakpoints_m": list(rings_bp),
        })
        m3_terms.append({
            "target_id": tid,
            "metric": metric,
            "transform": "ln_d",
            "floor_m": 1.0,
        })
        m4_terms.append({
            "target_id": tid,
            "metric": metric,
            "transform": "andersson_day",
            "floor_m": 1.0,
            "cutoff_mode": "fixed",
            "cutoff_m": float(default_cutoff_m),
            "cutoff_range_m": [100.0, 600.0, 25.0],
        })
        if d_mode == "both":
            for tlist in (m2_terms, m3_terms, m4_terms):
                tlist.append({
                    "target_id": tid,
                    "metric": "circuity_ratio",
                    "transform": "circuity_ratio",
                })

    return [
        m1,
        {
            "id": "m2_rings",
            "name": "M2: Discrete Distance Rings",
            "active": True,
            "covariates": [dict(c) for c in covariates],
            "spatial_terms": m2_terms,
            "interactions": [],
        },
        {
            "id": "m3_log",
            "name": "M3: Continuous Log-Distance",
            "active": True,
            "covariates": [dict(c) for c in covariates],
            "spatial_terms": m3_terms,
            "interactions": [],
        },
        {
            "id": "m4_ad",
            "name": "M4: Andersson–Day Censored",
            "active": True,
            "covariates": [dict(c) for c in covariates],
            "spatial_terms": m4_terms,
            "interactions": [],
        },
    ]
