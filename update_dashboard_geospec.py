#!/usr/bin/env python3
"""
Deterministic updater for the TDK/GeoSpec Interactive Dashboard (index.html)
and 1-file .geospec.json workspace bundles.

Run from Colab Notebooks/ (or anywhere):
  python update_dashboard_geospec.py
"""
from __future__ import annotations

import base64
import copy
import csv
import datetime
import hashlib
import json
import math
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_DIR = Path(__file__).resolve().parent
PLATFORM_DIR = REPO_DIR / "platform"
GEOSPEC_INPUT_DIR = REPO_DIR / "geospec_input"
RECIPES_DIR = REPO_DIR / "recipes"
SRC_DATA_DIR = GEOSPEC_INPUT_DIR if (GEOSPEC_INPUT_DIR / "kobanya_dataset.csv").exists() else RECIPES_DIR
INDEX_HTML = REPO_DIR / "index.html"

sys.path.insert(0, str(PLATFORM_DIR))
from py_engine.engine import GeoSpecSession  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv_cols(csv_path: Path) -> tuple[list[str], dict[str, list[str]], int]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        cols: dict[str, list[str]] = {h: [] for h in headers}
        n_rows = 0
        for row in reader:
            n_rows += 1
            for h in headers:
                cols[h].append(row[h])
    return headers, cols, n_rows


def decode_plotly_array(val: Any) -> list[float]:
    if isinstance(val, list):
        return [float(x) for x in val]
    if isinstance(val, dict) and "bdata" in val:
        raw = base64.b64decode(val["bdata"])
        dt = val.get("dtype", "f8")
        fmt = {"f8": "d", "f4": "f", "i4": "i", "i8": "q"}[dt]
        sz = {"f8": 8, "f4": 4, "i4": 4, "i8": 8}[dt]
        return [float(struct.unpack("<" + fmt, raw[i : i + sz])[0]) for i in range(0, len(raw), sz)]
    raise ValueError(f"Unsupported Plotly array format: {type(val)}")


def sig_stars(p: float | None) -> str:
    if p is None or not math.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def hp_pct(beta: float) -> float:
    return (math.exp(beta) - 1.0) * 100.0


def fmt_pct_label(pct: float, p: float | None) -> str:
    st = sig_stars(p)
    suffix = st if st else " (NS)"
    return f"<b>{pct:+.1f}%{suffix}</b>"


def generate_workspace_bundles() -> None:
    """Populate git-tracked recipes/ folder and write 1-file .geospec.json bundles matching parseWorkspaceBundle."""
    RECIPES_DIR.mkdir(parents=True, exist_ok=True)
    public_files = [
        "kobanya_dataset.csv",
        "leopoldstadt_dataset.csv",
        "kobanya_recipe.json",
        "leopoldstadt_recipe.json",
        "kobanya_paper_replication_recipe.json",
        "leopoldstadt_paper_replication_recipe.json",
    ]
    if SRC_DATA_DIR != RECIPES_DIR:
        for fname in public_files:
            src = SRC_DATA_DIR / fname
            if src.exists():
                shutil.copy2(src, RECIPES_DIR / fname)

    bundles = [
        ("kobanya", "kobanya_dataset.csv", "kobanya_recipe.json", "kobanya_best_practice.geospec.json"),
        ("leopoldstadt", "leopoldstadt_dataset.csv", "leopoldstadt_recipe.json", "leopoldstadt_best_practice.geospec.json"),
    ]
    for _city, csv_name, rec_name, out_name in bundles:
        headers, cols, n_rows = load_csv_cols(RECIPES_DIR / csv_name)
        recipe = json.loads((RECIPES_DIR / rec_name).read_text(encoding="utf-8"))
        bundle = {
            "kind": "geospec_workspace",
            "schema_version": "1.0.0",
            "seal_version": recipe.get("recipe_meta", {}).get("seal_version", "v1.0-sealed"),
            "saved_utc": recipe.get("recipe_meta", {}).get("sealed_utc", "2026-11-27T16:10:00Z"),
            "recipe": recipe,
            "dataset": {
                "fileName": csv_name,
                "headers": headers,
                "columns": cols,
                "nRows": n_rows,
            },
            "osm_bundle": None,
            "ui_state": {
                "activeSpecId": recipe.get("spatial_engine", {}).get("active_model_id", "m4_ad"),
                "covType": recipe.get("default_cov_type", "cluster"),
                "showPct": True,
            },
        }
        (RECIPES_DIR / out_name).write_text(
            json.dumps(bundle, ensure_ascii=False), encoding="utf-8"
        )
        stale_dup = GEOSPEC_INPUT_DIR / out_name
        if stale_dup.exists():
            stale_dup.unlink()
        print(f"Wrote recipes/{out_name} (nRows={n_rows})")


def run_geospec_city(city: str) -> dict[str, Any]:
    headers, cols, n_rows = load_csv_cols(SRC_DATA_DIR / f"{city}_dataset.csv")
    recipe = json.loads((SRC_DATA_DIR / f"{city}_recipe.json").read_text(encoding="utf-8"))
    sess = GeoSpecSession()
    sess.load_dataset(cols, "wgs84_lat", "wgs84_long")
    vm = sess.valid_coord_mask

    def fvec(c: str) -> np.ndarray:
        return np.array([float(v) for ok, v in zip(vm, cols[c]) if ok and str(v).strip() != ""], dtype=float)

    p = fvec("fajlagos_ar")
    area = fvec("alapterulet_nm")
    rooms = fvec("szobak_szama")
    lift = fvec("d_lift")
    new15 = fvec("d_epult_2015_utan")
    feluj = fvec("d_felujitott")
    bad = fvec("d_felujitando")
    szut = fvec("d_szuteren") if "d_szuteren" in cols else np.zeros_like(p)
    hm = fvec("dist_vasutallomas_hm")

    s1 = sess.run_stage_1_distances(recipe)
    s2 = sess.run_stage_2_descriptives_and_composition(recipe)
    s3 = sess.run_stage_3_ols_workbench(recipe, cov_type=recipe["default_cov_type"])
    s4_m4 = sess.run_stage_4_spatial_econometrics(recipe)
    s5 = sess.run_stage_5_gwr(recipe)
    s6 = sess.run_stage_6_ml_shap(recipe)

    # Exact raw <=500m vs >500m stats computed strictly on the analytic common sample (s3 M4 valid_indices)
    m4_vidx = set(s3["models"][-1]["valid_indices"])
    raw_in_500 = [float(cols["fajlagos_ar"][i]) for i in m4_vidx if float(cols["dist_vasutallomas_hm"][i]) <= 500.0]
    raw_out_500 = [float(cols["fajlagos_ar"][i]) for i in m4_vidx if float(cols["dist_vasutallomas_hm"][i]) > 500.0]
    m_in = float(np.mean(raw_in_500)) if raw_in_500 else 0.0
    m_out = float(np.mean(raw_out_500)) if raw_out_500 else 0.0
    raw_gap_pct = ((m_in - m_out) / m_out * 100.0) if m_out > 0 else 0.0

    s4_by_model: dict[str, Any] = {}
    for mid in ["m1_base", "m2_rings", "m3_log", "m4_ad"]:
        r_m = copy.deepcopy(recipe)
        r_m["spatial_engine"]["active_model_id"] = mid
        s4_by_model[mid] = sess.run_stage_4_spatial_econometrics(r_m)

    # Also compute HC3 standard errors for M4 SE stability comparison
    s3_hc3 = sess.run_stage_3_ols_workbench(recipe, cov_type="HC3")

    return {
        "city": city,
        "recipe": recipe,
        "n_csv": n_rows,
        "n_analytic": int(vm.sum()),
        "n_common": len(m4_vidx),
        "desc": {
            "mean_price": float(p.mean()),
            "med_price": float(np.median(p)),
            "med_area": float(np.median(area)),
            "mean_rooms": float(rooms.mean()),
            "lift_pct": float(lift.mean() * 100.0),
            "new15_pct": float(new15.mean() * 100.0),
            "feluj_pct": float(feluj.mean() * 100.0),
            "n_new15_common": int(sum(int(float(cols["d_epult_2015_utan"][i]) >= 0.5) for i in m4_vidx)),
            "n_feluj_common": int(sum(int(float(cols["d_felujitott"][i]) >= 0.5) for i in m4_vidx)),
            "n_bad_common": int(sum(int(float(cols["d_felujitando"][i]) >= 0.5) for i in m4_vidx)),
            "n_szut_geocoded": int(np.sum(szut >= 0.5)),
            "walk_mean": float(hm.mean()),
            "walk_med": float(np.median(hm)),
            "raw_le500_n": len(raw_in_500),
            "raw_le500_mean": round(m_in, 2),
            "raw_gt500_n": len(raw_out_500),
            "raw_gt500_mean": round(m_out, 2),
            "raw_le500_gap_pct": round(raw_gap_pct, 2),
        },
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s3_hc3": s3_hc3,
        "s4_m4": s4_m4,
        "s4_by_model": s4_by_model,
        "s5": s5,
        "s6": s6,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summarize_city_truth(res: dict[str, Any]) -> dict[str, Any]:
    s3_models = {}
    for m in res["s3"]["models"]:
        params_map = {}
        for pr in m["params"]:
            beta = float(pr["beta"])
            params_map[pr["name"]] = {
                "beta": round(beta, 6),
                "hp_pct": round(hp_pct(beta), 2),
                "se": round(float(pr["se"]), 6),
                "p": round(float(pr["p"]), 6),
            }
        s3_models[m["model_id"]] = {
            "n_obs": m["n_obs"],
            "k_params": m["k_params"],
            "r2": round(float(m["r2"]), 6),
            "adj_r2": round(float(m["adj_r2"]), 6),
            "aic": round(float(m["aic"]), 4),
            "cov_type_used": m["cov_type_used"],
            "params": params_map,
        }

    s4 = res["s4_m4"]
    comp = {c["model"].split()[0]: c for c in s4["model_comparison"]}
    sar_m4, sem_m4 = comp["SAR"], comp["SEM"]
    rho_val = float(sar_m4["spatial_param"])
    imp_sar = {x["variable"]: x for x in s4.get("impacts_sar", [])}
    first_imp = next(iter(imp_sar.values()), {})
    return {
        "descriptives": {
            "n_total": res["n_csv"],
            "n_geocoded": res["n_analytic"],
            "n_common": res["n_common"],
            **{k: round(v, 2) if isinstance(v, float) else v for k, v in res["desc"].items()},
        },
        "ols": s3_models,
        "spatial_stage4": {
            "primary_w_id": s4["primary_w_id"],
            "se_mode": s4["se_mode"],
            "moran_y": s4["moran_y"],
            "moran_resid": s4["moran_resid"],
            "anselin": s4["anselin"],
            "sar_rho": round(rho_val, 4),
            "sar_rho_se": sar_m4["spatial_se"],
            "sar_rho_p": sar_m4["spatial_p"],
            "sar_multiplier": round(1.0 / (1.0 - rho_val), 3) if abs(1.0 - rho_val) > 1e-6 else None,
            "sar_spillover_share_pct": round(float(first_imp.get("spillover_share_pct", 0.0)), 1),
            "sar_resid_moran_i": sar_m4["resid_moran_i"],
            "sem_lambda": round(float(sem_m4["spatial_param"]), 4),
            "sem_lambda_p": sem_m4["spatial_p"],
            "sem_resid_moran_i": sem_m4["resid_moran_i"],
            "sem_resid_moran_p": sem_m4["resid_moran_p"],
            "weights_robustness": s4["weights_robustness"],
        },
        "gwr_stage5": {
            "focal_model_id": res["s5"]["model_id"],
            "optimal_bw": res["s5"]["optimal_bw"],
            "global_ols_r2": res["s5"]["global_ols_r2"],
            "global_ols_adj_r2": res["s5"]["global_ols_adj_r2"],
            "gwr_r2": res["s5"]["gwr_r2"],
            "gwr_adj_r2": res["s5"]["gwr_adj_r2"],
            "param_summaries": res["s5"]["param_summaries"],
        },
        "ml_shap_stage6": {
            "engine": res["s6"]["engine"],
            "focal_model_id": res["s6"]["focal_model_id"],
            "n_obs": res["s6"]["n_obs"],
            "n_clusters": res["s6"]["n_clusters"],
            "cv_folds": res["s6"]["cv_folds"],
            "random_seed": res["s6"]["random_seed"],
            "local_accuracy_max_err": res["s6"]["local_accuracy_max_err"],
            "benchmark": res["s6"]["benchmark"],
            "grouped_importance": res["s6"]["grouped_importance"],
            "feature_importance": res["s6"]["feature_importance"],
        },
    }


def write_canonical_truth(kb: dict[str, Any], vi: dict[str, Any]) -> dict[str, Any]:
    """Seal and write recipes/canonical_truth.json with full SHA-256 lock and Stage 1-6 results."""
    tracked_files = [
        "kobanya_dataset.csv",
        "leopoldstadt_dataset.csv",
        "kobanya_recipe.json",
        "leopoldstadt_recipe.json",
        "kobanya_best_practice.geospec.json",
        "leopoldstadt_best_practice.geospec.json",
    ]
    sha_map = {fn: _file_sha256(RECIPES_DIR / fn) for fn in tracked_files}
    payload = {
        "schema_version": "1.0.0",
        "seal_version": "v1.0-sealed",
        "sealed_utc": kb["recipe"].get("recipe_meta", {}).get("sealed_utc", "2026-11-27T16:10:00Z"),
        "sha256": sha_map,
        "cities": {
            "kobanya": _summarize_city_truth(kb),
            "leopoldstadt": _summarize_city_truth(vi),
        },
    }
    out_path = RECIPES_DIR / "canonical_truth.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote recipes/canonical_truth.json (seal_version={payload['seal_version']})")
    return payload


# ---------------------------------------------------------------------------
# HTML & Plotly Patching Helpers
# ---------------------------------------------------------------------------

def replace_exact(html: str, old: str, new: str, label: str, *, optional: bool = False) -> str:
    if old not in html:
        if optional or new in html:
            return html
        raise RuntimeError(f"Expected anchor not found for [{label}]: {old[:120]!r}")
    return html.replace(old, new, 1)


def replace_section_between(
    html: str,
    start_marker: str,
    end_marker: str,
    new_block: str,
    label: str,
    *,
    after: str = "",
    max_lines: int = 220,
) -> str:
    search_from = 0
    if after:
        search_from = html.find(after)
        if search_from < 0:
            raise RuntimeError(f"Anchor 'after' not found for [{label}]: {after!r}")
    i = html.find(start_marker, search_from)
    if i < 0:
        raise RuntimeError(f"Start marker not found for [{label}]: {start_marker!r}")
    j = html.find(end_marker, i)
    if j < 0:
        raise RuntimeError(f"End marker not found for [{label}]: {end_marker!r}")
    old_span = html[i:j]
    n_lines = old_span.count("\n")
    if n_lines > max_lines:
        raise RuntimeError(f"Section [{label}] unexpectedly large ({n_lines} lines > max {max_lines}); aborting!")
    if "Plotly.newPlot(" in old_span:
        raise RuntimeError(f"Section [{label}] accidentally spans over a Plotly.newPlot chart!")
    return html[:i] + new_block + html[j:]


def patch_plotly_div(html: str, div_id: str, mutator) -> str:
    pattern = re.compile(
        r'(Plotly\.newPlot\(\s*"' + re.escape(div_id) + r'",\s*)(\[.*?\])(,\s*)(\{.*?\})(,\s*\{"responsive":\s*true\})',
        re.S,
    )
    m = pattern.search(html)
    if not m:
        raise RuntimeError(f"Plotly.newPlot call not found for div id={div_id}")
    traces = json.loads(m.group(2))
    layout = json.loads(m.group(4))
    mutator(traces, layout)
    new_call = (
        m.group(1)
        + json.dumps(traces, ensure_ascii=False, separators=(",", ":"))
        + m.group(3)
        + json.dumps(layout, ensure_ascii=False, separators=(",", ":"))
        + m.group(5)
    )
    return html[: m.start()] + new_call + html[m.end() :]


# ---------------------------------------------------------------------------
# Main Update Logic
# ---------------------------------------------------------------------------

def main() -> None:
    generate_workspace_bundles()
    kb = run_geospec_city("kobanya")
    vi = run_geospec_city("leopoldstadt")
    truth = write_canonical_truth(kb, vi)

    kb_m = {m["model_id"]: m for m in kb["s3"]["models"]}
    vi_m = {m["model_id"]: m for m in vi["s3"]["models"]}
    kb_m4_p = {r["name"]: r for r in kb_m["m4_ad"]["params"]}
    vi_m4_p = {r["name"]: r for r in vi_m["m4_ad"]["params"]}
    kb_m2_p = {r["name"]: r for r in kb_m["m2_rings"]["params"]}
    vi_m2_p = {r["name"]: r for r in vi_m["m2_rings"]["params"]}

    kb_s6 = kb["s6"]
    vi_s6 = vi["s6"]
    kb_grp = kb_s6["grouped_importance"]
    vi_grp = vi_s6["grouped_importance"]
    vi_feat_map = {r["feature"]: r for r in vi_s6["feature_importance"]}
    vi_outdoor_pct = (
        vi_feat_map.get("kulteri_alapterulet_nm", {}).get("rel_pct", 0.0)
        + vi_feat_map.get("d_kulteri_kapcsolat", {}).get("rel_pct", 0.0)
    )
    vi_p_300_500_hu = f"{vi_m2_p['ring_vagany_300_500']['p']:.3f}".replace(".", ",")
    vi_p_300_500_en = f"{vi_m2_p['ring_vagany_300_500']['p']:.3f}"
    kb_m4_vag_pct = hp_pct(kb_m4_p["AD(vagany,eucl,c=300m)"]["beta"])

    def hu1(val: float, signed: bool = False) -> str:
        s = f"{val:+.1f}" if signed else f"{val:.1f}"
        return s.replace(".", ",")

    kb_phys_hu = hu1(kb_grp["physical_pct"])
    kb_trk_hu = hu1(kb_grp["rail_track_pct"])
    kb_rail_hu = hu1(kb_grp["rail_pct"])
    kb_sta_hu = hu1(kb_grp["rail_station_pct"])
    vi_out_hu = hu1(vi_outdoor_pct)
    vi_trk_hu = hu1(vi_grp["rail_track_pct"])
    vi_rail_hu = hu1(vi_grp["rail_pct"])

    html = INDEX_HTML.read_text(encoding="utf-8")

    # 1. HEADER ACTION BUTTONS + GEOSPEC RECIPE DOWNLOAD DROPDOWN
    new_header_actions = """      <div class="flex flex-wrap items-center gap-2 min-w-0 max-w-full">
        <a href="./platform/index.html" class="inline-flex items-center gap-1.5 text-xs bg-teal-700 hover:bg-teal-600 text-white px-2.5 sm:px-3 py-1.5 rounded-lg font-medium transition shadow-sm">
          <span>GeoSpec Platform</span>
          <span class="text-[10px] text-teal-100 font-mono">WASM ↗</span>
        </a>
        <div class="relative inline-block text-left" id="recipe-dropdown-wrap">
          <button type="button" id="recipe-menu-btn" aria-haspopup="true" aria-expanded="false" aria-controls="recipe-dropdown-menu" class="inline-flex items-center gap-1.5 text-xs bg-orange-600 hover:bg-orange-500 text-white px-2.5 sm:px-3 py-1.5 rounded-lg font-semibold transition shadow-sm cursor-pointer">
            <span>Recept letöltése</span>
            <span class="text-[10px] text-orange-100 font-mono">GeoSpec ▾</span>
          </button>
          <div id="recipe-dropdown-menu" hidden role="menu" class="hidden fixed sm:absolute left-4 right-4 sm:left-auto sm:right-0 mt-2 sm:w-80 rounded-xl bg-white shadow-xl border border-slate-200 py-2 z-[60] text-xs text-slate-700">
            <div class="px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-400">1 kattintásos reprodukció (GeoSpec)</div>
            <a role="menuitem" href="./platform/index.html?study=kobanya" class="flex items-center justify-between px-3 py-2 hover:bg-teal-50 text-teal-800 font-semibold">
              <span>▶ Kőbánya megnyitása GeoSpec-ben</span><span class="font-mono text-[10px]">N=269 ↗</span>
            </a>
            <a role="menuitem" href="./platform/index.html?study=leopoldstadt" class="flex items-center justify-between px-3 py-2 hover:bg-teal-50 text-teal-800 font-semibold">
              <span>▶ Bécs megnyitása GeoSpec-ben</span><span class="font-mono text-[10px]">N=378 ↗</span>
            </a>
            <div class="my-1 border-t border-slate-100"></div>
            <div class="px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-400">Best-Practice receptek (_recipe.json)</div>
            <a role="menuitem" href="./recipes/kobanya_recipe.json" download="kobanya_recipe.json" class="flex items-center justify-between px-3 py-1.5 hover:bg-slate-50">
              <span>↓ Kőbánya Best-Practice recept</span><span class="font-mono text-[10px] text-slate-400">.json</span>
            </a>
            <a role="menuitem" href="./recipes/leopoldstadt_recipe.json" download="leopoldstadt_recipe.json" class="flex items-center justify-between px-3 py-1.5 hover:bg-slate-50">
              <span>↓ Bécs Leopoldstadt Best-Practice recept</span><span class="font-mono text-[10px] text-slate-400">.json</span>
            </a>
            <div class="my-1 border-t border-slate-100"></div>
            <div class="px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-400">Teljes 1-fájlos projektek (.geospec.json)</div>
            <a role="menuitem" href="./recipes/kobanya_best_practice.geospec.json" download="kobanya_best_practice.geospec.json" class="flex items-center justify-between px-3 py-1.5 hover:bg-slate-50">
              <span>↓ Kőbánya projekt (adat + recept)</span><span class="font-mono text-[10px] text-slate-400">.geospec.json</span>
            </a>
            <a role="menuitem" href="./recipes/leopoldstadt_best_practice.geospec.json" download="leopoldstadt_best_practice.geospec.json" class="flex items-center justify-between px-3 py-1.5 hover:bg-slate-50">
              <span>↓ Bécs projekt (adat + recept)</span><span class="font-mono text-[10px] text-slate-400">.geospec.json</span>
            </a>
            <div class="my-1 border-t border-slate-100"></div>
            <div class="px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-400">Adattáblák (.csv) &amp; Nov. 1 replikáció</div>
            <a role="menuitem" href="./recipes/kobanya_dataset.csv" download="kobanya_dataset.csv" class="flex items-center justify-between px-3 py-1.5 hover:bg-slate-50">
              <span>↓ Kőbánya nyers adatbázis</span><span class="font-mono text-[10px] text-slate-400">.csv</span>
            </a>
            <a role="menuitem" href="./recipes/leopoldstadt_dataset.csv" download="leopoldstadt_dataset.csv" class="flex items-center justify-between px-3 py-1.5 hover:bg-slate-50">
              <span>↓ Bécs nyers adatbázis</span><span class="font-mono text-[10px] text-slate-400">.csv</span>
            </a>
            <a role="menuitem" href="./recipes/kobanya_paper_replication_recipe.json" download="kobanya_paper_replication_recipe.json" class="flex items-center justify-between px-3 py-1.5 hover:bg-slate-50 text-slate-500">
              <span>↓ Kőbánya nov. 1 dolgozat replikáció</span><span class="font-mono text-[10px]">.json</span>
            </a>
            <a role="menuitem" href="./recipes/leopoldstadt_paper_replication_recipe.json" download="leopoldstadt_paper_replication_recipe.json" class="flex items-center justify-between px-3 py-1.5 hover:bg-slate-50 text-slate-500">
              <span>↓ Bécs nov. 1 dolgozat replikáció</span><span class="font-mono text-[10px]">.json</span>
            </a>
            <div class="my-1 border-t border-slate-100"></div>
            <div class="px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-emerald-700">Lepecsételt Igazságforrás (v1.0-sealed)</div>
            <a role="menuitem" href="./recipes/canonical_truth.json" download="canonical_truth.json" class="flex items-center justify-between px-3 py-1.5 hover:bg-emerald-50 text-emerald-800 font-semibold">
              <span>🔒 Kanonikus Eredmények + SHA-256</span><span class="font-mono text-[10px]">.json</span>
            </a>
          </div>
        </div>
        <a href="https://github.com/oliwerbig/kobanya-ingatlan-tdk/blob/main/Sz%C3%A1m%C3%ADt%C3%A1sok.ipynb" target="_blank" rel="noopener noreferrer" class="inline-flex items-center gap-1.5 text-xs bg-slate-900 hover:bg-slate-800 text-white px-2.5 sm:px-3 py-1.5 rounded-lg font-medium transition shadow-sm">
          <span>Számítások</span>
          <span class="text-[10px] text-slate-300 font-mono">.ipynb ↗</span>
        </a>
        <a href="https://github.com/oliwerbig/kobanya-ingatlan-tdk/blob/main/Vizualizacio.ipynb" target="_blank" rel="noopener noreferrer" class="inline-flex items-center gap-1.5 text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-2.5 sm:px-3 py-1.5 rounded-lg font-medium transition border border-slate-200">
          <span>Vizualizáció</span>
          <span class="text-[10px] text-slate-500 font-mono">.ipynb ↗</span>
        </a>
      </div>"""

    hdr_start = (
        '<div class="flex items-center gap-2 shrink-0">'
        if '<div class="flex items-center gap-2 shrink-0">' in html
        else (
            '<div class="flex flex-wrap items-center gap-2 shrink-0">'
            if '<div class="flex flex-wrap items-center gap-2 shrink-0">' in html
            else '<div class="flex flex-wrap items-center gap-2 min-w-0 max-w-full">'
        )
    )
    html = replace_section_between(
        html,
        hdr_start,
        "\n    </div>\n\n    <!-- NAVIGATION TABS",
        new_header_actions.lstrip(),
        "header_actions",
        after='id="top-header"' if 'id="top-header"' in html else "Kötöttpályás Közlekedési Externáliák",
        max_lines=80,
    )

    # 2. HERO THESIS STRIP + 6 KPI CARDS + REPRODUCIBILITY BAR
    new_hero = f"""<!-- TDK TÉZIS-ÖSSZEFOGLALÓ & FŐ MUTATÓK (FRISSÍTETT GEOSPEC BEST-PRACTICE v1.0-sealed) -->
  <section class="bg-slate-900 text-slate-100 border-b border-slate-800">
    <div class="max-w-7xl mx-auto px-4 py-4 space-y-4">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
        <div class="bg-slate-800/70 border border-slate-700/80 rounded-lg px-3.5 py-2.5">
          <div class="text-[10px] font-bold uppercase tracking-wider text-sky-400 mb-0.5">I. Állomásvölgy vs. vágányzaj</div>
          <p class="text-slate-300 leading-relaxed">Kőbányán a nyers ≤500 m állomás-völgy -21,1% (1,11M vs. 1,40M Ft/m²), de kontrollálva a vágányhatás eltűnik ({hu1(kb_m4_vag_pct, True)}% NS; TreeSHAP {kb_phys_hu}% fizikai vs. {kb_trk_hu}% vágány). Bécsben a 0–300 m zajdiszkont -16,9% – -17,7%, 300 m felett 0% (p = {vi_p_300_500_hu}).</p>
        </div>
        <div class="bg-slate-800/70 border border-slate-700/80 rounded-lg px-3.5 py-2.5">
          <div class="text-[10px] font-bold uppercase tracking-wider text-emerald-400 mb-0.5">II. Minőségi szakadék &amp; tudatos várostervezés</div>
          <p class="text-slate-300 leading-relaxed">Bécs Leopoldstadt korszerű/új állománya (92,6%) évtizedes tudatos városfejlesztés eredménye. Kőbányán a felújítás önmagában csak +6,7% (NS), a 2015 utáni új építés viszont +23,3% – +28,0% prémiumot ér el (kerülő medián: +141 m, max: +1 065 m).</p>
        </div>
        <div class="bg-slate-800/70 border border-slate-700/80 rounded-lg px-3.5 py-2.5">
          <div class="text-[10px] font-bold uppercase tracking-wider text-amber-400 mb-0.5">III. Mázsa tér &amp; önfinanszírozó LVC</div>
          <p class="text-slate-300 leading-relaxed">A New Wave Residence 6,8 ha-os modellje (880 lakás + 8 000 m² iroda; 86,5 Mrd Ft költség, 97,4 Mrd Ft árbevétel, 15,8% ROI) konzervatív +10% × 20% normatív értékmegragadással 1,84 Mrd Ft önkormányzati LVC-alapot képez.</p>
        </div>
      </div>

      <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 pt-2 border-t border-slate-800 text-center">
        <div class="bg-slate-800/40 rounded-lg py-2 px-2">
          <div class="text-[11px] text-slate-400">Geokódolt Minta</div>
          <div class="text-sm sm:text-base font-bold text-white">269 <span class="text-xs font-normal text-slate-400">Bp</span> / 378 <span class="text-xs font-normal text-slate-400">Bécs</span></div>
        </div>
        <div class="bg-slate-800/40 rounded-lg py-2 px-2">
          <div class="text-[11px] text-slate-400">Bécs Vágányzaj (0–300 m)</div>
          <div class="text-sm sm:text-base font-bold text-rose-400">-16,9% – -17,7% <span class="text-xs font-normal text-slate-400">(300m+: 0%)</span></div>
        </div>
        <div class="bg-slate-800/40 rounded-lg py-2 px-2">
          <div class="text-[11px] text-slate-400">Kőbánya Nyers vs. M4</div>
          <div class="text-sm sm:text-base font-bold text-emerald-400">-21,1% <span class="text-xs font-normal text-slate-400">/ {kb_m4_vag_pct:+.1f}% (NS)</span></div>
        </div>
        <div class="bg-slate-800/40 rounded-lg py-2 px-2">
          <div class="text-[11px] text-slate-400">2015+ Újépítés (Bp)</div>
          <div class="text-sm sm:text-base font-bold text-sky-400">+23,3% – +28,0%***</div>
        </div>
        <div class="bg-slate-800/40 rounded-lg py-2 px-2">
          <div class="text-[11px] text-slate-400">LVC Alap (New Wave)</div>
          <div class="text-sm sm:text-base font-bold text-amber-400">1,84 Mrd Ft <span class="text-xs font-normal text-slate-400">(15,8% ROI)</span></div>
        </div>
        <div class="bg-slate-800/40 rounded-lg py-2 px-2">
          <div class="text-[11px] text-slate-400">M4 R² (v1.0-sealed)</div>
          <div class="text-sm sm:text-base font-bold text-emerald-400">{vi_m['m4_ad']['r2']:.4f} <span class="text-xs font-normal text-slate-400">/ {kb_m['m4_ad']['r2']:.4f}</span></div>
        </div>
      </div>

      <div class="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-800/90 text-[11px]">
        <div class="flex flex-wrap items-center gap-1.5">
          <span class="text-slate-400 font-semibold uppercase tracking-wider mr-1">GeoSpec v1.0-sealed:</span>
          <a href="./platform/index.html?study=kobanya" class="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-teal-700 hover:bg-teal-600 text-white font-semibold transition">▶ Kőbánya GeoSpec-ben ↗</a>
          <a href="./platform/index.html?study=leopoldstadt" class="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-teal-700 hover:bg-teal-600 text-white font-semibold transition">▶ Bécs GeoSpec-ben ↗</a>
          <a href="./recipes/kobanya_recipe.json" download="kobanya_recipe.json" class="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-mono transition">↓ kobanya_recipe.json</a>
          <a href="./recipes/leopoldstadt_recipe.json" download="leopoldstadt_recipe.json" class="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-mono transition">↓ leopoldstadt_recipe.json</a>
          <a href="./recipes/canonical_truth.json" download="canonical_truth.json" class="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-emerald-900/60 hover:bg-emerald-800 text-emerald-100 border border-emerald-700 font-mono transition">🔒 canonical_truth.json</a>
        </div>
        <span class="text-slate-400">Kínálati árak (ingatlan.com / willhaben.at) · M1–M4 + épületszintű klaszterezés (Guard #2: n&lt;5 kiszűrve) · Silent Rail 2. fázis (BME × UTCN)</span>
      </div>
    </div>
  </section>"""

    html = replace_section_between(
        html,
        "<!-- TDK TÉZIS-ÖSSZEFOGLALÓ",
        '\n\n  <main class="max-w-7xl',
        new_hero,
        "hero_strip",
    )

    # 3. TAB 1: INTRO PARAGRAPH, 6 KPI CARDS, FIG 3 & PROSE
    html = replace_exact(
        html,
        "A kutatás empirikus alapját Budapest X. kerületének (Kőbánya, <strong>N=269</strong>) és Bécs II. kerületének (Leopoldstadt, <strong>N=378</strong>) szűrt, pontos geokódolt tranzakciós és kínálati lakáspiaci adatbázisa alkotja. Az alábbi modul bemutatja a két minta fizikai épületállományát, a lakhatási megfizethetőség strukturális különbségeit, a városszöveti és utcahálózati kanyargósságból fakadó gyalogos többletkerülőket (Network circuity / tortuosity), valamint bizonyítja a hedonikus modellezés alapfeltételeit: a logaritmikus normálitást és a multikollinearitás-mentességet (VIF &lt; 6,6).",
        "A kutatás empirikus alapját Budapest X. kerületének (Kőbánya, <strong>N = 269</strong>) és Bécs II. kerületének (Leopoldstadt, <strong>N = 378</strong>) házszámszinten geokódolt <strong>kínálati</strong> lakáshirdetési adatbázisa alkotja (<em>ingatlan.com</em>, <em>willhaben.at</em> — nem NAV/földhivatali tranzakciós árak). Bécs Leopoldstadt 92,6%-os korszerű/új állománya évtizedes tudatos önkormányzati várostervezés eredménye (ahogy Erő Zoltán is hangsúlyozta); a modul bemutatja a fizikai épületállományt, a gyalogos többletkerülőket (medián: +141 m, max: +1 065 m) és a hedonikus alapdiagnosztikát (VIF).",
        "tab1_intro",
        optional=True,
    )

    new_tab1_kpis = f"""<!-- 6 KÁRTYÁS PIACMORFOLÓGIAI KPI SZALAG -->
      <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
        <div class="bg-white p-3.5 rounded-xl border border-slate-200 shadow-sm">
          <div class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Analitikus Minta (N)</div>
          <div class="text-base font-extrabold text-slate-900 mt-0.5">269 <span class="text-xs font-normal text-slate-500">Bp</span> / 378 <span class="text-xs font-normal text-slate-500">Bécs</span></div>
          <div class="text-[10px] text-blue-600 font-medium mt-1">ingatlan.com / willhaben</div>
        </div>
        <div class="bg-white p-3.5 rounded-xl border border-slate-200 shadow-sm">
          <div class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Fajlagos Átlagár</div>
          <div class="text-base font-extrabold text-blue-600 mt-0.5">1,36 M Ft <span class="text-xs font-normal text-slate-500">/ 9 553 €</span></div>
          <div class="text-[10px] text-slate-500 mt-1">Kőbánya ≤500m völgy: -21,1%</div>
        </div>
        <div class="bg-white p-3.5 rounded-xl border border-slate-200 shadow-sm">
          <div class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Medián Lakásméret</div>
          <div class="text-base font-extrabold text-slate-900 mt-0.5">{kb['desc']['med_area']:.0f} m² <span class="text-xs font-normal text-slate-500">/ {vi['desc']['med_area']:.0f} m²</span></div>
          <div class="text-[10px] text-slate-500 mt-1">{kb['desc']['mean_rooms']:.2f} szoba / {vi['desc']['mean_rooms']:.2f} szoba</div>
        </div>
        <div class="bg-white p-3.5 rounded-xl border border-slate-200 shadow-sm">
          <div class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Lift &amp; Korszerű Állomány</div>
          <div class="text-base font-extrabold text-amber-600 mt-0.5">{kb['desc']['lift_pct']:.1f}% / {vi['desc']['lift_pct']:.1f}% <span class="text-xs font-normal text-slate-500">lift</span></div>
          <div class="text-[10px] text-slate-500 mt-1">Korszerű+új: 65,8% (Bp) / 92,6% (Bécs)</div>
        </div>
        <div class="bg-white p-3.5 rounded-xl border border-slate-200 shadow-sm">
          <div class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Állomás Gyalog</div>
          <div class="text-base font-extrabold text-indigo-600 mt-0.5">1 142 m <span class="text-xs font-normal text-slate-500">/ 718 m</span></div>
          <div class="text-[10px] text-slate-500 mt-1">15,2p (Bp) / 9,6p (Bécs)</div>
        </div>
        <div class="bg-white p-3.5 rounded-xl border border-slate-200 shadow-sm">
          <div class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Kerülőút (Δd) Medián</div>
          <div class="text-base font-extrabold text-red-600 mt-0.5">+141 m <span class="text-xs font-normal text-slate-500">(max +1 065 m)</span></div>
          <div class="text-[10px] text-slate-500 mt-1">Hálózati kanyargósság</div>
        </div>
      </div>"""

    html = replace_section_between(
        html,
        "<!-- 6 KÁRTYÁS PIACMORFOLÓGIAI KPI SZALAG -->",
        "\n\n      <!-- 2 OSZLOPOS GRID",
        new_tab1_kpis,
        "tab1_kpis",
    )

    def mut_fig3(traces, layout):
        traces[2]["name"] = "Kőbánya kerülőút (Medián: +141 m)"
        for ann in layout.get("annotations", []):
            if "143" in str(ann.get("text", "")):
                ann["text"] = "Kőbánya medián: +141 m"
                ann["x"] = 141.0
        for sh in layout.get("shapes", []):
            if abs(float(sh.get("x0", 0)) - 142.8) < 1.0:
                sh["x0"] = 141.0
                sh["x1"] = 141.0

    html = patch_plotly_div(html, "8e15b11e-863e-4f65-9ef1-9c3ca3f6f92b", mut_fig3)
    html = replace_exact(
        html,
        "medián: 143 m, max: 1 071 m",
        "medián: +141 m, max: +1 065 m",
        "tab1_detour_prose",
        optional=True,
    )

    # 4. TAB 2: HEDONIC HERO, KPI STRIP, FIG 7, 8, 9, 10 & TABLES
    html = replace_exact(
        html,
        "R² = 0.749 (Bécs) / 0.610 (Bp)",
        f"R² = {vi_m['m4_ad']['r2']:.4f} (Bécs) / {kb_m['m4_ad']['r2']:.4f} (Bp; replikáció: 0.6348)",
        "tab2_hero_r2",
        optional=True,
    )

    new_tab2_kpis = f"""<!-- TAB 2 HEDONIKUS KPI METRIC STRIP -->
      <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
          <div class="text-xs text-slate-500 font-medium">M4 Magyarázóerő (R²adj)</div>
          <div class="text-lg font-bold text-blue-700 mt-1">{vi_m['m4_ad']['adj_r2']*100:.1f}% <span class="text-xs font-normal text-slate-400">Bécs</span></div>
          <div class="text-[11px] text-slate-500 mt-0.5">Kőbánya: {kb_m['m4_ad']['adj_r2']*100:.1f}% (R²={kb_m['m4_ad']['r2']:.4f})</div>
        </div>
        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
          <div class="text-xs text-slate-500 font-medium">Új Építés Prémium (2015+)</div>
          <div class="text-lg font-bold text-emerald-600 mt-1">+{hp_pct(vi_m4_p['d_epult_2015_utan']['beta']):.1f}%*** <span class="text-xs font-normal text-slate-400">Bécs</span></div>
          <div class="text-[11px] text-slate-500 mt-0.5">Bp: +{hp_pct(kb_m4_p['d_epult_2015_utan']['beta']):.1f}% – +{hp_pct(kb_m2_p['d_epult_2015_utan']['beta']):.1f}%***</div>
        </div>
        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
          <div class="text-xs text-slate-500 font-medium">Földszint Diszkont</div>
          <div class="text-lg font-bold text-red-600 mt-1">{hp_pct(vi_m4_p['d_foldszint']['beta']):.1f}%*** <span class="text-xs font-normal text-slate-400">Bécs</span></div>
          <div class="text-[11px] text-slate-500 mt-0.5">Bp: {hp_pct(kb_m4_p['d_foldszint']['beta']):.1f}% (~0%, NS; n&lt;5 szűrve)</div>
        </div>
        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
          <div class="text-xs text-slate-500 font-medium">Erkély &amp; Terasz Prémium</div>
          <div class="text-lg font-bold text-indigo-600 mt-1">+{hp_pct(vi_m4_p['d_kulteri_kapcsolat']['beta']):.1f}%*** <span class="text-xs font-normal text-slate-400">Bécs</span></div>
          <div class="text-[11px] text-slate-500 mt-0.5">Bp: +{hp_pct(kb_m4_p['d_kulteri_kapcsolat']['beta']):.1f}% (TreeSHAP: {vi_outdoor_pct:.1f}% Bécs)</div>
        </div>
        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
          <div class="text-xs text-slate-500 font-medium">Vasúti Zaj (0–300 m)</div>
          <div class="text-lg font-bold text-red-600 mt-1">-16,9% – -17,7%***</div>
          <div class="text-[11px] text-slate-500 mt-0.5">Bécs M2; Bp kontrollált M4: {kb_m4_vag_pct:+.1f}% (NS)</div>
        </div>
        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
          <div class="text-xs text-slate-500 font-medium">Andersson–Day Határ</div>
          <div class="text-lg font-bold text-emerald-600 mt-1">300 m felett 0%</div>
          <div class="text-[11px] text-slate-500 mt-0.5">Bécs 300–500 m: +1,4% (p = {vi_p_300_500_hu} NS)</div>
        </div>
      </div>"""

    html = replace_section_between(
        html,
        "<!-- TAB 2 HEDONIKUS KPI METRIC STRIP -->",
        "\n\n      <!-- 7. ÁBRA KÁRTYA",
        new_tab2_kpis,
        "tab2_kpis",
    )

    # Fig 7 (M1-M4 Adj-R2 & AIC)
    m_ids = ["m1_base", "m2_rings", "m3_log", "m4_ad"]
    kb_adj = [round(kb_m[i]["adj_r2"], 4) for i in m_ids]
    kb_aic = [round(kb_m[i]["aic"], 2) for i in m_ids]
    vi_adj = [round(vi_m[i]["adj_r2"], 4) for i in m_ids]
    vi_aic = [round(vi_m[i]["aic"], 2) for i in m_ids]
    vi_daic = round(vi_aic[3] - vi_aic[0], 2)

    def mut_fig7(traces, layout):
        traces[0]["y"] = kb_adj
        traces[0]["text"] = [f"<b>{v:.3f}</b>" for v in kb_adj]
        traces[1]["y"] = kb_aic
        traces[2]["y"] = vi_adj
        traces[2]["text"] = [f"<b>{v:.3f}</b>" for v in vi_adj]
        traces[3]["y"] = vi_aic
        layout["yaxis2"]["range"] = [min(kb_aic) - 20, max(kb_aic) + 20]
        layout["yaxis4"]["range"] = [min(vi_aic) - 30, max(vi_aic) + 25]
        for ann in layout.get("annotations", []):
            txt = str(ann.get("text", ""))
            if "ΔAIC" in txt:
                ann["text"] = f"<b>ΔAIC = {vi_daic:.2f}!</b><br>(Drasztikus javulás M1-hez képest)"
                ann["y"] = vi_aic[3]
            elif "M2 Csúcs" in txt:
                ann["text"] = f"<b>M2 Csúcs: AIC = {kb_aic[1]:.1f}</b><br>(Legjobb illeszkedés Kőbányán)"
                ann["y"] = kb_aic[1]

    html = patch_plotly_div(html, "b64a9f1d-8cc1-4495-b7b3-21e9448daf2b", mut_fig7)

    html = replace_section_between(
        html,
        '        <div class="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700">',
        "\n        </div>\n      </div>\n\n      <!-- 8. ÁBRA KÁRTYA",
        f"""        <div class="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700">
          <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <h4 class="font-bold text-red-700 flex items-center gap-1.5 mb-1.5">
              <span>📈</span> 1. Bécs: Monoton Minőségi Ugrás (ΔAIC = {vi_daic:.2f}!)
            </h4>
            <p class="leading-relaxed text-slate-600">
              Bécs II. kerületében az M1 fizikai bázismodellhez képest a cenzorált térbeli változók drámai javulást adnak: az AIC <strong>{vi_aic[0]:.2f}-ről {vi_aic[3]:.2f}-re csökken</strong>, miközben a korrigált R² <strong>{vi_adj[0]:.4f}-ről {vi_adj[3]:.4f}-re ugrik</strong> (R² = {vi_m['m4_ad']['r2']:.4f}).
            </p>
          </div>
          <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <h4 class="font-bold text-blue-700 flex items-center gap-1.5 mb-1.5">
              <span>⚖️</span> 2. Kőbánya: Best-Practice ln(m²) + Klaszterezett Specifikáció
            </h4>
            <p class="leading-relaxed text-slate-600">
              A GeoSpec Best-Practice futtatásban (<code>ln(m²)</code>, épületszintű klaszterezés, <em>n</em> &lt; 5 ritka kategória kiszűrése) az M2 sávos modell adj-R²-e <strong>{kb_adj[1]:.4f}</strong>, míg a folytonos cenzorált M4 <strong>{kb_adj[3]:.4f}</strong>-et ér el (R² = <strong>{kb_m['m4_ad']['r2']:.4f}</strong>; a dolgozat-replikációs HC3 futamban: 0,6348).
            </p>
          </div>""",
        "fig7_prose",
        after="b64a9f1d-8cc1-4495-b7b3-21e9448daf2b",
        max_lines=30,
    )

    # Fig 8 (Halvorsen-Palmquist bars — dropping rare basement outlier n=3 per Guard #2)
    fig8_specs = [
        ("Új építés (2015 után)", "d_epult_2015_utan"),
        ("Felújított állapot (vs. átlagos)", "d_felujitott"),
        ("Erkély / terasz kapcsolat", "d_kulteri_kapcsolat"),
        ("Lift jelenléte", "d_lift"),
        ("Zárószint / tetőtér", "d_zaroszint"),
        ("Földszinti elhelyezkedés", "d_foldszint"),
        ("Felújítandó állapot (vs. átlagos)", "d_felujitando"),
        ("Panel építési technológia", "d_panel"),
    ]
    fig8_y = [lbl for lbl, _ in fig8_specs]
    kb_f8_x, kb_f8_txt = [], []
    vi_f8_x, vi_f8_txt = [], []
    for _lbl, key in fig8_specs:
        r_kb = kb_m4_p.get(key)
        if r_kb is not None:
            v = round(hp_pct(r_kb["beta"]), 2)
            kb_f8_x.append(v)
            kb_f8_txt.append(fmt_pct_label(v, r_kb["p"]))
        else:
            kb_f8_x.append(None)
            kb_f8_txt.append("")
        r_vi = vi_m4_p.get(key)
        if r_vi is not None:
            v = round(hp_pct(r_vi["beta"]), 2)
            vi_f8_x.append(v)
            vi_f8_txt.append(fmt_pct_label(v, r_vi["p"]))
        else:
            vi_f8_x.append(None)
            vi_f8_txt.append("")

    def mut_fig8(traces, layout):
        traces[0]["y"] = fig8_y
        traces[0]["x"] = kb_f8_x
        traces[0]["text"] = fig8_f8_txt = kb_f8_txt
        traces[1]["y"] = fig8_y
        traces[1]["x"] = vi_f8_x
        traces[1]["text"] = vi_f8_txt
        layout["xaxis"]["range"] = [-32, 48]
        layout["title"]["text"] = (
            "<b>8. Ábra: Fizikai Lakásjellemzők Halvorsen–Palmquist Implicit Árhatása (%)</b><br>"
            "<span style='font-size:12px;color:#64748b'>Szigorúan kizárólagos állapot-dummyk, n&lt;5 kiszűrve, épületszintű klaszterezett SE (*** p&lt;0.01, ** p&lt;0.05, * p&lt;0.1, NS: nem szignifikáns)</span>"
        )

    html = patch_plotly_div(html, "3caa3998-fc7e-4ca9-8ea1-cda30d6063c5", mut_fig8)
    html = replace_exact(
        html,
        "Minden lakásfizikai dummy pontos százalékos prémiuma és diszkontja az átlagos állapothoz viszonyítva [g = exp(β) - 1, HC3 szignifikancia-jelzésekkel].",
        "Minden lakásfizikai dummy százalékos prémiuma/diszkontja az átlagos állapothoz viszonyítva [g = exp(β) - 1, GeoSpec Best-Practice M4 klaszterezett SE, szigorúan kizárólagos állapotkategóriák].",
        "fig8_sub",
        optional=True,
    )
    html = replace_section_between(
        html,
        '        <div class="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700">',
        "\n        </div>\n      </div>\n\n      <!-- 9. ÁBRA KÁRTYA",
        f"""        <div class="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700">
          <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <h4 class="font-bold text-emerald-800 flex items-center gap-1.5 mb-1.5">
              <span>💎</span> 1. Tisztított Állapoti és Újépítési Prémiumok
            </h4>
            <p class="leading-relaxed text-slate-600">
              A szigorúan kizárólagos állapotkategóriák mellett (<em>d_felujitott = 0</em> minden 2015 utáni lakásnál) Kőbányán az új építés teljes prémiuma <strong>+{hp_pct(kb_m4_p['d_epult_2015_utan']['beta']):.1f}%***</strong> (M2-ben <strong>+{hp_pct(kb_m2_p['d_epult_2015_utan']['beta']):.1f}%***</strong>), míg a 2015 előtti felújítás <strong>+{hp_pct(kb_m4_p['d_felujitott']['beta']):.1f}% (NS)</strong>. Bécsben mindkét szegmens masszív felárat kap (új építés: <strong>+{hp_pct(vi_m4_p['d_epult_2015_utan']['beta']):.1f}%***</strong>, felújított: <strong>+{hp_pct(vi_m4_p['d_felujitott']['beta']):.1f}%**</strong>).
            </p>
          </div>
          <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <h4 class="font-bold text-red-800 flex items-center gap-1.5 mb-1.5">
              <span>🔻</span> 2. Erkély-érték és Fizikai Diszkontok
            </h4>
            <p class="leading-relaxed text-slate-600">
              Bécsben az erkély/terasz <strong>+{hp_pct(vi_m4_p['d_kulteri_kapcsolat']['beta']):.1f}%***</strong> prémiumot ad (Kőbánya: <strong>+{hp_pct(kb_m4_p['d_kulteri_kapcsolat']['beta']):.1f}%*</strong>), míg a <strong>földszinti elhelyezkedés (-{abs(hp_pct(vi_m4_p['d_foldszint']['beta'])):.1f}%***)</strong> és a <strong>felújítandó állapot ({hp_pct(kb_m4_p['d_felujitando']['beta']):.1f}%** ill. {hp_pct(vi_m4_p['d_felujitando']['beta']):.1f}%***)</strong> jelentős diszkonttal árazódik. Az <em>n</em> &lt; 5 esetszámú kőbányai szuterén kategória (<em>n</em> = 3) módszertani guard alapján kimaradt.
            </p>
          </div>""",
        "fig8_prose",
        after="3caa3998-fc7e-4ca9-8ea1-cda30d6063c5",
        max_lines=30,
    )

    # Fig 9 (M4 Forest plot)
    LABEL_MAP = {
        "log(alapterulet_nm)": "Alapterület ln(m²)",
        "szobak_szama": "Szobák száma (db)",
        "d_felujitando": "Felújítandó állapot dummy",
        "d_felujitott": "Felújított állapot dummy",
        "d_lift": "Lift jelenléte dummy",
        "d_kulteri_kapcsolat": "Erkély / terasz kapcsolat dummy",
        "kulteri_alapterulet_nm": "Kültéri méret (erkély/terasz m²)",
        "d_panel": "Panel építési technológia dummy",
        "d_foldszint": "Földszinti elhelyezkedés dummy",
        "d_zaroszint": "Zárószint / legfelső emelet dummy",
        "d_epult_2015_utan": "Új építés (2015 után) dummy",
        "AD(vagany,eucl,c=300m)": "Cenzorált vasúti közelség AD(300m)",
        "AD(fout,eucl,c=150m)": "Cenzorált főúti közelség AD(150m)",
        "AD(allomas,net,c=1125m)": "15-perces állomáselérés AD(1125m)",
    }

    def mut_fig9(traces, layout):
        for idx, (m4_res, color_hex) in enumerate([(kb["s3"]["models"][3], "#2563eb"), (vi["s3"]["models"][3], "#dc2626")]):
            params = [r for r in m4_res["params"] if r["name"] != "const"]
            params.sort(key=lambda r: r["beta"])
            traces[idx]["x"] = [round(r["beta"], 4) for r in params]
            traces[idx]["y"] = [LABEL_MAP.get(r["name"], r["label"]) for r in params]
            traces[idx]["error_x"]["array"] = [round(1.96 * r["se"], 4) for r in params]
            traces[idx]["marker"]["color"] = [color_hex if r["p"] < 0.05 else "#94a3b8" for r in params]
        layout["title"]["text"] = (
            "<b>9. Ábra: M4 Zászlóshajó Hedonikus Erdődiagram (Forest Plot, Kőbánya vs. Leopoldstadt)</b><br>"
            "<span style='font-size:12px;color:#64748b'>GeoSpec Best-Practice együtthatók és 95%-os épület-klaszterezett konfidenciaintervallumok (Szürke = p ≥ 0.05)</span>"
        )

    html = patch_plotly_div(html, "0d946b15-fa94-4153-90b9-c3416297022a", mut_fig9)
    html = replace_exact(
        html,
        "A két piacon becsült béta együtthatók és 95%-os megbízhatósági intervallumok MacKinnon–White HC3 robusztus standard hibákkal (Szürke = nem szignifikáns).",
        "A két piacon becsült M4 béta együtthatók és 95%-os épületszintű (koordináta-alapú) klaszterezett konfidenciaintervallumok (Szürke = nem szignifikáns).",
        "fig9_sub",
        optional=True,
    )
    html = replace_exact(
        html,
        "Bécsben a vasúti pálya közelsége sziklaszilárdan negatív (<strong>β = -0.2265***, p &lt; 0.001</strong>), és a főúti közelség is szignifikáns diszkontot képez (<strong>β = -0.0911***</strong>). Ezzel szemben Kőbányán a vasút közelsége önmagában nem szignifikáns a globális OLS modellben, ami a helyi térbeli struktúrák (SAR & GWR) szükségességét jelzi előre.",
        f"Bécsben a vasúti pálya közelsége sziklaszilárdan negatív (<strong>β = {vi_m4_p['AD(vagany,eucl,c=300m)']['beta']:.4f}***, p &lt; 0.001</strong>), míg Kőbányán a kontrollált globális vágányhatás statisztikailag nem szignifikáns (<strong>β = {kb_m4_p['AD(vagany,eucl,c=300m)']['beta']:.4f}, p = {kb_m4_p['AD(vagany,eucl,c=300m)']['p']:.3f}</strong>), mert az árkülönbséget az épületállomány állapota és a lokális térbeli struktúra (GWR / SAR) hordozza.",
        "fig9_prose",
        optional=True,
    )

    # Fig 10 (0-300m Acoustic decay)
    def mut_fig10(traces, layout):
        d_vi = decode_plotly_array(traces[0]["x"])
        d_kb = decode_plotly_array(traces[2]["x"])
        b_vi_ad = vi_m4_p["AD(vagany,eucl,c=300m)"]["beta"]
        b_kb_ad = kb_m4_p["AD(vagany,eucl,c=300m)"]["beta"]
        traces[0]["y"] = [round((math.exp(b_vi_ad * max(0.0, math.log(300.0 / max(1.0, d)))) - 1.0) * 100.0, 3) for d in d_vi]
        traces[2]["y"] = [round((math.exp(b_kb_ad * max(0.0, math.log(300.0 / max(1.0, d)))) - 1.0) * 100.0, 3) for d in d_kb]

        for trace_idx, m2_map in [(1, vi_m2_p), (3, kb_m2_p)]:
            pts = []
            arr_hi = []
            arr_lo = []
            txts = []
            for rng in ["ring_vagany_0_150", "ring_vagany_150_300", "ring_vagany_300_500"]:
                r = m2_map[rng]
                pct = hp_pct(r["beta"])
                hi = hp_pct(r["beta"] + 1.96 * r["se"]) - pct
                lo = pct - hp_pct(r["beta"] - 1.96 * r["se"])
                pts.append(round(pct, 2))
                arr_hi.append(round(max(0.0, hi), 2))
                arr_lo.append(round(max(0.0, lo), 2))
                txts.append(f"{pct:+.1f}%{sig_stars(r['p'])}")
            pts.append(0.0)
            arr_hi.append(0.0)
            arr_lo.append(0.0)
            txts.append("0% (Ref)")
            traces[trace_idx]["y"] = pts
            traces[trace_idx]["text"] = txts
            traces[trace_idx]["error_y"]["array"] = arr_hi
            traces[trace_idx]["error_y"]["arrayminus"] = arr_lo

        for ann in layout.get("annotations", []):
            txt = str(ann.get("text", ""))
            if "Bécs II. Leopoldstadt" in txt:
                ann["text"] = f"<b>1. Bécs II. Leopoldstadt: Szignifikáns Akusztikai Amortizáció (β = {b_vi_ad:.4f}***)</b>"
            elif "Budapest X. Kőbánya" in txt:
                ann["text"] = f"<b>2. Kőbánya: Állapot-kontroll után nincs szignifikáns zajdiszkont (β = {b_kb_ad:.4f}, NS)</b>"
            elif "300 m határ" in txt:
                p_300 = vi_m2_p["ring_vagany_300_500"]["p"]
                ann["text"] = f"300 m határ (0% zajhatás, p={p_300:.3f})"
            elif "Nyers" in txt or "Kőbánya ≤500m" in txt:
                p_kb = kb_m4_p["AD(vagany,eucl,c=300m)"]["p"]
                ann["text"] = f"Kőbánya ≤500m völgy: -21,1% (nyers) | M4 kontrollált: {kb_m4_vag_pct:+.1f}% (p={p_kb:.3f} NS)"

    html = patch_plotly_div(html, "959ffbee-cb91-459e-b6b3-5613efc2c90e", mut_fig10)
    html = replace_section_between(
        html,
        '        <div class="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700">',
        '\n        </div>\n        <div class="mt-3 p-3 bg-amber-50',
        f"""        <div class="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700">
          <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <h4 class="font-bold text-red-700 flex items-center gap-1.5 mb-1.5">
              <span>🔊</span> 1. Bécs II. Leopoldstadt: Szigorú 300 m-es Akusztikai Zóna
            </h4>
            <p class="leading-relaxed text-slate-600">
              Bécs sűrű szövetében a vasúti pálya mentén 0–150 m között <strong>{hp_pct(vi_m2_p['ring_vagany_0_150']['beta']):.1f}%**</strong>, 150–300 m között <strong>{hp_pct(vi_m2_p['ring_vagany_150_300']['beta']):.1f}%***</strong> az akusztikai diszkont. A 300 m-es határt átlépve az externália eltűnik (<strong>+{hp_pct(vi_m2_p['ring_vagany_300_500']['beta']):.1f}%, p = {vi_m2_p['ring_vagany_300_500']['p']:.3f}</strong>). Az M4 folytonos görbéje (β = {vi_m4_p['AD(vagany,eucl,c=300m)']['beta']:.4f}***) pontosan leírja a küszöbértéket.
            </p>
          </div>
          <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <h4 class="font-bold text-blue-700 flex items-center gap-1.5 mb-1.5">
              <span>🛡️</span> 2. Kőbánya: Állapot-konfounding vs. Silent Rail 2. Fázis
            </h4>
            <p class="leading-relaxed text-slate-600">
              Kőbányán az épületminőség és az örökölt zárványhatás kontrollálása után a vágányhatás nem szignifikáns (<strong>M4 β = {kb_m4_p['AD(vagany,eucl,c=300m)']['beta']:.4f}, {kb_m4_vag_pct:+.1f}% NS</strong>). Fontos korlát, hogy a puszta távolsági átlag még nem választja szét a pálya- és járműműszaki tényezőket (ív vs. egyenes, töltés vs. viadukt, korszerű motorvonat vs. fékező tehervonat), amelyeket a Silent Rail 2. fázis (BME × UTCN) mérési programja vizsgál tovább.
            </p>
          </div>""",
        "fig10_prose",
        after="959ffbee-cb91-459e-b6b3-5613efc2c90e",
        max_lines=30,
    )

    # Grand Hedonic Model Table + Stepwise Composition Table (Tab 2)
    step_rows_html = []
    for st in kb["s2"]["composition"]["steps"]:
        pct_s = hp_pct(st["beta"])
        st_sig = st.get("sig") or (" (NS)" if st["p"] >= 0.10 else "")
        step_rows_html.append(
            f"""                <tr class="hover:bg-slate-50">
                  <td class="py-2 px-3 font-bold text-slate-800">{st['step']}</td>
                  <td class="py-2 px-3 text-slate-600">{st['added_block']} (k = {st['k_params']})</td>
                  <td class="py-2 px-3 text-center font-mono text-slate-700">{st['beta']:+.4f}</td>
                  <td class="py-2 px-3 text-center font-mono text-slate-600">{st['se']:.4f}</td>
                  <td class="py-2 px-3 text-center font-mono text-slate-700">{st['beta']/st['se']:+.2f}</td>
                  <td class="py-2 px-3 text-center font-mono text-slate-500">{st['p']:.4f}{st_sig}</td>
                  <td class="py-2 px-3 text-center font-semibold text-slate-700">{pct_s:+.1f}%</td>
                  <td class="py-2 px-3 text-center font-mono font-bold text-emerald-700">{st['adj_r2']:.4f}</td>
                </tr>"""
        )

    new_hedonic_tables = f"""      <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
        <div class="flex flex-wrap items-center justify-between gap-2 mb-3 pb-3 border-b border-slate-100">
          <div class="flex flex-wrap items-center gap-x-3 gap-y-1.5 min-w-0">
            <span class="bg-blue-600 text-white font-black text-xs px-2.5 py-1.5 rounded-lg shadow-sm">ÖSSZEFOGLALÓ</span>
            <span class="bg-blue-50 text-blue-700 font-bold text-xs px-2.5 py-1 rounded-md border border-blue-200">Modell-rangsor &amp; Információs Kritériumok</span>
            <h4 class="text-base font-bold text-slate-900">Modellválasztási Nagytáblázat (M1–M4 GeoSpec Best-Practice Specifikációk)</h4>
          </div>
          <span class="text-xs text-slate-500 font-medium">Szigorúan M1–M4 · ln(m²) · épület-klaszterezett SE</span>
        </div>
        <p class="text-xs text-slate-500 mb-4">A négy ökonometriai specifikáció összehasonlítása (N<sub>Bp</sub> = 269, N<sub>Bécs</sub> = 378, <em>n</em> &lt; 5 ritka kategória kiszűrve, kizárólagos állapotkategóriák). A november 1-jei HC3 replikációs eredmények a <a href="./recipes/kobanya_paper_replication_recipe.json" download class="underline text-blue-700">replikációs receptben</a> reprodukálhatók (Bp M4 R² = 0,6348).</p>
        <div class="overflow-x-auto text-sm">
          <table class="w-full text-left border-collapse">
            <thead>
              <tr class="bg-slate-50 border-b border-slate-200 text-xs font-bold text-slate-600 uppercase">
                <th class="py-3 px-3">Modell</th>
                <th class="py-3 px-3">Specifikáció Típusa</th>
                <th class="py-3 px-3 text-center">Kőbánya R² / Adj-R²</th>
                <th class="py-3 px-3 text-center">Kőbánya AIC</th>
                <th class="py-3 px-3 text-center">Leopoldstadt R² / Adj-R²</th>
                <th class="py-3 px-3 text-center">Leopoldstadt AIC</th>
                <th class="py-3 px-3 text-center">Értékelés</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100 text-xs">
              <tr class="hover:bg-slate-50/50">
                <td class="py-2.5 px-3 font-bold text-slate-900">M1</td>
                <td class="py-2.5 px-3 text-slate-600">Fizikai alapmodell (nincs térbeli változó)</td>
                <td class="py-2.5 px-3 text-center font-semibold text-slate-700">{kb_m['m1_base']['r2']:.4f} / {kb_m['m1_base']['adj_r2']:.4f}</td>
                <td class="py-2.5 px-3 text-center text-slate-600">{kb_m['m1_base']['aic']:.2f}</td>
                <td class="py-2.5 px-3 text-center font-semibold text-slate-700">{vi_m['m1_base']['r2']:.4f} / {vi_m['m1_base']['adj_r2']:.4f}</td>
                <td class="py-2.5 px-3 text-center text-slate-600">{vi_m['m1_base']['aic']:.2f}</td>
                <td class="py-2.5 px-3 text-center"><span class="bg-slate-100 text-slate-600 px-2 py-0.5 rounded text-[10px]">Bázismodell</span></td>
              </tr>
              <tr class="hover:bg-slate-50/50">
                <td class="py-2.5 px-3 font-bold text-slate-900">M2</td>
                <td class="py-2.5 px-3 text-slate-600">Diszkrét sávos modell (zaj- és TOD-pufferek)</td>
                <td class="py-2.5 px-3 text-center font-bold text-blue-600">{kb_m['m2_rings']['r2']:.4f} / {kb_m['m2_rings']['adj_r2']:.4f}</td>
                <td class="py-2.5 px-3 text-center font-bold text-blue-600">{kb_m['m2_rings']['aic']:.2f}</td>
                <td class="py-2.5 px-3 text-center font-semibold text-slate-700">{vi_m['m2_rings']['r2']:.4f} / {vi_m['m2_rings']['adj_r2']:.4f}</td>
                <td class="py-2.5 px-3 text-center text-slate-600">{vi_m['m2_rings']['aic']:.2f}</td>
                <td class="py-2.5 px-3 text-center"><span class="bg-blue-100 text-blue-700 px-2 py-0.5 rounded text-[10px] font-semibold">Legjobb Bp illeszkedés</span></td>
              </tr>
              <tr class="hover:bg-slate-50/50">
                <td class="py-2.5 px-3 font-bold text-slate-900">M3</td>
                <td class="py-2.5 px-3 text-slate-600">Folytonos log-modell (ln-távolságok, GWR fókusz)</td>
                <td class="py-2.5 px-3 text-center font-semibold text-slate-700">{kb_m['m3_log']['r2']:.4f} / {kb_m['m3_log']['adj_r2']:.4f}</td>
                <td class="py-2.5 px-3 text-center text-slate-600">{kb_m['m3_log']['aic']:.2f}</td>
                <td class="py-2.5 px-3 text-center font-semibold text-slate-700">{vi_m['m3_log']['r2']:.4f} / {vi_m['m3_log']['adj_r2']:.4f}</td>
                <td class="py-2.5 px-3 text-center text-slate-600">{vi_m['m3_log']['aic']:.2f}</td>
                <td class="py-2.5 px-3 text-center"><span class="bg-amber-100 text-amber-700 px-2 py-0.5 rounded text-[10px]">GWR fókuszmodell</span></td>
              </tr>
              <tr class="bg-blue-50/60 font-semibold border-t-2 border-blue-200">
                <td class="py-3 px-3 font-black text-blue-900 flex items-center gap-1"><span>⭐</span> M4</td>
                <td class="py-3 px-3 font-medium text-blue-900">Hármas cenzorált zászlóshajó (Vasút 300m, Főút 150m, Állomás 1125m)</td>
                <td class="py-3 px-3 text-center font-bold text-blue-900">{kb_m['m4_ad']['r2']:.4f} / {kb_m['m4_ad']['adj_r2']:.4f}</td>
                <td class="py-3 px-3 text-center text-blue-800">{kb_m['m4_ad']['aic']:.2f}</td>
                <td class="py-3 px-3 text-center font-black text-red-600 text-sm">{vi_m['m4_ad']['r2']:.4f} / {vi_m['m4_ad']['adj_r2']:.4f}</td>
                <td class="py-3 px-3 text-center font-black text-red-600 text-sm">{vi_m['m4_ad']['aic']:.2f}</td>
                <td class="py-3 px-3 text-center"><span class="bg-emerald-600 text-white font-bold px-2 py-0.5 rounded shadow-sm text-[10px]">Zászlóshajó Csúcs</span></td>
              </tr>
            </tbody>
          </table>
        </div>
        <!-- LÉPCSŐZETES ROBUSZTUSSÁGI TÁBLÁZAT: KŐBÁNYAI VÁGÁNYHATÁS VS ÖSSZETÉTELI HATÁS -->
        <div class="mt-6 pt-5 border-t border-slate-200">
          <div class="flex items-center justify-between mb-2">
            <h5 class="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
              <span>🔍</span> Kompozíciós Lánc (Gelbach-felbontás): A Kőbányai M4 Vágányhatás Attenuációja
            </h5>
            <span class="text-[11px] text-slate-500 font-medium">GeoSpec Stage 2 · Épület-klaszterezett SE (k=4 → k=15)</span>
          </div>
          <p class="text-xs text-slate-500 mb-3">A nyers kőbányai állomásközeli völgy (≤500 m: -21,1%) a kontrollblokkok bevonásával lépcsőzetesen tisztul le: végső M4 vágányhatás β = {kb_m4_p['AD(vagany,eucl,c=300m)']['beta']:.4f} (attenuáció: {kb['s2']['composition']['final_attenuation_pct']:.1f}%, p = {kb_m4_p['AD(vagany,eucl,c=300m)']['p']:.3f} NS).</p>
          <div class="overflow-x-auto text-xs">
            <table class="w-full text-left border-collapse">
              <thead>
                <tr class="bg-blue-50/60 border-b border-blue-200 text-[11px] font-bold text-blue-900 uppercase">
                  <th class="py-2.5 px-3">Lépcső</th>
                  <th class="py-2.5 px-3">Bevont Kontrollblokk</th>
                  <th class="py-2.5 px-3 text-center">Vágány AD β</th>
                  <th class="py-2.5 px-3 text-center">Klaszter SE</th>
                  <th class="py-2.5 px-3 text-center">t-érték</th>
                  <th class="py-2.5 px-3 text-center">p-érték</th>
                  <th class="py-2.5 px-3 text-center">exp(β) - 1</th>
                  <th class="py-2.5 px-3 text-center">Korrigált R²</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-slate-100 text-[11px]">
{chr(10).join(step_rows_html)}
              </tbody>
            </table>
          </div>
        </div>
        <div class="mt-4 p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-600 leading-relaxed">
          <strong>Ökonometriai Módszertani Indoklás:</strong> A Best-Practice M4 modell szigorúan kizárólagos állapotkategóriákkal, <code>ln(m²)</code> alapterület-rugalmassággal, <em>n</em> &lt; 5 ritka kategória kiszűrésével (a 3 db kőbányai szuterén nélkül) és koordináta-alapú épületszintű klaszterezéssel fut, kiküszöbölve az azonos társasházi projektek ál-replikációját és a diszkrét sávhatárok ugrásait.
        </div>
      </div>"""

    html = replace_section_between(
        html,
        '      <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-200">\n        <div class="flex flex-wrap items-center justify-between gap-2 mb-3 pb-3 border-b border-slate-100">\n          <div class="flex flex-wrap items-center gap-x-3 gap-y-1.5 min-w-0">\n            <span class="bg-blue-600 text-white font-black text-xs px-2.5 py-1.5 rounded-lg shadow-sm">ÖSSZEFOGLALÓ</span>',
        "\n    </div>\n\n    <!-- TAB 3: TÉRÖKONOMETRIA",
        new_hedonic_tables,
        "hedonic_tables",
    )

    # 5. TAB 3: SPATIAL KPIs, FIG 12, FIG 13, FIG 22 & GRAND SPATIAL TABLE
    kb_s4 = kb["s4_m4"]
    vi_s4 = vi["s4_m4"]
    kb_sar_m4 = next(x for x in kb_s4["model_comparison"] if x["model"].startswith("SAR"))
    vi_sar_m4 = next(x for x in vi_s4["model_comparison"] if x["model"].startswith("SAR"))
    vi_sem_m4 = next(x for x in vi_s4["model_comparison"] if x["model"].startswith("SEM"))
    kb_imp_vag = next(x for x in kb_s4["impacts_sar"] if "vagany" in x["variable"])
    vi_imp_vag = next(x for x in vi_s4["impacts_sar"] if "vagany" in x["variable"])
    vi_idw_row = next((w for w in vi_s4["weights_robustness"] if w["w_id"] == "idw_2500"), {})
    vi_sar_verdict = "Classic OLS" if "Classic OLS" in vi_s4["anselin"]["verdict"] else vi_s4["anselin"]["recommended_model"]

    html = replace_exact(
        html,
        "β = -1.6257 (50.6% szign.)",
        f"GWR R² = {kb['s5']['gwr_r2']:.4f} (Bp) / {vi['s5']['gwr_r2']:.4f} (Bécs)",
        "tab3_hero_gwr",
        optional=True,
    )

    new_tab3_kpis = f"""<!-- TAB 3 SPATIAL KPI METRIC STRIP -->
      <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
          <div class="text-xs text-slate-500 font-medium">Moran's I Nyers Árak</div>
          <div class="text-lg font-bold text-blue-700 mt-1">{kb_s4['moran_y']['moran_i']:.3f}*** <span class="text-xs font-normal text-slate-400">Bp</span></div>
          <div class="text-[11px] text-slate-500 mt-0.5">Bécs knn5_dedup: {vi_s4['moran_y']['moran_i']:.3f}***</div>
        </div>
        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
          <div class="text-xs text-slate-500 font-medium">M4 OLS Reziduális Moran's I</div>
          <div class="text-lg font-bold text-emerald-600 mt-1">{kb_s4['moran_resid']['moran_i']:.3f} <span class="text-xs font-normal text-slate-400">Bp</span></div>
          <div class="text-[11px] text-slate-500 mt-0.5">Bécs knn5_dedup: {vi_s4['moran_resid']['moran_i']:+.3f} (NS)</div>
        </div>
        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
          <div class="text-xs text-slate-500 font-medium">Épületek Közti SAR Spillover</div>
          <div class="text-lg font-bold text-indigo-600 mt-1">{kb_imp_vag['spillover_share_pct']:.1f}% <span class="text-xs font-normal text-slate-400">Bp**</span></div>
          <div class="text-[11px] text-slate-500 mt-0.5">Bécs knn5_dedup: {vi_imp_vag['spillover_share_pct']:.1f}% (NS; ρ={vi_sar_m4['spatial_param']:.3f})</div>
        </div>
        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
          <div class="text-xs text-slate-500 font-medium">Anselin Döntési Fa (M4)</div>
          <div class="text-lg font-bold text-blue-700 mt-1">SAR (Bp) / {vi_sar_verdict} (Bécs)</div>
          <div class="text-[11px] text-slate-500 mt-0.5">Egységes knn5_dedup (0% helyazonos)</div>
        </div>
        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
          <div class="text-xs text-slate-500 font-medium">GWR R² (m3_log Fókusz)</div>
          <div class="text-lg font-bold text-emerald-600 mt-1">{kb['s5']['gwr_r2']*100:.1f}% <span class="text-xs font-normal text-slate-400">Bp</span></div>
          <div class="text-[11px] text-slate-500 mt-0.5">Bécs: {vi['s5']['gwr_r2']*100:.1f}% (Adj-R²: {kb['s5']['gwr_adj_r2']:.4f})</div>
        </div>
        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
          <div class="text-xs text-slate-500 font-medium">Védett TOD Potenciál</div>
          <div class="text-lg font-bold text-purple-600 mt-1">+10–23% Rent Gap</div>
          <div class="text-[11px] text-slate-500 mt-0.5">Zajvédett gyalogos körzetben</div>
        </div>
      </div>"""

    html = replace_section_between(
        html,
        "<!-- TAB 3 SPATIAL KPI METRIC STRIP -->",
        "\n\n      <!-- MODUL A: GLOBÁLIS",
        new_tab3_kpis,
        "tab3_kpis",
    )

    # Fig 12 (Moran's I bars & scatter legend names)
    def mut_fig12(traces, layout):
        traces[0]["name"] = f"Bécs (I = {vi_s4['moran_y']['moran_i']:.4f}***, z = {vi_s4['moran_y']['z']:.2f})"
        traces[1]["name"] = f"Kőbánya (I = {kb_s4['moran_y']['moran_i']:.4f}***, z = {kb_s4['moran_y']['z']:.2f})"
        traces[2]["name"] = f"Bécs Moran-lejtő ({vi_s4['moran_y']['moran_i']:.3f})"
        traces[3]["name"] = f"Kőbánya Moran-lejtő ({kb_s4['moran_y']['moran_i']:.3f})"
        kb_mi = [
            round(kb_s4["moran_y"]["moran_i"], 4),
            round(kb["s4_by_model"]["m1_base"]["moran_resid"]["moran_i"], 4),
            round(kb["s4_by_model"]["m2_rings"]["moran_resid"]["moran_i"], 4),
            round(kb["s4_by_model"]["m3_log"]["moran_resid"]["moran_i"], 4),
            round(kb["s4_by_model"]["m4_ad"]["moran_resid"]["moran_i"], 4),
        ]
        vi_mi = [
            round(vi_s4["moran_y"]["moran_i"], 4),
            round(vi["s4_by_model"]["m1_base"]["moran_resid"]["moran_i"], 4),
            round(vi["s4_by_model"]["m2_rings"]["moran_resid"]["moran_i"], 4),
            round(vi["s4_by_model"]["m3_log"]["moran_resid"]["moran_i"], 4),
            round(vi["s4_by_model"]["m4_ad"]["moran_resid"]["moran_i"], 4),
        ]
        traces[4]["y"] = kb_mi
        traces[4]["text"] = [f"{v:.3f}" for v in kb_mi]
        traces[5]["y"] = vi_mi
        traces[5]["text"] = [f"{v:.3f}" for v in vi_mi]

    html = patch_plotly_div(html, "363191f6-5d1b-4602-9d99-4139a2246fd3", mut_fig12)
    html = replace_section_between(
        html,
        '        <div class="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700">',
        "\n        </div>\n      </div>\n\n      <!-- MODUL B: SAR TÉRBELI HATÁSFELBONTÁS -->",
        f"""        <div class="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700">
          <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <h4 class="font-bold text-indigo-900 flex items-center gap-1.5 mb-1.5">
              <span>📍</span> 1. Nyers Térbeli Klaszterezettség (egységes knn5_dedup súlymátrix)
            </h4>
            <p class="leading-relaxed text-slate-600">
              A nyers lakásárak az azonos épületen belüli duplikációktól megtisztított <code>knn5_dedup</code> súlymátrixszal mindkét piacon szignifikáns klaszterezettséget mutatnak (Kőbánya: <strong>I = {kb_s4['moran_y']['moran_i']:.4f}***, z = {kb_s4['moran_y']['z']:.2f}</strong>; Bécs: <strong>I = {vi_s4['moran_y']['moran_i']:.4f}***, z = {vi_s4['moran_y']['z']:.2f}</strong>).
            </p>
          </div>
          <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <h4 class="font-bold text-emerald-800 flex items-center gap-1.5 mb-1.5">
              <span>📉</span> 2. Egzakt M<sub>x</sub>-korrigált Reziduális Moran's I és SAR/OLS Eltérés
            </h4>
            <p class="leading-relaxed text-slate-600">
              Az egzakt <em>M<sub>x</sub></em> reziduális diagnosztikával az M4 OLS után Bécsben a <code>knn5_dedup</code> mátrixon eltűnik az épületek közötti maradék autokorreláció (<strong>I = {vi_s4['moran_resid']['moran_i']:+.4f}, p = {vi_s4['moran_resid']['p']:.4f} NS</strong> → Classic OLS), míg Kőbányán valós épületek közötti térbeli függőség marad (<strong>I = {kb_s4['moran_resid']['moran_i']:.4f}***</strong>, SAR szűrés után: <strong>{kb_sar_m4['resid_moran_i']:.4f}</strong> NS).
            </p>
          </div>""",
        "fig12_prose",
        after="363191f6-5d1b-4602-9d99-4139a2246fd3",
        max_lines=30,
    )

    # Fig 13 (SAR direct/indirect/total impacts from GeoSpec Stage 4 — unified knn5_dedup)
    vi_imp_map = {x["variable"]: x for x in vi_s4["impacts_sar"]}
    kb_imp_map = {x["variable"]: x for x in kb_s4["impacts_sar"]}
    f13_items = [
        ("Bécs M4: Vágányzaj AD(300m)", vi_imp_map["AD(vagany,eucl,c=300m)"]),
        ("Bécs M4: Főúti zaj AD(150m)", vi_imp_map["AD(fout,eucl,c=150m)"]),
        ("Bécs M4: Állomás AD(1125m)", vi_imp_map["AD(allomas,net,c=1125m)"]),
        ("Kőbánya M4: Vágányzaj AD(300m)", kb_imp_map["AD(vagany,eucl,c=300m)"]),
        ("Kőbánya M4: Főúti zaj AD(150m)", kb_imp_map["AD(fout,eucl,c=150m)"]),
        ("Kőbánya M4: Állomás AD(1125m)", kb_imp_map["AD(allomas,net,c=1125m)"]),
    ]

    def mut_fig13(traces, layout):
        y_lbls = [lbl for lbl, _ in f13_items]
        d_vals = [round(it["direct"]["est"], 4) for _, it in f13_items]
        i_vals = [round(it["indirect"]["est"], 4) for _, it in f13_items]
        t_vals = [round(it["total"]["est"], 4) for _, it in f13_items]
        shares = [round(it["spillover_share_pct"], 1) for _, it in f13_items]
        for tr, vals in [(traces[0], d_vals), (traces[1], i_vals), (traces[2], t_vals)]:
            tr["y"] = y_lbls
            tr["x"] = vals
            tr["text"] = [f"{v:+.3f}" for v in vals]
        traces[3]["y"] = y_lbls
        traces[3]["x"] = shares
        traces[3]["text"] = [f"<b>{s:.1f}%</b>" for s in shares]
        traces[3]["marker"]["color"] = ["#94a3b8", "#94a3b8", "#94a3b8", "#2563eb", "#2563eb", "#2563eb"]
        for ann in layout.get("annotations", []):
            if "Bécs" in str(ann.get("text", "")) and "%" in str(ann.get("text", "")):
                ann["text"] = f"<b>Bécs knn5_dedup spillover: {vi_imp_vag['spillover_share_pct']:.1f}% (NS)</b>"

    html = patch_plotly_div(html, "38a4711e-cbfa-42fe-a0ce-476fd654497b", mut_fig13)

    # Fig 22 (OLS vs GWR R2 jump — m3_log focus from Stage 5)
    def mut_fig22(traces, layout):
        kb_ols_r2 = round(kb["s5"]["global_ols_r2"], 4)
        kb_gwr_r2 = round(kb["s5"]["gwr_r2"], 4)
        vi_ols_r2 = round(vi["s5"]["global_ols_r2"], 4)
        vi_gwr_r2 = round(vi["s5"]["gwr_r2"], 4)
        traces[0]["y"] = [kb_ols_r2, vi_ols_r2]
        traces[0]["text"] = [f"OLS (m3_log): {kb_ols_r2*100:.1f}%", f"OLS (m3_log): {vi_ols_r2*100:.1f}%"]
        traces[1]["y"] = [kb_gwr_r2, vi_gwr_r2]
        traces[1]["text"] = [
            f"<b>GWR: {kb_gwr_r2*100:.1f}%</b><br>(+{(kb_gwr_r2-kb_ols_r2)*100:.1f}%p)",
            f"<b>GWR: {vi_gwr_r2*100:.1f}%</b><br>(+{(vi_gwr_r2-vi_ols_r2)*100:.1f}%p)",
        ]

    html = patch_plotly_div(html, "484e9b34-272d-4a9b-934b-136c09d661cd", mut_fig22)

    # Grand Spatial Table (M1-M4 OLS vs SAR vs SEM + Anselin LM)
    spatial_rows = []
    spec_names = {
        "m1_base": "M1: Fizikai alapmodell",
        "m2_rings": "M2: Diszkrét sávos modell",
        "m3_log": "M3: Folytonos log-modell (GWR fókusz)",
        "m4_ad": "M4: Hármas cenzorált zászlóshajó",
    }
    for city_lbl, city_cls, res in [("Bp", "text-blue-600", kb), ("Bécs", "text-red-600", vi)]:
        for mid in m_ids:
            s4 = res["s4_by_model"][mid]
            comp = {c["model"].split()[0]: c for c in s4["model_comparison"]}
            ols, sar, sem = comp["OLS"], comp["SAR"], comp["SEM"]
            verd = s4["anselin"]["verdict"]
            badge_cls = "bg-blue-100 text-blue-800" if "SAR" in verd else ("bg-purple-100 text-purple-800" if "SEM" in verd else "bg-slate-100 text-slate-600")
            row_cls = "bg-blue-50/60 font-semibold border-t-2 border-blue-200" if mid == "m4_ad" else "hover:bg-slate-50/50"
            spatial_rows.append(
                f"""              <tr class="{row_cls}">
                <td class="py-2.5 px-3 font-bold text-slate-900"><span class="{city_cls} font-bold">{city_lbl}</span> {mid[:2].upper()}</td>
                <td class="py-2.5 px-3 text-slate-600">{spec_names[mid]}</td>
                <td class="py-2.5 px-3 text-center">{ols['r2_or_pseudo']:.4f}</td>
                <td class="py-2.5 px-3 text-center text-slate-500">{ols['aic']:.1f}</td>
                <td class="py-2.5 px-3 text-center font-bold text-blue-700">{sar['spatial_param']:.4f}</td>
                <td class="py-2.5 px-3 text-center font-bold text-blue-700">{sar['r2_or_pseudo']:.4f}</td>
                <td class="py-2.5 px-3 text-center font-black text-blue-700">{sar['aic']:.1f}</td>
                <td class="py-2.5 px-3 text-center text-purple-700">{sem['spatial_param']:.4f}</td>
                <td class="py-2.5 px-3 text-center text-purple-700">{sem['r2_or_pseudo']:.4f}</td>
                <td class="py-2.5 px-3 text-center text-purple-700">{sem['aic']:.1f}</td>
                <td class="py-2.5 px-3 text-center"><span class="{badge_cls} font-bold px-2 py-0.5 rounded text-[10px]">{verd}</span></td>
              </tr>"""
            )

    new_spatial_table = f"""        <div class="overflow-x-auto text-sm">
          <table class="w-full text-left border-collapse">
            <thead>
              <tr class="bg-slate-50 border-b border-slate-200 text-xs font-bold text-slate-600 uppercase">
                <th class="py-3 px-3">Modell</th>
                <th class="py-3 px-3">Specifikáció Típusa</th>
                <th class="py-3 px-3 text-center">OLS R²</th>
                <th class="py-3 px-3 text-center">OLS AIC</th>
                <th class="py-3 px-3 text-center">SAR ρ (Wy)</th>
                <th class="py-3 px-3 text-center">SAR Pseudo R²</th>
                <th class="py-3 px-3 text-center">SAR AIC</th>
                <th class="py-3 px-3 text-center">SEM λ (Wu)</th>
                <th class="py-3 px-3 text-center">SEM Pseudo R²</th>
                <th class="py-3 px-3 text-center">SEM AIC</th>
                <th class="py-3 px-3 text-center">Anselin LM Döntés</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100 text-xs">
{chr(10).join(spatial_rows)}
            </tbody>
          </table>
        </div>
        <div class="mt-4 p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-600 leading-relaxed">
          <strong>Ökonometriai Módszertani Eredmény (GeoSpec v1.0-sealed Stage 4 — egységes <code>knn5_dedup</code>):</strong> Kőbányán a deduplikált mátrixszal is a <strong>SAR modell</strong> a domináns (ρ = {kb_sar_m4['spatial_param']:.4f}**, RLM-Lag = {kb_s4['anselin']['rlm_lag']:.2f}, p = {kb_s4['anselin']['rlm_lag_p']:.4f}; {kb_imp_vag['spillover_share_pct']:.1f}% spillover). Bécsben az azonos társasházi duplikációk kiszűrésével (<code>knn5_dedup</code>) az épületek közötti reziduális spillover eltűnik (SAR ρ = {vi_sar_m4['spatial_param']:.4f}, p = {vi_sar_m4['spatial_p']:.4f} NS; Anselin: {vi_s4['anselin']['verdict']}). Robusztussági kontrollként az <code>idw_2500</code> ({vi_idw_row.get('colocated_pairs_pct', 32.3):.1f}% helyazonos társasházi pár) még mutat házon belüli korrelációt (ρ = {vi_idw_row.get('sar_rho', 0.4223):.4f}***, λ = {vi_idw_row.get('sem_lambda', 0.7422):.4f}***).
        </div>"""

    html = replace_section_between(
        html,
        '        <div class="overflow-x-auto text-sm">\n          <table class="w-full text-left border-collapse">\n            <thead>\n              <tr class="bg-slate-50 border-b border-slate-200 text-xs font-bold text-slate-600 uppercase">\n                <th class="py-3 px-3">Modell</th>\n                <th class="py-3 px-3">Specifikáció Típusa</th>\n                <th class="py-3 px-3 text-center">OLS R²',
        "\n      </div>\n\n    </div>\n\n    <!-- TAB 4: INTERAKTÍV TÉRKÉP",
        new_spatial_table,
        "spatial_table",
    )

    # Fig 13 prose card
    html = replace_section_between(
        html,
        '        <div class="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700">',
        "\n        </div>\n      </div>\n\n      <!-- MODUL",
        f"""        <div class="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700">
          <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <h4 class="font-bold text-blue-900 flex items-center gap-1.5 mb-1.5">
              <span>🏙️</span> 1. Bécs: Épületek között ~0% (<code>knn5_dedup</code> → Classic OLS)
            </h4>
            <p class="leading-relaxed text-slate-600">
              Bécs Leopoldstadtban a Guard #5 szerinti egységes <code>knn5_dedup</code> súlymátrixszal (0% azonos társasházon belüli pár) az épületek közötti maradék SAR spillover statisztikailag nulla (<strong>ρ = {vi_sar_m4['spatial_param']:.3f}, p = {vi_sar_m4['spatial_p']:.3f} NS</strong>; Classic OLS), vagyis a korszerű állomány és a 300 m-es zajhatár önmagában megmagyarázza az árakat. Az <code>idw_2500</code> ({vi_idw_row.get('colocated_pairs_pct', 32.3):.1f}% helyazonos kapcsolat) ρ = {vi_idw_row.get('sar_rho', 0.422):.3f}*** értéke a társasházon belüli lakáscsoportokból ered.
            </p>
          </div>
          <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <h4 class="font-bold text-slate-900 flex items-center gap-1.5 mb-1.5">
              <span>🧭</span> 2. Kőbánya: Valós épületek közötti átgyűrűzés ({kb_imp_vag['spillover_share_pct']:.1f}%)
            </h4>
            <p class="leading-relaxed text-slate-600">
              Kőbányán ugyanezen deduplikált épületmátrixon (<code>knn5_dedup</code>, 0% helyazonos pár) is valós, szignifikáns szomszédsági árhatás mérhető (SAR <strong>ρ = {kb_sar_m4['spatial_param']:.3f}**, p = {kb_sar_m4['spatial_p']:.4f}</strong>; <strong>{kb_imp_vag['spillover_share_pct']:.1f}%</strong> indirekt arány, {1.0/(1.0-kb_sar_m4['spatial_param']):.2f}× multiplikátor): a leromlott környezet lehúzza a szomszédos házakat.
            </p>
          </div>""",
        "fig13_prose",
        after="38a4711e-cbfa-42fe-a0ce-476fd654497b",
        max_lines=30,
    )

    # 6. TAB 4: DETOUR NUMBERS & TUNNEL STIGMA TYPO FIX
    html = replace_exact(
        html,
        '<div class="text-lg font-bold text-red-600 mt-1">+1 071 m (Bp)</div>',
        '<div class="text-lg font-bold text-red-600 mt-1">+1 065 m (Bp)</div>',
        "tab4_detour_kpi",
        optional=True,
    )
    html = replace_exact(
        html,
        "Utcahálózati Morfológia & Kétoldali Bejáratok (Max Δd = +1 071 m)",
        "Utcahálózati Morfológia & Kétoldali Bejáratok (Medián +141 m, Max Δd = +1 065 m)",
        "tab4_detour_h4",
        optional=True,
    )
    html = replace_exact(
        html,
        "(Δd = Hálózat - Légvonal, max +1 071 m)",
        "(Δd = Hálózat - Légvonal, medián +141 m, max +1 065 m)",
        "tab4_detour_p",
        optional=True,
    )
    html = replace_exact(
        html,
        "vasúti alagút-zárvány",
        "vasúti vágány-zárvány",
        "tab4_alagut_zarvany",
        optional=True,
    )

    # 7. TAB 5: SEALED STAGE 6 PURE-NUMPY GBRT & EXACT TREESHAP + NEW WAVE RESIDENCE LVC UPDATE
    new_tab5_kpis = f"""        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
          <div class="text-xs text-slate-500 font-medium">TreeSHAP Súly (Kőbánya)</div>
          <div class="text-lg font-bold text-blue-700 mt-1">{kb_phys_hu}% vs. {kb_trk_hu}%</div>
          <div class="text-[11px] text-slate-500 mt-0.5">Fizikai jellemzők vs. vágány (össz. vasút: {kb_rail_hu}%)</div>
        </div>
        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
          <div class="text-xs text-slate-500 font-medium">TreeSHAP Súly (Bécs)</div>
          <div class="text-lg font-bold text-emerald-600 mt-1">{vi_out_hu}% vs. {vi_trk_hu}%</div>
          <div class="text-[11px] text-slate-500 mt-0.5">Erkély/terasz vs. vágány (össz. vasút: {vi_rail_hu}%)</div>
        </div>
        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
          <div class="text-xs text-slate-500 font-medium">LVC Forrásalap (New Wave)</div>
          <div class="text-lg font-bold text-blue-700 mt-1">1,84 Mrd Ft</div>
          <div class="text-[11px] text-slate-500 mt-0.5">6,8 ha (+10% × 20%) · 15,8% ROI</div>
        </div>"""

    html = replace_section_between(
        html,
        '        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">\n          <div class="text-xs text-slate-500 font-medium">',
        "\n      </div>\n\n      <!-- MODUL 1:",
        new_tab5_kpis,
        "tab5_kpis",
        after="Alapterület (m²)</div>",
        max_lines=30,
    )

    # Fig 14 (Sealed Stage 6 GroupKFold CV Benchmark across OLS, Ridge-GCV, Pure-NumPy GBRT)
    kb_bmap = {b["model"]: b for b in kb_s6["benchmark"]}
    vi_bmap = {b["model"]: b for b in vi_s6["benchmark"]}
    f14_models = ["OLS Baseline", "Ridge (GCV)", "GBRT (Pure-NumPy)"]

    def mut_fig14(traces, layout):
        traces[0]["x"] = f14_models
        traces[0]["y"] = [round(kb_bmap[m]["cv_r2"], 4) for m in f14_models]
        traces[0]["text"] = [f"{kb_bmap[m]['cv_r2']:.3f}" for m in f14_models]
        traces[0]["textposition"] = "auto"
        traces[1]["x"] = f14_models
        traces[1]["y"] = [round(vi_bmap[m]["cv_r2"], 4) for m in f14_models]
        traces[1]["text"] = [f"{vi_bmap[m]['cv_r2']:.3f}" for m in f14_models]
        traces[1]["textposition"] = "auto"
        layout["title"]["text"] = (
            "<b>14. Ábra: GeoSpec Stage 6 Épületklaszteres GroupKFold CV R² Benchmark (v1.0-sealed, Seed=42)</b>"
        )
        layout["yaxis"] = {**(layout.get("yaxis") or {}), "range": [0, 0.7]}

    html = patch_plotly_div(html, "db64bd98-9507-4efd-9152-16ef7a8dfa8b", mut_fig14)
    html = replace_section_between(
        html,
        '        <div class="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700">',
        "\n        </div>\n      </div>\n\n      <!-- 7-MODELLES ML NAGYTÁBLÁZAT -->",
        f"""        <div class="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700">
          <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <h4 class="font-bold text-amber-900 flex items-center gap-1.5 mb-1.5">
              <span>🚀</span> 1. Szivárgásmentes Épületklaszteres Validáció (GeoSpec Stage 6)
            </h4>
            <p class="leading-relaxed text-slate-600">
              A <code>v1.0-sealed</code> adatbázison ({kb_s6['n_obs']} ill. {vi_s6['n_obs']} lakás, 5-fold épületklaszteres GroupKFold CV, fix <code>seed = {kb_s6['random_seed']}</code>) Kőbányán a Pure-NumPy GBRT érte el a legjobb mintán kívüli eredményt (<strong>CV R² = {kb_bmap['GBRT (Pure-NumPy)']['cv_r2']:.4f}</strong> vs. Ridge {kb_bmap['Ridge (GCV)']['cv_r2']:.4f}, in-sample R² = {kb_bmap['GBRT (Pure-NumPy)']['in_sample_r2']:.4f}). Bécsben a nagy teraszos tömbprojektek lineáris extrapolációja miatt a Ridge/OLS vezet (<strong>CV R² = {vi_bmap['Ridge (GCV)']['cv_r2']:.4f}</strong>; GBRT in-sample R² = {vi_bmap['GBRT (Pure-NumPy)']['in_sample_r2']:.4f}).
            </p>
          </div>
          <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <h4 class="font-bold text-slate-900 flex items-center gap-1.5 mb-1.5">
              <span>⚖️</span> 2. Determinisztikus TreeSHAP + Oksági Következtetés
            </h4>
            <p class="leading-relaxed text-slate-600">
              Az egzakt Lundberg TreeSHAP (maximális additív hiba: <strong>{kb_s6['local_accuracy_max_err']:.1e}</strong>) közvetlenül böngészőben, külső szerver nélkül reprodukálható a GeoSpec 6. fülén. A nemlineáris súlyok megerősítik az M1–M4 oksági eredményeket: Kőbányán a fizikai minőség (<strong>{kb_phys_hu}%</strong>), Bécsben a terasz és a 300 m-es csendes sáv (<strong>{vi_trk_hu}%</strong> vágány) dominál.
            </p>
          </div>""",
        "fig14_prose",
        after="db64bd98-9507-4efd-9152-16ef7a8dfa8b",
        max_lines=30,
    )

    # Fig 19 (Sealed Stage 6 Exact TreeSHAP Global Feature Shares — zero rare outliers)
    SHAP_LABEL_MAP = {
        "d_epult_2015_utan": "Új építés (2015 után) dummy",
        "log(alapterulet_nm)": "Alapterület ln(m²)",
        "kulteri_alapterulet_nm": "Kültéri méret (erkély/terasz m²)",
        "ln(allomas_net)": "ln(Vasútállomás gyalogtáv hm)",
        "ln(vagany_eucl)": "ln(Vasúti vágány távolság m)",
        "ln(fout_eucl)": "ln(Főút távolság m)",
        "d_felujitott": "Felújított állapot dummy",
        "d_felujitando": "Felújítandó állapot dummy",
        "d_kulteri_kapcsolat": "Erkély / terasz kapcsolat dummy",
        "d_foldszint": "Földszinti elhelyezkedés dummy",
        "d_zaroszint": "Zárószint / legfelső emelet dummy",
        "d_lift": "Lift jelenléte dummy",
        "szobak_szama": "Szobák száma (db)",
        "d_panel": "Panel építési technológia dummy",
    }
    kb_shap_m = {r["feature"]: r["rel_pct"] for r in kb_s6["feature_importance"]}
    vi_shap_m = {r["feature"]: r["rel_pct"] for r in vi_s6["feature_importance"]}
    all_shap_feats = sorted(
        set(kb_shap_m.keys()) | set(vi_shap_m.keys()),
        key=lambda fn: (kb_shap_m.get(fn, 0.0) + vi_shap_m.get(fn, 0.0)),
    )

    def mut_fig19(traces, layout):
        y_lbls = [SHAP_LABEL_MAP.get(fn, fn) for fn in all_shap_feats]
        traces[0]["name"] = "Budapest X. Kőbánya (Pure-NumPy GBRT)"
        traces[0]["y"] = y_lbls
        traces[0]["x"] = [round(kb_shap_m.get(fn, 0.0), 2) for fn in all_shap_feats]
        traces[1]["name"] = "Bécs II. Leopoldstadt (Pure-NumPy GBRT)"
        traces[1]["y"] = y_lbls
        traces[1]["x"] = [round(vi_shap_m.get(fn, 0.0), 2) for fn in all_shap_feats]
        layout["title"]["text"] = (
            "<b>19. Ábra: Egzakt TreeSHAP Globális Változó-fontosság (GeoSpec Stage 6, v1.0-sealed)</b><br>"
            "<span style='font-size:12px;color:#64748b'>Pure-NumPy GBRT (n_estimators=150, depth=3, seed=42) relatív |SHAP| magyarázóerő (%)</span>"
        )

    html = patch_plotly_div(html, "c3c9ebdb-4826-4f66-8203-2c3ea679120d", mut_fig19)
    html = replace_section_between(
        html,
        '        <div class="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700">',
        "\n        </div>\n      </div>\n\n      <!--",
        f"""        <div class="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700">
          <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <h4 class="font-bold text-purple-900 flex items-center gap-1.5 mb-1.5">
              <span>🎯</span> 1. Értékmeghatározó Attribútumok Eltérő Súlypontja
            </h4>
            <p class="leading-relaxed text-slate-600">
              A lepecsételt Stage 6 TreeSHAP elemzés pontosan visszaigazolja a hedonikus eredményeket: Kőbányán a fizikai jellemzők és állapot adják az ármagyarázat <strong>{kb_phys_hu}%-át</strong> (míg a vágányközelség mindössze <strong>{kb_trk_hu}%</strong>, összes vasút {kb_rail_hu}%). Bécsben az erkély/terasz (<strong>{vi_out_hu}%</strong>) és a vágányközelség (<strong>{vi_trk_hu}%</strong> — közel 5× akkora súly!) határozza meg a különbségeket.
            </p>
          </div>
          <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <h4 class="font-bold text-slate-900 flex items-center gap-1.5 mb-1.5">
              <span>🚂</span> 2. Átjárható Állomásközelség vs. Vágányzaj
            </h4>
            <p class="leading-relaxed text-slate-600">
              A két városrész összevetése rámutat, hogy elöregedett rozsdaövezetben (Kőbánya) a vevő elsősorban épületminőséget vásárol, ezért az állomáselérés súlya (<strong>{kb_sta_hu}%</strong>) megelőzi a vágányzajt ({kb_trk_hu}%). Amint a házak korszerűsödnek (Bécs, 92,6% modern/új állomány), a fizikai felülbírálás eltűnik és előtérbe lép a 300 m-es csendes sáv védelme.
            </p>
          </div>""",
        "tab5_shap_prose",
        after="c3c9ebdb-4826-4f66-8203-2c3ea679120d",
        max_lines=30,
    )

    html = replace_exact(
        html,
        '<h3 class="text-base font-bold text-slate-900">Értéknövekmény-elvonási (LVC) Szcenárió Kalkulátor</h3>',
        '<h3 class="text-base font-bold text-slate-900">Értékmegragadási (Land Value Capture, LVC) Szcenárió Kalkulátor</h3>',
        "fig15_h3",
        optional=True,
    )
    html = replace_exact(
        html,
        "Az állomási TOD prémiumból keletkező magántőke-felértékelődés önkormányzati visszanyerése 3 szakpolitikai adóelvonási szcenárió mentén (Konzervatív 10%, Mérsékelt 25%, Ambiciózus 50%).",
        "A New Wave Residence (6,8 ha) ütemezett fejlesztéséből és a csomóponti átminősítésből származó közösségi értékmegragadás (15%, 20% bázis és 25% LVC-ráta mellett).",
        "fig15_sub",
        optional=True,
    )

    def mut_fig15(traces, layout):
        layout["title"]["text"] = "<b>15. Ábra: Értékmegragadási (Land Value Capture, LVC) Önkormányzati Potenciál</b>"

    html = patch_plotly_div(html, "263da3a2-f4a8-4629-831d-f02a5cf6b3cf", mut_fig15)

    html = replace_section_between(
        html,
        '        <div class="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700">',
        "\n        </div>\n      </div>\n\n    </div></main>",
        """        <div class="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700">
          <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <h4 class="font-bold text-emerald-900 flex items-center gap-1.5 mb-1.5">
              <span>🏛️</span> 1. New Wave Residence (6,8 ha) Üzleti Mérleg és 1,84 Mrd Ft LVC-Alap
            </h4>
            <p class="leading-relaxed text-slate-600">
              A Suzuki et al. (2015, World Bank) módszertanán alapuló, 6,8 hektáros, 3 ütemes modellben (880 lakás + 8 000 m² iroda; 1,56 → 1,68 → 1,79M Ft/m²) <strong>86,5 Mrd Ft</strong> beruházási költség mellett <strong>97,4 Mrd Ft</strong> árbevétel és <strong>10,9 Mrd Ft</strong> fejlesztői eredmény (12,6% költségárrés, 15,8% ROI) realizálódik. Konzervatív, +10%-os csomóponti értéknövekedéssel és 20%-os kulccsal számolva <strong>1,84 Mrd Ft</strong> önfinanszírozó LVC-alap képződik gyalogos aluljáróra, zajvédelemre és közterületi zöldítésre.
            </p>
          </div>
          <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <h4 class="font-bold text-blue-900 flex items-center gap-1.5 mb-1.5">
              <span>🔄</span> 2. Normatív LVC + Jelzálog-alapú Adóhalasztás (Szociális Védelem)
            </h4>
            <p class="leading-relaxed text-slate-600">
              Az eseti Településrendezési Szerződések (TRSZ) helyett a normatív értékmegragadás (Land Value Capture, LVC) kiszámítható hozzájárulást rögzít az építtetői oldalon, miközben a régi, alacsonyabb jövedelmű vagy idős lakosokat a <strong>jelzálog-alapú adóhalasztás</strong> védi a kiszorítódástól (dzsentrifikáció-kontroll): a tulajdonost nem terheli azonnali likviditási kényszer, a hozzájárulás csak későbbi tulajdonosváltáskor vagy örökléskor esedékes.
            </p>
          </div>""",
        "tab5_lvc_prose",
        after="263da3a2-f4a8-4629-831d-f02a5cf6b3cf",
        max_lines=30,
    )

    # 8. RECIPE DROPDOWN TOGGLE JS (IDEMPOTENT)
    dropdown_js = """
    // === GEOSPEC RECIPE DOWNLOAD DROPDOWN ===
    (() => {
      const btn = document.getElementById("recipe-menu-btn");
      const menu = document.getElementById("recipe-dropdown-menu");
      if (!btn || !menu) return;
      function setOpen(open) {
        menu.hidden = !open;
        menu.classList.toggle("hidden", !open);
        btn.setAttribute("aria-expanded", open ? "true" : "false");
      }
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        setOpen(menu.hidden);
      });
      document.addEventListener("click", (e) => {
        if (!menu.hidden && !e.target.closest("#recipe-dropdown-wrap")) {
          setOpen(false);
        }
      });
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !menu.hidden) {
          setOpen(false);
          btn.focus();
        }
      });
    })();
"""
    if "GEOSPEC RECIPE DOWNLOAD DROPDOWN" not in html:
        html = replace_exact(
            html,
            "    // === MÁZSA TÉR ÉRTÉKVÖLGY 3D / SZINTVONAL / METSZET ===",
            dropdown_js + "\n    // === MÁZSA TÉR ÉRTÉKVÖLGY 3D / SZINTVONAL / METSZET ===",
            "inject_dropdown_js",
        )

    # -----------------------------------------------------------------------
    # Guardrail Assertions
    # -----------------------------------------------------------------------
    prose_lines = [line for line in html.splitlines() if len(line) < 1000]
    prose_blob = "\n".join(prose_lines)
    forbidden_prose = [
        "értéknövekmény-visszavon",
        "értéknövekmény-elvonás",
        "Kelenföld–Nyugati",
        "Kelenföld-Nyugati",
        "alagút-zárvány",
        "838 eFt",
        "tranzakciós és kínálati",
        "Szuterén Diszkont",
        "Szuterén szint dummy",
        "1 071 m",
        "1071m",
        "Medián: 143 m",
        "medián: 143 m",
        "0,738",
        "0.738",
        "80,1% vs. 5,0%",
        "34,0% vs. 19,4%",
        "41.3% Spillover",
    ]
    for bad in forbidden_prose:
        if bad.lower() in prose_blob.lower():
            raise AssertionError(f"Guardrail failed: forbidden phrase {bad!r} still present in prose!")

    # Structural integrity checks
    assert len(re.findall(r"Plotly\.newPlot\(", html)) == 23, "Lost a Plotly chart!"
    assert "GEO_KOBANYA" in html and "GEO_LEOPOLDSTADT" in html and "mazsa-stage" in html
    assert "./geospec_input/" not in html, "Found stale ./geospec_input/ link in index.html!"
    assert "v1.0-sealed" in html and "canonical_truth.json" in html, "Missing SSOT seal link/version in index.html!"
    assert len(html) > 1_400_000, f"index.html unexpectedly small: {len(html)}"

    # Verify every local link in header/hero resolves on disk
    for m in re.finditer(r'href="\./([^"?#]+)', html):
        rel = m.group(1)
        target = REPO_DIR / rel
        if not target.exists():
            raise AssertionError(f"Broken relative link in index.html: ./{rel}")

    INDEX_HTML.write_text(html, encoding="utf-8")
    print(f"Successfully updated {INDEX_HTML} ({len(html)} chars).")


if __name__ == "__main__":
    main()
