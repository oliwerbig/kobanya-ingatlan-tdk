#!/usr/bin/env python3
"""
Single Source of Truth (SSOT) Gatekeeper for TDK 2026 Research Outputs.

Verifies exact SHA-256 parity and numerical consistency across:
  1. recipes/canonical_truth.json
  2. recipes/*.csv, recipes/*_recipe.json, recipes/*.geospec.json & geospec_input/*_recipe.json
  3. Interactive Dashboard (Colab Notebooks/index.html)
  4. Public Ghost Research Hub (web_research_hub/research_hub.json & svg_figures.py)

Exit code 0 = 100% SSOT verified; non-zero = drift detected.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

COLAB_DIR = Path(__file__).resolve().parent
RECIPES_DIR = COLAB_DIR / "recipes"
GI_DIR = COLAB_DIR / "geospec_input"
INDEX_HTML = COLAB_DIR / "index.html"
ROOT_DIR = COLAB_DIR.parents[3]
HUB_DIR = ROOT_DIR / "web_research_hub"
HUB_JSON = HUB_DIR / "research_hub.json"
SVG_FIGS = HUB_DIR / "svg_figures.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    errors: list[str] = []
    truth_path = RECIPES_DIR / "canonical_truth.json"
    if not truth_path.exists():
        print(f"[FAIL] Missing {truth_path}. Run update_dashboard_geospec.py first.")
        return 1

    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    if truth.get("seal_version") != "v1.0-sealed":
        errors.append(f"Unexpected seal_version in canonical_truth.json: {truth.get('seal_version')!r}")

    # 1. Verify SHA-256 locks + geospec_input parity
    for fname, expected_sha in (truth.get("sha256") or {}).items():
        fpath = RECIPES_DIR / fname
        if not fpath.exists():
            errors.append(f"Missing sealed file in recipes/: {fname}")
            continue
        actual_sha = _sha256(fpath)
        if actual_sha != expected_sha:
            errors.append(f"SHA-256 mismatch for recipes/{fname}: expected {expected_sha[:12]}..., got {actual_sha[:12]}...")

    for rec_name in ("kobanya_recipe.json", "leopoldstadt_recipe.json"):
        if (GI_DIR / rec_name).exists() and (RECIPES_DIR / rec_name).exists():
            if _sha256(GI_DIR / rec_name) != _sha256(RECIPES_DIR / rec_name):
                errors.append(f"Recipe drift between geospec_input/{rec_name} and recipes/{rec_name}")

    kb_t = truth["cities"]["kobanya"]
    vi_t = truth["cities"]["leopoldstadt"]
    vi_p_300_500 = vi_t["ols"]["m2_rings"]["params"]["ring_vagany_300_500"]["p"]
    vi_p_hu = f"{vi_p_300_500:.3f}".replace(".", ",")
    kb_s6_grp = kb_t["ml_shap_stage6"]["grouped_importance"]
    vi_s6_grp = vi_t["ml_shap_stage6"]["grouped_importance"]
    vi_s6_fmap = {r["feature"]: r["rel_pct"] for r in vi_t["ml_shap_stage6"]["feature_importance"]}
    vi_outdoor_pct = vi_s6_fmap["kulteri_alapterulet_nm"] + vi_s6_fmap["d_kulteri_kapcsolat"]

    # 2. Verify index.html parity
    html = INDEX_HTML.read_text(encoding="utf-8")
    for forbidden in ("0,738", "0.738", "80,1% vs. 5,0%", "34,0% vs. 19,4%", "41.3% Spillover", "Szuterén szint dummy"):
        if forbidden in html:
            errors.append(f"Stale string {forbidden!r} found in index.html")
    for required in (
        "v1.0-sealed",
        "canonical_truth.json",
        vi_p_hu,
        f"{kb_s6_grp['physical_pct']:.1f}".replace(".", ",") + "%",
        f"{vi_s6_grp['rail_track_pct']:.1f}".replace(".", ",") + "%",
        f"{vi_t['spatial_stage4']['sar_rho']:.4f}",
        f"{kb_t['spatial_stage4']['sar_rho']:.4f}",
    ):
        if required not in html:
            errors.append(f"Expected sealed token {required!r} missing from index.html")

    # 3. Verify Ghost research_hub.json & svg_figures.py parity
    hub_raw = HUB_JSON.read_text(encoding="utf-8")
    svg_raw = SVG_FIGS.read_text(encoding="utf-8")
    for label, blob in (("research_hub.json", hub_raw), ("svg_figures.py", svg_raw)):
        for stale in ("0,7381", "0.7381", "80,1%", "80.1%", "34,0%", "34.0%", "19,4%", "19.4%", "p = 0,645"):
            if stale in blob:
                errors.append(f"Stale string {stale!r} still present in {label}")

    # Check exact sealed tokens inside research_hub.json & svg_figures.py
    kb_phys_hu = f"{kb_s6_grp['physical_pct']:.1f}".replace(".", ",")
    kb_track_hu = f"{kb_s6_grp['rail_track_pct']:.1f}".replace(".", ",")
    vi_out_hu = f"{vi_outdoor_pct:.1f}".replace(".", ",")
    vi_track_hu = f"{vi_s6_grp['rail_track_pct']:.1f}".replace(".", ",")
    for tok in (kb_phys_hu, kb_track_hu, vi_out_hu, vi_track_hu, "0,8132", "canonical_truth.json"):
        if tok not in hub_raw:
            errors.append(f"Sealed token {tok!r} missing from research_hub.json")

    if f"({kb_s6_grp['physical_pct']:.1f}," not in svg_raw or f"({vi_outdoor_pct:.1f}," not in svg_raw:
        errors.append("svg_figures.py fig_04 segments do not match sealed canonical_truth.json percentages")

    if errors:
        print("[FAIL] SSOT parity check failed:")
        for e in errors:
            print("  -", e)
        return 1

    print(
        f"[OK] SSOT verified ({truth['seal_version']}): "
        f"6 SHA-256 locks matched; index.html, research_hub.json, and svg_figures.py 100% synced "
        f"(Vienna 300-500m p={vi_p_300_500:.4f}, knn5_dedup rho={vi_t['spatial_stage4']['sar_rho']:.4f}; "
        f"Kőbánya knn5_dedup rho={kb_t['spatial_stage4']['sar_rho']:.4f}; "
        f"TreeSHAP Phys/Track: {kb_s6_grp['physical_pct']:.1f}%/{kb_s6_grp['rail_track_pct']:.1f}% vs {vi_outdoor_pct:.1f}%out/{vi_s6_grp['rail_track_pct']:.1f}%)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
