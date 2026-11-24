"""
Parametric OSM / GeoJSON Spatial GIS & CSR Multi-Source Dijkstra Engine
=======================================================================
Pure NumPy + SciPy implementation (zero osmnx/shapely/networkx):
- Parses Overpass API JSON (node, way, relation multipolygons) & GeoJSON
- Supports per-target interactive map feature exclusion (`excluded_osm_ids`)
- Four exact geometry handlers: point, line_segment, polygon_boundary, polygon_centroid
- Vectorized ray-casting containment (`inside_is_zero`)
- Low-RAM CSR street network + orthogonal edge snapping + virtual super-source Dijkstra
- On-demand single-point Dijkstra route replay (<10 ms) for map inspector
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .projection import LocalMetricProjection

# Explicit snap status codes (no silent 1.25x fallback fabrication)
SNAP_OK = 0
SNAP_DISCONNECTED = 1
SNAP_EXCEEDED = 2
SNAP_TARGET_EMPTY = 3
SNAP_FETCH_FAILED = 4


def _dedup_min_edges(
    rows: Any,
    cols: Any,
    weights: Any,
    n_cols: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate (row, col) edges by keeping the MINIMUM weight instead of summing."""
    r = np.asarray(rows, dtype=np.int64)
    c = np.asarray(cols, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    if len(r) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
    keys = r * int(n_cols) + c
    order = np.lexsort((w, keys))
    keys_s = keys[order]
    keep = np.empty(len(keys_s), dtype=bool)
    keep[0] = True
    keep[1:] = keys_s[1:] != keys_s[:-1]
    sel = order[keep]
    return r[sel].astype(np.int32), c[sel].astype(np.int32), w[sel]


def _dedup_min_csr(
    rows: Any,
    cols: Any,
    weights: Any,
    shape: Tuple[int, int],
) -> csr_matrix:
    """Build CSR matrix while taking minimum weight on duplicate (row, col) edges."""
    r, c, w = _dedup_min_edges(rows, cols, weights, shape[1])
    return csr_matrix((w, (r, c)), shape=shape, dtype=np.float64)


def _shoelace_area_and_centroid(ring: np.ndarray) -> Tuple[float, float, float]:
    """Compute signed area, cx, cy of a closed 2D ring via Green's theorem."""
    if len(ring) < 3:
        return 0.0, float(np.mean(ring[:, 0])), float(np.mean(ring[:, 1]))
    x = ring[:, 0]
    y = ring[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    signed_area = 0.5 * float(np.sum(cross))
    if abs(signed_area) < 1e-15:
        return 0.0, float(np.mean(x)), float(np.mean(y))
    cx = float(np.sum((x + x_next) * cross)) / (6.0 * signed_area)
    cy = float(np.sum((y + y_next) * cross)) / (6.0 * signed_area)
    return abs(signed_area), cx, cy


def _is_osm_area_way(pts: List[Tuple[float, float]], tags: Dict[str, Any]) -> bool:
    """Determine whether a closed OSM way represents an area/polygon vs linear loop."""
    is_closed = len(pts) >= 4 and abs(pts[0][0] - pts[-1][0]) < 1e-6 and abs(pts[0][1] - pts[-1][1]) < 1e-6
    if not is_closed:
        return False
    area_tag = str(tags.get("area", "")).lower()
    if area_tag == "no":
        return False
    if area_tag == "yes":
        return True
    if str(tags.get("junction", "")).lower() == "roundabout":
        return False
    if tags.get("railway") in ("rail", "light_rail", "subway", "tram", "narrow_gauge", "monorail"):
        return False
    if tags.get("highway") in (
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "living_street", "service",
        "pedestrian", "footway", "cycleway", "path", "track",
    ):
        return False
    if tags.get("waterway") in ("river", "canal", "stream", "drain", "ditch") and not tags.get("natural"):
        return False
    if tags.get("barrier"):
        return False
    return True


def _build_polygon_parts(
    outer_rings: List[List[Tuple[float, float]]],
    inner_rings: List[List[Tuple[float, float]]],
) -> List[Dict[str, Any]]:
    """Assign each inner hole ring to its smallest enclosing outer ring (handles island-in-hole)."""
    if not outer_rings:
        return []
    if len(outer_rings) == 1 or not inner_rings:
        return [{"outer": outer_rings[0], "inners": list(inner_rings)}] + [
            {"outer": r, "inners": []} for r in outer_rings[1:]
        ]

    outer_arrs = [np.asarray(r, dtype=np.float64) for r in outer_rings]
    outer_areas = [_shoelace_area_and_centroid(arr)[0] for arr in outer_arrs]
    order_by_area = sorted(range(len(outer_rings)), key=lambda i: outer_areas[i])
    parts: List[Dict[str, Any]] = [{"outer": r, "inners": []} for r in outer_rings]

    for ir in inner_rings:
        ir_arr = np.asarray(ir, dtype=np.float64)
        ir_area, cx, cy = _shoelace_area_and_centroid(ir_arr)
        # Test centroid + edge midpoints so concave rings & shared vertices match accurately
        sample_pts = [[cx, cy]]
        if len(ir_arr) >= 2:
            sample_pts.append(((ir_arr[0] + ir_arr[1]) * 0.5).tolist())
            mid_i = len(ir_arr) // 2
            sample_pts.append(((ir_arr[mid_i] + ir_arr[(mid_i + 1) % len(ir_arr)]) * 0.5).tolist())
        samples_np = np.asarray(sample_pts, dtype=np.float64)

        assigned = 0
        for oi in order_by_area:
            if outer_areas[oi] <= ir_area + 1e-15:
                continue
            if np.any(_points_in_ring(samples_np, outer_arrs[oi])):
                assigned = oi
                break
        parts[assigned]["inners"].append(ir)
    return parts


def _stitch_way_segments(way_coords_list: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
    """Stitch unordered OSM relation outer/inner member coordinate chains into closed rings."""
    if not way_coords_list:
        return []
    remaining = [list(w) for w in way_coords_list if len(w) >= 2]
    rings: List[List[Tuple[float, float]]] = []
    eps = 1e-6

    def _close(p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        return abs(p1[0] - p2[0]) < eps and abs(p1[1] - p2[1]) < eps

    while remaining:
        current = remaining.pop(0)
        changed = True
        while changed and not _close(current[0], current[-1]):
            changed = False
            for idx, seg in enumerate(remaining):
                if _close(current[-1], seg[0]):
                    current.extend(seg[1:])
                    remaining.pop(idx)
                    changed = True
                    break
                if _close(current[-1], seg[-1]):
                    current.extend(list(reversed(seg[:-1])))
                    remaining.pop(idx)
                    changed = True
                    break
                if _close(current[0], seg[-1]):
                    current = seg[:-1] + current
                    remaining.pop(idx)
                    changed = True
                    break
                if _close(current[0], seg[0]):
                    current = list(reversed(seg[1:])) + current
                    remaining.pop(idx)
                    changed = True
                    break
        rings.append(current)
    return rings


def _close_ring_tuples(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(pts) >= 3 and (
        abs(pts[0][0] - pts[-1][0]) > 1e-9 or abs(pts[0][1] - pts[-1][1]) > 1e-9
    ):
        return list(pts) + [pts[0]]
    return list(pts)


def parse_osm_elements(osm_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse Overpass API JSON (`out center;` or `out geom;`) into normalized feature dicts:
    {uid, name, tags, geom_type, rings (list of [(lon, lat), ...]), center (lon, lat)}.
    """
    elements = osm_json.get("elements", [])
    features: List[Dict[str, Any]] = []

    for el in elements:
        el_type = el.get("type")
        el_id = el.get("id")
        if not el_type or el_id is None:
            continue
        eff_type = str(el.get("orig_type") or el_type)
        uid = f"{eff_type}/{el_id}"
        tags = el.get("tags", {}) or {}
        name = tags.get("name") or tags.get("ref") or uid

        if el_type == "node":
            lat = el.get("lat")
            lon = el.get("lon")
            if lat is None or lon is None:
                continue
            features.append({
                "uid": uid,
                "name": name,
                "tags": tags,
                "geom_type": "point",
                "rings": [[(float(lon), float(lat))]],
                "inner_rings": [],
                "center": (float(lon), float(lat)),
            })
        elif el_type == "way":
            geom = el.get("geometry")
            center = el.get("center")
            if geom:
                pts = [(float(p["lon"]), float(p["lat"])) for p in geom if "lat" in p and "lon" in p]
                if len(pts) < 2:
                    continue
                is_area = _is_osm_area_way(pts, tags)
                if is_area:
                    pts = _close_ring_tuples(pts)
                c_lon = float(np.mean([p[0] for p in pts]))
                c_lat = float(np.mean([p[1] for p in pts]))
                if center and "lat" in center and "lon" in center:
                    c_lon, c_lat = float(center["lon"]), float(center["lat"])
                parts = [{"outer": pts, "inners": []}] if is_area else []
                features.append({
                    "uid": uid,
                    "name": name,
                    "tags": tags,
                    "geom_type": "polygon" if is_area else "line",
                    "rings": [pts],
                    "inner_rings": [],
                    "parts": parts,
                    "center": (c_lon, c_lat),
                })
            elif center and "lat" in center and "lon" in center:
                c_lon, c_lat = float(center["lon"]), float(center["lat"])
                features.append({
                    "uid": uid,
                    "name": name,
                    "tags": tags,
                    "geom_type": "point",
                    "rings": [[(c_lon, c_lat)]],
                    "inner_rings": [],
                    "center": (c_lon, c_lat),
                })
        elif el_type == "relation":
            members = el.get("members", [])
            outer_raw: List[List[Tuple[float, float]]] = []
            inner_raw: List[List[Tuple[float, float]]] = []
            point_pts: List[Tuple[float, float]] = []
            for m in members:
                if m.get("type") == "way" and m.get("geometry"):
                    pts = [(float(p["lon"]), float(p["lat"])) for p in m["geometry"] if "lat" in p and "lon" in p]
                    if len(pts) >= 2:
                        if m.get("role") == "inner":
                            inner_raw.append(pts)
                        else:
                            outer_raw.append(pts)
                elif m.get("type") == "node" and "lat" in m and "lon" in m:
                    point_pts.append((float(m["lon"]), float(m["lat"])))

            outer_rings = _stitch_way_segments(outer_raw)
            inner_rings = _stitch_way_segments(inner_raw)
            center = el.get("center")

            if outer_rings:
                is_poly = tags.get("type") in ("multipolygon", "boundary") or any(
                    _is_osm_area_way(r, tags) for r in outer_rings
                )
                if is_poly:
                    outer_rings = [_close_ring_tuples(r) for r in outer_rings]
                    inner_rings = [_close_ring_tuples(r) for r in inner_rings]
                all_pts = [p for r in outer_rings for p in r]
                c_lon = float(np.mean([p[0] for p in all_pts]))
                c_lat = float(np.mean([p[1] for p in all_pts]))
                if center and "lat" in center and "lon" in center:
                    c_lon, c_lat = float(center["lon"]), float(center["lat"])
                parts = _build_polygon_parts(outer_rings, inner_rings) if is_poly else []
                features.append({
                    "uid": uid,
                    "name": name,
                    "tags": tags,
                    "geom_type": "polygon" if is_poly else "line",
                    "rings": outer_rings,
                    "inner_rings": inner_rings,
                    "parts": parts,
                    "center": (c_lon, c_lat),
                })
            elif point_pts:
                c_lon = float(np.mean([p[0] for p in point_pts]))
                c_lat = float(np.mean([p[1] for p in point_pts]))
                features.append({
                    "uid": uid,
                    "name": name,
                    "tags": tags,
                    "geom_type": "point",
                    "rings": [[p] for p in point_pts],
                    "inner_rings": [],
                    "center": (c_lon, c_lat),
                })
            elif center and "lat" in center and "lon" in center:
                c_lon, c_lat = float(center["lon"]), float(center["lat"])
                features.append({
                    "uid": uid,
                    "name": name,
                    "tags": tags,
                    "geom_type": "point",
                    "rings": [[(c_lon, c_lat)]],
                    "inner_rings": [],
                    "center": (c_lon, c_lat),
                })

    return features


def parse_geojson_features(geojson: Dict[str, Any], prefix: str = "geojson") -> List[Dict[str, Any]]:
    """Parse RFC-7946 GeoJSON FeatureCollection / Feature / Geometry (including GeometryCollection) into normalized feature dicts."""
    if geojson.get("type") == "FeatureCollection":
        raw_feats = geojson.get("features", [])
    elif geojson.get("type") == "Feature":
        raw_feats = [geojson]
    else:
        raw_feats = [{"type": "Feature", "properties": {}, "geometry": geojson}]

    features: List[Dict[str, Any]] = []

    def _close_ring_pts(r_raw: List[Any]) -> List[Tuple[float, float]]:
        pts_r = [(float(c[0]), float(c[1])) for c in r_raw]
        return _close_ring_tuples(pts_r)

    def _append_geom(geom: Dict[str, Any], base_uid: str, name: str, props: Dict[str, Any], sub_counter: List[int]) -> None:
        if not geom:
            return
        gtype = geom.get("type")
        if gtype == "GeometryCollection":
            for sub in geom.get("geometries") or []:
                _append_geom(sub, base_uid, name, props, sub_counter)
            return
        coords = geom.get("coordinates")
        if not gtype or coords is None:
            return
        sub_idx = sub_counter[0]
        sub_counter[0] += 1
        uid = base_uid if sub_idx == 0 else f"{base_uid}#{sub_idx}"

        if gtype == "Point":
            pt = (float(coords[0]), float(coords[1]))
            features.append({
                "uid": uid, "name": name, "tags": props, "geom_type": "point",
                "rings": [[pt]], "inner_rings": [], "center": pt,
            })
        elif gtype == "MultiPoint":
            pts = [(float(c[0]), float(c[1])) for c in coords]
            if pts:
                c_lon = float(np.mean([p[0] for p in pts]))
                c_lat = float(np.mean([p[1] for p in pts]))
                features.append({
                    "uid": uid, "name": name, "tags": props, "geom_type": "point",
                    "rings": [[p] for p in pts], "inner_rings": [], "center": (c_lon, c_lat),
                })
        elif gtype == "LineString":
            pts = [(float(c[0]), float(c[1])) for c in coords]
            if len(pts) >= 2:
                features.append({
                    "uid": uid, "name": name, "tags": props, "geom_type": "line",
                    "rings": [pts], "inner_rings": [],
                    "center": (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))),
                })
        elif gtype == "MultiLineString":
            lines = [[(float(c[0]), float(c[1])) for c in line] for line in coords if len(line) >= 2]
            if lines:
                all_p = [p for l in lines for p in l]
                features.append({
                    "uid": uid, "name": name, "tags": props, "geom_type": "line",
                    "rings": lines, "inner_rings": [],
                    "center": (float(np.mean([p[0] for p in all_p])), float(np.mean([p[1] for p in all_p]))),
                })
        elif gtype == "Polygon":
            outer = _close_ring_pts(coords[0])
            inners = [_close_ring_pts(r) for r in coords[1:] if len(r) >= 3]
            if len(outer) >= 4:
                features.append({
                    "uid": uid, "name": name, "tags": props, "geom_type": "polygon",
                    "rings": [outer], "inner_rings": inners,
                    "parts": [{"outer": outer, "inners": inners}],
                    "center": (float(np.mean([p[0] for p in outer])), float(np.mean([p[1] for p in outer]))),
                })
        elif gtype == "MultiPolygon":
            outers, inners, parts = [], [], []
            for poly in coords:
                if poly and len(poly[0]) >= 3:
                    o_ring = _close_ring_pts(poly[0])
                    p_inners = [_close_ring_pts(r) for r in poly[1:] if len(r) >= 3]
                    outers.append(o_ring)
                    inners.extend(p_inners)
                    parts.append({"outer": o_ring, "inners": p_inners})
            if outers:
                all_p = [p for r in outers for p in r]
                features.append({
                    "uid": uid, "name": name, "tags": props, "geom_type": "polygon",
                    "rings": outers, "inner_rings": inners,
                    "parts": parts,
                    "center": (float(np.mean([p[0] for p in all_p])), float(np.mean([p[1] for p in all_p]))),
                })

    for idx, feat in enumerate(raw_feats):
        geom = feat.get("geometry") or {}
        props = feat.get("properties") or {}
        base_uid = str(feat.get("id") or props.get("id") or f"{prefix}/{idx}")
        name = str(props.get("name") or props.get("title") or base_uid)
        _append_geom(geom, base_uid, name, props, [0])

    return features


def features_to_preview_geojson(features: List[Dict[str, Any]], excluded_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build lightweight GeoJSON FeatureCollection for interactive Stage 1 map preview & toggling."""
    excl_set = set(excluded_ids or [])
    out_feats = []
    for f in features:
        gtype = f["geom_type"]
        rings = f["rings"]
        inners = f.get("inner_rings", [])
        if gtype == "point":
            if len(rings) == 1:
                geom = {"type": "Point", "coordinates": list(rings[0][0])}
            else:
                geom = {"type": "MultiPoint", "coordinates": [list(r[0]) for r in rings]}
        elif gtype == "line":
            if len(rings) == 1:
                geom = {"type": "LineString", "coordinates": [list(p) for p in rings[0]]}
            else:
                geom = {"type": "MultiLineString", "coordinates": [[list(p) for p in r] for r in rings]}
        else:
            def _closed_ring(ring_pts: List[Tuple[float, float]]) -> List[List[float]]:
                coords = [list(p) for p in ring_pts]
                if len(coords) >= 3 and (
                    abs(coords[0][0] - coords[-1][0]) > 1e-9 or abs(coords[0][1] - coords[-1][1]) > 1e-9
                ):
                    coords.append(list(coords[0]))
                return coords

            parts = f.get("parts")
            if parts and len(parts) > 1:
                geom = {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [_closed_ring(part["outer"])] + [_closed_ring(ir) for ir in part.get("inners", [])]
                        for part in parts
                    ],
                }
            elif parts and len(parts) == 1:
                coords = [_closed_ring(parts[0]["outer"])] + [_closed_ring(ir) for ir in parts[0].get("inners", [])]
                geom = {"type": "Polygon", "coordinates": coords}
            elif len(rings) == 1:
                coords = [_closed_ring(rings[0])] + [_closed_ring(ir) for ir in inners]
                geom = {"type": "Polygon", "coordinates": coords}
            else:
                geom = {"type": "MultiPolygon", "coordinates": [[_closed_ring(r)] for r in rings]}

        out_feats.append({
            "type": "Feature",
            "id": f["uid"],
            "properties": {
                "uid": f["uid"],
                "name": f["name"],
                "geom_type": gtype,
                "excluded": f["uid"] in excl_set,
                "tags": {k: v for k, v in list((f.get("tags") or {}).items())[:8]},
            },
            "geometry": geom,
        })
    return {"type": "FeatureCollection", "features": out_feats}


def _subdivide_segments(
    rings_xy: List[np.ndarray],
    max_len_m: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert polylines/rings into subdivided segments (seg_a, seg_b) of length <= max_len_m."""
    a_parts: List[np.ndarray] = []
    b_parts: List[np.ndarray] = []

    for pts in rings_xy:
        if len(pts) < 2:
            continue
        p0 = pts[:-1]
        p1 = pts[1:]
        vecs = p1 - p0
        lens = np.hypot(vecs[:, 0], vecs[:, 1])
        valid = lens > 1e-6
        if not np.any(valid):
            continue
        p0 = p0[valid]
        p1 = p1[valid]
        lens = lens[valid]

        short_mask = lens <= max_len_m
        if np.any(short_mask):
            a_parts.append(p0[short_mask])
            b_parts.append(p1[short_mask])

        long_idx = np.where(~short_mask)[0]
        for i in long_idx:
            if not math.isfinite(float(lens[i])):
                continue
            n_sub = min(4000, max(1, int(math.ceil(float(lens[i]) / max_len_m))))
            t = np.linspace(0.0, 1.0, n_sub + 1)[:, None]
            sub_pts = p0[i][None, :] + t * (p1[i] - p0[i])[None, :]
            a_parts.append(sub_pts[:-1])
            b_parts.append(sub_pts[1:])

    if not a_parts:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    return np.vstack(a_parts), np.vstack(b_parts)


def _point_to_segments_nearest(
    pts_xy: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    max_len_m: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact vectorized point-to-segment distance and closest point using midpoint cKDTree
    with triangle-inequality ball guarantee (cannot miss long straight segments).
    """
    n_pts = len(pts_xy)
    if len(seg_a) == 0:
        return np.full(n_pts, np.nan, dtype=np.float64), np.full((n_pts, 2), np.nan, dtype=np.float64)

    mids = 0.5 * (seg_a + seg_b)
    ab = seg_b - seg_a
    ab_len_sq = np.sum(ab * ab, axis=1)
    tree = cKDTree(mids)
    r0, nn_mid = tree.query(pts_xy, k=1)
    radii = r0 + 0.5 * max_len_m + 1e-6

    out_d = np.empty(n_pts, dtype=np.float64)
    out_xy = np.empty((n_pts, 2), dtype=np.float64)
    cand_lists = tree.query_ball_point(pts_xy, radii)
    for i, cands in enumerate(cand_lists):
        if not cands:
            out_d[i] = float(r0[i])
            out_xy[i] = mids[int(nn_mid[i])]
            continue
        idx = np.asarray(cands, dtype=np.int32)
        pa = pts_xy[i][None, :] - seg_a[idx]
        denom = np.maximum(ab_len_sq[idx], 1e-12)
        t = np.clip(np.sum(pa * ab[idx], axis=1) / denom, 0.0, 1.0)
        proj = seg_a[idx] + t[:, None] * ab[idx]
        dists = np.hypot(pts_xy[i, 0] - proj[:, 0], pts_xy[i, 1] - proj[:, 1])
        best = int(np.argmin(dists))
        out_d[i] = float(dists[best])
        out_xy[i] = proj[best]
    return out_d, out_xy


def _point_to_segments_distance(
    pts_xy: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    max_len_m: float = 50.0,
) -> np.ndarray:
    """Exact vectorized point-to-segment distance."""
    d, _ = _point_to_segments_nearest(pts_xy, seg_a, seg_b, max_len_m=max_len_m)
    return d


def _points_in_ring(
    pts_xy: np.ndarray,
    ring_xy: np.ndarray,
    pt_chunk: int = 512,
    edge_chunk: int = 2048,
) -> np.ndarray:
    """Chunked vectorized even-odd ray-casting point-in-polygon check with bbox prefilter."""
    n = len(pts_xy)
    if len(ring_xy) < 3 or n == 0:
        return np.zeros(n, dtype=bool)
    min_x, min_y = np.min(ring_xy, axis=0)
    max_x, max_y = np.max(ring_xy, axis=0)
    in_box = (
        (pts_xy[:, 0] >= min_x)
        & (pts_xy[:, 0] <= max_x)
        & (pts_xy[:, 1] >= min_y)
        & (pts_xy[:, 1] <= max_y)
    )
    if not np.any(in_box):
        return np.zeros(n, dtype=bool)

    box_idx = np.where(in_box)[0]
    pts_in = pts_xy[box_idx]
    m_pts = len(pts_in)

    rx1 = ring_xy[:, 0]
    ry1 = ring_xy[:, 1]
    rx2 = np.roll(rx1, -1)
    ry2 = np.roll(ry1, -1)
    n_edges = len(rx1)

    crossings_total = np.zeros(m_pts, dtype=np.int32)
    for ps in range(0, m_pts, pt_chunk):
        pe = min(m_pts, ps + pt_chunk)
        px = pts_in[ps:pe, 0][:, None]
        py = pts_in[ps:pe, 1][:, None]
        c_sum = np.zeros(pe - ps, dtype=np.int32)
        for es in range(0, n_edges, edge_chunk):
            ee = min(n_edges, es + edge_chunk)
            x1 = rx1[es:ee][None, :]
            y1 = ry1[es:ee][None, :]
            x2 = rx2[es:ee][None, :]
            y2 = ry2[es:ee][None, :]
            cond_y = (y1 > py) != (y2 > py)
            dy = y2 - y1
            x_intersect = (x2 - x1) * (py - y1) / np.where(np.abs(dy) > 1e-12, dy, 1e-12) + x1
            c_sum += np.sum(cond_y & (px < x_intersect), axis=1, dtype=np.int32)
        crossings_total[ps:pe] = c_sum

    res = np.zeros(n, dtype=bool)
    res[box_idx] = (crossings_total % 2) == 1
    return res


def _inside_polygons_mask(pts_xy: np.ndarray, proj_feats: List[Dict[str, Any]]) -> np.ndarray:
    """Check containment across polygon parts (supports MultiPolygon island-in-hole geometry)."""
    n_pts = len(pts_xy)
    inside = np.zeros(n_pts, dtype=bool)
    if n_pts == 0:
        return inside
    for f in proj_feats:
        if f["geom_type"] != "polygon":
            continue
        parts = f.get("parts_xy")
        if parts:
            for part in parts:
                in_part = _points_in_ring(pts_xy, part["outer"])
                if np.any(in_part):
                    for ir in part.get("inners", []):
                        in_part &= ~_points_in_ring(pts_xy, ir)
                    inside |= in_part
        else:
            in_outer = np.zeros(n_pts, dtype=bool)
            for r_xy in f["outer_xy"]:
                in_outer |= _points_in_ring(pts_xy, r_xy)
            in_inner = np.zeros(n_pts, dtype=bool)
            for r_xy in f.get("inner_xy", []):
                in_inner |= _points_in_ring(pts_xy, r_xy)
            inside |= in_outer & ~in_inner
    return inside


def _project_feature_rings(
    features: List[Dict[str, Any]],
    proj: LocalMetricProjection,
) -> List[Dict[str, Any]]:
    """Project all normalized features once onto local metric coordinates."""
    projected = []
    for f in features:
        outer_xy = []
        for r in f["rings"]:
            arr = np.asarray(r, dtype=np.float64)
            x, y = proj.forward(arr[:, 1], arr[:, 0])
            outer_xy.append(np.column_stack([x, y]))
        inner_xy = []
        for r in f.get("inner_rings", []):
            arr = np.asarray(r, dtype=np.float64)
            x, y = proj.forward(arr[:, 1], arr[:, 0])
            inner_xy.append(np.column_stack([x, y]))
        parts_xy = []
        for part in f.get("parts", []):
            o_arr = np.asarray(part["outer"], dtype=np.float64)
            ox, oy = proj.forward(o_arr[:, 1], o_arr[:, 0])
            i_list = []
            for ir in part.get("inners", []):
                ir_arr = np.asarray(ir, dtype=np.float64)
                ix, iy = proj.forward(ir_arr[:, 1], ir_arr[:, 0])
                i_list.append(np.column_stack([ix, iy]))
            parts_xy.append({"outer": np.column_stack([ox, oy]), "inners": i_list})
        cx, cy = proj.forward(f["center"][1], f["center"][0])
        projected.append({
            **f,
            "outer_xy": outer_xy,
            "inner_xy": inner_xy,
            "parts_xy": parts_xy,
            "center_xy": (float(cx), float(cy)),
        })
    return projected


def _edge_speed_kmh(tags: Dict[str, Any], routing_profile: str) -> float:
    """Compute hierarchy- & maxspeed-calibrated travel speed (km/h) for travel_time_min impedance."""
    hwy = str(tags.get("highway", "")).lower().strip()
    if routing_profile == "walk":
        if hwy == "steps":
            return 2.5
        if hwy in ("primary", "trunk", "primary_link", "trunk_link"):
            return 3.6
        if hwy in ("track", "path"):
            return 3.9
        return 4.5
    if routing_profile == "bike":
        if hwy == "steps":
            return 5.0
        if hwy in ("cycleway",):
            return 15.8
        if hwy in ("track", "path"):
            return 11.1
        if hwy in ("primary", "primary_link"):
            return 12.5
        return 15.0

    # drive profile
    raw_ms = str(tags.get("maxspeed", "")).lower().strip()
    if raw_ms:
        m = re.match(r"^(\d+(?:\.\d+)?)\s*(mph)?", raw_ms)
        if m:
            spd = float(m.group(1))
            if m.group(2) == "mph":
                spd *= 1.60934
            return max(5.0, min(130.0, spd))
    drive_defaults = {
        "motorway": 110.0,
        "motorway_link": 60.0,
        "trunk": 90.0,
        "trunk_link": 50.0,
        "primary": 60.0,
        "primary_link": 45.0,
        "secondary": 50.0,
        "secondary_link": 40.0,
        "tertiary": 45.0,
        "tertiary_link": 35.0,
        "unclassified": 40.0,
        "residential": 30.0,
        "living_street": 20.0,
        "service": 15.0,
    }
    return drive_defaults.get(hwy, 40.0)


def build_street_graph(
    overpass_graph_json: Dict[str, Any],
    proj: LocalMetricProjection,
    routing_profile: str = "walk",
) -> Dict[str, Any]:
    """
    Build low-RAM CSR street network from Overpass `out body;` / `out skel qt;` JSON.
    Supports exact edge snapping, hierarchy-calibrated travel_time_min, and contraflow bike routing.
    """
    elements = overpass_graph_json.get("elements", [])
    node_map: Dict[Any, Tuple[float, float]] = {}
    ways: List[Dict[str, Any]] = []

    for el in elements:
        etype = el.get("type")
        if etype == "node" and "lat" in el and "lon" in el:
            node_map[int(el["id"])] = (float(el["lon"]), float(el["lat"]))
        elif etype == "way":
            if el.get("nodes"):
                ways.append(el)
            elif el.get("geometry"):
                tags_g = el.get("tags", {}) or {}
                sep = f"{tags_g.get('layer', '0')}|{tags_g.get('bridge', '')}|{tags_g.get('tunnel', '')}"
                fake_ids = []
                for pt in el["geometry"]:
                    nid = f"{round(float(pt['lon']), 7):.7f},{round(float(pt['lat']), 7):.7f}|{sep}"
                    node_map[nid] = (float(pt["lon"]), float(pt["lat"]))
                    fake_ids.append(nid)
                ways.append({"type": "way", "id": el["id"], "nodes": fake_ids, "tags": tags_g})

    if not node_map or not ways:
        return {"empty": True, "n_nodes": 0, "n_edges": 0}

    # Keep only nodes referenced by highway ways
    used_ids: Dict[Any, int] = {}
    lon_list: List[float] = []
    lat_list: List[float] = []

    def _get_idx(nid: Any) -> Optional[int]:
        if nid in used_ids:
            return used_ids[nid]
        coord = node_map.get(nid)
        if coord is None:
            return None
        idx = len(lon_list)
        used_ids[nid] = idx
        lon_list.append(coord[0])
        lat_list.append(coord[1])
        return idx

    raw_u: List[int] = []
    raw_v: List[int] = []
    raw_dir: List[int] = []  # 0=both, 1=forward u->v, -1=reverse v->u
    raw_speed: List[float] = []
    raw_abutment: List[bool] = []

    for w in ways:
        nids = w.get("nodes", [])
        if len(nids) < 2:
            continue
        is_abutment = w.get("role") == "abutment"
        tags = w.get("tags", {}) or {}
        oneway_val = str(tags.get("oneway", "")).lower().strip()
        junction_val = str(tags.get("junction", "")).lower().strip()
        hwy_val = str(tags.get("highway", "")).lower().strip()
        bike_ow_val = str(tags.get("oneway:bicycle", "")).lower().strip()
        cyc_vals = [
            str(tags.get(ck, "")).lower().strip()
            for ck in ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both")
        ]
        has_opposite_cycleway = any("opposite" in cv for cv in cyc_vals)

        if is_abutment or routing_profile == "walk":
            w_dir = 0
        else:
            if oneway_val in ("yes", "true", "1"):
                w_dir = 1
            elif oneway_val in ("-1", "reverse"):
                w_dir = -1
            elif oneway_val in ("no", "false", "0"):
                w_dir = 0
            elif junction_val == "roundabout" or hwy_val in ("motorway", "motorway_link"):
                w_dir = 1
            else:
                w_dir = 0

            if routing_profile == "bike":
                if bike_ow_val in ("no", "false", "0") or has_opposite_cycleway:
                    w_dir = 0
                elif bike_ow_val in ("yes", "true", "1"):
                    w_dir = 1
                elif bike_ow_val in ("-1", "reverse"):
                    w_dir = -1

        spd_kmh = _edge_speed_kmh(tags, routing_profile)
        first_id = nids[0] if isinstance(nids[0], str) else int(nids[0])
        prev = _get_idx(first_id)
        for nid_raw in nids[1:]:
            nid = nid_raw if isinstance(nid_raw, str) else int(nid_raw)
            curr = _get_idx(nid)
            if prev is not None and curr is not None and prev != curr:
                raw_u.append(prev)
                raw_v.append(curr)
                raw_dir.append(w_dir)
                raw_speed.append(spd_kmh)
                raw_abutment.append(is_abutment)
            prev = curr

    if not raw_u:
        return {"empty": True, "n_nodes": 0, "n_edges": 0}

    node_lon = np.asarray(lon_list, dtype=np.float64)
    node_lat = np.asarray(lat_list, dtype=np.float64)
    nx, ny = proj.forward(node_lat, node_lon)
    node_xy = np.column_stack([nx, ny])

    u_arr = np.asarray(raw_u, dtype=np.int32)
    v_arr = np.asarray(raw_v, dtype=np.int32)
    dir_arr = np.asarray(raw_dir, dtype=np.int8)
    spd_arr = np.asarray(raw_speed, dtype=np.float64)
    ab_arr = np.asarray(raw_abutment, dtype=bool)
    edge_len = np.hypot(node_xy[v_arr, 0] - node_xy[u_arr, 0], node_xy[v_arr, 1] - node_xy[u_arr, 1])
    valid_e = (edge_len > 1e-4) | ab_arr
    u_arr = u_arr[valid_e]
    v_arr = v_arr[valid_e]
    dir_arr = dir_arr[valid_e]
    spd_arr = spd_arr[valid_e]
    edge_len = np.where(ab_arr[valid_e], np.maximum(edge_len[valid_e], 1e-6), edge_len[valid_e])
    edge_time_min = edge_len / (spd_arr * (1000.0 / 60.0))

    if len(u_arr) == 0:
        return {"empty": True, "n_nodes": 0, "n_edges": 0}

    n_nodes = len(node_xy)
    rev_rows: List[int] = []
    rev_cols: List[int] = []
    rev_weights: List[float] = []
    rev_times: List[float] = []
    fwd_rows: List[int] = []
    fwd_cols: List[int] = []
    fwd_weights: List[float] = []
    fwd_times: List[float] = []

    for u, v, d, length, tmin in zip(u_arr, v_arr, dir_arr, edge_len, edge_time_min):
        ui, vi, w, tm = int(u), int(v), float(length), float(tmin)
        if d == 0 or d == 1:
            fwd_rows.append(ui); fwd_cols.append(vi); fwd_weights.append(w); fwd_times.append(tm)
            rev_rows.append(vi); rev_cols.append(ui); rev_weights.append(w); rev_times.append(tm)
        if d == 0 or d == -1:
            fwd_rows.append(vi); fwd_cols.append(ui); fwd_weights.append(w); fwd_times.append(tm)
            rev_rows.append(ui); rev_cols.append(vi); rev_weights.append(w); rev_times.append(tm)

    g_rev = _dedup_min_csr(rev_rows, rev_cols, rev_weights, shape=(n_nodes, n_nodes))
    g_fwd = _dedup_min_csr(fwd_rows, fwd_cols, fwd_weights, shape=(n_nodes, n_nodes))
    g_time_rev = _dedup_min_csr(rev_rows, rev_cols, rev_times, shape=(n_nodes, n_nodes))
    g_time_fwd = _dedup_min_csr(fwd_rows, fwd_cols, fwd_times, shape=(n_nodes, n_nodes))

    # Subdivide long street edges to max 50 m for exact orthogonal edge snapping
    sub_a_list: List[np.ndarray] = []
    sub_b_list: List[np.ndarray] = []
    sub_edge_idx: List[int] = []
    sub_t0: List[float] = []
    sub_t1: List[float] = []
    max_len_m = 50.0

    for e_i, (u, v, L) in enumerate(zip(u_arr, v_arr, edge_len)):
        pa = node_xy[u]
        pb = node_xy[v]
        if L <= max_len_m:
            sub_a_list.append(pa[None, :])
            sub_b_list.append(pb[None, :])
            sub_edge_idx.append(e_i)
            sub_t0.append(0.0)
            sub_t1.append(1.0)
        else:
            k = int(math.ceil(L / max_len_m))
            ts = np.linspace(0.0, 1.0, k + 1)
            pts = pa[None, :] + ts[:, None] * (pb - pa)[None, :]
            sub_a_list.append(pts[:-1])
            sub_b_list.append(pts[1:])
            sub_edge_idx.extend([e_i] * k)
            sub_t0.extend(ts[:-1].tolist())
            sub_t1.extend(ts[1:].tolist())

    sub_a = np.vstack(sub_a_list)
    sub_b = np.vstack(sub_b_list)
    sub_mids = 0.5 * (sub_a + sub_b)
    edge_tree = cKDTree(sub_mids)

    return {
        "empty": False,
        "routing_profile": routing_profile,
        "n_nodes": n_nodes,
        "n_edges": len(u_arr),
        "node_xy": node_xy,
        "node_lon": node_lon,
        "node_lat": node_lat,
        "edge_u": u_arr,
        "edge_v": v_arr,
        "edge_dir": dir_arr,
        "edge_len": edge_len,
        "edge_speed_kmh": spd_arr,
        "edge_time_min": edge_time_min,
        "g_rev": g_rev,
        "g_fwd": g_fwd,
        "g_time_rev": g_time_rev,
        "g_time_fwd": g_time_fwd,
        "sub_a": sub_a,
        "sub_b": sub_b,
        "sub_edge_idx": np.asarray(sub_edge_idx, dtype=np.int32),
        "sub_t0": np.asarray(sub_t0, dtype=np.float64),
        "sub_t1": np.asarray(sub_t1, dtype=np.float64),
        "edge_tree": edge_tree,
        "max_seg_len": max_len_m,
    }


def _snap_points_to_edges(
    pts_xy: np.ndarray,
    graph: Dict[str, Any],
    max_snap_distance_m: float,
) -> Dict[str, np.ndarray]:
    """Orthogonally snap points (N, 2) onto nearest street graph edge within max_snap_distance_m."""
    n = len(pts_xy)
    edge_idx = np.full(n, -1, dtype=np.int32)
    edge_s = np.zeros(n, dtype=np.float64)
    snap_d = np.full(n, np.nan, dtype=np.float64)
    snap_xy = np.full((n, 2), np.nan, dtype=np.float64)

    if graph.get("empty", True) or n == 0:
        return {"edge_idx": edge_idx, "s": edge_s, "snap_d": snap_d, "snap_xy": snap_xy}

    tree: cKDTree = graph["edge_tree"]
    sub_a = graph["sub_a"]
    sub_b = graph["sub_b"]
    sub_ab = sub_b - sub_a
    sub_len_sq = np.maximum(np.sum(sub_ab * sub_ab, axis=1), 1e-12)
    sub_eidx = graph["sub_edge_idx"]
    sub_t0 = graph["sub_t0"]
    sub_t1 = graph["sub_t1"]
    max_seg = graph["max_seg_len"]

    r0, _ = tree.query(pts_xy, k=1)
    search_r = np.minimum(r0 + 0.5 * max_seg + 1e-6, max_snap_distance_m + 0.5 * max_seg + 1e-6)
    cands_all = tree.query_ball_point(pts_xy, search_r)

    for i, cands in enumerate(cands_all):
        if not cands:
            continue
        idx = np.asarray(cands, dtype=np.int32)
        pa = pts_xy[i][None, :] - sub_a[idx]
        t_local = np.clip(np.sum(pa * sub_ab[idx], axis=1) / sub_len_sq[idx], 0.0, 1.0)
        proj_pts = sub_a[idx] + t_local[:, None] * sub_ab[idx]
        dists = np.hypot(pts_xy[i, 0] - proj_pts[:, 0], pts_xy[i, 1] - proj_pts[:, 1])
        best_pos = int(np.argmin(dists))
        best_dist = float(dists[best_pos])
        if best_dist <= max_snap_distance_m:
            sub_id = int(idx[best_pos])
            edge_idx[i] = int(sub_eidx[sub_id])
            edge_s[i] = float(sub_t0[sub_id] + t_local[best_pos] * (sub_t1[sub_id] - sub_t0[sub_id]))
            snap_d[i] = best_dist
            snap_xy[i] = proj_pts[best_pos]

    return {"edge_idx": edge_idx, "s": edge_s, "snap_d": snap_d, "snap_xy": snap_xy}


def _collect_target_access_points(
    projected_features: List[Dict[str, Any]],
    geometry_handling: str,
    densify_step_m: float = 25.0,
    street_graph: Optional[Dict[str, Any]] = None,
    max_snap_m: float = 300.0,
) -> np.ndarray:
    """
    Collect target point coordinates for Dijkstra snapping with per-ring endpoints,
    plus exact street-vertex orthogonal projections and 2D target-street crossing points.
    Unified across Stage 1 batch distances and on-demand single-route replay.
    """
    pts: List[Tuple[float, float]] = []
    seg_a_list: List[np.ndarray] = []
    seg_b_list: List[np.ndarray] = []

    for f in projected_features:
        if geometry_handling == "polygon_centroid":
            if f["geom_type"] == "polygon":
                total_a = 0.0
                wx = 0.0
                wy = 0.0
                for r_xy in f["outer_xy"]:
                    a, cx, cy = _shoelace_area_and_centroid(r_xy)
                    total_a += a
                    wx += a * cx
                    wy += a * cy
                for r_xy in f.get("inner_xy", []):
                    a, cx, cy = _shoelace_area_and_centroid(r_xy)
                    total_a -= a
                    wx -= a * cx
                    wy -= a * cy
                pts.append((wx / total_a, wy / total_a) if total_a > 1e-6 else f["center_xy"])
            else:
                pts.append(f["center_xy"])
        elif geometry_handling == "point":
            if f["geom_type"] == "point":
                for r_xy in f["outer_xy"]:
                    pts.append((float(r_xy[0, 0]), float(r_xy[0, 1])))
            else:
                pts.append(f["center_xy"])
        else:
            all_rings = list(f["outer_xy"]) + list(f.get("inner_xy", []))
            if f["geom_type"] == "point":
                for r_xy in f["outer_xy"]:
                    pts.append((float(r_xy[0, 0]), float(r_xy[0, 1])))
            elif all_rings:
                had_seg = False
                for r_xy in all_rings:
                    sa_r, sb_r = _subdivide_segments([r_xy], max_len_m=densify_step_m)
                    if len(sa_r) > 0:
                        had_seg = True
                        seg_a_list.append(sa_r)
                        seg_b_list.append(sb_r)
                        pts.extend(map(tuple, sa_r.tolist()))
                        pts.append(tuple(sb_r[-1].tolist()))
                if not had_seg:
                    pts.append(f["center_xy"])
            else:
                pts.append(f["center_xy"])

    # Exact segment-to-graph snapping (node projections + crossing intersections)
    if (
        seg_a_list
        and street_graph is not None
        and not street_graph.get("empty", True)
        and geometry_handling in ("line_segment", "polygon_boundary")
    ):
        sa_all = np.vstack(seg_a_list)
        sb_all = np.vstack(seg_b_list)
        sab = sb_all - sa_all
        slen_sq = np.maximum(np.sum(sab * sab, axis=1), 1e-12)
        seg_mids = 0.5 * (sa_all + sb_all)
        seg_tree = cKDTree(seg_mids)

        # 1) Orthogonally project street nodes within range onto target segments
        node_xy = street_graph["node_xy"]
        node_r, node_nn = seg_tree.query(node_xy, k=1)
        cand_nodes = np.where(node_r <= (max_snap_m + densify_step_m))[0]
        if len(cand_nodes) > 0:
            cands_per_node = seg_tree.query_ball_point(node_xy[cand_nodes], r=max_snap_m + densify_step_m)
            for ni, s_cands in zip(cand_nodes, cands_per_node):
                if not s_cands:
                    continue
                s_idx = np.asarray(s_cands, dtype=np.int32)
                pa = node_xy[ni][None, :] - sa_all[s_idx]
                t_loc = np.clip(np.sum(pa * sab[s_idx], axis=1) / slen_sq[s_idx], 0.0, 1.0)
                proj_pts = sa_all[s_idx] + t_loc[:, None] * sab[s_idx]
                dists = np.hypot(node_xy[ni, 0] - proj_pts[:, 0], node_xy[ni, 1] - proj_pts[:, 1])
                best_k = int(np.argmin(dists))
                if dists[best_k] <= max_snap_m:
                    pts.append((float(proj_pts[best_k, 0]), float(proj_pts[best_k, 1])))

        # 2) Exact 2D intersections between target segments and nearby street sub-edges
        st_tree: cKDTree = street_graph["edge_tree"]
        st_a = street_graph["sub_a"]
        st_b = street_graph["sub_b"]
        st_ab = st_b - st_a
        cross_r = 0.5 * densify_step_m + 0.5 * float(street_graph["max_seg_len"]) + 1e-3
        hit_lists = st_tree.query_ball_point(seg_mids, r=cross_r)
        for ti, st_cands in enumerate(hit_lists):
            if not st_cands:
                continue
            p1 = sa_all[ti]
            d1 = sab[ti]
            q1 = st_a[st_cands]
            d2 = st_ab[st_cands]
            det = d1[0] * (-d2[:, 1]) - d1[1] * (-d2[:, 0])
            nz = np.abs(det) > 1e-9
            if not np.any(nz):
                continue
            rhs = q1[nz] - p1[None, :]
            det_z = det[nz]
            d2_z = d2[nz]
            t_param = (rhs[:, 0] * (-d2_z[:, 1]) - rhs[:, 1] * (-d2_z[:, 0])) / det_z
            u_param = (d1[0] * rhs[:, 1] - d1[1] * rhs[:, 0]) / det_z
            hit_m = (t_param >= -1e-6) & (t_param <= 1.0 + 1e-6) & (u_param >= -1e-6) & (u_param <= 1.0 + 1e-6)
            for tp in t_param[hit_m]:
                tc = float(max(0.0, min(1.0, tp)))
                pts.append((float(p1[0] + tc * d1[0]), float(p1[1] + tc * d1[1])))

    if not pts:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def _profile_base_m_per_min(routing_profile: str) -> float:
    if routing_profile == "drive":
        return 25.0 * 1000.0 / 60.0
    if routing_profile == "bike":
        return 15.0 * 1000.0 / 60.0
    return 4.5 * 1000.0 / 60.0


def compute_pairwise_network_distances(
    obs_xy: np.ndarray,
    valid_coord_mask: np.ndarray,
    street_graph: Optional[Dict[str, Any]],
    cutoff_m: float = 1500.0,
    max_snap_m: float = 300.0,
) -> Optional[csr_matrix]:
    """
    Build bounded-radius (cutoff_m) pairwise street-network distance CSR matrix across valid observations
    for barrier-aware Stage 4 W_net (GIS Roadmap #3). Groups co-located observations to unique nodes for speed.
    """
    if street_graph is None or street_graph.get("empty", True):
        return None

    pts_v = obs_xy[valid_coord_mask]
    n_v = len(pts_v)
    if n_v < 2:
        return csr_matrix((n_v, n_v), dtype=np.float64)

    obs_snap = _snap_points_to_edges(pts_v, street_graph, max_snap_m)
    edge_u = street_graph["edge_u"]
    edge_v = street_graph["edge_v"]
    edge_len = street_graph["edge_len"]
    edge_dir = street_graph["edge_dir"]
    g_fwd: csr_matrix = street_graph["g_fwd"]
    directed = street_graph["routing_profile"] != "walk"

    # Unique observation spatial groups (round to 0.1m so stacked condos compute Dijkstra once!)
    loc_keys: Dict[Tuple[int, int], int] = {}
    grp_of_obs = np.empty(n_v, dtype=np.int32)
    grp_rep_obs: List[int] = []
    obs_by_grp: List[List[int]] = []
    for i in range(n_v):
        if int(obs_snap["edge_idx"][i]) < 0:
            grp_of_obs[i] = -1
            continue
        k = (int(round(float(pts_v[i, 0]) * 10.0)), int(round(float(pts_v[i, 1]) * 10.0)))
        gid = loc_keys.get(k)
        if gid is None:
            gid = len(grp_rep_obs)
            loc_keys[k] = gid
            grp_rep_obs.append(i)
            obs_by_grp.append([i])
        else:
            obs_by_grp[gid].append(i)
        grp_of_obs[i] = gid

    n_grp = len(grp_rep_obs)
    if n_grp == 0:
        return csr_matrix((n_v, n_v), dtype=np.float64)

    # Pre-collect exit points on destination street nodes for fast batch lookup
    max_edge_L = float(np.max(edge_len)) if len(edge_len) > 0 else 50.0
    query_limit = float(cutoff_m + max_edge_L + max_snap_m)

    # Entry & exit node info per unique spatial group
    grp_entry: List[List[Tuple[int, float]]] = [[] for _ in range(n_grp)]
    grp_exit_by_node: Dict[int, List[Tuple[int, float]]] = {}
    grp_edge_info: List[Tuple[int, float, float, int]] = []
    grps_on_edge: Dict[int, List[int]] = {}

    unique_entry_nodes: set = set()
    for gid, rep_i in enumerate(grp_rep_obs):
        e_i = int(obs_snap["edge_idx"][rep_i])
        s_i = float(obs_snap["s"][rep_i])
        ds_i = float(obs_snap["snap_d"][rep_i])
        u_i = int(edge_u[e_i])
        v_i = int(edge_v[e_i])
        L_i = float(edge_len[e_i])
        d_dir = int(edge_dir[e_i])
        grp_edge_info.append((e_i, s_i, ds_i, d_dir))
        grps_on_edge.setdefault(e_i, []).append(gid)

        # Forward exit from obs i -> entry_node
        if d_dir == 0 or d_dir == -1:
            grp_entry[gid].append((u_i, ds_i + s_i * L_i))
            unique_entry_nodes.add(u_i)
        if d_dir == 0 or d_dir == 1:
            grp_entry[gid].append((v_i, ds_i + (1.0 - s_i) * L_i))
            unique_entry_nodes.add(v_i)

        # Entry into dest grp j <- exit_node
        if d_dir == 0 or d_dir == 1:
            grp_exit_by_node.setdefault(u_i, []).append((gid, ds_i + s_i * L_i))
        if d_dir == 0 or d_dir == -1:
            grp_exit_by_node.setdefault(v_i, []).append((gid, ds_i + (1.0 - s_i) * L_i))

    entry_list = sorted(unique_entry_nodes)
    node_to_row = {nid: r for r, nid in enumerate(entry_list)}
    dist_mat = dijkstra(g_fwd, directed=directed, indices=entry_list, limit=query_limit)
    if len(entry_list) == 1:
        dist_mat = dist_mat[None, :]

    exit_nodes_arr = np.asarray(sorted(grp_exit_by_node.keys()), dtype=np.int32)
    rows_out: List[int] = []
    cols_out: List[int] = []
    vals_out: List[float] = []

    for g1 in range(n_grp):
        best_g: Dict[int, float] = {}
        for u_node, d_ent in grp_entry[g1]:
            if d_ent > cutoff_m:
                continue
            r_u = node_to_row[u_node]
            d_row = dist_mat[r_u, exit_nodes_arr]
            reach_mask = np.isfinite(d_row) & ((d_ent + d_row) <= cutoff_m)
            for x_node, dg in zip(exit_nodes_arr[reach_mask], d_row[reach_mask]):
                base_d = d_ent + float(dg)
                for g2, d_ex in grp_exit_by_node[int(x_node)]:
                    if g2 == g1:
                        continue
                    cand_d = base_d + d_ex
                    if cand_d <= cutoff_m and (g2 not in best_g or cand_d < best_g[g2]):
                        best_g[g2] = cand_d

        # Same-edge direct pairs (no intermediate intersection traversal required)
        e1, s1, ds1, dir1 = grp_edge_info[g1]
        L1 = float(edge_len[e1])
        for g2 in grps_on_edge.get(e1, []):
            if g2 == g1:
                continue
            _, s2, ds2, _ = grp_edge_info[g2]
            if dir1 == 0 or (dir1 == 1 and s2 >= s1) or (dir1 == -1 and s2 <= s1):
                cand_d = ds1 + abs(s2 - s1) * L1 + ds2
                if cand_d <= cutoff_m and (g2 not in best_g or cand_d < best_g[g2]):
                    best_g[g2] = cand_d

        # Same co-located group pairs -> 0.0m distance
        obs1_list = obs_by_grp[g1]
        if len(obs1_list) > 1:
            for oi in obs1_list:
                for oj in obs1_list:
                    if oi != oj:
                        rows_out.append(oi); cols_out.append(oj); vals_out.append(0.0)

        for g2, dg12 in best_g.items():
            obs2_list = obs_by_grp[g2]
            for oi in obs1_list:
                for oj in obs2_list:
                    rows_out.append(oi); cols_out.append(oj); vals_out.append(float(dg12))

    return csr_matrix((vals_out, (rows_out, cols_out)), shape=(n_v, n_v), dtype=np.float64)


def compute_target_distances(
    obs_xy: np.ndarray,
    valid_coord_mask: np.ndarray,
    features: List[Dict[str, Any]],
    target_cfg: Dict[str, Any],
    proj: LocalMetricProjection,
    street_graph: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute Euclidean and/or Network Dijkstra distances, travel_time_min, and circuity for a single spatial target.
    Respects `excluded_osm_ids` toggled by the user in Stage 1 map preview.
    """
    n_total = len(obs_xy)
    excl = set(target_cfg.get("excluded_osm_ids", []) or [])
    kept_features = [f for f in features if f["uid"] not in excl]

    dist_eucl = np.full(n_total, np.nan, dtype=np.float64)
    dist_net = np.full(n_total, np.nan, dtype=np.float64)
    travel_time_min = np.full(n_total, np.nan, dtype=np.float64)
    circuity_delta = np.full(n_total, np.nan, dtype=np.float64)
    circuity_ratio = np.full(n_total, np.nan, dtype=np.float64)
    snap_status = np.full(n_total, SNAP_OK, dtype=np.int32)
    warnings: List[str] = []

    if not kept_features:
        snap_status[valid_coord_mask] = SNAP_TARGET_EMPTY
        warnings.append(f"Target '{target_cfg.get('id')}': 0 active features after filtering.")
        return {
            "dist_eucl": dist_eucl,
            "dist_net": dist_net,
            "travel_time_min": travel_time_min,
            "circuity_delta": circuity_delta,
            "circuity_ratio": circuity_ratio,
            "snap_status": snap_status,
            "n_active_features": 0,
            "n_excluded_features": len(features) - len(kept_features),
            "warnings": warnings,
        }

    proj_feats = _project_feature_rings(kept_features, proj)
    geom_mode = target_cfg.get("geometry_handling", "point")
    dist_mode = target_cfg.get("distance_modes", "euclidean")
    inside_is_zero = bool(target_cfg.get("inside_is_zero", True))
    max_snap_m = float(target_cfg.get("max_snap_distance_m", 300.0))

    obs_valid_xy = obs_xy[valid_coord_mask]
    n_valid = len(obs_valid_xy)
    inside_mask = np.zeros(n_valid, dtype=bool)
    if n_valid > 0 and geom_mode == "polygon_boundary" and inside_is_zero:
        inside_mask = _inside_polygons_mask(obs_valid_xy, proj_feats)

    # 1. Compute Euclidean Distance (always needed when dist_mode is 'euclidean' or 'both')
    if n_valid > 0:
        if geom_mode == "point":
            pt_coords = []
            for f in proj_feats:
                if f["geom_type"] == "point":
                    for r_xy in f["outer_xy"]:
                        pt_coords.append(r_xy[0])
                else:
                    pt_coords.append(f["center_xy"])
            t_arr = np.asarray(pt_coords, dtype=np.float64)
            eucl_v, _ = cKDTree(t_arr).query(obs_valid_xy, k=1)
        elif geom_mode == "polygon_centroid":
            cent_coords = []
            large_poly_warn = False
            for f in proj_feats:
                if f["geom_type"] == "polygon":
                    total_a = 0.0
                    wx = 0.0
                    wy = 0.0
                    for r_xy in f["outer_xy"]:
                        a, cx, cy = _shoelace_area_and_centroid(r_xy)
                        total_a += a
                        wx += a * cx
                        wy += a * cy
                        diam = float(np.hypot(np.ptp(r_xy[:, 0]), np.ptp(r_xy[:, 1])))
                        if diam > 250.0:
                            large_poly_warn = True
                    for r_xy in f.get("inner_xy", []):
                        a, cx, cy = _shoelace_area_and_centroid(r_xy)
                        total_a -= a
                        wx -= a * cx
                        wy -= a * cy
                    cent_coords.append((wx / total_a, wy / total_a) if total_a > 1e-6 else f["center_xy"])
                else:
                    cent_coords.append(f["center_xy"])
            if large_poly_warn:
                warnings.append(
                    f"Target '{target_cfg.get('id')}' uses polygon_centroid on polygons >250 m wide; "
                    "consider polygon_boundary for edge accuracy."
                )
            eucl_v, _ = cKDTree(np.asarray(cent_coords, dtype=np.float64)).query(obs_valid_xy, k=1)
        else:
            all_rings: List[np.ndarray] = []
            pt_fallback: List[Tuple[float, float]] = []
            for f in proj_feats:
                if f["geom_type"] == "point":
                    for r_xy in f["outer_xy"]:
                        pt_fallback.append((float(r_xy[0, 0]), float(r_xy[0, 1])))
                else:
                    all_rings.extend(f["outer_xy"])
                    all_rings.extend(f.get("inner_xy", []))

            eucl_v = np.full(n_valid, np.inf, dtype=np.float64)
            if all_rings:
                seg_a, seg_b = _subdivide_segments(all_rings, max_len_m=50.0)
                if len(seg_a) > 0:
                    eucl_v = np.minimum(eucl_v, _point_to_segments_distance(obs_valid_xy, seg_a, seg_b, 50.0))
            if pt_fallback:
                d_pt, _ = cKDTree(np.asarray(pt_fallback, dtype=np.float64)).query(obs_valid_xy, k=1)
                eucl_v = np.minimum(eucl_v, d_pt)

            if geom_mode == "polygon_boundary" and inside_is_zero:
                eucl_v[inside_mask] = 0.0

        dist_eucl[valid_coord_mask] = eucl_v

    # 2. Compute Network Dijkstra Distance + travel_time_min via unified snap + virtual super-source S_0
    if dist_mode in ("network_dijkstra", "both") and n_valid > 0:
        if street_graph is None or street_graph.get("empty", True):
            net_v = np.full(n_valid, np.nan, dtype=np.float64)
            time_v = np.full(n_valid, np.nan, dtype=np.float64)
            status_v = np.full(n_valid, SNAP_FETCH_FAILED, dtype=np.int32)
            if np.any(inside_mask):
                net_v[inside_mask] = 0.0
                time_v[inside_mask] = 0.0
                status_v[inside_mask] = SNAP_OK
            dist_net[valid_coord_mask] = net_v
            travel_time_min[valid_coord_mask] = time_v
            snap_status[valid_coord_mask] = status_v
            warnings.append(f"Target '{target_cfg.get('id')}': street routing graph unavailable.")
        else:
            t_access_xy = _collect_target_access_points(
                proj_feats, geom_mode, densify_step_m=25.0, street_graph=street_graph, max_snap_m=max_snap_m
            )
            obs_snap = _snap_points_to_edges(obs_valid_xy, street_graph, max_snap_m)
            tgt_snap = _snap_points_to_edges(t_access_xy, street_graph, max_snap_m)

            tgt_valid = tgt_snap["edge_idx"] >= 0
            if not np.any(tgt_valid):
                net_v = np.full(n_valid, np.nan, dtype=np.float64)
                time_v = np.full(n_valid, np.nan, dtype=np.float64)
                status_v = np.full(n_valid, SNAP_EXCEEDED, dtype=np.int32)
                if np.any(inside_mask):
                    net_v[inside_mask] = 0.0
                    time_v[inside_mask] = 0.0
                    status_v[inside_mask] = SNAP_OK
                dist_net[valid_coord_mask] = net_v
                travel_time_min[valid_coord_mask] = time_v
                snap_status[valid_coord_mask] = status_v
                warnings.append(
                    f"Target '{target_cfg.get('id')}': no target features snapped within {max_snap_m:.0f} m of street graph."
                )
            else:
                n_nodes = street_graph["n_nodes"]
                s0 = n_nodes
                g_rev: csr_matrix = street_graph["g_rev"]
                g_time_rev: csr_matrix = street_graph.get("g_time_rev", g_rev)
                edge_u = street_graph["edge_u"]
                edge_v = street_graph["edge_v"]
                edge_dir = street_graph["edge_dir"]
                edge_len = street_graph["edge_len"]
                edge_tmin = street_graph.get("edge_time_min")
                snap_mpm = _profile_base_m_per_min(street_graph["routing_profile"])

                aug_rows: List[int] = []
                aug_cols: List[int] = []
                aug_data: List[float] = []
                aug_tdata: List[float] = []
                same_edge_targets: Dict[int, List[Tuple[float, float]]] = {}

                for e_id, s_val, d_s in zip(
                    tgt_snap["edge_idx"][tgt_valid],
                    tgt_snap["s"][tgt_valid],
                    tgt_snap["snap_d"][tgt_valid],
                ):
                    e_i = int(e_id)
                    u_i = int(edge_u[e_i])
                    v_i = int(edge_v[e_i])
                    L = float(edge_len[e_i])
                    T_L = float(edge_tmin[e_i]) if edge_tmin is not None else (L / snap_mpm)
                    t_snap = float(d_s) / snap_mpm
                    d_dir = int(edge_dir[e_i])
                    same_edge_targets.setdefault(e_i, []).append((float(s_val), float(d_s)))
                    if d_dir == 0 or d_dir == 1:
                        aug_rows.append(s0); aug_cols.append(u_i)
                        aug_data.append(float(d_s + s_val * L))
                        aug_tdata.append(t_snap + float(s_val) * T_L)
                    if d_dir == 0 or d_dir == -1:
                        aug_rows.append(s0); aug_cols.append(v_i)
                        aug_data.append(float(d_s + (1.0 - s_val) * L))
                        aug_tdata.append(t_snap + (1.0 - float(s_val)) * T_L)

                directed_g = street_graph["routing_profile"] != "walk"
                ded_r, ded_c, ded_w = _dedup_min_edges(aug_rows, aug_cols, aug_data, n_nodes + 1)
                coo = g_rev.tocoo()
                aug_g = csr_matrix(
                    (np.concatenate([coo.data, ded_w]), (np.concatenate([coo.row, ded_r]), np.concatenate([coo.col, ded_c]))),
                    shape=(n_nodes + 1, n_nodes + 1),
                )
                d_from_s0 = dijkstra(aug_g, directed=directed_g, indices=s0)

                ded_tr, ded_tc, ded_tw = _dedup_min_edges(aug_rows, aug_cols, aug_tdata, n_nodes + 1)
                coo_t = g_time_rev.tocoo()
                aug_gt = csr_matrix(
                    (np.concatenate([coo_t.data, ded_tw]), (np.concatenate([coo_t.row, ded_tr]), np.concatenate([coo_t.col, ded_tc]))),
                    shape=(n_nodes + 1, n_nodes + 1),
                )
                t_from_s0 = dijkstra(aug_gt, directed=directed_g, indices=s0)

                net_v = np.full(n_valid, np.nan, dtype=np.float64)
                time_v = np.full(n_valid, np.nan, dtype=np.float64)
                status_v = np.full(n_valid, SNAP_OK, dtype=np.int32)

                for idx_i in range(n_valid):
                    if inside_mask[idx_i]:
                        net_v[idx_i] = 0.0
                        time_v[idx_i] = 0.0
                        status_v[idx_i] = SNAP_OK
                        continue
                    e_i = int(obs_snap["edge_idx"][idx_i])
                    if e_i < 0:
                        status_v[idx_i] = SNAP_EXCEEDED
                        continue
                    s_obs = float(obs_snap["s"][idx_i])
                    d_snap_obs = float(obs_snap["snap_d"][idx_i])
                    u_i = int(edge_u[e_i])
                    v_i = int(edge_v[e_i])
                    L = float(edge_len[e_i])
                    T_L = float(edge_tmin[e_i]) if edge_tmin is not None else (L / snap_mpm)
                    t_snap_obs = d_snap_obs / snap_mpm
                    d_dir = int(edge_dir[e_i])

                    best = np.inf
                    best_t = np.inf
                    if d_dir == 0 or d_dir == -1:
                        if np.isfinite(d_from_s0[u_i]):
                            best = min(best, d_snap_obs + s_obs * L + float(d_from_s0[u_i]))
                        if np.isfinite(t_from_s0[u_i]):
                            best_t = min(best_t, t_snap_obs + s_obs * T_L + float(t_from_s0[u_i]))
                    if d_dir == 0 or d_dir == 1:
                        if np.isfinite(d_from_s0[v_i]):
                            best = min(best, d_snap_obs + (1.0 - s_obs) * L + float(d_from_s0[v_i]))
                        if np.isfinite(t_from_s0[v_i]):
                            best_t = min(best_t, t_snap_obs + (1.0 - s_obs) * T_L + float(t_from_s0[v_i]))

                    if e_i in same_edge_targets:
                        for s_t, d_snap_t in same_edge_targets[e_i]:
                            if d_dir == 0 or (d_dir == 1 and s_t >= s_obs) or (d_dir == -1 and s_t <= s_obs):
                                best = min(best, d_snap_obs + abs(s_t - s_obs) * L + d_snap_t)
                                best_t = min(best_t, t_snap_obs + abs(s_t - s_obs) * T_L + (d_snap_t / snap_mpm))

                    if not np.isfinite(best):
                        status_v[idx_i] = SNAP_DISCONNECTED
                    else:
                        net_v[idx_i] = best
                        time_v[idx_i] = best_t if np.isfinite(best_t) else (best / snap_mpm)

                dist_net[valid_coord_mask] = net_v
                travel_time_min[valid_coord_mask] = time_v
                snap_status[valid_coord_mask] = status_v

                n_disc = int(np.sum(status_v == SNAP_DISCONNECTED))
                n_exc = int(np.sum(status_v == SNAP_EXCEEDED))
                if n_disc > 0 or n_exc > 0:
                    warnings.append(
                        f"Target '{target_cfg.get('id')}': {n_disc} disconnected, {n_exc} >{max_snap_m:.0f}m snap limit (kept as NaN, zero 1.25x fabrication)."
                    )

        both_valid = valid_coord_mask & np.isfinite(dist_net) & np.isfinite(dist_eucl)
        if np.any(both_valid):
            dn = dist_net[both_valid]
            de = dist_eucl[both_valid]
            zero_both = (de <= 1e-6) & (dn <= 1e-6)
            circuity_delta[both_valid] = np.where(zero_both, 0.0, np.maximum(0.0, dn - de))
            circuity_ratio[both_valid] = np.where(zero_both, 1.0, dn / np.maximum(1.0, de))

    return {
        "dist_eucl": dist_eucl,
        "dist_net": dist_net,
        "travel_time_min": travel_time_min,
        "circuity_delta": circuity_delta,
        "circuity_ratio": circuity_ratio,
        "snap_status": snap_status,
        "n_active_features": len(kept_features),
        "n_excluded_features": len(features) - len(kept_features),
        "warnings": warnings,
    }


def compute_single_route(
    obs_xy_single: Tuple[float, float],
    features: List[Dict[str, Any]],
    target_cfg: Dict[str, Any],
    proj: LocalMetricProjection,
    street_graph: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    On-demand single-point shortest route replay (<10 ms) when clicking an observation on the map.
    Shares exact unified segment-to-graph access snapping with Stage 1 batch distances and returns snap_reason & travel_time_min.
    """
    excl = set(target_cfg.get("excluded_osm_ids", []) or [])
    kept = [f for f in features if f["uid"] not in excl]
    if not kept:
        return {"ok": False, "reason": "no_active_features", "snap_reason": "SNAP_TARGET_EMPTY"}

    proj_feats = _project_feature_rings(kept, proj)
    geom_mode = target_cfg.get("geometry_handling", "point")
    inside_is_zero = bool(target_cfg.get("inside_is_zero", True))
    max_snap_m = float(target_cfg.get("max_snap_distance_m", 300.0))
    p_xy = np.asarray([obs_xy_single], dtype=np.float64)
    p_lat, p_lon = proj.inverse(obs_xy_single[0], obs_xy_single[1])

    is_inside = geom_mode == "polygon_boundary" and inside_is_zero and bool(_inside_polygons_mask(p_xy, proj_feats)[0])
    if is_inside:
        return {
            "ok": True,
            "target_id": target_cfg.get("id"),
            "inside_zero": True,
            "dist_eucl_m": 0.0,
            "euclidean_geojson": None,
            "network_geojson": None,
            "dist_net_m": 0.0,
            "travel_time_min": 0.0,
            "detour_m": 0.0,
            "snap_reason": "OK",
        }

    # Exact nearest Euclidean point on target
    t_access_xy = _collect_target_access_points(
        proj_feats, geom_mode, densify_step_m=25.0, street_graph=street_graph, max_snap_m=max_snap_m
    )
    eucl_d = np.inf
    nearest_tgt_xy = obs_xy_single
    if geom_mode in ("line_segment", "polygon_boundary"):
        all_rings: List[np.ndarray] = []
        pt_fallback: List[Tuple[float, float]] = []
        for f in proj_feats:
            if f["geom_type"] == "point":
                for r_xy in f["outer_xy"]:
                    pt_fallback.append((float(r_xy[0, 0]), float(r_xy[0, 1])))
            else:
                all_rings.extend(f["outer_xy"])
                all_rings.extend(f.get("inner_xy", []))
        if all_rings:
            seg_a, seg_b = _subdivide_segments(all_rings, max_len_m=50.0)
            if len(seg_a) > 0:
                d_seg, xy_seg = _point_to_segments_nearest(p_xy, seg_a, seg_b, 50.0)
                if float(d_seg[0]) < eucl_d:
                    eucl_d = float(d_seg[0])
                    nearest_tgt_xy = (float(xy_seg[0, 0]), float(xy_seg[0, 1]))
        if pt_fallback:
            pt_arr = np.asarray(pt_fallback, dtype=np.float64)
            d_pt, nn_pt = cKDTree(pt_arr).query(p_xy, k=1)
            if float(d_pt[0]) < eucl_d:
                eucl_d = float(d_pt[0])
                nearest_tgt_xy = (float(pt_arr[int(nn_pt[0]), 0]), float(pt_arr[int(nn_pt[0]), 1]))
    if not math.isfinite(eucl_d) and len(t_access_xy) > 0:
        d_eucl_arr, nn_idx = cKDTree(t_access_xy).query(p_xy, k=1)
        eucl_d = float(d_eucl_arr[0])
        nearest_tgt_xy = (float(t_access_xy[int(nn_idx[0]), 0]), float(t_access_xy[int(nn_idx[0]), 1]))

    t_lat, t_lon = proj.inverse(nearest_tgt_xy[0], nearest_tgt_xy[1])

    res: Dict[str, Any] = {
        "ok": True,
        "target_id": target_cfg.get("id"),
        "dist_eucl_m": round(eucl_d, 2),
        "euclidean_geojson": {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[float(p_lon), float(p_lat)], [float(t_lon), float(t_lat)]]},
            "properties": {"kind": "euclidean", "dist_m": round(eucl_d, 2)},
        },
        "network_geojson": None,
        "dist_net_m": None,
        "travel_time_min": None,
        "detour_m": None,
        "snap_reason": "SNAP_NO_GRAPH",
    }

    if street_graph is None or street_graph.get("empty", True):
        res["snap_reason"] = "SNAP_NO_GRAPH"
        return res
    if len(t_access_xy) == 0:
        res["snap_reason"] = "SNAP_TARGET_EMPTY"
        return res

    obs_snap = _snap_points_to_edges(p_xy, street_graph, max_snap_m)
    tgt_snap = _snap_points_to_edges(t_access_xy, street_graph, max_snap_m)

    if int(obs_snap["edge_idx"][0]) < 0 or not np.any(tgt_snap["edge_idx"] >= 0):
        res["snap_reason"] = "SNAP_EXCEEDED"
        return res

    n_nodes = street_graph["n_nodes"]
    g_fwd: csr_matrix = street_graph["g_fwd"]
    g_time_fwd: csr_matrix = street_graph.get("g_time_fwd", g_fwd)
    edge_u = street_graph["edge_u"]
    edge_v = street_graph["edge_v"]
    edge_dir = street_graph["edge_dir"]
    edge_len = street_graph["edge_len"]
    edge_tmin = street_graph.get("edge_time_min")
    snap_mpm = _profile_base_m_per_min(street_graph["routing_profile"])
    node_xy = street_graph["node_xy"]

    s_obs_node = n_nodes
    s_tgt_node = n_nodes + 1
    e_obs = int(obs_snap["edge_idx"][0])
    s_obs = float(obs_snap["s"][0])
    d_snap_obs = float(obs_snap["snap_d"][0])
    u_o, v_o, L_o, dir_o = int(edge_u[e_obs]), int(edge_v[e_obs]), float(edge_len[e_obs]), int(edge_dir[e_obs])
    T_o = float(edge_tmin[e_obs]) if edge_tmin is not None else (L_o / snap_mpm)
    t_snap_obs = d_snap_obs / snap_mpm

    aug_rows: List[int] = []
    aug_cols: List[int] = []
    aug_data: List[float] = []
    aug_tdata: List[float] = []

    if dir_o == 0 or dir_o == -1:
        aug_rows.append(s_obs_node); aug_cols.append(u_o); aug_data.append(float(d_snap_obs + s_obs * L_o))
        aug_tdata.append(t_snap_obs + s_obs * T_o)
    if dir_o == 0 or dir_o == 1:
        aug_rows.append(s_obs_node); aug_cols.append(v_o); aug_data.append(float(d_snap_obs + (1.0 - s_obs) * L_o))
        aug_tdata.append(t_snap_obs + (1.0 - s_obs) * T_o)

    tgt_valid_idx = np.where(tgt_snap["edge_idx"] >= 0)[0]
    best_same = np.inf
    best_same_t = np.inf
    best_same_xy = None
    tgt_best_exit: Dict[int, Tuple[float, np.ndarray, np.ndarray]] = {}

    for j in tgt_valid_idx:
        e_t = int(tgt_snap["edge_idx"][j])
        s_t = float(tgt_snap["s"][j])
        d_st = float(tgt_snap["snap_d"][j])
        u_t, v_t, L_t, dir_t = int(edge_u[e_t]), int(edge_v[e_t]), float(edge_len[e_t]), int(edge_dir[e_t])
        T_t = float(edge_tmin[e_t]) if edge_tmin is not None else (L_t / snap_mpm)
        t_st = d_st / snap_mpm
        if e_t == e_obs:
            if dir_o == 0 or (dir_o == 1 and s_t >= s_obs) or (dir_o == -1 and s_t <= s_obs):
                d_cand = d_snap_obs + abs(s_t - s_obs) * L_o + d_st
                t_cand = t_snap_obs + abs(s_t - s_obs) * T_o + t_st
                if d_cand < best_same:
                    best_same = d_cand
                    best_same_t = t_cand
                    best_same_xy = (tgt_snap["snap_xy"][j], t_access_xy[j])

        if dir_t == 0 or dir_t == 1:
            cost_u = float(s_t * L_t + d_st)
            aug_rows.append(u_t); aug_cols.append(s_tgt_node); aug_data.append(cost_u)
            aug_tdata.append(float(s_t * T_t + t_st))
            if u_t not in tgt_best_exit or cost_u < tgt_best_exit[u_t][0]:
                tgt_best_exit[u_t] = (cost_u, tgt_snap["snap_xy"][j], t_access_xy[j])
        if dir_t == 0 or dir_t == -1:
            cost_v = float((1.0 - s_t) * L_t + d_st)
            aug_rows.append(v_t); aug_cols.append(s_tgt_node); aug_data.append(cost_v)
            aug_tdata.append(float((1.0 - s_t) * T_t + t_st))
            if v_t not in tgt_best_exit or cost_v < tgt_best_exit[v_t][0]:
                tgt_best_exit[v_t] = (cost_v, tgt_snap["snap_xy"][j], t_access_xy[j])

    directed_g = street_graph["routing_profile"] != "walk"
    ded_r, ded_c, ded_w = _dedup_min_edges(aug_rows, aug_cols, aug_data, n_nodes + 2)
    coo = g_fwd.tocoo()
    aug_g = csr_matrix(
        (np.concatenate([coo.data, ded_w]), (np.concatenate([coo.row, ded_r]), np.concatenate([coo.col, ded_c]))),
        shape=(n_nodes + 2, n_nodes + 2),
    )
    dist_arr, preds = dijkstra(aug_g, directed=directed_g, indices=s_obs_node, return_predecessors=True)

    ded_tr, ded_tc, ded_tw = _dedup_min_edges(aug_rows, aug_cols, aug_tdata, n_nodes + 2)
    coo_t = g_time_fwd.tocoo()
    aug_gt = csr_matrix(
        (np.concatenate([coo_t.data, ded_tw]), (np.concatenate([coo_t.row, ded_tr]), np.concatenate([coo_t.col, ded_tc]))),
        shape=(n_nodes + 2, n_nodes + 2),
    )
    time_arr = dijkstra(aug_gt, directed=directed_g, indices=s_obs_node)

    graph_d = float(dist_arr[s_tgt_node])
    graph_t = float(time_arr[s_tgt_node])
    if best_same <= graph_d and best_same_xy is not None:
        coords_xy = [obs_xy_single, tuple(obs_snap["snap_xy"][0]), tuple(best_same_xy[0]), tuple(best_same_xy[1])]
        net_d = best_same
        net_t = best_same_t
    elif np.isfinite(graph_d):
        chain_nodes: List[int] = []
        curr = int(preds[s_tgt_node])
        while curr >= 0 and curr != s_obs_node:
            chain_nodes.append(curr)
            curr = int(preds[curr])
        chain_nodes.reverse()
        coords_xy = [obs_xy_single, tuple(obs_snap["snap_xy"][0])]
        for nid in chain_nodes:
            coords_xy.append((float(node_xy[nid, 0]), float(node_xy[nid, 1])))
        if chain_nodes and chain_nodes[-1] in tgt_best_exit:
            _, snap_end_xy, tgt_end_xy = tgt_best_exit[chain_nodes[-1]]
            coords_xy.append((float(snap_end_xy[0]), float(snap_end_xy[1])))
            coords_xy.append((float(tgt_end_xy[0]), float(tgt_end_xy[1])))
        net_d = graph_d
        net_t = min(best_same_t, graph_t) if np.isfinite(graph_t) else (graph_d / snap_mpm)
    else:
        res["snap_reason"] = "SNAP_DISCONNECTED"
        return res

    arr_xy = np.asarray(coords_xy, dtype=np.float64)
    w_lat, w_lon = proj.inverse(arr_xy[:, 0], arr_xy[:, 1])
    line_coords = [[float(lo), float(la)] for lo, la in zip(w_lon, w_lat)]
    res["dist_net_m"] = round(net_d, 2)
    res["travel_time_min"] = round(net_t, 2)
    res["detour_m"] = round(max(0.0, net_d - eucl_d), 2)
    res["snap_reason"] = "OK"
    res["network_geojson"] = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": line_coords},
        "properties": {
            "kind": "network_dijkstra",
            "dist_m": round(net_d, 2),
            "travel_time_min": round(net_t, 2),
            "detour_m": round(max(0.0, net_d - eucl_d), 2),
        },
    }
    return res
