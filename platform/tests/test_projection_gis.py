"""Unit tests for universal metric projection, parametric OSM/GeoJSON parsing, and CSR Dijkstra."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_engine.projection import LocalMetricProjection, compute_circular_centroid
from py_engine.spatial_gis import (
    SNAP_DISCONNECTED,
    SNAP_OK,
    build_street_graph,
    compute_single_route,
    compute_target_distances,
    parse_geojson_features,
    parse_osm_elements,
)


def test_kruger_tmerc_roundtrip_and_antimeridian() -> None:
    # Mid-latitude cluster (~47.5N, 19.1E)
    lats = np.array([47.48, 47.52, 47.49, 47.55, 47.45])
    lons = np.array([19.05, 19.12, 19.18, 19.09, 19.22])
    proj = LocalMetricProjection.from_points(lats, lons)
    assert proj.mode == "tmerc"
    x, y = proj.forward(lats, lons)
    lat_back, lon_back = proj.inverse(x, y)
    assert np.allclose(lats, lat_back, atol=1e-7)
    assert np.allclose(lons, lon_back, atol=1e-7)
    # Centroid sits near (0, 0) m
    assert abs(np.mean(x)) < 50.0 and abs(np.mean(y)) < 50.0

    # Antimeridian crossing (Fiji ~ 179.95E and -179.95W)
    a_lats = np.array([-16.8, -16.82, -16.79, -16.81])
    a_lons = np.array([179.95, -179.95, 179.98, -179.97])
    lat0, lon0, span = compute_circular_centroid(a_lats, a_lons)
    assert span < 0.2
    assert abs(abs(lon0) - 180.0) < 0.1
    p_anti = LocalMetricProjection.from_points(a_lats, a_lons)
    ax, ay = p_anti.forward(a_lats, a_lons)
    # Points separated by ~0.1 deg (~10 km), NOT 360 deg (~40,000 km)
    assert np.ptp(ax) < 15000.0
    alat_b, alon_b = p_anti.inverse(ax, ay)
    assert np.allclose(a_lats, alat_b, atol=1e-7)
    assert np.allclose(np.sin(np.radians(a_lons)), np.sin(np.radians(alon_b)), atol=1e-7)


def test_long_line_segment_and_polygon_hole_and_exclusion() -> None:
    proj = LocalMetricProjection(47.5, 19.1, 0.1)
    # Create a 600 m straight E-W railway line in meters, convert to WGS84
    line_xy = np.array([[-300.0, 0.0], [300.0, 0.0]])
    lat_l, lon_l = proj.inverse(line_xy[:, 0], line_xy[:, 1])
    osm_json = {
        "elements": [
            {
                "type": "way",
                "id": 101,
                "tags": {"railway": "rail", "name": "Main Line"},
                "geometry": [{"lat": float(la), "lon": float(lo)} for lo, la in zip(lon_l, lat_l)],
            },
            {
                "type": "way",
                "id": 102,
                "tags": {"railway": "rail", "name": "Abandoned Spur"},
                "geometry": [{"lat": float(lat_l[0] + 0.001), "lon": float(lon_l[0])}, {"lat": float(lat_l[1] + 0.001), "lon": float(lon_l[1])}],
            },
        ]
    }
    feats = parse_osm_elements(osm_json)
    assert len(feats) == 2

    # Observation directly north of midpoint (0, 25 m) -> exact distance MUST be 25 m, not ~300 m to endpoints!
    obs_xy = np.array([[0.0, 25.0], [100.0, 150.0]])
    valid_mask = np.array([True, True])
    res_all = compute_target_distances(
        obs_xy,
        valid_mask,
        feats,
        {"id": "rail", "geometry_handling": "line_segment", "distance_modes": "euclidean", "excluded_osm_ids": ["way/102"]},
        proj,
    )
    assert res_all["n_active_features"] == 1
    assert res_all["n_excluded_features"] == 1
    assert abs(res_all["dist_eucl"][0] - 25.0) < 0.2
    assert abs(res_all["dist_eucl"][1] - 150.0) < 0.2

    # Polygon boundary with interior hole & inside_is_zero=True
    outer = np.array([[-100, -100], [100, -100], [100, 100], [-100, 100], [-100, -100]], dtype=float)
    inner = np.array([[-30, -30], [30, -30], [30, 30], [-30, 30], [-30, -30]], dtype=float)
    ola, olo = proj.inverse(outer[:, 0], outer[:, 1])
    ila, ilo = proj.inverse(inner[:, 0], inner[:, 1])
    gj = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "id": "park1",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[float(x), float(y)] for x, y in zip(olo, ola)],
                    [[float(x), float(y)] for x, y in zip(ilo, ila)],
                ],
            },
        }],
    }
    pfeats = parse_geojson_features(gj)
    # Point at (50, 0) is inside outer and outside hole -> 0.0 m; point at (0, 0) is inside hole -> 30.0 m to hole ring!
    p_obs = np.array([[50.0, 0.0], [0.0, 0.0], [140.0, 0.0]])
    pres = compute_target_distances(
        p_obs,
        np.array([True, True, True]),
        pfeats,
        {"id": "park", "geometry_handling": "polygon_boundary", "inside_is_zero": True, "distance_modes": "euclidean"},
        proj,
    )
    assert abs(pres["dist_eucl"][0] - 0.0) < 1e-6
    assert abs(pres["dist_eucl"][1] - 30.0) < 0.2
    assert abs(pres["dist_eucl"][2] - 40.0) < 0.2


def test_virtual_supersource_dijkstra_and_single_route() -> None:
    proj = LocalMetricProjection(47.5, 19.1, 0.1)
    # Street L-shaped corridor: (0,0) -> (200,0) -> (200,200) and disconnected island at (1000,1000)->(1100,1000)
    pts_m = np.array([
        [0.0, 0.0], [200.0, 0.0], [200.0, 200.0],
        [1000.0, 1000.0], [1100.0, 1000.0],
    ])
    lats, lons = proj.inverse(pts_m[:, 0], pts_m[:, 1])
    graph_json = {
        "elements": [
            {"type": "node", "id": i + 1, "lat": float(lats[i]), "lon": float(lons[i])}
            for i in range(5)
        ] + [
            {"type": "way", "id": 10, "nodes": [1, 2, 3], "tags": {"highway": "residential"}},
            {"type": "way", "id": 11, "nodes": [4, 5], "tags": {"highway": "residential"}},
        ]
    }
    graph = build_street_graph(graph_json, proj, routing_profile="walk")
    assert not graph["empty"]

    # Target at (200, 200)
    t_lat, t_lon = proj.inverse(200.0, 200.0)
    t_feats = parse_osm_elements({"elements": [{"type": "node", "id": 99, "lat": float(t_lat), "lon": float(t_lon)}]})

    # Obs 0 at (0, 0): eucl = sqrt(200^2+200^2) = 282.84 m, net = 200 + 200 = 400 m
    # Obs 1 at (1050, 1000) on disconnected component -> NaN + SNAP_DISCONNECTED (no 1.25x fabrication!)
    obs_xy = np.array([[0.0, 0.0], [1050.0, 1000.0]])
    dres = compute_target_distances(
        obs_xy,
        np.array([True, True]),
        t_feats,
        {"id": "st", "geometry_handling": "point", "distance_modes": "both", "routing_profile": "walk", "max_snap_distance_m": 100.0},
        proj,
        graph,
    )
    assert dres["snap_status"][0] == SNAP_OK
    assert abs(dres["dist_eucl"][0] - math.hypot(200.0, 200.0)) < 0.5
    assert abs(dres["dist_net"][0] - 400.0) < 0.5
    assert dres["circuity_delta"][0] > 115.0
    assert dres["snap_status"][1] == SNAP_DISCONNECTED
    assert np.isnan(dres["dist_net"][1])

    # Click single-route replay for Obs 0
    route = compute_single_route(
        (0.0, 0.0),
        t_feats,
        {"id": "st", "geometry_handling": "point", "distance_modes": "both", "routing_profile": "walk", "max_snap_distance_m": 100.0},
        proj,
        graph,
    )
    assert route["ok"] is True
    assert route["network_geojson"] is not None
    assert abs(route["dist_net_m"] - 400.0) < 0.5
