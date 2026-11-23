"""
Universal Metric Projection (Pure NumPy, zero pyproj)
====================================================
Projects geographic WGS84 coordinates (lat, lon) onto a local conformal
Krüger 4th-order Transverse Mercator plane centered on the study area's circular
centroid (0, 0) m, eliminating UTM zone splits and 7-digit floating-point
cancellation. Falls back to Conformal-Sphere Oblique Azimuthal Equidistant (AEQD)
at high latitudes (|lat0| > 80 deg) or wide longitudinal spans (> 6 deg), and
provides CRS/swapped-lat-lon diagnostics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

# WGS84 ellipsoid constants
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = 2.0 * _WGS84_F - _WGS84_F * _WGS84_F
_WGS84_N = _WGS84_F / (2.0 - _WGS84_F)

# Krüger rectifying radius A and 4th-order forward/inverse series coefficients
_N = _WGS84_N
_N2 = _N * _N
_N3 = _N2 * _N
_N4 = _N3 * _N
_A_RECT = (_WGS84_A / (1.0 + _N)) * (1.0 + _N2 / 4.0 + _N4 / 64.0)

_ALPHA = (
    0.5 * _N - (2.0 / 3.0) * _N2 + (5.0 / 16.0) * _N3 + (41.0 / 180.0) * _N4,
    (13.0 / 48.0) * _N2 - (3.0 / 5.0) * _N3 + (557.0 / 1440.0) * _N4,
    (61.0 / 240.0) * _N3 - (103.0 / 140.0) * _N4,
    (49561.0 / 161280.0) * _N4,
)

_BETA = (
    0.5 * _N - (2.0 / 3.0) * _N2 + (37.0 / 96.0) * _N3 - (1.0 / 360.0) * _N4,
    (1.0 / 48.0) * _N2 + (1.0 / 15.0) * _N3 - (437.0 / 1440.0) * _N4,
    (17.0 / 480.0) * _N3 - (37.0 / 840.0) * _N4,
    (4397.0 / 161280.0) * _N4,
)

_DELTA = (
    2.0 * _N - (2.0 / 3.0) * _N2 - 2.0 * _N3 + (116.0 / 45.0) * _N4,
    (7.0 / 3.0) * _N2 - (8.0 / 5.0) * _N3 - (227.0 / 45.0) * _N4,
    (56.0 / 15.0) * _N3 - (136.0 / 35.0) * _N4,
    (4279.0 / 630.0) * _N4,
)

_SQRT_N_TERM = 2.0 * math.sqrt(_N) / (1.0 + _N)


def compute_circular_centroid(lat_deg: np.ndarray, lon_deg: np.ndarray) -> Tuple[float, float, float]:
    """Compute antimeridian-safe circular centroid (lat0, lon0) and longitudinal span (deg)."""
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    mask = np.isfinite(lat) & np.isfinite(lon)
    # Exclude explicit (0, 0) null-island placeholders if other valid points exist
    non_zero = mask & ~((np.abs(lat) < 1e-12) & (np.abs(lon) < 1e-12))
    if np.any(non_zero):
        mask = non_zero
    if not np.any(mask):
        raise ValueError("No valid geographic coordinates found for centroid calculation.")

    lat_v = lat[mask]
    lon_v = lon[mask]
    lon_rad = np.radians(lon_v)
    lon0_rad = float(np.arctan2(np.mean(np.sin(lon_rad)), np.mean(np.cos(lon_rad))))
    lon0_deg = math.degrees(lon0_rad)
    lat0_deg = float(np.mean(lat_v))

    dlon = np.arctan2(np.sin(lon_rad - lon0_rad), np.cos(lon_rad - lon0_rad))
    lon_span_deg = float(math.degrees(np.max(dlon) - np.min(dlon)))
    return lat0_deg, lon0_deg, lon_span_deg


def diagnose_coordinate_issues(lat_deg: np.ndarray, lon_deg: np.ndarray) -> Dict[str, Any]:
    """
    Detect common CRS mistakes (projected meters like UTM/EOV/WebMercator or swapped lat/lon).
    """
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    fin = np.isfinite(lat) & np.isfinite(lon)
    hints: List[str] = []
    looks_projected_crs = False
    suggested_swap = False

    if not np.any(fin):
        return {"looks_projected_crs": False, "suggested_swap": False, "hints": ["No finite numeric coordinates found."]}

    la = lat[fin]
    lo = lon[fin]
    max_abs = float(max(np.max(np.abs(la)), np.max(np.abs(lo))))
    if max_abs > 180.0:
        looks_projected_crs = True
        hints.append(
            f"Coordinates exceed [-180, 180] (max |coord|={max_abs:.1f}): values appear to be projected meters "
            "(e.g. UTM, EOV EPSG:23700, or Web Mercator) rather than WGS84 decimal degrees (EPSG:4326)."
        )
    else:
        lat_out = (np.abs(la) > 90.0) & (np.abs(la) <= 180.0)
        lon_in_lat = np.abs(lo) <= 90.0
        if np.any(lat_out & lon_in_lat):
            suggested_swap = True
            hints.append(
                "Some 'lat' values exceed [-90, 90] while 'lon' falls within [-90, 90]: "
                "latitude and longitude columns appear to be swapped."
            )

    return {
        "looks_projected_crs": looks_projected_crs,
        "suggested_swap": suggested_swap,
        "hints": hints,
    }


def _conformal_lat(phi_rad: np.ndarray) -> np.ndarray:
    """Compute WGS84 conformal latitude chi(phi)."""
    phi_c = np.clip(phi_rad, -1.5707963, 1.5707963)
    sin_p = np.sin(phi_c)
    q = np.arcsinh(np.tan(phi_c)) - _SQRT_N_TERM * np.arctanh(_SQRT_N_TERM * sin_p)
    return np.arctan(np.sinh(q))


def _meridian_arc_tmerc(phi_rad: float) -> float:
    """Compute conformal-rectified northing M(phi) for scale factor k0=1.0."""
    sin_p = math.sin(phi_rad)
    q = math.asinh(math.tan(phi_rad)) - _SQRT_N_TERM * math.atanh(_SQRT_N_TERM * sin_p)
    xi_p = math.atan(math.sinh(q))
    m = xi_p
    for j, a_j in enumerate(_ALPHA, start=1):
        m += a_j * math.sin(2.0 * j * xi_p)
    return _A_RECT * m


class LocalMetricProjection:
    """Local origin-centered conformal Transverse Mercator (or Oblique AEQD fallback) projection."""

    def __init__(self, lat0_deg: float, lon0_deg: float, lon_span_deg: float = 0.0) -> None:
        self.lat0_deg = float(lat0_deg)
        self.lon0_deg = float(lon0_deg)
        self.lon_span_deg = float(lon_span_deg)
        self.lat0_rad = math.radians(self.lat0_deg)
        self.lon0_rad = math.radians(self.lon0_deg)

        self.warnings: List[str] = []
        if abs(self.lat0_deg) > 80.0 or self.lon_span_deg > 6.0:
            self.mode = "aeqd"
            if self.lon_span_deg > 12.0:
                self.warnings.append(
                    f"Wide longitudinal span ({self.lon_span_deg:.2f} deg > 12 deg): using Conformal-Sphere Oblique AEQD "
                    "metric plane; verify planar distance distortion if spanning continental scales."
                )
        else:
            self.mode = "tmerc"

        self._m0 = _meridian_arc_tmerc(self.lat0_rad)
        sin0 = math.sin(self.lat0_rad)
        denom = 1.0 - _WGS84_E2 * sin0 * sin0
        self._rn0 = _WGS84_A / math.sqrt(denom)
        self._rm0 = _WGS84_A * (1.0 - _WGS84_E2) / (denom ** 1.5)
        self._cos_lat0 = math.cos(self.lat0_rad)

        self._chi0 = float(_conformal_lat(np.array([self.lat0_rad]))[0])
        self._sin_chi0 = math.sin(self._chi0)
        self._cos_chi0 = math.cos(self._chi0)
        if abs(self.lat0_deg) > 89.0 or self._cos_chi0 < 1e-6:
            self._r_conf = math.sqrt(self._rn0 * self._rm0)
        else:
            self._r_conf = self._rn0 * (self._cos_lat0 / self._cos_chi0)

        # Nearest standard UTM zone provenance
        zone = int((self.lon0_deg + 180.0) / 6.0) % 60 + 1
        self.nearest_utm_epsg = f"EPSG:{32600 + zone if self.lat0_deg >= 0 else 32700 + zone}"
        if self.mode == "aeqd":
            self.proj4_string = (
                f"+proj=aeqd +lat_0={self.lat0_deg:.6f} +lon_0={self.lon0_deg:.6f} "
                f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
            )
        else:
            self.proj4_string = (
                f"+proj=tmerc +lat_0={self.lat0_deg:.6f} +lon_0={self.lon0_deg:.6f} "
                f"+k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
            )

    @classmethod
    def from_points(cls, lat_deg: np.ndarray, lon_deg: np.ndarray) -> "LocalMetricProjection":
        """Build optimal projection directly from dataset coordinates."""
        lat0, lon0, span = compute_circular_centroid(lat_deg, lon_deg)
        return cls(lat0, lon0, span)

    def forward(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project WGS84 (lat, lon) degrees to local meters (x_east, y_north)."""
        lat = np.asarray(lat_deg, dtype=np.float64)
        lon = np.asarray(lon_deg, dtype=np.float64)
        scalar = lat.ndim == 0 and lon.ndim == 0
        lat_1d = np.atleast_1d(lat)
        lon_1d = np.atleast_1d(lon)

        phi = np.radians(np.clip(lat_1d, -89.9999, 89.9999))
        lam = np.radians(lon_1d)
        # Antimeridian-safe delta lambda in [-pi, pi]
        dlam = np.arctan2(np.sin(lam - self.lon0_rad), np.cos(lam - self.lon0_rad))

        if self.mode in ("aeqd", "enu"):
            chi = _conformal_lat(phi)
            dchi = chi - self._chi0
            sin_chi = np.sin(chi)
            cos_chi = np.cos(chi)
            sin_half_c2 = np.sin(0.5 * dchi) ** 2 + self._cos_chi0 * cos_chi * (np.sin(0.5 * dlam) ** 2)
            c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, sin_half_c2))))
            c = np.minimum(c, math.pi - 1e-7)
            k_prime = np.where(
                c < 1e-5,
                1.0 + (c * c) / 6.0 + (7.0 * (c ** 4)) / 360.0,
                c / np.sin(np.maximum(c, 1e-12)),
            )
            x = self._r_conf * k_prime * cos_chi * np.sin(dlam)
            y = self._r_conf * k_prime * (self._cos_chi0 * sin_chi - self._sin_chi0 * cos_chi * np.cos(dlam))
        else:
            sin_p = np.sin(phi)
            q = np.arcsinh(np.tan(phi)) - _SQRT_N_TERM * np.arctanh(_SQRT_N_TERM * sin_p)
            xi_p = np.arctan2(np.sinh(q), np.cos(dlam))
            eta_p = np.arctanh(np.sin(dlam) / np.cosh(q))

            xi = xi_p.copy()
            eta = eta_p.copy()
            for j, a_j in enumerate(_ALPHA, start=1):
                two_j = 2.0 * j
                xi += a_j * np.sin(two_j * xi_p) * np.cosh(two_j * eta_p)
                eta += a_j * np.cos(two_j * xi_p) * np.sinh(two_j * eta_p)

            x = _A_RECT * eta
            y = _A_RECT * xi - self._m0

        if scalar:
            return float(x[0]), float(y[0])
        return x.reshape(lat.shape), y.reshape(lon.shape)

    def inverse(self, x_m: np.ndarray, y_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse project local meters (x_east, y_north) back to WGS84 (lat, lon) degrees."""
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        scalar = x.ndim == 0 and y.ndim == 0
        x_1d = np.atleast_1d(x)
        y_1d = np.atleast_1d(y)

        if self.mode in ("aeqd", "enu"):
            rho = np.hypot(x_1d, y_1d)
            c = np.minimum(math.pi - 1e-9, rho / self._r_conf)
            sin_c = np.sin(c)
            cos_c = np.cos(c)
            safe_rho = np.maximum(rho, 1e-12)
            chi = np.where(
                rho < 1e-10,
                self._chi0,
                np.arcsin(np.clip(cos_c * self._sin_chi0 + (y_1d * sin_c * self._cos_chi0) / safe_rho, -1.0, 1.0)),
            )
            dlam = np.where(
                rho < 1e-10,
                0.0,
                np.arctan2(x_1d * sin_c, rho * self._cos_chi0 * cos_c - y_1d * self._sin_chi0 * sin_c),
            )
            phi = chi.copy()
            for j, d_j in enumerate(_DELTA, start=1):
                phi += d_j * np.sin(2.0 * j * chi)
        else:
            xi = (y_1d + self._m0) / _A_RECT
            eta = x_1d / _A_RECT

            xi_p = xi.copy()
            eta_p = eta.copy()
            for j, b_j in enumerate(_BETA, start=1):
                two_j = 2.0 * j
                xi_p -= b_j * np.sin(two_j * xi) * np.cosh(two_j * eta)
                eta_p -= b_j * np.cos(two_j * xi) * np.sinh(two_j * eta)

            chi = np.arcsin(np.clip(np.sin(xi_p) / np.cosh(eta_p), -1.0, 1.0))
            dlam = np.arctan2(np.sinh(eta_p), np.cos(xi_p))

            phi = chi.copy()
            for j, d_j in enumerate(_DELTA, start=1):
                phi += d_j * np.sin(2.0 * j * chi)

        lat_out = np.degrees(phi)
        lon_out = np.degrees(np.arctan2(np.sin(self.lon0_rad + dlam), np.cos(self.lon0_rad + dlam)))
        if scalar:
            return float(lat_out[0]), float(lon_out[0])
        return lat_out.reshape(x.shape), lon_out.reshape(y.shape)

    def metadata(self) -> Dict[str, Any]:
        """Return projection provenance dictionary for reproducibility manifests."""
        return {
            "mode": self.mode,
            "lat0_deg": round(self.lat0_deg, 6),
            "lon0_deg": round(self.lon0_deg, 6),
            "lon_span_deg": round(self.lon_span_deg, 4),
            "nearest_utm_epsg": self.nearest_utm_epsg,
            "proj4_string": self.proj4_string,
            "warnings": list(self.warnings),
        }
