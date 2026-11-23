"""
GeoSpec - Parametric Spatial Econometric & OSM Engine (Pure NumPy + SciPy)
=========================================================================
Zero-backend, WebAssembly-compatible computational kernel implementing:
- Local Krüger Transverse Mercator & ENU projections (projection.py)
- Parametric OSM / GeoJSON geometry & CSR multi-source Dijkstra GIS (spatial_gis.py)
- Dynamic formula design matrix, Box-Cox, Guard #2 sparse merger (design_matrix.py)
- OLS/HC3/CR1, Kennedy/HP, AD profile-MLE, Gelbach OVB, Heckman V (ols_kernel.py)
- Sparse CSR W, Moran/LISA, Anselin/Elhorst, ML SAR/SEM/SDM, Krinsky-Robb (spatial_kernel.py)
- Singularity-guarded batched Adaptive Bisquare GWR & Leung/IQR tests (gwr_kernel.py)
- Overleaf booktabs LaTeX, RFC-7946 QGIS GeoJSON & recipe exporters (exporters.py)
"""

__version__ = "1.0.0"

from .dag_cache import DAGCache, CancellationError, hash_payload
from .projection import LocalMetricProjection, compute_circular_centroid
from .engine import GeoSpecSession

__all__ = [
    "__version__",
    "DAGCache",
    "CancellationError",
    "hash_payload",
    "LocalMetricProjection",
    "compute_circular_centroid",
    "GeoSpecSession",
]
