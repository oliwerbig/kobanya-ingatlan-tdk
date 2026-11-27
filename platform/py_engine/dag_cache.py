"""
Content-addressed DAG memoization cache and cooperative cancellation helpers.
Prevents recomputing OSM geometries, Dijkstra graphs, or W eigenvalues when
editing model specifications or dragging Andersson-Day threshold sliders.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
from typing import Any, Callable, Dict, Optional


class CancellationError(Exception):
    """Raised when an active computation is superseded by a newer run_id."""


def _canonicalize(obj: Any) -> Any:
    """Convert Python/NumPy objects into deterministic JSON-serializable structures."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "__NaN__"
        if math.isinf(obj):
            return "__Inf__" if obj > 0 else "__-Inf__"
        return 0.0 if obj == 0.0 else round(obj, 12)
    if hasattr(obj, "tolist"):
        return _canonicalize(obj.tolist())
    if hasattr(obj, "item") and callable(obj.item):
        return _canonicalize(obj.item())
    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_canonicalize(v) for v in obj)
    return str(obj)


def hash_payload(obj: Any) -> str:
    """Return deterministic SHA-256 hex digest of any nested payload."""
    canonical = _canonicalize(obj)
    serialized = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class DAGCache:
    """Layered content-addressed LRU memoization cache for GeoSpec computation nodes."""

    LAYERS = ("geo", "osm", "graph", "dist", "weights", "design", "ols", "spatial", "gwr", "ml_shap")
    HEAVY_LAYERS = frozenset({"dist", "weights", "spatial", "gwr", "ml_shap"})
    DEFAULT_MAX_ENTRIES = 16
    HEAVY_MAX_ENTRIES = 8

    def __init__(self) -> None:
        self._store: Dict[str, OrderedDict[str, Any]] = {layer: OrderedDict() for layer in self.LAYERS}
        self._active_run_id: int = 0

    def start_run(self, run_id: int) -> None:
        """Register the latest active run_id for cooperative cancellation."""
        self._active_run_id = int(run_id)

    def check_cancelled(self, run_id: Optional[int]) -> None:
        """Raise CancellationError if run_id has been superseded.

        Note: In synchronous Pyodide Web Worker RPC execution, ``dag_client.js``
        also performs post-hoc stale-response filtering by ``run_id`` when a newer
        RPC finishes before or supersedes an older queued call.
        """
        if run_id is not None and int(run_id) < self._active_run_id:
            raise CancellationError(f"Run {run_id} superseded by {self._active_run_id}")

    def get(self, layer: str, key: str) -> Optional[Any]:
        """Retrieve cached artifact if present and refresh its LRU recency."""
        bucket = self._store.get(layer)
        if bucket is None or key not in bucket:
            return None
        bucket.move_to_end(key)
        return bucket[key]

    def set(self, layer: str, key: str, value: Any) -> Any:
        """Store artifact under layer and key with bounded LRU eviction."""
        if layer not in self._store:
            self._store[layer] = OrderedDict()
        bucket = self._store[layer]
        bucket[key] = value
        bucket.move_to_end(key)
        cap = self.HEAVY_MAX_ENTRIES if layer in self.HEAVY_LAYERS else self.DEFAULT_MAX_ENTRIES
        while len(bucket) > cap:
            bucket.popitem(last=False)
        return value

    def get_or_compute(
        self,
        layer: str,
        key: str,
        fn: Callable[[], Any],
        run_id: Optional[int] = None,
    ) -> Any:
        """Return cached value or execute fn and store result."""
        self.check_cancelled(run_id)
        cached = self.get(layer, key)
        if cached is not None:
            return cached
        val = fn()
        self.check_cancelled(run_id)
        return self.set(layer, key, val)

    def invalidate_from(self, layer: str) -> None:
        """Clear a layer and all downstream layers."""
        if layer not in self.LAYERS:
            return
        idx = self.LAYERS.index(layer)
        for name in self.LAYERS[idx:]:
            self._store[name].clear()

    def clear_all(self) -> None:
        """Reset the entire DAG cache."""
        for name in self.LAYERS:
            self._store[name].clear()
