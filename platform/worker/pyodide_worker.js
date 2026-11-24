/* GeoSpec Pyodide WebAssembly Worker
 * Loads Pyodide via SRI-verified fetch() + Blob URL, mounts NumPy + SciPy & py_engine,
 * and exposes the pure-Python GeoSpecSession via JSON-RPC with live progress callbacks.
 */

const PYODIDE_INDEX = "https://cdn.jsdelivr.net/pyodide/v0.27.4/full/";
const PYODIDE_JS_SRI_SHA384 = "tSKqtqt5i9t+BKiXGOGifQBi2UQGv0DKKTl7grFv4cS4c8bc/Zc+PI+1O2Bj+xtC";

let pyodide = null;
let initPromise = null;

const ENGINE_FILES = [
  "__init__.py",
  "dag_cache.py",
  "projection.py",
  "spatial_gis.py",
  "design_matrix.py",
  "ols_kernel.py",
  "spatial_kernel.py",
  "gwr_kernel.py",
  "exporters.py",
  "engine.py",
];

function arrayBufferToBase64(buf) {
  const bytes = new Uint8Array(buf);
  let bin = "";
  const chunk = 8192;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode(...bytes.subarray(i, Math.min(bytes.length, i + chunk)));
  }
  return btoa(bin);
}

async function loadSriVerifiedPyodideScript() {
  const resp = await fetch(PYODIDE_INDEX + "pyodide.js", { mode: "cors", credentials: "omit" });
  if (!resp.ok) {
    throw new Error(`Failed to fetch pyodide.js: HTTP ${resp.status}`);
  }
  const buf = await resp.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-384", buf);
  const actualB64 = arrayBufferToBase64(digest);
  if (actualB64 !== PYODIDE_JS_SRI_SHA384) {
    throw new Error(`SRI mismatch for pyodide.js: expected sha384-${PYODIDE_JS_SRI_SHA384}, got sha384-${actualB64}`);
  }
  const blobUrl = URL.createObjectURL(new Blob([buf], { type: "application/javascript" }));
  try {
    importScripts(blobUrl);
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

async function initPyodideEngine() {
  if (initPromise) return initPromise;
  initPromise = (async () => {
    try {
      self.postMessage({ type: "boot_status", status: "loading_runtime", message: "Loading WebAssembly Python runtime..." });
      await loadSriVerifiedPyodideScript();
      pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX });

      self.postMessage({ type: "boot_status", status: "loading_packages", message: "Loading NumPy & SciPy kernels..." });
      await pyodide.loadPackage(["numpy", "scipy"]);

      self.postMessage({ type: "boot_status", status: "loading_engine", message: "Mounting GeoSpec py_engine..." });
      pyodide.FS.mkdirTree("/geospec/py_engine");
      const baseEngineUrl = new URL("../py_engine/", self.location.href).href;

      await Promise.all(
        ENGINE_FILES.map(async (fname) => {
          const resp = await fetch(baseEngineUrl + fname);
          if (!resp.ok) throw new Error(`Failed to fetch ${fname}: HTTP ${resp.status}`);
          const text = await resp.text();
          pyodide.FS.writeFile(`/geospec/py_engine/${fname}`, text);
        })
      );

      self._postProgress = (pct, msg) => {
        self.postMessage({
          type: "progress",
          stage: "stage5",
          pct: Number(pct) || 0,
          message: String(msg || ""),
        });
      };

      await pyodide.runPythonAsync(`
import sys, json, math, base64
import js
if "/geospec" not in sys.path:
    sys.path.insert(0, "/geospec")
from py_engine import GeoSpecSession
_SESSION = GeoSpecSession()

def _json_safe(obj):
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "item") and callable(obj.item):
        return _json_safe(obj.item())
    if hasattr(obj, "tolist"):
        return _json_safe(obj.tolist())
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj

def _gwr_progress_cb(pct, msg=""):
    try:
        js.self._postProgress(float(pct), str(msg))
    except Exception:
        pass

def _rpc_dispatch(method, payload_json):
    payload = json.loads(payload_json) if payload_json else {}
    if method == "load_dataset":
        res = _SESSION.load_dataset(payload["columns"], payload["lat_col"], payload["lon_col"])
    elif method == "reset_session":
        res = _SESSION.reset_session()
    elif method == "ingest_target_raw":
        res = _SESSION.ingest_target_raw(
            payload["target_id"], payload["raw_payload"], payload.get("source_kind", "osm"), payload.get("excluded_osm_ids")
        )
    elif method == "remove_target":
        res = _SESSION.remove_target(payload["target_id"])
    elif method == "ingest_street_graph":
        res = _SESSION.ingest_street_graph(payload["profile"], payload["graph_json"])
    elif method == "run_stage_1":
        res = _SESSION.run_stage_1_distances(payload["recipe"], int(payload.get("run_id", 1)))
    elif method == "get_single_route":
        res = _SESSION.get_single_point_route(int(payload["obs_idx"]), payload["target_id"], payload["recipe"])
    elif method == "run_stage_2":
        res = _SESSION.run_stage_2_descriptives_and_composition(payload["recipe"], int(payload.get("run_id", 1)))
    elif method == "run_stage_3":
        res = _SESSION.run_stage_3_ols_workbench(payload["recipe"], payload.get("cov_type", "HC3"), int(payload.get("run_id", 1)))
    elif method == "run_robustness_matrix":
        res = _SESSION.run_robustness_matrix(payload["recipe"], int(payload.get("run_id", 1)))
    elif method == "run_stage_4":
        res = _SESSION.run_stage_4_spatial_econometrics(payload["recipe"], int(payload.get("run_id", 1)))
    elif method == "run_stage_5":
        res = _SESSION.run_stage_5_gwr(payload["recipe"], int(payload.get("run_id", 1)), progress_cb=_gwr_progress_cb)
    elif method == "generate_m1_m4":
        res = _SESSION.generate_m1_m4(
            payload["covariates"], payload["spatial_targets"], payload.get("default_rings"), float(payload.get("default_cutoff_m", 300.0))
        )
    elif method == "export_latex":
        res = _SESSION.export_latex_only(payload["recipe"], payload.get("kind", "ols_main"))
    elif method == "export_geojson":
        res = _SESSION.export_geojson_only(payload["recipe"])
    elif method == "export_bundle":
        out = _SESSION.export_publication_bundle(payload["recipe"], payload.get("osm_bundle"), payload.get("svg_figures"))
        zb = out.pop("zip_bytes", b"")
        out["zip_base64"] = base64.b64encode(zb).decode("ascii")
        res = out
    else:
        raise ValueError(f"Unknown RPC method: {method}")
    return json.dumps(_json_safe(res), allow_nan=False)
`);
      self.postMessage({ type: "boot_status", status: "ready", message: "Pyodide NumPy+SciPy kernel ready" });
    } catch (err) {
      initPromise = null;
      pyodide = null;
      self.postMessage({ type: "boot_status", status: "error", message: String(err?.message || err) });
      throw err;
    }
  })();
  return initPromise;
}

self.onmessage = async (event) => {
  const { id, type, method, payload } = event.data;
  if (type === "init") {
    try {
      await initPyodideEngine();
      self.postMessage({ id, ok: true });
    } catch (err) {
      self.postMessage({ id, ok: false, error: err.message });
    }
    return;
  }

  if (type === "rpc") {
    try {
      await initPyodideEngine();
      const fn = pyodide.globals.get("_rpc_dispatch");
      const resultJson = fn(method, JSON.stringify(payload || {}));
      const result = JSON.parse(resultJson);
      self.postMessage({ id, ok: true, result });
    } catch (err) {
      self.postMessage({ id, ok: false, error: String(err.message || err) });
    }
  }
};
