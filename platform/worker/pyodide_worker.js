/* GeoSpec Pyodide WebAssembly Worker
 * Loads Pyodide + NumPy + SciPy cleanly in a background thread and exposes
 * the pure-Python `py_engine` session via JSON RPC with run_id cancellation.
 */

const PYODIDE_INDEX = "https://cdn.jsdelivr.net/pyodide/v0.27.4/full/";
importScripts(PYODIDE_INDEX + "pyodide.js");

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

async function initPyodideEngine() {
  if (initPromise) return initPromise;
  initPromise = (async () => {
    self.postMessage({ type: "boot_status", status: "loading_runtime", message: "Loading WebAssembly Python runtime..." });
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

    await pyodide.runPythonAsync(`
import sys, json, math
if "/geospec" not in sys.path:
    sys.path.insert(0, "/geospec")
from py_engine import GeoSpecSession
_SESSION = GeoSpecSession()

def _json_safe(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj

def _rpc_dispatch(method, payload_json):
    payload = json.loads(payload_json) if payload_json else {}
    if method == "load_dataset":
        res = _SESSION.load_dataset(payload["columns"], payload["lat_col"], payload["lon_col"])
    elif method == "ingest_target_raw":
        res = _SESSION.ingest_target_raw(
            payload["target_id"], payload["raw_payload"], payload.get("source_kind", "osm"), payload.get("excluded_osm_ids")
        )
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
    elif method == "run_stage_4":
        res = _SESSION.run_stage_4_spatial_econometrics(payload["recipe"], int(payload.get("run_id", 1)))
    elif method == "run_stage_5":
        res = _SESSION.run_stage_5_gwr(payload["recipe"], int(payload.get("run_id", 1)))
    elif method == "generate_m1_m4":
        res = _SESSION.generate_m1_m4(
            payload["covariates"], payload["spatial_targets"], payload.get("default_rings"), float(payload.get("default_cutoff_m", 300.0))
        )
    elif method == "export_bundle":
        import base64
        out = _SESSION.export_publication_bundle(payload["recipe"])
        zb = out.pop("zip_bytes", b"")
        out["zip_base64"] = base64.b64encode(zb).decode("ascii")
        res = out
    else:
        raise ValueError(f"Unknown RPC method: {method}")
    return json.dumps(_json_safe(res), allow_nan=False)
`);
    self.postMessage({ type: "boot_status", status: "ready", message: "Pyodide NumPy+SciPy kernel ready" });
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
