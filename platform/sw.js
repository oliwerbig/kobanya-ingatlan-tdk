/* GeoSpec Service Worker — caches Pyodide WASM + pinned CDN assets cleanly (never opaque),
 * and full same-origin app shell + py_engine (network-first with timeout & offline/5xx fallback).
 */
const CACHE_NAME = "geospec-wasm-v13";
const FETCH_TIMEOUT_MS = 10000;

const CORE_SHELL_ASSETS = [
  "./",
  "./index.html",
  "./frontend/styles/theme.css",
  "./frontend/app.js",
  "./frontend/utils/sanitize.js",
  "./frontend/utils/i18n.js",
  "./frontend/state/dag_client.js",
  "./frontend/state/recipe_store.js",
  "./frontend/gis/osm_idb_cache.js",
  "./frontend/gis/overpass_client.js",
  "./frontend/components/map_inspector.js",
  "./frontend/components/stage1_data_osm.js",
  "./frontend/components/stage2_composition.js",
  "./frontend/components/stage3_model_composer.js",
  "./frontend/components/stage4_5_spatial_gwr.js",
  "./frontend/components/stage6_ml_shap.js",
  "./worker/pyodide_worker.js",
  "./py_engine/__init__.py",
  "./py_engine/dag_cache.py",
  "./py_engine/projection.py",
  "./py_engine/spatial_gis.py",
  "./py_engine/design_matrix.py",
  "./py_engine/ols_kernel.py",
  "./py_engine/spatial_kernel.py",
  "./py_engine/gwr_kernel.py",
  "./py_engine/ml_shap_kernel.py",
  "./py_engine/exporters.py",
  "./py_engine/engine.py",
];

const CDN_SHELL_ASSETS = [
  "https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css",
  "https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js",
  "https://cdn.plot.ly/plotly-2.35.2.min.js",
  "https://cdn.jsdelivr.net/pyodide/v0.27.4/full/pyodide.js",
];

const ALLOWED_CDN_PREFIXES = [
  "https://cdn.jsdelivr.net/pyodide/v0.27.4/full/",
  "https://unpkg.com/maplibre-gl@4.7.1/dist/",
  "https://cdn.plot.ly/plotly-2.35.2.min.js",
];

function isAllowedCdnUrl(url) {
  return ALLOWED_CDN_PREFIXES.some((prefix) => url === prefix || url.startsWith(prefix));
}

async function fetchWithTimeout(request, ms = FETCH_TIMEOUT_MS, init = undefined) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  try {
    return await fetch(request, init ? { ...init, signal: ctrl.signal } : { signal: ctrl.signal });
  } finally {
    clearTimeout(timer);
  }
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      await cache.addAll(CORE_SHELL_ASSETS);
      await Promise.allSettled(
        CDN_SHELL_ASSETS.map(async (url) => {
          try {
            const resp = await fetchWithTimeout(url, FETCH_TIMEOUT_MS, { mode: "cors", credentials: "omit" });
            if (resp && resp.status === 200 && resp.type !== "opaque") {
              await cache.put(url, resp);
            }
          } catch (err) {
            console.warn("SW CDN precache skip:", url, err);
          }
        })
      );
      await self.skipWaiting();
    })()
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    Promise.all([
      caches.keys().then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
      ),
      self.clients.claim(),
    ])
  );
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  const reqUrl = new URL(event.request.url);
  const url = event.request.url;

  const isCdn = isAllowedCdnUrl(url);
  const isSameOriginApp = reqUrl.origin === self.location.origin;

  if (!isCdn && !isSameOriginApp) {
    return;
  }

  if (isSameOriginApp) {
    event.respondWith(
      caches.open(CACHE_NAME).then(async (cache) => {
        try {
          const resp = await fetchWithTimeout(event.request, FETCH_TIMEOUT_MS);
          if (resp && resp.status === 200 && resp.type !== "opaque") {
            cache.put(event.request, resp.clone());
            return resp;
          }
          if (resp && resp.status >= 500) {
            const cached5 = await cache.match(event.request);
            if (cached5) return cached5;
          }
          return resp;
        } catch (err) {
          const cached = await cache.match(event.request);
          if (cached) return cached;
          if (event.request.mode === "navigate") {
            return (await cache.match("./index.html")) || Response.error();
          }
          return Response.error();
        }
      })
    );
    return;
  }

  event.respondWith(
    caches.open(CACHE_NAME).then(async (cache) => {
      const cached = await cache.match(event.request);
      if (cached && cached.type !== "opaque") return cached;
      try {
        const resp = await fetchWithTimeout(event.request, 30000);
        if (resp && resp.status === 200 && resp.type !== "opaque") {
          cache.put(event.request, resp.clone());
        }
        return resp;
      } catch (err) {
        return (cached && cached.type !== "opaque") ? cached : Response.error();
      }
    })
  );
});
