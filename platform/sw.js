/* GeoSpec Service Worker — caches Pyodide WASM + CDN assets (cache-first),
 * and full same-origin app shell + py_engine (network-first with offline fallback).
 */
const CACHE_NAME = "geospec-wasm-v5";

const CORE_SHELL_ASSETS = [
  "./",
  "./index.html",
  "./frontend/styles/theme.css",
  "./frontend/app.js",
  "./frontend/utils/sanitize.js",
  "./frontend/state/dag_client.js",
  "./frontend/state/recipe_store.js",
  "./frontend/gis/osm_idb_cache.js",
  "./frontend/gis/overpass_client.js",
  "./frontend/components/map_inspector.js",
  "./frontend/components/stage1_data_osm.js",
  "./frontend/components/stage2_composition.js",
  "./frontend/components/stage3_model_composer.js",
  "./frontend/components/stage4_5_spatial_gwr.js",
  "./worker/pyodide_worker.js",
  "./py_engine/__init__.py",
  "./py_engine/dag_cache.py",
  "./py_engine/projection.py",
  "./py_engine/spatial_gis.py",
  "./py_engine/design_matrix.py",
  "./py_engine/ols_kernel.py",
  "./py_engine/spatial_kernel.py",
  "./py_engine/gwr_kernel.py",
  "./py_engine/exporters.py",
  "./py_engine/engine.py",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(CORE_SHELL_ASSETS).catch(() => {}))
  );
  self.skipWaiting();
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

  const isCdn =
    url.includes("cdn.jsdelivr.net/pyodide/") ||
    url.includes("unpkg.com/maplibre-gl") ||
    url.includes("cdn.plot.ly");
  const isSameOriginApp = reqUrl.origin === self.location.origin;

  if (!isCdn && !isSameOriginApp) {
    return;
  }

  if (isSameOriginApp) {
    event.respondWith(
      caches.open(CACHE_NAME).then(async (cache) => {
        try {
          const resp = await fetch(event.request);
          if (resp && resp.status === 200) {
            cache.put(event.request, resp.clone());
          }
          return resp;
        } catch (err) {
          return (await cache.match(event.request)) || (await cache.match("./index.html")) || Response.error();
        }
      })
    );
    return;
  }

  event.respondWith(
    caches.open(CACHE_NAME).then(async (cache) => {
      const cached = await cache.match(event.request);
      if (cached) return cached;
      try {
        const resp = await fetch(event.request);
        if (resp && (resp.status === 200 || resp.type === "opaque")) {
          cache.put(event.request, resp.clone());
        }
        return resp;
      } catch (err) {
        return cached || Response.error();
      }
    })
  );
});
