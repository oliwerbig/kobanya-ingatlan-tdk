/* GeoSpec Service Worker — caches Pyodide WASM + NumPy/SciPy assets for fast repeat boots */
const CACHE_NAME = "geospec-wasm-v1";

self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = event.request.url;
  // Cache Pyodide CDN WASM/wheels and static GeoSpec py_engine files
  const isCacheable =
    url.includes("cdn.jsdelivr.net/pyodide/") ||
    url.includes("/py_engine/") ||
    url.includes("unpkg.com/maplibre-gl") ||
    url.includes("cdn.plot.ly");

  if (!isCacheable || event.request.method !== "GET") {
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
