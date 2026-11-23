/* GeoSpec Service Worker — caches Pyodide WASM + CDN assets, network-first for local py_engine */
const CACHE_NAME = "geospec-wasm-v2";

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
  const isCdn =
    url.includes("cdn.jsdelivr.net/pyodide/") ||
    url.includes("unpkg.com/maplibre-gl") ||
    url.includes("cdn.plot.ly");
  const isPyEngine = url.includes("/py_engine/");

  if ((!isCdn && !isPyEngine) || event.request.method !== "GET") {
    return;
  }

  if (isPyEngine) {
    event.respondWith(
      caches.open(CACHE_NAME).then(async (cache) => {
        try {
          const resp = await fetch(event.request);
          if (resp && resp.status === 200) {
            cache.put(event.request, resp.clone());
          }
          return resp;
        } catch (err) {
          return (await cache.match(event.request)) || Response.error();
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
