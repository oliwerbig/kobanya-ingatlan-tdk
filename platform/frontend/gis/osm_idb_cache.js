/* IndexedDB cache (7-day TTL, auto-pruning & clean DB close) + .osm_bundle.json offline export/import */

const DB_NAME = "geospec_osm_cache_v1";
const STORE_NAME = "queries";
const TTL_MS = 7 * 24 * 60 * 60 * 1000;

function openDB() {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      return reject(new Error("IndexedDB unavailable"));
    }
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "key" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function getCachedOSM(key) {
  let db = null;
  try {
    db = await openDB();
    const rec = await new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const req = tx.objectStore(STORE_NAME).get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
      tx.onabort = () => resolve(null);
    });
    if (!rec) return null;
    if (Date.now() - rec.savedAt > TTL_MS) {
      // Prune expired entry asynchronously without blocking return
      const delTx = db.transaction(STORE_NAME, "readwrite");
      delTx.objectStore(STORE_NAME).delete(key);
      await new Promise((res) => {
        delTx.oncomplete = () => res();
        delTx.onerror = () => res();
      });
      return null;
    }
    return rec.payload;
  } catch {
    return null;
  } finally {
    if (db) {
      try {
        db.close();
      } catch {}
    }
  }
}

async function _putRecord(db, key, payload) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).put({ key, savedAt: Date.now(), payload });
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error || new Error("IndexedDB write failed"));
    tx.onabort = () => reject(tx.error || new Error("IndexedDB write aborted"));
  });
}

export async function setCachedOSM(key, payload) {
  let db = null;
  try {
    db = await openDB();
    try {
      await _putRecord(db, key, payload);
    } catch (writeErr) {
      db.close();
      db = null;
      await pruneExpiredOSM();
      try {
        db = await openDB();
        await _putRecord(db, key, payload);
      } catch {
        // Ignore storage quota errors cleanly after pruning
      }
    }
  } catch {
    // Ignore IndexedDB unavailability
  } finally {
    if (db) {
      try {
        db.close();
      } catch {}
    }
  }
}

export async function pruneExpiredOSM() {
  let db = null;
  let removed = 0;
  try {
    db = await openDB();
    const now = Date.now();
    await new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const req = store.openCursor();
      req.onsuccess = () => {
        const cursor = req.result;
        if (!cursor) return;
        const rec = cursor.value;
        if (!rec || now - rec.savedAt > TTL_MS) {
          cursor.delete();
          removed++;
        }
        cursor.continue();
      };
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    });
  } catch {
    // Ignore
  } finally {
    if (db) {
      try {
        db.close();
      } catch {}
    }
  }
  return removed;
}

export async function clearAllCachedOSM() {
  let db = null;
  try {
    db = await openDB();
    await new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      tx.objectStore(STORE_NAME).clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    });
  } catch {
    // No-op
  } finally {
    if (db) {
      try {
        db.close();
      } catch {}
    }
  }
}

export function buildOsmBundle({ bbox = null, targets = [], rawOsmTargets = {}, rawGeojsonTargets = {}, rawStreetGraphs = {} }) {
  const targetEntries = {};
  for (const tgt of targets) {
    const tid = tgt.id;
    if (rawGeojsonTargets[tid]) {
      targetEntries[tid] = {
        target_cfg: { ...tgt },
        source_kind: "geojson",
        payload: rawGeojsonTargets[tid],
      };
    } else if (rawOsmTargets[tid]) {
      targetEntries[tid] = {
        target_cfg: { ...tgt },
        source_kind: "osm",
        payload: rawOsmTargets[tid],
      };
    }
  }
  return {
    kind: "geospec_osm_bundle",
    schema: "geospec.osm_bundle.v1",
    created_utc: new Date().toISOString(),
    bbox,
    targets: targetEntries,
    street_graphs: { ...rawStreetGraphs },
  };
}

export function parseOsmBundle(rawObj) {
  if (!rawObj || typeof rawObj !== "object" || Array.isArray(rawObj)) {
    throw new Error("Érvénytelen .osm_bundle.json formátum.");
  }
  const safeDict = (src) => {
    const out = {};
    if (!src || typeof src !== "object" || Array.isArray(src)) return out;
    for (const [k, v] of Object.entries(src)) {
      if (k === "__proto__" || k === "constructor" || k === "prototype") continue;
      Object.defineProperty(out, k, { value: v, enumerable: true, writable: true, configurable: true });
    }
    return out;
  };
  const targets = safeDict(rawObj.targets);
  const streetGraphs = safeDict(rawObj.street_graphs);
  if (Object.keys(targets).length === 0 && Object.keys(streetGraphs).length === 0) {
    throw new Error("Az .osm_bundle.json nem tartalmaz térbeli réteget vagy utcahálózatot.");
  }
  return {
    kind: "geospec_osm_bundle",
    schema: rawObj.schema || "geospec.osm_bundle.v1",
    bbox: rawObj.bbox || null,
    targets,
    street_graphs: streetGraphs,
  };
}
