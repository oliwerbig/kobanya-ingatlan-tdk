/* Browser-safe OSM client with Geofabrik Postpass primary SQL->GeoJSON engine,
 * fast-fail Overpass CORS mirror pool fallback, 2x2 auto-tiling, and IndexedDB caching.
 */
import { getCachedOSM, setCachedOSM } from "./osm_idb_cache.js";

const BACKEND_CACHE_VERSION = "v2";
const POSTPASS_URL = "https://postpass.geofabrik.de/api/interpreter";

const OVERPASS_MIRRORS = [
  "https://overpass-api.de/api/interpreter",
  "https://lz4.overpass-api.de/api/interpreter",
  "https://overpass.private.coffee/api/interpreter",
  "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
];

export const OSM_TARGET_PRESETS = [
  {
    preset_id: "rail",
    label: "Vasúti vágányok",
    osm_filter: [{ railway: ["rail", "light_rail"] }],
    element_types: ["way", "relation"],
    geometry_handling: "line_segment",
    inside_is_zero: false,
  },
  {
    preset_id: "transit_stops",
    label: "Vasútállomás / Tranzit megálló",
    osm_filter: [
      { railway: ["station", "halt", "tram_stop", "subway_entrance"] },
      { public_transport: ["station", "stop_position"] },
    ],
    element_types: ["node", "way"],
    geometry_handling: "point",
    inside_is_zero: false,
  },
  {
    preset_id: "major_roads",
    label: "Főútvonalak",
    osm_filter: [{ highway: ["motorway", "trunk", "primary", "secondary"] }],
    element_types: ["way"],
    geometry_handling: "line_segment",
    inside_is_zero: false,
  },
  {
    preset_id: "parks",
    label: "Parkok & Zöldfelületek",
    osm_filter: [
      { leisure: ["park", "garden", "nature_reserve"] },
      { landuse: ["recreation_ground", "village_green", "forest"] },
    ],
    element_types: ["way", "relation"],
    geometry_handling: "polygon_boundary",
    inside_is_zero: true,
  },
  {
    preset_id: "water",
    label: "Vízpart / Folyó / Tó",
    osm_filter: [
      { natural: ["water"] },
      { waterway: ["river", "canal", "riverbank"] },
    ],
    element_types: ["way", "relation"],
    geometry_handling: "polygon_boundary",
    inside_is_zero: true,
  },
  {
    preset_id: "schools_health",
    label: "Iskola / Egyetem / Kórház",
    osm_filter: [{ amenity: ["school", "university", "college", "hospital", "clinic"] }],
    element_types: ["node", "way", "relation"],
    geometry_handling: "polygon_boundary",
    inside_is_zero: true,
  },
  {
    preset_id: "industrial",
    label: "Ipari zóna / Barnamező",
    osm_filter: [{ landuse: ["industrial", "brownfield", "railway"] }],
    element_types: ["way", "relation"],
    geometry_handling: "polygon_boundary",
    inside_is_zero: true,
  },
];

function wrapLongitude(lon) {
  let x = ((Number(lon) + 180) % 360 + 360) % 360 - 180;
  if (x === -180 && Number(lon) > 0) x = 180;
  return Math.max(-179.999999, Math.min(179.999999, x));
}

export function padBbox(bbox, bufferM = 1500) {
  const latMid = 0.5 * (bbox.south + bbox.north);
  const dLat = bufferM / 111132.95;
  const dLon = bufferM / (111132.95 * Math.max(0.15, Math.cos((latMid * Math.PI) / 180)));
  return {
    south: Math.max(-89.9, bbox.south - dLat),
    west: wrapLongitude(bbox.west - dLon),
    north: Math.min(89.9, bbox.north + dLat),
    east: wrapLongitude(bbox.east + dLon),
  };
}

export function parseTagShorthand(shorthand) {
  // Supports "leisure=park;landuse=forest" or "amenity=school&operator=public"
  return shorthand
    .split(";")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((clause) => {
      const obj = {};
      for (const kv of clause.split("&")) {
        const [k, v] = kv.split("=").map((s) => (s || "").trim());
        if (k) obj[k] = v ? v.split("|").map((s) => s.trim()) : ["*"];
      }
      return obj;
    })
    .filter((o) => Object.keys(o).length > 0);
}

export function buildTargetOverpassQL(target, paddedBbox) {
  const b = `${paddedBbox.south.toFixed(6)},${paddedBbox.west.toFixed(6)},${paddedBbox.north.toFixed(6)},${paddedBbox.east.toFixed(6)}`;
  const types = target.element_types && target.element_types.length ? target.element_types : ["node", "way", "relation"];
  const clauses = [];

  for (const clause of target.osm_filter || []) {
    let filterExpr = "";
    for (const [k, vals] of Object.entries(clause)) {
      if (!vals || vals.length === 0 || (vals.length === 1 && vals[0] === "*")) {
        filterExpr += `["${k}"]`;
      } else if (vals.length === 1) {
        filterExpr += `["${k}"="${vals[0]}"]`;
      } else {
        filterExpr += `["${k}"~"^(${vals.join("|")})$"]`;
      }
    }
    for (const et of types) {
      const relFilter = et === "relation" ? '["type"="multipolygon"]' : "";
      clauses.push(`  ${et}${relFilter}${filterExpr}(${b});`);
    }
  }
  const outMode = target.geometry_handling === "point" ? "out center;" : "out geom;";
  return `[out:json][timeout:30];\n(\n${clauses.join("\n")}\n);\n${outMode}`;
}

export function buildStreetGraphOverpassQL(paddedBbox, profile = "walk") {
  const b = `${paddedBbox.south.toFixed(6)},${paddedBbox.west.toFixed(6)},${paddedBbox.north.toFixed(6)},${paddedBbox.east.toFixed(6)}`;
  if (profile === "drive") {
    return `[out:json][timeout:30];\nway["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|unclassified|residential|living_street|service)$"]["access"!="private"](${b});\nout tags;\n>;\nout skel qt;`;
  }
  if (profile === "bike") {
    return `[out:json][timeout:30];\nway["highway"~"^(cycleway|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|unclassified|residential|living_street|path|track|service)$"]["bicycle"!="no"]["access"!="private"]["service"!~"^(parking_aisle|driveway)$"](${b});\nout tags;\n>;\nout skel qt;`;
  }
  // Undirected pedestrian walk network: minimal out skel qt payload (60-80% smaller)
  return `[out:json][timeout:30];\nway["highway"~"^(primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|unclassified|residential|living_street|pedestrian|footway|path|cycleway|steps|service)$"]["foot"!="no"]["access"!="private"]["service"!~"^(parking_aisle|driveway)$"](${b});\n(._;>;);\nout skel qt;`;
}

function sqlLit(s) {
  return `'${String(s).replace(/'/g, "''")}'`;
}

export function buildTargetPostpassSQL(target, paddedBbox) {
  const env = `st_makeenvelope(${paddedBbox.west.toFixed(6)}, ${paddedBbox.south.toFixed(6)}, ${paddedBbox.east.toFixed(6)}, ${paddedBbox.north.toFixed(6)}, 4326)`;
  const types = target.element_types && target.element_types.length ? target.element_types : ["node", "way", "relation"];
  const osmTypes = [];
  if (types.includes("node")) osmTypes.push("'N'");
  if (types.includes("way")) osmTypes.push("'W'");
  if (types.includes("relation")) osmTypes.push("'R'");
  const typeClause = osmTypes.length ? `osm_type IN (${osmTypes.join(",")})` : "FALSE";

  let table;
  if (types.length === 1 && types[0] === "node") {
    table = "postpass_point";
  } else if (types.includes("node")) {
    table = "postpass_pointlinepolygon";
  } else {
    table = "postpass_linepolygon";
  }

  // Never clip target polygons with ST_Intersection(geom, env): avoids bbox-edge boundary artifacts & shifted centroids
  const geomExpr = target.geometry_handling === "point" ? "ST_Centroid(geom) AS geom" : "geom AS geom";

  const orClauses = [];
  for (const clause of target.osm_filter || []) {
    const andParts = [];
    for (const [k, vals] of Object.entries(clause)) {
      const kLit = sqlLit(k);
      if (!vals || vals.length === 0 || (vals.length === 1 && vals[0] === "*")) {
        andParts.push(`tags ? ${kLit}`);
      } else if (vals.length === 1) {
        andParts.push(`tags->>${kLit} = ${sqlLit(vals[0])}`);
      } else {
        andParts.push(`tags->>${kLit} IN (${vals.map(sqlLit).join(",")})`);
      }
    }
    if (andParts.length) {
      orClauses.push(andParts.length === 1 ? andParts[0] : `(${andParts.join(" AND ")})`);
    }
  }
  const whereExpr = orClauses.length ? orClauses.join(" OR ") : "FALSE";
  return `SELECT osm_type, osm_id, tags, ${geomExpr} FROM ${table} WHERE geom && ${env} AND ${typeClause} AND (${whereExpr})`;
}

export function postpassToOverpassTarget(fc) {
  const elements = [];
  const feats = fc?.features || [];
  const toGeomPts = (ring, closeRing = false) => {
    const pts = (ring || [])
      .filter((c) => Array.isArray(c) && c.length >= 2)
      .map((c) => ({ lon: Number(c[0]), lat: Number(c[1]) }));
    if (closeRing && pts.length >= 3) {
      const f = pts[0];
      const l = pts[pts.length - 1];
      if (Math.abs(f.lon - l.lon) > 1e-9 || Math.abs(f.lat - l.lat) > 1e-9) {
        pts.push({ lon: f.lon, lat: f.lat });
      }
    }
    return pts;
  };

  for (let idx = 0; idx < feats.length; idx++) {
    const feat = feats[idx];
    const props = feat?.properties || {};
    const rawType = props.osm_type;
    const elType =
      rawType === "N" || rawType === "node"
        ? "node"
        : rawType === "R" || rawType === "relation"
          ? "relation"
          : "way";
    const osmId = props.osm_id ?? idx + 1;
    const tags = props.tags || {};
    let subIdx = 0;
    const nextId = () => (subIdx++ === 0 ? osmId : `${osmId}#${subIdx}`);

    const unpackGeom = (g) => {
      if (!g || !g.type) return;
      if (g.type === "GeometryCollection") {
        for (const sub of g.geometries || []) unpackGeom(sub);
      } else if (g.type === "Point") {
        const c = g.coordinates;
        if (Array.isArray(c) && c.length >= 2) {
          const lon = Number(c[0]);
          const lat = Number(c[1]);
          elements.push({ type: elType, id: nextId(), lat, lon, center: { lat, lon }, tags });
        }
      } else if (g.type === "MultiPoint") {
        for (const c of g.coordinates || []) {
          if (Array.isArray(c) && c.length >= 2) {
            const lon = Number(c[0]);
            const lat = Number(c[1]);
            elements.push({ type: elType, id: nextId(), lat, lon, center: { lat, lon }, tags });
          }
        }
      } else if (g.type === "LineString") {
        const pts = toGeomPts(g.coordinates, false);
        if (pts.length >= 2) {
          if (elType === "relation") {
            elements.push({
              type: "relation",
              id: nextId(),
              tags,
              members: [{ type: "way", role: "outer", geometry: pts }],
            });
          } else {
            elements.push({ type: "way", id: nextId(), tags, geometry: pts });
          }
        }
      } else if (g.type === "MultiLineString") {
        if (elType === "relation") {
          const members = (g.coordinates || [])
            .map((line) => toGeomPts(line, false))
            .filter((pts) => pts.length >= 2)
            .map((pts) => ({ type: "way", role: "outer", geometry: pts }));
          if (members.length) {
            elements.push({ type: "relation", id: nextId(), tags, members });
          }
        } else {
          for (const line of g.coordinates || []) {
            const pts = toGeomPts(line, false);
            if (pts.length >= 2) {
              elements.push({ type: "way", id: nextId(), tags, geometry: pts });
            }
          }
        }
      } else if (g.type === "Polygon") {
        const rings = g.coordinates || [];
        if (elType === "way" && rings.length === 1) {
          const pts = toGeomPts(rings[0], true);
          if (pts.length >= 4) {
            elements.push({ type: "way", id: nextId(), tags, geometry: pts });
          }
        } else {
          const members = [];
          rings.forEach((ring, rIdx) => {
            const pts = toGeomPts(ring, true);
            if (pts.length >= 4) {
              members.push({
                type: "way",
                role: rIdx === 0 ? "outer" : "inner",
                geometry: pts,
              });
            }
          });
          if (members.length) {
            elements.push({ type: "relation", id: nextId(), tags: { ...tags, type: tags.type || "multipolygon" }, members });
          }
        }
      } else if (g.type === "MultiPolygon") {
        const members = [];
        for (const poly of g.coordinates || []) {
          (poly || []).forEach((ring, rIdx) => {
            const pts = toGeomPts(ring, true);
            if (pts.length >= 4) {
              members.push({
                type: "way",
                role: rIdx === 0 ? "outer" : "inner",
                geometry: pts,
              });
            }
          });
        }
        if (members.length) {
          elements.push({ type: "relation", id: nextId(), tags: { ...tags, type: tags.type || "multipolygon" }, members });
        }
      }
    };

    unpackGeom(feat?.geometry);
  }
  return { version: 0.6, generator: "Postpass", elements };
}

export function buildStreetGraphPostpassSQL(paddedBbox, profile = "walk") {
  const env = `st_makeenvelope(${paddedBbox.west.toFixed(6)}, ${paddedBbox.south.toFixed(6)}, ${paddedBbox.east.toFixed(6)}, ${paddedBbox.north.toFixed(6)}, 4326)`;
  let filterExpr;
  if (profile === "drive") {
    filterExpr =
      "tags->>'highway' IN ('motorway','motorway_link','trunk','trunk_link','primary','primary_link','secondary','secondary_link','tertiary','tertiary_link','unclassified','residential','living_street','service') AND COALESCE(tags->>'access','yes') <> 'private'";
  } else if (profile === "bike") {
    filterExpr =
      "tags->>'highway' IN ('cycleway','primary','primary_link','secondary','secondary_link','tertiary','tertiary_link','unclassified','residential','living_street','path','track','service') AND COALESCE(tags->>'bicycle','yes') <> 'no' AND COALESCE(tags->>'access','yes') <> 'private' AND COALESCE(tags->>'service','') NOT IN ('parking_aisle','driveway')";
  } else {
    filterExpr =
      "tags->>'highway' IN ('primary','primary_link','secondary','secondary_link','tertiary','tertiary_link','unclassified','residential','living_street','pedestrian','footway','path','cycleway','steps','service') AND COALESCE(tags->>'foot','yes') <> 'no' AND COALESCE(tags->>'access','yes') <> 'private' AND COALESCE(tags->>'service','') NOT IN ('parking_aisle','driveway')";
  }
  return `SELECT osm_id, tags, geom AS geom FROM postpass_line WHERE geom && ${env} AND ${filterExpr}`;
}

export function postpassToOverpassStreetGraph(fc) {
  const coordToId = new Map();
  const nodeElements = [];
  const wayElements = [];
  let nextNodeId = 1;
  let nextWayId = 1;

  const addLine = (coords, tags) => {
    if (!Array.isArray(coords) || coords.length < 2) return;
    const nids = [];
    for (const c of coords) {
      if (!Array.isArray(c) || c.length < 2) continue;
      const rLon = Number(Number(c[0]).toFixed(6));
      const rLat = Number(Number(c[1]).toFixed(6));
      const key = `${rLon.toFixed(6)},${rLat.toFixed(6)}`;
      let nid = coordToId.get(key);
      if (nid === undefined) {
        nid = nextNodeId++;
        coordToId.set(key, nid);
        nodeElements.push({ type: "node", id: nid, lon: rLon, lat: rLat });
      }
      nids.push(nid);
    }
    if (nids.length >= 2) {
      wayElements.push({ type: "way", id: nextWayId++, nodes: nids, tags });
    }
  };

  const unpackLineGeom = (g, tags) => {
    if (!g || !g.type) return;
    if (g.type === "LineString") addLine(g.coordinates, tags);
    else if (g.type === "MultiLineString") {
      for (const line of g.coordinates || []) addLine(line, tags);
    } else if (g.type === "GeometryCollection") {
      for (const sub of g.geometries || []) unpackLineGeom(sub, tags);
    }
  };

  for (const feat of fc?.features || []) {
    const tags = feat?.properties?.tags || {};
    unpackLineGeom(feat?.geometry, tags);
  }
  return { version: 0.6, generator: "Postpass", elements: [...nodeElements, ...wayElements] };
}

async function fetchPostpassSingle(sql, signal) {
  const timeoutSig = typeof AbortSignal !== "undefined" && AbortSignal.timeout ? AbortSignal.timeout(15000) : null;
  const reqSignal =
    signal && timeoutSig && AbortSignal.any ? AbortSignal.any([signal, timeoutSig]) : timeoutSig || signal;
  const resp = await fetch(POSTPASS_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8" },
    body: "data=" + encodeURIComponent(sql),
    signal: reqSignal,
  });
  if (!resp.ok) {
    const errTxt = await resp.text().catch(() => "");
    throw new Error(`Postpass HTTP ${resp.status}: ${errTxt.slice(0, 160)}`);
  }
  const data = await resp.json();
  if (!data || !Array.isArray(data.features)) {
    throw new Error("Postpass invalid GeoJSON response.");
  }
  return data;
}

async function fetchOverpassSingle(ql, signal) {
  let lastErr = null;
  for (let attempt = 0; attempt < OVERPASS_MIRRORS.length; attempt++) {
    const url = OVERPASS_MIRRORS[attempt];
    const timeoutSig = typeof AbortSignal !== "undefined" && AbortSignal.timeout ? AbortSignal.timeout(10000) : null;
    const reqSignal =
      signal && timeoutSig && AbortSignal.any ? AbortSignal.any([signal, timeoutSig]) : timeoutSig || signal;
    try {
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8" },
        body: "data=" + encodeURIComponent(ql),
        signal: reqSignal,
      });
      if (!resp.ok) {
        lastErr = new Error(`HTTP ${resp.status} from ${url}`);
        await new Promise((r) => setTimeout(r, 350 * (attempt + 1)));
        continue;
      }
      const data = await resp.json();
      if (data && data.remark && /runtime error|timed out|out of memory/i.test(data.remark)) {
        lastErr = new Error(`Overpass runtime remark: ${data.remark}`);
        continue;
      }
      return data;
    } catch (err) {
      if (signal && signal.aborted) throw err;
      lastErr = err;
    }
  }
  throw lastErr || new Error("All Overpass mirrors failed.");
}

export function buildTargetCacheKey(target, paddedBbox) {
  const ql = buildTargetOverpassQL(target, paddedBbox);
  const sql = buildTargetPostpassSQL(target, paddedBbox);
  const typesKey = JSON.stringify(target.element_types || ["node", "way", "relation"]);
  const geomKey = target.geometry_handling || "point";
  return `target::${BACKEND_CACHE_VERSION}::${target.id}::${geomKey}::${typesKey}::${ql}::${sql}`;
}

export async function fetchTargetOSM(target, rawBbox, { forceRefresh = false, signal } = {}) {
  const padded = padBbox(rawBbox, target.bbox_buffer_m ?? 800);
  const ql = buildTargetOverpassQL(target, padded);
  const sql = buildTargetPostpassSQL(target, padded);
  const cacheKey = buildTargetCacheKey(target, padded);
  if (!forceRefresh) {
    const cached = await getCachedOSM(cacheKey);
    if (cached) return { data: cached, fromCache: true };
  }
  try {
    const fc = await fetchPostpassSingle(sql, signal);
    const data = postpassToOverpassTarget(fc);
    await setCachedOSM(cacheKey, data);
    return { data, fromCache: false };
  } catch (ppErr) {
    if (signal && signal.aborted) throw ppErr;
    console.warn("Postpass target query failed, falling back to Overpass:", ppErr);
    const data = await fetchOverpassSingle(ql, signal);
    await setCachedOSM(cacheKey, data);
    return { data, fromCache: false };
  }
}

export async function fetchStreetGraphOSM(rawBbox, bufferM = 800, profile = "walk", { forceRefresh = false, signal } = {}) {
  const padded = padBbox(rawBbox, bufferM);
  const ql = buildStreetGraphOverpassQL(padded, profile);
  const cacheKey = `graph::${BACKEND_CACHE_VERSION}::${profile}::${padded.south.toFixed(4)},${padded.west.toFixed(4)},${padded.north.toFixed(4)},${padded.east.toFixed(4)}`;

  if (!forceRefresh) {
    const cached = await getCachedOSM(cacheKey);
    if (cached) return { data: cached, fromCache: true };
  }

  const sql = buildStreetGraphPostpassSQL(padded, profile);
  try {
    const fc = await fetchPostpassSingle(sql, signal);
    const data = postpassToOverpassStreetGraph(fc);
    await setCachedOSM(cacheKey, data);
    return { data, fromCache: false };
  } catch (ppErr) {
    if (signal && signal.aborted) throw ppErr;
    console.warn("Postpass street graph query failed, falling back to Overpass:", ppErr);
  }

  try {
    const data = await fetchOverpassSingle(ql, signal);
    await setCachedOSM(cacheKey, data);
    return { data, fromCache: false };
  } catch (err) {
    if (err.name === "AbortError") throw err;
    // 2x2 sub-tile fallback for large areas
    const midLat = 0.5 * (padded.south + padded.north);
    const midLon = 0.5 * (padded.west + padded.east);
    const subBoxes = [
      { south: padded.south, west: padded.west, north: midLat, east: midLon },
      { south: padded.south, west: midLon, north: midLat, east: padded.east },
      { south: midLat, west: padded.west, north: padded.north, east: midLon },
      { south: midLat, west: midLon, north: padded.north, east: padded.east },
    ];
    const seen = new Set();
    const mergedElements = [];
    for (const sb of subBoxes) {
      const subQl = buildStreetGraphOverpassQL(sb, profile);
      const part = await fetchOverpassSingle(subQl, signal);
      for (const el of part.elements || []) {
        const k = `${el.type}/${el.id}`;
        if (!seen.has(k)) {
          seen.add(k);
          mergedElements.push(el);
        }
      }
    }
    const merged = { version: 0.6, elements: mergedElements };
    await setCachedOSM(cacheKey, merged);
    return { data: merged, fromCache: false };
  }
}
