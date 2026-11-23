/* Browser-safe OSM client with Geofabrik Postpass primary SQL->GeoJSON engine,
 * fast-fail Overpass CORS mirror pool fallback, 2x2 auto-tiling, and IndexedDB caching.
 */
import { getCachedOSM, setCachedOSM } from "./osm_idb_cache.js";

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

export function padBbox(bbox, bufferM = 1500) {
  const latMid = 0.5 * (bbox.south + bbox.north);
  const dLat = bufferM / 111132.95;
  const dLon = bufferM / (111132.95 * Math.max(0.15, Math.cos((latMid * Math.PI) / 180)));
  return {
    south: Math.max(-89.9, bbox.south - dLat),
    west: bbox.west - dLon,
    north: Math.min(89.9, bbox.north + dLat),
    east: bbox.east + dLon,
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
  const outMode = target.geometry_handling === "point" ? "out center;" : `out geom(${b});`;
  return `[out:json][timeout:30];\n(\n${clauses.join("\n")}\n);\n${outMode}`;
}

export function buildStreetGraphOverpassQL(paddedBbox, profile = "walk") {
  const b = `${paddedBbox.south.toFixed(6)},${paddedBbox.west.toFixed(6)},${paddedBbox.north.toFixed(6)},${paddedBbox.east.toFixed(6)}`;
  if (profile === "drive") {
    return `[out:json][timeout:30];\nway["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service)$"]["access"!="private"](${b});\nout tags;\n>;\nout skel qt;`;
  }
  if (profile === "bike") {
    return `[out:json][timeout:30];\nway["highway"~"^(cycleway|primary|secondary|tertiary|unclassified|residential|living_street|path|track|service)$"]["bicycle"!="no"](${b});\nout tags;\n>;\nout skel qt;`;
  }
  // Undirected pedestrian walk network: minimal out skel qt payload (60-80% smaller)
  return `[out:json][timeout:30];\nway["highway"~"^(primary|secondary|tertiary|unclassified|residential|living_street|pedestrian|footway|path|cycleway|steps)$"]["foot"!="no"](${b});\n(._;>;);\nout skel qt;`;
}

function sqlLit(s) {
  return `'${String(s).replace(/'/g, "''")}'`;
}

export function buildTargetPostpassSQL(target, paddedBbox) {
  const env = `st_makeenvelope(${paddedBbox.west.toFixed(6)}, ${paddedBbox.south.toFixed(6)}, ${paddedBbox.east.toFixed(6)}, ${paddedBbox.north.toFixed(6)}, 4326)`;
  const types = target.element_types && target.element_types.length ? target.element_types : ["node", "way", "relation"];
  let table;
  let geomExpr;
  if (target.geometry_handling === "point") {
    table = "postpass_pointpolygon";
    geomExpr = "ST_Centroid(geom) AS geom";
  } else if (target.geometry_handling === "line_segment") {
    table = "postpass_line";
    geomExpr = `ST_Intersection(geom, ${env}) AS geom`;
  } else {
    table = types.includes("node") ? "postpass_pointlinepolygon" : "postpass_linepolygon";
    geomExpr = `ST_Intersection(geom, ${env}) AS geom`;
  }

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
  return `SELECT osm_type, osm_id, tags, ${geomExpr} FROM ${table} WHERE geom && ${env} AND (${whereExpr})`;
}

export function postpassToOverpassTarget(fc) {
  const elements = [];
  const feats = fc?.features || [];
  for (let idx = 0; idx < feats.length; idx++) {
    const feat = feats[idx];
    const props = feat?.properties || {};
    const rawType = props.osm_type;
    const elType = rawType === "N" ? "node" : rawType === "R" ? "relation" : "way";
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
        const pts = (g.coordinates || []).filter((c) => Array.isArray(c) && c.length >= 2);
        if (pts.length >= 2) {
          elements.push({
            type: "way",
            id: nextId(),
            tags,
            geometry: pts.map((c) => ({ lon: Number(c[0]), lat: Number(c[1]) })),
          });
        }
      } else if (g.type === "MultiLineString") {
        for (const line of g.coordinates || []) {
          const pts = (line || []).filter((c) => Array.isArray(c) && c.length >= 2);
          if (pts.length >= 2) {
            elements.push({
              type: "way",
              id: nextId(),
              tags,
              geometry: pts.map((c) => ({ lon: Number(c[0]), lat: Number(c[1]) })),
            });
          }
        }
      } else if (g.type === "Polygon") {
        const rings = g.coordinates || [];
        const members = [];
        rings.forEach((ring, rIdx) => {
          if (ring && ring.length >= 3) {
            members.push({
              type: "way",
              role: rIdx === 0 ? "outer" : "inner",
              geometry: ring.map((c) => ({ lon: Number(c[0]), lat: Number(c[1]) })),
            });
          }
        });
        if (members.length) {
          elements.push({ type: "relation", id: nextId(), tags: { ...tags, type: "multipolygon" }, members });
        }
      } else if (g.type === "MultiPolygon") {
        const members = [];
        for (const poly of g.coordinates || []) {
          (poly || []).forEach((ring, rIdx) => {
            if (ring && ring.length >= 3) {
              members.push({
                type: "way",
                role: rIdx === 0 ? "outer" : "inner",
                geometry: ring.map((c) => ({ lon: Number(c[0]), lat: Number(c[1]) })),
              });
            }
          });
        }
        if (members.length) {
          elements.push({ type: "relation", id: nextId(), tags: { ...tags, type: "multipolygon" }, members });
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
      "tags->>'highway' IN ('motorway','trunk','primary','secondary','tertiary','unclassified','residential','living_street','service') AND COALESCE(tags->>'access','yes') <> 'private'";
  } else if (profile === "bike") {
    filterExpr =
      "tags->>'highway' IN ('cycleway','primary','secondary','tertiary','unclassified','residential','living_street','path','track','service') AND COALESCE(tags->>'bicycle','yes') <> 'no'";
  } else {
    filterExpr =
      "tags->>'highway' IN ('primary','secondary','tertiary','unclassified','residential','living_street','pedestrian','footway','path','cycleway','steps') AND COALESCE(tags->>'foot','yes') <> 'no'";
  }
  return `SELECT osm_id, tags, ST_Intersection(geom, ${env}) AS geom FROM postpass_line WHERE geom && ${env} AND ${filterExpr}`;
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

export async function fetchTargetOSM(target, rawBbox, { forceRefresh = false, signal } = {}) {
  const padded = padBbox(rawBbox, target.bbox_buffer_m ?? 800);
  const ql = buildTargetOverpassQL(target, padded);
  const cacheKey = `target::${target.id}::${ql}`;
  if (!forceRefresh) {
    const cached = await getCachedOSM(cacheKey);
    if (cached) return { data: cached, fromCache: true };
  }
  const sql = buildTargetPostpassSQL(target, padded);
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

export async function fetchStreetGraphOSM(rawBbox, bufferM = 600, profile = "walk", { forceRefresh = false, signal } = {}) {
  const padded = padBbox(rawBbox, bufferM);
  const ql = buildStreetGraphOverpassQL(padded, profile);
  const cacheKey = `graph::${profile}::${padded.south.toFixed(4)},${padded.west.toFixed(4)},${padded.north.toFixed(4)},${padded.east.toFixed(4)}`;

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
