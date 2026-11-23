/* Browser-safe Overpass API client with CORS mirror pool, 2x2 auto-tiling,
 * HTTP-200 remark runtime-error detection, and IndexedDB caching.
 */
import { getCachedOSM, setCachedOSM } from "./osm_idb_cache.js";

const OVERPASS_MIRRORS = [
  "https://overpass-api.de/api/interpreter",
  "https://overpass.kumi.systems/api/interpreter",
  "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
  "https://overpass.private.coffee/api/interpreter",
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
      clauses.push(`  ${et}${filterExpr}(${b});`);
    }
  }
  const outMode = target.geometry_handling === "point" ? "out center;" : "out geom;";
  return `[out:json][timeout:45];\n(\n${clauses.join("\n")}\n);\n${outMode}`;
}

export function buildStreetGraphOverpassQL(paddedBbox, profile = "walk") {
  const b = `${paddedBbox.south.toFixed(6)},${paddedBbox.west.toFixed(6)},${paddedBbox.north.toFixed(6)},${paddedBbox.east.toFixed(6)}`;
  if (profile === "drive") {
    return `[out:json][timeout:60];\nway["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service)$"]["access"!="private"](${b});\nout tags;\n>;\nout skel qt;`;
  }
  if (profile === "bike") {
    return `[out:json][timeout:60];\nway["highway"~"^(cycleway|primary|secondary|tertiary|unclassified|residential|living_street|path|track|service)$"]["bicycle"!="no"](${b});\nout tags;\n>;\nout skel qt;`;
  }
  // Undirected pedestrian walk network: minimal out skel qt payload (60-80% smaller)
  return `[out:json][timeout:60];\nway["highway"]["highway"!~"^(motorway|motorway_link|trunk_link|raceway|construction)$"]["foot"!="no"](${b});\n(._;>;);\nout skel qt;`;
}

async function fetchOverpassSingle(ql, signal) {
  let lastErr = null;
  for (let attempt = 0; attempt < OVERPASS_MIRRORS.length; attempt++) {
    const url = OVERPASS_MIRRORS[attempt];
    const timeoutSig = typeof AbortSignal !== "undefined" && AbortSignal.timeout ? AbortSignal.timeout(15000) : null;
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
  const padded = padBbox(rawBbox, target.bbox_buffer_m ?? 1500);
  const ql = buildTargetOverpassQL(target, padded);
  const cacheKey = `target::${target.id}::${ql}`;
  if (!forceRefresh) {
    const cached = await getCachedOSM(cacheKey);
    if (cached) return { data: cached, fromCache: true };
  }
  const data = await fetchOverpassSingle(ql, signal);
  await setCachedOSM(cacheKey, data);
  return { data, fromCache: false };
}

export async function fetchStreetGraphOSM(rawBbox, bufferM = 1500, profile = "walk", { forceRefresh = false, signal } = {}) {
  const padded = padBbox(rawBbox, bufferM);
  const ql = buildStreetGraphOverpassQL(padded, profile);
  const cacheKey = `graph::${profile}::${padded.south.toFixed(4)},${padded.west.toFixed(4)},${padded.north.toFixed(4)},${padded.east.toFixed(4)}`;

  if (!forceRefresh) {
    const cached = await getCachedOSM(cacheKey);
    if (cached) return { data: cached, fromCache: true };
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
