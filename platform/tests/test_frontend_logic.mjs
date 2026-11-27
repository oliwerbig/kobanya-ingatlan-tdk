import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { buildOsmBundle, parseOsmBundle } from "../frontend/gis/osm_idb_cache.js";
import {
  buildStreetGraphOverpassQL,
  buildStreetGraphPostpassSQL,
  buildTargetCacheKey,
  buildTargetOverpassQL,
  buildTargetPostpassSQL,
  padBbox,
  parseTagShorthand,
  postpassToOverpassStreetGraph,
  postpassToOverpassTarget,
  splitAntimeridianBbox,
} from "../frontend/gis/overpass_client.js";
import * as recipeStore from "../frontend/state/recipe_store.js";
import {
  buildWorkspaceBundle,
  classifyIncomingJson,
  createDefaultRecipe,
  generateEmptyCSVTemplate,
  inferColumnProfile,
  inferGeojsonTargetDefaults,
  makeUniqueTargetId,
  parseCSVText,
  parseGeoJSONDataset,
  parseWorkspaceBundle,
  prepareIncomingBatch,
  slugifyTitle,
  validateGeojsonObject,
} from "../frontend/state/recipe_store.js";
import { escapeHtml, getPlotlyThemeLayout, parseJsonText, parseNumericScalar, sanitizeRecipe, stripBom } from "../frontend/utils/sanitize.js";
import { getLang, setLang, t, translateVerdict } from "../frontend/utils/i18n.js";

test("Overpass vs Postpass parity, unclipped geom, element_types & cache keys", () => {
  const bbox = { south: 47.4, west: 19.0, north: 47.5, east: 19.1 };
  const padded = padBbox(bbox, 800);
  const tgtPoly = {
    id: "parks",
    osm_filter: parseTagShorthand("leisure=park|garden"),
    element_types: ["way", "relation"],
    geometry_handling: "polygon_boundary",
    inside_is_zero: true,
  };

  const ql = buildTargetOverpassQL(tgtPoly, padded);
  assert.match(ql, /out geom;/);
  assert.doesNotMatch(ql, /out geom\(/);

  const sqlPoly = buildTargetPostpassSQL(tgtPoly, padded);
  assert.match(sqlPoly, /geom AS geom FROM postpass_linepolygon/);
  assert.doesNotMatch(sqlPoly, /ST_Intersection/i);
  assert.match(sqlPoly, /osm_type IN \('W','R'\)/);

  const sqlPt = buildTargetPostpassSQL(
    { ...tgtPoly, element_types: ["node", "way"], geometry_handling: "point" },
    padded
  );
  assert.match(sqlPt, /FROM postpass_pointlinepolygon/);
  assert.match(sqlPt, /osm_type IN \('N','W'\)/);

  const kPt = buildTargetCacheKey({ ...tgtPoly, geometry_handling: "point" }, bbox);
  const kPoly = buildTargetCacheKey(tgtPoly, bbox);
  const kNodeOnly = buildTargetCacheKey({ ...tgtPoly, element_types: ["node"] }, bbox);
  assert.notEqual(kPt, kPoly);
  assert.notEqual(kPoly, kNodeOnly);
  assert.ok(kPoly.startsWith("target::v2::"));

  const conv = postpassToOverpassTarget({
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        id: 123,
        properties: { osm_type: "way", osm_id: 123, tags: { name: "City Park", leisure: "park" } },
        geometry: {
          type: "Polygon",
          coordinates: [[[19.0, 47.4], [19.01, 47.4], [19.01, 47.41], [19.0, 47.4]]],
        },
      },
      {
        type: "Feature",
        id: 456,
        properties: { osm_type: "relation", osm_id: 456, tags: { name: "Island Park", type: "multipolygon" } },
        geometry: {
          type: "Polygon",
          coordinates: [
            [[19.02, 47.42], [19.03, 47.42], [19.03, 47.43], [19.02, 47.42]],
            [[19.022, 47.422], [19.028, 47.422], [19.028, 47.428], [19.022, 47.422]],
          ],
        },
      },
    ],
  });
  assert.equal(conv.elements[0].type, "way");
  assert.equal(conv.elements[0].id, 123);
  assert.equal(conv.elements[0].tags.name, "City Park");
  assert.equal(conv.elements[1].type, "relation");
  assert.equal(conv.elements[1].members[0].role, "outer");
  assert.equal(conv.elements[1].members[1].role, "inner");

  const convRel1Ring = postpassToOverpassTarget({
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        properties: { osm_type: "relation", osm_id: 789, tags: { name: "Single Ring Rel" } },
        geometry: { type: "Polygon", coordinates: [[[19.0, 47.4], [19.01, 47.4], [19.01, 47.41], [19.0, 47.4]]] },
      },
      {
        type: "Feature",
        properties: { osm_type: "node", osm_id: 101, tags: { name: "Node Pt" } },
        geometry: { type: "Point", coordinates: [19.0, 47.4] },
      },
    ],
  });
  assert.equal(convRel1Ring.elements[0].type, "relation");
  assert.equal(convRel1Ring.elements[1].type, "node");
});

test("RFC-4180 CSV parsing, comments, thousand separators, BOM & empty CSV template (no random generator)", () => {
  assert.equal(recipeStore.generateSampleStudy, undefined);

  const csv = [
    "# comment preamble line",
    'id;lat;lon;"note";price;price',
    'r1;47.5;19.0;"Line 1\nLine 2; ""quoted""";"1 234,56";99',
    'r2;47.6;19.1;"Simple";"2,500.75";101',
  ].join("\n");

  const parsed = parseCSVText("\uFEFF" + csv);
  assert.equal(parsed.nRows, 2);
  assert.deepEqual(parsed.headers, ["id", "lat", "lon", "note", "price", "price_2"]);
  assert.equal(parsed.columns.note[0], 'Line 1\nLine 2; "quoted"');

  const { profile } = inferColumnProfile(parsed.headers, parsed.columns);
  assert.equal(profile.price.isNumeric, true);

  const tpl = generateEmptyCSVTemplate();
  const tplParsed = parseCSVText(tpl);
  assert.equal(tplParsed.nRows, 2);
  assert.ok(tplParsed.headers.includes("lat"));
  assert.ok(tplParsed.headers.includes("lon"));
});

test("Sanitize utilities, BOM JSON parsing & offline .osm_bundle.json serialization", () => {
  const esc = escapeHtml(`<script>alert("xss")</script> & 'test'`);
  assert.doesNotMatch(esc, /[<>"]/);
  assert.match(esc, /&lt;script&gt;/);

  assert.equal(parseNumericScalar("1 234,56"), 1234.56);
  assert.equal(parseNumericScalar("1,234.56"), 1234.56);
  assert.ok(Number.isNaN(parseNumericScalar("")));
  assert.ok(Number.isNaN(parseNumericScalar("n/a")));

  assert.equal(stripBom("\uFEFF{\"ok\":1}"), '{"ok":1}');
  assert.deepEqual(parseJsonText("\uFEFF  {\"ok\": true} "), { ok: true });

  const polluted = JSON.parse(
    '{"schema_version":"1.0","__proto__":{"polluted":true},"recipe_meta":{"title":"<b>Bad</b>"},"spatial_targets":[{"id":"t1","name":"Park"}]}'
  );
  const clean = sanitizeRecipe(polluted, createDefaultRecipe());
  assert.equal(({}).polluted, undefined);
  assert.equal(clean.recipe_meta.title, "<b>Bad</b>");

  const bundle = buildOsmBundle({
    bbox: { south: 47.4, west: 19.0, north: 47.5, east: 19.1 },
    targets: [{ id: "park", name: "Park", source: "osm_preset", geometry_handling: "polygon_boundary", inside_is_zero: true }],
    rawOsmTargets: { park: { elements: [{ type: "way", id: 1 }] } },
    rawGeojsonTargets: {},
    rawStreetGraphs: { walk: { elements: [] } },
  });
  assert.equal(bundle.kind, "geospec_osm_bundle");
  const roundtrip = parseOsmBundle(bundle);
  assert.equal(roundtrip.targets.park.source_kind, "osm");
  assert.equal(roundtrip.targets.park.target_cfg.name, "Park");
  assert.equal(roundtrip.targets.park.payload.elements[0].id, 1);
  assert.throws(() => parseOsmBundle({ schema: "invalid" }));
});

test("Universal file classification, UTF-8 BOM JSON/GeoJSON, deterministic batch order & workspace roundtrip", () => {
  assert.equal(slugifyTitle("TDK 2026 — Budapest X. Kőbánya"), "tdk_2026_budapest_x_kobanya");
  assert.equal(slugifyTitle("   "), "geospec_study");

  const polyGj = {
    type: "FeatureCollection",
    features: [{ type: "Feature", properties: { name: "Park" }, geometry: { type: "Polygon", coordinates: [[[19, 47], [19.01, 47], [19.01, 47.01], [19, 47]]] } }],
  };
  const lineGj = {
    type: "FeatureCollection",
    features: [{ type: "Feature", properties: { name: "Rail" }, geometry: { type: "LineString", coordinates: [[19, 47], [19.01, 47.01]] } }],
  };
  const ptGj = {
    type: "FeatureCollection",
    features: [{ type: "Feature", properties: { name: "Stop", lat: 999 }, geometry: { type: "Point", coordinates: [19.12, 47.48] } }],
  };
  assert.deepEqual(inferGeojsonTargetDefaults(polyGj), { geometry_handling: "polygon_boundary", inside_is_zero: true });
  assert.deepEqual(inferGeojsonTargetDefaults(lineGj), { geometry_handling: "line_segment", inside_is_zero: false });
  assert.deepEqual(inferGeojsonTargetDefaults(ptGj), { geometry_handling: "point", inside_is_zero: false });

  // parseGeoJSONDataset strips BOM and does not double-push lat/lon properties
  const ptParsed = parseGeoJSONDataset("\uFEFF" + JSON.stringify(ptGj));
  assert.equal(ptParsed.nRows, 1);
  assert.equal(ptParsed.columns.lat.length, 1);
  assert.equal(ptParsed.columns.lat[0], 47.48);

  // GeoJSON classification rules
  assert.equal(classifyIncomingJson(ptGj, "pts.geojson", { hasDataset: false, batchHasDataset: false }).kind, "dataset_geojson");
  assert.equal(classifyIncomingJson(ptGj, "pts.geojson", { hasDataset: true, batchHasDataset: false }).kind, "target_geojson");
  assert.equal(classifyIncomingJson(ptGj, "pts.geojson", { hasDataset: false, batchHasDataset: true }).kind, "target_geojson");
  assert.equal(classifyIncomingJson(polyGj, "park.geojson", { hasDataset: false, batchHasDataset: false }).kind, "target_geojson");

  // Legacy OSM bundle (targets + street_graphs without kind and without dataset_schema)
  assert.equal(
    classifyIncomingJson({ targets: { t1: {} }, street_graphs: { walk: {} } }, "legacy.osm_bundle.json").kind,
    "osm_bundle"
  );
  // Minimal recipe keys
  assert.equal(classifyIncomingJson({ models: [{ id: "m1", covariates: [] }] }, "m.json").kind, "recipe");
  assert.equal(classifyIncomingJson({ spatial_targets: [] }, "t.json").kind, "recipe");
  assert.equal(classifyIncomingJson({ recipe_meta: { title: "X" } }, "r.json").kind, "recipe");
  // Unknown JSON throws cleanly
  assert.throws(() => classifyIncomingJson({ foo: "bar" }, "bad.json"), /Ismeretlen JSON/);

  // Real kobanya_recipe.json + kobanya_dataset.csv + stations (Point GeoJSON) + railways (LineString GeoJSON) in arbitrary order
  const here = path.dirname(fileURLToPath(import.meta.url));
  const rootDir = path.resolve(here, "../..");
  const kobRecipeText = fs.readFileSync(path.join(rootDir, "geospec_input/kobanya_recipe.json"), "utf8");
  const kobCsvText = fs.readFileSync(path.join(rootDir, "geospec_input/kobanya_dataset.csv"), "utf8");
  const kobStationsText = fs.readFileSync(path.join(rootDir, "geospec_input/kobanya_stations.geojson"), "utf8");
  const kobRailsText = fs.readFileSync(path.join(rootDir, "geospec_input/kobanya_railways.geojson"), "utf8");

  const osmBundleText = JSON.stringify(
    buildOsmBundle({
      targets: [{ id: "park", name: "Park", source: "osm_preset" }],
      rawOsmTargets: { park: { elements: [{ type: "way", id: 7 }] } },
      rawStreetGraphs: { walk: { elements: [] } },
    })
  );

  // Supply in reverse order with UTF-8 BOM prepended to JSONs
  const plan = prepareIncomingBatch(
    [
      { fileName: "kobanya_stations.geojson", text: "\uFEFF" + kobStationsText },
      { fileName: "snap.osm_bundle.json", text: "\uFEFF" + osmBundleText },
      { fileName: "kobanya_railways.geojson", text: kobRailsText },
      { fileName: "kobanya_recipe.json", text: "\uFEFF" + kobRecipeText },
      { fileName: "kobanya_dataset.csv", text: "\uFEFF" + kobCsvText },
    ],
    { hasDataset: false }
  );
  assert.deepEqual(
    plan.map((p) => p.kind),
    ["dataset_csv", "recipe", "osm_bundle", "target_geojson", "target_geojson"]
  );
  assert.equal(plan[1].data.spatial_engine.active_model_id, "m4_ad");

  // Workspace build & roundtrip (with dataset + osm_bundle, and recipe-only without dataset)
  const parsedCsv = parseCSVText(kobCsvText);
  const parsedRec = parseJsonText(kobRecipeText);
  const ws = buildWorkspaceBundle({
    recipe: parsedRec,
    fileName: "kobanya_dataset.csv",
    headers: parsedCsv.headers,
    columns: parsedCsv.columns,
    nRows: parsedCsv.nRows,
    osmBundle: JSON.parse(osmBundleText),
    activeSpecId: "m4_ad",
    covType: "cluster",
    showPct: true,
  });
  assert.equal(ws.kind, "geospec_workspace");
  assert.equal(classifyIncomingJson(ws, "kobanya.geospec.json").kind, "workspace");

  const wsParsed = parseWorkspaceBundle(ws);
  assert.equal(wsParsed.dataset.fileName, "kobanya_dataset.csv");
  assert.equal(wsParsed.dataset.nRows, parsedCsv.nRows);
  assert.equal(wsParsed.recipe.spatial_engine.active_model_id, "m4_ad");
  assert.equal(wsParsed.osm_bundle.targets.park.payload.elements[0].id, 7);
  assert.equal(wsParsed.ui_state.activeSpecId, "m4_ad");
  assert.equal(wsParsed.ui_state.covType, "cluster");
  assert.equal(wsParsed.ui_state.showPct, true);

  // Recipe-only workspace (dataset: null)
  const wsNoDs = parseWorkspaceBundle(
    buildWorkspaceBundle({ recipe: parsedRec, columns: null })
  );
  assert.equal(wsNoDs.dataset, null);
  assert.equal(wsNoDs.recipe.recipe_meta.title, parsedRec.recipe_meta.title);

  // Unique target id generation
  assert.equal(makeUniqueTargetId("vagany", parsedRec.spatial_targets), "vagany_2");

  // Empty FeatureCollection / invalid GeoJSON rejected even when dataset is loaded
  assert.throws(
    () => classifyIncomingJson({ type: "FeatureCollection", features: [] }, "empty.geojson", { hasDataset: true }),
    /nem tartalmaz Feature/
  );
  assert.throws(
    () => validateGeojsonObject({ type: "Feature", properties: {}, geometry: null }, "nogeom.geojson"),
    /nem tartalmaz érvényes geometriát/
  );
  assert.throws(
    () => validateGeojsonObject(parsedRec, "kobanya_recipe.json"),
    /Érvénytelen GeoJSON/
  );

  // Two Point GeoJSONs without prior dataset: first becomes dataset_geojson, second becomes target_geojson
  const ptBatch = prepareIncomingBatch(
    [
      { fileName: "obs.geojson", text: JSON.stringify(ptGj) },
      { fileName: "stops.geojson", text: kobStationsText },
    ],
    { hasDataset: false }
  );
  assert.deepEqual(
    ptBatch.map((p) => p.kind),
    ["dataset_geojson", "target_geojson"]
  );

  // parseOsmBundle filters prototype-polluting keys
  const craftedBundle = JSON.parse(
    '{"targets":{"__proto__":{"payload":{}},"park":{"payload":{"elements":[]}}},"street_graphs":{"constructor":{}}}'
  );
  const safeBundle = parseOsmBundle(craftedBundle);
  assert.equal(Object.prototype.hasOwnProperty.call(safeBundle.targets, "__proto__"), false);
  assert.equal(Object.prototype.hasOwnProperty.call(safeBundle.street_graphs, "constructor"), false);
  assert.ok(Object.prototype.hasOwnProperty.call(safeBundle.targets, "park"));
});

test("GIS-1, GIS-2, GIS-3: street graph filters, Postpass unclipped geom & antimeridian wrapping", () => {
  const padded = padBbox({ south: 0.0, west: -179.995, north: 0.1, east: 179.995 }, 5000);
  assert.ok(padded.west > 179.0 && padded.west < 180.0);
  assert.ok(padded.east < -179.0 && padded.east > -180.0);

  const studyPad = padBbox({ south: 47.47, west: 19.12, north: 47.50, east: 19.18 }, 800);
  const walkQl = buildStreetGraphOverpassQL(studyPad, "walk");
  const walkSql = buildStreetGraphPostpassSQL(studyPad, "walk");
  assert.match(walkQl, /\["access"!~"\^\(private\|no\)\$"\]/);
  assert.match(walkQl, /\["foot"~"\^\(yes\|designated\|permissive\)\$"\]/);
  assert.match(walkQl, /parking_aisle\|driveway/);
  assert.match(walkSql, /ST_ClipByBox2D\(/);
  assert.match(walkSql, /NOT IN \('private','no'\)/);
  assert.doesNotMatch(walkSql, /ST_Intersection/i);
  assert.match(walkSql, /NOT IN \('parking_aisle','driveway'\)/);

  const bikeSql = buildStreetGraphPostpassSQL(studyPad, "bike");
  assert.match(bikeSql, /COALESCE\(tags->>'access','yes'\) NOT IN \('private','no'\)/);
  assert.doesNotMatch(bikeSql, /ST_Intersection/i);

  const defRec = createDefaultRecipe();
  assert.equal(defRec.dataset_schema.cluster_spec.mode, "coordinate_radius");
  assert.equal(defRec.spatial_engine.weights[0].dedup_colocated, true);
});

test("Round-4: Antimeridian, Postpass abutment connectors, sanitizeRecipe hardening, atomic batch & i18n", () => {
  // 1. Antimeridian split & padBbox wrap
  const split = splitAntimeridianBbox({ south: -17.0, west: 179.5, north: -16.5, east: -179.5 });
  assert.equal(split.length, 2);
  assert.equal(split[0].east, 180.0);
  assert.equal(split[1].west, -180.0);
  const wrapped = padBbox({ south: 10.0, west: 179.99, north: 10.01, east: 179.995 }, 3000);
  assert.ok(wrapped.east >= -180 && wrapped.east <= 180);

  // 2. Postpass degree-1 bridge abutment vs grade-separated underpass isolation
  const osmGraph = postpassToOverpassStreetGraph({
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        properties: { tags: { highway: "residential" } },
        geometry: { type: "LineString", coordinates: [[19.000, 47.500], [19.002, 47.500]] },
      },
      {
        type: "Feature",
        properties: { tags: { highway: "footway", bridge: "yes", layer: "1" } },
        geometry: { type: "LineString", coordinates: [[19.002, 47.500], [19.003, 47.501]] },
      },
      {
        type: "Feature",
        properties: { tags: { highway: "footway", tunnel: "yes", layer: "-1" } },
        geometry: { type: "LineString", coordinates: [[19.001, 47.499], [19.001, 47.500], [19.001, 47.501]] },
      },
    ],
  });
  const ways = osmGraph.elements.filter((e) => e.type === "way");
  const abutments = ways.filter((w) => w.role === "abutment");
  assert.equal(abutments.length, 1);
  assert.equal(abutments[0].nodes.length, 2);

  // 3. sanitizeRecipe prototype stripping, clamping & getPlotlyThemeLayout override merge
  const poison = JSON.parse('{"__proto__":{"polluted":true},"guards_cfg":{"sparse_min_cell":9999,"hc3_leverage_warn":2.5}}');
  const cleanRec = sanitizeRecipe(poison, createDefaultRecipe());
  assert.equal(({}).polluted, undefined);
  assert.ok(cleanRec.guards_cfg.sparse_min_cell <= 200);
  assert.ok(cleanRec.guards_cfg.hc3_leverage_warn <= 0.999);
  const lay = getPlotlyThemeLayout({ font: { size: 14 }, xaxis: { title: "Dist" }, yaxis2: { overlaying: "y" } });
  assert.equal(lay.font.size, 14);
  assert.equal(lay.xaxis.title, "Dist");
  assert.ok(lay.xaxis.gridcolor);
  assert.equal(lay.yaxis2.overlaying, "y");

  // 4. prepareIncomingBatch atomic rejection on mixed valid + broken files
  assert.throws(() => {
    prepareIncomingBatch([
      { fileName: "ok_ws.json", text: JSON.stringify({ kind: "geospec_workspace", version: 1, recipe: createDefaultRecipe() }) },
      { fileName: "broken.json", text: "{malformed json" },
    ]);
  });

  // 5. i18n language switching, interpolation & verdict translation
  const prev = getLang();
  try {
    setLang("en");
    assert.equal(getLang(), "en");
    assert.match(t("map.layer_snap"), /Network/i);
    assert.equal(translateVerdict("Classic OLS (No spatial dependence)"), "Classic OLS (No spatial dependence)");
    setLang("hu");
    assert.equal(getLang(), "hu");
    assert.notEqual(translateVerdict("Classic OLS (No spatial dependence)"), "Classic OLS (No spatial dependence)");
  } finally {
    setLang(prev);
  }

  // 6. sanitizeRecipe preserves sparse_min_cell=0, adaptive_bisquare W & normalizes prevalence fraction
  const repEdge = sanitizeRecipe({
    guards_cfg: { sparse_min_cell: 0 },
    spatial_engine: {
      weights: [{ id: "ab", type: "adaptive_bisquare", k: 25 }],
      gwr: { prevalence_min_pct: 15.0 },
    },
  }, createDefaultRecipe());
  assert.equal(repEdge.guards_cfg.sparse_min_cell, 0);
  assert.equal(repEdge.spatial_engine.weights[0].type, "adaptive_bisquare");
  assert.equal(repEdge.spatial_engine.gwr.prevalence_min_pct, 0.15);

  const giDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "geospec_input");
  for (const rFile of [
    "kobanya_paper_replication_recipe.json",
    "leopoldstadt_paper_replication_recipe.json",
    "kobanya_recipe.json",
    "leopoldstadt_recipe.json",
  ]) {
    const absR = path.join(giDir, rFile);
    if (!fs.existsSync(absR)) continue;
    const rawR = JSON.parse(fs.readFileSync(absR, "utf8"));
    const sanR = sanitizeRecipe(rawR, createDefaultRecipe());
    assert.equal(sanR.guards_cfg.sparse_min_cell, rawR.guards_cfg.sparse_min_cell);
    assert.equal(sanR.spatial_engine.gwr.prevalence_min_pct, rawR.spatial_engine.gwr.prevalence_min_pct);
    assert.equal(sanR.spatial_targets.find((t) => t.id === "allomas").distance_modes, "network_dijkstra");
  }

  // 7. sanitizeRecipe preserves & clamps Stage 6 ml_shap config and seal metadata
  const mlSan = sanitizeRecipe({
    recipe_meta: { title: "Sealed", seal_version: "v1.0-sealed" },
    ml_shap: { focal_model_id: "m3_log", n_estimators: 9999, max_depth: 99, random_seed: 42 },
  }, createDefaultRecipe());
  assert.equal(mlSan.recipe_meta.seal_version, "v1.0-sealed");
  assert.equal(mlSan.ml_shap.focal_model_id, "m3_log");
  assert.equal(mlSan.ml_shap.n_estimators, 500);
  assert.equal(mlSan.ml_shap.max_depth, 6);
  assert.equal(mlSan.ml_shap.random_seed, 42);
});

