import test from "node:test";
import assert from "node:assert/strict";

import { buildOsmBundle, parseOsmBundle } from "../frontend/gis/osm_idb_cache.js";
import {
  buildTargetCacheKey,
  buildTargetOverpassQL,
  buildTargetPostpassSQL,
  padBbox,
  parseTagShorthand,
  postpassToOverpassTarget,
} from "../frontend/gis/overpass_client.js";
import {
  createDefaultRecipe,
  generateEmptyCSVTemplate,
  generateSampleStudy,
  inferColumnProfile,
  parseCSVText,
} from "../frontend/state/recipe_store.js";
import { escapeHtml, parseNumericScalar, sanitizeRecipe } from "../frontend/utils/sanitize.js";

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

test("RFC-4180 CSV parsing, comments, thousand separators, template & generic sandbox", () => {
  const csv = [
    "# comment preamble line",
    'id;lat;lon;"note";price;price',
    'r1;47.5;19.0;"Line 1\nLine 2; ""quoted""";"1 234,56";99',
    'r2;47.6;19.1;"Simple";"2,500.75";101',
  ].join("\n");

  const parsed = parseCSVText(csv);
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

  const sample = generateSampleStudy();
  assert.ok(sample.parsed.nRows >= 100);
  assert.ok(Object.keys(sample.geojsonTargets).length >= 2);
  assert.ok(sample.streetGraphs.walk?.elements?.length > 10);
  const serialized = JSON.stringify(sample).toLowerCase();
  assert.doesNotMatch(serialized, /kobanya|kőbánya|vienna|wien/);
});

test("Sanitize utilities & offline .osm_bundle.json serialization", () => {
  const esc = escapeHtml(`<script>alert("xss")</script> & 'test'`);
  assert.doesNotMatch(esc, /[<>"]/);
  assert.match(esc, /&lt;script&gt;/);

  assert.equal(parseNumericScalar("1 234,56"), 1234.56);
  assert.equal(parseNumericScalar("1,234.56"), 1234.56);
  assert.ok(Number.isNaN(parseNumericScalar("")));
  assert.ok(Number.isNaN(parseNumericScalar("n/a")));

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
  const roundtrip = parseOsmBundle(bundle);
  assert.equal(roundtrip.targets.park.source_kind, "osm");
  assert.equal(roundtrip.targets.park.target_cfg.name, "Park");
  assert.equal(roundtrip.targets.park.payload.elements[0].id, 1);
  assert.throws(() => parseOsmBundle({ schema: "invalid" }));
});
