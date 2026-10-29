"""Explicit semantic boundary; no terrain generation, segmentation or perception."""
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import re

from ..catalog import load
from ..contracts import JSON_SCHEMAS

SCHEMA = "rescue.upstream-semantics/v1"
AUTHORED_NOTE = "未接入上游地形语义；本层为显式作者标注（非自动分割）"
SEMANTIC_CLASSES = (
    "building", "road", "tree", "tree_cluster", "water", "open_area",
    "slope", "facility", "station", "coast", "underground", "bridge",
)
# Missing numeric/environmental inputs are never inferred from images.
REQUIRED_ATTRIBUTES = {cls: () for cls in SEMANTIC_CLASSES}
REQUIRED_ATTRIBUTES.update({
    "building": ("building.use", "building.floors", "building.structure", "building.capacity"),
    "road": ("road.width_m", "road.surface"),
    "station": ("station.kind",),
})
# Explicit adapter vocabulary, not changes to the read-only F046 workbook.
# Keep actual source labels rather than pretending every path is asphalt.
# Unknown values still fail the ordinary catalog/domain validation below.
SOURCE_ROAD_SURFACES = {
    "foot_path": "SC10 Route-A roads P01/P02/P03: source pedestrian network",
    "graded_earth": "SC10 Route-A E01_APRON: source entry apron",
    "native_gravel_track": "SC11 Route-A source track and entry spur",
    "authored_civil_preview": "SC08 Route-A source civil-preview carriageways",
    "stair_ramp": "SC08 Route-A source stair/ramp network",
    "grass": "GT04 source dry-placement reservations: grass-covered ground walks",
}
_DEFAULTS = {
    "building.use": "residential", "building.floors": 1,
    "building.structure": "mixed", "building.capacity": 4,
    "road.width_m": 5.0, "road.surface": "asphalt", "station.kind": "supply",
}


def rng_for(seed, stream, key=""):
    """Stable independent streams: Python's process-randomized hash is not used."""
    digest = hashlib.sha256(f"{seed}|{stream}|{key}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))


@dataclass
class SemanticLayer:
    source_kind: str
    entities: list
    by_class: dict
    bounds: tuple
    missing: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    source: dict = field(default_factory=dict)

    def to_dict(self):
        return deepcopy({"source_kind": self.source_kind, "source": self.source,
                         "entities": self.entities, "by_class": self.by_class,
                         "bounds": list(self.bounds), "missing": self.missing,
                         "warnings": self.warnings})


def _error(message):
    raise ValueError("上游语义无效 / Invalid upstream semantics: " + message)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _schema_check(value, schema, path="$"):
    """Interpret the keywords used by the shared upstream JSON Schema, stdlib only."""
    if schema is True:
        return
    if schema is False:
        _error(path + ": disallowed value")
    for alternatives in (schema.get("anyOf"),):
        if alternatives is not None:
            for alternative in alternatives:
                try:
                    _schema_check(value, alternative, path)
                    break
                except ValueError:
                    pass
            else:
                _error(path + ": no anyOf alternative matched")
    for subschema in schema.get("allOf", []):
        _schema_check(value, subschema, path)
    if "if" in schema:
        try:
            _schema_check(value, schema["if"], path)
        except ValueError:
            selected = schema.get("else", True)
        else:
            selected = schema.get("then", True)
        _schema_check(value, selected, path)
    expected = schema.get("type")
    valid_type = {"object": isinstance(value, dict), "array": isinstance(value, list),
                  "string": isinstance(value, str), "number": _number(value),
                  "integer": isinstance(value, int) and not isinstance(value, bool),
                  "boolean": isinstance(value, bool), "null": value is None}
    if expected is not None and not valid_type.get(expected, False):
        _error(path + ": expected " + str(expected))
    if "const" in schema and value != schema["const"]:
        _error(path + ": wrong constant")
    if "enum" in schema and value not in schema["enum"]:
        _error(path + ": outside enum")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or ("pattern" in schema and re.search(schema["pattern"], value) is None):
            _error(path + ": invalid string")
    if _number(value) and "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
        _error(path + ": below exclusive minimum")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                _error(path + ": missing " + key)
        properties = schema.get("properties", {})
        for key, item in value.items():
            if "propertyNames" in schema:
                _schema_check(key, schema["propertyNames"], path + ".<key>")
            _schema_check(item, properties.get(key, schema.get("additionalProperties", True)), path + "." + str(key))
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            _error(path + ": invalid array length")
        prefix = schema.get("prefixItems", [])
        for index, item in enumerate(value):
            _schema_check(item, prefix[index] if index < len(prefix) else schema.get("items", True), f"{path}[{index}]")


def _validate(doc):
    """Validate against Lane A's shared schema plus finite geometry/unique ids."""
    _schema_check(doc, JSON_SCHEMAS[SCHEMA])
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        _error("schema must be " + SCHEMA)
    source = doc.get("source")
    if not isinstance(source, dict) or source.get("kind") not in ("upstream_annotation", "authored"):
        _error("source.kind")
    for key in ("producer", "note"):
        if not isinstance(source.get(key), str):
            _error("source." + key)
    if "terrain_ref" not in source or (source["terrain_ref"] is not None and not isinstance(source["terrain_ref"], str)):
        _error("source.terrain_ref")
    frame = doc.get("frame")
    if not isinstance(frame, dict) or frame.get("crs") != "local_ENU_m":
        _error("frame.crs must be local_ENU_m")
    bounds = frame.get("bounds_m")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4 or not all(_number(n) for n in bounds):
        _error("frame.bounds_m must contain four finite numbers")
    if bounds[0:2] not in ([0, 0], (0, 0)) or bounds[2] <= 0 or bounds[3] <= 0:
        _error("bounds_m must be [0,0,W,H] with W,H positive")
    if not isinstance(doc.get("entities"), list):
        _error("entities must be an array")
    seen = set()
    for entity in doc["entities"]:
        if not isinstance(entity, dict) or not isinstance(entity.get("id"), str) or not entity["id"]:
            _error("entity.id")
        if entity["id"] in seen:
            _error("duplicate entity id: " + entity["id"])
        seen.add(entity["id"])
        if not isinstance(entity.get("class"), str):
            _error(entity["id"] + ": class")
        if entity.get("label_source") not in ("upstream", "authored"):
            _error(entity["id"] + ": label_source")
        if not isinstance(entity.get("attributes"), dict):
            _error(entity["id"] + ": attributes")
        geom = entity.get("geometry")
        if not isinstance(geom, dict) or geom.get("type") not in ("point", "polyline", "polygon"):
            _error(entity["id"] + ": geometry.type")
        coords = geom.get("coords")
        minimum = {"point": 1, "polyline": 2, "polygon": 3}[geom["type"]]
        if not isinstance(coords, list) or len(coords) < minimum or (geom["type"] == "point" and len(coords) != 1):
            _error(entity["id"] + ": geometry.coords length")
        for point in coords:
            if not isinstance(point, (list, tuple)) or len(point) != 2 or not all(_number(n) for n in point):
                _error(entity["id"] + ": coordinates must be finite [x,y] pairs")


def ingest(doc):
    """Validate, copy, and explicitly report every authored missing-input fill."""
    _validate(doc)
    catalog = load()
    missing, warnings, entities, by_class = [], [], [], {}
    bounds = tuple(doc["frame"]["bounds_m"])
    for original in doc["entities"]:
        entity = deepcopy(original)
        cls = entity["class"]
        if cls not in SEMANTIC_CLASSES:
            entity["original_class"] = cls
            entity["class"] = "unknown"
            warnings.append(f"{entity['id']}: unknown class {cls!r} retained as unknown / 未知类别保留")
        attrs = entity["attributes"]
        # The shared schema requires every non-catalog name to be explicit attr.*.
        proposed = [key for key in attrs if key not in catalog.fields_by_key]
        entity["proposed_attributes"] = proposed
        for key in proposed:
            warnings.append(f"{entity['id']}: {key} is explicitly proposed / 显式新增属性")
        for key in REQUIRED_ATTRIBUTES.get(entity["class"], ()):
            if key not in catalog.fields_by_key:
                _error("required attribute missing from catalog: " + key)
            if key not in attrs or attrs[key] is None:
                value = deepcopy(_DEFAULTS[key])
                attrs[key] = value
                missing.append({"entity_id": entity["id"], "class": entity["class"],
                                "attribute": key, "filled_with": value,
                                "fill_source": "authored_default"})
        for key, value in attrs.items():
            if value is None or key not in catalog.fields_by_key:
                continue
            domain = catalog.enum_domain(key)
            if catalog.fields_by_key[key].get("type") == "enum" and domain and value not in domain:
                if key == "road.surface" and isinstance(value, str) and value in SOURCE_ROAD_SURFACES:
                    warnings.append(f"{entity['id']}: road.surface {value!r} uses explicit source vocabulary extension: {SOURCE_ROAD_SURFACES[value]}; not a workbook enum or passability certification")
                else:
                    _error(f"{entity['id']}: {key} value {value!r} outside catalog domain")
        for key in ("building.capacity", "building.floors", "road.width_m"):
            if key in attrs and (not _number(attrs[key]) or attrs[key] < 0):
                _error(f"{entity['id']}: {key} must be nonnegative")
        if any(not (0 <= x <= bounds[2] and 0 <= y <= bounds[3]) for x, y in entity["geometry"]["coords"]):
            warnings.append(f"{entity['id']}: geometry extends outside frame / 几何超出边界")
        entities.append(entity)
        by_class.setdefault(entity["class"], []).append(entity)
    if doc["source"]["kind"] == "authored":
        warnings.append(AUTHORED_NOTE)
    if missing:
        warnings.append(f"{len(missing)} required attributes filled with explicit authored defaults / 缺失输入已显式记录")
    if not by_class.get("road"):
        warnings.append("No annotated road network; ground routing may fail / 缺少道路标注")
    if any(e["class"] == "slope" for e in entities):
        warnings.append("U07: slope region identity only; elevation ignored / 忽略高程")
    return SemanticLayer(doc["source"]["kind"], entities, by_class, bounds,
                         missing, warnings, deepcopy(doc["source"]))


def load_upstream(path):
    """Read an upstream export without changing it; ingest performs validation."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def authored_layer(scenario_id, seed, bounds=(0, 0, 2000, 2000)):
    """Return a reproducible labelled map, not a terrain or segmentation product."""
    catalog = load()
    if scenario_id not in catalog.scenarios:
        raise ValueError("未知情境 / Unknown scenario: " + str(scenario_id))
    if len(bounds) != 4 or tuple(bounds[:2]) != (0, 0) or not all(_number(n) for n in bounds) or min(bounds[2:]) <= 0:
        _error("authored bounds must be [0,0,W,H]")
    width, height = bounds[2:]
    r = rng_for(seed, "authored_layout", scenario_id)
    family = catalog.scenarios[scenario_id]["family"]
    wild = family == "野外" or scenario_id in ("SC07", "SC08", "SC09")
    underground = scenario_id in ("SC14", "SC15", "SC16")
    watery = scenario_id in ("SC03", "SC04", "SC05", "SC13")
    xmax = 0.38 if scenario_id == "SC05" else 0.90
    xs = [width * (0.08 + i * (xmax - 0.08) / 3) for i in range(4)]
    ys = [height * v for v in (0.10, 0.36, 0.63, 0.90)]
    entities = []
    surface = "snow" if scenario_id == "SC12" else "soil" if wild else "concrete" if underground else "asphalt"

    def add(eid, cls, typ, coords, attrs=None):
        entities.append({"id": eid, "class": cls,
                         "geometry": {"type": typ, "coords": [list(p) for p in coords]},
                         "attributes": attrs or {}, "label_source": "authored"})

    def rect(x0, y0, x1, y1):
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]

    road_index = 0
    for row in range(4):
        for col in range(4):
            for nr, nc in ((row, col + 1), (row + 1, col)):
                if nr >= 4 or nc >= 4:
                    continue
                road_index += 1
                add(f"road-{road_index}", "road", "polyline",
                    [(xs[col], ys[row]), (xs[nc], ys[nr])],
                    {"road.width_m": 3.0 if underground else (4.0 if wild else 7.0),
                     "road.surface": surface,
                     "road.from_node": f"junction-{row}-{col}",
                     "road.to_node": f"junction-{nr}-{nc}"})
    count = 8 if wild or underground else 12
    # Buildings stand beside road segments; entrances are on road-facing edges.
    for i in range(count):
        row, col = divmod(i, 3)
        x = (xs[col] + xs[col + 1]) / 2
        yroad = ys[row]
        bw = min(width * r.uniform(0.025, 0.045), (xs[col + 1] - xs[col]) * 0.32)
        bh = height * r.uniform(0.025, 0.045)
        y = yroad + height * 0.045
        use = "industrial" if scenario_id in ("SC15", "SC17", "SC18") else r.choice(["residential", "commercial", "public", "storage"])
        entrance = f"entrance-{i + 1}"
        add(f"building-{i + 1}", "building", "polygon", rect(x-bw/2, y, x+bw/2, y+bh),
            {"building.use": use, "building.structure": r.choice(["timber", "masonry", "RC", "steel"]),
             "building.floors": r.randint(1, 3), "building.capacity": r.randint(3, 9),
             "building.entrance_ids": [entrance]})
        add(entrance, "facility", "point", [(x, y)],
            {"entity.parent_id": f"building-{i + 1}", "attr.role": "entrance"})
        # Spur ending at the entrance; main-network intersection splitting occurs in truth.
        road_index += 1
        add(f"road-{road_index}", "road", "polyline", [(x, yroad), (x, y)],
            {"road.width_m": 2.8, "road.surface": surface})
    for i in range(5 if wild else 3):
        x = width * r.uniform(0.12, max(0.13, xmax-0.08))
        y = height * r.uniform(0.18, 0.82)
        add(f"trees-{i+1}", "tree_cluster", "polygon",
            rect(x, y, min(width, x+width*0.035), min(height, y+height*0.045)))
    add("open-area-1", "open_area", "polygon", rect(width*0.02, height*0.01, width*0.18, height*0.065))
    for i, (x, y, kind) in enumerate(((xs[0], ys[0], "supply"), (xs[0], ys[3], "charging")), 1):
        # Catalog enum spelling is authoritative even if a producer uses a synonym.
        domain = catalog.enum_domain("station.kind")
        if kind not in domain:
            kind = next((v for v in domain if ("charg" if i == 2 else "supply") in v), domain[0])
        add(f"station-{i}", "station", "point", [(x, y)],
            {"station.kind": kind, "station.capacity": 12, "station.power_W": 1500.0})
    # Authored rescue/medical receiving point (F084 domain "救援/医疗接收点"): handover needs a
    # compatible receiver; supply/charging stations never acknowledge a person handover.
    add("station-3", "station", "point", [(xs[0], ys[1])],
        {"station.kind": "medical", "station.capacity": 12, "station.power_W": 500.0,
         "attr.note_zh": "作者设定的医疗/救援接收点"})
    add("coordination-post", "facility", "point", [(xs[0], ys[0])], {"attr.role": "coordination_post"})
    if watery:
        wx = width * (0.40 if scenario_id == "SC05" else 0.72)
        add("water-1", "water", "polygon", rect(wx, 0, width, height))
        if scenario_id in ("SC04", "SC05", "SC13"):
            add("coast-1", "coast", "polyline", [(wx, 0), (wx, height)])
        if scenario_id == "SC03":
            add("bridge-1", "bridge", "polygon", rect(width*.70, height*.35, width*.92, height*.37))
    if wild and scenario_id != "SC11":
        add("slope-1", "slope", "polygon", rect(width*.42, height*.45, width*.95, height*.95),
            {"attr.role": "mountain_region", "attr.elevation_ignored": True})
    if underground:
        if scenario_id in ("SC14", "SC16"):
            add("water-1", "water", "polygon", rect(width*.60, height*.34, width*.90, height*.38))
        for i in range(2):
            add(f"underground-{i+1}", "underground", "polygon",
                rect(width*.06, ys[i+1]-height*.025, width*.94, ys[i+1]+height*.025),
                {"attr.role": {"SC14": "metro", "SC15": "mine_tunnel", "SC16": "cave"}[scenario_id]})
    return {"schema": SCHEMA,
            "source": {"kind": "authored", "producer": "rescue_env.authored_layer/v1",
                       "terrain_ref": None, "note": AUTHORED_NOTE},
            "frame": {"crs": "local_ENU_m", "bounds_m": list(bounds)}, "entities": entities}
