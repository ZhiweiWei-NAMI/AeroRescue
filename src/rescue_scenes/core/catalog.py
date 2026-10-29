"""Read-only catalog loader and independent stdlib XLSX count cross-check.

Chinese source text is retained verbatim. Normalization changes container shapes,
not vocabulary. Neither loading nor verification writes the source documents.
"""

from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import posixpath
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

from ..paths import PRIVATE_ROOT


_ROOT = Path(__file__).resolve().parent.parent
_JSON_NAME = "AeroRescue_任务行动能力状态字典_v1.json"
_XLSX_NAME = "AeroRescue_救援场景任务行动能力状态总目录_v1.xlsx"
EXPECTED = {
    "scenarios": 18, "overlays": 8, "tasks": 96, "sequences": 112,
    "steps": 523, "actions": 66, "capabilities": 49, "fields": 385,
    "sources": 26, "quotes": 15, "rules": 18,
}
_DECLARED = {
    "救援情境": "scenarios", "可叠加实验条件": "overlays",
    "任务类型": "tasks", "参考行动序列": "sequences",
    "原子操作/工具": "actions", "能力": "capabilities", "字段总数": "fields",
}
# Inspected sheet names: 11_原生类别81 is NOT another field table.
_SHEET_TABLES = {
    "01_": ("quotes", "需求ID"), "02_": ("sources", "来源ID"),
    "03_": ("scenarios", "情境ID"), "04_": ("overlays", "条件ID"),
    "05_": ("tasks", "任务ID"), "06_": ("sequences", "序列ID"),
    "07_": ("steps", "序列ID"), "08_": ("actions", "操作ID"),
    "09_": ("capabilities", "能力ID"), "10_": ("fields", "字段ID"),
    "13_": ("rules", "规则ID"),
}
_X = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _index(rows):
    return {row["id"]: row for row in rows}


class Catalog:
    """Normalized catalog, with raw dict records for all dictionary-backed tables."""

    def __init__(self, path):
        self.path = Path(path)
        self.xlsx_path = self.path.with_name(_XLSX_NAME)
        data = self.path.read_bytes()
        self.json_sha256 = sha256(data).hexdigest()
        self.raw = json.loads(data.decode("utf-8"))
        self.xlsx_sha256 = (sha256(self.xlsx_path.read_bytes()).hexdigest()
                            if self.xlsx_path.is_file() else None)
        self.counts_declared = self.raw["counts"]
        for name in ("scenarios", "tasks", "sequences", "actions",
                     "capabilities", "fields", "sources"):
            setattr(self, name, _index(self.raw[name]))
        self.fields_by_key = {row["key"]: row for row in self.fields.values()}
        self.overlays = {
            row[0]: {"id": row[0], "name": row[1],
                     "fields": row[2].split(), "tasks": row[3].split()}
            for row in self.raw["overlays"]
        }
        self.quotes = {
            row[0]: dict(zip(("id", "origin", "kind", "text", "scope", "constraint"), row))
            for row in self.raw["quotes"]
        }
        self.rules = {
            row[0]: {"id": row[0], "name": row[1], "inputs": row[2].split(),
                     "description": row[3], "outputs": row[4].split(),
                     "sources": row[5].split()}
            for row in self.raw["rules"]
        }
        self.steps = {seq_id: [] for seq_id in self.sequences}
        for row in self.raw["steps"]:
            options = list(row["actions"])
            self.steps.setdefault(row["sequence"], []).append({
                "step": row["step"], "options": options,
                "choose_one": ("按条件选择其中一项" in row["rule"] or len(options) > 1),
                "actor": row["actor"], "rule": row["rule"], "terminal": row["terminal"],
            })

    def task_methods(self, task_id):
        matches = [seq for seq in self.sequences.values() if task_id in seq["tasks"]]
        return ([seq["id"] for seq in matches if seq["kind"] == "参考主路径"]
                + [seq["id"] for seq in matches if seq["kind"] == "条件分支"])

    def enum_domain(self, key):
        domain = self.fields_by_key[key]["domain"].split("；", 1)[0]
        return [value.strip() for value in domain.split("/") if value.strip()]

    def _xlsx_counts(self):
        """Resolve workbook relationships, find headers, then count data rows."""
        sheet_counts, table_counts, problems = {}, {}, []
        with ZipFile(self.xlsx_path) as book:
            strings = []
            if "xl/sharedStrings.xml" in book.namelist():
                strings = ["".join(node.itertext()) for node in
                           ET.fromstring(book.read("xl/sharedStrings.xml")).findall("x:si", _X)]
            rels = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
            targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
            workbook = ET.fromstring(book.read("xl/workbook.xml"))
            for sheet in workbook.findall("x:sheets/x:sheet", _X):
                name = sheet.attrib["name"]
                target = targets[sheet.attrib[f"{{{_REL}}}id"]]
                member = (target.lstrip("/") if target.startswith("/") else
                          posixpath.normpath(posixpath.join("xl", target)))
                root = ET.fromstring(book.read(member))
                rows = []
                for row in root.findall("x:sheetData/x:row", _X):
                    values = []
                    for cell in row.findall("x:c", _X):
                        value = cell.find("x:v", _X)
                        if cell.attrib.get("t") == "inlineStr":
                            inline = cell.find("x:is", _X)
                            text = "" if inline is None else "".join(inline.itertext())
                        else:
                            text = "" if value is None else (value.text or "")
                            if cell.attrib.get("t") == "s" and text:
                                text = strings[int(text)]
                        values.append(text.strip())
                    rows.append(values)
                table = next((entry for prefix, entry in _SHEET_TABLES.items()
                              if name.startswith(prefix)), None)
                if table:
                    kind, header = table
                    header_index = next((i for i, values in enumerate(rows)
                                         if values and values[0] == header
                                         and sum(bool(v) for v in values) > 1), None)
                else:
                    # Other table-backed sheets share the labelled multi-column
                    # header pattern. 00_总览 is an editorial summary, not a table.
                    header_index = (None if name.startswith("00_") else next(
                        (i for i, values in enumerate(rows)
                         if sum(bool(v) for v in values) > 1), None))
                if header_index is None:
                    sheet_counts[name] = 0
                    if table:
                        problems.append(f"XLSX {name}: 未找到表头 / header not found")
                    continue
                count = sum(any(values) for values in rows[header_index + 1:])
                sheet_counts[name] = count
                if table:
                    if kind in table_counts:
                        problems.append(f"XLSX {name}: 重复目录表 / duplicate table {kind}")
                    table_counts[kind] = count
        return sheet_counts, table_counts, problems

    def verify(self):
        counts = {name: (sum(map(len, self.steps.values())) if name == "steps"
                         else len(getattr(self, name))) for name in EXPECTED}
        problems = []
        for kind, expected in EXPECTED.items():
            if counts[kind] != expected:
                problems.append(f"{kind}: {counts[kind]} != {expected}")
            if len(self.raw[kind]) != counts[kind]:
                problems.append(f"{kind}: 重复ID / duplicate IDs")
        if len(self.fields_by_key) != len(self.fields):
            problems.append("fields: 重复字段名 / duplicate keys")
        for declared, kind in _DECLARED.items():
            if self.counts_declared.get(declared) != counts[kind]:
                problems.append(f"counts.{declared}: 声明与实际不符 / declared count mismatch")
        xlsx_sheets, table_counts = {}, {}
        try:
            xlsx_sheets, table_counts, xlsx_problems = self._xlsx_counts()
            problems.extend(xlsx_problems)
            xlsx_matches_json = not xlsx_problems
            for kind in (entry[0] for entry in _SHEET_TABLES.values()):
                if table_counts.get(kind) != counts[kind]:
                    xlsx_matches_json = False
                    problems.append(f"XLSX {kind}: {table_counts.get(kind)} != JSON {counts[kind]}")
        except (OSError, BadZipFile, ET.ParseError, KeyError, ValueError, IndexError) as exc:
            xlsx_matches_json = False
            problems.append(f"XLSX 无法核对 / cannot cross-check: {exc}")
        return {"ok": not problems, "counts": counts, "expected": dict(EXPECTED),
                "xlsx_sheets": xlsx_sheets, "xlsx_matches_json": xlsx_matches_json,
                "problems": problems}


@lru_cache(maxsize=None)
def _load_cached(path):
    return Catalog(path)


def load(path=None):
    """Return the cached catalog at a canonical absolute JSON path."""
    if path is not None:
        path = Path(path)
    else:
        packaged = Path(__file__).resolve().parent.parent / "catalog" / _JSON_NAME
        path = packaged if packaged.is_file() else PRIVATE_ROOT / "rescue_scenes" / "catalog" / _JSON_NAME
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing private catalog input: {path}. "
                                "Set RESCUE_PRIVATE_ROOT to the external scene input root.")
    return _load_cached(str(path))


def verify():
    return load().verify()


def task_methods(task_id):
    return load().task_methods(task_id)


def enum_domain(key):
    return load().enum_domain(key)
