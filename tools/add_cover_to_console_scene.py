#!/usr/bin/env python3
"""Add a static ``cover_url`` to the rescue runs' console-scene.json files.

The AAS viewer header validator (``validateHeader`` in
``AeroAgentSim/frontend/src/feeds/http.ts``) passes ``scene`` through verbatim
and never rejects unknown keys, so an extra ``cover_url`` is safe. The scene
payload is served by ``authoring/replay.py:scene_header`` after a plain
``json.loads``.

This helper inserts the key textually immediately after the opening brace so
every other byte of the file (indent, key order, trailing newline) is kept.
It validates the JSON before and after, skips files that already carry
``cover_url``, and is safe to re-run.

It is intended to be run once by the operator:

    python3 tools/add_cover_to_console_scene.py            # patch in place
    python3 tools/add_cover_to_console_scene.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "scene_data"
    / "blockout_geometry_20261023"
    / "marble_aas_visual_20261024"
    / "runtime"
    / "out_cli"
)
RUN_DIR = re.compile(r"^(SC\d{2}|GT\d{2})_rescue\.scenario-", re.IGNORECASE)


def cover_entry(sid: str, pretty: bool) -> str:
    entry = f'"cover_url": "/data/rescue_covers/{sid.upper()}.jpg",'
    # A pretty-printed file already opens with "{\n"; start a new indented line.
    # A compact one is patched inline without inventing whitespace.
    return f"\n  {entry}" if pretty else entry


def patch(path: Path, sid: str, dry_run: bool) -> str:
    text = path.read_text(encoding="utf-8")
    document = json.loads(text)  # refuse to touch malformed JSON
    if "cover_url" in document:
        return "already set"
    brace = text.find("{")
    if brace < 0:
        raise ValueError(f"{path}: no JSON object")
    entry = cover_entry(sid, pretty=text[brace + 1 : brace + 2] == "\n")
    updated = text[: brace + 1] + entry + text[brace + 1 :]
    json.loads(updated)  # never write something that does not parse
    if not dry_run:
        path.write_text(updated, encoding="utf-8")
    return "patched"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT, help="out_cli directory")
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"not a directory: {args.root}")
    changed = 0
    for run in sorted(args.root.iterdir()):
        match = RUN_DIR.match(run.name)
        if not run.is_dir() or not match:
            continue
        scene = run / "console-scene.json"
        if not scene.is_file():
            print(f"skip {run.name}: no console-scene.json")
            continue
        status = patch(scene, match.group(1), args.dry_run)
        changed += status == "patched"
        print(f"{status}: {scene}")
    print(f"{changed} file(s) {'to patch' if args.dry_run else 'patched'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
