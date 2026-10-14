"""One small CLI from source acquisition to model export and planner comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapter import ParticipationFilter
from .compare import compare
from .data import acquire, prepare, read_jsonl, write_json, write_jsonl
from .episodes import generate, label_episode, export_events, read_platform_events
from .planner import load_planner
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    acq = commands.add_parser("acquire", help="Download only three pinned official SAREnv environments")
    acq.add_argument("--root", type=Path, default=Path("data/raw/sarenv"))
    prep = commands.add_parser("prepare", help="Parse source maps and report their actual semantics")
    prep.add_argument("--root", type=Path, default=Path("data/raw/sarenv"))
    prep.add_argument("--output", type=Path, default=Path("data/processed/sarenv-contexts.json"))
    demo = commands.add_parser("run", help="Run map-grounded procedural generation, planner labeling, CPU training, artifact reload and paired planning")
    demo.add_argument("--platform", type=Path, required=True)
    demo.add_argument("--root", type=Path, default=Path("data/raw/sarenv"))
    demo.add_argument("--output", type=Path, default=Path("runs/cpu-prototype"))
    demo.add_argument("--episodes", type=int, default=128)
    export = commands.add_parser("export", help="Convert normalized platform events into decision episodes and role records")
    source = export.add_mutually_exclusive_group(required=True)
    source.add_argument("--events", type=Path)
    source.add_argument("--run", type=Path, help="Read the actual platform run directory with its native WAL codec")
    export.add_argument("--platform", type=Path, required=True)
    export.add_argument("--run-id", required=True)
    export.add_argument("--map-id", required=True)
    export.add_argument("--split", choices=["train", "validation", "evaluation"], required=True)
    export.add_argument("--output", type=Path, required=True)
    fit = commands.add_parser("train", help="Train a compact reference filter from labeled JSONL")
    fit.add_argument("--records", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "acquire":
        result = acquire(args.root, ["1", "4", "6"])
        print(json.dumps({"files": len(result["files"]), "bytes": sum(f["bytes"] for f in result["files"]), "manifest": str(args.root / "source.json")}, indent=2))
    elif args.command == "prepare":
        result = prepare(args.root, args.output)
        print(json.dumps(result["environments"], ensure_ascii=False, indent=2))
    elif args.command == "train":
        print(json.dumps(train(args.records, args.output), indent=2))
    else:
        solve, planner_source = load_planner(args.platform)
        if args.command == "run":
            episodes = generate(args.root, args.episodes)
        else:
            events = read_platform_events(args.run, args.platform) if args.run else read_jsonl(args.events)
            episodes = export_events(events, args.run_id, args.split, args.map_id)
        args.output.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output / "episodes.jsonl", episodes)
        records = [row for episode in episodes for row in label_episode(episode, solve, planner_source)]
        records_path = args.output / "roles.jsonl"
        write_jsonl(records_path, records)
        if args.command == "export":
            write_jsonl(args.output / "platform-events.jsonl", events)
            print(json.dumps({"episodes": len(episodes), "role_records": len(records), "output": str(args.output)}))
            return
        training = train(records_path, args.output / "participation-filter.json")
        loaded = ParticipationFilter.load(args.output / "participation-filter.json")
        result = compare(episodes, loaded, solve)
        write_json(args.output / "comparison.json", result)
        write_json(args.output / "training-summary.json", training)
        print(json.dumps({"episodes": len(episodes), "role_records": len(records), "unknown_labels": sum(r["label"] is None for r in records),
                          "training": training, "comparison": result["summary"], "artifact": str(args.output / "participation-filter.json")}, indent=2))


if __name__ == "__main__":
    main()
