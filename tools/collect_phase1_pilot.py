#!/usr/bin/env python3
"""Collect the 24-episode phase-1 pilot decision dataset (record schema v2).

Runs 4 scenes x 6 seeds with the existing ``Environment`` and
``ForcedRuleDriver``, then writes ``decision_records.jsonl`` and
``pilot_summary.json`` under ``data/phase1_pilot/``.  Every candidate is rolled
out on a ``deepcopy(sim)`` clone; ``supervision`` carries the real counterfactual
utilities, the epsilon-acceptable set, paired VoI and the ISCC budget split.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerorescue.phase1.collector import DATA_DIR, SCENES, SEEDS, collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DATA_DIR)
    parser.add_argument("--scenes", nargs="+", default=list(SCENES), choices=SCENES)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    summary = collect(out_dir=args.out, scenes=tuple(args.scenes), seeds=tuple(args.seeds))
    print(json.dumps({key: summary[key] for key in
                      ("record_schema", "total_episodes", "total_records", "splits",
                       "regimes", "records_by_scene", "records_per_episode",
                       "candidates_per_record", "label_balance", "acceptable_set",
                       "paired_voi", "center_memory", "iscc_budget", "rollout_status",
                       "collection_wall_s")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
