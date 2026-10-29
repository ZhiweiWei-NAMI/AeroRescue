#!/usr/bin/env python3
"""Print only witnessed rescue/road milestones and explicitly identify misses."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rescue_scenes.evidence import chain_timeline


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('run_dir',type=Path); args=parser.parse_args()
    initial=json.loads((args.run_dir/'initial_truth.json').read_text())
    events=[json.loads(line) for line in (args.run_dir/'events.jsonl').read_text().splitlines() if line]
    timeline=chain_timeline(events,initial)
    print('Scripted v0.5 sensing; stages below are event evidence, not planned outcomes.')
    for row in timeline:
        detail=' '.join(f'{k}={v}' for k,v in row.items() if k not in {'t','seq','stage','target'})
        print(f"{row['t']:8.2f}s #{row['seq']:7d} {row['stage']:18s} {row['target']} {detail}")
    for person in (e for e in initial['entities'].values() if e['cls']=='human'):
        observed={r['stage'] for r in timeline if r['target']==person['id']}
        missing=[s for s in ('found','contact','extricated','loaded','transported','handover_ack') if s not in observed]
        print(person['id']+': '+('missing evidence: '+', '.join(missing) if missing else 'all rescue-chain stages witnessed'))
    road_stages={r['stage'] for r in timeline}
    print('Road chain missing evidence: '+(', '.join(s for s in ('blocked','cleared','passed') if s not in road_stages) or 'none'))
    result_path=args.run_dir/'result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        print('Failures by cause:',json.dumps(result['failures_by_cause'],ensure_ascii=False))
        print('World objectives:',json.dumps(result['objectives'],ensure_ascii=False))
    else:
        print('result.json absent: run did not close normally; timeline is partial.')


if __name__=='__main__': main()
