#!/usr/bin/env python3
"""Run the five real heterogeneous HF pairs on the same RGB-D wire protocol.

Dry smoke validates the local dataset/interface without downloading or loading
weights. Full runs require sufficient GPU memory (A100 40 GB or larger recommended
for the shared 7B receiver and backward Shadow Probes).
"""
from __future__ import annotations
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def pairs_from_manifest(manifest,pair_id):
    return [p for p in manifest['pairs'] if pair_id=='all' or p['pair_id']==pair_id]


def load_records(path):
    from aerorescue.phase1.schema_v2 import validate_phase1_v2_record
    records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for record in records:
        validate_phase1_v2_record(record)
        for view in record['model_input']['aerial_ground_views'].values():
            for key in ('rgb_path','depth_path'):
                resource=Path(view[key]);resource=resource if resource.is_absolute() else ROOT/resource
                if not resource.is_file():raise FileNotFoundError(resource)
                view[key]=str(resource)
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'runs/colab_bundle/colab_5pairs_manifest.json')
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/phase1_v2/decision_records_v2.jsonl')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'runs/colab_5pairs_outputs')
    parser.add_argument('--pair-id',default='all');parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--dry-run-smoke',action='store_true')
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text());pairs=pairs_from_manifest(manifest,args.pair_id)
    if not pairs:parser.error('unknown or empty pair selection')
    records=load_records(args.dataset)
    from aerorescue.iscc.hf_backends import source_family
    configured=[dict(pair_id=p['pair_id'],sender=p['sender_uav'],receiver=p['receiver_ugv'],family=source_family(p['sender_uav'])) for p in pairs]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summary=dict(schema='aerorescue.colab.5pairs.run_report/v2',timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        dataset_records=len(records),pairs_configured=configured,dry_run_smoke=args.dry_run_smoke,
        models_executed=False,status='schema_and_backend_imports_passed' if args.dry_run_smoke else 'running',results=[])
    status_file=args.output_dir/'colab_run_status.json'
    status_file.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    if args.dry_run_smoke:
        print(json.dumps(summary,ensure_ascii=False,indent=2));return 0
    import torch
    from aerorescue.iscc.hf_backends import HeterogeneousQwenRuntime
    from aerorescue.iscc.audited_pair0 import run_audited_pair0
    spec=importlib.util.spec_from_file_location('main_phase1_figures',ROOT/'tools/run_phase1_full_local.py')
    plotting=importlib.util.module_from_spec(spec);spec.loader.exec_module(plotting)
    failed=False
    for pair in pairs:
        destination=args.output_dir/pair['pair_id'];destination.mkdir(parents=True,exist_ok=True)
        runtime=None
        try:
            runtime=HeterogeneousQwenRuntime(pair['sender_uav'],pair['receiver_ugv'],args.device,revision=pair.get('revision'))
            (destination/'diagnostics').mkdir(exist_ok=True)
            result=run_audited_pair0(records,destination,plotting._plot_diagnostic_figures,runtime=runtime,pair_description=runtime.pair_description)
            result['model_provenance']=runtime.model_provenance
            (destination/'evaluation_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            summary['models_executed']=True
            summary['results'].append(dict(pair_id=pair['pair_id'],status='executed',audit_status=result['audit_status'],output=str(destination)))
        except Exception as exc:
            failed=True
            (destination/'failure.log').write_text(traceback.format_exc())
            summary['results'].append(dict(pair_id=pair['pair_id'],status='failed',error=type(exc).__name__+': '+str(exc)))
        finally:
            del runtime;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            status_file.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    summary['status']='completed_with_errors' if failed else 'executed_all_configured_pairs'
    status_file.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 1 if failed else 0


if __name__=='__main__':raise SystemExit(main())
