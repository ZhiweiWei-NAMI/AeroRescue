#!/usr/bin/env python3
"""Check portable Phase-1 review archive contents without trusting member paths."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aerorescue.phase1.schema_v2 import validate_phase1_v2_record


def main():
    archive=ROOT/'runs/colab_bundle/aerorescue_phase1_colab_bundle.tar.gz'
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        for entry in members:
            path=Path(entry.name)
            assert not path.is_absolute() and '..' not in path.parts
            assert not entry.issym() and not entry.islnk(),entry.name
        names={m.name for m in members if m.isfile()}
        def load_json(path):
            return json.load(tar.extractfile(path))
        rows=[json.loads(line) for line in tar.extractfile('data/phase1_v2/decision_records_v2.jsonl') if line.strip()]
        seq=[json.loads(line) for line in tar.extractfile('data/phase1_v2/sequential_diagnostic_records.jsonl') if line.strip()]
        for rec in rows:
            validate_phase1_v2_record(rec)
        for rec in rows+seq:
            paths=[v['rgb_path'] for v in rec['model_input']['aerial_ground_views'].values()]
            geometry=rec['supervision_and_eval']['geometry_paths']
            paths += [geometry['uav_geom_npz'],geometry['ground_geom_npz']]
            for path in paths:
                assert not Path(path).is_absolute(),path
                assert path in names,path
        evaluation=load_json('runs/phase1_local/evaluation_results.json')
        manifest=load_json('runs/colab_bundle/colab_5pairs_manifest.json')
        assert evaluation['audit_status']=='not_accepted'
        assert manifest['status']=='audit_not_accepted_colab_execution_unimplemented'
        for entry in members:
            if entry.isfile() and entry.name.endswith('.py'):
                compile(tar.extractfile(entry).read(),entry.name,'exec')
    digest=hashlib.sha256()
    with archive.open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):
            digest.update(block)
    report={'archive':str(archive),'status':'passed_portable_review_bundle_checks','bytes':archive.stat().st_size,
        'sha256':digest.hexdigest(),'regular_files':len(names),'decision_records':len(rows),'sequence_frames':len(seq),
        'all_sensor_resources_resolve_in_archive':True,'research_acceptance':False,'heterogeneous_model_execution_implemented':False}
    (archive.parent/'bundle_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (archive.parent/'aerorescue_phase1_colab_bundle.tar.gz.sha256').write_text(digest.hexdigest()+'  '+archive.name+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
