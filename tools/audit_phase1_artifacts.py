#!/usr/bin/env python3
"""Independently validate on-disk Phase-1 sensor, pose and twin evidence."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aerorescue.phase1 import schema_v2 as S
from aerorescue.iscc.geometry import se3_exp


def check_geometry(rec):
    paths=rec['supervision_and_eval']['geometry_paths']
    geometries=[dict(np.load(paths[f'{name}_geom_npz'])) for name in ('uav','ground')]
    cameras=[]
    for name,geom in zip(('uav','ground'),geometries):
        view=rec['model_input']['aerial_ground_views'][f'{name}_view']
        image=np.asarray(Image.open(view['rgb_path']).convert('RGB'))
        assert image.shape==(384,640,3)
        assert geom['depth'].shape==(384,640) and geom['depth'].dtype==np.float32
        assert geom['instance_id'].dtype==np.uint32
        valid=geom['valid_mask']
        assert np.isfinite(geom['depth']).all()
        assert np.array_equal(geom['depth']>0,valid)
        for key in ('normal_cam','normal_world'):
            assert geom[key].dtype==np.float32 and np.isfinite(geom[key]).all()
            assert np.max(np.abs(np.linalg.norm(geom[key][valid],axis=1)-1))<1e-3
        assert np.max(np.abs(geom['normal_world']@geom['R_c_from_w'].T-geom['normal_cam']))<1e-4
        assert np.array_equal(geom['instance_id']==0,~valid)
        camera=np.eye(4); camera[:3,:3]=geom['R_c_from_w']; camera[:3,3]=-camera[:3,:3]@geom['cam_pose_enu']
        cameras.append(camera)
    expected=cameras[1]@np.linalg.inv(cameras[0])
    truth=np.asarray(rec['supervision_and_eval']['true_pose_e_from_i']['T_e_from_i'])
    pose_error=float(np.max(np.abs(expected-truth)))
    assert pose_error<1e-8
    views=rec['model_input']['aerial_ground_views']
    ground=np.asarray(views['ground_view']['extrinsics_w_from_c'])
    assert np.max(np.abs(np.linalg.inv(cameras[1])-ground))<1e-8
    prior=np.linalg.inv(ground)@np.asarray(views['uav_view']['prior_extrinsics_w_from_c'])
    xi=rec['supervision_and_eval']['true_pose_e_from_i']['xi']
    assert np.max(np.abs(prior-se3_exp(xi)@truth))<1e-8
    source=geometries[0]
    assert np.array_equal(source['blind_mask']|source['covis_mask'],source['valid_mask'])
    assert not np.any(source['blind_mask']&source['covis_mask'])
    return pose_error


def main():
    data=ROOT/'data/phase1_v2'
    records=[json.loads(line) for line in (data/'decision_records_v2.jsonl').read_text().splitlines() if line.strip()]
    seq=[json.loads(line) for line in (data/'sequential_diagnostic_records.jsonl').read_text().splitlines() if line.strip()]
    errors=[]; pairs=defaultdict(list)
    for rec in records:
        S.validate_phase1_v2_record(rec)
        errors.append(check_geometry(rec))
        if rec['metadata']['sample_kind']=='challenge_matched':
            pairs[rec['metadata']['pair_group_id']].append(rec)
    pair_report=[]
    for pair_id,rows in pairs.items():
        assert len(rows)==2
        S.assert_matched_pair_noninterference(*rows)
        view=lambda r,role:np.asarray(Image.open(r['model_input']['aerial_ground_views'][role+'_view']['rgb_path']))
        ground_difference=int(np.any(view(rows[0],'ground')!=view(rows[1],'ground'),axis=-1).sum())
        aerial_difference=int(np.any(view(rows[0],'uav')!=view(rows[1],'uav'),axis=-1).sum())
        assert ground_difference==0 and aerial_difference>0
        pair_report.append({'pair_group_id':pair_id,'ground_different_pixels':ground_difference,'aerial_different_pixels':aerial_difference,
                            'equal_prior':[.5,.5],'ego_input_sha256':S.sha256_text(S.canonical_json(S.canonical_ego_input(rows[0])))})
    assert len(seq)==15
    assert [r['trajectory']['step'] for r in seq]==list(range(1,16))
    assert np.all(np.diff([r['metadata']['sim_time_s'] for r in seq])>0)
    sequence_overlap=[]
    for rec in seq:
        assert 'branch_evaluations' not in rec['supervision_and_eval']
        errors.append(check_geometry(rec))
        common=int(np.load(rec['supervision_and_eval']['geometry_paths']['uav_geom_npz'])['covis_mask'].sum())
        sequence_overlap.append(common)
    assert sequence_overlap[8]==sequence_overlap[9]==0
    assert all(sequence_overlap[i]>0 for i in (0,7,10,14))
    report={'status':'passed_sensor_pose_twin_artifact_checks','records':len(records),'sequence_frames':len(seq),
            'max_relative_pose_chain_error':max(errors),'pairs':pair_report,
            'sequence_covisible_pixels':sequence_overlap,
            'scope':'This artifact check does not certify missing RGB-only deployment, SOTA or Colab implementations.'}
    path=ROOT/'runs/phase1_local/artifact_validation.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
