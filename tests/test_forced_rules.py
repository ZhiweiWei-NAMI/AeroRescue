import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("forced_rules",ROOT/"tools/run_forced_rules.py")
rules=importlib.util.module_from_spec(spec)
spec.loader.exec_module(rules)


def test_sc09_forced_rules_use_native_actions_and_real_transition_sources(tmp_path):
    config=ROOT/"configs/SC09_upstream_allocated.json"
    original=config.read_bytes()
    output=tmp_path/"summary.json"
    report=rules.verify(config,output)
    assert config.read_bytes()==original
    assert report["rule_targets_satisfied"] is True
    assert report["all_objectives_satisfied"] is True
    assert report["camera_acceptance"] is False
    assert report["within_original_horizon"] is True
    assert all(a["status"]=="succeeded" for a in report["actions"])
    assert len([o for o in report["objectives"] if o["kind"]=="isolate_hazard" and o["satisfied"]])==2
    handover=[e for e in report["events"] if e["kind"]=="rescue.handover_acknowledged"]
    assert len(handover)==1
    assert handover[0]["after"]["confirmed"] is True
    assert handover[0]["action_instance"] in {a["id"] for a in report["actions"] if a["action"]=="A022" and a["status"]=="succeeded"}
    assert handover[0]["after"]["source_seq"] in handover[0]["after"]["write_seqs"]
    assert any(a["action"]=="A024" and a["status"]=="succeeded" for a in report["actions"])
    assert any(a["action"]=="A019" and a["status"]=="succeeded" for a in report["actions"])
    assert json.loads(output.read_text())["mode"]=="forced_rule_verification"
    assert output.stat().st_size<100_000


def test_empty_objectives_are_not_rule_success():
    from types import SimpleNamespace
    from rescue_scenes.forced_rules import ForcedRuleDriver
    sim=SimpleNamespace(agents={}, instances={}, t=0., events=[], evaluator=SimpleNamespace(update=lambda:None))
    env=SimpleNamespace(sim=sim, objectives=lambda:[])
    driver=ForcedRuleDriver(env, {'until_s':10})
    report=driver.snapshot()
    assert report['rule_targets_satisfied'] is False
    assert report['all_objectives_satisfied'] is False


def test_existing_upstream_product_refuses_changed_generation_inputs(tmp_path, monkeypatch):
    from rescue_scenes import upstream_scene
    import hashlib
    import pytest
    product=tmp_path/'tools/generated/upstream/SC09-1'
    product.mkdir(parents=True)
    catalog=tmp_path/'catalog.json';catalog.write_text('{}')
    generation=tmp_path/'rules.json';generation.write_text('{}')
    provenance={'catalog_sha256':hashlib.sha256(catalog.read_bytes()).hexdigest(),
                'config_sha256':hashlib.sha256(generation.read_bytes()).hexdigest(),
                'coverage_target_ids':[], 'input_roots':[]}
    (product/'scene.json').write_text('{}')
    (product/'scene-manifest.json').write_text(json.dumps({'generation':{'provenance':provenance}}))
    cfg={'background':{'id':'SC09'},'seed':1,'placement':{'upstream_generated_scene':{
        'catalog':str(catalog),'coverage_targets':['new-target'],'input_roots':[]}}}
    monkeypatch.setattr(upstream_scene,'ROOT',tmp_path)
    monkeypatch.setattr(upstream_scene,'generation_config',lambda config:str(generation))
    with pytest.raises(ValueError,match='Existing upstream inputs differ'):
        upstream_scene.load_scene(cfg)
    cfg['placement']['upstream_generated_scene']['coverage_targets']=[]
    catalog.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='Existing upstream inputs differ'):
        upstream_scene.load_scene(cfg)
    assert (product/'scene.json').read_text()=='{}'


def test_sc07_suppresses_scheduled_reignition_with_native_actions(tmp_path):
    report=rules.verify(ROOT/'configs/SC07_upstream_allocated.json',tmp_path/'sc07.json')
    assert report['all_objectives_satisfied'] is True
    assert report['within_original_horizon'] is True
    assert any(r.get('wait_for_fire_spread_s')==300 for r in report['rules'])
    assert any(a['action']=='A028' and a['status']=='succeeded' for a in report['actions'])
    suppression=[a for a in report['actions'] if a['action']=='A062']
    assert any(a['t'] < 300 for a in suppression)
    assert any(a['t'] > 300 and a['status']=='succeeded' for a in suppression)
    assert report['rule_sequence_finished_s'] > 300
