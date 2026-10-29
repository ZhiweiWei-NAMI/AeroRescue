"""Native API state/evidence checks for the explicitly isolated fixtures."""
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('platform_apis',Path(__file__).parents[1]/'tools/check_platform_apis.py')
apis=importlib.util.module_from_spec(spec);spec.loader.exec_module(apis)


def succeeded(case):
    assert all(s['status']=='succeeded' for s in case['steps']),[(s['action'],s['result']) for s in case['steps']]
    assert not case['contract_violations']
    for step in case['steps']:
        for event in step['events']:
            if event['kind'] in {'rescue.extricated','rescue.loaded','rescue.transport_started','rescue.transport_arrived','rescue.handover_acknowledged'}:
                payload=event['after']
                assert payload['source_seq'] in payload['write_seqs']
                source=next(e for e in step['events'] if e['seq']==payload['source_seq'])
                assert source['action_instance']==step['id']==event['action_instance']
                assert source['layer']=='world'


@pytest.mark.parametrize('loaded_first',[False,True])
def test_cargo_consumption_transport_and_release_preserve_person_mass(loaded_first):
    case=apis.cargo_person_case(loaded_first=loaded_first);succeeded(case)
    assert all(s['mass_after']['consistent'] for s in case['steps'])
    pickup=next(s for s in case['steps'] if s['action']=='A024')
    assert pickup['before']['station-1']['station.inventory']['medical_pack']-pickup['after']['station-1']['station.inventory']['medical_pack']==2
    assert pickup['after']['api-kit']['item.state']=='attached'
    aids=[s for s in case['steps'] if s['action']=='A019']
    assert [s['after']['api-kit']['item.quantity'] for s in aids]==[1,0]
    unload=next(s for s in case['steps'] if s['action']=='A025')
    assert unload['after'][apis.ACTOR]['platform.payload_item_ids']==[]
    assert unload['mass_after']['observed_total_kg']==75
    final=case['steps'][-1]
    assert final['after'][apis.PERSON]['human.handover_status']=='acknowledged'
    assert final['after']['api-receiver']['station.occupied']==1
    assert final['mass_after']['observed_total_kg']==0


def test_ground_movement_after_native_clearance_and_scripted_assessment():
    case=apis.ground_assessment_case();succeeded(case)
    assert case['actor_spec']['mobility']=='wheeled'
    assert 'A001' not in [s['action'] for s in case['steps']]
    for aid in ['A005','A006','A007']:
        step=next(s for s in case['steps'] if s['action']==aid)
        assert step['result']['observations']
    clear=next(s for s in case['steps'] if s['action']=='A012')
    assert clear['before']['road-1']['road.entry_state']=='blocked'
    assert clear['after']['road-1']['road.entry_state']=='open'
    assert clear['after']['road-1']['road.clear_width_m']==7
    assert clear['after']['api-debris']['obstacle.work_remaining']==0
    movement=case['steps'][-1]
    assert movement['before'][apis.ACTOR]['entity.pose']==[20,40]
    assert movement['after'][apis.ACTOR]['entity.pose']==[40,40]
    assessed=[o for o in case['objectives'] if o['kind']=='assess_all_buildings']
    assert assessed and all(o['satisfied'] for o in assessed)


@pytest.mark.parametrize('aid,cap',[('A013','C17'),('A016','C20')])
def test_native_heavy_clearance_and_lifting_restore_access(aid,cap):
    case=apis.clearing_case(aid,cap);succeeded(case);step=case['steps'][0]
    assert step['after']['api-debris']['obstacle.state']=='cleared'
    assert step['after']['api-debris']['obstacle.work_remaining']==0
    assert step['after']['road-1']['road.entry_state']=='open'
    if aid=='A013':assert step['after']['api-debris']['obstacle.volume_m3']==0
    if aid=='A016':assert step['after']['api-debris']['entity.pose']==[32,46]


def test_extrication_has_native_state_and_source_linked_events():
    case=apis.extrication_case();succeeded(case);step=case['steps'][0]
    assert step['before'][apis.PERSON]['human.buriedness']==20
    assert step['after'][apis.PERSON]['human.buriedness']==0
    assert 'extricated' in step['after'][apis.PERSON]['human.support_received']


def test_water_recovery_transport_and_confirmed_receiver_without_air_override():
    case=apis.water_case();succeeded(case)
    assert case['actor_spec']['mobility']=='water'
    first=case['steps'][0]
    assert first['before'][apis.PERSON]['human.location_context']=='water'
    assert first['after'][apis.PERSON]['human.carrier_id']==apis.ACTOR
    assert 'recovered_from_water' in first['after'][apis.PERSON]['human.support_received']
    assert all(s['mass_after']['consistent'] for s in case['steps'])
    final=case['steps'][-1]
    assert final['after'][apis.PERSON]['human.location_context']=='safe_point'
    assert final['after'][apis.PERSON]['human.pose']==[20,32]
    assert final['after']['api-receiver']['station.occupied']==1
    assert final['mass_after']['observed_total_kg']==0
