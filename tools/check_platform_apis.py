#!/usr/bin/env python3
"""Small isolated native API fixtures; no scenario rebuild, service or camera."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rescue_scenes.core.api import Environment
from rescue_scenes.core.sim.actions import MODELS

PERSON='human-building-1-1'
ACTOR='api-actor'


def setup(caps, kind='specialist_team', start=(20,40)):
    config={'scenario_id':'SC01','seed':1,'semantics':{'source':'demo'},'tick_s':1,'until_s':600,
            'agents':[{'id':ACTOR,'kind':kind,'start':list(start),'overrides':{
                'capabilities':caps,'payload_kg':200,'battery_Wh':2000,'sensors':['scripted_rgb'],
                'release_jam_rate':0.,'speed_mps':4.}}]}
    env=Environment.from_config(config)
    env.sim.agents[ACTOR].on_tick=lambda sim:None
    return env,config


def initial_fields(env,eid,fields):
    if env.sim.t != 0 or env.sim.instances:
        raise RuntimeError('Fixture initial fields may only be written before native execution')
    for key,value in fields.items():
        env.sim.world.set(eid,key,value,'authored_api_fixture')


def person(env,point=(20,40),trapped=False):
    initial_fields(env,PERSON,{'entity.pose':list(point),'human.mobility':'trapped' if trapped else 'immobile',
        'human.buriedness':20. if trapped else 0.,'human.handover_status':'not_started',
        'human.carrier_id':None,'human.destination_id':None,'human.location_context':'outside',
        'human.support_received':[],'human.alive':True,'human.health_value':.5,'attr.mass_kg':75.})


def station(env,point=(24,40),access=None):
    fields={'entity.kind':'station','entity.pose':list(point),'station.kind':'medical',
        'station.capacity':2,'station.occupied':0,'station.open_state':'open',
        'station.accepted_need_types':['water','extrication','medical_supplies','transport'],
        'attr.receiver_confirmation':True}
    if access:fields['station.access_region']={'type':'point','coords':[list(access)]}
    initial_fields(env,'api-receiver',fields)


def inventory(env):
    world=env.sim.world
    items=sum(world.get(e['id'],'item.mass_kg',0)*world.get(e['id'],'item.quantity',0)
              for e in world.entities_of('item') if world.get(e['id'],'item.holder_id')==ACTOR
              and world.get(e['id'],'item.state') not in {'lost','consumed'})
    humans=sum(world.get(e['id'],'attr.mass_kg',75.) for e in world.entities_of('human')
               if world.get(e['id'],'human.carrier_id')==ACTOR)
    observed=world.get(ACTOR,'platform.payload_used_kg',0.)
    return {'item_mass_kg':items,'person_mass_kg':humans,'expected_total_kg':items+humans,
            'observed_total_kg':observed,'consistent':abs(observed-items-humans)<1e-8}


def facts(env,ids):
    return {eid:deepcopy(env.sim.world.entities[eid]['fields']) for eid in ids if eid in env.sim.world.entities}


def run_action(env,aid,params,ids):
    sim=env.sim
    before=facts(env,ids); before_mass=inventory(env); start=len(sim.events)
    iid=sim.submit_action(ACTOR,aid,deepcopy(params));rec=sim.instances[iid]
    deadline=sim.t+float(rec.get('plan',{}).get('duration_s',0))+4*sim.config['tick_s']
    while rec['status'] in {'requested','accepted','running'} and sim.t<deadline:
        sim.step(min(1.,deadline-sim.t))
    if rec['status']=='rejected' and rec.get('result') is None:sim.step()
    result=rec.get('result')
    return {'action':aid,'id':iid,'actor':ACTOR,'input':params,'status':rec['status'],
            'required_capabilities':list(MODELS[aid].capabilities),
            'result':result.to_dict() if result else None,
            'before':before,'after':facts(env,ids),'mass_before':before_mass,'mass_after':inventory(env),
            'events':[e.to_dict() for e in sim.events[start:]],'t':sim.t}


def report(env,config,steps):
    return {'config':config,'actor_spec':dict(env.sim.agents[ACTOR].spec),
            'initialization':'authored_api_fixture fields written only at t=0 before first native action; no runtime truth mutation',
            'steps':steps,'final_truth':env.sim.world.to_dict(),
            'contract_violations':[e.to_dict() for e in env.sim.events if e.contract_violation]}


def cargo_person_case(loaded_first=False):
    env,config=setup(['C21','C22','C24','C25','C02','C46'])
    person(env);station(env)
    initial_fields(env,'api-kit',{'entity.kind':'item','entity.pose':[20,40],'item.kind':'medical_pack',
        'item.mass_kg':2.,'item.quantity':2,'item.state':'stock','item.condition':'usable',
        'item.release_state':'locked','item.holder_id':'station-1','attr.drop_error_m':0.,'attr.release_jam_rate':0.})
    ids=[ACTOR,PERSON,'api-kit','station-1','api-receiver'];steps=[]
    actions=[('A024',{'target':'api-kit'}),('A019',{'target':PERSON,'medical_item_id':'api-kit'}),
            ('A020',{'target':PERSON}),
            ('A021',{'target':PERSON,'goal':[24,40],'path':[[20,40],[24,40]],'destination_id':'api-receiver'}),
            ('A019',{'target':PERSON,'medical_item_id':'api-kit'}),('A025',{'target':'api-kit','drop_point':[24,40]}),
            ('A022',{'target':PERSON,'station_id':'api-receiver'})]
    if loaded_first:actions.insert(0,actions.pop(2))
    for aid,params in actions:
        steps.append(run_action(env,aid,params,ids))
    return report(env,config,steps)


def debris(env):
    initial_fields(env,'api-debris',{'entity.kind':'obstacle','entity.pose':[32,40],
        'obstacle.kind':'debris','obstacle.material':'mixed','obstacle.mass_kg':5.,
        'obstacle.volume_m3':1.,'obstacle.height_m':.5,'obstacle.state':'present',
        'obstacle.stability':'stable','obstacle.work_remaining':6.,'attr.initial_work':6.,
        'attr.blocked_width_m':7.,'obstacle.blocks_ids':['road-1']})
    initial_fields(env,'road-1',{'road.obstacle_ids':['api-debris']})


def ground_assessment_case():
    env,config=setup(['C02','C05','C12','C35','C04','C40','C44','C39','C16'],kind='ground_vehicle')
    debris(env);ids=[ACTOR,'api-debris','road-1','building-1'];steps=[]
    capture=run_action(env,'A005',{'target':'building-1','modality':'rgb','range_m':80},ids);steps.append(capture)
    obs=capture['result']['observations']
    inference=run_action(env,'A006',{'observations':obs},ids);steps.append(inference)
    localization=run_action(env,'A007',{'target':'building-1','observations':inference['result']['observations']},ids);steps.append(localization)
    steps.append(run_action(env,'A041',{'observations':inference['result']['observations']+localization['result']['observations']},ids))
    clear=run_action(env,'A012',{'target':'api-debris','duration_s':8,'disposal_location':[32,46]},ids);steps.append(clear)
    steps.append(run_action(env,'A041',{'observations':clear['result']['observations']},ids))
    steps.append(run_action(env,'A002',{'goal':[40,40],'path':[[20,40],[40,40]]},ids))
    result=report(env,config,steps)
    result['local_belief']=env.sim.agents[ACTOR].belief.to_view();result['objectives']=env.objectives()
    return result


def clearing_case(aid,cap):
    env,config=setup([cap]);debris(env)
    step=run_action(env,aid,{'target':'api-debris','duration_s':8,'disposal_location':[32,46]},[ACTOR,'api-debris','road-1'])
    return report(env,config,[step])


def extrication_case():
    env,config=setup(['C18','C20','C46','C22','C02']);person(env,trapped=True);station(env)
    ids=[ACTOR,PERSON,'api-receiver'];steps=[]
    for aid,params in [('A017',{'target':PERSON}),('A020',{'target':PERSON}),
            ('A021',{'target':PERSON,'goal':[24,40],'path':[[20,40],[24,40]],'destination_id':'api-receiver'}),
            ('A022',{'target':PERSON,'station_id':'api-receiver'})]:steps.append(run_action(env,aid,params,ids))
    return report(env,config,steps)


def water_case():
    env,config=setup(['C03','C22','C23','C46'],kind='boat',start=(12,20))
    person(env,(12,20));initial_fields(env,PERSON,{'human.location_context':'water','attr.flow_speed_mps':.2})
    initial_fields(env,'api-water',{'entity.kind':'water','entity.footprint':{'type':'polygon','coords':[[4,10],[30,10],[30,30],[4,30],[4,10]]}})
    station(env,(20,32),(20,28));ids=[ACTOR,PERSON,'api-receiver'];steps=[]
    for aid,params in [('A023',{'target':PERSON}),
            ('A021',{'target':PERSON,'goal':[20,28],'path':[[12,20],[20,28]],'destination_id':'api-receiver'}),
            ('A022',{'target':PERSON,'station_id':'api-receiver'})]:steps.append(run_action(env,aid,params,ids))
    return report(env,config,steps)


def run_all():
    value={'mode':'isolated_native_api_fixture','no_scene_configs_modified':True,'camera_enabled':False,'cases':{}}
    cases={'cargo_person':cargo_person_case,'ground_assessment':ground_assessment_case,
           'heavy_clearing':lambda:clearing_case('A013','C17'),'lifting':lambda:clearing_case('A016','C20'),
           'extrication':extrication_case,'water_receiver':water_case}
    for name,run in cases.items():
        try:
            value['cases'][name]=run()
        except Exception as exc:
            import traceback
            value['cases'][name]={'run_status':'fixture_error','error':str(exc),'traceback':traceback.format_exc()[-3000:],'steps':[]}
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    value=run_all();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    print(json.dumps({name:{'actions':[[s['action'],s['status']] for s in case['steps']],
                           'mass_mismatches':[(s['action'],s['mass_after']) for s in case['steps'] if not s['mass_after']['consistent']]}
                      for name,case in value['cases'].items()},ensure_ascii=False))

if __name__=='__main__':main()
