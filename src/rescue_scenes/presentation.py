"""Shared AAS registry and presentation, without importing the AAS runtime."""
import json
from copy import deepcopy
from pathlib import Path

BASE='rescue:Entity'
AGENT='rescue:Agent'
STATUS='rescue:Status'
EVENT='rescue.run.event'
KIND='rescue.kind'
POSITION='rescue.position_enu_m'
ORIENTATION='rescue.orientation_enu'
YAW='rescue.yaw_deg'
STATE='rescue.state_json'
BELIEF='rescue.agent_json'
# Readable scalar facts for the Inspector's operating summary (status/task/battery/position).
STATUS_F='rescue.status'
TASK_F='rescue.current_task'
BATTERY='rescue.battery_wh'
FIELDS=(KIND,POSITION,ORIENTATION,YAW,STATE,STATUS_F,TASK_F,BATTERY,BELIEF)
BUSINESS='scene_rescue.business'  # Events-tab narration; NOT under 'rescue.' (that prefix is the AAS rescue KPI bus, which requires data_json)
KPI_EVENT='rescue.event'     # existing AAS rescue KPI vocabulary (noticed / outcome)
TASK_EVENTS=('lowalt.task.created','lowalt.task.assigned','lowalt.task.completed','lowalt.task.failed')  # shared AAS task KPI population
HANDOFF='lowalt.handoff.transferred'
TYPES={'robot_dog':'rescue:Dog','uav':'rescue:UAV','ground_vehicle':'rescue:Vehicle',
       'specialist_team':'rescue:Team','coordinator':'rescue:Coordinator','compute_node':'rescue:Compute',
       'human':'rescue:Person','obstacle':'rescue:Debris','building':'rescue:Building',
       'station':'rescue:Station','road':'rescue:Road','open_area':'rescue:Region','cue':'rescue:Cue',
       'item':'rescue:Item','hazard':'rescue:Hazard'}
TYPES.update({f'building_{grade}':f'rescue:Building{grade}' for grade in ('D0','D1','D2','D3')})
TYPES.update(obstacle_partly_cleared='rescue:DebrisOpening',obstacle_cleared='rescue:DebrisCleared')
TYPES.update(clearance_loader='rescue:Loader')
TYPES.update(human_prone='rescue:PersonProne',human_supine='rescue:PersonSupine')  # produced static pose variants (P2)
TYPES.update(ambulance='rescue:Ambulance',ambulance_loaded='rescue:AmbulanceLoaded')  # RSC-008-C01 (P5) closed / loaded  # display kind of a ground_vehicle agent (config display.agent_models)
TYPES.update(shelter='rescue:Shelter')  # BLD-028-G02-D0 settlement unit as a compact staffed receiving point (per-entity display override)
TYPES.update(boat='rescue:Boat',boat_loaded='rescue:BoatLoaded',human_in_water='rescue:PersonInWater')  # NEW-BOAT-001-C01 (P3), NEW-HUM-W01 (P4)
TYPES.update(facility='rescue:Facility',fire_unit='rescue:FireUnit',fire_unit_deployed='rescue:FireUnitDeployed',
             fire_pump='rescue:FirePump',rope_rig='rescue:RopeRig',rope_rig_rigged='rescue:RopeRigRigged',
             human_snow_casualty='rescue:SnowCasualty',sled='rescue:Sled',sled_loaded='rescue:SledLoaded',
             tank='rescue:Tank',tank_leaking='rescue:TankLeaking',bund='rescue:Bund',bund_leaking='rescue:BundLeaking')
LOADED_VARIANT={'ambulance':'ambulance_loaded','boat':'boat_loaded','sled':'sled_loaded'}  # carrier display kind -> variant shown while it truly carries a person
AGENT_KINDS={'robot_dog','uav','ground_vehicle','specialist_team','coordinator','compute_node','clearance_loader','ambulance','boat'}
AGENT_KINDS.update({'fire_unit','sled'})
COLORS={'robot_dog':'#ffb84d','uav':'#49c9ff','ground_vehicle':'#69c873','specialist_team':'#f2e74c',
        'human':'#f06c72','obstacle':'#ad7654','building':'#93817c','station':'#55e8bc','coordinator':'#e4aaff','clearance_loader':'#f0c020','ambulance':'#ffffff','boat':'#ff7a3d'}


def kind_of(entity):
    return entity['fields']['platform.kind'] if entity['cls']=='platform' else entity['cls']


def type_of(entity,agent_models=None):
    # Stable per entity: AAS forbids lifecycle changes in physical (lockstep) calls, so
    # state-dependent visuals are separate predeclared display entities (display_extras).
    kind=(agent_models or {}).get(entity['id']) or kind_of(entity)
    if kind=='building' and 'attr.damage_grade' in entity['fields']:
        kind += '_'+entity['fields']['attr.damage_grade']
    return TYPES.get(kind,BASE)


HUMAN_STATES=('mobile','limited','immobile','trapped','contacted_mobile','contacted_limited','contacted_immobile',
              'contacted_trapped','extricated','loaded','transporting','transported_arrived','handover_ack')
OBSTACLE_STATES=('present','partly_cleared','cleared','moved')
HIDDEN=object()  # sentinel: position absent (engine retracts it), never a coordinate


def label_states(entity):
    f=entity['fields']; cls=entity['cls']
    if cls=='human': return HUMAN_STATES
    if cls=='obstacle': return OBSTACLE_STATES
    if cls=='station': return tuple(f"{f['station.kind']}_{s}" for s in ('open','restricted','closed'))
    if cls=='building':
        grade=f.get('attr.damage_grade')
        return tuple((f'{grade}_{v}' if grade else v) for v in ('none','minor','major','destroyed'))
    return ()


def human_pose(entity,human_models=None):
    """Display pose from world truth: in a configured location context (e.g. water) and not carried -> that
    context's model; trapped -> prone; immobile or carried -> supine; else standing base."""
    f=entity['fields']
    if f.get('human.location_context') in (human_models or {}) and not f.get('human.carrier_id'):
        return f['human.location_context']
    if f.get('human.mobility')=='trapped': return 'prone'
    if f.get('human.carrier_id') or f.get('human.mobility')=='immobile': return 'supine'
    return None


def state_model(entity,state_models=None):
    """One display kind selected by an explicit world field; no presentation flags."""
    rule=(state_models or {}).get(entity['id'])
    if rule is None:
        return None
    value=entity['fields'][rule['field']]
    if rule.get('positive'):
        value=value>0
    key=str(value).lower() if isinstance(value,bool) else str(value)
    return rule['values'].get(key)


def display_extras(entity,agent_models=None,human_models=None,state_models=None):
    """(id, type, role) of predeclared display entities for one world entity."""
    out=[]
    carrier=(agent_models or {}).get(entity['id'])
    rule=(state_models or {}).get(entity['id'])
    if rule:
        out.extend((entity['id']+'__state_'+kind,TYPES[kind],('state_model',kind))
                   for kind in sorted(set(rule['values'].values())) if kind!=carrier)
    if carrier in LOADED_VARIANT:
        out.append((entity['id']+'__model_loaded',TYPES[LOADED_VARIANT[carrier]],('model','loaded')))
    if entity['cls']=='human':
        out.extend((f"{entity['id']}__model_{context}",TYPES[kind],('model',context)) for context,kind in sorted((human_models or {}).items()))
        if carrier!='human_snow_casualty':
            out.append((entity['id']+'__model_prone',TYPES['human_prone'],('model','prone')))
            out.append((entity['id']+'__model_supine',TYPES['human_supine'],('model','supine')))
    if entity['cls']=='obstacle':
        out.append((entity['id']+'__model_opening',TYPES['obstacle_partly_cleared'],('model','partly_cleared')))
        out.append((entity['id']+'__model_cleared',TYPES['obstacle_cleared'],('model','cleared')))
    out.extend((f"{entity['id']}__{state}",STATUS,('label',state)) for state in label_states(entity))
    return out


def registry_snapshot():
    number={'type':'number'}; string={'type':'string'}
    schema={KIND:string,POSITION:{'type':'vector','length':3,'items':number},
            ORIENTATION:{'type':'vector','length':4,'items':number},YAW:number,STATE:string,
            STATUS_F:string,TASK_F:string,BATTERY:number,BELIEF:string}
    types=[{'id':BASE,'parents':[],'abstract':False},{'id':AGENT,'parents':[BASE],'abstract':False},
           {'id':STATUS,'parents':[BASE],'abstract':False}]
    types.extend({'id':typ,'parents':[AGENT if kind in AGENT_KINDS else BASE],'abstract':False} for kind,typ in TYPES.items())
    fields=[{'id':key,'declaring_type':AGENT if key==BELIEF else BASE,'schema':deepcopy(value),
             'metadata':{'role':'observation','clock':'canonical',**({'frame':'enu','unit':'m'} if key==POSITION else
                         {'unit':'Wh'} if key==BATTERY else {})}}
            for key,value in schema.items()]
    def event(mid,members,required=None):
        return {'id':mid,'kind':'event','schema':{'type':'record','members':members,'required':list(required or members),'extra':False},
                'result_schema':None,'feedback_schema':None,'cancel_support':False}
    vector={'type':'vector','length':3,'items':number}
    messages=[event(EVENT,{'data_json':string}),event(KPI_EVENT,{'data_json':string}),
              event(BUSINESS,{'text':string,'stage':string,'subject':string,'actor':string,'position':vector},['text','stage','subject','actor']),
              # Field names follow the installed lowalt task narratives/KPI reducer exactly.
              event(TASK_EVENTS[0],{'task':string,'kind':string}),
              event(TASK_EVENTS[1],{'task':string,'vehicle':string,'algorithm':string}),
              event(TASK_EVENTS[2],{'task':string,'vehicle':string,'latency_s':number}),
              event(TASK_EVENTS[3],{'task':string,'vehicle':string,'reason':string}),
              event(HANDOFF,{'handoff':string,'person':string,'receiver':string})]
    effective={row['id']:list(FIELDS if row['id']==AGENT or row['parents']==[AGENT] else FIELDS[:-1]) for row in types}
    return {'format':'aeroagentsim.aerograph-registry/v1','compiler_version':'1',
            'registry':{'revision':'rescue-scenes/v1','types':types,'fields':fields,'messages':messages,'schemas':{}},
            'details':{'effective_fields':effective,'relations':[],'normalizations':[],'exclusions':[],'producer_hints':{},
                       'statistics':{'types':len(types),'fields':len(fields),'messages':len(messages)},
                       'provenance':{'source':'v0.5 catalog fields projected through AAS plugin API'}}}


def ontology_concepts(types):
    """AeroGraph concept rows for the catalog projection (types only, no invented fields)."""
    rows=[]
    for row in types:
        parents=list(row['parents'])
        entry={'id':row['id'],'abstract':bool(row['abstract']),'ownFieldIds':[]}
        if len(parents)==1: entry['parent']=parents[0]
        elif parents: entry['parents']=parents
        else: entry['parent']=None
        rows.append(entry)
    return rows


def write_ontology(root, types):
    """Persist the v0.5 catalog projection as an AeroGraph source directory.

    The compiler records per-type file/pointer provenance, so the scenario can
    reference a real compiled hierarchy instead of a hand-written stub.
    """
    root=Path(root); (root/'entity-directory/data').mkdir(parents=True,exist_ok=True)
    (root/'semantic-directory/data').mkdir(parents=True,exist_ok=True)
    (root/'entity-directory/data/concepts.json').write_text(json.dumps({
        'schemaVersion':'aerorescue-catalog/1',
        'generatedFrom':'src/rescue_scenes/presentation.py registry_snapshot() type hierarchy',
        'concepts':ontology_concepts(types)},ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'semantic-directory/data/definitions.json').write_text('[]',encoding='utf-8')
    (root/'semantic-directory/data/relations.json').write_text('[]',encoding='utf-8')
    (root/'semantic-directory/data/schema-definitions.json').write_text('{"$defs":{}}',encoding='utf-8')
    return root


def scenario_registry(root, snapshot):
    """Registry spec: AeroGraph-compiled types plus explicit field/message overlays."""
    registry=snapshot['registry']
    static=[row for row in registry['types'] if not row['id'].startswith('rescue:Upstream_')]
    local=[{'id':row['id'],'parents':list(row['parents']),'abstract':bool(row['abstract'])}
           for row in registry['types'] if row['id'].startswith('rescue:Upstream_')]
    write_ontology(root,static)
    spec={'compile':{'root':str(Path(root).resolve()),'types':[row['id'] for row in static]}}
    if local: spec['types']=local
    spec['fields']=[{'id':row['id'],'type':row['declaring_type'],'schema':deepcopy(row['schema']),
                     'metadata':deepcopy(row['metadata'])} for row in registry['fields']]
    spec['messages']=deepcopy(registry['messages'])
    return spec


def bindings(models):
    result=[]
    for kind,typ in TYPES.items():
        if kind in {'road','open_area','cue','item','hazard'}:
            continue  # Background geometry already exists; these facts remain inspectable.
        visual={'kind':'marker','color':COLORS.get(kind,'#999999'),'scale':1}
        if kind in models:
            visual.update(kind='model',asset=models[kind]['url'],scale=models[kind].get('scale',1))
        if kind=='obstacle_cleared':
            visual={'kind':'label','color':'#999999','scale':0.5}
        result.append({'typeId':typ,'positionField':POSITION,'orientationField':ORIENTATION,'frame':'enu','visual':visual})
    result.append({'typeId':STATUS,'positionField':POSITION,'frame':'enu','visual':{'kind':'label','color':'#ffffff','scale':1}})
    return result


STAGE_SEVERITY={'blocked':'warning','task_failed':'warning','found':'notice','handover_ack':'notice','cleared':'notice',
                'road_open':'notice','passed':'notice'}


def event_narratives():
    """Events-tab narration for rescue.business (text is composed by the engine from committed evidence)."""
    return {BUSINESS:{'template':'{text}','template_zh':'{text}','category':'emergency','severity':'info',
                      'entities':['subject','actor'],'position':'position'}}
