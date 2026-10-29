"""AAS-owned lockstep wrapper around the vendored v0.5 runtime.

Truth poses/state are replay-only. Agent JSON contains their local beliefs and
own task records. Opt-in camera evidence comes from the kernel capture owner.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path

from aerokernel import Activate, Dependency, EntityRef, Partition
from aerokernel.engine import Timing
from aerokernel.sdk import ContextEngine
from aeroagentsim.engines.common import policies
from .backgrounds import Grid, digest
from .core.api import Environment
import re
from .evidence import ChainTracker, chain_timeline
from .presentation import (LOADED_VARIANT, BASE, AGENT, STATUS, EVENT, KIND, POSITION, ORIENTATION, YAW, STATE, BELIEF, FIELDS, HIDDEN, type_of, display_extras, human_pose, state_model,
                           STATUS_F, TASK_F, BATTERY, BUSINESS, KPI_EVENT, TASK_EVENTS, HANDOFF)


def encoded(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':'))


def position_xy(entity):
    pose=entity['fields'].get('entity.pose',entity['fields'].get('human.pose'))
    if pose is not None:
        return pose[:2]
    points=entity['geometry']['coords']
    if len(points)>1 and points[0]==points[-1]:
        points=points[:-1]
    return [sum(p[k] for p in points)/len(points) for k in (0,1)]


def state_label(entity):
    f=entity['fields']; cls=entity['cls']
    if cls=='human':
        if f.get('human.handover_status')=='acknowledged': return 'handover_ack'
        if f.get('human.handover_status')=='arrived': return 'transported_arrived'
        if f.get('human.handover_status')=='in_transit': return 'transporting'
        if f.get('human.carrier_id'): return 'loaded'
        support=f.get('human.support_received',[])
        if 'extricated' in support: return 'extricated'
        if 'contacted' in support: return 'contacted_'+str(f['human.mobility'])
        return str(f['human.mobility'])
    if cls=='obstacle': return f['obstacle.state']
    if cls=='building': return f"{f['attr.damage_grade']}_{f['building.visual_damage']}" if 'attr.damage_grade' in f else f['building.visual_damage']
    if cls=='road': return f['road.entry_state']
    if cls=='station': return f"{f['station.kind']}_{f['station.open_state']}"
    if cls=='platform': return f['platform.motion_state']
    return cls


def quat_product(a,b):
    ax,ay,az,aw=a; bx,by,bz,bw=b
    return [aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,
            aw*bz+ax*by-ay*bx+az*bw,aw*bw-ax*bx-ay*by-az*bz]


def rotate_vector(q,v):
    return quat_product(quat_product(q,[*v,0]),[-q[0],-q[1],-q[2],q[3]])[:3]


def snow_height(grid,features,e,n):
    """Use exact source cover controls on the two camps and their visible tread."""
    from .layout import inside, project
    for region in features['reservations']:
        if inside([e,n],region['polygon_en_m']):
            return grid.height(e,n)+region['snow_depth_vertical_m']
    route=features['visible_approach']; nodes=route['centerline_nodes']
    for i,(a,b) in enumerate(zip(nodes,nodes[1:])):
        point,t=project([e,n],a['top_enu_m'],b['top_enu_m'])
        width=next(row['visible_tread'] for row in route['width_schedule_m']
                   if int(row['from'][1:])<=i<int(row['through'][1:]))
        if math.dist([e,n],point)<=width/2+1e-6:
            return a['top_enu_m'][2]+t*(b['top_enu_m'][2]-a['top_enu_m'][2])
    return grid.height(e,n)


class RescueWorld(ContextEngine):
    version='rescue-scenes/v0.5-scripted-1'

    def __init__(self, context):
        self.build_context=context
        self.bundle=json.loads(Path(context.config['bundle_path']).read_text())
        self.config=self.bundle['config']
        self.grid=Grid(self.config['background']['grid'])
        if digest(self.config)!=self.bundle['config_digest']:
            raise ValueError('Scene config digest differs from materialized bundle')
        core=deepcopy(self.bundle['core'])  # semantics path is relative to the bundle (repeatable bundle bytes)
        core['semantics']['path']=str(Path(context.config['bundle_path']).parent/core['semantics']['path'])
        self.env=Environment.from_config(core)
        self.sim=self.env.sim
        self.rules=None
        if context.config.get('execution_mode')=='forced_rules':
            from .forced_rules import ForcedRuleDriver
            self.rules=ForcedRuleDriver(self.env,self.config)
        self.refs={ref.id:ref for ref in context.entities}
        self.agent_models=self.config['display'].get('agent_models',{})
        self.human_models=self.config['display'].get('human_models',{})
        self.state_models=self.config['display'].get('state_models',{})
        self.snow_features=self.bundle['audit'].get('source_features') if self.config['display'].get('source_snow_contact') else None
        # Floating models (boats, person in water) are drawn at the scene's water stage, not the submerged terrain.
        water=self.bundle['audit'].get('layout_network',{}).get('water')
        self.water_u=water['datum_U_m'] if water and water['id']==self.config['display'].get('water_surface_stage_id') else None
        expected=set(self.sim.world.entities)|{x[0] for e in self.sim.world.entities.values() for x in display_extras(e,self.agent_models,self.human_models,self.state_models)}
        if set(self.refs)!=expected:
            raise ValueError('Bootstrap entities differ from seeded world + display extras; rematerialize scenario')
        self.unpublished=set()
        self.initial_yaw={eid:math.radians(row['yaw_deg']) for eid,row in self.bundle['audit']['layout'].items()}
        if context.run_directory is None:
            raise ValueError('AAS run_directory is required for rescue artifacts')
        self.output=Path(context.run_directory)
        self.output.mkdir(parents=True,exist_ok=True)
        camera=self.config['display'].get('camera_observation')
        self.camera=None
        if camera and camera.get('enabled'):
            from .camera import CameraObservations
            self.camera=CameraObservations(self,camera,context.config['camera_observation'])
        self.event_file=(self.output/'events.jsonl').open('w',encoding='utf-8')
        self.cursor=0; self.created=set(); self.labels={}; self.label_generation=defaultdict(int)
        self.previous={}; self.yaw={}; self.last_values={}; self.closed=False
        self._published_static={}; self._encoded_states={}; self._display_extras={}; self._terrain_positions={}
        self._encoded_camera_records={}
        self.initial=self.sim.world.to_dict()
        (self.output/'initial_truth.json').write_text(encoded(self.initial))
        self.tracker=ChainTracker(self.initial); self.chain_t={}
        self.task_created={}; self.task_label={}; self.task_assigned=set(); self.task_done=set(); self.task_failed=set()
        (self.output/'scene-config.json').write_text(encoded(self.config))
        (self.output/'placement.json').write_text(encoded(self.bundle['audit']))
        source=Path(__file__).with_name('SOURCES.json')
        (self.output/'SOURCES.json').write_text(source.read_text())
        tick_ns=round(self.config['tick_s']*1e9)
        super().__init__(Partition(context.id,context.id,produces=FIELDS,
            consumes=(Dependency(POSITION),Dependency(YAW)) if self.camera else (),
            emits=(EVENT,KPI_EVENT,BUSINESS,*TASK_EVENTS,HANDOFF,*((self.camera.route['request_schema'],) if self.camera else ())),
            message_targets=('rescue.events',*((self.camera.route['capture_target'],) if self.camera else ())),lifecycle=True,lifecycle_reads=(BASE,AGENT,STATUS),
            timing=Timing('lockstep',step_ns=tick_ns,certified_hold=True)),policies=policies(FIELDS))

    def bootstrap(self,ctx):
        if ctx.now.ns!=0: raise ValueError('Rescue simulation requires AAS epoch at zero')
        self._publish(ctx); self._record(ctx)

    def step(self,ctx):
        target=ctx.now.ns/1e9
        if target<self.sim.t: raise ValueError('AAS clock moved backwards')
        # No independent timer and no stepping past the AAS frontier.
        if self.rules:
            self.rules.advance(target)
        else:
            self.sim.run(target)
        if self.camera:
            self.camera.periodic(ctx.now.ns)
        self._publish(ctx); self._record(ctx)

    def on_inputs(self,ctx):
        if self.camera:
            self.camera.receive(ctx)
            # Activation follows the pose publication's commit. The request's
            # source cut therefore includes this tick's camera-carrying actor.
            self.camera.request(ctx)
            # Receipt/request processing changes local task/belief overviews,
            # not physical truth. Poses/state were published by step already.
            self._publish(ctx,agents_only=True); self._record(ctx)

    def _set(self,ctx,ref,values):
        for key,value in values.items():
            token=(ref.id,ref.generation,key)
            if self.last_values.get(token)!=value:
                if value is HIDDEN:
                    # Not displayed in this state: explicit absence (no fact at bootstrap, retraction later).
                    # A parked far-away position would be linearly interpolated by the replay store.
                    if token in self.last_values:
                        ctx.retract(ref,key,ctx.policies[key].valid(ctx.now),'display variant inactive in this state')
                else:
                    ctx.set(ref,key,value)
                # Display transforms are freshly built flat numeric lists;
                # detach their container without recursively copying scalars.
                self.last_values[token]=(value if value is HIDDEN or type(value) in (str,int,float,bool,type(None))
                                         else value.copy() if key in (POSITION,ORIENTATION) else deepcopy(value))

    def _agent_values(self,eid,agent,fields):
        values={TASK_F:self._task_text(agent)}
        battery=fields.get('platform.battery_remaining_Wh')
        if isinstance(battery,(int,float)): values[BATTERY]=round(float(battery),3)
        # Bounded local overview. Complete belief/task copies are archived at close.
        overview={'id':eid,'kind':agent.kind,'current':agent.current,'queue':list(agent.queue),
            'capabilities':agent.spec['capabilities'],'belief_entries':len(agent.belief.entries),
            'current_task':agent.tasks.get(agent.current),
            'sensing':self.camera.sensing(eid) if self.camera else 'scripted_v0.5'}
        raw=encoded(overview)
        if self.camera:
            records=agent.records.get('camera_image',[])
            latest=records[-1] if records else None
            cached=self._encoded_camera_records.get(eid)
            # on_camera_observations replaces this detached overview snapshot;
            # capture/perception workers mutate their own report rows, not it.
            if cached is None or cached[0] is not latest:
                cached=(latest,encoded(records[-1:]))
                self._encoded_camera_records[eid]=cached
            # camera_image is the last key in the original compact JSON object.
            raw=raw[:-1]+',"camera_image":'+cached[1]+'}'
        values[BELIEF]=raw
        return values

    def _publish(self,ctx,labels=True,*,agents_only=False):
        # All refs (world entities + display extras) are predeclared and created at
        # bootstrap; later ticks only move/update them (no lifecycle in physical calls).
        display=self.config['display']
        world=self.sim.world
        if agents_only:
            # Preserve the same entity and field operation order; every omitted
            # display value is unchanged during camera receipt/request handling.
            for eid,agent in sorted(self.sim.agents.items()):
                ref=self.refs.get(eid)
                entity=world.entities.get(eid)
                if ref is not None and entity is not None:
                    self._set(ctx,ref,self._agent_values(eid,agent,entity['fields']))
            return
        # A carrier's loaded variant depends on other entities, not its own
        # field revision. Read the actual carried-person relation each publish.
        loaded={h['fields']['human.carrier_id'] for h in world.entities_of('human')
                if h['fields'].get('human.carrier_id')}
        for eid,entity in sorted(world.entities.items()):
            ref=self.refs.get(eid)
            if ref is None:
                if eid not in self.unpublished:
                    self.unpublished.add(eid)  # runtime-created world object: kept in events.jsonl, not rendered
                continue
            agent=self.sim.agents.get(eid); f=entity['fields']
            revision=(id(entity),world._field_revisions.get(eid,0),ref.generation)
            # Display configuration and imported transforms are fixed for a run.
            # Non-agent humans also stay unchanged unless their carrier moves.
            # Agent task/queue/camera state still projects on every invocation.
            carrier=(world.entities[f['human.carrier_id']]
                     if entity['cls']=='human' and f.get('human.carrier_id') else None)
            carrier_revision=(id(carrier),world._field_revisions.get(f['human.carrier_id'],0)) if carrier is not None else None
            display_revision=(revision,carrier_revision,eid in loaded)
            static=not agent and entity['cls']!='platform'
            if static and self._published_static.get(eid)==display_revision:
                continue
            xy=position_xy(carrier if carrier is not None else entity)
            agl=(display['uav_agl_m'] if agent and agent.kind=='uav' else
                 display['agent_ground_offset_m'] if agent else display['person_offset_m'] if entity['cls']=='human' else 0)
            # Terrain/display contact is fixed for the run. Cache only the last
            # base transform per entity, never task/telemetry/carrier projection.
            # Exact numeric spelling keeps signed zero and int/float distinctions.
            terrain_key=tuple((type(v),v.hex() if isinstance(v,float) else v) for v in (*xy,agl))
            terrain=self._terrain_positions.get(eid)
            if terrain is None or terrain[0]!=terrain_key:
                enu=self.grid.enu(xy,agl)
                if self.snow_features is not None:
                    enu[2]=snow_height(self.grid,self.snow_features,*enu[:2])+agl
                terrain=(terrain_key,tuple(enu))
                self._terrain_positions[eid]=terrain
            enu=list(terrain[1])
            afloat=(agent and agent.spec.get('mobility')=='water') or (entity['cls']=='human' and f.get('human.location_context')=='water' and not f.get('human.carrier_id'))
            if afloat and self.water_u is not None:
                enu=[enu[0],enu[1],self.water_u]
            old=self.previous.get(eid)
            if old is not None and math.dist(old[:2],enu[:2])>1e-6:
                self.yaw[eid]=math.atan2(enu[1]-old[1],enu[0]-old[0])
            self.previous[eid]=enu
            yaw=self.yaw.get(eid,self.initial_yaw.get(eid,0.0))
            # Y-up model (normalized to face east) -> ENU quaternion Rz(yaw)*Rx(90deg).
            half=yaw/2; root=math.sqrt(.5)
            quat=[root*math.cos(half),root*math.sin(half),root*math.sin(half),root*math.cos(half)]
            relation=self.bundle['audit']['layout'].get(eid,{}).get('relations',{})
            normal=relation.get('ground_normal_enu')
            if normal:
                nx,ny,nz=normal; scale=math.sqrt(2*(1+nz))
                tilt=[-ny/scale,nx/scale,0,scale/2]
                quat=quat_product(tilt,quat)
                if 'local_offset_m' in relation:
                    frame=quat_product(tilt,[0,0,math.sin(half),math.cos(half)])
                    offset=rotate_vector(frame,relation['local_offset_m'])
                    enu=[enu[i]+offset[i] for i in range(3)]
            state=state_label(entity)
            if eid not in self.created:
                ctx.create(ref); self.created.add(eid)
            shown=enu
            if entity['cls']=='obstacle' and f.get('obstacle.state')!='present':
                shown=HIDDEN
            pose=human_pose(entity,self.human_models) if entity['cls']=='human' else None
            if self.agent_models.get(eid)=='human_snow_casualty' and not f.get('human.carrier_id'):
                pose=None
            if pose and self.agent_models.get(f.get('human.carrier_id')) in LOADED_VARIANT:
                pose='in_carrier'  # drawn by the carrier's LOADED model (ambulance stretcher / boat deck), no second body
            if self.agent_models.get(eid) in LOADED_VARIANT:  # loaded variant while it actually carries a person
                pose='loaded' if eid in loaded else None
            if pose: shown=HIDDEN  # a state variant (display extra) is shown instead
            selected=state_model(entity,self.state_models)
            if selected is not None and selected!=self.agent_models.get(eid):
                shown=HIDDEN
            state_key=(revision,state)
            cached_state=self._encoded_states.get(eid)
            if cached_state is None or cached_state[0]!=state_key:
                cached_state=(state_key,encoded({'truth_for_rendering_only':True,'state':state,'fields':f}))
                self._encoded_states[eid]=cached_state
            values={KIND:agent.kind if agent else entity['cls'],POSITION:shown,ORIENTATION:quat,YAW:math.degrees(yaw),
                    STATE:cached_state[1]}
            values[STATUS_F]=self._status_text(entity,agent,state)
            if agent:
                values.update(self._agent_values(eid,agent,f))
            self._set(ctx,ref,values)
            obstacle_state=f.get('obstacle.state')
            if obstacle_state=='moved': obstacle_state='cleared'
            # Extras depend only on these declaration fields and fixed display
            # rules. Keep both exact JSON spellings; only active/pose may change.
            extras_key=(entity['cls'],f.get('attr.damage_grade'),f.get('station.kind'))
            extras=self._display_extras.get(eid)
            if extras is None or extras[0]!=extras_key:
                rows=[]
                for xid,typ,(role,when) in display_extras(entity,self.agent_models,self.human_models,self.state_models):
                    xref=self.refs.get(xid)
                    if xref is not None:
                        states=tuple(encoded({'entity':eid,'role':role,'shows':when,'active':active})
                                     for active in (False,True))
                        rows.append((xid,xref,role,when,states))
                extras=(extras_key,rows)
                self._display_extras[eid]=extras
            for xid,xref,role,when,states in extras[1]:
                if xid not in self.created:
                    ctx.create(xref); self.created.add(xid)
                model=role in ('model','state_model')
                if model:
                    active=(selected if role=='state_model' else obstacle_state if entity['cls']=='obstacle' else pose)==when
                    pos=enu if active else HIDDEN
                else:
                    active=state==when
                    pos=[enu[0],enu[1],enu[2]+display['status_height_m']] if active else HIDDEN
                self._set(ctx,xref,{KIND:role,POSITION:pos,ORIENTATION:quat if model else [0,0,0,1],
                                    YAW:math.degrees(yaw) if model else 0,
                                    STATE:states[active]})
            if static:
                self._published_static[eid]=display_revision

    def _record(self,ctx):
        for event in self.sim.events[self.cursor:]:
            if self.camera:
                self.camera.action(event)
            event_row=event.to_dict(); row=event_row; raw=encoded(row)
            self.event_file.write(raw+'\n')
            # Large initialization writes/evidence remain lossless in the JSONL.
            # Journal messages name the exact event sequence and digest instead.
            if len(raw.encode())>12288:
                row={'seq':row['seq'],'t':row['t'],'kind':row['kind'],'subject':row['subject'],
                     'artifact':'events.jsonl','sha256':digest(event_row),'payload_external':True}
                raw=encoded(row)
            ctx.emit(EVENT,{'data_json':raw},topic='rescue.events')
            for stage in self.tracker.feed(event_row):
                self._stage(ctx,stage)
            if event.family=='sm:task.status':
                self._task(ctx,event_row)
        self.cursor=len(self.sim.events); self.event_file.flush()
        if self.camera and self.camera.pending:
            ctx.ops.append(Activate(self.partition.id))

    # ---- projection of committed core evidence into the existing AAS Events/Results UI ----
    TASK_STATUS_ZH={'proposed':'生成','assigned':'已分配','accepted':'已接受','completed':'已完成','failed':'失败','canceled':'取消'}
    MEDICAL={'T029','T031','T033','T034','T035'}

    def _status_text(self,entity,agent,state):
        if not agent: return state
        f=entity['fields']; action=f.get('platform.current_action_id')
        return f"{state}" + (f" · {action}" if action else '')

    def _task_text(self,agent):
        task=agent.tasks.get(agent.current) if agent.current else None
        if task is None: return '空闲' if not agent.queue else f'排队 {len(agent.queue)}'
        row=task.to_dict() if hasattr(task,'to_dict') else task
        tid=row.get('type',''); name=self.sim.catalog.tasks.get(tid,{}).get('name','')
        target=row.get('target') or row.get('subject') or ''
        return f"{tid} {name} · {row.get('status','')}" + (f" · {target}" if target else '')

    def _emit_business(self,ctx,stage,subject,actor,text):
        payload={'text':text,'stage':stage,'subject':subject,'actor':actor or subject}
        pos=self.previous.get(subject) or self.previous.get(actor)
        if pos is not None: payload['position']=[float(v) for v in pos]
        ctx.emit(BUSINESS,payload,topic='rescue.events')

    def _stage(self,ctx,row):
        stage,target,actor=row['stage'],row['target'],row.get('actor') or row['target']
        self.chain_t.setdefault((stage,target),row['t'])
        text={'blocked':f"{row.get('vehicle')} 在 {target} 被残骸 {row.get('obstacle')} 阻断：轮式车辆不可通行",
              'partly_cleared':f"{actor} 清障中：{target} 部分清除（仍不足以通车）",
              'cleared':f"{actor} 完成清障：{target} 已清除",
              'road_open':f"道路 {target} 恢复通行",
              'passed':f"{row.get('vehicle')} 通过已清障路段 {target}",
              'found':f"{actor} 搜索发现受困人员 {target}（脚本化感知模型）",
              'contact':f"{actor} 与 {target} 建立联系",
              'extricated':f"{actor} 完成 {target} 解困",
              'loaded':f"{target} 已装载到 {row.get('carrier')}",
              'transporting':f"{actor} 转运 {target} 前往医疗点",
              'transported':f"{target} 已运抵医疗点（尚待接收确认）",
              'handover_ack':f"医疗点确认接收 {target}：移交完成"}.get(stage,f"{stage} {target}")
        subject=row.get('vehicle') if stage in ('blocked','passed') else target
        if subject not in self.refs: subject=target if target in self.refs else actor
        self._emit_business(ctx,stage,subject,actor if actor in self.refs else subject,text)
        if stage=='found':
            ctx.emit(KPI_EVENT,{'data_json':encoded({'kind':'noticed','event':target,'sensing':'scripted_v0.5'})},topic='rescue.events')
        if stage=='handover_ack':
            t0=self.chain_t.get(('found',target))
            contact=self.chain_t.get(('contact',target))
            ctx.emit(KPI_EVENT,{'data_json':encoded({'kind':'outcome','event':target,'success':True,
                'time_to_confirmation_s':None if t0 is None or contact is None else contact-t0,
                'time_to_delivery_s':None if t0 is None else row['t']-t0})},topic='rescue.events')
            # The actual receiving station comes from the arrival/handover itself
            # (human.destination_id), never from station enumeration order.
            receiver=self.sim.world.get(target,'human.destination_id')
            if not isinstance(receiver,str) or self.sim.world.entities.get(receiver,{}).get('cls')!='station':
                receiver=next((e['id'] for e in self.sim.world.entities_of('station')),'receiver')
            ctx.emit(HANDOFF,{'handoff':f'{target}->{receiver}','person':target,'receiver':receiver},topic='rescue.events')

    def _task(self,ctx,row):
        tid=row['subject']; status=row['after']; t=row['t']
        match=re.search(r'T\d{3}',tid or ''); cid=match.group(0) if match else 'T000'
        name=self.sim.catalog.tasks.get(cid,{}).get('name','')
        label=f"{cid} {name} [{tid}]"
        agent=row.get('actor') or ''
        if tid not in self.task_created:
            self.task_created[tid]=t; self.task_label[tid]=label
            ctx.emit(TASK_EVENTS[0],{'task':label,'kind':'medical' if cid in self.MEDICAL else 'incident_response'},topic='rescue.events')
        label=self.task_label[tid]
        if status in ('assigned','accepted') and tid not in self.task_assigned:
            self.task_assigned.add(tid)
            ctx.emit(TASK_EVENTS[1],{'task':label,'vehicle':agent,'algorithm':'v0.5 规则分配'},topic='rescue.events')
        if status=='completed' and tid not in self.task_done:
            self.task_done.add(tid)
            ctx.emit(TASK_EVENTS[2],{'task':label,'vehicle':agent,'latency_s':round(max(0.0,t-self.task_created[tid]),1)},topic='rescue.events')
        if status=='failed' and tid not in self.task_failed and tid not in self.task_done:
            self.task_failed.add(tid)
            ctx.emit(TASK_EVENTS[3],{'task':label,'vehicle':agent,'reason':str(row.get('text_zh',''))[:200]},topic='rescue.events')

    def close(self):
        if self.closed: return
        self.closed=True
        try:
            self.event_file.close()
            events=[e.to_dict() for e in self.sim.events]
            tasks=self.env.tasks(); coverage={}
            for tid in self.sim.catalog.tasks:
                rows=[t for t in tasks if t['type']==tid]
                unique={t['id'] for t in rows}
                executed={t['id'] for t in rows if any(h.get('status')=='executing' for h in t['history'])}
                completed={t['id'] for t in rows if t['status']=='completed' and t['holder']==t.get('accepted_by')}
                coverage[tid]={'generated':len(unique),'executed':len(executed),'completed':len(completed)}
            actions=[e for e in events if e['kind']=='action.result']
            action_results=[{'id':iid,'agent':record['agent_id'],'action_id':record['action_id'],
                             'status':record['status'],
                             'result':record['result'].to_dict() if record.get('result') is not None else None}
                            for iid,record in self.sim.instances.items()]
            failures=Counter()
            for record in action_results:
                r=record['result']
                if r is not None and r['status'] in {'failed','rejected','partial','canceled'}:
                    cause=r['failure_code'] if r['failure_code']!='none' else r['rejection_code']
                    failures[cause if cause!='none' else r['status']]+=1
            handovers=[e for e in events if e['layer']=='world' and e['field']=='human.handover_status' and e['after']=='acknowledged']
            result={'schema':'rescue.run-result/v1','sim_time_s':self.sim.t,'requested_until_s':self.config['until_s'],
                    'run_complete':self.sim.t>=self.config['until_s'],'seed':self.config['seed'],'config_digest':self.bundle['config_digest'],
                    'sensing':({aid:self.camera.sensing(aid) for aid in self.sim.agents}
                               if self.camera else 'scripted_v0.5'),
                    'objectives':self.env.objectives(),'tasks_by_catalog_id':coverage,
                    'tasks':tasks,'allocations':self.env.allocation(),'handovers':handovers,
                    'roads':[{'id':e['id'],'fields':e['fields']} for e in self.sim.world.entities_of('road')],
                    'obstacles':[{'id':e['id'],'fields':e['fields']} for e in self.sim.world.entities_of('obstacle')],
                    'road_events':[e for e in events if e['layer']=='world' and e['field'] in {'road.entry_state','obstacle.state'}],
                    'ledger':self.env.ledger(),'failures_by_cause':dict(failures),
                    'contract_violations':[e for e in events if e['contract_violation']],
                    'assumptions':self.config['assumptions'],'placement':self.bundle['audit'],
                    'chain_timeline':chain_timeline(events,json.loads((self.output/'initial_truth.json').read_text())),
                    'per_agent_actions':{aid:[r for r in action_results if r['agent']==aid] for aid in self.sim.agents}}
            if self.rules:
                report=self.rules.snapshot()
                result['execution_mode']='forced_rules'
                result['forced_rules']=report
                (self.output/'forced-rules.json').write_text(encoded(report))
            if self.camera:
                result['camera_observation']=self.camera.report()
            (self.output/'result.json').write_text(encoded(result))
            (self.output/'final_truth.json').write_text(encoded(self.sim.world.to_dict()))
            (self.output/'agent_beliefs.json').write_text(encoded({aid:a.belief.to_view() for aid,a in self.sim.agents.items()}))
        finally:
            if self.camera:
                self.camera.close()
            super().close()


def build(context):
    return RescueWorld(context)


build.config_schema={'type':'object','properties':{'bundle_path':{'type':'string'},'camera_observation':{'type':'object'},'execution_mode':{'enum':['autonomous','forced_rules']}},'required':['bundle_path'],'additionalProperties':False}
