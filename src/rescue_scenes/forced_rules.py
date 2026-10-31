"""Resumable explicit rules, shared by the verifier and existing AAS plugin."""
import math

from .core.sim.actions import MODELS, RECEIVING_STATION_KINDS


class ForcedRuleDriver:
    def __init__(self, env, config):
        self.env, self.config = env, config
        self.report = {}
        self.done = False
        self.sequence = self._sequence()
        try:
            next(self.sequence)
        except StopIteration:
            self.done = True

    def advance(self, target):
        sim = self.env.sim
        sim.run(target)
        if not self.done:
            try:
                next(self.sequence)
            except StopIteration:
                self.done = True
                self.finished_s = sim.t

    def snapshot(self):
        sim = self.env.sim
        sim.evaluator.update()
        report = self.report
        report.update(sim_time_s=sim.t, original_horizon_s=self.config['until_s'],
                      within_original_horizon=sim.t <= self.config['until_s'],
                      objectives=self.env.objectives(), camera_acceptance=False,
                      navigation_mode='explicit region positioning with recorded native air-motion overrides',
                      sensing_mode='native scripted sensing, not image inference', sequence_complete=self.done)
        covered = [o for o in report['objectives'] if o['kind'] in
                   {'isolate_hazard','suppress_fire','find_person','extricate','aid','handover','recover_from_water'}]
        report['rule_targets_satisfied'] = bool(covered) and all(o['satisfied'] for o in covered)
        report['all_objectives_satisfied'] = bool(report['objectives']) and all(o['satisfied'] for o in report['objectives'])
        report['events'] = [e.to_dict() for e in sim.events if e.kind.startswith('rescue.') or e.field == 'hazard.exclusion_zone']
        report['uncovered_objectives'] = [o for o in report['objectives'] if o not in covered]
        report['resource_gaps'] = [dict(target=r['target'],detail=r['blocked']) for r in report['rules'] if 'blocked' in r]
        report['resource_gaps'] += [dict(action=a['action'],actor=a['actor'],target=a['params'].get('target'),
                                        failure=a['failure'],detail=a['feedback'])
                                   for a in report['actions'] if a['failure'] in {'actuator','insufficient'}]
        report['pending_actions'] = [dict(id=iid,actor=r['agent_id'],action=r['action_id'],status=r['status'])
                                     for iid,r in sim.instances.items() if r['status'] in {'requested','accepted','running'}]
        report['rule_sequence_finished_s'] = getattr(self,'finished_s',None)
        return report

    def _sequence(self):
        env, sim = self.env, self.env.sim
        # The rule driver replaces scheduling, not native physical effects.
        for agent in sim.agents.values():
            agent.on_tick=lambda sim:None
        report=self.report
        report.update({"mode":"forced_rule_verification","capability_overrides":[],"actions":[],"rules":[],"scope":"hazard isolation, personnel rescue chains, building assessment and road access; other objectives remain reported"})

        from rescue_scenes.core.sim.actions import _items, _pose, SCRIPTED_SENSORS

        def point(eid):
            return list(_pose(sim,eid))

        def distance(a,b):
            return math.dist(list(a)[:2], list(b)[:2])

        state={'status':None,'observations':[],'result':None}

        def action(agent, aid, params, integrate=True):
            """Submit one native action and wait for its terminal status.

            ``integrate=False`` suppresses the automatic A041 follow-up so a
            caller can run the documented A005 -> A006 -> A007 -> A041 chain
            once instead of integrating after every intermediate step.
            """
            original_mobility=agent.spec.get("mobility")
            if aid in {"A001","A021"}:
                if original_mobility!="air":
                    report["capability_overrides"].append({"agent":agent.id,"action":aid,
                        "field":"mobility","before":original_mobility,"during_rule_action":"air"})
                agent.spec["mobility"]="air"
                params=dict(params,path=[point(agent.platform_eid),params["goal"]])
            required=set(MODELS[aid].capabilities)
            if aid=="A021":
                required.add({"air":"C01","water":"C03"}.get(agent.spec.get("mobility"),"C02"))
            missing=required-set(agent.spec.get("capabilities",[]))
            if missing:
                report["capability_overrides"].append({"agent":agent.id,"action":aid,"added":sorted(missing)})
                agent.spec["capabilities"]=sorted(set(agent.spec.get("capabilities",[]))|missing)
            iid=sim.submit_action(agent.id,aid,params)
            rec=sim.instances[iid]
            while rec["status"] in ("requested","accepted","running"):
                yield iid
            agent.spec["mobility"]=original_mobility
            result=rec.get("result")
            observations=[o.to_dict() if hasattr(o,"to_dict") else o for o in (result.observations if result else [])]
            state.update(status=rec["status"],observations=observations,result=result)
            row={"id":iid,"actor":agent.id,"action":aid,"params":params,
                 "status":rec["status"],"failure":getattr(result,"failure_code",rec.get("rejection_code")),
                 "feedback":getattr(result,"feedback_zh",None),"t":sim.t}
            report["actions"].append(row)
            if result and result.observations and integrate and aid!="A041":
                saved=dict(state)
                yield from action(agent,"A041",{"observations":observations})
                state.update(saved)  # callers keep the primary action's status
            return rec["status"]=="succeeded"

        def route_params(agent,target):
            return {"target":target,"goal":point(target)}

        def move(agent,target):
            # Explicit region positioning uses the native motion lifecycle;
            # real autonomous route/mobility feasibility is outside this test.
            # A scheduled target_moves may change the goal during a long leg.
            for _ in range(3):
                if distance(point(agent.platform_eid), point(target)) < .01:
                    return True
                if not (yield from action(agent, "A001", route_params(agent, target))):
                    return False
            return distance(point(agent.platform_eid), point(target)) < .01

        def move_xy(agent,xy):
            """Position at an explicit legal point (e.g. a building pickup point)."""
            if distance(point(agent.platform_eid),list(xy)) < .01:
                return True
            return (yield from action(agent,"A001",{"goal":list(xy)}))

        def sensor_modality(agent,modality):
            for sensor in agent.spec.get('sensors',[]):
                if isinstance(sensor,str) and SCRIPTED_SENSORS.get(sensor,sensor)==modality:
                    return True
                if isinstance(sensor,dict) and sensor.get('enabled',True) and sensor.get('modality')==modality:
                    return True
            return False

        def observer_agent():
            ordered=sorted(sim.agents.values(),key=lambda a:(a.spec.get('mobility')!='air',a.id))
            for modality in ('rgb','thermal','depth'):
                found=next((a for a in ordered if sensor_modality(a,modality)),None)
                if found is not None:
                    return found,modality
            return None,None

        def mass_of(person):
            return float(sim.world.get(person,'attr.mass_kg',75) or 75)

        objectives = env.objectives()

        # ---------------------------------------------------------------- buildings
        # assess_all_buildings is one world objective per building.  The native
        # chain is a scripted exterior observation (A005), scripted inference
        # (A006), local localisation (A007) and one belief integration (A041);
        # only the observation evidence, never a truth flag, satisfies the goal.
        observer,modality = observer_agent()
        assess=[o for o in objectives if o['kind']=='assess_all_buildings']
        if observer is not None and assess:
            radius=max(1.,min(250.,float(observer.spec.get('sensor_range_m') or 250.)))
            tries={}
            budget=3*len(assess)+8
            for _ in range(budget):
                sim.evaluator.update()
                pending=[]
                for objective in assess:
                    if sim.evaluator.status.get(objective['id'],{}).get('satisfied'):
                        continue
                    targets=objective['target'] if isinstance(objective['target'],list) else [objective['target']]
                    for building in targets:
                        if (building in sim.world.entities and sim.world.entities[building]['cls']=='building'
                                and building not in pending and tries.get(building,0)<3):
                            pending.append(building)
                if not pending:
                    break
                # Prefer the building that covers the most remaining targets.
                best=max(pending,key=lambda b:(sum(1 for other in pending if distance(point(b),point(other))<=radius),
                                               -pending.index(b)))
                tries[best]=tries.get(best,0)+1
                report['rules'].append({'objective':'assess_all_buildings','target':best,'actor':observer.id,
                                        'actions':['move','A005','A006','A007','A041']})
                if not (yield from move(observer,best)):
                    continue
                if not (yield from action(observer,'A005',{'target':best,'modality':modality,'range_m':radius},
                                          integrate=False)):
                    continue
                observations=state['observations']
                if observations:
                    yield from action(observer,'A006',{'target':best,'observations':observations},integrate=False)
                    yield from action(observer,'A007',{'target':best,'observations':observations},integrate=False)
                yield from action(observer,'A041',{'observations':observations})

        # ---------------------------------------------------------------- people
        people = sorted({o['target'] for o in objectives if o['kind'] in {'find_person', 'extricate', 'aid', 'handover', 'recover_from_water'}})
        for person in people:
            kinds = {o['kind'] for o in objectives if o['target'] == person}
            in_water = sim.world.get(person, 'human.location_context') == 'water'
            carriers = [a for a in sim.agents.values() if a.spec.get('payload_kg', 0) >= sim.world.get(person, 'attr.mass_kg', 75) and (a.spec.get('mobility') == 'water' if in_water else a.spec.get('mobility') in {'foot', 'legged', 'wheeled'})]
            if not carriers:
                report['rules'].append({'target': person, 'blocked': 'no existing carrier with sufficient physical payload'})
                continue
            carrier = max(carriers, key=lambda a: (a.kind == 'specialist_team', a.spec.get('payload_kg', 0)))
            report['rules'].append({'target': person, 'actor': carrier.id, 'actions': ['move', 'A009', 'A017 if trapped', 'A019 if aid', 'A023 or A020', 'A021', 'A022']})
            if 'aid' in kinds:
                if not next(_items(sim, carrier, ('medical_pack',)), None):
                    kits = [i['id'] for i in sim.world.entities_of('item') if sim.world.get(i['id'], 'item.kind') == 'medical_pack' and sim.world.get(i['id'], 'item.state', 'stock') in {'stock', 'reserved'} and (sim.world.get(i['id'], 'item.quantity', 1) > 0)]
                    if kits:
                        kit = kits[0]
                        source = sim.world.get(kit, 'item.holder_id') or kit
                        if (yield from move(carrier, source)):
                            yield from action(carrier, 'A024', {'target': kit})
            if not (yield from move(carrier, person)):
                continue
            yield from action(carrier, 'A005', {'target': person, 'modality': 'rgb'})
            yield from action(carrier, 'A009', {'target': person})
            if sim.world.get(person, 'human.mobility') == 'trapped':
                params = {'target': person}
                equipment = sim.world.get(person, 'attr.rope_equipment_id')
                if equipment:
                    params['equipment_id'] = equipment
                    if not (yield from move(carrier, equipment)):
                        continue
                if not (yield from action(carrier, 'A017', params)):
                    continue
                if equipment and (not (yield from move(carrier, person))):
                    continue
            if 'aid' in kinds:
                yield from action(carrier, 'A019', {'target': person})
            if not in_water and carrier.spec.get('mobility') == 'wheeled':
                # Native A020 keeps a wheeled goal outside the building: the
                # vehicle must first stand at the exterior pickup point, then
                # load within the existing 15 m casualty range.
                pickup = sim.world.navigation_goal(point(person), carrier.spec)
                if pickup is not None and distance(pickup, point(person)) > 1e-6:
                    yield from move_xy(carrier, pickup)
            loaded = yield from action(carrier, 'A023' if in_water else 'A020', {'target': person})
            if not loaded and not in_water:
                # Hand the casualty to an existing foot/legged carrier that can
                # legally enter, instead of forcing a wheeled loading failure.
                rescuers=[a for a in sim.agents.values() if a is not carrier
                          and a.spec.get('mobility') in {'foot','legged'}
                          and a.spec.get('payload_kg',0) >= mass_of(person)]
                if rescuers:
                    carrier=max(rescuers,key=lambda a:a.spec.get('payload_kg',0))
                    report['rules'].append({'target':person,'actor':carrier.id,'actions':['move','A020'],
                                            'detail':'interior casualty handed to a carrier that can enter'})
                    if (yield from move(carrier,person)):
                        loaded = yield from action(carrier,'A020',{'target':person})
            if not loaded:
                continue
            stations = [s['id'] for s in sim.world.entities_of('station') if sim.world.get(s['id'], 'station.kind') in RECEIVING_STATION_KINDS and sim.world.get(s['id'], 'station.open_state') == 'open' and (sim.world.get(s['id'], 'station.occupied', 0) < sim.world.get(s['id'], 'station.capacity', 0))]
            if not stations:
                report['rules'].append({'target':person,'blocked':'no open existing receiving station with free capacity'})
                continue
            stations.sort(key=lambda sid: distance(point(person)[:2], point(sid)[:2]))
            for station in stations:
                params = route_params(carrier, station)
                params.update(target=person, destination_id=station)
                if (yield from action(carrier, 'A021', params)):
                    if (yield from action(carrier, 'A022', {'target': person, 'station_id': station})):
                        break

        # ---------------------------------------------------------------- hazards
        for objective in objectives:
            if objective['kind'] not in {'isolate_hazard', 'suppress_fire'}:
                continue
            target = objective['target']
            fire = objective['kind'] == 'suppress_fire'
            unit = {'isolate_hazard': ('hazmat', 'fire', 'isolation'), 'suppress_fire': ('fire', 'hazmat')}[objective['kind']]
            agent = next((a for a in sorted(sim.agents.values(), key=lambda a: a.id) if a.spec.get('mobility') == 'air' or any((word in a.id.lower() or word in a.kind.lower() for word in unit))), None)
            if not agent:
                continue
            report['rules'].append({'objective': objective['id'], 'target': target, 'actor': agent.id,
                                    'actions': ['move', 'A062' if fire else 'A063']})
            params={'target':target}
            if not fire:
                if not (yield from move(agent, target)):
                    continue
                yield from action(agent, 'A063', params)
                continue
            if sim.world.get(agent.platform_eid,'equipment.kind')=='release':
                params['equipment_id']=agent.platform_eid
            # Include profile and overlay events as well as explicit disruptions,
            # matching the world schedule assembled by Simulation.
            from .core.world.profiles import OVERLAYS
            spreads=list(sim.config.get('disruptions',[]))
            spreads.extend(sim.world.profile.get('secondary_events',[]))
            for overlay in sim.config.get('overlays',[]):
                spreads.extend(OVERLAYS[overlay]['effects'].get('world',{}).get('secondary_events',[]))
            reignitions=sorted({float(when) for e in spreads
                                if e.get('kind')=='fire_spread' and e.get('target') in (None,target)
                                for when in [e.get('t',e.get('at_s',e.get('time_s',0)))]
                                if sim.t < float(when) < self.config['until_s']})
            # A real medium is required: carry conserved water stock in one trip,
            # return to the fire, and consume one unit per native A062 step.
            for _ in range(24):
                # The declared truth predicate is fire_intensity == 0.0. Once
                # extinguished, keep watching until authored re-ignitions pass.
                if float(sim.world.get(target,'hazard.fire_intensity',1) or 0) <= 1e-9:
                    future=next((t for t in reignitions if t > sim.t),None)
                    if future is None:
                        break
                    report['rules'].append({'objective':objective['id'],'target':target,
                                            'actor':agent.id,'wait_for_fire_spread_s':future})
                    while sim.t <= future:
                        yield None
                    continue
                equipment=params.get('equipment_id')
                if equipment and float(sim.world.get(equipment,'equipment.consumable_remaining',0) or 0) <= 0:
                    stations=sorted({sim.world.get(i['id'],'item.holder_id')
                                     for i in sim.world.entities_of('item')
                                     if sim.world.get(i['id'],'item.kind')=='water'
                                     and sim.world.get(i['id'],'item.state')=='stock'
                                     and float(sim.world.get(i['id'],'item.quantity',0) or 0)>0
                                     and sim.world.get(i['id'],'item.holder_id') in sim.world.entities
                                     and sim.world.entities[sim.world.get(i['id'],'item.holder_id')]['cls']=='station'})
                    if not stations:
                        report['rules'].append({'objective':objective['id'],'target':target,
                                                'blocked':'no existing station water stock for equipment refill'})
                        break
                    if not (yield from move(agent,stations[0])):
                        break
                    if not (yield from action(agent,'A028',{'target':stations[0],'station_id':stations[0],
                                                           'equipment_id':equipment,'item_kind':'water',
                                                           'quantity':agent.spec.get('equipment_refill_units',6)})):
                        break
                if 'equipment_id' not in params and not next(_items(sim, agent, ('water',)), None):
                    stock=[i['id'] for i in sim.world.entities_of('item')
                           if sim.world.get(i['id'],'item.kind')=='water'
                           and sim.world.get(i['id'],'item.state','stock') in {'stock','reserved'}
                           and float(sim.world.get(i['id'],'item.quantity',1) or 0) > 0]
                    if not stock:
                        report['rules'].append({'objective':objective['id'],'target':target,
                                                'blocked':'no existing firefighting medium or water stock'})
                        break
                    source=sim.world.get(stock[0],'item.holder_id') or stock[0]
                    if not (yield from move(agent, source)):
                        break
                    for kit in stock[:4]:
                        yield from action(agent,'A024',{'target':kit})
                if 'equipment_id' not in params:
                    water=next(_items(sim, agent, ('water',)), None)
                    if water is None:
                        break
                    params['water_item_id']=water
                if not (yield from move(agent, target)):
                    break
                if not (yield from action(agent,'A062',params)) and state['status']!='partial':
                    break
                params.pop('water_item_id',None)  # a consumed unit is refetched

        # ---------------------------------------------------------------- access
        for objective in objectives:
            if objective['kind'] != 'restore_access':
                continue
            road = objective['target']
            if road not in sim.world.entities:
                continue
            obstacles=[o for o in (sim.world.get(road,'road.obstacle_ids') or []) if o in sim.world.entities]
            if not obstacles:
                continue
            agent=(next((a for a in sorted(sim.agents.values(),key=lambda a:a.id) if 'clear' in a.id.lower()),None)
                   or next((a for a in sorted(sim.agents.values(),key=lambda a:a.id) if a.kind=='ground_vehicle'),None)
                   or next(iter(sorted(sim.agents.values(),key=lambda a:a.id)),None))
            if agent is None:
                continue
            report['rules'].append({'objective':objective['id'],'target':road,'actor':agent.id,
                                    'actions':['move','A013']})
            for obstacle in obstacles:
                if sim.world.get(road,'road.entry_state')=='open':
                    break
                if not (yield from move(agent,obstacle)):
                    continue
                for _ in range(12):
                    if sim.world.get(road,'road.entry_state')=='open':
                        break
                    if not (yield from action(agent,'A013',{'target':obstacle})) and state['status']!='partial':
                        break
