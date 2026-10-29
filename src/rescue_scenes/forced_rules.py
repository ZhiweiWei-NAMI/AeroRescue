"""Resumable explicit rules, shared by the verifier and existing AAS plugin."""
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
        report.update({"mode":"forced_rule_verification","capability_overrides":[],"actions":[],"rules":[],"scope":"hazard isolation and personnel rescue chains; other objectives remain reported"})

        def point(eid):
            from rescue_scenes.core.sim.actions import _pose
            return list(_pose(sim,eid))

        def action(agent, aid, params):
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
            row={"id":iid,"actor":agent.id,"action":aid,"params":params,
                 "status":rec["status"],"failure":getattr(result,"failure_code",rec.get("rejection_code")),
                 "feedback":getattr(result,"feedback_zh",None),"t":sim.t}
            report["actions"].append(row)
            if result and result.observations and aid!="A041":
                observations=[o.to_dict() if hasattr(o,"to_dict") else o for o in result.observations]
                yield from action(agent,"A041",{"observations":observations})
            return rec["status"]=="succeeded"

        def route_params(agent,target):
            return {"target":target,"goal":point(target)}

        def move(agent,target):
            # Explicit region positioning uses the native motion lifecycle;
            # real autonomous route/mobility feasibility is outside this test.
            if __import__("math").dist(point(agent.platform_eid),point(target)) < .01:
                return True
            return (yield from action(agent,"A001",route_params(agent,target)))

        objectives = env.objectives()
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
                from rescue_scenes.core.sim.actions import _items
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
            if not (yield from action(carrier, 'A023' if in_water else 'A020', {'target': person})):
                continue
            stations = [s['id'] for s in sim.world.entities_of('station') if sim.world.get(s['id'], 'station.kind') in RECEIVING_STATION_KINDS and sim.world.get(s['id'], 'station.open_state') == 'open' and (sim.world.get(s['id'], 'station.occupied', 0) < sim.world.get(s['id'], 'station.capacity', 0))]
            if not stations:
                report['rules'].append({'target':person,'blocked':'no open existing receiving station with free capacity'})
            stations.sort(key=lambda sid: __import__('math').dist(point(person)[:2], point(sid)[:2]))
            for station in stations:
                params = route_params(carrier, station)
                params.update(target=person, destination_id=station)
                if (yield from action(carrier, 'A021', params)):
                    if (yield from action(carrier, 'A022', {'target': person, 'station_id': station})):
                        break
        for objective in objectives:
            if objective['kind'] not in {'isolate_hazard', 'suppress_fire'}:
                continue
            target = objective['target']
            unit = {'isolate_hazard': ('hazmat', 'fire', 'isolation'), 'suppress_fire': ('fire', 'hazmat')}[objective['kind']]
            agent = next((a for a in sorted(sim.agents.values(), key=lambda a: a.id) if a.spec.get('mobility') == 'air' or any((word in a.id.lower() or word in a.kind.lower() for word in unit))), None)
            if not agent:
                continue
            fire = objective['kind'] == 'suppress_fire'
            report['rules'].append({'objective': objective['id'], 'target': target, 'actor': agent.id, 'actions': ['move', 'A062' if fire else 'A063']})
            if (yield from move(agent, target)):
                params={'target':target}
                if fire and sim.world.get(agent.platform_eid,'equipment.kind')=='release':
                    params['equipment_id']=agent.platform_eid
                yield from action(agent, 'A062' if fire else 'A063', params)
