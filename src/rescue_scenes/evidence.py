"""Rescue chain reconstruction from committed core events; never influences policy or world truth.

ChainTracker is incremental (the engine feeds each tick's new events to publish business events);
chain_timeline() runs the same tracker over a complete event list after the run.
"""
import math


class ChainTracker:
    def __init__(self, initial):
        self.entities=initial['entities']
        self.poses={eid:row['fields']['entity.pose'] for eid,row in self.entities.items() if row['cls']=='platform'}
        self.obstacles=[row for row in self.entities.values() if row['cls']=='obstacle']
        self.cleared=set(); self.seen=set(); self.timeline=[]

    def _note(self, out, event, stage, target, **extra):
        key=(stage,target)
        if key in self.seen: return
        self.seen.add(key)
        row={'t':event['t'],'seq':event['seq'],'stage':stage,'target':target,'actor':event.get('actor'),**extra}
        self.timeline.append(row); out.append(row)

    def feed(self, event):
        """Return the chain stages first evidenced by this event."""
        out=[]; entities=self.entities
        after=event['after']; target=event['subject']; field=event['field']
        if event['kind']=='action.result' and isinstance(after,dict):
            actor=entities.get(event['actor'],{})
            if after.get('failure_code')=='blocked' and actor.get('fields',{}).get('platform.kind')=='ground_vehicle':
                observed=set()
                for observation in after.get('observations',[]):
                    observed.update(observation.get('about',[]))
                    route=observation.get('values',{}).get('job.result_ref',{})
                    if isinstance(route,dict): observed.update(route.get('blocked_by',[]))
                for obstacle in self.obstacles:
                    for road_id in obstacle['fields']['obstacle.blocks_ids']:
                        if road_id in observed or obstacle['id'] in observed:
                            self._note(out,event,'blocked',road_id,action=after['action_id'],vehicle=event['actor'],obstacle=obstacle['id'])
            for observation in after.get('observations',[]):
                if observation.get('kind')=='interior_search_detection':
                    for human in observation['about']:
                        self._note(out,event,'found',human,sensing='scripted',observation=observation['id'])
        if event['layer']!='world': return out
        if field=='obstacle.state' and after=='partly_cleared':
            self._note(out,event,'partly_cleared',target)
        if field=='obstacle.state' and after in ('cleared','moved'):
            self.cleared.add(target); self._note(out,event,'cleared',target)
        if field=='road.entry_state' and after=='open' and event['before'] in ('blocked','closed','unsafe'):
            self._note(out,event,'road_open',target)
        if field=='human.support_received':
            for token,stage in (('contacted','contact'),('extricated','extricated')):
                if token in after and token not in (event['before'] or []): self._note(out,event,stage,target)
        if field=='human.carrier_id' and after is not None:
            self._note(out,event,'loaded',target,carrier=after)
        if field=='human.handover_status':
            stage={'in_transit':'transporting','arrived':'transported','acknowledged':'handover_ack'}.get(after)
            if stage: self._note(out,event,stage,target)
        if field!='entity.pose' or target not in self.poses: return out
        previous=self.poses[target]; self.poses[target]=after
        if entities[target]['fields']['platform.kind']!='ground_vehicle': return out
        for obstacle in self.obstacles:
            if obstacle['id'] not in self.cleared: continue
            f=obstacle['fields']; p=f['entity.pose']
            for road_id in f['obstacle.blocks_ids']:
                road=entities[road_id]; a,b=road['geometry']['coords'][:2]
                dx,dy=b[0]-a[0],b[1]-a[1]; norm=dx*dx+dy*dy
                if norm<=0: continue
                def project(q):
                    u=((q[0]-a[0])*dx+(q[1]-a[1])*dy)/norm
                    lateral=abs((q[0]-a[0])*dy-(q[1]-a[1])*dx)/math.sqrt(norm)
                    return u,lateral
                u,lateral=project(previous); v,other=project(after); cut,_=project(p)
                if min(u,v)<=cut<=max(u,v) and abs(u-v)>1e-9 and max(lateral,other)<=road['fields']['road.width_m']/2:
                    self._note(out,event,'passed',road_id,vehicle=target,obstacle=obstacle['id'])
        return out


def chain_timeline(events, initial):
    tracker=ChainTracker(initial)
    for event in events: tracker.feed(event)
    return sorted(tracker.timeline,key=lambda row:(row['t'],row['seq']))
