"""Deterministic discrete-time simulation; agents alone own their task copies."""
from copy import deepcopy
import hashlib
import heapq
import math
import random

from ..catalog import load
from ..contracts import Event, ActionResult, ActionOutcome, STATE_MACHINES, EVENT_FAMILIES, validate, reachable
from .ledger import Ledger, cost_record
from .network import Network
from .compute import Compute
from .evaluation import Evaluator


def camera_identity_alias(point, pose, spec):
    """Lazy camera import so core simulation keeps no camera dependency."""
    from ...camera import candidate_identity_alias
    return candidate_identity_alias(point, pose, spec)


def contact_response_pose(values):
    """The contacted person's pose as carried by the delivered A009 response.

    This is the contact measurement the effect returns to the agent, not a
    fresh world lookup: the identity decision must consume only what the agent
    was given (audit F-A finding 2).
    """
    if not isinstance(values, dict):
        return None
    pose = values.get('entity.pose') or values.get('human.pose')
    if not isinstance(pose, (list, tuple)) or len(pose) < 2:
        return None
    try:
        return [float(pose[0]), float(pose[1])]
    except (TypeError, ValueError):
        return None


def contact_identity_association(candidate_position, response_values, spec):
    """Whether an executed camera contact may claim the responder's identity.

    Both inputs are agent-side: ``candidate_position`` is this agent's own
    perception ground estimate (its belief) and ``response_values`` is the
    delivered contact response carrying the responder identity and contact
    pose. Returns ``(aliased, distance_m)``; neither value reads world truth.
    """
    pose = contact_response_pose(response_values)
    if candidate_position is None or pose is None:
        return False, None
    try:
        point = [float(candidate_position[0]), float(candidate_position[1])]
    except (TypeError, ValueError, IndexError):
        return False, None
    distance = math.hypot(point[0] - pose[0], point[1] - pose[1])
    if not math.isfinite(distance):
        return False, None
    return camera_identity_alias(point, pose, spec), round(distance, 3)


class Simulation:
    def __init__(self, world, agents, config):
        self.t = 0.0
        self.events = []
        self.world = world
        self.agents = {agent.id: agent for agent in agents}
        self.config = deepcopy(config)
        self.catalog = load()
        self.tasks_index = {}
        self.instances = {}
        # Keep the full archive above; tick work only visits live reservations.
        self._active_instances = {}
        self._heap = []
        self._schedule_seq = 0
        self._instance_seq = 0
        self._started = False
        self._action_context = None
        self._due_world = []
        self._family_specs = {f["id"]: f for f in EVENT_FAMILIES}
        self.ledger = Ledger(self)
        self.network = Network(self)
        self.compute = Compute(self)
        self.evaluator = Evaluator(self)
        for disruption in self.config.get("disruptions", []):
            self.schedule(float(disruption.get("t", 0)), self._enqueue_world, "inject", deepcopy(disruption))
        # Profile schedules are authored world events, not task proposals.
        from ..world.profiles import OVERLAYS
        profile = getattr(world, "profile", {})
        secondary_specs = list(profile.get("secondary_events", [])) if isinstance(profile, dict) else []
        for oid in config.get("overlays", []):
            secondary_specs.extend(OVERLAYS[oid]["effects"].get("world", {}).get("secondary_events", []))
        for secondary in secondary_specs:
            if isinstance(secondary, dict):
                when = secondary.get("t", secondary.get("at_s", secondary.get("time_s")))
                if when is not None:
                    self.schedule(float(when), self._enqueue_world, "secondary", deepcopy(secondary))
        for oid in config.get("overlays", []):
            effects = OVERLAYS[oid]["effects"].get("agents", {})
            for key, kind in (("equipment_faults", "equipment_fault"), ("wrong_evidence", "wrong_evidence"), ("stale_versions", "stale_version")):
                for spec in effects.get(key, []):
                    targets = [a for a in sorted(self.agents.values(), key=lambda a: a.id) if a.kind == spec.get("kind")][:int(spec.get("count", 1))]
                    for agent in targets:
                        item = dict(kind=kind, target=agent.id, params=deepcopy(spec))
                        self.schedule(float(spec.get("t", 0)), self._enqueue_world,
                                      "stale" if kind == "stale_version" else "inject", item)

    def rng(self, stream, key=""):
        seed = self.config.get("seed", getattr(self.world, "seed", 1))
        digest = hashlib.sha256(f"{seed}|{stream}|{key}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest, "big"))

    def emit(self, **kw):
        family = kw.get("family", "field.write")
        family_spec = self._family_specs.get(family, {})
        defaults = dict(seq=len(self.events) + 1, t=self.t, layer=family_spec.get("layer", "action"),
                        family=family, kind=family, grounding=family_spec.get("grounding", "catalog"),
                        refs=[], subject="simulation", field=None, before=None, after=None,
                        actor=None, task=None, task_rev=None, action_instance=None, cost=None,
                        visible_to=["*truth*"], backend_mode=None, scripted=False,
                        contract_violation=False, text_zh="仿真状态更新。")
        defaults.update(kw)
        defaults["seq"], defaults["t"] = len(self.events) + 1, self.t
        defaults["refs"] = list(dict.fromkeys(defaults.get("refs") or []))
        field = defaults.get("field")
        if field in self.catalog.fields_by_key:
            fid = self.catalog.fields_by_key[field]["id"]
            if fid not in defaults["refs"]:
                defaults["refs"].append(fid)
        before, after = defaults["before"], defaults["after"]
        if family == "belief.update":
            before = before.get("value") if isinstance(before, dict) else before
            after = after.get("value") if isinstance(after, dict) else after
        if field in STATE_MACHINES and before is not None and before != after:
            check = reachable if family == "belief.update" else validate
            if not check(field, before, after):
                defaults["contract_violation"] = True
        event = Event(**defaults)
        self.events.append(event)
        if defaults["task"]:
            tid = defaults["task"]
            index = self.tasks_index.setdefault(tid, dict(creator=None, type=None, history=[]))
            index["history"].append(dict(seq=event.seq, t=self.t, kind=event.kind,
                                         actor=event.actor, after=deepcopy(event.after)))
            for agent in self.agents.values():
                task = agent.tasks.get(tid)
                if task:
                    index["creator"] = task.get("creator")
                    index["type"] = task.get("type")
                    break
        return event

    def write_truth(self, eid, key, value, cause, actor=None, task_ref=None,
                    action_instance=None, visible_to=None, backend_mode=None, scripted=False):
        history_start = len(self.world.history)
        # WorldTruth.set owns the detached field/history/return snapshots.
        before, after = self.world.set(eid, key, value, {"rule": cause, "t": self.t, "scripted": scripted})
        # D01/D03 can derive additional physical changes in set(); emit them too.
        writes = self.world.history[history_start:]
        if not writes:
            writes = [dict(eid=eid, key=key, before=before, after=after, provenance=cause)]
        for write in writes:
            weid, wkey = write.get("eid", write.get("entity_id", eid)), write.get("key", write.get("field", key))
            previous, current = write.get("before"), write.get("after")
            if previous == current:
                continue
            family = "sm:" + wkey if wkey in STATE_MACHINES else "field.write"
            self.emit(layer="world", family=family, kind=wkey,
                      grounding="catalog" if wkey in self.catalog.fields_by_key else "proposed",
                      refs=[write.get("provenance", cause)], subject=weid,
                      field=wkey, before=previous, after=current, actor=actor,
                      task=task_ref[0] if task_ref else None,
                      task_rev=task_ref[1] if task_ref else None,
                      action_instance=action_instance,
                      visible_to=visible_to if visible_to is not None and weid == eid else ["*truth*"],
                      backend_mode=backend_mode, scripted=scripted,
                      text_zh=f"{weid} 的 {wkey} 已更新。")
            for agent in self.agents.values():
                if agent.platform_eid == weid and isinstance(agent.platform, dict):
                    agent.platform[wkey] = deepcopy(current)
                    aliases = {"entity.pose": ("pose",), "platform.battery_remaining_Wh": ("battery_Wh", "battery", "battery_remaining_Wh"),
                               "platform.motion_state": ("motion_state",), "platform.health_state": ("health", "health_state"),
                               "platform.payload_item_ids": ("payload",), "platform.payload_used_kg": ("payload_used_kg",)}.get(wkey, ())
                    for alias in aliases:
                        agent.platform[alias] = deepcopy(current)
        return before, after

    def _enqueue_world(self, kind, spec):
        """Queue one due world change on this simulation's own live list.

        A bound ``list.append`` captured in ``__init__`` keeps pointing at the
        list object that ``step`` swaps out, and at the *original* simulation
        after ``deepcopy(sim)``. Reading ``self._due_world`` at call time keeps
        every scheduled ``t > 0`` disruption bound to its own holder.
        """
        self._due_world.append((kind, spec))

    def schedule(self, delay_s, fn, *args):
        delay_s = float(delay_s)
        if not math.isfinite(delay_s) or delay_s < 0:
            raise ValueError("调度延时无效 / invalid scheduling delay")
        self._schedule_seq += 1
        heapq.heappush(self._heap, (self.t + delay_s, self._schedule_seq, fn, args))

    def start(self):
        if not self._started:
            self._started = True
            for agent in sorted(self.agents.values(), key=lambda a: a.id):
                agent.on_start(self)

    def step(self, dt=None):
        first = len(self.events)
        self.start()
        dt = float(self.config.get("tick_s", 5) if dt is None else dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("时间步长必须为正 / tick must be positive")
        end = self.t + dt
        while self._heap and self._heap[0][0] <= end:
            when, _, fn, args = heapq.heappop(self._heap)
            self.t = when
            fn(*args)
        self.t = end
        for instance in list(self._active_instances.values()):
            if instance["status"] == "running":
                self._progress(instance)
        self.network.step()
        self.compute.step()
        due_world, self._due_world = self._due_world, []
        for kind, spec in due_world:
            if kind == "inject":
                self.inject(spec)
            elif kind == "stale":
                self._stale_version(spec)
            else:
                self._secondary(spec)
        self._expire_tasks()
        for agent in sorted(self.agents.values(), key=lambda a: a.id):
            agent.on_tick(self)
        self.evaluator.update()
        return self.events[first:]

    def run(self, until_s):
        until_s = float(until_s)
        if not math.isfinite(until_s) or until_s < self.t:
            raise ValueError("结束时刻无效 / invalid end time")
        first = len(self.events)
        while self.t < until_s:
            self.step(min(float(self.config.get("tick_s", 5)), until_s - self.t))
        return self.events[first:]

    def _status(self, record, state, before=None, **extra):
        if before is None:
            before = record.get("status")
        record["status"] = state
        if state in ("accepted", "running"):
            self._active_instances[record["id"]] = record
        else:
            self._active_instances.pop(record["id"], None)
        payload = dict(status=state, rejection_code=record.get("rejection_code", "none"),
                       failure_code=record.get("failure_code", "none"))
        payload.update(extra)
        refs = [record["action_id"], "F235"]
        if state == "accepted":
            refs.extend(record["params"].get("_guard_capabilities", record["model"].capabilities))
        self.emit(layer="action", family="sm:action.status", kind="action.status",
                  refs=refs, subject=record["id"],
                  field="action.status", before=before, after=state,
                  actor=record["agent_id"], task=record["task_ref"][0] if record["task_ref"] else None,
                  task_rev=record["task_ref"][1] if record["task_ref"] else None,
                  action_instance=record["id"], visible_to=[record["agent_id"]],
                  backend_mode=record["model"].backend_mode, scripted=record["model"].backend_mode == "stub",
                  text_zh=f"{record['agent_id']} 的 {record['action_id']}：{state}。")
        for key in ("rejection_code", "failure_code"):
            if payload.get(key, "none") != "none":
                self.emit(layer="action", family="field.write", kind="action.result",
                          refs=[record["action_id"]], subject=record["id"], field="action." + key,
                          before="none", after=payload[key], actor=record["agent_id"],
                          task=record["task_ref"][0] if record["task_ref"] else None,
                          action_instance=record["id"], visible_to=[record["agent_id"]],
                          text_zh=f"动作返回原因：{payload[key]}。")

    def submit_action(self, agent_id, action_id, params, task_ref=None):
        from .actions import MODELS, PHYSICAL
        if agent_id not in self.agents or action_id not in MODELS:
            raise ValueError("智能体或动作编号无效 / unknown agent or action")
        if not isinstance(params, dict):
            raise ValueError("动作参数必须是对象 / action params must be an object")
        agent, model = self.agents[agent_id], MODELS[action_id]
        self._instance_seq += 1
        iid = f"act-{self._instance_seq:07d}"
        params = deepcopy(params)
        if action_id == 'A009':
            target = params.get('target', params.get('target_id'))
            if isinstance(target, str) and target.startswith('rescue-camera/') and '/person/' in target:
                estimate = agent.belief.get(target, 'belief.estimated_pose')
                kind = agent.belief.get(target, 'entity.kind')
                if target not in self.world.entities and kind and kind['value'] == 'human' and estimate:
                    params['camera_candidate_id'] = target
                    params['contact_position'] = deepcopy(estimate['value'])
        ref = tuple(task_ref) if task_ref else None
        # A PX4-managed vehicle owns its own flight: the kernel must neither
        # interpolate its pose nor settle the action on a scripted duration.
        external = bool(action_id in ("A001", "A021")
                        and getattr(agent, "motion_backend", "native") == "px4")
        params["_external_motion"] = external
        params.update(_instance_id=iid, _task_ref=ref)
        rec = dict(id=iid, agent_id=agent_id, action_id=action_id, params=params,
                   task_ref=ref, model=model, status=None, started_s=self.t,
                   progress_s=0.0, rejection_code="none", failure_code="none", cancel_requested=False,
                   external_motion=external)
        self.instances[iid] = rec
        self._status(rec, "requested")
        task = agent.tasks.get(ref[0]) if ref else None
        stale = bool(ref and (not task or task.get("rev") != ref[1]))
        try:
            ok, code = (False, "stale") if stale else model.guard(self, agent, params)
            if not stale and action_id == 'A009' and params.get('camera_candidate_id') and code == 'incompatible':
                # A009's ordinary guard rejects a belief-only ID at the target
                # class gate. Validate the remaining physical/resource gates;
                # the location is resolved only when the contact model executes.
                ok, code = self._camera_contact_guard(agent, params)
            if ok:
                plan = model.plan(self, agent, params)
                cost = cost_record(plan)
                duration = float(plan.get("duration_s", plan.get("time_s", 1.0)))
                if not math.isfinite(duration) or duration < 0:
                    raise ValueError("动作时长无效 / invalid action duration")
                cost["time_s"] = duration
                if action_id == 'A009' and params.get('camera_candidate_id') and params.get('energy_budget_Wh') is not None:
                    if cost['energy_Wh'] > float(params['energy_budget_Wh']):
                        ok, code = False, 'insufficient'
                rec["plan"] = dict(plan, duration_s=duration, **cost)
                params["_plan"] = deepcopy(rec["plan"])
                # Concurrent reservations cannot spend the same battery twice.
                battery = self.world.get(agent.platform_eid, "platform.battery_remaining_Wh")
                reserved = sum(r.get("plan", {}).get("energy_Wh", 0) for r in self._active_instances.values()
                               if r is not rec and r["agent_id"] == agent_id and r["status"] in ("accepted", "running"))
                if not agent.spec.get("mains_power", agent.spec.get("mobility") == "none") and battery is not None and float(battery) - reserved < cost["energy_Wh"]:
                    ok, code = False, "insufficient"
        except (TypeError, ValueError, OverflowError):
            ok, code = False, "incompatible"
        if not ok:
            rec["rejection_code"] = code
            rec["plan"] = dict(duration_s=0.0, **cost_record())
            self._status(rec, "rejected")
            self.schedule(float(self.config.get("tick_s", 5)), self._notify_rejected, iid)
            return iid
        if task and isinstance(task.get("method"), dict):
            method = task["method"]
            seq_id = params.get("seq_id", method.get("seq_id"))
            steps = self.catalog.steps.get(seq_id, [])
            step_index = int(method.get("step_index", 0))
            source_step = next((s for s in steps if s["step"] == params.get("step")), None)
            if source_step is None and 0 <= step_index < len(steps):
                source_step = steps[step_index]
            if source_step and action_id in source_step["options"]:
                rec["sequence_step"] = (seq_id, source_step["step"])
        self._status(rec, "accepted")
        self._status(rec, "running")
        if action_id in PHYSICAL:
            self.write_truth(agent.platform_eid, "platform.current_action_id", iid, action_id,
                             actor=agent_id, task_ref=ref, action_instance=iid, visible_to=[agent_id])
            self.write_truth(agent.platform_eid, "platform.busy_until_est_s", self.t + rec["plan"]["duration_s"], action_id,
                             actor=agent_id, task_ref=ref, action_instance=iid, visible_to=[agent_id])
        if hasattr(model, "start"):
            initial = model.start(self, agent, params)
            if initial is not None:
                self._apply_outcome(rec, initial)
        if task and task.get("status") == "accepted" and action_id not in {f"A{i:03d}" for i in range(50, 59)}:
            task["status"] = "executing"
            task.setdefault("history", []).append(dict(t=self.t, status="executing", note_zh="首个执行动作开始。"))
            self.emit(layer="task", family="sm:task.status", kind="task.status", refs=["F201", "KERNEL", task["type"]],
                      subject=task["id"], field="task.status", before="accepted", after="executing",
                      actor=agent_id, task=task["id"], task_rev=task["rev"], visible_to=[agent_id],
                      text_zh=f"{agent_id} 开始执行 {task['id']}。")
        if action_id in ("A001", "A002", "A003", "A021"):
            self.write_truth(agent.platform_eid, "platform.motion_state", "moving", action_id,
                             actor=agent_id, task_ref=ref, action_instance=iid, visible_to=[agent_id])
        if rec["external_motion"]:
            # No fixed completion timer: PX4 arrival settles the action through
            # complete_external_action; this bounded watchdog only stops a
            # vehicle that never reports.
            self.schedule(max(rec["plan"]["duration_s"] * 3.0, 180.0), self._external_watchdog, iid)
        else:
            self.schedule(rec["plan"]["duration_s"], self._complete, iid)
        return iid

    def complete_external_action(self, iid):
        """Settle a running externally-driven motion now (PX4 reached its goal).

        The normal completion path still runs: the model's effect, the ledger,
        the emitted ``action.result`` and ``on_action_result`` all execute, so a
        task cursor advances exactly as it does for a scripted arrival.
        """
        rec = self.instances.get(iid)
        if not rec or rec.get("status") != "running" or not rec.get("external_motion"):
            return False
        if rec.get("cancel_requested"):
            return False
        self._complete(iid)
        return rec.get("status") != "running"

    def _external_watchdog(self, iid):
        rec = self.instances.get(iid)
        if not rec or rec.get("status") != "running" or not rec.get("external_motion"):
            return
        self.cancel_action(iid, "外部飞控在超时窗口内未到达目标点。")

    def _camera_contact_guard(self, agent, params):
        from .actions import _equipment, _fields, _near
        if self.world.get(agent.platform_eid, 'platform.health_state', 'nominal') == 'failed':
            return False, 'incompatible'
        active = self.world.get(agent.platform_eid, 'platform.current_action_id')
        if active and active != params['_instance_id']:
            return False, 'busy'
        equipment = _equipment(self, agent, params)
        if params.get('equipment_id') and equipment is None:
            return False, 'incompatible'
        if equipment:
            fields = _fields(self, equipment)
            if not fields.get('equipment.enabled', True) or fields.get('equipment.health', 'ready') == 'failed':
                return False, 'incompatible'
            authority = fields.get('equipment.operator_authority', [])
            if isinstance(authority, str):
                authority = [authority]
            if authority and not ({agent.id, agent.kind, 'any'} & set(authority)):
                return False, 'unauthorized'
            if equipment != agent.platform_eid and not _near(
                    self, agent, equipment, fields.get('attr.operator_reach_m', 10.)):
                return False, 'incompatible'
        return True, 'none'

    def _camera_contact_effect(self, rec, agent):
        from .actions import _pose, _own_pose, _dist
        params = rec['params']
        point, own = params['contact_position'], _own_pose(self, agent)
        # This is a physical interaction query, never a perception association.
        # The existing A009 radius and response/alive model still apply.
        nearby = sorted((_dist(point, _pose(self, eid)), eid) for eid, entity in self.world.entities.items()
                        if entity['cls'] == 'human' and _dist(point, _pose(self, eid)) <= 35.
                        and _dist(own, _pose(self, eid)) <= 35.)
        if not nearby or _dist(own, point) > 35.:
            return ActionOutcome(status='failed', failure_code='sensor',
                feedback_zh='图像候选位置的接触范围内没有可确认人员；候选不代表真实人员。')
        target = nearby[0][1]
        # The 35 m above is the physical response radius, not an identity
        # radius: a stale candidate can otherwise be linked to a bystander
        # metres away. The identity decision below consumes only the delivered
        # contact response and this agent's own candidate belief; the physical
        # contact and its needs/report still stand either way.
        execution = dict(params, target=target, target_id=target)
        outcome = rec['model'].effect(self, agent, execution, self.rng('action', rec['id']))
        aliased = False
        if outcome.status == 'succeeded':
            for observation in outcome.observations:
                if observation.kind == 'human_response':
                    # The successful contact reports the responder's identity
                    # (about=[target]) and the contact pose it measured. That
                    # delivered measurement is the only identity and pose the
                    # association may use; never a second world lookup (audit
                    # F-A finding 2). A response that names another responder or
                    # lacks a delivered pose cannot claim this candidate.
                    named = observation.about if isinstance(observation.about, (list, tuple)) else []
                    observation.values.update({'entity.kind': 'human',
                        'observation.camera_candidate_id': params['camera_candidate_id']})
                    if target in named:
                        claimed, distance = contact_identity_association(
                            point, observation.values, agent.spec)
                    else:
                        claimed, distance = False, None
                    aliased = aliased or claimed
                    observation.values.update({
                        'observation.candidate_to_contact_pose_m': distance,
                        'observation.identity_alias': claimed})
            if aliased:
                from ..contracts import Observation
                outcome.observations.append(Observation(id=f"{rec['id']}/contact-association", by=agent.id,
                    t=self.t, about=[params['camera_candidate_id']], kind='human_contact_association',
                    values={'belief.entity_ref': target},
                    confidence=.9, model='A009_spatial_contact'))
        return outcome

    def _result(self, rec, outcome, cost):
        observations = outcome.observations
        evidence_ids = outcome.evidence_ids
        camera = rec.get("camera_observations")
        if camera:
            observations = list(observations) + deepcopy(camera)
            evidence_ids = list(dict.fromkeys(list(evidence_ids) + [row["id"] for row in camera]))
        return ActionResult(instance_id=rec["id"], action_id=rec["action_id"], agent_id=rec["agent_id"],
                            task_ref=rec["task_ref"], status=outcome.status,
                            rejection_code=rec["rejection_code"], failure_code=outcome.failure_code,
                            feedback_zh=outcome.feedback_zh, observations=observations,
                            evidence_ids=evidence_ids, cost=cost,
                            backend_mode=rec["model"].backend_mode,
                            params={k: deepcopy(v) for k, v in rec["params"].items() if not k.startswith("_")})

    def _notify_rejected(self, iid):
        rec = self.instances[iid]
        result = self._result(rec, ActionOutcome(status="rejected", feedback_zh=rec["params"].get(
            "_guard_feedback_zh", f"动作请求被拒绝：{rec['rejection_code']}。")), cost_record())
        rec["result"] = result
        self.agents[rec["agent_id"]].on_action_result(self, result)

    def _progress(self, rec):
        elapsed = max(0.0, self.t - rec["started_s"])
        if not rec.get("external_motion"):
            elapsed = min(elapsed, rec["plan"]["duration_s"])
        agent = self.agents[rec["agent_id"]]
        energy = rec["plan"]["energy_Wh"]
        if not rec.get("external_motion") and energy > 0 and not agent.spec.get("mains_power", agent.spec.get("mobility") == "none"):
            battery = self.world.get(agent.platform_eid, "platform.battery_remaining_Wh")
            if battery is not None and float(battery) < energy:
                supported = rec["plan"]["duration_s"] * max(0.0, float(battery)) / energy
                elapsed = min(elapsed, max(rec["progress_s"], supported))
        delta = elapsed - rec["progress_s"]
        if delta > 0 and hasattr(rec["model"], "progress"):
            outcome = rec["model"].progress(self, self.agents[rec["agent_id"]], rec["params"], elapsed, delta)
            if outcome is not None:
                self._apply_outcome(rec, outcome)
        rec["progress_s"] = elapsed

    def _apply_outcome(self, rec, outcome):
        agent, ref = self.agents[rec["agent_id"]], rec["task_ref"]
        first = len(self.events)
        for eid, key, value in outcome.truth_writes:
            self.write_truth(eid, key, value, rec["action_id"], actor=agent.id, task_ref=ref,
                             action_instance=rec["id"], backend_mode=rec["model"].backend_mode,
                             scripted=rec["model"].backend_mode == "stub")
        for key, value in outcome.local_writes:
            if key.startswith("platform.") or key == "entity.pose":
                self.write_truth(agent.platform_eid, key, value, rec["action_id"], actor=agent.id,
                                 task_ref=ref, action_instance=rec["id"], visible_to=[agent.id])
            else:
                before = deepcopy(agent.platform.get(key))
                agent.platform[key] = deepcopy(value)
                layer = "compute" if key.startswith(("compute.", "job.", "cache.", "service.")) else "comms" if key.startswith(("message.", "radio.", "link.")) else "knowledge"
                self.emit(layer=layer, family="field.write", kind=key,
                          grounding="catalog" if key in self.catalog.fields_by_key else "proposed",
                          refs=[rec["action_id"]], subject=agent.id, field=key,
                          before=before, after=deepcopy(value), actor=agent.id,
                          task=ref[0] if ref else None, task_rev=ref[1] if ref else None,
                          action_instance=rec["id"], visible_to=[agent.id],
                          backend_mode=rec["model"].backend_mode,
                          scripted=getattr(rec["model"], "scripted", False),
                          text_zh="动作更新本机资源或过程记录，不修改其他持有者副本。")
        # Rescue-chain rows derive from the writes just applied above, so a row
        # exists exactly when the state transition really happened (failed
        # outcomes write none of these fields).
        from .actions import record_rescue_chain
        record_rescue_chain(self, self.events[first:], rec)

    def _record_step(self, rec, status):
        if rec.get("sequence_step"):
            seq_id, step = rec["sequence_step"]
            ref = rec["task_ref"]
            self.emit(layer="action", family="field.write", kind="sequence.step", refs=[seq_id, rec["action_id"]],
                      subject=rec["id"], actor=rec["agent_id"], task=ref[0] if ref else None,
                      task_rev=ref[1] if ref else None, action_instance=rec["id"],
                      after=dict(seq_id=seq_id, step=step, action_id=rec["action_id"], status=status),
                      visible_to=[rec["agent_id"]], text_zh="记录实际执行过的方法步骤，不将选择当作执行。")

    def _release_platform(self, rec):
        agent = self.agents[rec["agent_id"]]
        if self.world.get(agent.platform_eid, "platform.current_action_id") == rec["id"]:
            self.write_truth(agent.platform_eid, "platform.current_action_id", None, "KERNEL",
                             actor=agent.id, action_instance=rec["id"], visible_to=[agent.id])
            self.write_truth(agent.platform_eid, "platform.busy_until_est_s", self.t, "KERNEL",
                             actor=agent.id, action_instance=rec["id"], visible_to=[agent.id])

    def _complete(self, iid):
        rec = self.instances[iid]
        if rec["status"] != "running" or rec["cancel_requested"]:
            return
        self._progress(rec)
        agent = self.agents[rec["agent_id"]]
        ref = rec["task_ref"]
        task = agent.tasks.get(ref[0]) if ref else None
        stale = bool(ref and (not task or task.get("rev") != ref[1] or task.get("status") in ("canceled", "expired")))
        if stale:
            self._finish_canceled(iid, "任务版本过期，隔离迟到结果。")
            return
        # PX4 reports battery remaining after the flight, not before it.
        available = self.world.get(agent.platform_eid, "platform.battery_remaining_Wh")
        if not rec.get("external_motion") and not agent.spec.get("mains_power", agent.spec.get("mobility") == "none") and available is not None and float(available) < rec["plan"]["energy_Wh"]:
            outcome = ActionOutcome(status="failed", failure_code="energy", feedback_zh="执行期间电量不足。")
            rec["params"]["_cancel_elapsed_s"] = rec["progress_s"]
            if hasattr(rec["model"], "cancel"):
                self._apply_outcome(rec, rec["model"].cancel(self, agent, rec["params"]))
        else:
            self._action_context = rec
            try:
                if rec['action_id'] == 'A009' and rec['params'].get('camera_candidate_id'):
                    outcome = self._camera_contact_effect(rec, agent)
                else:
                    outcome = rec["model"].effect(self, agent, rec["params"], self.rng("action", iid))
            finally:
                self._action_context = None
        self._apply_outcome(rec, outcome)
        rec["failure_code"] = outcome.failure_code
        self._status(rec, outcome.status)
        self._release_platform(rec)
        if rec["action_id"] in ("A001", "A002", "A003", "A021"):
            state = "blocked" if outcome.failure_code == "blocked" else "idle"
            self.write_truth(agent.platform_eid, "platform.motion_state", state, rec["action_id"], actor=agent.id,
                             task_ref=ref, action_instance=iid, visible_to=[agent.id])
        cost = cost_record(rec["plan"])
        if rec.get("external_motion") and rec["plan"]["duration_s"] > 0:
            # Record actual flight time; the other authored costs remain
            # bounded by the planned reservation.
            elapsed = max(0.0, self.t - rec["started_s"])
            ratio = min(1.0, elapsed / rec["plan"]["duration_s"])
            cost = cost_record({k: v * ratio for k, v in cost.items()})
            cost["time_s"] = elapsed
        if outcome.failure_code == "energy" and available is not None:
            cost["energy_Wh"] = min(cost["energy_Wh"], max(0.0, float(available)))
            cost["cpu_s"] *= rec["progress_s"] / max(.001, rec["plan"]["duration_s"])
        self.ledger.add(iid, agent.id, rec["action_id"], cost, ref)
        result = self._result(rec, outcome, cost)
        rec["result"] = result
        self._record_step(rec, outcome.status)
        self.emit(layer="action", family="field.write", kind="action.result", refs=[rec["action_id"], "F245"],
                  subject=iid, actor=agent.id, task=ref[0] if ref else None,
                  task_rev=ref[1] if ref else None, action_instance=iid,
                  after=result.to_dict(), backend_mode=rec["model"].backend_mode,
                  scripted=rec["model"].backend_mode == "stub", visible_to=[agent.id],
                  text_zh=outcome.feedback_zh or "动作结束，结果保留实际失败与不确定性。")
        agent.on_action_result(self, result)

    def cancel_action(self, instance_id, reason="用户请求安全停止"):
        rec = self.instances.get(instance_id)
        if not rec or rec["status"] not in ("accepted", "running") or rec["cancel_requested"]:
            return False
        rec["cancel_requested"] = True
        self.emit(layer="action", family="field.write", kind="action.cancel_requested", refs=["F244", rec["action_id"]],
                  subject=instance_id, field="action.cancel_requested", before=False, after=True,
                  actor=rec["agent_id"], task=rec["task_ref"][0] if rec["task_ref"] else None,
                  action_instance=instance_id, visible_to=[rec["agent_id"]], text_zh="动作将在下一安全点停止。")
        # A model may finish an irreversible step, or select its next safe stop.
        rec["params"]["_cancel_elapsed_s"] = max(0.0, self.t - rec["started_s"])
        rec["params"]["_remaining_s"] = max(0.0, rec["plan"]["duration_s"] - rec["params"]["_cancel_elapsed_s"])
        delay = float(self.config.get("tick_s", 5))
        if hasattr(rec["model"], "cancel_delay"):
            delay = max(delay, float(rec["model"].cancel_delay(self, self.agents[rec["agent_id"]], rec["params"])))
        self.schedule(delay, self._finish_canceled, instance_id, reason)
        return True

    def _finish_canceled(self, iid, reason):
        rec = self.instances[iid]
        if rec["status"] not in ("accepted", "running"):
            return
        self._progress(rec)
        rec["params"]["_cancel_elapsed_s"] = rec["progress_s"]
        if hasattr(rec["model"], "cancel"):
            cleanup = rec["model"].cancel(self, self.agents[rec["agent_id"]], rec["params"])
            if cleanup is not None:
                self._apply_outcome(rec, cleanup)
        duration = rec["plan"]["duration_s"]
        ratio = min(1.0, max(0.0, (self.t - rec["started_s"]) / duration)) if duration else 0.0
        cost = {k: v * ratio for k, v in cost_record(rec["plan"]).items()}
        cost = cost_record(cost)
        if rec.get("external_motion"):
            cost["time_s"] = max(0.0, self.t - rec["started_s"])
        agent = self.agents[rec["agent_id"]]
        available = self.world.get(agent.platform_eid, "platform.battery_remaining_Wh")
        if available is not None and not agent.spec.get("mains_power", agent.spec.get("mobility") == "none"):
            cost["energy_Wh"] = min(cost["energy_Wh"], max(0.0, float(available)))
        self._status(rec, "canceled")
        self._release_platform(rec)
        agent = self.agents[rec["agent_id"]]
        if rec["action_id"] in ("A001", "A002", "A003", "A021"):
            self.write_truth(agent.platform_eid, "platform.motion_state", "idle", "KERNEL", actor=agent.id,
                             action_instance=iid, visible_to=[agent.id])
        self.ledger.add(iid, agent.id, rec["action_id"], cost, rec["task_ref"])
        result = self._result(rec, ActionOutcome(status="canceled", feedback_zh=reason), cost)
        rec["result"] = result
        self._record_step(rec, "canceled")
        agent.on_action_result(self, result)

    def send_message(self, sender, receivers, kind, payload, size_bytes, task_ref=None, ttl_s=None, correlation_id=None):
        return self.network.send(sender, receivers, kind, payload, size_bytes, task_ref, ttl_s, correlation_id)

    def _expire_tasks(self):
        for agent in self.agents.values():
            for task in list(agent.tasks.values()):
                if task.get("status") not in ("proposed", "assigned", "received"):
                    continue
                deadline = task.get("deadline_s")
                window = task.get("commitment_window")
                lease = window.get("end_s") if isinstance(window, dict) else None
                limit = min([float(x) for x in (deadline, lease) if isinstance(x, (int, float))], default=math.inf)
                if self.t >= limit:
                    before = task["status"]
                    task["status"] = "expired"
                    task.setdefault("history", []).append(dict(t=self.t, status="expired", note_zh="本地任务副本超过期限。"))
                    self.emit(layer="task", family="sm:task.status", kind="task.status", refs=["F201", "KERNEL", task["type"]],
                              subject=task["id"], field="task.status", before=before, after="expired",
                              actor=agent.id, task=task["id"], task_rev=task["rev"], visible_to=[agent.id],
                              text_zh="本地任务副本已过期；其他持有者须通过消息得知。")

    def _secondary(self, spec):
        from ..world.truth import apply_secondary
        changes = apply_secondary(self.world, spec, self.t)
        for eid, key, before, after in changes:
            self.emit(layer="world", family="hazard.secondary", kind=key, refs=["F023", "WORLD"],
                      subject=eid, field=key, before=before, after=after,
                      backend_mode="stub", scripted=True, visible_to=["*truth*"],
                      text_zh="作者设定的次生事件改变了世界真值。")

    def _stale_version(self, disruption):
        """OV08's explicit protocol-version fault; never mutate a task holder."""
        import json
        agent = self.agents.get(disruption.get("target"))
        coordinator = next((a for a in self.agents.values() if a.kind == "coordinator"), None)
        if not agent or not coordinator:
            return
        task = agent.tasks.get(agent.current) if agent.current else None
        if task is None:
            task = next((t for t in agent.tasks.values() if t.get("rev", 0) > 1), None)
        if task is None or task.get("rev", 0) < 1:
            return
        stale = deepcopy(task)
        stale["rev"] = max(0, int(task["rev"]) - int(disruption.get("params", {}).get("revision_lag", 1)))
        payload = dict(op="assignment", task=stale, coordinator=coordinator.id,
                       fault_model="authored_stale_version_v1")
        self.emit(layer="comms", family="disruption.injected", kind="stale_version", grounding="proposed",
                  refs=["OV08", "F201"], subject=agent.id, after=dict(task=task["id"], sent_rev=stale["rev"]),
                  visible_to=["*truth*"], scripted=True, backend_mode="stub",
                  text_zh="显式注入旧版本标签消息；是否隔离由收件智能体判断。")
        self.send_message(coordinator.id, [agent.id], "assignment", payload,
                          len(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
                          task_ref=(task["id"], stale["rev"]), ttl_s=180)

    def inject(self, disruption):
        disruption = deepcopy(disruption)
        kind = disruption.get("kind")
        valid = {"aftershock_reblock", "link_outage", "low_battery", "service_failure", "target_moves",
                 "equipment_fault", "wrong_evidence", "fire_spread", "water_rise", "cancel_task"}
        if kind not in valid:
            raise ValueError("未知扰动 / unknown disruption")
        target, params = disruption.get("target"), disruption.get("params") or {}
        self.emit(layer="world", family="disruption.injected", kind=kind, grounding="proposed",
                  refs=["F023"], subject=target or "world", after=disruption,
                  visible_to=["*truth*"], backend_mode="stub", scripted=True,
                  text_zh=f"注入显式作者扰动：{kind}。")
        if kind == "link_outage":
            self.network.inject(disruption)
        elif kind == "service_failure":
            self.compute.inject(disruption)
        elif kind in ("aftershock_reblock", "target_moves", "fire_spread", "water_rise"):
            self._secondary(dict(kind=kind, target=target, params=params))
        elif kind == "low_battery":
            candidates = [self.agents[target]] if target in self.agents else ([a for a in self.agents.values() if a.spec.get("mobility") != "none"] if target is None else [])
            for agent in candidates:
                maximum = float(agent.spec.get("battery_Wh", 100))
                value = max(0, float(params.get("remaining_Wh", params.get("battery_Wh", maximum * .08))))
                self.write_truth(agent.platform_eid, "platform.battery_remaining_Wh", value, "WORLD", actor=agent.id, visible_to=[agent.id])
        elif kind == "equipment_fault":
            candidates = [self.agents[target]] if target in self.agents else ([a for a in self.agents.values() if a.spec.get("mobility") != "none"] if target is None else [])[:1]
            for agent in candidates:
                self.write_truth(agent.platform_eid, "platform.health_state", "degraded", "WORLD", visible_to=[agent.id])
                self.write_truth(agent.platform_eid, "equipment.health", params.get("health", "failed"), "WORLD", visible_to=[agent.id])
                self.write_truth(agent.platform_eid, "attr.fault_kind", params.get("equipment_kind", "platform"), "WORLD", visible_to=[agent.id])
                self.write_truth(agent.platform_eid, "attr.fault_recoverable", bool(params.get("recoverable", True)), "WORLD", visible_to=[agent.id])
        elif kind == "wrong_evidence":
            from ..contracts import Observation
            candidates = [self.agents[target]] if target in self.agents else [a for a in self.agents.values() if a.kind == "coordinator"][:1]
            for agent in candidates:
                key = params.get("key", params.get("field", "building.visual_damage"))
                known = agent.belief.entities()
                candidates = [eid for eid, fields in known.items() if fields.get("entity.kind", {}).get("value") == params.get("target_class", "building")]
                observed = [eid for eid in candidates if key in known[eid] and known[eid][key].get("source") != "prior"]
                eid = params.get("entity") or next(iter(observed or candidates), None)
                if eid is None:
                    continue
                value = params.get("value", "destroyed")
                prior = agent.belief.get(eid, key)
                if params.get("ensure_conflict") and prior and prior.get("value") == value:
                    value = "none" if value != "none" else "destroyed"
                obs = Observation(id=f"injected-{len(self.events)}", by="authored_injection", t=self.t,
                                  about=[eid], kind="authored_sensing", values={key: value},
                                  confidence=float(params.get("confidence", .8)), model="scripted_fault_v1")
                # Deliberately false evidence is still delivered and integrated through A041, never truth.
                self.submit_action(agent.id, "A041", {"observations": [obs.to_dict()]})
        elif kind == "cancel_task":
            aid = params.get("agent", disruption.get("agent", target))
            agent = self.agents.get(aid) if aid else next((a for a in self.agents.values() if a.current), None)
            if agent:
                tid = params.get("task", disruption.get("task", agent.current))
                if tid in agent.tasks:
                    task = agent.tasks[tid]
                    self.submit_action(agent.id, "A055", {"task_id": tid, "target": tid, "reason": "运行时取消请求"}, (tid, task["rev"]))
