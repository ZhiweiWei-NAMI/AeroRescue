"""Local, catalog-driven scripted agents; no policy has access to world truth.

Only build_agents/initial-prior construction crosses the initialization boundary.
At runtime the sole world reads below are this agent's own platform telemetry.
Action parameters are JSON values; task actions implement SPEC §6.7 directly.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
import math

from ..contracts import ActionOutcome, Message, validate
from .embodiment import EMBODIMENTS
from .knowledge import BeliefStore, camera_record_reference, prior_from_world, _record as _evidence_record
from .allocation import (GreedyCapabilityAccessAllocator, BiddingAllocator,
                         resolve_method, required_capabilities, belief_path, belief_position,
                         obstacle_approach_path, hazard_approach_path)
from .taskgen import candidates, is_meta_task, _Context, _TaskIndex, LINK_WEAK, UNKNOWN
from ..sim.actions import known_navigation_goal
from ..world.truth import BUILDING_NAVIGATION_FIELDS

TERMINAL = {"completed", "failed", "canceled", "expired"}
ACTIVE = {"proposed", "assigned", "received", "accepted", "executing", "blocked", "suspended"}
# Personnel care/load/transport/handover/water-recovery work may still proceed
# while a person is aboard; other work waits for the aboard person's handover.
PERSON_CARE_TASKS = {"T029", "T030", "T031", "T032", "T033", "T034", "T035", "T036"}
COST_KEYS = ("time_s", "energy_Wh", "bytes", "cpu_s")
# Mirrors the frozen action model's platform-occupancy boundary, not task type.
PLATFORM_ACTIONS = {"A001", "A002", "A003", "A004", "A021", "A062", "A063"} | {
    f"A{i:03d}" for i in range(8, 32)}
TELEMETRY = {
    "pose": "entity.pose", "battery_Wh": "platform.battery_remaining_Wh",
    "battery_max_Wh": "platform.battery_capacity_Wh",
    "motion_state": "platform.motion_state", "health": "platform.health_state",
    "payload": "platform.payload_item_ids", "reserve_Wh": "platform.reserve_Wh",
    "payload_used_kg": "platform.payload_used_kg", "return_site_id": "platform.return_site_id",
}
# These map observed feedback, not world conditions, onto catalog branches.
RECOVERY = {
    "blocked": "Q098", "energy": "Q102", "lost_link": "Q104",
    "timeout": "Q103", "memory": "Q106",
    "service": "Q107", "stale": "Q108", "conflict": "Q112",
    "partial_search": "Q110", "actuator": "Q111", "receipt": "Q111",
    "uncertain": "Q097", "no_capability": "Q100", "payload": "Q101",
}


def _dict(value):
    return value.to_dict() if hasattr(value, "to_dict") else dict(value)


def _failure_reason(code):
    if code in {"capability", "resource", "access", "network", "compute", "hazard", "stale", "canceled", "deadline"}:
        return code
    return {"no_capability": "capability", "incompatible": "capability", "unavailable": "capability",
            "busy": "resource", "energy": "resource", "insufficient": "resource", "payload": "resource",
            "blocked": "access", "lost_link": "network", "timeout": "network", "receipt": "access",
            "model": "compute", "service": "compute", "memory": "compute", "sensor": "resource",
            "actuator": "resource", "uncertain": "resource"}.get(code, "resource")


def _result_cause(result):
    """Rejections have a rejection_code and legitimately no failure_code."""
    for key in ("rejection_code", "failure_code"):
        code = result.get(key)
        if code and code != "none":
            return code
    return "none" if result.get("status") == "succeeded" else "model"


def may_resume_preempted(task, agent_id, current, resume_stack, tasks=None):
    """Whether a task reaching a terminal state may release the preemption stack.

    A preempted task may resume only when this agent really holds no accepted or
    executing task. A refused or failed replacement must not resume the
    suspended task while the preempting task is still executing, which would
    orphan that task in ``executing`` and keep its executor reserved on every
    peer's books.

    ``current`` alone is not ownership: after A056 suspends the original task
    the pointer still names it, so a replacement refused in that window used to
    leave the original suspended forever. When the holder's ``tasks`` table is
    supplied the check uses real ownership/status instead of the pointer; the
    pointer-only form remains only for callers that keep no table.
    """
    if not (resume_stack and not is_meta_task(task) and task.get("assigned_to") == agent_id):
        return False
    if tasks is None:
        return current is None
    return not any(other.get("assigned_to") == agent_id
                   and other.get("status") in {"accepted", "executing"}
                   and not is_meta_task(other)
                   for tid, other in tasks.items() if tid != task.get("id"))


def _value(store, eid, key, default=None):
    entry = store.get(eid, key)
    return entry.get("value", default) if entry else default


def _point(value):
    if isinstance(value, dict):
        value = value.get("position", value.get("xy", value.get("coords", [0, 0])))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [float(value[0]), float(value[1])]
    return [0.0, 0.0]


class AgentRuntime:
    """A single holder's tasks, belief, commitments and bounded method executor."""

    def __init__(self, spec, id, start, catalog):
        self.spec = deepcopy(spec)
        self.id = str(id)
        self.kind = self.spec["kind"]
        self.platform_eid = self.spec.get("platform_eid", self.id)
        self.catalog = catalog
        self.belief = BeliefStore(self.id)
        self.tasks = {}
        self._task_index = _TaskIndex()
        self._taskgen_history = {}
        self.current = None
        self.queue = []
        self.pending_actions = {}
        self.platform = {
            "pose": _point(start), "battery_Wh": self.spec.get("battery_Wh", 100),
            "battery_max_Wh": self.spec.get("battery_Wh", 100),
            "reserve_Wh": self.spec.get("reserve_Wh", self.spec.get("battery_Wh", 100) * .15),
            "motion_state": "idle", "health": "nominal", "payload": [], "payload_used_kg": 0,
        }
        self.records = {}
        self.local_jobs = {}
        self.received_messages = []
        self.sent_messages = {}
        self.public_objectives = []
        self.roster = {}
        self.coordinator_id = self.spec.get("coordinator_id")
        self.allocator = GreedyCapabilityAccessAllocator()
        self.allocation_decisions = []
        self._allocation_idle = set()
        self._controls = deque()
        self._proposal_pending = set()
        self._counter = 0
        self._started = False
        self._last_generate = -float("inf")
        self._last_report = -float("inf")
        self._last_allocate = -float("inf")
        self._reported_evidence = set()
        self._revisions = {}
        self._assignment_wait = {}
        self._rejected_agents = {}
        self._specialist_wait = {}
        self._specialist_work = deque()
        self._remote_work = deque()
        self._observations = []
        self._camera_candidates = {}
        self._camera_pending = {}
        self._bids = {}
        self._bid_requests = {}
        self._rule_seen = {}
        self._resume_stack = []
        self._remote_history_cursor = {}
        self._received_ids = set()
        self._allocation_state = {}
        self._report_entries = {}
        self._report_evidence = set()
        self._last_report_signature = None
        self._meta_cursor = 0

    @property
    def is_coordinator(self):
        """An explicit mission role does not change physical kind/capabilities."""
        return self.kind == "coordinator" or self.spec.get("mission_coordinator") is True

    def _work_pending(self):
        return any(c["action"] in PLATFORM_ACTIONS or c.get("purpose") in {
                   "method", "approach", "verify", "specialist_travel", "specialist_action",
                   "specialist_verify", "remote_action", "remote_send"} and not is_meta_task(
                   self.tasks.get((c.get("task") or [None])[0], {})) for c in self.pending_actions.values())

    def _coalesce_camera_candidates(self, detections):
        """Reuse an image candidate at the same measured point, never a truth ID."""
        observations, used = [], set()
        for detection in detections:
            values = detection['values']
            about = detection['about']
            if (len(about) != 1 or values.get('observation.association') != 'bbox_ground_projection'
                    or values.get('entity.kind') != 'human'):
                observations.append(detection)
                used.update(about)
                continue
            point = tuple(values['belief.estimated_pose'])
            tracks = self._camera_candidates.setdefault(point, [])
            target = next((eid for eid in tracks if eid not in used
                           and _value(self.belief, eid, 'belief.entity_ref', eid) not in used), about[0])
            if target not in tracks:
                tracks.append(target)
            used.add(target)  # Distinct detections in one frame remain distinct candidates.
            # Only an actual contact/report may have associated this local track
            # to a person; equal image estimates do not establish human identity.
            target = _value(self.belief, target, 'belief.entity_ref', target)
            used.add(target)
            if target == about[0]:
                observations.append(detection)
                continue
            associated = deepcopy(detection)
            associated['id'] += '/associated'
            associated['about'] = [target]
            associated['kind'] = 'authored_inference'
            associated['model'] = 'local_camera_candidate_coalescing'
            associated['values']['observation.association'] = 'same_local_candidate_position'
            associated['values']['observation.parent_evidence_ids'] = list(dict.fromkeys(
                values.get('observation.parent_evidence_ids', []) + [detection['id']]))
            observations.append(associated)
        return observations

    def on_camera_observations(self, sim, image, detections):
        """Return camera evidence to its originating task and A041 path."""
        self.belief.add_evidence(sim, image)
        image_record = image['values']['observation.camera_image']
        # The overview reads only [-1:]; history belongs to the camera report.
        self.records['camera_image'] = [camera_record_reference(image_record)]
        self.belief.put(sim, self.id, 'observation.camera_image', image_record, 1., 'camera_image',
                        evidence_ids=[image['id']], timestamp=image['t'])
        raw_evidence_ids = [row['id'] for row in detections]
        for detection in detections:
            self.belief.add_evidence(sim, detection)
        detections = self._coalesce_camera_candidates(detections)
        for detection in detections:
            self.belief.add_evidence(sim, detection)
        if image_record['action_instance'] is None:
            # Continuous camera evidence is not a task result or a task-scoped
            # integration barrier. Keep the latest repeated claim while an A041
            # batch is in flight; retain distinct claims and all raw evidence.
            for detection in detections:
                if not detection['about']:
                    continue
                claim = {key: value for key, value in detection['values'].items()
                         if not key.startswith('observation.')}
                label = detection['values'].get('observation.image_detection', {}).get('label')
                key = (tuple(detection['about']), detection['kind'], label,
                       json.dumps(claim, sort_keys=True))
                previous = self._camera_pending.get(key)
                if previous is None or detection['t'] >= previous['t']:
                    self._camera_pending[key] = deepcopy(detection)
            return
        instance = sim.instances[image_record['action_instance']]
        instance['camera_observations'] = deepcopy(detections)
        ref = instance['task_ref']
        task = self.tasks.get(ref[0]) if ref else None
        if task:
            waiting = task['progress'].get('pending_camera_analysis', {})
            waiting.pop(image_record['action_instance'], None)
            if not waiting:
                task['progress'].pop('pending_camera_analysis', None)
        if task and task['rev'] != ref[1]:
            return  # Keep the evidence, but isolate it from the replacement task.
        if task:
            task['progress']['latest_camera_observations'] = deepcopy(detections)
            task['result_evidence_ids'] = list(dict.fromkeys(
                task['result_evidence_ids'] + [image['id']] + raw_evidence_ids +
                [row['id'] for row in detections]))
        if instance.get('result') is None:
            # The action result will carry these detections through the normal
            # observation handler, including its task-scoped A041 integration.
            return
        if detections:
            self._observations = deepcopy(detections)
            if task:
                task['progress']['latest_observations'] = deepcopy(detections)
            self._controls.appendleft(('A041', {'observations': deepcopy(detections), 'evidence_ids': []},
                                       task['id'] if task else None, 'integrate', {}))

    def is_busy(self):
        return bool(self._work_pending() or self._specialist_work or self._remote_work or
                    (self.current and self.tasks.get(self.current, {}).get("status")
                     in {"accepted", "executing"}))

    def _emit(self, sim, family, text, *, layer="task", grounding="catalog", refs=(),
              subject=None, task=None, field=None, before=None, after=None):
        return sim.emit(layer=layer, family=family, kind=family, grounding=grounding,
                        refs=list(refs), subject=subject or self.id, field=field,
                        before=deepcopy(before), after=deepcopy(after), actor=self.id,
                        task=task["id"] if task else None,
                        task_rev=task["rev"] if task else None,
                        visible_to=[self.id], text_zh=text)

    def _task_field(self, sim, task, key, before, after, action):
        if before != after:
            self._emit(sim, "field.write", "任务持有者更新本机目录字段", refs=[action],
                       subject=task["id"], task=task, field=key, before=before, after=after)

    def _store_task(self, task):
        self.tasks[task["id"]] = task
        self._task_index.update(task)

    def _status(self, sim, task, status, note, via="A054"):
        before = task.get("status")
        if before == status:
            return
        # Never invent shortcuts when copying a remote holder's history.
        if before is not None and not validate("task.status", before, status):
            return
        task["status"] = status
        self._task_index.update(task)
        task["history"].append({"t": sim.t, "status": status, "note_zh": note})
        self._emit(sim, "sm:task.status", note, refs=["F201", via], subject=task["id"],
                   task=task, field="task.status", before=before, after=status)
        if status in TERMINAL:
            if self.current == task["id"]:
                self.current = None
            self.queue = [tid for tid in self.queue if tid != task["id"]]
            if may_resume_preempted(task, self.id, self.current, self._resume_stack, self.tasks):
                self._resume_preempted(sim)
        if not is_meta_task(task) and task.get("assigned_to") == self.id and self.coordinator_id != self.id:
            self._send_task_update(sim, task)

    def _owned_active(self):
        """Tasks this agent really holds in its physical execution lane.

        The ``current`` pointer is not ownership: A056 leaves it naming the
        suspended original, and a replacement refused in that window would
        otherwise look like an active commitment forever. Control/meta work runs
        in its own lane and neither blocks nor releases a physical preemption.
        """
        return {tid for tid, task in self.tasks.items()
                if task.get("assigned_to") == self.id
                and task.get("status") in {"accepted", "executing"}
                and not is_meta_task(task)}

    def _resume_preempted(self, sim):
        """Bounded return path from a preemption (audit F-A finding 1).

        Resume at most one still-suspended original through the existing A057
        path, and only when this agent really holds no accepted/executing task
        and no other commitment change is already queued. The stack entry is
        consumed either way, so this never becomes a retry loop.
        """
        if not self._resume_stack:
            return False
        if any(control[0] in {"A053", "A056", "A057", "A058"}
               and self.tasks.get(control[2], {}).get("status") not in TERMINAL
               for control in self._controls):
            return False  # a live queued commitment change keeps its own order
        if self._owned_active():
            return False
        while self._resume_stack:
            resume_id = self._resume_stack.pop()
            original = self.tasks.get(resume_id)
            if (not original or original.get("status") != "suspended"
                    or original.get("assigned_to") != self.id):
                continue
            if self._aboard_blocks(original):
                # Keep it suspended until the aboard person's handover.
                self._resume_stack.append(resume_id)
                return False
            if original["progress"].get("recovery"):
                self.current = resume_id
            else:
                self._controls.append(("A057", {"task_id": resume_id}, resume_id, "control", {}))
            return True
        return False

    def _submit(self, sim, action, params, *, task=None, purpose="method", **context):
        params = deepcopy(params)
        if action in {'A017', 'A062'} and 'equipment_id' not in params:
            equipment = self.spec.get('action_equipment', {}).get(action)
            if equipment is None:
                own = self.spec.get('equipment_ids', [])
                known = self.belief.entities(copy=False)
                choices = [eid for eid, fs in known.items() if 'equipment.kind' in fs
                           and (eid in own or any(entry.get('source') in {'observation', 'camera_image', 'message'} for entry in fs.values()))]
                for eid in sorted(choices, key=lambda eid: (eid not in own, eid!=self.platform_eid, eid)):
                    if eid!=self.platform_eid:
                        pose=self._target_point(eid)
                        if pose is None or math.dist(self.platform['pose'],pose)>_value(self.belief,eid,'attr.operator_reach_m',10.):
                            continue
                    if action=='A017' and _value(self.belief,eid,'equipment.kind')=='lift':
                        equipment=eid; break
                    if action=='A062' and _value(self.belief,eid,'equipment.consumable_type') in {'water','foam'}:
                        current=_value(self.belief,equipment,'equipment.consumable_remaining') if equipment else None
                        remaining=_value(self.belief,eid,'equipment.consumable_remaining')
                        if equipment is None or (isinstance(current,(int,float)) and current<=0
                                                 and isinstance(remaining,(int,float)) and remaining>0):
                            equipment=eid
            if equipment is not None:
                params['equipment_id']=equipment
        ref = (task["id"], task["rev"]) if task else None
        execution_ref = ref
        if task:
            params.setdefault("task_id", task["id"])
            params.setdefault("rev", task["rev"])
            # A local protocol's method manages the substantive holder's copy.
            # Its own progress/cost context stays distinct from the action guard.
            managed = self.tasks.get(params["task_id"])
            if managed and managed["id"] != task["id"] and action in {f"A{i:03d}" for i in range(50, 59)}:
                execution_ref = (managed["id"], managed["rev"])
                params["rev"] = managed["rev"]
            params["task_ref"] = list(execution_ref)
        instance = sim.submit_action(self.id, action, params, task_ref=execution_ref)
        self.pending_actions[instance] = {
            "action": action, "params": params, "task": ref,
            "purpose": purpose, **context,
        }
        return instance

    def _send(self, sim, receivers, op, payload=None, *, task=None, kind="task", correlation=None):
        receivers = [r for r in receivers if r]
        if not receivers:
            return None
        payload = {"op": op, **deepcopy(payload or {})}
        require_ack = op not in {"ack", "report", "brief"} or op == "report" and bool(payload.get("entries") or payload.get("evidence"))
        payload.setdefault("require_ack", require_ack)
        size = len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
        mid = sim.send_message(self.id, receivers, kind, payload, size,
                               task_ref=(task["id"], task["rev"]) if task else None,
                               ttl_s=180, correlation_id=correlation)
        self.sent_messages[mid] = {
            "id": mid, "kind": kind, "op": op, "payload": payload,
            "receivers": list(receivers), "sent_s": sim.t, "ttl_s": 180,
            "status": {receiver: "queued" for receiver in receivers},
            "ack_state": "pending" if payload["require_ack"] else "not_required",
            "timeout_s": 30, "retry_count": 0, "task_ref":
                (task["id"], task["rev"]) if task else None,
        }
        return mid

    def _refresh_telemetry(self, sim):
        # F109/F110 local telemetry exception, never another entity or world collection.
        for local_key, catalog_key in TELEMETRY.items():
            value = sim.world.get(self.platform_eid, catalog_key, self.platform.get(local_key))
            # WorldTruth.get already detaches mutable values, including defaults.
            self.platform[local_key] = value
        self.platform["pose"] = _point(self.platform.get("pose"))
        # Short and catalog aliases expose only the same local telemetry.
        for local_key, catalog_key in TELEMETRY.items():
            self.platform[catalog_key] = deepcopy(self.platform[local_key])
        self.platform["health_state"] = self.platform["health"]
        self.platform["battery_remaining_Wh"] = self.platform["battery_Wh"]

    def on_start(self, sim):
        if self._started:
            return
        self._started = True
        self._refresh_telemetry(sim)
        if self.spec.get("navigation_goals"):
            # Public deployment intent, not pre-created tasks or hidden victims.
            self.records["navigation_goals"] = deepcopy(self.spec["navigation_goals"])
            for goal in self.records["navigation_goals"]:
                if self._target_point(goal["target"]) is None:
                    raise ValueError("Deployment target is absent from the public map")
            self.belief.put(sim, self.id, "belief.current_goal_context",
                            deepcopy(self.records["navigation_goals"]), 1, "prior")
        if self.is_coordinator:
            if sim.config.get("allocator") == "bidding":
                self.allocator = BiddingAllocator()
            receivers = [aid for aid in self.roster if aid != self.id]
            self._send(sim, receivers, "brief", {"objectives": self.public_objectives,
                       "brief_zh": self.records.get("public_brief_zh", "按已知地图搜索与评估"),
                       "roster": self.roster}, kind="report")
        self._report(sim)

    def on_tick(self, sim):
        self.on_start(sim)
        self._refresh_telemetry(sim)
        for goal in self.records.get("navigation_goals", []):
            point = self._target_point(goal["target"], navigation=True)
            if point is not None and math.dist(self.platform["pose"], point) <= 2:
                goal["arrived"] = True
        self._expire(sim)
        self._assignment_timeouts(sim)
        self._handoff_timeouts(sim)
        if sim.t - self._last_generate >= 20:
            self._last_generate = sim.t
            self._generate(sim)
        if self.is_coordinator and sim.t - self._last_allocate >= 10:
            self._last_allocate = sim.t
            self._allocate(sim)
        # Management does not occupy the platform. Bound it independently of
        # the one serialized physical/method lane, including during long travel.
        control_pending = sum(c.get("purpose") not in {"method", "approach", "verify",
            "specialist_travel", "specialist_action", "specialist_verify", "remote_action"}
            for c in self.pending_actions.values())
        camera_ready = bool(self._camera_pending) and not any(
            c.get('purpose') == 'camera_integrate' for c in self.pending_actions.values())
        # A perpetually nonempty management queue must not starve periodic
        # evidence. Reserve one of the existing three control slots; this is
        # deliberately not a task-scoped barrier on physical execution.
        control_limit = 3 - int(camera_ready)
        if not self._work_pending():
            # Give a delivered commitment change this boundary before another
            # long approach/survey starts. Preserve A056 -> A053 queue order.
            integrations, changes, other = [], [], []
            for control in self._controls:
                task = self.tasks.get(control[2], {})
                if control[3] == "integrate":
                    integrations.append(control)
                elif control[0] in {"A053", "A056", "A057", "A058"} and not is_meta_task(task):
                    changes.append(control)
                else:
                    other.append(control)
            self._controls = deque(integrations + changes + other)
        if control_pending < control_limit:
            for _ in range(min(control_limit - control_pending, len(self._controls))):
                action, params, tid, purpose, context = self._controls.popleft()
                task = self.tasks.get(tid)
                if task and task["status"] in TERMINAL and purpose not in {"complete", "integrate"}:
                    continue
                if action in {"A053", "A056", "A057", "A058"} and self._work_pending() and not is_meta_task(task or {}):
                    self._controls.append((action, params, tid, purpose, context))
                    continue  # commitment changes wait for a safe action boundary
                if action == "A053" and task and not self._camera_contact_ready(sim, task):
                    # A contact queued before a nearby failed attempt observes
                    # the same cooldown. Wait locally, without failed A053 churn.
                    self._controls.append((action, params, tid, purpose, context))
                    continue
                self._submit(sim, action, params, task=task, purpose=purpose, **context)
                control_pending += 1
        if control_pending < 3 and camera_ready:
            observations = sorted(self._camera_pending.values(), key=lambda row: row['t'])
            self._camera_pending.clear()
            self._submit(sim, 'A041', {'observations': observations, 'evidence_ids': []},
                         purpose='camera_integrate')
        integrating = any(c.get("purpose") == "integrate" for c in self.pending_actions.values()) or any(
            c[3] == "integrate" for c in self._controls)
        changing_commitment = any(c["action"] in {"A053", "A056", "A057", "A058"}
            and not is_meta_task(self.tasks.get((c.get("task") or [None])[0], {}))
            for c in self.pending_actions.values())
        if not integrating and not self._work_pending() and not changing_commitment:
            if self._specialist_work:
                self._execute_specialist(sim)
            elif self._remote_work:
                self._execute_remote(sim)
            else:
                self._execute(sim)
        if not integrating:
            self._execute_meta(sim)
        if sim.t - self._last_report >= 60:
            self._report(sim)

    def _generate(self, sim):
        # Build one read-only context, not 96 copies of growing task histories.
        budget = 3 if self.is_coordinator else 1
        active = {(t["type"], str(t["target"])) for t in self._task_index.rows(ACTIVE)}
        for candidate in candidates(self, sim):
            typ, target = candidate["type"], str(candidate["target"])
            pair = (typ, target)
            if pair in active or pair in self._proposal_pending:
                continue
            refs = candidate["origin"]["belief_refs"]
            signature = []
            for eid, key in refs:
                entry = self.belief.get(eid, key)
                if eid in self.tasks:
                    original = self.tasks[eid]
                    value = (original["rev"], original["status"], original.get("failure_reason"))
                elif eid in self.sent_messages:
                    message = self.sent_messages[eid]
                    value = (message.get("ack_state"), message.get("retry_count", 0))
                elif eid in {self.id, self.platform_eid} and key in TELEMETRY.values():
                    value = self.platform.get(key)
                else:
                    value = entry.get("value") if entry else None
                    if typ == "T047" and key == "link.state" and value in LINK_WEAK:
                        # A fresh failed/degraded probe is a new sample window,
                        # even at the same pose with the same link state. The
                        # existing stale-sample guard still bounds probe cadence.
                        value = (value, entry.get("t"))
                signature.append((str(eid), key, repr(value)))
            # Protocol episodes are keyed to revision, not mutable progress or ACK timestamps.
            original = self.tasks.get(candidate.get("extra", {}).get("original_task_id", target))
            if is_meta_task(typ) and original:
                signature = [(original["id"], original["rev"])]
            signature = tuple(signature)
            if self._rule_seen.get(pair) == signature and typ not in {"T043", "T044"}:
                continue
            self._rule_seen[pair] = signature
            params = {**deepcopy(candidate.get("extra", {})), "type": typ, "task_type": typ,
                      "target": target, "origin": deepcopy(candidate["origin"]),
                      "priority": candidate["priority"]}
            self._proposal_pending.add(pair)
            self._controls.append(("A050", params, None, "propose", {"pair": pair}))
            budget -= 1
            if budget <= 0:
                break

    def _task_from_params(self, sim, params):
        supplied = deepcopy(params.get("task_record") or params.get("task") or {})
        if not isinstance(supplied, dict):
            supplied = {}
        typ = supplied.get("type", params.get("type", params.get("task_type")))
        if typ not in self.catalog.tasks:
            return None
        target = supplied.get("target", params.get("target", self.id))
        managed = self.tasks.get(params.get("original_task_id", target))
        if managed and is_meta_task(managed):
            return None  # no coordination/protocol task about another meta-task
        for existing in self._task_index.rows(ACTIVE, {typ}):
            if existing["type"] == typ and existing["target"] == target and existing["status"] in ACTIVE:
                return existing
        origin = supplied.get("origin", params.get("origin"))
        if not origin or not origin.get("why_zh") or not origin.get("rule_id"):
            return None
        self._counter += 1
        reopened = self.tasks.get(params.get("reopen_task_id"))
        if reopened and (reopened["status"] not in TERMINAL or reopened.get("retry_count", 0) >= 2):
            return None
        tid = reopened["id"] if reopened else supplied.get("id", f"{typ}@{self.id}#{self._counter}")
        task = {
            "id": tid, "type": typ, "name_zh": self.catalog.tasks[typ]["name"],
            "creator": self.id, "holder": self.id, "rev": 0, "status": None,
            "target": target, "goal_predicate": deepcopy(params.get("goal_predicate", {})),
            "required_capabilities": list(self.catalog.tasks[typ].get("capabilities", [])),
            "method": {"seq_id": None, "resolved_steps": [], "step_index": 0},
            "assigned_to": None, "accepted_by": None, "commitment_window": None,
            "interruptibility": params.get("interruptibility", "preemptible"),
            "priority": params.get("priority", 50), "deadline_s": params.get("deadline_s", sim.t + 3600),
            "dependency_ids": list(params.get("dependency_ids", [])), "progress": {},
            "result_evidence_ids": [], "failure_reason": "none", "origin": deepcopy(origin),
            "history": [], "costs": dict.fromkeys(COST_KEYS, 0),
            "parameters": {k: deepcopy(v) for k, v in params.items()
                           if not k.startswith("_") and k not in {
                               "origin", "task", "task_record", "task_id", "task_ref", "rev", "seq_id", "step"}},
        }
        if not task["goal_predicate"]:
            if typ in {"T043", "T044"}:
                task["goal_predicate"] = {"telemetry.battery_at_least": params.get(
                    "target_battery_Wh", self.platform["battery_max_Wh"] * .85)}
            else:
                task["goal_predicate"] = self._default_goal(typ, target)
        if reopened:
            task["creator"] = reopened["creator"]
            task["rev"] = reopened["rev"] + 1
            task["retry_count"] = reopened.get("retry_count", 0) + 1
            task["costs"] = deepcopy(reopened["costs"])
            task["prior_versions"] = deepcopy(reopened.get("prior_versions", [])) + [{
                "rev": reopened["rev"], "status": reopened["status"], "history": deepcopy(reopened["history"])}]
        if params.get("service_scope"):
            task["progress"]["service_scope"] = deepcopy(params["service_scope"])
        self._store_task(task)
        for record_key, catalog_key in {"type": "task.type", "creator": "task.creator_id",
                "target": "task.target_ref", "goal_predicate": "task.goal_predicate",
                "dependency_ids": "task.dependency_ids"}.items():
            self._task_field(sim, task, catalog_key, None, task[record_key], "A050")
        parent = self.tasks.get(params.get("parent_task_id"))
        if parent and parent["id"] != tid:
            parent["progress"].setdefault("generated_tasks", []).append(tid)
        self._status(sim, task, "proposed", f"{self.id} 根据本地证据提出{typ}", "A050")
        self._emit(sim, "taskgen.rule_fired", origin["why_zh"], grounding="proposed",
                   refs=[typ, "A050", "U13", "U14"], subject=tid, task=task, after=origin)
        return task

    @staticmethod
    def _default_goal(typ, target):
        explicit = {
            "T002": ("building.visual_damage", ["none", "minor", "major", "destroyed"]),
            "T014": ("belief.search_status", ["complete", "partial", "inconclusive"]),
            "T015": ("belief.search_status", ["complete", "partial", "inconclusive"]),
            "T022": ("obstacle.state", ["cleared", "moved"]),
            "T023": ("obstacle.state", ["cleared", "moved"]),
            "T032": ("human.mobility", ["limited", "immobile", "mobile"]),
            "T034": ("human.handover_status", ["arrived", "acknowledged"]),
            "T035": ("human.handover_status", ["acknowledged"]),
            "T036": ("human.handover_status", ["acknowledged"]),
            "T041": ("item.state", ["received"]),
        }
        if typ == "T030":
            return {"record.supply_receipt": True}
        if typ == "T085":
            return {"record.handoff_confirmed": True}
        if typ=='T093':
            return {f'hazard.fire_intensity@{target}':0.}
        if typ in explicit:
            key, values = explicit[typ]
            return {f"{key}@{target}": values}
        return {"record.method_evidence": True}

    def _initial_search_ready(self, sim, task):
        """Revalidate a queued first search against delivered completion/damage."""
        if task.get("type") != "T014":
            return True
        if _value(self.belief, str(task["target"]), "belief.search_status") == "complete":
            if not task["progress"].get("stale_search_cancel"):
                task["progress"]["stale_search_cancel"] = True
                self._controls.append(("A054", {"task_id": task["id"], "status": "canceled",
                    "reason": "stale", "note_zh": "已收到该建筑搜索完成报告；撤销重复初搜，保留成本且不新增搜索成功"},
                    task["id"], "complete", {}))
            return False
        priority = self.spec.get("casualty_work_policy", {}).get("damaged_search_priority")
        before = task.get("priority", 0)
        if (task["status"] == "proposed" and priority is not None and priority > before
                and _value(self.belief, str(task["target"]), "building.visual_damage") in {"major", "destroyed"}):
            # Initial proposals may predate the camera damage report. Promote
            # the waiting copy once, without A050/revision or a stale retry delay.
            task["priority"] = task["parameters"]["priority"] = priority
            self._task_field(sim, task, "task.priority", before, priority, "F202")
            self._allocation_state.get(task["id"], {}).pop("next_s", None)
        return True

    def _camera_contact_ready(self, sim, task, *, context=None):
        """Refresh waiting contact policy without reproposal or a new revision."""
        target = str(task.get("target", ""))
        if task.get("type") != "T027" or not target.startswith("rescue-camera/"):
            return True
        c = context if context is not None else _Context(self, sim)
        if not c.camera_candidate(target):
            human = c.v(target, "belief.entity_ref", target)
            if (human != target and c.v(human, "human.needs") not in UNKNOWN
                    and "contact" not in c.needs(human)):
                # Another actual contact may have resolved a queued image track.
                # Retire only this proven alias, never nearby unassociated people.
                if not task["progress"].get("stale_contact_cancel"):
                    task["progress"]["stale_contact_cancel"] = True
                    self._controls.append(("A054", {"task_id": task["id"], "status": "canceled",
                        "reason": "stale", "note_zh": "已收到该图像候选的实际人员关联及已确认需求；撤销重复联系，不新增成功记录"},
                        task["id"], "complete", {}))
                return False
            return True
        if c.camera_contact_retry(target) is None:
            return False
        priority = c.camera_contact_priority(target)
        before = task.get("priority", 0)
        if task["status"] == "proposed" and priority > before:
            task["priority"] = priority
            task["parameters"]["priority"] = priority
            self._task_field(sim, task, "task.priority", before, priority, "F202")
            # A one-time corroboration transition must not wait behind the old
            # rejection backoff while fresh, lower-priority survey work wins.
            self._allocation_state.get(task["id"], {}).pop("next_s", None)
        return True

    def _route_proposal(self, sim, task):
        if task["status"] != "proposed" or self._allocation_state.get(task["id"], {}).get("decided_rev") == task["rev"] or any(c[0] == "A052" and c[2] == task["id"] for c in self._controls):
            return
        local_protocol = is_meta_task(task) or bool(task["progress"].get("service_scope"))
        contact_ready = self._camera_contact_ready(sim, task)
        if task["progress"].get("stale_contact_cancel"):
            return  # The queued A054 owns this obsolete proposal, not A052.
        ok, _ = self._acceptance_check(task)
        active = self.tasks.get(self.current, {}).get("status") in {"accepted", "executing"}
        nonproposal_pending = any(c["action"] not in {"A050", "A041"} for c in self.pending_actions.values())
        own_maintenance = task["type"] in {"T043", "T044", "T048"} and task["creator"] == self.id
        # The existing two-capture priority also applies to a UAV's own proposal.
        # Physical work may still be pending: A053 waits for its real boundary,
        # then A056 checkpoints the lower-priority commitment. Do not race an
        # already queued or delivered replacement assignment.
        own_contact_preemption = (
            active and task["type"] == "T027" and task["priority"] == 81
            and str(task["target"]).startswith("rescue-camera/")
            and not any(c[0] in {"A052", "A053", "A056", "A057", "A058"}
                        and not is_meta_task(self.tasks.get(c[2], {})) for c in self._controls)
            and not any(c["action"] in {"A052", "A053", "A056", "A057", "A058"}
                        and not is_meta_task(self.tasks.get((c.get("task") or [None])[0], {}))
                        for c in self.pending_actions.values())
            and not any(t.get("assigned_to") == self.id
                        for t in self._task_index.rows({"assigned", "received"}, meta=False)))
        if local_protocol or contact_ready and ok and (
                own_maintenance or own_contact_preemption or not active and not nonproposal_pending):
            decision = {"task_id": task["id"], "agent_id": self.id, "decision": "self",
                        "reason_zh": "self：协议归提出者本地处理，A053另行核验能力" if local_protocol else "self：本机能力、可达估计与能源检查通过", "score": 0,
                        "considered": [{"agent": self.id, "ok": ok, "why_zh": "本地归属，非机群分配" if local_protocol else "本机检查通过"}]}
            self._record_decision(sim, task, decision)
            self._controls.append(("A052", {"task_id": task["id"], "agent_id": self.id},
                                   task["id"], "control", {}))
        elif not self.is_coordinator and not task["progress"].get("proposal_sent"):
            task["progress"]["proposal_sent"] = True
            self._send(sim, [self.coordinator_id], "proposal", {"task": task}, task=task)
        # Coordinator proposals remain on its own pending list for its allocator.

    def apply_task_action(self, sim, action_id, params):
        """A050–A058 effects; called only by the kernel's action effect models."""
        params = dict(params or {})
        task_id = params.get("task_id")
        if task_id is None and params.get("task_ref"):
            task_id = params["task_ref"][0]
        task = self.tasks.get(task_id)
        if action_id == "A050":
            task = self._task_from_params(sim, params)
            if task is None:
                return ActionOutcome(status="failed", failure_code="model", feedback_zh="候选缺少合法类型或本地来源")
            self._route_proposal(sim, task)
        elif action_id == "A051":
            candidate = params.get("task") or task
            if not isinstance(candidate, dict):
                return ActionOutcome(status="failed", failure_code="model", feedback_zh="出价缺少已收任务")
            ok, reason = self._acceptance_check(candidate)
            self._send(sim, [params.get("requester", self.coordinator_id)], "bid",
                       {"task_id": candidate["id"], "rev": candidate["rev"], "ok": ok,
                        "score": -min(self._distance(candidate), 1000000), "reason_zh": reason,
                        "telemetry": self._telemetry_report()}, task=candidate, kind="bid")
        elif task is None:
            return ActionOutcome(status="failed", failure_code="model", feedback_zh="本机没有该任务副本")
        elif params.get("rev", task["rev"]) != task["rev"]:
            return ActionOutcome(status="failed", failure_code="model", feedback_zh="忽略旧版任务管理请求")
        elif action_id == "A052":
            assignee = params.get("agent_id", params.get("assigned_to", params.get("assignee")))
            if not assignee or is_meta_task(task) and assignee != self.id:
                return ActionOutcome(status="failed", failure_code="model", feedback_zh="协调协议只可由提出者本地执行")
            if not self.is_coordinator and assignee != self.id:
                return ActionOutcome(status="failed", failure_code="model", feedback_zh="非协调者不可代替其他智能体分配")
            if task.get("assigned_to") == assignee and task["status"] in {"assigned", "received", "accepted", "executing"}:
                return ActionOutcome(status="succeeded", feedback_zh="同版本分配已记录，不产生重复承诺")
            if task["status"] not in {"proposed", "assigned"}:
                return ActionOutcome(status="failed", failure_code="model", feedback_zh="该版本已存在承诺，须先交接或撤销")
            old_assignee = task.get("assigned_to")
            previous_rev = task["rev"]
            task["rev"] += 1
            task["assigned_to"] = assignee
            self._task_field(sim, task, "task.revision", previous_rev, task["rev"], "A052")
            self._task_field(sim, task, "task.assigned_to", old_assignee, assignee, "A052")
            task["accepted_by"] = None
            task["assignment"] = {"delivery_state": "queued", "decision": "unknown", "retry_count": 0}
            self._revisions[task["id"]] = task["rev"]
            self._status(sim, task, "assigned", f"{self.id} 提交任务分配，等待收件", "A052")
            if old_assignee and old_assignee != assignee:
                self._send(sim, [old_assignee], "cancel", {"task_id": task["id"],
                           "rev": task["rev"], "reason": "新版改派，撤销旧承诺"}, task=task)
            if assignee == self.id:
                self._status(sim, task, "received", "本机收到自分配任务", "A065")
                self._controls.append(("A053", {"task_id": task["id"]}, task["id"], "control", {}))
            else:
                self._assignment_wait[task["id"]] = {"sent_s": None, "mid": None,
                                                    "retries": 0, "rev": task["rev"]}
                self._controls.appendleft(("A036", {"task_id": task["id"],
                    "receivers": [assignee], "kind": "assignment",
                    "payload": {"op": "assignment", "task": deepcopy(task), "coordinator": self.id,
                                "entries": self._target_entries(task["target"]), "evidence": self._target_evidence(task["target"])}},
                    task["id"], "assignment_send", {}))
        elif action_id == "A053":
            if task.get("accepted_by") == self.id and task["status"] in {"accepted", "executing"}:
                ok, reason = self._acceptance_check(task)
                return ActionOutcome(status="succeeded" if ok else "failed",
                                     failure_code="none" if ok else "energy" if reason == "energy" else "blocked",
                                     feedback_zh="重新核验现有本机承诺" if ok else "现有承诺不再满足本机条件")
            if task["status"] != "received":
                return ActionOutcome(status="failed", failure_code="model", feedback_zh="只有本机已收到的版本可决定接受")
            self._refresh_telemetry(sim)
            ok, reason = self._acceptance_check(task)
            previous = self.tasks.get(self.current, {})
            if ok and not is_meta_task(task) and previous.get("id") != task["id"] and previous.get("status") == "executing":
                # A053 has reached a safe action boundary. Suspend via the real
                # A056 model before establishing the replacement commitment.
                if previous["id"] not in self._resume_stack:
                    self._resume_stack.append(previous["id"])
                self._controls.appendleft(("A053", {"task_id": task["id"]}, task["id"], "control", {}))
                self._controls.appendleft(("A056", {"task_id": previous["id"]}, previous["id"], "control", {}))
                return ActionOutcome(status="partial", feedback_zh="等待安全挂起原承诺后再决定接受")
            task.setdefault("assignment", {})["decision"] = "accepted" if ok else "rejected"
            task["assignment"]["decision_reason"] = reason
            if ok:
                if not is_meta_task(task) and previous.get("id") != task["id"] and previous.get("status") == "suspended":
                    if previous["id"] not in self._resume_stack:
                        self._resume_stack.append(previous["id"])
                    rec = previous["progress"].get("recovery")
                    if rec and rec["seq_id"] == "Q102" and task["type"] == "T043":
                        rec["step_index"] = rec["resolved_steps"].index("A057")
                    self.current = None
                before_accepted = task.get("accepted_by")
                before_window = task.get("commitment_window")
                task["accepted_by"] = self.id
                task["commitment_window"] = [sim.t, task.get("deadline_s")]
                self._task_field(sim, task, "task.accepted_by", before_accepted, self.id, "A053")
                self._task_field(sim, task, "task.commitment_window", before_window, task["commitment_window"], "A053")
                self._select_method(sim, task)
                self._status(sim, task, "accepted", f"{self.id} 接受任务并登记承诺", "A053")
                if task["id"] not in self.queue:
                    self.queue.append(task["id"])
            else:
                task["failure_reason"] = _failure_reason(reason)
                self._status(sim, task, "failed", f"{self.id} 拒绝任务：{reason}", "A053")
                if self.kind == "robot_dog" and reason == "capability" and task["type"] == "T022":
                    self._branch_event(sim, task, "Q100", "机器狗没有额定清障工具，真实拒绝而非修改残骸状态")
            handoff_from = task["progress"].get("handoff_wait_release")
            if handoff_from:
                self._send(sim, [handoff_from], "handoff_ready", {"task": task, "accepted": ok}, task=task)
            if not is_meta_task(task) and self.coordinator_id and self.coordinator_id != self.id and not handoff_from:
                self._send(sim, [self.coordinator_id], "decision", {"task_id": task["id"],
                           "rev": task["rev"], "accepted": ok, "reason_zh": reason,
                           "task": task, "telemetry": self._telemetry_report()}, task=task)
        elif action_id == "A054":
            progress = params.get("progress")
            if isinstance(progress, dict):
                task["progress"].update(deepcopy(progress))
            desired = params.get("status")
            if desired == "completed" or params.get("complete"):
                if not self._goal_satisfied(task):
                    return ActionOutcome(status="partial", feedback_zh="本地证据尚不足以确认目标")
                self._status(sim, task, "completed", "持有者本地证据满足任务目标")
            elif desired in {"blocked", "failed", "canceled"}:
                task["failure_reason"] = _failure_reason(params.get("reason", task["failure_reason"]))
                self._status(sim, task, desired, params.get("note_zh", "按实际反馈更新任务状态"))
        elif action_id == "A055":
            for instance, context in list(self.pending_actions.items()):
                if context.get("task") and context["task"][0] == task["id"] and context["action"] != "A055":
                    sim.cancel_action(instance, params.get("reason", "安全抢占"))
            # Q102 requests safe cancellation, then A056 suspends; do not cancel its record.
            if not params.get("suspend") and not params.get("safe_stop_only"):
                self._status(sim, task, "canceled", "任务取消，保留已耗资源", "A055")
        elif action_id == "A056":
            if task["status"] == "executing" and task["interruptibility"] != "non_interruptible":
                task["progress"]["resume_checkpoint"] = deepcopy(task["method"])
                self._status(sim, task, "suspended", "任务安全挂起并保存检查点", "A056")
            else:
                return ActionOutcome(status="failed", failure_code="model", feedback_zh="当前任务不可挂起")
        elif action_id == "A057":
            ok, reason = self._acceptance_check(task, resuming=True)
            if task["status"] not in {"suspended", "blocked"} or not ok:
                return ActionOutcome(status="failed", failure_code="energy" if reason == "energy" else "blocked",
                                     feedback_zh="重新检查后尚不能恢复任务")
            self._status(sim, task, "executing", "重新检查资源与本地图后恢复任务", "A057")
            self.current = task["id"]
        elif action_id == "A058":
            receiver = params.get("agent_id", params.get("handoff_target_id"))
            if not receiver or receiver == self.id:
                return ActionOutcome(status="failed", failure_code="model", feedback_zh="交接缺少独立接收方")
            task["progress"]["handoff_pending"] = {"receiver": receiver, "rev": task["rev"] + 1}
            if task["status"] == "executing":
                self._controls.appendleft(("A056", {"task_id": task["id"]}, task["id"],
                                           "handoff_suspend", {"receiver": receiver}))
            elif task["status"] in {"suspended", "blocked"}:
                self._offer_handoff(sim, task, receiver)
            else:
                return ActionOutcome(status="failed", failure_code="blocked", feedback_zh="没有可交接的活动承诺")
        else:
            return ActionOutcome(status="failed", failure_code="model", feedback_zh="未知任务管理动作")
        return ActionOutcome(status="succeeded", feedback_zh="本机任务管理记录已更新",
                             evidence_ids=[f"record:{self.id}:{task['id']}:{task['rev']}:{action_id}"] if task else [])

    def _navigation_target(self, task):
        params = task.get("parameters", {})
        domain_target = task["target"]
        original = self.tasks.get(params.get("original_task_id", domain_target))
        if is_meta_task(task) and original and not is_meta_task(original):
            domain_target = original["target"]
        target = params.get("destination_id", params.get("destination", domain_target))
        if task['type']=='T094':
            stations=[eid for eid, fs in self.belief.entities(copy=False).items()
                      if _value(self.belief,eid,'station.kind') in {'supply','logistics','mixed'}]
            if stations:
                target=min(stations,key=lambda eid: math.dist(self.platform['pose'],self._target_point(eid)))
        if task["type"] in {"T030", "T039", "T040"}:
            target = params.get("receiver_id", _value(self.belief, str(task["target"]), "item.receiver_id", target))
        if task["type"] in {"T034", "T035", "T036"}:
            target = params.get("destination_id") or params.get("destination") or _value(
                self.belief, str(task["target"]), "human.destination_id")
            if not target:
                stations = [eid for eid, fs in self.belief.entities(copy=False).items()
                            if "station.kind" in fs]
                if stations:
                    target = min(stations, key=lambda eid: math.dist(self.platform["pose"], self._target_point(eid)))
        if task["type"] in {"T043", "T044"}:
            target = params.get("station_id")
            if target is None:
                stations = [eid for eid, fs in self.belief.entities(copy=False).items() if
                    fs.get("station.kind", {}).get("value") in {"charging", "charge", "supply", "charging_station"}
                    or isinstance(fs.get("station.power_W", {}).get("value"), (int, float))
                    and fs["station.power_W"]["value"] > 0]
                if stations:
                    target = min(stations, key=lambda e: math.dist(self.platform["pose"], self._target_point(e)))
        point = self._target_point(target, navigation=True)
        return (target if isinstance(target, str) else None), point

    def _distance(self, task):
        _, point = self._navigation_target(task)
        return math.dist(_point(self.platform["pose"]), point) if point is not None else float("inf")

    def _target_point(self, target, *, navigation=False):
        if isinstance(target, (list, tuple)):
            point = _point(target)
        elif isinstance(target, dict) and any(k in target for k in ("position", "xy", "coords")):
            point = _point(target)
        else:
            if isinstance(target, dict):
                target = target.get("id", target.get("entity_id", ""))
            point = belief_position(self.belief, target,
                                    own_ids={self.id, self.platform_eid}, own_pose=self.platform["pose"])
        return known_navigation_goal(self, point) if navigation else point

    def _search_route(self, task):
        """Sweep a public polygon, rather than treating its centroid as coverage."""
        spacing = self.spec.get("search_lane_spacing_m")
        if (not spacing or self.spec["mobility"] != "air" or task["type"] not in {"T001", "T009"}
                or task["progress"].get("recovery")):
            return None
        regions = self.spec.get("search_region_ids")
        if regions is not None and task["target"] not in regions:
            return None
        geometry = _value(self.belief, str(task["target"]), "entity.geometry", {})
        if geometry.get("type") != "polygon":
            return None
        polygon = geometry["coords"]
        # Parallel scanlines are clipped to the actual mapped polygon.  Only
        # their physical flight is requested; no search/detection result is set.
        low = min(p[1] for p in polygon)
        high = max(p[1] for p in polygon)
        count = max(1, math.ceil((high - low) / spacing))
        rows = []
        # An explicitly requested follow-up pass samples different clipped
        # lanes. The absent/zero parameter preserves the original sweep exactly.
        pass_index = max(0, int(task.get("parameters", {}).get("search_pass_index", 0)))
        lane_offset = (.5 + pass_index * .6180339887498949) % 1 if pass_index else .5
        for index in range(count):
            y = low + (index + lane_offset) * (high - low) / count
            intersections = []
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                if (a[1] > y) != (b[1] > y):
                    intersections.append(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
            intersections.sort()
            spans = []
            for a, b in zip(intersections[::2], intersections[1::2]):
                inset = min(1., (b - a) / 4)
                spans.extend([[a + inset, y], [b - inset, y]])
            if spans:
                rows.append(spans)
        if not rows:
            return None
        options = []
        for ordered in (rows, rows[::-1]):
            for reverse in (False, True):
                points = []
                for index, row in enumerate(ordered):
                    points.extend(row[::-1] if bool(index % 2) != reverse else row)
                options.append(points)
        start = _point(self.platform["pose"])
        points = min(options, key=lambda route: math.dist(start, route[0]))
        return [list(start)] + points

    def _obstacle_approach_params(self, target):
        route = obstacle_approach_path(self.belief, self.platform["pose"], target, self.spec)
        site = route["waypoints"][-1] if route["ok"] else self._target_point(target)
        params = {"goal": site, "goal_xy": site, "destination": site,
                  "approach_obstacle_id": target}
        if route["ok"]:
            # belief_path includes endpoint attachments; omit coincident pairs
            # before the existing supplied-path road matching in actions.py.
            points = [route["waypoints"][0]]
            for point in route["waypoints"][1:]:
                if math.dist(points[-1], point) > 1e-6:
                    points.append(point)
            if len(points) > 1:
                params["path"] = points
        return params

    def _approach_casualty(self, sim, task, action, params, *, work=None):
        if action not in {"A018", "A019"}:
            return False
        target = params.get("target", task["target"])
        site = self._target_point(target)
        if site is None or math.dist(self.platform["pose"], site) <= 15:
            return False
        movement = {"air": "A001", "water": "A003"}.get(self.spec["mobility"], "A002")
        context = {"work": work, "resume_stage": work["stage"]} if work is not None else {}
        self._submit(sim, movement, {"target": target, "goal": site, "goal_xy": site}, task=task,
                     purpose="specialist_travel" if work is not None else "approach", **context)
        return True

    def _aboard_blocks(self, task):
        """A carrier with an unacknowledged person aboard finishes that handover
        before unrelated work, as the A059 service path already requires."""
        if is_meta_task(task) or task["type"] in PERSON_CARE_TASKS:
            return False
        aboard = [eid for eid in self.belief.entity_ids("human")
                  if _value(self.belief, eid, "human.carrier_id") in {self.id, self.platform_eid}
                  and _value(self.belief, eid, "human.handover_status") != "acknowledged"]
        return bool(aboard) and str(task.get("target")) not in aboard

    def _acceptance_check(self, task, resuming=False):
        scope = task.get("progress", {}).get("service_scope") or task.get("parameters", {}).get("service_scope")
        if scope:
            action = scope["action_id"]
            needed = set(self.catalog.actions[action].get("capabilities", [])) | {"C02", "C43"}
            if not needed.issubset(self.spec["capabilities"]):
                return False, "capability"
            if not self.spec.get("available", True) or self.platform.get("health") == "failed":
                return False, "unavailable"
            route = (obstacle_approach_path(self.belief, self.platform["pose"], task["target"], self.spec)
                     if action in {"A012", "A013", "A016"} else
                     belief_path(self.belief, self.platform["pose"], self._target_point(task["target"]), self.spec))
            if not route["ok"] and self._distance(task) > 25:
                return False, "access"
        else:
            if task["type"] in {"T034", "T035"}:
                carrier = task.get("parameters", {}).get("carrier_id") or _value(self.belief, str(task["target"]), "human.carrier_id")
                if carrier and carrier not in {self.id, self.platform_eid}:
                    return False, "capability"
            ok, _, method = self.allocator.capability_filter(self.catalog, task, self.spec)
            if not ok:
                return False, "capability"
            row = self._telemetry_report()
            entities = self.belief.entities(copy=False)
            ok, _ = self.allocator.payload_filter(task, row, self.spec, method, entities,
                                                 identities={self.id, self.platform_eid})
            if not ok or self._aboard_blocks(task):
                return False, "payload"
            if not self.spec.get("available", True) or self.platform.get("health") == "failed":
                return False, "unavailable"
        # Keep ordinary checks and refusal precedence unchanged. A service scope
        # must also check the current commitment before it can return success.
        current = self.tasks.get(self.current, {})
        if not is_meta_task(task) and current and current.get("id") != task["id"] and current.get("status") in {"accepted", "executing"}:
            if current.get("interruptibility") == "non_interruptible":
                return False, "busy"
            if task.get("priority", 0) <= current.get("priority", 0) and not resuming:
                return False, "busy"
            if current.get("interruptibility") == "finish_current" and any(
                    c.get("task") and c["task"][0] == current["id"] and c.get("purpose") == "method"
                    for c in self.pending_actions.values()):
                return False, "busy"
        if scope:
            return True, "已收A059授权范围的实际工具与接近条件通过"
        physical = any(a in {"A001", "A002", "A003", "A008"} for a in method["resolved_steps"])
        estimate = max(.2, self._distance(task) * .001) if physical else .2
        battery = self.platform.get("battery_Wh", 0)
        reserve = self.platform.get("reserve_Wh", 0)
        if not self.spec.get("mains_power"):
            if task["type"] not in {"T043", "T044"} and battery < reserve + estimate:
                return False, "energy"
            if task["type"] in {"T043", "T044"} and battery < estimate:
                return False, "energy"
        ok, _, _ = self.allocator.access_filter(self, task, row, self.spec, method, entities=entities)
        if not ok:
            return False, "access"
        # Believed access only. Unknown access does not become a truth query.
        access = _value(self.belief, str(task["target"]), "belief.estimated_access")
        if access in (False, "blocked", "unsafe", "unreachable") and task["type"] not in {
                "T021", "T022", "T023", "T024", "T026", "T087"}:
            return False, "access"
        return True, "本机能力、能源与承诺检查通过"

    def fleet_view(self):
        view = {}
        for eid, fields in self.belief.entities(copy=False).items():
            entry = fields.get("belief.team_estimates")
            if entry and isinstance(entry["value"], dict):
                report = deepcopy(entry["value"])
                report.setdefault("id", eid)
                view[report["id"]] = report
        # A coordinator's own telemetry is local, not a remote peek.
        view[self.id] = self._telemetry_report()
        return view

    def _record_decision(self, sim, task, decision):
        # A rejection is observable evidence, not a permanent assignment lock.
        if decision.get("agent_id"):
            self._allocation_state.setdefault(task["id"], {})["decided_rev"] = task["rev"]
        self.allocation_decisions.append({"t": sim.t, **deepcopy(decision)})
        self._emit(sim, "allocation.decision", decision["reason_zh"], layer="allocation",
                   grounding="proposed", refs=["A052", task["type"]], subject=task["id"],
                   task=task, after=decision)

    def _allocate(self, sim):
        pending_assignments = {c[2] for c in self._controls if c[0] == "A052"}
        pending_assignments.update(c["task"][0] for c in self.pending_actions.values()
                                   if c["action"] == "A052" and c.get("task"))
        waiting = [t for t in self._task_index.rows({"proposed"}, meta=False)
                   if t["id"] not in pending_assignments
                   and self._allocation_state.get(t["id"], {}).get("decided_rev") != t["rev"]]
        if not waiting:
            return
        # Existing proposed pairs are suppressed by taskgen, so corroboration
        # must also reach the waiting backlog. Never revise an assigned task or
        # release a pending A052 reservation just to refresh its priority.
        context, ready = None, []
        for task in waiting:
            if (context is None and task.get("type") == "T027"
                    and str(task.get("target", "")).startswith("rescue-camera/")):
                context = _Context(self, sim)
            if (self._initial_search_ready(sim, task)
                    and self._camera_contact_ready(sim, task, context=context)):
                ready.append(task)
        waiting = ready
        if not waiting:
            return
        fleet = self.fleet_view()
        # Reserve assignments still awaiting delivery/acceptance, including A052
        # controls not yet submitted. A delivered matching current commitment
        # must reach commitment_filter: acceptance is not non_interruptibility.
        reserved = {c[1].get("agent_id") for c in self._controls if c[0] == "A052"}
        reserved.update(c["params"].get("agent_id") for c in self.pending_actions.values()
                        if c["action"] == "A052")
        for committed in self._task_index.rows(
                {"assigned", "received", "accepted", "executing"}, meta=False):
            aid = committed.get("assigned_to")
            reported = fleet.get(aid, {}).get("current_task")
            if (committed["status"] in {"assigned", "received"}
                    or not isinstance(reported, dict)
                    or reported.get("id") != committed["id"]
                    or reported.get("rev") != committed["rev"]):
                reserved.add(aid)
        fleet = {aid: row for aid, row in fleet.items() if aid not in reserved}
        idle = {aid for aid, row in fleet.items() if "current_task" in row
                and row["current_task"] is None and row.get("available") is True}
        newly_idle = idle - self._allocation_idle
        self._allocation_idle = idle
        proposals = []
        for task in waiting:
            state = self._allocation_state.get(task["id"], {})
            # Retry a waiting revision when a reported executor becomes free,
            # before a later, lower-priority proposal can take that executor.
            rejected = state.get("last_rejection", {})
            if sim.t >= state.get("next_s", 0) or newly_idle and rejected.get("rev") == task["rev"]:
                proposals.append(task)
        if not proposals:
            return
        for task in proposals:
            task["excluded_agents"] = sorted(self._rejected_agents.get(task["id"], set()))
        decisions = self.allocator.allocate(self, sim, proposals, fleet)
        for decision in decisions:
            d = _dict(decision)
            task = self.tasks[d["task_id"]]
            state = self._allocation_state.setdefault(task["id"], {})
            if not d.get("agent_id"):
                # Keep actual failed checks visible, without repeating an
                # unchanged rejection or consuming this revision's retry.
                if state.get("last_rejection") != d:
                    self._record_decision(sim, task, d)
                    state["last_rejection"] = deepcopy(d)
                state["next_s"] = sim.t + 60
                continue
            state["decided_rev"] = task["rev"]
            self._record_decision(sim, task, d)
            self._controls.append(("A052", {"task_id": task["id"], "agent_id": d["agent_id"]},
                                   task["id"], "control", {}))

    def _select_method(self, sim, task):
        method = resolve_method(self.catalog, task["type"], self.spec)
        if task['type']=='T032' and self.spec.get('action_equipment',{}).get('A017'):
            # A rope casualty has no observed structure to shore/break. Keep
            # the existing service scope: actual approach, extrication, assessment.
            method.update(resolved_steps=['A002','A017','A018','A054'], details=[])
            task['progress'].setdefault('inapplicable_steps',{})['Q032:2']='绳索人员解困，无已知支护/破拆结构目标'
        if task["progress"].get("handoff_wait_release"):
            method["step_index"] = min(task.get("method", {}).get("step_index", 0), len(method["resolved_steps"]))
        scope = task["progress"].get("service_scope")
        if scope:
            action = scope["action_id"]
            actions = ["A002", action] + (["A018"] if action == "A017" else [])
            method = {"seq_id": method["seq_id"], "resolved_steps": actions, "step_index": 0,
                      "details": [], "scope": "A059-authorized_service",
                      "note_zh": "只执行实际请求的专业服务，接近和结果核验另记成本，不声称执行完整参考路径"}
        task["method"] = deepcopy(method)
        task["required_capabilities"] = required_capabilities(self.catalog, task["type"], self.spec)
        self._emit(sim, "method.selected", "按本机构型选择目录方法及互斥动作", grounding="proposed",
                   refs=[task["type"], method["seq_id"]], subject=task["id"], task=task, after=method)

    def _execute(self, sim):
        if self.current is not None and self.tasks[self.current]['progress'].get('pending_camera_analysis'):
            if self.current not in self.queue:
                self.queue.append(self.current)
            self.current = None
        # Safety net for the bounded preemption return: a suspended original
        # must not wait forever when nothing this agent really owns executes
        # (e.g. its replacement was refused or canceled in the A056 window).
        if self._resume_preempted(sim):
            return
        if self.current is None:
            # A recovery A005 can yield to camera analysis while its task is
            # blocked. Keep that recovery runnable after the evidence arrives.
            self.queue = [tid for tid in self.queue if tid in self.tasks and
                          (self.tasks[tid]["status"] in {"accepted", "executing"} or
                           self.tasks[tid]["status"] == "blocked" and
                           self.tasks[tid]["progress"].get("recovery"))]
            physical_queue = [tid for tid in self.queue if not is_meta_task(self.tasks[tid])
                              and not self.tasks[tid]['progress'].get('pending_camera_analysis')]
            eligible = [tid for tid in physical_queue if all(self.tasks.get(dep, {}).get("status") == "completed"
                        for dep in self.tasks[tid].get("dependency_ids", []))]
            if not eligible:
                return
            self.current = max(eligible, key=lambda tid: self.tasks[tid]["priority"])
            self.queue.remove(self.current)
        task = self.tasks[self.current]
        self._advance_task(sim, task)

    def _execute_meta(self, sim):
        # A protocol may run alongside a physical commitment, never displace it.
        if any(is_meta_task(self.tasks.get((c.get("task") or [None])[0], {}))
               and c.get("purpose") == "method" for c in self.pending_actions.values()):
            return
        tasks = [t for t in self._task_index.rows({"accepted", "executing"}, meta=True)
                 if t.get("assigned_to") == self.id]
        if tasks:
            tasks.sort(key=lambda t: (-t["priority"], t["id"]))
            for task in tasks:
                method = task["progress"].get("recovery") or task["method"]
                index = method["step_index"]
                action = method["resolved_steps"][index] if index < len(method["resolved_steps"]) else None
                if action in PLATFORM_ACTIONS and self._work_pending():
                    continue
                self._advance_task(sim, task)
                break

    def _advance_task(self, sim, task):
        if task['progress'].get('pending_camera_analysis'):
            return
        if any(c.get("task") and c["task"][0] == task["id"] for c in self.pending_actions.values()) or any(
                c[2] == task["id"] and c[3] == "complete" for c in self._controls):
            return
        if task["status"] in TERMINAL:
            if self.current == task["id"]:
                self.current = None
            return
        if task["progress"].get("handoff_wait_release"):
            return  # Acceptance reserves work; the old holder must release it first.
        recovery = task["progress"].get("recovery")
        if recovery and recovery["seq_id"] == "Q102" and task["status"] == "suspended" and (
                recovery["step_index"] < recovery["resolved_steps"].index("A057")):
            return  # the separately proposed T043 owns travel/charging, once
        if task["status"] in {"blocked", "suspended"} and not recovery:
            self.current = None
            return
        if (not recovery and task["method"]["step_index"] == 0
                and not self._initial_search_ready(sim, task)):
            return
        if task["status"] == "accepted":
            self._status(sim, task, "executing", "开始第一个目录动作", "KERNEL")
        if (not recovery and task["type"] in {"T033", "T034", "T035"}
                and task["method"]["step_index"] == 0 and self._goal_satisfied(task)):
            # A delayed independent assignment may arrive after this carrier's
            # own transport/handover. Close it using the received outcome;
            # never reload an acknowledged casualty or replay the trip.
            self._finish_task(sim, task)
            return
        if not recovery and task.get("parameters", {}).get("conditional_sequence") == "Q112" and \
                not task["progress"].get("recovery_history") and self.belief.conflicts():
            self._recover(sim, task, "conflict")
            return
        if not recovery and not self.spec.get("mains_power") and task["type"] not in {"T043", "T044", "T045"} and \
                self.platform["battery_Wh"] <= self.platform["reserve_Wh"]:
            self._recover(sim, task, "energy")
            return
        method = recovery if recovery else task["method"]
        if not recovery and task["type"] == "T030" and self._prepare_supply(sim, task, method):
            return
        index = method["step_index"]
        if index >= len(method["resolved_steps"]):
            if recovery:
                self._finish_recovery(sim, task)
            else:
                self._finish_task(sim, task)
            return
        action = method["resolved_steps"][index]
        if not recovery and action == "A008" and not self._initial_search_ready(sim, task):
            return
        if (not recovery and task["type"] == "T027" and (index == 0 or action == "A009")
                and not self._camera_contact_ready(sim, task)):
            return
        # Independent proposals can arrive after another task already provided
        # the aid/handover. Retire only locally evidenced redundant work, not a
        # failed rescue: do not reload a receiver's person or consume a second kit.
        redundant = None
        if not recovery and task["type"] == "T033" and action == "A020" and _value(
                self.belief, str(task["target"]), "human.handover_status") == "acknowledged":
            redundant = "已收到该人员最终移交确认；撤销过时装载请求，不重复装载或新增成功记录"
        if (not recovery and task["type"] == "T031" and (index == 0 or action == "A019")
                and "treated" in _value(self.belief, str(task["target"]), "human.support_received", [])):
            redundant = "已有该人员实际救护回执；撤销重复救护请求，保留既有结果和耗材成本"
        if redundant:
            self._controls.append(("A054", {"task_id": task["id"], "status": "canceled",
                "reason": "stale", "note_zh": redundant}, task["id"], "complete", {}))
            return
        details = method.get("details", [])
        detail = details[index] if index < len(details) else {}
        params = self._step_params(task, action, method)
        if str(detail.get("actor", "")).startswith("远端"):
            waiting = task["progress"].get("remote_wait")
            if waiting and sim.t - waiting["sent_s"] > 120:
                self._recover(sim, task, "service")
            return  # Only the recipient executes these catalog actor roles.
        if action == "A036" and index + 1 < len(details) and str(details[index + 1].get("actor", "")).startswith("远端"):
            self._send_remote_method(sim, task, method, params)
            return
        if action == "A059" and self.kind != "specialist_team":
            original = detail.get("action_id", detail.get("original_action", detail.get("selected")))
            advance = 1
            if task["type"] == "T032" and index == 0:
                # The catalog support OR targets a structure/obstacle, not the
                # human. With no observed structural prerequisite it is not an
                # applicable service request; never run A015 on a person.
                original, advance = "A017", 3
                task["progress"].setdefault("inapplicable_steps", {})[f"{method['seq_id']}:1"] = (
                    "未收到需支护/破拆的结构目标；请求专业队核验并执行实际解困")
            if not original or original == "A059":
                following = details[index + 1] if index + 1 < len(details) else {}
                if following.get("delegated"):
                    original, advance = following["action_id"], 2
                else:
                    original = self._specialist_action(task)
            self._request_specialist(sim, task, original, params, method, advance=advance)
            return
        # Some catalog service methods start at the work site (A009/A020, etc.).
        # Supply a real, costed approach; it is not a fabricated catalog step.
        onsite = {"A008", "A009", "A011", "A012", "A013", "A014", "A015", "A016",
                  "A017", "A018", "A019", "A020", "A023", "A024", "A025", "A026", "A027", "A062", "A063"}
        if action in onsite and self.spec["mobility"] != "none":
            site = self._target_point(params.get("target", task["target"]), navigation=True)
            approach = {}
            if action in {'A062','A063'}:
                reach=self.spec.get('suppression_reach_m',50.) if action=='A062' else self.spec.get('isolation_reach_m',80.)
                route=hazard_approach_path(self.belief,self.platform['pose'],params['target'],self.spec,reach)
                if not route['ok']:
                    self._fail(sim,task,'access'); return
                site=route['waypoints'][-1]
            if task["type"] in {"T022", "T023", "T026"} and params.get("target") == task["target"]:
                approach = self._obstacle_approach_params(task["target"])
                site = approach["goal_xy"]
            if action == "A024":
                site = self._target_point(_value(self.belief, params.get("target"), "item.holder_id")) or site
            radius = 15 if action in {"A018", "A019"} else 10
            if action in {'A062','A063'}:
                radius=1e-6
            if action == "A020" and self.spec["mobility"] == "wheeled":
                person = self._target_point(params.get("target", task["target"]))
                if site is not None and person is not None and math.dist(site, person) > 1e-6:
                    radius = 2  # Match the movement model's existing arrival tolerance.
            if action == "A011" and task["type"] in {"T022", "T023", "T026"}:
                radius = 100  # Match A011's existing local measurement range.
            if site is not None and math.dist(self.platform["pose"], site) > radius:
                movement = {"air": "A001", "water": "A003"}.get(self.spec["mobility"], "A002")
                self._submit(sim, movement, {"target": params.get("target"), "goal": site, "goal_xy": site,
                                             **approach},
                             task=task, purpose="approach")
                return
        # In a main method, A054 records progress; only the final holder check closes it.
        if action == "A054":
            params["progress"] = {"method_step": index}
        if action == "A055" and (recovery or task["type"] in {"T045", "T083"}):
            params["safe_stop_only"] = True
            params["suspend"] = bool(recovery and recovery["seq_id"] == "Q102")
        self._emit(sim, "method.selected", f"执行目录步骤 {method['seq_id']}:{index + 1}",
                   grounding="proposed", refs=[method["seq_id"], action], task=task,
                   subject=task["id"], after={"seq_id": method["seq_id"],
                                               "step": detail.get("step", index + 1), "action": action})
        self._submit(sim, action, params, task=task, method_seq=method["seq_id"], step=index)

    def _prepare_supply(self, sim, task, method):
        """Resolve stock via a real A028 pickup, never by reading inventory."""
        item = task["parameters"].get("item_id")
        if item:
            if method["step_index"] == 0 and item in self.platform.get("payload", []):
                task["progress"].setdefault("steps", {})[f"{method['seq_id']}:0"] = {
                    "action": "A024", "status": "skipped", "why_zh": "A028已实际补给并挂载，不重复挂载"}
                method["step_index"] = 1
            return False
        if not set(self.catalog.actions["A028"].get("capabilities", [])).issubset(self.spec["capabilities"]):
            self._fail(sim, task, "capability")
            return True
        stations = [eid for eid, fs in self.belief.entities(copy=False).items()
                    if fs.get("station.kind", {}).get("value") in {"supply", "logistics", "mixed"}]
        if not stations:
            self._fail(sim, task, "resource")
            return True
        station = min(stations, key=lambda eid: math.dist(self.platform["pose"], self._target_point(eid)))
        point = self._target_point(station)
        if math.dist(self.platform["pose"], point) > 10:
            movement = {"air": "A001", "water": "A003"}.get(self.spec["mobility"], "A002")
            self._submit(sim, movement, {"target": station, "goal": point, "goal_xy": point}, task=task, purpose="approach")
        else:
            kind = next(iter(task["parameters"].get("need_types", ["water"])), "water")
            self._submit(sim, "A028", {"target": station, "station_id": station, "item_kind": kind},
                         task=task, purpose="supply_pickup", payload_before=list(self.platform.get("payload", [])), item_kind=kind)
        return True

    def _step_params(self, task, action, method):
        params = deepcopy(task.get("parameters", {}))
        destination_id, destination = self._navigation_target(task)
        index = method["step_index"]
        details = method.get("details", [])
        source_step = details[index].get("step", index + 1) if index < len(details) else index + 1
        defaults = {"target": task["target"], "target_id": task["target"],
                    "goal": destination, "destination": destination, "goal_xy": destination,
                    "seq_id": method["seq_id"], "step": source_step,
                    "task_id": task["id"], "rev": task["rev"],
                    "observations": deepcopy(task["progress"].get("latest_observations", self._observations)),
                    "kind": "report", "payload": {"op": "report", "require_ack": True, **self._report_payload()} if action == "A036" else {}}
        for key, value in defaults.items():
            params.setdefault(key, value)
        params.update(task_id=task["id"], rev=task["rev"], seq_id=method["seq_id"], step=source_step)
        if action == "A011":
            roads = _value(self.belief, str(task["target"]), "obstacle.blocks_ids", [])
            if roads:
                # Clearance methods verify the linked road, not the obstacle
                # as though it were a road. The link must come from evidence.
                road = min(roads, key=lambda eid: math.dist(self.platform["pose"], self._target_point(eid)))
                params.update(target=road, target_id=road)
        params["evidence_ids"] = list(dict.fromkeys(list(params.get("evidence_ids", [])) + task["result_evidence_ids"]))
        if action in {"A006", "A041", "A040", "A007", "A039"}:
            # Explicit inputs must not also be expanded from every old evidence id.
            if action == "A006":
                captured = [_dict(self.belief.evidence[eid]) for eid in task["result_evidence_ids"]
                            if eid in self.belief.evidence and _evidence_record(self.belief.evidence[eid]).get("kind") == "authored_sensing"]
                camera = task["progress"].get("latest_camera_observations", [])
                if camera:
                    params["observations"] = captured[-16:] + deepcopy(camera)
                elif captured:
                    params["observations"] = captured[-16:]
            if params.get("observations"):
                params["evidence_ids"] = []
        peer = params.get("peer_id") or (params.get("receiver", params.get("receiver_id"))
                if task["type"] in {f"T{i:03d}" for i in range(47, 57)} else None)
        # receiver_id can denote a human for delivery; only explicit comms peer
        # aliases override report recipients.
        if "receivers" not in params:
            params["receivers"] = [peer] if action in {"A032", "A035", "A036"} and params.get("receiver") else [self.coordinator_id] if self.coordinator_id else []
        if action in {"A001", "A002", "A003", "A021", "A066"}:
            params.update(goal=destination, goal_xy=destination, destination=destination)
            if destination_id:
                params["destination_id"] = destination_id
            if task['type'] == 'T036' and action == 'A003':
                # Q036 approaches the person first; A021 later goes to the receiver.
                pickup = self._target_point(task['target'], navigation=True)
                params.update(goal=pickup, goal_xy=pickup, destination=pickup)
                params.pop('destination_id', None)
            if task["type"] in {"T022", "T023", "T026"}:
                params.update(self._obstacle_approach_params(task["target"]))
            if action == "A001":
                search = self._search_route(task)
                if search:
                    params.update(path=search, goal=search[-1], goal_xy=search[-1], destination=search[-1])
                    params.pop("destination_id", None)
        if task['type']=='T094' and action=='A028':
            params.update(station_id=destination_id, equipment_id=task['target'],
                          item_kind=_value(self.belief,task['target'],'equipment.consumable_type','water'))
            if 'equipment_refill_units' in self.spec:
                params['quantity']=self.spec['equipment_refill_units']
        if action=='A005' and task['type'] in {'T006','T089','T093','T095'} and 'scripted_environment' in self.spec['sensors']:
            params.update(sensor_id='scripted_environment',modality='gas')
        original = self.tasks.get(params.get("original_task_id", task["target"]))
        protocol_progress = original["progress"] if original and is_meta_task(task) else task["progress"]
        if action in {"A035", "A037", "A065"}:
            if action == "A037":
                mid = params.get("message_id", task["progress"].get("last_sent_message_id"))
            elif action == "A065":
                mid = protocol_progress.get("last_received_message_id", params.get("message_id", protocol_progress.get("assignment_message_id")))
            else:
                mid = params.get("message_id")
            if mid:
                params["message_id"] = mid
        if action == "A036" and task["type"] in {"T053", "T054"}:
            original = self.sent_messages.get(params.get("message_id", task["target"]))
            if original:
                params.update(retry_of=original["id"], receivers=list(original["receivers"]),
                              kind=original["kind"], payload=deepcopy(original["payload"]))
        if action in {"A029", "A030"}:
            params["target_battery_Wh"] = self.platform["battery_max_Wh"] * .85
            params["platform_id"] = self.platform_eid
        if task["type"] in {"T043", "T044"} or task["progress"].get("recovery", {}).get("seq_id") == "Q102":
            stations = [eid for eid, fs in self.belief.entities(copy=False).items() if
                        fs.get("station.kind", {}).get("value") in {"charging", "charge", "supply", "charging_station"}]
            if stations:
                station = min(stations, key=lambda e: math.dist(self.platform["pose"], self._target_point(e)))
                params.update(target=station, target_id=station, station_id=station,
                              goal=self._target_point(station), destination=self._target_point(station),
                              goal_xy=self._target_point(station))
        if action in {"A020", "A021", "A022", "A023", "A017", "A018", "A019"}:
            params["human_id"] = task["target"]
        if action in {"A012", "A013", "A016"}:
            params["obstacle_id"] = task["target"]
        if action in {"A024", "A025", "A026", "A027"}:
            params.setdefault("item_id", task["target"])
            params["target"] = params["target_id"] = params["item_id"]
        if action == "A008":
            params["building_id"] = task["target"]
            params.setdefault("budget_s", 300 if task["type"] == "T014" else 450)
        if action == "A050" and (task["type"] in {"T078", "T086"} or method["seq_id"] == "Q112"):
            original = self.tasks.get(params.get("original_task_id"))
            if not original and method["seq_id"] == "Q112":
                original = next((t for t in self.tasks.values() if t["id"] != task["id"]
                    and t["target"] == task["target"] and t["status"] in TERMINAL), None)
            child_type = original["type"] if original else "T011" if method["seq_id"] == "Q112" else "T021"
            child_target = original["target"] if original else task["target"]
            params.update(type=child_type, task_type=child_type, target=child_target,
                          parent_task_id=task["id"],
                          origin={"rule_id": f"R-{child_type}-replan", "belief_refs": deepcopy(task["origin"]["belief_refs"]),
                                  "why_zh": "本地失败或冲突证据要求新版本复核，不沿用旧版完成结论"})
            params.pop("goal_predicate", None)
            if original and original["status"] in TERMINAL:
                params["reopen_task_id"] = original["id"]
        if action == "A051" and params.get("original_task_id") in self.tasks:
            params["task"] = deepcopy(self.tasks[params["original_task_id"]])
            params["requester"] = self.coordinator_id
        if action in {"A052", "A053", "A054", "A055", "A056", "A057", "A058"}:
            generated = task["progress"].get("generated_tasks", [])
            managed_id = generated[-1] if action == "A052" and generated else params.get("original_task_id")
            managed = self.tasks.get(managed_id)
            if action == "A052" and managed is None and self.is_coordinator:
                managed = next((t for t in self.tasks.values() if t["id"] != task["id"] and t["status"] == "proposed"), None)
            if managed:
                params.update(task_id=managed["id"], rev=managed["rev"])
                if action == "A052":
                    if managed.get("assigned_to"):
                        params["agent_id"] = managed["assigned_to"]
                    elif self.is_coordinator:
                        decision = next((d for d in reversed(self.allocation_decisions)
                                         if d["task_id"] == managed["id"] and d.get("agent_id")), None)
                        if decision:
                            params["agent_id"] = decision["agent_id"]
                if action == "A055" and task["type"] == "T083":
                    params["safe_stop_only"] = True
        if action == "A060":
            # Archive compact result records, never prior archive observations
            # (which themselves contain records and would grow exponentially).
            params["records"] = [{key: deepcopy(result.get(key)) for key in
                ("instance_id", "action_id", "task_ref", "status", "failure_code", "feedback_zh", "cost", "evidence_ids")}
                for result in self.records.get("action_results", [])
                if result.get("action_id") != "A060" and result.get("instance_id") not in
                   self.records.get("archived_action_ids", {})]
            params["observations"] = []
            params["evidence_ids"] = []
        return params

    def _goal_satisfied(self, task):
        if task['progress'].get('pending_camera_analysis'):
            return False
        for expression, expected in task["goal_predicate"].items():
            if expression == "telemetry.battery_at_least":
                if self.platform["battery_Wh"] < expected:
                    return False
                continue
            if expression == "record.handoff_confirmed":
                original = self.tasks.get(task.get("parameters", {}).get("original_task_id"), task)
                if bool(original["progress"].get("handoff_completed")) != bool(expected):
                    return False
                continue
            if expression == "record.supply_receipt":
                target = task.get("parameters", {}).get("receiver_id", task["target"])
                receipts = task["progress"].get("receipts", [])
                confirmed = any(r["receiver_id"] == target and r["status"] == "succeeded" for r in receipts)
                for eid, fields in self.belief.entities(copy=False).items():
                    if fields.get("item.state", {}).get("value") == "received" and \
                            fields.get("item.receiver_id", {}).get("value") == target:
                        confirmed = True
                if confirmed != bool(expected):
                    return False
                continue
            if expression == "record.service_result":
                scope = task["progress"].get("service_scope", {})
                if not scope.get("done") or scope.get("status") != "succeeded":
                    return False
                continue
            if expression == "record.method_evidence":
                method = task["method"]
                records = task["progress"].get("steps", {})
                if method["step_index"] < len(method["resolved_steps"]) or not records:
                    return False
                required_steps = [records.get(f"{method['seq_id']}:{i}")
                                  for i in range(len(method["resolved_steps"]))]
                if any(not record or record.get("status") != "succeeded" for record in required_steps):
                    return False
                continue
            if "@" not in expression:
                return False
            key, eid = expression.rsplit("@", 1)
            actual = _value(self.belief, eid, key)
            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return bool(task["goal_predicate"])

    def _finish_task(self, sim, task):
        if task["type"] == "T085" and not self._goal_satisfied(task):
            started = task["progress"].setdefault("handoff_wait_started", sim.t)
            if sim.t - started <= 120:
                return
            self._fail(sim, task, "network")
            return
        if self._goal_satisfied(task):
            self._controls.append(("A054", {"task_id": task["id"], "status": "completed"},
                                   task["id"], "complete", {}))
        else:
            code = "receipt" if task["type"] in {"T030", "T035", "T036", "T041"} else "uncertain"
            task["failure_reason"] = _failure_reason(code)
            self._status(sim, task, "blocked", "方法结束但本地目标证据不足，不能宣称完成")
            self._recover(sim, task, code)

    def _record_artifact(self, sim, result, context):
        if result["status"] != "succeeded":
            return
        action = result["action_id"]
        params = result.get("params") or context.get("params", {})
        evidence = list(params.get("evidence_ids", []))
        if action in {"A035", "A036"}:
            mid = self.platform.get("message.correlation_id")
            if mid:
                ref = result.get("task_ref") or context.get("task")
                owner_task = self.tasks.get(ref[0]) if ref else None
                if owner_task:
                    owner_task["progress"]["last_sent_message_id"] = mid
                receivers = params.get("receivers", [])
                self.sent_messages[mid] = {"id": mid, "kind": params.get("kind", "report"),
                    "op": params.get("payload", {}).get("op", "report"), "payload": deepcopy(params.get("payload", {})),
                    "receivers": [receivers] if isinstance(receivers, str) else list(receivers),
                    "sent_s": sim.t, "ack_state": "pending" if params.get("payload", {}).get("require_ack", params.get("kind") in {"task", "assignment", "control"}) else "not_required", "timeout_s": 60,
                    "ttl_s": params.get("ttl_s", 120),
                    "status": {receiver: "queued" for receiver in ([receivers] if isinstance(receivers, str) else receivers)},
                    "retry_count": 0, "task_ref": ref}
        for a, table in {"A006": "processed_evidence", "A039": "encoded_evidence"}.items():
            if action == a:
                self.records.setdefault(table, {}).update({eid: result["instance_id"] for eid in evidence})
        if action == "A036" and params.get("retry_of") in self.sent_messages:
            original = self.sent_messages[params["retry_of"]]
            original["retry_count"] = original.get("retry_count", 0) + 1
            original["sent_s"] = sim.t
            retry = self.sent_messages.get(self.platform.get("message.correlation_id"), {})
            retry["retry_count"] = original["retry_count"]
            retry["retry_root"] = original.get("retry_root", original["id"])
        if action == "A060":
            self.records.setdefault("archived_action_ids", {}).update({
                r["instance_id"]: result["instance_id"] for r in self.records.get("action_results", [])
                if r["instance_id"] != result["instance_id"]})
        if action == "A061":
            self.records["health_checked_at"] = sim.t
        if action == "A042":
            self.records["service_discovered_at"] = sim.t
        if action == "A038":
            self.records.setdefault("history_queries", {})[params.get("task_id", self.id)] = sim.t
        job_id = params.get("job_id")
        if job_id:
            job = self.local_jobs.setdefault(job_id, {"id": job_id})
            if action == "A044":
                job.update(status="queued", resources_reserved=True)
            elif action == "A006":
                job.update(status="completed", result_evidence_ids=result.get("evidence_ids", []))
            elif action == "A048":
                job["status"] = "canceled"
            elif action == "A049":
                job["resources_reserved"] = False
        # Only identified resources are recorded. A successful discovery without
        # advertisements does not invent a service, cache, or stock item.
        if action == "A045" and params.get("service_id"):
            self.records.setdefault("services", {})[params["service_id"]] = {
                "id": params["service_id"], "status": "unloaded" if params.get("unload") else "ready",
                "model_id": params.get("model_id"), "version": params.get("version"), "t": sim.t}
        if action == "A046" and params.get("cache_id"):
            self.records.setdefault("caches", {})[params["cache_id"]] = {
                "id": params["cache_id"], "content_ids": evidence,
                "model_version": params.get("model_version"),
                "reuse_compatible": params.get("reuse_compatible", False),
                "valid_until_s": params.get("valid_until_s", sim.t + 60), "eviction_state": "resident"}

    def _feedback_observations(self, sim, result, context):
        """Infer only postconditions guaranteed by a successful local action.

        No truth/event-log lookup: actuator success is local evidence. A017 does
        not tell limited vs immobile, so the specialist still runs A018 to learn
        mobility. These explicitly labelled receipts go through the real A041.
        """
        if result["status"] != "succeeded" and not (result["action_id"] == "A008" and result["status"] == "partial"):
            return []
        params = result.get("params") or context.get("params", {})
        target = params.get("target", params.get("human_id"))
        action = result["action_id"]
        values = {}
        if action == "A008":
            rows = []
            for observation in result.get("observations", []):
                observation = _dict(observation)
                if observation.get("kind") == "interior_search_detection":
                    for human in observation.get("about", []):
                        rows.append({"id": f"feedback:{result['instance_id']}:{human}", "by": self.id,
                            "t": sim.t, "about": [human], "kind": "action_feedback",
                            "values": {"entity.parent_id": target}, "confidence": observation.get("confidence", .8),
                            "model": "interior_search_context_v1"})
            return rows
        if action == "A028":
            fresh = set(self.platform.get("payload", [])) - set(context.get("payload_before", []))
            return [{"id": f"feedback:{result['instance_id']}:{item}", "by": self.id, "t": sim.t,
                     "about": [item], "kind": "action_feedback", "confidence": 1.,
                     "model": "successful_action_postcondition_v1", "values": {
                         "item.kind": context.get("item_kind", params.get("item_kind", "water")),
                         "item.state": "attached", "item.holder_id": self.id,
                         "entity.pose": deepcopy(self.platform["pose"])}} for item in sorted(fresh)]
        if action == "A017":
            values = {"human.buriedness": 0}
        elif action == "A019":
            supported = _value(self.belief, target, "human.support_received", [])
            values = {"human.support_received": list(dict.fromkeys([*supported, "treated"]))}
        elif action in {"A012", "A013", "A016"}:
            values = {"obstacle.state": "cleared", "obstacle.work_remaining": 0}
        elif action in {"A020", "A023"}:
            values = {"human.carrier_id": self.id, "human.location_context": "on_carrier"}
        elif action == "A021":
            values = {"human.handover_status": "arrived", "human.destination_id": params.get("station_id", params.get("destination_id")),
                      "entity.pose": deepcopy(self.platform["pose"])}
        elif action == "A022":
            values = {"human.handover_status": "acknowledged", "human.carrier_id": None,
                      "human.location_context": "safe_point", "human.destination_id": params.get("station_id", params.get("destination_id"))}
        if not values or not target:
            return []
        rows = []
        def receipt(data, suffix):
            return {"id": f"feedback:{result['instance_id']}:{suffix}", "by": self.id, "t": sim.t,
                    "about": [target], "kind": "action_feedback", "values": data,
                    "confidence": 1., "model": "successful_action_postcondition_v1"}
        before = _value(self.belief, target, "human.handover_status")
        after = values.get("human.handover_status")
        if before == "not_started" and after in {"arrived", "acknowledged"}:
            rows.append(receipt({"human.handover_status": "in_transit"}, "transit"))
        if before in {"not_started", "in_transit"} and after == "acknowledged":
            rows.append(receipt({"human.handover_status": "arrived"}, "arrival"))
        rows.append(receipt(values, "result"))
        return rows

    def on_action_result(self, sim, res):
        result = _dict(res)
        if result["action_id"] in PLATFORM_ACTIONS:
            self._refresh_telemetry(sim)
        context = self.pending_actions.pop(result["instance_id"], {})
        ref = context.get("task") or result.get("task_ref")
        task = self.tasks.get(ref[0]) if ref else None
        if context.get("pair"):
            self._proposal_pending.discard(tuple(context["pair"]))
        if task:
            for key in COST_KEYS:
                task["costs"][key] += (result.get("cost") or {}).get(key, 0) or 0
        self.records.setdefault("action_results", []).append(deepcopy(result))
        revised_by_assignment = bool(task and result["action_id"] == "A052" and
            result["status"] == "succeeded" and task["rev"] == ref[1] + 1)
        if task and ref[1] != task["rev"] and not revised_by_assignment:
            self.records.setdefault("stale_results", []).append(deepcopy(result))
            self._branch_event(sim, task, "Q108", "旧版动作结果隔离，不用于新版完成证据")
            return
        observations = list(result.get("observations", []))
        observations.extend(self._feedback_observations(sim, result, context))
        if observations:
            serialized = [_dict(o) for o in observations]
            for obs in observations:
                self.belief.add_evidence(sim, obs)
            self._observations = serialized
            if task:
                task["progress"]["latest_observations"] = serialized
            if task:
                task["result_evidence_ids"] = list(dict.fromkeys(task["result_evidence_ids"] + [o["id"] for o in serialized]))
            if result["action_id"] not in {"A041", "A040", "A007", "A011"}:
                self._controls.appendleft(("A041", {"observations": serialized,
                    "evidence_ids": []}, task["id"] if task else None,
                    "integrate", {}))
        if task:
            task["result_evidence_ids"] = list(dict.fromkeys(task["result_evidence_ids"] +
                                                list(result.get("evidence_ids") or [])))
            if result["action_id"] == "A027" and result["status"] == "succeeded":
                params = result.get("params") or context.get("params", {})
                task["progress"].setdefault("receipts", []).append({
                    "item_id": params.get("item_id"),
                    "receiver_id": params.get("receiver_id", task["target"]),
                    "status": "succeeded", "instance_id": result["instance_id"]})
        self._record_artifact(sim, result, context)
        purpose = context.get("purpose")
        if purpose == "integrate" and task and task["progress"].get("service_scope", {}).get("done") and \
                task["status"] == "blocked" and self._goal_satisfied(task):
            self._status(sim, task, "executing", "专业主体的本地证据已满足其持有副本目标", "A054")
            self._controls.appendleft(("A054", {"task_id": task["id"], "status": "completed"}, task["id"], "complete", {}))
        if purpose == "handoff_suspend":
            if task and result["status"] == "succeeded":
                self._offer_handoff(sim, task, context["receiver"])
            elif task:
                self._fail(sim, task, "resource")
            return
        if purpose == "handoff_release":
            if task and result["status"] == "succeeded":
                incoming = context["incoming"]
                task["progress"]["handoff_completed"] = True
                self._revisions[task["id"]] = incoming["rev"]
                self._send(sim, [incoming["accepted_by"]], "handoff_release", {
                    "task_id": incoming["id"], "rev": incoming["rev"]}, task=task)
                self._send(sim, [self.coordinator_id], "handoff_committed", {"task": incoming}, task=task)
            return
        if purpose == "assignment_send":
            if task and task["id"] in self._assignment_wait:
                waiting = self._assignment_wait[task["id"]]
                waiting["sent_s"] = sim.t
                if result["status"] == "succeeded":
                    waiting["mid"] = self.platform.get("message.correlation_id")
                    if waiting["mid"]:
                        self._controls.append(("A037", {"message_id": waiting["mid"], "timeout_s": 30},
                                               task["id"], "assignment_ack", {}))
                else:
                    task.setdefault("assignment", {})["last_send_error"] = result.get("rejection_code", result.get("failure_code"))
            return
        if purpose == "remote_send":
            if task and result["status"] == "succeeded":
                method = task["progress"].get("recovery") or task["method"]
                method["step_index"] += 1
                task["progress"].setdefault("steps", {})[f"{method['seq_id']}:{method['step_index'] - 1}"] = {
                    "action": "A036", "status": "succeeded", "instance_id": result["instance_id"]}
                # A036 has queued the input message; only its recipient runs
                # the remote steps. Do not duplicate transport here.
            elif task:
                self._recover(sim, task, _result_cause(result))
            return
        if purpose == "specialist_request":
            waiting = self._specialist_wait.get(context.get("request_id"))
            if waiting and result["status"] == "succeeded":
                # A059 has queued the service-contract message. Submission is
                # not service completion and must not send a second request.
                waiting["t"] = sim.t
            elif task:
                self._specialist_wait.pop(context.get("request_id"), None)
                if task["status"] == "executing":
                    self._status(sim, task, "blocked", "专业请求被拒绝，未执行工程动作")
                self._fail(sim, task, "capability")
            return
        if purpose in {"specialist_travel", "specialist_action", "specialist_verify", "remote_action"}:
            result["observations"] = [_dict(o) for o in observations]
            self._service_result(sim, result, context)
            return
        if purpose == "supply_pickup":
            fresh = set(self.platform.get("payload", [])) - set(context.get("payload_before", []))
            if task and result["status"] == "succeeded" and fresh:
                task["parameters"]["item_id"] = sorted(fresh)[0]
            elif task:
                self._fail(sim, task, _result_cause(result))
            return
        if purpose == "approach":
            if task and result["status"] != "succeeded":
                self._recover(sim, task, _result_cause(result), observed_action=result["action_id"])
            return
        if not task or purpose != "method":
            if task and purpose in {"control", "complete"} and result["status"] not in {"succeeded", "partial"}:
                reason = result.get("rejection_code", "none")
                if reason == "none":
                    reason = result.get("failure_code", "model")
                if result["action_id"] == "A053" and task["status"] == "received":
                    task["failure_reason"] = _failure_reason(reason)
                    task.setdefault("assignment", {})["decision"] = "rejected"
                    self._status(sim, task, "failed", f"接受接口拒绝：{reason}", "A053")
                    self._send(sim, [self.coordinator_id], "decision", {"task_id": task["id"],
                               "rev": task["rev"], "accepted": False, "reason_zh": reason, "task": task}, task=task)
                else:
                    self._fail(sim, task, reason)
            return
        if task["status"] in TERMINAL:
            return
        method = task["progress"].get("recovery") or task["method"]
        action = result["action_id"]
        status = result["status"]
        task["progress"].setdefault("steps", {})[f"{method['seq_id']}:{context.get('step', 0)}"] = {
            "action": action, "status": status, "instance_id": result["instance_id"],
            "evidence_ids": result.get("evidence_ids", []),
        }
        if status == "succeeded":
            method["step_index"] += 1
            if action == "A056" and method["seq_id"] == "Q102":
                self._controls.appendleft(("A050", {"type": "T043", "target": self.id,
                    "priority": 96, "maintenance": True,
                    "origin": {"rule_id": "R-T043-a", "belief_refs": [[self.id, "platform.battery_remaining_Wh"]],
                               "why_zh": "本机实际电量触发挂起后补能，充电仍须接口与能源模型检查"}},
                    None, "propose", {}))
        elif status == "partial" and action == "A008":
            method["step_index"] += 1
            self._branch_event(sim, task, "Q110", "部分搜索保留覆盖和漏检，继续整合并报告；补搜由新证据规则决定")
            task["progress"]["partial_search"] = True
        elif status == "partial" and action in {"A005", "A006"}:
            method["step_index"] += 1
            self._recover(sim, task, "uncertain", observed_action=action)
        else:
            code = _result_cause(result)
            feedback = result.get("feedback_zh", "")
            if code == "insufficient":
                memory_feedback = "内存" in feedback or "显存" in feedback or "memory" in feedback.lower()
                if memory_feedback and action in {"A044", "A045", "A006"}:
                    code = "memory"
                # A generic insufficient rejection does not prove low energy.
                # Preserve it unless the action supplies a specific cause.
            if action == "A027":
                code = "receipt"
            if action in {"A025", "A026"} and code in {"actuator", "model"}:
                code = "actuator"
            task["progress"]["last_failure"] = {"code": code, "action": action, "feedback_zh": feedback}
            if code == "blocked" and self._approach_casualty(
                    sim, task, action, context.get("params", {})):
                # The failed action and its cost remain recorded. Retry this
                # same method step only after reaching the newly reported site.
                return
            self._recover(sim, task, code, observed_action=action)

    def _branch_event(self, sim, task, seq_id, why):
        self._emit(sim, "branch.trigger", why, refs=[seq_id, task["type"]],
                   subject=task["id"], task=task, after={"seq_id": seq_id, "trigger": why})

    def _branch_method(self, seq_id):
        steps = self.catalog.steps.get(seq_id, [])
        resolved, details = [], []
        for step in steps:
            options = list(step["options"])
            possible = [a for a in options if set(self.catalog.actions[a].get("capabilities", [])).issubset(self.spec["capabilities"])]
            if "A001" in options and self.spec["mobility"] == "air":
                action = "A001"
            elif "A002" in options and self.spec["mobility"] in {"legged", "wheeled", "foot"}:
                action = "A002"
            else:
                action = (possible or options)[0]
            delegated = "专业/外部" in step.get("actor", "") and action not in possible
            resolved.append("A059" if delegated else action)
            details.append({"step": step["step"], "action_id": action, "actor": step.get("actor"),
                            "delegated": delegated})
        return {"seq_id": seq_id, "resolved_steps": resolved, "details": details, "step_index": 0}

    def _recover(self, sim, task, code, observed_action=None):
        if is_meta_task(task):
            self._fail(sim, task, code)
            return  # protocol failure is not a new recursive rescue protocol
        branch = RECOVERY.get(code)
        if code == "actuator" and observed_action not in {"A025", "A026", "A027"}:
            # Q111 diagnoses cargo release/receipt. A loading or medical-stock
            # failure does not authorize A025 on a human.
            branch = None
        if task["type"] == "T049" and code == "energy":
            branch = "Q105"
        if code == "blocked" and observed_action == "A008" and task["type"] == "T014" and "C17" not in self.spec["capabilities"]:
            branch = "Q099"
        if code == "no_capability" and not (self.kind == "robot_dog" and observed_action == "A012"):
            branch = None
        if not branch:
            self._fail(sim, task, code)
            return
        attempts = task["progress"].setdefault("retries", {})
        total = task["progress"].get("recovery_attempts", 0)
        if attempts.get(branch, 0) >= 2 or total >= 2:
            self._fail(sim, task, "capability" if code == "specialist" else code)
            return
        attempts[branch] = attempts.get(branch, 0) + 1
        task["progress"]["recovery_attempts"] = total + 1
        task["failure_reason"] = _failure_reason(code)
        if task["status"] == "executing" and branch not in {"Q102", "Q105"}:
            self._status(sim, task, "blocked", "实际动作反馈触发目录恢复分支")
        self._branch_event(sim, task, branch, self.catalog.sequences[branch]["trigger"])
        recovery = self._branch_method(branch)
        # Conditional sequences include their triggering action. It has just happened;
        # preserve it as evidence rather than replaying the failed trigger indefinitely.
        if observed_action and recovery["resolved_steps"] and recovery["resolved_steps"][0] == observed_action:
            recovery["step_index"] = 1
        old = task["progress"].get("recovery")
        if old:
            task["progress"].setdefault("recovery_history", []).append(deepcopy(old))
        task["progress"]["recovery"] = recovery
        task["progress"]["recovery_code"] = code
        task["progress"]["retry_action"] = observed_action
        if code == "memory":
            self.local_jobs.setdefault(task["id"], {}).update(status="rejected", failure_reason="memory")
        if code == "blocked":
            self.records["path_failure"] = {"target": task["target"], "failure_code": "blocked", "t": sim.t}
        if code == "energy":
            self.records["charging_needed"] = True
        if not is_meta_task(task):
            self.current = task["id"]

    def _finish_recovery(self, sim, task):
        recovery = task["progress"].pop("recovery")
        task["progress"].setdefault("recovery_history", []).append(recovery)
        code = task["progress"].get("recovery_code")
        if code in {"receipt", "actuator"} and not self._goal_satisfied(task):
            self._fail(sim, task, code)
            return
        if task["status"] in {"blocked", "suspended"}:
            self._controls.appendleft(("A057", {"task_id": task["id"]}, task["id"], "control", {}))
        # Partial search yields a report, not a fabricated absence or full coverage.
        if code in {"partial_search", "uncertain"}:
            if self._goal_satisfied(task):
                task["method"]["step_index"] = len(task["method"]["resolved_steps"])
            elif task["progress"].get("retries", {}).get(recovery["seq_id"], 0) >= 2:
                self._fail(sim, task, "sensor")

    def _fail(self, sim, task, reason):
        if task["status"] in TERMINAL:
            return
        task["failure_reason"] = _failure_reason(reason)
        if task["status"] == "suspended":
            self._status(sim, task, "canceled", f"无法安全恢复：{reason}，保留全部费用")
        else:
            if task["status"] in {"accepted", "proposed", "assigned"}:
                self._status(sim, task, "canceled", f"无法建立可执行承诺：{reason}")
            else:
                self._status(sim, task, "failed", f"恢复次数耗尽或条件不满足：{reason}")
        self._send_task_update(sim, task)

    def _expire(self, sim):
        for task in self._task_index.rows({"proposed", "assigned", "received"}):
            if task["status"] in {"proposed", "assigned", "received"} and task.get("deadline_s") is not None and sim.t >= task["deadline_s"]:
                self._status(sim, task, "expired", "未建立承诺的任务超过截止时间", "KERNEL")

    def _assignment_timeouts(self, sim):
        if not self.is_coordinator:
            return
        for tid, waiting in list(self._assignment_wait.items()):
            task = self.tasks.get(tid)
            if not task or task["status"] not in {"assigned", "received"} or task["rev"] != waiting["rev"]:
                self._assignment_wait.pop(tid, None)
                continue
            if waiting["sent_s"] is None or sim.t - waiting["sent_s"] < 90 * (2 ** waiting["retries"]):
                continue
            self._branch_event(sim, task, "Q109", "中心未收到分配接受回执，不假定已承诺")
            if waiting["retries"] < 2:
                waiting["retries"] += 1
                waiting["sent_s"] = sim.t
                params = {"task_id": task["id"], "receivers": [task["assigned_to"]],
                          "kind": "assignment", "payload": {"op": "assignment", "task": deepcopy(task), "coordinator": self.id,
                                "entries": self._target_entries(task["target"]), "evidence": self._target_evidence(task["target"])}}
                if waiting.get("mid"):
                    params["retry_of"] = waiting["mid"]
                self._controls.appendleft(("A036", params, task["id"], "assignment_send", {}))
            else:
                self._reassign(sim, task, task["assigned_to"], "分配回执超时")
                self._assignment_wait.pop(tid, None)

    def _handoff_timeouts(self, sim):
        for task in self._task_index.rows({"accepted", "suspended"}):
            progress = task["progress"]
            source = progress.get("handoff_wait_release")
            if source and task["status"] == "accepted":
                last = progress.setdefault("handoff_last_wait_s", sim.t)
                if sim.t - last >= 30:
                    retries = progress.get("handoff_retries", 0)
                    if retries >= 2:
                        self._controls.appendleft(("A055", {"task_id": task["id"],
                            "reason": "交接释放消息超时，未执行新承诺"}, task["id"], "control", {}))
                        progress.pop("handoff_wait_release", None)
                    else:
                        progress.update(handoff_retries=retries + 1, handoff_last_wait_s=sim.t)
                        self._send(sim, [source], "handoff_ready", {"task": task, "accepted": True}, task=task)
            pending = progress.get("handoff_pending")
            if isinstance(pending, dict) and task["status"] == "suspended" and not pending.get("releasing"):
                last = pending.get("sent_s", sim.t)
                if sim.t - last >= 30:
                    retries = pending.get("retries", 0)
                    if retries >= 2:
                        progress.pop("handoff_pending", None)
                        self._controls.appendleft(("A057", {"task_id": task["id"]}, task["id"], "control", {}))
                    else:
                        pending["retries"] = retries + 1
                        self._offer_handoff(sim, task, pending["receiver"])

    def _reassign(self, sim, task, rejected, reason):
        if is_meta_task(task) or task["status"] in TERMINAL:
            return
        # Busy or a now-occupied payload is not missing equipment/capability.
        # Delivered decision telemetry feeds the existing commitment/payload
        # filters; a carrier may become eligible after its actual handover.
        if reason not in {"busy", "payload"}:
            self._rejected_agents.setdefault(task["id"], set()).add(rejected)
        retries = task.get("retry_count", 0)
        old_assignee = task.get("assigned_to")
        self._status(sim, task, "canceled", reason + "；旧版本撤销，保留成本", "A054")
        if old_assignee:
            self._send(sim, [old_assignee], "cancel", {"task_id": task["id"],
                "rev": task["rev"] + 1, "reason": reason}, task=task)
        if retries >= 2:
            return
        state = self._allocation_state.setdefault(task["id"], {})
        state["next_s"] = sim.t + 30 * (2 ** retries)
        params = {**deepcopy(task.get("parameters", {})), "type": task["type"],
                  "target": task["target"], "reopen_task_id": task["id"],
                  "priority": task["priority"], "goal_predicate": deepcopy(task["goal_predicate"]),
                  "origin": {**deepcopy(task["origin"]), "why_zh": reason + "，基于原本地需求提出新版"}}
        self._controls.append(("A050", params, None, "propose", {}))

    def on_message(self, sim, msg):
        """Called on delivery only; never inspect another runtime or a network queue."""
        # Contract serialization already detaches the delivered message tree.
        message = msg.to_dict() if isinstance(msg, Message) else deepcopy(_dict(msg))
        mid = message["id"]
        if mid in self._received_ids:
            return
        self._received_ids.add(mid)
        self.received_messages.append(message)
        self.belief.received_messages = len(self.received_messages)
        payload = message.get("payload") or {}
        sender = message["sender"]
        op = payload.get("op", message.get("kind"))
        if op != "ack" and (message.get("ack_state") in {"pending", "timeout"} or payload.get("require_ack")):
            self._send(sim, [sender], "ack", {"message_id": mid}, kind="ack", correlation=mid)
        if op == "ack":
            original = self.sent_messages.get(payload.get("message_id"))
            if original and sender in original["receivers"]:
                acknowledgements = original.setdefault("acknowledgements", [])
                if sender not in acknowledgements:
                    acknowledgements.append(sender)
                original.setdefault("status", {})[sender] = "delivered"
                if set(original["receivers"]).issubset(acknowledgements):
                    original["ack_state"] = "acknowledged"
                ref = original.get("task_ref")
                acknowledged_task = self.tasks.get(ref[0]) if ref else None
                if acknowledged_task and acknowledged_task["rev"] == ref[1]:
                    acknowledged_task["progress"]["last_received_message_id"] = mid
                if original["op"] == "report" and original["ack_state"] == "acknowledged":
                    self.belief.mark_delivered(original["payload"].get("entries", []), evidence=original["payload"].get("evidence", []))
                    self.records.setdefault("shared_evidence", {}).update({
                        obs["id"]: original["id"] for obs in original["payload"].get("evidence", []) if obs.get("id")})
            return
        if op == "brief":
            self.public_objectives = deepcopy(payload.get("objectives", []))
            self.records["public_brief_zh"] = payload.get("brief_zh", "")
            self.roster.update(deepcopy(payload.get("roster", {})))
            self.records["fleet_manifest"] = deepcopy(self.roster)
            self.belief.put(sim, self.id, "belief.current_goal_context", self.public_objectives,
                            1, "message", [mid], sender)
        elif op == "report" or message.get("kind") == "report":
            self._merge_report(sim, payload, sender, mid)
        elif op == "proposal" and self.is_coordinator:
            incoming = payload.get("task", {})
            if is_meta_task(incoming):
                return
            known = self.tasks.get(incoming.get("id"))
            newer = bool(known and incoming.get("rev", -1) > known["rev"] and incoming.get("creator") == sender)
            if incoming.get("id") and incoming.get("origin") and (known is None or newer):
                incoming = deepcopy(incoming)
                incoming["holder"] = self.id
                self._store_task(incoming)
                self._emit(sim, "sm:task.status", "中心收到任务提议副本", refs=["A050", "F201"],
                           subject=incoming["id"], task=incoming, field="task.status", after=incoming["status"])
        elif op in {"assignment", "handoff"}:
            self._receive_assignment(sim, payload, sender, mid)
        elif op == "decision" and self.is_coordinator:
            task = self.tasks.get(payload.get("task_id"))
            if not task or task["rev"] != payload.get("rev") or task.get("assigned_to") != sender:
                return
            self._assignment_wait.pop(task["id"], None)
            if payload.get("telemetry"):
                self._merge_report(sim, {"telemetry": payload["telemetry"]}, sender, mid)
            if payload.get("accepted"):
                self._merge_task_report(sim, payload.get("task", {}), sender)
            else:
                task.setdefault("assignment", {})["decision"] = "rejected"
                task["assignment"]["decision_reason"] = payload.get("reason_zh")
                if self.roster.get(sender, {}).get("kind") == "robot_dog" and task["type"] == "T022" and \
                        payload.get("reason_zh") in {"capability", "no_capability"}:
                    self._branch_event(sim, task, "Q100", "收到机器狗无清障工具的拒绝，按能力改派")
                self._reassign(sim, task, sender, payload.get("reason_zh", "执行者拒绝"))
        elif op == "handoff_ready":
            incoming = payload.get("task", {})
            task = self.tasks.get(incoming.get("id"))
            pending = task["progress"].get("handoff_pending", {}) if task else {}
            if task and pending.get("receiver") == sender and pending.get("rev") == incoming.get("rev"):
                if payload.get("accepted"):
                    pending["releasing"] = True
                    self._controls.appendleft(("A055", {"task_id": task["id"], "reason": "新方已实际接受交接"},
                        task["id"], "handoff_release", {"incoming": deepcopy(incoming)}))
                else:
                    task["progress"].pop("handoff_pending", None)
                    if task["status"] == "suspended":
                        self._controls.appendleft(("A057", {"task_id": task["id"]}, task["id"], "control", {}))
        elif op == "handoff_release":
            task = self.tasks.get(payload.get("task_id"))
            if task and task["rev"] == payload.get("rev") and task["progress"].get("handoff_wait_release") == sender:
                task["progress"].pop("handoff_wait_release", None)
        elif op == "handoff_committed" and self.is_coordinator:
            incoming = payload.get("task", {})
            task = self.tasks.get(incoming.get("id"))
            if task and task.get("assigned_to") == sender and incoming.get("rev", -1) > task["rev"]:
                updated = deepcopy(incoming)
                updated["holder"] = self.id
                self._store_task(updated)
                self._emit(sim, "sm:task.status", "中心收到双方确认的交接新版本", refs=["A058", "F201"],
                    task=updated, subject=updated["id"], field="task.status", after=updated["status"])
        elif op == "task_update":
            self._merge_task_report(sim, payload.get("task", {}), sender)
        elif op == "cancel":
            task = self.tasks.get(payload.get("task_id"))
            if task and sender in {self.coordinator_id, task["creator"]} and payload.get("rev", -1) >= task["rev"]:
                self._revisions[task["id"]] = payload["rev"]
                self._controls.appendleft(("A055", {"task_id": task["id"], "reason": payload.get("reason")},
                                            task["id"], "control", {}))
        elif op == "bid_request":
            task = payload.get("task")
            if task:
                self._controls.append(("A051", {"task": task, "requester": sender}, None, "bid", {}))
        elif op in {"bid", "bid_response"} and self.is_coordinator:
            if payload.get("ok", True) and hasattr(self.allocator, "record_bid"):
                self.allocator.record_bid(payload.get("task_id"), sender,
                                          payload.get("score", 0), payload.get("rev"), received_s=sim.t)
            self._bids.setdefault(payload.get("task_id"), {})[sender] = deepcopy(payload)
            self.records.setdefault("bids", {}).setdefault(payload.get("task_id"), {})[sender] = deepcopy(payload)
            self._merge_report(sim, {"telemetry": payload.get("telemetry", {})}, sender, mid)
        elif op == "specialist_request":
            self._receive_specialist(sim, payload, sender, mid)
        elif op in {"specialist_result", "specialist_decision"}:
            self._receive_specialist_result(sim, payload, sender)
        elif op in {"job", "remote_job", "compute_request"} and self.kind == "compute_node":
            job_id = payload.get("job_id", mid)
            known = self.records.setdefault("remote_requests", {})
            if job_id not in known:
                known[job_id] = {"sender": sender, "received_s": sim.t}
                self._remote_work.append({"sender": sender, "payload": deepcopy(payload), "message_id": mid})
        elif op in {"job_result", "compute_result"}:
            ref = message.get("task_ref") or payload.get("task_ref")
            task = self.tasks.get(ref[0]) if ref else None
            if task and ref[1] != task["rev"]:
                self.records.setdefault("stale_results", []).append(message)
                self._branch_event(sim, task, "Q108", "远程结果版本过旧，隔离而不融合")
            else:
                waiting = task["progress"].get("remote_wait") if task else None
                if waiting and (waiting["node_id"] != sender or waiting["job_id"] != payload.get("job_id")):
                    return
                old_job = self.local_jobs.get(payload.get("job_id", mid), {})
                if old_job.get("status") == "completed":
                    return
                self.local_jobs[payload.get("job_id", mid)] = deepcopy(payload)
                if task and waiting:
                    if payload.get("status") == "succeeded":
                        method = task["progress"].get("recovery") or task["method"]
                        method["step_index"] = waiting["end_index"]
                        self._observations = deepcopy(payload.get("observations", []))
                        for r in payload.get("results", []):
                            step = r.get("params", {}).get("step", 1) - 1
                            task["progress"].setdefault("steps", {})[f"{method['seq_id']}:{step}"] = {
                                "action": r.get("action_id"), "status": r.get("status"),
                                "instance_id": r.get("instance_id"), "remote_actor": sender}
                        # The return message itself proves the remote A036
                        # sender step ran; its payload precedes that action result.
                        remote_send_index = waiting["end_index"] - 1
                        task["progress"].setdefault("steps", {})[f"{method['seq_id']}:{remote_send_index}"] = {
                            "action": "A036", "status": "succeeded", "message_id": mid, "remote_actor": sender}
                        task["result_evidence_ids"].extend(payload.get("evidence_ids", []))
                        self.local_jobs[payload["job_id"]]["status"] = "completed"
                        task["progress"].pop("remote_wait", None)
                        task["progress"]["last_received_message_id"] = mid
                    else:
                        code = "memory" if payload.get("rejection_code") == "insufficient" else "service"
                        self._recover(sim, task, code)
                self._merge_report(sim, payload, sender, mid)

    def _offer_handoff(self, sim, task, receiver):
        task["progress"]["handoff_pending"]["sent_s"] = sim.t
        offered = deepcopy(task)
        offered.update(rev=task["rev"] + 1, assigned_to=receiver, accepted_by=None, status="assigned")
        offered["progress"].pop("handoff_pending", None)
        offered["progress"].pop("handoff_completed", None)
        offered["progress"]["handoff_wait_release"] = self.id
        offered["history"] = [{"t": sim.t, "status": "assigned", "note_zh": "交接新版本等待接收方决定"}]
        self._send(sim, [receiver], "handoff", {"task": offered, "evidence": self._evidence_payload()}, task=task)

    def _receive_assignment(self, sim, payload, sender, mid):
        incoming = deepcopy(payload.get("task", {}))
        tid, rev = incoming.get("id"), incoming.get("rev", -1)
        if is_meta_task(incoming):
            return  # proposer-local protocols are never assignment traffic
        if not tid or not incoming.get("origin") or incoming.get("assigned_to") not in {None, self.id} or \
                sender != self.coordinator_id and payload.get("op") != "handoff":
            return
        known = self.tasks.get(tid)
        latest = max(self._revisions.get(tid, -1), known.get("rev", -1) if known else -1)
        if rev < latest or rev == latest and known and known["status"] != "assigned":
            self.records.setdefault("stale_assignments", []).append({"message_id": mid, "task_id": tid, "rev": rev})
            self._branch_event(sim, known or incoming, "Q109", "晚到或重复的旧分配不产生第二次承诺")
            if known and known.get("accepted_by") == self.id:
                if payload.get("op") == "handoff":
                    self._send(sim, [sender], "handoff_ready", {"accepted": True, "task": known}, task=known)
                else:
                    self._send(sim, [sender], "decision", {"task_id": tid, "rev": known["rev"],
                               "accepted": True, "task": known}, task=known)
            return
        if known and known["rev"] < rev:
            for instance, context in list(self.pending_actions.items()):
                if context.get("task") and context["task"][0] == tid:
                    sim.cancel_action(instance, "收到新版本分配，旧实例安全停止")
        incoming["holder"] = self.id
        incoming["status"] = "assigned"
        incoming["assigned_to"] = self.id
        incoming["accepted_by"] = None
        incoming["progress"]["assignment_message_id"] = mid
        self._store_task(incoming)
        self._revisions[tid] = rev
        self._status(sim, incoming, "received", "A065：分配消息已经实际送达", "A065")
        self._controls.appendleft(("A053", {"task_id": tid}, tid, "control", {}))
        if payload.get("evidence") or payload.get("entries"):
            self._merge_report(sim, {"evidence": payload.get("evidence", []),
                                    "entries": payload.get("entries", [])}, sender, mid)

    def _merge_task_report(self, sim, incoming, sender):
        task = self.tasks.get(incoming.get("id"))
        if task is None and self.is_coordinator and not is_meta_task(incoming) and (
                incoming.get("creator") == sender and incoming.get("holder") == sender and
                incoming.get("assigned_to") == sender and incoming.get("origin") and
                incoming.get("type") in self.catalog.tasks):
            task = deepcopy(incoming)
            task["holder"] = self.id
            task["status"] = None
            task["history"] = []
            self._store_task(task)
            self._status(sim, task, "proposed", "实际收到自执行者的任务副本及来源", "A050")
            self._status(sim, task, "assigned", "收到自分配记录", "A052")
            self._remote_history_cursor[(task["id"], task["rev"], sender)] = 0
        if not task or incoming.get("rev") != task["rev"] or sender != task.get("assigned_to"):
            return
        if incoming.get("assignment", {}).get("decision") == "rejected":
            # A rejection is a decision on the proposer's copy, not completion or
            # failure of the rescue work. The subsequent decision message reassigns.
            task.setdefault("assignment", {}).update(deepcopy(incoming["assignment"]))
            return
        # Use a per-version history cursor rather than timestamps: several legal
        # transitions can occur at the same simulation time. Late shorter reports
        # cannot regress this copy or overwrite newer costs/progress.
        cursor_key = (task["id"], task["rev"], sender)
        history = incoming.get("history", [])
        cursor = self._remote_history_cursor.get(cursor_key, 0)
        if len(history) < cursor:
            return
        for item in history[cursor:]:
            if item["status"] != task["status"] and item["status"] in {
                    "received", "accepted", "executing", "blocked", "suspended", "completed", "failed", "canceled"}:
                if task["status"] not in TERMINAL:
                    self._status(sim, task, item["status"], f"收到{sender}报告：{item['note_zh']}")
        self._remote_history_cursor[cursor_key] = len(history)
        for key in ("accepted_by", "commitment_window", "method", "result_evidence_ids", "failure_reason", "costs"):
            if key in incoming:
                task[key] = deepcopy(incoming[key])
        task["progress"].update(deepcopy(incoming.get("progress", {})))

    def _merge_report(self, sim, payload, sender, mid):
        route_keys = {"road.entry_state", "road.clear_width_m", "road.step_height_m", "obstacle.state"}
        location_keys = {"entity.pose", "human.pose", "belief.estimated_pose", "human.carrier_id",
                         "human.destination_id", "human.handover_status", "human.location_context"}
        forward_routes = self.is_coordinator and self.spec.get("share_route_updates", False)
        before_routes = {(eid, key): deepcopy(entry.get("value")) for (eid, key), entry in self.belief.entries.items()
                         if key in route_keys} if forward_routes else {}
        before_locations = {(eid, key): deepcopy(entry.get("value")) for (eid, key), entry in self.belief.entries.items()
                            if key in location_keys} if self.is_coordinator else {}
        telemetry = payload.get("telemetry")
        if telemetry:
            telemetry = deepcopy(telemetry)
            telemetry["id"] = sender
            self.belief.put(sim, sender, "belief.team_estimates", telemetry, 1, "message", [mid], sender)
        for obs in payload.get("evidence", []):
            self.belief.add_evidence(sim, obs)
        for entry in payload.get("entries", []):
            eid, key = entry.get("eid"), entry.get("key")
            if eid is not None and key:
                self.belief.merge_report_entry(sim, entry, sender, mid)
        for obs in payload.get("observations", []):
            obs = _dict(obs)
            self.belief.add_evidence(sim, obs)
            for eid in obs.get("about", []):
                values = obs.get("values", {})
                if isinstance(values.get(eid), dict):
                    values = values[eid]
                for key, value in values.items():
                    if "." in key:
                        self.belief.merge_report_entry(sim, {"eid": eid, "key": key, "value": value,
                            "confidence": obs.get("confidence", .5), "source": obs.get('source', 'observation'), "t": obs.get("t", sim.t),
                            "evidence_ids": [obs["id"]], "origin_agent": obs.get("by", sender)}, sender, mid)
        for task in payload.get("tasks", []):
            self._merge_task_report(sim, task, sender)
        if self.belief.conflicts():
            self.records["conflicting_evidence"] = True

        if forward_routes:
            changed = [{"eid": eid, "key": key, **deepcopy(entry)}
                       for (eid, key), entry in self.belief.entries.items()
                       if key in route_keys and ((eid, key) not in before_routes or
                                                before_routes[eid, key] != entry.get("value"))]
            receivers = [aid for aid in self.roster if aid not in {self.id, sender}]
            if changed and receivers:
                evidence_ids = dict.fromkeys(oid for row in changed for oid in row.get("evidence_ids", []))
                evidence = [_dict(self.belief.evidence[oid]) for oid in evidence_ids if oid in self.belief.evidence]
                self._send(sim, receivers, "report", {"entries": changed, "evidence": evidence}, kind="report")

        if self.is_coordinator:
            # The assessor cannot follow a transported casualty using the old
            # discovery alone. Forward received location reports through the
            # existing message path; no specialist reads the carrier's truth.
            changed = [{"eid": eid, "key": key, **deepcopy(entry)}
                       for (eid, key), entry in self.belief.entries.items()
                       if key in location_keys and (
                           _value(self.belief, eid, "entity.kind") == "human"
                           or self.belief.get(eid, "human.location_context") is not None
                           or self.belief.get(eid, "human.mobility") is not None)
                       and ((eid, key) not in before_locations or before_locations[eid, key] != entry.get("value"))]
            carriers = {row["value"] for row in changed if row["key"] == "human.carrier_id" and row["value"]}
            if telemetry and any(key == "human.carrier_id" and entry.get("value") == sender
                                 for (eid, key), entry in self.belief.entries.items()):
                carriers.add(sender)
            for carrier in sorted(carriers):
                entry = self.belief.get(carrier, "belief.team_estimates")
                if entry:
                    changed.append({"eid": carrier, "key": "belief.team_estimates", **deepcopy(entry)})
            receivers = [aid for aid, row in self.roster.items()
                         if row.get("kind") == "specialist_team" and aid not in {self.id, sender}]
            if changed and receivers:
                evidence_ids = dict.fromkeys(oid for row in changed for oid in row.get("evidence_ids", []))
                evidence = [_dict(self.belief.evidence[oid]) for oid in evidence_ids if oid in self.belief.evidence]
                self._send(sim, receivers, "report", {"entries": changed, "evidence": evidence}, kind="report")

    def _telemetry_report(self):
        current = self.tasks.get(self.current)
        if current:
            current = {key: deepcopy(current.get(key)) for key in ("id", "type", "rev", "status", "priority",
                       "interruptibility", "commitment_window", "deadline_s")}
        # Advertise explicit onsite-contact scope only when enabled; legacy
        # service-only specialists retain their previous telemetry representation.
        return {"id": self.id, "kind": self.kind,
                "spec": {key: deepcopy(self.spec[key]) for key in (
                    "kind", "mobility", "width_m", "step_height_m", "water_tolerance_m",
                    "speed_mps", "battery_Wh", "payload_kg", "external", "mains_power",
                    "sensors", "move_power_W", "length_m", "suppression_reach_m", "isolation_reach_m") if key in self.spec}
                    | ({"task_allowlist": deepcopy(self.spec["task_allowlist"])}
                       if self.kind == "specialist_team" and "T027" in (self.spec.get("task_allowlist") or ()) else {}),
                "pose": deepcopy(self.platform["pose"]), "battery_Wh": round(self.platform["battery_Wh"], 2),
                "battery_max_Wh": self.platform["battery_max_Wh"], "reserve_Wh": self.platform["reserve_Wh"],
                "health": self.platform["health"], "capabilities": list(self.spec["capabilities"]),
                "payload_used_kg": self.platform.get("payload_used_kg", 0),
                "payload": list(self.platform.get("payload", [])),
                "current_task": current, "current": self.current,
                "available": self.spec.get("available", True), "queue": list(self.queue)}

    def _target_entries(self, target):
        # A proven image-track alias without its person's response leaves the
        # assignee unable to retire an already resolved contact.
        targets = {target, _value(self.belief, target, "belief.entity_ref", target)}
        return [{"eid": eid, "key": key, **deepcopy(entry)}
                for (eid, key), entry in self.belief.entries.items() if eid in targets]

    def _target_evidence(self, target):
        targets = {target, _value(self.belief, target, "belief.entity_ref", target)}
        selected = []
        for obs in reversed(self.belief.evidence.values()):
            if targets.intersection(_evidence_record(obs).get("about", [])):
                selected.append(obs)
                if len(selected) == 16:
                    break
        return [_dict(obs) for obs in reversed(selected)]

    def _evidence_payload(self):
        return self.belief.report_delta(exclude_kinds={"local_archive"})["evidence"]

    def _report_payload(self):
        delta = self.belief.report_delta(exclude_kinds={"local_archive"})
        # Archives of task/action logs stay local rather than amplifying reports.
        delta["evidence"] = [e for e in delta["evidence"] if e.get("kind") != "local_archive"]
        return {"telemetry": self._telemetry_report(), **delta}

    def _report(self, sim):
        self._last_report = sim.t
        if self.coordinator_id and self.coordinator_id != self.id:
            if any(m.get("op") == "report" and m.get("ack_state") == "pending"
                   and sim.t - m["sent_s"] < m.get("ttl_s", 180)
                   for m in self.sent_messages.values()):
                return
            payload = self._report_payload()
            signature = (repr(payload["telemetry"]), tuple((e["eid"], e["key"], e.get("t")) for e in payload["entries"]),
                         tuple(e["id"] for e in payload["evidence"]))
            if signature == self._last_report_signature and not payload["entries"] and not payload["evidence"]:
                return
            self._last_report_signature = signature
            self._send(sim, [self.coordinator_id], "report", payload, kind="report")

    def _send_task_update(self, sim, task):
        if not is_meta_task(task) and self.coordinator_id and self.coordinator_id != self.id:
            self._send(sim, [self.coordinator_id], "task_update", {"task": task}, task=task)

    def _specialist_action(self, task):
        explicit = task.get("parameters", {}).get("specialist_action") or task.get("parameters", {}).get("requested_action_id")
        if explicit in self.catalog.actions:
            return explicit
        target = str(task["target"])
        if _value(self.belief, target, "human.mobility") == "trapped":
            return "A017"
        if _value(self.belief, target, "hazard.fire_intensity", 0):
            return "A062"
        if _value(self.belief, target, "obstacle.state") in {"present", "partly_cleared"}:
            return "A013"
        return "A018" if _value(self.belief, target, "entity.kind") == "human" else "A061"

    def _request_specialist(self, sim, task, action, params, method, advance=1):
        specialists = [aid for aid, item in self.roster.items() if item.get("kind") == "specialist_team"]
        # A carried person's transport/handover goes to the specialist team actually carrying them.
        carrier = _value(self.belief, str(task["target"]), "human.carrier_id")
        if carrier in specialists:
            specialists = [carrier] + [aid for aid in specialists if aid != carrier]
        if not specialists:
            self._fail(sim, task, "capability")
            return
        request_id = f"specialist:{task['id']}:{task['rev']}:{method['seq_id']}:{method['step_index']}"
        if request_id in self._specialist_wait:
            waiting = self._specialist_wait[request_id]
            if waiting.get("accepted"):
                if sim.t - waiting["t"] > 2400:
                    self._specialist_wait.pop(request_id, None)
                    self._fail(sim, task, "timeout")
                return
            if sim.t - waiting["t"] > 90:
                if waiting["retries"] >= 2:
                    self._fail(sim, task, "capability")
                else:
                    waiting["retries"] += 1
                    waiting["t"] = sim.t
                    self._send(sim, [waiting["agent"]], "specialist_request", waiting["payload"], task=task)
            return
        # Capture a detached service contract; never retain a mutable parent
        # record inside its own progress/retransmission payload.
        payload = {"request_id": request_id, "task": deepcopy(task), "action_id": action,
                   "params": deepcopy(params), "entries": self._target_entries(task["target"]),
                   "evidence": self._target_evidence(task["target"])}
        self._specialist_wait[request_id] = {"t": sim.t, "retries": 0, "agent": specialists[0],
                                              "payload": payload, "advance": advance}
        # A059 is a real catalog action; its model transports this specialist request.
        self._submit(sim, "A059", {**params, "receivers": [specialists[0]], "receiver": specialists[0],
                                 "specialist_id": specialists[0], "receiver_id": specialists[0], "kind": "task", "payload": {"op": "specialist_request", **payload},
                                 "action_id": action}, task=task, purpose="specialist_request", request_id=request_id)

    def _receive_specialist(self, sim, payload, sender, mid):
        if self.kind != "specialist_team":
            return
        request = deepcopy(payload)
        rid = request.get("request_id", mid)
        known = self.records.setdefault("specialist_requests", {})
        if rid in known:
            old = known[rid]
            if old["sender"] == sender:
                self._send(sim, [sender], "specialist_result" if old.get("result") else "specialist_decision",
                           old.get("result", old.get("decision", {"request_id": rid, "accepted": old["accepted"]})))
            return
        action = request.get("action_id", "A013")
        target = request.get("params", {}).get("target", request.get("task", {}).get("target"))
        for other_id, other in known.items():
            if other.get("accepted") and not other.get("result") and other.get("target") == target and other.get("action") == action:
                other.setdefault("listeners", []).append({"sender": sender, "request_id": rid})
                known[rid] = {"sender": sender, "accepted": True, "shared_with": other_id}
                self._send(sim, [sender], "specialist_decision", {"request_id": rid, "accepted": True,
                    "reason_zh": "同一目标同一专业动作已在执行，等待该实际服务结果，不重复作业"})
                return
        if request.get("evidence") or request.get("entries"):
            self._merge_report(sim, {"observations": request.get("evidence", []),
                                     "entries": request.get("entries", [])}, sender, mid)
        required = set(self.catalog.actions.get(action, {}).get("capabilities", [])) | {"C02"}
        self._refresh_telemetry(sim)
        goal = request.get("params", {}).get("goal_xy", request.get("params", {}).get("goal"))
        if action in {"A018", "A019"}:
            goal = self._target_point(target)
        route = (obstacle_approach_path(self.belief, self.platform["pose"], target, self.spec)
                 if action in {"A012", "A013", "A016"} else
                 belief_path(self.belief, self.platform["pose"], goal, self.spec))
        energy_ok = self.spec.get("mains_power") or self.platform["battery_Wh"] >= self.platform["reserve_Wh"] + route["length_m"] * .001
        # A team with a person aboard finishes that person's handover before serving anyone else.
        aboard = [eid for eid, fields in self.belief.entities(copy=False).items()
                  if _value(self.belief, eid, "human.carrier_id") in {self.id, self.platform_eid}
                  and _value(self.belief, eid, "human.handover_status") != "acknowledged"]
        current = self.tasks.get(self.current, {})
        # Admit only a higher-priority, safely preemptible ordinary commitment.
        # A053 still waits for the real action boundary, then A056 checkpoints it.
        may_preempt = (current.get("status") == "executing" and not is_meta_task(current)
                       and not current.get("progress", {}).get("service_scope")
                       and current.get("interruptibility") == "preemptible"
                       and request.get("task", {}).get("priority", 90) > current.get("priority", 0)
                       and not self._remote_work)
        busy = ((self.is_busy() and not may_preempt) or self._specialist_work
                or any(t["progress"].get("service_scope") for t in self._task_index.rows(ACTIVE)))
        rejection = ("unavailable" if not self.spec.get("available", True) or self.platform["health"] == "failed"
                     else "capability" if action not in self.catalog.actions or not required.issubset(self.spec["capabilities"])
                     else "payload" if aboard and str(target) not in aboard
                     else "access" if not route["ok"] else "energy" if not energy_ok
                     else "busy" if busy else "none")
        ok = rejection == "none"
        reasons = {"none": "可提供专业服务", "unavailable": "专业队不可用或平台失效",
                   "capability": "专业队缺少所请求的实际工具能力", "payload": "专业队正承载人员，须先完成其移交",
                   "access": "专业队没有已知可达路线", "energy": "专业队电量或预留不足",
                   "busy": "专业队当前承诺不能被该请求抢占"}
        decision = {"request_id": rid, "accepted": ok, "rejection_code": rejection,
                    "reason_zh": reasons[rejection]}
        known[rid] = {"sender": sender, "accepted": ok, "target": target, "action": action,
                      "decision": decision}
        self._send(sim, [sender], "specialist_decision", decision)
        if ok:
            incoming = request.get("task", {})
            target = request.get("params", {}).get("target", incoming.get("target"))
            service_id = f"service:{rid}"
            typ = {"A017": "T032", "A018": "T029", "A012": "T022", "A013": "T023"}.get(action,
                incoming.get("type") if not is_meta_task(incoming) else "T092")
            # An external team proposes its OWN substantive service record in
            # reaction to delivered A059, never accepts/assigns the meta parent.
            self._controls.appendleft(("A050", {"type": typ, "target": target,
                "task_record": {"id": service_id, "type": typ}, "priority": incoming.get("priority", 90),
                "service_scope": {"request_id": rid, "action_id": action, "done": False},
                "goal_predicate": {"record.service_result": True},
                "origin": {"rule_id": f"R-{typ}-service", "belief_refs": [[mid, "message.local_availability"]],
                           "why_zh": "实际收到A059服务请求，经本机可用性、工具和路线检查后建立服务范围"}},
                None, "propose", {}))
            self._specialist_work.append({"request_id": rid, "sender": sender, "request": request,
                                          "task_id": service_id, "stage": "travel", "results": []})

    def _execute_specialist(self, sim):
        work = self._specialist_work[0]
        params = deepcopy(work["request"].get("params", {}))
        service_task = self.tasks.get(work.get("task_id"))
        if service_task and service_task["status"] in TERMINAL:
            payload = {"request_id": work["request_id"], "status": "failed",
                       "failure_code": service_task.get("failure_reason", "capability"),
                       "reason_zh": "本地服务任务未获有效承诺或已终止，未执行服务动作",
                       "observations": [], "evidence_ids": [], "results": work["results"]}
            record = self.records["specialist_requests"][work["request_id"]]
            record["result"] = deepcopy(payload)
            self._send(sim, [work["sender"]], "specialist_result", payload)
            for listener in record.get("listeners", []):
                shared = {**payload, "request_id": listener["request_id"]}
                self.records["specialist_requests"][listener["request_id"]]["result"] = deepcopy(shared)
                self._send(sim, [listener["sender"]], "specialist_result", shared)
            self._specialist_work.popleft()
            return
        if not service_task or service_task["status"] not in {"accepted", "executing"}:
            return
        if service_task:
            self.current = service_task["id"]
            params.update(task_id=service_task["id"], rev=service_task["rev"])
            params.pop("task_ref", None)
            if service_task["status"] == "accepted":
                self._status(sim, service_task, "executing", "专业队开始实际到场与服务动作", "KERNEL")
        if work["stage"] == "travel":
            target = params.get("target", work["request"].get("task", {}).get("target"))
            if work["request"].get("action_id") in {"A012", "A013", "A016"}:
                params.update(self._obstacle_approach_params(target))
            if work["request"].get("action_id") in {"A018", "A019"}:
                # Request coordinates are a snapshot, not the person's location
                # for the lifetime of an accepted service contract.
                site = self._target_point(target)
                params.update(goal=site, goal_xy=site)
            aboard = target is not None and _value(self.belief, str(target), "human.carrier_id") in {self.id, self.platform_eid}
            if aboard:
                # The person is already aboard (own load): never walk back to their old reported position.
                # Transport/handover goes to the receiving point; any other service happens where we are.
                for key in ("goal", "goal_xy"):
                    params.pop(key, None)
                if not (work["request"].get("action_id") in {"A021", "A022"}
                        and any(params.get(k) for k in ("destination", "destination_id", "station_id"))):
                    params["goal"] = list(self.platform["pose"][:2])
            self._submit(sim, "A002", params, task=service_task, purpose="specialist_travel", work=work)
        else:
            action = "A018" if work["stage"] == "verify" else work["request"]["action_id"]
            if self._approach_casualty(sim, service_task, action, params, work=work):
                return
            self._submit(sim, action, params, task=service_task,
                         purpose="specialist_verify" if work["stage"] == "verify" else "specialist_action", work=work)

    def _receive_specialist_result(self, sim, payload, sender):
        rid = payload.get("request_id")
        waiting = self._specialist_wait.get(rid)
        if not waiting or waiting["agent"] != sender:
            return
        task = self.tasks.get(waiting["payload"]["task"]["id"])
        if not task or task["rev"] != waiting["payload"]["task"]["rev"]:
            return
        if payload.get("accepted") is False:
            if task["status"] == "executing":
                self._status(sim, task, "blocked", "专业队拒绝，任务保持受阻")
            self._fail(sim, task, payload.get("rejection_code", "capability"))
            return
        if payload.get("op") == "specialist_decision" or "status" not in payload:
            if payload.get("accepted"):
                waiting.update(accepted=True, t=sim.t)
            return
        self._specialist_wait.pop(rid, None)
        self._merge_report(sim, payload, sender, rid)
        if payload["status"] == "succeeded":
            method = task["progress"].get("recovery") or task["method"]
            index = method["step_index"]
            advance = waiting.get("advance", 1)
            for step in range(index, index + advance):
                task["progress"].setdefault("steps", {})[f"{method['seq_id']}:{step}"] = {
                    "action": method["resolved_steps"][step], "status": "skipped" if
                        f"{method['seq_id']}:{step}" in task["progress"].get("inapplicable_steps", {}) else "succeeded",
                    "specialist": sender, "evidence_ids": list(payload.get("evidence_ids", []))}
            method["step_index"] += advance
            task["result_evidence_ids"].extend(payload.get("evidence_ids", []))
        else:
            self._fail(sim, task, _result_cause(payload))

    def _send_remote_method(self, sim, task, method, params):
        nodes = [aid for aid, item in self.roster.items()
                 if item.get("kind") == "compute_node" and aid != self.id]
        if not nodes:
            self._fail(sim, task, "capability")
            return
        start = method["step_index"] + 1
        end = start
        details = method.get("details", [])
        while end < len(details) and str(details[end].get("actor", "")).startswith("远端"):
            end += 1
        job_id = f"remote:{task['id']}:{task['rev']}:{method['seq_id']}:{start}"
        remote = {"op": "remote_job", "job_id": job_id, "task_ref": [task["id"], task["rev"]],
                  "params": deepcopy(params), "seq_id": method["seq_id"],
                  "steps": deepcopy(details[start:end]), "start_index": start, "end_index": end}
        task["progress"]["remote_wait"] = {"job_id": job_id, "sent_s": sim.t,
                                                   "end_index": end, "node_id": nodes[0]}
        self.local_jobs[job_id] = {"status": "uploading", "node_id": nodes[0],
                                   "task_ref": [task["id"], task["rev"]],
                                   "input_evidence_ids": list(task["result_evidence_ids"])}
        self._submit(sim, "A036", {**params, "receivers": [nodes[0]], "payload": remote,
                                  "kind": "task", "node_id": nodes[0]},
                     task=task, purpose="remote_send", remote=remote)

    def _execute_remote(self, sim):
        work = self._remote_work[0]
        payload = work["payload"]
        steps = payload.get("steps") or [{"action_id": "A044", "step": 1},
                                         {"action_id": "A006", "step": 2}]
        index = work.get("index", 0)
        if index >= len(steps):
            return
        step = steps[index]
        action = step.get("action_id", step.get("resolved_action_id"))
        params = deepcopy(payload.get("params", payload))
        params.pop("task_id", None)
        params.pop("task_ref", None)
        params.pop("rev", None)
        params.update(node_id=self.id, message_id=work["message_id"],
                      seq_id=payload.get("seq_id"), step=step.get("step", index + 1))
        if action == "A036":
            result_payload = self._remote_result_payload(work, "succeeded")
            params.update(receivers=[work["sender"]], kind="result", payload={"op": "job_result", **result_payload})
        self._submit(sim, action, params, purpose="remote_action", work=work)

    @staticmethod
    def _remote_result_payload(work, status):
        return {"job_id": work["payload"].get("job_id", work["message_id"]),
                "task_ref": work["payload"].get("task_ref"), "status": status,
                "seq_id": work["payload"].get("seq_id"),
                "end_index": work["payload"].get("end_index"),
                "results": deepcopy(work.get("results", [])),
                "observations": [o for r in work.get("results", []) for o in r.get("observations", [])],
                "evidence_ids": [e for r in work.get("results", []) for e in r.get("evidence_ids", [])]}

    def _service_result(self, sim, result, context):
        work = context["work"]
        work.setdefault("results", []).append(deepcopy(result))
        purpose = context["purpose"]
        if purpose in {"specialist_action", "specialist_verify"} and _result_cause(result) == "blocked":
            service_task = self.tasks.get(work.get("task_id"))
            if service_task and self._approach_casualty(
                    sim, service_task, result["action_id"], context.get("params", {}), work=work):
                service_task["progress"]["last_failure"] = {
                    "code": "blocked", "action": result["action_id"], "feedback_zh": result.get("feedback_zh", "")}
                return
        if purpose == "specialist_travel":
            site = self._target_point(work["request"].get("params", {}).get("target"))
            radius = 15 if work["request"].get("action_id") in {"A018", "A019"} or context.get("resume_stage") == "verify" else 25
            near_worksite = site is not None and math.dist(self.platform["pose"], site) <= radius
            if result["status"] == "succeeded" or result.get("failure_code") == "blocked" and near_worksite:
                # A blocked approach may honestly stop within tool reach. The
                # failed travel result remains recorded; the tool has its own guard.
                work["stage"] = context.get("resume_stage", "perform")
                return
        if purpose == "specialist_action" and result["action_id"] == "A017" and result["status"] == "succeeded":
            work["stage"] = "verify"
            return
        if purpose == "remote_action" and result["status"] == "succeeded":
            work["index"] = work.get("index", 0) + 1
            steps = work["payload"].get("steps") or [None, None]
            if work["index"] < len(steps):
                return
        payload = {"status": result["status"], "failure_code": result.get("failure_code", "none"),
                   "rejection_code": result.get("rejection_code", "none"),
                   "feedback_zh": result.get("feedback_zh", ""),
                   "observations": result.get("observations", []), "evidence_ids": result.get("evidence_ids", []),
                   "results": work["results"]}
        if purpose.startswith("specialist"):
            payload["request_id"] = work["request_id"]
            payload["observations"] = [o for r in work["results"] for o in r.get("observations", [])]
            payload["evidence_ids"] = list(dict.fromkeys(o["id"] for o in payload["observations"]))
            service_task = self.tasks.get(work.get("task_id"))
            if service_task:
                service_task["progress"]["service_scope"].update(done=True, status=result["status"])
                if result["status"] == "succeeded":
                    self._controls.append(("A054", {"task_id": service_task["id"], "status": "completed"},
                                           service_task["id"], "complete", {}))
                else:
                    service_task["progress"]["last_failure"] = {
                        "code": _result_cause(result), "action": result["action_id"],
                        "feedback_zh": result.get("feedback_zh", "")}
                    self._fail(sim, service_task, _result_cause(result))
                if self.current == service_task["id"]:
                    self.current = None
            self._specialist_work.popleft()
            request_record = self.records["specialist_requests"][work["request_id"]]
            request_record["result"] = deepcopy(payload)
            self._send(sim, [work["sender"]], "specialist_result", payload)
            for listener in request_record.get("listeners", []):
                shared = {**payload, "request_id": listener["request_id"]}
                self.records["specialist_requests"][listener["request_id"]]["result"] = deepcopy(shared)
                self._send(sim, [listener["sender"]], "specialist_result", shared)
        else:
            payload = {**self._remote_result_payload(work, result["status"]),
                       "failure_code": result.get("failure_code", "none"),
                       "rejection_code": result.get("rejection_code", "none")}
            self._remote_work.popleft()
            if result["action_id"] != "A036" or result["status"] != "succeeded":
                self._send(sim, [work["sender"]], "job_result", payload, kind="result")


def build_agents(cfg, world, catalog):
    """Initialization bridge: create platforms and public-only priors, never hidden tasks."""
    requested = cfg.get("agents", "profile")
    if requested == "profile":
        requested = []
        counts = {}
        for item in world.profile.get("agent_mix", []):
            for _ in range(item.get("count", 1)):
                kind = item["kind"]
                counts[kind] = counts.get(kind, 0) + 1
                requested.append({"id": f"{kind}-{counts[kind]}", "kind": kind,
                                  "start": [100 + len(requested) * 5, 100], "overrides": {}})
    agents = []
    seen = set()
    for item in requested:
        aid, kind = item["id"], item["kind"]
        if aid in seen or kind not in EMBODIMENTS:
            raise ValueError("智能体编号重复或构型未知 / duplicate agent or unknown embodiment")
        seen.add(aid)
        spec = deepcopy(EMBODIMENTS[kind])
        spec.update(deepcopy(item.get("overrides", {})))
        spec.update(kind=kind, id=aid, platform_eid=aid)
        start = _point(item.get("start", [100, 100]))
        agent = AgentRuntime(spec, aid, start, catalog)
        fields = {
            "entity.pose": start, "platform.kind": kind,
            "platform.capability_ids": list(spec["capabilities"]), "platform.equipment_ids": list(spec.get('equipment_ids',[])),
            "platform.width_m": spec["width_m"], "platform.payload_capacity_kg": spec["payload_kg"],
            "platform.payload_used_kg": 0, "platform.payload_item_ids": [],
            "platform.battery_capacity_Wh": spec["battery_Wh"],
            "platform.battery_remaining_Wh": spec.get("battery_remaining_Wh", spec["battery_Wh"]),
            "platform.reserve_Wh": spec.get("reserve_Wh", spec["battery_Wh"] * .15),
            "platform.speed_limit_mps": spec["speed_mps"], "platform.motion_state": "idle",
            "platform.health_state": spec.get("health", "nominal"),
            "platform.water_tolerance_m": spec["water_tolerance_m"],
            "platform.current_action_id": None, "platform.busy_until_est_s": 0,
            "platform.concurrency_policy": "serial",
        }
        # Public station geometry gives the platform its announced rendezvous;
        # no live inventory, damage, occupants or hidden objectives are copied.
        fields.update(deepcopy(spec.get('initial_fields',{})))
        stations = [e for e in world.entities.values() if e.get("cls") == "station"]
        if stations:
            def public_position(entity):
                points = entity.get("geometry", {}).get("coords", [[0, 0]])
                return _point(points[0])
            fields["platform.return_site_id"] = min(stations, key=lambda e: math.dist(start, public_position(e)))["id"]
        # Overlay parameters are explicit configuration inputs, not sensed world state.
        if "OV05" in cfg.get("overlays", []):
            fields["platform.battery_remaining_Wh"] *= .22
        if "OV08" in cfg.get("overlays", []) and kind == "specialist_team":
            agent.spec["available"] = False
        if aid in world.entities:
            raise ValueError("平台编号与语义实体冲突 / platform id collides with semantic entity")
        world.entities[aid] = {"id": aid, "cls": "platform", "geometry": {"type": "point", "coords": [start]},
                               "fields": {}, "provenance": {}}
        for key, value in fields.items():
            world.set(aid, key, value, "authored")
        agents.append(agent)
    coordinator = next((a for a in agents if a.is_coordinator), None)
    roster = {a.id: {"id": a.id, "kind": a.kind, "capabilities": list(a.spec["capabilities"])} for a in agents}
    public = [deepcopy(_dict(o)) for o in world.objectives if _dict(o).get("public")]
    for agent in agents:
        agent.coordinator_id = coordinator.id if coordinator else None
        agent.spec["coordinator_id"] = agent.coordinator_id
        agent.roster = deepcopy(roster)  # Public fleet manifest contains no dynamic telemetry.
        agent.records["fleet_manifest"] = deepcopy(roster)
        # The existing public-prior whitelist is role-scoped only at this
        # initialization boundary. The runtime and platform retain their actual
        # embodiment, telemetry, equipment and action capability checks.
        prior_spec = agent.spec
        if agent is coordinator and agent.kind != "coordinator":
            prior_spec = {**agent.spec, "kind": "coordinator"}
        agent.belief = prior_from_world(world, prior_spec, public if agent is coordinator else [])
        for equipment in agent.spec.get('equipment_ids',[]):
            for key,value in world.entities[equipment]['fields'].items():
                if key.startswith('equipment.') or key in {'attr.operator_reach_m','attr.reach_m'}:
                    agent.belief.put(None,equipment,key,value,1.,'prior')
        # These generator-authored entrance/corridor envelopes are public map
        # geometry. Copy only these keys, never other attr.* or hidden state.
        for building in (e for e in world.entities.values() if e['cls'] == 'building' or
                         world.get(e['id'], 'attr.interior_access_via_entrance') is True):
            if building['cls'] != 'building':
                agent.belief.put(None, building['id'], 'building.entrance_ids',
                                 world.get(building['id'], 'building.entrance_ids'), 1., 'prior')
            for key in BUILDING_NAVIGATION_FIELDS:
                value = world.get(building["id"], key)
                if value is not None:
                    agent.belief.put(None, building["id"], key, value, 1., "prior")
        if agent is coordinator:
            agent.public_objectives = deepcopy(public)
            agent.records["public_brief_zh"] = world.profile.get("public_brief_zh", "全区域搜索与建筑评估")
    return agents
