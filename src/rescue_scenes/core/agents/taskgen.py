"""Agent-local, catalog-grounded task proposal rules (Python 3.10, stdlib).

There is deliberately no action submission or runtime loop in this module.
``Rule.when(agent, sim)`` returns ``(target, why_zh, belief_refs, extra)`` tuples;
``candidates``/``generate`` return proposal dictionaries suitable for A050.
The only simulation attribute read is its clock, ``sim.t``.  In particular,
objectives, other agents, compute/network internals and world state are not
policy inputs.  A received public fleet manifest is a capability advertisement,
not permission to inspect another agent.

Local input conventions (all optional except belief, id and spec):
* BeliefStore.get returns an entry, not its value; entities() groups entries.
* platform accepts catalog keys or their short names (battery_Wh is an alias).
* records.action_results contains locally received ActionResult dictionaries.
* records.fleet_manifest maps ids to public {kind, capabilities} advertisements.
* local_jobs, records.services and records.caches are LOCAL resource records.
  They accept catalog keys or short names, e.g. job.status or status.
* records.navigation_goals: [{target, purpose, arrived}]; inventory: {kind: qty}.
* processed_evidence, encoded_evidence, shared_evidence, uploads,
  archived_action_ids and history_queries are local artifact/id tables.
  An absent table is empty, never evidence of a successful operation.
* A public mission is a belief.current_goal_context belief or a DELIVERED
  mission/brief message. Assessment reads public objective records already in
  that local belief/message; the agent.public_objectives shortcut is not read.
* Mobile map priors/briefs provide routing context, not mission-wide proposals.
  Specialist service is runtime-owned; compute nodes propose only local compute
  or service work. Meta tasks never become inputs to task-about-task rules.

Rule priorities and numeric thresholds are proposed, uncalibrated policies.
Names and the Q097--Q112 membership table preserve the original catalog.
A rule proposes work; it does not assert completion, receipt, cure or safety.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from math import hypot, isfinite
from typing import Any, Callable

# F084 human.destination_id domain is "救援/医疗接收点"; mirrors sim.actions.RECEIVING_STATION_KINDS.
RECEIVING_STATION_KINDS = frozenset({"medical", "shelter"})


Trigger = tuple[str, str, list[tuple[str, str]], dict[str, Any]]
ACTIVE = frozenset({"proposed", "assigned", "received", "accepted", "executing",
                    "blocked", "suspended"})
TERMINAL_JOBS = frozenset({"completed", "failed", "canceled", "cancelled"})
UNKNOWN = (None, "unknown", "not_started", "unobserved", "unassessed", "")
# F264 catalog states are unknown/reachable/intermittent/unreachable.
# Additional strings are tolerated only as older local telemetry aliases.
LINK_DOWN = frozenset({"unreachable", "down", "unavailable", "disconnected"})
LINK_WEAK = LINK_DOWN | {"intermittent", "degraded"}
# These are control/information work, not new rescue needs.  All consumers use
# the same classification so a control task cannot generate controls about itself.
META_TASK_TYPES = frozenset(
    {"T007", "T008", "T047", "T090", "T091", "T092", "T096"}
    | {f"T{i:03d}" for i in range(51, 89)}
)
MOBILE_KINDS = frozenset({"uav", "robot_dog", "ground_vehicle", "boat"})
COMPUTE_TASK_TYPES = frozenset({f"T{i:03d}" for i in range(57, 69)} | {"T092", "T096"})
ARCHIVE_INTERNAL_ACTIONS = frozenset(
    {f"A{i:03d}" for i in range(32, 59)} | {"A007", "A011", "A059", "A060", "A061", "A064", "A065", "A066"}
)
ADMIN_EVIDENCE_KINDS = frozenset({
    "history_query", "local_archive", "specialist_request", "compute_submission",
    "abstract_link_probe", "transformed_evidence", "representation_adaptation",
})
# Housekeeping timestamps and accumulated provenance are not changing mission
# facts.  In particular, T075 must not maintain its own updated_at/staleness rows.
HOUSEKEEPING_KEYS = frozenset({
    "belief.updated_at_s", "belief.staleness_s", "belief.evidence_ids",
    "belief.source_agent_ids", "belief.current_goal_context", "belief.team_estimates",
    "belief.conflict_state", "observation.parent_evidence_ids", "observation.processing_job_id",
})
DYNAMIC_FACT_KEYS = frozenset({
    "entity.pose", "human.pose", "human.mobility", "human.injury", "human.needs",
    "human.alive", "human.carrier_id", "human.handover_status", "human.location_context",
    "belief.estimated_pose", "belief.human_presence", "belief.estimated_access",
    "cue.signal_state", "road.entry_state", "road.passability", "obstacle.state",
    "hazard.active", "hazard.fire_intensity", "hazard.gas_level", "hazard.water_depth_m",
})


def is_meta_task(task_or_type):
    """Return whether a type or task record is control/information maintenance.

    Accept a catalog type, a holder task record, or a catalog-qualified task row.
    Task ids are accepted too, without requiring access to any global task table.
    A T087 *target* human/obstacle remains a domain entity, not a meta task.
    """
    if isinstance(task_or_type, str):
        task_type = task_or_type.split("@", 1)[0].split("#", 1)[0]
    else:
        task_type = _get(task_or_type, "type", _get(task_or_type, "task_type", ""))
    return isinstance(task_type, str) and task_type in META_TASK_TYPES


@dataclass(frozen=True)
class Rule(Mapping):
    """Accessible both as ``rule.task_type`` and ``rule['task_type']``."""

    id: str
    task_type: str
    roles: list[str]
    when: Callable[[Any, Any], list[Trigger]]
    priority: int
    refs: list[str]

    def __getitem__(self, key):
        if key not in ("id", "task_type", "roles", "when", "priority", "refs"):
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(("id", "task_type", "roles", "when", "priority", "refs"))

    def __len__(self):
        return 6


def _get(obj, key, default=None):
    """Read plain dictionaries and the shared message/result dataclasses."""
    if type(obj) is dict and key in obj:
        return obj[key]
    if isinstance(obj, Mapping):
        if key in obj:
            return obj[key]
        nested = obj
        for part in key.split("."):
            if not isinstance(nested, Mapping) or part not in nested:
                break
            nested = nested[part]
        else:
            return nested
        short = key.rsplit(".", 1)[-1]
        if short in obj:
            return obj[short]
        # Resource rows may use catalog-qualified keys (job.status, etc.).
        qualified = [value for field, value in obj.items()
                     if isinstance(field, str) and field.endswith("." + short)]
        return qualified[0] if len(qualified) == 1 else default
    return getattr(obj, key.rsplit(".", 1)[-1], default)


def _dict(obj):
    if isinstance(obj, Mapping):
        return dict(obj)
    method = getattr(obj, "to_dict", None)
    return method() if callable(method) else {}


def _rows(table):
    if isinstance(table, Mapping):
        result = []
        for key, value in table.items():
            if type(value) is dict and "id" in value:
                # Rules only read these already-normalized local records.
                result.append(value)
                continue
            row = _dict(value)
            if row:
                row.setdefault("id", str(key))
                result.append(row)
        return result
    if isinstance(table, (list, tuple)):
        result = []
        for value in table:
            row = value if type(value) is dict else _dict(value)
            if row:
                result.append(row)
        return result
    return []


def _list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _num(value, default=0.0):
    try:
        result = float(value)
        return result if isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _tokens(value):
    if isinstance(value, Mapping):
        return {str(k).lower() for k, v in value.items() if v}
    if isinstance(value, str):
        return {part.strip().lower() for part in value.replace(",", " ").split()}
    return {str(part).lower() for part in _list(value)}


def _supply_kind(value):
    value = str(value).lower()
    return {"medical_supplies": "medical_pack", "medkit": "medical_pack",
            "med_kit": "medical_pack", "medicine": "medical_pack"}.get(value, value)


def _xy(value):
    if isinstance(value, Mapping):
        value = value.get("position", value.get("xy", value.get("coords")))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        if isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
            return (_num(value[0]), _num(value[1]))
    return None


def _id(row, fallback=""):
    return str(_get(row, "id") or _get(row, "instance_id") or
               _get(row, "evidence_id") or _get(row, "job_id") or fallback)


class _TaskIndex:
    """Runtime-maintained selections in the task table's original insertion order."""

    def __init__(self):
        self.order = {}
        self.keys = {}
        self.by_status = {}
        self.by_type = {}
        self.selections = {}
        self.meta_revision = 0

    def update(self, task):
        tid = task["id"]
        old = self.keys.get(tid)
        meta = old[1] if old is not None and old[2] == task["type"] else is_meta_task(task)
        key = (task["status"], meta, task["type"])
        self.selections.clear()
        if old is None:
            self.order[tid] = len(self.order)
        if key[1] != (old[1] if old is not None else is_meta_task(tid)):
            self.meta_revision += 1
        if old is not None:
            self.by_status[old[:2]].pop(tid, None)
            self.by_type[old[2]].pop(tid, None)
        self.keys[tid] = key
        self.by_status.setdefault(key[:2], {})[tid] = task
        self.by_type.setdefault(key[2], {})[tid] = task

    def rows(self, statuses, types=None, meta=None):
        key = (frozenset(statuses), None if types is None else frozenset(types), meta)
        if key not in self.selections:
            selected = {}
            for status in statuses:
                for flag in ((False, True) if meta is None else (meta,)):
                    selected.update(self.by_status.get((status, flag), {}))
            self.selections[key] = [selected[tid] for tid in sorted(selected, key=self.order.__getitem__)
                                   if types is None or selected[tid]["type"] in types]
        return self.selections[key]

    def typed(self, typ):
        bucket = self.by_type.get(typ, {})
        return [bucket[tid] for tid in sorted(bucket, key=self.order.__getitem__)]


class _Context:
    """A read-only projection of ONE runtime's knowledge and local records."""

    def __init__(self, agent, sim):
        self.agent = agent
        self.now = _num(getattr(sim, "t", 0))
        self.id = str(getattr(agent, "id", "agent"))
        self.spec = _dict(getattr(agent, "spec", {}))
        self.kind = str(getattr(agent, "kind", self.spec.get("kind", "")))
        self.is_coordinator = self.kind == "coordinator" or self.spec.get("mission_coordinator") is True
        self.platform = _dict(getattr(agent, "platform", {}))
        self.records = _dict(getattr(agent, "records", {}))
        self.task_index = getattr(agent, "_task_index", None)
        if self.task_index is None:
            self.tasks = _rows(getattr(agent, "tasks", {}))
            self.tasks_by_id = {_id(task): task for task in self.tasks}
        else:
            self.tasks_by_id = agent.tasks
            self.tasks = self.tasks_by_id.values()
        self.owned_targets = {str(_get(task, "target")) for task in self.task_rows()
                              if _get(task, "assigned_to") == self.id or _get(task, "accepted_by") == self.id}
        self.history = getattr(agent, "_taskgen_history", {})
        self.received = self._history_rows("received", getattr(agent, "received_messages", []))
        self.belief = getattr(agent, "belief", None)
        # Group the documented local entries without making 96 deep copies per
        # generation pass.  Entries are only read; no references escape a call.
        entries = getattr(self.belief, "entries", None)
        self.entity_index = getattr(self.belief, "entity_ids", None)
        if self.entity_index is not None:
            self.entities = self.belief.entities(copy=False)
        elif isinstance(entries, Mapping):
            self.entities = {}
            for pair, entry in entries.items():
                if isinstance(pair, tuple) and len(pair) == 2:
                    self.entities.setdefault(str(pair[0]), {})[pair[1]] = entry
        else:
            method = getattr(self.belief, "entities", None)
            self.entities = method() if callable(method) else {}
            if not isinstance(self.entities, Mapping):
                self.entities = {}
        self.services = _rows(self.records.get("services", {}))
        self.caches = _rows(self.records.get("caches", {}))
        self.fleet = [dict(self.spec, id=self.id)]
        self.fleet.extend(_rows(self.records.get("fleet_manifest", {})))
        self.fleet.extend(self.received_index["fleet"])
        for eid in (self.entity_index("platform") if self.entity_index is not None else self.entities):
            caps = self.v(eid, "platform.capability_ids")
            if caps:
                self.fleet.append({"id": eid, "capabilities": caps})

    def _history_rows(self, name, table):
        # Runtime result/message histories append detached records. Lightweight
        # callers still accept the documented mapping/list/dataclass inputs.
        if not hasattr(self.agent, "_taskgen_history") or not isinstance(table, list):
            return _rows(table)
        state = self.history.get(name)
        if state is None or state[0] is not table or state[1] > len(table):
            state = [table, 0, []]
            self.history[name] = state
        state[2].extend(_rows(table[state[1]:]))
        state[1] = len(table)
        return state[2]

    @cached_property
    def received_index(self):
        state = self.history.get("received_index")
        if state is None or state["rows"] is not self.received:
            state = {"rows": self.received, "cursor": 0, "fleet": [], "mission": [],
                     "objectives": [], "brief": [], "bids": [], "relay": [],
                     "forward": [], "damaged": [], "cancel": {}}
            self.history["received_index"] = state
        for message in self.received[state["cursor"]:]:
            payload = _get(message, "payload", {})
            kind = _get(message, "kind")
            is_mapping = isinstance(payload, Mapping)
            if kind in ("mission", "brief") or is_mapping and (
                    payload.get("op") in ("mission", "brief") or payload.get("brief") or payload.get("mission")):
                state["mission"].append(message)
            if kind in ("mission", "brief") or is_mapping and payload.get("op") in ("mission", "brief"):
                state["brief"].append(message)
            if not is_mapping:
                continue
            manifest = payload.get("fleet_manifest", payload.get("manifest", payload.get("roster")))
            if manifest is not None:
                state["fleet"].extend(_rows(manifest))
            advertised = payload.get("spec", payload.get("telemetry", {}))
            if isinstance(advertised, Mapping) and (
                    "capabilities" in advertised or "platform.capability_ids" in advertised):
                state["fleet"].append(dict(advertised))
            if payload.get("objectives"):
                state["objectives"].append(message)
            if payload.get("call_for_bids") or payload.get("command") == "bid" or payload.get("op") == "bid_request":
                state["bids"].append(message)
            if payload.get("relay_needed") or payload.get("relay_handoff"):
                state["relay"].append(message)
            if payload.get("forward_to", payload.get("next_hop")):
                state["forward"].append(message)
            if payload.get("checksum_ok") is False or payload.get("missing_fragments"):
                state["damaged"].append(message)
            if payload.get("command", payload.get("op")) in ("cancel", "preempt"):
                state["cancel"].setdefault(str(payload.get("task_id")), []).append(message)
        state["cursor"] = len(self.received)
        return state

    @cached_property
    def business_messages(self):
        return [message for message in self.sent if self.received_domain_message(message)]

    # Mutable task/belief selections last one generation pass; detached append-only
    # message and result projections retain cursors across passes.
    @cached_property
    def jobs(self):
        return [job for job in _rows(getattr(self.agent, "local_jobs", {}))
                if not self.meta_ref(_get(job, "task_ref"))]

    @cached_property
    def sent(self):
        return _rows(getattr(self.agent, "sent_messages", {}))

    @cached_property
    def results(self):
        return self._history_rows("results", self.records.get("action_results", []))

    @cached_property
    def result_index(self):
        rows = self.results
        revision = self.task_index.meta_revision if self.task_index is not None else None
        state = self.history.get("result_index")
        if state is None or state["rows"] is not rows or state["revision"] != revision:
            state = {"rows": rows, "revision": revision, "cursor": 0,
                     "failures": {}, "actions": {}, "admin_ids": set(), "domain": []}
            self.history["result_index"] = state
        for index in range(state["cursor"], len(rows)):
            row = rows[index]
            action = _get(row, "action_id")
            state["actions"].setdefault(action, []).append(row)
            for code in {_get(row, "failure_code"), _get(row, "rejection_code")}:
                state["failures"].setdefault(code, {})[index] = row
            if self.meta_ref(_get(row, "task_ref")) or action in ARCHIVE_INTERNAL_ACTIONS:
                # Result evidence_ids may name inputs; only actual observations
                # emitted by administrative actions are administrative evidence.
                ids = {_id(obs) for obs in _rows(_get(row, "observations", []))}
                dirty = getattr(self.belief, "_taskgen_evidence_dirty", None)
                if dirty is not None:
                    dirty.update((oid, None) for oid in ids - state["admin_ids"])
                state["admin_ids"].update(ids)
            else:
                state["domain"].append(row)
        state["cursor"] = len(rows)
        return state

    @cached_property
    def admin_evidence_ids(self):
        return self.result_index["admin_ids"]

    @cached_property
    def evidence_index(self):
        dirty = getattr(self.belief, "_taskgen_evidence_dirty", None)
        if self.task_index is None or dirty is None:
            return None
        # Consume result updates first: newly received administrative observations
        # can reclassify evidence that arrived before the corresponding result.
        self.admin_evidence_ids
        table = self.belief.evidence
        revision = self.task_index.meta_revision
        state = self.history.get("evidence_index")
        if state is None or state["table"] is not table or state["revision"] != revision:
            state = {"table": table, "revision": revision, "rows": {},
                     "targets": {}, "grouped": {}, "ordered": None}
            self.history["evidence_index"] = state
            dirty.update((oid, None) for oid in table)
        changed_targets = set()
        for oid in dirty:
            old = state["rows"].pop(oid, None)
            if old is not None:
                for target in _list(_get(old, "about", [])):
                    target = str(target)
                    state["targets"][target].pop(oid, None)
                    changed_targets.add(target)
            rows = _rows({oid: table[oid]}) if oid in table else []
            row = rows[0] if rows else None
            if row is not None and not self.admin_evidence(row) and (
                    self.kind not in MOBILE_KINDS or
                    _get(row, "by", _get(row, "from_agent", self.id)) == self.id):
                state["rows"][oid] = row
                for target in _list(_get(row, "about", [])):
                    target = str(target)
                    state["targets"].setdefault(target, {})[oid] = row
                    changed_targets.add(target)
            if old is not None or oid in state["rows"]:
                state["ordered"] = None
        order = self.belief._evidence_order.__getitem__
        for target in changed_targets:
            bucket = state["targets"][target]
            # Preserve even repeated about entries from the original projection.
            state["grouped"][target] = [row for oid in sorted(bucket, key=order)
                for row in [bucket[oid]] for item in _list(_get(row, "about", []))
                if str(item) == target]
        dirty.clear()
        return state

    @cached_property
    def evidence(self):
        state = self.evidence_index
        if state is not None:
            if state["ordered"] is None:
                state["ordered"] = [state["rows"][oid] for oid in sorted(
                    state["rows"], key=self.belief._evidence_order.__getitem__)]
            return state["ordered"]
        return [row for row in _rows(getattr(self.belief, "evidence", {}))
                if not self.admin_evidence(row) and (self.kind not in MOBILE_KINDS or
                    _get(row, "by", _get(row, "from_agent", self.id)) == self.id)]

    @cached_property
    def evidence_by_target(self):
        if self.evidence_index is not None:
            return self.evidence_index["grouped"]
        grouped = {}
        for row in self.evidence:
            for target in _list(_get(row, "about", [])):
                grouped.setdefault(str(target), []).append(row)
        return grouped

    @cached_property
    def conflicts(self):
        conflicts = getattr(self.belief, "conflicts", None)
        return _rows(conflicts()) if callable(conflicts) else []

    def meta_ref(self, ref):
        if isinstance(ref, Mapping):
            if is_meta_task(ref):
                return True
            task_id = ref.get("task_id", ref.get("id"))
        elif isinstance(ref, (list, tuple)):
            task_id = ref[0] if ref else None
        else:
            task_id = ref
        key = str(task_id)
        cache = self.__dict__.setdefault("_meta_refs", {})
        if key not in cache:
            cache[key] = is_meta_task(self.tasks_by_id.get(key, task_id))
        return cache[key]

    def admin_evidence(self, row):
        if (_id(row) in self.admin_evidence_ids or
                _get(row, "kind") in ADMIN_EVIDENCE_KINDS or self.meta_ref(_get(row, "task_ref"))):
            return True
        about = _list(_get(row, "about", []))
        if about and all(self.meta_ref(eid) for eid in about):
            return True
        values = _get(row, "values", {})
        if isinstance(values, Mapping) and values:
            keys = [str(key) for key in values]
            return all(key in HOUSEKEEPING_KEYS or key.startswith((
                "task.", "action.", "assignment.", "message.", "job.", "service.", "cache.", "link.", "attr.archive"))
                       for key in keys)
        return False

    def local_entry(self, entry):
        return (_get(entry, "source") in ("observation", "camera_image", "inference") and
                _get(entry, "from_agent") in (None, "", self.id))

    def relevant_entity(self, eid):
        """A mobile agent's map/brief is navigation context, not a task backlog."""
        eid = str(eid)
        cache = self.__dict__.setdefault("_relevant_cache", {})
        if eid not in cache:
            cache[eid] = self._relevant_entity(eid)
        return cache[eid]

    def _relevant_entity(self, eid):
        if self.meta_ref(eid):
            return False
        if self.is_coordinator:
            return True
        if eid in (self.id, str(getattr(self.agent, "platform_eid", self.id))) or eid in self.owned_targets:
            return True
        if self.kind == "compute_node":
            return False
        return any(key not in HOUSEKEEPING_KEYS and self.local_entry(entry)
                   for key, entry in self.entities.get(eid, {}).items())

    def fact_keys(self, eid):
        """Substantive domain facts only, never task/protocol maintenance rows."""
        if not self.relevant_entity(eid):
            return []
        return [str(key) for key in self.entities.get(eid, {})
                if key not in HOUSEKEEPING_KEYS and str(key).startswith((
                    "human.", "building.", "road.", "obstacle.", "hazard.", "cue.", "item.",
                    "belief.estimated_", "belief.human_presence", "belief.search_status", "observation.detections"))]

    def role_allows(self, number):
        if self.kind == "specialist_team":
            # A public brief alone is still not task evidence. Locally observed
            # clearance/loading/treatment needs and the actual carrier's onward
            # transport/handover use ordinary proposal/capability checks. A
            # mission may explicitly add onsite T027 contact; other specialist
            # services remain A059-scoped.
            return (number in (22, 23, 31, 33, 34, 35)
                    or number == 27 and "T027" in (self.spec.get("task_allowlist") or ()))
        task_type = f"T{number:03d}"
        if self.kind == "compute_node":
            return task_type in COMPUTE_TASK_TYPES
        if number in (1, 2, 3, 78, 80):
            return self.is_coordinator
        if number == 32:
            # Discoverers request T087.  The centre creates a single domain
            # extrication task, whose specialist action is still routed via A059.
            return self.is_coordinator or {"C18", "C19", "C20"}.issubset(self.caps())
        return self.is_coordinator or self.kind in MOBILE_KINDS

    def entry(self, eid, key):
        group = self.entities.get(str(eid), {})
        if key in group:
            return group[key]
        method = getattr(self.belief, "get", None)
        return method(str(eid), key) if callable(method) else None

    def v(self, eid, key, default=None):
        entry = self.entry(eid, key)
        return _get(entry, "value", default) if entry is not None else default

    def confidence(self, eid, key):
        return _num(_get(self.entry(eid, key), "confidence", 0))

    def age(self, eid, key):
        entry = self.entry(eid, key)
        return max(0.0, self.now - _num(_get(entry, "t", self.now)))

    def refs(self, eid, *keys):
        return [(str(eid), key) for key in keys]

    def is_kind(self, eid, kind):
        group = self.entities.get(eid, {})
        declared = self.v(eid, "entity.kind", self.v(eid, "entity.class"))
        if declared == kind:
            return True
        # A known field is evidence of the class; names/hidden ids are not.
        return any(str(key).startswith(kind + ".") for key in group)

    def ids(self, *kinds):
        # Per-pass memo (a _Context lives for one generation pass over unchanged beliefs).
        cache = self.__dict__.setdefault("_ids_cache", {})
        if kinds not in cache:
            candidates = self.entity_index(*kinds) if self.entity_index is not None else self.entities
            cache[kinds] = [eid for eid in sorted(candidates) if self.relevant_entity(eid)
                            and not (eid.startswith('rescue-camera/') and '/person/' in eid
                                     and self.v(eid, 'belief.entity_ref', eid) != eid)
                            and any(self.is_kind(eid, kind) for kind in kinds)]
        return list(cache[kinds])

    def camera_candidate(self, eid):
        """A projected local track is not an established human identity."""
        return (str(eid).startswith('rescue-camera/') and '/person/' in str(eid)
                and self.v(eid, 'belief.entity_ref', eid) == eid)

    def camera_contact_evidence(self, eid):
        """Delivered detections associated to this track, including coalesced rows."""
        cache = self.__dict__.setdefault("_camera_contact_evidence", {})
        if eid not in cache:
            reports = {_id(row): row for row in self.evidence_about(eid)}
            # Mobile proposal scans are own-only, but a received assignment or
            # presence report can carry a later camera capture from its observer.
            # Preserve that existing retry path using only delivered evidence.
            table = getattr(self.belief, "evidence", {})
            for oid in _list(_get(self.entry(eid, "belief.human_presence"), "evidence_ids", [])):
                if oid in table:
                    row = _dict(table[oid])
                    if eid in _list(_get(row, "about", [])):
                        reports[oid] = row
            rows = []
            for row in reports.values():
                values = _get(row, "values", {})
                detection = _get(values, "observation.image_detection", {})
                if (_get(row, "source") == "camera_image" and
                        str(_get(detection, "label", "")).lower() in {"person", "human"} and
                        _get(values, "observation.association") and
                        any(str(parent).endswith("/image") for parent in
                            _list(_get(values, "observation.parent_evidence_ids", [])))):
                    rows.append(row)
            cache[eid] = rows
        return cache[eid]

    def camera_contact_priority(self, eid):
        """Two captures of one local track outrank survey, not establish identity."""
        captures = {}
        for row in reversed(self.camera_contact_evidence(eid)):
            reporter = _get(row, "by")
            if not reporter:
                continue
            images = captures.setdefault(reporter, set())
            images.update(str(parent) for parent in _list(_get(
                _get(row, "values", {}), "observation.parent_evidence_ids", []))
                if str(parent).endswith("/image"))
            # Raw/associated copies, duplicate boxes and delivered replays of
            # one capture count once. Own captures and presence-linked reports
            # still need two distinct captures by the same reporting agent.
            if len(images) >= 2:
                return 81  # authored above T002=70 and T014=70/80
        return 64

    @cached_property
    def camera_contact_failures(self):
        """Holder-local failed contacts, not evidence that a person is absent."""
        failures, recorded = [], set()
        for result in self.result_index["actions"].get("A009", ()):
            if _get(result, "failure_code") != "sensor":
                continue
            params = _get(result, "params", {})
            target = str(_get(params, "camera_candidate_id", _get(params, "target", "")))
            ref = _get(result, "task_ref")
            tid = str(ref[0]) if isinstance(ref, (list, tuple)) and ref else str(_get(params, "task_id", ""))
            task = self.tasks_by_id.get(tid, {})
            if _get(task, "status") == "completed":
                continue  # a recovered contact is not a permanent failed site
            history = _rows(_get(task, "history", []))
            # ActionResult has no terminal timestamp; the received task history
            # supplies it. Without one, do not manufacture an elapsed cooldown.
            terminal = [row for row in history if _get(row, "status") in ("failed", "canceled")]
            when = max((_num(_get(row, "t")) for row in terminal), default=None)
            failures.append((target, _xy(_get(params, "contact_position")) or self.pose(target), when, tid))
            recorded.add(tid)
        for task in self.typed_tasks("T027"):
            if (_id(task) not in recorded and _get(task, "type") == "T027" and
                    _get(task, "status") in ("failed", "canceled") and
                    _get(task, "progress.last_failure.code") == "sensor" and
                    _get(task, "progress.last_failure.action") == "A009"):
                history = _rows(_get(task, "history", []))
                when = max((_num(_get(row, "t")) for row in history
                            if _get(row, "status") in ("failed", "canceled")), default=None)
                target = str(_get(task, "target"))
                failures.append((target, self.pose(target), when, _id(task)))
        return failures

    def camera_contact_retry(self, eid):
        """Shared site cooldown plus genuinely later acquired camera evidence."""
        if not self.camera_candidate(eid):
            return []
        policy = self.spec.get("camera_search_policy", {})
        cooldown = max(0., _num(policy.get("contact_cooldown_s", 120)))
        radius = max(0., _num(policy.get("contact_radius_m", 3)))
        point, refs = self.pose(eid), []
        # Coalescing is authored inference, but its parent image detection is
        # still acquired camera evidence. Do not use integration/report time.
        observed = max((_num(_get(row, "t"), -1) for row in
                        self.camera_contact_evidence(eid)), default=-1)
        for target, site, failed_at, tid in self.camera_contact_failures:
            nearby = point and site and hypot(point[0] - site[0], point[1] - site[1]) <= radius
            if target != eid and not nearby:
                continue
            if (failed_at is None or self.now < failed_at + cooldown or
                    observed <= failed_at):
                return None
            refs.extend(self.refs(tid, "task.status", "task.revision"))
        return refs

    def repeat_search(self, eid):
        """A completed camera pass is not measured regional coverage (Q009).

        This optional policy spends an authored time budget, not a catalog
        mandate to retry. Only the coordinator schedules follow-up passes.
        """
        policy = self.spec.get("camera_search_policy", {})
        if (not self.is_coordinator or not isinstance(policy, Mapping) or
                self.now >= _num(policy.get("repeat_budget_s")) or not self.mission()):
            return None
        # Respect explicit delivered mission/search closure. Hidden evaluator
        # objectives and the mere absence of further candidates are not inputs.
        goals = []
        for context in self.entities:
            for key in ("belief.current_goal_context", "objective.public_record", "public_objectives"):
                value = self.v(context, key)
                rows = _list(value) if not isinstance(value, Mapping) else [value]
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    if row.get("status") in ("completed", "canceled", "cancelled") and row.get("kind") in ("mission", "search"):
                        return None
                    if row.get("kind", row.get("type")) == "find_person":
                        goals.append(row)
        if goals and all(row.get("status") in ("completed", "canceled", "cancelled") or
                         any(_get(obs, "kind") == "human_response" for obs in
                             self.evidence_about(row.get("target", row.get("target_id")))) for row in goals):
            return None
        if self.v(eid, "belief.search_status") not in UNKNOWN + ("partial", "inconclusive"):
            return None
        passes = [task for task in self.typed_tasks("T009") if _get(task, "type") == "T009" and
                  _get(task, "target") == eid and _get(task, "status") == "completed" and
                  any(str(ref).startswith("rescue-camera/") and str(ref).endswith("/image")
                      for ref in _list(_get(task, "result_evidence_ids", [])))]
        if not passes or self.targeting(eid, {"T001", "T009"}):
            return None
        def finished(task):
            return max((_num(_get(row, "t"), -1) for row in _rows(_get(task, "history", []))
                        if _get(row, "status") == "completed"), default=-1)
        latest = max(passes, key=lambda task: (finished(task), _id(task)))
        ended = finished(latest)
        if ended < 0 or self.now < ended + max(0., _num(policy.get("repeat_cooldown_s"))):
            return None
        # The terminal task reference changes the runtime's value signature once
        # per pass; periodic frames/timestamps must not reset a retry episode.
        return self.refs(_id(latest), "task.status", "task.revision"), len(passes)

    def own(self, key, default=None):
        if key in self.platform:
            return self.platform[key]
        short = key.rsplit(".", 1)[-1]
        aliases = {
            "battery_remaining_Wh": ("battery_remaining_Wh", "battery_Wh"),
            "battery_capacity_Wh": ("battery_capacity_Wh", "battery_max_Wh", "battery_Wh_max"),
            "health_state": ("health_state", "health"),
            "pose": ("pose", "position"),
        }
        for alias in aliases.get(short, (short,)):
            if alias in self.platform:
                return self.platform[alias]
        value = self.v(self.id, key)
        if value is None:
            platform_eid = str(getattr(self.agent, "platform_eid", self.id))
            value = self.v(platform_eid, key)
        return default if value is None else value

    def battery(self):
        maximum = _num(self.own("platform.battery_capacity_Wh",
                                self.spec.get("battery_Wh", 0)))
        remaining = _num(self.own("platform.battery_remaining_Wh", maximum))
        reserve = _num(self.own("platform.reserve_Wh", maximum * 0.15))
        return remaining, maximum, reserve

    def low(self, fraction=0.25):
        remaining, maximum, reserve = self.battery()
        return maximum > 0 and remaining <= max(reserve, fraction * maximum)

    def caps(self):
        return set(_list(self.spec.get("capabilities", [])))

    def available(self, alternatives):
        """One advertised provider must have one full capability alternative."""
        if not alternatives:
            return True
        for provider in self.fleet:
            caps = set(_list(provider.get("capabilities", provider.get("platform.capability_ids", []))))
            if any(set(option).issubset(caps) for option in alternatives):
                return True
        return False

    def mission(self):
        return self._mission

    @cached_property
    def _mission(self):
        for eid in self.entities:
            value = self.v(eid, "belief.current_goal_context")
            if value not in UNKNOWN and value not in ({}, []):
                return self.refs(eid, "belief.current_goal_context")
            # The initial public brief is a LOCAL prior, not a world objective.
            if self.v(eid, "public_brief") or self.v(eid, "mission.public_brief"):
                return self.refs(eid, "public_brief")
        for message in self.received_index["mission"]:
            payload = _get(message, "payload", {})
            kind = _get(message, "kind")
            if kind in ("mission", "brief") or (isinstance(payload, Mapping) and
                    (payload.get("op") in ("mission", "brief") or payload.get("brief") or payload.get("mission"))):
                return self.refs(_id(message, self.id), "message.local_availability")
        return []

    def disputed(self, eid):
        facts = set(self.fact_keys(eid))
        if not facts:
            return False
        conflicts = [row for row in self.conflicts if _get(row, "eid") == eid]
        if conflicts:
            return any(_get(row, "key") in facts and _get(row, "state") in ("disputed", "insufficient")
                       for row in conflicts)
        return self.v(eid, "belief.conflict_state") in ("disputed", "insufficient")

    def conflict_refs(self, eid):
        facts = set(self.fact_keys(eid))
        refs = self.refs(eid, "belief.conflict_state")
        refs.extend((str(eid), str(_get(row, "key"))) for row in self.conflicts
                    if _get(row, "eid") == eid and _get(row, "key") in facts)
        return list(dict.fromkeys(refs))

    def task_rows(self, statuses=ACTIVE, types=None):
        # Never reason about a control task. Selections are stable within a pass;
        # runtime indexes avoid visiting completed control history at all.
        key = (frozenset(statuses), None if types is None else frozenset(types))
        cache = self.__dict__.setdefault("_task_rows", {})
        if key not in cache:
            if self.task_index is not None:
                cache[key] = self.task_index.rows(statuses, types, meta=False)
            else:
                cache[key] = [task for task in self.tasks if not is_meta_task(task)
                              and _get(task, "status") in statuses
                              and (types is None or _get(task, "type") in types)]
        return cache[key]

    def typed_tasks(self, typ):
        if self.task_index is not None:
            return self.task_index.typed(typ)
        return [task for task in self.tasks if _get(task, "type") == typ]

    @cached_property
    def active_by_target(self):
        grouped = {}
        for task in self.task_rows():
            grouped.setdefault(_get(task, "target"), []).append(task)
        return grouped

    def targeting(self, eid, types=None):
        return [task for task in self.active_by_target.get(eid, ())
                if types is None or _get(task, "type") in types]

    def marked(self, table, key):
        value = self.records.get(table, {})
        if isinstance(value, Mapping):
            return bool(value.get(key))
        return key in _list(value)

    def pose(self, eid):
        if str(eid) in (self.id, str(getattr(self.agent, "platform_eid", self.id))):
            return _xy(self.own("entity.pose"))
        return _xy(self.v(eid, "entity.pose", self.v(eid, "human.pose",
                   self.v(eid, "belief.estimated_pose"))))

    def distant(self, eid, threshold=15):
        start, end = _xy(self.own("entity.pose")), self.pose(eid)
        return bool(start and end and hypot(start[0] - end[0], start[1] - end[1]) > threshold)

    def possible_occupants(self, eid):
        presence = self.v(eid, "belief.human_presence")
        if presence in (True, "possible", "suspected", "present", "likely", "uncertain"):
            return True
        if isinstance(presence, Mapping) and (
                _num(presence.get("present")) + _num(presence.get("uncertain")) > 0.05):
            return True
        # Occupancy prior is independent of damage (G02), including D0 buildings.
        return (_num(self.v(eid, "building.capacity")) > 0 and
                self.v(eid, "building.use") not in ("vacant", "abandoned", "empty"))

    def needs(self, eid):
        return {_supply_kind(value) for value in
                _tokens(self.v(eid, "human.needs", self.v(eid, "belief.estimated_needs", [])))}

    def supported(self, eid):
        return {_supply_kind(value) for value in _tokens(self.v(eid, "human.support_received", []))}

    def result_failure(self, *codes):
        cache = self.__dict__.setdefault("_result_failures", {})
        if codes not in cache:
            selected = {}
            for code in codes:
                selected.update(self.result_index["failures"].get(code, {}))
            cache[codes] = [selected[index] for index in sorted(selected)
                            if not self.meta_ref(_get(selected[index], "task_ref"))]
        return list(cache[codes])

    def blocked(self):
        return self._blocked

    @cached_property
    def _blocked(self):
        result = self.result_failure("blocked")
        failure = self.records.get("path_failure")
        if isinstance(failure, Mapping) and _get(failure, "failure_code") == "blocked":
            result.append(dict(failure))
        tasks = [task for task in self.task_rows() if _get(task, "status") == "blocked" and
                 _get(task, "failure_reason") in ("blocked", "access", "path", "obstacle")]
        return result + tasks

    def obstructing(self, eid):
        cache = self.__dict__.setdefault("_obstructing_cache", {})
        if eid not in cache:
            cache[eid] = self._obstructing(eid)
        return cache[eid]

    def _obstructing(self, eid):
        if self.v(eid, "obstacle.state") in ("cleared", "moved"):
            return False
        blocked_ids = set(str(v) for v in _list(self.v(eid, "obstacle.blocks_ids", [])))
        for row in self.blocked():
            params = _get(row, "params", {})
            if not isinstance(params, Mapping):
                params = {}
            named = _list(params.get("blocked_by")) + _list(row.get("blocked_by"))
            if eid in named:
                return True
            if _get(row, "target", params.get("target")) in blocked_ids:
                return True
            for observation in _rows(_get(row, "observations", [])):
                if eid in _list(_get(observation, "about", [])):
                    return True
        # A received report can establish that a needed entrance is obstructed.
        # The motion goal may be on a different road beyond the blocked edge;
        # requiring a task whose target IS that edge loses the reported cause.
        # Use only a delivered actuator observation and the current local link
        # and state, never the world's road graph or an old cleared obstacle.
        if (self.v(eid, "obstacle.state") in ("present", "partly_cleared")
                and any(self.v(road, "road.entry_state") == "blocked" for road in blocked_ids)
                and any(_get(obs, "model") == "scripted_motion_blockage_v1"
                        for obs in self.evidence_by_target.get(str(eid), []))):
            return True
        for target in blocked_ids:
            if self.targeting(target) and self.v(target, "road.entry_state") == "blocked":
                return True
        return False

    def nearby_risk(self, eid):
        point = self.pose(eid)
        region = self.v(eid, "entity.region_id")
        for hazard in self.ids("hazard"):
            if self.v(hazard, "hazard.active") is False:
                continue
            active = (self.v(hazard, "hazard.active") is True or
                      _num(self.v(hazard, "hazard.fire_intensity")) > 0 or
                      _num(self.v(hazard, "hazard.gas_level")) > 0 or
                      _num(self.v(hazard, "hazard.water_depth_m")) > 0)
            if not active:
                continue
            other, other_region = self.pose(hazard), self.v(hazard, "entity.region_id")
            if region and region == other_region:
                return hazard
            if point and other and hypot(point[0] - other[0], point[1] - other[1]) < 150:
                return hazard
        return None

    def job_stale(self, job):
        ref = _get(job, "task_ref")
        if isinstance(ref, Mapping):
            task_id = ref.get("task_id", ref.get("id"))
            revision = ref.get("revision", ref.get("rev"))
        elif isinstance(ref, (list, tuple)) and len(ref) >= 2:
            task_id, revision = ref[:2]
        else:
            return False
        task = self.tasks_by_id.get(str(task_id))
        return bool(task and revision is not None and
                    _num(revision) < _num(_get(task, "rev", _get(task, "revision", 1))))

    def evidence_about(self, eid):
        return self.evidence_by_target.get(str(eid), [])

    @cached_property
    def _objective_rows(self):
        # L27perf12: one read-only projection per generation pass, not one full
        # belief/message scan per assessed building. Preserve source/row order
        # and the original belief-only coercion of objective kinds to strings.
        objectives = []
        for context_eid in self.entities:
            for key in ("objective.public_record", "public_objectives", "belief.current_goal_context"):
                value = self.v(context_eid, key)
                rows = _list(value) if not isinstance(value, Mapping) else [value]
                for row in rows:
                    if isinstance(row, Mapping):
                        objectives.append((str(row.get("kind", row.get("type", ""))),
                            row.get("target", row.get("target_id")), context_eid, key))
        for message in self.received_index["objectives"]:
            payload = _get(message, "payload", {})
            if not isinstance(payload, Mapping):
                continue
            for row in _rows(payload.get("objectives", [])):
                objectives.append((row.get("kind", row.get("type")),
                    row.get("target", row.get("target_id")), _id(message), "message.local_availability"))
        return objectives

    def objective_refs(self, eid, *kinds):
        """Match only a locally delivered/public objective, not an evaluator list."""
        cache = self.__dict__.setdefault("_objectives_by_kinds", {})
        if kinds not in cache:
            cache[kinds] = [(target, context_eid, key)
                           for kind, target, context_eid, key in self._objective_rows if kind in kinds]
        for target, context_eid, key in cache[kinds]:
            if target in (None, "*", "all", eid) or eid in _list(target):
                return self.refs(context_eid, key)
        return []

    def survey_refs(self):
        # L27perf12: the brief is unchanged throughout this _Context's pass.
        return list(self._survey_refs)

    @cached_property
    def _survey_refs(self):
        """Recognize a local public regional-assessment brief, not hidden goals."""
        def regional_survey(value):
            if not isinstance(value, str):
                return False
            text = value.lower()
            # A public brief ordering a search over a named area (区域…搜索) is a
            # regional search request; it reveals no casualty location or count.
            return ("survey" in text or "regional assessment" in text or "普查" in text or
                    "区域" in text and any(word in text for word in ("评估", "侦察", "勘查", "搜索")))

        for eid in self.entities:
            for key in ("public_brief", "mission.public_brief"):
                if regional_survey(self.v(eid, key)):
                    return self.refs(eid, key)
        for message in self.received_index["brief"]:
            payload = _get(message, "payload", {})
            if not isinstance(payload, Mapping):
                continue
            if _get(message, "kind") not in ("mission", "brief") and payload.get("op") not in ("mission", "brief"):
                continue
            if any(regional_survey(payload.get(key)) for key in ("brief", "brief_zh", "mission")):
                return self.refs(_id(message), "message.local_availability")
        return []

    def destination(self, eid):
        known = self.v(eid, "human.destination_id")
        if (known and self.is_kind(str(known), "station") and self.v(str(known), "station.open_state") not in ("closed", "offline")
                and self.v(str(known), "station.kind") in RECEIVING_STATION_KINDS):
            return str(known)
        # Static public station geometry is valid routing context for an owned
        # transport need, even though it cannot itself trigger mission proposals.
        # Only a known compatible receiver (F084 救援/医疗接收点), never supply/charging.
        stations = [target for target in self.entities if self.is_kind(target, "station")
                    and self.v(target, "station.kind") in RECEIVING_STATION_KINDS
                    and self.v(target, "station.open_state") not in ("closed", "offline")
                    and (self.v(target, "station.capacity") is None or
                         _num(self.v(target, "station.occupied")) < _num(self.v(target, "station.capacity")))]
        start = self.pose(eid) or self.pose(self.id)
        return min(stations, key=lambda target: (
            hypot(start[0] - self.pose(target)[0], start[1] - self.pose(target)[1])
            if start and self.pose(target) else float("inf"), target), default=None)

    def received_domain_message(self, message):
        """Protocol reliability may serve real business, never its own controls."""
        if self.meta_ref(_get(message, "task_ref")):
            return False
        payload = _get(message, "payload", {})
        if not isinstance(payload, Mapping):
            return False
        if _get(message, "retry_root") or payload.get("retry_root") or _get(message, "kind") == "ack":
            return False
        task = payload.get("task", {})
        task_ref = payload.get("task_ref", payload.get("task_id"))
        if is_meta_task(task) or self.meta_ref(task_ref):
            return False
        op = payload.get("op", _get(message, "op", ""))
        return op not in {"ack", "retry", "retransmit", "resend", "progress_ack", "specialist_decision",
                          "decision", "bid", "bid_request", "handoff_ready", "handoff_committed", "handoff_release"}

    def comms_backlog(self):
        return self._comms_backlog

    @cached_property
    def _comms_backlog(self):
        """Return only this holder's outstanding rescue reports/A059 requests."""
        reports = []
        for message in self.sent:
            payload = _get(message, "payload", {})
            if not isinstance(payload, Mapping) or payload.get("op", _get(message, "op")) != "report":
                continue
            if not self.received_domain_message(message) or _get(message, "ack_state") not in ("pending", "timeout"):
                continue
            artifacts = _rows(payload.get("evidence", [])) + _rows(payload.get("observations", []))
            entries = _rows(payload.get("entries", []))
            if artifacts or entries:
                domain_entries = any(
                    _get(entry, "key") not in HOUSEKEEPING_KEYS and
                    (str(_get(entry, "key", "")).startswith(("human.", "building.", "road.", "obstacle.", "hazard.", "cue.", "item.")) or
                     _get(entry, "key") in ("entity.pose", "entity.parent_id") and _get(entry, "eid") not in (self.id, str(getattr(self.agent, "platform_eid", self.id))))
                    for entry in entries)
                if not domain_entries and not any(not self.admin_evidence(row) for row in artifacts):
                    continue  # an ACK/probe/archive/diagnostic report is not rescue backlog
            age = max(0., self.now - _num(_get(message, "sent_s", self.now)))
            ttl = _get(message, "ttl_s")
            if ttl is not None and age >= _num(ttl):
                continue
            reports.append(message)
        # A059 is transported by its action model, not runtime._send, so an
        # unanswered local service contract need not occur in sent_messages.
        table = self.records.get("specialist_wait", getattr(self.agent, "_specialist_wait", {}))
        requests = []
        for request in _rows(table):
            if _get(request, "accepted") or _get(request, "status") in ("completed", "failed", "canceled", "rejected"):
                continue
            payload = _get(request, "payload", {})
            if not isinstance(payload, Mapping):
                continue
            task = payload.get("task", {})
            owner = self.tasks_by_id.get(_id(task), task)
            if owner and _get(owner, "status") not in ACTIVE:
                continue
            # A059 about actual rescue/engineering work is transport backlog;
            # diagnostics/archive/control receipts are not new rendezvous needs.
            action = payload.get("action_id", payload.get("requested_action_id"))
            if action not in {"A012", "A013", "A014", "A015", "A016", "A017", "A018", "A019", "A020", "A022", "A023", "A031", "A062", "A063"}:
                continue
            requests.append(request)
        return reports, requests

    def link_peer(self, eid):
        peer = self.v(eid, "link.receiver_id")
        if peer:
            return str(peer)
        prefix = self.id + "->"
        if str(eid).startswith(prefix):
            return str(eid)[len(prefix):]
        return str(self.records.get("probe_peer_id") or self.spec.get("coordinator_id") or "")

    def needed(self, task):
        if _get(task, "still_needed") is False or _get(task, "cancel_requested"):
            return False
        goal = _get(task, "goal_predicate", {})
        if isinstance(goal, Mapping) and goal:
            checked = False
            for expression, expected in goal.items():
                if "@" not in expression:
                    continue
                key, eid = expression.rsplit("@", 1)
                checked = True
                if self.v(eid, key) not in _list(expected):
                    return True
            if checked:
                return False
        eid = str(_get(task, "target", ""))
        if self.is_kind(eid, "human"):
            return (self.v(eid, "human.handover_status") != "acknowledged" or
                    self.v(eid, "human.mobility") == "trapped")
        if self.is_kind(eid, "building"):
            return self.v(eid, "belief.search_status") != "complete"
        if self.is_kind(eid, "obstacle"):
            return self.v(eid, "obstacle.state") not in ("cleared", "moved")
        return _get(task, "still_needed") is True


def _emit(target, why, refs, **extra):
    return (str(target), why, list(dict.fromkeys(refs)), extra)


def _assessment(c, n):
    out, mission = [], c.mission()
    if n == 1:
        for eid in c.ids("open_area", "region", "water", "building"):
            objectives = c.objective_refs(eid, "survey", "survey_area", "assess_region", "rapid_assessment",
                                          "regional_assessment")
            if not objectives and any(c.is_kind(eid, kind) for kind in ("open_area", "region", "water")):
                # A public regional survey includes mapped water, without
                # disclosing any hidden casualty or recovery requirement.
                objectives = c.survey_refs()
            if objectives and c.v(eid, "observation.coverage_estimate") in UNKNOWN:
                out.append(_emit(eid, "本机已知公共目标要求该区域初步评估，但尚无覆盖证据。",
                                 objectives + c.refs(eid, "entity.kind", "observation.coverage_estimate")))
    elif n == 2:
        for eid in c.ids("building"):
            objectives = c.objective_refs(eid, "assess_all_buildings", "assess_building", "building_assessment", "assessment")
            if objectives and c.v(eid, "building.visual_damage") in UNKNOWN:
                out.append(_emit(eid, "本机已知公共目标要求该建筑评估；外部损毁仍未知，不能按未损毁处理。",
                                 objectives + c.refs(eid, "entity.kind", "building.visual_damage")))
    elif n == 3 and mission:
        for eid in c.ids("road", "bridge"):
            if c.v(eid, "road.entry_state") in UNKNOWN or c.v(eid, "road.passability") in UNKNOWN:
                out.append(_emit(eid, "已知地图中该通路尚缺平台通行证据，需要灾后评估。",
                                 mission + c.refs(eid, "entity.kind", "road.entry_state", "road.passability")))
    elif n == 4:
        for eid in c.ids("building"):
            relevant = c.targeting(eid, {"T009", "T013", "T014", "T015", "T030", "T039", "T040"}) or (
                        c.possible_occupants(eid) and c.v(eid, "building.visual_damage") not in UNKNOWN)
            if relevant and (c.v(eid, "building.entrance_ids") in UNKNOWN or
                             c.v(eid, "belief.estimated_access") in UNKNOWN):
                out.append(_emit(eid, "搜索或交付需要接近此建筑，但入口净空及屋顶接近条件尚未核实。",
                                 c.refs(eid, "building.entrance_ids", "belief.estimated_access", "building.capacity")))
    elif n == 5:
        for eid in c.ids("water", "road", "building"):
            flooded = ((c.is_kind(eid, "water") and (c.evidence_about(eid) or c.targeting(eid))) or
                       c.v(eid, "building.flood_exposure") not in UNKNOWN + (False, "none", 0) or
                       _num(c.v(eid, "road.water_depth_m")) > 0)
            if flooded and c.v(eid, "belief.estimated_access") in UNKNOWN:
                out.append(_emit(eid, "本地已知水域或受淹迹象，但涉水条件未确认；需测量而非假造水深。",
                                 c.refs(eid, "entity.kind", "building.flood_exposure", "road.water_depth_m")))
    elif n == 6:
        for eid in c.ids("hazard", "building"):
            kind = str(c.v(eid, "hazard.kind", "")).lower()
            suspected = (kind in ("fire", "smoke", "gas", "hazmat", "wildfire") or
                         _num(c.v(eid, "hazard.smoke_density")) > 0 or
                         _num(c.v(eid, "hazard.gas_level")) > 0)
            if suspected and (c.v(eid, "hazard.footprint") in UNKNOWN or
                              c.age(eid, "hazard.intensity") > 60):
                out.append(_emit(eid, "收到火烟或危险源迹象，但其范围或强度证据不足，需受控侦察。",
                                 c.refs(eid, "hazard.kind", "hazard.smoke_density", "hazard.gas_level", "hazard.footprint")))
    elif n == 7:
        for eid, group in c.entities.items():
            if not c.relevant_entity(eid):
                continue
            for key in ("building.visual_damage", "road.entry_state", "road.topology_revision", "entity.revision"):
                entry = group.get(key)
                if entry and _get(entry, "source") in ("observation", "camera_image", "message", "inference"):
                    updated = _num(_get(entry, "t"))
                    mapped = _num(_get(c.records.get("map_revisions", {}), eid, -1), -1)
                    if updated > mapped and any(_get(v, "source") == "prior" for v in group.values()):
                        out.append(_emit(eid, "新收到灾后证据与已有地图需要对照，修订本地地图而非世界真值。",
                                         c.refs(eid, key, "entity.footprint"), map_revision=updated))
                        break
    elif n == 8:
        for eid in c.ids("building", "open_area", "region"):
            status = c.v(eid, "belief.search_status")
            assessed = c.v(eid, "building.visual_damage") not in UNKNOWN
            if (assessed or status in ("partial", "complete", "inconclusive")) and not c.v(eid, "building.worksite_record"):
                out.append(_emit(eid, "已有实际评估或搜索报告，但覆盖范围及未覆盖部分尚未登记。",
                                 c.refs(eid, "building.visual_damage", "belief.search_status", "building.worksite_record"),
                                 search_status=status, preserve_unknown=True))
    return out


def _search(c, n):
    out, mission = [], c.mission()
    if n == 9:
        # An explicit public regional search is a demand to gather evidence;
        # it cannot depend on already having a report from that region.
        def regional_search(value):
            if not isinstance(value, str):
                return False
            text = value.lower()
            return ("区域" in text and "搜索" in text or
                    any(term in text for term in ("regional search", "area search", "search area")))

        public_search = []
        for context_eid in c.entities:
            for key in ("public_brief", "mission.public_brief"):
                if regional_search(c.v(context_eid, key)):
                    public_search.extend(c.refs(context_eid, key))
        for message in c.received_index["brief"]:
            payload = _get(message, "payload", {})
            if isinstance(payload, Mapping) and any(
                    regional_search(payload.get(key)) for key in ("brief", "brief_zh", "mission")):
                public_search.extend(c.refs(_id(message), "message.local_availability"))
        repeat_enabled = (c.is_coordinator and
                          _num(_get(c.spec.get("camera_search_policy"), "repeat_budget_s")) > 0)
        for eid in c.ids("open_area", "tree_cluster", "slope", "coast", "water", "region"):
            reported = c.evidence_about(eid) or any(
                _get(entry, "source") in ("observation", "camera_image", "message", "inference") and key not in HOUSEKEEPING_KEYS
                for key, entry in c.entities[eid].items())
            prior_pass = repeat_enabled and any(_get(task, "type") == "T009" and _get(task, "target") == eid
                             and _get(task, "status") == "completed" and
                             any(str(ref).startswith("rescue-camera/") and str(ref).endswith("/image")
                                 for ref in _list(_get(task, "result_evidence_ids", []))) for task in c.typed_tasks("T009"))
            if prior_pass:
                retry = c.repeat_search(eid)
                if (reported or public_search) and retry is not None:
                    refs, index = retry
                    out.append(_emit(eid, "上次相机搜索已结束，但区域覆盖仍未知或不足；在声明预算内换视角复搜，未检出不等于无人。",
                                     c.refs(eid, "entity.kind", "belief.search_status", "observation.coverage_estimate") + refs + mission,
                                     search_pass_index=index, preserve_unknown=True, priority=64))
            elif (reported or public_search) and c.v(eid, "belief.search_status") in UNKNOWN:
                out.append(_emit(eid, "公共任务或现场报告要求搜索该区域，但尚无人员搜索覆盖。",
                                 c.refs(eid, "entity.kind", "belief.search_status", "observation.coverage_estimate") + public_search))
    elif n == 10:
        for eid in c.ids("cue"):
            if c.v(eid, "cue.kind") not in UNKNOWN and c.v(eid, "cue.related_human_id") in UNKNOWN:
                out.append(_emit(eid, "已观测到物品或信号线索，但人员关联未知，需解释并保留关联不确定性。",
                                 c.refs(eid, "cue.kind", "cue.location", "cue.related_human_id")))
    elif n == 11:
        for eid in c.ids("human", "cue", "building"):
            for key in ("belief.human_presence", "observation.detections", "cue.signal_state"):
                entry = c.entry(eid, key)
                value = c.v(eid, key)
                distribution_uncertain = isinstance(value, Mapping) and (
                    _num(value.get("uncertain")) > 0.25 or 0.2 < _num(value.get("present")) < 0.8)
                if entry and (c.confidence(eid, key) < 0.65 or distribution_uncertain or
                              value in ("occluded", "ambiguous", "suspected", "uncertain")):
                    out.append(_emit(eid, "疑似目标受遮挡或置信度不足，需要新增视角而不是把未检出当无人。",
                                     c.refs(eid, key), conditional_sequence="Q097"
                                     if value == "occluded" or c.v(eid, "human.visibility_context") == "occluded" else None))
                    break
    elif n == 12:
        for eid in c.ids("human"):
            observed = c.v(eid, "human.pose", c.v(eid, "entity.pose"))
            estimated = c.v(eid, "belief.estimated_pose")
            moving = c.v(eid, "human.mobility") == "mobile"
            if observed is not None and (estimated is None or
                                         (moving and c.age(eid, "belief.estimated_pose") > 30)):
                out.append(_emit(eid, "本地已有人员位置候选，但缺当前关联轨迹或移动目标位置已过时。",
                                 c.refs(eid, "human.pose", "entity.pose", "belief.estimated_pose", "human.mobility")))
    elif n == 13:
        for eid in c.ids("building", "underground", "slope", "tree_cluster"):
            access = c.v(eid, "belief.estimated_access")
            reported = c.evidence_about(eid) or any(
                _get(c.entry(eid, key), "source") in ("observation", "camera_image", "message", "inference") for key in c.fact_keys(eid))
            if (access in ("narrow", "limited", "partly_open") or c.is_kind(eid, "underground")) and (
                    reported or c.targeting(eid, {"T009", "T013", "T014", "T015"})):
                if c.v(eid, "belief.search_status") in UNKNOWN:
                    out.append(_emit(eid, "已知狭窄外部或地下接近区尚未搜索，需先验证可达空间。",
                                     c.refs(eid, "entity.kind", "belief.estimated_access", "belief.search_status") + mission))
    elif n == 14:
        for eid in c.ids("building"):
            # Occupancy evidence alone justifies interior search, including intact (D0) buildings;
            # access/safety are checked by A008 and capability by allocation.
            if c.possible_occupants(eid) and c.v(eid, "belief.search_status") in UNKNOWN:
                out.append(_emit(eid, "用途、容量或在场证据表明建筑内可能有人，但内部搜索仍未知（与受损程度无关）。",
                                 c.refs(eid, "building.visual_damage", "building.use", "building.capacity",
                                        "belief.human_presence", "belief.search_status"),
                                 priority=c.spec.get("casualty_work_policy", {}).get("damaged_search_priority", 80)
                                 if c.v(eid, "building.visual_damage") in ("major", "destroyed") else 70,
                                 goal_predicate={f"belief.search_status@{eid}": ["complete", "partial", "inconclusive"]}))
    elif n == 15:
        for eid in c.ids("building"):
            status = c.v(eid, "belief.search_status")
            if status in ("partial", "inconclusive"):
                out.append(_emit(eid, "上次抽象搜索报告为部分覆盖或仍不确定，需明确预算后补搜而非宣称无人。",
                                 c.refs(eid, "belief.search_status", "observation.coverage_estimate"),
                                 conditional_sequence="Q110", preserve_unknown=True))
    elif n == 16:
        for eid in c.ids("cue", "human", "building", "hazard", "open_area"):
            thermal_cue = (str(c.v(eid, "cue.kind", "")).lower() in
                           ("heat", "thermal", "heat_source", "campfire", "fire", "flare") or
                           "thermal" in _tokens(c.v(eid, "human.distress_signal", [])))
            illumination = c.v(eid, "hazard.illumination")
            lux = _get(illumination, "lux", illumination if isinstance(illumination, (int, float)) else None)
            dark = illumination in ("dark", "night", "low") or (lux is not None and _num(lux) < 10)
            if (thermal_cue or dark) and c.v(eid, "belief.search_status") != "complete":
                out.append(_emit(eid, "本地有热源线索或低照度搜索需求，申请已声明热像设备补充观测，不作透墙推断。",
                                 c.refs(eid, "cue.kind", "human.distress_signal", "hazard.illumination", "belief.search_status"), modality="thermal"))
    elif n == 17:
        for eid in c.ids("cue", "human", "building"):
            kind = str(c.v(eid, "cue.kind", "")).lower()
            if (kind in ("voice", "sound", "acoustic", "cry", "scream") or
                    _tokens(c.v(eid, "human.distress_signal", [])) & {"voice", "sound", "calling"}):
                if c.v(eid, "belief.estimated_pose") in UNKNOWN:
                    out.append(_emit(eid, "收到声音求救线索但声源位置未定，需声学测量并保留方向误差。",
                                     c.refs(eid, "cue.kind", "human.distress_signal", "belief.estimated_pose"), modality="acoustic"))
    elif n == 18:
        authorized_equipment = [eid for eid in c.ids("equipment") if
            str(c.v(eid, "equipment.kind", "")).lower() in ("rf", "rf_detector", "phone_locator") and
            (_tokens(c.v(eid, "equipment.operator_authority", [])) & {c.id.lower(), c.kind.lower(), "authorized", "granted"})]
        authorized = c.records.get("rf_authorized") is True or bool(authorized_equipment)
        if authorized:
            for eid in c.ids("cue", "human"):
                phone = (str(c.v(eid, "cue.kind", "")).lower() in ("rf", "phone", "mobile_phone", "radio") or
                         "phone" in _tokens(c.v(eid, "human.distress_signal", [])))
                if phone and c.v(eid, "cue.signal_state") not in ("resolved", "inactive"):
                    out.append(_emit(eid, "已获本地授权且存在未定位RF线索，使用协议匹配设备搜索，不把普通无线通信当手机定位。",
                                     c.refs(eid, "cue.kind", "cue.signal_state", "human.distress_signal") +
                                     [(equipment, "equipment.operator_authority") for equipment in authorized_equipment],
                                     modality="rf", authorization_record="rf_authorized" if not authorized_equipment else None))
    return out


def _engineering(c, n):
    out = []
    if n in (19, 20):
        purposes = ("observation", "relay") if n == 19 else ("travel", "access", "reroute")
        for goal in _rows(c.records.get("navigation_goals", [])):
            eid = _get(goal, "target")
            if eid and _get(goal, "purpose") in purposes and not _get(goal, "arrived", False):
                out.append(_emit(eid, "本机已形成尚未到达的合法接近目标，需要按已知地图规划实际转场。",
                                 c.refs(c.id, "belief.current_goal_context") + c.refs(eid, "entity.pose") +
                                 [(road, key) for road in c.ids("road") for key in ("road.entry_state", "road.clear_width_m", "road.step_height_m")],
                                 navigation_goal=dict(goal)))
        types = {"T001", "T002", "T009", "T049"} if n == 19 else {"T014", "T015", "T021", "T040"}
        for task in c.task_rows({"accepted", "blocked"}, types):
            eid = _get(task, "target")
            if eid and c.distant(eid):
                out.append(_emit(eid, "本机已接受的搜索或服务目标尚未到达，需先完成相应转场。",
                                 c.refs(_id(task), "task.status") + c.refs(eid, "entity.pose"),
                                 related_task_id=_id(task),
                                 conditional_sequence="Q098" if n == 20 and c.blocked() else None))
    elif n == 21:
        if c.blocked():
            for eid in c.ids("obstacle", "road"):
                if (c.obstructing(eid) or c.v(eid, "road.entry_state") == "blocked") and (
                        c.v(eid, "road.clear_width_m") is None and c.v(eid, "obstacle.volume_m3") is None):
                    out.append(_emit(eid, "本机移动反馈受阻，但障碍净空或尺度仍未知，需要接近验证后决定绕行。",
                                     c.refs(eid, "road.entry_state", "road.clear_width_m", "obstacle.volume_m3"),
                                     conditional_sequence="Q098"))
    elif n in (22, 23):
        for eid in c.ids("obstacle"):
            volume = c.v(eid, "obstacle.volume_m3")
            if volume is None or not c.obstructing(eid):
                continue
            heavy = _num(volume) > 0.5 or _num(c.v(eid, "obstacle.mass_kg")) > 80
            if heavy == (n == 23):
                search_blocked = bool(c.task_rows({"accepted", "executing", "blocked"}, {"T014", "T015"}))
                branch = None
                if heavy and search_blocked and "C17" not in c.caps():
                    branch = "Q099"
                elif not heavy and c.kind == "robot_dog" and "C16" not in c.caps():
                    branch = "Q100"
                out.append(_emit(eid, "本机受阻证据指向重型障碍，需额定工程清运。" if heavy else
                                 "本机受阻证据指向轻型残骸，需额定操纵主体移除后再测净空。",
                                 c.refs(eid, "obstacle.state", "obstacle.volume_m3", "obstacle.mass_kg", "obstacle.blocks_ids"),
                                 conditional_sequence=branch))
    elif n == 24:
        for eid in c.ids("obstacle", "building"):
            methods = _tokens(c.v(eid, "obstacle.clear_method", []))
            kind = c.v(eid, "obstacle.kind")
            if (methods & {"breach", "cut", "break"} or kind in ("blocked_door", "wall", "sealed_entrance")) and (
                    c.obstructing(eid) or c.targeting(eid)):
                out.append(_emit(eid, "必要入口被门墙阻断，绕行不足以进入，需授权破拆力量建立局部通路。",
                                 c.refs(eid, "obstacle.kind", "obstacle.clear_method", "obstacle.blocks_ids")))
    elif n == 25:
        for eid in c.ids("building"):
            if c.v(eid, "building.structural_condition") in ("compromised", "unstable", "unsafe", "critical", "at_risk") and (
                    c.possible_occupants(eid) or c.targeting(eid)):
                worksite = c.v(eid, "building.worksite_record", {})
                if not isinstance(worksite, Mapping) or not worksite.get("supported"):
                    out.append(_emit(eid, "现场结构证据不稳定且仍有搜索或人员需求，应先请求局部临时支撑并保留剩余危险。",
                                     c.refs(eid, "building.structural_condition", "building.capacity", "building.worksite_record")))
    elif n == 26:
        for eid in c.ids("obstacle"):
            if c.obstructing(eid) and (_tokens(c.v(eid, "obstacle.clear_method", [])) & {"lift", "hoist", "relocate"} or
                                      _num(c.v(eid, "obstacle.mass_kg")) > 500):
                out.append(_emit(eid, "阻路大块障碍需要吊升或移置，须由额定起重主体核验支撑与合法弃置区。",
                                 c.refs(eid, "obstacle.mass_kg", "obstacle.clear_method", "obstacle.disposal_location")))
    return out


def _personnel(c, n):
    out = []
    for eid in c.ids("human"):
        alive = c.v(eid, "human.alive")
        mobility = c.v(eid, "human.mobility")
        needs, supported = c.needs(eid), c.supported(eid)
        injury = c.v(eid, "human.injury")
        handover = c.v(eid, "human.handover_status")
        carrier = c.v(eid, "human.carrier_id")
        refs = c.refs(eid, "human.needs", "human.mobility", "human.handover_status")
        if n != 37 and (alive is False or (handover == "acknowledged" and n != 31)):
            continue
        if n == 27 and (c.v(eid, "human.needs") in UNKNOWN or "contact" in needs):
            retry_refs = c.camera_contact_retry(eid)
            if retry_refs is not None:
                out.append(_emit(eid, "已发现人员但需求或有效响应未知，需双向联系，未响应仍记未知。",
                                 refs + c.refs(eid, "human.distress_signal") + retry_refs))
        elif n == 28 and mobility in ("mobile", "limited") and "guided" not in supported and (
                needs & {"guidance", "evacuate", "evacuation", "shelter", "escape"} or c.nearby_risk(eid)):
            risk = c.nearby_risk(eid)
            out.append(_emit(eid, "可响应人员有避险需求或位于已知危险附近，需引导并确认实际响应和去向。",
                             refs + (c.refs(risk, "hazard.active", "hazard.footprint") if risk else [])))
        elif n == 29 and injury in UNKNOWN + ("not_modeled",) and (mobility in ("trapped", "immobile", "limited") or
                                                needs & {"medical_pack", "medical_supplies", "medical", "aid", "triage", "treatment"}):
            out.append(_emit(eid, "人员活动受限或提出医疗需求，但伤情尚无专业评估记录。",
                             refs + c.refs(eid, "human.injury", "human.last_assessed_s")))
        elif n == 30:
            missing = needs & {"water", "food", "medical_pack", "blanket", "supplies"} - supported
            if missing:
                items = [item for item in c.ids("item") if c.v(item, "item.receiver_id") == eid and
                         _supply_kind(c.v(item, "item.kind")) in missing and c.v(item, "item.state") != "lost"]
                pending_items = [item for item in items if c.v(item, "item.state") != "received"]
                # A known receipt satisfies this supply need without implying cure.
                received_kinds = {_supply_kind(c.v(item, "item.kind")) for item in items if c.v(item, "item.state") == "received"}
                missing -= received_kinds
                if missing:
                    extra = {"need_types": sorted(missing), "receiver_id": eid,
                             "goal_predicate": {"record.supply_receipt": True}}
                    item = next((item for item in pending_items if _supply_kind(c.v(item, "item.kind")) in missing), None)
                    if item:
                        extra.update(item_id=item, goal_predicate={f"item.state@{item}": ["received"]})
                    out.append(_emit(eid, "已确认人员基础物资需求尚未满足，需要真实投送并核验领取，不能以接触代替交付。",
                                     refs + c.refs(eid, "human.support_received") +
                                     (c.refs(item, "item.receiver_id", "item.state") if item else []), **extra))
        elif n == 31 and injury not in UNKNOWN + ("none", "uninjured", "healthy", "not_modeled", False) and "treated" not in supported:
            out.append(_emit(eid, "已有伤情证据且未获专业救护，需匹配资质和实际耗材的服务。",
                             refs + c.refs(eid, "human.injury", "human.support_received")))
        elif n == 32 and (mobility == "trapped" or _num(c.v(eid, "human.buriedness")) > 0):
            out.append(_emit(eid, "本地报告表明人员仍受困，需经A059请求专业支撑、破拆或吊升后解除受困，不等同最终移交。",
                             refs + c.refs(eid, "human.buriedness"), specialist_required=True,
                             requested_action_id="A017",
                             goal_predicate={f"human.mobility@{eid}": ["limited", "immobile", "mobile"],
                                             f"human.buriedness@{eid}": [0]}))
        elif (n == 33 and mobility in ("immobile", "limited", "mobile") and not carrier
              and _num(c.v(eid, "human.buriedness")) <= 0
              and c.v(eid, "human.location_context") not in ("water", "in_water", "in water", "afloat")
              and (needs & {"transport", "evacuate", "evacuation", "medical_transport"}
                   or mobility == "immobile" or "extricated" in supported)):
            destination = c.destination(eid)
            out.append(_emit(eid, "受困已解除或人员可搬运，但尚未装入合格载具，存在真实转运需求。",
                             refs + c.refs(eid, "human.carrier_id", "human.buriedness", "human.support_received"),
                             receiving_station_id=destination,
                             priority=c.spec.get("casualty_work_policy", {}).get("loading_priority", 85),
                             goal_predicate={f"human.location_context@{eid}": ["on_carrier"]}))
        elif n == 34 and carrier and handover not in ("arrived", "acknowledged"):
            destination = c.destination(eid)
            if destination and (c.is_coordinator or str(carrier) in (c.id, str(getattr(c.agent, "platform_eid", c.id)))):
                out.append(_emit(eid, "人员已有承载关系但尚未到达已知接收点，需要由实际承载者运输，不能直接完成移交。",
                                 refs + c.refs(eid, "human.carrier_id", "human.destination_id"),
                                 destination=destination, station_id=destination, carrier_id=carrier,
                                 priority=c.spec.get("casualty_work_policy", {}).get("transport_priority", 87),
                                 goal_predicate={f"human.handover_status@{eid}": ["arrived", "acknowledged"]}))
        elif n == 35 and handover == "arrived":
            destination = c.v(eid, "human.destination_id")
            if destination and (c.is_coordinator or str(carrier) in (c.id, str(getattr(c.agent, "platform_eid", c.id)))):
                out.append(_emit(eid, "人员已到达但接收端尚未确认，必须取得有效接收回执并解除承载占用后才算最终移交。",
                                 refs + c.refs(eid, "human.destination_id", "human.carrier_id"),
                                 destination=destination, station_id=destination, carrier_id=carrier,
                                 goal_predicate={f"human.handover_status@{eid}": ["acknowledged"],
                                                 f"human.carrier_id@{eid}": [None]}))
        elif n == 36 and c.v(eid, "human.location_context") in ("in_water", "in water", "water", "afloat") and (
                handover != "acknowledged" and not carrier):
            destination = c.destination(eid)
            out.append(_emit(eid, "已知人员仍在水中且未获回收，需有水域机动和回收设备的主体救助。",
                             refs + c.refs(eid, "human.location_context"),
                             destination=destination, station_id=destination))
        elif n == 37 and alive is False:
            phase = c.records.get("operation_phase")
            authorized = c.records.get("recovery_authorized") is True
            for context_eid in c.entities:
                context = c.v(context_eid, "belief.current_goal_context", {})
                if isinstance(context, Mapping):
                    phase = context.get("phase", phase)
                    authorized = authorized or context.get("recovery_authorized") is True
            if phase == "recovery" and authorized and handover != "acknowledged":
                out.append(_emit(eid, "本地明确进入授权恢复阶段且有死亡报告，安排专门遗体移交；不得计入存活救援。",
                                 refs + c.refs(eid, "human.alive") + c.refs(c.id, "belief.current_goal_context"),
                                 recovery_only=True))
    return out


def _logistics(c, n):
    out = []
    if n in (38, 39, 40, 41):
        for eid in c.ids("item"):
            state, holder = c.v(eid, "item.state"), c.v(eid, "item.holder_id")
            receiver = c.v(eid, "item.receiver_id")
            reserved = c.v(eid, "item.reservation_task_id")
            refs = c.refs(eid, "item.state", "item.holder_id", "item.receiver_id", "item.release_state")
            if n == 38 and state in ("stock", "reserved") and (reserved or (receiver and c.needs(receiver))):
                out.append(_emit(eid, "已知库存物品与本地交付需求或预留任务匹配，需实际取用挂载并检查容量接口。",
                                 refs + c.refs(eid, "item.mass_kg", "item.reservation_task_id"), receiver_id=receiver))
            elif n in (39, 40) and state in ("attached", "in_transit") and holder == c.id and receiver:
                airborne = c.spec.get("mobility") == "air" or c.kind == "uav"
                if airborne == (n == 39) and c.v(eid, "item.release_state") != "jammed":
                    out.append(_emit(eid, "本机已携带有明确接收者的物资，需实际释放到交付区；释放不代表领取。",
                                     refs, receiver_id=receiver, destination=receiver))
            elif n == 41 and ((state in ("released", "placed") and not c.v(eid, "item.receipt_id")) or
                              c.v(eid, "item.release_state") == "jammed"):
                out.append(_emit(eid, "已有释放、落点或卡滞反馈但尚无有效收货证据，需交付复核并报告实际物品状态。",
                                 refs + c.refs(eid, "item.receipt_id", "item.contact_pose"), receiver_id=receiver,
                                 conditional_sequence="Q111" if c.v(eid, "item.release_state") == "jammed" else None))
    elif n == 42:
        inventory = c.records.get("inventory", {})
        if isinstance(inventory, Mapping):
            for kind, quantity in inventory.items():
                if _num(quantity) <= 0 and any(_supply_kind(kind) in c.needs(h) for h in c.ids("human")):
                    out.append(_emit(c.id, "本机适用物资已耗尽，但已知人员仍有对应需求，需要从真实库存补给。",
                                     c.refs(c.id, "platform.payload_item_ids") +
                                     [(h, "human.needs") for h in c.ids("human") if _supply_kind(kind) in c.needs(h)],
                                     item_kind=kind))
        for eid in c.ids("equipment"):
            if c.v(eid, "equipment.consumable_remaining") is not None and (
                    _num(c.v(eid, "equipment.consumable_remaining")) <= 0 and c.v(eid, "equipment.enabled") is not False):
                out.append(_emit(eid, "本地器材记录显示适用耗材耗尽，继续作业前需实际补给。",
                                 c.refs(eid, "equipment.consumable_type", "equipment.consumable_remaining")))
    elif n == 43 and c.low():
        remaining, maximum, reserve = c.battery()
        out.append(_emit(c.id, "本机电量接近安全保留阈值，应暂停高耗能作业并按实际功率充电。",
                         c.refs(c.id, "platform.battery_remaining_Wh", "platform.reserve_Wh"),
                         target_battery_Wh=0.85 * maximum, reserve_Wh=reserve,
                         conditional_sequence="Q102", maintenance=True))
    elif n == 44 and c.low(0.35):
        swappable = c.spec.get("swappable_battery", c.records.get("swappable_battery", False))
        for station in c.ids("station"):
            stock = c.v(station, "station.inventory", {})
            connectors = _tokens(c.v(station, "station.connector_types", []))
            connector = str(c.spec.get("battery_connector", ""))
            compatible = bool(connector and connector.lower() in connectors)
            if (swappable and compatible and isinstance(stock, Mapping) and
                    _num(stock.get("battery", stock.get("batteries", 0))) > 0 and
                    c.v(station, "station.open_state") not in ("closed", "offline")):
                out.append(_emit(c.id, "本机低电量且已知换电站有兼容备用电池，申请真实换电服务并保留旧电池去向。",
                                 c.refs(c.id, "platform.battery_remaining_Wh") +
                                 c.refs(station, "station.inventory", "station.connector_types"), station_id=station, maintenance=True))
    elif n == 45:
        remaining, maximum, reserve = c.battery()
        risk = c.nearby_risk(c.id)
        jam = any(c.v(eid, "item.release_state") == "jammed" and c.v(eid, "item.holder_id") == c.id
                  for eid in c.ids("item"))
        if (maximum > 0 and remaining <= reserve) or risk or jam or c.own("platform.health_state") == "degraded":
            site = c.own("platform.return_site_id", c.records.get("return_site_id"))
            if site:
                out.append(_emit(site, "本机安全余量、危险接近或执行器反馈要求撤离，应实际返回已知安全点。",
                                 c.refs(c.id, "platform.battery_remaining_Wh", "platform.health_state", "platform.return_site_id"),
                                 conditional_sequence="Q111" if jam else
                                 ("Q102" if maximum > 0 and remaining <= reserve else None), maintenance=True))
    elif n == 46:
        for eid in c.ids("platform"):
            if c.v(eid, "platform.health_state") == "failed" or c.v(eid, "platform.motion_state") == "disabled":
                out.append(_emit(eid, "已收到平台失效或无法移动的本地证据，需要合格回收主体牵引运送。",
                                 c.refs(eid, "platform.health_state", "platform.motion_state")))
        if c.own("platform.health_state") == "failed" and not any(row[0] == c.id for row in out):
            out.append(_emit(c.id, "本机失效无法自行返站，需请求具备承载能力的外部回收主体。",
                             c.refs(c.id, "platform.health_state"), maintenance=True))
    return out


def _communication(c, n):
    out = []
    business_messages = c.business_messages
    reports, requests = c.comms_backlog() if n in (47, 48) else ([], [])
    links = [eid for eid in c.entities if c.entry(eid, "link.state") is not None
             and (c.is_coordinator or eid.startswith(c.id + "->") or c.relevant_entity(eid))]
    own_links = [eid for eid in links if c.local_entry(c.entry(eid, "link.state"))]
    if n == 47:
        for eid in own_links:
            peer = c.link_peer(eid)
            sample_key = "link.sampled_at_s" if c.entry(eid, "link.sampled_at_s") is not None else "link.state"
            if peer and (c.v(eid, "link.state") in UNKNOWN or c.age(eid, sample_key) > 45):
                out.append(_emit(eid, "本地链路样本未知或过时，需要向实际对端探测；位置变化是新采样理由而非元数据变化。",
                                 c.refs(eid, "link.state") + c.refs(c.id, "entity.pose"),
                                 receiver=peer, receiver_id=peer, peer_id=peer))
        if not own_links and (business_messages or requests or c.records.get("link_probe_due")):
            probe_messages = reports or business_messages
            peers = [str(peer) for message in probe_messages for peer in _list(_get(message, "receivers", []))
                     if str(peer) != c.id]
            peers.extend(str(_get(request, "agent")) for request in requests if _get(request, "agent"))
            peer = peers[0] if peers else c.records.get("probe_peer_id")
            if peer:
                out.append(_emit(peer, "本机有已知接收方的待发业务但尚无链路探测证据，需要实际采样而非读取全局网络。",
                                 c.refs(c.id, "link.state") + c.refs(_id(business_messages[0], c.id) if business_messages else c.id,
                                                                          "message.receiver_ids"),
                                 receiver=peer, receiver_id=peer, peer_id=peer))
    elif n == 48:
        if not reports and not requests:
            return []
        rendezvous = c.records.get("rendezvous_target") or c.own("platform.return_site_id")
        if not rendezvous or not c.pose(rendezvous) or not c.distant(rendezvous, threshold=10):
            return []
        for eid in links:
            # Only an own local sample can initiate self-relocation. A teammate's
            # reported outage is not telemetry of this platform's radio.
            if c.v(eid, "link.state") not in LINK_WEAK or not c.local_entry(c.entry(eid, "link.state")):
                continue
            peer = c.link_peer(eid)
            if not peer:
                continue
            refs = c.refs(eid, "link.state") + c.refs(rendezvous, "entity.pose")
            refs.extend(ref for message in reports for ref in c.refs(_id(message), "message.ack_state"))
            for request in requests:
                payload = _get(request, "payload", {})
                owner = _id(payload.get("task", {}))
                refs.extend(c.refs(owner or _id(request), "task.status" if owner else "message.ack_state"))
            out.append(_emit(rendezvous, "本机实测链路不可达或退化，且救援报告或A059请求仍待响应；优先返回已知会合点并实际重发缓存业务。",
                             refs, link_id=eid, receiver=peer, receiver_id=peer, peer_id=peer,
                             receivers=[c.spec.get("coordinator_id") or peer],
                             destination=rendezvous, maintenance=True,
                             pending_message_ids=[_id(message) for message in reports],
                             pending_request_ids=[_id(request) for request in requests]))
    elif n == 49:
        for message in c.received_index["relay"]:
            payload = _get(message, "payload", {})
            if c.received_domain_message(message) and (payload.get("relay_needed") or payload.get("relay_handoff")):
                target = payload.get("relay_target", payload.get("target", _id(message)))
                out.append(_emit(target, "实际收到中继需求或原中继退出通知，需要建立可用新链并保留逐跳业务证据。",
                                 c.refs(_id(message), "message.local_availability"),
                                 conditional_sequence="Q105" if payload.get("relay_handoff") else None))
        for eid in links:
            if c.v(eid, "link.state") in LINK_DOWN and business_messages:
                out.append(_emit(eid, "本地业务链路已不可用，申请空中中继持续转发而不是宣称全局连接恢复。",
                                 c.refs(eid, "link.state")))
    elif n == 50:
        relay_items = [eid for eid in c.ids("item") if c.v(eid, "item.kind") in ("relay", "relay_node") and
                       c.v(eid, "item.state") in ("stock", "reserved", "attached")]
        for eid in links:
            if relay_items and c.v(eid, "link.state") in LINK_WEAK:
                out.append(_emit(eid, "链路退化且本地已知可用实体中继节点，需要实际挂载布放并核验电源链路。",
                                 c.refs(eid, "link.state") + c.refs(relay_items[0], "item.kind", "item.state"), item_id=relay_items[0]))
    elif n == 51:
        for message in c.received_index["forward"]:
            payload = _get(message, "payload", {})
            if not c.received_domain_message(message):
                continue
            destination = payload.get("forward_to", payload.get("next_hop"))
            if destination and destination != c.id and not c.marked("forwarded_messages", _id(message)):
                out.append(_emit(_id(message), "上一跳消息已实际收件且包含下一跳目标，需要真实转发并等待终端回执。",
                                 c.refs(_id(message), "message.local_availability", "message.hop_trace"), receiver=destination))
    elif n == 52:
        for evidence in c.evidence:
            eid = _id(evidence)
            size = _num(_get(evidence, "payload_bytes", _get(evidence, "size_bytes", 0)))
            if size > 0 and not c.marked("uploads", eid):
                out.append(_emit(eid, "本机持有尚未上传的有字节证据，需要实际编码传输和完整性校验，不能共享磁盘冒充送达。",
                                 c.refs(eid, "observation.payload_bytes", "observation.content_hash"), evidence_ids=[eid]))
    elif n == 53:
        for message in business_messages:
            payload = _get(message, "payload", {})
            # Assignment delivery has its own bounded runtime timeout state.
            # A retransmitted copy is not fresh work for another retry task.
            if (_get(message, "retry_root") or payload.get("retry_root") or
                    payload.get("op", _get(message, "op")) == "assignment"):
                continue
            ack = _get(message, "ack_state")
            age = c.now - _num(_get(message, "sent_s", c.now))
            timeout = _num(_get(message, "ack_timeout_s", _get(message, "timeout_s", c.records.get("ack_timeout_s", 30))), 30)
            ttl = _get(message, "ttl_s")
            unexpired = ttl is None or age < _num(ttl)
            if (ack in ("pending", "timeout") and age >= timeout and unexpired and
                    _num(_get(message, "retry_count")) < 2):
                out.append(_emit(_id(message), "已发送业务超过回执时限且重试额度未耗尽，需要真实重传，缺回执不记成功。",
                                 c.refs(_id(message), "message.ack_state", "message.sent_s", "message.retry_count"),
                                 message_id=_id(message), max_retries=2))
        for message in c.received_index["damaged"]:
            payload = _get(message, "payload", {})
            if c.received_domain_message(message) and (payload.get("checksum_ok") is False or payload.get("missing_fragments")):
                out.append(_emit(_id(message), "实际收到的数据校验失败或缺片，需要向发送者请求补传，不使用未完整数据。",
                                 c.refs(_id(message), "message.received_fragments", "message.content_hash"),
                                 message_id=_id(message), conditional_sequence="Q103", max_retries=2))
    elif n == 54:
        down = any(c.v(eid, "link.state") in LINK_DOWN for eid in links)
        for message in business_messages:
            if not c.received_domain_message(message):
                continue
            states = _get(message, "status", {})
            values = list(states.values()) if isinstance(states, Mapping) else [states]
            age = c.now - _num(_get(message, "sent_s", c.now))
            ttl = _get(message, "ttl_s")
            cached = (any(state in ("queued", "partial") for state in values) or
                      bool(_get(message, "payload")) and _get(message, "ack_state") in ("pending", "timeout") and
                      not any(state in ("delivered", "expired", "failed") for state in values))
            if down and cached and (ttl is None or age < _num(ttl)):
                out.append(_emit(_id(message), "断联时本机仍保留未确认业务副本，需要按有效期存储携带并在恢复链路后实际投递。",
                                 c.refs(_id(message), "message.status", "message.ttl_s") +
                                 [(eid, "link.state") for eid in links], message_id=_id(message), conditional_sequence="Q104"))
    elif n == 55:
        for eid in links:
            if c.v(eid, "link.state") in ("intermittent", "degraded") and (
                    _num(c.v(eid, "link.loss_rate")) > 0.2 or _num(c.v(eid, "link.latency_s")) > 5):
                out.append(_emit(eid, "本地采样显示链路丢失率或时延超出业务预算，需要受设备约束调整信道功率或优先级。",
                                 c.refs(eid, "link.state", "link.loss_rate", "link.latency_s")))
    elif n == 56:
        pending = any(_get(m, "ack_state") in ("pending", "timeout") and c.received_domain_message(m) for m in business_messages)
        for eid in links:
            if c.v(eid, "link.state") in LINK_DOWN and (pending or c.task_rows()):
                out.append(_emit(eid, "本地已知失联且仍有承诺或未回执业务，需要恢复连接并刷新过时队友信息。",
                                 c.refs(eid, "link.state", "link.outage_duration_s"), conditional_sequence="Q104" if pending else None))
    return out


def _compute(c, n):
    out = []
    if n == 57:
        waiting = [job for job in c.jobs if _get(job, "status") in ("created", "rejected", "failed")]
        stale = c.now - _num(c.records.get("service_discovered_at", -1000), -1000) > 120
        if waiting and (not c.services or stale):
            job = waiting[0]
            out.append(_emit(_id(job, c.id), "本地作业等待服务且目录缺失或已过时，需要通过通信发现真实可用服务。",
                             c.refs(_id(job, c.id), "job.status", "job.service_id")))
    elif n == 58:
        for evidence in c.evidence:
            eid = _id(evidence)
            kind = _get(evidence, "kind", "")
            if kind in ("authored_sensing", "capture", "image", "sensor_capture") and not c.marked("processed_evidence", eid):
                if not any(eid in _list(_get(job, "input_evidence_ids", [])) and
                           _get(job, "status") not in TERMINAL_JOBS for job in c.jobs):
                    out.append(_emit(eid, "本机已有未处理的脚本采集证据，需要本地检测作业；这是脚本推断而非视觉能力。",
                                     c.refs(eid, "observation.processing_job_id", "observation.payload_ref"),
                                     evidence_ids=[eid], backend_mode="stub"))
    elif n == 59:
        for task in c.task_rows({"accepted", "executing", "blocked"}):
            target = _get(task, "target")
            if target and c.evidence_about(target) and not c.marked("reasoned_tasks", _id(task)):
                out.append(_emit(_id(task), "当前本机任务已有相关证据但尚未形成任务条件化推断，需要计上下文和实际推理成本。",
                                 c.refs(_id(task), "task.status", "task.goal_predicate") +
                                 c.refs(target, "belief.evidence_ids"), original_task_id=_id(task)))
    elif n == 60:
        for job in c.jobs:
            reason = str(_get(job, "failure_reason", _get(job, "rejection_code", ""))).lower()
            if _get(job, "status") in ("rejected", "failed") and reason in (
                    "memory", "oom", "insufficient_memory", "out_of_memory"):
                out.append(_emit(_id(job), "本地计算准入因内存不足被拒，需发现远程服务并实际传入输入、传回结果后才可决策。",
                                 c.refs(_id(job), "job.status", "job.failure_reason", "job.memory_request_bytes"),
                                 job_id=_id(job), conditional_sequence="Q106"))
        for result in c.result_failure("insufficient"):
            feedback = str(_get(result, "feedback_zh", "")).lower()
            if _get(result, "action_id") == "A044" and ("memory" in feedback or "内存" in feedback or "显存" in feedback):
                out.append(_emit(_id(result, c.id), "本机计算准入返回内存不足，需要比较远程执行而非把排队当完成。",
                                 c.refs(_id(result, c.id), "action.rejection_code"), conditional_sequence="Q106"))
    elif n == 61:
        for job in c.jobs:
            if _get(job, "status") in ("created", "rejected") and (
                    not _get(job, "node_id") or not _get(job, "service_id")) and c.services:
                out.append(_emit(_id(job), "本地作业尚未确定执行节点或模型，已收到候选服务信息，需按质量时延和预算选择。",
                                 c.refs(_id(job), "job.status", "job.service_id", "job.deadline_s"), job_id=_id(job)))
    elif n == 62:
        for job in c.jobs:
            if (_get(job, "status") == "created" and _get(job, "input_evidence_ids") and
                    not _get(job, "resources_reserved", False)):
                out.append(_emit(_id(job), "本地作业输入已就绪但尚未准入预留，需要真实排队执行并最终释放资源。",
                                 c.refs(_id(job), "job.status", "job.input_evidence_ids", "job.memory_request_bytes"), job_id=_id(job)))
    elif n == 63:
        for service in c.services:
            sid = _id(service)
            needed = any(_get(job, "service_id") == sid and _get(job, "status") not in TERMINAL_JOBS for job in c.jobs)
            if needed and _get(service, "status") in ("unloaded", "failed"):
                out.append(_emit(sid, "本地待执行作业依赖的指定版本模型服务尚未就绪，需要实际加载并更新资源。",
                                 c.refs(sid, "service.status", "service.version", "service.working_memory_bytes"),
                                 service_id=sid, operation="load"))
            elif (not needed and _get(service, "status") == "ready" and
                  _get(service, "unload_requested") is True):
                out.append(_emit(sid, "本地模型已有明确卸载请求且无活动作业依赖，需要实际释放模型占用。",
                                 c.refs(sid, "service.status", "service.version"), service_id=sid, operation="unload"))
    elif n == 64:
        for cache in c.caches:
            if (_get(cache, "reuse_compatible") is True and _get(cache, "eviction_state", "resident") == "resident" and
                    _num(_get(cache, "valid_until_s", c.now + 1)) > c.now):
                contents = set(_list(_get(cache, "content_ids", [])))
                for job in c.jobs:
                    if contents & set(_list(_get(job, "input_evidence_ids", []))) and _get(job, "status") == "created":
                        required = _get(job, "model_version")
                        if required is not None and required == _get(cache, "model_version"):
                            out.append(_emit(_id(cache), "本地有效缓存与待执行作业内容和模型版本兼容，可以复用并记录真实节省。",
                                             c.refs(_id(cache), "cache.reuse_compatible", "cache.model_version", "cache.valid_until_s"),
                                             cache_id=_id(cache), job_id=_id(job)))
                            break
    elif n == 65:
        for job in c.jobs:
            failed_service = any(_id(service) == _get(job, "service_id") and _get(service, "status") == "failed"
                                 for service in c.services)
            if (failed_service or _get(job, "failure_reason") == "service") and _get(job, "checkpoint_ref"):
                if _get(job, "checkpoint_compatible") is True and _num(_get(job, "retry_count")) < 2:
                    out.append(_emit(_id(job), "本地获知执行服务失效，且有明确兼容检查点，可请求迁移后恢复并计传输成本。",
                                     c.refs(_id(job), "job.failure_reason", "job.checkpoint_ref"),
                                     job_id=_id(job), conditional_sequence="Q107", max_retries=2))
    elif n == 66:
        for job in c.jobs:
            stale = c.job_stale(job)
            budget = _get(job, "budget_exhausted") is True
            deadline = _get(job, "deadline_s")
            overdue = deadline is not None and c.now > _num(deadline) and _get(job, "status") not in TERMINAL_JOBS
            if (stale or budget or overdue or _get(job, "status") == "failed") and _num(_get(job, "retry_count")) < 2:
                out.append(_emit(_id(job), "作业失败、预算耗尽或任务版本已变化，需要取消隔离旧结果后按当前预算重选，不覆盖新版本。",
                                 c.refs(_id(job), "job.status", "job.task_ref", "job.deadline_s", "job.failure_reason"),
                                 job_id=_id(job), isolate_stale=stale, conditional_sequence="Q108" if stale else None, max_retries=2))
    elif n == 67:
        groups = {}
        for evidence in c.evidence:
            if not c.marked("processed_evidence", _id(evidence)):
                for target in _list(_get(evidence, "about", [])):
                    groups.setdefault(str(target), []).append(evidence)
        for target, items in groups.items():
            sources = {_get(item, "by", _get(item, "from_agent")) for item in items}
            if len(items) >= 2 and (len(sources) >= 2 or c.records.get("batch_enabled") is True):
                out.append(_emit(target, "本机持有同一目标的多项输入，可按批处理或多源融合约束联合推理并保留各来源。",
                                 [(_id(item), "observation.parent_evidence_ids") for item in items],
                                 evidence_ids=[_id(item) for item in items]))
    elif n == 68:
        for job in c.jobs:
            if _get(job, "status") in TERMINAL_JOBS and _get(job, "resources_reserved") is True:
                out.append(_emit(_id(job), "本地终止作业仍持有资源预留，需要实际释放，不影响其他活动作业依赖。",
                                 c.refs(_id(job), "job.status", "job.finish_s"), job_id=_id(job)))
        dependencies = {value for job in c.jobs if _get(job, "status") not in TERMINAL_JOBS
                        for value in _list(_get(job, "cache_ids", []))}
        for cache in c.caches:
            expired = _get(cache, "valid_until_s") is not None and _num(_get(cache, "valid_until_s")) <= c.now
            invalid = _get(cache, "eviction_state") == "invalid"
            if (expired or invalid) and _id(cache) not in dependencies and _get(cache, "eviction_state") != "evicted":
                out.append(_emit(_id(cache), "本地缓存已过期或无效且无活动作业依赖，需要回收其真实占用。",
                                 c.refs(_id(cache), "cache.valid_until_s", "cache.eviction_state"), cache_id=_id(cache)))
    return out


def _knowledge(c, n):
    out = []
    if n == 69:
        for target in c.entities:
            if not c.relevant_entity(target) or not c.fact_keys(target):
                continue
            evidence = c.evidence_about(target)
            sources = {_get(row, "by", _get(row, "from_agent")) for row in evidence}
            frames = {repr(_get(row, "frame", "local_ENU_m")) for row in evidence}
            times = {_num(_get(row, "t", _get(row, "captured_at_s", 0))) for row in evidence}
            if len(sources) >= 2 and (len(frames) > 1 or len(times) > 1):
                out.append(_emit(target, "本机收到跨源且时间或坐标不同的目标证据，融合前需对齐并保留变换误差。",
                                 [(_id(row), "observation.captured_at_s") for row in evidence],
                                 evidence_ids=[_id(row) for row in evidence]))
    elif n == 70:
        for task in c.task_rows({"accepted", "executing", "blocked", "suspended"}):
            target = _get(task, "target")
            if c.evidence_about(target) and not c.marked("history_queries", _id(task)):
                out.append(_emit(_id(task), "当前任务目标在本机私有历史中有记录，需按任务条件检索本机有权访问的证据。",
                                 c.refs(_id(task), "task.target_ref", "task.revision") + c.refs(target, "belief.evidence_ids"),
                                 original_task_id=_id(task)))
    elif n == 71:
        for evidence in c.evidence:
            eid = _id(evidence)
            size = _num(_get(evidence, "payload_bytes", _get(evidence, "size_bytes", 0)))
            if size > _num(c.records.get("encoding_threshold_bytes", 4096), 4096) and not c.marked("encoded_evidence", eid):
                out.append(_emit(eid, "本机原始证据超过传输打包阈值且尚未编码，需要保留源链接、内容哈希和实际字节。",
                                 c.refs(eid, "observation.payload_bytes", "observation.content_hash"), evidence_ids=[eid]))
    elif n == 72:
        for eid, group in c.entities.items():
            facts = c.fact_keys(eid)
            if not facts:
                continue
            uncertain = [key for key in facts if _get(group[key], "source") != "prior" and
                         _num(_get(group[key], "confidence", 1), 1) < 0.6]
            status = c.v(eid, "belief.search_status")
            if uncertain or status == "inconclusive":
                out.append(_emit(eid, "本地证据低置信度或搜索仍不确定，需判断充分性；未检测不是不存在。",
                                 c.refs(eid, *(uncertain or ["belief.search_status"])),
                                 conditional_sequence="Q097" if c.v(eid, "cue.signal_state") == "occluded" else None))
    elif n == 73:
        for eid in c.entities:
            if c.fact_keys(eid) and c.disputed(eid):
                out.append(_emit(eid, "新旧报告在本地形成显式冲突，需要多源更正并将更正实际发送，不能按发送者身份当真值。",
                                 c.conflict_refs(eid) + c.refs(eid, "belief.source_agent_ids"),
                                 conditional_sequence="Q112"))
    elif n == 74:
        targets = {_get(task, "target") for task in c.task_rows()}
        for evidence in c.evidence:
            eid = _id(evidence)
            relevant = bool(set(_list(_get(evidence, "about", []))) & targets)
            if relevant and not c.marked("shared_evidence", eid):
                out.append(_emit(eid, "本机新证据与当前任务相关但尚未共享，需要按业务预算选择文本、裁剪或表示并真实传输。",
                                 c.refs(eid, "observation.payload_bytes", "observation.parent_evidence_ids"), evidence_ids=[eid]))
    elif n == 75:
        threshold = _num(c.records.get("belief_stale_after_s", 120), 120)
        for eid, group in c.entities.items():
            facts = c.fact_keys(eid)
            if not facts:
                continue
            stale = [key for key in facts if key in DYNAMIC_FACT_KEYS and
                     _get(group[key], "source") != "prior" and c.age(eid, key) > threshold]
            # Expired information is only useful to maintain for a live domain
            # need.  Static damage/use/geometry and our own metadata never age
            # into endless maintenance tasks.
            relevant = bool(c.targeting(eid)) or c.is_kind(eid, "human") or c.is_kind(eid, "hazard")
            if (stale and relevant) or c.disputed(eid):
                out.append(_emit(eid, "本地动态信息已过时或有冲突，需按来源和时间维护信念，不能继续当作最新事实。",
                                 c.refs(eid, *stale) + c.conflict_refs(eid),
                                 conditional_sequence="Q112" if c.disputed(eid) else None))
        for job in c.jobs:
            if c.job_stale(job):
                out.append(_emit(_id(job), "已收推理结果引用旧任务版本，应隔离为历史信息而非更新当前任务完成证据。",
                                 c.refs(_id(job), "job.task_ref", "job.result_ref"), conditional_sequence="Q108"))
    elif n == 76:
        pending = [row for row in c.result_index["domain"] if _get(row, "status") in ("succeeded", "partial", "failed", "rejected", "canceled")
                   and _get(row, "instance_id") and not c.marked("archived_action_ids", str(_get(row, "instance_id")))]
        last = _num(c.records.get("archived_at_s", c.records.get("archive_checked_at", 0)))
        requested = c.records.get("archive_requested") is True
        threshold = max(4, int(_num(c.records.get("archive_batch_size", 8), 8)))
        if pending and (requested or len(pending) >= threshold and c.now - last >= 120):
            instances = [str(_get(row, "instance_id")) for row in pending]
            out.append(_emit(instances[0], "本机已积累一批未归档的实际作业反馈，按批记录证据与费用链，不为管理动作递归建档。",
                             c.refs(instances[0], "action.status"),
                             action_instance=instances[0], action_instances=instances))
    return out


def _collaboration(c, n):
    out = []
    if n == 77:
        for task in c.task_rows({"received"}):
            if not _get(task, "interpreted", False):
                out.append(_emit(_id(task), "本机任务副本已实际收件但使命约束尚未解释，需要解析后决定是否承诺。",
                                 c.refs(_id(task), "task.status", "task.goal_predicate"), original_task_id=_id(task)))
        for message in c.received_index["brief"]:
            payload = _get(message, "payload", {})
            is_brief = (_get(message, "kind") in ("mission", "brief") or
                        isinstance(payload, Mapping) and payload.get("op") in ("mission", "brief"))
            if is_brief and not c.meta_ref(_get(message, "task_ref")) and not c.marked("interpreted_messages", _id(message)):
                out.append(_emit(_id(message), "实际收到新的使命简报，需要在本机解释目标与约束后纳入当前使命。",
                                 c.refs(_id(message), "message.local_availability"), message_id=_id(message)))
    elif n == 78 and c.mission():
        for eid in c.ids("building", "human", "obstacle"):
            complex_need = (c.v(eid, "human.mobility") == "trapped" or c.obstructing(eid) or
                            c.v(eid, "building.visual_damage") in ("major", "destroyed") or
                            c.v(eid, "belief.search_status") == "partial")
            if complex_need and not c.marked("decomposed_targets", eid):
                out.append(_emit(eid, "已知使命下出现受困、入口依赖或未覆盖搜索，需要基于本地证据分解可执行子目标和依赖。",
                                 c.mission() + c.refs(eid, "human.mobility", "building.visual_damage", "belief.search_status")))
    elif n == 79:
        for message in c.received_index["bids"]:
            payload = _get(message, "payload", {})
            if isinstance(payload, Mapping) and (payload.get("call_for_bids") or payload.get("command") == "bid" or
                                                  payload.get("op") == "bid_request"):
                if is_meta_task(payload.get("task")) or c.meta_ref(payload.get("task_id")) or c.meta_ref(_get(message, "task_ref")):
                    continue
                if not c.marked("bid_messages", _id(message)):
                    out.append(_emit(str(payload.get("task_id", _id(message))), "本机实际收到参与征询，需要按设备、电量及现有承诺给出可行出价或不参与理由。",
                                     c.refs(_id(message), "message.local_availability") +
                                     c.refs(c.id, "platform.capability_ids", "platform.battery_remaining_Wh"), message_id=_id(message)))
        for task in c.task_rows({"received"}):
            required = set(_list(_get(task, "required_capabilities", [])))
            if required and not required.issubset(c.caps()):
                out.append(_emit(_id(task), "实际收到的任务超出本机额定能力，需要明确不参与或提出能力匹配建议，不伪造工程完成。",
                                 c.refs(_id(task), "task.required_capability_ids") + c.refs(c.id, "platform.capability_ids"),
                                 conditional_sequence="Q100" if c.kind == "robot_dog" and
                                 _get(task, "type") == "T022" and "C16" not in c.caps() else None,
                                 original_task_id=_id(task)))
    elif n == 80:
        for task in c.task_rows({"proposed", "assigned", "failed"}):
            proposed = _get(task, "status") == "proposed" and not _get(task, "assigned_to")
            rejected = _get(task, "assignment.decision") == "rejected" or _get(task, "failure_reason") == "rejected"
            delivery = _get(task, "assignment.delivery_state", _get(task, "delivery_state"))
            assigned_times = [_num(_get(event, "t")) for event in _rows(_get(task, "history", []))
                              if _get(event, "status") == "assigned"]
            assigned_s = _get(task, "assigned_s", _get(task, "last_event_s", max(assigned_times, default=c.now)))
            age = c.now - _num(assigned_s)
            late = _get(task, "status") == "assigned" and delivery != "delivered" and age > 30
            if (proposed or rejected or late) and _get(task, "type") != "T080":
                out.append(_emit(_id(task), "本地提案等待分配、已被拒绝或分配尚未送达，需要仅依据已收机群信息协调并等待执行者接受。",
                                 c.refs(_id(task), "task.status", "assignment.delivery_state", "assignment.decision"),
                                 original_task_id=_id(task), conditional_sequence="Q109" if late else None))
        for result in c.result_failure("incompatible", "insufficient"):
            if _get(result, "action_id") == "A024":
                out.append(_emit(_id(result, c.id), "本地挂载反馈显示质量或接口超限，需要拆分可分物资或选择合适载具，不能复制库存。",
                                 c.refs(_id(result, c.id), "action.rejection_code"), conditional_sequence="Q101"))
    elif n == 81:
        for task in c.task_rows({"received"}):
            if not _get(task, "accepted_by") and _get(task, "assignment.decision", "unknown") == "unknown":
                out.append(_emit(_id(task), "任务在本机已收件但尚未承诺，必须按本机能力、资源和不可抢占承诺明确接受或拒绝。",
                                 c.refs(_id(task), "task.status", "assignment.decision") +
                                 c.refs(c.id, "platform.capability_ids", "platform.battery_remaining_Wh"), original_task_id=_id(task)))
    elif n == 82:
        for task in c.task_rows({"executing", "blocked"}):
            if _get(task, "type") in ("T076", "T082"):
                continue
            evidence = _get(task, "result_evidence_ids", [])
            progress = _get(task, "progress", {})
            if evidence or (isinstance(progress, Mapping) and progress):
                if not _get(task, "progress_report_acknowledged", False):
                    out.append(_emit(_id(task), "本机任务已有真实反馈或进度证据但未确认上报，需要核对目标并取得进度回执。",
                                     c.refs(_id(task), "task.progress", "task.result_evidence_ids"), original_task_id=_id(task)))
    elif n == 83:
        for task in c.task_rows({"accepted", "executing", "blocked"}):
            if _get(task, "type") in ("T043", "T044", "T045", "T083", "T091"):
                continue
            canceled = _get(task, "cancel_requested") is True
            for message in c.received_index["cancel"].get(_id(task), ()):
                payload = _get(message, "payload", {})
                if isinstance(payload, Mapping) and payload.get("task_id") == _id(task):
                    canceled = canceled or payload.get("command", payload.get("op")) in ("cancel", "preempt")
            interruptible = _get(task, "interruptibility") not in (False, "non_preemptible", "non_interruptible", "atomic")
            if canceled or (c.low() and interruptible):
                out.append(_emit(_id(task), "本机收到取消抢占指令或安全电量要求中断，需要在安全点停止并保留已耗成本。",
                                 c.refs(_id(task), "task.status", "task.interruptibility") +
                                 c.refs(c.id, "platform.battery_remaining_Wh"), original_task_id=_id(task),
                                 conditional_sequence="Q102" if c.low() else None))
    elif n == 84:
        remaining, maximum, reserve = c.battery()
        healthy = c.own("platform.health_state", "nominal") == "nominal"
        for task in c.task_rows({"suspended"}):
            target = _get(task, "target")
            access = c.v(target, "road.entry_state")
            if (healthy and (maximum <= 0 or remaining >= max(0.6 * maximum, reserve)) and
                    access not in ("blocked", "unsafe", "closed") and c.needed(task)):
                out.append(_emit(_id(task), "本机挂起任务仍有需要，补能后健康与已知通行条件允许恢复，应重验版本并从检查点继续。",
                                 c.refs(_id(task), "task.status", "task.resume_checkpoint") +
                                 c.refs(c.id, "platform.battery_remaining_Wh", "platform.health_state"),
                                 original_task_id=_id(task), conditional_sequence="Q102"
                                 if c.records.get("charging_needed") or _get(task, "suspension_reason") == "energy" else None))
    elif n == 85:
        for task in c.task_rows({"accepted", "executing", "blocked", "suspended"}):
            relay_exit = _get(task, "type") == "T049" and c.low(0.3)
            handoff = _get(task, "handoff_target_id")
            if relay_exit or handoff:
                out.append(_emit(_id(task), "本机承诺需要交接或中继电量要求退出，必须新方实际确认并接管后才释放旧方。",
                                 c.refs(_id(task), "task.handoff_target_id", "task.status") +
                                 c.refs(c.id, "platform.battery_remaining_Wh"), original_task_id=_id(task),
                                 handoff_target_id=handoff, conditional_sequence="Q105" if relay_exit else None))
    elif n == 86:
        for task in c.task_rows({"failed", "blocked", "completed"}):
            if _get(task, "type") in ("T076", "T080", "T082", "T086", "T088") or _num(_get(task, "retry_count")) >= 2:
                continue
            target = _get(task, "target")
            conflict = c.disputed(target)
            failed = _get(task, "status") == "failed"
            blocked = _get(task, "status") == "blocked" and _get(task, "failure_reason") in ("blocked", "access", "stale", "capability")
            if ((failed or blocked) and c.needed(task)) or (conflict and _get(task, "still_needed") is not False):
                out.append(_emit(_id(task), "本机任务失败或新证据推翻旧判断，目标在本地仍未满足，需要新版本重规划并撤销旧承诺。",
                                 c.refs(_id(task), "task.status", "task.failure_reason", "task.revision") +
                                 c.conflict_refs(target), original_task_id=_id(task),
                                 conditional_sequence="Q112" if conflict else
                                 ("Q098" if _get(task, "failure_reason") in ("blocked", "access") else None), max_retries=2))
        for job in c.jobs:
            if c.job_stale(job):
                out.append(_emit(_id(job), "已知作业结果已落后本机任务版本，应隔离旧结果并据当前证据重规划。",
                                 c.refs(_id(job), "job.task_ref", "job.result_ref"), job_id=_id(job), conditional_sequence="Q108"))
    elif n == 87:
        for eid in c.ids("obstacle", "human"):
            trapped = c.v(eid, "human.mobility") == "trapped" or _num(c.v(eid, "human.buriedness")) > 0
            volume = _num(c.v(eid, "obstacle.volume_m3"))
            obstacle = c.obstructing(eid)
            required = "C17" if volume > 0.5 else "C16"
            specialist_route = c.available([("C18", "C19", "C20")])
            incapable = (trapped and not {"C18", "C19", "C20"}.issubset(c.caps()) and
                         not (c.is_coordinator and specialist_route)) or (obstacle and required not in c.caps())
            if incapable:
                searches = c.task_rows({"accepted", "executing", "blocked"}, {"T014", "T015"})
                out.append(_emit(eid, "本机已观测受困或阻路需求超出额定装备，需以A059明确请求专业主体并等待响应及实际结果。",
                                 c.refs(eid, "human.mobility", "human.buriedness", "obstacle.volume_m3", "obstacle.blocks_ids") +
                                 c.refs(c.id, "platform.capability_ids"),
                                 requested_action_id="A017" if trapped else ("A013" if required == "C17" else "A012"),
                                 specialist_required=True,
                                 related_task_id=_id(searches[0]) if searches else None,
                                 conditional_sequence="Q099" if obstacle and searches else None))
    elif n == 88:
        for eid in c.ids("building", "open_area", "region"):
            status = c.v(eid, "belief.search_status")
            if status in ("complete", "partial", "inconclusive") and not c.marked("closed_worksites", eid):
                terminal = [t for t in c.task_rows({"completed", "failed"}, {"T014", "T015", "T013", "T009"})
                            if _get(t, "target") == eid]
                if terminal:
                    out.append(_emit(eid, "现场搜索阶段已产出终止报告，需要交代实际覆盖、未覆盖及残余危险并落实后续承担者。",
                                     c.refs(eid, "belief.search_status", "building.remaining_hazard", "building.worksite_record"),
                                     conditional_sequence="Q110" if status in ("partial", "inconclusive") else None,
                                     preserve_unknown=True))
    return out


def _support(c, n):
    out = []
    if n == 89:
        for eid in c.ids("hazard", "road", "building"):
            secondary = c.v(eid, "hazard.secondary_event")
            fire = _num(c.v(eid, "hazard.fire_intensity")) > 0
            reblocked = c.v(eid, "road.entry_state") == "blocked" and c.age(eid, "road.entry_state") < 30
            known_secondary = secondary not in UNKNOWN + (False, "none", [], {})
            if (known_secondary and c.v(eid, "hazard.active") is not False) or fire or reblocked:
                if not c.marked("hazard_monitoring", eid):
                    out.append(_emit(eid, "本地已知次生危险、活动火势或新阻塞，需要持续观测后报告，不能无代价获得全局变化。",
                                     c.refs(eid, "hazard.secondary_event", "hazard.fire_intensity", "road.entry_state")))
    elif n == 90:
        last = _num(c.records.get("health_checked_at", -1000), -1000)
        health = c.own("platform.health_state")
        abnormal = health in ("degraded", "failed", "recovering") or c.low(0.25)
        requested = c.records.get("health_monitoring_requested") is True
        # Nominal telemetry and the mere passage of time are not new work.  The
        # runtime already reports health periodically; an explicit monitor is
        # bounded and anomaly-driven rather than an initial high-volume task.
        if (abnormal or requested) and c.now - last >= 120:
            out.append(_emit(c.id, "本机异常遥测或明确监测请求需要实际复核健康与能源估计，不把例行上报扩增为诊断任务。",
                             c.refs(c.id, "platform.health_state", "platform.battery_remaining_Wh"), maintenance=True))
    elif n == 91:
        for task in c.task_rows({"blocked"}):
            reason = _get(task, "failure_reason")
            waiting = reason in ("dependency", "queue", "specialist", "lost_link", "network", "hazard", "access")
            if waiting and _get(task, "type") != "T091":
                out.append(_emit(_get(task, "target", c.id), "本机任务正等待依赖、专业响应或危险解除，需安全驻留观察并计持续资源开销。",
                                 c.refs(_id(task), "task.status", "task.dependency_ids", "task.failure_reason"),
                                 waiting_task_id=_id(task), maintenance=True))
        for station in c.ids("station"):
            if c.id in _list(c.v(station, "station.queue_ids", [])):
                out.append(_emit(station, "本机已知在服务站排队，需要安全等待并保持观测，排队不等于服务完成。",
                                 c.refs(station, "station.queue_ids", "station.occupied"), maintenance=True))
    elif n == 92:
        for eid in c.ids("equipment", "sensor"):
            key = "equipment.health" if c.is_kind(eid, "equipment") else "sensor.health"
            if c.v(eid, key) in ("degraded", "failed"):
                out.append(_emit(eid, "本地设备或传感器健康记录异常，需要诊断、隔离或报告无法恢复，不能强制成功。",
                                 c.refs(eid, key)))
        for service in c.services:
            if _get(service, "status") == "failed":
                out.append(_emit(_id(service), "本地收到服务失败反馈，需要诊断并根据检查点和兼容性决定恢复或替代。",
                                 c.refs(_id(service), "service.status"), service_id=_id(service), conditional_sequence="Q107"))
        for result in c.result_failure("sensor", "model", "actuator"):
            params = _get(result, "params", {})
            service_id = _get(params, "service_id")
            if c.kind == "compute_node" and not service_id:
                continue
            target = service_id or _get(params, "equipment_id", _get(params, "sensor_id", c.id))
            out.append(_emit(target, "本机动作返回传感、模型或执行器故障，需要针对本机明确资源实际诊断后才能继续依赖。",
                             c.refs(_id(result, c.id), "action.failure_code") +
                             c.refs(target, "service.status" if service_id else "platform.health_state"),
                             service_id=service_id, maintenance=True))
    elif n == 93:
        for eid in c.ids("hazard"):
            if _num(c.v(eid, "hazard.fire_intensity")) > 0 and c.v(eid, "hazard.active") is not False:
                out.append(_emit(eid, "本地证据显示仍有活动火势，需具备消防能力与实际介质的主体灭火后核验，允许复燃。",
                                 c.refs(eid, "hazard.fire_intensity", "hazard.active")))
    elif n == 94:
        for eid in c.ids("equipment"):
            kind = str(c.v(eid, "equipment.kind", "")).lower()
            medium = str(c.v(eid, "equipment.consumable_type", "")).lower()
            remaining = c.v(eid, "equipment.consumable_remaining")
            if (kind in ("fire_hose", "pump", "extinguisher", "fire_suppression") or medium in ("water", "foam")) and (
                    remaining is not None and _num(remaining) <= 0):
                out.append(_emit(eid, "消防设备实际灭火介质已耗尽，需要从已知水源或库存补充，不能无限喷水。",
                                 c.refs(eid, "equipment.kind", "equipment.consumable_type", "equipment.consumable_remaining")))
        fire_tasks = c.task_rows(types={"T093"})
        inventory = c.records.get("inventory", {})
        if fire_tasks and isinstance(inventory, Mapping) and "water" in inventory and _num(inventory["water"]) <= 0:
            out.append(_emit(c.id, "本机承担消防承诺但供水库存已空，需要真实补充灭火介质后恢复作业。",
                             c.refs(c.id, "platform.payload_item_ids") + c.refs(_id(fire_tasks[0]), "task.status"), item_kind="water"))
    elif n == 95:
        for eid in c.ids("hazard", "building"):
            active = (c.v(eid, "hazard.active") is True or _num(c.v(eid, "hazard.gas_level")) > 0 or
                      _num(c.v(eid, "hazard.fire_intensity")) > 0 or
                      c.v(eid, "building.structural_condition") in ("compromised", "unstable", "collapsed", "unsafe"))
            if active and not c.v(eid, "hazard.exclusion_zone"):
                out.append(_emit(eid, "已知危险源或不稳定结构尚未建立警戒隔离，需授权主体隔离并同步已知通行风险。",
                                 c.refs(eid, "hazard.active", "hazard.gas_level", "hazard.fire_intensity",
                                        "building.structural_condition", "hazard.exclusion_zone")))
    elif n == 96:
        adapters = _rows(c.records.get("representation_adapters", []))
        local_evidence = {_id(row) for row in c.evidence}
        for cache in c.caches:
            if (_get(cache, "eviction_state", "resident") != "resident" or
                    _num(_get(cache, "valid_until_s", c.now + 1)) <= c.now):
                continue
            cache_kind = str(_get(cache, "kind", "")).lower()
            if cache_kind not in ("features", "embedding", "feature", "latent", "representation", "kv"):
                continue
            for job in c.jobs:
                source_version, target_version = _get(cache, "model_version"), _get(job, "model_version")
                if (source_version is None or target_version is None or source_version == target_version or
                        _get(job, "status") != "created"):
                    continue
                contents = set(_list(_get(cache, "content_ids", [])))
                if not contents.intersection(_list(_get(job, "input_evidence_ids", []))):
                    continue
                for adapter in adapters:
                    proofs = set(_list(_get(adapter, "evidence_ids", [])))
                    exact = (_get(adapter, "source_version") == source_version and
                             _get(adapter, "target_version") == target_version and
                             _get(adapter, "compatible") is True and proofs and proofs.issubset(local_evidence))
                    # KV additionally needs explicit layer/dimension/prefix proof.
                    kv_ok = cache_kind != "kv" or all(_get(adapter, key) is True for key in
                                                                       ("layers_match", "dimensions_match", "prefix_match"))
                    if exact and kv_ok:
                        out.append(_emit(_id(cache), "本地表示与目标模型版本不同，但已有显式适配兼容证据，可请求受约束转换，不能通用复制KV。",
                                         c.refs(_id(cache), "cache.kind", "cache.model_version", "cache.content_ids") +
                                         c.refs(_id(job), "job.service_id", "job.input_evidence_ids") +
                                         [(proof, "observation.parent_evidence_ids") for proof in sorted(proofs)],
                                         cache_id=_id(cache), job_id=_id(job), adapter_id=_id(adapter),
                                         compatibility_evidence_ids=_list(_get(adapter, "evidence_ids"))))
                        break
    return out


# Q097--Q112 are catalog conditional paths, not unconditional task requests.
# Membership and Chinese trigger text are copied from the original JSON.
CONDITIONAL_TRIGGERS = {
    "Q097": {"tasks": ["T011", "T072"], "trigger": "遮挡/分辨率不足"},
    "Q098": {"tasks": ["T020", "T021", "T086"], "trigger": "移动实际发现阻碍"},
    "Q099": {"tasks": ["T014", "T023", "T087"], "trigger": "搜索入口受阻，当前主体无工程能力"},
    "Q100": {"tasks": ["T022", "T079", "T081"], "trigger": "DOG无夹爪/无额定清障能力"},
    "Q101": {"tasks": ["T030", "T038", "T080"], "trigger": "物资质量/接口超出当前构型"},
    "Q102": {"tasks": ["T043", "T045", "T083", "T084"], "trigger": "剩余能量接近安全保留阈值"},
    "Q103": {"tasks": ["T052", "T053"], "trigger": "接收数据不完整或校验失败"},
    "Q104": {"tasks": ["T054", "T056"], "trigger": "消息缓存时链路不可用"},
    "Q105": {"tasks": ["T049", "T085"], "trigger": "原中继需要退出"},
    "Q106": {"tasks": ["T058", "T060", "T061"], "trigger": "准入时内存不足"},
    "Q107": {"tasks": ["T060", "T065", "T092"], "trigger": "服务在执行中失败"},
    "Q108": {"tasks": ["T066", "T075", "T086"], "trigger": "推理期间任务版本已变化"},
    "Q109": {"tasks": ["T080", "T081"], "trigger": "中心已提交，执行方尚未收件"},
    "Q110": {"tasks": ["T014", "T015", "T088"], "trigger": "抽象搜索结束但覆盖不全或未检测"},
    "Q111": {"tasks": ["T039", "T041", "T045"], "trigger": "执行器故障或落点不满足条件"},
    "Q112": {"tasks": ["T073", "T075", "T086"], "trigger": "新报告与旧判断矛盾"},
}

# Each row: catalog name | proposer roles | proposed priority | provider anchor.
# '+' means capabilities on ONE provider; '|' means an alternative provider.
# These are availability anchors, NOT the allocator's full method requirements.
# Full action/method capability, access, consent and resource guards stay runtime
# responsibilities. 'any' permits a discovering agent to propose external work.
_TASK_TABLE = """
区域快速普查|coordinator|66|C01+C05
建筑外部损毁评估|coordinator|70|C01+C05
道路与通路灾后评估|coordinator|50|C01+C05|C02+C05
入口与屋顶交付区检查|any|55|C09
水域边界与涉水条件评估|any|50|C01+C05
火烟与危险源侦察|any|75|C05
灾前灾后变化与地图更新|any|45|C44
建筑/区域覆盖登记|any|30|C39
系统空中人员搜索|any|65|C01+C05
线索发现与语义解读|any|60|C12
疑似目标多视角复核|any|72|C05+C44
目标定位跟踪与跨视角关联|any|72|C04+C44
近地外部与狭窄区搜索|any|64|C02+C09
建筑内部初次抽象搜索|any|70|C15
建筑内部复搜/完整搜索|any|68|C15
热成像补充搜索|any|65|C06
声音线索搜索|any|67|C07
RF/手机线索搜索|any|64|C08
空中转场与观察点到达|uav|55|C01
地面转场与绕行|robot_dog ground_vehicle specialist_team|58|C02
障碍接近探测与路径验证|any|70|C09
轻型残骸移除|any|75|C16
重型道路清障|any|77|C17
入口开辟与局部破拆|any|78|C18
危险结构临时支撑|any|90|C19
吊升/移置大块障碍|any|80|C20
建立联系并确认需求|any|78|C11
安全警示与避险引导|any|88|C11
专业伤员评估与分级|any|86|C21
人员基础物资支援|any|79|C24
专业初步救护|any|94|C21
受困解除/解困|any|95|C18+C19+C20
人员装载|any|85|C22
人员转运|any|87|C22
救援/医疗最终移交|any|91|C22
水中人员救助回收|any|97|C03+C23
救援转恢复阶段的遗体移交|any|40|C22+C46
货物取用与挂载|any|59|C24
无人机精确物资投送|uav|74|C25
地面物资送达与卸放|ground_vehicle specialist_team|74|C26
交付复核与收货报告|any|83|C31
物资/器材补给|any|63|C27
平台充电|uav robot_dog ground_vehicle boat specialist_team|96|C28
平台换电|uav robot_dog ground_vehicle boat specialist_team|95|C29
返航/返站与安全撤离|uav robot_dog ground_vehicle boat specialist_team|100|C45
失效平台回收|any|88|C30
链路探测与质量估计|any|49|C31
通信会合与重定位|uav robot_dog ground_vehicle boat specialist_team|84|C31
空中中继保持|uav coordinator|76|C01+C32
通信节点布放|any|74|C33
多跳业务转发|any|62|C32
图像/证据数据上传|any|61|C31+C41
可靠回执与超时重传|any|84|C31
断联存储-携带-转发|uav robot_dog ground_vehicle|73|C34
信道/功率/传输优先级调整|any|69|C32
失联后的连接恢复|any|81|C31
计算与模型服务发现|any|47|C35
本地视觉检测|any|54|C12+C35
本地语义/任务推理|any|53|C13+C35
远程视觉复核/推理卸载|any|71|C38
执行节点与模型选择|any|56|C35
计算作业准入与排队|any|55|C35
模型服务加载/卸载|any|59|C37
上下文/表示缓存复用|any|38|C39
服务/作业迁移与恢复|any|80|C37+C38
预算耗尽/失败作业取消重试|any|79|C35
批处理/多源联合推理|any|51|C35+C44
计算资源释放与缓存清理|any|58|C35
跨源时间坐标对齐|any|60|C40
任务条件化本地历史检索|any|44|C39
图像裁剪/编码/证据打包|any|43|C41
置信度与证据充分性判断|any|67|C44
多源融合与冲突更正|any|82|C44
面向任务的信息选择与共享|any|57|C39+C41
局部信念与过时信息维护|any|64|C39
轨迹/证据归档与回放|any|20|C39
接收并解释当前使命|any|70|C43
自主任务分解与候选生成|coordinator|62|C42
本机参与筛选与出价|any|65|C43
多平台任务分配与协调|coordinator|72|C42
任务接受/拒绝与承诺|any|85|C43
进度监控与完成确认|any|46|C43
任务取消/抢占|any|99|C43
恢复原任务|any|78|C43
跨Agent任务与证据交接|any|86|C43
失败重规划与重派|any|83|C42
请求专业救援/工程力量|any|89|C46
工作现场闭合与阶段移交|any|39|C43
次生危险动态监视|any|87|C05
平台/能源/服务健康监视|any|48|C43
安全等待与驻留观察|any|52|C45
设备/传感器/服务故障诊断|any|92|C43
消防抑制/灭火|any|96|C47
消防供水与灭火耗材补充|any|89|C27
危险源隔离/警戒区建立|any|93|C48
跨模型特征/潜表示适配|any|42|C49
""".strip().splitlines()


def _evaluate(c, number):
    if number <= 8:
        return _assessment(c, number)
    if number <= 18:
        return _search(c, number)
    if number <= 26:
        return _engineering(c, number)
    if number <= 37:
        return _personnel(c, number)
    if number <= 46:
        return _logistics(c, number)
    if number <= 56:
        return _communication(c, number)
    if number <= 68:
        return _compute(c, number)
    if number <= 76:
        return _knowledge(c, number)
    if number <= 88:
        return _collaboration(c, number)
    return _support(c, number)


def _make_rule(number, row):
    name, role_text, priority_text, *provider_text = row.split("|")
    task_type = f"T{number:03d}"
    roles = role_text.split()
    alternatives = [tuple(option.split("+")) for option in provider_text if option]
    refs = [task_type, f"Q{number:03d}", "U13", "U14"]
    refs.extend(qid for qid, entry in CONDITIONAL_TRIGGERS.items() if task_type in entry["tasks"])
    if number in (1, 2, 8, 14, 15):
        refs.append("U06")
    if number in (14, 15):
        refs.extend(("U05", "G02"))
    if number in (30, 34, 35, 39, 40, 41):
        refs.append("U12")
    if 57 <= number <= 68 or number in (43, 44, 76):
        refs.append("U11")

    def when(agent, sim, *, context=None):
        c = context if context is not None else _Context(agent, sim)
        role = "coordinator" if c.is_coordinator else c.kind
        if ("any" not in roles and role not in roles and c.kind not in roles) or not c.role_allows(number):
            return []
        if not c.available(alternatives):
            return []
        seen, result = set(), []
        for target, why, belief_refs, extra in _evaluate(c, number):
            if target in seen or c.meta_ref(target):
                continue
            seen.add(target)
            details = dict(extra)
            details.setdefault("name_zh", name)
            if number in (11, 27) and c.camera_candidate(target):
                # Single-frame tracks remain below survey to bound churn. Two
                # associated captures justify one physical contact ahead of even
                # damage-prioritized T014 (80); they do not confirm a truth ID.
                # Keep frame IDs/timestamps out of the proposal value signature.
                if number == 27:
                    details["priority"] = c.camera_contact_priority(target)
                else:
                    details["priority"] = min(details.get("priority", int(priority_text)), 63)
            original = details.get("original_task_id", details.get("related_task_id", details.get("waiting_task_id")))
            if original:
                if c.meta_ref(original):
                    continue
                belief_refs = list(dict.fromkeys(belief_refs + c.refs(original, "task.status", "task.revision")))
            result.append((target, why, belief_refs, details))
        return result

    when.__name__ = f"when_{task_type}"
    when.__doc__ = f"{task_type} {name}: agent-local trigger; never an executable task request fallback."
    return Rule(f"R-{task_type}-a", task_type, roles, when, int(priority_text), refs)


RULES: list[Rule] = [_make_rule(number, row) for number, row in enumerate(_TASK_TABLE, 1)]
RULES_BY_TASK: dict[str, Rule] = {rule.task_type: rule for rule in RULES}
assert set(RULES_BY_TASK) == {f"T{number:03d}" for number in range(1, 97)}, (
    "Task generation registry must cover exactly T001-T096"
)
TASK_NAMES: dict[str, str] = {f"T{number:03d}": row.split("|", 1)[0]
                              for number, row in enumerate(_TASK_TABLE, 1)}


def _pending_proposals(agent):
    """Only the holder's task copies and in-flight A050 requests are inspected."""
    active = {(_get(task, "type"), str(_get(task, "target")))
              for task in _rows(getattr(agent, "tasks", {})) if _get(task, "status") in ACTIVE
              or _get(task, 'progress.pending_camera_analysis')}
    for action in _rows(getattr(agent, "pending_actions", {})):
        if _get(action, "action_id", _get(action, "action")) != "A050":
            continue
        params = _get(action, "params", {})
        if isinstance(params, Mapping):
            proposal = params.get("proposal", params)
            if isinstance(proposal, Mapping):
                active.add((proposal.get("type", proposal.get("task_type")), str(proposal.get("target"))))
    return active


def candidates(agent, sim, limit=None):
    """Return deterministic A050 proposal candidates; never mutate or submit.

    Runtime callers may instead iterate RULES and call when directly.  They must
    suppress duplicate (type, target) records, including pending A050 actions,
    before submitting.  This helper performs that same holder-local suppression
    and also removes duplicates inside this batch.  Completed/failed work is not
    globally blacklisted: fresh evidence may legitimately reopen a target.
    Runtime throttling, bounded retries and recording handled artifact ids keep
    periodic/lifecycle rules from being re-executed on unchanged local inputs.
    """
    suppressed = _pending_proposals(agent)
    context = _Context(agent, sim)
    result = []
    for rule in RULES:
        allowed = agent.spec.get("task_allowlist")
        if allowed is not None and rule.task_type not in allowed:
            continue  # Explicit mission scope; predicates still read local beliefs only.
        for target, why, belief_refs, extra in rule.when(agent, sim, context=context):
            key = (rule.task_type, str(target))
            if key in suppressed:
                continue
            suppressed.add(key)
            result.append({**dict(extra), "type": rule.task_type, "task_type": rule.task_type,
                           "target": target, "name_zh": TASK_NAMES[rule.task_type],
                           "why_zh": why, "belief_refs": belief_refs, "extra": dict(extra),
                           "priority": extra.get("priority", rule.priority), "rule_id": rule.id, "refs": list(rule.refs),
                           "origin": {"rule_id": rule.id, "why_zh": why, "belief_refs": belief_refs}})
    result.sort(key=lambda item: (-item["priority"], item["type"], str(item["target"])))
    return result if limit is None else result[:max(0, int(limit))]


def generate(agent, sim, limit=None):
    """Alias for pure candidate construction; A050 remains the runtime's job."""
    return candidates(agent, sim, limit=limit)
