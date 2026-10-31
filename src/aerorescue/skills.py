"""Unified decision primitives: the 66-action space, skill groups, candidates.

This module is the single place that answers three questions for any harness
(center sandbox, edge harness, scripted policy):

1. which catalog action ids exist (:data:`VALID_ACTION_IDS`, ``A001``-``A066``);
2. which human-readable skill a catalog method sequence belongs to
   (:data:`SKILL_LABELS`, :func:`sequence_skill`);
3. what this holder may execute next (:func:`build_decision_candidates`) and how
   that choice enters the one real state machine (:func:`execute_candidate`).

``execute_candidate`` never bypasses :meth:`AgentRuntime._submit`: every chosen
candidate becomes a kernel action instance in ``AgentRuntime.pending_actions``
and is settled by ``AgentRuntime.on_action_result``. A decision is therefore
auditable exactly like an internally selected step.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math

# A001-A063 are the frozen catalog actions; A064 adapts cross-model state,
# A065 receives a message, A066 replans around a locally observed obstacle.
VALID_ACTION_IDS = frozenset(f"A{i:03d}" for i in range(1, 67))

# Ten high-level skills plus the bounded recovery lane. Order is the report
# order; each label owns every main-path sequence classified into it.
SKILL_LABELS = (
    ("search_and_assess", "搜索与评估"),
    ("contact_person", "人员接触确认"),
    ("extricate_person", "人员解困"),
    ("pickup_supply", "物资获取"),
    ("aid_person", "人员救护"),
    ("load_and_evacuate", "装载与后送移交"),
    ("water_rescue", "水域救助"),
    ("suppress_fire", "火情压制"),
    ("isolate_hazard", "危险源隔离"),
    ("clear_obstacle", "通道清障"),
    ("recover_replan", "受阻恢复与重规划"),
)
SKILL_NAMES = tuple(name for name, _ in SKILL_LABELS)
SKILL_NAME_ZH = dict(SKILL_LABELS)
RECOVERY_SKILL = "recover_replan"

# First matching hint wins; order encodes "the defining physical act".
_SKILL_HINTS = (
    ("water_rescue", {"A023"}),
    ("suppress_fire", {"A062"}),
    ("isolate_hazard", {"A063"}),
    ("clear_obstacle", {"A011", "A012", "A013"}),
    ("extricate_person", {"A014", "A015", "A016", "A017"}),
    ("load_and_evacuate", {"A020", "A021", "A022"}),
    ("aid_person", {"A019", "A027"}),
    ("contact_person", {"A009", "A018"}),
    ("pickup_supply", {"A024", "A028"}),
    ("search_and_assess", {"A005", "A006", "A008", "A040", "A041"}),
)

# An onsite action is executed where the work is; a harness may always offer a
# costed approach to that site first.
ONSITE_ACTIONS = frozenset({
    "A008", "A009", "A011", "A012", "A013", "A014", "A015", "A016", "A017",
    "A018", "A019", "A020", "A023", "A024", "A025", "A026", "A027", "A062", "A063",
})
APPROACH_RADIUS_M = 10.0
MOVEMENT_BY_MOBILITY = {"air": "A001", "water": "A003", "legged": "A002",
                        "wheeled": "A002", "foot": "A002"}


def sequence_skill(seq_id, catalog=None):
    """Classify one catalog sequence id into a :data:`SKILL_NAMES` label."""
    if not isinstance(seq_id, str) or not seq_id:
        raise ValueError("方法序列编号无效 / invalid method sequence id")
    if catalog is None:
        from rescue_scenes.core.catalog import load
        catalog = load()
    row = catalog.sequences.get(seq_id)
    if row is None:
        raise ValueError(f"未知方法序列 / unknown method sequence: {seq_id}")
    if row.get("kind") == "条件分支":
        return RECOVERY_SKILL
    actions = {action for step in catalog.steps.get(seq_id, ())
               for action in step.get("options", ())}
    for skill, hints in _SKILL_HINTS:
        if actions & hints:
            return skill
    # The frozen label set has no separate information skill: a method made only
    # of sensing/report/control acts is observe-and-report work.
    return "search_and_assess"


def skill_sequences(catalog=None):
    """``{skill: (seq_id, ...)}`` derived from the catalog, never hand-listed."""
    if catalog is None:
        from rescue_scenes.core.catalog import load
        catalog = load()
    grouped = {name: [] for name in SKILL_NAMES}
    for seq_id in sorted(catalog.sequences):
        grouped[sequence_skill(seq_id, catalog)].append(seq_id)
    return {name: tuple(ids) for name, ids in grouped.items()}


def _digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _point(value):
    if isinstance(value, dict):
        value = value.get("position", value.get("xy", value.get("coords")))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            return None
    return None


@dataclass
class CandidateAction:
    """One executable decision, bound to the task revision it was built from.

    ``task_id``/``task_rev``/``method_seq``/``step_index`` are ``None`` only when
    the candidate is not bound to a task. ``purpose`` is ``"approach"`` (a costed
    move to the work site that never advances the method cursor) or ``"method"``
    (one catalog step). A recovery step still carries ``purpose="method"`` so the
    runtime's result handler treats it as a method step; ``method_scope`` keeps
    the main/recovery distinction visible to the harness.
    """

    candidate_id: str
    skill_name: str
    action_id: str
    purpose: str = "method"
    method_scope: str = "main"
    task_id: str | None = None
    task_rev: int | None = None
    task_type: str | None = None
    method_seq: str | None = None
    step_index: int | None = None
    target_id: str | None = None
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.action_id, str) or self.action_id not in VALID_ACTION_IDS:
            raise ValueError(f"动作编号超出 A001-A066 / unknown action: {self.action_id!r}")
        if self.purpose not in {"approach", "method"}:
            raise ValueError("候选目的须为 approach 或 method / invalid candidate purpose")
        if self.method_scope not in {"main", "recovery"}:
            raise ValueError("候选方法域须为 main 或 recovery / invalid method scope")
        if self.skill_name not in SKILL_NAMES:
            raise ValueError(f"未知技能标签 / unknown skill: {self.skill_name!r}")
        if self.task_id is None and any(value is not None for value in
                                       (self.task_rev, self.method_seq, self.step_index)):
            raise ValueError("未绑定任务的候选不得携带任务版本或步骤游标")
        if self.task_id is not None and (not isinstance(self.task_id, str) or not self.task_id
                or type(self.task_rev) is not int or self.task_rev < 0
                or not isinstance(self.method_seq, str) or not self.method_seq
                or type(self.step_index) is not int or self.step_index < 0):
            raise ValueError("任务候选须携带完整版本和方法游标 / incomplete candidate binding")
        if not isinstance(self.params, dict):
            raise ValueError("候选参数必须是对象 / candidate params must be an object")

    def to_dict(self):
        return {"candidate_id": self.candidate_id, "skill_name": self.skill_name,
                "task_id": self.task_id, "task_rev": self.task_rev,
                "task_type": self.task_type, "method_seq": self.method_seq,
                "step_index": self.step_index, "purpose": self.purpose,
                "method_scope": self.method_scope, "action_id": self.action_id,
                "target_id": self.target_id, "params": deepcopy(self.params)}

    def get(self, key, default=None):
        return self.to_dict().get(key, default)

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __contains__(self, key):
        return key in self.to_dict()


class CandidateSet(list):
    """A candidate list that also carries the decision ticket it was built with."""

    def __init__(self, rows=(), context=None):
        super().__init__(rows)
        self.context = context


@dataclass(frozen=True)
class DecisionContext:
    """Ticket proving which task view a candidate set was built from."""

    actor_id: str
    decision_id: str
    task_ref: tuple | None = None
    method_scope: str = "main"
    cursor: int = 0
    view_version: str = ""
    candidate_set_digest: str = ""

    def to_dict(self):
        return {"actor_id": self.actor_id, "decision_id": self.decision_id,
                "task_ref": list(self.task_ref) if self.task_ref else None,
                "method_scope": self.method_scope, "cursor": self.cursor,
                "view_version": self.view_version,
                "candidate_set_digest": self.candidate_set_digest}

    def stale(self, agent, candidate=None):
        """Refuse another holder's ticket, changed views and altered candidates."""
        if self.actor_id != agent.id or self.view_version != view_version(agent):
            return True
        current = build_decision_candidates(agent, agent.catalog)
        if not self.candidate_set_digest or self != current.context:
            return True
        if candidate is None:
            return False
        data = candidate.to_dict() if isinstance(candidate, CandidateAction) else dict(candidate)
        return not any(data == row.to_dict() for row in current)


def view_version(agent):
    """Digest of every task cursor this holder could decide from."""
    rows = []
    for task_id, task in sorted(agent.tasks.items()):
        method = task["progress"].get("recovery") or task.get("method") or {}
        rows.append([task_id, task.get("rev"), task.get("status"),
                     method.get("seq_id"), int(method.get("step_index", 0))])
    return _digest(rows)


def _active_tasks(agent):
    """Tasks this holder may still advance, mirroring ``_advance_task``.

    A suspended task without a recovery run has no next step: only its A057
    release path may resume it, so it is never offered as a candidate.
    """
    rows = []
    for task_id, task in sorted(agent.tasks.items()):
        if task.get("assigned_to") != agent.id:
            continue
        progress = task.get("progress") or {}
        if progress.get("handoff_wait_release") or progress.get("pending_camera_analysis"):
            continue
        if any(agent.tasks.get(dep, {}).get("status") != "completed"
               for dep in task.get("dependency_ids", ())):
            continue
        status = task.get("status")
        if status in {"accepted", "executing", "blocked"}:
            rows.append((task_id, task))
        elif status == "suspended" and progress.get("recovery"):
            rows.append((task_id, task))
    return rows


def _candidate(candidate_id, skill, action_id, **kw):
    return CandidateAction(candidate_id=candidate_id, skill_name=skill,
                           action_id=action_id, **kw)


def build_decision_candidates(agent, catalog, sim_time_s=0.0):
    """Read-only candidate set for this holder; never mutates a task or the world.

    Per active task the set contains the real approach (when the work site is
    out of range) and the current catalog step. ``step_index`` is always the
    cursor the candidate was built from, so a stale choice is refused instead of
    silently executing the wrong step.
    """
    if not hasattr(agent, "_step_params"):
        raise TypeError("build_decision_candidates requires an AgentRuntime")
    catalog = catalog if catalog is not None else getattr(agent, "catalog", None)
    if catalog is None:
        from rescue_scenes.core.catalog import load
        catalog = load()
    rows, issued = [], 0
    for task_id, task in _active_tasks(agent):
        recovery = task.get("progress", {}).get("recovery")
        method = recovery or task.get("method") or {}
        steps = list(method.get("resolved_steps") or ())
        index = int(method.get("step_index", 0))
        if index >= len(steps):
            continue  # the method lane's own completion step owns this task
        action = steps[index]
        if action not in catalog.actions:
            continue
        seq_id = method.get("seq_id")
        skill = sequence_skill(seq_id, catalog) if seq_id else "search_and_assess"
        scope = "recovery" if recovery else "main"
        common = dict(task_id=task_id, task_rev=int(task.get("rev", 0)),
                      task_type=task.get("type"), method_seq=seq_id,
                      step_index=index, method_scope=scope,
                      target_id=str(task.get("target")))
        approach = _approach_candidate(agent, task, action, method, common, sim_time_s)
        if approach is not None:
            issued += 1
            rows.append(_candidate(f"cand-{issued:03d}-approach", skill, approach.pop("action_id"),
                                   purpose="approach", params=approach, **common))
        issued += 1
        params = agent._step_params(task, action, method)
        if action == "A054":
            params["progress"] = {"method_step": index}
        if action == "A055":
            params["safe_stop_only"] = bool(recovery)
        rows.append(_candidate(f"cand-{issued:03d}-method", skill, action,
                               purpose="method", params=params, **common))
    digest = _digest([row.to_dict() for row in rows])
    context = DecisionContext(actor_id=agent.id,
                              decision_id=f"dec-{digest[:12]}",
                              task_ref=(rows[0].task_id, rows[0].task_rev) if rows and rows[0].task_id else None,
                              method_scope=rows[0].method_scope if rows else "main",
                              cursor=int(rows[0].step_index or 0) if rows else 0,
                              view_version=view_version(agent),
                              candidate_set_digest=digest)
    return CandidateSet(rows, context)


def _approach_candidate(agent, task, action, method, common, sim_time_s):
    """A real movement to the work site, or ``None`` when already there."""
    mobility = agent.spec.get("mobility")
    movement = MOVEMENT_BY_MOBILITY.get(mobility)
    if movement is None:
        return None
    params = agent._step_params(task, action, method)
    site = _point(params.get("goal")) or _point(params.get("goal_xy"))
    if site is None:
        target = params.get("target", task.get("target"))
        site = _point(agent._target_point(target, navigation=True))
    if site is None:
        return None
    pose = _point(agent.platform.get("pose"))
    if pose is None or math.dist(pose, site) <= APPROACH_RADIUS_M:
        return None
    return {"action_id": movement, "target": params.get("target", task.get("target")),
            "goal": list(site), "goal_xy": list(site), "destination": list(site)}


def execute_candidate(sim, agent, candidate, *, ticket=None):
    """Submit one candidate through the runtime's only submission path.

    Returns the kernel action instance id. A candidate whose task revision or
    method cursor moved since it was built is refused: rebuild the candidate set
    instead of executing a step the task no longer holds.
    """
    data = candidate.to_dict() if isinstance(candidate, CandidateAction) else dict(candidate)
    # Mapping inputs receive exactly the same validation as dataclass inputs.
    data = CandidateAction(**data).to_dict()
    action, purpose = data["action_id"], data["purpose"]
    if any(str(key).startswith('_') for key in data['params']):
        raise ValueError("候选参数不得设置内核内部字段 / candidate params contain kernel-internal members")
    if action not in agent.catalog.actions:
        raise ValueError("动作不在当前目录 / action is absent from the catalog")
    task_id = data["task_id"]
    task = agent.tasks.get(task_id) if task_id is not None else None
    if task_id is not None:
        if task is None or not any(tid == task_id for tid, _ in _active_tasks(agent)):
            raise ValueError("候选任务未由本机活动持有 / candidate task is not active and held")
        if data["task_rev"] != task.get("rev", 0):
            raise ValueError("候选任务版本已过期 / candidate task revision is stale")
        recovery = task.get("progress", {}).get("recovery")
        scope = "recovery" if recovery else "main"
        method = recovery or task.get("method") or {}
        cursor = int(method.get("step_index", 0))
        if (data["method_scope"] != scope or data["method_seq"] != method.get("seq_id")
                or data["step_index"] != cursor):
            raise ValueError("候选方法或步骤已过期 / candidate method cursor is stale")
        steps = method.get("resolved_steps") or []
        if purpose == "method" and (cursor >= len(steps) or action != steps[cursor]):
            raise ValueError("候选动作不是当前方法步骤 / candidate action is not the current step")
        if purpose == "approach" and action != MOVEMENT_BY_MOBILITY.get(agent.spec.get("mobility")):
            raise ValueError("接近候选必须是移动动作 / approach must be a movement action")
        if any(context.get("task") and context["task"][0] == task_id
               and context.get("purpose") in {"method", "approach"}
               for context in agent.pending_actions.values()):
            raise ValueError("任务已有待结算动作 / task already has a pending candidate action")
    if ticket is not None and ticket.stale(agent, data):
        raise ValueError("决策票据已过期或候选被修改 / decision ticket is stale or candidate was altered")
    return agent._submit(sim, action, deepcopy(data["params"]), task=task,
                         purpose=purpose, method_seq=data["method_seq"],
                         step=data["step_index"], method_scope=data["method_scope"],
                         candidate_id=data["candidate_id"])


# --------------------------------------------------------------------------- #
# Channel gating and the delivered-only memory envelope.
# --------------------------------------------------------------------------- #

# Bits follow the research convention: uplink (edge -> center), then downlink.
CHANNEL_MODES = {"B00": (False, False), "B10": (True, False),
                 "B01": (False, True), "B11": (True, True)}
DIRECTION_BIT = {"up": 0, "down": 1}


class CommChannelGate:
    """Per-holder byte budget for the four B-modes.

    ``B00`` is silent, ``B10`` uplink only, ``B01`` downlink only and ``B11``
    both. A refused delivery is *not* a network fault: the holder simply never
    spends the bytes and never reads the content.
    """

    def __init__(self, mode="B11", budget_bytes=0, *, b_down_bytes=None, b_up_bytes=None):
        if mode not in CHANNEL_MODES:
            raise ValueError(f"未知通道模式 / unknown channel mode: {mode!r}")
        budget_bytes = int(budget_bytes)
        if budget_bytes < 0:
            raise ValueError("字节预算不得为负 / byte budget must be nonnegative")
        self.mode = mode
        self.budget_bytes = budget_bytes
        self.budgets = {"down": budget_bytes if b_down_bytes is None else int(b_down_bytes),
                        "up": budget_bytes if b_up_bytes is None else int(b_up_bytes)}
        if min(self.budgets.values()) < 0:
            raise ValueError("字节预算不得为负 / byte budget must be nonnegative")
        self.spent = {"down": 0, "up": 0}
        self.denied = {"down": 0, "up": 0}

    def allows(self, direction):
        if direction not in DIRECTION_BIT:
            raise ValueError(f"未知方向 / unknown direction: {direction!r}")
        return CHANNEL_MODES[self.mode][DIRECTION_BIT[direction]]

    def budget(self, direction):
        return self.budgets[direction]

    def remaining(self, direction=None):
        if direction is not None:
            return max(0, self.budget(direction) - self.spent[direction])
        return {key: self.remaining(key) for key in self.spent}

    def admit(self, direction, size_bytes, now_s=None):
        """Charge ``size_bytes`` or refuse; never partially delivers."""
        if not self.allows(direction):
            self.denied[direction] += 1
            return False
        size_bytes = int(size_bytes)
        if size_bytes < 0:
            raise ValueError("字节数不得为负 / size must be nonnegative")
        if self.spent[direction] + size_bytes > self.budget(direction):
            self.denied[direction] += 1
            return False
        self.spent[direction] += size_bytes
        return True

    def to_dict(self):
        return {"mode": self.mode, "budgets": dict(self.budgets), "spent": dict(self.spent),
                "denied": dict(self.denied), "remaining": self.remaining()}


@dataclass
class MemoryEnvelope:
    """One bounded memory transfer; readable only after real delivery.

    The envelope is built from an actual :class:`~rescue_scenes.core.contracts.Message`:
    ``delivered_at_s`` stays ``None`` until ``Network._deliver`` set it, and
    ``valid_until_s`` is the message's own TTL horizon. An undelivered or expired
    envelope never enters an edge-readable context.
    """

    envelope_id: str
    sender: str
    receivers: tuple
    kind: str
    payload: dict
    size_bytes: int
    created_s: float
    valid_until_s: float
    delivered_at_s: float | None = None
    channel: str = "B11"
    digest: str = ""

    def __post_init__(self):
        self.receivers = tuple(self.receivers)
        if self.channel not in CHANNEL_MODES:
            raise ValueError(f"未知通道模式 / unknown channel mode: {self.channel!r}")
        if int(self.size_bytes) < 0:
            raise ValueError("字节数不得为负 / size must be nonnegative")
        if not self.digest:
            self.digest = self.compute_digest()

    def compute_digest(self):
        return _digest([self.envelope_id, self.sender, list(self.receivers), self.kind,
                        self.payload, int(self.size_bytes), float(self.created_s),
                        float(self.valid_until_s), self.channel])

    def verified(self):
        return self.digest == self.compute_digest()

    def delivered(self, now_s):
        if self.delivered_at_s is None:
            return False
        times = tuple(map(float, (self.created_s, self.delivered_at_s, now_s, self.valid_until_s)))
        return all(map(math.isfinite, times)) and times[0] <= times[1] <= times[2] < times[3]

    def readable(self, now_s, receiver=None):
        """Delivered, unexpired, intact and addressed to this receiver."""
        if receiver is not None and receiver not in self.receivers:
            return False
        return self.delivered(now_s) and self.verified()

    def to_dict(self):
        return {"envelope_id": self.envelope_id, "sender": self.sender,
                "receivers": list(self.receivers), "kind": self.kind,
                "payload": deepcopy(self.payload), "size_bytes": int(self.size_bytes),
                "created_s": float(self.created_s), "valid_until_s": float(self.valid_until_s),
                "delivered_at_s": self.delivered_at_s, "channel": self.channel,
                "digest": self.digest}

    @classmethod
    def from_dict(cls, data):
        """Rebuild a delivered envelope from its serialized form."""
        if isinstance(data, MemoryEnvelope):
            return data
        if not isinstance(data, dict):
            raise TypeError("memory envelope must be a mapping")
        if not isinstance(data.get("digest"), str) or not data["digest"]:
            raise ValueError("serialized memory envelope requires its original digest")
        return cls(envelope_id=str(data["envelope_id"]), sender=str(data.get("sender", "")),
                   receivers=tuple(data.get("receivers") or ()),
                   kind=str(data.get("kind", "")), payload=deepcopy(data.get("payload") or {}),
                   size_bytes=int(data.get("size_bytes") or 0),
                   created_s=float(data.get("created_s") or 0.0),
                   valid_until_s=float(data.get("valid_until_s") or 0.0),
                   delivered_at_s=(None if data.get("delivered_at_s") is None
                                   else float(data["delivered_at_s"])),
                   channel=str(data.get("channel") or "B11"),
                   digest=str(data.get("digest") or ""))

    @classmethod
    def from_message(cls, message, receiver=None, channel="B11", size_limit=None):
        """Bind an envelope to a real message; ``delivered_s`` is never invented."""
        msg = message.to_dict() if hasattr(message, "to_dict") else dict(message)
        size = int(msg.get("size_bytes") or 0)
        if size_limit is not None and size > int(size_limit):
            return None
        sent_s = float(msg.get("sent_s") or 0.0)
        ttl = msg.get("ttl_s")
        delivered = msg.get("delivered_s")
        receivers = tuple(msg.get("receivers") or ())
        statuses = msg.get("status") or {}
        if receiver is not None:
            receivers = (receiver,) if receiver in receivers else ()
        # A multicast message's global delivered_s does not prove every recipient
        # has its copy. Only receiver-specific delivery status grants readability.
        receivers = tuple(rid for rid in receivers if statuses.get(rid) == "delivered")
        if not receivers:
            delivered = None
        return cls(envelope_id=str(msg["id"]), sender=str(msg["sender"]),
                   receivers=receivers,
                   kind=str(msg.get("kind") or ""), payload=deepcopy(msg.get("payload") or {}),
                   size_bytes=size, created_s=sent_s,
                   valid_until_s=sent_s + (float(ttl) if ttl is not None else 0.0),
                   delivered_at_s=None if delivered is None else float(delivered),
                   channel=channel)


def readable_envelopes(envelopes, now_s, receiver=None):
    """Only the envelopes an edge may actually read at ``now_s``."""
    return [envelope for envelope in envelopes if envelope.readable(now_s, receiver)]
