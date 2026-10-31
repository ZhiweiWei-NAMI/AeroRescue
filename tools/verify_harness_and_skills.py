#!/usr/bin/env python
"""T1-T7 acceptance for the kernel/skills/center-edge harness contract.

Every trajectory runs the real kernel and the real ``AgentRuntime``; nothing is
scored from a stub. The script writes two deliverables:

* ``runs/verify/harness_skills_matrix.json`` - machine-readable assertion matrix;
* ``docs/harness-and-skills-acceptance.md`` - the same matrix as a report.

Run with the workspace virtualenv::

    .venv/bin/python tools/verify_harness_and_skills.py [--quiet]
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rescue_scenes.core.api import Environment                     # noqa: E402
from rescue_scenes.core.contracts import Observation               # noqa: E402
from rescue_scenes.px4_bridge import (                             # noqa: E402
    ARRIVAL_TOLERANCE_M, complete_reached_movements,
)
from aerorescue import edge_harness                                # noqa: E402
from aerorescue.skills import (                                    # noqa: E402
    CHANNEL_MODES, CandidateAction, CommChannelGate, DecisionContext, MemoryEnvelope,
    SKILL_LABELS, SKILL_NAMES, VALID_ACTION_IDS, build_decision_candidates,
    execute_candidate, readable_envelopes, sequence_skill, skill_sequences,
)

JSON_OUT = ROOT / "runs" / "verify" / "harness_skills_matrix.json"
DOC_OUT = ROOT / "docs" / "harness-and-skills-acceptance.md"
CASUALTY = "human-building-1-2"     # demo casualty that is immobile (loadable)
CASUALTY_MASS_KG = 82.5


class Trace:
    """Collects assertions for one trajectory and fails loudly on the first miss."""

    def __init__(self, tid, title, contract):
        self.tid, self.title, self.contract = tid, title, contract
        self.rows = []

    def expect(self, label, ok, **detail):
        ok = bool(ok)
        self.rows.append({"assertion": label, "ok": ok, "detail": detail})
        if not ok:
            raise AssertionError(f"{self.tid}: {label} :: {detail}")
        return ok

    def to_dict(self):
        return {"id": self.tid, "title": self.title, "contract": self.contract,
                "ok": all(row["ok"] for row in self.rows), "assertions": self.rows}


# --------------------------------------------------------------------------- #
# Harness helpers (authored scenario setup only; no result is ever asserted
# from a fabricated value).
# --------------------------------------------------------------------------- #

def demo_env(tick_s=2, disruptions=None):
    config = {"scenario_id": "SC01", "seed": 1, "semantics": {"source": "demo"},
              "tick_s": tick_s, "until_s": 60}
    if disruptions:
        config["disruptions"] = disruptions
    return Environment.from_config(config)


def isolate(sim, keep):
    """Only the holder under test ticks; this isolates a trajectory, not physics."""
    for agent in sim.agents.values():
        if agent is not keep:
            agent.on_tick = lambda _sim: None


def author_truth(sim, eid, key, value, rule="VERIFY-AUTHORED"):
    sim.world.set(eid, key, value, {"rule": rule, "t": sim.t})


def accept_task(agent, sim, task_type, target, priority=60, parameters=None, accept=True):
    """Create a holder task through the runtime's own task builder.

    ``rev``/``status`` follow the same transitions A052 -> A065 -> A053 use, so a
    verifier task is byte-identical to a delivered assignment.
    """
    task = agent._task_from_params(sim, {
        "type": task_type, "target": target, "priority": priority,
        "origin": {"rule_id": "VERIFY", "why_zh": f"验收轨迹 {task_type}@{target}",
                   "belief_refs": []}})
    if task is None:
        raise AssertionError(f"task builder refused {task_type}@{target}")
    for key, value in (parameters or {}).items():
        task["parameters"][key] = value
    task["assigned_to"] = agent.id
    task["rev"] = int(task.get("rev", 0)) + 1
    agent._status(sim, task, "assigned", "验收：等待收件", "A052")
    agent._status(sim, task, "received", "验收：本机收到", "A065")
    task["accepted_by"] = agent.id
    task["commitment_window"] = [sim.t, task.get("deadline_s")]
    agent._select_method(sim, task)
    if not accept:
        return task
    agent._status(sim, task, "accepted", "验收：登记承诺", "A053")
    if task["id"] not in agent.queue:
        agent.queue.append(task["id"])
    return task


def idle(sim, agent):
    return (not agent.pending_actions and not agent._controls
            and not any(rec["status"] in ("running", "accepted") for rec in sim.instances.values()))


def wait_idle(env, agent, max_s=600):
    sim, end = env.sim, env.sim.t + max_s
    while sim.t < end:
        if idle(sim, agent):
            return True
        sim.step()
    return idle(sim, agent)


def run_until(env, predicate, max_s=600):
    sim, end = env.sim, env.sim.t + max_s
    while sim.t < end:
        if predicate():
            return True
        sim.step()
    return predicate()


def deliver_report(sim, agent, observation, task=None):
    """Deliver authored situation evidence through the real A041 intake path."""
    return sim.submit_action(agent.id, "A041", {"observations": [observation.to_dict()]},
                             task_ref=(task["id"], task["rev"]) if task else None)


def situation_report(eid, values, source="rescue.center", suffix="1", now=0.0):
    return Observation(id=f"verify-report-{eid}-{suffix}", by=source, t=float(now), about=[eid],
                       kind="delivered_situation_report", values=dict(values),
                       confidence=.9, model="center_situation_report_v1")


def candidate_for(candidates, task_id, purpose=None):
    rows = [row for row in candidates if row.task_id == task_id]
    if purpose is not None:
        rows = [row for row in rows if row.purpose == purpose]
    return rows[0] if rows else None


def submit_and_settle(env, agent, candidate, ticket):
    iid = execute_candidate(env.sim, agent, candidate, ticket=ticket)
    wait_idle(env, agent)
    return iid, env.sim.instances[iid]


# --------------------------------------------------------------------------- #
# T1 - reconnaissance and building assessment (T011 / Q011)
# --------------------------------------------------------------------------- #

def t1():
    trace = Trace("T1", "T011/Q011 侦察与建筑评估闭环（approach 不推进游标，method 逐步推进）",
                  "R3.1 candidate binding; R3.2 execute_candidate -> pending_actions/on_action_result")
    env = demo_env()
    sim, agent = env.sim, env.sim.agents["uav-1"]
    isolate(sim, agent)
    agent.spec["task_allowlist"] = ["T011"]
    agent.policy_mode = "external_harness"
    task = accept_task(agent, sim, "T011", "building-1")
    method = task["method"]
    trace.expect("catalog resolves T011 to the real Q011 method",
                 method["seq_id"] == "Q011" and method["resolved_steps"][:2] == ["A001", "A005"],
                 seq=method["seq_id"], steps=method["resolved_steps"])

    candidates = build_decision_candidates(agent, sim.catalog, sim.t)
    trace.expect("candidate set carries a DecisionContext ticket",
                 isinstance(candidates.context, DecisionContext)
                 and len(candidates.context.candidate_set_digest) == 64,
                 decision=candidates.context.decision_id)
    approach = candidate_for(candidates, task["id"], "approach")
    method_candidate = candidate_for(candidates, task["id"], "method")
    trace.expect("out-of-range work site yields a real approach candidate",
                 approach is not None and approach.action_id == "A001"
                 and approach.step_index == 0 and approach.method_scope == "main"
                 and approach.task_rev == task["rev"],
                 action=getattr(approach, "action_id", None))
    trace.expect("method candidate binds the same revision and skill",
                 method_candidate is not None and method_candidate.action_id == "A001"
                 and method_candidate.skill_name == sequence_skill("Q011", sim.catalog),
                 skill=getattr(method_candidate, "skill_name", None))

    start_pose = list(sim.world.get("uav-1", "entity.pose"))
    iid, rec = submit_and_settle(env, agent, approach, candidates.context)
    trace.expect("approach enters AgentRuntime.pending_actions then settles",
                 rec["status"] == "succeeded" and rec["failure_code"] == "none",
                 status=rec["status"], failure=rec["failure_code"])
    trace.expect("approach never advances the method cursor",
                 task["method"]["step_index"] == 0, cursor=task["method"]["step_index"])
    trace.expect("approach really moved the holder to the work site",
                 math.dist(sim.world.get("uav-1", "entity.pose"), start_pose) > 10.0)
    stale_ticket = candidates.context
    try:
        execute_candidate(sim, agent, method_candidate, ticket=stale_ticket)
        refused = False
    except ValueError:
        refused = True
    trace.expect("a ticket from the pre-approach view is refused, not executed", refused)

    submitted, integrate = [], []
    for _ in range(12):
        candidates = build_decision_candidates(agent, sim.catalog, sim.t)
        rows = [row for row in candidates if row.task_id == task["id"] and row.purpose == "method"]
        if not rows:
            break
        cursor = task["method"]["step_index"]
        candidate = rows[0]
        before = set(sim.instances)
        iid, rec = submit_and_settle(env, agent, candidate, candidates.context)
        for new in sorted(set(sim.instances) - before):
            row = sim.instances[new]
            if row["action_id"] == "A041" and row["params"].get("seq_id") is None:
                integrate.append(new)
        trace.expect(f"method step {cursor} ({candidate.action_id}) advanced the cursor once",
                     task["method"]["step_index"] == cursor + 1
                     and rec["status"] == "succeeded",
                     status=rec["status"], failure=rec["failure_code"],
                     cursor=task["method"]["step_index"])
        submitted.append((cursor, candidate.action_id))
    trace.expect("the whole Q011 main path was executed",
                 [action for _, action in submitted] == ["A001", "A005", "A006", "A040", "A041", "A036"],
                 submitted=submitted)
    trace.expect("a control-lane A041 fusion ran while the method advanced",
                 len(integrate) >= 1, instances=integrate)
    trace.expect("A041 control fusion never adds a main method step",
                 set(task["progress"]["steps"]) == {f"Q011:{index}" for index in range(6)},
                 steps=sorted(task["progress"]["steps"]))
    trace.expect("A041 wrote the holder's belief (assessment evidence)",
                 any(event.family == "belief.update" and event.actor == "uav-1" for event in sim.events),
                 belief_events=sum(1 for event in sim.events
                                   if event.family == "belief.update" and event.actor == "uav-1"))

    # Without switching policy_mode back to internal_rule, external_harness
    # automatically finalizes the exhausted method via its own A054 lane.
    run_until(env, lambda: task["status"] in ("completed", "failed", "blocked"), max_s=120)
    trace.expect("the assessed task closes as completed under external_harness",
                 task["status"] == "completed" and agent.policy_mode == "external_harness",
                 status=task["status"], reason=task.get("failure_reason"))
    return trace


# --------------------------------------------------------------------------- #
# T2 - supply pickup, casualty load and handover (A028/A024/A020/A021/A022)
# --------------------------------------------------------------------------- #

def t2():
    trace = Trace("T2", "伤员装载/后送移交与物资获取（A020/A021/A022 + A028/A024）",
                  "R3.1 payload accounting; A022 acknowledged handover")
    env = demo_env()
    sim, agent = env.sim, env.sim.agents["ground_vehicle-1"]
    isolate(sim, agent)
    agent.spec["task_allowlist"] = ["T038", "T030", "T033", "T035"]
    author_truth(sim, CASUALTY, "attr.mass_kg", CASUALTY_MASS_KG)
    author_truth(sim, "station-3", "attr.receiver_confirmation", True)
    author_truth(sim, "station-3", "station.accepted_need_types",
                 ["water", "medical_supplies", "transport"])
    deliver_report(sim, agent, situation_report(CASUALTY, {
        "entity.kind": "human", "entity.pose": [36.0, 52.0], "human.mobility": "immobile",
        "human.needs": ["water", "transport"], "human.location_context": "interior"},
        now=sim.t))
    wait_idle(env, agent)
    trace.expect("delivered report reached the holder belief through A041",
                 agent.belief.get(CASUALTY, "entity.pose") is not None)

    # --- A020: load the real casualty; payload must include its authored mass.
    agent.policy_mode = "internal_rule"
    load = accept_task(agent, sim, "T033", CASUALTY, priority=70)
    run_until(env, lambda: load["status"] in ("completed", "failed", "blocked"), max_s=240)
    loaded = sim.world.get("ground_vehicle-1", "platform.payload_used_kg")
    trace.expect("A020 loads the casualty and payload equals its real attr.mass_kg",
                 load["status"] == "completed"
                 and sim.world.get(CASUALTY, "human.carrier_id") == "ground_vehicle-1"
                 and abs(loaded - CASUALTY_MASS_KG) < 1e-9,
                 payload=loaded, expected=CASUALTY_MASS_KG, status=load["status"])

    # --- A021 + A022: transport to a confirming receiver and acknowledge.
    handover = accept_task(agent, sim, "T035", CASUALTY, priority=80,
                           parameters={"destination_id": "station-3"})
    run_until(env, lambda: handover["status"] in ("completed", "failed", "blocked"), max_s=240)
    after = sim.world.get("ground_vehicle-1", "platform.payload_used_kg")
    trace.expect("A022 acknowledges the handover and payload returns to zero",
                 handover["status"] == "completed"
                 and sim.world.get(CASUALTY, "human.handover_status") == "acknowledged"
                 and sim.world.get(CASUALTY, "human.carrier_id") is None
                 and after == 0.0,
                 status=handover["status"], payload=after,
                 handover=sim.world.get(CASUALTY, "human.handover_status"))
    trace.expect("the receiver's occupancy really changed",
                 int(sim.world.get("station-3", "station.occupied")) >= 1,
                 occupied=sim.world.get("station-3", "station.occupied"))

    # --- A028: real station stock moves into the platform payload.
    agent.policy_mode = "external_harness"
    supply = accept_task(agent, sim, "T038", "station-1",
                         parameters={"station_id": "station-1", "item_kind": "water"})
    agent.current = supply["id"]
    candidates = build_decision_candidates(agent, sim.catalog, sim.t)
    approach = candidate_for(candidates, supply["id"], "approach")
    if approach is not None:
        submit_and_settle(env, agent, approach, candidates.context)
        candidates = build_decision_candidates(agent, sim.catalog, sim.t)
    iid, rec = submit_and_settle(env, agent, candidate_for(candidates, supply["id"], "method"),
                                 candidates.context)
    items = list(sim.world.get("ground_vehicle-1", "platform.payload_item_ids"))
    water_mass = sum(float(sim.world.get(item, "item.mass_kg")) for item in items)
    trace.expect("A028 transfers one real stock instance and charges its mass",
                 rec["status"] == "succeeded" and len(items) == 1
                 and sim.world.get("ground_vehicle-1", "platform.payload_used_kg") == water_mass
                 and sim.world.get(items[0], "item.holder_id") == "ground_vehicle-1",
                 items=items, payload=sim.world.get("ground_vehicle-1", "platform.payload_used_kg"))

    # --- A024: mount one existing item instance through the candidate primitive.
    mount = accept_task(agent, sim, "T030", CASUALTY,
                        parameters={"item_id": "item-station-1-medical_pack-1"})
    pack = "item-station-1-medical_pack-1"
    pack_mass = float(sim.world.get(pack, "item.mass_kg"))
    candidate = CandidateAction(
        candidate_id="cand-a024", skill_name="pickup_supply", action_id="A024",
        purpose="method", method_scope="main", task_id=mount["id"], task_rev=mount["rev"],
        task_type="T030", method_seq=mount["method"]["seq_id"], step_index=0,
        target_id=pack, params={"target": pack, "item_id": pack})
    iid, rec = submit_and_settle(env, agent, candidate, None)
    payload = sim.world.get("ground_vehicle-1", "platform.payload_used_kg")
    trace.expect("A024 mounts the existing item and adds exactly its mass",
                 rec["status"] == "succeeded" and payload == water_mass + pack_mass
                 and sim.world.get(pack, "item.state") == "attached",
                 payload=payload, expected=water_mass + pack_mass)
    return trace


# --------------------------------------------------------------------------- #
# T3 - blocked road triggers the Q098 recovery lane
# --------------------------------------------------------------------------- #

def t3():
    trace = Trace("T3", "道路受阻 failure_code=blocked 触发 Q098 恢复闭环",
                  "R2.3 recovery cursor; runtime._recover skips the repeated trigger step")
    env = demo_env()
    sim, agent = env.sim, env.sim.agents["ground_vehicle-1"]
    isolate(sim, agent)
    agent.spec["task_allowlist"] = ["T021"]
    agent.policy_mode = "external_harness"
    author_truth(sim, "road-1", "road.entry_state", "closed", rule="VERIFY-AFTERSHOCK")
    task = accept_task(agent, sim, "T021", "road-2")
    trace.expect("catalog resolves T021 to Q021 starting with the movement step",
                 task["method"]["seq_id"] == "Q021"
                 and task["method"]["resolved_steps"][0] == "A002",
                 steps=task["method"]["resolved_steps"])

    candidates = build_decision_candidates(agent, sim.catalog, sim.t)
    iid, rec = submit_and_settle(env, agent, candidate_for(candidates, task["id"], "method"),
                                 candidates.context)
    recovery = task["progress"].get("recovery")
    trace.expect("the real movement model reports failure_code=blocked",
                 rec["status"] == "failed" and rec["failure_code"] == "blocked",
                 status=rec["status"], failure=rec["failure_code"])
    trace.expect("recovery is Q098 and starts after the repeated trigger step",
                 recovery is not None and recovery["seq_id"] == "Q098"
                 and recovery["step_index"] == 1
                 and recovery["resolved_steps"] == ["A002", "A041", "A066", "A002", "A054"],
                 seq=(recovery or {}).get("seq_id"), cursor=(recovery or {}).get("step_index"))
    trace.expect("the blocked task is honestly blocked, not failed",
                 task["status"] == "blocked", status=task["status"])

    executed = []
    candidates = build_decision_candidates(agent, sim.catalog, sim.t)
    rows = [row for row in candidates if row.task_id == task["id"] and row.purpose == "method"]
    cursor = (task["progress"].get("recovery") or task["method"])["step_index"]
    iid, rec = submit_and_settle(env, agent, rows[0], candidates.context)
    executed.append((cursor, rows[0].action_id, rec["status"]))
    trace.expect("the recovery lane starts by integrating the discovered blockage",
                 executed == [(1, "A041", "succeeded")], executed=executed)

    # A clearing crew physically opens the road and reports it; the delivered
    # evidence enters through the same A041 path, so the A066 replan can find a
    # passable route and the lane finishes A041 -> A066 -> A002 -> A054.
    author_truth(sim, "road-1", "road.entry_state", "open", rule="VERIFY-CLEARING")
    deliver_report(sim, agent, situation_report("road-1", {
        "road.entry_state": "open", "road.clear_width_m": 7.0},
        source="rescue.clearing_crew", suffix="clearing", now=sim.t))
    wait_idle(env, agent)
    for _ in range(4):
        candidates = build_decision_candidates(agent, sim.catalog, sim.t)
        rows = [row for row in candidates if row.task_id == task["id"] and row.purpose == "method"]
        if not rows:
            break
        cursor = (task["progress"].get("recovery") or task["method"])["step_index"]
        candidate = rows[0]
        iid, rec = submit_and_settle(env, agent, candidate, candidates.context)
        executed.append((cursor, candidate.action_id, rec["status"]))
        if candidate.action_id == "A054":
            break
    trace.expect("the cleared recovery lane runs A041 -> A066 -> A002 -> A054",
                 [action for _, action, _ in executed] == ["A041", "A066", "A002", "A054"]
                 and [cursor for cursor, _, _ in executed] == [1, 2, 3, 4]
                 and all(status == "succeeded" for _, _, status in executed),
                 executed=executed)
    return trace


# --------------------------------------------------------------------------- #
# T4 - high-priority preemption and the resume stack
# --------------------------------------------------------------------------- #

def t4():
    trace = Trace("T4", "高优任务抢占与 _resume_stack 挂起恢复闭环",
                  "A053 preemption -> A056 checkpoint -> may_resume_preempted -> A057")
    env = demo_env()
    sim, agent = env.sim, env.sim.agents["ground_vehicle-1"]
    isolate(sim, agent)
    agent.spec["task_allowlist"] = ["T021", "T033", "T035"]
    agent.policy_mode = "external_harness"
    author_truth(sim, CASUALTY, "attr.mass_kg", CASUALTY_MASS_KG)
    author_truth(sim, "station-3", "attr.receiver_confirmation", True)
    author_truth(sim, "station-3", "station.accepted_need_types",
                 ["water", "medical_supplies", "transport"])
    deliver_report(sim, agent, situation_report(CASUALTY, {
        "entity.kind": "human", "entity.pose": [36.0, 52.0], "human.mobility": "immobile",
        "human.needs": ["water", "transport"], "human.location_context": "interior"},
        now=sim.t))
    wait_idle(env, agent)

    original = accept_task(agent, sim, "T021", "road-2", priority=50)
    agent.current = original["id"]
    candidates = build_decision_candidates(agent, sim.catalog, sim.t)
    iid, rec = submit_and_settle(env, agent, candidate_for(candidates, original["id"], "method"),
                                 candidates.context)
    trace.expect("the original commitment is executing with a real action",
                 original["status"] == "executing" and agent.current == original["id"]
                 and rec["status"] == "succeeded",
                 status=original["status"])

    replacement = accept_task(agent, sim, "T033", CASUALTY, priority=90, accept=False)
    agent._controls.appendleft(("A053", {"task_id": replacement["id"]},
                               replacement["id"], "control", {}))
    run_until(env, lambda: original["status"] == "suspended", max_s=120)
    trace.expect("A053 preempts at the safe boundary and A056 suspends the original",
                 original["status"] == "suspended"
                 and agent._resume_stack == [original["id"]],
                 status=original["status"], stack=list(agent._resume_stack))
    trace.expect("A056 wrote the suspended checkpoint",
                 isinstance(original["progress"].get("resume_checkpoint"), dict),
                 checkpoint=bool(original["progress"].get("resume_checkpoint")))
    trace.expect("the replacement was really accepted after the suspension",
                 replacement["status"] in ("accepted", "executing"),
                 status=replacement["status"])

    for _ in range(6):
        candidates = build_decision_candidates(agent, sim.catalog, sim.t)
        rows = [row for row in candidates if row.task_id == replacement["id"]]
        if not rows:
            break
        iid, rec = submit_and_settle(env, agent, rows[0], candidates.context)
        if replacement["method"]["step_index"] >= len(replacement["method"]["resolved_steps"]):
            break
    # The harness ran the method; the runtime's own A054 lane closes it.
    agent.policy_mode = "internal_rule"
    run_until(env, lambda: replacement["status"] in ("completed", "failed", "blocked"), max_s=180)
    trace.expect("the preempting rescue completes", replacement["status"] == "completed",
                 status=replacement["status"])
    trace.expect("a carried casualty keeps the original suspended until handover",
                 original["status"] == "suspended",
                 status=original["status"], stack=list(agent._resume_stack))

    handover = accept_task(agent, sim, "T035", CASUALTY, priority=95,
                           parameters={"destination_id": "station-3"})
    run_until(env, lambda: handover["status"] in ("completed", "failed", "blocked"), max_s=240)
    trace.expect("the handover completes and the receiver acknowledges",
                 handover["status"] == "completed"
                 and sim.world.get(CASUALTY, "human.handover_status") == "acknowledged",
                 status=handover["status"])
    run_until(env, lambda: original["status"] == "executing" and not agent._resume_stack, max_s=120)
    trace.expect("may_resume_preempted releases the original through A057",
                 original["status"] == "executing" and agent._resume_stack == []
                 and agent.current == original["id"],
                 status=original["status"], stack=list(agent._resume_stack),
                 current=agent.current)
    trace.expect("the resume really used the A057 commitment path",
                 any(row["action_id"] == "A057" and row["status"] == "succeeded"
                     for row in sim.instances.values()))
    return trace


# --------------------------------------------------------------------------- #
# T5 - PX4 external motion embodiment
# --------------------------------------------------------------------------- #

def t5():
    trace = Trace("T5", "PX4 external_motion 具身飞控闭环",
                  "R2.3 external motion; complete_external_action; 2 m arrival tolerance")
    env = demo_env()
    sim, agent = env.sim, env.sim.agents["uav-1"]
    isolate(sim, agent)
    agent.spec["task_allowlist"] = ["T011"]
    agent.policy_mode = "external_harness"
    agent.motion_backend = "px4"
    task = accept_task(agent, sim, "T011", "building-1")
    agent.current = task["id"]
    candidates = build_decision_candidates(agent, sim.catalog, sim.t)
    candidate = candidate_for(candidates, task["id"], "method")
    iid = execute_candidate(sim, agent, candidate, ticket=candidates.context)
    rec = sim.instances[iid]
    trace.expect("A001 under motion_backend=px4 is an external motion",
                 rec["external_motion"] is True and rec["params"].get("_external_motion") is True
                 and rec["params"]["_motion"]["external"] is True,
                 action=rec["action_id"])
    goal = rec["params"]["_motion"]["goal"]
    trace.expect("the external leg keeps the real goal for the flight controller",
                 isinstance(goal, (list, tuple)) and len(goal) >= 2, goal=goal)
    pose_before = list(sim.world.get("uav-1", "entity.pose"))
    duration = rec["plan"]["duration_s"]
    watchdog = [when - sim.t for when, _, fn, _ in sim._heap
                if getattr(fn, "__name__", "") == "_external_watchdog"]
    trace.expect("no fixed completion timer: only a bounded watchdog is armed",
                 len(watchdog) == 1 and watchdog[0] >= max(duration * 3.0, 180.0) - 1e-6,
                 watchdog=watchdog, duration=duration)

    # The control lane must stay alive while the flight controller owns the pose.
    agent._controls.append(("A041", {"observations": [situation_report(
        "building-1", {"building.visual_damage": "major"}).to_dict()], "evidence_ids": []},
        task["id"], "integrate", {}))
    sim.run(sim.t + duration + 4)
    trace.expect("the scripted interpolation never moved the PX4-managed UAV",
                 sim.world.get("uav-1", "entity.pose") == pose_before,
                 pose=sim.world.get("uav-1", "entity.pose"))
    trace.expect("the planned duration alone does not complete the action",
                 rec["status"] == "running", status=rec["status"])
    trace.expect("on_tick still drains the control queue during external motion",
                 any(row["action_id"] == "A041" and row["params"].get("seq_id") is None
                     and row["status"] == "succeeded" for row in sim.instances.values()))

    # Telemetry 3 m short of the goal must not settle the leg.
    sim.world.set("uav-1", "entity.pose", [goal[0] - 3.0, goal[1]],
                  {"rule": "PX4-EMBODIMENT", "t": sim.t})
    trace.expect("arrival outside the 2 m tolerance is not a completion",
                 complete_reached_movements(sim, {"uav-1"}) == [] and rec["status"] == "running",
                 tolerance=ARRIVAL_TOLERANCE_M)
    sim.world.set("uav-1", "entity.pose", [goal[0] - 1.0, goal[1]],
                  {"rule": "PX4-EMBODIMENT", "t": sim.t})
    completed = complete_reached_movements(sim, {"uav-1"})
    trace.expect("telemetry within 2 m settles the action through complete_external_action",
                 completed == [iid] and rec["status"] == "succeeded",
                 completed=completed, status=rec["status"])
    trace.expect("the native pose was not overwritten by a scripted completion",
                 sim.world.get("uav-1", "entity.pose") == [goal[0] - 1.0, goal[1]],
                 pose=sim.world.get("uav-1", "entity.pose"))
    trace.expect("the task cursor advanced to the next catalog step",
                 task["method"]["step_index"] == 1, cursor=task["method"]["step_index"])
    sim.step()
    following = [row.action_id for row in build_decision_candidates(agent, sim.catalog, sim.t)]
    trace.expect("the next decision offers A005, not a repeated A001",
                 following and following[0] == "A005", candidates=following)
    trace.expect("the platform reports idle after the settled external arrival",
                 sim.world.get("uav-1", "platform.motion_state") == "idle",
                 motion_state=sim.world.get("uav-1", "platform.motion_state"))
    return trace


# --------------------------------------------------------------------------- #
# T6 - scheduled world disruption at t=10 and deepcopy isolation
# --------------------------------------------------------------------------- #

def t6():
    trace = Trace("T6", "t=10 定时世界扰动触发与 deepcopy(sim) 隔离",
                  "R2.3 _enqueue_world binds to its own holder")
    env = demo_env(tick_s=5, disruptions=[
        {"t": 10, "kind": "link_outage", "target": None, "params": {"duration_s": 20}}])
    sim = env.sim
    clone = deepcopy(sim)
    trace.expect("every scheduled callback rebinds to the copied simulation",
                 clone._heap and all(getattr(fn, "__self__", None) is clone
                                     for _, _, fn, _ in clone._heap),
                 heap=len(clone._heap))
    sim.run(10)
    fired = [event for event in sim.events
             if event.family == "disruption.injected" and event.kind == "link_outage"]
    trace.expect("the t=10 disruption really fires on the original simulation",
                 len(fired) == 1 and fired[0].t == 10.0,
                 events=[(event.kind, event.t) for event in fired])
    trace.expect("the due list is consumed, not orphaned",
                 sim._due_world == [], due=sim._due_world)

    clone.run(10)
    clone_fired = [event for event in clone.events
                   if event.family == "disruption.injected" and event.kind == "link_outage"]
    trace.expect("the copy fires the same disruption on its own due list",
                 len(clone_fired) == 1 and clone_fired[0].t == 10.0,
                 events=[(event.kind, event.t) for event in clone_fired])
    trace.expect("the original keeps exactly one injection: the copy never wrote into it",
                 sum(1 for event in sim.events if event.family == "disruption.injected"
                     and event.kind == "link_outage") == 1)
    trace.expect("the copy really ran its own clock", clone.t == 10.0 and sim.t == 10.0,
                 original=sim.t, copy=clone.t)
    return trace


# --------------------------------------------------------------------------- #
# T7 - CommChannelGate byte budgets and delivered-only memory envelopes
# --------------------------------------------------------------------------- #

def t7():
    trace = Trace("T7", "B00/B10/B01/B11 字节预算门控与 MemoryEnvelope 送达校验",
                  "R2.6 comm budget gate; delivered-only edge memory")
    for mode, (up, down) in CHANNEL_MODES.items():
        gate = CommChannelGate(mode, 64)
        trace.expect(f"{mode} opens exactly its declared directions",
                     gate.allows("down") is down and gate.allows("up") is up,
                     mode=mode, down=down, up=up)
    silent = CommChannelGate("B00", 64)
    trace.expect("B00 admits nothing in either direction",
                 silent.admit("down", 1) is False and silent.admit("up", 1) is False
                 and silent.remaining() == {"down": 64, "up": 64},
                 state=silent.to_dict())
    downlink = CommChannelGate("B01", 16)
    trace.expect("B01 spends downlink bytes and refuses uplink",
                 downlink.admit("down", 10) is True and downlink.remaining("down") == 6
                 and downlink.admit("down", 7) is False and downlink.admit("up", 1) is False,
                 state=downlink.to_dict())
    uplink = CommChannelGate("B10", 16, b_up_bytes=8)
    trace.expect("B10 honours the uplink-only budget",
                 uplink.admit("up", 8) is True and uplink.admit("up", 1) is False
                 and uplink.admit("down", 1) is False,
                 state=uplink.to_dict())
    both = CommChannelGate("B11", 32, b_down_bytes=8, b_up_bytes=24)
    trace.expect("B11 keeps independent direction budgets",
                 both.admit("down", 8) and both.admit("up", 24)
                 and not both.admit("down", 1) and not both.admit("up", 1),
                 state=both.to_dict())

    env = demo_env()
    sim, agent = env.sim, env.sim.agents["uav-1"]
    isolate(sim, agent)
    agent.spec["task_allowlist"] = []
    mid = sim.network.send("coordinator-1", ["uav-1"], "report", {"op": "report", "n": 1}, 32,
                           ttl_s=180)
    message = sim.network.messages[mid]
    pending = MemoryEnvelope.from_message(message.to_dict())
    trace.expect("an undelivered message never becomes a readable envelope",
                 pending.delivered_at_s is None
                 and pending.readable(sim.t, receiver="uav-1") is False,
                 delivered_at=pending.delivered_at_s)
    run_until(env, lambda: message.delivered_s is not None, max_s=60)
    delivered = MemoryEnvelope.from_message(message.to_dict())
    trace.expect("the delivered envelope is readable for its receiver only",
                 delivered.delivered_at_s is not None
                 and delivered.readable(sim.t, receiver="uav-1")
                 and not delivered.readable(sim.t, receiver="uav-2"),
                 delivered_at=delivered.delivered_at_s)
    expired = MemoryEnvelope.from_dict(dict(delivered.to_dict(), valid_until_s=sim.t - 1))
    trace.expect("an expired envelope is not readable",
                 expired.readable(sim.t, receiver="uav-1") is False)
    tampered = MemoryEnvelope.from_dict(dict(delivered.to_dict(), payload={"op": "forged"}))
    trace.expect("a tampered envelope fails its digest check",
                 tampered.verified() is False and tampered.readable(sim.t, receiver="uav-1") is False)
    trace.expect("readable_envelopes filters the unreadable rows",
                 [row.envelope_id for row in readable_envelopes(
                     [pending, delivered, expired], sim.t, receiver="uav-1")] == [delivered.envelope_id])

    situation = {"now_s": sim.t, "vehicle": "uav-1",
                 "memory": [pending.to_dict(), delivered.to_dict(), expired.to_dict()]}
    observation = edge_harness._observation(situation, "uav-1", {})
    trace.expect("the edge harness exposes only the delivered envelope",
                 [row["envelope_id"] for row in observation["memory"]] == [delivered.envelope_id]
                 and observation["memory_unreadable"] == 2,
                 memory=[row["envelope_id"] for row in observation["memory"]],
                 unreadable=observation["memory_unreadable"])

    gated = sim.agents["robot_dog-1"]
    gated.comm_budget_gate = CommChannelGate("B01", 0)
    before = len(gated.received_messages)
    gated.on_message(sim, deepcopy(message))
    trace.expect("a holder whose downlink budget is exhausted never reads the message",
                 len(gated.received_messages) == before
                 and gated.records.get("comm_gate_denied"),
                 denied=gated.records.get("comm_gate_denied"))
    sent = gated._send(sim, ["coordinator-1"], "report", {"op": "report", "n": 2})
    trace.expect("B01 refuses uplink sends entirely", sent is None)
    return trace


# --------------------------------------------------------------------------- #
# Primitives: the frozen 66-action space, the 11 skill labels, refusal paths
# --------------------------------------------------------------------------- #

def primitives():
    trace = Trace("T0", "统一原语：66 动作集、11 个 Skill 标签、越界拒绝",
                  "R3.1 action space and skill mapping")
    trace.expect("the action space is exactly A001-A066",
                 VALID_ACTION_IDS == frozenset(f"A{i:03d}" for i in range(1, 67))
                 and len(VALID_ACTION_IDS) == 66
                 and {"A064", "A065", "A066"} <= VALID_ACTION_IDS,
                 size=len(VALID_ACTION_IDS))
    trace.expect("ten rescue skills plus the recovery lane are declared",
                 len(SKILL_LABELS) == 11 and len(SKILL_NAMES) == 11
                 and SKILL_NAMES[:10] == ("search_and_assess", "contact_person", "extricate_person",
                                          "pickup_supply", "aid_person", "load_and_evacuate",
                                          "water_rescue", "suppress_fire", "isolate_hazard",
                                          "clear_obstacle")
                 and SKILL_NAMES[10] == "recover_replan")
    from rescue_scenes.core.catalog import load
    catalog = load()
    grouped = skill_sequences(catalog)
    covered = {seq for ids in grouped.values() for seq in ids}
    trace.expect("every catalog sequence is classified exactly once",
                 covered == set(catalog.sequences)
                 and sum(len(ids) for ids in grouped.values()) == len(catalog.sequences),
                 sequences=len(covered))
    trace.expect("all eleven labels own at least one sequence",
                 all(grouped[name] for name in SKILL_NAMES),
                 sizes={name: len(ids) for name, ids in grouped.items()})
    trace.expect("recovery branches are the recovery skill",
                 sequence_skill("Q098", catalog) == "recover_replan"
                 and sequence_skill("Q011", catalog) == "search_and_assess"
                 and sequence_skill("Q033", catalog) == "load_and_evacuate")
    for bad in ("A000", "A067", "a001", "B001"):
        try:
            CandidateAction(candidate_id="bad", skill_name="search_and_assess", action_id=bad)
            refused = False
        except ValueError:
            refused = True
        trace.expect(f"candidate construction refuses {bad}", refused)
    try:
        CandidateAction(candidate_id="bad", skill_name="search_and_assess", action_id="A001",
                        task_rev=1)
        refused = False
    except ValueError:
        refused = True
    trace.expect("an unbound candidate cannot carry a task revision", refused)
    return trace


TRAJECTORIES = (primitives, t1, t2, t3, t4, t5, t6, t7)


def build_report(traces, elapsed_s):
    trajectories = [trace.to_dict() for trace in traces]
    passed = sum(1 for row in trajectories if row["ok"])
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "contract": "astra Round 3 frozen kernel/skills/harness contract (R2.3/R2.4/R2.6/R3.1/R3.6)",
        "elapsed_s": round(elapsed_s, 2),
        "summary": {"trajectories": len(trajectories), "passed": passed,
                    "failed": len(trajectories) - passed,
                    "assertions": sum(len(row["assertions"]) for row in trajectories),
                    "assertions_passed": sum(1 for row in trajectories
                                             for item in row["assertions"] if item["ok"])},
        "trajectories": trajectories,
    }


def render_doc(report):
    lines = [
        "# Harness 与 Skills 验收矩阵（T0 ~ T7）",
        "",
        "本文件由 `tools/verify_harness_and_skills.py` 实测生成，禁止手工编辑。",
        "所有断言均运行真实 `Simulation` 内核与真实 `AgentRuntime`，无桩、无硬编码分数。",
        "",
        f"- 生成时间：`{report['generated_at']}`",
        f"- 契约：`{report['contract']}`",
        f"- 结果：**{report['summary']['passed']}/{report['summary']['trajectories']} 轨迹通过**，"
        f"{report['summary']['assertions_passed']}/{report['summary']['assertions']} 断言通过",
        f"- 耗时：{report['elapsed_s']} s",
        "",
        "## 1. 轨迹总览",
        "",
        "| 轨迹 | 覆盖契约 | 断言数 | 结果 |",
        "| :--- | :--- | ---: | :--- |",
    ]
    for row in report["trajectories"]:
        lines.append(f"| `{row['id']}` {row['title']} | {row['contract']} | "
                     f"{len(row['assertions'])} | {'✅ PASS' if row['ok'] else '❌ FAIL'} |")
    lines += ["", "## 2. 逐条断言", ""]
    for row in report["trajectories"]:
        lines += [f"### {row['id']} · {row['title']}", "",
                  f"契约：`{row['contract']}`", "",
                  "| # | 断言 | 结果 | 实测细节 |", "| ---: | :--- | :--- | :--- |"]
        for index, item in enumerate(row["assertions"], start=1):
            detail = json.dumps(item["detail"], ensure_ascii=False, sort_keys=True)
            if len(detail) > 220:
                detail = detail[:217] + "..."
            detail = detail.replace("|", "\\|")
            lines.append(f"| {index} | {item['assertion']} | "
                         f"{'✅' if item['ok'] else '❌'} | `{detail}` |")
        lines.append("")
    lines += [
        "## 3. 冻结契约与实现位置",
        "",
        "| 契约 | 实现 | 验收 |",
        "| :--- | :--- | :--- |",
        "| R2.3 定时世界扰动 | `Simulation._enqueue_world`（`kernel.py`）在调用时读取自身 `_due_world`，"
        "`deepcopy(sim)` 后回调重绑到副本 | `T6` |",
        "| R2.3 外部运动 | `submit_action` 标记 `external_motion`，只挂有界 watchdog；"
        "`ActionModel.start/_move_progress/_effect_A001` 跳过脚本插值；"
        "`Simulation.complete_external_action` 走正常结算 | `T5` |",
        "| R2.4 决策边界 | `AgentRuntime.policy_mode`/`motion_backend`/`comm_budget_gate`；"
        "`external_harness` 仅停内部选步，消息/控制队列/A041 融合仍活 | `T1`,`T4`,`T5` |",
        "| R2.6 通道预算 | `CommChannelGate`（B00/B10/B01/B11）门控 `_send`/`on_message` | `T7` |",
        "| R3.1 统一候选 | `CandidateAction`/`DecisionContext`/`MemoryEnvelope`/"
        "`build_decision_candidates`（`skills.py`） | `T0`,`T1`,`T7` |",
        "| R3.6 统一执行 | `execute_candidate` → `AgentRuntime._submit` → `pending_actions` → "
        "`on_action_result` | `T1`,`T3`,`T4`,`T5` |",
        "| PX4 具身 | `engine._sync_px4` → `px4_bridge.complete_reached_movements`（2 m 2D/显式高度 3D 容差） | `T5` |",
        "",
        "## 4. 复现",
        "",
        "```bash",
        ".venv/bin/python tools/verify_harness_and_skills.py",
        ".venv/bin/python -m pytest tests/test_sandbox_and_harness.py -q",
        "```",
        "",
        "> 说明：验收脚本对场景真值的写入（伤员 `attr.mass_kg`、接收站 `attr.receiver_confirmation`、"
        "道路封闭/清除）属于“作者场景设定”，与 `configs/` 中的场景数据同性质；"
        "所有结论仍由内核真实执行产生。",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    parser.add_argument("--no-write", action="store_true",
                        default=os.environ.get("AERO_VERIFY_NO_WRITE") == "1",
                        help="verify without updating generated report files")
    args = parser.parse_args(argv)
    started = time.time()
    traces, failures = [], []
    for factory in TRAJECTORIES:
        try:
            trace = factory()
        except AssertionError as exc:
            failures.append(str(exc))
            trace = Trace(factory.__name__.upper(), factory.__doc__ or "", "failed")
            trace.rows.append({"assertion": str(exc), "ok": False, "detail": {}})
        traces.append(trace)
        if not args.quiet:
            state = "PASS" if trace.to_dict()["ok"] else "FAIL"
            print(f"[{state}] {trace.tid} {trace.title}")
    report = build_report(traces, time.time() - started)
    if not args.no_write:
        JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
        JSON_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
        DOC_OUT.write_text(render_doc(report), encoding="utf-8")
    summary = report["summary"]
    print(f"{summary['passed']}/{summary['trajectories']} trajectories, "
          f"{summary['assertions_passed']}/{summary['assertions']} assertions passed "
          f"in {report['elapsed_s']} s")
    if not args.no_write:
        print(f"wrote {JSON_OUT.relative_to(ROOT)} and {DOC_OUT.relative_to(ROOT)}")
    for failure in failures:
        print("FAILED:", failure)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
