"""F-B: platform-consumable rescue-chain events tied to applied world writes.

The frontend rescue feed counts handedOver from rows whose kind is
``rescue.handover_acknowledged`` and extricated from ``rescue.extricated``.
These tests drive the real Simulation/action pipeline (no stubs of the kernel)
with the compact authored demo world and assert that one row is emitted per
actual state transition, carrying the real person, carrier and receiver.
"""
import pytest

from rescue_scenes.core.api import Environment
from rescue_scenes.engine import RescueWorld

PERSON = "human-building-1-1"          # trapped + serious injury in the demo
PERSON_IMMOBILE = "human-building-1-2"
TEAM = "specialist_team-1"
RECEIVER = "station-3"                 # demo medical receiving point
KINDS = ("rescue.extricated", "rescue.loaded", "rescue.transport_started",
         "rescue.transport_arrived", "rescue.handover_acknowledged")


def build_env():
    env = Environment.from_config({"scenario_id": "SC01", "seed": 1,
                                   "semantics": {"source": "demo"},
                                   "tick_s": 1, "until_s": 2000})
    # Only the carrier team: no coordinator means no autonomous allocation, and
    # no periodic task generation can race the actions this test drives itself.
    team = env.sim.agents[TEAM]
    team._last_generate = float("inf")
    env.sim.agents = {TEAM: team}
    return env


def place(env, person, point, mobility="immobile"):
    """Put the person and the carrier team at one on-road point."""
    world = env.sim.world
    world.set(person, "human.buriedness", 0.0, "authored_demo")
    world.set(person, "human.mobility", mobility, "authored_demo")
    world.set(person, "human.handover_status", "not_started", "authored_demo")
    world.set(person, "human.carrier_id", None, "authored_demo")
    world.set(person, "human.destination_id", None, "authored_demo")
    world.set(person, "entity.pose", list(point), "authored_demo")
    world.set(TEAM, "entity.pose", list(point), "authored_demo")
    return world


def add_station(env, sid, point, accepted=None, capacity=4, confirmation=True):
    world = env.sim.world
    world.set(sid, "entity.kind", "station", "authored_demo")
    world.set(sid, "entity.pose", list(point), "authored_demo")
    world.set(sid, "station.kind", "medical", "authored_demo")
    world.set(sid, "station.capacity", capacity, "authored_demo")
    world.set(sid, "station.occupied", 0, "authored_demo")
    world.set(sid, "station.open_state", "open", "authored_demo")
    world.set(sid, "station.accepted_need_types",
              list(accepted or ["water", "medical_supplies", "transport"]), "authored_demo")
    world.set(sid, "attr.receiver_confirmation", bool(confirmation), "authored_demo")
    return world


def run_action(env, action_id, params, limit_s=1500.0, task_ref=None):
    sim = env.sim
    iid = sim.submit_action(TEAM, action_id, params, task_ref=task_ref)
    rec = sim.instances[iid]
    while rec["status"] in ("requested", "accepted", "running") and sim.t < limit_s:
        sim.run(min(limit_s, sim.t + 30.0))
    if rec["status"] == "rejected" and "result" not in rec:
        # The kernel delivers rejection results one tick later.
        sim.run(min(limit_s, sim.t + max(1.0, sim.config["tick_s"])))
    return iid, rec


def chain_events(env, kind):
    return [row for row in (event.to_dict() for event in env.sim.events) if row["kind"] == kind]


def source_event(env, row):
    """The world write this milestone was derived from, by its cited seq."""
    return next(event for event in env.sim.events if event.seq == row["after"]["source_seq"])


def attach_task(env, task_id="T017@specialist_team-1#1", task_type="T017", rev=1):
    """A minimal holder-local task record so the action has a real task_ref."""
    from rescue_scenes.core.agents.runtime import COST_KEYS
    env.sim.agents[TEAM].tasks[task_id] = {
        "id": task_id, "type": task_type, "rev": rev, "status": "executing",
        "assigned_to": TEAM, "costs": dict.fromkeys(COST_KEYS, 0), "progress": {},
        "result_evidence_ids": [], "method": {}, "parameters": {}, "history": []}
    return task_id, rev


def transport_params(person, point, receiver):
    # A literal goal keeps the test independent of the demo team's local belief
    # copy; destination_id is still the actual receiving station.
    return {"target": person, "goal": list(point), "destination_id": receiver}


def test_full_chain_emits_one_event_per_transition():
    env = build_env()
    world = place(env, PERSON, (20.0, 40.0), mobility="trapped")
    world.set(PERSON, "human.buriedness", 20.0, "authored_demo")
    world.set(RECEIVER, "attr.receiver_confirmation", True, "authored_demo")

    iid17, rec17 = run_action(env, "A017", {"target": PERSON})
    assert rec17["result"].status == "succeeded", rec17["result"].feedback_zh

    iid20, rec20 = run_action(env, "A020", {"target": PERSON})
    assert rec20["result"].status == "succeeded", rec20["result"].feedback_zh

    iid21, rec21 = run_action(env, "A021", transport_params(PERSON, (16.0, 44.0), RECEIVER))
    assert rec21["result"].status == "succeeded", rec21["result"].feedback_zh
    preconditions = {key: world.get(PERSON, key) for key in
                     ("human.carrier_id", "human.handover_status", "human.destination_id",
                      "human.location_context", "human.mobility")}
    preconditions.update(team={
        key: world.get(TEAM, key, default) for key, default in
        (("platform.health_state", "nominal"), ("platform.current_action_id", None))})
    assert preconditions["human.carrier_id"] == TEAM, preconditions
    assert preconditions["team"]["platform.health_state"] != "failed", preconditions
    assert preconditions["team"]["platform.current_action_id"] is None, preconditions

    iid22, rec22 = run_action(env, "A022", {"target": PERSON, "station_id": RECEIVER})
    assert rec22["result"].status == "succeeded", (preconditions, rec22["result"].rejection_code,
                                                   rec22["result"].feedback_zh)
    assert world.get(PERSON, "human.handover_status") == "acknowledged"

    rows = {kind: chain_events(env, kind) for kind in KINDS}
    assert all(len(found) == 1 for found in rows.values()), {k: len(v) for k, v in rows.items()}

    expected_action = {"rescue.extricated": "A017", "rescue.loaded": "A020",
                       "rescue.transport_started": "A021", "rescue.transport_arrived": "A021",
                       "rescue.handover_acknowledged": "A022"}
    expected_instance = {"rescue.extricated": iid17, "rescue.loaded": iid20,
                         "rescue.transport_started": iid21, "rescue.transport_arrived": iid21,
                         "rescue.handover_acknowledged": iid22}
    for kind, found in rows.items():
        event = found[0]
        payload = event["after"]
        assert event["subject"] == PERSON
        assert payload["person"] == PERSON
        assert payload["action_id"] == expected_action[kind]
        assert event["action_instance"] == expected_instance[kind]
        assert event["actor"] == TEAM
        assert payload["carrier"] == TEAM
        assert event["seq"] > 0 and event["t"] > 0
        assert isinstance(payload["source_seq"], int) and payload["source_seq"] > 0
        assert payload["source_seq"] in payload["write_seqs"]
        assert payload["write_seqs"]
        # Provenance is the originating write's, not sim.emit's defaults.
        write = source_event(env, event)
        assert (event["task"], event["task_rev"], event["backend_mode"],
                event["scripted"]) == (write.task, write.task_rev,
                                       write.backend_mode, write.scripted)

    assert "receiver" not in rows["rescue.extricated"][0]["after"]
    assert "receiver" not in rows["rescue.loaded"][0]["after"]
    for kind in ("rescue.transport_started", "rescue.transport_arrived",
                 "rescue.handover_acknowledged"):
        assert rows[kind][0]["after"]["receiver"] == RECEIVER
    assert rows["rescue.handover_acknowledged"][0]["after"]["confirmed"] is True


def test_milestone_carries_the_originating_task_and_revision():
    env = build_env()
    world = place(env, PERSON, (20.0, 40.0), mobility="trapped")
    world.set(PERSON, "human.buriedness", 20.0, "authored_demo")
    task_ref = attach_task(env)

    _, rec = run_action(env, "A017", {"target": PERSON}, task_ref=task_ref)
    assert rec["result"].status == "succeeded", rec["result"].feedback_zh
    row = chain_events(env, "rescue.extricated")[0]
    write = source_event(env, row)
    assert write.task == task_ref[0] and write.task_rev == task_ref[1]
    assert row["task"] == task_ref[0]
    assert row["task_rev"] == task_ref[1]
    assert row["after"]["task_id"] == task_ref[0]


@pytest.mark.parametrize("mode", ["stub", "camera_image"])
def test_milestone_provenance_matches_scripted_and_camera_source_writes(monkeypatch, mode):
    from rescue_scenes.core.sim.actions import MODELS
    # The milestone copies whatever provenance the originating write carried:
    # a scripted (stub) backend or the camera backend.
    monkeypatch.setattr(MODELS["A017"], "backend_mode", mode)
    env = build_env()
    world = place(env, PERSON, (20.0, 40.0), mobility="trapped")
    world.set(PERSON, "human.buriedness", 20.0, "authored_demo")

    _, rec = run_action(env, "A017", {"target": PERSON})
    assert rec["result"].status == "succeeded", rec["result"].feedback_zh
    row = chain_events(env, "rescue.extricated")[0]
    write = source_event(env, row)
    assert row["backend_mode"] == mode == write.backend_mode
    assert row["scripted"] is (mode == "stub")
    assert row["scripted"] == write.scripted


def test_partial_extrication_emits_no_extricated_event():
    env = build_env()
    world = place(env, PERSON, (20.0, 40.0), mobility="trapped")
    world.set(PERSON, "human.buriedness", 100000.0, "authored_demo")

    _, rec = run_action(env, "A017", {"target": PERSON})
    assert rec["result"].status == "partial", rec["result"].feedback_zh
    assert chain_events(env, "rescue.extricated") == []
    assert world.get(PERSON, "human.buriedness") > 0


def test_rejected_loading_emits_no_loaded_event():
    env = build_env()
    place(env, PERSON, (20.0, 40.0), mobility="trapped")

    _, rec = run_action(env, "A020", {"target": PERSON})
    assert rec["status"] == "rejected"
    assert rec["rejection_code"] == "incompatible"
    assert chain_events(env, "rescue.loaded") == []


def test_receiver_is_the_actual_station_when_two_stations_exist():
    env = build_env()
    world = place(env, PERSON_IMMOBILE, (24.0, 40.0))
    add_station(env, "station-4", (28.0, 40.0))
    assert next(entity["id"] for entity in world.entities_of("station")) == "station-1"

    _, rec20 = run_action(env, "A020", {"target": PERSON_IMMOBILE})
    assert rec20["result"].status == "succeeded", rec20["result"].feedback_zh
    _, rec21 = run_action(env, "A021", transport_params(PERSON_IMMOBILE, (28.0, 40.0), "station-4"))
    assert rec21["result"].status == "succeeded", rec21["result"].feedback_zh
    _, rec22 = run_action(env, "A022", {"target": PERSON_IMMOBILE, "station_id": "station-4"})
    assert rec22["result"].status == "succeeded", rec22["result"].feedback_zh

    ack = chain_events(env, "rescue.handover_acknowledged")
    assert len(ack) == 1
    assert ack[0]["after"]["receiver"] == "station-4"
    assert world.get(PERSON_IMMOBILE, "human.destination_id") == "station-4"

    # engine._stage must name the same actual receiver in lowalt.handoff.transferred.
    class Capture:
        def __init__(self):
            self.rows = []

        def emit(self, kind, payload, topic=None):
            self.rows.append((kind, payload))

    world_plugin = object.__new__(RescueWorld)
    world_plugin.sim = env.sim
    world_plugin.chain_t = {}
    world_plugin.previous = {}
    world_plugin.refs = {PERSON_IMMOBILE: None, TEAM: None}
    ctx = Capture()
    RescueWorld._stage(world_plugin, ctx, {"stage": "handover_ack", "target": PERSON_IMMOBILE,
                                           "actor": TEAM, "t": env.sim.t})
    handoffs = [payload for kind, payload in ctx.rows if kind == "lowalt.handoff.transferred"]
    assert handoffs and handoffs[-1]["receiver"] == "station-4"


def test_incompatible_station_is_refused_without_handover_event():
    env = build_env()
    world = place(env, PERSON_IMMOBILE, (16.0, 44.0))
    world.set(RECEIVER, "station.accepted_need_types", ["water"], "authored_demo")
    world.set(RECEIVER, "attr.receiver_confirmation", True, "authored_demo")

    _, rec20 = run_action(env, "A020", {"target": PERSON_IMMOBILE})
    assert rec20["result"].status == "succeeded", rec20["result"].feedback_zh
    _, rec21 = run_action(env, "A021", transport_params(PERSON_IMMOBILE, (16.0, 44.0), RECEIVER))
    assert rec21["result"].status == "succeeded", rec21["result"].feedback_zh
    assert chain_events(env, "rescue.transport_arrived")

    _, rec22 = run_action(env, "A022", {"target": PERSON_IMMOBILE, "station_id": RECEIVER})
    assert rec22["result"].status == "failed"
    assert rec22["result"].failure_code == "incompatible"
    assert "兼容" in rec22["result"].feedback_zh
    assert chain_events(env, "rescue.handover_acknowledged") == []
    assert world.get(PERSON_IMMOBILE, "human.handover_status") == "arrived"


@pytest.mark.parametrize("confirmation,occupied,failure", [
    (False, 0, "timeout"),      # receiver never confirms
    (True, 4, "actuator"),      # capacity full
])
def test_failed_handover_emits_no_handover_event(confirmation, occupied, failure):
    env = build_env()
    world = place(env, PERSON_IMMOBILE, (16.0, 44.0))
    world.set(RECEIVER, "station.capacity", 4, "authored_demo")
    world.set(RECEIVER, "station.occupied", occupied, "authored_demo")
    world.set(RECEIVER, "attr.receiver_confirmation", confirmation, "authored_demo")

    _, rec20 = run_action(env, "A020", {"target": PERSON_IMMOBILE})
    assert rec20["result"].status == "succeeded", rec20["result"].feedback_zh
    _, rec21 = run_action(env, "A021", transport_params(PERSON_IMMOBILE, (16.0, 44.0), RECEIVER))
    assert rec21["result"].status == "succeeded", rec21["result"].feedback_zh

    _, rec22 = run_action(env, "A022", {"target": PERSON_IMMOBILE, "station_id": RECEIVER})
    assert rec22["result"].status == "failed"
    assert rec22["result"].failure_code == failure
    assert chain_events(env, "rescue.handover_acknowledged") == []
    assert world.get(PERSON_IMMOBILE, "human.handover_status") == "arrived"
