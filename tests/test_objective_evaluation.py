"""Focused tests for Evaluator._objective, especially isolate_hazard truth semantics.

Defect under test: truth.py declares isolate_hazard with
desired={"hazard.exclusion_zone": True}, while actions.py _effect_A063 writes the
exclusion-zone polygon (dict/list) to hazard.exclusion_zone.  The generic
desired-field branch used to run first and compare `polygon == True`, so the
objective was unreachable (SC17 objective-12, SC09 objectives 4-5).
"""
from copy import deepcopy

import pytest

from rescue_scenes.core.sim.evaluation import Evaluator


class StubWorld:
    """Minimal stand-in for WorldTruth: only what Evaluator._objective reads."""

    def __init__(self, entities=None, objectives=None):
        self.entities = entities or {}
        self.objectives = objectives or []

    def get(self, eid, key, default=None):
        value = self.entities.get(eid, {}).get("fields", {}).get(key, default)
        if type(value) in (str, int, float, bool, type(None)):
            return value
        return deepcopy(value)


class StubSim:
    def __init__(self, world):
        self.world = world


POLYGON = {"kind": "polygon",
           "coords": [[340.0, 210.0], [380.0, 210.0], [380.0, 250.0], [340.0, 250.0], [340.0, 210.0]]}
POLYGON_LIST = [[340.0, 210.0], [380.0, 210.0], [380.0, 250.0], [340.0, 250.0], [340.0, 210.0]]


def make_evaluator(entities):
    return Evaluator(StubSim(StubWorld(entities=entities)))


def hazard(hid, **fields):
    return {"id": hid, "cls": "hazard", "fields": fields}


def isolate_objective(hid):
    return {"id": "objective-isolate-" + hid, "kind": "isolate_hazard",
            "target": hid, "desired": {"hazard.exclusion_zone": True}}


def test_isolate_hazard_satisfied_by_polygon_dict():
    evaluator = make_evaluator({"hazard-1": hazard("hazard-1", **{"hazard.exclusion_zone": POLYGON})})
    progress, note = evaluator._objective(isolate_objective("hazard-1"))
    assert progress == 1.0
    assert note == "核对危险区隔离设置。"


def test_isolate_hazard_satisfied_by_polygon_list():
    evaluator = make_evaluator({"hazard-1": hazard("hazard-1", **{"hazard.exclusion_zone": POLYGON_LIST})})
    progress, _ = evaluator._objective(isolate_objective("hazard-1"))
    assert progress == 1.0


@pytest.mark.parametrize("fields", [
    {},                                         # field never written
    {"hazard.exclusion_zone": None},
    {"hazard.exclusion_zone": {}},
    {"hazard.exclusion_zone": []},
])
def test_isolate_hazard_unsatisfied_without_established_zone(fields):
    evaluator = make_evaluator({"hazard-1": hazard("hazard-1", **fields)})
    progress, _ = evaluator._objective(isolate_objective("hazard-1"))
    assert progress == 0.0


def test_isolate_hazard_unknown_target_is_not_satisfied():
    evaluator = make_evaluator({})
    progress, _ = evaluator._objective(isolate_objective("hazard-missing"))
    assert progress == 0.0


def test_isolate_hazard_two_targets_one_isolated_is_not_fully_satisfied():
    """Current aggregation for this kind is all-or-nothing, not a mean."""
    evaluator = make_evaluator({
        "hazard-1": hazard("hazard-1", **{"hazard.exclusion_zone": POLYGON}),
        "hazard-2": hazard("hazard-2"),
    })
    objective = {"id": "objective-isolate-both", "kind": "isolate_hazard",
                 "target": ["hazard-1", "hazard-2"], "desired": {"hazard.exclusion_zone": True}}
    progress, _ = evaluator._objective(objective)
    assert progress == 0.0
    assert progress < 1.0


def test_non_isolate_kind_still_uses_generic_desired_comparison():
    """handover must keep the `actual == expected` desired-field semantics."""
    entities = {
        "human-1": {"id": "human-1", "cls": "human",
                    "fields": {"human.handover_status": "acknowledged"}},
        "human-2": {"id": "human-2", "cls": "human",
                    "fields": {"human.handover_status": "pending"}},
    }
    evaluator = make_evaluator(entities)
    objective = lambda hid: {"id": "objective-handover-" + hid, "kind": "handover",
                             "target": hid,
                             "desired": {"human.handover_status": "acknowledged"}}
    assert evaluator._objective(objective("human-1")) == (1.0, "根据目标声明的真值字段逐项核对。")
    assert evaluator._objective(objective("human-2")) == (0.0, "根据目标声明的真值字段逐项核对。")
