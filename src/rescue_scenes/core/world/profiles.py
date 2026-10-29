"""Catalog-backed scenario vocabulary and explicitly authored model parameters.

All numerical parameters are ``authored_uncalibrated``: they are neither measured
rescue data nor a calibrated disaster/vision model.  This module declares data;
world truth, sensing, networking, compute and agent owners apply the parameters.
Callers should deepcopy a profile/overlay before applying configuration overrides.

Parameter contract (seconds, metres, Wh and probabilities unless stated):
* damage_model.intensity is a uniform [low, high] normalized intensity range;
  fragility maps exact building.structure values to intensity multipliers.
  obstacle_rate is the per-source (building/tree) obstruction probability.
  water_depth_m and wind_mps are authored environmental values; smoke_density
  and gas_level
  are normalized [0, 1] severity proxies, not physical concentrations.
* occupancy_prior.default/by_use are per-capacity occupancy probabilities;
  max_per_building caps sampled occupants. Sampling is independent of damage.
* secondary_events contain {t, kind, target?, params}. Times are absolute since
  run start. A missing target means the owning model selects a matching entity
  deterministically (seeded, stable id order); an empty candidate set is a no-op.
  aftershock_reblock params describe debris; water_rise.delta_m is incremental;
  fire_spread.intensity_delta is incremental; target_moves.offset_m is [dx, dy]
  and the owning world model must constrain the result to a legal location.
* Overlay world.sensing_miss_rate/confusion_rate are absolute authored rate
  overrides (combine by max), not image processing. secondary_events append.
* network.loss_rate and latency_s override base settings (combine by max).
  outage_windows [{start_s, end_s, target}] append; target='*' selects all links.
  bandwidth_factor multiplies bandwidth, without implying a packet simulator.
* compute.memory_capacity_factor/cpu_capacity_factor multiply node capacities.
  service_failures [{t, target, recover_after_s}] schedule an unavailable service;
  target='*' selects a deterministically chosen service, not a forced job failure.
* energy.initial_battery_fraction sets mobile platforms' initial charge fraction
  (mains-powered coordinators/nodes excluded); inventory_factor and
  consumable_factor multiply starting quantities, rounding inventory down.
* agents commitments reserve an existing agent's availability, not world-created
  tasks. kind/count select the first matching agents in stable id order;
  start_s/end_s and interruptibility describe the reserved interval. deadline_s
  is a relative deadline assigned by the agent to newly generated work. conflicts
  list shared-resource reservations, not predetermined allocation outcomes.
  equipment_faults, wrong_evidence and stale_versions are scheduled local-agent
  injections; they must not alter world truth to match false/stale evidence.
* Optional profile inventory_prior maps exact item.kind values to station stock
  overrides. specialist_roles maps an embodiment kind to descriptive team roles;
  these labels grant no capabilities and do not imply automatic availability.
"""

from copy import deepcopy

from ..catalog import load


__all__ = ["PROFILES", "OVERLAYS"]


def _damage(intensity, fragility, obstacle_rate, *, water=0.0, wind=0.0,
            smoke=0.0, gas=0.0):
    return {
        "intensity": list(intensity),
        "fragility": dict(zip(("timber", "masonry", "RC", "steel", "mixed"),
                              fragility)),
        "obstacle_rate": obstacle_rate,
        "water_depth_m": water,
        "wind_mps": wind,
        "smoke_density": smoke,
        "gas_level": gas,
    }


def _occupancy(default, by_use, maximum):
    return {
        "default": default,
        "by_use": dict(zip(("residential", "commercial", "public", "industrial",
                            "storage", "shelter"), by_use)),
        "max_per_building": maximum,
    }


def _mix(*, uav=2, dogs=2, vehicles=1, teams=1, boats=0):
    counts = (("coordinator", 1), ("uav", uav), ("robot_dog", dogs),
              ("ground_vehicle", vehicles), ("specialist_team", teams),
              ("compute_node", 1), ("boat", boats))
    return [{"kind": kind, "count": count} for kind, count in counts if count]


def _event(t, kind, **params):
    return {"t": t, "kind": kind, "params": params}


# Exact names, families and task ids are attached from Catalog below. Each row
# independently authors its model and occupancy prior; there is no inferred
# mapping from visual damage to occupants or from objectives to agent tasks.
_SCENARIO_PARAMETERS = {
    "SC01": {
        "hazards": ["earthquake"],
        "affected_classes": ["building", "road", "bridge", "facility"],
        "damage_model": _damage((0.35, 0.90), (1.00, 1.30, 0.65, 0.55, 1.05), 0.48),
        "occupancy_prior": _occupancy(0.10, (0.18, 0.08, 0.12, 0.06, 0.02, 0.20), 6),
        "secondary_events": [_event(420, "aftershock_reblock", volume_m3=1.8,
                                    clear_width_m=0.35, step_height_m=0.35)],
        "agent_mix": _mix(),
        "public_brief_zh": "对震后街区全部建筑开展评估，搜索失联人员并核实道路入口；警惕余震，不预设建筑内人数。",
    },
    "SC02": {
        "hazards": ["wind"],
        "affected_classes": ["building", "road", "tree", "tree_cluster"],
        "damage_model": _damage((0.20, 0.65), (1.25, 0.80, 0.25, 0.45, 0.90),
                                0.42, wind=18.0),
        "occupancy_prior": _occupancy(0.08, (0.14, 0.05, 0.09, 0.04, 0.02, 0.22), 5),
        "secondary_events": [_event(540, "aftershock_reblock", volume_m3=0.9,
                                    obstacle_kind="tree", clear_width_m=0.45,
                                    step_height_m=0.25)],
        "agent_mix": _mix(),
        "public_brief_zh": "核查风灾街区屋顶、附属物、倒树与道路，搜索求助线索；飞行和通行须遵守风况限制。",
    },
    "SC03": {
        "hazards": ["flood"],
        "affected_classes": ["building", "road", "water", "bridge"],
        "damage_model": _damage((0.15, 0.60), (1.10, 0.80, 0.35, 0.40, 0.80),
                                0.25, water=0.65, wind=4.0),
        "occupancy_prior": _occupancy(0.09, (0.16, 0.04, 0.10, 0.04, 0.02, 0.24), 6),
        "secondary_events": [_event(480, "water_rise", delta_m=0.25)],
        "agent_mix": _mix(boats=2),
        "public_brief_zh": "评估受淹建筑、道路和桥梁，搜索受困与落水人员；按水深阈值派遣船只，机器狗不默认可涉水。",
    },
    "SC04": {
        "hazards": ["tsunami", "flood"],
        "affected_classes": ["building", "road", "coast", "water", "facility"],
        "damage_model": _damage((0.35, 0.85), (1.30, 1.05, 0.55, 0.60, 1.10),
                                0.50, water=1.10, wind=12.0),
        "occupancy_prior": _occupancy(0.07, (0.10, 0.04, 0.08, 0.03, 0.02, 0.26), 5),
        "secondary_events": [_event(600, "water_rise", delta_m=0.35),
                             _event(780, "target_moves", offset_m=[35.0, 15.0])],
        "agent_mix": _mix(boats=2, teams=2),
        "public_brief_zh": "巡查沿海淹没区、漂移线索和残骸，协同船只实施受控搜索与交接；水位变化为作者模型，非流体仿真。",
    },
    "SC05": {
        "hazards": ["wind"],
        "affected_classes": ["water", "coast", "facility"],
        "damage_model": _damage((0.03, 0.15), (0.65, 0.35, 0.15, 0.20, 0.45),
                                0.05, water=2.0, wind=8.0),
        "occupancy_prior": _occupancy(0.03, (0.04, 0.03, 0.05, 0.02, 0.01, 0.10), 3),
        "secondary_events": [_event(240, "target_moves", offset_m=[45.0, 20.0])],
        "agent_mix": _mix(uav=3, dogs=1, boats=2),
        "public_brief_zh": "搜索开放水域落水线索，由救援船承担水上回收并确认交接；岸上车辆和机器狗不得替代船只回收。",
    },
    "SC06": {
        "hazards": ["fire"],
        "affected_classes": ["building", "road", "facility"],
        "damage_model": _damage((0.30, 0.80), (1.40, 0.85, 0.45, 0.70, 1.00),
                                0.32, wind=3.0, smoke=0.72),
        "occupancy_prior": _occupancy(0.10, (0.17, 0.10, 0.15, 0.06, 0.02, 0.18), 6),
        "secondary_events": [_event(360, "fire_spread", intensity_delta=0.20,
                                    smoke_delta=0.12)],
        "agent_mix": _mix(teams=2),
        "specialist_roles": {"specialist_team": ["urban_rescue", "fire"]},
        "public_brief_zh": "评估火烟影响并搜索求助线索，设置警戒并请求消防专业队；灭火消耗和复燃均需按状态模型核实。",
    },
    "SC07": {
        "hazards": ["fire", "wind"],
        "affected_classes": ["tree", "tree_cluster", "building", "road", "open_area"],
        "damage_model": _damage((0.20, 0.75), (1.45, 0.65, 0.30, 0.50, 0.95),
                                0.38, wind=11.0, smoke=0.65),
        "occupancy_prior": _occupancy(0.06, (0.10, 0.04, 0.08, 0.03, 0.01, 0.20), 4),
        "secondary_events": [_event(300, "fire_spread", intensity_delta=0.25,
                                    smoke_delta=0.15)],
        "agent_mix": _mix(uav=3, teams=2),
        "specialist_roles": {"specialist_team": ["wildland_rescue", "fire"]},
        "public_brief_zh": "搜索林地与城郊交界求助线索，核查逃生道路并协调消防队；火线变化为抽象事件，不是精细火传播。",
    },
    "SC08": {
        "hazards": ["landslide"],
        "affected_classes": ["slope", "building", "road", "bridge", "tree_cluster"],
        "damage_model": _damage((0.30, 0.85), (1.10, 1.20, 0.60, 0.55, 1.10),
                                0.62, water=0.12),
        "occupancy_prior": _occupancy(0.07, (0.13, 0.04, 0.07, 0.04, 0.02, 0.18), 5),
        "secondary_events": [_event(510, "aftershock_reblock", volume_m3=3.5,
                                    clear_width_m=0.20, step_height_m=0.50)],
        "agent_mix": _mix(vehicles=2, teams=2),
        "public_brief_zh": "评估堆积区、建筑和阻断道路，搜索受困线索并请求适配设备；仅使用坡区平面语义，不模拟坡体动力。",
    },
    "SC09": {
        "hazards": ["volcano", "gas"],
        "affected_classes": ["building", "road", "slope", "open_area"],
        "damage_model": _damage((0.15, 0.55), (1.00, 0.70, 0.35, 0.40, 0.80),
                                0.28, wind=7.0, smoke=0.55, gas=0.35),
        "occupancy_prior": _occupancy(0.05, (0.08, 0.03, 0.06, 0.03, 0.01, 0.20), 4),
        "secondary_events": [],
        "agent_mix": _mix(teams=2),
        "public_brief_zh": "核查喷发后可见损毁、灰尘遮挡及禁入区，搜索求助线索；烟密度参数仅作灰尘遮挡代理，不代表火山动力或气体浓度。",
    },
    "SC10": {
        "hazards": ["none"],
        "affected_classes": ["slope", "tree_cluster", "road", "open_area"],
        "damage_model": _damage((0.0, 0.0), (0.70, 0.60, 0.35, 0.35, 0.60), 0.18),
        "occupancy_prior": _occupancy(0.03, (0.05, 0.02, 0.04, 0.01, 0.01, 0.12), 3),
        "secondary_events": [_event(720, "target_moves", offset_m=[25.0, -20.0])],
        "agent_mix": _mix(uav=3, dogs=3),
        "public_brief_zh": "搜索山地林区失踪线索，结合道路与坡区身份规划接近；平面投影忽略高程，不认证真实山地通行能力。",
    },
    "SC11": {
        "hazards": ["wind"],
        "affected_classes": ["open_area", "road", "facility"],
        "damage_model": _damage((0.02, 0.12), (0.90, 0.45, 0.15, 0.25, 0.60),
                                0.10, wind=10.0),
        "occupancy_prior": _occupancy(0.02, (0.03, 0.01, 0.03, 0.01, 0.01, 0.08), 2),
        "secondary_events": [_event(900, "target_moves", offset_m=[50.0, 10.0])],
        "agent_mix": _mix(uav=3, vehicles=2),
        "public_brief_zh": "在荒漠平面搜索稀疏线索，持续核查能源、补给和风况；沙尘与距离影响为显式作者参数，非真实荒漠基准复现。",
    },
    "SC12": {
        "hazards": ["wind"],
        "affected_classes": ["slope", "road", "open_area", "facility"],
        "damage_model": _damage((0.02, 0.18), (0.75, 0.50, 0.20, 0.30, 0.60),
                                0.28, wind=14.0),
        "occupancy_prior": _occupancy(0.02, (0.04, 0.01, 0.04, 0.01, 0.01, 0.10), 3),
        "secondary_events": [],
        "agent_mix": _mix(dogs=3, teams=2),
        "public_brief_zh": "搜索雪地高山区域失踪线索并协调专业接近；本模型仅使用平面坡区与障碍，不认证严寒、雪崩或陡坡救援。",
    },
    "SC13": {
        "hazards": ["wind", "landslide"],
        "affected_classes": ["coast", "slope", "road", "water", "open_area"],
        "damage_model": _damage((0.05, 0.25), (0.85, 0.65, 0.25, 0.35, 0.70),
                                0.25, wind=9.0),
        "occupancy_prior": _occupancy(0.03, (0.05, 0.02, 0.04, 0.01, 0.01, 0.12), 3),
        "secondary_events": [_event(600, "target_moves", offset_m=[20.0, 10.0])],
        "agent_mix": _mix(dogs=3, boats=1),
        "public_brief_zh": "区分岸上、坡区和水域的失踪线索，按可达性协调地面队与船只；不得把所有滨海目标当作落水目标。",
    },
    "SC14": {
        "hazards": ["gas", "flood"],
        "affected_classes": ["underground", "building", "road", "facility"],
        "damage_model": _damage((0.10, 0.40), (0.90, 0.80, 0.40, 0.45, 0.75),
                                0.35, water=0.18, smoke=0.12, gas=0.25),
        "occupancy_prior": _occupancy(0.08, (0.08, 0.10, 0.18, 0.04, 0.02, 0.16), 6),
        "secondary_events": [_event(660, "water_rise", delta_m=0.12)],
        "agent_mix": _mix(uav=1, dogs=3, teams=2),
        "inventory_prior": {"relay": 8},
        "public_brief_zh": "由机器狗与专业队搜索地下设施，部署中继并核实入口和危险线索；室内行动为抽象任务，不是三维地下导航复现。",
    },
    "SC15": {
        "hazards": ["gas", "landslide"],
        "affected_classes": ["underground", "road", "facility"],
        "damage_model": _damage((0.15, 0.55), (1.00, 0.95, 0.50, 0.50, 0.90),
                                0.48, water=0.08, gas=0.65),
        "occupancy_prior": _occupancy(0.05, (0.03, 0.02, 0.07, 0.14, 0.02, 0.12), 5),
        "secondary_events": [_event(450, "aftershock_reblock", volume_m3=2.5,
                                    clear_width_m=0.30, step_height_m=0.40)],
        "agent_mix": _mix(uav=1, dogs=4, teams=2),
        "inventory_prior": {"relay": 10},
        "public_brief_zh": "以机器狗、中继和专业队核查隧道矿井的失联、气体与阻塞线索；危险区进入须受控，不宣称完整矿山处置能力。",
    },
    "SC16": {
        "hazards": ["flood"],
        "affected_classes": ["underground", "slope", "road", "water"],
        "damage_model": _damage((0.0, 0.0), (0.90, 0.75, 0.40, 0.45, 0.70),
                                0.40, water=0.10),
        "occupancy_prior": _occupancy(0.02, (0.02, 0.01, 0.04, 0.01, 0.01, 0.08), 3),
        "secondary_events": [_event(840, "water_rise", delta_m=0.10)],
        "agent_mix": _mix(uav=1, dogs=4, teams=2),
        "inventory_prior": {"relay": 10},
        "public_brief_zh": "利用机器狗与中继搜索天然洞穴失踪线索，核查狭窄入口和积水；非规则三维几何与低照度感知不在本阶段验证范围。",
    },
    "SC17": {
        "hazards": ["fire", "gas"],
        "affected_classes": ["road", "facility", "building", "bridge"],
        "damage_model": _damage((0.20, 0.60), (1.15, 0.80, 0.40, 0.65, 0.85),
                                0.55, smoke=0.45, gas=0.30),
        "occupancy_prior": _occupancy(0.09, (0.05, 0.07, 0.12, 0.20, 0.03, 0.14), 7),
        "secondary_events": [_event(420, "fire_spread", intensity_delta=0.15,
                                    smoke_delta=0.10)],
        "agent_mix": _mix(vehicles=2, teams=2),
        "public_brief_zh": "核查交通工业事故的车辆设备阻塞、火烟与多伤员线索，按能力请求专业队；大型设备是否可用须获回执。",
    },
    "SC18": {
        "hazards": ["gas"],
        "affected_classes": ["facility", "building", "road", "open_area"],
        "damage_model": _damage((0.05, 0.25), (0.70, 0.50, 0.20, 0.30, 0.55),
                                0.15, wind=5.0, gas=0.85),
        "occupancy_prior": _occupancy(0.06, (0.06, 0.05, 0.09, 0.14, 0.02, 0.12), 5),
        "secondary_events": [],
        "agent_mix": _mix(teams=2),
        "public_brief_zh": "检测危险物质和气体线索，实施警戒、专业力量请求与受控接近；参数不是浓度测量，不包含完整CBRN处置。",
    },
}


# Empty component maps explicitly mean that this overlay does not change that
# component. The schedule/selector vocabulary is proposed implementation data;
# only each overlay's id/name/fields/tasks are the original catalog vocabulary.
_OVERLAY_EFFECTS = {
    "OV01": {
        "world": {"sensing_miss_rate": 0.30, "confusion_rate": 0.18},
        "network": {}, "compute": {}, "energy": {}, "agents": {},
    },
    "OV02": {
        "world": {},
        "network": {"loss_rate": 0.30, "latency_s": 2.5, "bandwidth_factor": 0.40},
        "compute": {}, "energy": {}, "agents": {},
    },
    "OV03": {
        "world": {},
        "network": {"outage_windows": [
            {"start_s": 60, "end_s": 180, "target": "*"},
            {"start_s": 420, "end_s": 540, "target": "*"},
        ]},
        "compute": {}, "energy": {}, "agents": {},
    },
    "OV04": {
        "world": {}, "network": {},
        "compute": {"memory_capacity_factor": 0.25, "cpu_capacity_factor": 0.50,
                    "service_failures": [{"t": 120, "target": "*",
                                          "recover_after_s": 180}]},
        "energy": {}, "agents": {},
    },
    "OV05": {
        "world": {}, "network": {}, "compute": {},
        "energy": {"initial_battery_fraction": 0.22, "inventory_factor": 0.35,
                   "consumable_factor": 0.30},
        "agents": {},
    },
    "OV06": {
        "world": {}, "network": {}, "compute": {}, "energy": {},
        "agents": {
            "commitments": [
                {"kind": "ground_vehicle", "count": 1, "start_s": 0, "end_s": 180,
                 "interruptibility": "non_interruptible",
                 "reason_zh": "车辆已有不可中断运输承诺"},
                {"kind": "specialist_team", "count": 1, "start_s": 0, "end_s": 300,
                 "interruptibility": "finish_current",
                 "reason_zh": "专业队需完成当前外部作业"},
            ],
            "deadline_s": 240,
            "conflicts": [{"resource": "station.charging_slot", "capacity": 1,
                           "start_s": 0, "end_s": 300}],
        },
    },
    "OV07": {
        "world": {"secondary_events": [
            _event(180, "aftershock_reblock", volume_m3=2.0,
                   clear_width_m=0.25, step_height_m=0.35),
            _event(360, "target_moves", offset_m=[30.0, -15.0]),
        ]},
        "network": {}, "compute": {}, "energy": {}, "agents": {},
    },
    "OV08": {
        "world": {}, "network": {}, "compute": {}, "energy": {},
        "agents": {
            "equipment_faults": [{"t": 90, "kind": "uav", "count": 1,
                                  "equipment_kind": "release", "health": "failed",
                                  "recoverable": True}],
            "wrong_evidence": [{"t": 150, "kind": "uav", "count": 1,
                                "target_class": "building",
                                "field": "building.visual_damage", "value": "destroyed",
                                "ensure_conflict": True, "confidence": 0.85}],
            "stale_versions": [{"t": 210, "kind": "robot_dog", "count": 1,
                                "revision_lag": 1}],
        },
    },
}


_catalog = load()
PROFILES = {
    scenario_id: {
        "id": scenario_id,
        "name_zh": _catalog.scenarios[scenario_id]["name"],
        "family": _catalog.scenarios[scenario_id]["family"],
        "calibration": "authored_uncalibrated",
        **deepcopy(parameters),
        "catalog_tasks": list(_catalog.scenarios[scenario_id]["tasks"]),
    }
    for scenario_id, parameters in _SCENARIO_PARAMETERS.items()
}

OVERLAYS = {
    overlay_id: {
        "id": overlay_id,
        "name_zh": _catalog.overlays[overlay_id]["name"],
        "fields": list(_catalog.overlays[overlay_id]["fields"]),
        "tasks": list(_catalog.overlays[overlay_id]["tasks"]),
        "calibration": "authored_uncalibrated",
        "effects": deepcopy(effects),
    }
    for overlay_id, effects in _OVERLAY_EFFECTS.items()
}
