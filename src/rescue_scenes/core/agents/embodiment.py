"""Authored, extensible platform specifications (SPEC §6.1).

Dimensions and rates are uncalibrated simulation parameters, not equipment
certifications.  Capabilities describe installed interfaces, not perfect outcomes.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite


class EmbodimentSpec(dict):
    """A JSON-compatible dict; attribute reads tolerate the §4.3 shorthand."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


EMBODIMENTS: dict[str, dict] = {}
CAPABILITY_IDS = frozenset(f"C{i:02d}" for i in range(1, 50))
_REQUIRED = {
    "kind", "label_zh", "mobility", "width_m", "step_height_m",
    "water_tolerance_m", "speed_mps", "battery_Wh", "payload_kg",
    "radio_range_m", "capabilities", "sensors", "external",
}
_MOBILITIES = {"air", "legged", "wheeled", "water", "foot", "none"}


def register(kind: str, spec: dict) -> dict:
    """Register an independent, validated dict specification and return it.

    Re-registering a kind deliberately replaces it.  Extra authored parameters
    (e.g. a reserve or availability policy) are preserved for extensions.
    """
    if not isinstance(kind, str) or not kind:
        raise ValueError("具身类型不能为空 / embodiment kind must be nonempty")
    if not isinstance(spec, dict):
        raise TypeError("具身规格必须是字典 / embodiment spec must be a dict")
    result = EmbodimentSpec(deepcopy(spec))
    if "kind" in result and result["kind"] != kind:
        raise ValueError("具身类型不一致 / inconsistent embodiment kind")
    result["kind"] = kind
    missing = _REQUIRED - result.keys()
    if missing:
        raise ValueError(f"缺少具身字段 / missing embodiment fields: {sorted(missing)}")
    if result["mobility"] not in _MOBILITIES:
        raise ValueError("未知机动模式 / unknown mobility")
    for key in ("width_m", "step_height_m", "water_tolerance_m", "speed_mps",
                "battery_Wh", "payload_kg", "radio_range_m"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"参数必须为非负有限数 / finite nonnegative number: {key}")
        if not isfinite(value) or value < 0:
            raise ValueError(f"参数必须为非负有限数 / finite nonnegative number: {key}")
    caps = result["capabilities"]
    if not isinstance(caps, (list, tuple)) or any(c not in CAPABILITY_IDS for c in caps):
        raise ValueError("未知能力 / unknown catalog capability")
    if not isinstance(result["sensors"], (list, tuple)):
        raise ValueError("传感器必须为列表 / sensors must be a list")
    if not isinstance(result["external"], bool):
        raise ValueError("external 必须为布尔值 / external must be boolean")
    result["capabilities"] = list(dict.fromkeys(caps))
    result["sensors"] = list(result["sensors"])
    EMBODIMENTS[kind] = result
    return result


def _spec(kind, label, mobility, dimensions, capabilities, sensors=(), external=False):
    width, step, water, speed, battery, payload, radio = dimensions
    caps = capabilities.split()
    # Contract reconciliation: A050/A051 require C42 in the original catalog,
    # while §6.4 requires every autonomous embodiment to propose its own tasks.
    # C42 here is the local task-generation/bidding interface, not omniscience.
    if "C42" not in caps:
        caps.append("C42")
    # Every runtime must integrate its own returned observations via A041 (§6.7).
    # C39/C44 are that local evidence/belief interface, not extra sensing equipment.
    caps.extend(c for c in ("C39", "C44") if c not in caps)
    # Mobile agents expose a specialist-request interface, not specialist tools.
    # Their integrated charging connector implements A029/C28 on their own pack.
    if mobility not in {"none"}:
        caps.extend(c for c in ("C46", "C28") if c not in caps)
    return register(kind, {
        "kind": kind, "label_zh": label, "mobility": mobility,
        "width_m": width, "step_height_m": step, "water_tolerance_m": water,
        "speed_mps": speed, "battery_Wh": battery, "payload_kg": payload,
        "radio_range_m": radio, "capabilities": caps,
        "sensors": list(sensors), "external": external,
        # Zero capacity with mains_power is explicitly unlimited, not an empty pack.
        "mains_power": mobility == "none", "calibration": "authored_uncalibrated",
    })


_spec("coordinator", "中心协调", "none", (1.0, 0, 0, 0, 0, 0, 2200),
      "C31 C32 C35 C39 C42 C43 C44 C46 C13 C14 C38")
_spec("uav", "无人机", "air", (0.7, 0, 0, 12, 180, 2, 1200),
      "C01 C04 C05 C09 C24 C25 C31 C32 C33 C34 C35 C39 C40 C41 C43 C44 C45 C12 C11",
      ("scripted_rgb", "scripted_range"))
_spec("robot_dog", "机器狗", "legged", (0.5, 0.3, 0.15, 1.5, 600, 5, 650),
      "C02 C04 C05 C09 C11 C15 C31 C34 C35 C39 C41 C43 C44 C45 C12 C06 C07",
      ("scripted_rgb", "scripted_range", "scripted_thermal", "scripted_acoustic"))
# A dog deliberately has neither light-manipulation C16 nor engineering C17.
_spec("ground_vehicle", "地面车辆", "wheeled", (2.2, 0.08, 0.3, 6, 18000, 800, 1800),
      "C02 C04 C05 C09 C14 C16 C22 C24 C26 C27 C31 C32 C33 C34 C35 C36 C37 C39 C43 C45 C28 C29",
      ("scripted_rgb", "scripted_range"))
_spec("boat", "救援船", "water", (1.8, 0, 10, 4, 12000, 500, 1600),
      "C03 C05 C23 C22 C31 C43 C45 C24 C26", ("scripted_rgb",))
# Contract reconciliation: §6.6 requires specialists to travel; the catalog's
# A002 ground-travel interface requires C02 even for a foot-mobile external team.
_spec("specialist_team", "专业救援队", "foot", (0.6, 0.4, 0.5, 1.2, 1200, 120, 1600),
      "C16 C17 C18 C19 C20 C21 C22 C30 C46 C47 C48 C11 C31 C43 C08 C10 C02",
      ("scripted_rf", "scripted_environment"), external=True)
# Contract reconciliation: §6.6/§6.7 make the compute host an AgentRuntime;
# its A053/A054 accept/progress task-management interfaces require C43.
_spec("compute_node", "边缘计算节点", "none", (0.5, 0, 0, 0, 0, 0, 1800),
      "C35 C36 C37 C38 C39 C49 C31 C12 C13 C43")

# The frozen table already covers all 49; fail visibly if a future edit loses one.
# Any future missing capability must be assigned to a plausible kind explicitly,
# never silently added to a dog (especially C16/C17) or to every platform.
_missing = CAPABILITY_IDS - {
    cap for spec in EMBODIMENTS.values() for cap in spec["capabilities"]
}
if _missing:
    raise RuntimeError(f"具身能力覆盖不完整 / uncovered capabilities: {sorted(_missing)}")
