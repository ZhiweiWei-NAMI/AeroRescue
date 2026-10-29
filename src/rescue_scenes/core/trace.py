"""Catalog coverage, separating executed evidence from available contracts.

This module never starts a simulation. Only the caller's completed coverage sweep
can supply execution evidence. Missing parallel-lane registries mean interface
coverage, not an assumption that their implementation exists or ran.
"""

from importlib import import_module
from collections.abc import Mapping

from . import __version__
from .catalog import load
from .contracts import EVENT_FAMILIES, STATE_MACHINES


# EDITORIAL TABLE, NOT RUN COVERAGE. The only hand-assigned coverage statuses.
# These statements describe scope/design, not claims that a quote was executed.
QUOTE_STATUS = {
    "U01": {"status": "modeled", "detail_zh": "统一可编程场景按配置与分层接口建模；实际运行范围另看扫描证据。"},
    "U02": {"status": "deferred", "detail_zh": "本期明确不实现视觉、图像生成或感知流水线；传感与识别仅为标注 stub 的脚本模型。"},
    "U03": {"status": "modeled", "detail_zh": "行为、通信与计算采用抽象模型；时延、能源与资源占用单独记账，非底层高保真控制。"},
    "U04": {"status": "interface_only", "detail_zh": "后端边界有契约；本期不接入 Gazebo 或 ns-3，不宣称运行过这些后端。"},
    "U05": {"status": "modeled", "detail_zh": "室内活动由 A008 抽象搜索任务表达，允许漏检、部分结果和失败，不是室内视觉闭环。"},
    "U06": {"status": "modeled", "detail_zh": "所有建筑进入潜在评估范围；人员占用独立于损毁采样，不把完好建筑当作无人。"},
    "U07": {"status": "modeled", "detail_zh": "默认二维 2×2 千米平面；消费上游语义区域但忽略高程，不修改上游地形。"},
    "U08": {"status": "modeled", "detail_zh": "任务由各持有者保存本地副本，当前任务、分配、接受与承诺分开记录。"},
    "U09": {"status": "modeled", "detail_zh": "清障、通信与计算均保留目录任务类型；是否生成并执行须由扫描逐项证明。"},
    "U10": {"status": "modeled", "detail_zh": "采用具体 Task→Sequence→Action→Capability→Field 映射，不以抽象能力面替代可执行动作。"},
    "U11": {"status": "modeled", "detail_zh": "统一成本契约记录时间、能源、字节与 CPU 秒，失败和中断也保留已耗成本。"},
    "U12": {"status": "modeled", "detail_zh": "接触、释放、落地、领取与确认移交是不同状态；arrived 不等于 acknowledged。"},
    "U13": {"status": "modeled", "detail_zh": "本地知识与消息共享分离；特征与 KV 共享仅为抽象兼容性模型，不宣称节省真实通信。"},
    "U14": {"status": "modeled", "detail_zh": "中心仅能获得已送达报告和自身观测；本地历史不自动成为中心知识。"},
    "U15": {"status": "interface_only", "detail_zh": "保留原目录来源限制；未取得完整历史与独立原始讨论，不杜撰归因或宣称历史全覆盖。"},
}


def _value(record, name, default=None):
    return record.get(name, default) if isinstance(record, Mapping) else getattr(record, name, default)


def _registry(module_name, attribute, default):
    """Read only public registries named by SPEC; do not depend on internals."""
    qualified = f"{__package__}.{module_name}"
    try:
        module = import_module(qualified)
    except ModuleNotFoundError as exc:
        # An absent lane is legitimate during assembly. An unrelated missing
        # dependency inside an existing lane is an error, not silent coverage.
        if exc.name == qualified or qualified.startswith(str(exc.name) + "."):
            return default
        raise
    return getattr(module, attribute, default)


def _ids(sweep, key):
    values = sweep.get(key, ())
    return set(values or ())


def _events(sweep):
    """Runs may retain events or just counts; counts prove no individual refs."""
    for run in sweep.get("runs", ()) or ():
        events = _value(run, "events", ())
        if not isinstance(events, (list, tuple)):
            continue
        for event in events:
            if isinstance(event, Mapping):
                yield event
            elif hasattr(event, "to_dict"):
                yield event.to_dict()


def _links(record):
    refs = []
    for key in ("sources", "tasks", "actions", "capabilities", "field_ids", "scenes", "refs"):
        values = record.get(key, ())
        if isinstance(values, (tuple, list)):
            refs.extend(value for value in values if isinstance(value, str))
    return list(dict.fromkeys(refs))


def build(sweep=None):
    """Build traceability from a completed §5.3 sweep, or contract coverage only.

    Tasks count as executed only in the intersection of generated and executed
    task types. Terminal action attempts (including failure/rejection) count;
    success is never inferred. `links` contains catalog IDs, not private paths.
    """
    sweep = {} if sweep is None else sweep
    cat = load()
    models = _registry("sim.actions", "MODELS", {})
    task_rules = _registry("agents.taskgen", "RULES", [])
    profiles = _registry("world.profiles", "PROFILES", {})
    overlays = _registry("world.profiles", "OVERLAYS", {})
    required_attributes = _registry("world.semantics", "REQUIRED_ATTRIBUTES", {})
    generated = _ids(sweep, "tasks_generated")
    executed = {
        "scenarios": _ids(sweep, "scenarios"),
        "overlays": _ids(sweep, "overlays"),
        "tasks": generated & _ids(sweep, "tasks_executed"),
        "sequences": _ids(sweep, "sequences"),
        "actions": _ids(sweep, "actions_terminal"),
        "capabilities": _ids(sweep, "capabilities_used"),
        "fields": _ids(sweep, "fields_written"),
        "event_families": _ids(sweep, "event_families"),
        "rules": set(),
    }
    executed_steps = set()
    for value in sweep.get("steps", ()) or ():
        if isinstance(value, (tuple, list)) and len(value) == 2:
            executed_steps.add((str(value[0]), str(value[1])))
    # Rule execution needs explicit emitted rule references; output-field overlap
    # alone cannot prove a particular generator/derivation ran.
    for event in _events(sweep):
        for ref in event.get("refs", ()) or ():
            if ref in cat.rules:
                executed["rules"].add(ref)

    model_actions = set(models)
    modeled_steps = set()
    for sequence, steps in cat.steps.items():
        for step in steps:
            if any(option in model_actions for option in step["options"]):
                modeled_steps.add((sequence, str(step["step"])))
    modeled_sequences = {
        sequence for sequence, steps in cat.steps.items()
        if steps and all((sequence, str(step["step"])) in modeled_steps for step in steps)
    }
    rule_task_types = {_value(rule, "task_type") for rule in task_rules}
    modeled_tasks = {task for task in rule_task_types if task in cat.tasks
                     and cat.task_methods(task)}
    model_capabilities = set()
    model_field_keys = set(STATE_MACHINES)
    for model in models.values():
        model_capabilities.update(_value(model, "capabilities", ()) or ())
        model_field_keys.update(_value(model, "writes", ()) or ())
    for keys in required_attributes.values():
        model_field_keys.update(keys)
    # Broad catalog readgroups are possible dependencies, not proof that a model
    # actually reads every field. Do not turn them into invented model coverage.
    modeled_fields = {cat.fields_by_key[key]["id"] for key in model_field_keys
                      if key in cat.fields_by_key}
    modeled = {
        "scenarios": set(profiles), "overlays": set(overlays), "tasks": modeled_tasks,
        "sequences": modeled_sequences, "actions": model_actions,
        "capabilities": model_capabilities, "fields": modeled_fields,
        # The catalog's descriptive rules alone are not executable registrations.
        "rules": set(), "event_families": {family["id"] for family in EVENT_FAMILIES},
    }
    items = {}

    def append(kind, identifier, name, status, detail, links):
        items.setdefault(kind, []).append({"id": identifier, "name_zh": name,
                                          "status": status, "detail_zh": detail,
                                          "links": links})

    for kind in ("scenarios", "overlays", "tasks", "sequences", "actions",
                 "capabilities", "fields", "rules"):
        for identifier, record in getattr(cat, kind).items():
            evidence_id = record["key"] if kind == "fields" else identifier
            was_executed = (identifier in executed[kind] or evidence_id in executed[kind])
            status = ("executed" if was_executed else
                      "modeled" if identifier in modeled[kind] else "interface_only")
            detail = {
                "executed": "已完成扫描中有执行证据；不代表动作成功或目标满足。",
                "modeled": "已存在模型或状态契约，扫描未证明本项执行。",
                "interface_only": "仅有目录、配置或接口依赖；未取得可执行模型或运行证据。",
            }[status]
            if kind == "tasks" and status == "executed":
                detail = "扫描同时记录了智能体生成与执行此任务类型；不代表真值目标已满足。"
            if kind == "overlays" and identifier == "OV01":
                status = "deferred"
                detail = ("视觉不确定性流水线延期；叠加效果仅以作者设定的脚本感测噪声建模。"
                          + ("扫描已应用该脚本噪声，不计作视觉执行。" if was_executed else ""))
            name = record.get("label", record.get("name", identifier))
            append(kind, identifier, name, status, detail, _links(record))

    for sequence, steps in cat.steps.items():
        for step in steps:
            identity = (sequence, str(step["step"]))
            status = ("executed" if identity in executed_steps else
                      "modeled" if identity in modeled_steps else "interface_only")
            append("steps", f"{sequence}:{step['step']}",
                   f"{cat.sequences[sequence]['name']} · 第{step['step']}步", status,
                   "扫描已执行本步骤。" if status == "executed" else
                   "至少一个条件选项有动作模型，未证明执行。" if status == "modeled" else
                   "步骤仅在目录中定义，尚无可用动作模型。",
                   [sequence] + step["options"])

    for identifier, record in cat.quotes.items():
        editorial = QUOTE_STATUS[identifier]
        append("quotes", identifier, record["text"], editorial["status"],
               "【人工编辑范围判定，非运行覆盖】" + editorial["detail_zh"], [identifier])
    for family in EVENT_FAMILIES:
        status = "executed" if family["id"] in executed["event_families"] else "modeled"
        detail = family["description_zh"]
        if family["grounding"] == "proposed":
            detail += " 项目新增契约：" + family["gap_closed_zh"]
        append("event_families", family["id"], family["name_zh"], status,
               detail, list(family["refs"]))

    summary = {}
    for kind, rows in items.items():
        summary[kind] = {"total": len(rows)}
        for status in ("executed", "modeled", "interface_only", "deferred"):
            summary[kind][status] = sum(row["status"] == status for row in rows)
    return {"generated_at_version": __version__, "catalog": cat.verify(),
            "items": items, "summary": summary}
