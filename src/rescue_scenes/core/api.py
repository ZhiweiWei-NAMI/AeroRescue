"""Programmable env-v0.5 entry point. No hidden-truth planner or forced demos."""
from copy import deepcopy
from collections import defaultdict
import math

from .catalog import load
from .world.semantics import authored_layer, ingest, load_upstream
from .world.demo import DEMO_AGENTS, DEMO_TRUTH_OPTIONS, compact_demo_layer
from .world.truth import generate, derive_objectives
from .world.profiles import PROFILES, OVERLAYS
from .agents.runtime import build_agents
from .sim.kernel import Simulation


_DEFAULTS = dict(seed=1, overlays=[], semantics={"source": "authored", "path": None},
                 bounds_m=[0, 0, 2000, 2000], agents="profile", disruptions=[],
                 allocator="greedy", tick_s=5, until_s=3600)
_ALLOWED = set(_DEFAULTS) | {"scenario_id", "truth_options"}


def _error(zh, en):
    raise ValueError(zh + " / " + en)


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        _error("数值无效：" + name, "invalid number: " + name)
    return value


def _config(cfg):
    if not isinstance(cfg, dict):
        _error("环境配置必须为对象", "environment config must be an object")
    unknown = set(cfg) - _ALLOWED
    if unknown:
        _error("未知配置键：" + ", ".join(sorted(unknown)), "unknown config keys: " + ", ".join(sorted(unknown)))
    result = deepcopy(_DEFAULTS)
    result.update(deepcopy(cfg))
    if "truth_options" in result and not isinstance(result["truth_options"], dict):
        _error("真值配置须为对象", "truth_options must be an object")
    if not isinstance(result.get("scenario_id"), str) or result["scenario_id"] not in PROFILES:
        _error("缺少或未知情境编号", "missing or unknown scenario_id")
    if isinstance(result["seed"], bool) or not isinstance(result["seed"], int):
        _error("种子必须为整数", "seed must be an integer")
    overlays = result["overlays"]
    if not isinstance(overlays, list) or any(not isinstance(oid, str) or oid not in OVERLAYS for oid in overlays):
        _error("叠加条件无效", "invalid overlays")
    result["overlays"] = list(dict.fromkeys(overlays))
    semantics = result["semantics"]
    if not isinstance(semantics, dict) or set(semantics) - {"source", "path"}:
        _error("语义配置键无效", "invalid semantics config keys")
    result["semantics"] = dict(_DEFAULTS["semantics"], **semantics)
    if result["semantics"]["source"] not in ("authored", "upstream", "demo"):
        _error("语义来源无效", "invalid semantics source")
    if result["semantics"]["source"] == "upstream" and not isinstance(result["semantics"]["path"], str):
        _error("上游语义需要本地路径", "upstream semantics requires a local path")
    bounds = result["bounds_m"]
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        _error("边界须含四个数值", "bounds_m requires four numbers")
    for value in bounds:
        _number(value, "bounds_m")
    if bounds[0] != 0 or bounds[1] != 0 or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        _error("边界须为[0,0,宽,高]", "bounds must be [0,0,width,height]")
    result["bounds_m"] = list(bounds)
    if result["allocator"] not in ("greedy", "bidding"):
        _error("分配器无效", "invalid allocator")
    _number(result["tick_s"], "tick_s", positive=True)
    _number(result["until_s"], "until_s")
    if result["agents"] != "profile":
        if not isinstance(result["agents"], list) or not result["agents"]:
            _error("智能体配置无效", "invalid agents")
        seen = set()
        for agent in result["agents"]:
            if not isinstance(agent, dict) or set(agent) - {"id", "kind", "start", "overrides"}:
                _error("智能体配置键无效", "invalid agent config keys")
            if not isinstance(agent.get("id"), str) or not agent["id"] or agent["id"] in seen:
                _error("智能体编号为空或重复", "empty or duplicate agent id")
            seen.add(agent["id"])
            if not isinstance(agent.get("kind"), str):
                _error("缺少智能体类型", "missing agent kind")
            if not isinstance(agent.get("overrides", {}), dict):
                _error("智能体覆盖参数须为对象", "agent overrides must be an object")
            start = agent.get("start")
            if start is not None:
                if not isinstance(start, (list, tuple)) or len(start) != 2:
                    _error("起点须为二维坐标", "start must be a 2D point")
                for value in start:
                    _number(value, "start")
                if not (bounds[0] <= start[0] <= bounds[2] and bounds[1] <= start[1] <= bounds[3]):
                    _error("起点超出区域", "start is outside bounds")
    if not isinstance(result["disruptions"], list):
        _error("扰动须为列表", "disruptions must be a list")
    valid_disruptions = {"aftershock_reblock", "link_outage", "low_battery", "service_failure", "target_moves", "equipment_fault", "wrong_evidence", "fire_spread", "water_rise", "cancel_task"}
    for disruption in result["disruptions"]:
        if not isinstance(disruption, dict) or set(disruption) - {"t", "kind", "target", "params"}:
            _error("扰动配置键无效", "invalid disruption config keys")
        if not isinstance(disruption.get("kind"), str) or disruption["kind"] not in valid_disruptions or not isinstance(disruption.get("params", {}), dict):
            _error("扰动配置无效", "invalid disruption")
        _number(disruption.get("t", 0), "disruption.t")
    return result


def _record(value):
    return value.to_dict() if hasattr(value, "to_dict") else deepcopy(value)


def _geometry(fields, default=None):
    if isinstance(fields.get("entity.geometry"), dict):
        return deepcopy(fields["entity.geometry"])
    for key in ("entity.footprint", "road.centerline"):
        value = fields.get(key)
        if isinstance(value, dict) and "coords" in value:
            return deepcopy(value)
        if isinstance(value, (list, tuple)) and value:
            return dict(type="polyline" if key.startswith("road") else "polygon", coords=deepcopy(value))
    pose = fields.get("entity.pose", fields.get("human.pose"))
    if isinstance(pose, (list, tuple)) and len(pose) >= 2:
        return dict(type="point", coords=[list(pose[:2])])
    return deepcopy(default or {"type": "point", "coords": []})


def _style(cls, fields, belief=False):
    damage = fields.get("building.visual_damage", "unknown")
    return dict(damage=damage, state=fields.get("obstacle.state", fields.get("road.entry_state", fields.get("human.mobility"))),
                uncertain=belief, label_source="belief" if belief else "truth", class_name=cls)


class Environment:
    def __init__(self, sim, config, semantic_doc):
        self.sim = sim
        self.config = deepcopy(config)
        self._semantic_doc = deepcopy(semantic_doc)

    @classmethod
    def from_config(cls, cfg):
        config = _config(cfg)
        semantics = config["semantics"]
        options = deepcopy(config.get("truth_options", {}))
        if semantics["source"] == "demo":
            # Compact authored rescue demo: pinned hidden truth + capable responders (labelled).
            doc, options = compact_demo_layer(), deepcopy(DEMO_TRUTH_OPTIONS)
            if config["agents"] == "profile":
                config["agents"] = deepcopy(DEMO_AGENTS)
        else:
            doc = (load_upstream(semantics["path"]) if semantics["source"] == "upstream" else
                   authored_layer(config["scenario_id"], config["seed"], bounds=tuple(config["bounds_m"])))
        layer = ingest(doc)
        if semantics["source"] in ("upstream", "demo"):
            config["bounds_m"] = list(layer.bounds)
        world = generate(doc, config["scenario_id"], config["seed"], overlays=config["overlays"], options=options)
        if not world.objectives:
            world.objectives = derive_objectives(world)
        agents = build_agents(config, world, load())
        sim = Simulation(world, agents, config)
        env = cls(sim, config, doc)
        sim.emit(layer="upstream", family="upstream.semantics_ingested", kind="upstream.semantics_ingested",
                 grounding="proposed", refs=[config["scenario_id"]], subject="semantic-layer", after=dict(source=layer.source_kind, entities=len(layer.entities), missing=layer.missing, warnings=layer.warnings),
                 visible_to=[a.id for a in agents], scripted=layer.source_kind == "authored",
                 text_zh="载入显式语义标注；未运行图像分割或视觉感知。")
        # Emit initial writes grouped by their real provenance, not catalog-wide lists.
        grouped = defaultdict(list)
        for entity in world.entities.values():
            for key, value in entity.get("fields", {}).items():
                cause = entity.get("provenance", {}).get(key, "authored")
                grouped[str(cause)].append(dict(entity=entity["id"], key=key, value=deepcopy(value)))
        for cause, writes in grouped.items():
            fids = [sim.catalog.fields_by_key[w["key"]]["id"] for w in writes if w["key"] in sim.catalog.fields_by_key]
            refs = ([cause] if cause in sim.catalog.rules or cause in sim.catalog.actions else []) + fids
            sim.emit(layer="world", family="gen.rule", kind="gen.rule", refs=list(dict.fromkeys(refs)),
                     subject=cause, after=dict(rule=cause, count=len(writes), writes=writes),
                     visible_to=["*truth*"], backend_mode="stub", scripted=True,
                     text_zh=f"按 {cause} 初始化真值字段；不生成智能体任务。")
        for oid in config["overlays"]:
            sim.emit(layer="world", family="overlay.applied", kind="overlay.applied", refs=[oid], subject=oid,
                     after=deepcopy(OVERLAYS[oid]), visible_to=["*truth*"], scripted=True,
                     text_zh=f"应用目录叠加条件 {oid} 的作者模型参数。")
        for objective in world.objectives:
            obj = _record(objective)
            sim.emit(layer="world", family="objective.declared", kind="objective.declared", grounding="proposed",
                     refs=obj.get("refs", []), subject=obj["id"], after=obj,
                     visible_to=[a.id for a in agents if a.kind == "coordinator"] if obj.get("public") else ["*truth*"],
                     text_zh="声明世界目标；它不是任务，不直接触发分配。")
        sim.start()
        sim.evaluator.update()
        return env

    def step(self, n=1):
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            _error("步数须为非负整数", "step count must be a nonnegative integer")
        events = []
        for _ in range(n):
            events.extend(event.to_dict() for event in self.sim.step())
        return events

    def run(self, until_s):
        return [event.to_dict() for event in self.sim.run(until_s)]

    def inject(self, d):
        first = len(self.sim.events)
        self.sim.inject(d)
        return [event.to_dict() for event in self.sim.events[first:]]

    def events(self, since_seq=0, layer=None):
        if isinstance(since_seq, bool) or not isinstance(since_seq, int) or since_seq < 0:
            _error("事件序号无效", "invalid event sequence")
        return [event.to_dict() for event in self.sim.events if event.seq > since_seq and (layer is None or event.layer == layer)]

    def tasks(self):
        records = []
        for agent in sorted(self.sim.agents.values(), key=lambda a: a.id):
            for task in agent.tasks.values():
                row = _record(task)
                row["holder"] = agent.id
                index = self.sim.tasks_index.setdefault(row["id"], dict(creator=row.get("creator"), type=row.get("type"), history=[]))
                index["creator"], index["type"] = row.get("creator"), row.get("type")
                row["global_index"] = deepcopy(index)
                records.append(row)
        return records

    def allocation(self):
        result = []
        for event in self.sim.events:
            if event.family == "allocation.decision":
                row = deepcopy(event.after) if isinstance(event.after, dict) else {"decision": deepcopy(event.after)}
                row.update(t=event.t, seq=event.seq, actor=event.actor, task_id=event.task or row.get("task_id"), text_zh=event.text_zh)
                result.append(row)
        return result

    def objectives(self):
        return [dict(_record(obj), **deepcopy(self.sim.evaluator.status.get(_record(obj)["id"], {}))) for obj in self.sim.world.objectives]

    def ledger(self):
        return self.sim.ledger.to_dict()

    def snapshot(self, view="truth"):
        if not isinstance(view, str):
            _error("视角必须为字符串", "view must be a string")
        sim = self.sim
        if view == "central":
            owner = next((a for a in sorted(sim.agents.values(), key=lambda a: a.id) if a.kind == "coordinator"), None)
            if owner is None:
                _error("没有中心协调智能体", "no coordinator agent")
        elif view.startswith("agent:"):
            owner = sim.agents.get(view[6:])
            if owner is None:
                _error("未知智能体视角", "unknown agent view")
        elif view == "truth":
            owner = None
        else:
            _error("未知视角", "unknown view")
        entities, grouped = [], {}
        if owner is None:
            for entity in sim.world.entities.values():
                fields = deepcopy(entity.get("fields", {}))
                entities.append(dict(id=entity["id"], cls=entity["cls"], geometry=deepcopy(entity["geometry"]),
                                     label_zh=fields.get("entity.id", entity["id"]), fields=fields,
                                     style=_style(entity["cls"], fields)))
            graph = deepcopy(sim.world.graph)
            knowledge = {}
        else:
            grouped = owner.belief.entities()
            # Everything, including public geometry, comes from this holder's copied prior/evidence.
            for eid, entries in grouped.items():
                fields = {key: deepcopy(entry["value"]) for key, entry in entries.items()}
                cls_name = fields.get("entity.kind", next((prefix for prefix in ("human", "obstacle", "building", "platform", "hazard", "item") if any(k.startswith(prefix + ".") for k in fields)), "unknown"))
                if cls_name == "unknown" and not any(k in fields for k in ("entity.geometry", "entity.pose", "entity.footprint", "road.centerline")):
                    continue
                entities.append(dict(id=eid, cls=cls_name, geometry=_geometry(fields), label_zh=eid,
                                     fields=deepcopy(entries), style=_style(cls_name, fields, belief=True)))
            graph = self._known_graph(grouped)
            knowledge = owner.belief.to_view()
        agent_rows = []
        for agent in sorted(sim.agents.values(), key=lambda a: a.id):
            if owner is None or owner.id == agent.id:
                values = {key: sim.world.get(agent.platform_eid, key) for key in ("entity.pose", "platform.battery_remaining_Wh", "platform.battery_capacity_Wh", "platform.motion_state", "platform.health_state")}
                current = agent.tasks.get(agent.current) if agent.current else None
                queue = list(getattr(agent, "queue", []))
                knows = len(agent.belief.to_view().get("entries", []))
            else:
                entries = grouped.get(agent.platform_eid, grouped.get(agent.id, {}))
                values = {key: entry["value"] for key, entry in entries.items()}
                report = values.get("belief.team_estimates", {})
                if isinstance(report, dict):
                    for name, key in (("pose", "entity.pose"), ("battery_Wh", "platform.battery_remaining_Wh"),
                                      ("battery_max_Wh", "platform.battery_capacity_Wh"), ("health", "platform.health_state"),
                                      ("motion_state", "platform.motion_state")):
                        if name in report:
                            values[key] = deepcopy(report[name])
                current = report.get("current_task") if isinstance(report, dict) else None
                if not current:
                    current = next((task for task in owner.tasks.values() if task.get("assigned_to") == agent.id and task.get("status") in ("accepted", "executing", "blocked", "suspended")), None)
                queue, knows = list(report.get("queue", [])) if isinstance(report, dict) else [], None
            current_summary = ({key: current.get(key) for key in ("id", "type", "name_zh", "status", "rev")} if current else None)
            agent_rows.append(dict(id=agent.id, kind=agent.kind, label_zh=agent.spec.get("label_zh", agent.kind),
                                   pose=values.get("entity.pose"), battery_Wh=values.get("platform.battery_remaining_Wh"),
                                   battery_max_Wh=values.get("platform.battery_capacity_Wh", agent.spec.get("battery_Wh")),
                                   motion_state=values.get("platform.motion_state", "unknown"), health=values.get("platform.health_state", "unknown"),
                                   capabilities=list(agent.spec.get("capabilities", [])), current_task=current_summary, queue=queue, knows=knows))
        if owner is None:
            objectives = self.objectives()
        else:
            entry = owner.belief.get("mission", "public_objectives") or owner.belief.get(owner.id, "belief.current_goal_context")
            objectives = deepcopy(entry["value"]) if entry and isinstance(entry.get("value"), list) else []
        return dict(t=sim.t, view=view, map=dict(bounds=list(self.config["bounds_m"]), entities=entities, graph=graph),
                    agents=agent_rows, knowledge=knowledge, objectives=objectives)

    @staticmethod
    def _known_graph(grouped):
        nodes, edges = {}, {}
        for eid, entries in grouped.items():
            values = {k: v["value"] for k, v in entries.items()}
            centerline = values.get("road.centerline")
            if isinstance(centerline, dict):
                centerline = centerline.get("coords")
            a, b = values.get("road.from_node"), values.get("road.to_node")
            if a and b and isinstance(centerline, (list, tuple)) and len(centerline) >= 2:
                nodes[a], nodes[b] = list(centerline[0]), list(centerline[-1])
                edges[eid] = dict(a=a, b=b, road_id=eid,
                                  length_m=sum(math.dist(x, y) for x, y in zip(centerline, centerline[1:])))
        return dict(nodes=nodes, edges=edges)


# Configurations supply conditions, never task/action scripts or fabricated coverage.
# The named wrapper keeps display metadata outside the strict env-config schema.
DEMO_CONFIGS = [
    dict(id=f"demo-{sid.lower()}", title_zh=profile["name_zh"],
         config=dict(scenario_id=sid, seed=10 + index, agents="profile", tick_s=5, until_s=600))
    for index, (sid, profile) in enumerate(sorted(PROFILES.items()))
]


def _stress_fleet(scenario, overrides):
    """Public hardware/availability parameters, not pre-seeded tasks or beliefs."""
    result, counts = [], defaultdict(int)
    for item in PROFILES[scenario]["agent_mix"]:
        for _ in range(item["count"]):
            kind = item["kind"]
            counts[kind] += 1
            result.append(dict(id=f"{kind}-{counts[kind]}", kind=kind,
                               start=[100 + len(result) * 5, 100],
                               overrides=deepcopy(overrides.get(kind, {}))))
    return result


def _stress(name, title, scenario, overlays, disruptions, agents="profile", until=1200):
    return dict(id=name, title_zh=title,
                config=dict(scenario_id=scenario, seed=83 + len(name), overlays=overlays,
                            agents=agents, disruptions=disruptions, tick_s=5, until_s=until))


DEMO_CONFIGS += [
    _stress("stress-access-reassign", "通路受阻、承诺冲突与重新分配", "SC01", ["OV06", "OV07"],
            [dict(t=110, kind="aftershock_reblock", params=dict(clear_width_m=.2, volume_m3=6, work_remaining=240)),
             dict(t=160, kind="low_battery", target="ground_vehicle-1", params=dict(remaining_Wh=2))]),
    _stress("stress-comms-retry", "消息丢失、断联排队与重试", "SC14", ["OV02", "OV03"],
            [dict(t=250, kind="link_outage", params=dict(duration_s=170))]),
    _stress("stress-partial-charge", "部分搜索、低电挂起与充电恢复", "SC02", ["OV01", "OV05"],
            [dict(t=250, kind="low_battery", target="uav-1", params={})],
            agents=_stress_fleet("SC02", {"uav": {"battery_Wh": 60}, "robot_dog": {"battery_Wh": 120}}), until=1500),
    _stress("stress-release-receipt", "释放机构故障、重试与未签收", "SC03", ["OV05", "OV08"],
            [dict(t=320, kind="target_moves", params=dict(offset=[100, 40])),
             dict(t=160, kind="equipment_fault", target="uav-1", params=dict(equipment_kind="release", recoverable=True))],
            agents=_stress_fleet("SC03", {"uav": {"release_jam_rate": .75}}), until=1500),
    _stress("stress-compute-migrate", "内存准入、远端服务故障与迁移", "SC16", ["OV02", "OV04"],
            [dict(t=300, kind="service_failure", target="compute_node-1", params=dict(duration_s=120, recoverable=True)),
             dict(t=420, kind="link_outage", params=dict(duration_s=90))],
            agents=_stress_fleet("SC16", {"uav": {"compute": {"memory_capacity_bytes": 16 * 1024 * 1024}},
                                           "robot_dog": {"compute": {"memory_capacity_bytes": 16 * 1024 * 1024}}}), until=1200),
    _stress("stress-stale-conflict", "迟到版本隔离、冲突证据与重开", "SC08", ["OV03", "OV08"],
            [dict(t=280, kind="wrong_evidence", target="uav-1", params=dict(key="building.visual_damage", value="minor")),
             dict(t=500, kind="target_moves", params=dict(offset=[30, 15]))]),
    _stress("stress-specialist-cancel", "专业队不可用与任务取消", "SC06", ["OV06", "OV07"],
            [dict(t=210, kind="cancel_task", params={}),
             dict(t=240, kind="equipment_fault", target="specialist_team-1", params=dict(recoverable=False)),
             dict(t=360, kind="fire_spread", params=dict(intensity_delta=.3))],
            agents=_stress_fleet("SC06", {"specialist_team": {"available": False}})),
]
# Compact authored rescue demo (world.demo): pinned hidden truth, agents must discover and act.
DEMO_CONFIGS.insert(0, dict(id="rescue-demo", title_zh="紧凑救援演示（作者设定世界）：搜索→解困→装载→转运→接收确认",
                            config=dict(scenario_id="SC01", seed=1, agents="profile", tick_s=5, until_s=2400,
                                        semantics={"source": "demo"})))


def coverage_sweep(configs=None):
    """Execute agent loops and aggregate ONLY observable execution records.

    No action is submitted here, no task is fabricated, and selecting a method
    alone never counts its sequence/steps as executed. Missing branches stay missing.
    """
    catalog = load()
    buckets = {key: set() for key in ("scenarios", "overlays", "tasks_generated", "tasks_executed", "sequences", "steps", "actions_terminal", "capabilities_used", "fields_written", "event_families", "rules")}
    branches, runs = defaultdict(set), []
    selected = DEMO_CONFIGS if configs is None else configs
    for index, named in enumerate(selected):
        cfg = named.get("config", named)
        config_id = named.get("id", f"run-{index+1}")
        if "config" not in named:
            cfg = {k: v for k, v in cfg.items() if k not in ("id", "title_zh", "name_zh", "name")}
        env = Environment.from_config(cfg)
        # Avoid serializing the entire event log and global task histories twice.
        env.sim.run(env.config["until_s"])
        records = [row for agent in env.sim.agents.values() for row in agent.tasks.values()]
        types = {row["id"]: row["type"] for row in records}
        executed_task_ids = set()
        run_events = env.sim.events
        evidence_events, evidence_seen = [], set()
        # Even scenario selection is witnessed by the actual semantics-ingested event.
        for emitted in run_events:
            event = vars(emitted)
            family, kind, after = event["family"], event["kind"], event.get("after")
            refs = event.get("refs", [])
            is_task_status = event.get("field") == "task.status"
            buckets["event_families"].add(family)
            buckets["scenarios"].update(ref for ref in refs if ref in catalog.scenarios)
            buckets["rules"].update(ref for ref in refs if ref in catalog.rules)
            # Keep first real witnesses, not all repeated telemetry payloads. This
            # also lets trace.build verify rule refs without rerunning a sweep.
            witness = (family, kind, tuple(refs), event.get("field"), after if isinstance(after, str) else None)
            if witness not in evidence_seen:
                evidence_seen.add(witness)
                evidence_events.append({key: deepcopy(event[key]) for key in ("seq", "t", "layer", "family", "kind", "refs", "subject", "field", "actor", "task")})
            if family == "overlay.applied":
                buckets["overlays"].update(ref for ref in refs if ref in catalog.overlays)
            if is_task_status and after == "proposed":
                # A fired rule whose A050 was rejected did not generate a record.
                tid = event.get("task") or event.get("subject")
                if tid in types:
                    buckets["tasks_generated"].add(types[tid])
            if kind == "action.status" and after in ("succeeded", "partial", "failed", "canceled", "rejected"):
                buckets["actions_terminal"].update(ref for ref in refs if ref in catalog.actions)
            if kind == "action.status" and after == "accepted":
                buckets["capabilities_used"].update(ref for ref in refs if ref in catalog.capabilities)
            if event.get("field") in catalog.fields_by_key:
                buckets["fields_written"].add(catalog.fields_by_key[event["field"]]["id"])
            if family == "gen.rule" and isinstance(after, dict):
                for write in after.get("writes", []):
                    if write.get("key") in catalog.fields_by_key:
                        buckets["fields_written"].add(catalog.fields_by_key[write["key"]]["id"])
                buckets["rules"].update(ref for ref in refs if ref in catalog.rules)
            if kind in ("sequence.step", "method.step") and isinstance(after, dict):
                seq = after.get("seq_id", after.get("sequence"))
                step = after.get("step")
                if seq in catalog.sequences and step is not None:
                    buckets["sequences"].add(seq)
                    buckets["steps"].add((seq, step))
                    tid = event.get("task")
                    if tid in types:
                        executed_task_ids.add(tid)
                        buckets["tasks_executed"].add(types[tid])
            if family == "branch.trigger":
                tid = event.get("task")
                branch_ids = [ref for ref in refs if ref in catalog.sequences]
                if tid:
                    branches[tid].update(branch_ids or [kind])
            if is_task_status and after in ("blocked", "failed", "suspended", "canceled", "completed"):
                tid = event.get("task") or event.get("subject")
                branches[tid].add(after)
        runs.append(dict(config_id=config_id, events=evidence_events, event_count=len(run_events),
                         events_retained="first_real_witness_per_coverage_signature",
                         tasks=[{key: deepcopy(row.get(key)) for key in ("id", "type", "holder", "rev", "status")} for row in records],
                         objectives=env.objectives(),
                         tasks_executed=sorted({types[tid] for tid in executed_task_ids})))
    result = {key: sorted(values) for key, values in buckets.items()}
    result.update(branches={key: sorted(values) for key, values in sorted(branches.items())}, runs=runs)
    return result
