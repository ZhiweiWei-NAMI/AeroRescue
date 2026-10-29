"""Truth-only E01 evaluation, deliberately separate from task completion."""
from copy import deepcopy


def _dict(value):
    return value.to_dict() if hasattr(value, "to_dict") else value


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


class Evaluator:
    def __init__(self, sim):
        self.sim = sim
        self.status = {}
        self.metrics = {}
        self._event_cursor = 0
        self._assessed = set()
        self._found = set()
        self._false_reports = set()
        self._extricated = set()
        self._completion_times = {}
        self._seen_results = set()
        self._violation_count = 0
        self._initial_trapped = {eid for eid, e in sim.world.entities.items()
                                 if e["cls"] == "human" and sim.world.get(eid, "human.mobility") == "trapped"}

    def _read_results(self):
        world = self.sim.world
        new_events = self.sim.events[self._event_cursor:]
        self._event_cursor = len(self.sim.events)
        self._violation_count += sum(bool(event.contract_violation) for event in new_events)
        observations = [event.after for event in new_events
                        if event.kind == "camera_image.detection" and event.backend_mode == "camera_image"]
        completed = {event.action_instance for event in new_events if event.kind == "action.result" and event.action_instance}
        for iid in completed:
            record = self.sim.instances.get(iid, {})
            result = record.get("result")
            if iid in self._seen_results or result is None:
                continue
            self._seen_results.add(iid)
            observations.extend(result.observations)
        for observation in observations:
            obs = _dict(observation)
            values = obs.get("values", {})
            for eid in obs.get("about", []):
                entity = world.entities.get(eid)
                reported = values.get(eid, values)
                if not isinstance(reported, dict):
                    continue
                if entity and entity["cls"] == "building":
                    if any(key in reported and reported[key] not in (None, "unknown")
                           for key in ("building.visual_damage", "building.structural_condition")):
                        self._assessed.add(eid)
                if entity and entity["cls"] == "human":
                    if obs.get("source") != "camera_image" and any(
                            key.startswith("human.") or key == "entity.pose" for key in reported):
                        self._found.add(eid)
                elif eid.startswith(("human-", "person-", "false-")):
                    self._false_reports.add(eid)
            # Authored search reports can describe individually detected people, never misses.
            if obs.get("source") == "camera_image":
                continue
            for key in ("detections", "detected", "humans", "detected_ids"):
                for detection in values.get(key, []) if isinstance(values.get(key), list) else []:
                    eid = detection.get("id") if isinstance(detection, dict) else detection
                    if eid in world.entities and world.entities[eid]["cls"] == "human":
                        self._found.add(eid)
                    elif eid:
                        self._false_reports.add(str(eid))

    def _objective(self, objective):
        world = self.sim.world
        kind, target = objective.get("kind"), objective.get("target")
        desired = objective.get("desired", {})
        if isinstance(target, list):
            targets = target
        elif target in world.entities:
            targets = [target]
        else:
            targets = []
        if kind == "assess_all_buildings":
            buildings = targets or [eid for eid, entity in world.entities.items() if entity["cls"] == "building"]
            progress = len(self._assessed.intersection(buildings)) / len(buildings) if buildings else 1.0
            return progress, "已取得外观评估证据的建筑比例；不代表室内搜索。"
        if kind == "find_person":
            progress = float(bool(targets) and all(eid in self._found for eid in targets))
            return progress, "人员是否实际出现在已执行传感模型的检测证据中。"
        # isolate_hazard 的 desired 只是声明本 kind 的谓词（hazard.exclusion_zone: True），
        # 其真值是"已建立非空隔离区"（A063 写入 polygon），不能按 desired 字面做 == 比较。
        if kind == "isolate_hazard":
            return float(bool(targets) and all(bool(world.get(e, "hazard.exclusion_zone")) for e in targets)), "核对危险区隔离设置。"
        if desired and targets:
            checks = []
            for eid in targets:
                for key, expected in desired.items():
                    actual = world.get(eid, key)
                    if key == "human.support_received":
                        expected_values = expected if isinstance(expected, list) else [expected]
                        checks.append(all(value in (actual or []) for value in expected_values))
                    else:
                        checks.append(actual in expected if isinstance(expected, list) else actual == expected)
            return _mean(float(c) for c in checks), "根据目标声明的真值字段逐项核对。"
        if kind == "extricate":
            return float(bool(targets) and all(world.get(e, "human.mobility") in ("immobile", "limited", "mobile") for e in targets)), "解困只核对不再受困，不等于医疗或移交。"
        if kind == "aid":
            return float(bool(targets) and all("treated" in (world.get(e, "human.support_received", []) or []) for e in targets)), "核对已实施支持，不以发现或接触代替。"
        if kind == "handover":
            return float(bool(targets) and all(world.get(e, "human.handover_status") == "acknowledged" for e in targets)), "只有接收方确认的移交才满足目标。"
        if kind == "deliver_supplies":
            return float(bool(targets) and all(world.get(e, "item.state") == "received" for e in targets)), "释放、投放和接触不等于物资签收。"
        if kind == "restore_access":
            return float(bool(targets) and all(world.get(e, "road.entry_state") == "open" for e in targets)), "按道路真值判断通行恢复。"
        if kind == "restore_link":
            links = list(self.sim.network.links.values())
            if target:
                links = [v for k, v in self.sim.network.links.items() if str(target) in str(k)]
            return float(bool(links) and all(v.get("state", v.get("link.state")) in ("reachable", "up", "good", "connected") for v in links)), "按实际链路状态核对，不把发送当作送达。"
        if kind == "suppress_fire":
            return float(bool(targets) and all(float(world.get(e, "hazard.fire_intensity", 1)) <= .05 for e in targets)), "按火强真值核对；后续复燃可使目标重新不满足。"
        if kind == "recover_from_water":
            return float(bool(targets) and all(world.get(e, "human.location_context") not in (None, "in_water", "in water", "water") for e in targets)), "脱离水域不等于完成后续移交。"
        return 0.0, "没有可核验的真值谓词，不推定成功。"

    def update(self):
        sim, world = self.sim, self.sim.world
        self._read_results()
        for objective in world.objectives:
            obj = _dict(objective)
            progress, explanation = self._objective(obj)
            progress = max(0.0, min(1.0, float(progress)))
            value = dict(id=obj["id"], satisfied=progress >= 1.0, progress=progress, evidence_zh=explanation)
            previous = self.status.get(obj["id"])
            if value != previous:
                self.status[obj["id"]] = value
                if value["satisfied"] and obj["id"] not in self._completion_times:
                    self._completion_times[obj["id"]] = sim.t
                sim.emit(layer="evaluation", family="objective.status", kind="objective.status", grounding="proposed",
                         refs=["E01"] + list(obj.get("refs", [])), subject=obj["id"], before=previous, after=value,
                         visible_to=["*truth*"], text_zh=f"世界目标 {obj['id']} 的真值满足度已更新。")
        buildings = [eid for eid, entity in world.entities.items() if entity["cls"] == "building"]
        # Casualty denominators: generated ordinary residents (attr.population_role) are not casualties.
        humans = [eid for eid, entity in world.entities.items() if entity["cls"] == "human" and world.get(eid, "attr.population_role") != "ordinary"]
        items = [eid for eid, entity in world.entities.items() if entity["cls"] == "item"]
        for eid in self._initial_trapped:
            if world.get(eid, "human.mobility") != "trapped":
                self._extricated.add(eid)
        ledger = sim.ledger.totals
        metrics = {
            "evaluation.assessment_coverage": len(self._assessed.intersection(buildings)) / len(buildings) if buildings else 0.0,
            "evaluation.search_coverage": _mean(min(1.0, max(0.0, float(world.get(e, "building.search_coverage_truth", 0) or 0))) for e in buildings),
            "evaluation.detection_counts": dict(found=len(self._found.intersection(humans)), missed=len(set(humans) - self._found), false_positive=len(self._false_reports), denominator=len(humans)),
            "evaluation.support_counts": dict(extricated=len(self._extricated), treated=sum("treated" in (world.get(e, "human.support_received", []) or []) for e in humans),
                                              handover=sum(world.get(e, "human.handover_status") == "acknowledged" for e in humans),
                                              supplies_received=sum(world.get(e, "item.state") == "received" for e in items)),
            "evaluation.response_time_s": {obj["id"]: self._completion_times.get(obj["id"]) for obj in map(_dict, world.objectives)},
            "evaluation.energy_Wh": ledger["energy_Wh"],
            "evaluation.communication_bytes": dict(transmitted=ledger["bytes"]),
            "evaluation.compute_cost": dict(cpu_s=ledger["cpu_s"]),
            "evaluation.constraint_violations": dict(contract=self._violation_count,
                                                     depleted_platforms=sum(float(world.get(a.platform_eid, "platform.battery_remaining_Wh", 1) or 0) <= 0 for a in sim.agents.values() if a.spec.get("mobility") != "none")),
        }
        for key, value in metrics.items():
            if key not in self.metrics or self.metrics[key] != value:
                before = deepcopy(self.metrics.get(key))
                self.metrics[key] = deepcopy(value)
                field = sim.catalog.fields_by_key.get(key, {})
                sim.emit(layer="evaluation", family="eval.update", kind=key, refs=["E01", field.get("id", "E01")],
                         subject="evaluation", field=key, before=before, after=value, visible_to=["*truth*"],
                         text_zh="独立评估器更新真值指标；任务完成不直接计为救援成功。")

    def to_dict(self):
        return dict(objectives=deepcopy(list(self.status.values())), metrics=deepcopy(self.metrics))
