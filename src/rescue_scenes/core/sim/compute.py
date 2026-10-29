"""Stateful abstract compute nodes, services, queues and network-gated offload.

Action plans pay orchestration time/energy only (zero bytes and CPU). The network
owns all transport charges. This module records actual executed CPU and its energy
once, including failed/canceled work, in the same ledger. A006's independent stub
inference action is not executed or charged here. No remote API invokes an agent:
requests, admission failures, checkpoints and results must be delivered messages.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

from rescue_scenes.core.contracts import ActionOutcome, Observation
from rescue_scenes.core.world.profiles import OVERLAYS


MIB = 1024 * 1024
TERMINAL = frozenset(("completed", "failed", "canceled"))


class Compute:
    backend_mode = "compute_model"

    def __init__(self, sim):
        self.sim = sim
        self.options = {"memory_capacity_factor": 1.0, "cpu_capacity_factor": 1.0, "service_failures": []}
        for oid in sim.config.get("overlays", []):
            effect = OVERLAYS[oid]["effects"].get("compute", {})
            for key in ("memory_capacity_factor", "cpu_capacity_factor"):
                self.options[key] *= float(effect.get(key, 1.0))
            self.options["service_failures"].extend(deepcopy(effect.get("service_failures", [])))
        self.nodes = {}
        self.jobs = {}
        self.caches = {}
        self.known_services = {aid: {} for aid in sim.agents}
        self.job_views = {aid: {} for aid in sim.agents}
        self.results = {aid: {} for aid in sim.agents}
        self.isolated_results = {aid: {} for aid in sim.agents}
        self.selections = {}
        self.latest_job = {}
        self._request_timeouts = set()
        self._loads = {}
        self._serial = 0
        self._load_serial = 0
        self._seen_messages = set()
        self._recoveries = []
        for aid, agent in sorted(sim.agents.items()):
            spec = agent.spec
            caps = set(spec.get("capabilities", []))
            if not caps.intersection(("C13", "C35", "C36", "C37", "C49")):
                continue
            edge = spec.get("kind", getattr(agent, "kind", "")) == "compute_node"
            cfg = dict(spec.get("compute", {}))
            memory = int(cfg.get("memory_capacity_bytes", 512 * MIB if edge else 64 * MIB)
                         * self.options["memory_capacity_factor"])
            node = {"id": aid, "host_agent": aid,
                    "cpu_capacity": max(0.001, float(cfg.get("cpu_capacity", 4.0 if edge else 1.0))
                                        * self.options["cpu_capacity_factor"]),
                    "memory_capacity_bytes": max(0, memory), "memory_used_bytes": 0,
                    "cpu_allocated": 0.0, "max_concurrency": max(1, int(cfg.get("max_concurrency", 1))),
                    "availability": "available", "services": {}, "queue": [], "running": [],
                    "storage_capacity_bytes": int(cfg.get("storage_capacity_bytes", 256 * MIB)),
                    "energy_Wh": 0.0}
            self.nodes[aid] = node
            definitions = cfg.get("services", {"inference": {"model": "authored_stub_v1", "memory_bytes": 16 * MIB}})
            if isinstance(definitions, list):
                definitions = {s.get("id", "inference"): s for s in definitions}
            for sid, definition in definitions.items():
                self._new_service(node, sid, definition)
                if definition.get("autoload", True):
                    self._load(node, sid)
        for failure in self.options["service_failures"]:
            sim.schedule(max(0.0, float(failure["t"]) - sim.t), self._overlay_failure, deepcopy(failure))

    def _overlay_failure(self, failure):
        target = failure.get("target", "*")
        services = [(nid, sid) for nid, node in sorted(self.nodes.items()) for sid in sorted(node["services"])]
        if target == "*":
            if not services:
                return
            nid, sid = self.sim.rng("compute.overlay_failure", str(failure)).choice(services)
        elif self._node(target):
            node = self._node(target)
            if not node["services"]:
                return
            nid, sid = node["id"], sorted(node["services"])[0]
        else:
            matching = [(n, s) for n, s in services if s == target or n + ":" + s == target]
            if not matching:
                return
            nid, sid = matching[0]
        params = {"service_id": sid, "recoverable": failure.get("recover_after_s") is not None}
        if failure.get("recover_after_s") is not None:
            params["duration_s"] = failure["recover_after_s"]
        self.inject({"kind": "service_failure", "target": nid, "params": params})

    def _emit(self, subject, key, before, after, refs, visible=None, text="计算状态更新", **kw):
        machines = ("job.status", "service.status", "compute.availability", "cache.eviction_state")
        return self.sim.emit(layer="compute", family="sm:" + key if key in machines else "field.write",
                             kind=key, grounding="catalog", refs=list(refs), subject=subject, field=key,
                             before=before, after=deepcopy(after), visible_to=visible or ["*truth*"],
                             backend_mode=self.backend_mode, scripted=False, text_zh=text, **kw)

    def _branch(self, subject, branch, actor, text):
        self.sim.emit(layer="compute", family="branch.trigger", kind="branch.trigger", grounding="catalog",
                      refs=[branch], subject=subject, actor=actor, visible_to=[actor],
                      backend_mode=self.backend_mode, scripted=False, text_zh=text)

    def _node(self, node_id):
        if node_id in self.nodes:
            return self.nodes[node_id]
        for node in self.nodes.values():
            if node["host_agent"] == node_id:
                return node
        return None

    def _service_status(self, node, sid, status):
        service = node["services"][sid]
        before = service["status"]
        if before == status:
            return
        service["status"] = status
        self._emit(node["id"] + ":" + sid, "service.status", before, status, ["F317"],
                   [node["host_agent"]], "服务状态为 " + status)

    def _availability(self, node, status):
        before = node["availability"]
        if before != status:
            node["availability"] = status
            self._emit(node["id"], "compute.availability", before, status, ["F304"],
                       [node["host_agent"]], "计算节点状态为 " + status)

    def _set_status(self, job, status, delivered=False):
        before = job["status"]
        if before == status:
            return
        job["status"] = status
        node = self._node(job["node_id"])
        local = node is not None and node["host_agent"] == job["requester"]
        visible = [job["requester"]] if local or delivered or status == "uploading" else [node["host_agent"]] if node else [job["requester"]]
        self._emit(job["id"], "job.status", before, status, ["F336"], visible,
                   "作业状态为 " + status, actor=visible[0], task=job["task_ref"][0] if job["task_ref"] else None)
        if local or delivered or status == "uploading":
            self.job_views.setdefault(job["requester"], {})[job["id"]] = self._job_public(job)

    def _new_service(self, node, sid, params):
        if sid not in node["services"]:
            node["services"][sid] = {
                "id": sid, "model": params.get("model", params.get("model_id", "authored_stub_v1")),
                "version": str(params.get("version", "1")), "status": "unloaded",
                "memory_bytes": max(0, int(params.get("memory_bytes", params.get("weights_bytes", 16 * MIB)))),
                "load_s": max(0.0, float(params.get("load_s", 2.0))), "reserved": False, "fault": None,
                "compatible_representations": list(params.get("compatible_representations", ["json", "text"]))}
        return node["services"][sid]

    def _load(self, node, sid):
        service = node["services"][sid]
        if service["status"] in ("loading", "ready", "busy"):
            return {"ok": True, "status": service["status"]}
        fault = service.get("fault")
        if (node["availability"] == "offline" or service["status"] == "draining"
                or (fault and not fault.get("recoverable"))):
            return {"ok": False, "failure_reason": "model_unavailable"}
        needed = 0 if service["reserved"] else service["memory_bytes"]
        if node["memory_used_bytes"] + needed > node["memory_capacity_bytes"]:
            self._service_status(node, sid, "failed")
            self._branch(node["id"] + ":" + sid, "Q106", node["host_agent"], "模型内存准入不足")
            return {"ok": False, "failure_reason": "memory"}
        node["memory_used_bytes"] += needed
        service["reserved"] = True
        self._service_status(node, sid, "loading")
        self._load_serial += 1
        self._loads[(node["id"], sid)] = {"id": self._load_serial, "start": self.sim.t,
                                         "due": self.sim.t + service["load_s"]}
        return {"ok": True, "status": "loading"}

    def _finish_load(self, node, sid):
        load = self._loads.pop((node["id"], sid), None)
        if load is None:
            return
        elapsed = max(0.0, min(self.sim.t, load["due"]) - load["start"])
        cpu = elapsed * min(1.0, node["cpu_capacity"])
        energy = cpu * float(self.options.get("energy_Wh_per_cpu_s", 0.002))
        node["energy_Wh"] += energy
        self.sim.ledger.add("load:%s:%s:%d" % (node["id"], sid, load["id"]), node["host_agent"],
                            "COMPUTE", {"time_s": elapsed, "energy_Wh": energy, "bytes": 0, "cpu_s": cpu})

    def _node_public(self, node):
        return {"id": node["id"], "host_agent": node["host_agent"], "cpu_capacity": node["cpu_capacity"],
                "memory_capacity_bytes": node["memory_capacity_bytes"], "memory_used_bytes": node["memory_used_bytes"],
                "availability": node["availability"], "queue_length": len(node["queue"]),
                "services": {sid: {k: deepcopy(v) for k, v in s.items() if k != "reserved"}
                             for sid, s in node["services"].items()}, "observed_s": float(self.sim.t)}

    @staticmethod
    def _job_public(job):
        keys = ("id", "requester", "node_id", "service_id", "status", "task_ref", "queue_enter_s",
                "start_s", "finish_s", "failure_reason", "cpu_s", "executed_cpu_s", "memory_request_bytes",
                "result", "checkpoint_ref")
        return {k: deepcopy(job.get(k)) for k in keys}

    def query_nodes(self, agent_id=None):
        if agent_id is None:
            return [self._node_public(n) for n in self.nodes.values()]
        known = deepcopy(self.known_services.get(agent_id, {}))
        for node in self.nodes.values():
            if node["host_agent"] == agent_id:
                known[node["id"]] = self._node_public(node)
        return list(known.values())

    def query_services(self, node_id=None, agent_id=None):
        return [{"node_id": n["id"], **deepcopy(service)} for n in self.query_nodes(agent_id)
                if node_id is None or n["id"] == node_id for service in n["services"].values()]

    def _send(self, sender, receiver, op, body, size=None, task_ref=None, correlation_id=None):
        payload = {"_compute": {"op": op, **deepcopy(body)}, "require_ack": False}
        if op == "job_result":
            job = body.get("job", {})
            payload.update(op="compute_result", job_id=body.get("job_id"), status=body.get("status"),
                           action_status="succeeded" if body.get("status") == "completed" else "failed",
                           task_ref=job.get("task_ref"), result=deepcopy(job.get("result")),
                           failure_code="none" if body.get("status") == "completed" else "model",
                           failure_reason=job.get("failure_reason", "none"), observations=[],
                           evidence_ids=list((job.get("result") or {}).get("evidence_ids", [])))
        encoded = len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
        # Encoded metadata is real traffic even for a declared zero-byte input.
        return self.sim.send_message(sender, [receiver], "result" if op.endswith("result") else "control",
                                     payload, max(encoded, int(size or 0)), task_ref=task_ref,
                                     ttl_s=float(self.options.get("message_ttl_s", 180)), correlation_id=correlation_id)

    def discover_services(self, agent_id, params=None):
        params = params or {}
        selected = params.get("node_id", params.get("target_node"))
        ids = [selected] if selected else sorted(self.nodes)
        sent = []
        for nid in ids:
            node = self._node(nid)
            if node is None:
                continue
            if node["host_agent"] == agent_id:
                self.known_services.setdefault(agent_id, {})[node["id"]] = self._node_public(node)
            else:
                sent.append(self._send(agent_id, node["host_agent"], "discover", {"node_id": node["id"]}))
        return {"ok": bool(sent or self.query_nodes(agent_id)), "pending_messages": sent,
                "known": self.query_nodes(agent_id)}

    def select_execution(self, agent_id, params=None):
        params = params or {}
        sid = params.get("service_id", "inference")
        memory = int(params.get("memory_request_bytes", params.get("memory_bytes", 1 * MIB)))
        candidates = []
        for node in self.query_nodes(agent_id):
            if params.get("remote") and node["host_agent"] == agent_id:
                continue
            if params.get("node_id") and node["id"] != params["node_id"]:
                continue
            service = node["services"].get(sid)
            if not service or service["status"] not in ("ready", "busy", "loading", "unloaded"):
                continue
            if node["availability"] == "offline" or node["memory_capacity_bytes"] - node["memory_used_bytes"] < memory:
                continue
            estimate = (node["queue_length"] + 1) * float(params.get("cpu_s", 5.0)) / max(0.001, node["cpu_capacity"])
            candidates.append((estimate, node["id"], node))
        if not candidates:
            return {"ok": False, "failure_reason": "model_unavailable", "known_only": True}
        estimate, nid, node = min(candidates, key=lambda item: (item[0], item[1]))
        decision = {"ok": True, "node_id": nid, "service_id": sid, "estimated_runtime_s": estimate,
                    "local": node["host_agent"] == agent_id, "estimate_s": node["observed_s"], "known_only": True}
        self.selections[agent_id] = deepcopy(decision)
        return decision

    def submit_job(self, agent_id, params, task_ref=None):
        self._serial += 1
        jid = "job-%06d" % self._serial
        chosen = self.selections.get(agent_id, {})
        nid = params.get("node_id", params.get("target_node", params.get("host", chosen.get("node_id", agent_id))))
        node = self._node(nid)
        if node:
            nid = node["id"]
        cpu = float(params.get("cpu_s", 5.0))
        memory = int(params.get("memory_request_bytes", params.get("memory_bytes", 1 * MIB)))
        request = float(params.get("cpu_request", node["cpu_capacity"] if node else 1.0))
        deadline = float(params.get("deadline_s", self.sim.t + 300))
        if not math.isfinite(cpu) or cpu < 0 or memory < 0 or not math.isfinite(request) or request <= 0 or not math.isfinite(deadline):
            raise ValueError("作业资源非法 / invalid job resources")
        task_ref = task_ref or params.get("_task_ref", params.get("task_ref"))
        params = {k: deepcopy(v) for k, v in params.items() if not k.startswith("_")}
        job = {"id": jid, "requester": agent_id, "node_id": nid,
               "service_id": params.get("service_id", chosen.get("service_id", "inference")), "status": "created",
               "task_ref": tuple(task_ref) if task_ref else None, "params": deepcopy(params),
               "cpu_s": cpu, "remaining_cpu_s": cpu, "executed_cpu_s": 0.0,
               "memory_request_bytes": memory, "reserved": False, "cpu_allocation": 0.0,
               "created_s": float(self.sim.t), "queue_enter_s": None, "start_s": None, "finish_s": None,
               "deadline_s": deadline, "last_s": float(self.sim.t),
               "failure_reason": "none", "result": None, "checkpoint_ref": params.get("checkpoint_ref"),
               "request_message": None, "result_message": None, "charged": False}
        self.jobs[jid] = job
        self.latest_job[agent_id] = jid
        params.setdefault("service_id", job["service_id"])
        self.job_views.setdefault(agent_id, {})[jid] = self._job_public(job)
        if node is None:
            self._fail(job, "model_unavailable", notify=False)
        elif node["host_agent"] == agent_id:
            self._admit(job, node)
        else:
            self._set_status(job, "uploading")
            job["request_message"] = self._send(agent_id, node["host_agent"], "submit",
                                                 {"job_id": jid, "node_id": nid, "params": params},
                                                 size=params.get("input_bytes", 0), task_ref=job["task_ref"], correlation_id=jid)
        return jid

    def _admit(self, job, node, delivered_params=None):
        if job["status"] not in ("created", "uploading"):
            return
        if delivered_params is not None:
            # The host consumes the delivered input copy, never the requester's
            # pre-upload object in the bookkeeping table.
            job["params"] = deepcopy(delivered_params)
            job["service_id"] = delivered_params.get("service_id", "inference")
            job["cpu_s"] = job["remaining_cpu_s"] = float(delivered_params.get("cpu_s", 5.0))
            job["memory_request_bytes"] = int(delivered_params.get("memory_request_bytes", delivered_params.get("memory_bytes", MIB)))
            job["checkpoint_ref"] = deepcopy(delivered_params.get("checkpoint_ref"))
        sid = job["service_id"]
        service = node["services"].get(sid)
        if node["availability"] == "offline" or service is None or service["status"] in ("failed", "draining"):
            self._fail(job, "model_unavailable")
            return
        checkpoint = job.get("checkpoint_ref") or {}
        if ((job["params"].get("version") is not None and str(job["params"]["version"]) != service["version"])
                or (checkpoint and (checkpoint.get("model") != service["model"] or str(checkpoint.get("version")) != service["version"]))
                or job["params"].get("representation", "json") not in service["compatible_representations"]):
            self._fail(job, "input")
            return
        weights = 0 if service["reserved"] else service["memory_bytes"]
        if node["memory_used_bytes"] + weights + job["memory_request_bytes"] > node["memory_capacity_bytes"]:
            self._branch(job["id"], "Q106", node["host_agent"], "作业内存准入不足，不能把排队当完成")
            self._fail(job, "memory")
            return
        if service["status"] == "unloaded" and not self._load(node, sid)["ok"]:
            self._fail(job, "memory")
            return
        node["memory_used_bytes"] += job["memory_request_bytes"]
        job["reserved"] = True
        job["queue_enter_s"] = float(self.sim.t)
        node["queue"].append(job["id"])
        self._set_status(job, "queued")

    def _start(self, job, node):
        service = node["services"][job["service_id"]]
        if service["status"] == "loading":
            if job["status"] == "queued":
                self._set_status(job, "loading")
            return False
        if service["status"] != "ready" or node["availability"] == "offline":
            return False
        allocation = min(float(job["params"].get("cpu_request", node["cpu_capacity"])),
                         node["cpu_capacity"] - node["cpu_allocated"])
        if allocation <= 0:
            return False
        job["start_s"] = job["last_s"] = float(self.sim.t)
        job["cpu_allocation"] = allocation
        node["cpu_allocated"] += allocation
        node["queue"].remove(job["id"])
        node["running"].append(job["id"])
        self._set_status(job, "running")
        self._service_status(node, job["service_id"], "busy")
        if node["availability"] == "available":
            self._availability(node, "busy")
        return True

    def _advance(self, job):
        if job["status"] != "running":
            return
        stop = min(float(self.sim.t), job["deadline_s"])
        elapsed = max(0.0, stop - job["last_s"])
        consumed = min(job["remaining_cpu_s"], elapsed * job["cpu_allocation"])
        node = self._node(job["node_id"])
        agent = self.sim.agents[node["host_agent"]]
        energy_limited = False
        if not agent.spec.get("mains_power", False):
            battery = self.sim.world.get(agent.platform_eid, "platform.battery_remaining_Wh", None)
            rate = float(self.options.get("energy_Wh_per_cpu_s", 0.002))
            if battery is not None and rate > 0:
                accrued = sum(j["executed_cpu_s"] * rate for j in self.jobs.values()
                              if j["node_id"] == node["id"] and not j["charged"])
                allowed = max(0.0, float(battery) - accrued) / rate
                energy_limited = allowed < consumed
                consumed = min(consumed, allowed)
        job["executed_cpu_s"] += consumed
        job["remaining_cpu_s"] = max(0.0, job["remaining_cpu_s"] - consumed)
        job["last_s"] = stop
        if energy_limited:
            self._fail(job, "model_unavailable")
        elif job["remaining_cpu_s"] <= 1e-9:
            self._complete_execution(job)
        elif self.sim.t >= job["deadline_s"]:
            self._fail(job, "timeout")

    def _charge(self, job):
        if job["charged"]:
            return
        job["charged"] = True
        node = self._node(job["node_id"])
        if node is None or job["executed_cpu_s"] <= 0:
            return
        cpu = job["executed_cpu_s"]
        energy = cpu * float(self.options.get("energy_Wh_per_cpu_s", 0.002))
        node["energy_Wh"] += energy
        self.sim.ledger.add("compute:" + job["id"], node["host_agent"], "COMPUTE",
                            {"time_s": cpu / max(0.001, job["cpu_allocation"]), "energy_Wh": energy,
                             "bytes": 0, "cpu_s": cpu}, task_ref=job["task_ref"])

    def _release(self, job):
        node = self._node(job["node_id"])
        if node is None:
            return
        if job["id"] in node["queue"]:
            node["queue"].remove(job["id"])
        if job["id"] in node["running"]:
            node["running"].remove(job["id"])
            node["cpu_allocated"] = max(0.0, node["cpu_allocated"] - job["cpu_allocation"])
        if job["reserved"]:
            node["memory_used_bytes"] = max(0, node["memory_used_bytes"] - job["memory_request_bytes"])
            job["reserved"] = False
        service = node["services"].get(job["service_id"])
        if service and service["status"] == "busy" and not any(self.jobs[j]["service_id"] == job["service_id"] for j in node["running"]):
            self._service_status(node, job["service_id"], "ready")
        if not node["running"] and node["availability"] == "busy":
            self._availability(node, "available")

    def _run_service(self, job):
        # Operates exclusively on supplied input, never on WorldTruth/hidden occupants.
        data = deepcopy(job["params"].get("input", {}))
        operation = job["params"].get("operation", "summary")
        if operation == "sum" and isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            output = {"sum": sum(data)}
        else:
            encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            output = {"input_digest": hashlib.sha256(encoded).hexdigest(), "processed_input": data}
        return {"job_id": job["id"], "task_ref": job["task_ref"], "output": output,
                "evidence_ids": list(job["params"].get("input_evidence_ids", [])),
                "backend_mode": self.backend_mode, "scripted": True}

    def _complete_execution(self, job):
        job["finish_s"] = float(self.sim.t)
        job["result"] = self._run_service(job)
        self._charge(job)
        self._release(job)
        node = self._node(job["node_id"])
        if node["host_agent"] == job["requester"]:
            self._set_status(job, "completed")
            self.results.setdefault(job["requester"], {})[job["id"]] = deepcopy(job["result"])
        else:
            self._set_status(job, "downloading")
            self._send_job_result(job, "completed")

    def _send_job_result(self, job, status):
        node = self._node(job["node_id"])
        if node is None or node["host_agent"] == job["requester"]:
            return
        job["result_message"] = self._send(node["host_agent"], job["requester"], "job_result",
                                             {"job_id": job["id"], "status": status,
                                              "job": self._job_public(job)},
                                             size=job["params"].get("output_bytes", 0),
                                             task_ref=job["task_ref"], correlation_id=job["id"])

    def _fail(self, job, reason, notify=True):
        if job["status"] in TERMINAL:
            return
        job["failure_reason"] = reason
        job["finish_s"] = float(self.sim.t)
        self._charge(job)
        self._release(job)
        self._set_status(job, "failed")
        if notify:
            self._send_job_result(job, "failed")
        if self._node(job["node_id"]) is None:
            self.job_views[job["requester"]][job["id"]] = self._job_public(job)

    def query_job(self, agent_id, params):
        jid = params if isinstance(params, str) else params.get("job_id")
        job = self.jobs.get(jid)
        if job and self._node(job["node_id"]) and self._node(job["node_id"])["host_agent"] == agent_id:
            return self._job_public(job)
        return deepcopy(self.job_views.get(agent_id, {}).get(jid, {}))

    def manage_model(self, agent_id, params):
        node = self._node(params.get("node_id", agent_id))
        if node is None:
            return {"ok": False, "failure_reason": "model_unavailable"}
        if node["host_agent"] != agent_id:
            mid = self._send(agent_id, node["host_agent"], "manage_model", {"params": params})
            return {"ok": True, "pending": True, "message_id": mid}
        sid = params.get("service_id", "inference")
        service = self._new_service(node, sid, params)
        mode = params.get("mode", params.get("operation", "load"))
        if mode in ("load", "reload", "recover"):
            if mode == "reload" and service["status"] == "ready":
                self._service_status(node, sid, "draining")
                if service["reserved"]:
                    node["memory_used_bytes"] -= service["memory_bytes"]
                    service["reserved"] = False
                self._service_status(node, sid, "unloaded")
            if params.get("version") and str(params["version"]) != service["version"] and service["status"] != "unloaded":
                return {"ok": False, "failure_reason": "input"}
            if service["status"] == "unloaded":
                service["version"] = str(params.get("version", service["version"]))
                service["model"] = params.get("model", params.get("model_id", service["model"]))
                service["memory_bytes"] = max(0, int(params.get("memory_bytes", params.get("weights_bytes", service["memory_bytes"]))))
            return self._load(node, sid)
        if mode != "unload" or service["status"] == "busy":
            return {"ok": False, "failure_reason": "model_unavailable"}
        if service["status"] == "unloaded":
            return {"ok": True, "status": "unloaded"}
        if service["status"] != "ready":
            return {"ok": False, "failure_reason": "model_unavailable"}
        self._service_status(node, sid, "draining")
        if service["reserved"]:
            node["memory_used_bytes"] -= service["memory_bytes"]
            service["reserved"] = False
        self._service_status(node, sid, "unloaded")
        return {"ok": True, "status": "unloaded"}

    def _release_cache(self, cache):
        if cache.get("memory_reserved"):
            node = self._node(cache["owner"])
            if node:
                node["memory_used_bytes"] = max(0, node["memory_used_bytes"] - cache["size_bytes"])
            cache["memory_reserved"] = False

    def manage_cache(self, agent_id, params):
        cid = params.get("cache_id", "cache:" + agent_id)
        cache = self.caches.get(cid)
        mode = params.get("mode", params.get("operation", "store"))
        if cache is not None and cache["owner"] != agent_id:
            return {"ok": False, "failure_reason": "input"}
        if mode in ("evict", "invalidate"):
            if cache is None or cache["state"] != "resident":
                return {"ok": False, "failure_reason": "input"}
            for state in (["evicting", "evicted"] if mode == "evict" else ["invalid"]):
                before = cache["state"]
                cache["state"] = state
                self._emit(cid, "cache.eviction_state", before, state, ["A046", "F358"], [agent_id], "缓存状态为 " + state)
            self._release_cache(cache)
            return {"ok": True, "cache_id": cid, "state": cache["state"]}
        if mode in ("reuse", "lookup"):
            compatible = bool(cache and cache["state"] == "resident" and cache["valid_until_s"] > self.sim.t
                              and cache["version"] == str(params.get("version", "1"))
                              and cache["representation"] == params.get("representation", "json")
                              and cache.get("prefix") == params.get("prefix")
                              and cache.get("model") == params.get("model_id"))
            if compatible:
                cache["hit_count"] += 1
            return {"ok": compatible, "cache_id": cid, "hit": compatible,
                    "data": deepcopy(cache["data"]) if compatible else None}
        if mode not in ("store", "cache") or (cache is not None and cache["state"] != "resident"):
            return {"ok": False, "failure_reason": "input"}
        encoded = len(json.dumps(params.get("data", {}), ensure_ascii=False, default=str).encode("utf-8"))
        size = max(encoded, int(params.get("size_bytes", encoded)))
        node = self._node(agent_id)
        kind = params.get("kind", "summary")
        if kind not in ("raw", "crop", "summary", "embedding", "features", "KV"):
            return {"ok": False, "failure_reason": "input"}
        old_memory = cache["size_bytes"] if cache and cache.get("memory_reserved") else 0
        if kind == "KV" and (node is None or node["memory_used_bytes"] - old_memory + size > node["memory_capacity_bytes"]):
            return {"ok": False, "failure_reason": "memory"}
        limit = node["storage_capacity_bytes"] if node else int(self.options.get("cache_capacity_bytes", 16 * MIB))
        used = sum(c["size_bytes"] for key, c in self.caches.items()
                   if key != cid and c["owner"] == agent_id and c["state"] == "resident")
        if used + size > limit:
            return {"ok": False, "failure_reason": "memory"}
        if cache:
            self._release_cache(cache)
        if kind == "KV":
            node["memory_used_bytes"] += size
        self.caches[cid] = {"id": cid, "owner": agent_id, "state": "resident", "size_bytes": size,
                            "kind": kind, "memory_reserved": kind == "KV", "prefix": params.get("prefix"), "model": params.get("model_id"),
                            "version": str(params.get("version", "1")), "representation": params.get("representation", "json"),
                            "valid_until_s": float(params.get("valid_until_s", self.sim.t + 300)),
                            "hit_count": 0, "data": deepcopy(params.get("data", {}))}
        return {"ok": True, "cache_id": cid, "state": "resident"}

    def cancel_job(self, agent_id, params):
        jid = params.get("job_id")
        job = self.jobs.get(jid)
        if job is None or agent_id not in (job["requester"], self._node(job["node_id"])["host_agent"] if self._node(job["node_id"]) else None):
            return {"ok": False, "failure_reason": "input"}
        node = self._node(job["node_id"])
        if node and node["host_agent"] != agent_id:
            mid = self._send(agent_id, node["host_agent"], "cancel", {"job_id": jid}, task_ref=job["task_ref"])
            return {"ok": True, "pending": True, "message_id": mid}
        if job["status"] in TERMINAL:
            return {"ok": True, "status": job["status"]}
        self._advance(job)
        if job["status"] in TERMINAL:
            return {"ok": True, "status": job["status"]}
        job["failure_reason"] = "canceled"
        job["finish_s"] = float(self.sim.t)
        self._charge(job)
        self._release(job)
        self._set_status(job, "canceled")
        self._send_job_result(job, "canceled")
        return {"ok": True, "status": "canceled"}

    def migrate_job(self, agent_id, params):
        job = self.jobs.get(params.get("job_id"))
        target = self._node(params.get("target_node", params.get("node_id")))
        if job is None or target is None or job["requester"] != agent_id:
            return {"ok": False, "failure_reason": "input"}
        source = self._node(job["node_id"])
        if source and source["host_agent"] != agent_id:
            mid = self._send(agent_id, source["host_agent"], "migrate", {"job_id": job["id"], "target_node": target["id"]},
                             task_ref=job["task_ref"], correlation_id=job["id"])
            return {"ok": True, "pending": True, "message_id": mid}
        return self._migrate_at_source(job, target)

    def _migrate_at_source(self, job, target):
        source = self._node(job["node_id"])
        if source is None:
            return {"ok": False, "failure_reason": "model_unavailable"}
        service = source["services"].get(job["service_id"])
        if job["status"] == "completed":
            return {"ok": False, "failure_reason": "input"}
        self._advance(job)
        if job["status"] in ("completed", "downloading"):
            return {"ok": False, "failure_reason": "input"}
        # A new attempt has a new machine/ledger identity; terminal jobs never resurrect.
        checkpoint = {"source_job": job["id"], "remaining_cpu_s": job["remaining_cpu_s"],
                      "model": service["model"] if service else None, "version": service["version"] if service else None}
        job["checkpoint_ref"] = checkpoint
        self._emit(job["id"], "job.checkpoint_ref", None, checkpoint, ["A047", "F347"],
                   [source["host_agent"]], "保存可传输的作业检查点")
        if job["status"] not in TERMINAL:
            self.cancel_job(source["host_agent"], {"job_id": job["id"]})
        params = deepcopy(job["params"])
        params.update(node_id=target["id"], cpu_s=checkpoint["remaining_cpu_s"], checkpoint_ref=checkpoint,
                      version=checkpoint["version"], restart_cpu_s=float(self.options.get("restart_cpu_s", 0.5)))
        # The source sends the checkpoint to the requester; only its delivery initiates
        # a replacement upload. This also works when the target equals the requester.
        mid = self._send(source["host_agent"], job["requester"], "migration_result",
                         {"source_job": job["id"], "params": params},
                         size=max(256, int(job["params"].get("checkpoint_bytes", 4096))),
                         task_ref=job["task_ref"], correlation_id=job["id"])
        return {"ok": True, "pending": True, "message_id": mid}

    def release_compute(self, agent_id, params):
        job = self.jobs.get(params.get("job_id"))
        if job:
            if job["requester"] != agent_id:
                return {"ok": False, "failure_reason": "input"}
            node = self._node(job["node_id"])
            if node and node["host_agent"] != agent_id:
                mid = self._send(agent_id, node["host_agent"], "release", {"job_id": job["id"]}, task_ref=job["task_ref"])
                return {"ok": True, "pending": True, "message_id": mid}
            if job["status"] not in TERMINAL:
                return {"ok": False, "failure_reason": "input"}
            self._release(job)
        if params.get("cache_id"):
            return self.manage_cache(agent_id, {"cache_id": params["cache_id"], "mode": "evict"})
        return {"ok": True}

    def on_message(self, msg, receiver):
        if receiver not in msg.receivers or msg.status.get(receiver) != "delivered":
            return
        protocol = msg.payload.get("_compute")
        token = (msg.id, receiver)
        if not isinstance(protocol, dict) or token in self._seen_messages:
            return
        self._seen_messages.add(token)
        op = protocol.get("op")
        node = self._node(receiver)
        if op == "discover" and node and node["availability"] != "offline":
            self._send(receiver, msg.sender, "discover_result", {"node": self._node_public(node)}, correlation_id=msg.id)
        elif op == "discover_result":
            advertised = protocol.get("node", {})
            if advertised.get("host_agent") == msg.sender:
                self.known_services.setdefault(receiver, {})[advertised["id"]] = deepcopy(advertised)
        elif op == "submit" and node:
            job = self.jobs.get(protocol.get("job_id"))
            if job and job["requester"] == msg.sender and job["node_id"] == node["id"]:
                self._admit(job, node, delivered_params=protocol.get("params", {}))
        elif op == "job_result":
            job = self.jobs.get(protocol.get("job_id"))
            if job is None or receiver != job["requester"]:
                return
            host = self._node(job["node_id"])
            if host is None or msg.sender != host["host_agent"]:
                return
            result_status = protocol.get("status")
            view = deepcopy(protocol.get("job", {}))
            view["status"] = result_status
            ref = job["task_ref"]
            runtime = self.sim.agents.get(receiver)
            task = runtime.tasks.get(ref[0]) if runtime is not None and ref else None
            stale = bool(task and (task.get("rev") != ref[1] or task.get("status") in ("canceled", "expired")))
            late = job["id"] in self._request_timeouts or (result_status == "completed" and job["status"] in ("failed", "canceled"))
            if result_status == "completed" and job["status"] == "downloading":
                self._set_status(job, "completed", delivered=True)
            if stale or late:
                view["isolated"] = "stale_revision" if stale else "late_terminal_result"
                self.isolated_results.setdefault(receiver, {})[job["id"]] = deepcopy(view)
                if stale and not late:
                    self.job_views.setdefault(receiver, {})[job["id"]] = dict(view, result=None)
                msg.payload.update(isolated=view["isolated"], evidence_ids=[], observations=[])
                if late:
                    msg.payload.update(status="failed", action_status="failed")
                if stale:
                    self._branch(job["id"], "Q108", receiver, "过期任务版本的计算结果仅作历史记录")
                return
            self.job_views.setdefault(receiver, {})[job["id"]] = view
            if result_status == "completed" and job["status"] == "completed":
                self.results.setdefault(receiver, {})[job["id"]] = deepcopy(view.get("result"))
        elif op in ("cancel", "release", "migrate") and node:
            job = self.jobs.get(protocol.get("job_id"))
            if job is None or job["requester"] != msg.sender or job["node_id"] != node["id"]:
                return
            if op == "cancel":
                self.cancel_job(receiver, {"job_id": job["id"]})
            elif op == "release" and job["status"] in TERMINAL:
                self._release(job)
            elif op == "migrate":
                target = self._node(protocol.get("target_node"))
                if target:
                    self._migrate_at_source(job, target)
        elif op == "migration_result":
            job = self.jobs.get(protocol.get("source_job"))
            if job and receiver == job["requester"] and self._node(job["node_id"])["host_agent"] == msg.sender:
                params = deepcopy(protocol["params"])
                params["cpu_s"] += params.pop("restart_cpu_s", 0.5)
                checkpoint = params.get("checkpoint_ref", {})
                params["input_bytes"] = max(int(params.get("input_bytes", 0)), int(job["params"].get("checkpoint_bytes", 4096)))
                new = self.submit_job(receiver, params, task_ref=job["task_ref"])
                self.job_views.setdefault(receiver, {}).setdefault(job["id"], {})["migrated_to"] = new
        elif op in ("manage_model", "recover") and node:
            params = deepcopy(protocol.get("params", {}))
            params["node_id"] = node["id"]
            result = self.manage_model(receiver, params) if op == "manage_model" else self.diagnose_recover(receiver, params)
            self._send(receiver, msg.sender, "model_result", {"result": result, "node": self._node_public(node)}, correlation_id=msg.id)
        elif op == "model_result" and protocol.get("node", {}).get("host_agent") == msg.sender:
            self.known_services.setdefault(receiver, {})[protocol["node"]["id"]] = deepcopy(protocol["node"])

    def step(self):
        for due, nid, sid in list(self._recoveries):
            if self.sim.t >= due:
                self._recoveries.remove((due, nid, sid))
                node = self.nodes[nid]
                if node["availability"] == "offline":
                    self._availability(node, "available")
                if sid in node["services"] and node["services"][sid]["status"] == "failed":
                    self._load(node, sid)
        for (nid, sid), load in list(self._loads.items()):
            if self.sim.t >= load["due"]:
                node = self.nodes[nid]
                self._finish_load(node, sid)
                if node["availability"] != "offline" and node["services"][sid]["status"] == "loading":
                    node["services"][sid]["fault"] = None
                    self._service_status(node, sid, "ready")
        for job in list(self.jobs.values()):
            node = self._node(job["node_id"])
            requester_view = self.job_views.get(job["requester"], {}).get(job["id"], {})
            if (node and node["host_agent"] != job["requester"] and self.sim.t >= job["deadline_s"]
                    and requester_view.get("status") not in TERMINAL and job["id"] not in self._request_timeouts):
                self._request_timeouts.add(job["id"])
                before = requester_view.get("status", "uploading")
                requester_view.update(status="failed", failure_reason="timeout", result=None)
                self._emit(job["id"] + ":requester", "job.status", before, "failed", ["F336"],
                           [job["requester"]], "请求方等待远程结果超时")
            if job["status"] == "running":
                self._advance(job)
            elif job["status"] not in TERMINAL and self.sim.t >= job["deadline_s"]:
                # Downloaded-but-lost results do not become locally usable by timeout.
                self._fail(job, "timeout", notify=job["status"] != "uploading")
            if job["status"] == "uploading" and job["request_message"]:
                message = self.sim.network.messages.get(job["request_message"])
                if message and all(s in ("failed", "expired") for s in message.status.values()):
                    self._fail(job, "network", notify=False)
                    self.job_views[job["requester"]][job["id"]] = self._job_public(job)
            elif job["status"] == "downloading" and job["result_message"]:
                message = self.sim.network.messages.get(job["result_message"])
                if message and all(s in ("failed", "expired") for s in message.status.values()):
                    self._fail(job, "network", notify=False)
        for node in self.nodes.values():
            if node["availability"] == "offline":
                continue
            for jid in list(node["queue"]):
                if len(node["running"]) >= node["max_concurrency"]:
                    break
                job = self.jobs[jid]
                service = node["services"].get(job["service_id"])
                if not service or service["status"] == "failed":
                    self._fail(job, "model_unavailable")
                else:
                    self._start(job, node)
        for cache in self.caches.values():
            if cache["state"] == "resident" and self.sim.t >= cache["valid_until_s"]:
                cache["state"] = "invalid"
                self._release_cache(cache)
                self._emit(cache["id"], "cache.eviction_state", "resident", "invalid", ["F358"], [cache["owner"]], "缓存已过期")

    def inject(self, disruption):
        if disruption.get("kind") != "service_failure":
            return False
        params = disruption.get("params", {})
        target = disruption.get("target")
        nodes = [self._node(target)] if target and self._node(target) else list(self.nodes.values()) if not target else []
        if not nodes and target:
            nodes = [n for n in self.nodes.values() if target in n["services"]]
        for node in nodes:
            ids = [params["service_id"]] if params.get("service_id") else [target] if target in node["services"] else list(node["services"])
            for sid in ids:
                if sid not in node["services"]:
                    continue
                for jid in list(node["running"]):
                    if self.jobs[jid]["service_id"] == sid:
                        self._advance(self.jobs[jid])
                self._finish_load(node, sid)
                node["services"][sid]["fault"] = {"recoverable": bool(params.get("recoverable", params.get("duration_s") is not None))}
                self._service_status(node, sid, "failed")
                self._branch(node["id"] + ":" + sid, "Q107", node["host_agent"], "服务失效，已耗计算成本保留")
                for job in list(self.jobs.values()):
                    if job["node_id"] == node["id"] and job["service_id"] == sid and job["status"] in ("queued", "loading", "running"):
                        self._fail(job, "model_error")
                duration = params.get("duration_s")
                if duration is not None and params.get("recoverable", True):
                    self._recoveries.append((self.sim.t + max(0.0, float(duration)), node["id"], sid))
            if params.get("offline"):
                self._availability(node, "offline")
        return True

    def diagnose_recover(self, agent_id, params):
        node = self._node(params.get("node_id", agent_id))
        if node is None:
            return {"ok": False, "failure_reason": "model_unavailable"}
        if node["host_agent"] != agent_id:
            mid = self._send(agent_id, node["host_agent"], "recover", {"params": params})
            return {"ok": True, "pending": True, "message_id": mid}
        sid = params.get("service_id", "inference")
        service = node["services"].get(sid)
        if not service or service["status"] != "failed" or not (service.get("fault") or {}).get("recoverable"):
            return {"ok": False, "failure_reason": "model_unavailable"}
        if node["availability"] == "offline":
            self._availability(node, "available")
        return self._load(node, sid)

    recover_service = diagnose_recover

    def action_guard(self, agent, action_id, params):
        aid = agent.id
        if action_id in ("A047", "A048", "A049") and params.get("job_id"):
            if not self.query_job(aid, params):
                return False, "unauthorized"
        if action_id == "A044":
            try:
                cpu = float(params.get("cpu_s", 5.0))
                memory = int(params.get("memory_request_bytes", params.get("memory_bytes", MIB)))
                request = float(params.get("cpu_request", 1.0))
                deadline = float(params.get("deadline_s", self.sim.t + 300))
                if cpu < 0 or not math.isfinite(cpu) or memory < 0 or request <= 0 or not math.isfinite(request) or not math.isfinite(deadline):
                    return False, "incompatible"
            except (TypeError, ValueError):
                return False, "incompatible"
            selected = self.selections.get(aid, {})
            nid = params.get("node_id", params.get("target_node", params.get("host", selected.get("node_id", aid))))
            node = self._node(nid)
            # Local telemetry is authoritative. Remote admission is deliberately
            # NOT inspected here; only the delivered host request can decide it.
            if node and node["host_agent"] == aid:
                service = node["services"].get(params.get("service_id", selected.get("service_id", "inference")))
                if service is None:
                    return False, "incompatible"
                weights = 0 if service["reserved"] else service["memory_bytes"]
                if node["memory_used_bytes"] + weights + memory > node["memory_capacity_bytes"]:
                    self._branch(params.get("_instance_id", aid), "Q106", aid, "本机内存准入不足，需要消息发现远程服务")
                    return False, "insufficient"
        return True, "none"

    def action_plan(self, agent, action_id, params):
        return {"duration_s": float(params.get("duration_s", 1.0)), "energy_Wh": 0.0005,
                "bytes": 0, "cpu_s": 0.0}

    def action_effect(self, agent, action_id, params, rng):
        task_ref = params.get("_task_ref", params.get("task_ref"))
        params = {k: deepcopy(v) for k, v in params.items() if not k.startswith("_")}
        methods = {"A042": self.discover_services, "A043": self.select_execution,
                   "A045": self.manage_model, "A046": self.manage_cache,
                   "A047": self.migrate_job, "A048": self.cancel_job, "A049": self.release_compute,
                   "A061": self.diagnose_recover}
        if action_id == "A044":
            jid = self.submit_job(agent.id, params, task_ref=task_ref)
            result = self.query_job(agent.id, {"job_id": jid})
            ok = result.get("status") != "failed"
            reason = result.get("failure_reason", "none")
            observation = Observation(id="submission:" + jid, by=agent.id, t=self.sim.t, about=[jid],
                                      kind="compute_submission", values={"job.status": result.get("status"),
                                      "job.failure_reason": reason, "job.service_id": result.get("service_id")},
                                      confidence=1.0, model=self.backend_mode)
            return ActionOutcome(status="succeeded" if ok else "failed", failure_code="none" if ok else "model",
                                 local_writes=[("compute.queue_ids", [jid])], observations=[observation],
                                 feedback_zh=(jid + " 已提交，入队不等于执行完成") if ok else "作业准入失败：" + reason)
        if action_id in ("A047", "A048", "A049") and not params.get("job_id"):
            params["job_id"] = self.latest_job.get(agent.id)
        result = methods[action_id](agent.id, params)
        ok = result.get("ok", False)
        pending = result.get("pending", False) or bool(result.get("pending_messages"))
        writes = [("compute.node_id", result["node_id"]), ("job.service_id", result["service_id"])] if action_id == "A043" and ok else []
        return ActionOutcome(status="succeeded" if ok else "failed", failure_code="none" if ok else "model", local_writes=writes,
                             feedback_zh=("计算请求已排队，等待网络送达" if pending else "计算资源操作已完成") if ok else "计算资源操作失败：" + result.get("failure_reason", "model_unavailable"))

    def to_dict(self):
        return {"backend_mode": self.backend_mode, "nodes": deepcopy(list(self.nodes.values())),
                "jobs": [self._job_public(j) for j in self.jobs.values()], "caches": deepcopy(list(self.caches.values()))}
