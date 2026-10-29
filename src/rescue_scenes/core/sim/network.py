"""Seeded, store-and-forward radio transport (an abstract model, not ns-3).

Accounting boundary: action plans pay orchestration time/energy and have zero
transport bytes/CPU. This module alone charges actual transmission attempts,
including every relay hop and retry, through the unified ledger. It never charges
CPU or action orchestration again. A business ACK is itself a delivered message;
a radio contact, transmission, or local copy is not an application ACK.
"""
from __future__ import annotations

from copy import deepcopy
import heapq
import json
import math

from rescue_scenes.core.contracts import ActionOutcome, Message, Observation
from rescue_scenes.core.world.profiles import OVERLAYS


KINDS = frozenset("cue report evidence task bid assignment ack correction result control".split())
TERMINAL = frozenset(("delivered", "expired", "failed"))


class Network:
    backend_mode = "abstract_model"

    def __init__(self, sim):
        self.sim = sim
        self.options = {"loss_rate": 0.01, "latency_s": 0.1, "bandwidth_factor": 1.0}
        for oid in sim.config.get("overlays", []):
            effect = OVERLAYS[oid]["effects"].get("network", {})
            for key in ("loss_rate", "latency_s"):
                if key in effect:
                    self.options[key] = max(self.options[key], float(effect[key]))
            self.options["bandwidth_factor"] *= float(effect.get("bandwidth_factor", 1.0))
            self.options.setdefault("outage_windows", []).extend(deepcopy(effect.get("outage_windows", [])))
        self.messages = {}
        # Archives remain complete; only transport/ACK work stays in the tick set.
        # Numeric send order also preserves ordering when retry reopens an old ACK.
        self._pending_messages = {}
        self.links = {}
        self.relays = {}
        self.radios = {}
        self.outages = []
        self._states = {}
        self._flights = {}
        self._delivered = set()
        self._acks = {}
        self._ack_deadlines = {}
        self._serial = 0
        for aid, agent in sorted(sim.agents.items()):
            spec = agent.spec
            self.radios[aid] = {
                "enabled": True, "channel": spec.get("radio_channel", "rescue"),
                "tx_power_dbm": 20.0, "routing_mode": "relay", "relay_enabled": False,
                "range_m": float(spec.get("radio_range_m", 1000.0)),
                "bandwidth_bps": float(spec.get("radio_bandwidth_bps", 1_000_000)),
            }
        self.outages.extend(deepcopy(self.options.get("outage_windows", [])))

    def _emit(self, subject, key, before, after, refs, visible=None, text="网络状态更新", **kw):
        family = "sm:" + key if key in ("message.status", "message.ack_state", "link.state") else "field.write"
        return self.sim.emit(layer="comms", family=family, kind=key, grounding="catalog",
                             refs=list(refs), subject=subject, field=key, before=before, after=after,
                             visible_to=visible or ["*truth*"], backend_mode=self.backend_mode,
                             scripted=False, text_zh=text, **kw)

    def _status(self, msg, receiver, state):
        before = msg.status[receiver]
        if before == state:
            return
        msg.status[receiver] = state
        if state in TERMINAL:
            self._retire(msg)
        self._emit(msg.id, "message.status", before, state, ["F285"], [msg.sender],
                   "消息向 %s 的状态为 %s" % (receiver, state), actor=msg.sender,
                   task=msg.task_ref[0] if msg.task_ref else None)

    def _retire(self, msg):
        if msg.ack_state != "pending" and all(status in TERMINAL for status in msg.status.values()):
            self._pending_messages.pop(int(msg.id[4:]), None)

    def _endpoint(self, eid):
        compute = getattr(self.sim, "compute", None)
        if compute is not None and eid in compute.nodes:
            return compute.nodes[eid].get("host_agent") or eid
        return eid

    def _pose(self, eid):
        eid = self._endpoint(eid)
        if eid in self.relays:
            return self.relays[eid]["position"]
        agent = self.sim.agents.get(eid)
        if agent is None:
            return None
        pose = self.sim.world.get(getattr(agent, "platform_eid", eid), "entity.pose", None)
        if isinstance(pose, dict):
            pose = pose.get("position", pose.get("xy", [pose.get("x"), pose.get("y")]))
        if isinstance(pose, (tuple, list)) and len(pose) >= 2 and all(isinstance(x, (float, int)) for x in pose[:2]):
            return tuple(pose[:2])
        platform = getattr(agent, "platform", {})
        pose = platform.get("pose") if isinstance(platform, dict) else None
        return tuple(pose[:2]) if isinstance(pose, (tuple, list)) and len(pose) >= 2 else None

    def _radio(self, eid):
        eid = self._endpoint(eid)
        if eid in self.relays:
            r = self.relays[eid]
            return {"enabled": r["active"], "channel": r["channel"], "range_m": r["range_m"],
                    "bandwidth_bps": r["bandwidth_bps"]}
        radio = dict(self.radios.get(eid, {"enabled": False}))
        agent = self.sim.agents.get(eid)
        if agent:
            platform = getattr(agent, "platform_eid", eid)
            radio["enabled"] = radio["enabled"] and self.sim.world.get(platform, "radio.enabled", True)
            radio["enabled"] = radio["enabled"] and self.sim.world.get(platform, "platform.health_state", "nominal") != "failed"
            battery = self.sim.world.get(platform, "platform.battery_remaining_Wh", None)
            if battery is not None and battery <= 0 and not agent.spec.get("mains_power", False):
                radio["enabled"] = False
        return radio

    @staticmethod
    def _matches(window, a, b):
        target = window.get("target")
        if target is None or target in ("*", "all"):
            return True
        if isinstance(target, (list, tuple)):
            return set(target) == {a, b}
        return target in (a, b, a + "->" + b, b + "->" + a)

    def _outage(self, a, b, start=None, end=None):
        start = self.sim.t if start is None else start
        end = start if end is None else end
        for w in self.outages:
            lo = float(w.get("start_s", w.get("t", 0)))
            hi = float(w.get("end_s", lo + float(w.get("duration_s", 60))))
            if self._matches(w, a, b) and ((lo <= start < hi) or (start < hi and end > lo)):
                return True
        return False

    def _edge(self, a, b):
        if a == b:
            return {"bandwidth_bps": float("inf"), "latency_s": 0.0, "loss_rate": 0.0}
        ra, rb = self._radio(a), self._radio(b)
        pa, pb = self._pose(a), self._pose(b)
        if not ra.get("enabled") or not rb.get("enabled") or pa is None or pb is None:
            return None
        if ra.get("channel") != rb.get("channel") or self._outage(a, b):
            return None
        distance = math.dist(pa, pb)
        limit = min(float(ra.get("range_m", 0)), float(rb.get("range_m", 0)))
        if limit <= 0 or distance > limit:
            return None
        loss = float(self.options["loss_rate"])
        latency = float(self.options["latency_s"])
        bandwidth = min(float(ra.get("bandwidth_bps", 1e6)), float(rb.get("bandwidth_bps", 1e6)))
        bandwidth = float(self.options.get("bandwidth_bps", bandwidth)) * self.options["bandwidth_factor"]
        if bandwidth <= 0:
            return None
        return {"bandwidth_bps": bandwidth, "latency_s": max(0.0, latency),
                "loss_rate": max(0.0, min(1.0, loss)), "distance_m": distance}

    def check_link(self, sender, receiver):
        """Model-derived link estimate; callers publish it only via probe/delivery."""
        a, b = self._endpoint(sender), self._endpoint(receiver)
        relay_ids = {rid for rid, r in self.relays.items() if r["active"]}
        relay_ids.update(aid for aid, radio in self.radios.items() if radio.get("relay_enabled"))
        vertices = sorted(relay_ids | {a, b})
        heap = [(0.0, (a,))]
        best = {a: 0.0}
        route = None
        while heap:
            cost, path = heapq.heappop(heap)
            current = path[-1]
            if cost > best.get(current, float("inf")):
                continue
            if current == b:
                route = list(path)
                break
            for nxt in vertices:
                if nxt in path:
                    continue
                edge = self._edge(current, nxt)
                if edge is None:
                    continue
                new = cost + edge["latency_s"] + 8192.0 / edge["bandwidth_bps"]
                if new < best.get(nxt, float("inf")):
                    best[nxt] = new
                    heapq.heappush(heap, (new, path + (nxt,)))
        edges = [self._edge(x, y) for x, y in zip(route or [], (route or [])[1:])]
        loss = 1.0 - math.prod(1.0 - e["loss_rate"] for e in edges)
        state = ("unknown" if self._pose(a) is None or self._pose(b) is None else "unreachable") if route is None else ("intermittent" if loss >= 0.1 else "reachable")
        previous = self.links.get((sender, receiver), {"state": "unknown", "sample_count": 0})
        result = {"sender": sender, "receiver": receiver, "state": state, "route": route or [],
                  "goodput_bps": min((e["bandwidth_bps"] for e in edges), default=1e12 if route else 0.0),
                  "latency_s": sum(e["latency_s"] for e in edges), "loss_rate": loss,
                  "sample_count": previous["sample_count"], "sampled_at_s": previous.get("sampled_at_s"),
                  "quality_source": "abstract_model"}
        self.links[(sender, receiver)] = result
        if previous["state"] != state:
            self._emit(sender + "->" + receiver, "link.state", previous["state"], state, ["F264"],
                       text="抽象链路状态为 " + state)
        return deepcopy(result)

    def probe_link(self, sender, receiver):
        result = self.check_link(sender, receiver)
        result["sample_count"] += 1
        result["sampled_at_s"] = self.sim.t
        self.links[(sender, receiver)] = deepcopy(result)
        self._emit(sender + "->" + receiver, "link.sample_count", result["sample_count"] - 1,
                   result["sample_count"], ["A032", "F270"], [sender], "完成抽象链路探测", actor=sender)
        return result

    def configure_radio(self, agent_id, params):
        radio = self.radios[agent_id]
        for key in ("enabled", "channel", "tx_power_dbm", "routing_mode", "relay_enabled", "range_m", "bandwidth_bps"):
            if key in params:
                # Power is metadata; increasing it does not magically repair an outage.
                agent = self.sim.agents[agent_id]
                value = params[key]
                if key == "range_m":
                    value = max(0.0, min(float(value), float(agent.spec.get("radio_range_m", 1000))))
                if key == "bandwidth_bps":
                    value = max(0.0, min(float(value), float(agent.spec.get("radio_bandwidth_bps", 1_000_000))))
                radio[key] = value
                if key not in ("range_m", "bandwidth_bps"):
                    self.sim.write_truth(agent.platform_eid, "radio." + key, params[key], "A033", actor=agent_id)
        return deepcopy(radio)

    def add_relay(self, agent_id, relay_id=None, position=None, range_m=None, active=True):
        """Register a physically deployed relay; A034 owns inventory consumption."""
        relay_id = relay_id or "relay-%d" % (len(self.relays) + 1)
        position = position if position is not None else self._pose(agent_id)
        if position is None:
            raise ValueError("中继位置未知 / unknown relay position")
        self.relays[relay_id] = {"id": relay_id, "owner": agent_id, "position": tuple(position[:2]),
                                 "range_m": float(range_m or self.options.get("relay_range_m", 1200)),
                                 "active": bool(active), "channel": self._radio(agent_id).get("channel", "rescue"),
                                 "bandwidth_bps": float(self.options.get("bandwidth_bps", 1e6))}
        return relay_id

    def queued_bytes(self, sender):
        return sum(m.size_bytes for m in self._pending_messages.values() if m.sender == sender
                   for status in m.status.values() if status not in TERMINAL)

    def send(self, sender, receivers, kind, payload, size_bytes, task_ref=None, ttl_s=None, correlation_id=None):
        if kind not in KINDS:
            raise ValueError("消息类型非法 / invalid message kind: " + str(kind))
        receivers = [receivers] if isinstance(receivers, str) else list(receivers)
        receivers = list(dict.fromkeys(receivers))
        if not receivers or not isinstance(payload, dict):
            raise ValueError("消息需要收件人和字典内容 / receivers and dictionary payload required")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError("字节数须为非负整数 / size_bytes must be a nonnegative integer")
        ttl_s = float(self.options.get("ttl_s", 120.0) if ttl_s is None else ttl_s)
        if ttl_s < 0 or not math.isfinite(ttl_s):
            raise ValueError("有效期非法 / invalid TTL")
        if self.queued_bytes(sender) + size_bytes * len(receivers) > int(self.options.get("queue_limit_bytes", 64 * 1024 * 1024)):
            raise ValueError("发送队列已满 / network queue full")
        # A caller may pad a packet, but cannot transfer a large payload for free.
        size_bytes = max(size_bytes, len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")))
        if self.queued_bytes(sender) + size_bytes * len(receivers) > int(self.options.get("queue_limit_bytes", 64 * 1024 * 1024)):
            raise ValueError("发送队列已满 / network queue full")
        self._serial += 1
        mid = "msg-%06d" % self._serial
        ack = "pending" if payload.get("require_ack", kind in ("assignment", "task", "control")) and kind != "ack" else "not_required"
        msg = Message(id=mid, kind=kind, sender=sender, receivers=receivers, payload=deepcopy(payload),
                      size_bytes=size_bytes, task_ref=tuple(task_ref) if task_ref else None,
                      correlation_id=correlation_id, ttl_s=ttl_s, sent_s=float(self.sim.t), delivered_s=None,
                      status={rid: "created" for rid in receivers}, ack_state=ack, retry_count=0, hop_trace=[])
        self.messages[mid] = msg
        self._pending_messages[self._serial] = msg
        self._acks[mid] = set()
        if ack == "pending":
            self._ack_deadlines[mid] = self.sim.t + float(self.options.get("ack_timeout_s", ttl_s))
        for rid in receivers:
            self._states[(mid, rid)] = {"current": self._endpoint(sender), "attempt": 0,
                                        "remaining": size_bytes, "next_s": self.sim.t, "hop": 0}
            self._status(msg, rid, "queued")
        return mid

    def _charge(self, msg, receiver, state, source, nbytes, duration):
        owner = self.relays.get(source, {}).get("owner", self._endpoint(source))
        if owner not in self.sim.agents:
            owner = msg.sender
        energy = duration * float(self.options.get("radio_power_W", 1.0)) / 3600.0
        energy += nbytes * float(self.options.get("tx_energy_Wh_per_byte", 1e-9))
        self.sim.ledger.add("net:%s:%s:%d:%d" % (msg.id, receiver, state["hop"], state["attempt"]),
                            owner, "NETWORK", {"time_s": duration, "energy_Wh": energy,
                                               "bytes": nbytes, "cpu_s": 0.0}, task_ref=msg.task_ref)

    def _start(self, msg, rid, state, busy):
        if state["next_s"] > self.sim.t:
            return
        current, destination = state["current"], self._endpoint(rid)
        link = self.check_link(current, destination)
        if not link["route"]:
            if not state.get("outage_noted"):
                state["outage_noted"] = True
                self.sim.emit(layer="comms", family="branch.trigger", kind="branch.trigger",
                              grounding="catalog", refs=["Q104"], subject=msg.id, actor=msg.sender,
                              visible_to=[msg.sender], backend_mode=self.backend_mode, scripted=False,
                              text_zh="链路不可用，消息保留在有有效期的队列中")
            return
        if current == destination:
            if msg.status[rid] == "queued":
                self._status(msg, rid, "transmitting")
            self._deliver(msg, rid)
            return
        nxt = link["route"][1]
        if current in busy:
            return
        edge = self._edge(current, nxt)
        if edge is None:
            return
        if state.get("retry_next") not in (None, nxt):
            # Fragments buffered by the old next hop are not magically available
            # at a newly selected relay after a route change.
            state["remaining"] = msg.size_bytes
        state["retry_next"] = nxt
        duration = state["remaining"] * 8.0 / edge["bandwidth_bps"] + edge["latency_s"]
        if msg.status[rid] == "queued":
            self._status(msg, rid, "transmitting")
        state["attempt"] += 1
        self._flights[(msg.id, rid)] = {"source": current, "next": nxt, "start": self.sim.t,
                                        "finish": self.sim.t + duration, "edge": edge,
                                        "bytes": state["remaining"]}
        busy.add(current)

    def _finish(self, msg, rid, state, flight):
        source, nxt = flight["source"], flight["next"]
        self._charge(msg, rid, state, source, flight["bytes"], flight["finish"] - flight["start"])
        interrupted = self._outage(source, nxt, flight["start"], flight["finish"]) or self._edge(source, nxt) is None
        mtu = max(1, int(self.options.get("mtu_bytes", 1500)))
        rng = self.sim.rng("network.loss", "%s|%s|%s|%s|%s" % (msg.id, rid, state["hop"], state["attempt"], source))
        missing = 0
        for offset in range(0, flight["bytes"], mtu):
            amount = min(mtu, flight["bytes"] - offset)
            if interrupted or rng.random() < flight["edge"]["loss_rate"]:
                missing += amount
        if interrupted and flight["bytes"] == 0:
            missing = 1
        if missing:
            if missing < flight["bytes"] and msg.status[rid] == "transmitting":
                self._status(msg, rid, "partial")
                self._emit(msg.id, "message.received_fragments", 0,
                           math.ceil((msg.size_bytes - missing) / mtu), ["F281", "Q103"], [msg.sender], "消息分片不完整")
            if state["attempt"] > int(self.options.get("retry_limit", 2)):
                self._status(msg, rid, "failed")
                return
            state["remaining"] = min(flight["bytes"], missing)
            state["next_s"] = self.sim.t + float(self.options.get("retry_delay_s", 1.0))
            msg.retry_count += 1
            return
        msg.hop_trace.append({"receiver": rid, "from": source, "to": nxt, "t": float(self.sim.t),
                              "attempts": state["attempt"]})
        state.update(current=nxt, attempt=0, remaining=msg.size_bytes, next_s=self.sim.t, hop=state["hop"] + 1)
        if nxt == self._endpoint(rid):
            self._deliver(msg, rid)

    def _deliver(self, msg, receiver):
        token = (msg.id, receiver)
        if token in self._delivered:
            return
        self._delivered.add(token)
        self._status(msg, receiver, "delivered")
        msg.delivered_s = float(self.sim.t)
        self._emit(msg.id, "message.local_availability", False, True, ["F292"], [receiver],
                   "收件人已得到消息的本地副本", actor=receiver)
        # The dedicated family is required for traceability, not a truth/knowledge shortcut.
        self.sim.emit(layer="comms", family="msg.local_copy", kind="msg.local_copy", grounding="catalog",
                      refs=["F292"], subject=msg.id, field=None, before=None, after={"receiver": receiver},
                      actor=receiver, visible_to=[receiver], backend_mode=self.backend_mode, scripted=False,
                      text_zh=receiver + " 收到完整消息副本")
        if msg.kind == "ack":
            original_id = msg.correlation_id or msg.payload.get("message_id")
            original = self.messages.get(original_id)
            if original and receiver == original.sender and msg.sender in original.receivers:
                self._acks[original.id].add(msg.sender)
                if set(original.receivers) <= self._acks[original.id] and original.ack_state == "pending":
                    before = original.ack_state
                    original.ack_state = "acknowledged"
                    self._retire(original)
                    self._emit(original.id, "message.ack_state", before, "acknowledged", ["F288", "A037"],
                               [original.sender], "业务确认消息已送达")
        delivered = deepcopy(msg)
        compute = getattr(self.sim, "compute", None)
        if compute is not None:
            compute.on_message(delivered, receiver)
        runtime = self.sim.agents.get(receiver)
        if runtime is not None:
            runtime.on_message(self.sim, delivered)

    def step(self):
        # Complete only a hop whose entire serialization and latency have elapsed.
        for token, flight in list(self._flights.items()):
            msg, rid = self.messages[token[0]], token[1]
            expiry = msg.sent_s + msg.ttl_s
            if flight["finish"] <= self.sim.t and flight["finish"] < expiry:
                del self._flights[token]
                self._finish(msg, rid, self._states[token], flight)
        for msg in [self._pending_messages[key] for key in sorted(self._pending_messages)]:
            for rid in msg.receivers:
                if msg.status[rid] not in TERMINAL and self.sim.t >= msg.sent_s + msg.ttl_s:
                    flight = self._flights.pop((msg.id, rid), None)
                    if flight:
                        elapsed = max(0.0, min(self.sim.t, msg.sent_s + msg.ttl_s) - flight["start"])
                        sent = min(flight["bytes"], int(elapsed * flight["edge"]["bandwidth_bps"] / 8))
                        self._charge(msg, rid, self._states[(msg.id, rid)], flight["source"], sent, elapsed)
                    self._status(msg, rid, "failed" if msg.status[rid] == "partial" else "expired")
            if msg.ack_state == "pending" and self.sim.t >= self._ack_deadlines.get(msg.id, float("inf")):
                msg.ack_state = "timeout"
                self._retire(msg)
                self._emit(msg.id, "message.ack_state", "pending", "timeout", ["F288", "A037"],
                           [msg.sender], "等待业务确认超时")
        busy = {f["source"] for f in self._flights.values()}
        for msg in [self._pending_messages[key] for key in sorted(self._pending_messages)]:
            for rid in msg.receivers:
                token = (msg.id, rid)
                if msg.status[rid] not in TERMINAL and token not in self._flights:
                    self._start(msg, rid, self._states[token], busy)

    def acknowledge(self, message_id, receiver):
        msg = self.messages[message_id]
        if (message_id, receiver) not in self._delivered:
            raise ValueError("未收到完整消息，不能确认 / cannot acknowledge an undelivered message")
        return self.send(receiver, [msg.sender], "ack", {"message_id": message_id, "require_ack": False},
                         64, task_ref=msg.task_ref, correlation_id=message_id)

    def await_ack(self, message_id, timeout_s=None):
        msg = self.messages[message_id]
        if timeout_s is not None and msg.ack_state == "pending":
            self._ack_deadlines[message_id] = min(self._ack_deadlines[message_id], self.sim.t + max(0, float(timeout_s)))
        return {"message_id": message_id, "ack_state": msg.ack_state, "status": deepcopy(msg.status)}

    def retry(self, message_id, receivers=None):
        original = self.messages[message_id]
        receivers = receivers or [r for r in original.receivers if r not in self._acks[message_id]]
        if not receivers:
            return message_id
        new = self.send(original.sender, receivers, original.kind, original.payload, original.size_bytes,
                        original.task_ref, original.ttl_s, original.correlation_id or original.id)
        self.messages[new].retry_count = original.retry_count + 1
        if original.ack_state == "timeout":
            original.ack_state = "pending"
            self._pending_messages[int(original.id[4:])] = original
            self._ack_deadlines[message_id] = self.sim.t + original.ttl_s
            self._emit(message_id, "message.ack_state", "timeout", "pending", ["F288", "A037"],
                       [original.sender], "重新等待业务确认")
        return new

    def forward_message(self, sender, message_id, receivers):
        msg = self.messages[message_id]
        if (message_id, sender) not in self._delivered:
            raise ValueError("上跳未完整送达 / previous hop not delivered")
        remaining_ttl = msg.ttl_s - (self.sim.t - msg.sent_s)
        if remaining_ttl <= 0:
            raise ValueError("消息已过期 / message expired")
        mid = self.send(sender, receivers, msg.kind, msg.payload, msg.size_bytes, msg.task_ref,
                        remaining_ttl, msg.correlation_id or msg.id)
        self.messages[mid].hop_trace = deepcopy(msg.hop_trace)
        return mid

    def inject(self, disruption):
        if disruption.get("kind") != "link_outage":
            return False
        params = disruption.get("params", {})
        start = float(params.get("start_s", self.sim.t))
        self.outages.append({"target": disruption.get("target"), "start_s": start,
                             "end_s": start + max(0.0, float(params.get("duration_s", 60)))})
        return True

    def _relay_item(self, agent, params):
        candidates = [params["item_id"]] if params.get("item_id") else [e["id"] for e in self.sim.world.entities_of("item")]
        for eid in candidates:
            holder = self.sim.world.get(eid, "item.holder_id")
            state = self.sim.world.get(eid, "item.state")
            if (self.sim.world.get(eid, "item.kind") == "relay" and holder in (agent.id, agent.platform_eid)
                    and state in ("stock", "reserved", "attached", "in_transit")
                    and self.sim.world.get(eid, "item.quantity", 1) > 0):
                return eid
        return None

    def action_guard(self, agent, action_id, params):
        mid = params.get("message_id")
        if action_id in ("A035", "A037", "A065"):
            msg = self.messages.get(mid)
            if msg is None:
                return False, "insufficient"
            if action_id == "A035" and (mid, agent.id) not in self._delivered:
                return False, "insufficient"
            if action_id == "A035" and self.sim.t >= msg.sent_s + msg.ttl_s:
                return False, "stale"
            if action_id == "A037" and msg.sender != agent.id:
                return False, "unauthorized"
            if action_id == "A065" and agent.id not in msg.receivers:
                return False, "unauthorized"
        if action_id == "A034" and self._relay_item(agent, params) is None:
            return False, "insufficient"
        if action_id in ("A035", "A036"):
            receivers = params.get("receivers", params.get("receiver_ids", params.get("receiver", params.get("target"))))
            if not receivers or not isinstance(receivers, (str, list, tuple)):
                return False, "incompatible"
            receivers = [receivers] if isinstance(receivers, str) else receivers
            if not all(isinstance(r, str) and r for r in receivers):
                return False, "incompatible"
            if action_id == "A036":
                payload = params.get("payload", {})
                if not isinstance(payload, dict) or params.get("kind", "report") not in KINDS:
                    return False, "incompatible"
                try:
                    size = int(params.get("size_bytes", 0))
                    ttl = float(120 if params.get("ttl_s") is None else params["ttl_s"])
                    if size < 0 or ttl < 0 or not math.isfinite(ttl):
                        return False, "incompatible"
                except (TypeError, ValueError):
                    return False, "incompatible"
                size = max(size, len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")))
                if self.queued_bytes(agent.id) + size * len(receivers) > int(self.options.get("queue_limit_bytes", 64 * 1024 * 1024)):
                    return False, "insufficient"
                original = self.messages.get(params.get("retry_of"))
                if params.get("retry_of") and (original is None or original.sender != agent.id):
                    return False, "unauthorized"
        return True, "none"

    def action_plan(self, agent, action_id, params):
        duration = params.get("duration_s", params.get("timeout_s", 5.0) if action_id == "A037" else 1.0)
        return {"duration_s": float(duration), "energy_Wh": 0.0005, "bytes": 0, "cpu_s": 0.0}

    def action_effect(self, agent, action_id, params, rng):
        aid = agent.id
        mid = params.get("message_id")
        ref = params.get("_task_ref", params.get("task_ref"))
        if action_id == "A032":
            target = params.get("receiver", params.get("target", params.get("node_id")))
            if not target:
                return ActionOutcome(status="failed", failure_code="lost_link", feedback_zh="没有指定链路探测目标")
            result = self.probe_link(aid, target)
            obs = Observation(id="probe:%s:%s:%s" % (aid, target, self.sim.t), by=aid, t=self.sim.t,
                              about=[aid + "->" + target], kind="abstract_link_probe",
                              values={"link.state": result["state"], "link.goodput_bps": result["goodput_bps"],
                                      "link.latency_s": result["latency_s"], "link.loss_rate": result["loss_rate"]},
                              confidence=1.0, model=self.backend_mode)
            return ActionOutcome(status="succeeded", observations=[obs], feedback_zh="抽象链路探测完成，不保证后续消息送达")
        if action_id == "A033":
            self.configure_radio(aid, params)
            return ActionOutcome(status="succeeded", feedback_zh="无线接口已配置，恢复与否由后续传输决定")
        if action_id == "A034":
            item = self._relay_item(agent, params)
            if item is None:
                return ActionOutcome(status="failed", failure_code="actuator", feedback_zh="没有实际携带可用中继")
            pose = params.get("position", params.get("pose", self._pose(aid)))
            current = self._pose(aid)
            if pose is None or current is None or math.dist(tuple(pose[:2]), current) > float(agent.spec.get("deploy_reach_m", 5)):
                return ActionOutcome(status="failed", failure_code="blocked", feedback_zh="中继布放点不在实际可达范围")
            self.add_relay(aid, params.get("relay_id", item), pose, params.get("range_m"))
            state = self.sim.world.get(item, "item.state")
            # Preserve the catalog item machine instead of teleporting stock to placed.
            transitions = {"stock": ["reserved", "attached", "in_transit", "placed"],
                           "reserved": ["attached", "in_transit", "placed"],
                           "attached": ["in_transit", "placed"], "in_transit": ["placed"]}
            for after in transitions[state]:
                self.sim.write_truth(item, "item.state", after, "A034", actor=aid, task_ref=ref)
            return ActionOutcome(status="succeeded", truth_writes=[(item, "item.holder_id", None),
                                 (item, "entity.pose", list(pose)), (item, "radio.enabled", True),
                                 (item, "radio.relay_enabled", True)], feedback_zh="已消耗实际携带节点并布放中继")
        if action_id in ("A035", "A036"):
            receivers = params.get("receivers", params.get("receiver_ids", params.get("receiver", params.get("target"))))
            if action_id == "A035":
                mid = self.forward_message(aid, mid, receivers)
            elif params.get("retry_of"):
                mid = self.retry(params["retry_of"], receivers)
            else:
                payload = params.get("payload", {})
                size = int(params.get("size_bytes", len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))))
                mid = self.send(aid, receivers, params.get("kind", "report"), payload, size,
                                task_ref=ref, ttl_s=params.get("ttl_s"), correlation_id=params.get("correlation_id"))
            return ActionOutcome(status="succeeded", local_writes=[("message.correlation_id", mid)],
                                 feedback_zh="消息已进入网络队列；尚未视为对方收到")
        if action_id == "A037":
            result = self.await_ack(mid)
            ack = result["ack_state"]
            if ack == "pending":
                # The bounded action has actually spent its plan's waiting time.
                self.messages[mid].ack_state = ack = "timeout"
                self._retire(self.messages[mid])
                self._emit(mid, "message.ack_state", "pending", "timeout", ["F288", "A037"],
                           [aid], "业务确认在本次等待预算内未送达")
            return ActionOutcome(status="succeeded" if ack in ("acknowledged", "not_required") else "partial",
                                 failure_code="timeout" if ack == "timeout" else "none",
                                 feedback_zh="业务确认状态：" + ack)
        if action_id == "A065":
            received = (mid, aid) in self._delivered
            return ActionOutcome(status="succeeded" if received else "failed",
                                 failure_code="none" if received else "lost_link",
                                 feedback_zh="完整消息已有本地副本" if received else "消息未实际送达，禁止提前消费")
        return ActionOutcome(status="failed", failure_code="model", feedback_zh="不支持的网络操作")

    def to_dict(self):
        return {"backend_mode": self.backend_mode, "messages": [m.to_dict() for m in self.messages.values()],
                "links": list(deepcopy(self.links).values()), "relays": list(deepcopy(self.relays).values()),
                "outages": deepcopy(self.outages)}
