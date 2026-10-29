"""Agent-local evidence and belief stores; no shared or truth-backed knowledge.

Only ``prior_from_world`` crosses the initialization boundary.  It copies a
positive whitelist of public map attributes and the owner's platform telemetry;
no runtime belief operation reads a world object.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import is_dataclass
from functools import lru_cache
from math import isfinite


SOURCES = frozenset({"prior", "observation", "camera_image", "message", "inference"})
PUBLIC_CLASSES = frozenset({
    "building", "road", "tree", "tree_cluster", "water", "open_area", "slope",
    "facility", "station", "coast", "underground", "bridge",
})
# Positive whitelist: never copy arbitrary ``fields``/``attributes`` or attr.*.
# In particular there is no damage, occupancy, obstacle, hazard, live inventory,
# clear width, passability or search-coverage field in this table.
PUBLIC_ATTRIBUTES = {
    "building": frozenset({"building.use", "building.structure", "building.floors",
                            "building.capacity", "building.entrance_ids"}),
    "road": frozenset({"road.centerline", "road.width_m", "road.surface",
                        "road.from_node", "road.to_node", "attr.allowed_mobilities", "attr.professional_only"}),
    "station": frozenset({"station.kind", "station.capacity", "station.power_W",
                           "station.connector_types", "station.accepted_need_types",
                           "station.access_region"}),
    "open_area": frozenset({"region.land_use", "attr.ground_access_isolated"}),
    "facility": frozenset({"attr.blocks_ground_navigation"}), "tree": frozenset(), "tree_cluster": frozenset(),
    "water": frozenset(), "slope": frozenset(), "coast": frozenset(),
    "underground": frozenset(), "bridge": frozenset(),
}
# Successive telemetry/status snapshots are state changes, not competing claims.
_TEMPORAL_KEYS = frozenset({
    "observation.camera_image",
    "entity.pose", "human.pose", "belief.estimated_pose", "belief.team_estimates", "belief.search_status",
    "belief.updated_at_s", "belief.staleness_s", "belief.current_goal_context",
    "belief.conflict_state", "human.mobility", "human.handover_status",
    "human.support_received", "human.carrier_id", "human.destination_id",
    "human.location_context", "human.buriedness", "item.state", "item.release_state", "obstacle.state",
    "obstacle.work_remaining", "obstacle.volume_m3", "road.entry_state", "road.clear_width_m",
    "road.step_height_m", "belief.estimated_access",
})
_TEMPORAL_PREFIXES = ("platform.", "task.", "action.", "assignment.", "message.", "job.", "service.", "cache.", "equipment.", "hazard.")
_DERIVED_EVIDENCE_KINDS = frozenset({"action_feedback", "authored_inference", "localization"})

OWN_TELEMETRY = frozenset({
    "entity.pose", "platform.kind", "platform.capability_ids", "platform.equipment_ids",
    "platform.width_m", "platform.height_m", "platform.swept_radius_m",
    "platform.payload_capacity_kg", "platform.payload_used_kg", "platform.payload_item_ids",
    "platform.battery_capacity_Wh", "platform.battery_remaining_Wh", "platform.reserve_Wh",
    "platform.speed_limit_mps", "platform.energy_model", "platform.motion_state",
    "platform.health_state", "platform.current_action_id", "platform.busy_until_est_s",
    "platform.return_site_id", "platform.water_tolerance_m", "platform.concurrency_policy",
})


def camera_record_reference(row):
    """Compact output view; full rows live in result.json's camera report.

    Resolve request_id in camera_observation.observations (or pending for a
    request whose receipt has not arrived). Keep lifecycle/status fields inline.
    This is NOT a replacement for live evidence: its serialized size affects
    network transmission and evidence-processing actions.
    """
    omitted = {"receipt", "request", "source_cut", "acquired", "camera",
               "calibration", "camera_pose", "detections", "detection_observations",
               "perception_inference"}
    reference = {key: deepcopy(value) for key, value in row.items() if key not in omitted}
    reference["record_ref"] = {"file": "result.json", "field": "camera_observation",
                               "request_id": row["request_id"]}
    if "receipt" in row:
        reference["receipt"] = {key: deepcopy(value) for key, value in row["receipt"].items()
                                if key != "record"}
    return reference


def _entry_snapshot(key, entry):
    if key != "observation.camera_image" or entry is None:
        return deepcopy(entry)
    return {name: camera_record_reference(value) if name == "value" else deepcopy(value)
            for name, value in entry.items()}


def _record(value):
    if isinstance(value, dict):
        return value
    if is_dataclass(value) and hasattr(value, "__dict__"):
        # Observation.to_dict uses recursive asdict; unchanged-evidence scans
        # need only a borrowed record and should not copy its entire payload.
        return vars(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    # World entities may be dataclasses, while the frozen catalog is plain dicts.
    return vars(value)


@lru_cache(maxsize=256)
def _report_intermediates(key, before, after):
    """Legal knowledge-state projection, not evidence of physical execution."""
    if key not in {"belief.search_status", "human.handover_status", "human.mobility",
                   "item.state", "item.release_state", "obstacle.state", "road.entry_state"}:
        return ()
    if not isinstance(before, str) or not isinstance(after, str) or before == after:
        return ()
    from ..contracts import STATE_MACHINES
    machine = STATE_MACHINES.get(key, {})
    if before not in machine.get("states", ()) or after not in machine.get("states", ()):
        return ()
    adjacency = {}
    for source, target, _via in machine.get("transitions", ()):
        adjacency.setdefault(source, []).append(target)
    queue, seen = [(before, ())], {before}
    for state, path in queue:
        for target in adjacency.get(state, ()):
            if target == after:
                return path
            if target not in seen:
                seen.add(target)
                queue.append((target, path + (target,)))
    # No legal path is invented.  A genuinely incompatible report remains a
    # visible contract diagnostic rather than being coerced to a valid value.
    return ()


class BeliefStore:
    """A holder's private entries, evidence and explicit conflicting reports."""

    def __init__(self, agent_id):
        self.agent_id = str(agent_id)
        self.entries: dict[tuple[str, str], dict] = {}
        # put() replaces entries and per-entity groups, so a borrowed grouped
        # snapshot keeps the same membership/entry references as before.
        self._entities: dict[str, dict] = {}
        self._entity_order: dict[str, int] = {}
        self._entity_kinds: dict[str, set[str]] = {}
        self._entities_by_kind: dict[str, set[str]] = {}
        self._kind_selections: dict[tuple[str, ...], tuple[str, ...]] = {}
        self.evidence: dict[str, object] = {}
        self.received_messages = 0
        self._conflicts: dict[tuple[str, str], dict] = {}
        self._reported: dict[tuple[str, str], dict] = {}
        self._reported_evidence: dict[str, object] = {}
        # Evidence writes go through add_evidence; ACKs refresh only their IDs.
        # Keep pending kinds separate so excluded archives never join scans.
        self._evidence_order: dict[str, int] = {}
        self._pending_evidence: dict[str, dict] = {}
        self._pending_evidence_by_kind: dict[object, dict[str, dict]] = {}
        # Task generation consumes/clears this independent change journal.
        self._taskgen_evidence_dirty: dict[str, None] = {}
        # Revisions are local invalidation tokens, never world versions.
        self.revision = 0

    def get(self, eid, key):
        """Return the local entry, never a value looked up in world truth."""
        return self.entries.get((eid, key))

    def put(self, sim, eid, key, value, confidence, source,
            evidence_ids=(), from_agent=None, *, timestamp=None,
            origin_source=None, origin_agent=None, state_projection=False):
        """Replace a local entry only when its content/provenance changes.

        An unchanged telemetry poll is not new evidence and does not refresh its
        timestamp or allocate event snapshots.  Callers integrating a genuinely
        newer observation may supply its evidence ``timestamp`` explicitly.
        ``source`` describes local acquisition; origin fields retain provenance
        across a delivered message without relabeling it a local observation.
        """
        if source not in SOURCES or origin_source is not None and origin_source not in SOURCES:
            raise ValueError(f"未知认知来源 / unknown belief source: {source}")
        confidence = float(confidence)
        if not isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("置信度须在0到1之间 / confidence must be in [0, 1]")
        now = float(sim.t) if sim is not None else 0.0
        ids = list(dict.fromkeys(evidence_ids or ()))
        reports = [_record(self.evidence[str(item)]) for item in ids if str(item) in self.evidence]
        if source in {"observation", "camera_image"} or source == "message" and origin_source in {"observation", "camera_image"}:
            if reports and all(report.get("kind") in _DERIVED_EVIDENCE_KINDS for report in reports):
                if source in {"observation", "camera_image"}:
                    source = "inference"
                else:
                    origin_source = "inference"
            # A041 may finish after capture.  Evidence age is its original t,
            # not the integration/delivery time.  This consults local evidence
            # only; unsupported/legacy timestamps remain explicitly clock-dated.
            if timestamp is None:
                times = [report["t"] for report in reports
                         if isinstance(report.get("t"), (int, float))
                         and isfinite(report["t"]) and report["t"] >= 0]
                if times:
                    timestamp = max(times)
        when = now if timestamp is None else float(timestamp)
        if not isfinite(when) or when < 0:
            raise ValueError("证据时间须为非负有限数 / invalid evidence timestamp")
        ref = (str(eid), str(key))
        before = self.entries.get(ref)
        # Check the prior boundary before copying anything or recording conflicts.
        if before is not None and before["source"] != "prior" and source == "prior":
            return before
        if before is not None and ids and ids == before.get("evidence_ids") and source != "message":
            if ref[1] == "belief.updated_at_s":
                return before
            if (ref[1] == "belief.search_status" and value == "in_progress"
                    and before["value"] in {"partial", "complete", "inconclusive"} and when <= before["t"]):
                # A041 prepends in_progress even when replaying the same final
                # report.  Do not fabricate another search pass from that replay.
                for report in reports:
                    values = report.get("values", {})
                    groups = values.get("entities") if isinstance(values, dict) else None
                    nested = groups.get(ref[0], {}) if isinstance(groups, dict) else {}
                    reported_state = values.get(ref[1], nested.get(ref[1])) if isinstance(values, dict) and isinstance(nested, dict) else None
                    if reported_state == before["value"]:
                        return before
        metadata = {"confidence": confidence, "source": source,
                    "evidence_ids": ids, "from_agent": from_agent}
        if origin_source is not None:
            metadata["origin_source"] = origin_source
        if origin_agent is not None:
            metadata["origin_agent"] = str(origin_agent)
        if state_projection:
            metadata["state_projection"] = True
        if (before is not None and before["value"] == value
                and all(before.get(name) == item for name, item in metadata.items())
                and all(name in metadata or name not in before
                        for name in ("origin_source", "origin_agent", "state_projection"))
                and (timestamp is None or before["t"] == when)):
            return before
        entry = {"value": deepcopy(value), **metadata, "t": when}
        # A prior refinement and a successive state snapshot are not competing
        # claims.  Keep distinct non-temporal alternatives, including late ones.
        if (before is not None and before["source"] != "prior" and source != "prior"
                and ref[1] not in _TEMPORAL_KEYS and not ref[1].startswith(_TEMPORAL_PREFIXES)
                and before["value"] != value):
            conflict = self._conflicts.get(ref)
            alternatives = conflict.get("alternatives") if conflict is not None else None
            if alternatives is None:
                alternatives = [deepcopy(before)]
            if conflict is not None and conflict.get("state") == "resolved" and entry in alternatives:
                return before
            if entry not in alternatives:
                self._conflicts[ref] = {
                    "eid": ref[0], "key": ref[1], "state": "disputed", "t": now,
                    "alternatives": alternatives + [deepcopy(entry)],
                }
        # Delayed reports/replayed old observations retain their evidence but
        # cannot roll state back.  Same-time transport cannot displace direct
        # local evidence.  Replaying a reviewed alternative does not reopen it.
        if (before is not None and (when < before["t"] or
                source == "message" and when == before["t"]
                and before["source"] in {"observation", "camera_image", "inference"})):
            return before
        self.entries[ref] = entry
        eid, key = ref
        fields = {**self._entities.get(eid, {}), key: entry}
        self._entities[eid] = fields
        if eid not in self._entity_order:
            self._entity_order[eid] = len(self._entity_order)
        if before is None or key in {"entity.kind", "entity.class"}:
            # Prefix membership is deliberately inclusive: consumers retain
            # their original kind predicates (camera and taskgen differ).
            kinds = {name.split(".", 1)[0] for name in fields if "." in name}
            for name in ("entity.kind", "entity.class"):
                declared = fields.get(name, {}).get("value")
                if isinstance(declared, str):
                    kinds.add(declared)
            previous_kinds = self._entity_kinds.get(eid, set())
            if kinds != previous_kinds:
                for kind in previous_kinds - kinds:
                    self._entities_by_kind[kind].remove(eid)
                for kind in kinds - previous_kinds:
                    self._entities_by_kind.setdefault(kind, set()).add(eid)
                self._entity_kinds[eid] = kinds
                changed_kinds = kinds ^ previous_kinds
                for selection in tuple(self._kind_selections):
                    if any(kind in changed_kinds for kind in selection):
                        del self._kind_selections[selection]
        self.revision += 1
        if sim is not None:
            # Detached output snapshots only for an actual update. Camera rows
            # use report references here, never in live entries/report_delta:
            # shrinking wire payloads would change network costs and delivery.
            sim.emit(
                layer="knowledge", family="belief.update", kind="belief.update",
                grounding="catalog", refs=["A041", "F188", "F189", "F190"],
                subject=ref[0], field=ref[1], before=_entry_snapshot(ref[1], before),
                after=_entry_snapshot(ref[1], entry),
                actor=self.agent_id, visible_to=[self.agent_id],
                text_zh=(f"{self.agent_id} 依据已送达最终状态投影未独立观测的中间态"
                         if state_projection else f"{self.agent_id} 更新对 {ref[0]} 的本地认知"),
            )
        return entry

    def merge_report_entry(self, sim, entry, from_agent, evidence_id=None):
        """Merge one delivered flattened entry, preserving evidence age/origin.

        Only this explicit delivered payload is read.  Origin is a reported
        claim, not authenticated truth; immediate transport remains ``message``.
        """
        if not isinstance(entry, dict) or entry.get("eid") is None or not entry.get("key"):
            return None
        # Match A041 integration: an unmeasured/indeterminate field is retained
        # in evidence, not a newer physical state that erases a measured value.
        # None remains meaningful (for example, releasing human.carrier_id).
        if entry.get("value") == "unknown":
            return self.get(str(entry["eid"]), str(entry["key"]))
        original_source = entry.get("origin_source", entry.get("source", "message"))
        if original_source not in SOURCES:
            original_source = "message"
        original_agent = entry.get("origin_agent") or entry.get("from_agent") or from_agent
        timestamp = entry.get("t")
        ids = entry.get("evidence_ids") or ([evidence_id] if evidence_id else [])
        if timestamp is None and original_source in {"observation", "camera_image"}:
            reports = [_record(self.evidence[str(item)]) for item in ids if str(item) in self.evidence]
            times = [report["t"] for report in reports if isinstance(report.get("t"), (int, float))
                     and isfinite(report["t"]) and report["t"] >= 0]
            if times:
                timestamp = max(times)
        previous = self.get(str(entry["eid"]), str(entry["key"]))
        when = float(timestamp) if timestamp is not None else float(sim.t) if sim is not None else 0.0
        timestamp = when  # projections and final entry must share one evidence time
        if (previous is not None and isfinite(when) and when >= previous["t"]
                and not (when == previous["t"] and previous["source"] in {"observation", "camera_image", "inference"})
                and isinstance(previous["value"], str) and isinstance(entry.get("value"), str)):
            for intermediate in _report_intermediates(str(entry["key"]), previous["value"], entry["value"]):
                self.put(sim, entry["eid"], entry["key"], intermediate, entry.get("confidence", .5),
                         "message", ids, from_agent, timestamp=when, origin_source=original_source,
                         origin_agent=original_agent, state_projection=True)
        # Legacy reports lacking both entry/evidence time stay delivery-dated.
        return self.put(sim, entry["eid"], entry["key"], entry.get("value"),
                        entry.get("confidence", .5), "message", ids,
                        from_agent, timestamp=timestamp, origin_source=original_source,
                        origin_agent=original_agent)

    def add_evidence(self, sim, obs_or_report):
        """Store a local copy only; A041/A040 perform the actual belief writes."""
        report = _record(obs_or_report)
        evidence_id = report.get("id") or report.get("evidence_id")
        if not evidence_id:
            evidence_id = f"{self.agent_id}:evidence:{len(self.evidence) + 1}"
        evidence_id = str(evidence_id)
        previous = self.evidence.get(evidence_id)
        if previous is not None and _record(previous) == report:
            return evidence_id
        self.evidence[evidence_id] = deepcopy(obs_or_report)
        if evidence_id not in self._evidence_order:
            self._evidence_order[evidence_id] = len(self._evidence_order)
        self._refresh_pending_evidence(evidence_id)
        self._taskgen_evidence_dirty[evidence_id] = None
        return evidence_id

    def _refresh_pending_evidence(self, evidence_id):
        """Reconcile one stored record against its exact last acknowledgment."""
        previous = self._pending_evidence.pop(evidence_id, None)
        if previous is not None:
            kind = previous.get("kind")
            pending = self._pending_evidence_by_kind[kind]
            del pending[evidence_id]
            if not pending:
                del self._pending_evidence_by_kind[kind]
        item = self.evidence.get(evidence_id)
        if item is None:
            return
        record = _record(item)
        if type(record) is not dict or "id" not in record:
            record = dict(record)
            record.setdefault("id", evidence_id)
        if self._reported_evidence.get(evidence_id) != record:
            self._pending_evidence[evidence_id] = record
            self._pending_evidence_by_kind.setdefault(record.get("kind"), {})[evidence_id] = record

    def entities(self, *, copy=True):
        """Group local entries; default snapshots remain detached.

        ``copy=False`` is for read-only policy scans within one synchronous hook;
        consumers must not mutate or retain the borrowed entries.
        """
        if copy:
            # Copy each entry separately, as in the original grouped snapshot.
            return {eid: {key: deepcopy(entry) for key, entry in fields.items()}
                    for eid, fields in self._entities.items()}
        return dict(self._entities)

    def entity_ids(self, *kinds):
        """Ordered candidates by declared kind/class or known field prefix.

        These are inclusive buckets, not a replacement for a caller's exact
        class predicate. Union order is first entity insertion, even if a kind
        is learned later or changes. The immutable selection survives unrelated
        telemetry/evidence writes.
        """
        selected = self._kind_selections.get(kinds)
        if selected is None:
            ids = set()
            for kind in kinds:
                ids.update(self._entities_by_kind.get(kind, ()))
            selected = tuple(sorted(ids, key=self._entity_order.__getitem__))
            self._kind_selections[kinds] = selected
        return selected

    def staleness(self, t):
        return {ref: max(0.0, float(t) - entry["t"])
                for ref, entry in self.entries.items()}

    def conflicts(self):
        return [deepcopy(value) for _, value in sorted(self._conflicts.items())
                if value["state"] in {"disputed", "insufficient"}]

    def resolve_conflict(self, eid, key, sufficient=True):
        """Mark a reviewed conflict; the caller still integrates evidence via A040."""
        conflict = self._conflicts.get((eid, key))
        if conflict is not None:
            conflict["state"] = "resolved" if sufficient else "insufficient"

    def report_delta(self, exclude_kinds=()):
        """Return undelivered local entries/evidence without marking them sent.

        Calling this is not delivery: a lost report must remain eligible for a
        retry.  Acknowledgments must pass their exact payload to mark_delivered.
        Neither priors nor received entries are promoted to local observations.
        ``exclude_kinds`` keeps local-only archives out before any payload copy;
        excluding evidence does not falsely acknowledge or remove it.
        """
        excluded = frozenset(exclude_kinds)
        # Filter before sorting: acknowledged history need not participate in
        # the report's ordering work. Still compare contents (get() is mutable).
        changed = ((ref, entry) for ref, entry in self.entries.items()
                   if entry["source"] in {"observation", "camera_image", "inference"}
                   and self._reported.get(ref) != entry)
        entries = [{"eid": eid, "key": key, **deepcopy(entry)}
                   for (eid, key), entry in sorted(changed)]
        # Kind buckets avoid both acknowledged history and excluded pending
        # archives. Replacements/late ACKs may reinsert a candidate, so restore
        # the evidence dictionary's original insertion order before copying.
        pending = [(self._evidence_order[evidence_id], record)
                   for kind, records in self._pending_evidence_by_kind.items()
                   if kind not in excluded
                   for evidence_id, record in records.items()]
        evidence = [deepcopy(dict(record)) for _, record in sorted(pending)]
        return {"entries": entries, "evidence": evidence}

    def mark_reported(self, refs=None):
        """Legacy explicit acknowledgment of current entries, not send intent."""
        refs = self.entries if refs is None else refs
        for ref in refs:
            ref = tuple(ref)
            if ref in self.entries and self._reported.get(ref) != self.entries[ref]:
                self._reported[ref] = deepcopy(self.entries[ref])

    def mark_delivered(self, entries, evidence_ids=(), *, evidence=()):
        """Acknowledge exact flattened entries and, preferably, evidence snapshots.

        A newer local entry must remain local-only.  Out-of-order old ACKs may
        cause harmless retransmission, but never mark unsent newer content as
        delivered.  Evidence IDs alone are safe for immutable observation IDs;
        use ``evidence=payload['evidence']`` for exact snapshot acknowledgment.
        """
        for flattened in entries:
            if not isinstance(flattened, dict) or "eid" not in flattened or "key" not in flattened:
                continue
            ref = (str(flattened["eid"]), str(flattened["key"]))
            snapshot = {key: value for key, value in flattened.items() if key not in {"eid", "key"}}
            if self._reported.get(ref) != snapshot:
                self._reported[ref] = deepcopy(snapshot)
        for item in evidence:
            record = _record(item)
            evidence_id = record.get("id") or record.get("evidence_id")
            if evidence_id is not None and self._reported_evidence.get(str(evidence_id)) != record:
                self._reported_evidence[str(evidence_id)] = deepcopy(record)
                self._refresh_pending_evidence(str(evidence_id))
        for evidence_id in evidence_ids:
            item = self.evidence.get(str(evidence_id))
            if item is not None:
                record = dict(_record(item))
                record.setdefault("id", str(evidence_id))
                if self._reported_evidence.get(str(evidence_id)) != record:
                    self._reported_evidence[str(evidence_id)] = deepcopy(record)
                    self._refresh_pending_evidence(str(evidence_id))

    def to_view(self):
        return {
            "entries": [{"eid": eid, "key": key, **_entry_snapshot(key, entry)}
                        for (eid, key), entry in sorted(self.entries.items())],
            "evidence_count": len(self.evidence),
            "received_messages": self.received_messages,
            "local_only": [[eid, key] for (eid, key), entry in sorted(self.entries.items())
                           if entry["source"] in {"observation", "camera_image", "inference"}
                           and self._reported.get((eid, key)) != entry],
            "conflicts": self.conflicts(),
        }


def _center(geometry):
    coords = geometry.get("coords", geometry.get("coordinates", []))
    if not coords:
        return None
    if isinstance(coords[0], (int, float)):
        return list(coords[:2])
    while coords and isinstance(coords[0], (list, tuple)) and coords[0] and isinstance(coords[0][0], (list, tuple)):
        coords = coords[0]
    points = [point for point in coords if isinstance(point, (list, tuple)) and len(point) >= 2]
    if not points:
        return None
    return [sum(point[i] for point in points) / len(points) for i in (0, 1)]


def prior_from_world(world, agent_spec, public_objectives, *, sim=None):
    """Build a detached, strictly public initial store, plus own telemetry.

    ``agent_spec`` is a dict augmented with ``id`` and ``platform_eid`` by the
    builder.  Reading entity containers happens only here at initialization;
    geometry and whitelisted static fields are copied, not retained as aliases.
    The world graph, hidden objectives and dynamic entities are never copied.
    """
    if not isinstance(agent_spec, dict):
        raise TypeError("agent_spec must be a dict")
    agent_id = agent_spec.get("id", agent_spec.get("agent_id", agent_spec.get("kind", "agent")))
    store = BeliefStore(agent_id)
    entities = getattr(world, "entities", {})
    rows = entities.values() if isinstance(entities, dict) else entities
    for raw in rows:
        entity = _record(raw)
        cls = entity.get("cls", entity.get("class"))
        if cls not in PUBLIC_CLASSES:
            continue
        eid = entity["id"]
        geometry = entity.get("geometry") or {}
        store.put(sim, eid, "entity.kind", cls, 1.0, "prior")
        store.put(sim, eid, "entity.geometry", geometry, 1.0, "prior")
        point = _center(geometry)
        if point is not None:
            store.put(sim, eid, "entity.pose", point, 1.0, "prior")
        fields = entity.get("fields", entity.get("attributes", {}))
        for key in sorted(PUBLIC_ATTRIBUTES[cls]):
            if key in fields:
                store.put(sim, eid, key, fields[key], 1.0, "prior")

    platform_eid = agent_spec.get("platform_eid", str(agent_id))
    # Exactly one platform is consulted.  Never seed another agent's telemetry.
    for key in sorted(OWN_TELEMETRY):
        value = world.get(platform_eid, key, None)
        if value is not None:
            store.put(sim, platform_eid, key, value, 1.0, "prior")

    if agent_spec.get("kind") == "coordinator":
        profile = getattr(world, "profile", {})
        brief = profile.get("public_brief_zh", "") if isinstance(profile, dict) else ""
        store.put(sim, "mission", "public_brief", brief, 1.0, "prior")
        objectives = []
        for raw in public_objectives or ():
            objective = _record(raw)
            if objective.get("public") is not True:
                continue
            public = {key: deepcopy(objective[key]) for key in
                      ("id", "kind", "target", "desired", "priority", "public", "rationale_zh", "refs")
                      if key in objective}
            objectives.append(public)
            store.put(sim, f"objective:{objective['id']}", "objective.public_record", public, 1.0, "prior")
        store.put(sim, "mission", "public_objectives", objectives, 1.0, "prior")
    return store
