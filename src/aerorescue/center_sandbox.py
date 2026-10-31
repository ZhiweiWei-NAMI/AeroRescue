"""Center sandbox: a LangGraph factory that plans and dispatches rescue tasks.

The graph is a pure function of the delivered situation report plus the model
reply: it never reads kernel state, and every effect is a typed ``outputs``
command handled by :class:`rescue_scenes.engine.RescueWorld`.

Input situation (``trigger.payload.data_json`` or a plain mapping)::

    {"now_s": 12.0, "coordinator": "coordinator-1",
     "reports": [{"report_id": "r1", "kind": "person", "target": "human-1",
                  "status": "confirmed", "task_type": "T029", "priority": 80}],
     "candidates": [...optional explicit candidate rows...],
     "belief": {...}, "vehicles": [{"id": "uav-1", "capabilities": ["C01"],
                  "battery_remaining_Wh": 900, "payload_kg": 0,
                  "payload_capacity_kg": 2.5}]}

Output: ``rescue.command.propose_task`` (A050) per candidate and
``rescue.command.assign_task`` (A052) per accepted assignment.
"""
import json
import math

from aeroagentsim.agents.langgraph_client import JournalClient
from rescue_scenes.core.agents import taskgen
from rescue_scenes.core.catalog import load

from .skills import CHANNEL_MODES, CommChannelGate, MemoryEnvelope, sequence_skill

PROPOSE = 'rescue.command.propose_task'
ASSIGN = 'rescue.command.assign_task'
TARGET = 'rescue'
CATALOG = load()
# Report kinds that map to a catalog task type without re-running taskgen predicates.
REPORT_TASKS = {'person': 'T029', 'casualty': 'T029', 'survivor': 'T029', 'aid': 'T031',
                'building': 'T010', 'damage': 'T010', 'structure': 'T010',
                'road': 'T021', 'debris': 'T021', 'obstacle': 'T021'}

SYSTEM_PROMPT = """You are the rescue command-center sandbox. Assign each pending
candidate task to at most one available vehicle, respecting capability, battery and
payload constraints. Candidates carry a catalog task type, a target, required
capabilities (all-of) and optional payload_kg; vehicles carry capabilities,
battery_remaining_Wh and payload capacity. Never assign a vehicle that lacks any
required capability, is below the battery floor, or cannot carry the payload.
Return ONLY one JSON object:
{"assignments": [{"candidate": "<candidate id>", "vehicle": "<vehicle id>"}], "reason": "<short reason>"}
Use [] when no vehicle can take the work. Assign each candidate and each vehicle at
most once; do not invent ids."""


def _situation(state):
    """The delivered situation, never kernel internals."""
    trigger = state.get('trigger') or {}
    payload = trigger.get('payload') or {}
    data = payload.get('data_json') if isinstance(payload, dict) else None
    if isinstance(data, str) and data.strip():
        return json.loads(data)
    if isinstance(payload, dict) and payload and 'data_json' not in payload:
        return payload
    return state.get('situation') or {}


def _skill(task_type):
    """The main method's skill label; the center never invents its own."""
    try:
        return sequence_skill(CATALOG.task_methods(task_type)[0], CATALOG)
    except (AttributeError, IndexError, KeyError, ValueError):
        return 'search_and_assess'


def _candidate(row, index, source=''):
    typ = row.get('task_type') or row.get('type')
    if typ not in taskgen.TASK_NAMES:
        raise ValueError(f'candidate {index} has no catalog task type: {typ!r}')
    target = row.get('target')
    if target is None or str(target) == '':
        raise ValueError(f'candidate {index} has no target')
    rule = taskgen.RULES_BY_TASK.get(typ)
    priority = row.get('priority', rule.priority if rule else 50)
    origin = row.get('origin') if isinstance(row.get('origin'), dict) else {}
    return {'id': str(row.get('candidate_id') or row.get('id') or f'cand-{index:03d}'),
            'type': typ, 'target': str(target), 'priority': float(priority),
            'name_zh': taskgen.TASK_NAMES[typ],
            'why_zh': str(row.get('why_zh') or row.get('reason') or origin.get('why_zh') or
                          f'{typ} {taskgen.TASK_NAMES[typ]} at {target}'),
            'rule_id': str(row.get('rule_id') or origin.get('rule_id') or
                           (rule.id if rule else 'CENTER-SANDBOX')),
            'capabilities': sorted(set(row.get('capabilities') or []) |
                                   set(CATALOG.tasks[typ].get('capabilities') or [])),
            'skill_name': _skill(typ),
            'payload_kg': row.get('payload_kg'),
            'position_enu': row.get('position_enu'),
            'source_report': row.get('report_id') or source}


def derive_candidates(situation, runtime=None, sim=None):
    """Explicit candidates, real ``taskgen.candidates``, or delivered reports.

    With a live holder (``runtime`` + ``sim``) the center uses the same pure
    candidate rules the edge runtime uses, so a sandbox proposal cannot invent a
    task type the catalog rules would not raise from the same local evidence.
    """
    rows = situation.get('candidates')
    if rows is None and runtime is not None and sim is not None:
        rows = [dict(row, task_type=row.get('type')) for row in taskgen.candidates(runtime, sim)]
    if rows is None:
        rows = []
        for report in situation.get('reports') or []:
            kind = str(report.get('kind') or '').lower()
            typ = report.get('task_type') or REPORT_TASKS.get(kind)
            if typ is None or report.get('status') in {'rejected', 'resolved', 'duplicate'}:
                continue
            rows.append({**report, 'type': typ, 'report_id': report.get('report_id')})
    candidates, seen, ids = [], set(), set()
    for index, row in enumerate(rows):
        candidate = _candidate(row, index)
        key = (candidate['type'], candidate['target'])
        if key in seen:
            continue
        if candidate['id'] in ids:
            raise ValueError(f"duplicate candidate id: {candidate['id']}")
        ids.add(candidate['id'])
        seen.add(key)
        candidates.append(candidate)
    return candidates


def generate_candidates(state, runtime=None, sim=None):
    situation = _situation(state)
    return {'situation': situation,
            'candidates': derive_candidates(situation, runtime=runtime, sim=sim)}


def _vehicles(situation):
    rows = situation.get('vehicles') or situation.get('fleet') or []
    vehicles, seen = [], set()
    for index, row in enumerate(rows):
        identity = str(row.get('id') or row.get('vehicle') or '')
        if not identity or identity in seen:
            continue
        seen.add(identity)
        vehicles.append({'id': identity, 'capabilities': list(row.get('capabilities') or []),
                         'battery_remaining_Wh': row.get('battery_remaining_Wh'),
                         'payload_kg': row.get('payload_kg', 0.0),
                         'payload_capacity_kg': row.get('payload_capacity_kg'),
                         'available': row.get('available', True)})
    return vehicles


def build_graph(client: JournalClient, options: dict):
    from langgraph.graph import END, START, StateGraph

    battery_floor = float(options.get('battery_min_Wh', 20.0))
    center = options.get('coordinator', 'coordinator-1')
    target = options.get('engine', TARGET)
    # Downlink byte budget and channel mode for the memory envelopes.
    channel_mode = options.get('channel_mode', 'B11')
    if channel_mode not in CHANNEL_MODES:
        raise ValueError(f'unknown channel mode: {channel_mode!r}')
    b_down_bytes = int(options.get('b_down_bytes', 4096))
    if b_down_bytes < 0:
        raise ValueError('b_down_bytes must be nonnegative')
    memory_ttl_s = float(options.get('memory_ttl_s', 180.0))
    runtime, sim = options.get('runtime'), options.get('sim')

    def plan_observation(state):
        situation = state['situation']
        now_s = float(situation.get('now_s') or 0.0)
        gate = CommChannelGate(channel_mode, b_down_bytes=b_down_bytes)
        memory, dropped = [], 0
        for candidate in state['candidates']:
            payload = {'op': 'candidate', 'candidate_id': candidate['id'],
                       'type': candidate['type'], 'target': candidate['target'],
                       'priority': candidate['priority'], 'why_zh': candidate['why_zh'],
                       'skill_name': candidate.get('skill_name')}
            envelope = MemoryEnvelope(
                envelope_id=f"mem-{candidate['id']}", sender=center, receivers=(target,),
                kind='candidate', payload=payload,
                size_bytes=len(json.dumps(payload, ensure_ascii=False).encode('utf-8')),
                created_s=now_s, valid_until_s=now_s + memory_ttl_s, channel=channel_mode)
            if gate.admit('down', envelope.size_bytes, now_s):
                memory.append(envelope.to_dict())
            else:
                dropped += 1
        return {'now_s': situation.get('now_s'), 'candidates': state['candidates'],
                'vehicles': _vehicles(situation),
                # Undelivered plans: the edge reads them only after the kernel's
                # Network._deliver stamps delivered_s.
                'memory': memory, 'memory_budget': gate.to_dict(), 'memory_dropped': dropped,
                'reports': [row for row in situation.get('reports') or []
                            if row.get('status') not in {'rejected', 'duplicate'}]}

    async def plan_and_dispatch(state):
        observation = plan_observation(state)
        candidates = {row['id']: row for row in observation['candidates']}
        vehicles = {row['id']: row for row in observation['vehicles']}

        def validate(value):
            if not isinstance(value, dict) or set(value) != {'assignments', 'reason'}:
                raise ValueError('plan requires exactly assignments and reason')
            if not isinstance(value['assignments'], list):
                raise ValueError('assignments must be a list')
            used_candidates, used_vehicles = set(), set()
            for row in value['assignments']:
                if not isinstance(row, dict) or set(row) != {'candidate', 'vehicle'}:
                    raise ValueError('assignment requires exactly candidate and vehicle')
                candidate, vehicle = candidates.get(row['candidate']), vehicles.get(row['vehicle'])
                if candidate is None:
                    raise ValueError(f"unknown candidate: {row['candidate']}")
                if vehicle is None:
                    raise ValueError(f"unknown vehicle: {row['vehicle']}")
                if row['candidate'] in used_candidates or row['vehicle'] in used_vehicles:
                    raise ValueError('candidate and vehicle may be assigned once per plan')
                used_candidates.add(row['candidate'])
                used_vehicles.add(row['vehicle'])
                if vehicle['available'] is not True:
                    raise ValueError(f"vehicle is unavailable: {vehicle['id']}")
                for name, number in (('battery_remaining_Wh', vehicle['battery_remaining_Wh']),
                                     ('payload_kg', vehicle['payload_kg']),
                                     ('payload_capacity_kg', vehicle['payload_capacity_kg']),
                                     ('required_payload_kg', candidate['payload_kg'])):
                    if number is not None and (type(number) not in (int, float)
                            or not math.isfinite(number) or number < 0):
                        raise ValueError(f'{name} must be finite and nonnegative')
                required = set(candidate['capabilities'])
                missing = required - set(vehicle['capabilities'])
                if missing:
                    raise ValueError(f"vehicle {vehicle['id']} lacks required capability "
                                     f"{','.join(sorted(missing))} of {candidate['id']} (ALL-of)")
                battery = vehicle['battery_remaining_Wh']
                if battery is not None and float(battery) < battery_floor:
                    raise ValueError(f"vehicle {vehicle['id']} is below the battery floor")
                payload = candidate['payload_kg']
                capacity = vehicle['payload_capacity_kg']
                if payload is not None and capacity is not None:
                    free = float(capacity) - float(vehicle['payload_kg'] or 0.0)
                    if float(payload) > free:
                        raise ValueError(f"vehicle {vehicle['id']} cannot carry {payload} kg")

        plan = await client.ask_json('center_plan', SYSTEM_PROMPT, observation, validate)
        outputs = []
        for candidate in observation['candidates']:
            outputs.append({'kind': 'command', 'schema': PROPOSE, 'target': target, 'payload': {
                'actor': center, 'type': candidate['type'], 'target': candidate['target'],
                'priority': candidate['priority'], 'candidate_id': candidate['id'],
                'why_zh': candidate['why_zh'],
                'origin_json': json.dumps({'rule_id': candidate['rule_id'],
                                           'why_zh': candidate['why_zh'],
                                           'belief_refs': [candidate['source_report']]
                                           if candidate['source_report'] else []},
                                          ensure_ascii=False)}})
        for row in plan['assignments']:
            candidate = candidates[row['candidate']]
            outputs.append({'kind': 'command', 'schema': ASSIGN, 'target': target, 'payload': {
                'actor': center, 'agent_id': row['vehicle'], 'type': candidate['type'],
                'target': candidate['target'], 'candidate_id': candidate['id'],
                'task_id': None}})
        return {'outputs': outputs, 'plan': plan}

    def generate(state):
        return generate_candidates(state, runtime=runtime, sim=sim)

    graph = StateGraph(dict)
    graph.add_node('generate_candidates', generate)
    graph.add_node('plan_and_dispatch', plan_and_dispatch)
    graph.add_edge(START, 'generate_candidates')
    graph.add_edge('generate_candidates', 'plan_and_dispatch')
    graph.add_edge('plan_and_dispatch', END)
    return graph
