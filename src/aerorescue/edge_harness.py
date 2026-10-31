"""Edge embodiment harness: one local model chooses the next executable action.

The graph is a LangGraph factory (``build_graph(client, options)``) over the
vehicle's own observation: telemetry, camera perception, the delivered memory
envelopes and the tasks assigned by the center. It returns typed ``outputs``:
``rescue.command.submit_action`` for native catalog actions (A001-A066) and the
mapped ``adapters.px4.*`` command for flight directives.
"""
from copy import deepcopy
import json
import math

from aeroagentsim.agents.langgraph_client import JournalClient
from rescue_scenes.px4_bridge import command_schemas

from .skills import VALID_ACTION_IDS, MemoryEnvelope, CommChannelGate

SUBMIT = 'rescue.command.submit_action'
TARGET = 'rescue'

SYSTEM_PROMPT = """You are one rescue vehicle's embodied harness. You see only your
own telemetry, your camera detections and the tasks the center assigned to you.
Choose the next executable actions. Allowed actions:
- a native catalog action A001..A066 with its params (for example
  {"action": "A001", "params": {"goal": [e, n, u]}} or
  {"action": "A009", "params": {"target": "human-1"}});
- a PX4 flight command from this exact list: {px4_commands}
  with the native payload (goto needs position_enu [e,n,u] and yaw_deg,
  north=0 clockwise; entity is the AAS entity id, e.g. "uav-1").
Respect battery and payload limits; do not repeat a completed task; prefer
inspection before contact and contact before transport.
You may instead pick a delivered decision candidate by id:
{"candidate": "<candidate id>", "params": {...optional overrides...}}.
When you pick an action yourself, echo the candidate's purpose, method_seq and
step_index if the observation gave them, so the runtime can verify the step.
Return ONLY one JSON object:
{"actions": [{"action": "<id>", "params": {...}, "purpose": "method|approach",
              "method_seq": "Q0xx", "step_index": 0}], "reason": "<short reason>"}
Use [] when no action is currently executable."""


def _situation(state):
    trigger = state.get('trigger') or {}
    payload = trigger.get('payload') or {}
    data = payload.get('data_json') if isinstance(payload, dict) else None
    if isinstance(data, str) and data.strip():
        return json.loads(data)
    if isinstance(payload, dict) and payload and 'data_json' not in payload:
        return payload
    return state.get('situation') or {}


def _observation(situation, vehicle, options):
    telemetry = situation.get('telemetry') or {}
    perception = situation.get('perception') or {}
    now_s = float(situation.get('now_s') or 0.0)
    # Only delivered, unexpired, intact envelopes are edge-readable.
    envelopes, unreadable = [], 0
    for row in situation.get('memory') or []:
        try:
            envelope = MemoryEnvelope.from_dict(row)
            readable = (envelope.readable(now_s, receiver=vehicle)
                        and CommChannelGate(envelope.channel).allows('down'))
        except (KeyError, TypeError, ValueError, OverflowError):
            readable = False
        if readable:
            envelopes.append(envelope.to_dict())
        else:
            unreadable += 1
    return {'vehicle': vehicle,
            'now_s': situation.get('now_s'),
            'telemetry': telemetry,
            'candidates': situation.get('candidates') or [],
            'memory': envelopes,
            'memory_unreadable': unreadable,
            'perception': {'detections': perception.get('detections') or [],
                           'observations': perception.get('observations') or [],
                           'sensing': perception.get('sensing')},
            'assignments': situation.get('assignments') or [],
            'capabilities': situation.get('capabilities') or options.get('capabilities') or [],
            'px4_vehicle': situation.get('px4_vehicle') or options.get('px4_vehicle'),
            'battery_min_Wh': float(options.get('battery_min_Wh', 20.0))}


def _task_ref(assignments, params):
    target = str(params.get('target') or params.get('target_id') or '')
    matches = [row for row in assignments if row.get('task_id') and target
               and str(row.get('target')) == target
               and row.get('status') not in {'completed', 'failed', 'canceled'}]
    if len(matches) != 1:
        return None
    return [str(matches[0]['task_id']), int(matches[0].get('rev', 1))]


def build_graph(client: JournalClient, options: dict):
    from langgraph.graph import END, START, StateGraph

    vehicle = options.get('vehicle', 'uav-1')
    px4_target = options.get('px4_engine', 'px4')
    engine = options.get('engine', TARGET)
    # The installed adapter owns its backend name; options may override it.
    px4_schemas = command_schemas(options.get('px4_command_prefix'))
    prompt = SYSTEM_PROMPT.replace('{px4_commands}', ', '.join(sorted(px4_schemas)))

    async def decide(state):
        situation = _situation(state)
        observation = _observation(situation, vehicle, options)
        assignments = observation['assignments']
        candidates = {str(row.get('candidate_id')): row
                      for row in observation['candidates'] if isinstance(row, dict)}

        def validate(value):
            if not isinstance(value, dict) or set(value) != {'actions', 'reason'}:
                raise ValueError('decision requires exactly actions and reason')
            if not isinstance(value['actions'], list):
                raise ValueError('actions must be a list')
            for index, row in enumerate(value['actions']):
                if not isinstance(row, dict) or not ({'action', 'candidate'} & set(row)):
                    raise ValueError('each action requires an action or candidate')
                if not set(row) <= {'action', 'params', 'candidate', 'purpose',
                                    'method_seq', 'method_scope', 'step_index', 'skill_name'}:
                    raise ValueError('action carries an unsupported member')
                if not isinstance(row.get('params', {}), dict):
                    raise ValueError('action params must be an object')
                if 'candidate' not in row and isinstance(row.get('action'), str):
                    target = row.get('params', {}).get('target')
                    matches = [candidate for candidate in candidates.values()
                               if candidate.get('action_id') == row['action']
                               and all(key not in row or row[key] == candidate.get(key)
                                       for key in ('purpose', 'method_seq', 'method_scope', 'step_index'))
                               and (target is None or target == (candidate.get('params') or {}).get('target'))]
                    if len(matches) == 1:
                        row['candidate'] = matches[0]['candidate_id']
                    elif len(matches) > 1:
                        raise ValueError('action matches multiple candidates; select a candidate id')
                if 'candidate' in row:
                    picked = candidates.get(str(row['candidate']))
                    if picked is None:
                        raise ValueError(f"unknown candidate: {row['candidate']}")
                    binding = {'action': picked.get('action_id'), **{
                        key: picked.get(key) for key in ('purpose', 'method_seq', 'method_scope',
                                                        'step_index', 'skill_name')}}
                    for key, expected in binding.items():
                        if key in row and row[key] != expected:
                            raise ValueError(f'candidate and {key} disagree')
                    row.update(binding)
                    row['params'] = {**deepcopy(picked.get('params') or {}), **row.get('params', {})}
                row.setdefault('params', {})
                action = row.get('action')
                if not isinstance(action, str):
                    raise ValueError('action must be a string')
                if any(str(key).startswith('_') for key in row['params']):
                    raise ValueError('action params may not set kernel-internal members')
                if action in px4_schemas:
                    # Match by suffix so a configured prefix is not hard-coded.
                    name = action.rsplit('.', 1)[-1]
                    if name == 'goto':
                        goal = row['params'].get('position_enu')
                        if (not isinstance(goal, list) or len(goal) != 3
                                or any(type(v) not in (int, float) or not math.isfinite(v) for v in goal)):
                            raise ValueError(f'{action} requires finite position_enu [e,n,u]')
                        row['params'].setdefault('yaw_deg', 0.0)
                    elif name == 'takeoff':
                        row['params'].setdefault('altitude_m', 12.0)
                    # AAS adapter commands index config['vehicles'] by entity id,
                    # not by the backend vehicle name.
                    if row['params'].get('entity', vehicle) != vehicle:
                        raise ValueError('PX4 command must address this vehicle')
                    row['params']['entity'] = vehicle
                    numeric = 'yaw_deg' if name == 'goto' else 'altitude_m' if name == 'takeoff' else None
                    if numeric:
                        number = row['params'][numeric]
                        if (type(number) not in (int, float) or not math.isfinite(number)
                                or numeric == 'altitude_m' and number <= 0):
                            raise ValueError(f'{numeric} must be finite and valid')
                elif action not in VALID_ACTION_IDS:
                    raise ValueError(f'action is neither A001-A066 nor a PX4 flight command: {action!r}')
                if row.get('purpose') not in (None, 'approach', 'method'):
                    raise ValueError(f"purpose must be approach or method: {row['purpose']!r}")

        decision = await client.ask_json('edge_decide', prompt, observation, validate)
        outputs = []
        for row in decision['actions']:
            if row['action'] in px4_schemas:
                outputs.append({'kind': 'command', 'schema': row['action'],
                                'target': px4_target, 'payload': row['params']})
                continue
            params = dict(row['params'])
            picked = candidates.get(str(row.get('candidate')))
            ref = ([picked['task_id'], picked['task_rev']] if picked and picked.get('task_id')
                   else _task_ref(assignments, params))
            outputs.append({'kind': 'command', 'schema': SUBMIT, 'target': engine, 'payload': {
                'actor': vehicle, 'action_id': row['action'],
                'params_json': json.dumps(params, ensure_ascii=False),
                'task_ref_json': None if ref is None else json.dumps(ref),
                'candidate_id': row.get('candidate'), 'skill_name': row.get('skill_name'),
                'purpose': row.get('purpose'), 'method_seq': row.get('method_seq'),
                'method_scope': row.get('method_scope'), 'step_index': row.get('step_index')}})
        return {'outputs': outputs, 'decision': decision}

    graph = StateGraph(dict)
    graph.add_node('decide', decide)
    graph.add_edge(START, 'decide')
    graph.add_edge('decide', END)
    return graph
