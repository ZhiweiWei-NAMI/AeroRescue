"""WP2/WP3: center sandbox, edge harness, PX4 bridge, external commands, Qwen perception.

The T1-T7 section at the end re-runs the frozen-contract acceptance trajectories
from ``tools/verify_harness_and_skills.py`` so a regression fails in CI, not only
in the manual verification script.
"""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import struct
import zlib

import numpy as np
import pytest

from aerokernel import (
    BindingManifest, CommandRequest, EntityRef, Instant, Kernel, MemoryRegistry, Partition,
)
from aerokernel.binding import BindingRule, LifecycleRule
from aerokernel.engine import Timing
from aerokernel.sdk import ContextEngine
from aeroagentsim.engines.common import policies
from aeroagentsim.agents.langgraph import execute_graph
from aeroagentsim.agents.langgraph_client import JournalClient, ScriptedProvider
from aeroagentsim.agents.provider import ProviderError

from aerorescue.center_sandbox import ASSIGN, PROPOSE, build_graph as build_center
from aerorescue.edge_harness import SUBMIT, build_graph as build_edge
from aerorescue.skills import (
    CHANNEL_MODES, CandidateAction, CommChannelGate, MemoryEnvelope, SKILL_NAMES,
    VALID_ACTION_IDS, build_decision_candidates, execute_candidate, readable_envelopes,
)
from rescue_scenes import perception_qwen
from rescue_scenes.backgrounds import digest
from rescue_scenes.camera import CameraObservations
from rescue_scenes.core.api import Environment
from rescue_scenes.core.world.demo import DEMO_TRUTH_OPTIONS, compact_demo_layer
from rescue_scenes.engine import COMMANDS, RECEIPT, RescueWorld, command_descriptors
from rescue_scenes.presentation import (
    AGENT, BASE, BELIEF, FIELDS, POSITION, YAW, display_extras, registry_snapshot, type_of,
)
from rescue_scenes.px4_bridge import (
    PX4_PREFIX, PX4_SCHEMAS, PX4EmbodimentBridge, command_schemas, compass_deg_to_enu_yaw,
    enu_yaw_to_compass_deg, motion_state, movement_commands, yaw_from_quaternion,
)

RUN, EPOCH = 'run-wp2', '0'


def completion(value, tokens=2):
    message = {'role': 'assistant', 'content': json.dumps(value)}
    return {'message': message, 'raw': {'choices': [{'message': message}]},
            'usage': {'completion_tokens': tokens}}


def client(responses, *, calls=4, tokens=4000, retries=0, prompt_bytes=20000):
    provider = ScriptedProvider('scripted', responses)
    budget = {'wall_timeout_s': 30.0, 'max_calls': calls, 'max_tokens': tokens,
              'max_retries': retries, 'max_prompt_bytes': prompt_bytes, 'recursion_limit': 8}
    return JournalClient(provider, 'scripted', budget), provider


def initial(situation, schema='rescue.center.request', topic='rescue.center'):
    return {'observation': {'valid_at': {'ns': '0', 'microstep': 0}, 'known_at': {},
                            'fields': [], 'relations': [], 'events': [], 'actions': []},
            'trigger': {'id': 'msg-1', 'schema': schema, 'topic': topic,
                        'payload': {'data_json': json.dumps(situation)}, 'at_ns': 0}}


CENTER_SITUATION = {
    'now_s': 12.0, 'coordinator': 'coordinator-1',
    'reports': [{'report_id': 'r1', 'kind': 'person', 'target': 'human-1',
                 'status': 'confirmed', 'priority': 80, 'position_enu': [20.0, 40.0, 0.0]}],
    'vehicles': [
        {'id': 'uav-1', 'capabilities': ['C01', 'C05'], 'battery_remaining_Wh': 900.0,
         'payload_kg': 0.0, 'payload_capacity_kg': 2.5},
        # The real specialist_team embodiment carries the whole T029 requirement;
        # the center now checks ALL-of, so a C21-only declaration is refused.
        {'id': 'specialist_team-1', 'capabilities': ['C21', 'C31', 'C43', 'C46'],
         'battery_remaining_Wh': 1200.0, 'payload_kg': 0.0, 'payload_capacity_kg': 120.0}],
}


def test_center_graph_proposes_and_assigns_through_execute_graph():
    journal, provider = client({'center_plan': [completion(
        {'assignments': [{'candidate': 'cand-000', 'vehicle': 'specialist_team-1'}],
         'reason': 'only the specialist team carries clinical capability'})]})
    result = execute_graph(build_center, journal, {}, initial(CENTER_SITUATION))
    assert [row['schema'] for row in result['outputs']] == [PROPOSE, ASSIGN]
    propose, assign = (row['payload'] for row in result['outputs'])
    assert (propose['actor'], propose['type'], propose['target']) == ('coordinator-1', 'T029', 'human-1')
    assert propose['priority'] == 80.0 and propose['candidate_id'] == 'cand-000'
    assert json.loads(propose['origin_json'])['rule_id'] == 'R-T029-a'
    assert assign == {'actor': 'coordinator-1', 'agent_id': 'specialist_team-1',
                      'type': 'T029', 'target': 'human-1', 'candidate_id': 'cand-000',
                      'task_id': None}
    assert provider.calls == 1
    (record,) = journal.records
    assert record['identity'][1] == 'plan_and_dispatch' and record['validation'] == {'valid': True}
    assert record['request']['messages'][1]['content'].startswith('{')
    assert json.loads(record['request']['messages'][1]['content'])['candidates'][0]['id'] == 'cand-000'


def test_center_graph_rejects_capability_violation_then_retries():
    journal, _ = client({'center_plan': [
        completion({'assignments': [{'candidate': 'cand-000', 'vehicle': 'uav-1'}], 'reason': 'wrong'}),
        completion({'assignments': [{'candidate': 'cand-000', 'vehicle': 'specialist_team-1'}],
                    'reason': 'corrected'})]}, retries=1)
    result = execute_graph(build_center, journal, {}, initial(CENTER_SITUATION))
    assert len(result['outputs']) == 2
    records = journal.records
    assert [record['validation'] for record in records] == [
        {'valid': False, 'code': 'TOOL_REJECTED'}, {'valid': True}]
    # Capabilities are ALL-of: one missing capability rejects the assignment.
    assert 'lacks required capability C21' in records[0]['rejection']['message']
    assert '(ALL-of)' in records[0]['rejection']['message']


def test_center_graph_without_vehicles_still_proposes_only():
    situation = {**CENTER_SITUATION, 'vehicles': []}
    journal, _ = client({'center_plan': [completion({'assignments': [], 'reason': 'no vehicle'})]})
    result = execute_graph(build_center, journal, {}, initial(situation))
    assert [row['schema'] for row in result['outputs']] == [PROPOSE]


EDGE_SITUATION = {
    'now_s': 12.0, 'vehicle': 'uav-1', 'px4_vehicle': 'px4_0',
    'telemetry': {'position_enu': [20.0, 40.0, 12.0], 'battery_remaining_Wh': 900.0, 'payload_kg': 0.0},
    'perception': {'detections': [{'label': 'person', 'confidence': 0.9,
                                   'bbox_xyxy': [10, 20, 30, 40]}], 'sensing': 'camera_image'},
    'assignments': [{'task_id': 'T029@coordinator-1#1', 'type': 'T029', 'target': 'human-1',
                     'status': 'assigned', 'rev': 1}],
    'capabilities': ['C01', 'C05'],
}


def test_edge_graph_emits_native_action_and_px4_flight_command():
    decision = {'actions': [
        {'action': 'A009', 'params': {'target': 'human-1'}},
        {'action': f'{PX4_PREFIX}.goto', 'params': {'position_enu': [20.0, 40.0, 12.0], 'yaw_deg': 90.0}}],
        'reason': 'contact the detected person after repositioning'}
    journal, provider = client({'edge_decide': [completion(decision)]})
    result = execute_graph(build_edge, journal, {'vehicle': 'uav-1', 'px4_engine': 'px4'},
                           initial(EDGE_SITUATION, 'rescue.edge.telemetry', 'rescue.edge.uav-1'))
    native, flight = result['outputs']
    assert native['schema'] == SUBMIT and native['target'] == 'rescue'
    assert native['payload']['actor'] == 'uav-1' and native['payload']['action_id'] == 'A009'
    assert json.loads(native['payload']['params_json']) == {'target': 'human-1'}
    assert json.loads(native['payload']['task_ref_json']) == ['T029@coordinator-1#1', 1]
    assert flight == {'kind': 'command', 'schema': f'{PX4_PREFIX}.goto', 'target': 'px4',
                      'payload': {'position_enu': [20.0, 40.0, 12.0], 'yaw_deg': 90.0,
                                  'entity': 'uav-1'}}
    assert provider.calls == 1


def test_edge_graph_rejects_malformed_px4_goto():
    journal, _ = client({'edge_decide': [completion({'actions': [
        {'action': f'{PX4_PREFIX}.goto', 'params': {'yaw_deg': 10.0}}], 'reason': 'bad'})]})
    with pytest.raises(ProviderError) as caught:
        execute_graph(build_edge, journal, {'vehicle': 'uav-1'},
                      initial(EDGE_SITUATION, 'rescue.edge.telemetry', 'rescue.edge.uav-1'))
    assert caught.value.code == 'TOOL_REJECTED'


def demo_environment():
    return Environment.from_config({'scenario_id': 'SC01', 'seed': 1,
                                    'semantics': {'source': 'demo'}, 'tick_s': 1, 'until_s': 60})


def test_px4_bridge_enu_compass_round_trip():
    assert enu_yaw_to_compass_deg(0.0) == pytest.approx(90.0)
    assert enu_yaw_to_compass_deg(math.pi / 2) == pytest.approx(0.0)
    assert enu_yaw_to_compass_deg(math.pi) == pytest.approx(270.0)
    assert enu_yaw_to_compass_deg(-math.pi / 2) == pytest.approx(180.0)
    for degrees in (0.0, 45.0, 90.0, 180.0, 270.0, 359.9):
        assert enu_yaw_to_compass_deg(compass_deg_to_enu_yaw(degrees)) == pytest.approx(degrees)
    for yaw in (-math.pi, -1.0, 0.0, 1.0, math.pi):
        back = compass_deg_to_enu_yaw(enu_yaw_to_compass_deg(yaw))
        assert abs((back - yaw + math.pi) % (2 * math.pi) - math.pi) < 1e-12
    # The native attitude quaternion is retained and its heading is used.
    assert yaw_from_quaternion([0.0, 0.0, 0.0, 1.0]) == pytest.approx(0.0)
    assert yaw_from_quaternion([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]) \
        == pytest.approx(math.pi / 2)
    assert yaw_from_quaternion([0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8)]) \
        == pytest.approx(math.pi / 4)


def test_px4_bridge_syncs_pose_battery_and_motion_state():
    env = demo_environment()
    world = env.sim.world
    bridge = PX4EmbodimentBridge(env, {'uav-1': 'px4_0'})
    capacity = world.get('uav-1', 'platform.battery_capacity_Wh')
    applied = bridge.sync([{'vehicle': 'px4_0', 'sim_ns': 0,
                            'position_enu': [40.0, 50.0, 12.0], 'velocity_enu': [3.0, 4.0, 0.0],
                            'attitude_quat': [0.0, 0.0, 0.0, 1.0],
                            'battery': {'remaining_fraction': 0.5, 'voltage_v': 22.2},
                            'armed': True, 'flight_mode': 'AUTO.MISSION', 'landed_state': 'in_air'}])
    assert applied[0]['entity.pose'] == world.get('uav-1', 'entity.pose')
    assert world.get('uav-1', 'entity.pose') == [40.0, 50.0]
    assert world.get('uav-1', 'platform.battery_remaining_Wh') == pytest.approx(0.5 * capacity)
    assert world.get('uav-1', 'platform.motion_state') == 'moving'
    assert bridge.last['uav-1']['flight_mode'] == 'AUTO.MISSION'
    bridge.sync([{'vehicle': 'px4_0', 'velocity_enu': [0.0, 0.0, 0.0],
                  'armed': True, 'landed_state': 'on_ground'}])
    assert world.get('uav-1', 'platform.motion_state') == 'landed'
    bridge.sync([{'vehicle': 'px4_0', 'armed': False}])
    assert world.get('uav-1', 'platform.motion_state') == 'idle'
    assert motion_state({'armed': True, 'velocity_enu': [0.1, 0.1, 0.0]}) == 'holding'


def test_px4_bridge_command_sequence_and_action_goal():
    rows = movement_commands('uav-1', [10.0, 20.0, 12.0], math.pi / 2, land=True)
    assert [row['schema'] for row in rows] == [f'{PX4_PREFIX}.{name}'
                                               for name in ('arm', 'takeoff', 'goto', 'land')]
    assert rows[2]['payload'] == {'entity': 'uav-1', 'position_enu': [10.0, 20.0, 12.0], 'yaw_deg': 0.0}
    assert rows[1]['payload']['altitude_m'] == 12.0
    assert all(row['target'] == 'px4' for row in rows)
    airborne = movement_commands('uav-1', [10.0, 20.0, 12.0], 0.0, airborne=True)
    assert [row['schema'] for row in airborne] == [
        f'{PX4_PREFIX}.goto', f'{PX4_PREFIX}.hold']

    env = demo_environment()
    bridge = PX4EmbodimentBridge(env, {'uav-1': 'px4_0'})
    rows = bridge.commands_for_action('uav-1', 'A001', {'goal': [40.0, 50.0, 12.0]})
    pose = env.sim.world.get('uav-1', 'entity.pose')
    expected = math.atan2(50.0 - pose[1], 40.0 - pose[0])
    assert rows[2]['payload']['yaw_deg'] == pytest.approx(enu_yaw_to_compass_deg(expected))
    with pytest.raises(ValueError):
        bridge.commands_for_action('uav-1', 'A009', {'goal': [0.0, 0.0, 0.0]})


def png_bytes(width, height):
    raw = b''.join(b'\x00' + b'\x00\x00\x00' * width for _ in range(height))

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack(
            '>I', zlib.crc32(tag + data) & 0xffffffff)

    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def test_perception_offline_fallback_is_deterministic_and_contract_clean():
    width, height = 64, 48
    calibration = {'width': width, 'height': height, 'camera_pose': {}, 'calibration': True,
                   'perception_metadata': {},
                   'perception_options': {'endpoint': 'http://127.0.0.1:9/v1/chat/completions',
                                          'timeout_s': 2.0,
                                          'fallback_detections': [
                                              {'label': 'person', 'confidence': 0.8,
                                               'bbox_xyxy': [0.25, 0.25, 0.5, 0.5]},
                                              {'label': 'not_a_label', 'confidence': 0.9,
                                               'bbox_xyxy': [0.0, 0.0, 0.1, 0.1]}]}}
    detections = perception_qwen.detect(png_bytes(width, height), calibration)
    assert detections == [{'label': 'person', 'confidence': 0.8,
                           'bbox_xyxy': [16.0, 12.0, 32.0, 24.0]}]
    CameraObservations.validate_detections(detections, width, height)
    assert calibration['perception_metadata']['inference_mode'] == 'offline_fallback'
    assert calibration['perception_metadata']['fallback_detections'] == 1

    empty = {'width': width, 'height': height, 'perception_metadata': {},
             'perception_options': {'endpoint': 'http://127.0.0.1:9/v1/chat/completions'}}
    assert perception_qwen.detect(png_bytes(width, height), empty) == []
    assert empty['perception_metadata']['fallback_reason'] in {'URLError', 'ConnectionError'}


def test_perception_parses_fenced_model_reply_and_rejects_bad_bytes():
    rows = perception_qwen._content_rows({'choices': [{'finish_reason': 'stop', 'message': {
        'role': 'assistant', 'content': '```json\n[{"label": "road_debris", "confidence": 0.5, '
                                        '"bbox_xyxy": [0.1, 0.1, 0.4, 0.4]}]\n```'}}]})
    assert perception_qwen._pixels(rows, 100, 50) == [
        {'label': 'road_debris', 'confidence': 0.5, 'bbox_xyxy': [10.0, 5.0, 40.0, 20.0]}]
    with pytest.raises(ValueError):
        perception_qwen._content_rows({'choices': [{'finish_reason': 'length',
                                                    'message': {'role': 'assistant', 'content': ''}}]})
    with pytest.raises(ValueError):
        perception_qwen.detect(b'not a png', {'width': 10, 'height': 10, 'perception_options': {}})


def tiny_grid(path):
    e = np.arange(0.0, 121.0); n = np.arange(0.0, 81.0)
    np.savez(path, E=e.reshape(1, -1), N=n.reshape(1, -1), U=np.zeros((len(n), len(e))),
             cell_class=np.zeros((len(n) - 1, len(e) - 1), dtype=np.int16),
             class_names=np.array(['ground'], dtype='<U8'))
    return path


def rescue_world_bundle(directory):
    doc, options = compact_demo_layer(), deepcopy(DEMO_TRUTH_OPTIONS)
    (directory / 'semantics.json').write_text(json.dumps(doc))
    grid = tiny_grid(directory / 'grid.npz')
    config = {'schema': 'rescue.scene/v1', 'background': {'id': 'TEST', 'grid': str(grid)},
              'scenario_id': 'SC01', 'seed': 1, 'tick_s': 1.0, 'until_s': 20.0,
              'display': {'uav_agl_m': 12, 'agent_ground_offset_m': 0.15,
                          'person_offset_m': 0.15, 'status_height_m': 3},
              'assumptions': ['wp2 test'], 'placement': {}}
    core = {'scenario_id': 'SC01', 'seed': 1, 'semantics': {'source': 'demo', 'path': 'semantics.json'},
            'bounds_m': [0.0, 0.0, 120.0, 80.0], 'agents': 'profile', 'tick_s': 1.0,
            'until_s': 20.0, 'truth_options': options}
    bundle = {'config': config, 'core': core, 'audit': {'layout': {}},
              'config_digest': digest(config)}
    path = directory / 'bundle.json'
    path.write_text(json.dumps(bundle))
    return path, core


def build_rescue_world(tmp_path, commands=(), engine_config=None):
    from aeroagentsim.platform.plugins import EngineBuild

    bundle_path, core = rescue_world_bundle(tmp_path)
    env = Environment.from_config(core)
    entities, seen = [], set()
    for eid, row in sorted(env.sim.world.entities.items()):
        for ref_id, typ in [(eid, type_of(row, {}))] + [
                (xid, xtyp) for xid, xtyp, _ in display_extras(row, {}, {}, {})]:
            if ref_id not in seen:
                seen.add(ref_id)
                entities.append(EntityRef(RUN, EPOCH, ref_id, 0, typ))

    snapshot = registry_snapshot()
    snapshot['registry']['messages'].extend(command_descriptors())
    registry = MemoryRegistry.from_data(snapshot['registry'])
    manifest = BindingManifest(
        RUN, EPOCH, tuple(entities),
        rules=(BindingRule('rescue', BASE, FIELDS[:-1], '*'),
               BindingRule('rescue', AGENT, (BELIEF,), '*')),
        lifecycle=(LifecycleRule('rescue', BASE, '*'),),
        bootstrap_commands=tuple(commands))
    build = EngineBuild('rescue', {'bundle_path': str(bundle_path), 'external_commands': True,
                                  **(engine_config or {})},
                        registry, manifest, tuple(entities), {},
                        partitions={}, run_directory=tmp_path / 'run')
    world = RescueWorld(build)
    build.partitions['rescue'] = world.partition
    kernel = Kernel(root_seed=7)
    kernel.bind(registry, manifest, (world,))
    return world, kernel


def test_scenario_plugin_console_scene_is_retained_for_replay(tmp_path):
    scene={'id':'test rescue scene','city':{'kind':'glb','url':'/data/scenes/test.glb'}}
    world,kernel=build_rescue_world(tmp_path,engine_config={'console_scene':scene})
    try:
        assert json.loads((tmp_path/'run/console-scene.json').read_text())==scene
        assert json.loads((tmp_path/'run/scene-source.json').read_text())=={'kind':'fixed-glb','id':'run'}
    finally:
        world.close()


PROPOSE_PAYLOAD = {'actor': 'coordinator-1', 'type': 'T029', 'target': 'human-building-1-1',
                   'priority': 80.0, 'candidate_id': 'cand-000', 'why_zh': 'center plan',
                   'origin_json': json.dumps({'rule_id': 'CENTER-SANDBOX',
                                              'why_zh': 'center plan', 'belief_refs': []})}
ASSIGN_PAYLOAD = {'actor': 'coordinator-1', 'agent_id': 'uav-1', 'type': 'T029',
                  'target': 'human-building-1-1', 'candidate_id': 'cand-000', 'task_id': None}


def test_rescue_world_external_commands_and_receipts(tmp_path):
    submit = {'actor': 'uav-1', 'action_id': 'A005',
              'params_json': json.dumps({'target': 'building-1', 'modality': 'rgb'}),
              'task_ref_json': None}
    commands = (CommandRequest(COMMANDS['propose_task'], 'rescue', Instant(1), PROPOSE_PAYLOAD),
                CommandRequest(COMMANDS['assign_task'], 'rescue', Instant(1), ASSIGN_PAYLOAD),
                CommandRequest(COMMANDS['submit_action'], 'rescue', Instant(1), submit))
    world, kernel = build_rescue_world(tmp_path, commands)
    kernel.start()
    with pytest.raises(ValueError):
        world.handle_external_command('unknown_kind', {})
    rejected = world.handle_external_command(
        'submit_action', {'actor': 'ghost', 'action_id': 'A001', 'params': {}})
    assert rejected['status'] == 'rejected' and rejected['command_id'].startswith('injected-')
    injected = world.handle_external_command(
        'submit_action', {'actor': 'uav-1', 'action_id': 'A001', 'params': {'goal': [40.0, 50.0, 12.0]}})
    assert injected['status'] == 'submitted'
    assert injected['instance_id'] in world.command_links
    kernel.run_until(12 * 10 ** 9)
    receipts = world.command_receipts
    by_command = {row['command_id']: row for row in receipts}
    assert by_command[rejected['command_id']]['status'] == 'rejected'
    assert by_command[injected['command_id']]['status'] in {'succeeded', 'rejected', 'failed'}
    assert {'A005', 'A050', 'A052'} <= {row['action_id'] for row in receipts}
    # The typed schema member (params_json) survives the WAL delivery intact.
    sensor = next(row for row in receipts if row['action_id'] == 'A005')
    assert world.sim.instances[sensor['instance_id']]['params']['target'] == 'building-1'
    plan = [row for row in receipts if row['action_id'] in {'A050', 'A052'}]
    assert [row['status'] for row in plan] == ['succeeded', 'succeeded']
    assert json.loads(plan[1]['result_json'])['status'] == 'succeeded'
    assert len({row['instance_id'] for row in plan}) == 2
    assert world.pending_assignments == {} and world.command_links == {}
    # Deferred assignments append new action events while the proposal result
    # is recorded; the journal must retain those events too.
    recorded = [json.loads(line) for line in (tmp_path / 'run' / 'events.jsonl').read_text().splitlines()]
    assert [row['seq'] for row in recorded] == [event.seq for event in world.sim.events]
    kernel.close()
    result = json.loads((tmp_path / 'run' / 'result.json').read_text())
    assert result['external_commands']['receipt_schema'] == RECEIPT
    assert result['external_commands']['undelivered'] == []
    assert len(result['external_commands']['receipts']) == len(receipts)
    assert result['external_commands']['command_instances'][injected['command_id']] == \
        injected['instance_id']


def test_perception_closes_camera_inference_contract_with_local_endpoint():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.update(path=self.path, body=body)
            content = json.dumps([{'label': 'person', 'confidence': 0.75,
                                   'bbox_xyxy': [0.5, 0.5, 0.75, 0.75]}])
            payload = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {
                'role': 'assistant', 'content': content}}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        endpoint = f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
        calibration = {'width': 80, 'height': 60, 'perception_metadata': {},
                       'perception_options': {'endpoint': endpoint, 'model': 'Qwen3.5-2B-Base'}}
        detector = object.__new__(CameraObservations)  # the real camera-side wrapper
        detections, elapsed = detector.detect(
            'rescue_scenes.perception_qwen:detect', png_bytes(80, 60), calibration, 80, 60)
        assert detections == [{'label': 'person', 'confidence': 0.75,
                               'bbox_xyxy': [40.0, 30.0, 60.0, 45.0]}]
        assert elapsed >= 0.0
        assert seen['path'] == '/v1/chat/completions'
        assert seen['body']['model'] == 'Qwen3.5-2B-Base'
        image = seen['body']['messages'][1]['content'][1]
        assert image['type'] == 'image_url'
        assert image['image_url']['url'].startswith('data:image/png;base64,')
        assert calibration['perception_metadata']['inference_mode'] == 'qwen_vl'
        assert calibration['perception_metadata']['model'] == 'Qwen3.5-2B-Base'
    finally:
        server.shutdown()
        server.server_close()


def test_model_server_resolves_local_weights_and_reports_completions():
    from aerorescue import model_server

    with pytest.raises(SystemExit):
        model_server.resolve_model('not-a/local/repo/name')
    completion = model_server._completion('Qwen3.5-0.8B-Base', '{"actions": []}', 7, 3)
    assert completion['choices'][0]['message'] == {'role': 'assistant', 'content': '{"actions": []}'}
    assert completion['usage'] == {'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10}
    assert completion['model'] == 'Qwen3.5-0.8B-Base'


def test_center_and_edge_outputs_drive_the_rescue_world(tmp_path):
    center_journal, _ = client({'center_plan': [completion(
        {'assignments': [{'candidate': 'cand-000', 'vehicle': 'specialist_team-1'}],
         'reason': 'the specialist team carries the clinical capability'})]})
    center = execute_graph(build_center, center_journal, {}, initial(CENTER_SITUATION))
    edge_journal, _ = client({'edge_decide': [completion({'actions': [
        {'action': 'A005', 'params': {'target': 'building-1', 'modality': 'rgb'}}],
        'reason': 'inspect the assigned building'})]})
    edge = execute_graph(build_edge, edge_journal, {'vehicle': 'uav-1'},
                         initial(EDGE_SITUATION, 'rescue.edge.telemetry', 'rescue.edge.uav-1'))
    commands = [CommandRequest(row['schema'], row['target'], Instant(1), row['payload'])
                for row in center['outputs'] + edge['outputs']]
    world, kernel = build_rescue_world(tmp_path, commands)
    kernel.start()
    kernel.run_until(15 * 10 ** 9)
    receipts = {row['action_id']: row['status'] for row in world.command_receipts}
    assert receipts.get('A050') == 'succeeded', world.command_receipts
    assert receipts.get('A052') == 'succeeded', world.command_receipts
    assert receipts.get('A005') == 'succeeded', world.command_receipts
    kernel.close()


class PX4Source(ContextEngine):
    """Minimal producer of the registered PX4 telemetry fields (test double)."""

    version = 'test-px4/1'

    def __init__(self, ref, samples):
        self.ref, self.samples = ref, samples
        fields = tuple(sorted(samples))
        super().__init__(Partition('px4', 'px4', produces=fields,
                                   timing=Timing('lockstep', step_ns=10 ** 9, certified_hold=True)),
                         policies=policies(fields))

    def bootstrap(self, ctx):
        for field, value in self.samples.items():
            ctx.set(self.ref, field, value)


PX4_RECORD = {'type': 'record', 'members': {'remaining_fraction': {'type': 'number'},
                                            'voltage_v': {'type': 'number'}},
              'required': ['remaining_fraction', 'voltage_v'], 'extra': False}


def build_px4_world(tmp_path, position=(55.0, 60.0, 25.0), attitude=(0.0, 0.0, 0.7071067811865476, 0.7071067811865476)):
    from aeroagentsim.platform.plugins import EngineBuild

    bundle_path, core = rescue_world_bundle(tmp_path)
    env = Environment.from_config(core)
    entities, seen = [], set()
    for eid, row in sorted(env.sim.world.entities.items()):
        for ref_id, typ in [(eid, type_of(row, {}))] + [
                (xid, xtyp) for xid, xtyp, _ in display_extras(row, {}, {}, {})]:
            if ref_id not in seen:
                seen.add(ref_id)
                entities.append(EntityRef(RUN, EPOCH, ref_id, 0, typ))
    uav_type = type_of(env.sim.world.entities['uav-1'], {})
    snapshot = registry_snapshot()
    snapshot['registry']['messages'].extend(command_descriptors())
    snapshot['registry']['fields'].extend([
        {'id': 'px4.position_enu', 'declaring_type': uav_type, 'metadata': {'frame': 'enu'},
         'schema': {'type': 'vector', 'length': 3, 'items': {'type': 'number'}}},
        {'id': 'px4.velocity_enu', 'declaring_type': uav_type, 'metadata': {'frame': 'enu'},
         'schema': {'type': 'vector', 'length': 3, 'items': {'type': 'number'}}},
        {'id': 'px4.attitude_quat', 'declaring_type': uav_type, 'metadata': {'frame': 'enu'},
         'schema': {'type': 'vector', 'length': 4, 'items': {'type': 'number'}}},
        {'id': 'px4.battery', 'declaring_type': uav_type, 'metadata': {},
         'schema': deepcopy(PX4_RECORD)},
        {'id': 'px4.armed', 'declaring_type': uav_type, 'metadata': {},
         'schema': {'type': 'boolean'}},
    ])
    registry = MemoryRegistry.from_data(snapshot['registry'])
    uav_ref = next(ref for ref in entities if ref.id == 'uav-1')
    source = PX4Source(uav_ref, {
        'px4.position_enu': list(position), 'px4.attitude_quat': list(attitude),
        'px4.velocity_enu': [0.0, 0.0, 0.0], 'px4.armed': True,
        'px4.battery': {'remaining_fraction': 0.5, 'voltage_v': 22.2}})
    manifest = BindingManifest(
        RUN, EPOCH, tuple(entities),
        rules=(BindingRule('rescue', BASE, FIELDS[:-1], '*'),
               BindingRule('rescue', AGENT, (BELIEF,), '*'),
               BindingRule('px4', uav_type,
                           ('px4.position_enu', 'px4.velocity_enu', 'px4.attitude_quat',
                            'px4.battery', 'px4.armed'), '*')),
        lifecycle=(LifecycleRule('rescue', BASE, '*'),))
    config = {'bundle_path': str(bundle_path), 'external_commands': True,
              'px4_vehicles': {'uav-1': 'px4_0'},
              'px4_fields': {'uav-1': {'position': 'px4.position_enu',
                                       'velocity': 'px4.velocity_enu',
                                       'attitude': 'px4.attitude_quat',
                                       'battery': 'px4.battery', 'armed': 'px4.armed'}}}
    build = EngineBuild('rescue', config, registry, manifest, tuple(entities), {},
                        partitions={}, run_directory=tmp_path / 'run')
    world = RescueWorld(build)
    build.partitions['rescue'] = world.partition
    kernel = Kernel(root_seed=7)
    kernel.bind(registry, manifest, (world, source))
    return world, kernel, uav_ref


def test_px4_telemetry_is_the_single_writer_of_native_pose(tmp_path):
    world, kernel, uav_ref = build_px4_world(tmp_path)
    # Declared consumption of the registered PX4 fields, not guessed IDs.
    consumed = {(dep.field, dep.id_pattern) for dep in world.partition.consumes}
    assert {('px4.position_enu', 'uav-1'), ('px4.battery', 'uav-1'),
            ('px4.attitude_quat', 'uav-1')} <= consumed
    # The scripted agent tick must not move a PX4-managed UAV.
    before = deepcopy(world.sim.world.get('uav-1', 'entity.pose'))
    world.sim.agents['uav-1'].on_tick(world.sim)
    assert world.sim.world.get('uav-1', 'entity.pose') == before
    kernel.start()
    kernel.run_until(3 * 10 ** 9)
    pose = world.px4.pose['uav-1']
    assert pose['position_enu'] == [55.0, 60.0, 25.0]
    assert pose['attitude'] == [0.0, 0.0, 0.7071067811865476, 0.7071067811865476]
    assert pose['yaw_rad'] == pytest.approx(math.pi / 2)
    # Publication uses the real 3D altitude and heading, never the display AGL.
    assert world.previous['uav-1'] == [55.0, 60.0, 25.0]
    assert world.last_values[(uav_ref.id, uav_ref.generation, POSITION)][2] == 25.0
    assert world.last_values[(uav_ref.id, uav_ref.generation, YAW)] == pytest.approx(90.0)
    capacity = world.sim.world.get('uav-1', 'platform.battery_capacity_Wh')
    assert world.sim.world.get('uav-1', 'platform.battery_remaining_Wh') == pytest.approx(0.5 * capacity)
    assert world.sim.world.get('uav-1', 'platform.motion_state') == 'holding'
    # Native scripted movement of a PX4-managed UAV is refused, not silently applied.
    refused = world.handle_external_command(
        'submit_action', {'actor': 'uav-1', 'action_id': 'A001', 'params': {'goal': [10.0, 20.0, 12.0]}})
    assert refused['status'] == 'rejected' and 'PX4 adapter' in refused['result_json']
    kernel.close()


def test_command_schema_json_members_are_decoded(tmp_path):
    world, kernel, _ = build_px4_world(tmp_path)
    submitted = world.handle_external_command('submit_action', {
        'actor': 'uav-1', 'action_id': 'A005',
        'params_json': json.dumps({'target': 'building-1', 'modality': 'rgb'}),
        'task_ref_json': None})
    assert submitted['status'] == 'submitted'
    instance = world.sim.instances[submitted['instance_id']]
    assert instance['params']['target'] == 'building-1'
    assert instance['params']['modality'] == 'rgb'
    proposed = world.handle_external_command('propose_task', {
        'actor': 'coordinator-1', 'type': 'T010', 'target': 'building-2', 'priority': 60.0,
        'candidate_id': 'cand-7', 'why_zh': 'schema keys',
        'origin_json': json.dumps({'rule_id': 'R-T010-a', 'why_zh': 'schema keys',
                                   'belief_refs': ['r7']})})
    assert proposed['status'] == 'submitted'
    params = world.sim.instances[proposed['instance_id']]['params']
    assert params['origin'] == {'rule_id': 'R-T010-a', 'why_zh': 'schema keys',
                                'belief_refs': ['r7']}
    assert params['priority'] == 60.0
    kernel.close()


def test_assign_task_without_in_flight_proposal_is_rejected_not_queued(tmp_path):
    world, kernel, _ = build_px4_world(tmp_path)
    result = world.handle_external_command('assign_task', {
        'actor': 'coordinator-1', 'agent_id': 'uav-1', 'type': 'T010',
        'target': 'building-9', 'candidate_id': 'cand-9', 'task_id': None})
    assert result['status'] == 'rejected'
    assert 'in-flight A050' in result['result_json']
    assert world.pending_assignments == {}
    # A deferred assignment whose proposal completed without a task terminates
    # as failed and is never re-queued.
    world.pending_assignments[('coordinator-1', 'T010', 'building-9')].append(
        {'payload': {'actor': 'coordinator-1', 'agent_id': 'uav-1', 'type': 'T010',
                     'target': 'building-9', 'candidate_id': 'cand-9', 'task_id': None},
         'command_id': 'cmd-deferred'})
    world._deferred({'actor': 'coordinator-1', 'type': 'T010', 'target': 'building-9'}, None, True)
    assert world.pending_assignments == {}
    failed = [row for row in world.command_receipts if row['command_id'] == 'cmd-deferred']
    assert len(failed) == 1 and failed[0]['status'] == 'failed'
    assert 'without a matching local task' in failed[0]['result_json']
    # A failed proposal likewise terminates its deferred assignment.
    world.pending_assignments[('coordinator-1', 'T010', 'building-8')].append(
        {'payload': {'actor': 'coordinator-1', 'agent_id': 'uav-1', 'type': 'T010',
                     'target': 'building-8', 'candidate_id': 'cand-8', 'task_id': None},
         'command_id': 'cmd-failed-proposal'})
    world._deferred({'actor': 'coordinator-1', 'type': 'T010', 'target': 'building-8'}, None, False)
    assert world.pending_assignments == {}
    terminated = [row for row in world.command_receipts if row['command_id'] == 'cmd-failed-proposal']
    assert len(terminated) == 1 and terminated[0]['status'] == 'failed'
    assert 'did not complete' in terminated[0]['result_json']
    kernel.close()


@pytest.mark.parametrize('landed,velocity,expected', [
    ('ON_GROUND', [0.0, 0.0, 0.0], 'landed'),
    ('GROUND', [0.0, 0.0, 0.0], 'landed'),
    ('LANDING', [0.0, 0.0, -0.2], 'landed'),
    ('IN_AIR', [3.0, 4.0, 0.0], 'moving'),
    ('TAKING_OFF', [0.0, 0.0, 1.0], 'holding'),
])
def test_motion_state_normalizes_uppercase_landed_states(landed, velocity, expected):
    assert motion_state({'armed': True, 'landed_state': landed, 'velocity_enu': velocity}) == expected
    assert motion_state({'armed': True, 'landed': landed}) == ('landed' if expected == 'landed' else 'holding')


def test_px4_command_schemas_follow_the_registered_adapter():
    from aeroagentsim.adapters.schemas import PX4_GAZEBO

    assert PX4_PREFIX == f'adapters.{PX4_GAZEBO}'
    assert PX4_SCHEMAS == command_schemas() == command_schemas(PX4_PREFIX)
    assert {f'{PX4_PREFIX}.{name}' for name in ('arm', 'takeoff', 'goto', 'hold', 'land')} == PX4_SCHEMAS
    assert command_schemas('adapters.custom') == frozenset(
        f'adapters.custom.{name}' for name in ('arm', 'takeoff', 'goto', 'hold', 'land'))
    custom = movement_commands('uav-1', [1.0, 2.0, 3.0], 0.0, prefix='adapters.custom')
    assert [row['schema'] for row in custom] == ['adapters.custom.arm', 'adapters.custom.takeoff',
                                                 'adapters.custom.goto', 'adapters.custom.hold']


@pytest.mark.parametrize('action', ['A000', 'A067', 'A100', 'a001', 'B001'])
def test_edge_graph_rejects_out_of_range_action_ids(action):
    journal, _ = client({'edge_decide': [completion({'actions': [
        {'action': action, 'params': {'target': 'human-1'}}], 'reason': 'bad'})]})
    with pytest.raises(ProviderError) as caught:
        execute_graph(build_edge, journal, {'vehicle': 'uav-1'},
                      initial(EDGE_SITUATION, 'rescue.edge.telemetry', 'rescue.edge.uav-1'))
    assert caught.value.code == 'TOOL_REJECTED'


@pytest.mark.parametrize('action', ['A001', 'A063', 'A064', 'A065', 'A066'])
def test_edge_graph_accepts_the_full_66_action_space(action):
    journal, _ = client({'edge_decide': [completion({'actions': [
        {'action': action, 'params': {'target': 'human-1'}}], 'reason': 'in range'})]})
    result = execute_graph(build_edge, journal, {'vehicle': 'uav-1'},
                           initial(EDGE_SITUATION, 'rescue.edge.telemetry', 'rescue.edge.uav-1'))
    assert result['outputs'][0]['payload']['action_id'] == action
    assert action in VALID_ACTION_IDS and len(VALID_ACTION_IDS) == 66


def test_edge_graph_picks_a_delivered_candidate_and_carries_its_binding():
    candidate = {'candidate_id': 'cand-001-method', 'skill_name': 'search_and_assess',
                 'task_id': 'T011@uav-1#2', 'task_rev': 2, 'task_type': 'T011',
                 'method_seq': 'Q011', 'step_index': 1, 'purpose': 'method',
                 'method_scope': 'main', 'action_id': 'A005',
                 'target_id': 'building-1', 'params': {'target': 'building-1'}}
    situation = dict(EDGE_SITUATION, candidates=[candidate])
    journal, _ = client({'edge_decide': [completion({'actions': [
        {'candidate': 'cand-001-method', 'action': 'A005'}, ], 'reason': 'take the candidate'})]})
    result = execute_graph(build_edge, journal, {'vehicle': 'uav-1'},
                           initial(situation, 'rescue.edge.telemetry', 'rescue.edge.uav-1'))
    payload = result['outputs'][0]['payload']
    assert (payload['action_id'], payload['candidate_id']) == ('A005', 'cand-001-method')
    assert (payload['purpose'], payload['method_seq'], payload['step_index']) == ('method', 'Q011', 1)
    assert payload['skill_name'] == 'search_and_assess'


def test_edge_graph_rejects_an_unknown_candidate_id():
    journal, _ = client({'edge_decide': [completion({'actions': [
        {'candidate': 'cand-missing', 'action': 'A005'}], 'reason': 'bad'})]})
    with pytest.raises(ProviderError) as caught:
        execute_graph(build_edge, journal, {'vehicle': 'uav-1'},
                      initial(EDGE_SITUATION, 'rescue.edge.telemetry', 'rescue.edge.uav-1'))
    assert caught.value.code == 'TOOL_REJECTED'


def test_edge_graph_honours_a_custom_px4_prefix():
    journal, _ = client({'edge_decide': [completion({'actions': [
        {'action': 'adapters.custom.goto', 'params': {'position_enu': [1.0, 2.0, 3.0],
                                                      'yaw_deg': 0.0}}], 'reason': 'custom'})]})
    result = execute_graph(build_edge, journal,
                           {'vehicle': 'uav-1', 'px4_command_prefix': 'adapters.custom'},
                           initial(EDGE_SITUATION, 'rescue.edge.telemetry', 'rescue.edge.uav-1'))
    assert result['outputs'][0]['schema'] == 'adapters.custom.goto'
    assert result['outputs'][0]['payload']['entity'] == 'uav-1'


def test_edge_graph_applies_goto_defaults_under_a_custom_prefix():
    journal, _ = client({'edge_decide': [completion({'actions': [
        {'action': 'adapters.custom.goto', 'params': {'position_enu': [4.0, 5.0, 6.0]}}],
        'reason': 'custom goto defaults'})]})
    result = execute_graph(build_edge, journal,
                           {'vehicle': 'uav-1', 'px4_command_prefix': 'adapters.custom'},
                           initial(EDGE_SITUATION, 'rescue.edge.telemetry', 'rescue.edge.uav-1'))
    payload = result['outputs'][0]['payload']
    assert payload['yaw_deg'] == 0.0 and payload['entity'] == 'uav-1'


def test_edge_prompt_names_the_registered_px4_schemas():
    from aerorescue import edge_harness

    journal, _ = client({'edge_decide': [completion({'actions': [], 'reason': 'none'})]})
    execute_graph(edge_harness.build_graph, journal, {'vehicle': 'uav-1'},
                  initial(EDGE_SITUATION, 'rescue.edge.telemetry', 'rescue.edge.uav-1'))
    (record,) = journal.records
    prompt = record['request']['messages'][0]['content']
    for schema in PX4_SCHEMAS:
        assert schema in prompt
    assert 'adapters.px4.arm' not in prompt


def test_px4_native_position_outside_the_scripted_grid_still_publishes(tmp_path):
    world, kernel, _ = build_px4_world(tmp_path, position=(500.0, 520.0, 30.0))
    kernel.start()
    kernel.run_until(2 * 10 ** 9)
    assert world.previous['uav-1'] == [500.0, 520.0, 30.0]
    kernel.close()


def test_engine_config_schema_exposes_the_opt_in_keys():
    from rescue_scenes.engine import build

    properties = build.config_schema['properties']
    assert {'external_commands', 'px4_vehicles', 'px4_fields'} <= set(properties)
    assert build.config_schema['additionalProperties'] is False
    assert build.config_schema['required'] == ['bundle_path']


# --------------------------------------------------------------------------- #
# T1-T7: frozen-contract acceptance trajectories.
#
# These re-run the exact trajectories of ``tools/verify_harness_and_skills.py``
# (real kernel, real AgentRuntime, real PX4 bridge) so the contract cannot
# regress without a test failure. Each trajectory is one test for a readable
# failure, and the script itself writes the published matrix.
# --------------------------------------------------------------------------- #

def acceptance_module():
    path = Path(__file__).resolve().parents[1] / 'tools' / 'verify_harness_and_skills.py'
    spec = importlib.util.spec_from_file_location('verify_harness_and_skills', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ACCEPTANCE = acceptance_module()


@pytest.mark.parametrize('trajectory', [ACCEPTANCE.t1, ACCEPTANCE.t2, ACCEPTANCE.t3,
                                        ACCEPTANCE.t4, ACCEPTANCE.t5, ACCEPTANCE.t6,
                                        ACCEPTANCE.t7], ids=['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'])
def test_frozen_contract_trajectory(trajectory):
    trace = trajectory()
    report = trace.to_dict()
    assert report['ok'], [row for row in report['assertions'] if not row['ok']]
    assert len(report['assertions']) >= 3


def test_skills_primitives_cover_the_catalog():
    trace = ACCEPTANCE.primitives()
    assert trace.to_dict()['ok']
    assert len(VALID_ACTION_IDS) == 66 and len(SKILL_NAMES) == 11


def test_comm_channel_gate_modes_and_envelope_delivery():
    assert CHANNEL_MODES == {'B00': (False, False), 'B10': (True, False),
                             'B01': (False, True), 'B11': (True, True)}
    for mode, (up, down) in CHANNEL_MODES.items():
        gate = CommChannelGate(mode, 8)
        assert (gate.allows('down'), gate.allows('up')) == (down, up)
        assert gate.admit('down', 4) is down
        assert gate.admit('up', 4) is up
        assert gate.admit('down', 100) is False  # never exceeds the byte budget
    envelope = MemoryEnvelope(envelope_id='mem-1', sender='coordinator-1', receivers=('uav-1',),
                              kind='candidate', payload={'op': 'candidate'}, size_bytes=12,
                              created_s=0.0, valid_until_s=60.0, delivered_at_s=None)
    assert envelope.readable(10.0, receiver='uav-1') is False   # not delivered yet
    envelope.delivered_at_s = 5.0
    assert envelope.readable(10.0, receiver='uav-1') is True
    assert envelope.readable(10.0, receiver='uav-2') is False   # wrong receiver
    assert envelope.readable(61.0, receiver='uav-1') is False   # expired
    tampered = MemoryEnvelope.from_dict(dict(envelope.to_dict(), payload={'op': 'forged'}))
    assert tampered.readable(10.0, receiver='uav-1') is False
    assert [row.envelope_id for row in readable_envelopes([tampered, envelope], 10.0, 'uav-1')] \
        == ['mem-1']


def test_candidate_action_rejects_unknown_ids_and_unbound_bindings():
    for action in ('A000', 'A067', 'a001', 'B001'):
        with pytest.raises(ValueError):
            CandidateAction(candidate_id='bad', skill_name='search_and_assess', action_id=action)
    with pytest.raises(ValueError):
        CandidateAction(candidate_id='bad', skill_name='search_and_assess', action_id='A001',
                        task_rev=1)
    with pytest.raises(ValueError):
        CandidateAction(candidate_id='bad', skill_name='not_a_skill', action_id='A001')
    with pytest.raises(ValueError):
        CandidateAction(candidate_id='bad', skill_name='search_and_assess', action_id='A001',
                        purpose='integrate')


def test_execute_candidate_refuses_a_stale_cursor():
    env = ACCEPTANCE.demo_env()
    sim, agent = env.sim, env.sim.agents['uav-1']
    ACCEPTANCE.isolate(sim, agent)
    agent.spec['task_allowlist'] = ['T011']
    agent.policy_mode = 'external_harness'
    task = ACCEPTANCE.accept_task(agent, sim, 'T011', 'building-1')
    candidates = build_decision_candidates(agent, sim.catalog, sim.t)
    method = [row for row in candidates if row.purpose == 'method'][0]
    agent.current = task['id']
    execute_candidate(sim, agent, method, ticket=candidates.context)
    ACCEPTANCE.wait_idle(env, agent)
    assert task['method']['step_index'] == 1
    with pytest.raises(ValueError):
        # The same candidate is now one step behind the task.
        execute_candidate(sim, agent, method)
    with pytest.raises(ValueError):
        execute_candidate(sim, agent, method, ticket=candidates.context)


def decision_task():
    env = ACCEPTANCE.demo_env()
    agent = env.sim.agents['uav-1']
    ACCEPTANCE.isolate(env.sim, agent)
    agent.spec['task_allowlist'] = ['T011']
    agent.policy_mode = 'external_harness'
    task = ACCEPTANCE.accept_task(agent, env.sim, 'T011', 'building-1')
    candidates = build_decision_candidates(agent, env.sim.catalog, env.sim.t)
    method = next(row for row in candidates if row.purpose == 'method')
    return env, agent, task, candidates, method


def test_channel_bits_are_uplink_then_downlink():
    assert CommChannelGate('B10', 8).admit('up', 8)
    assert not CommChannelGate('B10', 8).admit('down', 1)
    assert CommChannelGate('B01', 8).admit('down', 8)
    assert not CommChannelGate('B01', 8).admit('up', 1)


@pytest.mark.parametrize('member,value', [
    ('actor_id', 'ground_vehicle-1'), ('candidate_set_digest', 'forged'),
    ('cursor', 999), ('task_ref', ('another-task', 1)), ('decision_id', 'forged'),
])
def test_decision_ticket_checks_all_binding_members(member, value):
    from dataclasses import replace
    env, agent, _, candidates, method = decision_task()
    with pytest.raises(ValueError):
        execute_candidate(env.sim, agent, method,
                          ticket=replace(candidates.context, **{member: value}))
    assert not env.sim.instances


@pytest.mark.parametrize('member,value', [
    ('action_id', 'A066'), ('method_scope', 'recovery'), ('method_seq', 'Q098'),
    ('task_rev', None), ('step_index', None),
])
def test_candidate_cannot_change_the_current_task_step(member, value):
    env, agent, _, _, method = decision_task()
    data = method.to_dict()
    data[member] = value
    with pytest.raises(ValueError):
        execute_candidate(env.sim, agent, data)
    assert not env.sim.instances


def test_ticket_rejects_mutated_candidate_parameters():
    env, agent, _, candidates, method = decision_task()
    data = method.to_dict()
    data['params']['goal'] = [0, 0]
    with pytest.raises(ValueError):
        execute_candidate(env.sim, agent, data, ticket=candidates.context)


def test_candidate_step_cannot_be_submitted_twice_before_its_result():
    env, agent, task, candidates, method = decision_task()
    execute_candidate(env.sim, agent, method, ticket=candidates.context)
    with pytest.raises(ValueError, match='pending'):
        execute_candidate(env.sim, agent, method)
    assert ACCEPTANCE.wait_idle(env, agent)
    assert task['method']['step_index'] == 1


@pytest.mark.parametrize('choice', [{'candidate': 'recovery-1'}, {'action': 'A066'}])
def test_edge_candidate_choice_preserves_parameters_task_and_recovery_scope(choice):
    candidate = {'candidate_id': 'recovery-1', 'skill_name': 'recover_replan',
                 'task_id': 'held-task', 'task_rev': 4, 'method_seq': 'Q098',
                 'step_index': 2, 'purpose': 'method', 'method_scope': 'recovery',
                 'action_id': 'A066', 'params': {'target': 'road-1', 'seq_id': 'Q098'}}
    journal, _ = client({'edge_decide': [completion({
        'actions': [choice], 'reason': 'recover'})]})
    result = execute_graph(build_edge, journal, {'vehicle': 'uav-1'},
                           initial(dict(EDGE_SITUATION, candidates=[candidate])))
    payload = result['outputs'][0]['payload']
    assert json.loads(payload['params_json']) == candidate['params']
    assert json.loads(payload['task_ref_json']) == ['held-task', 4]
    assert payload['method_scope'] == 'recovery'


@pytest.mark.parametrize('override', [{'purpose': 'approach'}, {'step_index': 99},
                                     {'method_scope': 'recovery'}, {'method_seq': 'Q098'}])
def test_edge_rejects_overridden_candidate_binding(override):
    _, _, _, _, method = decision_task()
    journal, _ = client({'edge_decide': [completion({'actions': [
        {'candidate': method.candidate_id, **override}], 'reason': 'tampered'})]})
    with pytest.raises(ProviderError):
        execute_graph(build_edge, journal, {'vehicle': 'uav-1'},
                      initial(dict(EDGE_SITUATION, candidates=[method.to_dict()])))


def test_engine_routes_task_ref_without_purpose_through_runtime(tmp_path):
    world, kernel = build_rescue_world(tmp_path)
    try:
        agent = world.sim.agents['uav-1']
        ACCEPTANCE.isolate(world.sim, agent)
        agent.policy_mode = 'external_harness'
        agent.spec['task_allowlist'] = ['T011']
        task = ACCEPTANCE.accept_task(agent, world.sim, 'T011', 'building-1')
        method = next(row for row in build_decision_candidates(agent, world.sim.catalog)
                      if row.purpose == 'method')
        payload = {'actor': agent.id, 'action_id': method.action_id,
                   'params': method.params, 'task_ref': [task['id'], task['rev']]}
        stale = world.handle_external_command('submit_action', dict(
            payload, task_ref=[task['id'], task['rev'] - 1], purpose='method'))
        assert stale['status'] == 'rejected'
        wrong = world.handle_external_command('submit_action', dict(payload, step_index=99))
        assert wrong['status'] == 'rejected'
        submitted = world.handle_external_command('submit_action', payload)
        assert submitted['status'] == 'submitted'
        assert agent.pending_actions[submitted['instance_id']]['purpose'] == 'method'
        assert ACCEPTANCE.wait_idle(world.env, agent)
        assert task['method']['step_index'] == 1
    finally:
        kernel.close()


@pytest.mark.parametrize('action,params,ref', [
    ('A067', {}, None), ('A001', {'_external_motion': True}, None),
    ('A001', {}, ['task-only']), ('A001', {}, 'task'),
])
def test_engine_rejects_malformed_action_and_binding_before_submission(tmp_path, action, params, ref):
    world, kernel = build_rescue_world(tmp_path)
    try:
        before = len(world.sim.instances)
        result = world.handle_external_command('submit_action', {
            'actor': 'uav-1', 'action_id': action, 'params': params, 'task_ref': ref})
        assert result['status'] == 'rejected'
        assert len(world.sim.instances) == before
    finally:
        kernel.close()


def test_memory_requires_original_digest_and_causal_delivery_times():
    envelope = MemoryEnvelope('mem', 'coordinator-1', ('uav-1',), 'report', {}, 8,
                              created_s=5, valid_until_s=20, delivered_at_s=10)
    assert not envelope.readable(9, 'uav-1')
    assert not envelope.readable(20, 'uav-1')
    envelope.delivered_at_s = 4
    assert not envelope.readable(12, 'uav-1')
    row = envelope.to_dict()
    row.pop('digest')
    with pytest.raises(ValueError, match='digest'):
        MemoryEnvelope.from_dict(row)
    from aerorescue.edge_harness import _observation
    observation = _observation({'now_s': 12, 'memory': [row]}, 'uav-1', {})
    assert observation['memory'] == [] and observation['memory_unreadable'] == 1


def test_multicast_delivery_does_not_make_another_receiver_readable():
    env = ACCEPTANCE.demo_env()
    sim = env.sim
    mid = sim.network.send('coordinator-1', ['uav-1', 'ground_vehicle-1'],
                           'report', {'op': 'report'}, 8, ttl_s=120)
    message = sim.network.messages[mid]
    sim.network._deliver(message, 'uav-1')
    envelope = MemoryEnvelope.from_message(message)
    assert envelope.readable(sim.t, 'uav-1')
    assert not envelope.readable(sim.t, 'ground_vehicle-1')
    pending = MemoryEnvelope.from_message(message, receiver='ground_vehicle-1')
    assert not pending.readable(sim.t, 'ground_vehicle-1')


def test_px4_arrival_requires_native_altitude_for_a_3d_goal():
    from rescue_scenes.px4_bridge import complete_reached_movements
    env, agent, task, _, method = decision_task()
    agent.motion_backend = 'px4'
    data = method.to_dict()
    data['params']['goal'] = [36.0, 52.0, 12.0]
    iid = execute_candidate(env.sim, agent, data)
    record = env.sim.instances[iid]
    goal = record['params']['_motion']['goal']
    env.sim.world.set(agent.platform_eid, 'entity.pose', list(goal), {'rule': 'TEST-PX4'})
    assert complete_reached_movements(env.sim, [agent.id]) == []
    assert complete_reached_movements(env.sim, [agent.id], positions_enu={agent.id: [36, 52, 0]}) == []
    assert record['status'] == 'running' and task['method']['step_index'] == 0
    assert complete_reached_movements(env.sim, [agent.id], positions_enu={agent.id: [36, 52, 12]}) == [iid]
    assert task['method']['step_index'] == 1
    assert complete_reached_movements(env.sim, [agent.id], positions_enu={agent.id: [36, 52, 12]}) == []


def test_px4_partial_telemetry_retains_armed_state():
    env = demo_environment()
    bridge = PX4EmbodimentBridge(env)
    bridge.apply_telemetry('uav-1', {'armed': True, 'velocity_enu': [1, 0, 0]})
    bridge.apply_telemetry('uav-1', {'velocity_enu': [2, 0, 0]})
    assert env.sim.world.get('uav-1', 'platform.motion_state') == 'moving'


def test_center_explicit_candidates_cannot_weaken_catalog_capabilities():
    situation = deepcopy(CENTER_SITUATION)
    situation['candidates'] = [{'id': 'weak', 'type': 'T029', 'target': 'human-1',
                                'capabilities': ['C01']}]
    journal, _ = client({'center_plan': [completion({'assignments': [
        {'candidate': 'weak', 'vehicle': 'uav-1'}], 'reason': 'weakened requirements'})]})
    with pytest.raises(ProviderError):
        execute_graph(build_center, journal, {}, initial(situation))


def test_center_rejects_duplicate_candidate_ids():
    from aerorescue.center_sandbox import derive_candidates
    with pytest.raises(ValueError, match='duplicate candidate'):
        derive_candidates({'candidates': [{'id': 'same', 'type': 'T029', 'target': target}
                                          for target in ('human-1', 'human-2')]})


@pytest.mark.parametrize('member', ['battery_remaining_Wh', 'payload_capacity_kg', 'payload_kg'])
@pytest.mark.parametrize('number', [float('nan'), -1.0])
def test_center_invalid_resources_cannot_bypass_constraints(member, number):
    situation = deepcopy(CENTER_SITUATION)
    situation['vehicles'][1][member] = number
    journal, _ = client({'center_plan': [completion({'assignments': [
        {'candidate': 'cand-000', 'vehicle': 'specialist_team-1'}], 'reason': 'bad resource'})]})
    with pytest.raises((ProviderError, ValueError)):
        execute_graph(build_center, journal, {}, initial(situation))


def test_perception_rejects_an_object_without_detections():
    with pytest.raises(ValueError, match='requires detections'):
        perception_qwen._content_rows({'choices': [{'message': {'content': '{"unexpected": 1}'}}]})


def test_model_completion_preserves_truncation_for_perception():
    from aerorescue.model_server import _completion
    reply = _completion('test-model', '[]', 3, 2, finish_reason='length')
    assert reply['choices'][0]['finish_reason'] == 'length'
    with pytest.raises(ValueError, match='exhausted max_tokens'):
        perception_qwen._content_rows(reply)


def test_execute_candidate_cannot_inject_external_motion_flags():
    env, agent, _, _, method = decision_task()
    data = method.to_dict()
    data['params']['_external_motion'] = True
    with pytest.raises(ValueError, match='kernel-internal'):
        execute_candidate(env.sim, agent, data)
    assert not env.sim.instances


@pytest.mark.parametrize('last_token,reason', [(5, 'length'), (9, 'stop')])
def test_local_generation_distinguishes_token_limit_from_eos(last_token, reason):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import threading
    from aerorescue.model_server import LocalQwen

    class Inputs(dict):
        def to(self, _device):
            return self

    class Processor:
        def __call__(self, **_kwargs):
            return Inputs(input_ids=np.array([[1]]))

        def batch_decode(self, _tokens, **_kwargs):
            return ['[]']

    engine = object.__new__(LocalQwen)
    engine.lock = threading.Lock()
    engine._prompt = lambda _messages: ('prompt', [])
    engine.processor = Processor()
    engine.torch = SimpleNamespace(inference_mode=nullcontext)
    engine.model = SimpleNamespace(device='cpu', generation_config=SimpleNamespace(eos_token_id=9),
                                   generate=lambda **_kwargs: np.array([[1, 4, last_token]]))
    assert engine.complete([], 2, 0)['finish_reason'] == reason
