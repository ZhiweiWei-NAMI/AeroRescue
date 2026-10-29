"""Opt-in rescue requester for the installed AAS capture owner.

Perception contract: ``module:function(png: bytes, calibration: dict) -> list``.
Each detection is exactly ``{label: str, confidence: float, bbox_xyxy: [x0,y0,x1,y1]}``.
Coordinates are pixels, top-left outer-edge origin. No entity IDs, world labels,
render picking, depth or truth containers are supplied to the callable.

Periodic inference keeps only the latest waiting frame per camera, with a global
``perception_queue_size`` bound (default: camera count, at least one). Actions
have priority and are never dropped. ``perception_workers`` (8) bounds concurrent
calls; ``perception_action_reserve`` (up to 2, leaving one periodic worker) keeps
periodic calls from occupying all workers. Action inference starts on receipt,
while delivery still respects ``analysis_latency_s`` as a minimum delay. Results
arrive at the first tick at or after that delay when the call has finished;
``action_analysis_blocking`` (False) instead makes action frames stall the
lockstep at their due tick, for replays that need a wall-independent timeline
given the same detections. With a single worker no
capacity is reserved. ``analysis_min_interval_s`` (0) limits periodic call starts
per camera in simulation time; actions bypass it. Existing ``analysis_stride`` remains an
optional periodic eligibility filter, independent of ``analysis_latency_s``.
``perception_options`` passes Magpie's max_image_edge (768), max_tokens (4096),
timeout_s (300), reasoning_effort ('low'); the verified model route is unchanged.
Every capture still uses the committed scene cut; only inference is reused by
content hash. Reports distinguish real renders, unique PNGs, new inference
completions, cached analysis, skipped reasons, intervals, call walls and waits.
Expected throughput: no periodic inference waits/backlog; capture cost is
unchanged. The measured 1671 s/300 simulated s capture path and 143 s core-only
baseline preclude claiming 5-10x from these two files. SC01 outcome and new wall
cost must be measured centrally, not inferred from scheduling or cache hits.

Central validation commands from scene_sim (not executed by this lane):
python tools/make_run.py SC01 --seed 1
python - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path('configs/SC01.json').read_text())
cfg.update(seed=1, tick_s=0.5, until_s=300)
camera = cfg['display'].setdefault('camera_observation', {})
camera.update(enabled=True, replace_scripted=True,
              capture_interval_s=0.5, analysis_stride=1,
              perception='rescue_scenes.perception_magpie:detect',
              perception_workers=6, perception_queue_size=1,
              analysis_min_interval_s=0,
              perception_options={'max_image_edge': 768, 'max_tokens': 4096})
Path('/tmp/L27infer-SC01.json').write_text(json.dumps(cfg))
camera.pop('capture_interval_s')
Path('/tmp/L27infer-SC01-action.json').write_text(json.dumps(cfg))
PY
python tools/make_run.py SC01 --seed 1 --config /tmp/L27infer-SC01-action.json
python tools/make_run.py SC01 --seed 1 --config /tmp/L27infer-SC01.json
Read camera_observation in the returned run_dir/result.json; compare the default
seed-1 chain_timeline/objectives and inspect each frame's source_cut/camera_pose.
"""
from collections import Counter, OrderedDict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import hashlib
from importlib import import_module
import json
import math
import sys
from statistics import mean, median
from time import perf_counter

from aerokernel import CommandRequest, Stamp
from aerokernel.state import Fact
from aerokernel.values import thaw
from aeroagentsim.observations.artifacts import ArtifactStore
from aeroagentsim.observations.contracts import CaptureRequest

from .presentation import POSITION, YAW
from .core.agents.knowledge import camera_record_reference

# Authored camera contact-site radius (the shared SC01/GT04 policy suppresses a
# failed camera contact at a locally recorded site within 3 m). A re-sighting of
# an already contacted person inside this radius is that person, not a new
# contact target. Never used to establish a person's identity before contact.
CONTACT_SITE_RADIUS_M = 3.0


def contact_site_radius(spec):
    """Validated authored contact-site radius from an agent's local spec."""
    policy = spec.get('camera_search_policy') if isinstance(spec, dict) else None
    value = policy.get('contact_radius_m', CONTACT_SITE_RADIUS_M) if isinstance(policy, dict) else CONTACT_SITE_RADIUS_M
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        return CONTACT_SITE_RADIUS_M
    return float(value)


def contacted_people(local, kind_ids):
    """Local people this agent already contacted, posed at the delivered contact."""
    ids = [eid for eid in kind_ids[('human',)]
           if not eid.startswith('rescue-camera/') and 'human.needs' in local[eid]]
    rows = {}
    for eid in ids:
        fields = local[eid]
        pose = next((fields[key] for key in ('entity.pose', 'human.pose') if key in fields), None)
        rows[eid] = fields if pose is None else {**fields, 'belief.estimated_pose': pose}
    return ids, rows


def candidate_identity_alias(point, pose, spec):
    """Whether an executed camera contact may claim this person's identity.

    The physical contact radius (A009, 35 m) is not an identity radius; a stale
    candidate can be metres away from the person actually present.
    """
    if point is None or pose is None:
        return False
    return math.dist(point[:2], pose[:2]) <= contact_site_radius(spec)


def contact_site_matches(position, contacted, radius, used):
    """People already contacted whose delivered pose is within `radius` of this
    detection's own ground estimate. Never uncontacted people, never a person
    already consumed by another detection in the same frame."""
    if position is None or not contacted:
        return []
    matches = []
    for eid, fields in contacted.items():
        if eid in used:
            continue
        pose = next((fields[key]['value'] for key in
                     ('belief.estimated_pose', 'human.pose', 'entity.pose') if key in fields), None)
        if pose is None:
            continue
        distance = math.dist(position, pose[:2])
        if distance <= radius:
            matches.append((distance, eid, 'human'))
    return sorted(matches)


def configure(scenario, snapshot, config, directory, console_path, runtime, aas):
    """Extend only an enabled generated scenario with existing capture schemas."""
    from aeroagentsim.observations.schemas import (
        METADATA_SCHEMAS, REQUEST, CAPTURE_RECEIPT, STORAGE_RESULT,
        UPLOAD_RECEIPT, UPLOAD_VERIFIED,
    )
    from .backgrounds import digest

    owner = 'rescue_camera_capture'
    record_type = 'rescue:CaptureRecord'
    request_schema = 'rescue.camera.request'
    fields = {name: 'rescue.camera.' + name for name in METADATA_SCHEMAS}
    registry = snapshot['registry']
    registry['types'].append({'id': record_type, 'parents': [], 'abstract': False})
    registry['fields'].extend({'id': fields[name], 'declaring_type': record_type,
                               'schema': deepcopy(schema), 'metadata': {}}
                              for name, schema in METADATA_SCHEMAS.items())
    for name, kind, schema, result in (
        (request_schema, 'command', REQUEST, CAPTURE_RECEIPT),
        ('rescue.camera.storage_result', 'command', STORAGE_RESULT, UPLOAD_RECEIPT),
        ('rescue.camera.upload_verified', 'event', UPLOAD_VERIFIED, None),
    ):
        registry['messages'].append({'id': name, 'kind': kind, 'schema': deepcopy(schema),
                                    'result_schema': deepcopy(result), 'feedback_schema': None,
                                    'cancel_support': False})
    snapshot['details']['effective_fields'][record_type] = list(fields.values())
    snapshot['details']['statistics'].update({key: len(registry[key]) for key in ('types', 'fields', 'messages')})
    console = json.loads(console_path.read_text())
    urls = {console['city']['url']}
    urls.update(row['url'] for row in console.get('splats', []))
    urls.update(binding['visual']['asset'] for binding in scenario['presentation']
                if binding['visual'].get('kind') == 'model'
                and binding['visual'].get('asset', '').startswith('/'))
    for name in ('roads',):
        if name in console:
            urls.add(console[name]['url'])
    if 'hdri' in console:
        urls.add(console['hdri'])
    asset_digest = 'sha256:' + digest({'scene': console, 'presentation': scenario['presentation'],
                                       'display_assets': json.loads((directory / 'display-assets.json').read_text())})
    inventory_path = directory / 'capture-assets.json'
    inventory_path.write_text(json.dumps({'format': 'aeroagentsim.capture-assets/v1',
        'environment': 'viewer-default/v1', 'asset_id': asset_digest,
        'files': [{'url': url} for url in sorted(urls)]}, ensure_ascii=False))
    renderer = {'mode': 'traffic_browser', 'node_modules': str(aas / 'frontend/node_modules'),
                'frontend_dist': str(runtime / 'frontend_dist'), 'timeout_s': config.get('timeout_s', 120),
                'software_gl': False, **config.get('renderer', {}), 'asset_digest': asset_digest,
                'console_scene_path': str(console_path.resolve()),
                'capture_assets_path': str(inventory_path.resolve())}
    scenario['engines'][owner] = {'plugin': 'traffic_camera_capture', 'config': {
        'request_schema': request_schema, 'storage_result_schema': 'rescue.camera.storage_result',
        'accepted_schema': 'rescue.camera.upload_verified', 'accepted_topic': 'rescue.camera.uploads',
        'record_type': record_type, 'record_id_prefix': 'rescue-image/', 'fields': fields,
        'run_directory': '@run', 'renderer': renderer}}
    scenario['bindings']['rules'].append({'writer': owner, 'type': record_type,
                                           'fields': list(fields.values()), 'ids': '*'})
    scenario['bindings']['lifecycle'].append({'controller': owner, 'type': record_type, 'ids': '*'})
    scenario['engines']['rescue']['config']['camera_observation'] = {
        'capture_target': owner, 'request_schema': request_schema, 'asset_digest': asset_digest}


class CameraObservations:
    def __init__(self, world, config, route):
        self.world, self.cfg, self.route = world, config, route
        self.actions = set(config.get('actions', ['A005', 'A008', 'A009', 'A010', 'A011', 'A018']))
        self.cameras = config.get('agents', {'uav-1': {'pitch_deg': -60, 'fov_deg': 70}})
        if config.get('replace_scripted'):
            for actor in self.cameras:
                world.sim.agents[actor].spec['camera_replace_scripted'] = True
        self.capture_interval_s = config.get('capture_interval_s')
        if self.capture_interval_s is not None:
            self.capture_interval_s = float(self.capture_interval_s)
            tick_ns = round(world.config['tick_s'] * 1e9)
            interval_ns = round(self.capture_interval_s * 1e9) if math.isfinite(self.capture_interval_s) else 0
            if (not math.isfinite(self.capture_interval_s) or self.capture_interval_s <= 0
                    or interval_ns <= 0 or not math.isclose(interval_ns / 1e9, self.capture_interval_s, rel_tol=0, abs_tol=1e-12)):
                raise ValueError('camera capture_interval_s must be positive, finite and representable in nanoseconds')
            if interval_ns < tick_ns or interval_ns % tick_ns:
                raise ValueError('camera capture_interval_s must be an integer multiple of tick_s; set tick_s to a compatible value')
            self.interval_ns = interval_ns
            self.analysis_stride = config.get('analysis_stride', 1)
            if type(self.analysis_stride) is not int or self.analysis_stride < 1:
                raise ValueError('camera analysis_stride must be an integer >= 1')
            self.periodic_agents = config.get('periodic_agents', list(self.cameras))
            if len(set(self.periodic_agents)) != len(self.periodic_agents) or any(actor not in self.cameras for actor in self.periodic_agents):
                raise ValueError('camera periodic_agents must be distinct configured camera agents')
        # Runtime-only scheduling knob: a shorter GIL switch interval lets the
        # render/perception worker threads hand results back to the tick loop
        # sooner. It changes no capture time, latency or perception semantics.
        switch_s = config.get('python_switch_interval_s')
        if switch_s is not None:
            if type(switch_s) not in (int, float) or not math.isfinite(switch_s) or switch_s <= 0:
                raise ValueError('camera python_switch_interval_s must be positive and finite')
            sys.setswitchinterval(float(switch_s))
        self.pending = deque()
        self.submitted = deque()
        self.commands = {}
        self.jobs = {}
        self.records = []
        self.store = ArtifactStore(world.output)
        self.analysis_latency_s = float(config.get('analysis_latency_s', 30))
        if not math.isfinite(self.analysis_latency_s) or self.analysis_latency_s < 0:
            raise ValueError('camera analysis_latency_s must be finite and nonnegative')
        self.perception_workers = config.get('perception_workers', 8)
        self.perception_action_reserve = config.get('perception_action_reserve',
                                                    min(2, self.perception_workers - 1))
        if (type(self.perception_action_reserve) is not int
                or not 0 <= self.perception_action_reserve < self.perception_workers):
            raise ValueError('camera perception_action_reserve must be an integer >= 0 and < perception_workers')
        self.periodic_worker_limit = self.perception_workers - self.perception_action_reserve
        self.action_analysis_blocking = config.get('action_analysis_blocking', False)
        if type(self.action_analysis_blocking) is not bool:
            raise ValueError('camera action_analysis_blocking must be a boolean')
        self.workers = ThreadPoolExecutor(max_workers=self.perception_workers,
                                          thread_name_prefix='rescue-perception')
        # Only periodic work is replaceable. Task-gating action work is retained
        # and dispatched first; periodic calls leave capacity for receipt-time
        # action dispatch instead of making actions wait for unrelated inference.
        self.perception_queue_size = config.get('perception_queue_size', max(1, len(self.cameras)))
        if type(self.perception_queue_size) is not int or self.perception_queue_size < 1:
            raise ValueError('camera perception_queue_size must be an integer >= 1')
        self.analysis_min_interval_s = float(config.get('analysis_min_interval_s', 0))
        if not math.isfinite(self.analysis_min_interval_s) or self.analysis_min_interval_s < 0:
            raise ValueError('camera analysis_min_interval_s must be finite and nonnegative')
        self.action_queue = deque()
        self.periodic_queue = OrderedDict()
        self.running = {}
        self.running_periodic = set()
        self.last_analysis_s = {}
        self.inference_calls = []
        self.detector_cache = {}
        # Only the last optical frame per agent; moving cameras replace it.
        self._projection_pixels = {}
        self.origin_enu = [world.grid.e[0], world.grid.n[0]]
        # Static public terrain, attached only to positions already in this
        # holder's map. No person/state labels enter this initialization.
        for actor in self.cameras:
            agent = world.sim.agents[actor]
            for eid, fields in agent.belief.entities(copy=False).items():
                position = fields.get('entity.pose', {}).get('value')
                if position is not None:
                    agent.belief.put(world.sim, eid, 'map.ground_height_enu_m',
                                     world.grid.enu(position[:2])[2], 1., 'prior')

    def action(self, event):
        if (event.kind == 'action.status' and event.after == 'running'
                and event.actor in self.cameras):
            instance = self.world.sim.instances[event.action_instance]
            if instance['action_id'] in self.actions:
                self.pending.append((event.actor, instance['id']))
                agent = self.world.sim.agents[event.actor]
                ref = instance['task_ref']
                task = agent.tasks.get(ref[0]) if ref else None
                if task and task['rev'] == ref[1]:
                    task['progress'].setdefault('pending_camera_analysis', {})[instance['id']] = {
                        'action_instance': instance['id'], 'target': task['target']}

    def periodic(self, now_ns):
        self.pump_inference()
        # Called once after each lockstep step; k starts at 1, not bootstrap.
        if self.capture_interval_s is not None and now_ns > 0 and now_ns % self.interval_ns == 0:
            index = now_ns // self.interval_ns
            for actor in self.periodic_agents:
                self.pending.append((actor, None, index))

    def emit(self, kind, actor, payload, text):
        self.world.sim.emit(layer='knowledge', family=kind, kind=kind, grounding='proposed',
            refs=[], subject=actor, actor=actor, visible_to=[actor], after=deepcopy(payload),
            backend_mode='camera_image', scripted=False, text_zh=text)

    def request(self, ctx):
        while self.pending:
            pending = self.pending.popleft()
            actor, instance_id = pending[:2]
            periodic_index = pending[2] if len(pending) == 3 else None
            ref = self.world.refs[actor]
            ctx.get(ref, POSITION)
            ctx.get(ref, YAW)
            pose = ctx.view.field((ref, POSITION), ctx.now)
            yaw = ctx.view.field((ref, YAW), ctx.now)
            if not isinstance(pose, Fact) or not isinstance(yaw, Fact):
                raise ValueError(f'Camera actor has no committed pose: {actor}')
            mount = self.cameras[actor]
            offset = mount.get('offset_enu_m', [0, 0, 0])
            camera = {'revision': 'rescue-agent-camera/v1', 'frame': 'enu',
                      'pose_enu': [float(pose.value[k]) + offset[k] for k in range(3)],
                      'yaw_deg': float(yaw.value) + mount.get('yaw_offset_deg', 0),
                      'pitch_deg': mount['pitch_deg'], 'fov_deg': mount['fov_deg'],
                      'calibration': True}
            request_suffix = instance_id if periodic_index is None else f'periodic/{periodic_index}'
            request = CaptureRequest(f'rescue-camera/{actor}/{request_suffix}', ref.run_id, ref,
                ctx.view.cut, camera, self.route['asset_digest'],
                Stamp('canonical', ctx.view.cut.instant.ns, 1, 'canonical'),
                self.cfg.get('width', 1920), self.cfg.get('height', 1080), self.cfg.get('timeout_s', 120))
            agent = self.world.sim.agents[actor]
            instance = self.world.sim.instances[instance_id] if instance_id is not None else None
            target = instance['params'].get('target', instance['params'].get('target_id')) if instance else None
            ground_eid = target
            ground = agent.belief.get(target, 'map.ground_height_enu_m')
            if ground is None:
                # Own position and the public terrain map establish a local
                # horizontal ground plane when the action has no mapped target.
                height = self.world.grid.enu(agent.platform['pose'])[2]
                agent.belief.put(self.world.sim, actor, 'map.ground_height_enu_m', height, 1., 'prior')
                ground_eid = actor
                ground = agent.belief.get(actor, 'map.ground_height_enu_m')
            captured_s = request.source_cut.instant.ns / 1e9
            tick = self.world.sim.config.get('tick_s', 5)
            due_s = math.ceil((captured_s + self.analysis_latency_s) / tick) * tick
            self.jobs[request.request_id] = {'request': request, 'instance': instance_id,
                'captured_s': captured_s, 'analysis_due_s': due_s,
                'ground_height_enu_m': ground['value'], 'ground_height_entity': ground_eid}
            if periodic_index is not None:
                self.jobs[request.request_id].update(periodic_index=periodic_index,
                    analyze=(periodic_index - 1) % self.analysis_stride == 0)
            self.world.sim.schedule(due_s - self.world.sim.t, self.deliver_analysis, request.request_id)
            ref = instance['task_ref'] if instance else None
            task = agent.tasks.get(ref[0]) if ref else None
            if task and task['rev'] == ref[1]:
                task['progress']['pending_camera_analysis'][instance_id].update(
                    request_id=request.request_id, analysis_due_s=due_s)
            ctx.submit(CommandRequest(self.route['request_schema'], self.route['capture_target'],
                                      ctx.now, request.to_command_data()))
            self.submitted.append(request.request_id)
            self.emit('camera_image.requested', actor, {
                **camera_record_reference({'request_id': request.request_id}),
                'request': {'request_id': request.request_id},
                'action_instance': instance_id, 'analysis_latency_s': self.analysis_latency_s,
                'analysis_due_s': due_s}, '请求在已提交源切面拍摄本机相机图像。')

    def receive(self, ctx):
        for dirty in ctx.dirty:
            if dirty.kind != 'receipt':
                continue
            payload = thaw(dirty.payload)
            command, status = payload['command_id'], payload['status']
            if status == 'submitted':
                self.commands[command] = self.submitted.popleft()
            elif status in {'succeeded', 'failed', 'rejected', 'canceled'}:
                identity = self.commands.pop(command)
                # This is the terminal receipt of our submitted capture command,
                # not the upload/storage_result route. The capture owner succeeds
                # that command only after rendering and storing this run's frame.
                self.deliver(identity, status, payload['result'], own_capture=True)

    def deliver(self, identity, status, receipt, *, own_capture=False):
        """Capture receipt hook: submit image work, never wait for a detector."""
        job = self.jobs[identity]
        request = job['request']
        agent = self.world.sim.agents[request.actor.id]
        row = {'source': 'camera_image', 'agent': agent.id, 'action_instance': job['instance'],
               'request_id': identity, 'observation_id': identity + '/image',
               'status': status, 'receipt': receipt, 'captured_s': job['captured_s'],
               'analysis_latency_s': self.analysis_latency_s, 'analysis_due_s': job['analysis_due_s'],
               'ground_height_enu_m': job['ground_height_enu_m'],
               'ground_height_entity': job['ground_height_entity'], 'ground_height_source': 'public_terrain_prior'}
        if 'periodic_index' in job:
            row.update(capture_kind='periodic', periodic_index=job['periodic_index'])
        job['row'] = row
        self.records.append(row)
        if status != 'succeeded':
            row['request'] = request.to_data()
            return
        if receipt['request_id'] != identity or receipt['status'] != 'succeeded':
            raise ValueError('Rescue camera receipt/request mismatch')
        record = receipt['record']
        # Only our in-process capture owner can hand off already validated bytes.
        # Receipts/uploads alone never grant trust; an evicted handoff takes the
        # unchanged disk validation path. No private metadata enters evidence.
        captured = (self.store.take_capture(record['digest'], expected=(request.width, request.height))
                    if own_capture else None)
        if captured is None:
            png = self.store.read(record['digest'], expected=(request.width, request.height),
                                  trusted_png=own_capture and record['renderer_mode'] == 'browser')
            content_sha256 = hashlib.sha256(png).hexdigest()
        else:
            png, content_sha256 = captured.png, captured.sha256
        # AAS's legacy digest slot is a request-addressed artifact identifier.
        # Record a separate content hash; do not claim that identifier is SHA256.
        image = {'artifact_id': record['digest'], 'digest': record['digest'],
                 'content_sha256': content_sha256, 'byte_count': len(png),
                 'source_cut': record['request']['source_cut'], 'acquired': record['request']['acquired'],
                 'camera': record['request']['camera'], 'calibration': record['calibration'],
                 'camera_pose': record['camera_pose'], 'renderer_mode': record['renderer_mode']}
        row.update(image)
        job['image'] = image
        detector = self.cfg.get('perception')
        if not job.get('analyze', True):
            self.skip_inference(job, 'analysis_stride')
        elif detector is None:
            row['perception_status'] = 'not_configured'
            row['message'] = 'image acquired; perception model not configured'
        else:
            job['png'] = png
            row['perception_status'] = 'queued'
            if job['instance'] is not None:
                self.action_queue.append(identity)
            else:
                previous = self.periodic_queue.pop(agent.id, None)
                if previous is not None:
                    self.skip_inference(self.jobs[previous], 'superseded_by_latest_frame', identity)
                self.periodic_queue[agent.id] = identity
                while len(self.periodic_queue) > self.perception_queue_size:
                    _, displaced = self.periodic_queue.popitem(last=False)
                    self.skip_inference(self.jobs[displaced], 'periodic_queue_limit', identity)
            self.pump_inference()
        self.emit('camera_image.analysis_pending', agent.id, camera_record_reference(row),
                  '实际 PNG 已取得；推理调度状态已记录，尚未交付检测。')

    @staticmethod
    def skip_inference(job, reason, replacement=None):
        row = job['row']
        row.update(perception_status='not_scheduled', perception_skip_reason=reason,
                   message='image acquired; perception not scheduled: ' + reason)
        if replacement is not None:
            row['perception_replacement_request_id'] = replacement
        job.pop('png', None)

    def cached_inference(self, job):
        cached = self.detector_cache.get(job['image']['content_sha256'])
        # Cache inference only, never capture. Failed calls remain real failures
        # and a later actual capture can retry via the normal recovery path.
        if cached is not None and cached[1].done() and cached[1].exception() is not None:
            return None
        return cached

    def start_inference(self, identity):
        job = self.jobs[identity]
        row, request = job['row'], job['request']
        cached = self.cached_inference(job)
        row['perception_cache_hit'] = cached is not None
        if cached is None:
            metadata = {}
            calibration = deepcopy(job['image']['calibration'])
            calibration.update(camera_pose=deepcopy(job['image']['camera_pose']),
                perception_options=deepcopy(self.cfg.get('perception_options', {})),
                perception_metadata=metadata)
            future = self.workers.submit(self.detect, self.cfg['perception'], job['png'],
                                          calibration, request.width, request.height)
            cached = self.detector_cache[job['image']['content_sha256']] = (identity, future, metadata)
            self.running[identity] = future
            if job['instance'] is None:
                self.running_periodic.add(identity)
            self.inference_calls.append({'request_id': identity, 'agent': request.actor.id,
                'captured_s': job['captured_s'], 'started_s': self.world.sim.t,
                'future': future, 'metadata': metadata})
            self.last_analysis_s[request.actor.id] = self.world.sim.t
        row['perception_cache_request_id'], job['future'], job['inference_metadata'] = cached
        row.update(perception_status='pending', inference_scheduled_s=self.world.sim.t)
        job.pop('png', None)

    def pump_inference(self):
        # Only the simulation thread changes queues/rows. Worker completion is
        # polled at existing ticks/receipts, not allowed to mutate agent state.
        self.running = {identity: future for identity, future in self.running.items() if not future.done()}
        self.running_periodic.intersection_update(self.running)
        for identity in list(self.action_queue):
            if len(self.running) >= self.perception_workers and self.cached_inference(self.jobs[identity]) is None:
                continue
            self.action_queue.remove(identity)
            self.start_inference(identity)
        if self.action_queue:
            return
        for actor, identity in list(self.periodic_queue.items()):
            job = self.jobs[identity]
            cached = self.cached_inference(job)
            if cached is None:
                if (len(self.running) >= self.perception_workers
                        or len(self.running_periodic) >= self.periodic_worker_limit):
                    break
                if self.world.sim.t - self.last_analysis_s.get(actor, -math.inf) < self.analysis_min_interval_s:
                    continue
            del self.periodic_queue[actor]
            self.start_inference(identity)

    def detect(self, detector, png, calibration, width, height):
        started = perf_counter()
        try:
            module, name = detector.split(':', 1)
            detections = getattr(import_module(module), name)(png, calibration)
            self.validate_detections(detections, width, height)
            return detections, perf_counter() - started
        finally:
            calibration['perception_metadata']['wall_s'] = perf_counter() - started

    def later_identical(self, job):
        """Latest later periodic frame sharing this frame's inference call.

        While the call is pending any later identical frame qualifies (it will
        wait for the outcome); once finished, only frames already due do, so
        the result is not held back."""
        if job['instance'] is not None or 'future' not in job:
            return None
        pending = not job['future'].done()
        return max((other for other in self.jobs.values() if other is not job
                    and other['instance'] is None and other.get('future') is job['future']
                    and other['request'].actor.id == job['request'].actor.id
                    and other['captured_s'] > job['captured_s']
                    and (pending or other['analysis_due_s'] <= self.world.sim.t)),
                   key=lambda other: other['captured_s'], default=None)

    def deliver_analysis(self, identity):
        """Deliver at or after the due tick; only blocking action frames stall lockstep."""
        self.pump_inference()
        job = self.jobs[identity]
        if 'row' not in job:
            # Async capture: the receipt for this cut has not arrived yet. The
            # image cannot be analysed before it exists; defer and record it.
            job['receipt_deferred_ticks'] = job.get('receipt_deferred_ticks', 0) + 1
            self.world.sim.schedule(self.world.sim.config.get('tick_s', 5), self.deliver_analysis, identity)
            return
        request, row = job['request'], job['row']
        if job.get('receipt_deferred_ticks'):
            row['receipt_deferred_ticks'] = job['receipt_deferred_ticks']
        later = self.later_identical(job)
        if (job['instance'] is None or not self.action_analysis_blocking) and later is None and (
                row.get('perception_status') == 'queued' or ('future' in job and not job['future'].done())):
            # analysis_latency_s is a minimum delivery delay, not a sampling
            # interval. Keep the actual source cut and declare any late delivery.
            row['analysis_deferred_ticks'] = row.get('analysis_deferred_ticks', 0) + 1
            self.world.sim.schedule(self.world.sim.config.get('tick_s', 5), self.deliver_analysis, identity)
            return
        agent = self.world.sim.agents[request.actor.id]
        row.update(analysis_delivered_s=self.world.sim.t, analysis_wall_wait_s=0.,
                   analysis_delivery_lateness_s=max(0., self.world.sim.t - job['analysis_due_s']))
        detections = []
        waiting = row.get('perception_status') == 'queued' or ('future' in job and not job['future'].done())
        started = perf_counter()
        # Action frames are never dropped. Blocking ones wait here at their due
        # tick only if their actual inference (or a hash-identical capture's) needs it.
        while row.get('perception_status') == 'queued':
            wait(tuple(self.running.values()), return_when=FIRST_COMPLETED)
            self.pump_inference()
        if later is not None:
            # A later hash-identical periodic frame shares this inference call and
            # carries its outcome (detections or failure). This frame still
            # delivers its own image at its due tick, so one slow or failed call
            # cannot release hundreds of identical results in a single tick.
            row.update(perception_status='superseded_by_identical_frame',
                       detections_superseded_by=later['request'].request_id,
                       perception_inference=deepcopy(job['inference_metadata']))
        elif 'future' in job:
            try:
                detected, elapsed = job['future'].result()
                detections = deepcopy(detected)
                row['perception_wall_s'] = elapsed
                row['perception_status'] = 'succeeded'
            except Exception as error:
                row.update(perception_status='failed', perception_error=f'{type(error).__name__}: {error}')
            finally:
                row['analysis_wall_wait_s'] = perf_counter() - started if waiting else 0.
                row['perception_inference'] = deepcopy(job['inference_metadata'])
                if 'wall_s' in job['inference_metadata']:
                    row['perception_wall_s'] = job['inference_metadata']['wall_s']
            if row['perception_status'] == 'failed':
                self.emit('camera_image.perception_failed', agent.id, camera_record_reference(row),
                          '图像已取得；感知调用失败，未生成检测。')
        self.jobs.pop(identity)
        self.pump_inference()
        row['detections'] = detections
        derived = self.observations(agent, job, job['image'], detections) if row['status'] == 'succeeded' else []
        row['detection_observations'] = deepcopy(derived)
        # Wall timings are diagnostics only. Including them in agent evidence
        # would change report byte counts/network costs with worker jitter.
        evidence_row = {key: deepcopy(value) for key, value in row.items()
                        if key not in {'analysis_wall_wait_s', 'perception_wall_s', 'perception_inference',
                                       'inference_scheduled_s', 'analysis_delivery_lateness_s'}}
        observation = {'id': identity + '/image', 'by': agent.id,
            't': request.source_cut.instant.ns / 1e9, 'about': [agent.id], 'kind': 'camera_image',
            'source': 'camera_image', 'model': 'committed_browser', 'confidence': 1.0,
            'values': {'observation.camera_image': evidence_row}}
        agent.on_camera_observations(self.world.sim, observation, derived)
        self.emit('camera_image.analysis_delivered', agent.id, {
            'request_id': identity, 'action_instance': job['instance'],
            'analysis_latency_s': self.analysis_latency_s, 'analysis_due_s': job['analysis_due_s'],
            'analysis_delivered_s': self.world.sim.t, 'analysis_wall_wait_s': row['analysis_wall_wait_s'],
            'analysis_delivery_lateness_s': row['analysis_delivery_lateness_s'],
            **({'perception_wall_s': row['perception_wall_s']} if 'perception_wall_s' in row else {})},
            '交付实际相机证据；周期帧可晚于最早交付 tick，墙钟耗时仅供诊断。')
        if row['status'] != 'succeeded':
            self.emit('camera_image.failed', agent.id, camera_record_reference(row),
                      '图像采集失败；未生成检测。')
            return
        # report().observations retains the full row, including receipt, route
        # metadata and diagnostics. Live evidence above stays unchanged: network
        # costs and A039 hashing depend on its original serialized contents.
        image_event = {**observation, 'values': {
            'observation.camera_image': camera_record_reference(evidence_row)}}
        self.emit('camera_image.acquired', agent.id, image_event,
                  row.get('message', '已取得实际 PNG 图像与相机标定。'))
        for detection in derived:
            # Evaluation consumes about/source and domain claims; image-space
            # provenance is recoverable by this ID in the full frame's derived
            # observations, not repeated for every detection event.
            detection_event = {**detection,
                'values': {key: value for key, value in detection['values'].items()
                           if not key.startswith('observation.')},
                'record_ref': {**image_event['values']['observation.camera_image']['record_ref'],
                               'observation_id': detection['id']}}
            self.emit('camera_image.detection', agent.id, detection_event,
                      '图像空间检测进入本机证据；关联仅使用本地认知或搜索范围。')

    @staticmethod
    def validate_detections(detections, width, height):
        if not isinstance(detections, list):
            raise TypeError('Perception must return a list of image-space detections')
        for detection in detections:
            if not isinstance(detection, dict) or set(detection) != {'label', 'confidence', 'bbox_xyxy'}:
                raise ValueError('Detection requires only label, confidence, bbox_xyxy; no truth IDs')
            if not isinstance(detection['label'], str) or not detection['label']:
                raise ValueError('Detection label must be a nonempty string')
            confidence, box = detection['confidence'], detection['bbox_xyxy']
            if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError('Detection confidence must be in [0,1]')
            if (not isinstance(box, (list, tuple)) or len(box) != 4
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                    or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
                raise ValueError('Detection bbox_xyxy must use in-frame pixel coordinates')

    def observations(self, agent, job, image, detections):
        observations = []
        if not detections:
            return observations
        local = agent.belief.entities(copy=False)
        kind_ids = {kinds: list(agent.belief.entity_ids(*kinds))
                    for kinds in (('human',), ('building',), ('road', 'obstacle'))}
        track_ids = [eid for eid in agent.belief.entity_ids('human', 'building', 'road', 'obstacle')
                     if eid.startswith('rescue-camera/') and '/person/' in eid]
        # Periodic detections can arrive before their A041 batch is integrated.
        # They are already delivered local evidence, not additional detections.
        for pending in agent._camera_pending.values():
            fields = pending['values']
            if fields.get('entity.kind') == 'human' and 'belief.estimated_pose' in fields:
                for eid in pending['about']:
                    if eid not in local:
                        local[eid] = {key: {'value': value} for key, value in fields.items()}
                        local[eid]['map.ground_height_enu_m'] = {
                            'value': fields['observation.ground_height_enu_m']}
                        for ids in kind_ids.values():
                            ids.append(eid)
                        if eid.startswith('rescue-camera/') and '/person/' in eid:
                            track_ids.append(eid)
        for eid in track_ids:
            fields = local[eid]
            estimate = fields.get('belief.estimated_pose', {})
            # Preserve the ground plane used to locate an unconfirmed track;
            # the camera may now be above different terrain.
            for oid in reversed(estimate.get('evidence_ids', [])):
                evidence = agent.belief.evidence.get(oid)
                if evidence is None:
                    continue
                evidence_values = evidence['values'] if isinstance(evidence, dict) else evidence.values
                ground = evidence_values.get('observation.ground_height_enu_m')
                if ground is not None:
                    local[eid] = {**fields, 'map.ground_height_enu_m': {'value': ground}}
                    break
        used_people = set()
        prior_tracks = set(track_ids)
        # Reuse pixels, never associations. repr keeps signed zero and numeric
        # spelling distinct; only the latest exact camera/calibration is held.
        optical_key = repr((image['camera_pose']['T_world_enu_from_camera_optical'],
                            image['calibration']['K'], self.origin_enu,
                            self.cfg.get('association_height_m', 1)))
        cached = self._projection_pixels.get(agent.id)
        if cached is None or cached[0] != optical_key:
            cached = self._projection_pixels[agent.id] = (optical_key, {})
        pixels = cached[1]
        # New local tracks extend each per-delivery list without reprojecting
        # its old prefix. Each detection still performs the original box test.
        projections = {}
        candidates = None
        candidate_ids = None
        contacted = None
        contacted_ids = None
        instance = self.world.sim.instances[job['instance']] if job['instance'] is not None else None
        target = instance['params'].get('target', instance['params'].get('target_id')) if instance else None
        for i, detection in enumerate(detections):
            about, values = [], {'observation.image_detection': deepcopy(detection),
                'observation.parent_evidence_ids': [job['request'].request_id + '/image'],
                'observation.analysis_latency_s': self.analysis_latency_s,
                'observation.analysis_due_s': job['analysis_due_s']}
            if detection['label'].lower() in {'person', 'human'}:
                # A crowded box is not evidence of another person. Rank local
                # tracks by distance to its centre instead of minting a new ID.
                matches = self.projected_matches(local, ('human',), detection['bbox_xyxy'], image,
                                                  job['ground_height_enu_m'], _projections=projections,
                                                  _entity_ids=kind_ids[('human',)], _pixels=pixels)
                association = 'projected_local_belief'
                if not matches:
                    # An unconfirmed track was placed at the box's foot, not at
                    # association_height_m. Reuse it when its ground point is
                    # inside this box's calibrated ground footprint. This uses
                    # the measured box itself, not an arbitrary world radius.
                    if candidates is None:
                        candidate_ids = [eid for eid in kind_ids[('human',)]
                                         if eid.startswith('rescue-camera/') and '/person/' in eid]
                        candidates = {eid: local[eid] for eid in candidate_ids}
                    matches = self.projected_matches(candidates, ('human',), detection['bbox_xyxy'], image,
                                                      job['ground_height_enu_m'], height_m=0,
                                                      _projections=projections,
                                                      _entity_ids=candidate_ids, _pixels=pixels)
                    association = 'bbox_ground_footprint'
                if not matches:
                    # A person this agent already contacted has a delivered
                    # contact pose. Test that foot point against this box the
                    # same way, so a re-sighting is not a fresh contact target.
                    if contacted is None:
                        contacted_ids, contacted = contacted_people(local, kind_ids)
                    matches = self.projected_matches(contacted, ('human',), detection['bbox_xyxy'], image,
                                                      job['ground_height_enu_m'], height_m=0,
                                                      _projections=projections,
                                                      _entity_ids=contacted_ids, _pixels=pixels)
                    association = 'contacted_ground_footprint'
                if not matches:
                    # A detector box puts a contacted person's foot roughly a
                    # metre from the delivered contact pose, so both strict
                    # tests above can miss and mint a duplicate contact track.
                    # Test this box's own ground estimate against the authored
                    # contact-site radius, only for people already contacted.
                    if contacted is None:
                        contacted_ids, contacted = contacted_people(local, kind_ids)
                    box = detection['bbox_xyxy']
                    # One ground estimate serves both remaining fallbacks.
                    position = self.ground_position(box, image, job['ground_height_enu_m'])
                    matches = contact_site_matches(position, contacted,
                                                   contact_site_radius(agent.spec), used_people)
                    if matches:
                        association = 'contacted_contact_site_radius'
                    if not matches and position is not None:
                        # A foot estimate can jitter outside the next box. Use
                        # its measured bottom-edge ground extent, not a fixed
                        # metric/pixel tolerance or a true person's position.
                        corners = [self.ground_position([x, box[1], x, box[3]], image,
                                                        job['ground_height_enu_m'])
                                   for x in (box[0], box[2])]
                        if all(point is not None for point in corners):
                            radius = max(math.dist(position, point) for point in corners)
                            matches = self.projected_matches(
                                candidates, ('human',), box, image, job['ground_height_enu_m'],
                                height_m=0, _projections=projections, _entity_ids=candidate_ids,
                                _pixels=pixels, _ground_support=(position, radius))
                            # Relaxed support cannot consume another detection
                            # in this frame, including a newly minted track.
                            matches = [match for match in matches
                                       if match[1] in prior_tracks and match[1] not in used_people]
                            association = 'bbox_ground_uncertainty'
                            if not matches:
                                matches = [match for match in self.projected_matches(
                                    contacted, ('human',), box, image, job['ground_height_enu_m'],
                                    height_m=0, _projections=projections, _entity_ids=contacted_ids,
                                    _pixels=pixels, _ground_support=(position, radius))
                                    if match[1] not in used_people]
                                association = 'contacted_ground_uncertainty'
                available = [match for match in matches if match[1] not in used_people]
                if available:
                    about = [available[0][1]]
                    used_people.add(about[0])
                    values.update({'entity.kind': 'human', 'belief.human_presence': 'suspected',
                                   'observation.association': association})
                    if len(matches) > 1:
                        values['observation.association_candidates'] = [match[1] for match in matches]
                elif matches:
                    # Keep the raw overlapping detection but do not turn a
                    # second box on an already-used track into another identity.
                    values['observation.localization_status'] = 'ambiguous_same_frame_track'
                    values['observation.association_candidates'] = [match[1] for match in matches]
                else:
                    if position is not None:
                        # This ID belongs only to belief. It is never inserted
                        # into the physical world or counted as a found person.
                        about = [job['request'].request_id + f'/person/{i}']
                        used_people.add(about[0])
                        values.update({'entity.kind': 'human', 'belief.estimated_pose': position,
                            'belief.human_presence': 'suspected',
                            'observation.ground_height_enu_m': job['ground_height_enu_m'],
                            'observation.association': 'bbox_ground_projection'})
                        local[about[0]] = {key: {'value': value} for key, value in values.items()}
                        local[about[0]]['map.ground_height_enu_m'] = {'value': job['ground_height_enu_m']}
                        kind_ids[('human',)].append(about[0])
                        if candidates is not None:
                            candidates[about[0]] = local[about[0]]
                            candidate_ids.append(about[0])
                    else:
                        values['observation.localization_status'] = 'ray_does_not_intersect_local_ground'
                        if target in local:
                            about = [target]
                            values.update({'belief.human_presence': 'suspected',
                                           'observation.association': 'unlocalized_search_scope'})
            elif detection['label'] in {'building_damage_none', 'building_damage_minor',
                                         'building_damage_major', 'building_damage_destroyed'}:
                matches = self.projected_matches(local, ('building',), detection['bbox_xyxy'], image,
                                                  job['ground_height_enu_m'], _projections=projections,
                                                  _entity_ids=kind_ids[('building',)], _pixels=pixels)
                if matches:
                    about = [matches[0][1]]
                    values.update({'building.visual_damage': detection['label'].removeprefix('building_damage_'),
                                   'observation.association': 'projected_local_belief'})
            elif detection['label'] == 'road_debris':
                matches = self.projected_matches(local, ('road', 'obstacle'), detection['bbox_xyxy'], image,
                                                  job['ground_height_enu_m'], _projections=projections,
                                                  _entity_ids=kind_ids[('road', 'obstacle')], _pixels=pixels)
                if matches:
                    _, eid, kind = matches[0]
                    about = [eid]
                    values['observation.association'] = 'projected_local_belief'
                    if kind == 'obstacle':
                        values['obstacle.state'] = 'present'
                    # Debris on a known road is image evidence, not a measured
                    # clearance or a claim that a particular vehicle is blocked.
            observations.append({'id': job['request'].request_id + f'/detection/{i}', 'by': agent.id,
                't': job['request'].source_cut.instant.ns / 1e9, 'about': about, 'kind': 'camera_image',
                'source': 'camera_image', 'values': values,
                'confidence': detection['confidence'] * self.cfg.get('confidence', .6),
                'model': self.cfg['perception']})
        return observations

    def projected_matches(self, local, kinds, box, image, ground_height, *, height_m=None,
                          _projections=None, _entity_ids=None, _pixels=None, _ground_support=None):
        key = (id(local), kinds, height_m)
        cached = _projections.get(key) if _projections is not None else None
        cursor, points = cached if cached is not None else (0, [])
        ids = list(local) if _entity_ids is None else _entity_ids
        if cursor < len(ids):
            prefixes = [(candidate, candidate + '.') for candidate in kinds]
            for index in range(cursor, len(ids)):
                eid = ids[index]
                fields = local[eid]
                if fields.get('belief.entity_ref', {}).get('value', eid) != eid:
                    continue  # A contacted local track has been associated to its observed person.
                kind = fields.get('entity.kind', {}).get('value')
                if kind not in kinds:
                    kind = next((candidate for candidate, prefix in prefixes
                                 if any(key.startswith(prefix) for key in fields)), None)
                if kind is None:
                    continue
                position = None
                for field in ('belief.estimated_pose', 'human.pose', 'entity.pose'):
                    if field in fields:
                        position = fields[field]['value']
                        break
                if position is not None:
                    ground = fields.get('map.ground_height_enu_m', {}).get('value', ground_height)
                    pixel_key = (eid, height_m)
                    previous = _pixels.get(pixel_key) if _pixels is not None else None
                    ground_key = repr(ground)
                    # Belief writes replace position values; keep the object,
                    # not its id, so replacement cannot alias an old cache key.
                    if previous is not None and previous[0] is position and previous[1] == ground_key:
                        pixel = previous[2]
                    else:
                        pixel = self.project(position, image, ground, height_m=height_m)
                        if _pixels is not None:
                            _pixels[pixel_key] = (position, ground_key, pixel)
                    if pixel is not None:
                        points.append((pixel, eid, kind))
            if _projections is not None:
                _projections[key] = (len(ids), points)
        centre = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        if _ground_support is not None:
            position, radius = _ground_support
            matches = []
            for pixel, eid, kind in points:
                fields = local[eid]
                estimate = next((fields[field]['value'] for field in ('belief.estimated_pose', 'human.pose', 'entity.pose')
                                 if field in fields), None)
                if estimate is not None and math.dist(position, estimate[:2]) <= radius:
                    matches.append((math.dist(pixel, centre), eid, kind))
            return sorted(matches)
        return sorted((math.dist(pixel, centre), eid, kind) for pixel, eid, kind in points
                      if box[0] <= pixel[0] <= box[2] and box[1] <= pixel[1] <= box[3])

    def project(self, position, image, ground_height, *, height_m=None):
        if ground_height is None:
            return None
        enu = [position[k] + self.origin_enu[k] for k in (0, 1)] + [
            ground_height + (self.cfg.get('association_height_m', 1) if height_m is None else height_m)]
        matrix = image['camera_pose']['T_world_enu_from_camera_optical']
        delta = [enu[k] - matrix[k][3] for k in range(3)]
        # Keep the same sum/products in the same order, without three generator
        # objects for every accumulated person-track projection.
        optical = [sum((matrix[0][j] * delta[0], matrix[1][j] * delta[1], matrix[2][j] * delta[2]))
                   for j in range(3)]
        if optical[2] <= 0:
            return None
        k = image['calibration']['K']
        return [k[0] * optical[0] / optical[2] + k[2], k[4] * optical[1] / optical[2] + k[5]]

    def ground_position(self, box, image, ground_height):
        k = image['calibration']['K']
        y = (box[3] - k[5]) / k[4]
        x = ((box[0] + box[2]) / 2 - k[2] - k[1] * y) / k[0]
        matrix = image['camera_pose']['T_world_enu_from_camera_optical']
        origin = [matrix[i][3] for i in range(3)]
        ray = [matrix[i][0] * x + matrix[i][1] * y + matrix[i][2] for i in range(3)]
        if ray[2] == 0:
            return None
        distance = (ground_height - origin[2]) / ray[2]
        if distance <= 0:
            return None
        return [origin[i] + distance * ray[i] - self.origin_enu[i] for i in (0, 1)]

    def report(self):
        captured = [row for row in self.records if row['status'] == 'succeeded']
        completed = [call for call in self.inference_calls if call['future'].done()]
        walls = [call['metadata']['wall_s'] for call in completed if 'wall_s' in call['metadata']]
        intervals, by_agent = [], {}
        for actor in self.cameras:
            # Source-frame spacing, not wall time or analysis_latency_s. Cache
            # hits are not new inference calls and cannot inflate this metric.
            times = sorted(call['captured_s'] for call in self.inference_calls if call['agent'] == actor)
            gaps = [b - a for a, b in zip(times, times[1:])]
            intervals.extend(gaps)
            by_agent[actor] = {'inference_calls': len(times),
                'effective_inference_interval_sim_s': mean(gaps) if gaps else None}
        frames_with_analysis = sum(row.get('perception_status') == 'succeeded' or (
            row['request_id'] in self.jobs and 'future' in self.jobs[row['request_id']]
            and self.jobs[row['request_id']]['future'].done()
            and self.jobs[row['request_id']]['future'].exception() is None) for row in captured)
        # Reused detections are explicit evidence reuse, not inference throughput.
        frames_analyzed = sum(call['future'].exception() is None for call in completed)
        skipped = Counter(row['perception_skip_reason'] for row in self.records if 'perception_skip_reason' in row)
        waits = [row.get('analysis_wall_wait_s', 0.) for row in self.records]
        report = {'source': 'camera_image', 'perception': self.cfg.get('perception'),
                'analysis_latency_s': self.analysis_latency_s,
                'perception_workers': self.perception_workers,
                'python_switch_interval_s': sys.getswitchinterval(),
                'perception_action_reserve': self.perception_action_reserve,
                'periodic_worker_limit': self.periodic_worker_limit,
                'action_analysis_blocking': self.action_analysis_blocking,
                'perception_queue_size': self.perception_queue_size,
                'analysis_min_interval_s': self.analysis_min_interval_s,
                'inference_policy': 'action_priority_latest_per_camera',
                'perception_options': deepcopy(self.cfg.get('perception_options', {})),
                'perception_unique_images': len(self.detector_cache),
                'frames_captured': len(captured),
                'unique_images': len({row['content_sha256'] for row in captured}),
                'frames_analyzed': frames_analyzed, 'analyzed_frames': frames_analyzed,
                'frames_with_analysis': frames_with_analysis,
                'perception_cache_hits': sum(row.get('perception_cache_hit', False) for row in captured),
                'frames_analysis_delivered': sum(row.get('perception_status') == 'succeeded' for row in captured),
                'inference_calls': len(self.inference_calls), 'inference_calls_completed': len(completed),
                'inference_calls_failed': sum(call['future'].exception() is not None for call in completed),
                'inference_wall_mean_s': mean(walls) if walls else None,
                'inference_wall_median_s': median(walls) if walls else None,
                'effective_inference_interval_sim_s': mean(intervals) if intervals else None,
                'inference_by_agent': by_agent,
                'skipped_counts': dict(skipped), 'frames_skipped': sum(skipped.values()),
                'frames_detections_superseded': sum('detections_superseded_by' in row for row in self.records),
                'not_analyzed_frames': sum(row.get('perception_status') in {'not_scheduled', 'not_configured'} for row in captured),
                'frames_inference_queued': sum(row.get('perception_status') == 'queued' for row in captured),
                'frames_analysis_pending': sum(row.get('perception_status') == 'pending' for row in captured),
                'analysis_wait_count': sum(value > 0 for value in waits),
                'analysis_wall_wait_s': sum(waits), 'analysis_wall_wait_max_s': max(waits, default=0.),
                'replace_scripted': self.cfg.get('replace_scripted', False),
                'sensing_by_agent': {actor: self.sensing(actor) for actor in self.cameras},
                'inference_requests': [{key: deepcopy(value) for key, value in call.items() if key != 'future'}
                                       for call in completed],
                'observations': self.records,
                'pending': [job['request'].to_data() for job in self.jobs.values()]}
        if self.capture_interval_s is not None:
            report.update(capture_interval_s=self.capture_interval_s, analysis_stride=self.analysis_stride,
                periodic_frames=sum(row.get('capture_kind') == 'periodic' for row in self.records),
                action_frames=sum(row.get('capture_kind') != 'periodic' for row in self.records))
        return report

    def sensing(self, actor):
        if actor not in self.cameras:
            return 'scripted_v0.5'
        detector = self.cfg.get('perception')
        if detector == 'rescue_scenes.perception_magpie:detect':
            label = 'camera_image+deepseek via magpie'
        else:
            label = 'camera_image+' + detector if detector else 'camera_image (perception not configured)'
        if not self.cfg.get('replace_scripted'):
            label += '+scripted_v0.5'
        return label

    def close(self):
        # Do not advance simulated time or deliver unfinished analysis at close.
        self.workers.shutdown(wait=False, cancel_futures=True)
