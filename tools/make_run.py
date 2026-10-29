#!/usr/bin/env python3
"""Materialize one config, launch the existing AAS CLI, then register its replay.

No package install, server lifecycle, frontend build, detector, or ns-3 process.
--materialize-only writes inputs without invoking the simulation.
"""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import os
import signal
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
# In the exported src layout, scripts live under scene_env and the package
# lives in the clone's src directory; locally the package is beside tools.
sys.path[:0]=[str(ROOT),str(ROOT.parent/'src')]
from rescue_scenes.paths import AAS_ROOT, DATA_ROOT, PRIVATE_ROOT, expand
from rescue_scenes.backgrounds import materialize, digest
from rescue_scenes.assets import stage_models
from rescue_scenes.core.world.truth import generate
from rescue_scenes.core.agents.runtime import build_agents
from rescue_scenes.core.catalog import load
from rescue_scenes.presentation import BASE,AGENT,BELIEF,FIELDS,type_of,display_extras,bindings,registry_snapshot,event_narratives

AAS=AAS_ROOT
RUNTIME=DATA_ROOT/'blockout_geometry_20261023/marble_aas_visual_20261024/runtime'
GENERATED=(DATA_ROOT/'scene_sim/tools/generated' if 'RESCUE_DATA_ROOT' in os.environ
           else ROOT/'tools/generated')


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def stage_background(config,console,runtime):
    bg=config['background']; delivery=Path(bg['delivery_root'])/bg['id']
    entries=[(console['city']['url'],delivery/'geometry')]
    entries.extend((row['url'],delivery/'splats') for row in console.get('splats',[]))
    for url,source_dir in entries:
        if not url.startswith('/data/'):
            raise ValueError('Background asset must be a local /data/ URL: '+url)
        target=runtime/'frontend_dist'/url.lstrip('/')
        if target.is_file(): continue  # Reuse the accepted background's staged bytes.
        source=source_dir/Path(url).name
        if not source.is_file(): raise FileNotFoundError(f'Background asset missing in runtime and delivery: {source}')
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)


def conclude_if_incomplete(run_dir, generated, runtime, cfg, config_digest):
    """Keep a compact conclusion for an incomplete attempt, then drop its bulk data."""
    result=json.loads((run_dir/'result.json').read_text())
    objectives=result['objectives']; camera=result.get('camera_observation') or {}
    camera_enabled=bool(cfg['display'].get('camera_observation',{}).get('enabled'))
    camera_ok=(not camera_enabled or camera.get('frames_captured',0)>0 and
               camera.get('inference_calls_completed',0)>camera.get('inference_calls_failed',0))
    if result.get('run_complete') and objectives and all(row['satisfied'] for row in objectives) and camera_ok:
        return None
    if (run_dir.resolve().parent!=(runtime/'out_cli').resolve() or
            generated.resolve().parent!=GENERATED.resolve()):
        raise ValueError('Cleanup target is outside this run output or generated-input directory')
    summary={'run_id':run_dir.name,'status':'incomplete','seed':cfg['seed'],
             'tick_s':cfg['tick_s'],'until_s':cfg['until_s'],'config_digest':config_digest,
             'objectives':objectives,'failures_by_cause':result.get('failures_by_cause',{}),
             'camera':{key:camera.get(key) for key in ('frames_captured','frames_analysis_delivered',
                       'inference_calls','inference_calls_completed','inference_calls_failed')},
             'tasks':{},'rescue_events':[]}
    task_status={}; generated_counts={}
    for line in (run_dir/'events.jsonl').open():
        row=json.loads(line); kind=row.get('kind',''); after=row.get('after')
        task_type=str(row.get('subject','')).split('@')[0]
        if row.get('field')=='task.target_ref':
            generated_counts[task_type]=generated_counts.get(task_type,0)+1
        if row.get('family')=='sm:task.status':
            key=task_type+':'+str(after);task_status[key]=task_status.get(key,0)+1
        if kind.startswith('rescue.'):
            summary['rescue_events'].append({key:row.get(key) for key in
                ('seq','t','kind','subject','actor','task','task_rev','action_instance','backend_mode','scripted','after')})
    summary['tasks']={'generated':generated_counts,'status':task_status}
    output=runtime/'conclusions'/run_dir.name;output.mkdir(parents=True,exist_ok=True)
    write(output/'summary.json',summary)
    report=subprocess.run([sys.executable,str(ROOT/'tools/summarize_run.py'),str(run_dir)],
                          capture_output=True,text=True)
    (output/'summary.txt').write_text(report.stdout)
    shutil.rmtree(run_dir)
    return {'run_id':run_dir.name,'run_dir':None,'status':'incomplete',
            'summary_dir':str(output),'raw_artifacts_deleted':True,'generated_inputs_retained':str(generated),'url':None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('SID'); parser.add_argument('--seed',type=int)
    parser.add_argument('--config',type=Path); parser.add_argument('--runtime',type=Path,default=RUNTIME)
    parser.add_argument('--aas',type=Path,default=AAS); parser.add_argument('--materialize-only',action='store_true')
    parser.add_argument('--forced-rules',action='store_true')
    parser.add_argument('--scripted-sensing',action='store_true',help='Disable camera capture explicitly; no image acceptance')
    parser.add_argument('--bundle',type=Path,help='Reuse verified materialized inputs without scene generation')
    parser.add_argument('--timeout',type=float,default=300,help='AAS subprocess timeout in seconds')
    args=parser.parse_args()
    cfg=expand(json.loads((args.config or ROOT/'configs'/f'{args.SID}.json').read_text()))
    asset_map=cfg.get('placement',{}).get('asset_map')
    if 'RESCUE_PRIVATE_ROOT' in os.environ and asset_map and not Path(asset_map).is_absolute():
        cfg['placement']['asset_map']=str(PRIVATE_ROOT/asset_map)
    if cfg['background']['id']!=args.SID: raise ValueError('SID and config background differ')
    if args.seed is not None: cfg['seed']=args.seed
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory=GENERATED/f"{args.SID}-{cfg['seed']}-{stamp}"
    if args.bundle:
        source=args.bundle.resolve()
        bundle=json.loads(source.read_text())
        upstream=bundle['audit'].get('upstream')
        if upstream:
            product=Path(upstream['scene_json'])
            manifest=json.loads(Path(upstream['manifest']).read_text())
            scene=json.loads(product.read_text())
            expected_id=cfg['placement']['upstream_generated_scene'].get('upstream_scene_id',args.SID)
            if (scene['scene_id']!=expected_id or scene['seed']!=cfg['seed'] or
                    hashlib.sha256(product.read_bytes()).hexdigest()!=manifest['generation']['scene_sha256']):
                raise ValueError('Reused scene id/seed/manifest hash differs')
        check=json.loads(json.dumps(cfg))
        check['background']['grid']=bundle['config']['background']['grid']
        if digest(check)!=bundle['config_digest']:
            raise ValueError('--bundle config differs from requested config')
        directory.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source.parent/bundle['core']['semantics']['path'],directory/'semantics.json')
        # Only display mode changes; the materialized scene and native core stay identical.
        cfg=bundle['config']
    else:
        bundle_path,bundle=materialize(cfg,directory)
    if args.scripted_sensing:
        cfg['display'].setdefault('camera_observation',{})['enabled']=False
        bundle['config']=cfg
        bundle['config_digest']=digest(cfg)
    bundle_path=directory/'bundle.json'
    write(bundle_path,bundle)
    # Artifact names use the project id; the core scenario_id is a hazard
    # profile and upstream_scene_id independently identifies imported terrain.
    SID=bundle['config']['background']['id']
    # Generate initial entities only: no Simulation, start hook or forward step.
    doc=json.loads((directory/'semantics.json').read_text())
    world=generate(doc,bundle['core']['scenario_id'],bundle['core']['seed'],options=bundle['core']['truth_options'])
    build_agents(bundle['core'],world,load())
    models,asset_audit=stage_models(Path(os.environ.get('RESCUE_ASSET_ROOT',str(ROOT))),directory/'assets',cfg)
    snapshot=registry_snapshot(); imported_types={}; imported_presentation=[]
    if 'upstream' in bundle['audit']:
        from rescue_scenes.upstream_scene import bind_assets
        imported_types,imported_presentation=bind_assets(bundle,directory,snapshot,models,asset_audit)
    write(directory/'display-assets.json',asset_audit)
    write(directory/'registry.snapshot.json',snapshot)
    scenario={'format':'aeroagentsim.scenario/v1','id':f'{SID}-rescue-{stamp}','provenance':'full',
              'registry':{'snapshot':'registry.snapshot.json'},
              'entities':[{'id':eid,'type':imported_types.get(eid) or type_of(row,cfg['display'].get('agent_models')),'facts':{}} for eid,row in sorted(world.entities.items())]+
                         [{'id':xid,'type':typ,'facts':{}} for _,row in sorted(world.entities.items()) for xid,typ,_r in display_extras(row,cfg['display'].get('agent_models'),cfg['display'].get('human_models'),cfg['display'].get('state_models'))],
              'bindings':{'rules':[{'writer':'rescue','type':BASE,'fields':list(FIELDS[:-1]),'ids':'*'},
                                   {'writer':'rescue','type':AGENT,'fields':[BELIEF],'ids':'*'}],
                          'lifecycle':[{'controller':'rescue','type':BASE,'ids':'*'}],'commands':[]},
              'engines':{'rescue':{'plugin':'rescue_scenes','config':{'bundle_path':str(bundle_path.resolve())}}},
              'presentation':bindings(models)+imported_presentation,
              'event_narratives':event_narratives(),
              'clock_mappings':[{'clock_id':'canonical','mapping_id':'canonical'}],
              'run':{'seed':cfg['seed'],'until_ns':round(cfg['until_s']*1e9),'advance_ns':round(cfg['tick_s']*1e9),'pacing':'fast'},
              'outputs':{'durability':'flush'}}
    if args.forced_rules:
        scenario['engines']['rescue']['config']['execution_mode']='forced_rules'
    scenario_path=directory/f'{SID}_rescue.scenario.json'
    console=json.loads(Path(cfg['background']['console_source']).read_text())
    mission=cfg['display'].get('mission_label','earthquake rescue')
    camera=cfg['display'].get('camera_observation')
    sensing='camera image + scripted sensing' if camera and camera.get('enabled') else 'scripted sensing'
    console['id']=f"{SID} {mission} · generated layout · {sensing} · seed {cfg['seed']}"
    # Frame the generated incident (source building + its debris) instead of a fixed authored view (three = [E,U,-N]).
    layout=bundle['audit']['layout']; offset=cfg['display']['camera_offset_enu']
    focus=[layout[eid]['enu'] for eid,row in layout.items() if row['kind']=='obstacle']
    focus+=[layout[row['relations']['source']]['enu'] for row in layout.values() if row['kind']=='obstacle']
    if 'camera_focus_entities' in cfg['display']:
        focus=[layout[eid]['enu'] for eid in cfg['display']['camera_focus_entities']]
    elif not focus:
        # Non-building missions have no debris; frame their real deployment,
        # without inventing an incident or changing any task state.
        focus=[layout[agent['id']]['enu'] for agent in cfg['agents']]
    if not focus:
        raise ValueError(f'{SID}: no camera focus entities or agent deployment configured')
    target=[sum(p[k] for p in focus)/len(focus) for k in range(3)]
    three=lambda p:[p[0],p[2],-p[1]]
    console['camera']={'position':three([target[k]+offset[k] for k in range(3)]),'target':three(target),'fixed':True}
    console['attribution']+='; disaster entities and v0.5 sensing/action/network models are authored simulation assumptions'
    if camera and camera.get('enabled'):
        console['attribution']+='; configured agent cameras provide rendered image evidence through '+str(camera.get('perception'))
    console_path=directory/'console-scene.json'; write(console_path,console)
    if camera and camera.get('enabled'):
        # Capture uses the selected --aas and the same staged scene/models as
        # this run. Disabled configurations retain the original scenario bytes.
        sys.path[:0]=[str(args.aas/'src'),str(args.aas/'aerokernel')]
        from rescue_scenes.camera import configure
        configure(scenario,snapshot,camera,directory,console_path,args.runtime.resolve(),args.aas.resolve())
        write(directory/'registry.snapshot.json',snapshot)
    write(scenario_path,scenario)
    if args.materialize_only:
        print(json.dumps({'scenario':str(scenario_path),'console':str(console_path),'assets':asset_audit},ensure_ascii=False))
        return
    runtime=args.runtime.resolve(); (runtime/'out_cli').mkdir(parents=True,exist_ok=True)
    stage_background(cfg,console,runtime)
    asset_dir=runtime/'frontend_dist/data/rescue_assets'; asset_dir.mkdir(parents=True,exist_ok=True)
    for row in models.values(): shutil.copy2(row['file'],asset_dir/Path(row['file']).name)
    env=os.environ.copy(); env['PYTHONPATH']=os.pathsep.join([str(args.aas.resolve()/'src'),str(args.aas.resolve()/'aerokernel'),str(ROOT/'site'),str(ROOT),str(ROOT.parent/'src'),env.get('PYTHONPATH','')])
    command=[os.environ.get('RESCUE_PYTHON',str(args.aas/'.venv/bin/python')),'-m','aeroagentsim.services.cli','run',str(scenario_path),'--out',str(runtime/'out_cli')]
    proc=subprocess.Popen(command,cwd=args.aas,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          text=True,start_new_session=True)
    try:
        stdout,stderr=proc.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL)  # Only the process group created above belongs to this launch.
        stdout,stderr=proc.communicate()
        write(directory/'failure.json',{'stage':'aas','status':'timeout','timeout_s':args.timeout,
              'argv':command,'cwd':str(args.aas.resolve()),'pid':proc.pid,'returncode':proc.returncode,
              'cleanup':'task-owned subprocess group killed and waited','generated_inputs_retained':str(directory),
              'stdout_tail':stdout[-4000:],'stderr_tail':stderr[-4000:]})
        raise SystemExit('AAS timeout; compact diagnostic and valid inputs retained: '+str(directory/'failure.json'))
    proc.stdout,proc.stderr=stdout,stderr
    (directory/'aas.stdout.log').write_text(proc.stdout); (directory/'aas.stderr.log').write_text(proc.stderr)
    if proc.returncode:
        sys.stderr.write(proc.stderr); sys.stderr.write(proc.stdout)
        raise SystemExit(f'AAS failed ({proc.returncode}); exact logs: {directory}')
    run_value=json.loads(proc.stdout.strip().splitlines()[-1])['run']
    run_dir=Path(run_value)
    if not run_dir.is_absolute(): run_dir=runtime/'out_cli'/run_dir.name
    if not run_dir.is_dir(): raise FileNotFoundError(f'AAS returned a nonexistent run directory: {run_dir}')
    conclusion=(None if args.forced_rules else conclude_if_incomplete(run_dir,directory,runtime,cfg,bundle['config_digest']))
    rule_status=None
    if args.forced_rules:
        rule_report=json.loads((run_dir/'forced-rules.json').read_text())
        rule_status='complete' if rule_report['all_objectives_satisfied'] else 'partial'
        if rule_status=='partial':
            removed=[]
            # This run was just returned by this task's subprocess. Keep replay records and valid inputs.
            if run_dir.resolve().parent != (runtime/'out_cli').resolve():
                raise ValueError('Rule cleanup target outside this task runtime')
            for artifact in run_dir.rglob('*'):
                if artifact.is_file() and (artifact.suffix.lower() in {'.png','.jpg','.jpeg'} or
                        artifact.suffix=='.log' and artifact.stat().st_size>1_000_000):
                    removed.append(str(artifact.resolve())); artifact.unlink()
            output=runtime/'conclusions'/run_dir.name
            output.mkdir(parents=True,exist_ok=True)
            write(output/'summary.json',dict(rule_report,removed_bulk=removed,generated_inputs_retained=str(directory)))
    if conclusion is not None:
        print(json.dumps(conclusion,ensure_ascii=False)); return
    shutil.copy2(console_path,run_dir/'console-scene.json')
    shutil.copy2(directory/'display-assets.json',run_dir/'display-assets.json')
    write(run_dir/'scene-source.json',{'kind':'fixed-glb','id':run_dir.name})
    # Use an actual staged console path in the existing TSV registration format.
    replay_scene=runtime/'frontend_dist/data/rescue_runs'/run_dir.name/'console-scene.json'
    replay_scene.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(console_path,replay_scene)
    with (runtime/'run_scenes.tsv').open('a') as handle:
        handle.write(f'{run_dir.name}\t{replay_scene.relative_to(runtime.parent)}\n')
    print(json.dumps({'run_id':run_dir.name,'run_dir':str(run_dir),
                      'execution_mode':'forced_rules' if args.forced_rules else 'autonomous','status':rule_status or 'complete',
                      'url':f"http://127.0.0.1:18931/?run={run_dir.name}&sel={cfg['agents'][0]['id']}"},ensure_ascii=False))


if __name__=='__main__': main()
