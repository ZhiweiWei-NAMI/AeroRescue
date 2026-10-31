"""Read lane A2's map and stage display-only GLB origin/heading wrappers.

Original files remain untouched. Each wrapper transforms complete scene roots
(including skins), with source hash and transform recorded in display-assets.json.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import struct

from .glb_runtime import optimize, merge_by_material, opaque_blend, semantic_textures, quantize_geometry, write_glb
from .paths import PRIVATE_ROOT

# Display stand-ins: team = accepted adult human model; coordinator (command post) = container unit BLD-028-G02.
FLOATING={'boat','boat_loaded','human_in_water'}  # waterline-origin production assets (P3, P4)
ALIASES={'human':'person','specialist_team':'person','coordinator':'shelter','obstacle':'debris','station':'medical_point'}


def normalization_parameters(row,reference=None,size=None,waterline=False):
    """Shared AAS origin/heading transform, in its actual composition order."""
    if row['up']!='+y':raise ValueError('A2 GLB display contract requires +Y up')
    ref=reference or row
    turn={'+x':0,'-x':math.pi,'+z':math.pi/2,'-z':-math.pi/2}[row['forward']]
    lo,hi=ref['boundsM']['min'],ref['boundsM']['max']
    translate=([-v for v in ref['origin_m']] if 'origin_m' in ref else
               [-(lo[0]+hi[0])/2,0 if waterline else -lo[1],-(lo[2]+hi[2])/2])
    scale=[1,1,1] if size is None else [size[0]/ref['sizeM'][0],1,size[1]/ref['sizeM'][1]]
    return translate,turn,scale


def normalize_glb(source, target, row, size=None, reference=None, max_px=1024, waterline=False):
    data=source.read_bytes()
    magic,version,length=struct.unpack_from('<4sII',data)
    if magic!=b'glTF' or version!=2 or length!=len(data):
        raise ValueError(f'Invalid GLB: {source}')
    chunks=[]; offset=12
    while offset<len(data):
        count,kind=struct.unpack_from('<II',data,offset); offset+=8
        chunks.append((kind,data[offset:offset+count])); offset+=count
    doc=json.loads(chunks[0][1]); scene=doc['scenes'][doc.get('scene',0)]
    translate,turn,scale=normalization_parameters(row,reference,size,waterline)
    nodes=doc.setdefault('nodes',[]); index=len(nodes)
    nodes.append({'name':'rescue_center_ground','translation':translate,'children':list(scene['nodes'])})
    nodes.append({'name':'rescue_forward_east','rotation':[0,math.sin(turn/2),0,math.cos(turn/2)],'children':[index]})
    nodes.append({'name':'rescue_authored_footprint','scale':scale,'children':[index+1]})
    scene['nodes']=[index+2]
    doc,binary,runtime=optimize(doc,chunks[1][1] if len(chunks)>1 else b'',max_px)
    # Display-only batching and semantic texture budget (same chain as upstream assets; skinned/animated/morph
    # geometry is left unmerged by merge_by_material).
    doc,binary,merge=merge_by_material(doc,binary); doc,binary,alpha=opaque_blend(doc,binary); doc,binary,textures=semantic_textures(doc,binary)
    doc,binary,quantize=quantize_geometry(doc,binary)
    runtime={**runtime,'merge':merge,'alpha_mode':alpha,'semantic_textures':textures,'quantize':quantize}
    write_glb(target,doc,binary)
    return {'source_sha256':hashlib.sha256(data).hexdigest(),'translation':translate,'heading_y_rad':turn,'scale':scale,'runtime_variant':runtime,
            'note':'display-only origin/heading/footprint wrapper; simulation dimensions remain explicit embodiment/config inputs'}


def stage_models(root, directory, config):
    root=PRIVATE_ROOT if 'RESCUE_PRIVATE_ROOT' in os.environ else Path(root)
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    path=root/'assets/asset_map.json'
    if not path.exists():
        return {},{'mode':'labelled_glyphs','reason':'lane A2 asset map absent at materialization'}
    source=json.loads(path.read_text()); kinds=source['kinds']; assets=source.get('assets',{})
    if isinstance(assets,list): assets={a['asset_id']:a for a in assets}
    production=root/'assets/production_map.json'  # produced assets (same record format as A2's map)
    if production.exists():
        extra=json.loads(production.read_text()); kinds={**kinds,**extra['kinds']}; assets={**assets,**extra['assets']}
    chosen={key:kinds[name] for key,name in {**{k:k for k in ('robot_dog','uav','ground_vehicle','specialist_team','coordinator')},**ALIASES}.items() if name in kinds}
    for kind in set(config.get('display',{}).get('agent_models',{}).values()):
        chosen[kind]=kinds[kind]
    from .presentation import LOADED_VARIANT
    for kind,loaded in LOADED_VARIANT.items():
        if kind in config.get('display',{}).get('agent_models',{}).values():
            chosen[loaded]=kinds[loaded]
    for kind in config.get('display',{}).get('human_models',{}).values():
        chosen[kind]=kinds[kind]
    for rule in config.get('display',{}).get('state_models',{}).values():
        for kind in rule['values'].values():
            chosen[kind]=kinds[kind]
    for kind in ('human_prone','human_supine'):  # truth-driven pose variants of every person
        if kind in kinds: chosen[kind]=kinds[kind]
    if 'building' in kinds:
        base=kinds['building']
        for grade in ('D0','D1','D2','D3'):
            aid=base.get('states',{}).get(grade)
            if aid in assets: chosen['building_'+grade]=assets[aid]
            elif aid==base['asset_id']: chosen['building_'+grade]=base
    if 'debris' in kinds:
        aid=kinds['debris'].get('states',{}).get('opening')
        if aid in assets: chosen['obstacle_partly_cleared']=assets[aid]
    models={}; audit={'mode':'available_A2_models_else_labelled_glyphs','assets':{},'gaps':source.get('gaps',[])}
    for key,row in chosen.items():
        if key in config.get('display',{}).get('model_origins',{}):
            row={**row,'origin_m':config['display']['model_origins'][key]}
        source_path=root/row['file']
        if not source_path.is_file():
            audit['assets'][key]={'mode':'labelled_glyph','reason':'mapped GLB absent','file':row['file']}
            continue
        # External resources are copied by A2 only when actually supplied; never
        # declare a complete asset on the basis of an unresolved URI.
        if row.get('external_uris'):
            audit['assets'][key]={'mode':'labelled_glyph','reason':'GLB has external resource URIs','uris':row['external_uris']}
            continue
        filename=key+'-'+row['asset_id']+'.glb'; target=directory/filename
        size=None  # Real model footprint: the generator places entities with asset_map sizes, never a stretched model.
        reference=None
        if key.startswith('building_') and 'building' in kinds:
            reference=assets.get(kinds['building'].get('states',{}).get('D0'),kinds['building'])
        if row.get('runtime_static'):
            # AAS static entity layer rejects SkinnedMesh: use the faithful static export (same bounds).
            source_path=root/row['runtime_static']['file']
        transform=normalize_glb(source_path,target,row,size,reference,512 if key.startswith('building_') else 1024,key in FLOATING)
        # Content-addressed name so browsers never reuse a stale cached variant.
        digest8=hashlib.sha256(target.read_bytes()).hexdigest()[:8]
        final=target.with_name(target.stem+'-'+digest8+'.glb'); target.replace(final); target=final; filename=final.name
        models[key]={'url':'/data/rescue_assets/'+filename,'scale':1,'file':str(target)}
        audit['assets'][key]={'asset_id':row['asset_id'],'source':str(source_path),'transform':transform}
    return models,audit
