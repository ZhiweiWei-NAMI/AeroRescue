"""Local Qwen-VL camera perception for the AAS capture owner.

Contract (see ``rescue_scenes.camera``): ``detect(png, calibration) -> list`` of
``{label, confidence, bbox_xyxy}`` in pixels of the calibrated capture. The
callable receives image bytes and camera geometry only; it never sees entity
ids, world truth or scene metadata.

The model call targets a local OpenAI-compatible multimodal endpoint
(``AERO_QWEN_VL_ENDPOINT`` / ``AERO_QWEN_VL_MODEL``, default
``http://127.0.0.1:8788/v1/chat/completions`` and ``Qwen3.5-2B-Base``). When the
endpoint is unreachable the detector returns the operator-declared
``perception_options.fallback_detections`` (normalized [0,1] boxes) and records
that fallback in ``perception_metadata``; with no declaration it returns no
detections. A malformed model reply is a real failure and propagates.
"""
import base64
from io import BytesIO
import json
import math
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

DEFAULT_ENDPOINT='http://127.0.0.1:8788/v1/chat/completions'
DEFAULT_MODEL='Qwen3.5-2B-Base'
MAX_TOKENS=1024
TIMEOUT_S=120
MAX_IMAGE_EDGE=768
LABELS=frozenset({
    'person', 'building_damage_none', 'building_damage_minor',
    'building_damage_major', 'building_damage_destroyed', 'road_debris',
})
PROMPT="""Detect only objects visibly supported by the supplied rendered image.
Return ONLY a JSON array, with [] if there are no supported detections. Each item
must have exactly these keys: label, confidence, bbox_xyxy. confidence is a number
in [0,1]. bbox_xyxy is [x0,y0,x1,y1] normalized to [0,1] over the supplied image,
with top-left origin, x increasing right and y increasing down.
Use ONLY these labels:
- person: a visible human body or head, including standing, seated or lying people
  and people swimming, floating or holding a flotation ring in water; small
  distant people count.
- building_damage_none: an intact building, with no visible structural damage.
- building_damage_minor: a standing building with visible cracks or spalling,
  without partial collapse.
- building_damage_major: a partially collapsed building with substantial roof,
  wall or floor loss, but a recognizable standing structure remains.
- building_damage_destroyed: a building reduced to a rubble pile with little or
  no standing structure.
- road_debris: visible rubble, fragments or other debris on a road surface.
For building grades, box the whole visible building or its collapsed remains;
choose one grade per building. Box road debris itself, not the whole road. Debris
presence does not establish vehicle passability. Do not infer people hidden
inside buildings, unseen objects, identities, needs or injuries. Do not return
entity IDs, explanations, markdown or any labels outside this list.
"""


def _options(calibration):
    options=calibration.get('perception_options') or {}
    if not isinstance(options,dict):
        raise TypeError('perception_options must be a dict')
    endpoint=options.get('endpoint') or os.environ.get('AERO_QWEN_VL_ENDPOINT') or DEFAULT_ENDPOINT
    model=options.get('model') or os.environ.get('AERO_QWEN_VL_MODEL') or DEFAULT_MODEL
    max_tokens=options.get('max_tokens',MAX_TOKENS)
    timeout_s=options.get('timeout_s',TIMEOUT_S)
    max_image_edge=options.get('max_image_edge',MAX_IMAGE_EDGE)
    for name,value in (('max_tokens',max_tokens),('max_image_edge',max_image_edge)):
        if type(value) is not int or value<=0:
            raise ValueError(f'{name} must be a positive integer')
    if type(timeout_s) not in (int,float) or not math.isfinite(timeout_s) or timeout_s<=0:
        raise ValueError('timeout_s must be a finite positive number')
    for name,value in (('endpoint',endpoint),('model',model)):
        if not isinstance(value,str) or not value.strip():
            raise ValueError(f'{name} must be a nonempty string')
    if not endpoint.startswith(('http://','https://')):
        raise ValueError('endpoint must be an absolute HTTP(S) URL')
    return endpoint,model,max_tokens,timeout_s,max_image_edge


def _request_image(png,max_image_edge):
    # Pillow is optional. PNG IHDR still yields the actual transmitted size.
    if png[:8]!=b'\x89PNG\r\n\x1a\n' or png[12:16]!=b'IHDR' or len(png)<24:
        raise ValueError('Qwen-VL image detector requires PNG image bytes')
    width=int.from_bytes(png[16:20],'big'); height=int.from_bytes(png[20:24],'big')
    try:
        from PIL import Image
    except ImportError:
        return png,width,height
    with Image.open(BytesIO(png)) as image:
        if max(image.size)<=max_image_edge:
            return png,image.width,image.height
        image.thumbnail((max_image_edge,max_image_edge),Image.Resampling.LANCZOS)
        output=BytesIO(); image.save(output,format='PNG')
        return output.getvalue(),image.width,image.height


def _content_rows(completion):
    choice=completion['choices'][0]
    if choice.get('finish_reason')=='length':
        raise ValueError('Qwen-VL detection response exhausted max_tokens')
    content=choice['message']['content']
    if isinstance(content,list):
        content=''.join(part.get('text','') for part in content if isinstance(part,dict))
    content=str(content).strip()
    fenced=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',content,flags=re.DOTALL|re.IGNORECASE)
    rows=json.loads(fenced.group(1) if fenced else content)
    if isinstance(rows,dict):
        if 'detections' not in rows:
            raise ValueError('Qwen-VL object response requires detections')
        rows=rows['detections']
    if not isinstance(rows,list):
        raise TypeError('Qwen-VL detector must return a JSON array')
    return rows


def _pixels(rows,width,height):
    detections=[]
    for row in rows:
        if not isinstance(row,dict):
            continue  # A malformed row is not a detection; the rest of the reply stands.
        label,confidence,box=row.get('label'),row.get('confidence'),row.get('bbox_xyxy')
        if label not in LABELS:
            continue
        if type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<=confidence<=1:
            continue
        if (not isinstance(box,(list,tuple)) or len(box)!=4
                or any(type(value) not in (int,float) or not math.isfinite(value) for value in box)):
            continue
        pixels=[min(limit,max(0.,value*limit)) for value,limit in zip(box,(width,height,width,height))]
        if pixels[0]>=pixels[2] or pixels[1]>=pixels[3]:
            continue
        detections.append({'label':label,'confidence':confidence,'bbox_xyxy':pixels})
    return detections


def _complete(endpoint,model,png,options,timeout_s):
    image_png,image_width,image_height=_request_image(png,options['max_image_edge'])
    data_url='data:image/png;base64,'+base64.b64encode(image_png).decode('ascii')
    payload={'model':model,'max_tokens':options['max_tokens'],'stream':False,'messages':[
        {'role':'system','content':PROMPT},
        {'role':'user','content':[
            {'type':'text','text':'Detect the visible objects in this image.'},
            {'type':'image_url','image_url':{'url':data_url}}]}]}
    request=Request(endpoint,data=json.dumps(payload).encode('utf-8'),
                    headers={'Content-Type':'application/json'},method='POST')
    # Never inherit the shell's remote proxy for a local endpoint.
    with build_opener(ProxyHandler({})).open(request,timeout=timeout_s) as response:
        completion=json.load(response)
    return _content_rows(completion),image_width,image_height


def _fallback(calibration,metadata,reason):
    """Deterministic operator-declared detections, never inferred from truth."""
    options=calibration.get('perception_options') or {}
    rows=options.get('fallback_detections') or []
    if not isinstance(rows,list):
        raise TypeError('fallback_detections must be a list')
    detections=_pixels(rows,calibration['width'],calibration['height'])
    if isinstance(metadata,dict):
        metadata.update({'inference_mode':'offline_fallback','fallback_reason':reason,
                         'fallback_declared':len(rows),'fallback_detections':len(detections)})
    return detections


def detect(png,calibration):
    """Return pixel boxes for one calibrated capture (see module docstring)."""
    width,height=calibration['width'],calibration['height']
    endpoint,model,max_tokens,timeout_s,max_image_edge=_options(calibration)
    options={'max_tokens':max_tokens,'max_image_edge':max_image_edge}
    metadata=calibration.get('perception_metadata')
    if isinstance(metadata,dict):
        metadata.update({'model':model,'endpoint':endpoint,'max_tokens':max_tokens,
                         'timeout_s':timeout_s,'max_image_edge':max_image_edge})
    try:
        rows,image_width,image_height=_complete(endpoint,model,png,options,timeout_s)
    except HTTPError:
        raise  # The endpoint answered: a rejected/failed call is a real failure.
    except (URLError,ConnectionError,TimeoutError) as exc:
        return _fallback(calibration,metadata,type(exc).__name__)
    if isinstance(metadata,dict):
        metadata.update({'inference_mode':'qwen_vl','inference_image_width':image_width,
                         'inference_image_height':image_height})
    return _pixels(rows,width,height)
