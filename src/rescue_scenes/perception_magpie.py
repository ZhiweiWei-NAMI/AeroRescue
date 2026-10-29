"""Image-only DeepSeek detections through the local Magpie gateway."""
import base64
from io import BytesIO
import json
import math
import os
import re
from urllib.request import ProxyHandler, Request, build_opener


ENDPOINT = os.environ.get('MAGPIE_ENDPOINT', 'http://127.0.0.1:3425/v1/chat/completions')
# Magpie route verified (2026-10-27, unique-text probe) to pass the image to
# DeepSeek V4.1 Flash. opencode-go/deepseek-v4.1-flash also saw images but its
# upstream credential became invalid (HTTP 401) on 2026-10-27; it remains
# selectable through perception_options.model or MAGPIE_MODEL.
MODEL = os.environ.get('MAGPIE_MODEL', 'workbuddy/deepseek-v4.1-flash')
MAX_TOKENS = 4096
TIMEOUT_S = 300
MAX_IMAGE_EDGE = 768
REASONING_EFFORT = 'low'
LABELS = frozenset({
    'person', 'building_damage_none', 'building_damage_minor',
    'building_damage_major', 'building_damage_destroyed', 'road_debris',
})
PROMPT = """Detect only objects visibly supported by the supplied rendered image.
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


def _request_image(png, max_image_edge):
    # Pillow is optional in the selected AAS interpreter. PNG IHDR still gives
    # actual transmitted dimensions when resizing is unavailable.
    if png[:8] != b'\x89PNG\r\n\x1a\n' or png[12:16] != b'IHDR' or len(png) < 24:
        raise ValueError('Magpie image detector requires PNG image bytes')
    width = int.from_bytes(png[16:20], 'big')
    height = int.from_bytes(png[20:24], 'big')
    try:
        from PIL import Image
    except ImportError:
        return png, width, height
    with Image.open(BytesIO(png)) as image:
        if max(image.size) <= max_image_edge:
            return png, image.width, image.height
        image.thumbnail((max_image_edge, max_image_edge), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format='PNG')
        return output.getvalue(), image.width, image.height


def _options(calibration):
    options = calibration.get('perception_options', {})
    if not isinstance(options, dict):
        raise TypeError('perception_options must be a dict')
    max_image_edge = options.get('max_image_edge', MAX_IMAGE_EDGE)
    max_tokens = options.get('max_tokens', MAX_TOKENS)
    timeout_s = options.get('timeout_s', TIMEOUT_S)
    reasoning_effort = options.get('reasoning_effort', REASONING_EFFORT)
    model = options.get('model', MODEL)
    for name, value in (('max_image_edge', max_image_edge), ('max_tokens', max_tokens)):
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if (type(timeout_s) not in (int, float) or not math.isfinite(timeout_s)
            or timeout_s <= 0):
        raise ValueError('timeout_s must be a finite positive number')
    for name, value in (('reasoning_effort', reasoning_effort), ('model', model)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} must be a nonempty string')
    return max_image_edge, max_tokens, timeout_s, reasoning_effort, model


def detect(png, calibration):
    """Return pixel boxes in the original calibrated capture's dimensions.

    Transport, model and malformed-response errors propagate to the S17 caller.
    Neither calibration poses nor any simulation metadata enter the prompt.
    """
    width, height = calibration['width'], calibration['height']
    max_image_edge, max_tokens, timeout_s, reasoning_effort, model = _options(calibration)
    image_png, image_width, image_height = _request_image(png, max_image_edge)
    metadata = calibration.get('perception_metadata')
    if isinstance(metadata, dict):
        metadata.update({
            'model': model,
            'inference_image_width': image_width,
            'inference_image_height': image_height,
            'inference_image_bytes': len(image_png),
            'max_image_edge': max_image_edge,
            'max_tokens': max_tokens,
            'timeout_s': timeout_s,
            'reasoning_effort': reasoning_effort,
        })
    data_url = 'data:image/png;base64,' + base64.b64encode(image_png).decode('ascii')
    payload = {'model': model, 'max_tokens': max_tokens, 'reasoning_effort': reasoning_effort, 'messages': [
        {'role': 'system', 'content': PROMPT},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': 'Detect the visible objects in this image.'},
            {'type': 'image_url', 'image_url': {'url': data_url}},
        ]},
    ]}
    request = Request(ENDPOINT, data=json.dumps(payload).encode('utf-8'),
                      headers={'Content-Type': 'application/json'}, method='POST')
    # Do not inherit the shell's remote HTTP(S)_PROXY for the local gateway.
    with build_opener(ProxyHandler({})).open(request, timeout=timeout_s) as response:
        completion = json.load(response)
    choice = completion['choices'][0]
    if choice.get('finish_reason') == 'length':
        raise ValueError('Magpie image detection response exhausted max_tokens')
    content = choice['message']['content'].strip()
    fenced = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', content, flags=re.DOTALL | re.IGNORECASE)
    rows = json.loads(fenced.group(1) if fenced else content)
    if not isinstance(rows, list):
        raise TypeError('Magpie image detector must return a JSON array')
    detections = []
    for row in rows:
        if isinstance(row, dict):
            # Retain the existing object response schema as well as compact rows.
            label = row.get('label')
            if label not in LABELS:
                continue
            confidence, box = row.get('confidence'), row.get('bbox_xyxy')
        elif isinstance(row, list) and len(row) == 6:
            label, confidence, *box = row
            if label not in LABELS:
                continue
        else:
            continue  # A malformed row is not a detection; the rest of the reply stands.
        if (type(confidence) not in (int, float) or not math.isfinite(confidence)
                or not 0 <= confidence <= 1):
            continue
        if (not isinstance(box, list) or len(box) != 4
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in box)):
            continue
        pixels = [min(limit, max(0., value * limit))
                  for value, limit in zip(box, (width, height, width, height))]
        if pixels[0] >= pixels[2] or pixels[1] >= pixels[3]:
            continue
        detections.append({'label': label, 'confidence': confidence, 'bbox_xyxy': pixels})
    return detections
