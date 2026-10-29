"""Locations of external inputs and generated scene data.

The defaults retain the original local layout; environment variables can move
these roots independently when the scene code is used elsewhere.
"""
import os
from pathlib import Path
import re

_PKG = Path(__file__).resolve()
# Relocatable: installed package may sit in <dist>/src/rescue_scenes or beside tools/.
_ROOT = _PKG.parents[2] if _PKG.parents[1].name == 'src' else _PKG.parents[1]
SCENE_ROOT = Path(os.environ.get('RESCUE_SCENE_ROOT', str(_ROOT))).expanduser()
AAS_ROOT = Path(os.environ.get('RESCUE_AAS_ROOT', '/mnt/data2/weizhiwei/aeroagentsim/AeroAgentSim')).expanduser()
PRIVATE_ROOT = Path(os.environ.get('RESCUE_PRIVATE_ROOT', os.environ.get('RESCUE_ASSET_ROOT', str(SCENE_ROOT)))).expanduser()
DATA_ROOT = Path(os.environ.get('RESCUE_DATA_ROOT', str(SCENE_ROOT.parent))).expanduser()
GENERATOR_ROOT = Path(os.environ.get('RESCUE_GENERATOR_ROOT', str(SCENE_ROOT.parent / 'upstream/scene-v0.6'))).expanduser()
PYTHON = os.environ.get('RESCUE_PYTHON', '/mnt/data2/weizhiwei/rescue-generator-stage1/envs/cut4/bin/python')

ROOT = SCENE_ROOT

_ROOTS = {'RESCUE_AAS_ROOT': AAS_ROOT, 'RESCUE_PRIVATE_ROOT': PRIVATE_ROOT,
          'RESCUE_DATA_ROOT': DATA_ROOT, 'RESCUE_GENERATOR_ROOT': GENERATOR_ROOT,
          'RESCUE_PYTHON': PYTHON}
_PATTERN = re.compile(r'\$\{(RESCUE_[A-Z_]+)\}')


def expand(value):
    """Expand rescue-root placeholders in nested JSON values, preserving types."""
    if isinstance(value, str):
        return _PATTERN.sub(lambda match: str(_ROOTS[match.group(1)])
                            if match.group(1) in _ROOTS else match.group(0), value)
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value
