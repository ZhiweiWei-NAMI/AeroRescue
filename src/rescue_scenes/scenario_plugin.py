"""The thirteen materialized rescue missions in the standard AAS template catalog.

Templates retain their authored forced-rule execution mode and local scene/asset
provenance. They use the separately registered rescue_scenes engine plugin.
"""
from pathlib import Path

from aeroagentsim.plugin_api import ScenarioPlugin, TemplateSource

SCENES=('SC01','SC03','SC04','SC05','SC07','SC08','SC09','SC10','SC11','SC12','SC13','SC17','GT04')


def rescue_knobs(document):
    """Pointer-backed controls that only change fields the frozen bundle does not own."""
    run=document['run']
    config=document['engines']['rescue']['config']
    return [
        {
            'id': 'until_ns',
            'label': 'Run horizon (ns)',
            'label_zh': '运行时长（纳秒）',
            'description': '1 second = 1,000,000,000 ns. A shorter run is a partial mission; the saved bundle and its full-mission target remain fixed.',
            'description_zh': '1 秒 = 1,000,000,000 纳秒。缩短运行得到部分任务记录；冻结输入及完整任务目标保持固定。',
            'type': 'integer',
            'default': run['until_ns'],
            'min': run['advance_ns'],
            'max': run['until_ns'],
            'pointer': '/run/until_ns',
        },
        {
            'id': 'execution_mode',
            'label': 'Execution mode',
            'label_zh': '执行模式',
            'description': 'forced_rules runs the authored reference driver; autonomous uses the existing local agent runtime with scripted sensing.',
            'description_zh': 'forced_rules 使用已编写的参考规则驱动；autonomous 使用现有本地智能体运行时与脚本感知。',
            'type': 'choice',
            'default': config['execution_mode'],
            'options': ['forced_rules', 'autonomous'],
            'options_zh': ['参考规则驱动', '本地智能体自主运行'],
            'pointer': '/engines/rescue/config/execution_mode',
        },
        {
            'id': 'pacing',
            'label': 'Run pacing',
            'label_zh': '运行节奏',
            'description': 'fast runs as quickly as possible; realtime paces simulated time against wall time.',
            'description_zh': 'fast 尽快执行；realtime 按墙钟时间推进仿真。',
            'type': 'choice',
            'default': run['pacing'],
            'options': ['fast', 'realtime'],
            'options_zh': ['尽快执行', '实时推进'],
            'pointer': '/run/pacing',
        },
    ]


def plugin() -> ScenarioPlugin:
    root=Path(__file__).with_name('templates')
    return ScenarioPlugin(name='rescue_scenes',templates=tuple(
        TemplateSource(root/('rescue-'+sid.lower()),extra_knobs=rescue_knobs) for sid in SCENES))
