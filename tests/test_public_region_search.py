from types import SimpleNamespace

from rescue_scenes.core.agents.taskgen import _Context, _search

BRIEF = "对乡村受灾区域开展空中和近地搜索，依据实际观测确认人员需求、通路与接收条件后组织救援。"


def context(brief=None, message=False, observed=False):
    entries = {("plot-1", "entity.kind"): {"value": "region", "source": "prior", "t": 0}}
    if observed:
        entries[("plot-1", "observation.detections")] = {"value": [], "source": "observation", "t": 0}
    received = []
    if brief and message:
        received.append({"id": "mission-1", "kind": "brief", "payload": {"brief_zh": brief}})
    elif brief:
        entries[("mission-1", "public_brief")] = {"value": brief, "source": "prior", "t": 0}
    agent = SimpleNamespace(id="coordinator-1", kind="coordinator", spec={}, platform={},
                            records={}, tasks={}, received_messages=received,
                            belief=SimpleNamespace(entries=entries))
    return _Context(agent, SimpleNamespace(t=0))


def test_public_regional_search_does_not_require_a_report_first():
    rows = _search(context(BRIEF), 9)
    assert [row[0] for row in rows] == ["plot-1"]
    assert rows[0][2]


def test_delivered_regional_search_brief_is_sufficient():
    assert [row[0] for row in _search(context(BRIEF, message=True), 9)] == ["plot-1"]


def test_assessment_brief_does_not_authorize_personnel_search():
    assert _search(context("对受灾区域开展建筑评估。"), 9) == []


def test_static_map_alone_does_not_authorize_search():
    assert _search(context(), 9) == []


def test_actual_report_still_triggers_search_without_a_public_brief():
    assert [row[0] for row in _search(context(observed=True), 9)] == ["plot-1"]
