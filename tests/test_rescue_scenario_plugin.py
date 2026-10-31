import json
from pathlib import Path

import pytest

from aeroagentsim.authoring.templates import TemplateRegistry
from rescue_scenes.scenario_plugin import SCENES, plugin, rescue_knobs

KPIS = [
    "tasks_created",
    "tasks_completed",
    "tasks_failed",
    "completion_rate",
    "mean_task_latency_s",
    "handoffs_completed",
    "rescue_time_to_confirmation_s",
    "rescue_time_to_delivery_s",
]
TEMPLATE_IDS = ["rescue-" + sid.lower() for sid in SCENES]


def registry() -> TemplateRegistry:
    return TemplateRegistry(plugins=(plugin(),))


def source_of(sid):
    return next(source for source in plugin().templates if source.directory.name == "rescue-" + sid.lower())


def scenario_of(sid):
    return json.loads((source_of(sid).directory / "scenario.json").read_text())


def test_rescue_plugin_registers_thirteen_native_missions():
    provider = plugin()
    assert provider.name == "rescue_scenes"
    reg = TemplateRegistry(plugins=(provider,))
    for sid, source in zip(SCENES, provider.templates):
        metadata = json.loads((source.directory / "template.yaml").read_text())
        scenario = json.loads((source.directory / "scenario.json").read_text())
        assert metadata["id"] == "rescue-" + sid.lower()
        assert metadata["category"] == "emergency"
        assert scenario["engines"]["rescue"]["plugin"] == "rescue_scenes"
        assert scenario["engines"]["rescue"]["config"]["execution_mode"] == "forced_rules"
        compiled = source.directory / scenario["registry"]["snapshot"]
        assert json.loads(compiled.read_text())["registry"]["revision"] == "aerograph-compiler/1"
        reg.apply(metadata["id"])
        assert Path(scenario["engines"]["rescue"]["config"]["bundle_path"]).is_file()
    assert len(provider.templates) == 13
    assert len(reg.list()) == 13


def test_public_metadata_exposes_three_working_controls_and_selected_kpis():
    reg = registry()
    for sid in SCENES:
        tid = "rescue-" + sid.lower()
        metadata = reg.get(tid)
        scenario = scenario_of(sid)
        run = scenario["run"]
        config = scenario["engines"]["rescue"]["config"]

        assert metadata["knobs"] == reg.get(tid)["knobs"]
        knobs = {knob["id"]: knob for knob in metadata["knobs"]}
        assert list(knobs) == ["until_ns", "execution_mode", "pacing"]

        horizon = knobs["until_ns"]
        assert horizon["pointer"] == "/run/until_ns"
        assert horizon["type"] == "integer"
        assert horizon["default"] == run["until_ns"]
        assert horizon["min"] == run["advance_ns"]
        assert horizon["max"] == run["until_ns"]
        assert horizon["min"] <= horizon["default"] <= horizon["max"]

        mode = knobs["execution_mode"]
        assert mode["pointer"] == "/engines/rescue/config/execution_mode"
        assert mode["type"] == "choice"
        assert mode["default"] == config["execution_mode"] == "forced_rules"
        assert mode["options"] == ["forced_rules", "autonomous"]

        pacing = knobs["pacing"]
        assert pacing["pointer"] == "/run/pacing"
        assert pacing["type"] == "choice"
        assert pacing["default"] == run["pacing"] == "fast"
        assert pacing["options"] == ["fast", "realtime"]

        assert metadata["kpis"] == KPIS
        assert metadata["knobs"][0]["min"] == run["advance_ns"]


def test_public_metadata_ids_are_unique():
    assert len(set(TEMPLATE_IDS)) == 13 == len(TEMPLATE_IDS)


def test_apply_changes_only_the_declared_fields_and_preserves_frozen_inputs():
    reg = registry()
    for sid in SCENES:
        tid = "rescue-" + sid.lower()
        original = scenario_of(sid)
        prepared = reg.apply(tid, {
            "until_ns": 60000000000,
            "execution_mode": "autonomous",
            "pacing": "realtime",
        })
        document = prepared.document
        assert document["run"]["until_ns"] == 60000000000
        assert document["engines"]["rescue"]["config"]["execution_mode"] == "autonomous"
        assert document["run"]["pacing"] == "realtime"
        # Frozen materialized input, authored presentation and display scene stay untouched.
        assert document["engines"]["rescue"]["config"]["bundle_path"] == original["engines"]["rescue"]["config"]["bundle_path"]
        applied_scene = document["engines"]["rescue"]["config"]["console_scene"]
        original_scene = original["engines"]["rescue"]["config"]["console_scene"]
        # PyYAML legitimately normalises numeric-looking strings, so compare authored fields structurally.
        for key in ("id", "city", "camera", "groundY", "attribution"):
            assert applied_scene.get(key) == original_scene.get(key), (tid, key)
        assert [s["url"] for s in applied_scene.get("splats", [])] == [s["url"] for s in original_scene.get("splats", [])]
        assert len(applied_scene.get("splats", [])) == len(original_scene.get("splats", []))
        assert document["entities"] == original["entities"]
        assert document["presentation"] == original["presentation"]
        assert document["provenance"] == original["provenance"]
        assert document["run"]["advance_ns"] == original["run"]["advance_ns"]


def test_apply_rejects_wrong_choice_wrong_type_and_out_of_range_values():
    reg = registry()
    tid = "rescue-sc01"
    run = scenario_of("SC01")["run"]
    with pytest.raises((ValueError, TypeError)):
        reg.apply(tid, {"execution_mode": "teleport"})
    with pytest.raises((ValueError, TypeError)):
        reg.apply(tid, {"pacing": "warp"})
    with pytest.raises((ValueError, TypeError)):
        reg.apply(tid, {"until_ns": "not-a-number"})
    with pytest.raises((ValueError, TypeError)):
        reg.apply(tid, {"until_ns": run["advance_ns"] - 1})
    with pytest.raises((ValueError, TypeError)):
        reg.apply(tid, {"until_ns": run["until_ns"] + run["advance_ns"]})
    with pytest.raises((ValueError, TypeError)):
        reg.apply(tid, {"unknown_knob": 1})


def test_each_guide_is_present_nonempty_and_scene_specific():
    for sid in SCENES:
        tid = "rescue-" + sid.lower()
        guide = source_of(sid).directory / "README.md"
        assert guide.is_file(), tid
        text = guide.read_text(encoding="utf-8")
        assert text.strip()
        assert tid in text
        assert "Initialize" in text and "Start" in text
        assert "until_ns" in text and "execution_mode" in text and "pacing" in text
        assert "60000000000" in text or "nanosecond" in text.lower()
        # Scene-specific, not a generic heading.
        metadata = json.loads((source_of(sid).directory / "template.yaml").read_text())
        assert (metadata.get("title_zh") or metadata["title"]) in text


def test_guide_endpoint_returns_200_through_the_actual_app(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from aeroagentsim.services.app import create_app

    templates_root = tmp_path / "templates"
    templates_root.mkdir()
    # A self-contained copy keeps the app's registry scope inside the temporary root.
    import shutil
    shutil.copytree(source_of("SC01").directory, templates_root / "rescue-sc01")
    app = create_app(root=tmp_path / "runs", templates_root=templates_root)
    with TestClient(app) as client:
        response = client.get("/v1/templates/rescue-sc01/guide")
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["id"] == "rescue-sc01"
        assert "Initialize" in payload["markdown"]


def test_rescue_knobs_provider_reads_authoritative_run_and_config():
    document = scenario_of("SC03")
    knobs = rescue_knobs(document)
    assert knobs[0]["default"] == document["run"]["until_ns"] == 2400000000000
    assert knobs[0]["max"] == document["run"]["until_ns"]
    assert knobs[0]["min"] == document["run"]["advance_ns"] == 1000000000
    assert knobs[1]["default"] == document["engines"]["rescue"]["config"]["execution_mode"]
    assert knobs[2]["default"] == document["run"]["pacing"]
