import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location("rescue_make_run",Path(__file__).parents[1]/"tools/make_run.py")
make_run=importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_run)


def case(tmp_path,monkeypatch,satisfied=False,camera=False):
    runtime=tmp_path/"runtime";run=runtime/"out_cli"/"SC01_rescue.scenario-test"
    generated_root=tmp_path/"generated";generated=generated_root/"attempt"
    run.mkdir(parents=True);generated.mkdir(parents=True)
    (run/"frame.png").write_bytes(b"large frame placeholder")
    (generated/"bundle.json").write_text("{}")
    (run/"events.jsonl").write_text("")
    result={"run_complete":True,"objectives":[{"kind":"handover","target":"person-1","satisfied":satisfied}],
            "failures_by_cause":{},"camera_observation":{"frames_captured":0,"inference_calls_completed":0,"inference_calls_failed":0}}
    (run/"result.json").write_text(json.dumps(result))
    cfg={"seed":1,"tick_s":0.5,"until_s":3600,"display":{"camera_observation":{"enabled":camera}}}
    monkeypatch.setattr(make_run,"GENERATED",generated_root)
    return runtime,run,generated,cfg


def test_failed_attempt_drops_frames_but_retains_generated_inputs_and_compact_result(tmp_path,monkeypatch):
    runtime,run,generated,cfg=case(tmp_path,monkeypatch)
    source=tmp_path/"retained-source.glb";source.write_bytes(b"original source")
    output=make_run.conclude_if_incomplete(run,generated,runtime,cfg,"config-digest")
    assert output["raw_artifacts_deleted"] is True
    assert not run.exists() and (generated/"bundle.json").exists()
    summary=json.loads((Path(output["summary_dir"])/"summary.json").read_text())
    assert summary["objectives"][0]["target"]=="person-1"
    assert summary["status"]=="incomplete" and summary["config_digest"]=="config-digest"
    assert "observations" not in summary["camera"]
    assert source.read_bytes()==b"original source"


def test_complete_attempt_is_retained(tmp_path,monkeypatch):
    runtime,run,generated,cfg=case(tmp_path,monkeypatch,satisfied=True)
    assert make_run.conclude_if_incomplete(run,generated,runtime,cfg,"digest") is None
    assert (run/"frame.png").exists() and generated.exists()


def test_zero_frame_camera_attempt_is_not_accepted(tmp_path,monkeypatch):
    runtime,run,generated,cfg=case(tmp_path,monkeypatch,satisfied=True,camera=True)
    output=make_run.conclude_if_incomplete(run,generated,runtime,cfg,"digest")
    assert output["status"]=="incomplete" and not run.exists()


def test_cleanup_refuses_a_source_directory_outside_generated_outputs(tmp_path,monkeypatch):
    runtime,run,generated,cfg=case(tmp_path,monkeypatch)
    source=tmp_path/"retained-source";source.mkdir()
    with pytest.raises(ValueError):
        make_run.conclude_if_incomplete(run,source,runtime,cfg,"digest")
    assert source.exists() and run.exists()
