"""Incremental, out-of-process catalog coverage measurement.

Runs the DEMO_CONFIGS one by one and, after each run, atomically writes
evidence/coverage/coverage.json (trace items + per-run outcome metrics) so the explorer can
show partial, clearly labelled coverage while the sweep is still running. Execution evidence
comes only from what actually happened in the runs:
  * events (task proposals, terminal action states, field writes, families, rule refs), and
  * kernel action instances: a catalog sequence step counts as executed only when an action
    instance carrying that (seq_id, step) reached a terminal state.
Task outcome counts (generated / executed / completed / failed) are reported separately, so
"executed" never reads as "rescue achieved".

Usage: python3 -m rescue_env.coverage_job [--out evidence/coverage] [--only id,id]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from copy import deepcopy
from pathlib import Path

from . import __version__, trace
from .provenance import stamp
from .api import DEMO_CONFIGS, Environment
from .catalog import load

ROOT = Path(__file__).resolve().parents[1]
STAMP = {}  # source revision + start timestamp of this sweep, set in main()
TERMINAL_ACTION = {"succeeded", "partial", "failed", "canceled", "rejected"}


def _write(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str))
    os.replace(tmp, path)


def run_one(named: dict, index: int) -> dict:
    cat = load()
    cfg = named.get("config", named)
    if "config" not in named:
        cfg = {k: v for k, v in cfg.items() if k not in ("id", "title_zh", "name_zh", "name")}
    config_id = named.get("id", f"run-{index + 1}")
    started = time.time()
    env = Environment.from_config(cfg)
    env.sim.run(env.config["until_s"])
    wall = round(time.time() - started, 2)
    sim = env.sim
    b = {k: set() for k in ("scenarios", "overlays", "tasks_generated", "tasks_executed", "sequences",
                            "actions_terminal", "capabilities_used", "fields_written", "event_families")}
    steps, witnesses = set(), {}
    records = [row for agent in sim.agents.values() for row in agent.tasks.values()]
    types = {row["id"]: row["type"] for row in records}

    def witness(kind, ident, ev, task=None):
        witnesses.setdefault(f"{kind}|{ident}", {"config_id": config_id, "event_seq": ev.seq, "t": ev.t,
                                                 **({"task_id": task} if task else {})})

    action_terminal = {}
    for ev in sim.events:
        fam, kind, after, refs = ev.family, ev.kind, ev.after, list(ev.refs or [])
        b["event_families"].add(fam)
        witness("event_families", fam, ev)
        for r in refs:
            if r in cat.scenarios and fam == "upstream.semantics_ingested":
                b["scenarios"].add(r); witness("scenarios", r, ev)
            if r in cat.overlays and fam == "overlay.applied":
                b["overlays"].add(r); witness("overlays", r, ev)
            if r in cat.rules:
                witness("rules", r, ev)
        if ev.field == "task.status" and after == "proposed":
            tid = ev.task or ev.subject
            if tid in types:
                b["tasks_generated"].add(types[tid])
        if kind == "action.status" and after in TERMINAL_ACTION:
            for r in refs:
                if r in cat.actions:
                    b["actions_terminal"].add(r); witness("actions", r, ev)
            if ev.action_instance:
                action_terminal[ev.action_instance] = ev
        if kind == "action.status" and after == "accepted":
            for r in refs:
                if r in cat.capabilities:
                    b["capabilities_used"].add(r); witness("capabilities", r, ev)
        if ev.field in cat.fields_by_key:
            fid = cat.fields_by_key[ev.field]["id"]
            b["fields_written"].add(fid); witness("fields", fid, ev)
        if fam == "gen.rule" and isinstance(after, dict):
            for w in after.get("writes", []):
                if w.get("key") in cat.fields_by_key:
                    fid = cat.fields_by_key[w["key"]]["id"]
                    b["fields_written"].add(fid); witness("fields", fid, ev)

    # Catalog step execution from real action instances (terminal state + recorded seq/step).
    executed_task_ids = set()
    for iid, rec in sim.instances.items():
        ev = action_terminal.get(iid)
        params = rec.get("params") or {}
        seq, step = params.get("seq_id"), params.get("step")
        if ev is None or seq not in cat.sequences or step is None:
            continue
        b["sequences"].add(seq); steps.add((seq, str(step)))
        task_id = params.get("task_id") or (rec.get("task_ref") or [None])[0]
        witness("sequences", seq, ev, task_id); witness("steps", f"{seq}:{step}", ev, task_id)
        if task_id in types:
            executed_task_ids.add(task_id)
            b["tasks_executed"].add(types[task_id])
            witness("tasks", types[task_id], ev, task_id)

    # Outcomes per task type: count logical tasks by the holder that executed them.
    outcome = collections.defaultdict(collections.Counter)
    seen = set()
    for row in records:
        if row["id"] in seen:
            continue
        holder_rows = [r for r in records if r["id"] == row["id"]]
        own = next((r for r in holder_rows if r.get("holder") == r.get("accepted_by")), None) or \
            next((r for r in holder_rows if r.get("holder") == r.get("creator")), row)
        seen.add(row["id"])
        outcome[row["type"]]["generated"] += 1
        if row["id"] in executed_task_ids:
            outcome[row["type"]]["executed"] += 1
        if own.get("status") in ("completed", "failed", "canceled", "expired", "blocked", "suspended"):
            outcome[row["type"]][own["status"]] += 1
    objectives = env.objectives()
    world_changes = collections.Counter(ev.field for ev in sim.events
                                        if ev.layer == "world" and ev.field and not str(ev.field).startswith("platform."))
    return {
        "config_id": config_id, "config": deepcopy(cfg), "title_zh": named.get("title_zh", config_id), "wall_s": wall,
        "sim_s": env.config["until_s"], "event_count": len(sim.events),
        "contract_violations": sum(1 for ev in sim.events if ev.contract_violation),
        "buckets": {k: sorted(v) for k, v in b.items()}, "steps": sorted(steps),
        "witnesses": witnesses, "task_outcomes": {k: dict(v) for k, v in sorted(outcome.items())},
        "objectives": {"total": len(objectives), "satisfied": sum(1 for o in objectives if o.get("satisfied")),
                       "satisfied_kinds": dict(collections.Counter(o.get("kind") for o in objectives if o.get("satisfied")))},
        "world_fields_changed": dict(world_changes.most_common(25)),
    }


def merge(runs: list[dict]) -> dict:
    sweep = {k: set() for k in ("scenarios", "overlays", "tasks_generated", "tasks_executed", "sequences",
                                "actions_terminal", "capabilities_used", "fields_written", "event_families")}
    steps = set()
    for r in runs:
        for k, v in r["buckets"].items():
            sweep[k].update(v)
        steps.update(tuple(s) for s in r["steps"])
    out = {k: sorted(v) for k, v in sweep.items()}
    out["steps"] = sorted(steps)
    # trace.build reads rule refs from run events; give it minimal real witnesses.
    out["runs"] = [{"events": [{"refs": [w.split("|", 1)[1]]} for w in r["witnesses"] if w.startswith("rules|")]}
                   for r in runs]
    return out


def build_report(runs: list[dict], total: int, finished: bool) -> dict:
    sweep = merge(runs)
    result = trace.build(sweep)
    outcomes = collections.defaultdict(collections.Counter)
    for r in runs:
        for t, c in r["task_outcomes"].items():
            outcomes[t].update(c)
    for kind, rows in result["items"].items():
        for row in rows:
            ws = [dict(r["witnesses"][f"{kind}|{row['id']}"], title_zh=r["title_zh"]) for r in runs
                  if f"{kind}|{row['id']}" in r["witnesses"]]
            links = row.get("links", [])
            row["links"] = {"catalog": links if isinstance(links, list) else links, "executed_in": ws[:5]}
            if kind == "tasks":
                c = outcomes.get(row["id"], {})
                row["outcomes"] = dict(c)
                if c:
                    row["detail_zh"] += (f" 实测：生成 {c.get('generated', 0)}，执行 {c.get('executed', 0)}，"
                                         f"完成 {c.get('completed', 0)}，失败 {c.get('failed', 0)}，"
                                         f"取消 {c.get('canceled', 0)}，受阻/挂起 {c.get('blocked', 0) + c.get('suspended', 0)}。")
    tasks_completed = sorted(t for t, c in outcomes.items() if c.get("completed"))
    result["measured"] = {
        "version": __version__, **STAMP, "occurrence_vs_completion_zh": "生成/执行为出现次数；完成为任务状态 completed；均不等于世界目标已满足。",
        "partial": not finished, "runs_done": len(runs), "runs_total": total,
        "tasks_generated": len(sweep["tasks_generated"]), "tasks_executed": len(set(sweep["tasks_generated"]) & set(sweep["tasks_executed"])),
        "tasks_completed": len(tasks_completed), "tasks_completed_ids": tasks_completed,
        "configs": [r.get("config") for r in runs],
        "runs": [{k: r[k] for k in ("config_id", "title_zh", "wall_s", "sim_s", "event_count", "contract_violations",
                                    "objectives", "world_fields_changed")} | {"task_outcomes": r["task_outcomes"]}
                 for r in runs],
    }
    result["pending"] = False
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "evidence" / "coverage"))
    ap.add_argument("--only", default="")
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    STAMP.update(stamp())
    configs = DEMO_CONFIGS
    if args.only:
        wanted = set(args.only.split(","))
        configs = [c for c in configs if c.get("id") in wanted]
    runs = []
    _write(out / "coverage.json", {**build_report([], len(configs), False), "running": configs[0].get("id") if configs else None})
    for i, named in enumerate(configs):
        try:
            runs.append(run_one(named, i))
        except Exception as exc:  # a failing demo is reported, never hidden
            runs.append({"config_id": named.get("id"), "config": deepcopy(named.get("config")), "title_zh": named.get("title_zh"), "wall_s": 0, "sim_s": 0,
                         "event_count": 0, "contract_violations": 0, "buckets": {}, "steps": [], "witnesses": {},
                         "task_outcomes": {}, "objectives": {"error": repr(exc)}, "world_fields_changed": {}})
        nxt = configs[i + 1].get("id") if i + 1 < len(configs) else None
        _write(out / "coverage.json", {**build_report(runs, len(configs), nxt is None), "running": nxt})
        _write(out / f"run-{i + 1:02d}-{named.get('id')}.json", deepcopy(runs[-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
