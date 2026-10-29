#!/usr/bin/env python3
"""Explicit rule-driven verification; not an autonomous-policy acceptance run.

Uses original scenario geometry, objectives and native action/evaluator pipeline.
Capability and navigation assumptions are overridden and recorded. No truth flags,
poses, supplies, receiving capacity or action outcomes are patched.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rescue_scenes.paths import expand
from rescue_scenes.backgrounds import materialize
from rescue_scenes.core.api import Environment
from rescue_scenes.core.sim.actions import MODELS, RECEIVING_STATION_KINDS


def verify(config, output):
    raw=config.read_bytes()
    cfg=expand(json.loads(raw))
    with tempfile.TemporaryDirectory(prefix="rescue-forced-") as tmp:
        directory=Path(tmp)
        _,bundle=materialize(cfg,directory)
        core=deepcopy(bundle["core"])
        core["semantics"]["path"]=str(directory/core["semantics"]["path"])
        env=Environment.from_config(core)
        sim=env.sim
        from rescue_scenes.forced_rules import ForcedRuleDriver
        driver=ForcedRuleDriver(env,cfg)
        while not driver.done:
            driver.advance(sim.t+cfg['tick_s'])
        report=driver.snapshot()
        report.update(config=str(config.resolve()), config_sha256=hashlib.sha256(raw).hexdigest())
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps({"mode":report["mode"],"rules":len(report["rules"]),"passed":report["rule_targets_satisfied"],"actions":[[a["action"],a["status"],a["feedback"]] for a in report["actions"]],"output":str(output)},ensure_ascii=False))
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    verify(args.config,args.output)


if __name__=="__main__":main()
