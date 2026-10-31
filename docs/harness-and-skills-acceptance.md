# Harness 与 Skills 验收矩阵（T0 ~ T7）

本文件由 `tools/verify_harness_and_skills.py` 实测生成，禁止手工编辑。
所有断言均运行真实 `Simulation` 内核与真实 `AgentRuntime`，无桩、无硬编码分数。

- 生成时间：`2026-10-30T11:40:55-0700`
- 契约：`astra Round 3 frozen kernel/skills/harness contract (R2.3/R2.4/R2.6/R3.1/R3.6)`
- 结果：**8/8 轨迹通过**，84/84 断言通过
- 耗时：0.43 s

## 1. 轨迹总览

| 轨迹 | 覆盖契约 | 断言数 | 结果 |
| :--- | :--- | ---: | :--- |
| `T0` 统一原语：66 动作集、11 个 Skill 标签、越界拒绝 | R3.1 action space and skill mapping | 10 | ✅ PASS |
| `T1` T011/Q011 侦察与建筑评估闭环（approach 不推进游标，method 逐步推进） | R3.1 candidate binding; R3.2 execute_candidate -> pending_actions/on_action_result | 19 | ✅ PASS |
| `T2` 伤员装载/后送移交与物资获取（A020/A021/A022 + A028/A024） | R3.1 payload accounting; A022 acknowledged handover | 6 | ✅ PASS |
| `T3` 道路受阻 failure_code=blocked 触发 Q098 恢复闭环 | R2.3 recovery cursor; runtime._recover skips the repeated trigger step | 6 | ✅ PASS |
| `T4` 高优任务抢占与 _resume_stack 挂起恢复闭环 | A053 preemption -> A056 checkpoint -> may_resume_preempted -> A057 | 9 | ✅ PASS |
| `T5` PX4 external_motion 具身飞控闭环 | R2.3 external motion; complete_external_action; 2 m arrival tolerance | 12 | ✅ PASS |
| `T6` t=10 定时世界扰动触发与 deepcopy(sim) 隔离 | R2.3 _enqueue_world binds to its own holder | 6 | ✅ PASS |
| `T7` B00/B10/B01/B11 字节预算门控与 MemoryEnvelope 送达校验 | R2.6 comm budget gate; delivered-only edge memory | 16 | ✅ PASS |

## 2. 逐条断言

### T0 · 统一原语：66 动作集、11 个 Skill 标签、越界拒绝

契约：`R3.1 action space and skill mapping`

| # | 断言 | 结果 | 实测细节 |
| ---: | :--- | :--- | :--- |
| 1 | the action space is exactly A001-A066 | ✅ | `{"size": 66}` |
| 2 | ten rescue skills plus the recovery lane are declared | ✅ | `{}` |
| 3 | every catalog sequence is classified exactly once | ✅ | `{"sequences": 112}` |
| 4 | all eleven labels own at least one sequence | ✅ | `{"sizes": {"aid_person": 3, "clear_obstacle": 9, "contact_person": 3, "extricate_person": 2, "isolate_hazard": 1, "load_and_evacuate": 4, "pickup_supply": 6, "recover_replan": 16, "search_and_assess": 66, "suppress_fi...` |
| 5 | recovery branches are the recovery skill | ✅ | `{}` |
| 6 | candidate construction refuses A000 | ✅ | `{}` |
| 7 | candidate construction refuses A067 | ✅ | `{}` |
| 8 | candidate construction refuses a001 | ✅ | `{}` |
| 9 | candidate construction refuses B001 | ✅ | `{}` |
| 10 | an unbound candidate cannot carry a task revision | ✅ | `{}` |

### T1 · T011/Q011 侦察与建筑评估闭环（approach 不推进游标，method 逐步推进）

契约：`R3.1 candidate binding; R3.2 execute_candidate -> pending_actions/on_action_result`

| # | 断言 | 结果 | 实测细节 |
| ---: | :--- | :--- | :--- |
| 1 | catalog resolves T011 to the real Q011 method | ✅ | `{"seq": "Q011", "steps": ["A001", "A005", "A006", "A040", "A041", "A036"]}` |
| 2 | candidate set carries a DecisionContext ticket | ✅ | `{"decision": "dec-7cba1b846c25"}` |
| 3 | out-of-range work site yields a real approach candidate | ✅ | `{"action": "A001"}` |
| 4 | method candidate binds the same revision and skill | ✅ | `{"skill": "search_and_assess"}` |
| 5 | approach enters AgentRuntime.pending_actions then settles | ✅ | `{"failure": "none", "status": "succeeded"}` |
| 6 | approach never advances the method cursor | ✅ | `{"cursor": 0}` |
| 7 | approach really moved the holder to the work site | ✅ | `{}` |
| 8 | a ticket from the pre-approach view is refused, not executed | ✅ | `{}` |
| 9 | method step 0 (A001) advanced the cursor once | ✅ | `{"cursor": 1, "failure": "none", "status": "succeeded"}` |
| 10 | method step 1 (A005) advanced the cursor once | ✅ | `{"cursor": 2, "failure": "none", "status": "succeeded"}` |
| 11 | method step 2 (A006) advanced the cursor once | ✅ | `{"cursor": 3, "failure": "none", "status": "succeeded"}` |
| 12 | method step 3 (A040) advanced the cursor once | ✅ | `{"cursor": 4, "failure": "none", "status": "succeeded"}` |
| 13 | method step 4 (A041) advanced the cursor once | ✅ | `{"cursor": 5, "failure": "none", "status": "succeeded"}` |
| 14 | method step 5 (A036) advanced the cursor once | ✅ | `{"cursor": 6, "failure": "none", "status": "succeeded"}` |
| 15 | the whole Q011 main path was executed | ✅ | `{"submitted": [[0, "A001"], [1, "A005"], [2, "A006"], [3, "A040"], [4, "A041"], [5, "A036"]]}` |
| 16 | a control-lane A041 fusion ran while the method advanced | ✅ | `{"instances": ["act-0000004", "act-0000006"]}` |
| 17 | A041 control fusion never adds a main method step | ✅ | `{"steps": ["Q011:0", "Q011:1", "Q011:2", "Q011:3", "Q011:4", "Q011:5"]}` |
| 18 | A041 wrote the holder's belief (assessment evidence) | ✅ | `{"belief_events": 37}` |
| 19 | the assessed task closes as completed under external_harness | ✅ | `{"reason": "none", "status": "completed"}` |

### T2 · 伤员装载/后送移交与物资获取（A020/A021/A022 + A028/A024）

契约：`R3.1 payload accounting; A022 acknowledged handover`

| # | 断言 | 结果 | 实测细节 |
| ---: | :--- | :--- | :--- |
| 1 | delivered report reached the holder belief through A041 | ✅ | `{}` |
| 2 | A020 loads the casualty and payload equals its real attr.mass_kg | ✅ | `{"expected": 82.5, "payload": 82.5, "status": "completed"}` |
| 3 | A022 acknowledges the handover and payload returns to zero | ✅ | `{"handover": "acknowledged", "payload": 0.0, "status": "completed"}` |
| 4 | the receiver's occupancy really changed | ✅ | `{"occupied": 1}` |
| 5 | A028 transfers one real stock instance and charges its mass | ✅ | `{"items": ["item-station-1-water-2"], "payload": 1.0}` |
| 6 | A024 mounts the existing item and adds exactly its mass | ✅ | `{"expected": 1.5, "payload": 1.5}` |

### T3 · 道路受阻 failure_code=blocked 触发 Q098 恢复闭环

契约：`R2.3 recovery cursor; runtime._recover skips the repeated trigger step`

| # | 断言 | 结果 | 实测细节 |
| ---: | :--- | :--- | :--- |
| 1 | catalog resolves T021 to Q021 starting with the movement step | ✅ | `{"steps": ["A002", "A011", "A041", "A036"]}` |
| 2 | the real movement model reports failure_code=blocked | ✅ | `{"failure": "blocked", "status": "failed"}` |
| 3 | recovery is Q098 and starts after the repeated trigger step | ✅ | `{"cursor": 1, "seq": "Q098"}` |
| 4 | the blocked task is honestly blocked, not failed | ✅ | `{"status": "blocked"}` |
| 5 | the recovery lane starts by integrating the discovered blockage | ✅ | `{"executed": [[1, "A041", "succeeded"], [2, "A066", "succeeded"], [3, "A002", "succeeded"], [4, "A054", "succeeded"]]}` |
| 6 | the cleared recovery lane runs A041 -> A066 -> A002 -> A054 | ✅ | `{"executed": [[1, "A041", "succeeded"], [2, "A066", "succeeded"], [3, "A002", "succeeded"], [4, "A054", "succeeded"]]}` |

### T4 · 高优任务抢占与 _resume_stack 挂起恢复闭环

契约：`A053 preemption -> A056 checkpoint -> may_resume_preempted -> A057`

| # | 断言 | 结果 | 实测细节 |
| ---: | :--- | :--- | :--- |
| 1 | the original commitment is executing with a real action | ✅ | `{"status": "executing"}` |
| 2 | A053 preempts at the safe boundary and A056 suspends the original | ✅ | `{"stack": ["T021@ground_vehicle-1#1"], "status": "suspended"}` |
| 3 | A056 wrote the suspended checkpoint | ✅ | `{"checkpoint": true}` |
| 4 | the replacement was really accepted after the suspension | ✅ | `{"status": "accepted"}` |
| 5 | the preempting rescue completes | ✅ | `{"status": "completed"}` |
| 6 | a carried casualty keeps the original suspended until handover | ✅ | `{"stack": ["T021@ground_vehicle-1#1"], "status": "suspended"}` |
| 7 | the handover completes and the receiver acknowledges | ✅ | `{"status": "completed"}` |
| 8 | may_resume_preempted releases the original through A057 | ✅ | `{"current": "T021@ground_vehicle-1#1", "stack": [], "status": "executing"}` |
| 9 | the resume really used the A057 commitment path | ✅ | `{}` |

### T5 · PX4 external_motion 具身飞控闭环

契约：`R2.3 external motion; complete_external_action; 2 m arrival tolerance`

| # | 断言 | 结果 | 实测细节 |
| ---: | :--- | :--- | :--- |
| 1 | A001 under motion_backend=px4 is an external motion | ✅ | `{"action": "A001"}` |
| 2 | the external leg keeps the real goal for the flight controller | ✅ | `{"goal": [34.8, 51.0]}` |
| 3 | no fixed completion timer: only a bounded watchdog is armed | ✅ | `{"duration": 2.3333928333333334, "watchdog": [180.0]}` |
| 4 | the scripted interpolation never moved the PX4-managed UAV | ✅ | `{"pose": [10.0, 38.0]}` |
| 5 | the planned duration alone does not complete the action | ✅ | `{"status": "running"}` |
| 6 | on_tick still drains the control queue during external motion | ✅ | `{}` |
| 7 | arrival outside the 2 m tolerance is not a completion | ✅ | `{"tolerance": 2.0}` |
| 8 | telemetry within 2 m settles the action through complete_external_action | ✅ | `{"completed": ["act-0000001"], "status": "succeeded"}` |
| 9 | the native pose was not overwritten by a scripted completion | ✅ | `{"pose": [33.8, 51.0]}` |
| 10 | the task cursor advanced to the next catalog step | ✅ | `{"cursor": 1}` |
| 11 | the next decision offers A005, not a repeated A001 | ✅ | `{"candidates": ["A005"]}` |
| 12 | the platform reports idle after the settled external arrival | ✅ | `{"motion_state": "idle"}` |

### T6 · t=10 定时世界扰动触发与 deepcopy(sim) 隔离

契约：`R2.3 _enqueue_world binds to its own holder`

| # | 断言 | 结果 | 实测细节 |
| ---: | :--- | :--- | :--- |
| 1 | every scheduled callback rebinds to the copied simulation | ✅ | `{"heap": 2}` |
| 2 | the t=10 disruption really fires on the original simulation | ✅ | `{"events": [["link_outage", 10.0]]}` |
| 3 | the due list is consumed, not orphaned | ✅ | `{"due": []}` |
| 4 | the copy fires the same disruption on its own due list | ✅ | `{"events": [["link_outage", 10.0]]}` |
| 5 | the original keeps exactly one injection: the copy never wrote into it | ✅ | `{}` |
| 6 | the copy really ran its own clock | ✅ | `{"copy": 10.0, "original": 10.0}` |

### T7 · B00/B10/B01/B11 字节预算门控与 MemoryEnvelope 送达校验

契约：`R2.6 comm budget gate; delivered-only edge memory`

| # | 断言 | 结果 | 实测细节 |
| ---: | :--- | :--- | :--- |
| 1 | B00 opens exactly its declared directions | ✅ | `{"down": false, "mode": "B00", "up": false}` |
| 2 | B10 opens exactly its declared directions | ✅ | `{"down": false, "mode": "B10", "up": true}` |
| 3 | B01 opens exactly its declared directions | ✅ | `{"down": true, "mode": "B01", "up": false}` |
| 4 | B11 opens exactly its declared directions | ✅ | `{"down": true, "mode": "B11", "up": true}` |
| 5 | B00 admits nothing in either direction | ✅ | `{"state": {"budgets": {"down": 64, "up": 64}, "denied": {"down": 1, "up": 1}, "mode": "B00", "remaining": {"down": 64, "up": 64}, "spent": {"down": 0, "up": 0}}}` |
| 6 | B01 spends downlink bytes and refuses uplink | ✅ | `{"state": {"budgets": {"down": 16, "up": 16}, "denied": {"down": 1, "up": 1}, "mode": "B01", "remaining": {"down": 6, "up": 16}, "spent": {"down": 10, "up": 0}}}` |
| 7 | B10 honours the uplink-only budget | ✅ | `{"state": {"budgets": {"down": 16, "up": 8}, "denied": {"down": 1, "up": 1}, "mode": "B10", "remaining": {"down": 16, "up": 0}, "spent": {"down": 0, "up": 8}}}` |
| 8 | B11 keeps independent direction budgets | ✅ | `{"state": {"budgets": {"down": 8, "up": 24}, "denied": {"down": 1, "up": 1}, "mode": "B11", "remaining": {"down": 0, "up": 0}, "spent": {"down": 8, "up": 24}}}` |
| 9 | an undelivered message never becomes a readable envelope | ✅ | `{"delivered_at": null}` |
| 10 | the delivered envelope is readable for its receiver only | ✅ | `{"delivered_at": 12.0}` |
| 11 | an expired envelope is not readable | ✅ | `{}` |
| 12 | a tampered envelope fails its digest check | ✅ | `{}` |
| 13 | readable_envelopes filters the unreadable rows | ✅ | `{}` |
| 14 | the edge harness exposes only the delivered envelope | ✅ | `{"memory": ["msg-000006"], "unreadable": 2}` |
| 15 | a holder whose downlink budget is exhausted never reads the message | ✅ | `{"denied": [{"bytes": 32, "message": "msg-000006", "t": 12.0}, {"bytes": 46, "op": "report", "receivers": ["coordinator-1"], "t": 12.0}]}` |
| 16 | B01 refuses uplink sends entirely | ✅ | `{}` |

## 3. 冻结契约与实现位置

| 契约 | 实现 | 验收 |
| :--- | :--- | :--- |
| R2.3 定时世界扰动 | `Simulation._enqueue_world`（`kernel.py`）在调用时读取自身 `_due_world`，`deepcopy(sim)` 后回调重绑到副本 | `T6` |
| R2.3 外部运动 | `submit_action` 标记 `external_motion`，只挂有界 watchdog；`ActionModel.start/_move_progress/_effect_A001` 跳过脚本插值；`Simulation.complete_external_action` 走正常结算 | `T5` |
| R2.4 决策边界 | `AgentRuntime.policy_mode`/`motion_backend`/`comm_budget_gate`；`external_harness` 仅停内部选步，消息/控制队列/A041 融合仍活 | `T1`,`T4`,`T5` |
| R2.6 通道预算 | `CommChannelGate`（B00/B10/B01/B11）门控 `_send`/`on_message` | `T7` |
| R3.1 统一候选 | `CandidateAction`/`DecisionContext`/`MemoryEnvelope`/`build_decision_candidates`（`skills.py`） | `T0`,`T1`,`T7` |
| R3.6 统一执行 | `execute_candidate` → `AgentRuntime._submit` → `pending_actions` → `on_action_result` | `T1`,`T3`,`T4`,`T5` |
| PX4 具身 | `engine._sync_px4` → `px4_bridge.complete_reached_movements`（2 m 2D/显式高度 3D 容差） | `T5` |

## 4. 复现

```bash
.venv/bin/python tools/verify_harness_and_skills.py
.venv/bin/python -m pytest tests/test_sandbox_and_harness.py -q
```

> 说明：验收脚本对场景真值的写入（伤员 `attr.mass_kg`、接收站 `attr.receiver_confirmation`、道路封闭/清除）属于“作者场景设定”，与 `configs/` 中的场景数据同性质；所有结论仍由内核真实执行产生。
