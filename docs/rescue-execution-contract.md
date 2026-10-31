# 救援交付执行契约

主控：本会话助手；执行：远程 Codex。用户已授权远程 Codex，替代此前直接操作限定。继续现有交付计划，不扩大范围。

## 强制边界
1. 检验成功采用显式规则指定区域、执行者、动作顺序。自主调度、任务竞争、能力分派与普通地面寻路不得阻断规则模式。已有原生 air-motion 验证覆盖应记录清楚，不称为原始地面可达性或自主能力通过。
2. 继续现有 tools/run_forced_rules.py 和 tools/make_run.py，将规则接入现有 AAS 运行入口及浏览器回放。复用服务，不新建替代展示工程，不将 JSON 结果冒充浏览器交付。
3. 保留真实动作效果、耗材消耗、装载、运输、接收确认。禁止直接改成功标志、伪造观察与交接。无接收站、物资或容量属于真实资源缺口，记录后推进其他目标；不得暗中补造。
4. 已有完整场景只核对 scene id、seed、manifest/hash 一次后复用 scene_json。只有输入改变、缺失或损坏才重建。不删除有效生成输入。生成子进程设合理超时，异常清理本任务子进程；低 CPU 不能直接判定死锁。
5. 仅执行改动相关测试和一次必要回归。通过后无新修改或具体疑点不重复。禁止新增防御性框架、冗余校验层、多代理评审链及无关重构。
6. 同一失败做一次有证据的修复后重试；仍失败则记录精确阻点，继续独立事项。每个实质里程碑更新 execution-status.md：已完成、证据、下一步、阻点。主控读取并纠偏。
7. 失败仅留紧凑诊断，清理本次大帧、截图、大日志，异常与超时同样处理。删除前核对绝对路径和归属，不动他人任务、原始资源、有效生成输入及成功回放。
8. 分别报告全部原目标、规则覆盖目标、原时间窗口、条件覆盖、相机感知及浏览器实际验证。空目标不能判成功，脚本感知不能称视觉验收。
9. 对已经授权的可逆实现不反复询问。不扩展资源授权、不发布、不干扰其他项目。主控最终验收，执行者持续推进至可交付或具体可复现阻点。

## 现场与纠偏
工作树 /mnt/data2/weizhiwei/aeroagentsim/AeroRescue/scene_data/scene_sim-delivery；提交基线 d162777。保留并继续未提交规则工具和测试。AAS 工作树 /mnt/data2/weizhiwei/aeroagentsim/wt/delivery-20261028。Python /mnt/data2/weizhiwei/aeroagentsim/AeroAgentSim/.venv/bin/python。证据 ../delivery_20261028/continuation。
SC04、SC11、SC12、SC13 现有报告全部目标通过；SC04/SC13 使用规则移动覆盖。SC09 记录目标通过，旧字段需补齐。SC03 人员链通过但建筑未全通过。SC07 旧报告仍是旧移动失败，应使用当前规则重验。SC05 水上回收通过但无 station。SC17 生成目录为空，先前停在上游生成，不能宣称已证明死锁。
SC01 用 configs/SC01_upstream_expanded.json，不能用只有 UAV 的 allocated。GT04 用 ../delivery_20261028/wp1/GT04_archived_condition.json。
旧 forced_rules_summary.json 的 SC13 off_known_roads 归因未经充分证明，且 SC04/SC13 已通过规则定位，必须修正过时归因。此前完整测试 45 passed 后又有规则修改，需要最终版本一次回归。
已有成功回放 SC07_rescue.scenario-4a9520bc202e，曾公布 http://127.0.0.1:18931/?run=SC07_rescue.scenario-4a9520bc202e&sel=ground_vehicle-1 ，须实际检查，不能沿用未核实说法。

## 立即执行
先读当前规则、运行入口和相关测试，在 execution-status.md 写明接收契约，随后直接完成规则入口接入与最小可查看回放、补齐证据并做必要测试。禁止重新全项目通读或另起庞大规划。实际模型路由如实报告。结束写 execution-result.md，说明改动、测试、浏览器证据、真实资源缺口与未完成项。
