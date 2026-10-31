# AeroRescue × AeroAgentSim 工程化总体推进计划与交付契约

更新时间：2026-10-30
状态：基于 `sol`（架构/缺口审计）与 `ds`（AAS 插件与代码摸底）双路核验定稿

---

## 一、 核心架构诊断：现状与必须打通的 4 个断层

通过对 `AeroAgentSim`（HEAD `ea012e67`）与 `AeroRescue`（HEAD `76782f0`）的完整代码审计，当前工程存在以下 4 个核心断层，必须在**不复制 AAS 内核、严格遵循 AAS Plugin 契约**的前提下打通：

| 断层维度 | 当前代码现状 | 目标工程形态（AAS Plugin 兼容） |
|---|---|---|
| **1. `RescueWorld` 外部命令与回执边界** | `src/rescue_scenes/engine.py`（`RescueWorld`）目前在 `Partition` 中仅发布渲染字段与事件，`bindings.commands=[]`，外部引擎无法通过 AAS Kernel 合法下发 `A001~A063` 动作或接收完成回执。 | 在 `RescueWorld` 的 `Partition` 中声明标准命令 Schema（`rescue.command.submit_action` / `rescue.command.propose_task` / `rescue.command.assign_task`），将外部决策转换为原生 `submit_action` 并按原生 `action.result` 发射回执。 |
| **2. 中心式沙箱（任务生成 `taskgen` + 规划 + 调度）** | `core/agents/taskgen.py`（96 条规则）与 `allocation.py`（Greedy/Bidding）目前封死在 `AgentRuntime` 内部脚本循环中；与 AAS 的 `langgraph` 引擎（`grants + budget + WAL`）未打通。 | 构建 **Center Sandbox Plugin**（基于 AAS `langgraph` 引擎 + 本地 `Qwen3.5-2B` 服务）：<br>① **TaskGen 层**：基于已投递的汇报与中心局部 Belief 运行候选任务生成（`A050`）；<br>② **Planner/Scheduler 层**：结合能力、库存、容量与道路可达性约束完成任务规划与派发（`A052`），全程受 AAS `grants/budget/WAL` 审计。 |
| **3. 端侧具身 Harness（PX4 容器 + 本地开源模型）** | `AeroRescue` 内无 PX4 接入，UAV 运动由 `core/sim/actions.py` 内部离散位移直接写死；而 AAS 已具备成熟的 `px4-gazebo` Lockstep 容器引擎（`aeroagentsim/px4-gazebo:standalone`）与 `adapters.<px4>.<action>` 命令。 | 构建 **Embodied UAV Harness**：<br>① **飞控物理所有权移交 PX4**：UAV 实体的 `position/velocity/attitude/battery/armed/mode/landed` 由 AAS 原生 PX4 容器引擎唯一写入；`RescueWorld` 通过 `consumes` 订阅已提交的 PX4 位姿并驱动救援业务效果；<br>② **端侧模型接入**：通过 AAS `langgraph` / HTTP Adapter 接入本地 `Qwen3.5-0.8B`（结合本机相机多视角图像与局部状态选择技能/动作）。 |
| **4. 13 个典型场景的全量跑通缺口（B 类资源/生成缺口）** | 16 项原生平台 API 已闭环，但 13 个场景（`SC01, 03, 04, 05, 07, 08, 09, 10, 11, 12, 13, 17, GT04`）存在明确的配置/生成阻点（如 `SC05` 缺接收站、`SC17` 上游生成 `KeyError: 'medical'`、`SC01/GT04` 耗材/容量与搬运接力策略）。 | 逐一修复场景配置缺口与 `SC17` 生成器字典映射，补齐多主体协同策略（如室内重伤员由具备承载能力的队员/载具接力至室外上车点），实现 13 个场景原生闭环通过。 |

---

## 二、 兼容 AAS Plugin 思路的总体架构设计

```
+-----------------------------------------------------------------------------------+
|                        AeroAgentSim (AAS) DES Lockstep Kernel                     |
|                   (唯一全局时钟、Fact 版本库、命令路由、WAL 审计与回放)              |
+-------------------+-----------------------------------+---------------------------+
                    |                                   |
      [1. 中心沙箱插件 (Center Sandbox)]      [2. 端侧具身 Harness (Edge Embodied)]
   aeroagentsim.engines: langgraph         aeroagentsim.engines: langgraph + px4
   +---------------------------------+     +---------------------------------------+
   | 中心模型: 本地 Qwen3.5-2B Server |     | 端侧模型: 本地 Qwen3.5-0.8B Server     |
   | - 观察: 已送达战报 + 中心 Belief |     | - 观察: 自身遥测 + 相机感知 + 分配任务|
   | - TaskGen: 触发候选任务 (A050)   |     | - 动作决策: 救援动作 + 航点指令       |
   | - Planner: 约束规划与调度 (A052) |     +-------------------+-------------------+
   +----------------+----------------+                         |
                    | rescue.command.assign_task               | adapters.<px4>.goto_location
                    v                                          v
+-------------------+------------------------------------------+--------------------+
|   rescue_scenes (RescueWorld ContextEngine)      |   PX4/Gazebo Lockstep Engine   |
| - 唯一救援业务状态 Owner (伤员/建筑/障碍/库存/任务)| - Docker: px4-gazebo:standalone|
| - 消费 PX4 发布的真实 UAV 位姿与电量              | - 唯一飞控物理状态 Writer       |
| - 执行 A001~A063 原生状态流转并向 Kernel 发回执    |   (pose/vel/att/battery/landed)|
+--------------------------------------------------+--------------------------------+
```

### 核心设计准则（防膨胀、零内核复制）
1. **单一事实源**：绝不在 `aerorescue` 中另起炉灶写第二个仿真器或时钟循环，所有状态读写、命令投递与模型调用全部走 AAS `Partition` 与 `JournalClient`。
2. **模型服务解耦**：本地 `Qwen3.5-2B`（中心）与 `Qwen3.5-0.8B`（端侧）通过轻量级 OpenAI 兼容推理端点暴露，通过环境变量 `AEROAGENTSIM_PROVIDER_PROFILES` 注册为 `center-qwen` 与 `edge-qwen`，场景配置仅引用 `provider: {mode: live, profile: ...}`，保证 WAL 可完整记录并支持无模型离线回放。
3. **PX4 与地面/水面混合具身**：UAV 接入 AAS 原生 `aeroagentsim/px4-gazebo:standalone` 容器（真实 MAVSDK/Gazebo 闭环）；地面车辆、工程机械、救援艇与搜救犬继续使用 `RescueWorld` 经验证的道路网/水域物理运动模型。

---

## 三、 四阶段推进路线图（Work Packages）与多代理协同分工

| 阶段 | 工单内容 | 写作用域 (Write Set) | 负责通道 / 协同方式 | 里程碑验收标准 |
|---|---|---|---|---|
| **WP1**<br>13 场景生成与 B 类缺口清零 | 1. 修复 `SC17` 上游生成 `facility_targets['medical']` 映射错误并生成冻结 `scene.json`<br>2. 补齐 `SC05` 岸边开放接收站与泊位配置<br>3. 核准 `SC01 / SC03 / SC07 / GT04` 耗材库存、载具接力与建筑评估规则，跑通 13 场景基线 | `configs/*.json`<br>`scene_data/upstream/`<br>`src/rescue_scenes/forced_rules.py` | • **`ds` 子代理**：批量核查 13 场景配置并实现修复<br>• **`sol` 子代理**：审计修复前后的 `initial_truth.json` 与资源守恒断言 | 13 个场景全部成功物化且原生动作链无 B 类配置阻断 |
| **WP2**<br>`RescueWorld` 命令/回执总线与中心沙箱插件 | 1. 在 `RescueWorld` 暴露 `rescue.command.*` 命令接收与原生动作回执映射<br>2. 实现 `aerorescue.center_sandbox:build_graph`（封装 `taskgen` 候选提议 + Qwen3.5-2B 规划调度 + `A050/A052` 下发） | `src/rescue_scenes/engine.py`<br>`src/aerorescue/center_sandbox.py`<br>`tools/make_run.py` | • **`ds` 子代理**：编写命令处理与 `center_sandbox.py`<br>• **`sol` 子代理**：审查 `grants/budget/WAL` 契约合规性 | 中心沙箱通过 `langgraph` 引擎驱动任务生成（`A050`）与分配（`A052`），WAL 记录完整 |
| **WP3**<br>端侧开源模型与原生相机感知闭环 | 1. 启动本地 `Qwen3.5-2B / 0.8B` OpenAI 兼容服务端脚本<br>2. 实现 `aerorescue.edge_harness:build_graph` 与本地 Qwen-VL 相机感知插件（替代失效的 401 远端接口），补齐“图像→请求→推理→观测(A006/A041)”完整关联证据（消除 D 类待定项） | `src/aerorescue/model_server.py`<br>`src/aerorescue/edge_harness.py`<br>`src/rescue_scenes/perception_qwen.py` | • **`ds` 子代理**：实现本地 Qwen 服务与端侧 Harness<br>• **外部 `claude` CLI**：协同联调相机捕获与多模态 JSON Schema 校验 | 端侧 Agent 基于真实渲染相机图像与局部状态自主完成感知上报与动作执行 |
| **WP4**<br>PX4 容器具身桥接与端到端联调 | 1. 实现 `src/rescue_scenes/px4_bridge.py`，将 UAV 实体的移动动作（`A001`）桥接至 AAS 原生 PX4 容器命令（`arm/takeoff/goto_location/hold/land`）<br>2. 绑定 ENU 坐标/高度基准与 `pose_clock_id`，在代表性场景跑通 `PX4 SITL + Center 2B + Edge 0.8B` 全闭环 | `src/rescue_scenes/px4_bridge.py`<br>`src/rescue_scenes/engine.py`<br>`tests/test_px4_embodied_harness.py` | • **外部 `claude` CLI**：操作 `aeroagentsim/px4-gazebo:standalone` 容器联调<br>• **`sol` 子代理**：审计单写者（Single-Writer）位姿约束与时钟对齐 | UAV 由真实 PX4 容器驱动飞行，配合中心沙箱与地面力量完成救援任务并支持 Web 回放 |

---

## 四、 最终工程交付内容清单（Deliverables）

| 序号 | 交付物名称 | 路径 | 验收口径 |
|---|---|---|---|
| **D1** | **13 场景全通配置与生成包** | `configs/*.json` + `scene_data/` | 全部 13 个场景（`SC01..13, SC17, GT04`）物化与原生动作链验证通过，附 `runs/verify/scenes_13_matrix.json` |
| **D2** | **`RescueWorld` 原生命令与 PX4 位姿桥** | `src/rescue_scenes/engine.py` + `src/rescue_scenes/px4_bridge.py` | 兼容 AAS `Partition` 单写者契约；支持外部命令驱动与 PX4 容器实时位姿/电量同步 |
| **D3** | **中心式沙箱插件 (Center Sandbox)** | `src/aerorescue/center_sandbox.py` | 兼容 AAS `langgraph` 引擎；集成动态任务生成（`taskgen`）、资源约束规划与任务派发，接入本地 `Qwen3.5-2B` |
| **D4** | **端侧具身 Harness 与本地视觉感知** | `src/aerorescue/edge_harness.py` + `src/rescue_scenes/perception_qwen.py` | 接入本地 `Qwen3.5-0.8B` 与真实相机渲染帧，消除平台 API 矩阵中的唯一 D 类项（`camera_image_inference`） |
| **D5** | **一键启动与回归验证套件** | `tools/make_run.py` + `tools/serve_local_qwen.py` + `tests/` | 支持 `--execution-mode {forced_rules, autonomous, llm_sandbox}` 与 `--px4-uav` 旋钮；全量 `pytest` 通过 + 浏览器回放可查 |
