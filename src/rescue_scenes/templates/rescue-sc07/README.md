# rescue-sc07 · SC07 野火/林地城市交界（火势抑制） / SC07 Rescue mission

## 场景概述 / Scene overview

**中文** — 林地-城市交界火线，需压制火势并评估建筑；显式规则驱动与脚本感知，使用本机保留的场景输入及三维资产。

**English** — Native rescue actions with authored forced rules and scripted sensing. Uses retained local scene bundles and assets.

| 项目 / Item | 内容 / Value |
| :--- | :--- |
| 场景 ID / Scene ID | `rescue-sc07` |
| 灾害与环境 / Hazard & environment | SC07 野火/林地城市交界（火势抑制） |
| 地形资产 / Terrain asset | `/data/scenes/SC07_appearance_IMAGEGEN_C01only_yup.glb`（真实米制 GLB，保留原始尺度与朝向 / source metres, no rescale） |
| 高斯泼溅 / Splat overlay | 含 1 个高斯泼溅（SPZ）叠加层 / 1 Gaussian-splat (SPZ) overlay(s) |
| 智能体阵容 / Agent roster | UAV×1, Team×1, Coordinator×1, FireUnit×1, Station×1 |
| 授权运行时长 / Authored horizon | 3600 秒（3600000000000 ns；最短步长 500000000 ns） / 3600 s (3600000000000 ns; minimum step 500000000 ns) |
| 执行模式 / Execution mode | `forced_rules`（参考规则驱动 / authored reference driver） |

场景中的所有受困人员、道路、建筑与灾害角色都由冻结的救援 bundle 在 **Initialize (t=0)** 时发布；草稿预览不伪造这些运行时角色，只显示真实地形与相机。 / All victims, roads, buildings and disaster actors are published by the frozen bundle at **Initialize (t=0)**; the draft preview never fabricates runtime actors and only shows the real terrain and camera.

## 运行步骤 / How to run

1. 打开或新建工作区，进入 **Scenario controls**。 / Open a workspace and go to **Scenario controls**.
2. 修改任一控制值（例如把 `until_ns` 改为 `60000000000`，即 60 秒）。 / Change one control (e.g. `until_ns` to `60000000000`, i.e. 60 s).
3. 点击 **Apply controls to draft**，再执行 **Validate**。 / Click **Apply controls to draft**, then **Validate**.
4. **Save version** 保存为新版本；旧版本与已记录运行保持冻结。 / **Save version** freezes a new version; previous versions and recorded runs stay pinned.
5. **Initialize latest version (t=0)**，让引擎发布该版本的智能体与任务。 / **Initialize latest version (t=0)** so the engine publishes this version's actors and tasks.
6. **Start** 运行；回放与地形/泼溅变换与作者坐标一致。 / **Start** the run; replay terrain and overlays agree with the authored transforms.

## 三项可用控制 / The three working controls

| 控制 / Control | 指针 / Pointer | 类型 / Type | 默认 / Default | 范围 / Range |
| :--- | :--- | :--- | :--- | :--- |
| 运行时长 `until_ns` | `/run/until_ns` | integer | `3600000000000` | `500000000` – `3600000000000` ns |
| 执行模式 `execution_mode` | `/engines/rescue/config/execution_mode` | choice | `forced_rules` | `forced_rules`, `autonomous` |
| 运行节奏 `pacing` | `/run/pacing` | choice | `fast` | `fast`, `realtime` |

- 纳秒换算 / Nanoseconds: `1 s = 1,000,000,000 ns`，因此 `60000000000` = 60 s。
- `forced_rules` 运行已编写的参考规则驱动；`autonomous` 使用现有本地智能体运行时与脚本感知（不是未接入的 LLM/PX4 能力）。 / `forced_rules` runs the authored reference driver; `autonomous` uses the existing local agent runtime with scripted sensing (not a future LLM/PX4 capability).
- 缩短时长只得到部分任务记录；冻结的完整任务目标不变。若在移交前结束，`rescue_time_to_confirmation_s` / `rescue_time_to_delivery_s` 会保持为空（null）。 / A shorter horizon is a partial run; the frozen full-mission target is unchanged. Ending before handover leaves the confirmation/delivery times null.

## 冻结输入与重播种 / Frozen inputs and reseeding

本模板的种子、布局、资产清单与场景描述都已在 `scenario.json` 中冻结并带摘要。真正重新播种（世界重生成）需要重新物化 bundle、实体、资产绑定与来源信息；仅修改运行种子不会改变世界。 / This template's seed, layout, asset manifest and scene descriptor are frozen and digested in `scenario.json`. Real world reseeding requires rematerializing the bundle, entities, asset bindings and provenance; changing a run seed alone does not change the world.

## 预览与相机 / Preview and camera

- 草稿预览显示本场景的真实米制地形 `/data/scenes/SC07_appearance_IMAGEGEN_C01only_yup.glb` 与SPZ 叠加层；运行时角色在 Initialize 之后才出现。 / The draft preview shows this scene's real metric terrain and its SPZ overlay; runtime actors appear only after Initialize.
- 场景带有作者设定的起始相机（`camera.fixed`），加载时保留该视角；轨道旋转、缩放与 **Fit scene** 仍可手动使用。 / The scene ships an authored starting camera (`camera.fixed`) kept at load; orbit, zoom and **Fit scene** remain manually available.
- 若地形或泼溅资源加载失败，预览会显示可见的加载错误；该失败不会伪造替代地形。 / If terrain or splat assets fail, the preview shows a visible load error instead of substituting a fake scene.

## 指标说明 / KPIs

`tasks_created`, `tasks_completed`, `tasks_failed`, `completion_rate`, `mean_task_latency_s`, `handoffs_completed`, `rescue_time_to_confirmation_s`, `rescue_time_to_delivery_s`

- 任务完成率基于 `lowalt.task.*` 任务事件，而非“任务成功”布尔。 / Task completion is based on `lowalt.task.*` events, not a success flag.
- `handoffs_completed` 只统计已确认接收的 `lowalt.handoff.transferred` 事件。 / Handoffs are acknowledged receiving events.
- 确认/送达时长需要真实事件链（发现 → 联系 → 移交确认）；缺失时保持 null。 / Confirmation/delivery intervals require the real event chain; unmeasured values stay null.
