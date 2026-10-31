# AeroRescue 工作区主线规范（Workspace Instructions）

本工作区（`/mnt/data2/weizhiwei/aeroagentsim/AeroRescue`）包含**科研主线**与**工程主线**两大公用核心资产，所有主代理、子代理（`sol` / `ds` / `astra`）与外部 `claude` CLI 在执行任务前须以此为准：

## 1. 权威公用财产索引

| 主线维度 | 权威文档路径 | 核心定位与内容 |
| :--- | :--- | :--- |
| **科研主线（顶刊/顶会理论与实验）** | [`docs/MAINLINE_AGENTIC_ISCC_BLUEPRINT.md`](docs/MAINLINE_AGENTIC_ISCC_BLUEPRINT.md) | 面向多模型变体（Multi-Variant，5 Pairs）与多空地视角（Aerial-Ground）的 Agentic-Native ISCC：$\mathrm{SE}(3)$ 射影雅可比与跨锚点秩-6 相关 EIV 建模、拉回低秩 Fisher 度量、三大核心定理（Woodbury 秩-6 折损、空间梯度二分律、Wiener 收缩门控防负迁移）、四阶段协议与 Algorithm 1、3 类退化场景（Regimes A/B/C）× 零稻草人 SOTA 对比 × 4 项机理可解释性诊断。 |
| **工程主线（AAS 插件、PX4 具身与中心沙箱）** | [`docs/engineering-master-plan.md`](docs/engineering-master-plan.md) | 13 个典型灾害救援场景（`SC01..SC17, GT04`）全通、`RescueWorld` 命令/回执总线、基于 `langgraph` 的中心沙箱（任务生成 `A050` + 规划调度 `A052`）、端侧具身 Harness（接入 AAS 原生 `px4-gazebo` 容器与开源 VLM）。 |
| **代码与资产目录布局** | [`docs/migration-layout.md`](docs/migration-layout.md) | 所有三维场景、GLB、高斯泼溅、LOD 缓存与上游生成器 `scene-v0.6` 均已自包含归拢于 `scene_data/` 下，不依赖工作区外的 `blender-data` 或旧工作树。 |

## 2. 子代理角色与审计修复铁律

- **`astra` (`magpie-responses/auto/gpt-6-astra`, `xhigh`)**：负责架构精简提速与科研学术讨论（含阶段 1 方向与长期计划 3 轮+ Debate）。
- **`ds` (`magpie/auto/DeepSeek-V4.1-Flash`)**：负责日常功能初版开发、脚本实现与大规模日志/数据摸底。
- **`sol` (`magpie-responses/auto/gpt-6.1-sol`, `high`)**：**唯一主体审计子代理 + 审计未通过后的直接修复者**。所有代码、数据防泄漏、视觉真实性把关与实验结果必须经 `sol (high)` 审计；**一旦审计不通过，后续修复直接交由 `sol (high)` 执行**，禁止让普通开发代理反复试错。

## 3. 开发与实验铁律

1. **零稻草人与零伪造**：所有定理验证、SOTA 基线（KVComm、Vision Wormhole、BOLT、Which2comm、Where2comm、RDcomm、VisionZip）与闭环指标必须基于真实模型特征、真实射影几何与真实仿真轨迹运行，严禁使用随机张量或硬编码分数冒充实验结果。
2. **反防御性膨胀（Anti-Bloat）**：算法核（$\mathrm{SE}(3)$ 雅可比 $\mathbf{G}_k$、Woodbury RLS 滤波、Algorithm 1 三元预算求解、Wiener 收缩门控）保持纯粹、向量化的矩阵实现（PyTorch/NumPy），禁止包裹冗余抽象层。
3. **自包含资产边界**：所有新采集的空地多视角 RGB-D/法向数据、预提取特征张量与实验报告统一存放于本工作区 `data/` 与 `runs/` 下。
