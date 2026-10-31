# 阶段 1 本机执行契约与长期完备计划（Astra 3-Round Debate 定稿版）

> **状态**：经 Lead 与 `astra` (`gpt-6-astra`, `xhigh`) 3 轮深度交锋收敛定稿（2026-10-31）  
> **审计铁律**：所有阶段代码、数据防泄漏、视觉把关与实验结果均由唯一主体审计子代理 **`sol` (`gpt-6.1-sol`, `high`)** 正式审计；若审计发现未通过项，直接由 `sol (high)` 接手修复。  
> **本阶段推进终止节点**：本机阶段 1（多场景空地真图渲染与分支采样数据集 + 视觉把关 + Pair 0 `Qwen3.5-0.8B ↔ Qwen3.5-2B` 真实 Decoder 闭环 + ISCC 核心算子与 4 项机理诊断 + Colab 5 Pairs 套件打包）全部通过 `sol (high)` 验收，停在 **「等待 Colab 运行」** 节点。

---

## 一、 3 轮 Debate 收敛的 8 条冻结实现契约（Frozen Implementation Contracts）

### 契约 ①：双世界 Non-Interference、Matched Hidden-State Pairs 与三轴正交划分
1. **模型输入白名单（Allowlist）与零泄漏断言**：
   - 模型输入 `model_input` 仅包含：`mission_brief`（场景级静态任务简报，严禁含当前参考动作或目标实体 ID）、`center_delivered_context`（仅限经网络投递到 `coordinator.belief` 且 `source != 'prior'` 的战报与初始静态站点表）、`edge_telemetry_and_belief`（本机位姿、电量、载荷、名义能力、本机已观测局部 belief）、`aerial_ground_views`（无任何调试文字/边框的 $640\times 384$ 传感器 RGB 图像、传感器自身光轴 Z 深度 `depth_path`（单通道 Float32 米制 `.npy`，仅包含自身有效观测与天空零值）、带噪先验外参 $\widehat{\mathbf{T}}_{e\leftarrow i}$、内参 $\mathbf{K}_i, \mathbf{K}_e$）、`public_candidate_menu`（仅含 `{candidate_id, option_id, action_id, task_type, target_anchor, params}`，彻底剔除 `source`、`priority`、`oracle_feasible`）。
   - 监督与真值评估数据（分支收益、真值位姿 $\mathbf{T}_{e\leftarrow i}^\star$、离线真值深度/法向/实例掩码及真值共视掩码）独立存放于 `supervision_and_eval` 字段，推理与评测算子签名仅接收 `model_input`。
2. **Matched Hidden-State Pairs（配对隐状态断面）**：
   - 从 $t=0$ 构造仅在地面前景建筑物遮挡后方存在左右走廊阻断状态差异（或开阔院落遮挡区域伤员分布差异，分项独立控制）的成对世界 $(\omega^{(+)}, \omega^{(-)})$。
   - **Non-Interference 硬断言**：在断面 $t$，地面历史、地面传感器 RGB 图像与地面完整序列化输入满足 `sha256(ground_rgb_plus) == sha256(ground_rgb_minus)` 且 `sha256(canonical_json(ego_input_plus)) == sha256(canonical_json(ego_input_minus))`。
   - **50% 硬上界充要条件**：在冻结的窗口 $H=120\text{s}$ 与奖励权重下，强制检验成对状态的最优宏动作集合互不相交（$A_+^\star \cap A_-^\star = \emptyset$）且先验等概（$0.5 / 0.5$），使任何仅依赖地面 Ego 输入的策略在配对样本上的硬准确率理论上界严格 $\le 50\%$。
3. **三轴正交字段**：
   - `split ∈ {train, val, test}`：按场景与几何/布局祖先簇严格隔离——`train = {SC01, SC09}`，`val = {SC07, SC10}`，`test = {SC08, GT04, SC05(水域对照)}`，同一场景的所有 seed、双胞胎对与分支永远留在同一 `split`；
   - `sample_kind ∈ {natural, challenge_matched}`：独立标识自然轨迹断面（含无收益/过期/失锁断面）与配对挑战断面；
   - `physical_regime ∈ {A, B, C, water}`：由实测空地三维共视重叠率、视角正交度与特征谱有效秩判定。

### 契约 ②：宏动作 Option FSM、纯核心克隆、具名收益字典与稳定外生 CRN
1. **宏动作 Option 与合法续推器 $\rho_{\mathrm{legit}}$**：
   - 候选菜单由 $K \in [4, 6]$ 个具名宏动作 Option 构成（如沿左/右走廊机动+区域感知、前方受阻路段清障、受困目标接触/解困/急救/装载/交接、原地保持与视角扫描），底层由合法的 `rho_legit` 控制器调用名义能力原语序列执行（仅读 agent 自身 `belief`，**彻底禁用 `ForcedRuleDriver` 的临时 `air` 跨墙移动覆盖与能力偷加**）。
2. **纯核心克隆 `clone_core_env` 与 4 项隔离断言**：
   - 剥离外部钩子深拷贝 `Environment` 核心状态，重绑事件堆回调与 `sim` 引用；强制通过 4 项单测断言：① 无操作续推同轨；② 反复克隆哈希一致；③ 打乱分支执行顺序结果一致；④ 兄弟分支状态变异后父状态哈希不变。
3. **稳定外生共同随机流（CRN）与单次 $240\text{s}$ 前缀切片**：
   - 外生随机源按 `(scene_id, seed, actor_id, event_family, purpose, target_anchor, quantized_t, local_seq)` 键索引，绝不使用随分支控制流变化的 `action_id`。
   - 单次向前推进 $240\text{s}$，在 $H \in \{60\text{s}, 120\text{s}(\text{主}), 240\text{s}\}$ 三个前缀切点记录具名增量收益字典 `named_rewards = {person_found, contacted, extricated, aided, handed_over, road_cleared, hazard_handled, action_failed, energy_Wh, active_busy_s}`（按 `unique(entity_id, milestone)` 去重，时间项记录实际作业繁忙时长 `active_busy_s` 与首次达成时刻）。

### 契约 ③：全链路坐标系、$640\times 384$ 整除网格与阻尼单应性雅可比
1. **分辨率与 `Qwen3.5` Post-Merger 网格运行时断言**：
   - 本机 `Qwen3.5-0.8B-Base`（raw ViT `width=768, depth=12` $\to$ post-merger `d_i=1024`）与 `Qwen3.5-2B-Base`（raw ViT `width=1024, depth=24` $\to$ post-merger `d_e=2048`）均采用 `patch_size=16, merge_size=2`。
   - 图像统一渲染为 **$640 \times 384$**，运行时强制断言处理器输出 `image_grid_thw == (1, 24, 40)` 且 post-merger 视觉 token 数严格等于 $12 \times 20 = 240$，像素到 post-merger 特征网格的坐标导数严格为 $\mathbf{A}_i = \mathbf{A}_e = \frac{1}{32}\mathbf{I}_2$。
2. **$\mathrm{SE}(3)$ 左乘扰动与阻尼伪逆**：
   - 统一接收端相机坐标系下的左乘扰动 $\widehat{\mathbf{T}}_{e\leftarrow i} = \exp(\boldsymbol{\xi}^\wedge)\mathbf{T}_{e\leftarrow i}^\star$，$\mathbf{J}_{\mathrm{se3}}(\mathbf{p}_k^{(e)}) = \big[-[\mathbf{p}_k^{(e)}]_\times \;\; \mathbf{I}_3\big] \in \mathbb{R}^{3\times 6}$；仅在变更线性化参考位姿时通过伴随变换 $\mathrm{Ad}(\Delta \mathbf{T})$ 传输协方差并叠加马尔可夫过程噪声。
   - 深度 $d_k = Z_c > 0$ 为光学相机坐标系线性 $Z$ 深度，表面法向转换至源相机系 $\mathbf{n}_k^{(i)} = \mathbf{R}_{i\leftarrow w}\mathbf{n}_k^{(w)}$；像素单应性雅可比 $\mathbf{H}_{\mathrm{px}}(k) = \mathbf{J}_{\mathrm{proj}}(\mathbf{p}_k^{(e)})\mathbf{R}_{e\leftarrow i}\big(\mathbf{I}_3 - \frac{\mathbf{p}_k^{(i)}\mathbf{n}_k^{(i)\top}}{\mathbf{n}_k^{(i)\top}\mathbf{p}_k^{(i)}}\big)\mathbf{K}_i^{-1}\mathbf{E}_{3\times 2} d_k \in \mathbb{R}^{2\times 2}$，特征网格雅可比 $\mathbf{G}_{f,k} = \mathbf{A}_e \mathbf{G}_{\mathrm{px}, k}, \mathbf{H}_{f,k} = \mathbf{A}_e \mathbf{H}_{\mathrm{px}}(k)\mathbf{A}_i^{-1}$；在掠射角或高条件数处采用阻尼伪逆 $\mathbf{H}_{f,k}^\dagger = (\mathbf{H}_{f,k}^\top\mathbf{H}_{f,k} + \lambda_H \mathbf{I}_2)^{-1}\mathbf{H}_{f,k}^\top$ 并标记几何可靠性权重。

### 契约 ④：薄 SVD 竖直堆叠 $\mathbf{B}_Y$ 与基于真实 Decoder Logits 的残差校准策略
1. **正确的薄 SVD 竖直堆叠**：
   - 令第 $j$ 个盲区槽位的加权雅可比块为 $\mathbf{B}_j = \mathbf{F}_\theta^{1/2}\mathbf{J}_{Y,j} \in \mathbb{R}^{K \times d_e}$，定义**竖直堆叠矩阵**：
     $$\mathbf{B}_Y = \begin{bmatrix} \mathbf{B}_1 \\ \mathbf{B}_2 \\ \vdots \\ \mathbf{B}_k \end{bmatrix} \in \mathbb{R}^{(kK) \times d_e}, \qquad \mathbf{B}_Y^\top \mathbf{B}_Y = \sum_{j=1}^k \mathbf{J}_{Y,j}^\top \mathbf{F}_\theta \mathbf{J}_{Y,j} = \mathbf{S}_{\mathrm{ch}} \in \mathbb{R}^{d_e \times d_e}$$
   - 直接在未截断的 $\mathbf{B}_Y \mathbf{A}_{\mathrm{ref}} \in \mathbb{R}^{(kK)\times r}$ 上计算稳定秩 $\mathrm{srank}(\mathbf{B}_Y \mathbf{A}_{\mathrm{ref}}) = \frac{\|\mathbf{B}_Y \mathbf{A}_{\mathrm{ref}}\|_F^2}{\|\mathbf{B}_Y \mathbf{A}_{\mathrm{ref}}\|_2^2}$ 用于机理诊断 1（绝不先截断 4 维再自证低秩）；再对 $\mathbf{B}_Y$ 做 Thin SVD 取前 $r_{\mathrm{act}}=4$ 个右奇异向量构成 $\mathbf{U}_{\mathrm{act}} \in \mathbb{R}^{d_e \times 4}$。
2. **基于真实 Decoder Logits 的残差校准策略与 Shadow Probe 导数**：
   - 将接收端部署策略显式定义为残差校准形式：
     $$\mathbf{z}_{\mathrm{dec}}(\mathbf{v}) = \mathbf{z}_{\mathrm{ego}}(h_t) + \underbrace{\Big[\mathbf{z}_{\mathrm{aug}}\big(h_t, \mathbf{v}, \widehat{\mathbf{P}}_{\mathrm{anchor}}\big) - \mathbf{z}_{\mathrm{aug}}\big(h_t, \mathbf{0}, \widehat{\mathbf{P}}_{\mathrm{anchor}}\big)\Big]}_{\mathcal{R}_\theta(h_t, \mathbf{v}, \widehat{\mathbf{P}}_{\mathrm{anchor}})}$$
     其中 $\mathbf{z}_{\mathrm{aug}}(h_t, \mathbf{v}, \widehat{\mathbf{P}}_{\mathrm{anchor}})$ 为向真实 `Qwen3.5-2B` 24 层语言 Decoder 追加软视觉槽位 $\mathbf{v} = \widehat{\mathbf{Y}}_T^{\mathrm{gated}}$ 与公共候选走廊三维位置嵌入 $\widehat{\mathbf{P}}_{\mathrm{anchor}}$ 后的候选动作符号 logits。
   - 该定义在数学与代码上恒满足 $\mathcal{R}_\theta(h_t, \mathbf{0}, \cdot) \equiv \mathbf{0}$（即 $\mathbf{z}_{\mathrm{dec}}(\mathbf{0}) = \mathbf{z}_{\mathrm{ego}}(h_t)$），同时在 $\mathbf{v}=\mathbf{0}$ 处的 Shadow Probe 雅可比 $\mathbf{J}_Y = \frac{\partial \mathbf{z}_{\mathrm{aug}}}{\partial \mathbf{v}}\big|_{\mathbf{v}=\mathbf{0}}$ 非零可微，彻底消除 $w=0$ 时的查询死锁！当门控全关（$\mathbf{w}=\mathbf{0}$）且无需探测时，直接短路返回 $\mathbf{z}_{\mathrm{ego}}(h_t)$。

### 契约 ⑤：固定接收端子空间 RLS 持久状态与发送端预测 Schur 补闭合
1. **持久状态驻留固定接收端子空间 $\mathbf{U}_{\mathrm{rec}} \in \mathbb{R}^{d_e \times p}$ ($p=32$)**：
   - 在 `train` split 上提取接收端 post-merger 视觉特征的固定正交基 $\mathbf{U}_{\mathrm{rec}} \in \mathbb{R}^{2048 \times 32}$ 与发送端固定正交基 $\mathbf{U}_{\mathrm{src}} \in \mathbb{R}^{1024 \times 32}$（带固定版本号 `basis_version`）。
   - 在线 RLS 滤波器持久维护固定子空间坐标下的映射行向量 $\widehat{\mathbf{A}}_t \in \mathbb{R}^{p \times r}$ ($32\times 32$) 及逐行精度矩阵 $\{\mathbf{\Lambda}_{m, t}\}_{m=1}^p$（避免因 $\mathbf{U}_{\mathrm{act}, t}$ 每步旋转导致历史协方差失配），由 $\widehat{\mathbf{Y}}_T = \mathbf{U}_{\mathrm{rec}}\widehat{\mathbf{A}}_t \widehat{\mathbf{X}}_T$ 显式重建 $d_e=2048$ 维接收端特征。
2. **发送端预测 Schur 补 $\mathbf{\Omega}_{\mathrm{pred}}$ 与接收端实测 $\mathbf{\Omega}_{\mathrm{meas}}$**：
   - 发送端空间梯度 $\mathbf{D}_{i, k} = \nabla_{\mathbf{u}} \mathbf{x}_k \in \mathbb{R}^{r \times 2}$ 定义在 $r=32$ 维源 PCA 坐标上；发送端维护仅按已 ACK 锚点同步更新的代理后验协方差 $\mathbf{P}_{\mathrm{pred}, t-1}$。
   - 热启动（`mode=query`）下，接收端下发投影查询基 $\mathbf{V}_{\mathrm{query}} = \widehat{\mathbf{A}}_{t-1}^\top (\mathbf{U}_{\mathrm{rec}}^\top \mathbf{U}_{\mathrm{act}, t}) \in \mathbb{R}^{r \times 4}$、查询奇异值权重 $\boldsymbol{\lambda}_{\mathrm{act}} \in \mathbb{R}^4$ 与 $\Sigma_{\xi, t-1}$；发送端构造 $1\times 6$ 代理行 $\widehat{\mathbf{m}}_{\ell, k}^{(i)} = \mathbf{v}_\ell^\top \mathbf{D}_{i, k} \mathbf{H}_{f, k}^\dagger \mathbf{G}_{f, k}$，计算 $\mathbf{\Omega}_{\ell}^{\mathrm{pred}}$ 下的 Schur 补秩-1 条件创新收益；接收端收到锚点后用本地真实梯度 $\mathbf{D}_{e, k} = \mathbf{U}_{\mathrm{rec}}^\top \nabla_{\mathbf{v}}\mathbf{y}_k \in \mathbb{R}^{p\times 2}$ 构造 $\mathbf{\Omega}_{m, t}^{\mathrm{meas}}$ 执行 Woodbury RLS 更新，并量化报告 $\mathbf{\Omega}_{\mathrm{pred}}$ 与 $\mathbf{\Omega}_{\mathrm{meas}}$ 的校准差距。
   - 冷启动或失锁时下发 `mode=bootstrap`，发送端采用“几何 D-最优边缘 + 多样性表征锚点 + 保底盲区预算 $b_s \ge 0.25 B_t$”；零共视时直接令 $b_g=b_a=0$ 并触发主动视角动作 $a_t^{\mathrm{view}}$。所有下行查询、位姿协方差、ACK 与上行负载均精确计入字节账本。

### 契约 ⑥：经验校准正部收缩门控、All-Off 短路与三项核心风险指标
1. 采用经验校准正部收缩门控 $w_{\ell, j}^{\mathrm{cal}} = \max\big(0, 1 - \frac{\kappa_{\mathrm{gate}}\hat{\sigma}_{\ell, j}^2}{\hat{\mu}_{\ell, j}^2 + \epsilon}\big)$（$\kappa_{\mathrm{gate}}$ 仅在 `val` split 上校准）；当 $\mathbf{w}=\mathbf{0}$ 时短路返回同一次前向的 $\mathbf{z}_{\mathrm{ego}}$。
2. 强制报告三项安全与覆盖指标：
   - **Negative Transfer Rate (NTR, %)**：同根配对评估中 $\hat{Q}_H^\rho(s_t, a_{\mathrm{coop}}) < \hat{Q}_H^\rho(s_t, a_{\mathrm{ego}}) - \epsilon_Q$ 的断面比例；
   - **CVaR-10% Regret**：最差 $10\%$ 断面的平均物理后悔值；
   - **Active Communication Coverage (%)**：非零门控（$\|\mathbf{w}\|_1 > 0$）生效的断面比例及平均实耗字节数（Bytes/Step）。

### 契约 ⑦：独立 $\mathbf{A}_{\mathrm{ref}}$ 拟合、真实 `Qwen3.5-2B` 语言 Decoder 闭环与公平 Ego 对照
1. **$\mathbf{A}_{\mathrm{ref}}$ 独立特征监督**：在 `train` split 上用成对无遮挡特征独立闭式拟合参考映射 $\mathbf{A}_{\mathrm{ref}}$（分别报告同图跨编码器 teacher 残差与跨视角同表面残差），绝不与决策层联合训练。
2. **真实 `Qwen3.5-2B` 语言 Decoder 与公平训练**：在 `train` split 上使用含通信 dropout（$p_{\mathrm{drop}}=0.5$，确保 `Ego-Only` 分支获得同等训练监督！）与位姿/量化扰动的数据微调轻量级软 token 读取投影层，驱动真实的 24 层 `Qwen3.5-2B` 语言 Decoder 输出动作决策并提取真实跨层注意力与层间输出敏感度。

### 契约 ⑧：复用 `engine.py` 场景显示变换、四通道渲染 QA 与 Colab 套件边界
1. 复用 `src/rescue_scenes/engine.py`（552–672 行）的实体状态变体选择、高程贴合与法向倾斜变换，将背景建筑网格与动态伤员/路障/车辆统一拼入同切面三维网格，输出无任何调试文字/边框的 $640\times 384$ `RGB + Float32 Depth + Float32 Normal + Uint32 Instance-ID`；共视与盲区采用深度自适应容差 $\epsilon_{\mathrm{abs}} + \epsilon_{\mathrm{rel}} Z$、法向一致性及 post-merger token 占用率校验。
2. 本机完成 6 代表性场景（`SC01, SC09, SC07, SC10, SC08, GT04`）+ `SC05` 水域对照的真实图像数据集、分层 Contact Sheets、Pair 0 完整闭环实验与 4 张实测机理诊断图谱，并经 **`sol (high)`** 审计通过后，打包同接口 Colab 5 Pairs 执行套件，停在 **「等待 Colab 运行」** 节点。


## 二、2026-10-31 用户授权的 RGB-D 边界修订

- `model_input.aerial_ground_views.{uav_view,ground_view}.depth_path` 是相应 RGB-D 传感器自身观测的 Float32 光轴 Z 深度（米，天空/无效点为 0），允许在线读取；文件必须是单数组 `.npy`，不能是带实例 ID、法向真值或共视真值的 NPZ 容器。
- 在线法向由自身深度反投影后的邻域切向量叉积求得。`instance_id`、`covis_mask`、真值外参、分支收益只供离线监督/评估，禁止进入发送端候选筛选和接收端配准。
- 发送端候选只依赖其 RGB-D、公共候选菜单和解码后的接收端查询。接收端只能用自身 RGB-D 与解码后的上行锚点匹配，成功项通过真实 ACK 更新发送代理状态；全部控制、锚点、坐标、量化参数及 ACK 以实际序列化长度计费。
- train/val/test 都构造单因素路障交换挑战：地面历史、RGB、深度与其他模型输入完全相同；人员位置固定；只有左右路障状态及其直接道路阻断效应变化。样本所属场景与布局祖先始终留在原 split。
- 本修订不授权使用测试集调参或宣称预设排名；`ACCEPTED` 必须同时满足输入隔离、实际传输、真实模型决策与预注册测试指标。
