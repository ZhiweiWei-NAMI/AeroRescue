# AeroRescue 文献/数据调研（论文 framing）

**日期**：2026-10-14（Asia/Shanghai）  
**约束**：不编造结果；本地包仅证明「SAREnv 子集 + 程序 episode + CPU 逻辑回归参与过滤 + 平台运动/通信」可跑通，**未**证明优于 baseline，**未**完成中心微调/PDDL 闭环评估。

---

## A. 场景特色与挑战（硬点）

1. **时限 + 淹没/失效动态**：洪水/震后救援中任务有截止（水位上涨、可通行窗），迟到即失败（BiG-MRTA 洪水包投送设定；Ghassemi & Chowdhury 2021）。
2. **异构能力 × 角色链**：侦察→确认→中继→投送→评估构成时序依赖，单机无法覆盖全角色；需联盟/参与决策而非一次性全局分配（Drones 2025 时序任务链；AAMAS 2024 联盟形成）。
3. **分布式参与 vs 中心规划**：边缘机只见本地状态+任务通告，需做 **join/keep-or-drop**；中心见候选全集再发可调用计划——与「全员参与拍卖/全局 LLM 调度」不同。
4. **通信昂贵且间歇**：联盟形成与拍卖以消息轮次换协调质量；参与过滤可降候选规模与通信负载（CBBA 2009；Afghah et al. ACC 2018）。
5. **地图先验有、参与标签无**：SAREnv 等提供 OSM 要素 + 失踪概率热图，**不**提供「谁该参与何种角色」监督（Bullock et al. Drones 2025 / namurproject）。
6. **感知标签 ≠ 任务分配标签**：FloodNet / AIFloodSense / UrbanSARFloods 标淹水像素或损伤类，**不能**直接训 participation F1。
7. **自组织稳定性**：边际效用/Shapley 激励 join/leave，但多为仿真效用，非机载有界决策器 + 中心可调用计划接口（Lu et al. AAMAS 2024）。
8. **评估割裂**：现有工作多报路径覆盖/makespan/联盟效用/感知 IoU；少有统一报 **participation F1 × makespan × 通信字节/轮次 × 任务成功率**。

---

## B. 相关工作：做了什么 / 没做什么

| 工作 | 做了什么 | 缺口（相对 AeroRescue：边缘 Jev 过滤 + 中心规划发计划） |
|------|----------|----------------------------------------------------------|
| Queralta et al., *Collaborative Multi-Robot Search and Rescue*, IEEE Access **2020** https://ieeexplore.ieee.org/document/9220149 | SAR 多机器人调研：分配、覆盖、中心/分布式规划、感知 | 无边缘「是否参与」学习过滤；无 PDDL/LLM 混合接口 |
| Choi, Brunet, How, *Consensus-Based Decentralized Auctions (CBBA)*, IEEE TRO **2009** https://ieeexplore.ieee.org/document/5072249 | 分布式拍卖+共识，冲突消解，有近似最优界 | 每机仍参与多轮出价；非轻量 keep/drop；无任务语义+环境语义条件过滤 |
| Afghah et al., *Coalition Formation… Heterogeneous UAV Networks*, ACC **2018** https://arxiv.org/abs/1711.00214 | Leader 建盟、follower 竞价 join；信誉过滤自私机 | 效用/信誉规则，非学习型有界决策器；无中心可调用计划层 |
| Lu et al., *Task-Driven Multi-UAV Coalition Formation*, AAMAS **2024** https://arxiv.org/abs/2403.05108 | 任务驱动联盟、边际效用偏好、Shapley、Nash 稳定 | 仿真效用最大化；无洪水地图/平台运动通信；无 participation 监督指标 |
| Drones **2025**, *Task Allocation… Temporal Task Chains* https://www.mdpi.com/2504-446X/9/8/574 | 侦察–投送–评估时序链；动态 leader + 多轮协商 | 「是否申请参与」为规则能力匹配，非学习过滤；缺参与 F1 |
| Ghassemi & Chowdhury, *BiG-MRTA* 灾害响应 **2021** + https://github.com/adamslab-ub/BiG-MRTA | 洪水包投送：截止、航程、载荷；图匹配在线分配 | **中心/集中式式匹配**；假定任务已知；无边缘参与过滤、无角色语义 |
| MDPI Drones **2024**, *Collaborative Task Allocation… SAR*（震后 PSOGWO）https://www.mdpi.com/2504-446X/8/4/138 | 多 UAV 救援路由、时间窗、元启发 | 集中优化；无分布式参与决策；无真实地图基准绑定 |
| Bullock et al., *SAREnv*, Drones **2025**, 9(9):628 https://www.mdpi.com/2504-446X/9/9/628 ；数据 https://github.com/namurproject/SAREnv | 60 场景、概率图、覆盖规划 baseline（圈/披萨/贪心/随机） | **仅搜索路径**；无联盟/参与/投送/通信指标 |
| AUSPEX, Front. Robot. AI **2025** https://www.frontiersin.org/articles/10.3389/frobt.2025.1583479 | ROS2 救援框架：UP/PDDL/HTN、ALNS、Say’n’Fly LLM-modulo | 规划在中心/地面；**无**边缘轻量参与过滤器；LLM 用于动作剪枝非 join 决策 |
| Liu et al., *LLM+P*, **2023** https://arxiv.org/abs/2304.11477 | NL→PDDL→经典规划器→NL；保证可解性优于纯 LLM | 家务/IPC 域；非 UAV SAR；非多机参与过滤 |
| Zhang et al., *LaMMA-P*, **2024** https://arxiv.org/abs/2409.20560 | LLM 分解+分配+PDDL（Fast Downward）；MAT-THOR | 室内异构；全观假设；无通信负载；无洪水 |
| Kannan et al., *SMART-LLM*, IROS **2024** https://arxiv.org/abs/2309.10062 | 单 LLM 分解/组联盟/出计划（对话式调度） | **正是要对比的「聊天 LLM 调度器」**；非有界 join 过滤；无 SAR 地图 |
| MTU-LLM（异质 SAR 分配）https://cdnintech.com/articles/619/… | LLM 生成分配与路径脚本 | 开发加速向；非边缘过滤+符号加速对照设计 |

**总缺口（可写进 related work）**：  
几乎无人同时做 **(i) 边缘轻量、非聊天式 join-or-not 决策**，**(ii) 中心较大推理/PDDL 出可调用计划**，**(iii) 在真实地理先验 + 程序/平台 episode 上用 participation F1 + makespan + 通信 + 成功率四元组评估**。

---

## C. 数据集：能支撑什么 / 不能支撑什么

| 数据 | 标签实际有什么 | 能支撑的实验 | **不能**支撑 |
|------|----------------|--------------|--------------|
| **SAREnv** (namurproject; Bullock et al. 2025) DOI data.bris；GitHub https://github.com/namurproject/SAREnv | OSM GeoJSON 要素、失踪概率 heatmap.npy、场景元数据；路径评估指标（累积探测概率、时间折扣、发现分） | 环境语义特征、搜索先验、覆盖/路径规划 baseline、多尺度地形 | **参与/角色/联盟标签**；洪水淹没截止；投送成功；通信迹 |
| **本地 AeroRescue 用法**（README，2026-10-14） | SAREnv 子集 env 1/4/6 + **程序生成** UAV/能力/承诺/任务语义；规划器派生 keep/drop（最优 makespan±20% 条件） | 参与过滤管线、候选缩减、规划层 makespan/可行性对照；平台 WAL 导出接口 | 不可当「真实参与真值」发表；3 地点不可做统计外推；中文任务 OOD |
| **BiG-MRTA flood sim** https://github.com/adamslab-ub/BiG-MRTA | 任务坐标、淹没截止、航程/载荷；完成率 vs ILP | 截止约束下分配 makespan/完成率；与匈牙利式匹配对照 | 无 OSM 语义；无角色链；无参与监督；非真实洪水遥感 |
| **FloodNet** Rahnemoonfar et al. IEEE Access **2021** https://arxiv.org/abs/2012.02951 | UAV 高分影像；淹水建筑/道路等分割、分类、VQA | 视觉感知模块、环境语义输入 | 任务分配/参与 |
| **AIFloodSense** IEEE DataPort / Kaggle | 470 航拍、230 事件；分割+分类+VQA | 泛化洪水场景理解 | 同上 |
| **UrbanSARFloods** Zhao et al. **2024** https://arxiv.org/abs/2406.04111 ；https://github.com/jie666-6/UrbanSARFloods | Sentinel-1 强度+相干；开阔/城市淹水像素 | 大范围淹水地图作环境层 | 无人机角色/参与 |
| **Sen1Floods11** 等 | 开阔水域洪水掩膜 | 遥感洪水范围 | MRTA/参与 |

**结论**：公开数据最多支撑 **环境语义 + 搜索先验**；**participation 标签必须程序派生或平台日志标注**（本地已采用前者；后者需真实 run）。

---

## D. Baseline 对比怎么设计

### D.1 方法轴（建议固定 6 类）

| ID | 方法 | 代表 |
|----|------|------|
| B0 | 全候选进中心（无过滤） | 本地 comparison 中的 full candidates |
| B1 | 规则距离/能量阈值 | 距离门控、电量门控（本地已有等数量距离规则对照） |
| B2 | 分布式拍卖/共识 | CBBA（Choi et al. 2009） |
| B3 | 集中匹配 | Hungarian / BiG-MRTA 式最大权匹配 |
| B4 | 联盟博弈 join/leave | MUCFC-CFG（AAMAS 2024）或 ACC 2018 leader-follower |
| B5 | 聊天式 LLM 调度 | SMART-LLM / 纯 LLM 分配（**对照，非主方法**） |
| **Ours-edge** | 学习型轻量参与过滤（Jev / 逻辑回归 / 小模型） | 仅本地状态+通告 → keep/drop/角色 |
| **Ours-center** | 过滤后候选上：有界调度 / PDDL / LLM→PDDL | LLM+P、LaMMA-P、AUSPEX-UP |

### D.2 指标轴（必须同表）

| 指标 | 定义要点 | 谁负责测 |
|------|----------|----------|
| **Participation F1 / Precision / Recall** | 相对规划派生或人工角色标签；未知标签不计 | AeroRescue |
| **候选缩减比** | \|C_filter\|/\|C_full\| | 过滤后 |
| **Makespan / 计划成本** | 同物理资格后规划目标 | 中心规划器 |
| **计划可行性率** | 过滤后仍可解比例 | 规划层 |
| **通信负载** | 消息条数或字节（通告+出价+计划下发） | AeroAgentSim |
| **任务成功率** | 确认 survivors / 投送到位 / 截止内完成 | **仅平台闭环**；勿用程序 episode 冒充 |
| **端到端时延** | 过滤推理 + 中心规划 + 下发 | 分层报，勿只报总时间 |

### D.3 公平条件（写作必写）

1. **同一物理资格门**先于所有方法（能力、航程、busy）。  
2. **同一地图与 episode 种子**；train/val/eval **按地图与模板切开**，禁止同 episode 帧级随机拆。  
3. 规则基线与学习过滤 **保留相同候选数** 时再比 makespan（本地已做）。  
4. LLM 基线固定温度/模型/预算（token 或墙钟）；PDDL 固定求解器与时限。  
5. **分层声明**：规划层指标 ≠ 飞行/ns-3 成功率；未跑平台闭环不得写 success rate 赢。  
6. 参与建议 ≠ 投送许可（本地 README 已强调）。

### D.4 本地已跑事实（勿夸大）

- 192 程序 episode，2304 角色记录；CPU 逻辑回归 273 参；eval 上学习过滤保留 19/20 可解，漏 1 个可行组合；**未显示优于等数量距离规则**；加入端侧后总耗时高于全候选。  
→ 论文可写「管线可复现」，不可写「SOTA」。

---

## E. 中心对齐选项 × 边缘 Jev 如何接

```
任务通告(任务语义+环境摘要)
        │
        ▼
┌───────────────────┐
│ 边缘 Jev：轻量过滤 │  输入：本机状态+通告（无全局真值）
│ keep/drop + 角色建议 │  输出：roles, available_after_s, reason
└─────────┬─────────┘
          │ 仅保留 keep 的候选（降通信/搜索空间）
          ▼
┌───────────────────┐
│ 中心：较大推理和/或 │
│ PDDL/HTN/有界调度  │  输出：可调用计划 → 平台执行运动/消息
└───────────────────┘
```

| 中心选项 | 适用 | 与 Jev 接口 | 风险 |
|----------|------|-------------|------|
| **A. 平台现有任务图 + 有界调度**（现状） | 工程闭环最快 | `participation_filter.json` → propose() | 表达力有限 |
| **B. LLM→PDDL→Fast Downward**（LLM+P / LaMMA-P） | 长时序角色链、自然语言任务 | Jev 输出写成 PDDL objects/init 中的可用 agent 集合 | PDDL 错译；需 domain 专家文件 |
| **C. Unified Planning / HTN**（AUSPEX 路径） | 符号可验证、可换引擎 | 过滤后 agent 集 = problem 参数 | 动态重规划成本 |
| **D. 微调中等推理模型直接出计划** | 语义灵活 | 仅见 Jev 保留子集，降低上下文 | 无正确性保证；贵；**本地 Qwen LoRA 入口未训** |
| **E. ALNS / 拍卖在过滤后子集** | 多点巡视/投送 | 子集上跑 CBBA/匹配 | 仍可能通信重 |

**推荐论文叙事**：边缘 = **learned bounded participation**（非聊天调度）；中心 = **symbolic acceleration（PDDL/HTN）或现有有界调度**，LLM 只做 NL↔符号翻译（LLM-modulo），避免 SMART-LLM 式端到端聊天调度。

---

## F. 可发方向（贡献句）

1. **边缘有界参与决策**：在任务语义×环境语义条件下，用轻量模型做 join-or-not/角色保留，相对 CBBA/全候选/距离规则，在 **participation F1 与通信负载** 上可公平对比（标签由规划派生或平台日志，须明示）。  
2. **过滤–规划协同架构**：将学习过滤与中心 PDDL/有界调度接口化（可调用计划），在 SAREnv 地理先验 + AeroAgentSim 运动/通信上给出 **候选缩减–makespan–可行性** 分层评估，区别于纯联盟博弈或纯 LLM 调度。  
3. **基准协议**：发布「无参与标签地图 + 程序/日志标签协议 + 方法×指标矩阵」，填补 SAREnv/FloodNet 等只服务感知或覆盖、不服务自组织参与的空白。

---

## 关键链接速查

- SAREnv 论文：https://www.mdpi.com/2504-446X/9/9/628  
- SAREnv 代码：https://github.com/namurproject/SAREnv  
- Queralta SAR 综述：https://ieeexplore.ieee.org/document/9220149  
- CBBA：https://ieeexplore.ieee.org/document/5072249  
- AAMAS 2024 联盟：https://arxiv.org/abs/2403.05108  
- ACC 2018 联盟：https://arxiv.org/abs/1711.00214  
- BiG-MRTA：https://github.com/adamslab-ub/BiG-MRTA  
- LLM+P：https://arxiv.org/abs/2304.11477  
- LaMMA-P：https://arxiv.org/abs/2409.20560  
- SMART-LLM：https://arxiv.org/abs/2309.10062  
- AUSPEX：https://www.frontiersin.org/articles/10.3389/frobt.2025.1583479  
- FloodNet：https://arxiv.org/abs/2012.02951  
- UrbanSARFloods：https://arxiv.org/abs/2406.04111  

