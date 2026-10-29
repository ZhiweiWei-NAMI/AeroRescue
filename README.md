# AeroRescue

救援研究的**数据 + 训练 + adapter**仓库。仿真、运动执行和通信留在 [AeroAgentSim](https://github.com/ZhiweiWei-NAMI/AeroAgentSim)；本仓库不复制仿真器或规划器内核。

## Architecture

- **Edge**: Jev-style lightweight participation filter — task + environment semantics; decide **join or not** (keep/drop). Not an edge LLM that writes roles.
- **Center**: larger reasoning and/or PDDL-style planning, then publish **callable plans**.
- **Sim / motion / comms**: stay in AeroAgentSim. This repo is **data + training + adapter only**.

## 目前能跑什么

官方 SAREnv 小子集 → 来源核验与地图摘要 → 明确标为程序生成的救援任务扩展 → 调用 AeroAgentSim 的有界规划器生成角色标签 → 实际 CPU 优化 → 保存并重新加载参与筛选模型 → 相同问题上的成对规划比较。另有真实平台日志导出和可选 Qwen3.5-0.8B LoRA 入口。

CPU 模型是哈希文本与结构化特征的逻辑回归，只有 273 个参数；不是 Qwen 微调、语言推理模型，也不证明机载部署效果。Qwen 入口尚未训练；当前环境没有已验证的 GPU、PyTorch 或模型权重。中心仍使用平台原有的模型任务图与有界调度器，没有声称中心模型已微调。

## 运行

Python 3.11+，必需依赖只有 NumPy。与 AeroAgentSim 并列放置：

```bash
python -m pip install -e .
aerorescue acquire
aerorescue prepare
OPENBLAS_NUM_THREADS=1 aerorescue run --platform ../aeroagentsim-work
```

本地重建 CPU 原型（不要把大型 dumps / journals / secrets 提交进 Git）：

```bash
OPENBLAS_NUM_THREADS=1 aerorescue run --platform ../AeroAgentSim \
  --output runs/cpu-prototype --episodes 64
```

`runs/cpu-prototype/` 会写出 episodes.jsonl、roles.jsonl、participation-filter.json、training-summary.json 和 comparison.json。原始 SAREnv 子集、运行结果、模型权重、平台 journals/WAL 和密钥都不进入 Git（见 `.gitignore`）。文献调研见 `docs/AeroRescue-literature-survey-2026-10-14.md`。全流程使用实际程序运行，不包含测试套件。

## 接回平台

安装本包到平台 Python 环境，或把本仓库 `src` 加入平台进程的 PYTHONPATH。在救援场景的 `engines.tasks.config.rescue` 下设置：

```yaml
participation_filter: /absolute/path/AeroRescue/runs/cpu-prototype/participation-filter.json
```

平台每个独立 UAV harness 加载 `ParticipationFilter.load(path)`，调用 `propose(local_request)`，得到 roles、available_after_s、reason。中心任务图和真实运动/消息执行仍由平台负责。去掉该字段恢复平台模型 API。参与建议不是投送许可；投送仍须收到 positive 报告。CPU 模型主要在英文程序模板训练，平台中文任务属于分布外输入，必须通过实际运行观察，不能直接宣称泛化成功。

## 数据与标签边界

- 下载源：[官方 SAREnv](https://github.com/namurproject/SAREnv)，固定 commit `0f2d344d17c0b9c18cec2176dd9b171beba727d6`。只下载环境 1、4、6 的 GeoJSON 和 NPY，加 LICENSE、README，共 32,691,150 bytes；不下载全库或大型档案。
- `data/raw/sarenv/source.json` 保存每个来源 URL、SHA256、文件大小与获取时间；GeoJSON 解析 Git LFS pointer 后逐个核对 SHA256。代码仓库许可 MIT；底层地图保留 [OpenStreetMap contributors / ODbL attribution](https://www.openstreetmap.org/copyright)。上游 LICENSE 与 README 原样随数据保存。
- GeoJSON 是实际地点的地图要素，概率图是 SAREnv 生成的搜索先验。它们不提供真实洪水、受困者、人员需求、角色参与或协同计划标签。生成器从概率图选取有来源的坐标；无人机、能力、已有承诺和任务语义均是另加的程序情景。没有把任意真实图像贴到模拟地点。
- 标签使用完整候选规划、强制包含某个机—角色配对、移除配对三种求解。能参与某个不超过最优 makespan 20% 的计划才是正样本；没有被最终最优计划选中不等于负样本。无解的完整问题和当前无法确定的角色保持 null。suspected＋投送情景的规划标签对应“之后确认为 positive”的条件分支，仅表明应保留潜在投送者，不预测观察结果，也不授权提前投送。
- 模型输入只有本机状态和任务通告。标签、其他 UAV 状态、计划结果和隐藏真值不进入模型输入。未来物理占用可由研究规划器排程；平台当前阶段只允许立即可派遣 UAV，两者必须区分。
- train/validation/evaluation 使用不同地图、不同语言模板和不同能力组合；同一 episode 不拆帧随机分割。阈值仅由 validation 选择。这里只有 3 个来源地点，不能据此做统计结论。

## 实际运行记录（2026-10-14）

192 个程序 episode，2,304 个角色记录，560 个未知标签。CPU 优化执行 300 步，加权训练损失 0.6931 → 0.4027；参与阈值 0.35。

32 个 evaluation 问题中，全候选可解 20 个；学习筛选保留 19 个。各方法先应用相同物理资格条件，再比较候选数量：全候选 163 → 学习筛选 136。相同保留数量的距离规则也保留 19 个。学习筛选漏掉一个可行组合，且加入端侧推理后总耗时高于全候选。这是数据/训练/载入管线的可运行样例，未显示方法优于基线。详细数据来自运行文件，不是编造分数。

上述 comparison 是规划层的计划可行性与 makespan，没有把它称为真实飞行闭环、ns-3 结果或任务成功率。真实平台模型 API / 通信运行和截图由 AeroAgentSim 交付；本仓库的 adapter 已接入，仍需要该实际平台运行。

## 导出真实平台 episode

可直接读取平台 run 目录的已提交 WAL，使用原生 codec 解码并按 message_id 去重；也接受按实际事件顺序保存的 JSONL，每行 `{schema, payload: {data_json: "..."}}`。需要 rescue.local.request 和带完整 problem/candidate_universe 的 rescue.event/plan_computed；不猜测遗漏的物理状态。

```bash
aerorescue export --run ../aeroagentsim-work/runs/actual-run-id --platform ../aeroagentsim-work \
  --run-id actual-run-id --map-id rooftop-flood --split train --output data/platform/run-id
```

保留原始本机请求、真实规划输入、实际最优计划和候选全集。不同时间的旧局部请求不会被后来物理状态硬打标签，此类记录输出 null；初始 inspect/relay 阶段缺少 deliver 也不自动标负。真实日志不伪装成程序生成数据，程序数据不冒充真实平台运行。端侧训练要求分别取得 train、validation 数据；单次演示不够支持模型结论。

## 可选 Qwen LoRA

```bash
python -m pip install -e '.[peft]'
python -m aerorescue.peft_train --data runs/cpu-prototype/roles.jsonl \
  --output artifacts/qwen_lora --allow-download
```

默认使用官方 [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) 的内置 Qwen3_5ForConditionalGeneration，只训练语言 decoder 的 LoRA 参数，不训练视觉塔。监督目标为 keep/drop，只对回答 token 计算损失；未观察标签跳过，evaluation 不参与训练。默认要求 CUDA，CPU 必须显式 `--allow-cpu`；默认只读本地缓存，联网下载须显式 `--allow-download`。禁用远端代码与凭据读取。此入口目前仅完成语法/帮助命令核对，未做权重训练或推理验证。实际训练后，平台 participation_filter 可指向 adapter 目录；默认阈值 0.5 尚未校准，推理机器也须有 CUDA 和本地基础权重，同一进程共享只读模型。

该入口及模型选择参考 [Transformers Qwen3.5](https://github.com/huggingface/transformers/tree/v5.3.0/src/transformers/models/qwen3_5) 和 [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora)。未把未运行的可选代码算作已验证训练结果。

## AeroAgentSim 引擎插件：rescue_scenes

本仓库同时提供 `rescue_scenes` 引擎插件（entry point `aeroagentsim.engines`），安装后平台按
`engines.rescue.plugin: rescue_scenes` 直接加载，不复制平台内核。

```bash
pip install -e .        # 在 AeroAgentSim 的 venv 中
```

外部数据根（场景输入/素材/缓存）通过环境变量指向，代码与数据分离：

```bash
export RESCUE_SCENE_ROOT=/path/rescue-data
export RESCUE_DATA_ROOT=/path/project
export RESCUE_AAS_ROOT=/path/AeroAgentSim
export RESCUE_GENERATOR_ROOT=/path/project/upstream/scene-v0.6
export RESCUE_LOD_CACHE=/path/rescue-data/display_lod_cache
python tools/make_run.py SC09 --config configs/SC09_upstream_allocated.json --forced-rules --scripted-sensing
```
