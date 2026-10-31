# 代码与数据布局（AeroRescue 工作区自包含版）

## 1. 代码（纳入 Git 版本管理）
- `src/rescue_scenes/`：AeroAgentSim 救援场景引擎插件（entry point `aeroagentsim.engines`）
- `src/aerorescue/`：救援数据导出、评估与训练适配模块
- `tools/`：`make_run.py`、`run_forced_rules.py`、`check_platform_apis.py`、`summarize_run.py`、`register_production.py`
- `configs/`：13 个典型灾害救援场景配置（`SC01` ~ `SC17`、`GT04`）及 `asset_allocation.json`、`rule_inventory.json`
- `tests/`：56 项单元与契约回归测试
- `docs/`：研究计划、文献综述与平台 API 闭环验收报告

## 2. 场景与三维资产（位于 `AeroRescue/scene_data/`，通过 `.gitignore` 排除）
所有原分散在 `blender-data` 与 `rescue-generator-stage1` 的有效资产已统一归拢至本工作区 `scene_data/` 下：
- `scene_data/rescue-data/`：`assets/`（含生产资产与 GLB 链接）、`bundles/SC09-verified-input/`、`display_lod_cache/`、`tools/generated/upstream/`
- `scene_data/upstream/scene-v0.6/`：上游场景生成器
- `scene_data/scene_sim/`：场景 GLB 资产库（`assets/`）、背景几何与布局（`tools/generated/background_roots/`）、目录快照（`dev/step3`、`dev/step4`、`sources/`、`configs/`）
- `scene_data/blockout_geometry_20261023/`：各场景导出网格、GLB、console-scene 与前端运行时资源（`marble_aas_visual_20261024/runtime`）
- `scene_data/ground-templates-20261022-v1/`：GT02 地形模板元数据

## 3. 默认路径（零环境变量即可运行，亦可通过环境变量覆盖）
`src/rescue_scenes/paths.py` 已默认指向工作区内 `scene_data/`：
```bash
RESCUE_DATA_ROOT=/mnt/data2/weizhiwei/aeroagentsim/AeroRescue/scene_data
RESCUE_SCENE_ROOT=/mnt/data2/weizhiwei/aeroagentsim/AeroRescue/scene_data/rescue-data
RESCUE_AAS_ROOT=/mnt/data2/weizhiwei/aeroagentsim/AeroAgentSim
RESCUE_GENERATOR_ROOT=/mnt/data2/weizhiwei/aeroagentsim/AeroRescue/scene_data/upstream/scene-v0.6
RESCUE_LOD_CACHE=/mnt/data2/weizhiwei/aeroagentsim/AeroRescue/scene_data/rescue-data/display_lod_cache
```
