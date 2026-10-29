# 代码/数据布局与运行方式（迁移后）

## 代码（仓库）
- AeroRescue：`src/rescue_scenes/` 引擎插件（entry point `aeroagentsim.engines`）
- AeroRescue：`tools/`（make_run、run_forced_rules、check_platform_apis、summarize_run）、`tests/`、`configs/`、`docs/`
- AeroAgentSim：平台本体（仅修复 RescueReducer camera 载荷缺陷）

## 数据（不進 Git，位于 blender-data）
`rescue-backgrounds-18-20261023/rescue-data/`
- `assets/` 资产目录 + `glb` 符号链接（真实 GLB 在 `scene_sim/assets/glb`）
- `display_lod_cache/` 显示 LOD 缓存
- `bundles/SC09-verified-input/` 可复现输入
- `tools/generated/upstream/SC09-1/` 上游场景产物
- `scene_sim-delivery-history.bundle` 旧仓库完整历史
- 场景源/上游生成器：`scene_sim/`、`upstream/scene-v0.6/`

## 环境变量
```
RESCUE_SCENE_ROOT=<project>/rescue-data
RESCUE_DATA_ROOT=<project>
RESCUE_AAS_ROOT=<AeroAgentSim>
RESCUE_GENERATOR_ROOT=<project>/upstream/scene-v0.6
RESCUE_LOD_CACHE=<project>/rescue-data/display_lod_cache
```

## 验证（2026-10-29）
- entry point 发现：rescue_scenes -> rescue_scenes.engine:build
- pytest：56 passed
- check_platform_apis：6 组原生 API 全部 succeeded，载荷一致
- 端到端 SC09：5/5 目标、16 动作全部成功（SC09_rescue.scenario-8dd8371cd60c）
