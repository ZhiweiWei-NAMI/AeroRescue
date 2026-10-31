# 平台功能/API闭环验收

本次按合法调用、真实效果和原生事件验收平台功能，不采用场景总通过率。机器矩阵：[platform-api-matrix.json](platform-api-matrix.json)。
实测/保留证据覆盖16项逻辑闭环；相机图像推理1项为D（关联证据不足，待验证），未计API失败。新增独立C缺陷1项，已修复；前轮reducer C缺陷1项已修复并复用证据，当前未留开放C。

## 平台功能闭环

| 功能/API | 具体闭环证据 | 状态 |
|---|---|---|
| scripted_perception_region_assessment / A005, A006, A007, A041 | A005原生RGB观测→A006脚本推断→A007本地定位→A041本机认知；三个建筑评估谓词满足，保留原生观测及父证据。 | closed |
| camera_image_inference / configured CameraObservations/perception plugin | 仅复用camera_verified/归档计数；缺完整保留图像—请求—推理返回关联。未启动图像推理，不泛称所有感知模式通过。 | D，非失败 |
| contact / A009 | SC09 seq409、act-0000006：human.support_received []→contacted。 | closed |
| extrication / A017 | A017受困作业量20→0，生成extricated并保留source_seq/write_seqs。SC13既有链保留。 | closed |
| aid / A019 | A019两次实际医疗包数量2→1→0，治疗support/health真实变化；物资质量同步正确。 | closed |
| supply_pickup / A024, A025 | A024实际源站库存减少2单位、物资stock→attached、持有者→actor、物资质量4kg；A025释放空包不留下虚假载荷。 | closed |
| person_loading / A020 | A020载入75kg；轮式在[36,48]被合法拒绝，显式外侧点[36,44]调用成功。 | closed |
| water_recovery / A023 | 原生A023：水中人员→boat承载，flow0.2m/s；原场景缺站点与API闭环分开。 | closed |
| transport / A021 | A021真实on-road地面/水域路线，人员随载具移动并arrived；无air移动覆盖。 | closed |
| handover / A022 | 开放医疗站capacity2/occupied0/confirmation=true，A022接收后occupied1、carrier解除、载荷0；保留真实回执来源。 | closed |
| fire_suppression / A062 | SC07已有fire-unit-1正确绑定，原生A062水6→1.5，火强度→0；未补造水。 | closed |
| hazard_isolation / A063 | SC09 A063两次设置真实非空exclusion_zone，状态写与动作实例可定位。 | closed |
| access_restoration / A012, A013, A016 | 注册映射A012轻清、A013重清、A016吊移：work6→0、blocked→open、净空→7m；A013体积1→0，A016移至[32,46]。A014为破拆、A015为支护，仅纠正注册映射，不扩大补测。 | closed |
| ground_movement / A002 | 原生A002，wheeled+C02，合法road-1 [20,40]→[40,40]，清障后宽7m/step0，无air覆盖。 | closed |
| run_entry_events_replay / existing tools/make_run.py, AAS CLI, RescueWorld, events.jsonl, browser replay | 复用SC09真实CLI/原生日志/Chromium accepted=true及成功JPEG；32×播放0→1645、末尾60:00，39对象、6事件、9资产200。 | closed |
| camera_metrics_protocol / RescueReducer.consume | 前轮reducer限九个实际JSON协议；真实camera无data_json载荷不报错、不计旧协议，legacy rescue.event仍计数；前轮2 passed。 | closed |
| combined_person_cargo_payload / A024, A019, A020, A021, A025, A022 | 先载人和后载人两种顺序均通过；总载荷始终等于剩余item质量+75kg在载人员；交接后0。 | closed |

每行合法能力/范围/库存/容量、实际输入、前后字段、原生seq、动作实例和来源链接均在机器矩阵；fixture详细证据为 [platform-api-fixture-after.json](platform-api-fixture-after.json)。各fixture只声明所需能力，初始化在t=0且首个动作前完成；运行中仅动作模型产生状态变化。
水上fixture显式初始提供水域[4,10]～[30,30]、岸边接收站[20,32]、水中泊位[20,28]、容量2和接收确认；boat保持water机动，不是air定位。

### C类单点修复与复验

独立ID **C-D03-PERSON-MASS**：原D03重算platform载荷只计items。先A020载75kg人员、A024取4kg药品、A019耗2kg后，应77kg，实测2kg；后续应75kg而实测0kg。此前A019无独立payload写的单项疑点被统一D03同步反证，未对A019/A021逐个打补丁。
只在 `rescue_scenes/core/world/truth.py::_derive_inventory` 的platform分支加入实际aboard人员质量。修复后，运输、耗药、卸载、移交各边界质量一致。共用该缺陷ID的多行不重复计缺陷。补丁：[platform-api-d03-person-mass.patch](platform-api-d03-person-mass.patch)。
失败：[platform-api-loaded-first-before.json](platform-api-loaded-first-before.json)；成功：[platform-api-fixture-after.json](platform-api-fixture-after.json)。相关回归 **20 passed in2.23s，exit0**，覆盖新增fixture、原生救援事件链和现有规则测试；未重跑全场景、完整SC09或49项旧全量基线。
前轮独立ID **C-METRICS-CAMERA-SCHEMA** 已修复；本轮仅引用既有patch/真实请求fixture测试和浏览器证据，不重复修复或重启服务。

## 场景/平台配置缺口（B，分离于API闭环）

- SC05缺接收站；本轮合法隔离水域fixture的A023→A021→A022闭环，SC05原配置保持不变。
- SC01/GT04已有药品、接收空位和承载容量不足的原生拒绝，属于B。fixture明确配置合法药品、200kg载荷和接收空位，未补造原场景资源。
- SC07首轮未绑定已有消防装备属于B；正确绑定后有限水消耗/火强度变化已被原事件证明。
- SC09原规则记录的C24合法能力配置补充属于B；新fixture只在初始actor中声明必要能力，没有给所有动作授所有能力。
13份原配置字节SHA全部未变：[platform-api-input-verification.json](platform-api-input-verification.json)，对照主控 [platform-api-input-baseline.json](platform-api-input-baseline.json)。

## 可忽略的策略/调度问题（A）

SC01失败A020实际actor为ground_vehicle-1，规则A001将它送到人员中心，恢复wheeled机动后调用A020触发“须先到建筑外侧入口”分支。该分支要求既有pickup点且actor距其>2m，不是合法前置下拒绝。独立轮式fixture在内部[36,48]正确拒绝、在真实外侧[36,44]成功：[platform-api-wheel-loading.json](platform-api-wheel-loading.json)。将这次未按入口定位归A，不归API故障。
未发动作、错误顺序、优先级/执行者争用不用于本次API失败判定；本轮未优化自主策略。原地面寻路的旧air覆盖不作为A002证据，A002已另用合法道路补测。

## D、服务及证据范围

相机图像推理只保留历史压缩计数/链摘要，缺完整图像—请求—推理—观测关联，因此D，不判失败。本轮未运行相机推理或GPU。脚本感知、历史真实浏览器截图和相机推理分层报告。
初始主控只读观察18931未监听；随后共享会话/主控恢复服务，最新页面与SC09 header200，见 [platform-api-service-restoration.json](platform-api-service-restoration.json)。该恢复不是本执行者操作；本执行者没有启动、停止、重启服务，也未重做浏览器已通过项。

## 真实命令与保留

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/data2/weizhiwei/aeroagentsim/wt/delivery-20261028/src:/mnt/data2/weizhiwei/aeroagentsim/wt/delivery-20261028/aerokernel:site:. \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/mnt/data2/weizhiwei/aeroagentsim/AeroAgentSim/.venv/bin/python tools/check_platform_apis.py \
  --output /mnt/data2/weizhiwei/blender-data/.bridge/outputs/task_87e8c4faa74443bc802e1777c49-dc0aa11c77bde995/platform-api-fixture-after.json
```

相关回归使用同一指定Python/PYTHONPATH，执行 `-m pytest tests/test_platform_api_closure.py tests/test_rescue_chain_events.py tests/test_forced_rules.py -q -p no:cacheprovider --basetemp=/mnt/data2/weizhiwei/blender-data/.bridge/outputs/task_87e8c4faa74443bc802e1777c49-dc0aa11c77bde995/pytest-platform-related`。日志：[platform-related-tests.log](platform-related-tests.log)。
首次fixture工具误用sim._started阻止t=0初始化，改为t=0且无动作后实跑成功；这项工具初始化失败已留小日志，不计平台C。
本任务before/loaded-first-before等大快照已在核对绝对目录归属后缩为必要输入/前后载荷/原生seq/动作来源，移除冗余final_truth/local_belief及无关载荷：[platform-api-diagnostic-compaction.json](platform-api-diagnostic-compaction.json)。原有效scene_json、旧成功回放/截图、其他任务文件未动。所有本任务小运行已退出；本次验收矩阵可审阅，停止扩展。
