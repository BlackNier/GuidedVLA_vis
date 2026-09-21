# LIBERO-Plus stage 标注数据导出

此增量基于 BlackNier/GuidedVLA_vis 的 `f46c8e334d08e5914e8304d4f585a1792d32a001`。
新增功能只在仿真客户端执行，不要求修改策略服务器或重新训练。

## 使用

保留现有推理命令。状态导出默认开启；建议给单条或批量入口追加：

```bash
--attention-dump-interval 5 \
--stage-plan-path /absolute/path/libero_9_stage_plan_attention_weighted.json
```

例如（先按原流程启动策略服务器）：

```bash
examples/libero_plus/.venv/bin/python examples/libero_plus/main.py \
  --host 127.0.0.1 --port 8000 \
  --task-suite-name libero_10 --task-ids 0 --num-trials-per-task 1 \
  --attention-dump-interval 5 \
  --stage-plan-path /absolute/path/libero_9_stage_plan_attention_weighted.json
```

这里 task-ids 是评估套件的真实 ID，不是语义 stage plan 的 ID；程序保存计划原文和哈希，不擅自推断二者的映射。
计划省略时仍可导出，其哈希为 null。task_count 与 tasks 长度不符只警告，不改动原计划。
`--no-export-stage-data` 可关闭新增导出，恢复原有动作前图像序列。
批量入口 `examples/libero_plus/eval_libero_plus.py` 同样接受以上新参数。
`--attention-dump-interval` 控制 PNG/PT 的抽样间隔，不影响每步状态和完整视频。

## 输出

单条入口默认目录：

```text
data/libero_plus/rollout_images/rollout_<instruction>_epXX_<success|failure>/
  agentview/000000.png ...       # 每 N 帧 + 最后观测
  wristview/000000.png ...       # 同步
  wristview.mp4                 # 完整腕部视角；主视角在原 video_out_path
  annotation/
    states.npz
    observations.jsonl
    actions.jsonl
    policy_calls.jsonl
    frame_map.json
    metadata.json
    scene.xml
    task.bddl
    stage_plan.json             # 提供 --stage-plan-path 时保存
    manifest.json               # 最后写入，带文件哈希和完整性标记
```

批量入口的图像/状态输出放在 `<results_base_dir>/<task_slug>/rollout_images/`，
attention 放在 `<results_base_dir>/<task_slug>/rollout_attention/`，避免不同任务组共用默认导出目录。

## 每个文件的含义

- `states.npz`：数组第一维是 obs_id。含原始 `states`、显式 `qpos/qvel/ctrl/act`、
  所有 body 的 `body_position_world/body_rotation_world`、`obs_id/env_step/sim_time`。
  ndarray 均为数值，可 `np.load(..., allow_pickle=False)`。
- `observations.jsonl`：每个观测一行，末端位姿、实际夹爪开合、对象/fixture 根 body 位姿、
  对象相对夹爪位姿、左右指腹和整个夹爪接触标志、接触对/距离/约束索引/重算接触力。
  `policy_observation_state` 保存模型使用的机器人状态；其 quaternion 为 xyzw。
- `actions.jsonl`：每个成功返回的 env.step 一行，记录原样送给环境的动作、源/目标 obs_id、
  policy_call_id、chunk 内索引及动作后 done。env.step 抛异常时不伪造已完成动作。
- `policy_calls.jsonl`：模型每次调用的输入 obs_id、返回的完整动作块及 replan_steps。
  未实际执行的预测动作仍可在此查到，不计入 actions.jsonl。
- `frame_map.json`：obs_id 与两视角视频帧号、环境步号、仿真时间及抽样 PNG 相对路径对应。
- `metadata.json`：任务/实际 prompt、套件、类别、实际 BDDL 引用和副本哈希、解析后的场景定义、
  seed 和初始化状态索引/哈希、软件与仓库版本、控制频率、图像变换、controller/action_spec、
  物体/碰撞体/关节/body 名称与索引、结果和结束原因、阶段计划哈希。

## 时间和终止约定

- 零起始帧号，`obs_0 --action_0--> obs_1 ... --action_(T-1)--> obs_T`。
- 正常成功和超时均为 T 个动作、T+1 个状态/视频帧；最后观测也强制保存双视角 PNG。
- 视频 10 FPS 是播放速度；物理时间使用 sim_time，控制频率单独保存。
- 等待步不输出；首个 obs_id 为 0，env_step 为已完成的等待步数。
- 最后观测没有新策略推理，其 attention 为 null，不能当成全零 attention。
  action chunk 中间帧仍沿用原代码的 attention 重用逻辑。
- 推理/step 异常保存已有前缀并标 `termination_reason=error`、`complete=false`；
  如果动作成功但随后的采集失败，可能缺少最后状态，该记录也不会被认作完整。
- 初始化或磁盘写入失败会抛出错误，不留下新的完整性标记。
- 已有视频但没有完整 annotation 数据的 rollout 会重新执行，无法从旧视频补回精确仿真接触。
  推荐第一次使用新输出目录，便于保留旧结果作对照。

## 接触与抓持证据

接触和派生位姿在共享只读 MjModel、独立 MjData 中以当前 qpos/qvel 重算，不调用活跃 sim.forward()，
不修改在线仿真的 qpos、求解器缓存、控制器或随机数状态。
这是**观测时刻的几何/动力学重算结果**，不是控制步内部每个物理子步的接触历史。
`force_torque` 是重算值，不是精确记录原积分器最后子步的力。

- `left_finger/right_finger/any_gripper`：存在碰撞候选，可能包含 margin/gap 的近接触。
- 同名 `_active`：至少一个接触有 `efc_address >= 0`，不等价于正压力阈值或稳定抓持。
- `bilateral`：左右指腹均接触同一实例，不能单独作为已夹稳的真值。
- 如果机器人缺少某个 fingerpad group，其对应标志及 bilateral 为 null，不能按 false 使用。
- 物体随夹爪移动且 `position_in_gripper/rotation_in_gripper` 持续稳定，才提供搬运抓持证据。
- 门/抽屉等使用对应关节 qpos 或子 body 位姿；fixture 根 body 可能完全不动。

针对当前 stage 计划，离线先用抽样图定位操作对象，再用持续接近、双侧接触+共同运动、
门/抽屉关节变化等确定候选起始步，最后按 frame_map 提取完整视频邻域复核。
连续数步确认后回溯到事件开始帧，不把“确认时刻”当起始帧。
本增量只导出证据，不自动生成 stage 标签或将接触布尔量冒充稳定抓持标签。

## 读取关节示例

```python
import json
import numpy as np
from pathlib import Path

folder = Path('/path/to/rollout/annotation')
meta = json.loads((folder / 'metadata.json').read_text())
with np.load(folder / 'states.npz', allow_pickle=False) as data:
    joint_name = '替换成 metadata 中的真实关节名'
    j = meta['joint_names'].index(joint_name)
    begin = meta['joint_qposadr'][j]
    end = meta['joint_qposadr'][j + 1] if j + 1 < len(meta['joint_names']) else data['qpos'].shape[1]
    positions = data['qpos'][:, begin:end].copy()
```

世界位置单位 m；hinge 关节为 rad，slide 为 m。旋转矩阵是 local-to-world。
完整 qpos 中自由关节四元数遵循 MuJoCo wxyz，不能与 observation 的 xyzw 混用。

## 限制与依赖

使用仓库现有 MuJoCo 3.2.3 原生绑定和 robosuite 的 `_model/_data` 包装，无新增推理依赖。
当前对象/夹爪注册适配单臂 LIBERO 接口。变更机器人接口或切换到 mujoco-py 后需要适配。
额外每步 mj_forward 和视频输出增加 CPU/存储开销，尚未测量完整 LIBERO rollout 性能。

`states` 不是全仿真 checkpoint。XML/资源、模型扰动、控制器内部状态、随机图像噪声和求解器历史
不全包含在扁平 state 中；不能承诺位级重放或重新生成完全一致的噪声图像。
本数据包已直接保存每步所需运动和接触证据，通常无需重跑仿真判断边界。

## 验证

```bash
python -m pytest -q examples/libero_plus/tests/test_stage_export.py
```

测试使用原生 MuJoCo 小场景验证接触、实例区分、位姿更新和不修改在线状态；
使用策略/编码器替身验证评估循环的 T+1 对齐、终止/超时/异常、动作块映射及完整性检查。
完整 GuidedVLA 模型 + LIBERO-Plus 渲染 rollout 需要在你的 GPU 评估环境中执行。
