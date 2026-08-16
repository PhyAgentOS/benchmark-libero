# LIBERO Benchmark 节点

把 LIBERO 包一层的 Dora 节点。进程内持有 LIBERO env，只做测评：批次展开 → 逐 episode 跑 → 结果落盘。

```
benchmark 节点（本节点）                     policy 节点（不改）
├─ 进程内 import libero，直接持有 env
├─ proprio_state / image/main / image/second ──观测──▶ 缓存 + tick 推理
├─ action ◀────────────────────────────────────动作──  JointCommand 7D
├─ policy_command ─────────────────────────────命令──▶ start / set_instruction / reset_scene / stop
├─ 判定 env.env._check_success()
└─ benchmark_status（流式）/ benchmark_result（批次末）
```

不经过 TargetWS、不经过 mujoco_sim。判定、动作、重置全走 LIBERO 官方接口。
设计规范见《Benchmark 接入手册》v14（PhyAgentOS 文档库），本节点是它的参考实现。

**已验证**（2026-08-16，完整 Forge 两节点）：

| 策略 | suite | 结果 | 耗时 | 平均延迟 |
|:---|:---|:---|---:|---:|
| VLA-JEPA | libero_spatial | 4/4 | 20.0 s | 16.3 ms |
| LingBot-VA | libero_10 | 2/2 | 128.7 s | 221 ms |

两个策略结构完全不同（VLA-JEPA 有 `observation.state`，LingBot-VA 没有；chunk 7 vs 16；延迟差 13 倍），都跑通。
更大样本（离线闭环，同一套 adapter）：VLA-JEPA 在 libero_spatial 100 episodes = **94%**（渲染 256）/ **96%**（渲染 224），对照 OpenVLA 的 84.7%。

---

## 1. 环境

### 1.1 占位符约定

本文用以下占位符指代与部署机相关的路径，落地时替换成你自己的：

| 占位符 | 含义 |
|:---|:---|
| `$BENCH_PY` | 评测用 python 解释器的**绝对路径**（该环境须同时具备 libero / dora / forge_msgs / forge_policy / lerobot / torch+CUDA） |
| `$DORA_BIN` | `dora` 命令行所在目录 |
| `$FORGE_RUNTIME` | 本仓库（forge_runtime）根目录 |
| `$LEROBOT_RUNNER` | lerobot_runner 仓库根目录 |
| `$MODELS` | 策略权重目录 |
| `$WORK` | 存放本次评测三个配置文件的工作目录 |
| `$PKG_LINK` | 给 policy 节点用的包软链目录（见 §2.2） |

### 1.2 两个环境的分工

**一个环境未必同时具备节点依赖和 `dora` 命令行**，实测常见的情况是分在两处：

| 角色 | 要求 |
|:---|:---|
| **节点运行环境**（`$BENCH_PY`） | 两个节点的 python 都用它。须有 libero / dora(python 模块) / forge_msgs / forge_policy / lerobot / torch+CUDA |
| **`dora` CLI**（`$DORA_BIN`） | 只用来拉起 dataflow。**两边 dora 必须同版本**（如同为 0.4.1 / message 0.7.0），否则注册节点时报 version mismatch |

```bash
export PATH=$DORA_BIN:$PATH     # 拉起用它
# 节点里写 $BENCH_PY 的绝对路径（见 dataflow）
```

自检：

```bash
$BENCH_PY -c "import dora, forge_msgs, forge_policy, libero.libero, lerobot; print('ok')"
dora --version
```

依赖版本要求：

| 包 | 版本 | 说明 |
|:---|:---|:---|
| python | 3.12 | 不要用 3.8，理由见 §7 坑 7 |
| dora-rs / dora-rs-cli | **两者必须同版本** | 混版本会在注册节点时报 version mismatch |
| forge-msgs / forge-policy | — | `forge_policy` 提供 policy 节点的事件循环，处理 `policy_command` |
| robosuite / mujoco | 1.4.0 + 3.x | robosuite 的三参数 `mj_fullM` 只有 mujoco 3.x 认 |

**策略权重建议全部本地化**到 `$MODELS`，并导 `HF_HUB_OFFLINE=1`，避免运行时依赖网络。

---

## 2. 整体配置：三个文件，一张对应表

跑一次评测要三个文件，**它们之间有字段必须逐字对齐，对不上不报错、只掉分**：

```
bench.yaml          benchmark 节点配置（本节点的，见 config.example.yaml）
policy_xxx.yaml     policy 节点配置（lerobot_runner 的）
dataflow.yaml       dora 拓扑：两个节点怎么连
```

### 2.1 必须对齐的字段

| 概念 | bench.yaml | policy_xxx.yaml | dataflow.yaml |
|:---|:---|:---|:---|
| **策略标识** | `policy_id` | `policy.policy_id` | — |
| **相机通道** | （硬编码在 adapter，见 §9） | `image_inputs` 的**左键** | `inputs:` 里的通道名 |
| **相机→模型特征** | — | `image_inputs` 的**右值** | — |
| **proprio 关节名** | `config.py` 的 `STATE_JOINTS` | `state_joints` | — |
| **动作关节** | — | `joints`（7 个） | — |
| **渲染分辨率** | `camera_height/width` | （policy 内部自己 resize） | — |
| **语言指令** | `send_instruction: true` | `policy.instruction`（会被逐 episode 覆盖） | — |

`policy_id` 对不上 → policy 忽略我们所有命令，整批用初始指令跑完。
`image_inputs` 左键对不上通道名 → policy 收不到图。
`STATE_JOINTS` 与 `state_joints` 对不上 → `JointState` 按名字查表**静默取到空值**。

### 2.2 VLA-JEPA 的完整可用配置

`bench.yaml`：

```yaml
suite: libero_spatial
task_ids: "0-1"                # all / "0-9" / "0,2-4,8" / [0,1,2]；留空=全集
init_state_ids: "0-1"
num_runs: 1
max_steps: 300
num_steps_wait: 10
control_mode: relative
seed: 0
layout_mode: per_episode       # 见 §7 坑 1，换这个会改变分数
camera_height: 256
camera_width: 256
policy_id: vla_jepa_libero     # ← 必须与 policy 配置一致
send_instruction: true
result_dir: $WORK/results
record_dir: null
```

`policy_vla_jepa.yaml`：

```yaml
joints:                         # 动作 7 维
  - {name: joint1, mode: position, unit: radians}
  # ... joint2..joint7 同理
state_joints: [joint1, joint2, joint3, joint4, joint5, joint6, joint7, gripper]   # proprio 8 维

policy:
  type: vla_jepa
  policy_id: vla_jepa_libero    # ← 必须与 bench.yaml 一致
  auto_start: false             # 由 benchmark 发 start
  instruction: "..."            # 占位，会被逐 episode 覆盖
  device: cuda
  enable_world_model: false     # 推理不需要 V-JEPA2，省显存
  pretrained_path: $MODELS/vla_jepa_libero

image_inputs:
  image/main: image             # 左=dora 通道名，右=checkpoint 特征名去掉 observation.images. 前缀
  image/second: image2
```

`dataflow.yaml`：

```yaml
nodes:
  - id: libero_benchmark
    path: $BENCH_PY
    args: -u $FORGE_RUNTIME/packages/nodes/benchmarks/libero/main.py
    env:
      BENCHMARK_CONFIG: $WORK/bench.yaml
      MUJOCO_GL: egl
      PYOPENGL_PLATFORM: egl
      CUDA_VISIBLE_DEVICES: "5"
    inputs:
      action: policy/action
    outputs: [proprio_state, image/main, image/second, policy_command, benchmark_status, benchmark_result]

  - id: policy
    path: $BENCH_PY
    args: -u -m lerobot_inference infer --config $WORK/policy_vla_jepa.yaml
    env:
      PYTHONPATH: $PKG_LINK          # lerobot_inference 包（见下）
      HF_HUB_OFFLINE: "1"
      CUDA_VISIBLE_DEVICES: "5"
    inputs:
      tick: dora/timer/millis/20
      policy_command:
        source: libero_benchmark/policy_command
        queue_size: 32                # 必须显式加深，见 §7 坑 4
      proprio_state: libero_benchmark/proprio_state
      image/main: libero_benchmark/image/main
      image/second: libero_benchmark/image/second
    outputs: [action, policy_command_status]
```

**`PYTHONPATH=$PKG_LINK` 是必需的**：policy 节点的包名是 `lerobot_inference`，但仓库目录叫 `lerobot_runner`，要软链一次：

```bash
mkdir -p $PKG_LINK && ln -sfn $LEROBOT_RUNNER $PKG_LINK/lerobot_inference
```

**dora 的 `path` 必须是绝对路径**，`~` 不展开；若 home 是软链，用 `readlink -f` 取真实路径。

### 2.3 LingBot-VA 的差异

只有三处不同，其余照抄：

```yaml
policy:
  type: lingbot_va
  policy_id: lingbot_va_libero
  pretrained_path: $MODELS/lingbot_va_libero_long
  # checkpoint 里的 wan_pretrained_path 指向 HF repo（本地缓存没有），必须覆盖：
  wan_pretrained_path: $MODELS/lingbot_va_base_frozen
# 无 state_joints —— LingBot-VA 是纯视频-动作模型，checkpoint 没有 observation.state
```

bench.yaml 里把 `suite` 改成 `libero_10`、`policy_id` 改成 `lingbot_va_libero`。

---

## 3. 怎么跑

### 3.1 stub 冒烟（验时序，不需要 GPU/策略）

stub 是个假 policy，只回空动作。**`success_rate` 恒为 0 是预期行为。**

```bash
export PATH=$DORA_BIN:$PATH
cd $FORGE_RUNTIME/packages/nodes/benchmarks/libero
dora run dataflow_stub.yaml          # stub 不需要 GPU 与策略权重
```

**要确认的四件事**（不是看跑完就算过）：

1. 每个 episode 的 `set_instruction` 内容**不一样**
2. stub 退出时打印的 `obs == act`（1:1 锁步）
3. `t/i/r` 索引与配置里的批次一致
4. 结果 JSON 落盘且 `episodes[]` 条数 == `total_episodes`

### 3.2 真策略两节点（验口径，出真分数）

```bash
export PATH=$DORA_BIN:$PATH
cd $WORK             # 三个配置文件所在目录
dora run dataflow.yaml
```

`$WORK` 下需要三个文件：`bench.yaml` + `policy_<策略>.yaml` + `dataflow.yaml`，内容见 §2.2 / §2.3。

**第一次接新 checkpoint 必须先小规模对基线**（2 task × 2 init-state），拿 `success_rate` 对已发表数字。stub 验的是时序，真策略验的是口径，两件事不能合并。

### 3.3 接一个新 checkpoint 前必核三项

错了都不报错，只会安静地掉分：

| 项 | 怎么核 |
|:---|:---|
| **proprio 语义** | **读 checkpoint 的归一化统计量**，别信配置里的文字描述。`observation.state` 的 min/mean/max 逐维量纲直接暴露它是关节角还是末端位姿 |
| **相机别名** | policy 配置 `image_inputs` 左键要能对上我们的输出通道名 |
| **分辨率** | 读 checkpoint `config.json`：有 `resize_images_to` 就喂原生分辨率（policy 自己缩）；没有才需要自己对齐 `input_features` |

> 实例：VLA-JEPA 的 policy YAML 注释写 "observation.state 为 8 维（7 关节角 + gripper）"，**是错的**。归一化统计量显示它是 `eef_pos(3) + axis_angle(3) + gripper_qpos(2)`——第 3 维均值 0.79（LIBERO 桌面高度，米），第 4 维均值 2.98 ≈ π（夹爪朝下的 axis-angle），第 7/8 维完美对称正负小量（两根手指）。

---

## 4. 代码清单

```
main.py                      批次调度核心（通用，不认识 LIBERO）
config.py                    配置模型 + 关节名常量 + 校验 + 种子派生
result.py                    BenchmarkExecutionResultV1 组装与落盘
adapters/libero_adapter.py   ★ LIBERO 特有的全部逻辑
adapters/__init__.py         空
config.example.yaml          配置模板（全字段带注释）
dataflow.example.yaml        真策略 dataflow 模板
dataflow_stub.yaml           stub 冒烟用（gitignored）
stub_policy.py               假 policy（gitignored）
pyproject.toml               包声明；LIBERO/robosuite 刻意不写进依赖，由评测环境提供
```

### 4.1 `main.py` — 批次调度核心

**不认识 LIBERO**，换 benchmark 基本不动。

| 单元 | 职责 |
|:---|:---|
| `BenchmarkRunner` | 持有批次计划与当前 episode 状态 |
| `.start_next_episode()` | 取下一个 `(task, init_state, run)`；派生 episode 种子；发 `set_instruction` + `reset_scene`；调 adapter 重置；发首帧观测 |
| `.on_action()` | 收到动作 → 校验有限性 → `env.step` → **每步判定** → 未结束就发下一帧观测 |
| `.advance()` | 推进到下一 episode；**任何异常都收敛成"批次失败"**，绝不冲出事件循环 |
| `._record_episode()` | 只记录，不推进（推进由调用方显式 `advance()`） |
| `.fail_current()` | 异常路径的唯一入口：记为 failed 并推进 |
| `.finish()` | 批次收尾。**落盘优先**于发消息和关 env |
| `main()` | 加载配置 → 建 adapter → **fail fast 校验** → 建 Node → 事件循环，`finally` 里保证 `finish()` |

时序契约：**一次观测换一个动作**。发布观测后置 `awaiting_action=True`，收到 action 才 step。`awaiting_action=False` 时到达的多余 action **直接丢弃**（policy tick 通常快于我们步进）。action chunk 由 policy 内部排队，benchmark 不感知。

### 4.2 `config.py` — 配置与校验

| 单元 | 职责 |
|:---|:---|
| `STATE_JOINTS` | proprio 的关节名列表，**必须与 policy 的 `state_joints` 逐字一致** |
| `STATE_SEMANTICS` | 8 维的语义标注（eef_pos/axis_angle/gripper），与 `STATE_JOINTS` 逐位对应，仅作文档 |
| `parse_id_list()` | 解析 `all` / `"0-9"` / `"0,2-4,8"` / `[0,1,2]` / `None` |
| `derive_episode_seed()` | 把 `(seed, task, init, run)` 映射成稳定种子。**刻意不用内建 `hash()`**（对 str 带 `PYTHONHASHSEED` 随机化，跨进程不一致，正好毁掉复现性） |
| `BenchmarkNodeConfig` | 配置数据类 |
| `load_config()` | 读 YAML + 校验 `layout_mode` / `num_runs` / `max_steps` |
| `resolve_and_validate_plan()` | 回填 `all`，**逐 task** 校验 id 范围。不同 task 的 init state 数未必相同，不能拿 `task_ids[0]` 代表全体 |

### 4.3 `adapters/libero_adapter.py` — LIBERO 特有逻辑

**换 benchmark 只改这一个文件。**

| 单元 | 职责 |
|:---|:---|
| `num_tasks` / `num_init_states()` / `task_language()` | 批次展开需要的元信息 |
| `_ensure_env()` | **同 task 复用 env，换 task 才重建**（建 env 约 2 s） |
| `_sim()` | 沿包装链找到真正持有 `sim` 的那一层 |
| `capture_layout()` / `restore_layout()` | 家具布局的快照与恢复（`body_pos` + `body_quat`），见 §7 坑 1 |
| `reset_episode()` | 按 `layout_mode` 处理布局 → `env.reset()` → `set_init_state()` → 预热 → **设 `use_delta`** |
| `step()` | 送 7 维动作，返回 `StepOutcome(obs, reward, done, success)` |
| `check_success()` | **逐层兜底**找 `_check_success`（在 `env.env` 上，不在 `OffScreenRenderEnv` 上） |
| `to_state()` | 8D proprio：`eef_pos(3) + axis_angle(3) + gripper_qpos(2)` |
| `to_images()` | 取两路图并 **`[::-1,::-1]` 翻转** |
| `quat_to_axisangle()` | robosuite 四元数是 `(x,y,z,w)` 顺序 |

### 4.4 `result.py` — 结果组装

| 单元 | 职责 |
|:---|:---|
| `EpisodeRecord` | 单 episode 记录，含 `episode_seed`、`success_source` |
| `BenchmarkResult` | 批次聚合。`success_rate` **分母用计划总数**（中途取消时不虚高）；`mean_policy_latency_ms` 用 `is not None` 过滤（不能用真值判断，会丢掉恰好 0.0 的） |
| `.dump()` | 落 `<result_dir>/<suite>_<时间戳>.json` |

---

## 5. 结果

落在 `result_dir`，schema 为 `BenchmarkExecutionResultV1`。

批次级：

| 字段 | 说明 |
|:---|:---|
| `status` | `succeeded` / `failed` / `cancelled` |
| `successes` / `total_episodes` / `success_rate` | **分母是计划总数**，不是已完成数 |
| `completed_episodes` / `valid_episodes` | 已完成 / 无 error 的 |
| `timed_out_episodes` | 超时截断数 |
| `layout_mode` / `seed` | **复现口径**，缺了无法复现场景 |
| `num_steps` / `mean_policy_latency_ms` / `elapsed_s` | 步数 / 平均推理延迟 / 总耗时 |

逐 episode（`episodes[]`）：`episode_id`、`task_id`、`init_state_id`、`run_index`、`task_description`、`success`、`termination`（`success`/`failed`/`timed_out`）、`num_steps`、`return_value`、`episode_seed`、`mean_policy_latency_ms`、`success_source`、`error_code`/`error_message`。

`first_attempt_*` / `final_*` / `assisted_*` / `episodes_replanned` / `recovered_after_replan` 一律 `null`——本节点不做恢复重试，留给上层填。

批次中途 Ctrl-C 或 dora STOP：`status=cancelled`，**已完成的 episode 照常落盘**。

---

## 6. 排错

| 症状 | 原因 |
|:---|:---|
| 跑起来没反应、卡住不动 | policy 没进 running（`policy_id` 对不上，或没收到 `start`）；也可能 policy 卡死——本节点**没有超时看门狗**，会静默悬停 |
| 首个 episode 用了上一句指令 | dataflow 里 `policy_command` 的 `queue_size` 没加深 |
| 分数异常低但不报错 | 按顺序查：proprio 语义 → 图像朝向 → 相机别名 → 分辨率 → `control_mode` |
| `AttributeError: _check_success` | 判定在 `env.env` 上，不在 `OffScreenRenderEnv` 上 |
| `AttributeError: 'BenchmarkNodeConfig' object has no attribute ...` | 改过 `config.py` 但漏了字段。**改配置模型后必须跑一次真 dataflow**，`py_compile` 查不出来 |
| `TypeError: mj_fullM` | mujoco 是 2.3.7，需要 3.x（`conda create --clone` 会让版本号说谎，认功能别认版本号） |
| `os.path.join(None, ...)` TypeError | `CONDA_PREFIX` 没设，adapter 已用 `sys.prefix` 兜底 |
| `Could not initiate node from environment variable` | 对端节点先崩了，看它的 traceback，不是本节点的问题 |
| version mismatch | `dora-rs` 与 `dora-rs-cli` 版本不一致 |
| 同配置两次跑分不同 | 检查 `layout_mode` 是否为 `legacy` |
| `ProcessorMigrationError` | checkpoint 是旧版归一化格式，跑 lerobot 的 `migrate_policy_normalization.py` |

---

## 7. 改代码前必读的七个坑

都是实测踩出来的，**每一条都不报错**，症状是分数安静地烂掉。

1. **`set_init_state` 恢复不了家具**。无关节的 fixture（炉子、柜体）位姿存在 `model.body_pos` / `model.body_quat` 里，属于**模型**；`set_init_state()` 只恢复 `qpos/qvel`，属于**状态**。`env.reset()` 每次重新采样家具摆放，恢复不到。实测同一 `(task, init_state)` 在不同 reset 次序下家具漂 **1.5 cm**、渲染图 **10% 像素不同**，而 `qpos+qvel`、相机、光照全部逐位相同——只看状态发现不了。
   由 `layout_mode` 控制：`per_episode`（默认，布局 = `(seed,task,init,run)` 的纯函数）/ `fixed`（每 task 一套，做状态分支用）/ `legacy`（旧行为）。实测同 episode 身份隔两局重跑：per_episode 图像差异 **0.000%**，legacy **12.653%**。

2. **图像必须翻转**。LIBERO 离屏渲染上下颠倒，adapter 里做了 `[::-1, ::-1]`。`libero_sim_node.py` 那个雏形不翻，是因为它的 obs 来自 `server.py`（那边已翻过）。本节点直接读 env，**必须自己翻**。

3. **成功判定的属性路径**。`OffScreenRenderEnv` 上**没有** `_check_success`，实现在被包一层的 `env.env` 上；`info` 是空字典，`info["is_success"]` 那条路也不通。

4. **`policy_command` 的 queue_size 必须显式加深**。episode 边界连发 `set_instruction` + `reset_scene`，默认深度会把前一条挤掉，表现为**首个 episode 拿着上一句指令跑完**——照样出分，不报错。

5. **`use_delta` 必须在 `env.reset()` 之后设**。reset 会把 controller 覆盖回默认值。

6. **`CONDA_PREFIX`**。conda 版 mujoco 用它定位 `include/`，而该变量只有 `conda activate` 才会设——dora 直接拉起解释器时没有。adapter 已用 `sys.prefix` 兜底。

7. **不要盲从"LIBERO 必须 py3.8"**。`LIBERO/setup.py` 与 robosuite 1.4.0 的 `python_requires` 都只写 `>=3`。py3.8 上要给 forge_msgs 补四层兼容，其中 `zip(strict=True)`（11 处，含必经路径）**只能全局改写 `builtins.zip`**，会波及同进程里 env/numpy 的每一次 zip 调用，不可接受。py3.12 下全部归零。

另：episode 间用 `reset_scene` 而不是 `reset`——后者把 policy 打回 idle 相，之后不再响应观测。

---

## 8. 验证状态

**已验证**：完整 Forge 两节点闭环（VLA-JEPA + LingBot-VA，结构完全不同的两个策略）、adapter 口径（三条独立证据：归一化统计量量纲、原始 demo 字段结构、实测分数高于已发表基线）、`policy_command` 四条命令时序、1:1 锁步、批次展开与 `t/i/r` 索引、场景可复现、越界 fail fast、异常路径必落盘、结果 schema。

**未验证**：dora 两节点只跑过 4 个和 2 个 episode，那个 100 episode 的数字是**离线闭环**跑的——长批次下的 dora 链路稳定性（跨 10 个 task 反复重建 env、几百 episode 的内存表现）没有证据。suite 只测过 `libero_spatial` 和 `libero_10`，`object` / `goal` / `90` / `100` 没碰过。

---

## 9. 已知限制

**功能未实现**

- `record_dir` 的 mp4 落盘（配置项已留，无代码）
- 资源占用（CPU/GPU 峰值）记录
- wall-clock 看门狗——policy 卡死时节点静默悬停，只能靠 dora STOP 兜底

**设计上的欠缺**

- 相机别名（`image/main`、`image/second`）与 obs key（`agentview_image`、`robot0_eye_in_hand_image`）**硬编码在 adapter**，换 checkpoint 要改代码，应改为从 config 读
- 启动竞态：`start` 命令与首帧观测走不同通道且无 ack，靠通道投递顺序侥幸成立
- 预热步在 `use_delta` 设置**之前**执行，`control_mode: absolute` 时预热用的是默认 delta 语义（当前预热动作全 0 所以无害）
- 无启动自检：分辨率 / proprio 维度 / 别名与 checkpoint 的交叉核对尚未实现
