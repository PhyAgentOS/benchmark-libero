# LIBERO Benchmark 节点

把 LIBERO 包一层的 Dora 节点，进程内持有 env，只提供 benchmark 服务。

```
benchmark 节点（本节点）                     policy 节点（不改）
├─ 内嵌 LIBERO env（进程内 import libero）
├─ proprio_state / image/main / image/second ──观测──▶ 缓存 + tick 推理
├─ action ◀────────────────────────────────────动作──  JointCommand 7D
├─ policy_command ─────────────────────────────命令──▶ start/set_instruction/reset_scene/stop
├─ 判定 env.env._check_success()
└─ benchmark_status（流式）/ benchmark_result（批次末）
```

不经过 TargetWS、不经过 mujoco_sim——判定、动作、重置全走 LIBERO 官方接口，零适配数学。

## 运行

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
BENCHMARK_CONFIG=./config.yaml dora run dataflow.example.yaml
```

配置项见 `config.example.yaml`。批次 = `task_ids × init_state_ids × num_runs` 串行展开，
支持 `all` / `"0-9"` / `"0,2-4,8"` / `[0,1,2]`。

结果落 `result_dir`，schema 为 `BenchmarkExecutionResultV1`；恢复/重试相关字段本节点
一律填 `null`（不做恢复重试，由上层填充）。

## 环境要求

需要一个**同时**装了 LIBERO 和 dora 的 Python 环境：

| 包 | 版本 | 备注 |
|:---|:---|:---|
| python | ≥3.10 | 见下方"为什么不是 3.8" |
| dora-rs / dora-rs-cli | 两者必须同版本 | 0.4.1 走 message format v0.7.0；混用 0.5.0 CLI 会在注册节点时报 version mismatch |
| forge-msgs | 1.0.1 | 依赖 pydantic ≥2 与 pyarrow |
| libero / robosuite / mujoco | robosuite 1.4.0 + mujoco 3.x | robosuite 的三参数 `mj_fullM` 只有 mujoco 3.x 认，2.3.7 会 TypeError |

参考环境 `forge_bench`（由 `dtvla` 克隆 + 装 dora-rs/forge-msgs 得到）。

### 为什么不是 Python 3.8

LIBERO 常见的装法是 py3.8，但 `LIBERO/setup.py` 与 robosuite 1.4.0 的 `python_requires`
都只写了 `>=3`——py3.8 不是硬约束。而在 py3.8 上跑本节点要给 forge_msgs 补四层：

1. `typing.Self`（10 个文件）——可 shim
2. `list[str]` 注解需 pydantic 运行时求值——需 `eval_type_backport`
3. `itertools.pairwise`——可 shim
4. **`zip(strict=True)`（11 处，含 `joint.py` 的必经路径）——只能全局改写 `builtins.zip`**

第 4 条会波及同进程里 LIBERO/robosuite/mujoco/numpy 的每一次 zip 调用。py3.12 下这些
全部归零，所以选 py3.12。

## 几个会静默出错的地方

都是实测踩出来的，改代码前先读：

- **图像必须翻转**。LIBERO 离屏渲染上下颠倒，`adapters/libero_adapter.py` 里做了
  `[::-1, ::-1]`。`lerobot_runner` 那个 `libero_sim_node.py` 雏形不翻，是因为它的 obs
  来自 `server.py`（那边已经翻过）。本节点直接读 env，**必须自己翻**，否则策略看到倒立
  世界，分数会安静地烂掉。
- **成功判定的属性路径**。`OffScreenRenderEnv` 上**没有** `_check_success`（接入手册
  §3.3.1 写的是它，照抄会 AttributeError），实现在被包一层的 `env.env` 上；`info` 是空
  字典，`info["is_success"]` 那条路也不通。
- **`policy_command` 的 queue_size 必须显式加深**。episode 边界会连发
  `set_instruction` + `reset_scene`，默认队列深度会把前一条挤掉，表现为**首个 episode
  拿着上一句指令跑完**——照样出分，不报错。
- **指令必须逐 episode 下发**。LIBERO 每个 task 指令不同，而 policy 配置里的
  `instruction` 是写死的一句。关掉 `send_instruction` 等于整批用同一句错指令推理。
- **`CONDA_PREFIX`**。conda 版 mujoco 用它定位 `include/`，而该变量只有 `conda activate`
  才会设——dora 直接拉起解释器时没有。adapter 里已用 `sys.prefix` 兜底。
- **episode 间用 `reset_scene` 而不是 `reset`**。`reset` 会把 policy 打回 idle 相，之后
  不再响应观测；`reset_scene` 清动作队列和观测缓存但保持 running。

## 状态

已验证（stub policy + 真 LIBERO env，`dora run` 全链路）：

- [x] 批次展开 task×init_state×num_runs，episode_id 的 `t/i/r` 索引正确
- [x] 逐 episode 下发 instruction（实测两个 task 的指令确实不同）
- [x] 观测/动作 1:1 锁步（obs 数 == act 数）
- [x] 翻转、8D proprio 拼接、成功判定接口
- [x] `max_steps` 截断记 `timed_out`（与 `failed` 区分）
- [x] 批次聚合 + `BenchmarkExecutionResultV1` 落盘 + `stop` 收尾

未做：

- [ ] **接真 policy 跑分**——目前只用 stub 验调度闭环，success_rate 恒为 0 是预期
      （stub 发空动作）。真实分数需接 VLA-JEPA checkpoint 后对照已发表基线核对
- [ ] `record_dir` 的 mp4 落盘（配置项已留，未实现）
- [ ] 资源占用（CPU/GPU 峰值）记录
- [ ] 全 suite 跑分

## 与接入手册的偏差

`PhyAgentOS-main/docs/forge/Benchmark接入手册.md`（v13，2026-08-13）描述的是
**三方架构**（benchmark → `sim_command` → 不改动的 Simulator → Policy），而本节点按
2026-08-14 拍板的**两节点方案**实现（benchmark 内嵌 LIBERO env，不用 mujoco_sim）。

手册的 §3.1 / §3.4 / §5 / §7.1 / §8-P6 与本实现不一致，**手册尚未同步**。
