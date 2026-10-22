# LIBERO Benchmark Node

[English](README.md) | [简体中文](README_zh.md)

A Dora wrapper around LIBERO that owns its environment in-process and performs evaluation:
expand a batch, run episodes and persist results.

```text
Benchmark Node (this repository)                 Policy Node (unchanged)
  import libero; own env in-process
  proprio_state / image/main / image/second ---> observation cache + tick inference
  action <------------------------------------- 7D JointCommand
  policy_command -----------------------------> start / set_instruction / reset_scene / stop
  evaluate env.env._check_success()
  benchmark_status (stream) / benchmark_result (end of batch)
```

Evaluation, actions and resets use official LIBERO interfaces directly, without TargetWS or mujoco_sim.
The repository identifies this as a reference implementation of the PhyAgentOS Benchmark Integration Guide v14.

Repository separation does not change the Dora port contract:

- Input: `action`, a policy-generated 7D `JointCommand`.
- Outputs: `proprio_state` (8D `JointState`), `image/main`, `image/second`,
  `policy_command`, `benchmark_status`, `benchmark_result`.

**Historical repository results** (2026-08-16, complete Forge two-Node dataflow):

| Policy | Suite | Result | Elapsed | Mean latency |
|:---|:---|:---|---:|---:|
| VLA-JEPA | libero_spatial | 4/4 | 20.0 s | 16.3 ms |
| LingBot-VA | libero_10 | 2/2 | 128.7 s | 221 ms |

Both policy structures ran: VLA-JEPA has `observation.state`, LingBot-VA does not; chunk lengths are 7 vs 16,
with approximately 13-fold latency difference. Larger offline runs using the same adapter reported
VLA-JEPA libero_spatial success over 100 episodes of **94%** (rendering 256) / **96%** (rendering 224),
compared with the cited OpenVLA baseline of 84.7%.

---

## Node, Endpoint and Tool API

| Item | Current implementation |
| --- | --- |
| Production Nodes | 1: `libero_benchmark`; source `main.py`, packaged entrypoint `dist/libero_benchmark` |
| Tool support | Direct Endpoint provider with `serve: true`; default autorun registers no Endpoint |
| Endpoints | 1 in serve mode: `benchmark.controller` |
| Operations | 2: `describe` (Query), `run` (Session, stoppable, status supported; each has maximum concurrency 1) |
| Direct HTTP Tool API | 0; Dora `tool_in/tool_out` connect to Gateway |
| Gateway Tools | Defined by Skill/Gateway `tools.specs`; 2 operations do not imply 2 repository-fixed Tools |

| Tool ID | Endpoint / operation | Provider |
| --- | --- | --- |
| Configure in `tools.specs` | `benchmark.controller / describe` (Query) | `libero_benchmark` in serve mode |
| Configure in `tools.specs` | `benchmark.controller / run` (Session) | The same Node |

Current source has no independent `bench_endpoint` Node and does not receive Tool requests through legacy
`bench_cmd/bench_report`. Session stop takes effect at an episode boundary; acceptance is not completion.
Read the terminal invocation result for the final outcome.

Implementation: [main.py](main.py), [descriptor](bench_tool/descriptor.py),
[Endpoint methods](bench_tool/endpoints.py), [Dora loop](bench_tool/node.py).

Compatibility: current dependency declarations are `dora-rs==0.4.1`, `forge-msgs==1.2.0`, `forge-tool[dora]==1.0.0`. See [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock). These declarations do not guarantee cross-version interoperability; also check the build revision and Skill lock when using artifacts.

Changes to Node entrypoints, Endpoints or Tool APIs must update this section and the associated configuration/contracts. Follow the [Node README contract](https://github.com/PhyAgentOS/forge/blob/master/CONTRIBUTING.md#node-readme-contract).

## 1. Environment

### 1.1 Path placeholders

Replace the deployment-specific path placeholders below with your own locations:

| Placeholder | Meaning |
|:---|:---|
| `$BENCH_PY` | Absolute evaluation Python path; environment includes libero / dora / forge_msgs / forge_policy / lerobot / torch+CUDA |
| `$DORA_BIN` | Directory containing the `dora` CLI |
| `$LIBERO_NODE_ROOT` | This repository root |
| `$LEROBOT_RUNNER` | LeRobot runner repository root |
| `$MODELS` | Policy weights directory |
| `$WORK` | Working directory containing the three evaluation configuration files |
| `$PKG_LINK` | Package symlink directory for the policy Node; see section 2.2 |

Standalone repository:

```bash
git clone https://github.com/PhyAgentOS/benchmark-libero.git libero
cd libero
uv sync --all-groups
uv run pytest -q
```

`uv sync` installs base Node dependencies; the evaluation environment supplies LIBERO/robosuite and policy inference dependencies.

### 1.2 Two environment roles

Node dependencies and the `dora` CLI may live in separate environments:

| Role | Requirements |
|:---|:---|
| Node runtime (`$BENCH_PY`) | Both Nodes use this Python; install libero / dora module / forge_msgs / forge_policy / lerobot / torch+CUDA |
| `dora` CLI (`$DORA_BIN`) | Launches the dataflow; CLI and Node Dora versions must match, e.g. 0.4.1 / message 0.7.0, or registration fails with version mismatch |

```bash
export PATH=$DORA_BIN:$PATH     # Launcher CLI
# Use the absolute $BENCH_PY path in Node configuration; see dataflow below
```

Check the environment:

```bash
$BENCH_PY -c "import dora, forge_msgs, forge_policy, libero.libero, lerobot; print('ok')"
dora --version
```

Version requirements:

| Package | Version | Notes |
|:---|:---|:---|
| python | 3.12 | Avoid 3.8; see pitfall 7 in section 7 |
| dora-rs / dora-rs-cli | Matching versions | Mixed versions fail Node registration |
| forge-msgs / forge-policy | See locks | `forge_policy` supplies the policy event loop and handles `policy_command` |
| robosuite / mujoco | 1.4.0 + 3.x | The three-argument `mj_fullM` used by robosuite requires MuJoCo 3.x |

Download all weights to `$MODELS` and export `HF_HUB_OFFLINE=1` to avoid runtime network dependencies.
Sources are listed in section 1.3.

---

### 1.3 Policy weight sources

Sources below are official Hugging Face repositories. For offline deployment, download to `$MODELS`
and pin revisions (`huggingface-cli download <repo> --revision <hash>`), then export `HF_HUB_OFFLINE=1`.

**VLA-JEPA** (Qwen3-VL backbone and flow-matching DiT action head)

| Component | Download | HF repository ID | Local directory | Size | License |
|:---|:---|:---|:---|---:|:---|
| Policy checkpoint | https://huggingface.co/lerobot/VLA-JEPA-LIBERO | `lerobot/VLA-JEPA-LIBERO` | `vla_jepa_libero` | 5.8 G | Apache-2.0 |
| Qwen3-VL backbone | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct | `Qwen/Qwen3-VL-2B-Instruct` | `qwen3_vl_2b_instruct` | 4.0 G | Apache-2.0 |
| V-JEPA2 encoder | https://huggingface.co/facebook/vjepa2-vitl-fpc64-256 | `facebook/vjepa2-vitl-fpc64-256` | `vjepa2_vitl_fpc64_256` | 6.0 G | MIT |

```bash
huggingface-cli download lerobot/VLA-JEPA-LIBERO          --local-dir $MODELS/vla_jepa_libero
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct        --local-dir $MODELS/qwen3_vl_2b_instruct
huggingface-cli download facebook/vjepa2-vitl-fpc64-256   --local-dir $MODELS/vjepa2_vitl_fpc64_256   # Optional for inference
```

> V-JEPA2 is a training-only world model, unused during inference. Set `enable_world_model: false`
> to skip loading its 6 G of weights and associated GPU memory, as in the documented runs.
> Checkpoint `qwen_model_name` / `jepa_encoder_name` reference the backbone/encoder paths;
> explicitly override them with local `$MODELS` paths for offline use.

**LingBot-VA** (video-action world model without `observation.state`)

| Component | Download | HF repository ID | Local directory | Size | License |
|:---|:---|:---|:---|---:|:---|
| Policy checkpoint | https://huggingface.co/lerobot/lingbot_va_libero_long | `lerobot/lingbot_va_libero_long` | `lingbot_va_libero_long` | 9.5 G | See repository |
| Frozen Wan VAE + UMT5 | https://huggingface.co/robbyant/lingbot-va-base | `robbyant/lingbot-va-base` | `lingbot_va_base_frozen` | 14 G | See repository |

```bash
huggingface-cli download lerobot/lingbot_va_libero_long --local-dir $MODELS/lingbot_va_libero_long
huggingface-cli download robbyant/lingbot-va-base       --local-dir $MODELS/lingbot_va_base_frozen
```

The frozen directory must contain `text_encoder/` (UMT5, 11 G), `vae/` (Wan VAE, 2.7 G) and `tokenizer/` (21 M).

> The checkpoint's `config.json` sets `wan_pretrained_path` to online repository
> `robbyant/lingbot-va-posttrain-libero-long`. Override it with local
> `$MODELS/lingbot_va_base_frozen` in the policy configuration or offline runs will attempt a network download.

**Other policy weights** (repository status as of 2026-08-16)

| Policy | Status |
|:---|:---|
| ACT | Supplied training artifacts; convert with `lerobot convert`; no public checkpoint source specified |
| Pi0.5 | Supplied training artifacts; also requires a local PaliGemma tokenizer |
| Diffusion | Official LeRobot Hub, e.g. `lerobot/diffusion_pusht`: https://huggingface.co/lerobot/diffusion_pusht (Apache-2.0). PushT domain: 2D state/action and 96x96 images; not usable for LIBERO |

---

## 2. Configuration: three files

Evaluation uses three files whose matching fields must agree exactly. Mismatches may lower scores without errors:

```text
bench.yaml          Benchmark Node configuration; see config.example.yaml
policy_xxx.yaml     LeRobot policy Node configuration
dataflow.yaml       Dora topology connecting benchmark and policy
```

### 2.1 Fields that must match

| Concept | bench.yaml | policy_xxx.yaml | dataflow.yaml |
|:---|:---|:---|:---|
| Policy identity | `policy_id` | `policy.policy_id` | — |
| Camera channels | Hardcoded in adapter; section 9 | Left-side `image_inputs` keys | `inputs:` channel names |
| Camera-to-feature mapping | — | Right-side `image_inputs` aliases | — |
| Proprioceptive joint names | `STATE_JOINTS` in `config.py` | `state_joints` | — |
| Action joints | — | `joints` (7) | — |
| Render resolution | `camera_height/width` | Policy resizes internally | — |
| Language instruction | `send_instruction: true` | `policy.instruction`, overwritten per episode | — |

A mismatched `policy_id` makes the policy ignore commands and use its initial instruction for the entire batch.
Mismatched `image_inputs` keys prevent image delivery. Mismatched `STATE_JOINTS` / `state_joints`
make name-based `JointState` lookups silently return empty values.

### 2.2 Complete VLA-JEPA configuration

`bench.yaml`：

```yaml
suite: libero_spatial
task_ids: "0-1"                # all / "0-9" / "0,2-4,8" / [0,1,2]; empty means all
init_state_ids: "0-1"
num_runs: 1
max_steps: 300
num_steps_wait: 10
control_mode: relative
seed: 0
layout_mode: per_episode       # Pitfall 1 in section 7; changing this changes scores
camera_height: 256
camera_width: 256
policy_id: vla_jepa_libero     # Must match the policy configuration
send_instruction: true
result_dir: $WORK/results
record_dir: null
```

`policy_vla_jepa.yaml`：

```yaml
joints:                         # 7 action dimensions
  - {name: joint1, mode: position, unit: radians}
  # ... configure joint2..joint7 similarly
state_joints: [joint1, joint2, joint3, joint4, joint5, joint6, joint7, gripper]   # 8D proprioception

policy:
  type: vla_jepa
  policy_id: vla_jepa_libero    # Must match bench.yaml
  auto_start: false             # Benchmark sends start
  instruction: "..."            # Placeholder; overwritten per episode
  device: cuda
  enable_world_model: false     # V-JEPA2 is unnecessary for inference
  pretrained_path: $MODELS/vla_jepa_libero

image_inputs:
  image/main: image             # Left: Dora channel; right: checkpoint feature without observation.images. prefix
  image/second: image2
```

`dataflow.yaml`：

```yaml
nodes:
  - id: libero_benchmark
    path: $BENCH_PY
    args: -u $LIBERO_NODE_ROOT/main.py
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
      PYTHONPATH: $PKG_LINK          # lerobot_inference package; see below
      HF_HUB_OFFLINE: "1"
      CUDA_VISIBLE_DEVICES: "5"
    inputs:
      tick: dora/timer/millis/20
      policy_command:
        source: libero_benchmark/policy_command
        queue_size: 32                # Explicit depth required; pitfall 4 in section 7
      proprio_state: libero_benchmark/proprio_state
      image/main: libero_benchmark/image/main
      image/second: libero_benchmark/image/second
    outputs: [action, policy_command_status]
```

**`PYTHONPATH=$PKG_LINK` is required:** the import name is `lerobot_inference`, while the checkout
folder is named differently. Create its package symlink once:

```bash
mkdir -p $PKG_LINK && ln -sfn $LEROBOT_RUNNER $PKG_LINK/lerobot_inference
```

Dora `path` values must be absolute; `~` is not expanded. Resolve symlinked home paths with `readlink -f`.

### 2.3 LingBot-VA differences

Change these three policy fields; retain the remaining configuration:

```yaml
policy:
  type: lingbot_va
  policy_id: lingbot_va_libero
  pretrained_path: $MODELS/lingbot_va_libero_long
  # Override the checkpoint's online HF repository with a local frozen-weight directory
  wan_pretrained_path: $MODELS/lingbot_va_base_frozen
# No state_joints: LingBot-VA is video-action only, without observation.state
```

In bench.yaml, set `suite` to `libero_10` and `policy_id` to `lingbot_va_libero`.

---

## 3. Running

### 3.1 Stub smoke test: timing without GPU/policy weights

The fake policy returns empty actions. A constant `success_rate` of 0 is expected.

```bash
export PATH=$DORA_BIN:$PATH
cd $LIBERO_NODE_ROOT
dora run dataflow_stub.yaml          # Stub needs neither GPU nor policy weights
```

Check all four properties, rather than just process completion:

1. Each episode receives its own `set_instruction` content.
2. The stub reports `obs == act` at exit (1:1 lockstep).
3. `t/i/r` indices match the configured batch.
4. Result JSON is persisted and `episodes[]` length equals `total_episodes`.

### 3.2 Real-policy two-Node evaluation

```bash
export PATH=$DORA_BIN:$PATH
cd $WORK             # Directory containing all three configuration files
dora run dataflow.yaml
```

`$WORK` needs `bench.yaml`, `policy_<policy>.yaml` and `dataflow.yaml`; see sections 2.2 / 2.3.

Start each new checkpoint with a small baseline comparison (2 tasks x 2 initial states), comparing
`success_rate` against published results. Stub testing checks timing; real-policy testing checks
observation/action conventions. Neither replaces the other.

### 3.3 Three checks for a new checkpoint

These mismatches may silently reduce scores:

| Item | Check |
|:---|:---|
| Proprioceptive semantics | Read checkpoint normalization statistics, not YAML descriptions; per-dimension min/mean/max reveal joint-angle vs end-effector-pose units |
| Camera aliases | Left-side `image_inputs` keys must match output channel names |
| Resolution | Inspect checkpoint `config.json`; with `resize_images_to`, provide native resolution for policy-side resizing; otherwise match `input_features` |

> Example: a VLA-JEPA YAML comment describing 8D state as seven joint angles plus gripper was incorrect.
> Statistics indicate `eef_pos(3) + axis_angle(3) + gripper_qpos(2)`: dimension 3 mean 0.79 is tabletop
> height in metres; dimension 4 mean 2.98 is approximately pi for downward orientation; dimensions 7/8
> are symmetric opposite-sign finger values.

---

## 4. Code map

```text
main.py                      Entry assembly for the generic batch runner
config.py                    Configuration, joint constants, validation and seed derivation
result.py                    BenchmarkExecutionResultV1 assembly/persistence
adapters/libero_adapter.py   LIBERO-specific implementation
adapters/__init__.py         Empty
config.example.yaml          Annotated configuration template
dataflow.example.yaml        Real-policy dataflow template
dataflow_stub.yaml           Git-ignored stub smoke fixture
stub_policy.py               Git-ignored fake policy
pyproject.toml               Package metadata; evaluation environment supplies LIBERO/robosuite
```

### 4.1 `main.py`: batch orchestration

The generic batch runner is benchmark-independent; the entrypoint assembles it with the LIBERO adapter.

| Unit | Responsibility |
|:---|:---|
| `BenchmarkRunner` | Batch plan and current episode state |
| `.start_next_episode()` | Select next `(task, init_state, run)`, derive seed, send `set_instruction` / `reset_scene`, reset adapter and publish first observation |
| `.on_action()` | Validate finite action, call `env.step`, check success every step and publish the next observation unless finished |
| `.advance()` | Advance episodes; exceptions become batch failure rather than escaping the event loop |
| `._record_episode()` | Record only; callers explicitly invoke `advance()` |
| `.fail_current()` | Central exception path: record failure and advance |
| `.finish()` | Persist before sending final messages or closing the environment |
| `main()` | Load config, create/validate adapter, create Node, run event loop; guarantee `finish()` in `finally` |

Timing contract: one observation per action. Publishing sets `awaiting_action=True`; stepping waits
for an action. Extra actions with `awaiting_action=False` are discarded because policy ticks may be
faster than environment steps. The policy owns action-chunk queues; benchmark is unaware of them.

### 4.2 `config.py`: configuration and validation

| Unit | Responsibility |
|:---|:---|
| `STATE_JOINTS` | Proprioceptive names; must exactly match policy `state_joints` |
| `STATE_SEMANTICS` | Documentation-only 8D annotations aligned with `STATE_JOINTS`: eef_pos/axis_angle/gripper |
| `parse_id_list()` | Parse `all`, `"0-9"`, `"0,2-4,8"`, `[0,1,2]`, `None` |
| `derive_episode_seed()` | Stable seed from `(seed, task, init, run)`; avoids built-in `hash()` and per-process string randomization via `PYTHONHASHSEED` |
| `BenchmarkNodeConfig` | Configuration dataclass |
| `load_config()` | Load YAML and validate `layout_mode`, `num_runs`, `max_steps` |
| `resolve_and_validate_plan()` | Expand `all` and validate IDs per task; tasks may have different initial-state counts |

### 4.3 `adapters/libero_adapter.py`: LIBERO integration

Keep benchmark-specific environment logic in the adapter when adding another benchmark.

| Unit | Responsibility |
|:---|:---|
| `num_tasks` / `num_init_states()` / `task_language()` | Metadata for batch expansion |
| `_ensure_env()` | Reuse environment within a task; rebuild on task change (approximately 2 seconds in documented setup) |
| `_sim()` | Locate the wrapper layer owning `sim` |
| `capture_layout()` / `restore_layout()` | Snapshot/restore fixture `body_pos` / `body_quat`; see pitfall 1 |
| `reset_episode()` | Apply layout mode, reset environment, restore initial state, warm up, then set `use_delta` |
| `step()` | Apply 7D action; return `StepOutcome(obs, reward, done, success)` |
| `check_success()` | Find `_check_success` through wrappers, on `env.env` rather than `OffScreenRenderEnv` |
| `to_state()` | 8D `eef_pos(3) + axis_angle(3) + gripper_qpos(2)` |
| `to_images()` | Read two images and flip with `[::-1,::-1]` |
| `quat_to_axisangle()` | robosuite quaternion order is `(x,y,z,w)` |

### 4.4 `result.py`: result assembly

| Unit | Responsibility |
|:---|:---|
| `EpisodeRecord` | Per-episode record including `episode_seed` and `success_source` |
| `BenchmarkResult` | Batch aggregation; `success_rate` divides by planned total, including cancelled work; latency filtering uses `is not None` to retain exact 0.0 values |
| `.dump()` | Write `<result_dir>/<suite>_<timestamp>.json` |

---

## 5. Results

Results are written to `result_dir` using `BenchmarkExecutionResultV1`.

Batch fields:

| Field | Meaning |
|:---|:---|
| `status` | `succeeded` / `failed` / `cancelled` |
| `successes` / `total_episodes` / `success_rate` | Denominator is planned total, not completed count |
| `completed_episodes` / `valid_episodes` | Completed episodes / episodes without errors |
| `timed_out_episodes` | Timeout truncations |
| `layout_mode` / `seed` | Required scene reproducibility parameters |
| `num_steps` / `mean_policy_latency_ms` / `elapsed_s` | Steps / mean inference latency / elapsed time |

Each `episodes[]` entry includes `episode_id`, `task_id`, `init_state_id`, `run_index`,
`task_description`, `success`, `termination` (`success`/`failed`/`timed_out`), `num_steps`,
`return_value`, `episode_seed`, `mean_policy_latency_ms`, `success_source`, `error_code`/`error_message`.

`first_attempt_*`, `final_*`, `assisted_*`, `episodes_replanned` and `recovered_after_replan`
remain `null`: this Node does not perform recovery retries; upper layers own those fields.

Ctrl-C or Dora STOP mid-batch yields `status=cancelled`; completed episodes are still persisted.

---

## 6. Troubleshooting

| Symptom | Cause/check |
|:---|:---|
| No progress after startup | Policy is not running: mismatched `policy_id` or missing `start`; a hung policy also stalls silently because no timeout watchdog exists |
| First episode uses the previous instruction | Increase `policy_command` input `queue_size` |
| Low scores without errors | Check proprioceptive semantics, image orientation, camera aliases, resolution and `control_mode`, in that order |
| `AttributeError: _check_success` | Method belongs to `env.env`, not `OffScreenRenderEnv` |
| Missing `BenchmarkNodeConfig` attribute | A configuration field was omitted; run a real dataflow after configuration-model changes, not just `py_compile` |
| `TypeError: mj_fullM` | MuJoCo 3.x functionality is required; a cloned environment's version label alone is insufficient |
| `os.path.join(None, ...)` TypeError | `CONDA_PREFIX` missing; adapter falls back to `sys.prefix` |
| `Could not initiate node from environment variable` | Inspect the peer Node's earlier traceback |
| Version mismatch | Node/CLI Dora versions differ |
| Same configuration, different scores | Check for `layout_mode: legacy` |
| `ProcessorMigrationError` | Migrate historical normalization using LeRobot `migrate_policy_normalization.py` |

---

## 7. Seven implementation pitfalls

The repository documents these as silent causes of score degradation:

1. **`set_init_state` does not restore fixtures.** Jointless fixture poses live in model `body_pos/body_quat`;
   restoring state `qpos/qvel` does not undo fixture resampling by `env.reset()`.
   Historical observations reported 1.5 cm fixture drift and 10% differing pixels despite identical state,
   cameras and lighting. `layout_mode` selects `per_episode` (default; layout is a function of
   `(seed,task,init,run)`), `fixed` (one layout per task) or legacy behavior. Repeated episode identities
   reportedly differed by 0.000% pixels in per_episode mode versus 12.653% in legacy mode.

2. **Flip images.** This direct-environment adapter applies `[::-1, ::-1]` to offscreen images.
   The earlier `libero_sim_node.py` prototype received already-flipped observations from `server.py`.

3. **Check success on the correct wrapper.** `_check_success` is on `env.env`, not `OffScreenRenderEnv`;
   `info` is empty, so `info["is_success"]` is unavailable.

4. **Set `policy_command` queue depth explicitly.** Episode boundaries emit `set_instruction` and
   `reset_scene` back-to-back; default depth may drop the instruction and silently reuse the previous one.

5. **Set `use_delta` after `env.reset()`.** Reset restores the controller default.

6. **Handle missing `CONDA_PREFIX`.** Direct Dora interpreter startup may lack conda activation;
   the adapter falls back to `sys.prefix` to locate MuJoCo includes.

7. **Use the documented Python 3.12 environment.** LIBERO and robosuite 1.4.0 declare `>=3`, not a
   hard Python 3.8 requirement. Supporting 3.8 would require multiple Forge compatibility changes,
   including 11 `zip(strict=True)` uses; globally replacing `builtins.zip` would affect env/numpy too.

Between episodes, use `reset_scene`, not `reset`: the latter returns the policy to idle and stops observation handling.

---

## 8. Historical test matrix

These are repository-reported combinations as of 2026-08-16. **Two-Node** means benchmark and policy
communicate through Dora; **offline** means the policy drives the environment in-process through the
same adapter, without Dora.

### 8.1 Reported scores

| Policy | Suite | Scale | Rendering | Mode | success_rate | Mean latency | Elapsed |
|:---|:---|:---|---:|:---|---:|---:|---:|
| VLA-JEPA | libero_spatial | 2 tasks x 2 init = 4 ep | 256 | Two-Node | 4/4 = 100% | 16.3 ms | 20 s |
| VLA-JEPA | libero_spatial | 10 tasks x 10 init = 100 ep | 256 | Offline | 94/100 = 94.0% | 15–28 ms | 643 s |
| VLA-JEPA | libero_spatial | 10 tasks x 10 init = 100 ep | 224 | Offline | 96/100 = 96.0% | 15–28 ms | 603 s |
| LingBot-VA | libero_10 | 2 tasks x 1 init = 2 ep | 256 | Two-Node | 2/2 = 100% | 221 ms | 129 s |
| LingBot-VA | libero_10 | 2 tasks x 2 init = 4 ep | 256 | Offline | 3/4 = 75% | 245–310 ms | 322 s |

Shared parameters: `max_steps=300`, `num_steps_wait=10`, `control_mode=relative`, `seed=0`, `layout_mode=per_episode`.

The cited OpenVLA LIBERO-Spatial baseline is 84.7%; reported VLA-JEPA runs achieved 94–96%.
These historical comparisons do not establish equivalence of every evaluation convention.

For rendering 224 vs 256, three failures were shared (t5_i0/i4/i5), three were unique to 256,
and one to 224; the reported exact McNemar test gave p = 0.63. Default rendering remains 256.

All six VLA-JEPA failures in the 256 run were task-5 `timed_out` outcomes, with no collision failure reported.
Successful episodes took at most 137 steps, while failures exhausted 300; this suggests policy stalls
rather than merely insufficient time in those runs.

### 8.2 Failed or untested combinations

| Policy | Status | Reason |
|:---|:---|:---|
| Diffusion (`diffusion_synth`) | Loads/infers (398 ms) | PushT: 2D state/action, 96x96 images; incompatible with LIBERO |
| Diffusion (`diffusion_pusht`) | Load failure | `ProcessorMigrationError`; run LeRobot normalization migration |
| ACT | Untested | No supplied training checkpoint available in the documented environment |
| Pi0.5 | Untested | No supplied training checkpoint available in the documented environment |
| SmolVLA (`smolvla_libero`) | Untested | Config declares 6D state but statistics are 8D; model needs 3 cameras while LIBERO provides 2 |

### 8.3 Coverage and gaps

The repository reports checks of the full two-Node loop, adapter conventions using normalization
statistics/original LIBERO demonstrations/scores, four `policy_command` sequences, 1:1 lockstep,
batch expansion and `t/i/r` indexing, scene reproducibility, fail-fast bounds validation,
exception-path persistence and result schema.

The two policies differ in state inputs, chunk length (7 vs 16) and approximately 13-fold latency;
both were exercised, rather than only one policy architecture.

Remaining gaps:

- Two-Node runs covered only 4 and 2 episodes. The 100-episode results were offline; long Dora batches,
  repeated environment rebuilding across 10 tasks and hundreds of episodes of memory behavior remain unestablished.
- Only `libero_spatial` and `libero_10` were exercised; `object`, `goal`, `90`, `100` were not.
- `num_runs > 1` was not exercised; per_episode mode would vary layouts across runs, but that remains unverified.

---

## 9. Known limitations

Unimplemented features:

- MP4 output for `record_dir` (configuration exists, implementation absent).
- Peak CPU/GPU resource tracking.
- Wall-clock watchdog; a hung policy stalls the Node until Dora STOP.

Design limitations:

- Camera aliases `image/main`, `image/second` and observation keys `agentview_image`, `robot0_eye_in_hand_image`
  are hardcoded in the adapter; changing checkpoints may require code changes.
- Startup has no acknowledgment between separate start/observation channels and relies on delivery ordering.
- Warmup occurs before `use_delta` is set. With `control_mode: absolute`, warmup still uses default delta
  semantics; currently its all-zero actions avoid an effect.
- No startup cross-check of resolution, proprioceptive dimensions or aliases against the checkpoint.


## 10. Serve mode and standalone Node bundle

The same `main.py` / `libero_benchmark` entrypoint selects two modes:

- `serve: false` (default): autorun one configured batch, then exit.
- `serve: true`: resident Node directly registering `benchmark.controller`, with `describe` Query and `run` Session.

Connect the Gateway provider output to this Node's `tool_in` and its `tool_out` back to the Gateway provider input.
Retain `action` input and policy observations, images and `policy_command` dataflow.
`bench_tool/node.py` composes the Endpoint handler and evaluation loop in the same process; no separate Endpoint Node is needed.

Call `describe` before `run`. Session uses stop at episode boundaries; neither accepted nor stop accepted
is terminal. Results include primary_metric, success_rate, successes, total_episodes, completed_episodes,
result_path and episode details; see [result assembly](bench_tool/result_shape.py).

### 10.1 Build the standalone Node

From the repository root:

    ./scripts/build_pyinstaller.sh

Output:

    dist/libero_benchmark

The PyInstaller one-file build collects _core, LIBERO, robosuite, MuJoCo Python modules, Dora,
Forge messages and required package data. No other Forge checkout is required at build time.

The Node excludes LINGBOT policies/checkpoints. Large LIBERO scene/object assets are host runtime assets
located through standard LIBERO configuration; do not commit models or private machine paths.

### 10.2 Host environment

For headless LIBERO:

    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl

The host needs working EGL/OpenGL drivers (libEGL/libGL and vendor drivers) and installed LIBERO assets.
The benchmark Node itself does not require CUDA; the separate policy Node uses GPU/CUDA according to its configuration.

After building, run from an arbitrary temporary directory:

    unset PYTHONPATH
    /tmp/libero_benchmark

Verify that the executable does not depend on a source checkout, private Python paths or external PYTHONPATH.

### 10.3 Compatibility and troubleshooting

- BENCH_NOT_READY: wait for benchmark initialization, then retry describe; do not modify the runtime manager.
- HTTP 503 / operation concurrency ... exhausted: an invocation remains active; reject immediately without queueing or starting another execution.
- Startup timeout: allow sufficient profile `startup_timeout_s` for LIBERO/MuJoCo initialization.
- result_path: use the terminal Tool invocation result; accepted alone does not establish completion.

## 11. Endpoint and deployment versions

Current source uses `libero_benchmark` itself as the provider in serve mode.
[`benchmark.controller`](bench_tool/descriptor.py) declares `run` as Session.
There is no `bench_endpoint/` directory or `bench_endpoint` executable.

Historical separate-Endpoint, Query/Action or `bench_cmd/bench_report` instructions do not configure
this version. Configure Query/Session ToolSpecs and provider channels in Gateway, matching Gateway,
`forge-tool`, `forge-msgs`, Dora and consumer versions. Source capability does not establish historical
Skill-locked binary capability.

The benchmark Node computes scores/results; Gateway/agent neither substitutes its own scoring nor aggregates across invocations.
