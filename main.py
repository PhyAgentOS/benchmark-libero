#!/usr/bin/env python3
"""LIBERO benchmark 节点（Dora）。

架构（两节点，2026-08-14 定稿，手册 v14）：

    benchmark 节点（本节点，进程内持有 LIBERO env）
      ├─ 输出 proprio_state(8D JointState) + image/main + image/second
      ├─ 输入 action(JointCommand 7D) → env.step
      ├─ 判定 env.env._check_success()
      └─ 批次 task×init_state×num_runs → benchmark_status / benchmark_result
            ▲ dora 共享内存
            ▼
    policy 节点（不改）

不经过 TargetWS、不经过 mujoco_sim——判定/动作/重置全走 LIBERO 官方接口。

时序：action 驱动。发布一次观测 → 等 policy 的 action → step → 再发布观测。
policy 内部的 action chunk 由它自己排队，每 tick 吐一步，本节点按 1:1 消费。
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np
import pyarrow as pa
from dora import Node
from forge_msgs import Image, JointCommand, JointState, PolicyCommand

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from adapters.libero_adapter import LiberoAdapter  # noqa: E402
from config import (  # noqa: E402
    STATE_JOINTS,
    BenchmarkNodeConfig,
    derive_episode_seed,
    load_config,
    resolve_and_validate_plan,
)
from result import BenchmarkResult, EpisodeRecord  # noqa: E402


def _log(msg: str) -> None:
    print(f"[libero_bench] {msg}", flush=True)


class BenchmarkRunner:
    """批次调度核心。通用逻辑，不认识 LIBERO——benchmark 特有的都在 adapter 里。"""

    def __init__(self, node: Node, cfg: BenchmarkNodeConfig, adapter: LiberoAdapter) -> None:
        self.node = node
        self.cfg = cfg
        self.adapter = adapter
        self.plan = cfg.episode_plan()
        self.result = BenchmarkResult(
            suite=cfg.suite,
            control_mode=cfg.control_mode,
            max_steps=cfg.max_steps,
            total_episodes=len(self.plan),
            layout_mode=cfg.layout_mode,
            seed=cfg.seed,
        )
        self.plan_index = -1
        self.started_at = time.time()

        # 当前 episode 状态
        self.step_idx = 0
        self.episode_return = 0.0
        self.episode_seed: int | None = None
        self.awaiting_action = False
        self.obs_sent_at: float | None = None
        self.latencies: list[float] = []
        self.finished = False

    # ---- policy 生命周期 ----

    def _policy_command(self, command: str, inputs: dict[str, Any] | None = None) -> None:
        msg = PolicyCommand.from_inputs(
            policy_id=self.cfg.policy_id, command=command, inputs=inputs or {}
        )
        self.node.send_output("policy_command", msg.to_arrow())

    # ---- 观测发布 ----

    def _publish_obs(self, obs: dict[str, Any]) -> None:
        state = self.adapter.to_state(obs)
        self.node.send_output(
            "proprio_state",
            JointState.from_np(state, STATE_JOINTS, "position").to_arrow(),
        )
        for output_id, frame in self.adapter.to_images(obs).items():
            self.node.send_output(output_id, Image.from_numpy(frame, "rgb8").to_arrow())

        self.obs_sent_at = time.time()
        self.awaiting_action = True

    def _emit_status(self, text: str) -> None:
        self.node.send_output("benchmark_status", pa.array([text]))
        _log(text)

    # ---- episode 推进 ----

    def advance(self) -> None:
        """推进到下一个 episode。任何异常都必须收敛成"批次失败"，绝不能冲出事件循环。

        （旧实现里 _record_episode 直接递归调用 start_next_episode，一旦这里抛错，
        异常是在事件循环的 except 分支内二次抛出的，会冲出循环导致 finish() 从不
        执行、整批结果不落盘。）
        """
        if self.finished:
            return
        try:
            self.start_next_episode()
        except Exception as exc:  # noqa: BLE001
            _log(f"启动下一 episode 失败：{type(exc).__name__}: {exc}")
            self.finish("failed")

    def start_next_episode(self) -> bool:
        """返回 False 表示批次已跑完。"""
        self.plan_index += 1
        if self.plan_index >= len(self.plan):
            self.finish("succeeded")
            return False

        task_id, init_state_id, run_index = self.plan[self.plan_index]
        instruction = self.adapter.task_language(task_id)
        self.episode_seed = derive_episode_seed(
            self.cfg.seed, task_id, init_state_id, run_index
        )

        # 指令必须逐 episode 下发：LIBERO 每个 task 指令不同，policy 配置里那句是写死的。
        # 顺带 reset_scene 清掉上一局残留的 action 队列与观测缓存（reset 会掉出 running 相，
        # 不能用）。
        if self.cfg.send_instruction:
            self._policy_command("set_instruction", {"instruction": instruction})
        self._policy_command("reset_scene")

        obs = self.adapter.reset_episode(
            task_id=task_id,
            init_state_id=init_state_id,
            num_steps_wait=self.cfg.num_steps_wait,
            control_mode=self.cfg.control_mode,
            episode_seed=self.episode_seed,
        )

        self.step_idx = 0
        self.episode_return = 0.0
        self.latencies = []
        self._emit_status(
            f"episode {self.plan_index + 1}/{len(self.plan)} 开始 "
            f"t{task_id}_i{init_state_id}_r{run_index} :: {instruction}"
        )
        self._publish_obs(obs)
        return True

    def on_action(self, command: JointCommand) -> None:
        if self.finished or not self.awaiting_action:
            return  # 非 1:1 的多余 action（policy tick 快于我们）直接丢弃

        action = np.asarray(command.position, dtype=np.float32).reshape(-1)
        if action.size < 7 or not np.all(np.isfinite(action)):
            self.fail_current("invalid_action", f"action 非法：size={action.size}")
            return

        if self.obs_sent_at is not None:
            self.latencies.append((time.time() - self.obs_sent_at) * 1000.0)
        self.awaiting_action = False

        outcome = self.adapter.step(action)
        self.step_idx += 1
        self.episode_return += outcome.reward

        # 每步查一次判定（手册 §3.4 规则 1），不是 episode 结束才查。
        if outcome.success or outcome.done:
            self._record_episode(
                success=outcome.success,
                termination="success" if outcome.success else "failed",
            )
            self.advance()
        elif self.step_idx >= self.cfg.max_steps:
            # 超时截断算失败，但 termination 单独记，便于区分"没做到"和"没做完"。
            self._record_episode(success=False, termination="timed_out")
            self.advance()
        else:
            self._publish_obs(outcome.obs)

    # ---- episode 收尾 ----

    def _episode_meta(self) -> tuple[int, int, int, str]:
        task_id, init_state_id, run_index = self.plan[self.plan_index]
        return task_id, init_state_id, run_index, self.adapter.task_language(task_id)

    def _record_episode(
        self,
        success: bool,
        termination: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """只记录，不推进。推进由调用方显式 advance()——见 advance() 的注释。"""
        task_id, init_state_id, run_index, instruction = self._episode_meta()
        record = EpisodeRecord(
            episode_id=f"{self.cfg.suite}_t{task_id}_i{init_state_id}_r{run_index}",
            suite=self.cfg.suite,
            task_id=task_id,
            init_state_id=init_state_id,
            run_index=run_index,
            task_description=instruction,
            success=success,
            termination=termination,
            num_steps=self.step_idx,
            return_value=self.episode_return,
            episode_seed=self.episode_seed,
            mean_policy_latency_ms=(
                sum(self.latencies) / len(self.latencies) if self.latencies else None
            ),
            error_code=error_code,
            error_message=error_message,
        )
        self.result.episodes.append(record)
        self.awaiting_action = False

        done_n = len(self.result.episodes)
        self._emit_status(
            f"{record.episode_id} {termination} "
            f"({self.step_idx} 步) | 累计 {self.result.successes}/{done_n} "
            f"= {self.result.successes / done_n:.1%}"
        )

    def fail_current(self, code: str, message: str) -> None:
        """当前 episode 记为失败并推进。异常路径的唯一入口。"""
        if self.finished or self.plan_index < 0 or self.plan_index >= len(self.plan):
            return
        _log(f"episode 失败：{code} {message}")
        self._record_episode(
            success=False, termination="failed", error_code=code, error_message=message
        )
        self.advance()

    # ---- 批次收尾 ----

    def finish(self, status: str) -> None:
        if self.finished:
            return
        self.finished = True
        self.result.status = status
        self.result.elapsed_s = time.time() - self.started_at

        payload = self.result.to_dict()
        stamp = time.strftime("%Y%m%d_%H%M%S")

        # 落盘优先：即使后续发消息/关 env 出问题，结果也必须先安全落地。
        path = None
        try:
            path = self.result.dump(self.cfg.result_dir, stamp)
        except Exception as exc:  # noqa: BLE001
            _log(f"结果落盘失败：{type(exc).__name__}: {exc}")

        for action in (
            lambda: self.node.send_output(
                "benchmark_result", pa.array([json.dumps(payload, ensure_ascii=False)])
            ),
            lambda: self._policy_command("stop"),
            self.adapter.close,
        ):
            try:
                action()
            except Exception as exc:  # noqa: BLE001
                _log(f"收尾步骤失败（忽略）：{type(exc).__name__}: {exc}")

        _log(
            f"批次 {status}：{self.result.successes}/{self.result.total_episodes} "
            f"= {payload['success_rate']:.1%}，耗时 {self.result.elapsed_s:.0f}s"
        )
        if path:
            _log(f"结果已落盘：{path}")


def main() -> int:
    config_path = os.environ.get("BENCHMARK_CONFIG", os.path.join(_HERE, "config.yaml"))
    cfg = load_config(config_path)

    adapter = LiberoAdapter(
        suite_name=cfg.suite,
        camera_height=cfg.camera_height,
        camera_width=cfg.camera_width,
        seed=cfg.seed,
        layout_mode=cfg.layout_mode,
    )
    # fail fast：越界的 task/init_state 必须在建 node 之前炸掉，不能留到 episode 推进时。
    resolve_and_validate_plan(cfg, adapter)

    node = Node()
    runner = BenchmarkRunner(node, cfg, adapter)
    _log(
        f"suite={cfg.suite} tasks={cfg.task_ids} init_states={len(cfg.init_state_ids)} "
        f"num_runs={cfg.num_runs} layout_mode={cfg.layout_mode} seed={cfg.seed} "
        f"→ {len(runner.plan)} episodes"
    )

    rc = 0
    try:
        runner._policy_command("start")
        runner.advance()

        for event in node:
            if event["type"] == "INPUT":
                if event["id"] == "action":
                    try:
                        runner.on_action(JointCommand.from_arrow(event["value"]))
                    except Exception as exc:  # noqa: BLE001 - 单步失败不该打死整批
                        runner.fail_current("step_error", f"{type(exc).__name__}: {exc}")
            elif event["type"] == "STOP":
                runner.finish("cancelled")
                break
            elif event["type"] == "ERROR":
                _log(f"dora error: {event.get('error')}")
                runner.finish("failed")
                rc = 1
                break

            if runner.finished:
                break
    finally:
        # 无论怎么退出（异常/中断/提前 break），结果都必须落盘。
        runner.finish("cancelled")

    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
