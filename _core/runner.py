"""批次调度核心。通用逻辑，**不认识任何具体 benchmark**。

时序契约：action 驱动的 1:1 锁步。发布一次观测 → 等 policy 的 action → step →
再发布观测。policy 内部的 action chunk 由它自己排队，本节点按 1:1 消费，
`awaiting_action == False` 时到达的多余 action 直接丢弃。
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pyarrow as pa
from forge_msgs import Image, JointCommand, JointState, PolicyCommand

from .adapter import default_aggregate
from .config import BaseBenchmarkConfig, derive_episode_seed
from .video import EpisodeRecorder
from .result import BenchmarkResult, EpisodeRecord


class BenchmarkRunner:
    def __init__(
        self,
        node,
        cfg: BaseBenchmarkConfig,
        adapter,
        log=print,
        *,
        result_tag: str | None = None,
        close_adapter_on_finish: bool = True,
    ) -> None:
        self.node = node
        self.cfg = cfg
        self.adapter = adapter
        self._log = log
        # 两个开关都只服务于 serve 模式，默认值即 autorun 的原行为：
        #
        # result_tag=None            -> 结果文件名沿用 `<suite>_<时间戳>.json`，与旧版逐字一致。
        #                               serve 模式传 invocation_id：文件名的时间戳**只到秒**，
        #                               同一秒内的两次 invocation 会写到同一路径、前一次被静默覆盖
        #                               （autorun 一个进程只跑一批，撞不上，所以这个坑只在 serve 里出现）。
        # close_adapter_on_finish=True -> finish() 收尾时关 adapter，与旧版一致。
        #                               serve 模式传 False：Skill Runtime 的生命周期长于单次
        #                               invocation，关了 env 下一次 bench.run 就要重载整个场景；
        #                               真正的释放交给 run_node 退出时统一做。
        self.result_tag = result_tag
        self.close_adapter_on_finish = close_adapter_on_finish
        self.plan = cfg.episode_plan()
        self.result = BenchmarkResult(
            benchmark=cfg.benchmark,
            suite=cfg.suite,
            max_steps=cfg.max_steps,
            total_episodes=len(self.plan),
            control_mode=getattr(cfg, "control_mode", ""),
            layout_mode=getattr(cfg, "layout_mode", "per_episode"),
            seed=cfg.seed,
            reproducibility=getattr(adapter, "reproducibility", lambda: {})(),
        )
        self.plan_index = -1
        self.started_at = time.time()

        self.step_idx = 0
        self.episode_return = 0.0
        self.episode_seed: int | None = None
        self.last_metrics: dict[str, Any] = {}
        self.awaiting_action = False
        self.obs_sent_at: float | None = None
        self.latencies: list[float] = []
        self.finished = False
        # serve 模式的停止请求：置位后当前 episode 跑完就不再开下一集，批次以
        # stopped 收尾（已有完成集作 partial）。消费点在 start_next_episode。
        self.stop_requested = False
        self.stop_reason: str | None = None
        # finish() 时带上的摘要；endpoint 的七字段输出读它。
        self.failure_summary: str | None = None
        # finish() 里 dump() 的最终落盘路径。只做记录，不参与判分。
        self.result_path: str | None = None

        self._joint_names = list(adapter.state_joint_names())
        self.recorder = EpisodeRecorder(
            cfg.record_dir, fps=getattr(adapter, "RECORD_FPS", 20.0), log=self._log
        )

    # ---- policy 生命周期 ----

    def policy_command(self, command: str, inputs: dict[str, Any] | None = None) -> None:
        msg = PolicyCommand.from_inputs(
            policy_id=self.cfg.policy_id, command=command, inputs=inputs or {}
        )
        self.node.send_output("policy_command", msg.to_arrow())

    # ---- 观测发布 ----

    def _publish_obs(self, obs: dict[str, Any]) -> None:
        state = self.adapter.to_state(obs)
        if state is not None:
            self.node.send_output(
                "proprio_state",
                JointState.from_np(
                    np.asarray(state, dtype=np.float64), self._joint_names, "position"
                ).to_arrow(),
            )
        frames = self.adapter.to_images(obs)
        for output_id, frame in frames.items():
            self.node.send_output(output_id, Image.from_numpy(frame, "rgb8").to_arrow())
        self.recorder.add(frames)

        self.obs_sent_at = time.time()
        self.awaiting_action = True

    def emit_status(self, text: str) -> None:
        self.node.send_output("benchmark_status", pa.array([text]))
        self._log(text)

    # ---- episode 推进 ----

    def request_stop(self, reason: str | None = None) -> None:
        """请求在 episode 边界停，不在动作中途停 —— episode 判分要求环境状态
        干净，中途掐掉会让当前集半途而废（那才是数据污染）。"""
        if self.finished:
            return
        self.stop_requested = True
        self.stop_reason = reason

    def advance(self) -> None:
        """推进到下一 episode。任何异常都收敛成"批次失败"，绝不冲出事件循环。"""
        if self.finished:
            return
        try:
            self.start_next_episode()
        except Exception as exc:  # noqa: BLE001
            self._log(f"启动下一 episode 失败：{type(exc).__name__}: {exc}")
            self.finish("failed", failure_summary=f"启动下一 episode 失败：{type(exc).__name__}: {exc}")

    def start_next_episode(self) -> bool:
        self.plan_index += 1
        if self.plan_index >= len(self.plan):
            self.finish("succeeded")
            return False

        # stop 的消费点：边界到了就不再开下一集，批次以 stopped 收尾。
        # 已有完成集会在 result 里,由 endpoint 转成 partial 输出。
        if self.stop_requested:
            self.finish("stopped", failure_summary=self.stop_reason)
            return False

        task_id, init_state_id, run_index = self.plan[self.plan_index]
        # 可选钩子：指令依赖整个 episode 身份而非仅 task 时用它（CALVIN 的每条
        # 序列有自己的首个子任务指令）。不实现就退回按 task 取。
        ep_instr = getattr(self.adapter, "episode_instruction", None)
        instruction = (
            ep_instr(task_id, init_state_id, run_index)
            if callable(ep_instr)
            else self.adapter.task_language(task_id)
        )
        self.episode_seed = derive_episode_seed(
            self.cfg.seed, task_id, init_state_id, run_index
        )

        # 指令必须逐 episode 下发：多数 benchmark 每个 task 指令不同，而 policy
        # 配置里那句是写死的。reset_scene 清上一局残留的队列与缓存（不能用 reset，
        # 那会把 policy 打回 idle 相）。
        if self.cfg.send_instruction:
            self.policy_command("set_instruction", {"instruction": instruction})
        self.policy_command("reset_scene")

        obs = self.adapter.reset_episode(
            task_id=task_id,
            init_state_id=init_state_id,
            run_index=run_index,
            episode_seed=self.episode_seed,
        )

        self.step_idx = 0
        self.episode_return = 0.0
        self.last_metrics = {}
        self.latencies = []
        self.recorder.reset()
        self.emit_status(
            f"episode {self.plan_index + 1}/{len(self.plan)} 开始 "
            f"t{task_id}_i{init_state_id}_r{run_index} :: {instruction}"
        )
        self._publish_obs(obs)
        return True

    def on_action(self, command: JointCommand) -> None:
        if self.finished or not self.awaiting_action:
            return  # 非 1:1 的多余 action 直接丢弃

        action = np.asarray(command.position, dtype=np.float32).reshape(-1)
        if action.size == 0 or not np.all(np.isfinite(action)):
            self.fail_current("invalid_action", f"action 非法：size={action.size}")
            return

        if self.obs_sent_at is not None:
            self.latencies.append((time.time() - self.obs_sent_at) * 1000.0)
        self.awaiting_action = False

        outcome = self.adapter.step(action)
        self.step_idx += 1
        self.episode_return += outcome.reward
        self.last_metrics = outcome.metrics or {}

        # 每步查一次判定，不是 episode 结束才查
        if outcome.success or outcome.done:
            if outcome.success:
                termination = "success"
            elif self.last_metrics:
                # 长程任务：环境说结束了但没全对 → 部分完成
                termination = "partial"
            else:
                termination = "failed"
            self._record_episode(success=outcome.success, termination=termination)
            self.advance()
        elif self.step_idx >= self.cfg.max_steps:
            self._record_episode(success=False, termination="timed_out")
            self.advance()
        else:
            # 链式任务的子任务边界：换指令并让 policy 丢掉上一子任务的动作队列
            if outcome.next_instruction:
                if self.cfg.send_instruction:
                    self.policy_command(
                        "set_instruction", {"instruction": outcome.next_instruction}
                    )
                self.policy_command("reset_scene")
                self.emit_status(f"  ↳ 子任务切换 :: {outcome.next_instruction}")
            self._publish_obs(outcome.obs)

    # ---- episode 收尾 ----

    def _record_episode(
        self,
        success: bool,
        termination: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """只记录，不推进。推进由调用方显式 advance()。"""
        task_id, init_state_id, run_index = self.plan[self.plan_index]
        record = EpisodeRecord(
            episode_id=f"{self.cfg.suite}_t{task_id}_i{init_state_id}_r{run_index}",
            suite=self.cfg.suite,
            task_id=task_id,
            init_state_id=init_state_id,
            run_index=run_index,
            task_description=self.adapter.task_language(task_id),
            success=success,
            termination=termination,
            num_steps=self.step_idx,
            return_value=self.episode_return,
            episode_seed=self.episode_seed,
            mean_policy_latency_ms=(
                sum(self.latencies) / len(self.latencies) if self.latencies else None
            ),
            success_source=getattr(self.adapter, "SUCCESS_SOURCE", ""),
            metrics=self.last_metrics,
            error_code=error_code,
            error_message=error_message,
        )
        video = self.recorder.save(
            f"ep_t{task_id}_i{init_state_id}_r{run_index}_{termination}"
        )
        if video:
            record.metrics = {**record.metrics, "video": video}

        self.result.episodes.append(record)
        self.awaiting_action = False

        done_n = len(self.result.episodes)
        self.emit_status(
            f"{record.episode_id} {termination} ({self.step_idx} 步) | "
            f"累计 {self.result.successes}/{done_n}"
        )

    def fail_current(self, code: str, message: str) -> None:
        """当前 episode 记为失败并推进。异常路径的唯一入口。"""
        if self.finished or not 0 <= self.plan_index < len(self.plan):
            return
        self._log(f"episode 失败：{code} {message}")
        self._record_episode(
            success=False, termination="failed", error_code=code, error_message=message
        )
        self.advance()

    # ---- 批次收尾 ----

    def finish(self, status: str, failure_summary: str | None = None) -> None:
        if self.finished:
            return
        self.finished = True
        self.failure_summary = failure_summary
        self.result.status = status
        self.result.elapsed_s = time.time() - self.started_at

        # 聚合：adapter 有 aggregate 就用它，否则默认成功率
        ep_metrics = [e.metrics for e in self.result.episodes]
        agg = getattr(self.adapter, "aggregate", None)
        try:
            if callable(agg):
                primary, extra = agg(ep_metrics)
            else:
                primary, extra = default_aggregate(
                    self.result.successes, self.result.total_episodes, ep_metrics
                )
        except Exception as exc:  # noqa: BLE001
            self._log(f"指标聚合失败，退回默认：{type(exc).__name__}: {exc}")
            primary, extra = default_aggregate(
                self.result.successes, self.result.total_episodes, ep_metrics
            )
        self.result.primary_metric = primary
        self.result.metrics = extra

        payload = self.result.to_dict()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if self.result_tag:
            # 只有 serve 模式会传 tag；autorun 走不到这里，文件名与旧版逐字一致。
            safe = "".join(c for c in str(self.result_tag) if c.isalnum() or c in "-_")[:32]
            if safe:
                stamp = f"{stamp}_{safe}"

        # 落盘优先：即使后续发消息/关 env 出问题，结果也必须先安全落地。
        path = None
        try:
            path = self.result.dump(self.cfg.result_dir, stamp)
        except Exception as exc:  # noqa: BLE001
            self._log(f"结果落盘失败：{type(exc).__name__}: {exc}")
        self.result_path = path

        closers = (self.adapter.close,) if self.close_adapter_on_finish else ()
        for action in (
            lambda: self.node.send_output(
                "benchmark_result", pa.array([json.dumps(payload, ensure_ascii=False)])
            ),
            lambda: self.policy_command("stop"),
            *closers,
        ):
            try:
                action()
            except Exception as exc:  # noqa: BLE001
                self._log(f"收尾步骤失败（忽略）：{type(exc).__name__}: {exc}")

        pm = self.result.primary_metric
        self._log(
            f"批次 {status}：{pm.get('name','?')} = {pm.get('value')} "
            f"（{self.result.successes}/{self.result.total_episodes} 成功），"
            f"耗时 {self.result.elapsed_s:.0f}s"
        )
        if path:
            self._log(f"结果已落盘：{path}")
