"""对端(benchmark 节点)的镜像状态。

这一层是整个方案 A 的关键:endpoint 侧**永远不等 benchmark 侧的回执**。

原因是 `binding.dispatch_input` 是用 `asyncio.Runner().run(...)` 同步跑到底的
(照抄 relative_pose_policy_node/runner.py 的 `_handle_tool`),而它本身就在 dora
事件循环里。如果 `start()` 去 await 一条来自 bench_report 的消息,那条消息只能
由事件循环送达,而事件循环正被它自己堵着 —— **必然死锁,而且完全静默**。

所以:capabilities 由 benchmark 节点**主动推**,这里只做缓存;所有 endpoint
方法只读缓存、只往 outbox 里塞命令,立刻返回。
"""

from __future__ import annotations

import logging
from typing import Any

from .descriptor import (
    ERR_INVALID_ARGUMENT,
    ERR_NODE_UNAVAILABLE,
    endpoint_error,
    tool_error,
)
from .invocations import InvocationTable
from . import wire

logger = logging.getLogger(__name__)

# 心跳阈值。benchmark 节点每 1s 发一次,连续 3 个周期没有就判它掉了。
# 判定结果是 unknown 而不是 failed:副作用已经发生(GPU 时间烧了、result.json
# 可能已落盘),Agent 不能盲目重试。
DEFAULT_HEARTBEAT_TIMEOUT_S = 3.0


class BenchPeer:
    """capabilities 缓存 + 心跳判活 + 出站命令队列 + invocation 表。"""

    def __init__(
        self,
        *,
        table: InvocationTable | None = None,
        heartbeat_timeout_s: float = DEFAULT_HEARTBEAT_TIMEOUT_S,
    ) -> None:
        self.table = table or InvocationTable()
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.capabilities: dict[str, Any] | None = None
        self.last_seen: float | None = None
        self.outbox: list[dict[str, Any]] = []

    # ---- 判活 ----

    def alive(self, now: float) -> bool:
        return self.last_seen is not None and (now - self.last_seen) <= self.heartbeat_timeout_s

    def ready(self, now: float) -> bool:
        """gateway 的 readiness 看这个:仿真器没起来之前不该是 ready。"""
        return self.capabilities is not None and self.alive(now)

    # ---- 出站 ----

    def send(self, command: dict[str, Any]) -> None:
        self.outbox.append(command)

    def drain(self) -> list[dict[str, Any]]:
        items = self.outbox
        self.outbox = []
        return items

    # ---- 入站 ----

    def on_report(self, message: dict[str, Any], now: float) -> None:
        """处理一条 bench_report。任何一条消息都算"对端还活着"。"""
        self.last_seen = now
        kind = message.get("type")

        if kind == wire.REPORT_CAPABILITIES:
            self.capabilities = {k: v for k, v in message.items() if k not in ("v", "type")}
            return

        if kind == wire.REPORT_HEARTBEAT:
            return

        invocation_id = message.get("invocation_id")
        if not isinstance(invocation_id, str) or self.table.by_id(invocation_id) is None:
            # 陈旧回执(比如上一轮进程的)。丢掉,但别静默 —— 它通常意味着
            # invocation_id 关联写错了。
            logger.warning("丢弃无主 bench_report: type=%s id=%r", kind, invocation_id)
            return

        if kind == wire.REPORT_ACCEPTED:
            self.table.mark_running(invocation_id, {"phase_detail": "running"})
        elif kind == wire.REPORT_REJECTED:
            self.table.finish(
                invocation_id,
                "failed",
                error=tool_error(
                    str(message.get("code") or ERR_INVALID_ARGUMENT),
                    str(message.get("message") or "benchmark 节点拒绝了该计划"),
                ),
            )
        elif kind == wire.REPORT_PROGRESS:
            self.table.update_details(
                invocation_id,
                {
                    "episode_index": message.get("episode_index"),
                    "total": message.get("total"),
                    "successes": message.get("successes"),
                },
            )
        elif kind == wire.REPORT_RESULT:
            self._apply_result(invocation_id, message)
        else:
            logger.warning("未知 bench_report 类型:%r", kind)

    def _apply_result(self, invocation_id: str, message: dict[str, Any]) -> None:
        status = str(message.get("status") or "failed")
        payload = message.get("payload")
        outputs = dict(payload) if isinstance(payload, dict) else {}
        # benchmark 侧的批次状态 → Forge 的 phase
        phase = {"succeeded": "completed", "failed": "failed", "cancelled": "cancelled"}.get(
            status, "failed"
        )
        error = None
        if phase == "failed":
            error = tool_error(
                "BENCH_RUN_FAILED",
                str(message.get("message") or "批次未正常结束"),
                details={"status": status},
            )
        self.table.finish(invocation_id, phase, outputs=outputs, error=error)

    # ---- 每 tick 调一次 ----

    def tick(self, now: float) -> None:
        """对端掉线时把在跑的 invocation 收成 unknown。

        不做这件事的后果:benchmark 节点崩了以后 endpoint 永远报 running,
        Agent 永远轮不到终态,挂在那里。
        """
        item = self.table.active()
        if item is None or self.alive(now):
            return
        self.table.finish(
            item.key.invocation_id,
            "unknown",
            error=tool_error(
                ERR_NODE_UNAVAILABLE,
                f"benchmark 节点已 {self.heartbeat_timeout_s:.0f}s 无心跳;"
                "本次执行的最终结果无法确定,禁止盲目重试",
            ),
        )


# ---- 计划归一化与校验 ----
#
# 这段逻辑和 3.11 侧 `_core/config.py:resolve_and_validate_plan` 是重复的,
# **这是刻意的**:endpoint 拿缓存的 capabilities 就能挡住绝大多数越界,把错误
# 变成同步的 BENCH_INVALID_ARGUMENT,而不是"接受了、跑起来了、然后失败"。
# 3.11 侧那一份是权威兜底,不能删。


def normalize_plan(arguments: dict[str, Any], capabilities: dict[str, Any]) -> dict[str, Any]:
    tasks = capabilities.get("tasks") or []
    if not tasks:
        raise endpoint_error(ERR_INVALID_ARGUMENT, "capabilities 里没有任何 task")
    counts = {int(t["task_id"]): int(t["num_init_states"]) for t in tasks}

    task_ids = arguments.get("task_ids")
    if task_ids is None:
        task_ids = sorted(counts)
    task_ids = [int(t) for t in task_ids]
    if not task_ids:
        raise endpoint_error(ERR_INVALID_ARGUMENT, "task_ids 不能是空数组;要全集请传 null")
    bad = [t for t in task_ids if t not in counts]
    if bad:
        raise endpoint_error(
            ERR_INVALID_ARGUMENT,
            f"越界 task_ids: {bad};可用范围 0..{max(counts)}",
            details={"available_task_ids": sorted(counts)},
        )

    init_state_ids = arguments.get("init_state_ids")
    if init_state_ids is None:
        init_state_ids = list(range(min(counts[t] for t in task_ids)))
    init_state_ids = [int(i) for i in init_state_ids]
    if not init_state_ids:
        raise endpoint_error(
            ERR_INVALID_ARGUMENT, "init_state_ids 不能是空数组;要全集请传 null"
        )
    for t in task_ids:
        # 逐 task 校验:不同 task 的 init state 数未必相同,拿 task_ids[0]
        # 代表全体是错的(3.11 侧那份注释里踩过)。
        bad = [i for i in init_state_ids if not 0 <= i < counts[t]]
        if bad:
            raise endpoint_error(
                ERR_INVALID_ARGUMENT,
                f"task {t} 只有 {counts[t]} 个 init state,越界 init_state_ids: {bad}",
            )

    defaults = capabilities.get("defaults") or {}
    num_runs = int(arguments.get("num_runs", defaults.get("num_runs", 1)))
    max_steps = int(arguments.get("max_steps", defaults.get("max_steps", 300)))
    seed = int(arguments.get("seed", defaults.get("seed", 0)))
    if num_runs < 1:
        raise endpoint_error(ERR_INVALID_ARGUMENT, f"num_runs 必须 ≥1,收到 {num_runs}")
    if max_steps < 1:
        raise endpoint_error(ERR_INVALID_ARGUMENT, f"max_steps 必须 ≥1,收到 {max_steps}")

    limits = capabilities.get("limits") or {}
    total = len(task_ids) * len(init_state_ids) * num_runs
    cap_total = int(limits.get("max_total_episodes", 0) or 0)
    if cap_total and total > cap_total:
        # 没有这道闸,Agent 会一口气开 150 集然后跑 8 小时,而且不报错。
        raise endpoint_error(
            ERR_INVALID_ARGUMENT,
            f"本次将跑 {total} 集,超过上限 {cap_total};请缩小 task_ids/init_state_ids/num_runs",
            details={"requested_episodes": total, "max_total_episodes": cap_total},
        )
    cap_steps = int(limits.get("max_max_steps", 0) or 0)
    if cap_steps and max_steps > cap_steps:
        raise endpoint_error(
            ERR_INVALID_ARGUMENT, f"max_steps {max_steps} 超过上限 {cap_steps}"
        )

    return {
        "task_ids": task_ids,
        "init_state_ids": init_state_ids,
        "num_runs": num_runs,
        "seed": seed,
        "max_steps": max_steps,
        "total_episodes": total,
    }


__all__ = ["DEFAULT_HEARTBEAT_TIMEOUT_S", "BenchPeer", "normalize_plan"]
