"""benchmark.controller 的两个 operation 实现：describe(Query)与 run(Session)。

**start() 只登记不执行**：session 挂进 pending 队列，由 node.py 的事件循环拾取
并跑第一个 episode 的 reset（阻塞、重建仿真场景）。endpoint 方法本身永远不
await 外部 I/O，也不碰 dora —— 这是从 bench_endpoint 双节点架构继承过来的
死锁纪律：dispatch_input 是同步跑到底的，任何一步去等 dora 消息都会把事件
循环自己堵死，而且完全静默。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from forge_tool import (
    ToolAccepted,
    ToolContext,
    ToolControlResponse,
    ToolEvent,
    ToolExecutionKey,
    ToolExecutionStatus,
    ToolRequest,
    ToolResult,
    ToolResultResponse,
)

from _core.config import BaseBenchmarkConfig
from _core.session import BenchSession

from .descriptor import (
    ERR_BUSY,
    ERR_CONFLICT,
    ERR_INVALID_ARGUMENT,
    ERR_UNKNOWN_INVOCATION,
    endpoint_error,
    tool_error,
)
from .invocations import InvocationTable
from .result_shape import outputs_from_result, stopped_summary

# describe 的输出白名单。capabilities 里还有 benchmark 内部的其它字段
# （reproducibility 之类），直接透传等于把内部细节漏给 Agent，而且 schema 会对不上。
_DESCRIBE_FIELDS = (
    "benchmark",
    "suite",
    "policy_id",
    "num_tasks",
    "tasks",
    "joint_names",
    "camera_ids",
    "defaults",
    "limits",
)


@dataclass
class SessionHandle:
    """一次 run session 在端点侧与循环侧之间传递的唯一把手。

    生命周期：start() 创建 → pending 队列 → 循环拾取跑批 → on_finished 收终态。
    `reported` 保证终态只报一次（handler 的重复 start 会再来要 accepted，但
    on_finished 由循环驱动，双保险）。
    """

    key: ToolExecutionKey
    plan: dict[str, Any]
    arguments_json: str
    session: BenchSession
    emitter: Any = None
    stop_reason: str | None = None
    reported: bool = False


class BenchDescribeEndpoint:
    """QueryToolEndpoint：进程内直读 capabilities，永不等待。"""

    def __init__(self, caps: dict[str, Any]) -> None:
        self.caps = caps

    async def query(self, request: ToolRequest, context: ToolContext) -> ToolResult:
        caps = self.caps or {}
        outputs = {name: caps.get(name) for name in _DESCRIBE_FIELDS if name in caps}
        return ToolResult(status="succeeded", outputs=outputs)


class BenchSessionEndpoint:
    """SessionToolEndpoint：start 只登记，stop 在 episode 边界落停。"""

    def __init__(
        self,
        table: InvocationTable,
        node: Any,
        cfg: BaseBenchmarkConfig,
        adapter: Any,
        caps: dict[str, Any],
        *,
        emit: Callable[[SessionHandle, ToolEvent], None],
        log: Callable[[str], None] = print,
    ) -> None:
        self.table = table
        self.node = node
        self.cfg = cfg
        self.adapter = adapter
        self.caps = caps
        self._emit = emit
        self._log = log
        self._handles: dict[str, SessionHandle] = {}
        self._pending: list[SessionHandle] = []

    # ---- 循环侧接口（同步，不进 tool 分发）----

    def take_pending(self) -> list[SessionHandle]:
        items, self._pending = self._pending, []
        return items

    def on_finished(self, handle: SessionHandle) -> None:
        """把跑完的 session 收成终态，然后发终态事件。

        **顺序是硬约束**：先 table.finish（落 ToolResult），再 emit 终态
        ToolEvent。反过来 forge_tool 的 handler 在收事件时抛
        FORGE_ENDPOINT_TERMINAL_RESULT_UNAVAILABLE。
        """
        if handle.reported:
            return
        handle.reported = True
        result = handle.session.runner.result
        phase = {"succeeded": "completed", "failed": "failed", "stopped": "stopped"}.get(
            result.status, "failed"
        )
        payload = result.to_dict()
        completed = int(payload.get("completed_episodes") or 0)
        total = int(payload.get("total_episodes") or 0)

        failure_summary = handle.session.runner.failure_summary
        if phase == "failed":
            failure_summary = failure_summary or "批次未正常结束"
        elif phase == "stopped":
            failure_summary = stopped_summary(
                completed=completed,
                total=total,
                reason=handle.session.runner.stop_reason or handle.stop_reason,
            )
        outputs = outputs_from_result(
            run_id=handle.key.invocation_id,
            result=result,
            result_file=handle.session.result_path,
            failure_summary=failure_summary,
        )

        error = None
        if phase == "failed":
            error = tool_error(
                "BENCH_RUN_FAILED",
                failure_summary or "批次未正常结束",
                details={"status": result.status},
            )
        self.table.finish(handle.key.invocation_id, phase, outputs=outputs, error=error)

        event_type = {
            "completed": "executor_completed",
            "failed": "executor_failed",
            "stopped": "stopped",
        }[phase]
        self._emit(
            handle,
            ToolEvent(
                type=event_type,
                data={
                    "invocation_id": handle.key.invocation_id,
                    "status": result.status,
                    "result_file": handle.session.result_path,
                },
            ),
        )

    # ---- SessionToolEndpoint SPI ----

    async def start(
        self,
        request: ToolRequest,
        context: ToolContext,
        events: Any,
    ) -> ToolAccepted:
        key = context.execution_key
        arguments_json = json.dumps(
            dict(request.arguments),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        existing = self._handles.get(key.invocation_id)
        if existing is not None:
            if existing.arguments_json != arguments_json:
                # handler 的重复 key 缓存承接的是同参数重放；异参重放必须在这里
                # 拒掉，绝不能把新参数塞进一个已经登记的 session。
                raise endpoint_error(
                    ERR_CONFLICT,
                    "execution key 已用于一次参数不同的 session；重复 key 必须带相同参数",
                )
            return ToolAccepted(
                details={"invocation_id": key.invocation_id, "idempotent": True}
            )

        # 先查忙，再校验参数：忙的时候连参数都不用看，而且 retryable 语义不同。
        active = self.table.active()
        if active is not None:
            raise endpoint_error(
                ERR_BUSY,
                f"已有 session {active.key.invocation_id} 在跑；一台机器只有一个仿真器",
                details={"active_invocation_id": active.key.invocation_id},
            )

        plan = normalize_plan(dict(request.arguments), self.caps)

        # ⚠️ 这行以下就有副作用了（session 进队列、adapter.cfg 被回写），
        # 再往后不许抛 ToolEndpointError。BenchSession 内部还有一道权威兜底
        # 校验（对 adapter 的真实清单），它抛的 ValueError 在这里转成同步拒绝。
        try:
            session = BenchSession(
                self.node, self.cfg, self.adapter, plan, key.invocation_id, log=self._log
            )
        except ValueError as exc:
            raise endpoint_error(ERR_INVALID_ARGUMENT, f"{type(exc).__name__}: {exc}") from None

        self.table.create(key, plan)
        handle = SessionHandle(
            key=key,
            plan=plan,
            arguments_json=arguments_json,
            session=session,
            emitter=events,
        )
        self._handles[key.invocation_id] = handle
        self._pending.append(handle)
        return ToolAccepted(
            details={
                "invocation_id": key.invocation_id,
                "total_episodes": plan["total_episodes"],
            }
        )

    async def stop(
        self,
        key: ToolExecutionKey,
        reason: str | None = None,
    ) -> ToolControlResponse:
        item = self.table.get(key)
        if item is None:
            return ToolControlResponse(
                command="stop",
                status="rejected",
                error=tool_error(
                    ERR_UNKNOWN_INVOCATION, f"没有 session {key.invocation_id!r} 的记录"
                ),
            )
        if item.terminal:
            # 已经终结：不是错误，只是没得可停。
            return ToolControlResponse(command="stop", status="terminal")
        if not item.stop_requested:
            item.stop_requested = True
            self.table.mark_stopping(key.invocation_id, {"phase_detail": "stopping"})
            handle = self._handles.get(key.invocation_id)
            if handle is not None:
                handle.stop_reason = reason
                handle.session.request_stop(reason)
        # accepted 只表示"停止请求被收下"，不表示已经停了 —— 调用方必须继续看
        # status/result。这是 ToolControlResponse 文档里写死的语义。
        return ToolControlResponse(command="stop", status="accepted", details={"reason": reason})

    async def status(self, key: ToolExecutionKey) -> ToolExecutionStatus:
        return self.table.status_for(key)

    async def result(self, key: ToolExecutionKey) -> ToolResultResponse:
        return self.table.result_for(key)


# ---- 计划归一化与校验 ----
#
# 这段逻辑和 _core/config.py:resolve_and_validate_plan 是重复的，**这是刻意的**：
# endpoint 拿到 capabilities 就能挡住绝大多数越界，把错误变成同步的
# BENCH_INVALID_ARGUMENT，而不是"接受了、跑起来了、然后失败"。_core 那份是
# 权威兜底（对 adapter 的真实清单），不能删。


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
        raise endpoint_error(ERR_INVALID_ARGUMENT, "task_ids 不能是空数组；要全集请传 null")
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
            ERR_INVALID_ARGUMENT, "init_state_ids 不能是空数组；要全集请传 null"
        )
    for t in task_ids:
        # 逐 task 校验：不同 task 的 init state 数未必相同，拿 task_ids[0]
        # 代表全体是错的（_core 侧那份注释里踩过）。
        bad = [i for i in init_state_ids if not 0 <= i < counts[t]]
        if bad:
            raise endpoint_error(
                ERR_INVALID_ARGUMENT,
                f"task {t} 只有 {counts[t]} 个 init state，越界 init_state_ids: {bad}",
            )

    defaults = capabilities.get("defaults") or {}
    num_runs = int(arguments.get("num_runs", defaults.get("num_runs", 1)))
    max_steps = int(arguments.get("max_steps", defaults.get("max_steps", 300)))
    seed = int(arguments.get("seed", defaults.get("seed", 0)))
    if num_runs < 1:
        raise endpoint_error(ERR_INVALID_ARGUMENT, f"num_runs 必须 ≥1，收到 {num_runs}")
    if max_steps < 1:
        raise endpoint_error(ERR_INVALID_ARGUMENT, f"max_steps 必须 ≥1，收到 {max_steps}")

    limits = capabilities.get("limits") or {}
    total = len(task_ids) * len(init_state_ids) * num_runs
    cap_total = int(limits.get("max_total_episodes", 0) or 0)
    if cap_total and total > cap_total:
        # 没有这道闸，Agent 会一口气开满然后跑 8 小时，而且不报错。
        raise endpoint_error(
            ERR_INVALID_ARGUMENT,
            f"本次将跑 {total} 集，超过上限 {cap_total}；请缩小 task_ids/init_state_ids/num_runs",
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


__all__ = ["BenchDescribeEndpoint", "BenchSessionEndpoint", "SessionHandle", "normalize_plan"]
