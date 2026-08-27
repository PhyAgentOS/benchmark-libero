"""两个 operation 的实现:bench.describe(Query)与 bench.run(Action)。

**所有方法都不 await 外部 I/O**,只读 BenchPeer 的缓存、只往 outbox 塞命令。
理由见 peer.py 顶部那段死锁分析。
"""

from __future__ import annotations

from typing import Any

from forge_tool import (
    ToolAccepted,
    ToolContext,
    ToolControlResponse,
    ToolExecutionKey,
    ToolExecutionStatus,
    ToolRequest,
    ToolResult,
    ToolResultResponse,
)

from .descriptor import (
    ERR_BUSY,
    ERR_NOT_READY,
    ERR_UNKNOWN_INVOCATION,
    endpoint_error,
    tool_error,
)
from .peer import BenchPeer, normalize_plan
from . import wire

# describe 的输出白名单。capabilities 是 benchmark 节点推上来的,直接透传等于
# 把内部字段(record_dir 之类)漏给 Agent,而且 schema 会对不上。
_DESCRIBE_FIELDS = (
    "benchmark",
    "suite",
    "policy_id",
    "env_cfg_type",
    "action_type",
    "num_tasks",
    "tasks",
    "joint_names",
    "camera_ids",
    "defaults",
    "limits",
)


class BenchDescribeEndpoint:
    """QueryToolEndpoint:只读缓存,永不等待。"""

    def __init__(self, peer: BenchPeer, clock) -> None:
        self.peer = peer
        self.clock = clock

    async def query(self, request: ToolRequest, context: ToolContext) -> ToolResult:
        now = self.clock()
        if not self.peer.ready(now):
            # Query 的失败是终态 ToolResult,不是异常 —— 异常是"执行开始前的拒绝",
            # 而 Query 没有"开始执行"这个阶段。
            return ToolResult(
                status="failed",
                error=tool_error(
                    ERR_NOT_READY,
                    "benchmark 节点尚未上报 capabilities(仿真器可能还在启动)",
                ),
            )
        caps = self.peer.capabilities or {}
        outputs = {name: caps.get(name) for name in _DESCRIBE_FIELDS if name in caps}
        return ToolResult(status="succeeded", outputs=outputs)


class BenchRunEndpoint:
    """ActionToolEndpoint:start 只登记不执行。"""

    def __init__(self, peer: BenchPeer, clock) -> None:
        self.peer = peer
        self.clock = clock

    async def start(
        self,
        request: ToolRequest,
        context: ToolContext,
        events: Any,
    ) -> ToolAccepted:
        now = self.clock()
        if not self.peer.ready(now):
            raise endpoint_error(
                ERR_NOT_READY, "benchmark 节点尚未就绪,无法开始评测"
            )

        # 先查忙,再校验参数:忙的时候连参数都不用看,而且 retryable 语义不同。
        active = self.peer.table.active()
        if active is not None:
            raise endpoint_error(
                ERR_BUSY,
                f"已有 invocation {active.key.invocation_id} 在跑;一台机器只有一个仿真器",
                details={"active_invocation_id": active.key.invocation_id},
            )

        plan = normalize_plan(dict(request.arguments), self.peer.capabilities or {})

        # ⚠️ 这行以下就有副作用了(命令会被发出去),再往后不许抛 ToolEndpointError。
        key = context.execution_key
        self.peer.table.create(key, plan)
        self.peer.send(wire.run_command(key.invocation_id, plan))
        return ToolAccepted(
            details={
                "invocation_id": key.invocation_id,
                "total_episodes": plan["total_episodes"],
            }
        )

    async def cancel(
        self,
        key: ToolExecutionKey,
        reason: str | None = None,
    ) -> ToolControlResponse:
        item = self.peer.table.get(key)
        if item is None:
            return ToolControlResponse(
                command="cancel",
                status="rejected",
                error=tool_error(
                    ERR_UNKNOWN_INVOCATION, f"没有 invocation {key.invocation_id!r} 的记录"
                ),
            )
        if item.terminal:
            # 已经终结:不是错误,只是没得可取消。
            return ToolControlResponse(command="cancel", status="terminal")
        if not item.cancel_requested:
            item.cancel_requested = True
            self.peer.send(wire.cancel_command(key.invocation_id, reason))
        # accepted 只表示"取消请求被收下",不表示已经停了 —— 调用方必须继续看
        # status/result。这是 ToolControlResponse 文档里写死的语义。
        return ToolControlResponse(command="cancel", status="accepted")

    async def status(self, key: ToolExecutionKey) -> ToolExecutionStatus:
        return self.peer.table.status_for(key)

    async def result(self, key: ToolExecutionKey) -> ToolResultResponse:
        return self.peer.table.result_for(key)


__all__ = ["BenchDescribeEndpoint", "BenchRunEndpoint"]
