"""端点描述符与错误码。

`benchmark.controller` 上挂两个 operation：

- describe = Query：读 adapter 能力清单（进程内直读，无缓存/无心跳）；
- run = Session：一个 session = 一个批次，stoppable（stop 在 episode 边界落地）。

两个 operation 挂在同一个 endpoint 上，**不要拆成两个 endpoint**：它们共享
同一张 invocation 表与同一份 capabilities，拆开就要在两个对象之间同步状态。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from forge_tool import (
    TOOL_ENDPOINT_PROTOCOL,
    ToolEndpointDescriptor,
    ToolEndpointError,
    ToolError,
    ToolOperationDescriptor,
)

BENCH_ENDPOINT_ID = "benchmark.controller"
OP_DESCRIBE = "describe"
OP_RUN = "run"

BENCH_DESCRIPTOR = ToolEndpointDescriptor(
    protocol_version=TOOL_ENDPOINT_PROTOCOL,
    endpoint_id=BENCH_ENDPOINT_ID,
    operations=(
        ToolOperationDescriptor(name=OP_DESCRIBE, semantics="query", max_concurrency=1),
        # session 的硬约束（forge_tool descriptor 校验，写错构造时就抛）：
        # cancellable 必须 False（session 只认 stop）、status_supported 必须 True。
        # stoppable=True 才是 session 的停法：stop → 相位 stopping → 批次在
        # episode 边界落停 → 终态 stopped。
        # max_concurrency=1 是硬约束：一台机器一个仿真器，第二个 run 只能拒，
        # 绝不排队 —— 排队会让 Agent 以为"已接受"，然后等一个几十分钟后才开始的东西。
        ToolOperationDescriptor(
            name=OP_RUN,
            semantics="session",
            stoppable=True,
            status_supported=True,
            max_concurrency=1,
        ),
    ),
)

# 错误码。retryable 的语义是"同样的参数过一会儿再来可能成"，
# 不是"这次失败了但可以再试" —— 参数错了重试多少次都一样，必须是 False。
ERR_BUSY = "FORGE_ENDPOINT_BUSY"
ERR_CONFLICT = "FORGE_EXECUTION_CONFLICT"
ERR_NOT_READY = "BENCH_NOT_READY"
ERR_INVALID_ARGUMENT = "BENCH_INVALID_ARGUMENT"
ERR_UNKNOWN_INVOCATION = "BENCH_UNKNOWN_INVOCATION"

_RETRYABLE = {ERR_BUSY, ERR_NOT_READY}


def tool_error(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> ToolError:
    return ToolError(
        code=code,
        message=message,
        retryable=code in _RETRYABLE,
        details=dict(details or {}),
    )


def endpoint_error(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> ToolEndpointError:
    """构造"执行开始前"的拒绝。

    ⚠️ 只能在**副作用发生之前**抛。一旦 session 已建、已进 pending 队列，
    就必须走终态 ToolResult，不能再抛这个异常 —— 那会让 Agent 以为什么都没
    发生，而实际上仿真器已经在跑了。
    """
    return ToolEndpointError(tool_error(code, message, details=details))


__all__ = [
    "BENCH_DESCRIPTOR",
    "BENCH_ENDPOINT_ID",
    "ERR_BUSY",
    "ERR_CONFLICT",
    "ERR_INVALID_ARGUMENT",
    "ERR_NOT_READY",
    "ERR_UNKNOWN_INVOCATION",
    "OP_DESCRIBE",
    "OP_RUN",
    "endpoint_error",
    "tool_error",
]
