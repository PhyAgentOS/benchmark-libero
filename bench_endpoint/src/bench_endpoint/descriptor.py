"""端点描述符与错误码。

`bench.describe` 是 Query,`bench.run` 是 Action —— **两个 operation 挂在同一个
endpoint 上**,不要拆成两个 endpoint:它们共享同一份 capabilities 缓存和同一张
invocation 表,拆开就要在两个对象之间同步状态。
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

BENCH_ENDPOINT_ID = "bench.runner"
OP_DESCRIBE = "describe"
OP_RUN = "run"

BENCH_DESCRIPTOR = ToolEndpointDescriptor(
    protocol_version=TOOL_ENDPOINT_PROTOCOL,
    endpoint_id=BENCH_ENDPOINT_ID,
    operations=(
        ToolOperationDescriptor(name=OP_DESCRIBE, semantics="query", max_concurrency=1),
        # max_concurrency=1 是硬约束:一台机器一个 Isaac Sim,第二个 run 只能拒,
        # 绝不排队 —— 排队会让 Agent 以为"已接受",然后等一个几十分钟后才开始的东西。
        ToolOperationDescriptor(
            name=OP_RUN,
            semantics="action",
            cancellable=True,
            status_supported=True,  # action 必须支持 status,否则 descriptor 校验就炸
            max_concurrency=1,
        ),
    ),
)

# 错误码。retryable 的语义是"同样的参数过一会儿再来可能成",
# 不是"这次失败了但可以再试" —— 参数错了重试多少次都一样,必须是 False。
ERR_BUSY = "FORGE_BUSY"
ERR_NOT_READY = "BENCH_NOT_READY"
ERR_INVALID_ARGUMENT = "BENCH_INVALID_ARGUMENT"
ERR_NODE_UNAVAILABLE = "BENCH_NODE_UNAVAILABLE"
ERR_UNKNOWN_INVOCATION = "BENCH_UNKNOWN_INVOCATION"

_RETRYABLE = {ERR_BUSY, ERR_NOT_READY, ERR_NODE_UNAVAILABLE}


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

    ⚠️ 只能在**副作用发生之前**抛。一旦 bench_cmd 已经发出去,就必须走终态
    ToolResult,不能再抛这个异常 —— 那会让 Agent 以为什么都没发生,而实际上
    仿真器已经在跑了。
    """
    return ToolEndpointError(tool_error(code, message, details=details))


__all__ = [
    "BENCH_DESCRIPTOR",
    "BENCH_ENDPOINT_ID",
    "ERR_BUSY",
    "ERR_INVALID_ARGUMENT",
    "ERR_NODE_UNAVAILABLE",
    "ERR_NOT_READY",
    "ERR_UNKNOWN_INVOCATION",
    "OP_DESCRIBE",
    "OP_RUN",
    "endpoint_error",
    "tool_error",
]
