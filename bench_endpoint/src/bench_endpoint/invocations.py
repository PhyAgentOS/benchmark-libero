"""一次 bench.run 的状态表。

**不变式(handler 依赖它,破了会在 forge_tool 内部抛)**:
只要 `status()` 返回的 phase 是终态,`result()` 就必须能给出 available 的
ToolResult,且两者按 `validate_execution_result` 的映射对齐:

    completed → succeeded    failed → failed
    cancelled → cancelled    unknown → unknown

所以这里把 phase 和 result **成对写入**,不提供只改其中一个的接口。
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from forge_tool import (
    ToolError,
    ToolExecutionKey,
    ToolExecutionStatus,
    ToolResult,
    ToolResultResponse,
)

from .descriptor import ERR_UNKNOWN_INVOCATION, tool_error

logger = logging.getLogger(__name__)

# phase → ToolResult.status,与 forge_tool 的 _TERMINAL_PHASE_RESULTS 一致
_TERMINAL = {
    "completed": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
    "unknown": "unknown",
}

# 留几条历史。够 Agent 对账即可 —— 留多了没用,Agent 不会回头看上上次。
DEFAULT_RETENTION = 8


@dataclass
class Invocation:
    key: ToolExecutionKey
    plan: dict[str, Any]
    phase: str = "accepted"
    result: ToolResult | None = None
    details: dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False

    @property
    def terminal(self) -> bool:
        return self.phase in _TERMINAL

    def status(self) -> ToolExecutionStatus:
        error = self.result.error if (self.result is not None and self.result.error) else None
        # ToolExecutionStatus 的校验:只有 failed/unknown 允许带 error,
        # 其余带了就抛。所以这里必须按 phase 过滤,不能无脑透传。
        if self.phase not in ("failed", "unknown"):
            error = None
        elif error is None:
            error = tool_error("BENCH_INTERNAL", f"{self.phase} 缺少 error")
        return ToolExecutionStatus(phase=self.phase, error=error, details=dict(self.details))


class InvocationTable:
    """按 invocation_id 索引。同一时刻最多一个非终态条目。"""

    def __init__(self, retention: int = DEFAULT_RETENTION) -> None:
        self._items: OrderedDict[str, Invocation] = OrderedDict()
        self._retention = retention

    # ---- 查询 ----

    def get(self, key: ToolExecutionKey) -> Invocation | None:
        return self._items.get(key.invocation_id)

    def by_id(self, invocation_id: str) -> Invocation | None:
        return self._items.get(invocation_id)

    def active(self) -> Invocation | None:
        for item in self._items.values():
            if not item.terminal:
                return item
        return None

    # ---- 写入 ----

    def create(self, key: ToolExecutionKey, plan: dict[str, Any]) -> Invocation:
        if self.active() is not None:
            raise RuntimeError("已有未终结的 invocation;调用方必须先拒掉")
        item = Invocation(key=key, plan=dict(plan), details={"phase_detail": "dispatched"})
        self._items[key.invocation_id] = item
        while len(self._items) > self._retention:
            oldest, _ = next(iter(self._items.items()))
            # 只淘汰已终结的,免得把在跑的那条挤掉
            victim = self._items[oldest]
            if not victim.terminal:
                break
            self._items.pop(oldest)
        return item

    def mark_running(self, invocation_id: str, details: dict[str, Any] | None = None) -> None:
        item = self._items.get(invocation_id)
        if item is None or item.terminal:
            return
        item.phase = "running"
        if details:
            item.details.update(details)

    def update_details(self, invocation_id: str, details: dict[str, Any]) -> None:
        item = self._items.get(invocation_id)
        if item is None or item.terminal:
            return
        item.details.update(details)

    def finish(
        self,
        invocation_id: str,
        phase: str,
        *,
        outputs: dict[str, Any] | None = None,
        error: ToolError | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """把一条 invocation 写成终态。phase 与 result.status 成对写入。"""
        if phase not in _TERMINAL:
            raise ValueError(f"{phase!r} 不是终态")
        item = self._items.get(invocation_id)
        if item is None:
            return
        if item.terminal:
            # 已经终结了还收到终态 —— 按协议不能改写，但**必须吼一声**：
            # 最常见的成因是心跳阈值太紧，把正常的长阻塞误判成掉线，于是真结果
            # 到达时被丢掉，而仿真器其实跑完了、result.json 也落盘了。
            # 静默返回的话这种劈叉永远查不出来。
            logger.warning(
                "invocation %s 已是终态 %s，丢弃迟到的 %s 回执"
                "（多半是 heartbeat_timeout_s 设得太小）",
                invocation_id,
                item.phase,
                phase,
            )
            return
        status = _TERMINAL[phase]
        if status in ("failed", "unknown") and error is None:
            error = tool_error("BENCH_INTERNAL", f"{phase} 未给出 error")
        if status not in ("failed", "unknown"):
            error = None
        item.phase = phase
        item.result = ToolResult(status=status, outputs=dict(outputs or {}), error=error)
        if details:
            item.details.update(details)

    # ---- 对 handler 的两个出口 ----

    def status_for(self, key: ToolExecutionKey) -> ToolExecutionStatus:
        item = self.get(key)
        if item is None:
            return ToolExecutionStatus(
                phase="unknown",
                error=tool_error(
                    ERR_UNKNOWN_INVOCATION,
                    f"没有 invocation {key.invocation_id!r} 的记录",
                ),
            )
        return item.status()

    def result_for(self, key: ToolExecutionKey) -> ToolResultResponse:
        item = self.get(key)
        if item is None:
            # 刻意不用 not_found:上面的 status_for 已经把未知 key 报成终态
            # unknown 了,handler 会立刻来要结果,此时返回 not_found 就自相矛盾。
            return ToolResultResponse(
                status="available",
                result=ToolResult(
                    status="unknown",
                    error=tool_error(
                        ERR_UNKNOWN_INVOCATION,
                        f"没有 invocation {key.invocation_id!r} 的记录",
                    ),
                ),
            )
        if not item.terminal or item.result is None:
            return ToolResultResponse(status="pending")
        return ToolResultResponse(status="available", result=item.result)


__all__ = ["DEFAULT_RETENTION", "Invocation", "InvocationTable"]
