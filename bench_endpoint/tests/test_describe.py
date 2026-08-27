"""bench.describe:只读缓存,永不等待。"""

from __future__ import annotations

import asyncio

from forge_tool import ToolContext, ToolExecutionKey, ToolRequest

from bench_endpoint.descriptor import BENCH_ENDPOINT_ID, ERR_NOT_READY, OP_DESCRIBE
from bench_endpoint.endpoints import BenchDescribeEndpoint


def _context(invocation_id: str = "inv-describe") -> ToolContext:
    return ToolContext(
        execution_key=ToolExecutionKey(invocation_id=invocation_id, attempt_id="a1"),
        tool_id="bench.describe",
        implementation_id="example.benchmark",
        endpoint_id=BENCH_ENDPOINT_ID,
        operation=OP_DESCRIBE,
    )


def _describe(peer, clock):
    endpoint = BenchDescribeEndpoint(peer, clock)
    return asyncio.run(endpoint.query(ToolRequest(arguments={}), _context()))


def test_describe_before_capabilities_is_not_ready(cold_peer, clock) -> None:
    result = _describe(cold_peer, clock)
    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == ERR_NOT_READY
    # 必须是 retryable:仿真器起来之后同样的调用就会成功
    assert result.error.retryable is True


def test_describe_returns_task_inventory(peer, clock) -> None:
    result = _describe(peer, clock)
    assert result.status == "succeeded"
    outputs = result.outputs
    assert outputs["benchmark"] == "example_benchmark"
    assert outputs["suite"] == "build_tower"
    assert outputs["num_tasks"] == 2
    assert [t["task_id"] for t in outputs["tasks"]] == [0, 1]
    assert outputs["tasks"][0]["num_init_states"] == 3
    assert outputs["limits"]["max_total_episodes"] == 6


def test_describe_does_not_leak_internal_fields(peer, clock) -> None:
    """capabilities 直接透传会把 result_dir 之类漏给 Agent。"""
    outputs = _describe(peer, clock).outputs
    assert "result_dir" not in outputs
    assert "v" not in outputs
    assert "type" not in outputs


def test_describe_goes_not_ready_when_peer_goes_silent(peer, clock) -> None:
    """心跳断了以后 describe 也要退回 not_ready,不能拿陈旧 capabilities 继续答。"""
    clock.advance(10.0)
    result = _describe(peer, clock)
    assert result.status == "failed"
    assert result.error.code == ERR_NOT_READY
