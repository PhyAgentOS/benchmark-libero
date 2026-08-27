"""bench.run:L0-A 的主验收面。

覆盖:accepted / 并发 busy / 参数越界 / 上限 / 结果回填 / cancel / 掉线 unknown,
以及 handler 依赖的那条不变式(终态 status ⇒ available result)。
"""

from __future__ import annotations

import asyncio

import pytest
from forge_tool import (
    ToolContext,
    ToolEndpointError,
    ToolExecutionKey,
    ToolRequest,
)

from bench_endpoint import wire
from bench_endpoint.descriptor import (
    BENCH_ENDPOINT_ID,
    ERR_BUSY,
    ERR_INVALID_ARGUMENT,
    ERR_NODE_UNAVAILABLE,
    ERR_NOT_READY,
    ERR_UNKNOWN_INVOCATION,
    OP_RUN,
)
from bench_endpoint.endpoints import BenchRunEndpoint

from conftest import plan_arguments


def _key(invocation_id: str) -> ToolExecutionKey:
    return ToolExecutionKey(invocation_id=invocation_id, attempt_id="a1")


def _context(invocation_id: str) -> ToolContext:
    return ToolContext(
        execution_key=_key(invocation_id),
        tool_id="bench.run",
        implementation_id="example.benchmark",
        endpoint_id=BENCH_ENDPOINT_ID,
        operation=OP_RUN,
    )


def _start(endpoint: BenchRunEndpoint, invocation_id: str, **overrides):
    request = ToolRequest(arguments=plan_arguments(**overrides))
    return asyncio.run(endpoint.start(request, _context(invocation_id), events=None))


@pytest.fixture
def endpoint(peer, clock) -> BenchRunEndpoint:
    return BenchRunEndpoint(peer, clock)


# ---- 接受 ----


def test_start_accepts_and_emits_one_run_command(endpoint, peer) -> None:
    accepted = _start(endpoint, "inv-1")
    assert accepted.details["invocation_id"] == "inv-1"
    assert accepted.details["total_episodes"] == 1

    commands = peer.drain()
    assert len(commands) == 1
    assert commands[0]["type"] == wire.CMD_RUN
    assert commands[0]["invocation_id"] == "inv-1"
    assert commands[0]["plan"]["task_ids"] == [0]


def test_start_does_not_block_on_peer_acknowledgement(endpoint, peer) -> None:
    """start 返回时对端一个字都还没回 —— 这正是要的行为(否则死锁)。"""
    _start(endpoint, "inv-1")
    item = peer.table.by_id("inv-1")
    assert item is not None
    assert item.phase == "accepted"
    assert item.terminal is False


def test_null_ids_expand_to_full_set(endpoint, peer) -> None:
    _start(endpoint, "inv-1", task_ids=None, init_state_ids=None, num_runs=1)
    plan = peer.drain()[0]["plan"]
    assert plan["task_ids"] == [0, 1]
    # 逐 task 取最小值:task 1 只有 2 个 init state
    assert plan["init_state_ids"] == [0, 1]
    assert plan["total_episodes"] == 4


def test_omitted_seed_and_max_steps_use_capability_defaults(endpoint, peer) -> None:
    request = ToolRequest(
        arguments={
            "task_ids": [0],
            "init_state_ids": [0],
            "num_runs": 1,
        }
    )
    accepted = asyncio.run(
        endpoint.start(request, _context("inv-defaults"), events=None)
    )

    assert accepted.details["invocation_id"] == "inv-defaults"
    plan = peer.drain()[0]["plan"]
    assert plan["seed"] == 0
    assert plan["max_steps"] == 300


# ---- 拒绝 ----


def test_second_concurrent_run_is_busy(endpoint, peer) -> None:
    _start(endpoint, "inv-1")
    with pytest.raises(ToolEndpointError) as excinfo:
        _start(endpoint, "inv-2")
    error = excinfo.value.error
    assert error.code == ERR_BUSY
    assert error.retryable is True
    assert error.details["active_invocation_id"] == "inv-1"
    # 被拒的那次绝不能留下痕迹,更不能发命令
    assert peer.table.by_id("inv-2") is None
    assert len(peer.drain()) == 1


def test_run_before_ready_is_rejected(cold_peer, clock) -> None:
    endpoint = BenchRunEndpoint(cold_peer, clock)
    with pytest.raises(ToolEndpointError) as excinfo:
        _start(endpoint, "inv-1")
    assert excinfo.value.error.code == ERR_NOT_READY


def test_out_of_range_task_id_is_invalid_argument(endpoint, peer) -> None:
    with pytest.raises(ToolEndpointError) as excinfo:
        _start(endpoint, "inv-1", task_ids=[7])
    error = excinfo.value.error
    assert error.code == ERR_INVALID_ARGUMENT
    # 参数错了重试多少次都一样,必须是 False
    assert error.retryable is False
    assert peer.drain() == []


def test_init_state_is_validated_per_task(endpoint) -> None:
    """task 0 有 3 个 init state、task 1 只有 2 个;拿 task_ids[0] 代表全体是错的。"""
    with pytest.raises(ToolEndpointError) as excinfo:
        _start(endpoint, "inv-1", task_ids=[0, 1], init_state_ids=[2])
    assert excinfo.value.error.code == ERR_INVALID_ARGUMENT
    assert "task 1" in excinfo.value.error.message


def test_episode_budget_is_enforced(endpoint) -> None:
    """没有这道闸,Agent 会一口气开满然后跑几小时,而且不报错。"""
    with pytest.raises(ToolEndpointError) as excinfo:
        _start(endpoint, "inv-1", task_ids=[0], init_state_ids=[0, 1, 2], num_runs=5)
    error = excinfo.value.error
    assert error.code == ERR_INVALID_ARGUMENT
    assert error.details["requested_episodes"] == 15
    assert error.details["max_total_episodes"] == 6


def test_empty_id_list_is_rejected(endpoint) -> None:
    """空数组和 null 是两回事:null=全集,空数组=什么都不跑,后者一定是填错了。"""
    with pytest.raises(ToolEndpointError):
        _start(endpoint, "inv-1", task_ids=[])


# ---- 生命周期 ----


def test_result_is_pending_until_report_arrives(endpoint, peer) -> None:
    _start(endpoint, "inv-1")
    response = asyncio.run(endpoint.result(_key("inv-1")))
    assert response.status == "pending"


def test_report_result_becomes_terminal_and_available(endpoint, peer, clock) -> None:
    _start(endpoint, "inv-1")
    peer.on_report(
        {
            "v": wire.WIRE_VERSION,
            "type": wire.REPORT_ACCEPTED,
            "invocation_id": "inv-1",
        },
        clock(),
    )
    assert asyncio.run(endpoint.status(_key("inv-1"))).phase == "running"

    peer.on_report(
        {
            "v": wire.WIRE_VERSION,
            "type": wire.REPORT_RESULT,
            "invocation_id": "inv-1",
            "status": "succeeded",
            "payload": {
                "status": "succeeded",
                "success_rate": 0.5,
                "successes": 1,
                "total_episodes": 2,
                "completed_episodes": 2,
                "result_path": "/tmp/build_tower_x.json",
            },
        },
        clock(),
    )

    status = asyncio.run(endpoint.status(_key("inv-1")))
    response = asyncio.run(endpoint.result(_key("inv-1")))
    assert status.phase == "completed"
    assert response.status == "available"
    assert response.result.status == "succeeded"
    assert response.result.outputs["success_rate"] == 0.5
    assert response.result.outputs["result_path"] == "/tmp/build_tower_x.json"


def test_rejected_report_fails_the_invocation(endpoint, peer, clock) -> None:
    """endpoint 侧放行了但 3.11 侧兜底拒了 —— 必须变成终态 failed,不是异常。"""
    _start(endpoint, "inv-1")
    peer.on_report(
        {
            "v": wire.WIRE_VERSION,
            "type": wire.REPORT_REJECTED,
            "invocation_id": "inv-1",
            "code": ERR_INVALID_ARGUMENT,
            "message": "越界 init_state_ids: [9]",
        },
        clock(),
    )
    response = asyncio.run(endpoint.result(_key("inv-1")))
    assert response.status == "available"
    assert response.result.status == "failed"
    assert response.result.error.code == ERR_INVALID_ARGUMENT


def test_terminal_status_always_has_available_result(endpoint, peer, clock) -> None:
    """forge_tool 的 handler 依赖这条不变式,破了会在 SDK 内部抛。"""
    _start(endpoint, "inv-1")
    peer.on_report(
        {
            "v": wire.WIRE_VERSION,
            "type": wire.REPORT_RESULT,
            "invocation_id": "inv-1",
            "status": "failed",
            "message": "批次中途崩了",
            "payload": {"status": "failed", "completed_episodes": 1, "total_episodes": 4},
        },
        clock(),
    )
    status = asyncio.run(endpoint.status(_key("inv-1")))
    response = asyncio.run(endpoint.result(_key("inv-1")))
    assert status.phase == "failed"
    assert status.error is not None
    assert response.status == "available"
    assert response.result.status == "failed"


def test_unknown_invocation_status_and_result_agree(endpoint) -> None:
    status = asyncio.run(endpoint.status(_key("never-existed")))
    response = asyncio.run(endpoint.result(_key("never-existed")))
    assert status.phase == "unknown"
    assert status.error.code == ERR_UNKNOWN_INVOCATION
    # 不能返回 not_found:status 已经报了终态,handler 会立刻来要结果
    assert response.status == "available"
    assert response.result.status == "unknown"


def test_run_after_terminal_is_allowed(endpoint, peer, clock) -> None:
    """L3 的单元版:上一次跑完之后,同一个进程必须还能接第二次。"""
    _start(endpoint, "inv-1")
    peer.on_report(
        {
            "v": wire.WIRE_VERSION,
            "type": wire.REPORT_RESULT,
            "invocation_id": "inv-1",
            "status": "succeeded",
            "payload": {"status": "succeeded", "success_rate": 1.0},
        },
        clock(),
    )
    peer.drain()
    accepted = _start(endpoint, "inv-2")
    assert accepted.details["invocation_id"] == "inv-2"
    assert peer.drain()[0]["invocation_id"] == "inv-2"


# ---- 取消 ----


def test_cancel_emits_command_once_and_is_idempotent(endpoint, peer) -> None:
    _start(endpoint, "inv-1")
    peer.drain()
    first = asyncio.run(endpoint.cancel(_key("inv-1"), "user asked"))
    second = asyncio.run(endpoint.cancel(_key("inv-1"), "user asked again"))
    assert first.status == "accepted"
    assert second.status == "accepted"
    commands = peer.drain()
    assert len(commands) == 1
    assert commands[0]["type"] == wire.CMD_CANCEL


def test_cancel_after_terminal_reports_terminal(endpoint, peer, clock) -> None:
    _start(endpoint, "inv-1")
    peer.on_report(
        {
            "v": wire.WIRE_VERSION,
            "type": wire.REPORT_RESULT,
            "invocation_id": "inv-1",
            "status": "succeeded",
            "payload": {},
        },
        clock(),
    )
    response = asyncio.run(endpoint.cancel(_key("inv-1")))
    assert response.status == "terminal"
    assert response.error is None


def test_cancel_unknown_invocation_is_rejected(endpoint) -> None:
    response = asyncio.run(endpoint.cancel(_key("never-existed")))
    assert response.status == "rejected"
    assert response.error.code == ERR_UNKNOWN_INVOCATION


# ---- 掉线 ----


def test_peer_silence_marks_running_invocation_unknown(endpoint, peer, clock) -> None:
    """不做这件事,benchmark 节点崩了以后 Agent 会永远轮询到 running。"""
    _start(endpoint, "inv-1")
    clock.advance(10.0)
    peer.tick(clock())

    status = asyncio.run(endpoint.status(_key("inv-1")))
    response = asyncio.run(endpoint.result(_key("inv-1")))
    assert status.phase == "unknown"
    assert status.error.code == ERR_NODE_UNAVAILABLE
    assert response.result.status == "unknown"
    # unknown ≠ failed:副作用可能已经发生,SKILL.md 里要禁止盲目重试
    assert response.result.error.retryable is True


def test_tick_does_not_touch_terminal_invocations(endpoint, peer, clock) -> None:
    _start(endpoint, "inv-1")
    peer.on_report(
        {
            "v": wire.WIRE_VERSION,
            "type": wire.REPORT_RESULT,
            "invocation_id": "inv-1",
            "status": "succeeded",
            "payload": {},
        },
        clock(),
    )
    clock.advance(60.0)
    peer.tick(clock())
    assert asyncio.run(endpoint.status(_key("inv-1"))).phase == "completed"
