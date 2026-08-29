"""benchmark.controller/run(session)与 /describe(query)的主验收面。

覆盖：accepted / 并发 busy / 同 key 幂等与异参冲突 / 参数越界 / 预算上限 /
stop 全流程（accepted→stopping→episode 边界→stopped）/ 终态落表与事件顺序 /
describe 白名单 / 七字段契约，以及 handler 依赖的不变式（终态 status ⇒
available result）。
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from forge_msgs import JointCommand
from forge_tool import (
    ToolContext,
    ToolEndpointError,
    ToolExecutionKey,
    ToolRequest,
)

from bench_tool.descriptor import (
    BENCH_ENDPOINT_ID,
    ERR_BUSY,
    ERR_CONFLICT,
    ERR_INVALID_ARGUMENT,
    ERR_UNKNOWN_INVOCATION,
    OP_RUN,
)
from bench_tool.endpoints import BenchDescribeEndpoint

from conftest import JOINTS, drive, plan_arguments


def _key(invocation_id: str) -> ToolExecutionKey:
    return ToolExecutionKey(invocation_id=invocation_id, attempt_id="a1")


def _context(invocation_id: str) -> ToolContext:
    return ToolContext(
        execution_key=_key(invocation_id),
        tool_id="benchmark.controller.run",
        implementation_id="forge.benchmark.libero",
        endpoint_id=BENCH_ENDPOINT_ID,
        operation=OP_RUN,
    )


def _start(endpoint, invocation_id: str, **overrides):
    request = ToolRequest(arguments=plan_arguments(**overrides))
    return asyncio.run(endpoint.start(request, _context(invocation_id), events=None))


def _feed_actions(session, max_actions: int = 100) -> None:
    """喂零动作直到批次结束（不重复调用 start）。"""
    action = JointCommand.from_np(np.zeros(len(JOINTS)), JOINTS, "position")
    for _ in range(max_actions):
        if session.finished:
            return
        session.on_action(action)
    raise AssertionError("批次没有在预期步数内结束")


# ---- 接受 ----


def test_start_accepts_and_queues_one_session(endpoint, adapter) -> None:
    accepted = _start(endpoint, "inv-1")
    assert accepted.details["invocation_id"] == "inv-1"
    assert accepted.details["total_episodes"] == 1

    handles = endpoint.take_pending()
    assert len(handles) == 1
    handle = handles[0]
    assert handle.key.invocation_id == "inv-1"
    assert handle.plan["task_ids"] == [0]
    # start 只登记不执行：adapter 一个 reset 都还没跑，阻塞留给事件循环
    assert adapter.resets == []
    assert handle.session.finished is False
    # 表里是 accepted，不是 running —— running 由循环拾取时标记
    assert endpoint.table.by_id("inv-1").phase == "accepted"


def test_null_ids_expand_to_full_set(endpoint) -> None:
    _start(endpoint, "inv-1", task_ids=None, init_state_ids=None, num_runs=1)
    plan = endpoint.take_pending()[0].plan
    assert plan["task_ids"] == [0, 1]
    # 逐 task 取最小值：task 1 只有 2 个 init state
    assert plan["init_state_ids"] == [0, 1]
    assert plan["total_episodes"] == 4


def test_omitted_seed_and_max_steps_use_capability_defaults(endpoint) -> None:
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
    plan = endpoint.take_pending()[0].plan
    assert plan["seed"] == 0
    assert plan["max_steps"] == 300


# ---- 拒绝 ----

def test_second_concurrent_run_is_busy(endpoint) -> None:
    _start(endpoint, "inv-1")
    with pytest.raises(ToolEndpointError) as excinfo:
        _start(endpoint, "inv-2")
    error = excinfo.value.error
    assert error.code == ERR_BUSY
    assert error.retryable is True
    assert error.details["active_invocation_id"] == "inv-1"
    # 被拒的那次绝不能留下痕迹
    assert endpoint.table.by_id("inv-2") is None
    assert len(endpoint.take_pending()) == 1


def test_duplicate_key_with_same_arguments_is_idempotent(endpoint) -> None:
    first = _start(endpoint, "inv-1")
    second = _start(endpoint, "inv-1")
    assert first.details["invocation_id"] == "inv-1"
    assert second.details["invocation_id"] == "inv-1"
    assert second.details["idempotent"] is True
    # 不重复排队
    assert len(endpoint.take_pending()) == 1


def test_duplicate_key_with_different_arguments_is_conflict(endpoint) -> None:
    _start(endpoint, "inv-1")
    with pytest.raises(ToolEndpointError) as excinfo:
        _start(endpoint, "inv-1", seed=99)
    error = excinfo.value.error
    assert error.code == ERR_CONFLICT
    assert error.retryable is False


def test_out_of_range_task_id_is_invalid_argument(endpoint) -> None:
    with pytest.raises(ToolEndpointError) as excinfo:
        _start(endpoint, "inv-1", task_ids=[7])
    error = excinfo.value.error
    assert error.code == ERR_INVALID_ARGUMENT
    # 参数错了重试多少次都一样，必须是 False
    assert error.retryable is False
    assert endpoint.take_pending() == []


def test_init_state_is_validated_per_task(endpoint) -> None:
    """task 0 有 3 个 init state、task 1 只有 2 个；拿 task_ids[0] 代表全体是错的。"""
    with pytest.raises(ToolEndpointError) as excinfo:
        _start(endpoint, "inv-1", task_ids=[0, 1], init_state_ids=[2])
    assert excinfo.value.error.code == ERR_INVALID_ARGUMENT
    assert "task 1" in excinfo.value.error.message


def test_episode_budget_is_enforced(endpoint) -> None:
    """没有这道闸，Agent 会一口气开满然后跑几小时，而且不报错。"""
    with pytest.raises(ToolEndpointError) as excinfo:
        _start(endpoint, "inv-1", task_ids=[0], init_state_ids=[0, 1, 2], num_runs=5)
    error = excinfo.value.error
    assert error.code == ERR_INVALID_ARGUMENT
    assert error.details["requested_episodes"] == 15
    assert error.details["max_total_episodes"] == 6


def test_empty_id_list_is_rejected(endpoint) -> None:
    """空数组和 null 是两回事：null=全集，空数组=什么都不跑，后者一定是填错了。"""
    with pytest.raises(ToolEndpointError):
        _start(endpoint, "inv-1", task_ids=[])


# ---- 生命周期 ----


def test_result_is_pending_until_finished(endpoint) -> None:
    _start(endpoint, "inv-1")
    assert asyncio.run(endpoint.result(_key("inv-1"))).status == "pending"


def test_unknown_invocation_status_and_result_agree(endpoint) -> None:
    status = asyncio.run(endpoint.status(_key("never-existed")))
    response = asyncio.run(endpoint.result(_key("never-existed")))
    assert status.phase == "unknown"
    assert status.error.code == ERR_UNKNOWN_INVOCATION
    # 不能返回 not_found：status 已经报了终态，handler 会立刻来要结果
    assert response.status == "available"
    assert response.result.status == "unknown"


def test_run_after_terminal_is_allowed(endpoint, adapter) -> None:
    """同一个进程必须还能接第二次。"""
    _start(endpoint, "inv-1", init_state_ids=[0])
    handle = endpoint.take_pending()[0]
    drive(handle.session)
    endpoint.on_finished(handle)

    accepted = _start(endpoint, "inv-2")
    assert accepted.details["invocation_id"] == "inv-2"
    assert len(endpoint.take_pending()) == 1


# ---- stop ----


def test_stop_active_session_is_accepted_and_marks_stopping(endpoint) -> None:
    _start(endpoint, "inv-1")
    first = asyncio.run(endpoint.stop(_key("inv-1"), "agent asked"))
    assert first.status == "accepted"
    item = endpoint.table.by_id("inv-1")
    assert item.phase == "stopping"
    assert item.stop_requested is True
    handle = endpoint.take_pending()[0]
    assert handle.session.runner.stop_requested is True
    assert handle.session.runner.stop_reason == "agent asked"
    # 幂等：重复 stop 不重复副作用
    second = asyncio.run(endpoint.stop(_key("inv-1"), "agent asked again"))
    assert second.status == "accepted"


def test_stop_after_terminal_reports_terminal(endpoint) -> None:
    _start(endpoint, "inv-1")
    handle = endpoint.take_pending()[0]
    drive(handle.session)
    endpoint.on_finished(handle)
    response = asyncio.run(endpoint.stop(_key("inv-1")))
    assert response.status == "terminal"
    assert response.error is None


def test_stop_unknown_invocation_is_rejected(endpoint) -> None:
    response = asyncio.run(endpoint.stop(_key("never-existed")))
    assert response.status == "rejected"
    assert response.error.code == ERR_UNKNOWN_INVOCATION


# ---- 终态：七字段契约 ----


def test_completed_batch_reports_succeeded_with_seven_fields(endpoint, adapter) -> None:
    _start(endpoint, "inv-1", init_state_ids=[0, 1])
    handle = endpoint.take_pending()[0]
    drive(handle.session)
    endpoint.on_finished(handle)

    status = asyncio.run(endpoint.status(_key("inv-1")))
    result = asyncio.run(endpoint.result(_key("inv-1")))
    assert status.phase == "completed"
    assert result.status == "available"
    assert result.result.status == "succeeded"
    assert result.result.error is None

    outputs = result.result.outputs
    # 七字段契约：多一个少一个都和 skill 树的 output_schema 对不上
    assert set(outputs) == {
        "run_id",
        "status",
        "completed_episodes",
        "total_episodes",
        "success_rate",
        "result_file",
        "failure_summary",
    }
    assert outputs["run_id"] == "inv-1"
    assert outputs["status"] == "succeeded"
    assert outputs["completed_episodes"] == 2
    assert outputs["total_episodes"] == 2
    assert outputs["success_rate"] == 1.0
    assert outputs["result_file"].endswith("inv-1.json")
    assert outputs["failure_summary"] is None

    # 终态事件（类型 + 载荷）
    invocation_id, event = endpoint.emit_recorder.events[-1]
    assert invocation_id == "inv-1"
    assert event.type == "executor_completed"
    assert event.data["status"] == "succeeded"


def test_stop_before_pickup_runs_zero_episodes_and_reports_stopped(endpoint, adapter) -> None:
    """stop 落在 accepted 与拾取之间：一集都不跑，输出纯 stopped。"""
    _start(endpoint, "inv-1", init_state_ids=[0, 1])
    asyncio.run(endpoint.stop(_key("inv-1"), "stopped before pickup"))
    handle = endpoint.take_pending()[0]
    drive(handle.session)
    endpoint.on_finished(handle)

    result = asyncio.run(endpoint.result(_key("inv-1")))
    assert result.status == "available"
    assert result.result.status == "stopped"
    assert result.result.error is None  # stopped 的 ToolResult 不许带 error
    outputs = result.result.outputs
    assert outputs["status"] == "stopped"
    assert outputs["completed_episodes"] == 0
    assert outputs["total_episodes"] == 2
    assert "stopped before pickup" in outputs["failure_summary"]

    assert endpoint.emit_recorder.events[-1][1].type == "stopped"


def test_stop_mid_episode_lands_at_boundary_and_reports_partial(endpoint, adapter) -> None:
    """stop 是 episode 边界语义：当前集跑完收分（1/3），后面不再开。"""
    _start(endpoint, "inv-1", init_state_ids=[0, 1, 2])
    handle = endpoint.take_pending()[0]
    handle.session.start()
    asyncio.run(endpoint.stop(_key("inv-1"), "agent asked"))
    _feed_actions(handle.session)
    endpoint.on_finished(handle)

    status = asyncio.run(endpoint.status(_key("inv-1")))
    result = asyncio.run(endpoint.result(_key("inv-1")))
    assert status.phase == "stopped"
    assert result.result.status == "stopped"
    assert result.result.error is None
    outputs = result.result.outputs
    assert outputs["status"] == "partial"
    assert outputs["completed_episodes"] == 1
    assert outputs["total_episodes"] == 3
    assert outputs["success_rate"] == pytest.approx(1.0 / 3.0)
    assert "agent asked" in outputs["failure_summary"]

    assert endpoint.emit_recorder.events[-1][1].type == "stopped"


def test_failed_batch_reports_failed_with_error(endpoint) -> None:
    _start(endpoint, "inv-1")
    handle = endpoint.take_pending()[0]
    handle.session.runner.finish("failed", failure_summary="仿真器崩了")
    endpoint.on_finished(handle)

    status = asyncio.run(endpoint.status(_key("inv-1")))
    result = asyncio.run(endpoint.result(_key("inv-1")))
    assert status.phase == "failed"
    assert status.error is not None
    assert result.result.status == "failed"
    assert result.result.error.code == "BENCH_RUN_FAILED"
    assert "仿真器崩了" in result.result.error.message
    assert result.result.outputs["status"] == "failed"
    assert result.result.outputs["failure_summary"] == "仿真器崩了"

    assert endpoint.emit_recorder.events[-1][1].type == "executor_failed"


# ---- describe ----


def test_describe_returns_whitelisted_capabilities(caps) -> None:
    # 注入一个内部字段，证明白名单确实在过滤
    endpoint = BenchDescribeEndpoint({**caps, "result_dir": "/tmp/should-not-leak"})
    result = asyncio.run(endpoint.query(ToolRequest(arguments={}), _context("inv-x")))
    assert result.status == "succeeded"
    assert result.error is None
    outputs = result.outputs
    assert outputs["num_tasks"] == 2
    assert {t["task_id"]: t["num_init_states"] for t in outputs["tasks"]} == {0: 3, 1: 2}
    assert outputs["defaults"] == {"num_runs": 1, "seed": 0, "max_steps": 300}
    assert "result_dir" not in outputs
    assert "reproducibility" not in outputs  # 内部字段不得泄漏
