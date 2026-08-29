"""事件循环与注册租约：验的是「接得上」，不是「算得对」。

L1 里最阴的三种失败（endpoint_id 拼错、operations 映射写反、registry.response
忘了 acknowledge）症状完全一样：gateway 起得来但 tool 永远 not ready。在这里
先把节点侧摘干净。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from forge_msgs import JointCommand, ToolMessage
from forge_tool import (
    EndpointRegistryResponse,
    ToolContext,
    ToolExecutionKey,
    ToolRequest,
    make_control_request_envelope,
    make_endpoint_registry_response_envelope,
    make_invoke_request_envelope,
)
from forge_tool.dora import tool_envelope_to_message, tool_message_to_envelope

from bench_tool.descriptor import BENCH_ENDPOINT_ID, OP_RUN
from bench_tool.node import ACTION_INPUT_ID, TOOL_INPUT_ID, TOOL_OUTPUT_ID, ServeToolLoop

from conftest import JOINTS, plan_arguments


@pytest.fixture
def loop(node, cfg_template, adapter) -> ServeToolLoop:
    return ServeToolLoop(node, cfg_template, adapter, log=lambda _msg: None)


def _tool_messages(node) -> list[ToolMessage]:
    return [
        ToolMessage.from_arrow(value) for oid, value in node.sent if oid == TOOL_OUTPUT_ID
    ]


def _invoke_message(loop: ServeToolLoop, invocation_id: str, **overrides) -> object:
    context = ToolContext(
        execution_key=ToolExecutionKey(invocation_id=invocation_id, attempt_id="a1"),
        tool_id="benchmark.controller.run",
        implementation_id="forge.benchmark.libero",
        endpoint_id=BENCH_ENDPOINT_ID,
        operation=OP_RUN,
    )
    envelope = make_invoke_request_envelope(
        ToolRequest(arguments=plan_arguments(**overrides)),
        context,
        request_id=f"req-{invocation_id}",
        endpoint_instance_id=loop.endpoint_instance_id,
    )
    return tool_envelope_to_message(envelope).to_arrow()


def _invoke(loop, invocation_id: str, **overrides) -> None:
    loop.process_event(
        {"type": "INPUT", "id": TOOL_INPUT_ID, "value": _invoke_message(loop, invocation_id, **overrides)}
    )


def _action_value() -> object:
    return JointCommand.from_np(np.zeros(len(JOINTS)), JOINTS, "position").to_arrow()


# ---- 注册与租约 ----


def test_loop_registers_on_first_event(loop, node) -> None:
    assert loop.process_event({"type": "INPUT", "id": "tick", "value": None}) is None
    messages = _tool_messages(node)
    assert messages and messages[0].message_type == "endpoint.register"


def test_registry_response_is_acknowledged(loop, node) -> None:
    loop.process_event({"type": "INPUT", "id": "tick", "value": None})
    registration = _tool_messages(node)[0]
    envelope = tool_message_to_envelope(registration)
    response = make_endpoint_registry_response_envelope(
        EndpointRegistryResponse(
            operation="register",
            status="accepted",
            registry_revision=1,
            lease_ttl_ms=3000,
        ),
        envelope,
    )
    loop.process_event(
        {"type": "INPUT", "id": TOOL_INPUT_ID, "value": tool_envelope_to_message(response).to_arrow()}
    )
    assert loop.lease.pending is None
    assert loop.lease.next_registration > 0


def test_timeout_event_drives_lease_and_pickup(loop, node) -> None:
    """空闲时没有任何输入，续租与 pending 拾取全靠 tick。"""
    assert loop.process_event(None) is None
    assert any(m.message_type == "endpoint.register" for m in _tool_messages(node))


# ---- invoke → accepted → 跑批 ----


def test_invoke_returns_accepted_then_runs_the_batch(loop, node, adapter) -> None:
    _invoke(loop, "inv-1", init_state_ids=[0, 1])
    messages = _tool_messages(node)
    assert messages[-1].message_type == "tool.invoke.response"
    assert "inv-1" in messages[-1].payload_json
    # dispatch 内不跑 reset —— 拾取发生在下一个事件
    assert adapter.resets == []

    loop.process_event(None)  # tick：拾取 pending，跑 blocking reset
    assert loop._active is not None
    assert loop._active.key.invocation_id == "inv-1"
    assert adapter.resets, "拾取后 episode 0 必须已 reset"

    # 2 集 × 2 步 = 4 个 action
    for _ in range(4):
        loop.process_event({"type": "INPUT", "id": ACTION_INPUT_ID, "value": _action_value()})

    assert loop._active is None  # 批次跑完回到 idle
    item = loop.table.by_id("inv-1")
    assert item.phase == "completed"
    assert item.result is not None
    assert item.result.outputs["status"] == "succeeded"

    events = [
        json.loads(m.payload_json)
        for m in _tool_messages(node)
        if m.message_type == "tool.event"
    ]
    types = [e["type"] for e in events]
    assert types.count("progress") == 2
    assert types[-1] == "executor_completed"


def test_busy_through_handler(loop, node) -> None:
    _invoke(loop, "inv-1")
    loop.process_event(None)  # 拾取，进入 running
    _invoke(loop, "inv-2")
    responses = [
        m for m in _tool_messages(node) if m.message_type == "tool.invoke.response"
    ]
    # handler 在进 endpoint 前用自己的 max_concurrency=1 检查拦截（endpoint 里
    # 的 FORGE_ENDPOINT_BUSY 是直调层的兜底码，race 场景才会出现）
    assert "FORGE_ENDPOINT_CAPACITY" in responses[-1].payload_json
    assert '"retryable":true' in responses[-1].payload_json


def test_second_run_after_completion_is_accepted(loop, node) -> None:
    _invoke(loop, "inv-1")
    loop.process_event(None)
    for _ in range(2):
        loop.process_event({"type": "INPUT", "id": ACTION_INPUT_ID, "value": _action_value()})
    assert loop.table.by_id("inv-1").phase == "completed"

    _invoke(loop, "inv-2")
    responses = [
        m for m in _tool_messages(node) if m.message_type == "tool.invoke.response"
    ]
    assert "inv-2" in responses[-1].payload_json


def test_idle_action_is_dropped(loop) -> None:
    assert (
        loop.process_event({"type": "INPUT", "id": ACTION_INPUT_ID, "value": _action_value()})
        is None
    )
    assert loop._active is None


# ---- stop：handler 全流程 ----


def test_stop_control_request_flows_through_handler(loop, node) -> None:
    _invoke(loop, "inv-1", init_state_ids=[0, 1])
    loop.process_event(None)  # running，episode 0 开跑

    context = ToolContext(
        execution_key=ToolExecutionKey(invocation_id="inv-1", attempt_id="a1"),
        tool_id="benchmark.controller.run",
        implementation_id="forge.benchmark.libero",
        endpoint_id=BENCH_ENDPOINT_ID,
        operation=OP_RUN,
    )
    envelope = make_control_request_envelope(
        "stop",
        context,
        request_id="ctl-1",
        endpoint_instance_id=loop.endpoint_instance_id,
        reason="agent asked",
    )
    loop.process_event(
        {"type": "INPUT", "id": TOOL_INPUT_ID, "value": tool_envelope_to_message(envelope).to_arrow()}
    )
    messages = _tool_messages(node)
    assert messages[-1].message_type == "tool.control.response"
    assert '"accepted"' in messages[-1].payload_json
    assert loop.table.by_id("inv-1").phase == "stopping"

    # 跑完当前集 → 边界 → stopped（2 集中完成 1 集 → partial）
    for _ in range(4):
        loop.process_event({"type": "INPUT", "id": ACTION_INPUT_ID, "value": _action_value()})
    item = loop.table.by_id("inv-1")
    assert item.phase == "stopped"
    assert item.result.outputs["status"] == "partial"
    assert item.result.outputs["completed_episodes"] == 1

    events = [
        json.loads(m.payload_json)
        for m in _tool_messages(node)
        if m.message_type == "tool.event"
    ]
    assert events[-1]["type"] == "stopped"


# ---- dora 生命周期 ----


def test_stop_event_finishes_session_and_unregisters(loop, node) -> None:
    _invoke(loop, "inv-1", init_state_ids=[0, 1])
    loop.process_event(None)  # running
    # 跑完 episode 0（2 步）再 STOP：dora STOP 是立即终止，已完成的 1 集照常收分
    for _ in range(2):
        loop.process_event({"type": "INPUT", "id": ACTION_INPUT_ID, "value": _action_value()})
    assert loop.process_event({"type": "STOP"}) == 0

    item = loop.table.by_id("inv-1")
    assert item.phase == "stopped"
    assert item.result.outputs["status"] == "partial"
    assert item.result.outputs["completed_episodes"] == 1

    messages = _tool_messages(node)
    assert any(m.message_type == "endpoint.unregister" for m in messages)


def test_stop_without_session_unregisters_cleanly(loop, node) -> None:
    assert loop.process_event({"type": "STOP"}) == 0
    assert any(
        m.message_type == "endpoint.unregister" for m in _tool_messages(node)
    )


def test_timeout_error_is_not_fatal(loop) -> None:
    assert loop.process_event({"type": "ERROR", "error": "Receiver timed out"}) is None


def test_error_event_fails_session_and_exits(loop, node) -> None:
    _invoke(loop, "inv-1")
    loop.process_event(None)  # running
    assert loop.process_event({"type": "ERROR", "error": "runtime boom"}) == 1
    assert loop.table.by_id("inv-1").phase == "failed"
