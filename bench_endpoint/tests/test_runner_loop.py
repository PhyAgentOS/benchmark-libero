"""事件循环与注册租约。

这一层验的是"**接得上**",不是"算得对" —— L1 里最阴的三种失败(endpoint_id
拼错、providers 映射写反、registry.response 忘了 acknowledge)症状完全一样:
gateway 起得来但 tool 永远 not ready。在这里先把 endpoint 侧摘干净。
"""

from __future__ import annotations

import pytest

from bench_endpoint import wire
from bench_endpoint.descriptor import BENCH_DESCRIPTOR, BENCH_ENDPOINT_ID
from bench_endpoint.endpoint_lease import EndpointLease
from bench_endpoint.runner import CMD_OUTPUT_ID, TOOL_OUTPUT_ID


# ---- 线格式 ----


def test_wire_roundtrip() -> None:
    payload = {"type": wire.CMD_RUN, "invocation_id": "inv-1", "plan": {"num_runs": 2}}
    assert wire.decode(wire.encode(payload)) == {**payload, "v": wire.WIRE_VERSION}


def test_wire_rejects_version_mismatch() -> None:
    """两个节点分别升级,没有版本号的话第一次不兼容变更就是静默错乱。"""
    import json

    import pyarrow as pa

    bad = pa.array([json.dumps({"v": 999, "type": "heartbeat"})])
    with pytest.raises(wire.WireError):
        wire.decode(bad)


def test_wire_rejects_non_json() -> None:
    import pyarrow as pa

    with pytest.raises(wire.WireError):
        wire.decode(pa.array(["not json at all"]))


# ---- 描述符 ----


def test_descriptor_shape() -> None:
    assert BENCH_DESCRIPTOR.endpoint_id == BENCH_ENDPOINT_ID
    ops = {op.name: op for op in BENCH_DESCRIPTOR.operations}
    assert ops["describe"].semantics == "query"
    assert ops["run"].semantics == "action"
    # action 必须支持 status,否则 forge_tool 的 descriptor 校验直接抛
    assert ops["run"].status_supported is True
    assert ops["run"].cancellable is True
    # 一台机器一个仿真器
    assert ops["run"].max_concurrency == 1


# ---- 租约 ----


def test_lease_emits_registration_then_backs_off() -> None:
    lease = EndpointLease(BENCH_DESCRIPTOR, "instance-1")
    first = lease.poll(0.0)
    assert first is not None
    assert first.message_type == "endpoint.register"
    # 回执没到之前不重发
    assert lease.poll(0.1) is None


def test_lease_renews_at_one_third_of_ttl() -> None:
    lease = EndpointLease(BENCH_DESCRIPTOR, "instance-1")
    request = lease.poll(0.0)
    from forge_tool import make_endpoint_registry_response_envelope
    from forge_tool import EndpointRegistryResponse

    response = make_endpoint_registry_response_envelope(
        EndpointRegistryResponse(
            operation="register",
            status="accepted",
            registry_revision=1,
            lease_ttl_ms=3000,
        ),
        request,
    )
    assert lease.acknowledge(response, 0.0) is True
    # ttl/3 = 1s:所以 tick 周期必须 ≤1000ms,否则续不上
    assert lease.poll(0.5) is None
    assert lease.poll(1.1) is not None


# ---- 事件循环 ----


def test_runner_registers_on_first_drive(runner, node) -> None:
    runner.run([])
    outputs = [oid for oid, _ in node.sent]
    assert outputs == [TOOL_OUTPUT_ID]


def test_tick_drives_lease_and_liveness(runner, node, peer, clock) -> None:
    """空闲时没有任何输入,续租只能靠 tick —— 这是和参考实现唯一的实质差别。"""
    runner.process_event({"type": "INPUT", "id": "tick", "value": None})
    assert any(oid == TOOL_OUTPUT_ID for oid, _ in node.sent)


def test_report_updates_peer_through_the_loop(runner, node, peer, clock) -> None:
    heartbeat = wire.encode(
        {"type": wire.REPORT_HEARTBEAT, "invocation_id": None, "busy": False}
    )
    clock.advance(2.0)
    runner.process_event({"type": "INPUT", "id": "bench_report", "value": heartbeat})
    assert peer.last_seen == clock()
    assert peer.ready(clock()) is True


def test_malformed_report_does_not_escape_the_loop(runner, node) -> None:
    """一条坏消息不该打死节点 —— 打死了整条 dataflow 一起没。"""
    import pyarrow as pa

    result = runner.process_event(
        {"type": "INPUT", "id": "bench_report", "value": pa.array(["garbage"])}
    )
    assert result is None


def test_outbox_is_flushed_on_bench_cmd(runner, node, peer) -> None:
    peer.send(wire.run_command("inv-1", {"task_ids": [0]}))
    runner.process_event({"type": "INPUT", "id": "tick", "value": None})
    commands = node.commands()
    assert len(commands) == 1
    assert commands[0]["invocation_id"] == "inv-1"
    assert any(oid == CMD_OUTPUT_ID for oid, _ in node.sent)


def test_stop_unregisters(runner, node) -> None:
    assert runner.process_event({"type": "STOP"}) == 0
    assert len(node.sent) >= 1
