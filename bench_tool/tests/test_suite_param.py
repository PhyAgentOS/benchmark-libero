"""describe / run 的 suite 参数：选 suite、白名单外同步拒绝、按 suite 缓存 capabilities。"""

from __future__ import annotations

import asyncio

import pytest
from forge_tool import ToolContext, ToolEndpointError, ToolExecutionKey, ToolRequest

from bench_tool.descriptor import BENCH_ENDPOINT_ID, ERR_INVALID_ARGUMENT, OP_DESCRIBE, OP_RUN
from bench_tool.endpoints import BenchDescribeEndpoint, select_capabilities
from bench_tool.node import ServeToolLoop

from conftest import MultiSuiteAdapter, plan_arguments


def _context(invocation_id: str, operation: str = OP_RUN) -> ToolContext:
    return ToolContext(
        execution_key=ToolExecutionKey(invocation_id=invocation_id, attempt_id="a1"),
        tool_id=f"benchmark.controller.{operation}",
        implementation_id="forge.benchmark.libero",
        endpoint_id=BENCH_ENDPOINT_ID,
        operation=operation,
    )


@pytest.fixture
def multi_adapter() -> MultiSuiteAdapter:
    return MultiSuiteAdapter()


@pytest.fixture
def multi_loop(node, cfg_template, multi_adapter) -> ServeToolLoop:
    cfg_template.suites = ["fake_suite", "big_suite"]
    cfg_template.max_steps_by_suite = {"big_suite": 77}
    return ServeToolLoop(node, cfg_template, multi_adapter, log=lambda _msg: None)


def _describe(loop: ServeToolLoop, **arguments):
    describe = BenchDescribeEndpoint(loop.caps, loop.capabilities_for)
    return asyncio.run(
        describe.query(ToolRequest(arguments=arguments), _context("q", OP_DESCRIBE))
    ).outputs


def _start(loop: ServeToolLoop, invocation_id: str, **overrides):
    request = ToolRequest(arguments=plan_arguments(**overrides))
    return asyncio.run(loop.endpoint.start(request, _context(invocation_id), events=None))


# ---- describe ----


def test_describe_default_lists_suites(multi_loop) -> None:
    out = _describe(multi_loop)
    assert out["suite"] == "fake_suite"
    assert out["suites"] == ["fake_suite", "big_suite"]
    assert out["num_tasks"] == 2


def test_describe_other_suite(multi_loop) -> None:
    out = _describe(multi_loop, suite="big_suite")
    assert out["suite"] == "big_suite"
    assert [t["num_init_states"] for t in out["tasks"]] == [1, 1, 4]
    assert out["defaults"]["max_steps"] == 77


def test_describe_rejects_suite_outside_whitelist(multi_loop) -> None:
    with pytest.raises(ToolEndpointError) as excinfo:
        _describe(multi_loop, suite="libero_90")
    assert excinfo.value.error.code == ERR_INVALID_ARGUMENT
    assert excinfo.value.error.details["available_suites"] == ["fake_suite", "big_suite"]


def test_capabilities_are_cached_per_suite(multi_loop) -> None:
    first = multi_loop.capabilities_for("big_suite")
    assert multi_loop.capabilities_for("big_suite") is first
    assert multi_loop.capabilities_for("fake_suite") is multi_loop.caps


def test_operations_are_wired_with_caps_lookup(multi_loop) -> None:
    assert multi_loop.endpoint.caps_for == multi_loop.capabilities_for


# ---- select_capabilities ----


def test_select_without_lookup_rejects_non_default(caps) -> None:
    assert select_capabilities({}, caps, None) is caps
    assert select_capabilities({"suite": caps["suite"]}, caps, None) is caps
    with pytest.raises(ToolEndpointError):
        select_capabilities({"suite": "other"}, caps, None)


def test_select_rejects_non_string_suite(multi_loop) -> None:
    with pytest.raises(ToolEndpointError):
        select_capabilities({"suite": 3}, multi_loop.caps, multi_loop.capabilities_for)


# ---- run ----


def test_run_without_suite_uses_default(multi_loop) -> None:
    _start(multi_loop, "inv-1")
    handle = multi_loop.endpoint.take_pending()[0]
    assert handle.plan["suite"] == "fake_suite"
    assert handle.session.cfg.suite == "fake_suite"


def test_run_other_suite_validates_against_that_suite(multi_loop) -> None:
    # task 2 / init 3 只在 big_suite 存在
    accepted = _start(multi_loop, "inv-1", suite="big_suite", task_ids=[2], init_state_ids=[3])
    assert accepted.details["total_episodes"] == 1
    handle = multi_loop.endpoint.take_pending()[0]
    assert handle.plan["suite"] == "big_suite"
    assert handle.session.cfg.suite == "big_suite"


def test_run_other_suite_omitted_max_steps_uses_suite_default(multi_loop) -> None:
    args = {"suite": "big_suite", "task_ids": [0], "init_state_ids": [0], "num_runs": 1}
    asyncio.run(
        multi_loop.endpoint.start(ToolRequest(arguments=args), _context("inv-1"), events=None)
    )
    assert multi_loop.endpoint.take_pending()[0].plan["max_steps"] == 77


def test_run_rejects_bounds_of_wrong_suite(multi_loop) -> None:
    with pytest.raises(ToolEndpointError) as excinfo:
        _start(multi_loop, "inv-1", suite="big_suite", task_ids=[0], init_state_ids=[1])
    assert excinfo.value.error.code == ERR_INVALID_ARGUMENT
    assert multi_loop.table.by_id("inv-1") is None


def test_run_rejects_suite_outside_whitelist(multi_loop, multi_adapter) -> None:
    with pytest.raises(ToolEndpointError) as excinfo:
        _start(multi_loop, "inv-1", suite="libero_90")
    assert excinfo.value.error.code == ERR_INVALID_ARGUMENT
    assert multi_loop.table.by_id("inv-1") is None
    assert multi_adapter.cfg is None  # 拒绝前不得回写 adapter


def test_single_suite_profile_rejects_any_other_suite(endpoint) -> None:
    """旧 profile（没配 suites）+ 不支持切换的 adapter：只认默认 suite。"""
    request = ToolRequest(arguments=plan_arguments(suite="big_suite"))
    with pytest.raises(ToolEndpointError) as excinfo:
        asyncio.run(endpoint.start(request, _context("inv-1"), events=None))
    assert excinfo.value.error.code == ERR_INVALID_ARGUMENT
    assert excinfo.value.error.details["available_suites"] == ["fake_suite"]
