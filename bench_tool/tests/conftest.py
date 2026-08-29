"""bench_tool 的公共夹具。**不需要 dora、不需要 GPU、不起仿真器。**

事件循环是拿合成的 event 字典列表喂 process_event 驱动的。FakeAdapter/FakeNode
等直接复用 _core/tests/conftest.py 里的类 —— 两个包测的是同一批行为，不复制。
（fixture 不能复用：pytest 只加载本目录及祖先目录的 conftest。）
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

import pytest

# pytest 把本文件也注册成顶层模块 conftest，直接 `from conftest import ...`
# 会解析到自身（circular import）。用独立模块名加载 _core 的 conftest，
# 共享它的类与驱动函数；fixture 在本文件重新定义（pytest 不跨非祖先
# conftest 继承 fixture）。
_CORE_TESTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "_core", "tests")
)
_spec = importlib.util.spec_from_file_location(
    "_core_tests_conftest", os.path.join(_CORE_TESTS, "conftest.py")
)
assert _spec is not None and _spec.loader is not None
_core_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core_conftest)

# noqa: F401 —— 重导出给本包测试用
JOINTS = _core_conftest.JOINTS
FakeAdapter = _core_conftest.FakeAdapter
FakeNode = _core_conftest.FakeNode
drive = _core_conftest.drive
make_plan = _core_conftest.make_plan

from bench_tool.endpoints import BenchSessionEndpoint  # noqa: E402
from bench_tool.invocations import InvocationTable  # noqa: E402
from _core.config import BaseBenchmarkConfig  # noqa: E402
from _core.session import bench_limits, capabilities_payload  # noqa: E402


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class EmitRecorder:
    """收集 endpoint 发的事件：(invocation_id, ToolEvent) 列表。"""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def __call__(self, handle: Any, event: Any) -> None:
        self.events.append((handle.key.invocation_id, event))


@pytest.fixture
def adapter() -> FakeAdapter:
    return FakeAdapter()


@pytest.fixture
def node() -> FakeNode:
    return FakeNode()


@pytest.fixture
def cfg_template(tmp_path) -> BaseBenchmarkConfig:
    """模板 cfg。测试要证明它**不会被 session 改动**。"""
    cfg = BaseBenchmarkConfig()
    cfg.benchmark = "fake"
    cfg.suite = "fake_suite"
    cfg.policy_id = "fake_policy"  # forge_msgs.PolicyCommand 要求非空
    cfg.task_ids = []
    cfg.init_state_ids = []
    cfg.num_runs = 1
    cfg.seed = 0
    cfg.max_steps = 300
    cfg.serve = True
    cfg.max_total_episodes = 6
    cfg.max_max_steps = 2000
    cfg.result_dir = str(tmp_path / "results")
    cfg.record_dir = str(tmp_path / "videos")
    return cfg


@pytest.fixture
def caps(cfg_template, adapter) -> dict[str, Any]:
    return capabilities_payload(cfg_template, adapter, limits=bench_limits(cfg_template))


@pytest.fixture
def endpoint(node, cfg_template, adapter, caps) -> BenchSessionEndpoint:
    recorder = EmitRecorder()
    endpoint = BenchSessionEndpoint(
        InvocationTable(), node, cfg_template, adapter, caps,
        emit=recorder, log=lambda _msg: None,
    )
    endpoint.emit_recorder = recorder  # type: ignore[attr-defined]
    return endpoint


def plan_arguments(**overrides: Any) -> dict[str, Any]:
    args = {
        "task_ids": [0],
        "init_state_ids": [0],
        "num_runs": 1,
        "seed": 0,
        "max_steps": 300,
    }
    args.update(overrides)
    return args
