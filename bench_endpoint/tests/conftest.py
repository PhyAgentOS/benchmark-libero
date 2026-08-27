"""L0-A 的公共夹具。**不需要 dora、不需要 GPU、不起 Isaac Sim。**

事件循环是拿合成的 event 字典列表喂 `process_event` 驱动的,做法和
relative_pose_policy_node/tests/test_runner.py 一致。
"""

from __future__ import annotations

from typing import Any

import pytest

from bench_endpoint import wire
from bench_endpoint.peer import BenchPeer
from bench_endpoint.runner import BenchEndpointRunner


class FakeNode:
    """收集 send_output,替代 dora.Node。"""

    def __init__(self) -> None:
        self.sent: list[tuple[str, Any]] = []

    def send_output(self, output_id: str, value: Any, /) -> None:
        self.sent.append((output_id, value))

    def commands(self) -> list[dict[str, Any]]:
        """把发到 bench_cmd 上的命令解回 dict。"""
        return [wire.decode(v) for oid, v in self.sent if oid == "bench_cmd"]


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


CAPABILITIES = {
    "v": wire.WIRE_VERSION,
    "type": wire.REPORT_CAPABILITIES,
    "benchmark": "example_benchmark",
    "suite": "build_tower",
    "policy_id": "demo_policy",
    "env_cfg_type": "arx_x5",
    "action_type": "joint",
    "num_tasks": 2,
    "tasks": [
        {"task_id": 0, "name": "build_tower", "language": "build a tower", "num_init_states": 3},
        {"task_id": 1, "name": "other", "language": "do other", "num_init_states": 2},
    ],
    "joint_names": ["j0", "j1"],
    "camera_ids": ["image/head", "image/left_wrist", "image/right_wrist"],
    "defaults": {"num_runs": 1, "seed": 0, "max_steps": 300},
    "limits": {"max_total_episodes": 6, "max_max_steps": 2000},
    # 内部字段,describe 必须把它过滤掉
    "result_dir": "/tmp/should-not-leak",
}


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


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def peer(clock: FakeClock) -> BenchPeer:
    p = BenchPeer(heartbeat_timeout_s=3.0)
    p.on_report(CAPABILITIES, clock())
    return p


@pytest.fixture
def cold_peer() -> BenchPeer:
    """还没收到 capabilities 的对端。"""
    return BenchPeer(heartbeat_timeout_s=3.0)


@pytest.fixture
def node() -> FakeNode:
    return FakeNode()


@pytest.fixture
def runner(node: FakeNode, peer: BenchPeer, clock: FakeClock) -> BenchEndpointRunner:
    return BenchEndpointRunner(node, peer=peer, clock=clock)
