"""_core 共享核心的契约测试夹具。**不 import 任何仿真器,不占 GPU。**

跑法(仓库根或 benchmarks/ 下都行):
    <python> -m pytest packages/nodes/benchmarks/_core/tests -q

`_core` 由 LIBERO / RoboCasa / CALVIN 共用,所以这里的 fake adapter 只依赖
`_core/adapter.py` 声明的协议,不带任何 benchmark 特化。
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pytest

# benchmarks/ 进 sys.path,才能 `import _core.*`。
# 各节点 main.py 里那套「父目录只在导入期间挂上」的讲究这里不需要:
# 测试进程不 import 任何与节点目录同名的仿真器包。
_BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from _core.adapter import StepOutcome  # noqa: E402
from _core.config import BaseBenchmarkConfig  # noqa: E402

JOINTS = ["j0", "j1"]


class FakeNode:
    """替代 dora.Node,收集 send_output。"""

    def __init__(self) -> None:
        self.sent: list[tuple[str, Any]] = []

    def send_output(self, output_id: str, value: Any, /) -> None:
        self.sent.append((output_id, value))

    def outputs(self, output_id: str) -> list[Any]:
        return [v for oid, v in self.sent if oid == output_id]


class FakeAdapter:
    """两个 task,init state 数**故意不同**(3 / 2)——逐 task 校验的回归靠这个。

    `self.cfg` 初始为 None:用来钉住「判据必须是 hasattr 而不是 is not None」。
    `reset_episode` 把拿到的 `cfg.seed` 记下来,用于证明 Tool plan 的 seed
    真的到了 adapter(LIBERO 就是靠 cfg.seed 播家具布局的)。
    """

    RECORD_FPS = 20.0

    def __init__(self, *, succeed: bool = True, steps_per_episode: int = 2) -> None:
        self.succeed = succeed
        self.steps_per_episode = steps_per_episode
        self.cfg: Any = None
        self.closed = 0
        self.resets: list[dict[str, Any]] = []
        self._step_idx = 0

    # ---- 清单 ----
    @property
    def num_tasks(self) -> int:
        return 2

    def num_init_states(self, task_id: int) -> int:
        return 3 if task_id == 0 else 2

    def task_language(self, task_id: int) -> str:
        return f"fake task {task_id}"

    @property
    def tasks(self) -> list[str]:
        return ["alpha", "beta"]

    def state_joint_names(self) -> list[str]:
        return list(JOINTS)

    def reproducibility(self) -> dict[str, Any]:
        return {"fake": True}

    # ---- 执行 ----
    def reset_episode(
        self, task_id: int, init_state_id: int, run_index: int, episode_seed: int
    ) -> dict[str, Any]:
        self._step_idx = 0
        self.resets.append(
            {
                "task_id": task_id,
                "init_state_id": init_state_id,
                "run_index": run_index,
                "episode_seed": episode_seed,
                # adapter 在 reset 时回读 cfg —— 记下它当时看到的值
                "cfg_seed": getattr(self.cfg, "seed", None),
                "cfg_max_steps": getattr(self.cfg, "max_steps", None),
            }
        )
        return {"state": np.zeros(2)}

    def step(self, action: np.ndarray) -> StepOutcome:
        self._step_idx += 1
        done = self._step_idx >= self.steps_per_episode
        return StepOutcome(
            obs={"state": np.zeros(2)},
            reward=1.0 if (done and self.succeed) else 0.0,
            done=done,
            success=bool(done and self.succeed),
        )

    def to_state(self, obs: dict[str, Any]) -> np.ndarray:
        return np.zeros(2, dtype=np.float64)

    def to_images(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        return {}

    def close(self) -> None:
        self.closed += 1


class NoCfgAdapter(FakeAdapter):
    """没有 cfg 属性的 adapter —— 回写必须优雅跳过,不能抛。"""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        del self.cfg


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
    cfg.result_dir = str(tmp_path / "results")
    cfg.record_dir = str(tmp_path / "videos")
    return cfg


def make_plan(**overrides: Any) -> dict[str, Any]:
    plan = {
        "task_ids": [0],
        "init_state_ids": [0],
        "num_runs": 1,
        "seed": 0,
        "max_steps": 8,
    }
    plan.update(overrides)
    return plan


def drive(session, max_actions: int = 200) -> None:
    """喂零动作直到批次结束。1:1 锁步,每个 action 推一步。"""
    from forge_msgs import JointCommand

    session.start()
    for _ in range(max_actions):
        if session.finished:
            return
        session.on_action(
            JointCommand.from_np(np.zeros(len(JOINTS)), JOINTS, "position")
        )
    raise AssertionError("批次没有在预期步数内结束")
