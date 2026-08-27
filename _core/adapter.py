"""Benchmark adapter 契约。

调度核心只认这个接口，**不 import 任何仿真器**——所以换 MuJoCo / SAPIEN / PyBullet
对核心是透明的。接一个新 benchmark = 实现这个协议 + 写配置，核心不动。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class StepOutcome:
    """单步结果。

    `success` 是主判定（长程任务里表示"整条链全部完成"）；benchmark 特有的成绩
    放 `metrics`，例如 CALVIN 的 `{"chain_length": 3, "subtasks": [...]}`。
    核心不解释 metrics 的内容，原样带进结果文件。
    """

    obs: dict[str, Any]
    reward: float
    done: bool
    success: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    # 长程/链式任务用：本步之后指令换了（如 CALVIN 一个 episode 串 5 个子任务，
    # 每个子任务有自己的语言指令）。非 None 时核心会重发 set_instruction 并让
    # policy 清队列，语义等价于 CALVIN 官方 rollout 里的 model.reset()。
    next_instruction: str | None = None


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """每个 benchmark 必须实现的十个方法。"""

    # ---- 批次展开需要的元信息 ----
    @property
    def num_tasks(self) -> int: ...

    def num_init_states(self, task_id: int) -> int: ...

    def task_language(self, task_id: int) -> str: ...

    # ---- episode 生命周期 ----
    def reset_episode(
        self, task_id: int, init_state_id: int, run_index: int, episode_seed: int
    ) -> dict[str, Any]:
        """重置到指定初始状态，返回首帧观测。"""
        ...

    def step(self, action: np.ndarray) -> StepOutcome: ...

    def close(self) -> None: ...

    # ---- 观测转换 ----
    def to_state(self, obs: dict[str, Any]) -> np.ndarray | None:
        """proprio 向量；返回 None 表示该 benchmark 不产生 proprio（纯视觉策略）。"""
        ...

    def to_images(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """{dora 通道名: HWC uint8}。"""
        ...

    # ---- 可选钩子（有默认实现，benchmark 不需要就别写）----
    def state_joint_names(self) -> list[str]:
        """proprio 的关节名，必须与 policy 配置的 state_joints 逐字一致。"""
        ...

    def aggregate(self, episode_metrics: list[dict[str, Any]]) -> tuple[dict, dict]:
        """把逐 episode 的 metrics 聚合成 (primary_metric, metrics)。

        不实现时核心用默认：primary_metric = success_rate。
        CALVIN 这类要覆盖成 avg_sequence_length。
        """
        ...


def default_aggregate(
    successes: int, total_episodes: int, episode_metrics: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """默认聚合：成功率。分母用**计划总数**，中途取消时不虚高。"""
    denom = total_episodes or 1
    return (
        {"name": "success_rate", "value": successes / denom, "max": 1.0},
        {},
    )
