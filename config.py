"""Configuration model for the LIBERO benchmark node."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

# Policy 的 state_joints 顺序（见 policy_vla_jepa.yaml）。JointState 按名字查表，
# 名字对不上会静默取到空值，所以这里必须和 policy 配置逐字一致。
STATE_JOINTS: list[str] = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "gripper",
]

# LIBERO 的 8D 观测语义顺序：eef_pos(3) + axis_angle(3) + gripper_qpos(2)。
# 与 STATE_JOINTS 逐位对应——名字是标签，位置才是契约。
STATE_SEMANTICS: list[str] = [
    "eef_pos_x",
    "eef_pos_y",
    "eef_pos_z",
    "eef_ax_x",
    "eef_ax_y",
    "eef_ax_z",
    "gripper_qpos_0",
    "gripper_qpos_1",
]


def parse_id_list(spec: Any, default: Iterable[int]) -> list[int]:
    """解析 episode 规格：'all' / '0-9' / '0,2-4,8' / [0,1,2] / None。"""
    if spec is None or spec == "all":
        return list(default)
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]

    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


@dataclass
class BenchmarkNodeConfig:
    suite: str = "libero_spatial"
    task_ids: list[int] = field(default_factory=list)
    init_state_ids: list[int] = field(default_factory=list)
    num_runs: int = 1
    max_steps: int = 300
    num_steps_wait: int = 10
    control_mode: str = "relative"
    seed: int = 0
    camera_height: int = 256
    camera_width: int = 256

    policy_id: str = "vla_jepa_libero"
    # 每个 episode 是否把 task 的语言指令下发给 policy。LIBERO 每个 task 指令不同，
    # 关掉就等于整批用同一句指令推理，分数没有意义。
    send_instruction: bool = True

    result_dir: str = "results/benchmark"
    record_dir: str | None = None

    @property
    def total_episodes(self) -> int:
        return len(self.task_ids) * len(self.init_state_ids) * self.num_runs

    def episode_plan(self) -> list[tuple[int, int, int]]:
        """展开批次：task_ids × init_state_ids × num_runs 的笛卡尔积，串行执行。"""
        return [
            (task_id, init_state_id, run_index)
            for task_id in self.task_ids
            for init_state_id in self.init_state_ids
            for run_index in range(self.num_runs)
        ]


def load_config(path: str | Path) -> BenchmarkNodeConfig:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    cfg = BenchmarkNodeConfig(
        suite=data.get("suite", "libero_spatial"),
        num_runs=int(data.get("num_runs", 1)),
        max_steps=int(data.get("max_steps", 300)),
        num_steps_wait=int(data.get("num_steps_wait", 10)),
        control_mode=str(data.get("control_mode", "relative")),
        seed=int(data.get("seed", 0)),
        camera_height=int(data.get("camera_height", 256)),
        camera_width=int(data.get("camera_width", 256)),
        policy_id=str(data.get("policy_id", "vla_jepa_libero")),
        send_instruction=bool(data.get("send_instruction", True)),
        result_dir=str(data.get("result_dir", "results/benchmark")),
        record_dir=data.get("record_dir"),
    )
    # task/init_state 的默认全集由 adapter 在拿到 suite 后回填（见 main.py）。
    cfg.task_ids = parse_id_list(data.get("task_ids"), [])
    cfg.init_state_ids = parse_id_list(data.get("init_state_ids"), [])
    return cfg
