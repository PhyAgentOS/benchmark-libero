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

LAYOUT_MODES = ("per_episode", "fixed", "legacy")


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


def derive_episode_seed(
    base_seed: int, task_id: int, init_state_id: int, run_index: int
) -> int:
    """把 episode 身份映射成一个稳定的种子。

    刻意不用内建 `hash()`：它对 str/bytes 带 PYTHONHASHSEED 随机化，跨进程不一致，
    正好会毁掉我们想要的复现性。这里用固定的整数混合，纯算术、跨进程恒定。
    """
    h = (int(base_seed) & 0xFFFF) * 1_000_003
    h ^= (int(task_id) + 1) * 73_856_093
    h ^= (int(init_state_id) + 1) * 19_349_663
    h ^= (int(run_index) + 1) * 83_492_791
    return h & 0x7FFF_FFFF


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

    # 家具（无关节的 fixture）位姿存在 model.body_pos/body_quat 里，属于"模型"而非
    # "状态"，而 set_init_state 只恢复 qpos/qvel——所以 env.reset() 重新采样的家具
    # 摆放不会被 init_state 恢复。实测同一 (task, init_state) 不同 reset 次序下炉子
    # 会漂 1.5 cm、渲染图 10% 像素不同。三种口径：
    #   per_episode（默认）：每 episode 按身份重新 seed → 布局是 (task,init,run) 的纯函数
    #   fixed             ：每个 task 只用一套布局 → init_state 唯一确定场景
    #   legacy            ：旧行为，布局取决于"该 task 内第几次 reset"（复现已发表口径时用）
    layout_mode: str = "per_episode"

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

    layout_mode = str(data.get("layout_mode", "per_episode"))
    if layout_mode not in LAYOUT_MODES:
        raise ValueError(f"layout_mode 必须是 {LAYOUT_MODES} 之一，收到 {layout_mode!r}")

    cfg = BenchmarkNodeConfig(
        suite=data.get("suite", "libero_spatial"),
        num_runs=int(data.get("num_runs", 1)),
        max_steps=int(data.get("max_steps", 300)),
        num_steps_wait=int(data.get("num_steps_wait", 10)),
        control_mode=str(data.get("control_mode", "relative")),
        seed=int(data.get("seed", 0)),
        camera_height=int(data.get("camera_height", 256)),
        camera_width=int(data.get("camera_width", 256)),
        layout_mode=layout_mode,
        policy_id=str(data.get("policy_id", "vla_jepa_libero")),
        send_instruction=bool(data.get("send_instruction", True)),
        result_dir=str(data.get("result_dir", "results/benchmark")),
        record_dir=data.get("record_dir"),
    )
    if cfg.num_runs < 1:
        raise ValueError(f"num_runs 必须 ≥1，收到 {cfg.num_runs}")
    if cfg.max_steps < 1:
        raise ValueError(f"max_steps 必须 ≥1，收到 {cfg.max_steps}")

    # task/init_state 的默认全集由 adapter 在拿到 suite 后回填（见 main.py resolve_plan）。
    cfg.task_ids = parse_id_list(data.get("task_ids"), [])
    cfg.init_state_ids = parse_id_list(data.get("init_state_ids"), [])
    return cfg


def resolve_and_validate_plan(cfg: BenchmarkNodeConfig, adapter: Any) -> None:
    """把 'all'/空配置回填成真实全集，并逐 task 校验 id 范围。

    必须 fail fast：越界 id 如果留到 episode 推进时才炸，异常会在事件循环的
    except 分支里再次抛出，冲出循环导致整批结果不落盘。
    """
    n_tasks = adapter.num_tasks
    if not cfg.task_ids:
        cfg.task_ids = list(range(n_tasks))

    bad = [t for t in cfg.task_ids if not 0 <= t < n_tasks]
    if bad:
        raise ValueError(f"suite {cfg.suite!r} 只有 {n_tasks} 个 task，越界 task_ids: {bad}")

    # 逐 task 取 init state 数：不同 task 的数量未必相同，不能拿 task_ids[0] 代表全体。
    counts = {t: adapter.num_init_states(t) for t in cfg.task_ids}
    if not cfg.init_state_ids:
        cfg.init_state_ids = list(range(min(counts.values())))

    for t in cfg.task_ids:
        bad = [i for i in cfg.init_state_ids if not 0 <= i < counts[t]]
        if bad:
            raise ValueError(
                f"task {t} 只有 {counts[t]} 个 init state，越界 init_state_ids: {bad}"
            )

    if not cfg.task_ids or not cfg.init_state_ids:
        raise ValueError("批次为空：task_ids 或 init_state_ids 解析后没有任何元素")
