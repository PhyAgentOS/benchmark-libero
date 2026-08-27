"""LIBERO 节点配置：共有字段来自 _core，这里只加 LIBERO 特有的。"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

def _import_core(name):
    """导 _core.<name>；父目录只临时挂上，避免遮蔽同名仿真器包（见 main.py 注释）。"""
    import importlib

    try:
        return importlib.import_module("_core." + name)
    except ImportError:
        p = str(Path(__file__).resolve().parents[1])
        sys.path.insert(0, p)
        try:
            return importlib.import_module("_core." + name)
        finally:
            if p in sys.path:
                sys.path.remove(p)


_m = _import_core("config")
BaseBenchmarkConfig = _m.BaseBenchmarkConfig
fill_base = _m.fill_base
read_yaml = _m.read_yaml
derive_episode_seed = _m.derive_episode_seed
resolve_and_validate_plan = _m.resolve_and_validate_plan

# proprio 的关节名。JointState 按名字查表，名字对不上会静默取到空值，
# 所以必须和 policy 配置的 state_joints 逐字一致。
STATE_JOINTS: list[str] = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper",
]

# 8D 观测的语义顺序：eef_pos(3) + axis_angle(3) + gripper_qpos(2)。
# 与 STATE_JOINTS 逐位对应——名字是标签，位置才是契约。
STATE_SEMANTICS: list[str] = [
    "eef_pos_x", "eef_pos_y", "eef_pos_z",
    "eef_ax_x", "eef_ax_y", "eef_ax_z",
    "gripper_qpos_0", "gripper_qpos_1",
]

LAYOUT_MODES = ("per_episode", "fixed", "legacy")


@dataclass
class LiberoConfig(BaseBenchmarkConfig):
    benchmark: str = "libero"
    suite: str = "libero_spatial"
    num_steps_wait: int = 10
    control_mode: str = "relative"
    camera_height: int = 256
    camera_width: int = 256
    # 家具（无关节 fixture）位姿存在 model.body_pos/body_quat 里，属于"模型"而非
    # "状态"，set_init_state 只恢复 qpos/qvel 恢复不到。三种口径见 README §7 坑 1。
    layout_mode: str = "per_episode"
    policy_id: str = "vla_jepa_libero"


def load_config(path: str | Path) -> LiberoConfig:
    data = read_yaml(path)
    cfg = LiberoConfig()
    fill_base(cfg, data)

    cfg.num_steps_wait = int(data.get("num_steps_wait", cfg.num_steps_wait))
    cfg.control_mode = str(data.get("control_mode", cfg.control_mode))
    cfg.camera_height = int(data.get("camera_height", cfg.camera_height))
    cfg.camera_width = int(data.get("camera_width", cfg.camera_width))
    cfg.layout_mode = str(data.get("layout_mode", cfg.layout_mode))
    if cfg.layout_mode not in LAYOUT_MODES:
        raise ValueError(f"layout_mode 必须是 {LAYOUT_MODES} 之一，收到 {cfg.layout_mode!r}")
    return cfg


# Backward-compatible name used by the original standalone LIBERO node consumers.
BenchmarkNodeConfig = LiberoConfig
