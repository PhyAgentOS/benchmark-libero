"""LIBERO adapter：env 生命周期、观测转换、成功判定。

benchmark 特有的东西全部收敛在这里；_core 的调度不认识 LIBERO。
换 benchmark 就换这个模块（手册 §5.2）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

def _import_core(name):
    """导 _core.<name>；父目录只临时挂上，避免遮蔽同名仿真器包（见 main.py 注释）。"""
    import importlib

    try:
        return importlib.import_module("_core." + name)
    except ImportError:
        p = str(Path(__file__).resolve().parents[2])
        sys.path.insert(0, p)
        try:
            return importlib.import_module("_core." + name)
        finally:
            if p in sys.path:
                sys.path.remove(p)


_m = _import_core("adapter")
StepOutcome = _m.StepOutcome

# LIBERO 预热用的空动作：前 6 维为 0，gripper 维 -1.0 保持张开。
DUMMY_ACTION: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """LIBERO/robosuite 的四元数是 (x, y, z, w) 顺序。"""
    quat = np.asarray(quat, dtype=np.float32).reshape(4).copy()
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * np.arccos(w) / den)).astype(np.float32)


class LiberoAdapter:
    """进程内持有 LIBERO env，不经过 TargetWS、不经过 mujoco_sim。"""

    SUCCESS_SOURCE = "env.env._check_success"

    def __init__(self, cfg) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        # conda 版 mujoco 用 CONDA_PREFIX 定位 include/bin，取不到就是
        # os.path.join(None, ...) 当场 TypeError。而该变量只有 conda activate
        # 才会设——dora 直接拉起解释器时没有。sys.prefix 等价且不依赖启动方式。
        os.environ.setdefault("CONDA_PREFIX", sys.prefix)

        from libero.libero import benchmark, get_libero_path

        self.cfg = cfg
        self._get_libero_path = get_libero_path
        self.suite = benchmark.get_benchmark_dict()[cfg.suite]()

        self._env = None
        self._env_task_id: int | None = None
        self._init_states_cache: dict[int, np.ndarray] = {}
        self._task_layout: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # ---- 给结果文件的复现口径 ----

    def reproducibility(self) -> dict[str, Any]:
        return {
            "layout_mode": self.cfg.layout_mode,
            "control_mode": self.cfg.control_mode,
            "num_steps_wait": self.cfg.num_steps_wait,
            "camera": [self.cfg.camera_height, self.cfg.camera_width],
        }

    def state_joint_names(self) -> list[str]:
        from config import STATE_JOINTS

        return STATE_JOINTS

    # ---- 批次展开需要的元信息 ----

    @property
    def num_tasks(self) -> int:
        return len(self.suite.tasks)

    def num_init_states(self, task_id: int) -> int:
        return int(np.asarray(self._init_states(task_id)).shape[0])

    def task_language(self, task_id: int) -> str:
        return str(self.suite.get_task(task_id).language)

    def _init_states(self, task_id: int) -> np.ndarray:
        if task_id not in self._init_states_cache:
            self._init_states_cache[task_id] = np.asarray(
                self.suite.get_task_init_states(task_id)
            )
        return self._init_states_cache[task_id]

    # ---- env 生命周期 ----

    def _ensure_env(self, task_id: int):
        """同一个 task 复用 env；换 task 才重建（建 env 约 2s）。"""
        if self._env is not None and self._env_task_id == task_id:
            return self._env

        self.close()
        from libero.libero.envs import OffScreenRenderEnv

        task = self.suite.get_task(task_id)
        bddl = os.path.join(
            self._get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=int(self.cfg.camera_height),
            camera_widths=int(self.cfg.camera_width),
        )
        env.seed(self.cfg.seed)
        self._env = env
        self._env_task_id = task_id
        return env

    # ---- 场景布局：fixture 位姿在 model 里，不在 state 里 ----

    def _sim(self):
        for target in (getattr(self._env, "env", None), self._env):
            sim = getattr(target, "sim", None)
            if sim is not None:
                return sim
        raise RuntimeError("LIBERO env 上找不到 sim")

    def capture_layout(self) -> tuple[np.ndarray, np.ndarray]:
        m = self._sim().model
        return (
            np.array(m.body_pos, dtype=np.float64).copy(),
            np.array(m.body_quat, dtype=np.float64).copy(),
        )

    def restore_layout(self, layout: tuple[np.ndarray, np.ndarray]) -> None:
        m = self._sim().model
        m.body_pos[:] = layout[0]
        m.body_quat[:] = layout[1]

    # ---- episode ----

    def reset_episode(
        self, task_id: int, init_state_id: int, run_index: int, episode_seed: int
    ) -> dict[str, Any]:
        env = self._ensure_env(task_id)

        # per_episode：把摆放随机数钉在 episode 身份上，布局与"该 task 内第几次
        # reset"解耦——否则改 num_runs / init_state_ids 会让同一个 init_state_id
        # 拿到另一套家具位置，跨配置的结果不可比。
        if self.cfg.layout_mode == "per_episode":
            env.seed(int(episode_seed))

        env.reset()

        if self.cfg.layout_mode == "fixed":
            layout = self._task_layout.get(task_id)
            if layout is None:
                self._task_layout[task_id] = self.capture_layout()
            else:
                self.restore_layout(layout)

        obs = env.set_init_state(self._init_states(task_id)[init_state_id])

        for _ in range(max(0, self.cfg.num_steps_wait)):
            obs, _, _, _ = env.step(DUMMY_ACTION)

        # use_delta 必须在 reset 之后设，reset 会把 controller 覆盖回默认值。
        use_delta = self.cfg.control_mode.lower() not in {"absolute", "abs"}
        for robot in env.robots:
            robot.controller.use_delta = use_delta

        return obs

    def step(self, action: np.ndarray) -> StepOutcome:
        env = self._require_env()
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        if act.size < 7 or not np.all(np.isfinite(act)):
            raise ValueError(f"LIBERO action must contain 7 finite values, got size={act.size}")
        act = act[:7]
        obs, reward, done, _info = env.step(act.tolist())
        success = self.check_success()
        return StepOutcome(
            obs=obs,
            reward=float(reward),
            done=bool(done),
            success=success,
            metrics={"success": success},
        )

    def check_success(self) -> bool:
        """实测：OffScreenRenderEnv 上没有 _check_success（旧手册写的是它，照抄会
        AttributeError），实现在被包一层的 env.env 上；info 是空字典。逐层兜底。"""
        env = self._require_env()
        for target in (getattr(env, "env", None), env):
            checker = getattr(target, "_check_success", None)
            if callable(checker):
                return bool(checker())
        raise RuntimeError("LIBERO env 上找不到 _check_success，无法判定成功")

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # noqa: BLE001 关闭失败不该影响批次收尾
                pass
        self._env = None
        self._env_task_id = None

    def _require_env(self):
        if self._env is None:
            raise RuntimeError("env 尚未创建，先调用 reset_episode")
        return self._env

    # ---- 观测转换 ----

    def to_state(self, obs: dict[str, Any]) -> np.ndarray:
        """8D proprio：eef_pos(3) + axis_angle(3) + gripper_qpos(2)。"""
        gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
        if gripper.size == 0:
            gripper = np.zeros(2, dtype=np.float32)
        elif gripper.size == 1:
            gripper = np.repeat(gripper, 2)
        return np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(3),
                quat_to_axisangle(obs["robot0_eef_quat"]),
                gripper[:2],
            ]
        ).astype(np.float64)

    def to_images(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """LIBERO 离屏渲染上下颠倒，必须 [::-1,::-1] 才匹配训练分布。

        坑：libero_sim_node.py 那个雏形不翻，是因为它的 obs 来自 server.py（已翻过）。
        本节点直接读 env，翻转必须自己做，否则策略看到倒立世界，分数安静地烂掉。
        """
        return {
            "image/main": np.ascontiguousarray(
                np.asarray(obs["agentview_image"], dtype=np.uint8)[::-1, ::-1]
            ),
            "image/second": np.ascontiguousarray(
                np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)[::-1, ::-1]
            ),
        }
