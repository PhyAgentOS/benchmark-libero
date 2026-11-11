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
    # OSC_POSE：3 平移 + 3 旋转 + 1 夹爪。核心按它在 step 前校验，维度不符整批 failed。
    ACTION_DIM = 7

    @property
    def action_dim(self) -> int:
        return self.ACTION_DIM

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
        self._benchmark_dict = benchmark.get_benchmark_dict()
        # suite 对象按名字懒加载并缓存。当前 suite 永远从 self.cfg.suite 读：
        # serve 模式下 BenchSession 会把 adapter.cfg 换成本次 run 的 cfg。
        self._suites: dict[str, Any] = {}
        # 默认 suite 和白名单里每个名字都在启动时校验：写错要当场炸，
        # 不能等到某次 run 换过去才发现
        allowed = getattr(cfg, "allowed_suites", None)
        for name in allowed() if callable(allowed) else [cfg.suite]:
            self._load_suite(name)

        # 下面三份缓存的键都是 (suite, task_id)。只按 task_id 缓存的话，
        # 先跑 libero_10 的 task 3 再跑 libero_spatial 的 task 3 会复用上一个
        # suite 的 env / init state / 布局，分数照出、不报任何错。
        self._env = None
        self._env_key: tuple[str, int] | None = None
        self._init_states_cache: dict[tuple[str, int], np.ndarray] = {}
        self._task_layout: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    # ---- suite ----

    def _load_suite(self, name: str):
        if name not in self._suites:
            factory = self._benchmark_dict.get(name)
            if factory is None:
                raise ValueError(
                    f"未知 LIBERO suite {name!r}；可用: {sorted(self._benchmark_dict)}"
                )
            self._suites[name] = factory()
        return self._suites[name]

    @property
    def suite(self):
        """当前 suite 对象（= self.cfg.suite）。"""
        return self._load_suite(self.cfg.suite)

    def suite_view(self, name: str) -> "_SuiteView":
        """某个 suite 的只读清单视图，供 describe / plan 校验用，不碰 env。"""
        return _SuiteView(self, name)

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
        return self._num_tasks(self.cfg.suite)

    def num_init_states(self, task_id: int) -> int:
        return self._num_init_states(self.cfg.suite, task_id)

    def task_language(self, task_id: int) -> str:
        return self._task_language(self.cfg.suite, task_id)

    def _num_tasks(self, suite: str) -> int:
        return len(self._load_suite(suite).tasks)

    def _num_init_states(self, suite: str, task_id: int) -> int:
        return int(np.asarray(self._init_states(task_id, suite)).shape[0])

    def _task_language(self, suite: str, task_id: int) -> str:
        return str(self._load_suite(suite).get_task(task_id).language)

    def _init_states(self, task_id: int, suite: str | None = None) -> np.ndarray:
        key = (suite or self.cfg.suite, task_id)
        if key not in self._init_states_cache:
            self._init_states_cache[key] = np.asarray(
                self._load_suite(key[0]).get_task_init_states(task_id)
            )
        return self._init_states_cache[key]

    # ---- env 生命周期 ----

    def _ensure_env(self, task_id: int):
        """同一个 (suite, task) 复用 env；换任何一个就重建（建 env 约 2s）。"""
        key = (self.cfg.suite, task_id)
        if self._env is not None and self._env_key == key:
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
        self._env_key = key
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
            layout_key = (self.cfg.suite, task_id)
            layout = self._task_layout.get(layout_key)
            if layout is None:
                self._task_layout[layout_key] = self.capture_layout()
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
        # 严格等于 7：多出来的维度不再截断。截断会把维度错配的 policy 静默跑成 0 分。
        if act.size != self.ACTION_DIM or not np.all(np.isfinite(act)):
            raise ValueError(
                f"LIBERO action must contain exactly {self.ACTION_DIM} finite values, got size={act.size}"
            )
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
        self._env_key = None

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


class _SuiteView:
    """LiberoAdapter 在某个 suite 上的只读清单视图。

    只回答 describe / plan 校验要问的问题（task 数、init state 数、指令），
    不建 env、不改 adapter.cfg。其余属性（action_dim、state_joint_names、
    reproducibility…）与 suite 无关，原样转给 adapter。
    """

    def __init__(self, adapter: LiberoAdapter, suite: str) -> None:
        adapter._load_suite(suite)  # 名字不对当场抛 ValueError
        self._adapter = adapter
        self.suite_name = suite

    @property
    def num_tasks(self) -> int:
        return self._adapter._num_tasks(self.suite_name)

    def num_init_states(self, task_id: int) -> int:
        return self._adapter._num_init_states(self.suite_name, task_id)

    def task_language(self, task_id: int) -> str:
        return self._adapter._task_language(self.suite_name, task_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)
