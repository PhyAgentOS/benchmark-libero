"""LiberoAdapter 的多 suite 缓存与 LiberoConfig 的官方步数口径。

不装 LIBERO 也能跑：adapter 用 object.__new__ 绕过 __init__（那里要 import libero），
suite 对象和 OffScreenRenderEnv 都是假的。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from adapters.libero_adapter import LiberoAdapter
from config import LiberoConfig, load_config


class _FakeSuite:
    def __init__(self, name: str, init_counts: list[int]) -> None:
        self.name = name
        self.init_counts = init_counts
        self.tasks = [f"{name}-{i}" for i in range(len(init_counts))]

    def get_task(self, task_id: int):
        return SimpleNamespace(
            language=f"{self.name} task {task_id}",
            problem_folder=self.name,
            bddl_file=f"{task_id}.bddl",
        )

    def get_task_init_states(self, task_id: int) -> np.ndarray:
        return np.zeros((self.init_counts[task_id], 4))


class _FakeEnv:
    def __init__(self, bddl_file_name: str, **_kw) -> None:
        self.bddl = bddl_file_name
        self.closed = False

    def seed(self, _seed: int) -> None:
        pass

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def adapter(monkeypatch) -> LiberoAdapter:
    envs = types.ModuleType("libero.libero.envs")
    envs.OffScreenRenderEnv = _FakeEnv
    monkeypatch.setitem(sys.modules, "libero", types.ModuleType("libero"))
    monkeypatch.setitem(sys.modules, "libero.libero", types.ModuleType("libero.libero"))
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs)

    a = object.__new__(LiberoAdapter)
    a.cfg = LiberoConfig(suite="libero_spatial", suites=["libero_spatial", "libero_10"])
    a._get_libero_path = lambda _key: "/bddl"
    a._benchmark_dict = {
        "libero_spatial": lambda: _FakeSuite("libero_spatial", [3, 3]),
        "libero_10": lambda: _FakeSuite("libero_10", [5, 5, 5]),
    }
    a._suites = {}
    a._env = None
    a._env_key = None
    a._init_states_cache = {}
    a._task_layout = {}
    return a


def test_unknown_suite_is_rejected_with_choices(adapter) -> None:
    with pytest.raises(ValueError, match="libero_spatial"):
        adapter.suite_view("libero_nope")


def test_suite_view_reads_other_suite_without_touching_cfg(adapter) -> None:
    view = adapter.suite_view("libero_10")
    assert view.num_tasks == 3
    assert view.num_init_states(0) == 5
    assert view.task_language(1) == "libero_10 task 1"
    assert view.action_dim == 7  # 与 suite 无关的属性转给 adapter
    assert adapter.cfg.suite == "libero_spatial"
    assert adapter.num_tasks == 2


def test_init_states_cache_is_keyed_by_suite(adapter) -> None:
    assert adapter.num_init_states(0) == 3
    adapter.cfg.suite = "libero_10"
    # 只按 task_id 缓存的话这里会拿到 spatial 的 3
    assert adapter.num_init_states(0) == 5
    assert set(adapter._init_states_cache) == {("libero_spatial", 0), ("libero_10", 0)}


def test_env_is_rebuilt_when_suite_changes_for_same_task(adapter) -> None:
    env_a = adapter._ensure_env(0)
    assert adapter._ensure_env(0) is env_a  # 同 (suite, task) 复用
    assert env_a.bddl.endswith("libero_spatial/0.bddl")

    adapter.cfg.suite = "libero_10"
    env_b = adapter._ensure_env(0)
    assert env_b is not env_a
    assert env_a.closed
    assert env_b.bddl.endswith("libero_10/0.bddl")
    assert adapter._env_key == ("libero_10", 0)


def test_close_forgets_env_key(adapter) -> None:
    adapter._ensure_env(0)
    adapter.close()
    assert adapter._env_key is None
    assert adapter._ensure_env(0) is not None


def test_libero_config_uses_official_steps_for_other_suites(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "suite: libero_spatial\n"
        "suites: [libero_spatial, libero_object, libero_goal, libero_10, libero_90]\n"
        "max_steps: 250\n"
        "max_steps_by_suite: {libero_goal: 333}\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.default_max_steps() == 250  # 默认 suite 沿用 profile 的 max_steps
    assert cfg.default_max_steps("libero_object") == 280
    assert cfg.default_max_steps("libero_goal") == 333
    assert cfg.default_max_steps("libero_10") == 520
    assert cfg.default_max_steps("libero_90") == 400
