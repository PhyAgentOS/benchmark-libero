"""动作维度契约：adapter 声明 action_dim，核心在 step 之前逐个 action 校验。

维度不符是 policy 与 benchmark 的契约错配。旧行为是每集记 step_error 后继续跑，
整批静默出 0 分；现在必须整批 failed 并带 action_dim_mismatch。
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
from forge_msgs import JointCommand

from _core import session as S
from _core.adapter import ACTION_DIM_MISMATCH

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_core_tests_conftest", os.path.join(_here, "conftest.py")
)
assert _spec is not None and _spec.loader is not None
_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conftest)
JOINTS = _conftest.JOINTS
FakeAdapter = _conftest.FakeAdapter
drive = _conftest.drive
make_plan = _conftest.make_plan


class CountingAdapter(FakeAdapter):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.steps = 0

    def step(self, action):
        self.steps += 1
        return super().step(action)


def _action(n: int) -> JointCommand:
    return JointCommand.from_np(np.zeros(n), [f"a{i}" for i in range(n)], "position")


def test_matching_action_dim_runs_normally(cfg_template, node) -> None:
    adapter = CountingAdapter()
    sess = S.BenchSession(node, cfg_template, adapter, make_plan(init_state_ids=[0, 1]), "inv-1")
    drive(sess)
    assert sess.runner.result.status == "succeeded"
    assert sess.runner.result.error_code is None
    assert adapter.steps == 4


@pytest.mark.parametrize("bad_dim", [1, 7])
def test_mismatched_action_dim_aborts_the_whole_batch(cfg_template, node, bad_dim) -> None:
    adapter = CountingAdapter()
    sess = S.BenchSession(
        node, cfg_template, adapter, make_plan(init_state_ids=[0, 1, 2]), "inv-1"
    )
    sess.start()
    sess.on_action(_action(bad_dim))

    runner = sess.runner
    assert sess.finished
    assert runner.result.status == "failed"
    assert runner.failure_code == ACTION_DIM_MISMATCH
    assert f"{bad_dim}" in runner.failure_summary and f"{len(JOINTS)}" in runner.failure_summary
    # 错配的 action 绝不能进环境
    assert adapter.steps == 0
    # 当前集留证据，后面的集不再开（不是 3 集各记一次失败）
    assert len(runner.result.episodes) == 1
    assert len(adapter.resets) == 1
    ep = runner.result.episodes[0]
    assert ep.error_code == ACTION_DIM_MISMATCH and ep.termination == "failed"
    payload = runner.result.to_dict()
    assert payload["error_code"] == ACTION_DIM_MISMATCH
    assert payload["valid_episodes"] == 0


def test_actions_after_abort_are_ignored(cfg_template, node) -> None:
    adapter = CountingAdapter()
    sess = S.BenchSession(node, cfg_template, adapter, make_plan(), "inv-1")
    sess.start()
    sess.on_action(_action(7))
    sess.on_action(_action(len(JOINTS)))
    assert adapter.steps == 0
    assert len(node.outputs("benchmark_result")) == 1


def test_undeclared_action_dim_keeps_legacy_behaviour(cfg_template, node) -> None:
    """没声明 action_dim 的 adapter：核心不拦，维度检查仍归 adapter.step。"""
    adapter = CountingAdapter()
    adapter.action_dim = None
    sess = S.BenchSession(node, cfg_template, adapter, make_plan(), "inv-1")
    sess.start()
    sess.on_action(_action(7))
    assert adapter.steps == 1


@pytest.mark.parametrize("bad", [0, -3, 7.0, "7", True])
def test_invalid_declared_action_dim_is_rejected_up_front(cfg_template, node, bad) -> None:
    adapter = FakeAdapter()
    adapter.action_dim = bad
    with pytest.raises(ValueError, match="action_dim"):
        S.BenchSession(node, cfg_template, adapter, make_plan(), "inv-1")


def test_capabilities_expose_action_dim(cfg_template, adapter) -> None:
    assert S.capabilities_payload(cfg_template, adapter)["action_dim"] == len(JOINTS)
    adapter.action_dim = None
    assert S.capabilities_payload(cfg_template, adapter)["action_dim"] is None
