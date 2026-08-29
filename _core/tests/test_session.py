"""BenchSession 的契约测试 —— 只覆盖 session 自己拥有的语义。

事件循环/idle 状态/并发拒绝属于 bench_tool/node.py 的 serve loop，不在这里测。

这里盯的都是「改错了照样出分、不报错」的地方：跑一遍是发现不了的。
"""

from __future__ import annotations

import importlib.util
import os
import re

import pytest

from _core import session as S

# 仓库里有两个 tests/conftest.py（_core 与 bench_tool），pytest 都把顶层模块
# 名注册成 "conftest"，`from conftest import ...` 会被后加载的覆盖。用独立模块
# 名显式加载本目录的 conftest（与 bench_tool/tests/conftest.py 用的是同一个
# 名字和同一个文件，sys.modules 里撞了也拿到同一对象）。
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_core_tests_conftest", os.path.join(_here, "conftest.py")
)
assert _spec is not None and _spec.loader is not None
_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conftest)
JOINTS = _conftest.JOINTS
FakeAdapter = _conftest.FakeAdapter
FakeNode = _conftest.FakeNode
NoCfgAdapter = _conftest.NoCfgAdapter
drive = _conftest.drive
make_plan = _conftest.make_plan


# ============================================================ 1. cfg 隔离


def test_session_does_not_mutate_the_template_cfg(cfg_template, node, adapter) -> None:
    """模板 cfg 被改脏的话，下一次 invocation 会带着上一次的参数跑，而且不报错。"""
    before = {
        "task_ids": list(cfg_template.task_ids),
        "init_state_ids": list(cfg_template.init_state_ids),
        "num_runs": cfg_template.num_runs,
        "seed": cfg_template.seed,
        "max_steps": cfg_template.max_steps,
    }
    S.BenchSession(
        node, cfg_template, adapter,
        make_plan(task_ids=[1], init_state_ids=[1], num_runs=2, seed=99, max_steps=42),
        "inv-1",
    )
    after = {
        "task_ids": list(cfg_template.task_ids),
        "init_state_ids": list(cfg_template.init_state_ids),
        "num_runs": cfg_template.num_runs,
        "seed": cfg_template.seed,
        "max_steps": cfg_template.max_steps,
    }
    assert before == after, f"模板 cfg 被污染：{before} → {after}"


def test_session_cfg_is_a_distinct_object(cfg_template, node, adapter) -> None:
    sess = S.BenchSession(node, cfg_template, adapter, make_plan(), "inv-1")
    assert sess.cfg is not cfg_template


# ============================================================ 2. adapter.cfg 回写


def test_adapter_cfg_points_at_session_cfg(cfg_template, node, adapter) -> None:
    """LiberoAdapter 在 reset 时回读 self.cfg 的 seed/layout_mode/control_mode。

    不指到 session 的 cfg，run 传的 seed 就到不了 adapter —— 家具布局用错
    种子，**分数照出、不报任何错**。
    """
    assert adapter.cfg is None, "夹具前提：初始为 None"
    sess = S.BenchSession(node, cfg_template, adapter, make_plan(seed=7), "inv-1")
    assert adapter.cfg is sess.cfg
    assert adapter.cfg is not cfg_template


def test_writeback_predicate_is_hasattr_not_none_check(cfg_template, node) -> None:
    """判据必须是 hasattr。用 `is not None` 的话，cfg 初始为 None 的 adapter
    会静默拿不到回写通道 —— 上一版就是这里写错，被这条测试抓到的。"""
    adapter = FakeAdapter()          # cfg 初始 None
    sess = S.BenchSession(node, cfg_template, adapter, make_plan(), "inv-1")
    assert adapter.cfg is sess.cfg, "cfg 为 None 时也必须完成回写"


def test_adapter_without_cfg_attribute_is_skipped(cfg_template, node) -> None:
    """没有 cfg 属性的 adapter 不该被强行塞一个，也不该抛。"""
    adapter = NoCfgAdapter()
    assert not hasattr(adapter, "cfg")
    S.BenchSession(node, cfg_template, adapter, make_plan(), "inv-1")
    assert not hasattr(adapter, "cfg")


# ============================================================ 3. plan 真正进入本次执行


def test_plan_fields_reach_this_runner(cfg_template, node, adapter) -> None:
    sess = S.BenchSession(
        node, cfg_template, adapter,
        make_plan(task_ids=[1], init_state_ids=[0, 1], num_runs=2, seed=5, max_steps=12),
        "inv-1",
    )
    assert sess.cfg.task_ids == [1]
    assert sess.cfg.init_state_ids == [0, 1]
    assert sess.cfg.num_runs == 2
    assert sess.cfg.seed == 5
    assert sess.cfg.max_steps == 12
    # runner 拿到的是同一个 cfg，而不是模板
    assert sess.runner.cfg is sess.cfg
    assert len(sess.runner.plan) == 1 * 2 * 2


def test_plan_seed_reaches_the_adapter_at_reset(cfg_template, node, adapter) -> None:
    """最关键的一条：seed 要在 **reset 时**被 adapter 读到。

    判据不用 max_steps 而用 seed —— LIBERO 是靠 cfg.seed 播家具布局的，
    seed 串了会静默换场景。
    """
    sess = S.BenchSession(node, cfg_template, adapter, make_plan(seed=1234), "inv-1")
    drive(sess)
    assert adapter.resets, "至少要 reset 一次"
    assert adapter.resets[0]["cfg_seed"] == 1234, (
        f"adapter 在 reset 时看到的 seed 是 {adapter.resets[0]['cfg_seed']}，"
        "不是 run 传的 1234"
    )
    assert adapter.resets[0]["cfg_max_steps"] == 8


def test_episode_seed_follows_the_plan_seed(cfg_template, node, adapter) -> None:
    """不同 base seed 必须导出不同的 episode_seed；同 plan 必须可复现。"""
    a = S.BenchSession(node, cfg_template, adapter, make_plan(seed=0), "inv-a")
    drive(a)
    seed_a = adapter.resets[-1]["episode_seed"]

    adapter_b = FakeAdapter()
    b = S.BenchSession(node, cfg_template, adapter_b, make_plan(seed=1), "inv-b")
    drive(b)
    seed_b = adapter_b.resets[-1]["episode_seed"]

    adapter_c = FakeAdapter()
    c = S.BenchSession(node, cfg_template, adapter_c, make_plan(seed=0), "inv-c")
    drive(c)
    seed_c = adapter_c.resets[-1]["episode_seed"]

    assert seed_a != seed_b, "换 base seed 必须换 episode_seed"
    assert seed_a == seed_c, "同 plan 必须复现同一个 episode_seed"


def test_out_of_range_plan_is_rejected(cfg_template, node, adapter) -> None:
    """_core 侧是权威兜底：endpoint 的 capabilities 可能对不上 adapter 的真实清单。"""
    with pytest.raises(ValueError):
        S.BenchSession(node, cfg_template, adapter, make_plan(task_ids=[9]), "inv-1")


def test_init_state_validated_per_task(cfg_template, node, adapter) -> None:
    """task 0 有 3 个 init state、task 1 只有 2 个；拿 task_ids[0] 代表全体是错的。"""
    with pytest.raises(ValueError):
        S.BenchSession(
            node, cfg_template, adapter,
            make_plan(task_ids=[0, 1], init_state_ids=[2]), "inv-1",
        )


# ============================================================ 4. result_tag / result_path


_UNTAGGED = re.compile(r"^fake_suite_\d{8}_\d{6}\.json$")


def test_result_path_carries_the_invocation_id(cfg_template, node, adapter) -> None:
    sess = S.BenchSession(node, cfg_template, adapter, make_plan(), "inv-abc123")
    drive(sess)
    path = sess.result_path
    assert path and path.endswith(".json")
    assert "inv-abc123" in path
    assert not _UNTAGGED.match(path.rsplit("/", 1)[-1]), "serve 模式不该退回无 tag 的旧文件名"


def test_two_sessions_never_share_a_result_path(cfg_template, node, adapter) -> None:
    """回归：文件名的时间戳**只到秒**，同一秒两次 invocation 会静默覆盖。"""
    paths = []
    for inv in ("inv-a", "inv-b"):
        a = FakeAdapter()
        sess = S.BenchSession(node, cfg_template, a, make_plan(), inv)
        drive(sess)
        paths.append(sess.result_path)
    assert paths[0] != paths[1], f"两次 invocation 写到了同一路径：{paths[0]}"


def test_session_does_not_close_the_adapter(cfg_template, node, adapter) -> None:
    """Skill Runtime 的生命周期长于单次 session；关了 env 下一次要重载整个场景。"""
    sess = S.BenchSession(node, cfg_template, adapter, make_plan(), "inv-1")
    drive(sess)
    assert adapter.closed == 0
    assert sess.runner.close_adapter_on_finish is False


# ============================================================ 5. 终态 result 与 request_stop


def test_batch_result_after_drive(cfg_template, node, adapter) -> None:
    sess = S.BenchSession(node, cfg_template, adapter, make_plan(init_state_ids=[0, 1]), "inv-1")
    drive(sess)
    result = sess.runner.result
    assert result.status == "succeeded"
    assert result.total_episodes == 2
    assert result.successes == 2
    assert sess.result_path and sess.result_path.endswith(".json")


def test_legacy_benchmark_result_channel_still_emitted(cfg_template, node, adapter) -> None:
    """旧的 benchmark_result 通道要保留，否则独立跑的下游脚本静默拿不到东西。"""
    sess = S.BenchSession(node, cfg_template, adapter, make_plan(), "inv-1")
    drive(sess)
    assert len(node.outputs("benchmark_result")) == 1


def test_request_stop_lands_at_episode_boundary(cfg_template, node, adapter) -> None:
    """stop 不是 mid-air 掐断：当前集跑完收分，后面的集不再开。"""
    import numpy as np
    from forge_msgs import JointCommand

    sess = S.BenchSession(
        node, cfg_template, adapter, make_plan(init_state_ids=[0, 1, 2]), "inv-1"
    )
    sess.start()  # episode 0 开跑，awaiting_action
    sess.request_stop("agent asked to stop")
    action = JointCommand.from_np(np.zeros(len(JOINTS)), JOINTS, "position")
    while not sess.finished:
        sess.on_action(action)

    result = sess.runner.result
    assert result.status == "stopped"
    assert len(result.episodes) == 1  # 当前集收分，后面的不再开
    assert sess.runner.stop_reason == "agent asked to stop"
    assert adapter.closed == 0


def test_stop_requested_before_start_runs_zero_episodes(cfg_template, node, adapter) -> None:
    """stop 落在 accepted 与拾取之间：一集都不该跑，adapter 一次 reset 都没有。"""
    sess = S.BenchSession(node, cfg_template, adapter, make_plan(init_state_ids=[0, 1]), "inv-1")
    sess.request_stop("stopped before pickup")
    drive(sess)
    result = sess.runner.result
    assert result.status == "stopped"
    assert len(result.episodes) == 0
    assert adapter.resets == []


def test_finish_records_failure_summary(cfg_template, node, adapter) -> None:
    sess = S.BenchSession(node, cfg_template, adapter, make_plan(), "inv-1")
    sess.runner.finish("failed", failure_summary="仿真器崩了")
    assert sess.runner.result.status == "failed"
    assert sess.runner.failure_summary == "仿真器崩了"


def test_progress_reports_only_on_episode_boundary(cfg_template, node, adapter) -> None:
    sess = S.BenchSession(node, cfg_template, adapter, make_plan(init_state_ids=[0, 1]), "inv-1")
    sess.start()
    first = sess.progress_report()
    assert first is not None and first["episode_index"] == 0
    assert sess.progress_report() is None  # 同一集内不重复发


# ============================================================ 6. 两次连续 session


def test_second_session_is_fully_independent(cfg_template, node, adapter) -> None:
    """第二次 invocation 必须用自己的 cfg / result，不继承第一次的状态。"""
    first = S.BenchSession(node, cfg_template, adapter, make_plan(seed=0, init_state_ids=[0]), "inv-1")
    drive(first)

    second = S.BenchSession(
        node, cfg_template, adapter,
        make_plan(seed=42, init_state_ids=[1], max_steps=16), "inv-2",
    )
    drive(second)

    # 独立的 cfg 对象与取值
    assert second.cfg is not first.cfg
    assert second.cfg.seed == 42 and first.cfg.seed == 0
    assert second.cfg.init_state_ids == [1] and first.cfg.init_state_ids == [0]
    assert second.cfg.max_steps == 16 and first.cfg.max_steps == 8
    # adapter 现在指向第二次的 cfg
    assert adapter.cfg is second.cfg
    # 独立的 result 与落盘路径
    assert second.runner.result is not first.runner.result
    assert second.result_path != first.result_path
    assert second.runner.result.status == "succeeded"
    # 第二次确实换了 init state，不是把第一次重跑一遍
    assert adapter.resets[-1]["init_state_id"] == 1
    assert adapter.resets[-1]["cfg_seed"] == 42
    # 模板始终未被污染
    assert cfg_template.seed == 0 and cfg_template.init_state_ids == []
    # 全程没关过 adapter
    assert adapter.closed == 0


# ============================================================ capabilities


def test_capabilities_carries_per_task_counts(cfg_template, adapter) -> None:
    caps = S.capabilities_payload(cfg_template, adapter, limits={"max_total_episodes": 6})
    assert "type" not in caps  # 进程内直读，不需要线协议的类型戳
    assert caps["num_tasks"] == 2
    assert {t["task_id"]: t["num_init_states"] for t in caps["tasks"]} == {0: 3, 1: 2}
    assert caps["tasks"][0]["name"] == "alpha"
    assert caps["joint_names"] == ["j0", "j1"]
    assert caps["defaults"] == {"num_runs": 1, "seed": 0, "max_steps": 300}
    assert caps["limits"]["max_total_episodes"] == 6
    # benchmark 特有口径走 reproducibility 透传，_core 不认识它们
    assert caps["reproducibility"] == {"fake": True}


def test_capabilities_camera_ids_fall_back_to_empty(cfg_template, adapter) -> None:
    """LIBERO 的 config 没有 camera_map，adapter 也没有 camera_ids 钩子 —— 给空表，
    不为此改 adapter 协议。"""
    assert S.capabilities_payload(cfg_template, adapter)["camera_ids"] == []
