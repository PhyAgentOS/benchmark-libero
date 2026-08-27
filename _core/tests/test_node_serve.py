"""node.py 的模式分叉与 serve loop 状态机。

session 自己的语义在 test_session.py,这里只测 node 层:idle/active 转换、
并发拒绝、cancel、以及 adapter 的最终释放。
"""

from __future__ import annotations

import numpy as np
import pytest
from forge_msgs import JointCommand

from _core import node as N
from _core import session as S
from conftest import JOINTS, FakeAdapter, FakeNode, make_plan


# ---------------------------------------------------------------- 事件构造


def cmd(payload: dict) -> dict:
    return {"type": "INPUT", "id": N.CMD_INPUT_ID, "value": S.encode(payload)}


def run_cmd(invocation_id: str, **plan_overrides) -> dict:
    return cmd({"type": S.CMD_RUN, "invocation_id": invocation_id,
                "plan": make_plan(**plan_overrides)})


def cancel_cmd(invocation_id: str) -> dict:
    return cmd({"type": S.CMD_CANCEL, "invocation_id": invocation_id})


def action_event() -> dict:
    return {
        "type": "INPUT",
        "id": "action",
        "value": JointCommand.from_np(np.zeros(len(JOINTS)), JOINTS, "position").to_arrow(),
    }


def reports(node: FakeNode) -> list[dict]:
    return [S.decode(v) for oid, v in node.sent if oid == N.REPORT_OUTPUT_ID]


def kinds(node: FakeNode) -> list[str]:
    return [r["type"] for r in reports(node)]


def drive_to_completion(loop: N.ServeLoop, limit: int = 200) -> None:
    for _ in range(limit):
        if not loop.busy:
            return
        loop.on_event(action_event())
    raise AssertionError("invocation 没有在预期步数内结束")


# ================================================ 1-2. 模式分叉


def test_serve_false_takes_the_autorun_path(cfg_template, adapter, monkeypatch) -> None:
    cfg_template.serve = False
    called = []
    monkeypatch.setattr(N, "_run_autorun", lambda *a, **k: called.append("autorun") or 0)
    monkeypatch.setattr(N, "_run_serve", lambda *a, **k: called.append("serve") or 0)
    assert N.run_node(cfg_template, adapter) == 0
    assert called == ["autorun"], "serve=False 必须走 autorun"


def test_missing_serve_attribute_takes_autorun(adapter, monkeypatch) -> None:
    """连 serve 属性都没有的老 cfg 也必须走 autorun,不能抛。"""
    class OldCfg:
        pass

    called = []
    monkeypatch.setattr(N, "_run_autorun", lambda *a, **k: called.append("autorun") or 0)
    monkeypatch.setattr(N, "_run_serve", lambda *a, **k: called.append("serve") or 0)
    N.run_node(OldCfg(), adapter)
    assert called == ["autorun"]


def test_serve_true_takes_the_serve_path(cfg_template, adapter, monkeypatch) -> None:
    cfg_template.serve = True
    called = []
    monkeypatch.setattr(N, "_run_autorun", lambda *a, **k: called.append("autorun") or 0)
    monkeypatch.setattr(N, "_run_serve", lambda *a, **k: called.append("serve") or 0)
    N.run_node(cfg_template, adapter)
    assert called == ["serve"]


def test_serve_starts_idle_and_runs_nothing(cfg_template, node, adapter) -> None:
    """serve 起来后**不自动执行 benchmark**,等 bench_cmd。"""
    loop = N.ServeLoop(node, cfg_template, adapter)
    assert loop.busy is False
    assert loop.session is None
    assert adapter.resets == [], "启动阶段一个 episode 都不该跑"
    # 空转若干轮:超时事件、tick,都不该启动任何东西
    loop.on_event(None)
    loop.on_event({"type": "ERROR", "error": "Timeout event stream error: Receiver timed out"})
    loop.on_event({"type": "INPUT", "id": "tick", "value": None})
    assert loop.busy is False and adapter.resets == []


def test_timeout_error_is_not_fatal(cfg_template, node, adapter) -> None:
    """dora 的 next(timeout=) 超时表达成 ERROR;认错了会把整批判失败。"""
    loop = N.ServeLoop(node, cfg_template, adapter)
    timeout_evt = {"type": "ERROR", "error": "Timeout event stream error: Receiver timed out"}
    assert loop.on_event(timeout_evt) is None
    real_err = {"type": "ERROR", "error": "node crashed"}
    assert loop.on_event(real_err) == 1


# ================================================ 3-4. run → active → idle


def test_run_creates_session_and_goes_active(cfg_template, node, adapter) -> None:
    loop = N.ServeLoop(node, cfg_template, adapter)
    loop.on_event(run_cmd("inv-1"))
    assert loop.busy is True
    assert loop.session.invocation_id == "inv-1"
    assert S.REPORT_ACCEPTED in kinds(node)
    assert adapter.resets, "run 之后应当已经 reset 了第一集"


def test_completion_returns_to_idle_and_reports_result(cfg_template, node, adapter) -> None:
    loop = N.ServeLoop(node, cfg_template, adapter)
    loop.on_event(run_cmd("inv-1"))
    drive_to_completion(loop)

    assert loop.busy is False and loop.session is None, "跑完必须回 idle"
    results = [r for r in reports(node) if r["type"] == S.REPORT_RESULT]
    assert len(results) == 1
    assert results[0]["invocation_id"] == "inv-1"
    assert results[0]["payload"]["status"] == "succeeded"
    assert results[0]["payload"]["success_rate"] == 1.0


def test_action_when_idle_is_dropped(cfg_template, node, adapter) -> None:
    """上一批的迟到动作不该打死节点,也不该凭空建 session。"""
    loop = N.ServeLoop(node, cfg_template, adapter)
    loop.on_event(action_event())
    assert loop.busy is False and adapter.resets == []


# ================================================ 5. 第二次 run


def test_second_run_gets_a_brand_new_session(cfg_template, node, adapter) -> None:
    loop = N.ServeLoop(node, cfg_template, adapter)
    loop.on_event(run_cmd("inv-1", seed=0, init_state_ids=[0]))
    first = loop.session
    first_cfg = first.cfg
    drive_to_completion(loop)
    first_path = first.result_path

    loop.on_event(run_cmd("inv-2", seed=42, init_state_ids=[1], max_steps=16))
    second = loop.session
    assert second is not first
    assert second.cfg is not first_cfg
    assert second.cfg.seed == 42 and second.cfg.init_state_ids == [1]
    assert adapter.cfg is second.cfg, "adapter 必须跟到第二次的 cfg"
    drive_to_completion(loop)

    assert loop.busy is False
    assert second.result_path != first_path
    results = [r for r in reports(node) if r["type"] == S.REPORT_RESULT]
    assert [r["invocation_id"] for r in results] == ["inv-1", "inv-2"]
    assert all(r["payload"]["status"] == "succeeded" for r in results)
    assert adapter.closed == 0, "invocation 级 finish 不该关 adapter"


# ================================================ 6. 并发拒绝


def test_second_run_while_busy_is_rejected(cfg_template, node, adapter) -> None:
    loop = N.ServeLoop(node, cfg_template, adapter)
    loop.on_event(run_cmd("inv-1"))
    active = loop.session
    resets_before = len(adapter.resets)

    loop.on_event(run_cmd("inv-2"))

    rejected = [r for r in reports(node) if r["type"] == S.REPORT_REJECTED]
    assert len(rejected) == 1
    assert rejected[0]["invocation_id"] == "inv-2"
    assert rejected[0]["code"] == "FORGE_BUSY"
    assert "inv-1" in rejected[0]["message"]
    # 当前 invocation 一点没被动
    assert loop.session is active
    assert loop.session.invocation_id == "inv-1"
    assert len(adapter.resets) == resets_before, "被拒的那次不该 reset 任何 episode"


def test_rejected_run_does_not_queue(cfg_template, node, adapter) -> None:
    """不排队:第一次跑完之后,被拒的那次也不该自己冒出来。"""
    loop = N.ServeLoop(node, cfg_template, adapter)
    loop.on_event(run_cmd("inv-1"))
    loop.on_event(run_cmd("inv-2"))
    drive_to_completion(loop)
    assert loop.busy is False and loop.session is None


def test_invalid_plan_is_rejected_without_touching_state(cfg_template, node, adapter) -> None:
    loop = N.ServeLoop(node, cfg_template, adapter)
    loop.on_event(run_cmd("inv-bad", task_ids=[9]))
    rejected = [r for r in reports(node) if r["type"] == S.REPORT_REJECTED]
    assert len(rejected) == 1
    assert rejected[0]["code"] == "BENCH_INVALID_ARGUMENT"
    assert loop.busy is False and adapter.resets == []


def test_malformed_command_does_not_escape(cfg_template, node, adapter) -> None:
    import pyarrow as pa

    loop = N.ServeLoop(node, cfg_template, adapter)
    assert loop.on_event({"type": "INPUT", "id": N.CMD_INPUT_ID, "value": pa.array(["garbage"])}) is None
    assert loop.busy is False


# ================================================ 7. cancel


def test_cancel_terminates_the_current_invocation(cfg_template, node, adapter) -> None:
    loop = N.ServeLoop(node, cfg_template, adapter)
    loop.on_event(run_cmd("inv-1", init_state_ids=[0, 1, 2]))
    assert loop.busy is True

    loop.on_event(cancel_cmd("inv-1"))

    assert loop.busy is False and loop.session is None
    results = [r for r in reports(node) if r["type"] == S.REPORT_RESULT]
    assert len(results) == 1
    p = results[0]["payload"]
    assert p["status"] == "cancelled"
    assert p["completed_episodes"] < p["total_episodes"]
    assert adapter.closed == 0, "cancel 是 invocation 级,不该关 adapter"


def test_cancel_for_another_invocation_is_ignored(cfg_template, node, adapter) -> None:
    """乱取消比不取消更糟:id 对不上就不许动当前这次。"""
    loop = N.ServeLoop(node, cfg_template, adapter)
    loop.on_event(run_cmd("inv-1"))
    active = loop.session
    loop.on_event(cancel_cmd("inv-999"))
    assert loop.session is active and loop.busy is True
    assert not [r for r in reports(node) if r["type"] == S.REPORT_RESULT]


def test_cancel_when_idle_is_a_noop(cfg_template, node, adapter) -> None:
    loop = N.ServeLoop(node, cfg_template, adapter)
    loop.on_event(cancel_cmd("inv-1"))
    assert loop.busy is False


def test_run_after_cancel_works(cfg_template, node, adapter) -> None:
    loop = N.ServeLoop(node, cfg_template, adapter)
    loop.on_event(run_cmd("inv-1", init_state_ids=[0, 1, 2]))
    loop.on_event(cancel_cmd("inv-1"))
    loop.on_event(run_cmd("inv-2"))
    assert loop.busy is True and loop.session.invocation_id == "inv-2"
    drive_to_completion(loop)
    assert loop.busy is False


# ================================================ 8. adapter 生命周期


class ScriptedNode(FakeNode):
    """按脚本吐事件的假 dora Node;脚本放完后可选地抛异常。"""

    def __init__(self, events, raise_after: Exception | None = None) -> None:
        super().__init__()
        self._events = list(events)
        self._raise_after = raise_after

    def next(self, timeout=None):  # noqa: ARG002 与 dora.Node 签名对齐
        if self._events:
            return self._events.pop(0)
        if self._raise_after is not None:
            raise self._raise_after
        return {"type": "STOP"}


def test_stop_closes_adapter_exactly_once(cfg_template, adapter) -> None:
    node = ScriptedNode([run_cmd("inv-1"), *[action_event()] * 4, {"type": "STOP"}])
    rc = N.serve_forever(node, cfg_template, adapter)
    assert rc == 0
    assert adapter.closed == 1, f"STOP 后 adapter 应恰好关一次,实际 {adapter.closed}"


def test_exception_closes_adapter_exactly_once(cfg_template, adapter) -> None:
    node = ScriptedNode([run_cmd("inv-1")], raise_after=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        N.serve_forever(node, cfg_template, adapter)
    assert adapter.closed == 1, f"异常退出后 adapter 应恰好关一次,实际 {adapter.closed}"


def test_dora_error_closes_adapter_exactly_once(cfg_template, adapter) -> None:
    node = ScriptedNode([run_cmd("inv-1"), {"type": "ERROR", "error": "node crashed"}])
    rc = N.serve_forever(node, cfg_template, adapter)
    assert rc == 1
    assert adapter.closed == 1


def test_stop_mid_invocation_reports_a_terminal_result(cfg_template, adapter) -> None:
    """STOP 打断一次 invocation,调用方也必须拿到终态,不能悬着。"""
    node = ScriptedNode([run_cmd("inv-1", init_state_ids=[0, 1, 2]), {"type": "STOP"}])
    N.serve_forever(node, cfg_template, adapter)
    results = [r for r in reports(node) if r["type"] == S.REPORT_RESULT]
    assert len(results) == 1
    assert results[0]["payload"]["status"] == "cancelled"
    assert adapter.closed == 1


def test_full_run_over_serve_forever_closes_once(cfg_template, adapter) -> None:
    """完整跑完一次再 STOP:invocation 级 finish 没关,退出时关且只关一次。"""
    node = ScriptedNode([run_cmd("inv-1"), *[action_event()] * 6])
    N.serve_forever(node, cfg_template, adapter)
    results = [r for r in reports(node) if r["type"] == S.REPORT_RESULT]
    assert len(results) == 1 and results[0]["payload"]["status"] == "succeeded"
    assert adapter.closed == 1


# ================================================ 心跳


def test_heartbeat_emits_capabilities_and_beat(cfg_template, node, adapter) -> None:
    loop = N.ServeLoop(node, cfg_template, adapter)
    assert loop.heartbeat(1000.0) is True
    ks = kinds(node)
    assert S.REPORT_CAPABILITIES in ks and S.REPORT_HEARTBEAT in ks
    # 同一周期内不重复发
    assert loop.heartbeat(1000.1) is False
    assert loop.heartbeat(1000.0 + S.HEARTBEAT_PERIOD_S) is True


def test_heartbeat_reports_busy_flag(cfg_template, node, adapter) -> None:
    loop = N.ServeLoop(node, cfg_template, adapter)
    loop.heartbeat(1000.0)
    beat = [r for r in reports(node) if r["type"] == S.REPORT_HEARTBEAT][-1]
    assert beat["busy"] is False and beat["invocation_id"] is None

    loop.on_event(run_cmd("inv-1"))
    loop.heartbeat(2000.0)
    beat = [r for r in reports(node) if r["type"] == S.REPORT_HEARTBEAT][-1]
    assert beat["busy"] is True and beat["invocation_id"] == "inv-1"


def test_describe_command_reports_capabilities(cfg_template, node, adapter) -> None:
    loop = N.ServeLoop(node, cfg_template, adapter)
    loop.on_event(cmd({"type": S.CMD_DESCRIBE}))
    caps = [r for r in reports(node) if r["type"] == S.REPORT_CAPABILITIES]
    assert len(caps) == 1
    assert caps[0]["num_tasks"] == 2
    assert {t["task_id"]: t["num_init_states"] for t in caps[0]["tasks"]} == {0: 3, 1: 2}
