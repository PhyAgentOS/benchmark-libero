"""Dora 节点入口。各 benchmark 的 main.py 只需调用 run_node()。

两种模式,由 `cfg.serve` 分叉:

- **autorun(默认)**:起来就把配置里的批次跑完然后退出。三家 benchmark 现有的
  config.yaml 都不含 `serve` 键 → `serve=False` → 走 `_run_autorun()`,
  **那个函数体与改造前逐字相同**。
- **serve**:等 `bench_cmd`,收到一条跑一批,跑完回 idle 继续等。给 Forge Tool 用。

两条路共用同一个 BenchmarkRunner,判分逻辑完全一致。
"""

from __future__ import annotations

import time
from typing import Any, Callable

from dora import Node
from forge_msgs import JointCommand

from .config import BaseBenchmarkConfig, resolve_and_validate_plan
from .runner import BenchmarkRunner
from .session import (
    CMD_CANCEL,
    CMD_DESCRIBE,
    CMD_RUN,
    HEARTBEAT_PERIOD_S,
    REPORT_ACCEPTED,
    REPORT_HEARTBEAT,
    REPORT_REJECTED,
    BenchSession,
    WireError,
    capabilities_payload,
    decode,
    encode,
)

CMD_INPUT_ID = "bench_cmd"
REPORT_OUTPUT_ID = "bench_report"


def run_node(
    cfg: BaseBenchmarkConfig,
    adapter: Any,
    log: Callable[[str], None] = print,
) -> int:
    """按 cfg.serve 分叉。默认(False)走 autorun,即改造前的行为。"""
    if getattr(cfg, "serve", False):
        return _run_serve(cfg, adapter, log)
    return _run_autorun(cfg, adapter, log)


# ======================================================================
# autorun —— 函数体自 baseline 0dd7c90 逐字未变，只是换了函数名
# ======================================================================


def _run_autorun(
    cfg: BaseBenchmarkConfig,
    adapter: Any,
    log: Callable[[str], None] = print,
) -> int:
    """事件循环。无论怎么退出，结果都必须落盘。"""
    # fail fast：越界的 task/init_state 必须在建 Node 之前炸掉
    resolve_and_validate_plan(cfg, adapter)

    node = Node()
    runner = BenchmarkRunner(node, cfg, adapter, log=log)
    log(
        f"benchmark={cfg.benchmark} suite={cfg.suite} tasks={cfg.task_ids} "
        f"init_states={len(cfg.init_state_ids)} num_runs={cfg.num_runs} "
        f"seed={cfg.seed} → {len(runner.plan)} episodes"
    )

    rc = 0
    try:
        runner.policy_command("start")
        runner.advance()

        for event in node:
            if event["type"] == "INPUT":
                if event["id"] == "action":
                    try:
                        runner.on_action(JointCommand.from_arrow(event["value"]))
                    except Exception as exc:  # noqa: BLE001 单步失败不该打死整批
                        runner.fail_current("step_error", f"{type(exc).__name__}: {exc}")
            elif event["type"] == "STOP":
                runner.finish("cancelled")
                break
            elif event["type"] == "ERROR":
                log(f"dora error: {event.get('error')}")
                runner.finish("failed")
                rc = 1
                break

            if runner.finished:
                break
    finally:
        runner.finish("cancelled")

    return rc


# ======================================================================
# serve —— 等 bench_cmd，一次跑一批，跑完回 idle
# ======================================================================


def _limits(cfg: BaseBenchmarkConfig) -> dict[str, int]:
    return {
        "max_total_episodes": int(getattr(cfg, "max_total_episodes", 0) or 0),
        "max_max_steps": int(getattr(cfg, "max_max_steps", 0) or 0),
    }


def _is_timeout(event: Any) -> bool:
    """dora 把 `next(timeout=)` 的超时表达成 ERROR 事件，不是返回 None。

    照抄文档里的 `if event is None` 会让超时走进致命错误分支，把整批判失败。
    两种形态都要认。
    """
    if not isinstance(event, dict) or event.get("type") != "ERROR":
        return False
    return "timed out" in str(event.get("error", "")).lower()


class ServeLoop:
    """serve 模式的状态机。**不碰 dora**,事件由外面喂进来,所以可以单测。

    只有一个可变状态:`self.session`。None = idle,非 None = 正在跑一次
    invocation。一台机器一个仿真器,所以同时最多一个。
    """

    def __init__(
        self,
        node: Any,
        cfg: BaseBenchmarkConfig,
        adapter: Any,
        log: Callable[[str], None] = print,
    ) -> None:
        self.node = node
        self.cfg = cfg
        self.adapter = adapter
        self._log = log
        self.session: BenchSession | None = None
        self.caps = capabilities_payload(cfg, adapter, limits=_limits(cfg))
        self._last_heartbeat = 0.0

    @property
    def busy(self) -> bool:
        return self.session is not None

    # ---- 出站 ----

    def report(self, payload: dict[str, Any]) -> None:
        self.node.send_output(REPORT_OUTPUT_ID, encode(payload))

    def heartbeat(self, now: float) -> bool:
        """每个周期发一次 capabilities + heartbeat。发了返回 True。

        capabilities **重复发**是有意的(内容不变,幂等):endpoint 可能比本节点
        晚起、或中途重启;只推一次的话它永远等不到,tool 永远 not ready。
        """
        if now - self._last_heartbeat < HEARTBEAT_PERIOD_S:
            return False
        self._last_heartbeat = now
        self.report(self.caps)
        self.report(
            {
                "type": REPORT_HEARTBEAT,
                "invocation_id": self.session.invocation_id if self.session else None,
                "busy": self.busy,
            }
        )
        return True

    # ---- 入站 ----

    def on_event(self, event: Any) -> int | None:
        """处理一个 dora 事件。返回非 None 表示外层循环该退出,值就是退出码。"""
        if event is None or _is_timeout(event):
            return None

        etype = event.get("type") if isinstance(event, dict) else None
        if etype == "STOP":
            self._terminate("cancelled")
            return 0
        if etype == "ERROR":
            self._log(f"dora error: {event.get('error')}")
            return 1
        if etype != "INPUT":
            return None

        eid = event.get("id")
        if eid == CMD_INPUT_ID:
            self._on_command(event["value"])
        elif eid == "action":
            self._on_action(event["value"])
        # 其它输入(tick 之类)只当心跳,不做任何事 —— **绝不在 tick 上推进 episode**
        return None

    def _on_action(self, value: Any) -> None:
        session = self.session
        if session is None:
            return  # 上一批的迟到动作,丢掉
        try:
            session.on_action(JointCommand.from_arrow(value))
        except Exception as exc:  # noqa: BLE001 单步失败不该打死整批
            session.fail("step_error", f"{type(exc).__name__}: {exc}")

        progress = session.progress_report()
        if progress is not None:
            self.report(progress)
        if session.finished:
            self.report(session.result_report())
            self._log(f"invocation {session.invocation_id} 结束，回到 idle")
            self.session = None

    def _on_command(self, value: Any) -> None:
        try:
            message = decode(value)
        except WireError as exc:
            self._log(f"丢弃非法 bench_cmd：{exc}")
            return

        kind = message.get("type")
        if kind == CMD_DESCRIBE:
            self.report(capabilities_payload(self.cfg, self.adapter, limits=_limits(self.cfg)))
        elif kind == CMD_CANCEL:
            self._on_cancel(message)
        elif kind == CMD_RUN:
            self._on_run(message)
        else:
            self._log(f"未知 bench_cmd 类型：{kind!r}")

    def _on_cancel(self, message: dict[str, Any]) -> None:
        session = self.session
        if session is None:
            return
        if message.get("invocation_id") != session.invocation_id:
            # 取消的不是当前这次 —— 不动它。乱取消比不取消更糟。
            self._log(f"忽略 cancel：{message.get('invocation_id')!r} 不是当前 invocation")
            return
        self._log(f"收到 cancel：{session.invocation_id}")
        session.cancel()
        self.report(session.result_report())
        self.session = None

    def _on_run(self, message: dict[str, Any]) -> None:
        invocation_id = str(message.get("invocation_id") or "")
        plan = message.get("plan")
        if not invocation_id or not isinstance(plan, dict):
            self._log("bench_cmd run 缺少 invocation_id 或 plan")
            return

        if self.busy:
            # endpoint 侧已经挡了并发,走到这里说明两边状态不同步。
            # **必须回 rejected 且不动当前 invocation**：静默忽略会让调用方
            # 永远等一个不会来的结果;排队或覆盖则会毁掉正在跑的那一批。
            current = self.session.invocation_id  # type: ignore[union-attr]
            self.report(
                {
                    "type": REPORT_REJECTED,
                    "invocation_id": invocation_id,
                    "code": "FORGE_BUSY",
                    "message": f"已有 invocation {current} 在跑",
                }
            )
            return

        try:
            session = BenchSession(self.node, self.cfg, self.adapter, plan, invocation_id, log=self._log)
        except Exception as exc:  # noqa: BLE001 计划非法不该打死节点
            self.report(
                {
                    "type": REPORT_REJECTED,
                    "invocation_id": invocation_id,
                    "code": "BENCH_INVALID_ARGUMENT",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            return

        self.report({"type": REPORT_ACCEPTED, "invocation_id": invocation_id})
        self._log(
            f"invocation {invocation_id} 开跑：tasks={session.cfg.task_ids} "
            f"init_states={session.cfg.init_state_ids} num_runs={session.cfg.num_runs} "
            f"→ {len(session.runner.plan)} episodes"
        )
        # 进长阻塞之前先补一拍心跳：start() 里的 reset 会重建仿真场景，
        # 事件循环会整个停住几十秒，期间发不出任何东西。
        self.report({"type": REPORT_HEARTBEAT, "invocation_id": invocation_id, "busy": True})
        self.session = session
        session.start()
        self.report({"type": REPORT_HEARTBEAT, "invocation_id": invocation_id, "busy": True})

    # ---- 收尾 ----

    def _terminate(self, status: str) -> None:
        session = self.session
        if session is None:
            return
        if not session.finished:
            session.cancel()
        self.report(session.result_report())
        self.session = None

    def shutdown(self) -> None:
        """进程退出前调一次。**不关 adapter** —— 那是 serve_forever 的事。"""
        self._terminate("cancelled")


def serve_forever(
    node: Any,
    cfg: BaseBenchmarkConfig,
    adapter: Any,
    log: Callable[[str], None] = print,
) -> int:
    """serve 模式的外层循环。node 从外面传进来,测试可以喂假的。"""
    loop = ServeLoop(node, cfg, adapter, log=log)
    log(
        f"serve 模式：benchmark={cfg.benchmark} suite={cfg.suite} "
        f"{loop.caps['num_tasks']} 个 task，等待 bench_cmd"
    )
    rc = 0
    try:
        while True:
            # **必须带 timeout**：bench_cmd 是低频输入，空闲时若无限阻塞，
            # 心跳发不出去，endpoint 会把本节点判成掉线。
            event = node.next(timeout=HEARTBEAT_PERIOD_S)
            loop.heartbeat(time.time())
            outcome = loop.on_event(event)
            if outcome is not None:
                rc = outcome
                break
    finally:
        loop.shutdown()
        # **唯一关 adapter 的地方。** invocation 级的 finish() 不关(runner 建的时候
        # 传了 close_adapter_on_finish=False)，否则下一次 bench.run 要重载整个场景。
        adapter.close()
    return rc


def _run_serve(
    cfg: BaseBenchmarkConfig,
    adapter: Any,
    log: Callable[[str], None] = print,
) -> int:
    return serve_forever(Node(), cfg, adapter, log=log)
