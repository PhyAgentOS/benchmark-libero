"""serve 模式的 dora 事件循环：工具面（forge_tool）+ 数据面（批次推进）同进程。

结构照抄 kai0_runner/inference/session_runner.py（其上溯是
relative_pose_policy_node/runner.py）：
- ToolEndpointHandler + DoraToolEndpointBinding：tool_in → 分发 → tool_out；
- EndpointLease：空闲时靠 node.next(timeout=) 的 tick 驱动续租；
- 数据面：pending start 拾取（阻塞 reset 在这里，dispatch 之外）、action 驱动
  episode 锁步、终态收进 invocation 表 + 发终态事件。

关键纪律（继承自双节点架构的死锁分析）：endpoint 方法在 dispatch 内同步跑到底，
只登记不执行；任何 dora 发送都在 dispatch 之外。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Callable, Iterable

from forge_msgs import JointCommand, ToolMessage
from forge_tool import ToolEndpointHandler, ToolEnvelope, ToolEvent
from forge_tool.dora import (
    DoraToolEndpointBinding,
    tool_envelope_to_message,
    tool_message_to_envelope,
)

from _core.config import BaseBenchmarkConfig
from _core.session import bench_limits, capabilities_payload

from .descriptor import BENCH_DESCRIPTOR, OP_DESCRIBE, OP_RUN
from .endpoints import BenchDescribeEndpoint, BenchSessionEndpoint, SessionHandle
from .invocations import InvocationTable
from .lease import EndpointLease

TOOL_INPUT_ID = "tool_in"
TOOL_OUTPUT_ID = "tool_out"
ACTION_INPUT_ID = "action"

# node.next(timeout=) 的周期。空闲时没有任何输入，续租与 pending 拾取全靠它。
# lease_ttl_ms 由 gateway 的 skill profile 决定（skill 树里为本节点调到 120000：
# episode reset 会阻塞事件循环数十秒）；EndpointLease 在 ttl/3 处重注册，
# 所以 tick 周期必须 ≤ ttl/3，取 1s。
TICK_PERIOD_S = 1.0


class ServeToolLoop:
    """serve 模式状态机。**不碰 dora**，事件由外面喂进来，所以可以单测。

    只有两个可变状态：endpoint 的 pending 队列与 `self._active`。
    None = idle，非 None = 正在跑一次 session。一台机器一个仿真器，
    所以同时最多一个。
    """

    def __init__(
        self,
        node: Any,
        cfg: BaseBenchmarkConfig,
        adapter: Any,
        log: Callable[[str], None] = print,
        *,
        endpoint_instance_id: str | None = None,
    ) -> None:
        self.node = node
        self.cfg = cfg
        self.adapter = adapter
        self._log = log
        self.endpoint_instance_id = endpoint_instance_id or str(uuid.uuid4())
        self.caps = capabilities_payload(cfg, adapter, limits=bench_limits(cfg))
        self.table = InvocationTable()
        self._async = asyncio.Runner()
        self.endpoint = BenchSessionEndpoint(
            self.table,
            self.node,
            self.cfg,
            self.adapter,
            self.caps,
            emit=self._emit_event,
            log=log,
        )
        self.handler = ToolEndpointHandler(
            BENCH_DESCRIPTOR,
            endpoint_instance_id=self.endpoint_instance_id,
            operations={
                OP_DESCRIBE: BenchDescribeEndpoint(self.caps),
                OP_RUN: self.endpoint,
            },
        )
        self.binding = DoraToolEndpointBinding(self.handler, event_sink=self._send_batch)
        self.lease = EndpointLease(BENCH_DESCRIPTOR, self.endpoint_instance_id)
        self._active: SessionHandle | None = None
        self._closed = False

    # ---- 出站 ----

    async def _send_batch(self, batch: Any) -> None:
        self.node.send_output(TOOL_OUTPUT_ID, batch)

    def _send_envelope(self, envelope: ToolEnvelope) -> None:
        self.node.send_output(TOOL_OUTPUT_ID, tool_envelope_to_message(envelope).to_arrow())

    def _drive_lease(self) -> None:
        request = self.lease.poll(time.monotonic())
        if request is not None:
            self._send_envelope(request)

    def _emit_event(self, handle: SessionHandle, event: ToolEvent) -> None:
        """把事件交给 handler 发的 emitter，跑在 dispatch 之外。

        终态事件的 emit 会回调 handler 校验"result 已可用、事件不越序"，
        所以 on_finished 里必须**先落表再 emit**。
        """
        emitter = handle.emitter
        if emitter is None:
            return
        try:
            self._async.run(emitter.emit(event))
        except Exception:  # noqa: BLE001 事件发不出去不该打死批次
            self._log(f"tool 事件发送失败（{event.type}），忽略")

    # ---- 入站 ----

    def _handle_tool(self, value: Any) -> None:
        message = ToolMessage.from_arrow(value)
        envelope = tool_message_to_envelope(message)
        if envelope.message_type == "endpoint.registry.response":
            # 漏了这一步的症状：gateway 起得来、GET /tools 也有，但 ready 永远
            # false，而且不报任何错。
            self.lease.acknowledge(envelope, time.monotonic())
            return
        for batch in self._async.run(self.binding.dispatch_input(value)):
            self.node.send_output(TOOL_OUTPUT_ID, batch)

    # ---- 数据面 ----

    def _pick_up_pending(self) -> None:
        """把 start() 登记的新 session 拾起来开跑。

        session.start() 里的 reset 会重建仿真场景，事件循环会整个停住几十秒。
        必须在 tool 分发之外执行（ToolAccepted 已经发出去了，这里阻塞不影响
        握手），且每轮先于事件分发 —— 上一轮已经把租续好了，阻塞期间不需要再续。
        """
        for handle in self.endpoint.take_pending():
            self.table.mark_running(handle.key.invocation_id, {"phase_detail": "running"})
            self._active = handle
            self._log(
                f"session {handle.key.invocation_id} 开跑：tasks={handle.session.cfg.task_ids} "
                f"init_states={handle.session.cfg.init_state_ids} num_runs={handle.session.cfg.num_runs} "
                f"→ {len(handle.session.runner.plan)} episodes"
            )
            handle.session.start()
            if handle.session.finished:
                self._on_finished(handle)

    def _on_action(self, value: Any) -> None:
        handle = self._active
        if handle is None:
            return  # 上一批的迟到动作，丢掉
        try:
            handle.session.on_action(JointCommand.from_arrow(value))
        except Exception as exc:  # noqa: BLE001 单步失败不该打死整批
            handle.session.fail("step_error", f"{type(exc).__name__}: {exc}")
        self._on_progress(handle)
        if handle.session.finished:
            self._on_finished(handle)

    def _on_progress(self, handle: SessionHandle) -> None:
        if handle.session.finished:
            return
        progress = handle.session.progress_report()
        if progress is None:
            return
        self.table.update_details(handle.key.invocation_id, dict(progress))
        self._emit_event(handle, ToolEvent(type="progress", data=dict(progress)))

    def _on_finished(self, handle: SessionHandle) -> None:
        if handle.reported:
            return
        if self._active is handle:
            self._active = None
        self.endpoint.on_finished(handle)
        self._log(f"session {handle.key.invocation_id} 结束，回到 idle")

    def _terminate_active(self, status: str, reason: str) -> None:
        """dora STOP/ERROR 时把在跑的 session 立即收成终态（不等 episode 边界）。"""
        handle = self._active
        if handle is None or handle.session.finished:
            return
        handle.stop_reason = reason
        handle.session.runner.finish(status, failure_summary=reason)
        self._on_finished(handle)

    # ---- 事件循环 ----

    def process_event(self, event: Any) -> int | None:
        """处理一个 dora 事件。返回非 None 表示外层循环该退出，值就是退出码。"""
        if event is None or _is_timeout(event):
            self._drive_lease()
            self._pick_up_pending()
            return None

        etype = event.get("type") if isinstance(event, dict) else None
        self._drive_lease()
        if etype == "STOP":
            self._terminate_active("stopped", "节点被 dora 停止")
            unregister = self.lease.stop()
            if unregister is not None:
                self._send_envelope(unregister)
            self._async.close()
            self._closed = True
            return 0
        if etype == "ERROR":
            self._log(f"dora error: {event.get('error')}")
            self._terminate_active("failed", f"dora error: {event.get('error')}")
            self._async.close()
            self._closed = True
            return 1
        if etype != "INPUT":
            return None

        self._pick_up_pending()
        eid = event.get("id")
        try:
            if eid == TOOL_INPUT_ID:
                self._handle_tool(event.get("value"))
            elif eid == ACTION_INPUT_ID:
                self._on_action(event.get("value"))
            # 其它输入（tick 之类）只当续租心跳，不做任何事 ——
            # **绝不在 tick 上推进 episode**
        except (TypeError, ValueError) as exc:
            self._log(f"忽略非法输入：{exc}")
        return None

    def shutdown(self) -> None:
        """进程退出前调一次（幂等）。**不关 adapter** —— 那是 run_serve_tool 的事。"""
        self._terminate_active("stopped", "节点异常退出")
        if not self._closed:
            self._async.close()
            self._closed = True

    def run(self, events: Iterable[dict[str, Any]]) -> int:
        self._drive_lease()
        for event in events:
            outcome = self.process_event(event)
            if outcome is not None:
                return outcome
        return 0


def _is_timeout(event: Any) -> bool:
    """dora 把 `next(timeout=)` 的超时表达成 ERROR 事件，不是返回 None。

    照抄文档里的 `if event is None` 会让超时走进致命错误分支，把整批判失败。
    两种形态都要认。
    """
    if not isinstance(event, dict) or event.get("type") != "ERROR":
        return False
    return "timed out" in str(event.get("error", "")).lower()


def run_serve_tool(
    cfg: BaseBenchmarkConfig,
    adapter: Any,
    log: Callable[[str], None] = print,
    *,
    node: Any = None,
) -> int:
    """serve 模式外层循环。node 从外面传进来，测试可以喂假的。"""
    from dora import Node  # 延迟导入：loop 层（测试）不依赖 dora 运行时

    dora_node = node if node is not None else Node()
    loop = ServeToolLoop(dora_node, cfg, adapter, log=log)
    log(
        f"serve 模式：benchmark={cfg.benchmark} suite={cfg.suite} "
        f"{loop.caps['num_tasks']} 个 task，endpoint {BENCH_DESCRIPTOR.endpoint_id} 等待 tool_in"
    )
    rc = 0
    try:
        while True:
            # **必须带 timeout**：tool_in 是低频输入，空闲时若无限阻塞，
            # 续租与 pending 拾取都发不出去。
            event = dora_node.next(timeout=TICK_PERIOD_S)
            outcome = loop.process_event(event)
            if outcome is not None:
                rc = outcome
                break
    finally:
        loop.shutdown()
        # **唯一关 adapter 的地方。** session 级的 finish() 不关（runner 建的时候
        # 传了 close_adapter_on_finish=False），否则下一次 run 要重载整个场景。
        adapter.close()
    return rc


__all__ = [
    "ACTION_INPUT_ID",
    "TICK_PERIOD_S",
    "TOOL_INPUT_ID",
    "TOOL_OUTPUT_ID",
    "ServeToolLoop",
    "run_serve_tool",
]
