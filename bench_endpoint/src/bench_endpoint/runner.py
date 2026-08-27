"""dora 事件循环。结构照抄 relative_pose_policy_node/runner.py。

和参考实现的**唯一实质差别**:那边每收到一个 `joint_state`(高频)就顺手续租,
本节点空闲时一个输入都没有,所以续租只能靠 `tick` 驱动。lease_ttl_ms 默认 3000,
EndpointLease 会在 ttl/3 后重新注册 → tick 周期必须 ≤1000ms。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from forge_tool import ToolEndpointHandler, ToolEnvelope
from forge_tool._tool_message import ToolMessage
from forge_tool.dora import (
    DoraToolEndpointBinding,
    tool_envelope_to_message,
    tool_message_to_envelope,
)

from .config import BenchEndpointConfig
from .descriptor import BENCH_DESCRIPTOR, OP_DESCRIBE, OP_RUN
from .endpoints import BenchDescribeEndpoint, BenchRunEndpoint
from .endpoint_lease import EndpointLease
from .invocations import InvocationTable
from .peer import BenchPeer
from . import wire

logger = logging.getLogger(__name__)

CMD_OUTPUT_ID = "bench_cmd"
TOOL_OUTPUT_ID = "tool_out"


class BenchEndpointRunner:
    def __init__(
        self,
        node: Any,
        config: BenchEndpointConfig | None = None,
        *,
        peer: BenchPeer | None = None,
        endpoint_instance_id: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.node = node
        self.config = config or BenchEndpointConfig()
        self.clock = clock
        self.peer = peer or BenchPeer(
            table=InvocationTable(self.config.retention),
            heartbeat_timeout_s=self.config.heartbeat_timeout_s,
        )
        self.endpoint_instance_id = endpoint_instance_id or str(uuid.uuid4())
        self._async = asyncio.Runner()
        self.handler = ToolEndpointHandler(
            BENCH_DESCRIPTOR,
            endpoint_instance_id=self.endpoint_instance_id,
            operations={
                OP_DESCRIBE: BenchDescribeEndpoint(self.peer, self.clock),
                OP_RUN: BenchRunEndpoint(self.peer, self.clock),
            },
        )
        self.binding = DoraToolEndpointBinding(self.handler, event_sink=self._send_batch)
        self.lease = EndpointLease(BENCH_DESCRIPTOR, self.endpoint_instance_id)

    # ---- 出站 ----

    async def _send_batch(self, batch: object) -> None:
        self.node.send_output(TOOL_OUTPUT_ID, batch)

    def _send_envelope(self, envelope: ToolEnvelope) -> None:
        self.node.send_output(TOOL_OUTPUT_ID, tool_envelope_to_message(envelope).to_arrow())

    def _drive_lease(self) -> None:
        request = self.lease.poll(self.clock())
        if request is not None:
            self._send_envelope(request)

    def _flush_commands(self) -> None:
        """把 endpoint 方法塞进 outbox 的命令发出去。

        必须在 tool 分发**之后**调:`start()` 是在分发过程中往 outbox 里塞的。
        """
        for command in self.peer.drain():
            self.node.send_output(CMD_OUTPUT_ID, wire.encode(command))

    # ---- 入站 ----

    def _handle_tool(self, value: object) -> None:
        envelope = tool_message_to_envelope(ToolMessage.from_arrow(value))
        if envelope.message_type == "endpoint.registry.response":
            # 漏了这一步的症状:gateway 起得来、GET /tools 也有,但 ready 永远
            # false,而且不报任何错。
            self.lease.acknowledge(envelope, self.clock())
            return
        batches = self._async.run(self.binding.dispatch_input(value))
        for batch in batches:
            self.node.send_output(TOOL_OUTPUT_ID, batch)

    def _handle_report(self, value: object) -> None:
        try:
            message = wire.decode(value)
        except wire.WireError as exc:
            logger.warning("丢弃非法 bench_report:%s", exc)
            return
        self.peer.on_report(message, self.clock())

    def process_event(self, event: dict[str, Any]) -> int | None:
        event_type = event.get("type")
        if event_type == "STOP":
            unregister = self.lease.stop()
            if unregister is not None:
                self._send_envelope(unregister)
            self._async.close()
            return 0
        if event_type == "ERROR":
            logger.error("Dora node error: %s", event.get("error", "unknown"))
            self._async.close()
            return 1
        if event_type != "INPUT":
            return None

        self._drive_lease()
        try:
            event_id = event.get("id")
            if event_id == "tool_in":
                self._handle_tool(event["value"])
            elif event_id == "bench_report":
                self._handle_report(event["value"])
            elif event_id == "tick":
                self.peer.tick(self.clock())
        except (TypeError, ValueError) as exc:
            logger.warning("忽略非法输入:%s", exc)
        self._drive_lease()
        self._flush_commands()
        return None

    def run(self, events: Iterable[dict[str, Any]]) -> int:
        self._drive_lease()
        self._flush_commands()
        for event in events:
            result = self.process_event(event)
            if result is not None:
                return result
        return 0


def run_endpoint(config: BenchEndpointConfig) -> int:
    from dora import Node

    logging.basicConfig(level=config.log_level, format="[bench_endpoint] %(message)s")
    node = Node()
    return BenchEndpointRunner(node, config).run(node)


__all__ = ["CMD_OUTPUT_ID", "TOOL_OUTPUT_ID", "BenchEndpointRunner", "run_endpoint"]
