"""bench_endpoint ↔ benchmark node 之间的 JSON 线格式。

刻意用最朴素的 `pa.array([json.dumps(...)])`:仓库里 `benchmark_status` 和
`benchmark_result` 已经是这个形态(`_core/runner.py:117` 和 `:347`),不引入新概念。

**每条消息都带 `v`**:两个节点跑在不同 Python(3.11 / 3.12)、是两个独立包、
会分别升级。没有版本号的话,第一次不兼容变更就是静默错乱——收到不认识的字段
默默忽略,然后分数不对但谁都不报错。
"""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa

WIRE_VERSION = 1

# endpoint → benchmark(通道 bench_cmd)
CMD_RUN = "run"
CMD_CANCEL = "cancel"
CMD_DESCRIBE = "describe"

# benchmark → endpoint(通道 bench_report)
REPORT_CAPABILITIES = "capabilities"
REPORT_ACCEPTED = "accepted"
REPORT_REJECTED = "rejected"
REPORT_PROGRESS = "progress"
REPORT_HEARTBEAT = "heartbeat"
REPORT_RESULT = "result"


class WireError(ValueError):
    """线上消息不合法。调用方应当丢弃该消息并记一条日志,不要让它冲出事件循环。"""


def encode(payload: dict[str, Any]) -> pa.Array:
    """把一条消息编成 dora 通道上的载体。"""
    if not isinstance(payload, dict):
        raise WireError("payload must be a dict")
    body = dict(payload)
    body.setdefault("v", WIRE_VERSION)
    return pa.array([json.dumps(body, ensure_ascii=False)])


def decode(value: Any) -> dict[str, Any]:
    """把 dora 事件的 value 解回 dict。

    容忍三种形态:pyarrow Array / list / 裸字符串。dora 在不同路径下交付的
    东西不完全一致,而这里判错的代价是整条通道静默失效,所以宁可多认几种。
    """
    raw: Any = value
    if isinstance(raw, (pa.Array, pa.ChunkedArray)):
        items = raw.to_pylist()
    elif isinstance(raw, pa.RecordBatch | pa.Table):
        raise WireError("bench 通道只收字符串数组,不收 RecordBatch")
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    if len(items) != 1:
        raise WireError(f"bench 通道每条消息只应有 1 个元素,收到 {len(items)}")
    text = items[0]
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8")
    if not isinstance(text, str):
        raise WireError(f"bench 通道载体必须是字符串,收到 {type(text).__name__}")
    try:
        body = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WireError(f"bench 通道载体不是合法 JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise WireError("bench 通道载体必须是 JSON 对象")
    version = body.get("v")
    if version != WIRE_VERSION:
        raise WireError(f"线协议版本不匹配:本端 {WIRE_VERSION},收到 {version!r}")
    if not isinstance(body.get("type"), str):
        raise WireError("消息缺少 type 字段")
    return body


# ---- endpoint 侧构造的命令 ----


def run_command(invocation_id: str, plan: dict[str, Any]) -> dict[str, Any]:
    return {"v": WIRE_VERSION, "type": CMD_RUN, "invocation_id": invocation_id, "plan": plan}


def cancel_command(invocation_id: str, reason: str | None = None) -> dict[str, Any]:
    return {
        "v": WIRE_VERSION,
        "type": CMD_CANCEL,
        "invocation_id": invocation_id,
        "reason": reason,
    }


def describe_command() -> dict[str, Any]:
    return {"v": WIRE_VERSION, "type": CMD_DESCRIBE}


__all__ = [
    "CMD_CANCEL",
    "CMD_DESCRIBE",
    "CMD_RUN",
    "REPORT_ACCEPTED",
    "REPORT_CAPABILITIES",
    "REPORT_HEARTBEAT",
    "REPORT_PROGRESS",
    "REPORT_REJECTED",
    "REPORT_RESULT",
    "WIRE_VERSION",
    "WireError",
    "cancel_command",
    "decode",
    "describe_command",
    "encode",
    "run_command",
]
