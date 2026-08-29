"""descriptor 形状：session 语义的硬约束都在 forge_tool 的校验里，
写死在这里防止手滑改回去（改回 action 语义会让 skill 树整套契约静默错乱）。"""

from __future__ import annotations

from bench_tool.descriptor import BENCH_DESCRIPTOR, BENCH_ENDPOINT_ID, OP_DESCRIBE, OP_RUN


def test_descriptor_shape() -> None:
    assert BENCH_DESCRIPTOR.endpoint_id == BENCH_ENDPOINT_ID
    ops = {op.name: op for op in BENCH_DESCRIPTOR.operations}
    assert set(ops) == {OP_DESCRIBE, OP_RUN}

    assert ops[OP_DESCRIBE].semantics == "query"

    run = ops[OP_RUN]
    assert run.semantics == "session"
    # session 的硬约束：stoppable=True；cancellable 必须 False、status_supported
    # 必须 True（forge_tool 的 descriptor 校验强制，写错构造时就抛）
    assert run.stoppable is True
    assert run.cancellable is False
    assert run.status_supported is True
    # 一台机器一个仿真器
    assert run.max_concurrency == 1
