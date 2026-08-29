"""Dora 节点入口（autorun）。serve 模式已迁到 bench_tool/node.py，main.py 分叉。

- **autorun（默认）**：起来就把配置里的批次跑完然后退出。函数体与改造前逐字相同。
- **serve**：见 bench_tool/node.py —— serve 需要 forge_tool，而 _core 保持不依赖它。
"""

from __future__ import annotations

from typing import Any, Callable

from dora import Node
from forge_msgs import JointCommand

from .config import BaseBenchmarkConfig, resolve_and_validate_plan
from .runner import BenchmarkRunner


def run_node(
    cfg: BaseBenchmarkConfig,
    adapter: Any,
    log: Callable[[str], None] = print,
) -> int:
    """autorun 入口：起来就把配置里的批次跑完然后退出。"""
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
