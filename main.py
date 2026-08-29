#!/usr/bin/env python3
"""LIBERO benchmark 节点（Dora）。

本节点进程内持有 LIBERO env 与 Forge session endpoint，与不改动的 policy 节点
经 dora 通信。批次调度、锁步、落盘全在 _core 里，本文件只负责组装。
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# 父目录**只在导入 _core 期间**挂上：节点目录与仿真器包同名（libero/robocasa），
# 若长期留在 sys.path，会把 editable 安装的真包遮成命名空间包，
# 表现为「环境未注册」而不是 ImportError——极难排查。
sys.path.insert(0, _PARENT)
try:
    from _core.node import run_node  # noqa: E402
    import _core.adapter, _core.config, _core.result, _core.runner, _core.video  # noqa: F401,E402
finally:
    if _PARENT in sys.path:
        sys.path.remove(_PARENT)
from adapters.libero_adapter import LiberoAdapter  # noqa: E402
from config import load_config  # noqa: E402


def _log(msg: str) -> None:
    print(f"[libero_bench] {msg}", flush=True)


def main() -> int:
    config_path = os.environ.get("BENCHMARK_CONFIG", os.path.join(_HERE, "config.yaml"))
    cfg = load_config(config_path)
    _log(f"layout_mode={cfg.layout_mode} camera={cfg.camera_height}x{cfg.camera_width}")
    adapter = LiberoAdapter(cfg)
    if getattr(cfg, "serve", False):
        # 延迟导入：autorun 不需要 forge_tool，serve 才要
        from bench_tool.node import run_serve_tool  # noqa: E402

        return run_serve_tool(cfg, adapter, log=_log)
    return run_node(cfg, adapter, log=_log)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
