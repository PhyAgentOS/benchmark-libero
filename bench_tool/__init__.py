"""LIBERO benchmark 的进程内 Forge 工具端点（benchmark.controller）。

一个进程同时跑两件事：

- 数据面：批次调度、锁步、落盘（_core，不 import forge_tool）；
- 工具面：describe(Query) + run(Session) 两个 operation，经 dora 的
  tool_in / tool_out 通道与 gateway 通信（本包，依赖 forge_tool）。

两个面都活在同一个 dora 事件循环里。endpoint 的 start() 只登记不执行，
阻塞的 episode reset 由事件循环在 tool 分发之外拾取，见 node.py。

依赖方向：bench_tool → _core，_core 不反向依赖。
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
