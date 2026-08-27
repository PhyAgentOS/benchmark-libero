"""Forge Tool endpoint,把 benchmark 节点包成 bench.describe / bench.run。

本包**只跑在 3.12**(forge_bench env),因为 forge_tool 用了 PEP 695 的
`type X = ...` 语法。benchmark 节点跑在 Isaac Sim 的 3.11 里,两边靠朴素 JSON
通道对话,benchmark 侧一行 forge_tool 都不 import。
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
