"""bench_endpoint 的节点配置。第一版只有三个旋钮,不做通用框架。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .invocations import DEFAULT_RETENTION
from .peer import DEFAULT_HEARTBEAT_TIMEOUT_S


@dataclass
class BenchEndpointConfig:
    heartbeat_timeout_s: float = DEFAULT_HEARTBEAT_TIMEOUT_S
    retention: int = DEFAULT_RETENTION
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchEndpointConfig:
        cfg = cls()
        cfg.heartbeat_timeout_s = float(data.get("heartbeat_timeout_s", cfg.heartbeat_timeout_s))
        cfg.retention = int(data.get("retention", cfg.retention))
        cfg.log_level = str(data.get("log_level", cfg.log_level))
        if cfg.heartbeat_timeout_s <= 0:
            raise ValueError(f"heartbeat_timeout_s 必须 >0,收到 {cfg.heartbeat_timeout_s}")
        if cfg.retention < 1:
            raise ValueError(f"retention 必须 ≥1,收到 {cfg.retention}")
        return cfg


def load_config(path: str | None) -> BenchEndpointConfig:
    if not path:
        return BenchEndpointConfig()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} 顶层必须是映射")
    return BenchEndpointConfig.from_dict(data)


__all__ = ["BenchEndpointConfig", "load_config"]
