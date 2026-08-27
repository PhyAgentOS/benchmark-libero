"""Map configured LIBERO assets into the frozen package at runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _configured_assets() -> Path | None:
    config_dir = Path(
        os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    )
    config_file = config_dir / "config.yaml"
    if not config_file.is_file():
        return None
    try:
        import yaml

        data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    value = data.get("assets")
    if not value:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))).strip()).resolve()
    return path if path.is_dir() else None


def _link_assets() -> None:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if not bundle_root:
        return
    assets = _configured_assets()
    if assets is None:
        return
    target = Path(bundle_root) / "libero" / "libero" / "assets"
    if target.exists() or target.is_symlink():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(assets, target_is_directory=True)
    except OSError:
        pass


_link_assets()
