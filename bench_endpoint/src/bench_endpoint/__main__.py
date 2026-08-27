"""节点入口:python -m bench_endpoint --config bench_endpoint.yaml"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .runner import run_endpoint


def main() -> int:
    parser = argparse.ArgumentParser(prog="bench_endpoint")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置路径")
    args = parser.parse_args()
    return run_endpoint(load_config(args.config))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
