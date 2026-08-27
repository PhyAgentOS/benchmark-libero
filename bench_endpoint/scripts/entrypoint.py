"""PyInstaller entrypoint for the standalone bench_endpoint node."""

from bench_endpoint.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
