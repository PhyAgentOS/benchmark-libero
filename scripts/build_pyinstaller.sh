#!/usr/bin/env bash
# Build the standalone LIBERO benchmark node.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
NODE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
DIST_DIR=$NODE_ROOT/dist
WORK_DIR=$NODE_ROOT/build/pyinstaller
PYTHON_BIN=${PYTHON_BIN:-python3.12}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: python3.12 was not found; set PYTHON_BIN in this script or invoke it with a compatible shell" >&2
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import importlib

for name in ("libero", "robosuite", "mujoco", "dora", "forge_msgs", "pyarrow", "scipy", "numba"):
    importlib.import_module(name)
    print(f"build dependency available: {name}")
PY

BUILD_ENV=$(mktemp -d /tmp/libero_benchmark_pyinstaller.XXXXXX)
cleanup() {
    rm -rf "$BUILD_ENV"
}
trap cleanup EXIT

echo "==> creating isolated PyInstaller environment"
"$PYTHON_BIN" -m venv --system-site-packages "$BUILD_ENV"
"$BUILD_ENV/bin/python" -m pip install --disable-pip-version-check --no-input --index-url https://download.pytorch.org/whl/cpu --no-deps torch==2.5.1+cpu
"$BUILD_ENV/bin/python" -m pip install \
    --disable-pip-version-check \
    --no-input \
    pyinstaller==6.19.0 \
    dora-rs==0.4.1 \
    forge-msgs==1.0.1 \
    pyarrow==23.0.1 \
    pydantic==2.12.5 \
    pyyaml==6.0.3

rm -rf "$DIST_DIR" "$WORK_DIR"
mkdir -p "$DIST_DIR" "$WORK_DIR"

echo "==> building one-file executable"
LIBERO_NODE_ROOT="$NODE_ROOT" \
PYTHONPATH="$NODE_ROOT:$NODE_ROOT/.." \
    "$BUILD_ENV/bin/pyinstaller" \
    --noconfirm \
    --clean \
    --distpath "$DIST_DIR" \
    --workpath "$WORK_DIR" \
    "$SCRIPT_DIR/libero_benchmark.spec"

DIST_FILE="$DIST_DIR/libero_benchmark"
if [[ ! -x "$DIST_FILE" ]]; then
    echo "ERROR: executable not found: $DIST_FILE" >&2
    exit 1
fi

echo "==> build successful: $DIST_FILE"
ls -lh "$DIST_FILE"
