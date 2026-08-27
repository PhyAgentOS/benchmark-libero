#!/usr/bin/env bash
# Build bench_endpoint as a standalone Forge/Dora node executable.
#
# Usage:
#   ./scripts/build_pyinstaller.sh --forge-gateway-src /path/to/forge_gateway
#
# The gateway source is a build-time input because forge-tool is currently
# distributed with that repository. The exact gateway commit is pinned below;
# no checkout path is embedded in the resulting executable.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPEC_FILE="${SCRIPT_DIR}/bench_endpoint.spec"
DIST_DIR="${ROOT}/dist"
WORK_DIR="${ROOT}/build/pyinstaller"
EXPECTED_GATEWAY_COMMIT="94a5bc6f17cd87a7165eb02c6834bce76ab6a844"

usage() {
    echo "usage: $0 --forge-gateway-src PATH" >&2
    exit 2
}

GATEWAY_PATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --forge-gateway-src)
            [[ $# -ge 2 ]] || usage
            GATEWAY_PATH="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done
[[ -n "${GATEWAY_PATH}" ]] || usage

# Accept a repository root (the documented form) and derive its source tree.
# Resolving the path here keeps the checkout location out of the executable.
GATEWAY_PATH="$(cd "${GATEWAY_PATH}" && pwd)"
GATEWAY_ROOT="$(git -C "${GATEWAY_PATH}" rev-parse --show-toplevel)"
GATEWAY_SRC="${GATEWAY_ROOT}/src"
GATEWAY_HEAD="$(git -C "${GATEWAY_ROOT}" rev-parse HEAD)"
if [[ "${GATEWAY_HEAD}" != "${EXPECTED_GATEWAY_COMMIT}" ]]; then
    echo "ERROR: forge_gateway HEAD must be ${EXPECTED_GATEWAY_COMMIT}, got ${GATEWAY_HEAD}" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "ERROR: ${PYTHON_BIN} was not found" >&2
    exit 1
}

BUILD_ENV="$(mktemp -d /tmp/bench_endpoint_pyinstaller.XXXXXX)"
cleanup() {
    rm -rf "${BUILD_ENV}"
}
trap cleanup EXIT

echo "==> creating isolated Python 3.12 build environment"
"${PYTHON_BIN}" -m venv "${BUILD_ENV}"
"${BUILD_ENV}/bin/python" -m pip install \
    --disable-pip-version-check \
    --no-input \
    "pyinstaller==6.19.0" \
    "dora-rs==0.4.1" \
    "forge-msgs==1.0.1" \
    "pyarrow==23.0.1" \
    "pydantic==2.12.5" \
    "pyyaml==6.0.3"

FORGE_GATEWAY_SRC="${GATEWAY_SRC}" \
PYTHONPATH="${ROOT}/src:${GATEWAY_SRC}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${BUILD_ENV}/bin/python" - <<'PY'
import os
from pathlib import Path

import bench_endpoint  # noqa: F401
import forge_tool

gateway_src = Path(os.environ["FORGE_GATEWAY_SRC"]).resolve()
tool_file = Path(forge_tool.__file__).resolve()
if gateway_src not in tool_file.parents:
    raise SystemExit(f"forge_tool resolved outside pinned gateway source: {tool_file}")
print(f"forge_tool source: {tool_file}")
PY

rm -rf "${DIST_DIR}" "${WORK_DIR}"
mkdir -p "${DIST_DIR}" "${WORK_DIR}"

echo "==> building one-file executable"
BENCH_ENDPOINT_ROOT="${ROOT}" \
FORGE_GATEWAY_SRC="${GATEWAY_SRC}" \
PYTHONPATH="${ROOT}/src:${GATEWAY_SRC}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${BUILD_ENV}/bin/pyinstaller" \
    --noconfirm \
    --clean \
    --distpath "${DIST_DIR}" \
    --workpath "${WORK_DIR}" \
    "${SPEC_FILE}"

DIST_FILE="${DIST_DIR}/bench_endpoint"
if [[ ! -x "${DIST_FILE}" ]]; then
    echo "ERROR: executable not found: ${DIST_FILE}" >&2
    exit 1
fi

echo "==> build successful: ${DIST_FILE}"
ls -lh "${DIST_FILE}"
