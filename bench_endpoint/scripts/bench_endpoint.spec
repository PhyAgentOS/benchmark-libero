# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the standalone bench_endpoint node."""

import os

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


_root_dir = os.environ["BENCH_ENDPOINT_ROOT"]
_gateway_src = os.environ["FORGE_GATEWAY_SRC"]
_entry = os.path.join(_root_dir, "scripts", "entrypoint.py")
_hooks_dir = os.path.join(_root_dir, "scripts", "hooks")

# These packages are imported dynamically by the Forge/Dora binding or by
# extension modules. Keep the list explicit so the one-file node has no
# checkout or PYTHONPATH dependency at runtime.
_packages = (
    "bench_endpoint",
    "forge_tool",
    "forge_msgs",
    "dora",
    "pydantic",
    "yaml",
)
hiddenimports = [
    module
    for package in _packages
    for module in collect_submodules(package)
    if not (
        ".tests" in module
        or ".test" in module
        or module.endswith(".conftest")
        or module.endswith("_tests")
        or ".benchmark" in module
    )
]

datas = []
binaries = []
for package in ("forge_msgs", "dora", "pydantic", "yaml"):
    datas.extend(collect_data_files(package))
    binaries.extend(collect_dynamic_libs(package))

a = Analysis(
    [_entry],
    pathex=[os.path.join(_root_dir, "src"), _gateway_src],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[_hooks_dir],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "tests",
        "pyarrow.tests",
        "pyarrow.conftest",
        "numpy._core._multiarray_tests",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="bench_endpoint",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
