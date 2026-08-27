# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller one-file specification for the LIBERO benchmark node."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


_node_root = Path(os.environ["LIBERO_NODE_ROOT"]).resolve()
_entry = _node_root / "scripts" / "entrypoint.py"
_hooks = _node_root / "scripts" / "hooks"


def _module_dir(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"required build dependency is not importable: {name}")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


_libero_dir = _module_dir("libero.libero")
_robosuite_dir = _module_dir("robosuite")


def _add_data(source: Path, destination: str) -> None:
    if not source.exists():
        raise SystemExit(f"required package data is missing: {source}")
    datas.append((str(source), destination))


def _without_tests(name: str) -> bool:
    return (
        ".tests" not in name
        and not name.endswith(".test")
        and not name.endswith("_test")
        and not name.endswith("conftest")
    )


hiddenimports = [
    "_core.adapter",
    "_core.config",
    "_core.node",
    "_core.result",
    "_core.runner",
    "_core.session",
    "_core.video",
    "scipy._external.array_api_compat.numpy.fft",
    "scipy._external.array_api_compat.numpy.linalg",
    "numpy.core._multiarray_tests",
]
for package in (
    "libero.libero",
    "robosuite",
    "mujoco",
    "dora",
    "forge_msgs",
    "gymnasium",
    "yaml",
    "PIL",
    "cv2",
    "OpenGL",
    "scipy",
    "numba",
):
    hiddenimports.extend(
        module for module in collect_submodules(package) if _without_tests(module)
    )

datas = []
for package in ("dora", "forge_msgs", "pydantic", "yaml", "PIL"):
    datas.extend(collect_data_files(package))

# Small, versioned LIBERO metadata is included.  The large scene/object asset
# tree remains a host/runtime asset and is linked by the runtime hook.
_add_data(_libero_dir / "bddl_files", "libero/libero/bddl_files")
_add_data(_libero_dir / "init_files", "libero/libero/init_files")

# LIBERO's Panda environment needs these robosuite resources only.
_add_data(
    _robosuite_dir / "models" / "assets" / "robots" / "panda",
    "robosuite/models/assets/robots/panda",
)
_add_data(
    _robosuite_dir / "models" / "assets" / "grippers" / "panda_gripper.xml",
    "robosuite/models/assets/grippers",
)
_add_data(
    _robosuite_dir / "models" / "assets" / "grippers" / "meshes" / "panda_gripper",
    "robosuite/models/assets/grippers/meshes/panda_gripper",
)
_add_data(
    _robosuite_dir / "models" / "assets" / "mounts" / "rethink_mount.xml",
    "robosuite/models/assets/mounts",
)
_add_data(
    _robosuite_dir / "models" / "assets" / "mounts" / "meshes" / "rethink_mount",
    "robosuite/models/assets/mounts/meshes/rethink_mount",
)
_add_data(
    _robosuite_dir / "models" / "assets" / "arenas",
    "robosuite/models/assets/arenas",
)
_add_data(
    _robosuite_dir / "controllers" / "config",
    "robosuite/controllers/config",
)
_add_data(
    _robosuite_dir / "models" / "assets" / "mounts" / "null_mount.xml",
    "robosuite/models/assets/mounts",
)
_add_data(
    _robosuite_dir / "models" / "assets" / "base.xml",
    "robosuite/models/assets",
)

binaries = []
for package in ("mujoco", "dora", "forge_msgs", "pyarrow", "scipy", "numba"):
    binaries.extend(collect_dynamic_libs(package))

# NumPy from the conda benchmark environment links against BLAS/LAPACK and
# Fortran support libraries that live next to the interpreter rather than
# inside numpy/. Include them by discovered prefix, never by a checkout path.
_runtime_lib_dir = Path(getattr(sys, "base_prefix", sys.prefix)) / "lib"
for _pattern in (
    "libblas.so*",
    "liblapack.so*",
    "libopenblas*.so*",
    "libgfortran.so*",
    "libquadmath.so*",
    "libgcc_s.so*",
):
    for _library in _runtime_lib_dir.glob(_pattern):
        binaries.append((str(_library), "."))

a = Analysis(
    [str(_entry)],
    pathex=[str(_node_root), str(_node_root.parent)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(_hooks)],
    runtime_hooks=[str(_hooks / "pyi_rth_libero_assets.py")],
    hooksconfig={},
    excludes=[
        "pytest",
        "tests",
        "pyarrow.tests",
        "pyarrow.conftest",
        "mujoco.introspect",
        "robosuite.demos",
        "robosuite.scripts",
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
    name="libero_benchmark",
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
