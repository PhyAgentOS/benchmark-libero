"""PyArrow hook without the optional test corpus."""

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return (
        "/tests/" in normalized
        or normalized.startswith("pyarrow/tests/")
        or normalized.endswith("_tests.pxd")
        or normalized.endswith("_tests.pyx")
        or normalized.endswith("_tests.so")
    )


def _is_test_item(item: tuple[str, str]) -> bool:
    return _is_test_path(item[0]) or _is_test_path(item[1])


hiddenimports = collect_submodules(
    "pyarrow",
    filter=lambda name: "tests" not in name and not name.endswith("conftest"),
)
datas = [item for item in collect_data_files("pyarrow") if not _is_test_item(item)]
binaries = [
    item for item in collect_dynamic_libs("pyarrow") if not _is_test_item(item)
]
