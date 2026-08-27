"""Collect NumPy's package-local binaries without requiring conda metadata.

The build environment intentionally reuses the host benchmark environment via
``venv --system-site-packages``. In that layout PyInstaller detects the base
conda installation, but cannot use its conda metadata from the temporary venv.
Collecting NumPy's package-local shared libraries is sufficient for the frozen
node and keeps the build independent of the host conda prefix.
"""

from PyInstaller.utils.hooks import collect_dynamic_libs


binaries = collect_dynamic_libs("numpy")
