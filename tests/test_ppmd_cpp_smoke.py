"""Smoke tests for the `_ppmd_cpp` pybind11 extension (Phase 1).

These tests are intentionally minimal: they only verify the extension can be
imported and that its `version()` function returns the expected sentinel string.
They skip cleanly if:
  * The extension has not been built yet, or
  * The interpreter is the system `/bin/python3.8` (which lacks `Python.h`
    and cannot host the extension built against `.venv-smoke` Python 3.12).
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Ensure the build directory is on sys.path so the extension can be imported
# regardless of how unittest is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUILD_DIR = _REPO_ROOT / "scripts" / "ppmd_cpp"
if str(_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_DIR))


def _running_under_system_python38() -> bool:
    exe = os.path.realpath(sys.executable)
    return exe == "/bin/python3.8" or exe.endswith("/python3.8") and "venv" not in exe


class PpmdCppSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        if _running_under_system_python38():
            self.skipTest(
                "running under system /bin/python3.8 which has no Python.h; "
                "use .venv-smoke/bin/python instead"
            )
        try:
            import _ppmd_cpp  # noqa: F401
        except ImportError as e:
            self.skipTest(f"_ppmd_cpp extension not built or not importable: {e}")

    def test_ppmd_cpp_extension_imports(self) -> None:
        import _ppmd_cpp

        self.assertTrue(hasattr(_ppmd_cpp, "version"))

    def test_ppmd_cpp_version_string(self) -> None:
        import _ppmd_cpp

        v = _ppmd_cpp.version()
        self.assertIsInstance(v, str)
        self.assertEqual(v, "0.0.1")


if __name__ == "__main__":
    unittest.main()
