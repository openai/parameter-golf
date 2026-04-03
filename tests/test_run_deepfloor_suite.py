from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

from tools import run_deepfloor_suite


class RunDeepFloorSuiteTests(unittest.TestCase):
    def test_resolve_python_bin_accepts_existing_path(self) -> None:
        resolved = run_deepfloor_suite.resolve_python_bin(sys.executable)
        self.assertEqual(resolved, Path(sys.executable))

    def test_resolve_python_bin_uses_path_lookup_for_command_name(self) -> None:
        with mock.patch("tools.run_deepfloor_suite.shutil.which", return_value="/usr/bin/python3") as which:
            resolved = run_deepfloor_suite.resolve_python_bin("python3")
        which.assert_called_once_with("python3")
        self.assertEqual(resolved, Path("/usr/bin/python3"))

    def test_resolve_python_bin_raises_for_missing_command(self) -> None:
        with mock.patch("tools.run_deepfloor_suite.shutil.which", return_value=None):
            with self.assertRaises(FileNotFoundError):
                run_deepfloor_suite.resolve_python_bin("definitely-not-a-python-bin")


if __name__ == "__main__":
    unittest.main()
