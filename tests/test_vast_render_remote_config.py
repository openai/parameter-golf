from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deploy.vast.render_remote_config import render_remote_config


class RenderRemoteConfigTests(unittest.TestCase):
    def test_rewrites_runner_and_output_paths(self):
        with tempfile.TemporaryDirectory() as tempdir:
            workdir = Path(tempdir) / "workspace"
            config = {
                "runner": {
                    "workdir": "/tmp/old",
                    "script_path": "records/demo/train_gpt.py",
                    "python_bin": "python",
                    "activate_script": None,
                    "gpus": 1,
                    "logs_dir": "logs",
                },
                "fixed_env": {
                    "SEED": 1337,
                    "TRAIN_BATCH_TOKENS": 524288,
                },
                "search": {
                    "output_root": "search_runs/demo",
                },
            }
            rendered = render_remote_config(
                config,
                workdir=workdir,
                python_bin=workdir / ".venv-remote/bin/python",
                gpus=8,
                logs_dir=workdir / "logs",
                fixed_env_overrides={"TRAIN_BATCH_TOKENS": 123456},
            )

            self.assertEqual(rendered["runner"]["workdir"], str(workdir))
            self.assertEqual(rendered["runner"]["python_bin"], str(workdir / ".venv-remote/bin/python"))
            self.assertEqual(rendered["runner"]["gpus"], 8)
            self.assertEqual(rendered["runner"]["script_path"], str((workdir / "records/demo/train_gpt.py").resolve()))
            self.assertEqual(rendered["runner"]["logs_dir"], str((workdir / "logs").resolve()))
            self.assertEqual(rendered["search"]["output_root"], str((workdir / "search_runs/demo").resolve()))
            self.assertEqual(rendered["fixed_env"]["TRAIN_BATCH_TOKENS"], 123456)


if __name__ == "__main__":
    unittest.main()
