from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from search.config import RunnerConfig
from search.runner import PreparedRun, build_prepared_run, launch_run


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        base = Path(self.tempdir.name)
        fake_venv = base / "venv" / "bin"
        fake_venv.mkdir(parents=True, exist_ok=True)
        self.runner = RunnerConfig(
            workdir=base,
            script_path=base / "train_gpt.py",
            python_bin=fake_venv / "python",
            activate_script=fake_venv / "activate",
            gpus=1,
            logs_dir=base / "logs",
        )
        self.fixed_env = {"SEED": 1337, "ITERATIONS": 1000}

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_builds_single_gpu_command(self):
        prepared = build_prepared_run(
            self.runner,
            self.fixed_env,
            {"MATRIX_LR": 0.02, "WARMDOWN_FRACTION": 0.25},
            run_id="demo_0000",
            output_root=Path(self.tempdir.name) / "out",
        )
        self.assertIn(f"exec {self.runner.python_bin}", prepared.display_command)
        self.assertIn("WARMDOWN_ITERS=250", prepared.display_command)

    def test_builds_multi_gpu_command(self):
        runner = RunnerConfig(
            workdir=self.runner.workdir,
            script_path=self.runner.script_path,
            python_bin=self.runner.python_bin,
            activate_script=self.runner.activate_script,
            gpus=8,
            logs_dir=self.runner.logs_dir,
        )
        prepared = build_prepared_run(
            runner,
            self.fixed_env,
            {"MATRIX_LR": 0.02},
            run_id="demo_0001",
            output_root=Path(self.tempdir.name) / "out",
        )
        self.assertIn("-m torch.distributed.run --standalone --nproc_per_node=8", prepared.display_command)

    def test_launch_run_streams_stdout_and_invokes_callback(self):
        base = Path(self.tempdir.name)
        stdout_path = base / "out" / "demo.stdout"
        log_path = base / "logs" / "demo.txt"
        status_path = base / "out" / "run_status" / "demo.json"
        prepared = PreparedRun(
            run_id="demo",
            run_env={},
            argv=[
                "/bin/bash",
                "-lc",
                "printf 'line1\\nline2\\n'",
            ],
            display_command="printf 'line1\\nline2\\n'",
            log_path=log_path,
            stdout_path=stdout_path,
            status_path=status_path,
        )
        seen: list[str] = []

        completed = launch_run(
            prepared,
            workdir=base,
            tee_to_stdout=False,
            on_output_line=seen.append,
        )

        self.assertEqual(completed.return_code, 0)
        self.assertEqual(seen, ["line1", "line2"])
        self.assertEqual(stdout_path.read_text(encoding="utf-8"), "line1\nline2\n")


if __name__ == "__main__":
    unittest.main()
