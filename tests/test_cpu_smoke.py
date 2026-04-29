import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "cpu_smoke.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cpu_smoke", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CpuSmokeHelpersTest(unittest.TestCase):
    def test_resolve_repo_path_keeps_absolute_paths(self):
        module = load_module()
        absolute = "/tmp/example.file"
        self.assertEqual(module.resolve_repo_path(absolute), Path(absolute))

    def test_require_existing_file_raises_clear_error(self):
        module = load_module()
        missing = REPO_ROOT / "tmp_missing_tokenizer.model"
        with self.assertRaises(FileNotFoundError) as ctx:
            module.require_existing_file(missing, "tokenizer")
        self.assertIn("tokenizer", str(ctx.exception))
        self.assertIn(str(missing), str(ctx.exception))

    def test_write_summary_creates_json_file(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = module.write_summary(
                results_dir=Path(tmpdir),
                mode="train",
                payload={"steps": 2, "losses": [1.0, 0.5]},
            )
            self.assertTrue(out_path.exists())
            self.assertEqual(out_path.suffix, ".json")
            self.assertIn("train", out_path.name)


if __name__ == "__main__":
    unittest.main()