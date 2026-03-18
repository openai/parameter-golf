import importlib.util
import sys
import types
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "data" / "cached_challenge_fineweb.py"


def load_module():
    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = lambda *args, **kwargs: None

    previous = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = fake_hf
    try:
        spec = importlib.util.spec_from_file_location("cached_challenge_fineweb_under_test", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = previous


class CachedChallengeFineWebPathTests(unittest.TestCase):
    def test_local_path_for_remote_strips_multi_segment_prefix(self):
        module = load_module()
        module.REMOTE_ROOT_PREFIX = "exports/v1"

        path = module.local_path_for_remote(
            "exports/v1/datasets/fineweb10B_sp1024/fineweb_train_000000.bin"
        )

        self.assertEqual(
            path,
            module.DATASETS_DIR / "fineweb10B_sp1024" / "fineweb_train_000000.bin",
        )

    def test_manifest_path_uses_repo_root_when_remote_prefix_is_empty(self):
        module = load_module()
        module.REMOTE_ROOT_PREFIX = ""

        self.assertEqual(module.manifest_path(), module.ROOT / "manifest.json")


if __name__ == "__main__":
    unittest.main()
