from __future__ import annotations

import unittest
from pathlib import Path

from search.config import load_search_config


ROOT = Path(__file__).resolve().parents[1]


class SearchConfigPresetTests(unittest.TestCase):
    def test_v2_wd_sliding_local_has_weight_decay_knobs(self):
        config = load_search_config(ROOT / "search_configs/metastack_v2_wd_sliding_local.yaml")
        self.assertIn("MUON_WEIGHT_DECAY", config.search_space)
        self.assertIn("SCALAR_WEIGHT_DECAY", config.search_space)
        self.assertEqual(config.fixed_env["TOKEN_WEIGHT_DECAY"], 0.0)
        self.assertEqual(config.fixed_env["HEAD_WEIGHT_DECAY"], 0.0)
        self.assertIn("2026-03-20_MetaStack_v2_WD/train_gpt.py", str(config.runner.script_path))


if __name__ == "__main__":
    unittest.main()
