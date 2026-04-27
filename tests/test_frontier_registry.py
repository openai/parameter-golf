from __future__ import annotations

import unittest

from research.frontier_registry import FRONTIER_PRESETS
from research_only.hybrid_registry import RESEARCH_ONLY_PRESETS


class FrontierRegistryTest(unittest.TestCase):
    def test_lane_separation_marks_stable_and_challenger(self) -> None:
        self.assertEqual(FRONTIER_PRESETS["sp8192_mainline_base"].lane, "stable")
        self.assertTrue(FRONTIER_PRESETS["sp8192_mainline_base"].submission_safe)
        self.assertEqual(FRONTIER_PRESETS["challenger_prefix_matcher"].lane, "challenger")
        self.assertFalse(FRONTIER_PRESETS["challenger_prefix_matcher"].submission_safe)
        self.assertTrue(FRONTIER_PRESETS["challenger_prefix_matcher"].requires_manual_rule_review)

    def test_legacy_presets_are_not_branch_of_record(self) -> None:
        self.assertEqual(FRONTIER_PRESETS["control_verified_sota"].lane, "legacy")
        self.assertFalse(FRONTIER_PRESETS["control_verified_sota"].submission_safe)

    def test_research_only_presets_stay_out_of_frontier_registry(self) -> None:
        for name in RESEARCH_ONLY_PRESETS:
            self.assertNotIn(name, FRONTIER_PRESETS)


if __name__ == "__main__":
    unittest.main()

