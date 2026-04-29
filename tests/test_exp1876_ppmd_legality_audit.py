import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = REPO_ROOT / "audits" / "exp_1876_ppmd" / "ppmd_legality_audit.py"
AUDIT_OUTPUT = REPO_ROOT / "audits" / "exp_1876_ppmd" / "audit_outputs" / "static_provenance.json"
DENOMINATOR_AUDIT_OUTPUT = REPO_ROOT / "audits" / "exp_1876_ppmd" / "audit_outputs" / "denominator_audit.json"
COVERAGE_AUDIT_OUTPUT = REPO_ROOT / "audits" / "exp_1876_ppmd" / "audit_outputs" / "coverage_audit.json"
ARTIFACT_CAP_BYTES = 16_000_000
EXPECTED_VAL_TARGET_TOKENS = 40_540_160
EXPECTED_FULL_TARGET_BYTES = 151_078_222
EXPECTED_PPM_SUBSET_TOKENS = 8_000_000
EXPECTED_FIRST_8M_TARGET_BYTES = 29_365_687


def load_audit_module():
    spec = importlib.util.spec_from_file_location("exp1876_ppmd_legality_audit", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load exp_1876 PPM-D legality audit module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Exp1876PpmdLegalityAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_audit_module()
        cls.report = cls.module.build_report()
        cls.module.write_report(cls.report)

    def test_bootstrap_decodes_to_merged_source(self):
        bootstrap = self.report["bootstrap"]
        inputs = self.report["inputs"]
        self.assertTrue(bootstrap["decoded_matches_merged_source"])
        self.assertEqual(
            bootstrap["decoded_sha256"],
            inputs["train_gpt_merged_py"]["sha256"],
        )
        self.assertEqual(
            bootstrap["decoded_bytes"],
            inputs["train_gpt_merged_py"]["bytes"],
        )
        self.assertIn("lzma", bootstrap["decoder"])
        self.assertTrue(AUDIT_OUTPUT.exists())
        output = json.loads(AUDIT_OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(output["bootstrap"]["decoded_sha256"], bootstrap["decoded_sha256"])

    def test_artifact_total_under_16mb(self):
        accounting = self.report["artifact_accounting"]
        self.assertEqual(accounting["cap_bytes"], ARTIFACT_CAP_BYTES)
        self.assertEqual(
            accounting["actual_total_bytes"],
            accounting["model_bytes"] + accounting["code_bytes"],
        )
        self.assertLess(accounting["actual_total_bytes"], ARTIFACT_CAP_BYTES)
        self.assertGreater(accounting["headroom_bytes"], 0)
        self.assertEqual(accounting["log_model_bytes"], accounting["model_bytes"])
        self.assertEqual(accounting["log_code_bytes"], accounting["code_bytes"])
        self.assertEqual(accounting["log_total_bytes"], accounting["actual_total_bytes"])

    def test_static_score_before_update_patterns_detected(self):
        static = self.report["static_evidence"]
        self.assertEqual(static["_loss_bpb"]["status"], "PASS")
        self.assertTrue(static["_loss_bpb"]["probes"]["uses_byte_denominator"])

        ppm = static["_ppm_mixture_bpb"]
        self.assertEqual(ppm["status"], "PASS")
        self.assertTrue(ppm["probes"]["token_logp_divided_by_n_bytes"])
        self.assertTrue(ppm["score_before_update_order_proved"])
        self.assertLess(ppm["positions"]["score_mix_nll"], ppm["positions"]["update_count"])

        ttt = static["eval_val_ttt_score_before_update"]
        self.assertEqual(ttt["status"], "PASS")
        self.assertTrue(ttt["score_before_update_order_proved"])
        self.assertLess(ttt["positions"]["score_loss_sum"], ttt["positions"]["training_gate"])
        self.assertLess(ttt["positions"]["ppm_record_nll"], ttt["positions"]["training_gate"])

        timed = static["timed_eval_wraps_eval_val_ttt"]
        self.assertEqual(timed["status"], "PASS")
        self.assertTrue(timed["wrapped"])

        for check_name in (
            "_loss_bpb",
            "_ppm_mixture_bpb",
            "eval_val_ttt_score_before_update",
            "timed_eval_wraps_eval_val_ttt",
        ):
            self.assertTrue(static[check_name]["evidence"], check_name)
            for item in static[check_name]["evidence"]:
                self.assertIn("line", item)
                self.assertIn("text", item)

    def test_ppm_subset_cap_detected(self):
        subset = self.report["static_evidence"]["ppm_subset_selection"]
        self.assertEqual(subset["status"], "PASS")
        self.assertTrue(subset["subset_cap_detected"])
        self.assertEqual(subset["hparam_default"], 8_000_000)
        self.assertEqual(subset["log_ppm_time_tokens"], 8_000_000)
        self.assertEqual(self.report["log_extracts"]["ppm_mix"]["bytes"], 29_365_687)
        self.assertAlmostEqual(self.report["log_extracts"]["ppm_mix"]["mix_bpb"], 0.994872)
        self.assertTrue(subset["evidence"])


class Exp1876PpmdLegalityPhase2AuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_audit_module()
        cls.denominator = cls.module.build_denominator_audit()
        cls.module.write_denominator_audit(cls.denominator)
        cls.coverage = cls.module.build_coverage_audit()
        cls.module.write_coverage_audit(cls.coverage)

    def test_validation_token_and_byte_counts_match_log_or_known_reference(self):
        computed = self.denominator["computed"]
        references = self.denominator["references"]
        checks = self.denominator["checks"]

        self.assertEqual(computed["target_token_count"], EXPECTED_VAL_TARGET_TOKENS)
        self.assertEqual(references["log_val_tokens"], EXPECTED_VAL_TARGET_TOKENS)
        self.assertEqual(computed["full_target_bytes"], EXPECTED_FULL_TARGET_BYTES)
        self.assertEqual(references["known_full_target_bytes"], EXPECTED_FULL_TARGET_BYTES)
        self.assertTrue(checks["target_token_count_matches_log"]["passed"])
        self.assertTrue(checks["full_target_bytes_matches_known_reference"]["passed"])

        self.assertTrue(DENOMINATOR_AUDIT_OUTPUT.exists())
        output = json.loads(DENOMINATOR_AUDIT_OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(output["computed"]["full_target_bytes"], EXPECTED_FULL_TARGET_BYTES)

    def test_first_8m_ppm_byte_count_matches_log(self):
        computed = self.denominator["computed"]
        references = self.denominator["references"]
        checks = self.denominator["checks"]

        self.assertEqual(computed["first_8m_target_position_bytes"], EXPECTED_FIRST_8M_TARGET_BYTES)
        self.assertEqual(references["log_ppm_mix_bytes"], EXPECTED_FIRST_8M_TARGET_BYTES)
        self.assertEqual(computed["ppm_subset_tokens"], EXPECTED_PPM_SUBSET_TOKENS)
        self.assertEqual(references["log_ppm_time_tokens"], EXPECTED_PPM_SUBSET_TOKENS)
        self.assertTrue(checks["first_8m_ppm_bytes_match_log"]["passed"])

    def test_world_size_1_and_8_position_streams_are_equivalent(self):
        comparison = self.coverage["comparisons"]["world_size_1_vs_8"]
        ws1 = self.coverage["world_sizes"]["1"]
        ws8 = self.coverage["world_sizes"]["8"]

        self.assertTrue(comparison["selected_position_streams_equivalent"])
        self.assertTrue(comparison["scored_position_sets_equivalent"])
        self.assertEqual(ws1["selected_positions"]["sha256_i64le"], ws8["selected_positions"]["sha256_i64le"])
        self.assertEqual(ws1["scored_positions"]["sha256_i64le"], ws8["scored_positions"]["sha256_i64le"])
        self.assertTrue(ws8["selected_positions"]["exact_0_to_7999999"])
        self.assertEqual(ws8["selected_positions"]["count"], EXPECTED_PPM_SUBSET_TOKENS)
        self.assertEqual(ws8["selected_positions"]["first"], 0)
        self.assertEqual(ws8["selected_positions"]["last"], EXPECTED_PPM_SUBSET_TOKENS - 1)

        self.assertTrue(COVERAGE_AUDIT_OUTPUT.exists())
        output = json.loads(COVERAGE_AUDIT_OUTPUT.read_text(encoding="utf-8"))
        self.assertTrue(output["comparisons"]["world_size_1_vs_8"]["selected_position_streams_equivalent"])

    def test_no_missing_or_duplicate_scored_positions(self):
        checks = self.coverage["checks"]
        self.assertTrue(checks["world_size_1_no_missing_or_duplicate_scored_positions"]["passed"])
        self.assertTrue(checks["world_size_8_no_missing_or_duplicate_scored_positions"]["passed"])

        for world_size in ("1", "8"):
            scored = self.coverage["world_sizes"][world_size]["scored_positions"]
            self.assertEqual(scored["count"], EXPECTED_VAL_TARGET_TOKENS)
            self.assertEqual(scored["missing_count"], 0)
            self.assertEqual(scored["duplicate_count"], 0)
            self.assertEqual(scored["first"], 0)
            self.assertEqual(scored["last"], EXPECTED_VAL_TARGET_TOKENS - 1)
            self.assertTrue(scored["covers_exact_target_range"])


NORMALIZATION_AUDIT_OUTPUT = REPO_ROOT / "audits" / "exp_1876_ppmd" / "audit_outputs" / "normalization_audit.json"


class Exp1876PpmdLegalityPhase3NormalizationTest(unittest.TestCase):
    """Phase 3: formal normalization proofs and synthetic counterexamples."""

    @classmethod
    def setUpClass(cls):
        cls.module = load_audit_module()

    # ------------------------------------------------------------------
    # PPM-D byte distribution normalization (should sum to 1)
    # ------------------------------------------------------------------

    def test_ppm_only_byte_distribution_normalizes_on_synthetic_state(self):
        """Feed several synthetic byte histories into a PPM-D order-5 model,
        enumerate all 256 byte probs, assert sum approx 1.0 +/- 1e-12."""
        histories = [
            b"",                          # empty — pure uniform
            b"hello",                     # short context
            b"abcabc",                    # repeated pattern
            b"the quick brown fox jumps over the lazy dog",  # longer context
            bytes(range(256)) * 2,        # all bytes seen
        ]
        for order in (1, 3, 5):
            for history in histories:
                result = self.module.synthetic_ppm_d_normalization_test(order, history)
                self.assertAlmostEqual(
                    result["prob_sum"], 1.0, places=12,
                    msg=f"PPM-D order={order} history={history[:20]!r}... sum={result['prob_sum']}"
                )
                self.assertEqual(result["num_bytes_scored"], 256)
                self.assertTrue(result["normalizes"])

    # ------------------------------------------------------------------
    # Neural byte component is NOT a valid distribution
    # ------------------------------------------------------------------

    def test_current_neural_byte_component_is_observed_path_only(self):
        """exp(log(p_NN(token))/n_bytes) is constant for all 256 byte values,
        hence cannot be a conditional distribution over bytes."""
        test_cases = [
            (0.01, 3),
            (0.5, 1),
            (0.001, 5),
            (0.1, 2),
        ]
        for token_prob, n_bytes in test_cases:
            result = self.module.synthetic_neural_byte_counterexample(token_prob, n_bytes)
            # The nn byte "probability" is the same value regardless of byte index
            nn_val = token_prob ** (1.0 / n_bytes)
            self.assertAlmostEqual(result["nn_byte_prob"], nn_val, places=12)
            # Sum over 256 bytes is 256 * nn_val, which is almost certainly != 1
            expected_sum = 256.0 * nn_val
            self.assertAlmostEqual(result["sum_over_256"], expected_sum, places=10)
            self.assertFalse(result["is_normalized"],
                             msg=f"p={token_prob}, n={n_bytes}: sum={result['sum_over_256']} should NOT be ~1")
            # The value is constant across all bytes
            self.assertTrue(result["constant_across_bytes"])

    def test_token_probability_spread_counterexample_is_not_a_256_way_distribution(self):
        """For several (token_prob, n_bytes) pairs, verify the neural component
        and the mixture both fail to sum to 1 over 256 byte values."""
        test_cases = [
            (0.01, 3),
            (0.1, 2),
            (0.5, 4),
            (0.001, 5),
            (0.99, 1),
        ]
        for token_prob, n_bytes in test_cases:
            result = self.module.synthetic_neural_byte_counterexample(token_prob, n_bytes)
            nn_sum = result["sum_over_256"]
            # Very unlikely to be exactly 1 for arbitrary token_prob/n_bytes
            self.assertNotAlmostEqual(nn_sum, 1.0, places=6,
                                      msg=f"p={token_prob}, n={n_bytes}: sum={nn_sum} unexpectedly ≈ 1")

    def test_mixture_counterexample_not_normalized(self):
        """The full mixture lambda*nn + (1-lambda)*ppm does NOT sum to 1 when
        the neural component is the geometric-mean decomposition."""
        # Construct a uniform PPM distribution (proper, sums to 1)
        ppm_probs = [1.0 / 256.0] * 256
        test_cases = [
            (0.01, 3, 0.9),
            (0.1, 2, 0.5),
            (0.5, 4, 0.7),
        ]
        for token_prob, n_bytes, lam in test_cases:
            result = self.module.synthetic_mixture_counterexample(
                token_prob, n_bytes, ppm_probs, lam
            )
            self.assertFalse(result["is_normalized"],
                             msg=f"Mixture with p={token_prob}, n={n_bytes}, λ={lam}: "
                                 f"sum={result['mixture_sum']} should NOT be ~1")
            # PPM component alone does sum to 1
            self.assertAlmostEqual(result["ppm_component_sum"], 1.0, places=12)
            # Neural component alone does NOT sum to 1
            self.assertFalse(result["nn_component_normalized"])

    # ------------------------------------------------------------------
    # Score-before-update ordering (static trace verification)
    # ------------------------------------------------------------------

    def test_score_first_trace_static_order_passes(self):
        """Verify that in the PPM-D loop, the context update occurs AFTER
        mix_nll -= log_mix (score accumulation)."""
        report = self.module.build_report()
        ppm = report["static_evidence"]["_ppm_mixture_bpb"]
        self.assertEqual(ppm["status"], "PASS")
        self.assertTrue(ppm["score_before_update_order_proved"])
        # The score operation (mix_nll -= log_mix) must have a lower source
        # position than the update operation (d[b]=d.get(b,0)+1)
        self.assertLess(ppm["positions"]["score_mix_nll"],
                        ppm["positions"]["update_count"])

    # ------------------------------------------------------------------
    # Full normalization audit JSON output
    # ------------------------------------------------------------------

    def test_build_normalization_audit_produces_valid_output(self):
        """build_normalization_audit() produces a well-formed JSON report
        with all required proof obligations."""
        audit = self.module.build_normalization_audit()
        self.module.write_normalization_audit(audit)

        self.assertIn("ppmd_normalization_proof", audit)
        self.assertIn("neural_byte_counterproof", audit)
        self.assertIn("mixture_counterproof", audit)
        self.assertIn("score_before_update_verification", audit)
        self.assertIn("verdict", audit)

        # PPM-D normalizes
        ppmd = audit["ppmd_normalization_proof"]
        self.assertTrue(ppmd["all_histories_normalize"])

        # Neural does NOT normalize
        nn = audit["neural_byte_counterproof"]
        self.assertTrue(nn["counterexample_found"])

        # Mixture does NOT normalize
        mix = audit["mixture_counterproof"]
        self.assertTrue(mix["counterexample_found"])

        # Score-before-update passes
        sbu = audit["score_before_update_verification"]
        self.assertTrue(sbu["passed"])

        # Output file is written
        self.assertTrue(NORMALIZATION_AUDIT_OUTPUT.exists())
        output = json.loads(NORMALIZATION_AUDIT_OUTPUT.read_text(encoding="utf-8"))
        self.assertTrue(output["ppmd_normalization_proof"]["all_histories_normalize"])
        self.assertTrue(output["neural_byte_counterproof"]["counterexample_found"])


if __name__ == "__main__":
    unittest.main()