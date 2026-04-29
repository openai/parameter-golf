import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "eval_path_a_ppmd.py"


def load_module():
    spec = importlib.util.spec_from_file_location("eval_path_a_ppmd", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load eval_path_a_ppmd.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PathAPpmdEvalCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_module()

    def make_candidates_and_tries(self):
        # token 1 has a SentencePiece leading-space marker whose literal space
        # is emitted only after a non-boundary previous token.
        token_bytes = [b"a", b"cat", b"c", b""]
        has_space = [False, True, False, False]
        candidates = [self.m.candidate_bytes_for_token(i, token_bytes, has_space) for i in range(len(token_bytes))]
        boundary_root, non_boundary_root = self.m.build_candidate_tries(candidates)
        return candidates, boundary_root, non_boundary_root

    def test_candidate_token_bytes_include_sentencepiece_space(self):
        token_bytes = [b"word"]
        has_space = [True]
        candidate = self.m.candidate_bytes_for_token(0, token_bytes, has_space)
        self.assertEqual(candidate.after_boundary, b"word")
        self.assertEqual(candidate.after_non_boundary, b" word")

    def test_path_a_mixture_normalizes_over_tokens(self):
        _, boundary_root, _ = self.make_candidates_and_tries()
        state = self.m.PPMDState(order=2)
        state.update_bytes(b"abacadabra")
        z, target_q, terminal_count = self.m.trie_partial_z_and_target(boundary_root, state, target_id=1)
        self.assertEqual(terminal_count, 4)
        self.assertGreater(z, 0.0)
        self.assertGreater(target_q, 0.0)

        # PPM component normalized by construction over token IDs.
        ppm_probs = []
        for target_id in range(4):
            _, q, _ = self.m.trie_partial_z_and_target(boundary_root, state, target_id=target_id)
            ppm_probs.append(q / z)
        self.assertAlmostEqual(sum(ppm_probs), 1.0, places=12)

        # Mixing with a normalized neural token distribution remains normalized.
        neural_probs = [0.5, 0.25, 0.125, 0.125]
        lam = 0.7
        mix_sum = sum(lam * n + (1.0 - lam) * p for n, p in zip(neural_probs, ppm_probs))
        self.assertAlmostEqual(mix_sum, 1.0, places=12)

    def test_score_first_state_hash_changes_only_after_update(self):
        candidates, boundary_root, non_boundary_root = self.make_candidates_and_tries()
        state = self.m.PPMDState(order=2)
        state.update_bytes(b"aaaa")
        before = state.state_digest()
        record = self.m.path_a_score_position(
            state,
            boundary_root,
            non_boundary_root,
            target_id=1,
            prev_is_boundary=False,
            neural_nll_nats=-math.log(0.25),
            actual_target_bytes=candidates[1].after_non_boundary,
            lambda_hi=0.5,
            lambda_lo=0.5,
        )
        self.assertEqual(record["state_digest_before"], before)
        self.assertEqual(record["state_digest_after_score"], before)
        self.assertNotEqual(record["state_digest_after_update"], before)
        self.assertTrue(record["state_changed_only_after_update"])
        self.assertGreater(record["p_mix_target"], 0.0)
        self.assertLessEqual(record["p_mix_target"], 1.0)

    def test_rank_sharded_vocab_reduction_matches_single_rank(self):
        _, boundary_root, _ = self.make_candidates_and_tries()
        state = self.m.PPMDState(order=3)
        state.update_bytes(b"the cat sat")
        full = self.m.trie_partial_z_and_target(boundary_root, state, target_id=2)
        shard0 = self.m.trie_partial_z_and_target(boundary_root, state, target_id=2, shard_start=0, shard_end=2)
        shard1 = self.m.trie_partial_z_and_target(boundary_root, state, target_id=2, shard_start=2, shard_end=4)
        combined = self.m.combine_path_a_partials([shard0, shard1])
        self.assertAlmostEqual(combined[0], full[0], places=14)
        self.assertAlmostEqual(combined[1], full[1], places=14)
        self.assertEqual(combined[2], full[2])

    def test_single_shard_score_is_rejected_without_reduction(self):
        candidates, boundary_root, non_boundary_root = self.make_candidates_and_tries()
        state = self.m.PPMDState(order=2)
        with self.assertRaisesRegex(ValueError, "single vocab shard"):
            self.m.path_a_score_position(
                state,
                boundary_root,
                non_boundary_root,
                target_id=0,
                prev_is_boundary=True,
                neural_nll_nats=-math.log(0.5),
                actual_target_bytes=candidates[0].after_boundary,
                shard_start=0,
                shard_end=2,
            )

    def test_ppmd_byte_distribution_normalizes_after_context_updates(self):
        state = self.m.PPMDState(order=3)
        state.update_bytes(b"abracadabra")
        probs = state.byte_probs()
        self.assertEqual(len(probs), 256)
        self.assertAlmostEqual(sum(probs), 1.0, places=12)
        self.assertTrue(all(p >= 0.0 for p in probs))

    def test_single_byte_probability_matches_full_distribution(self):
        state = self.m.PPMDState(order=4)
        state.update_bytes(b"the quick brown fox jumps over the lazy dog")
        full = state.byte_probs()
        for b in (0, 32, ord("e"), ord("z"), 255):
            self.assertAlmostEqual(state.byte_prob(b), full[b], places=15)

        virtual = state.clone_virtual().fork_and_update(ord("!"))
        full_virtual = virtual.byte_probs()
        for b in (0, 33, ord("o"), ord("q"), 255):
            self.assertAlmostEqual(virtual.byte_prob(b), full_virtual[b], places=15)

    def test_audit_schema_for_prefix_eval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "smoke.json"
            report = self.m.run_core_smoke(out, 4)
            self.assertTrue(out.exists())
            loaded = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(loaded["mode"], "core-smoke")
            self.assertEqual(loaded["positions"], report["positions"])
            self.assertIn("candidate_token_evals", loaded["cost_estimate"])
            self.assertIn("candidate_trie_stats", loaded)

    def test_score_path_a_arrays_returns_bpb_and_samples(self):
        candidates, _, _ = self.make_candidates_and_tries()
        target_ids = [0, 1, 2]
        prev_ids = [-1, 0, 1]
        nll_nats = [-math.log(0.5), -math.log(0.25), -math.log(0.25)]
        is_boundary = [False, False, False, True]
        report = self.m.score_path_a_arrays(
            target_ids,
            prev_ids,
            nll_nats,
            candidates,
            is_boundary,
            order=2,
            lambda_hi=0.5,
            lambda_lo=0.5,
            normalization_sample_every=1,
        )
        self.assertEqual(report["positions"], 3)
        self.assertGreater(report["total_bits"], 0.0)
        self.assertGreater(report["total_bytes"], 0)
        self.assertGreater(report["bpb"], 0.0)
        self.assertEqual(len(report["normalization_samples"]), 3)
        self.assertTrue(all(sample["score_first"] for sample in report["normalization_samples"]))

    def test_cli_help_runs_without_exp_import(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--core-smoke", proc.stdout)

    def test_default_fresh_eval_is_guarded(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--max-positions", "1"],
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("intentionally guarded", proc.stderr)


if __name__ == "__main__":
    unittest.main()