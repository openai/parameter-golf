import importlib.util
import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "eval_path_b_ppmd.py"


def load_module():
    spec = importlib.util.spec_from_file_location("eval_path_b_ppmd", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load eval_path_b_ppmd.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PathBPpmdEvalPrimitivesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_module()

    def make_synthetic_sequences(self):
        pieces = ["a", "ab", "ac", "b", "<pad>"]
        return [
            self.m.token_byte_sequences_from_piece(i, piece, is_special=(piece == "<pad>"))
            for i, piece in enumerate(pieces)
        ]

    def test_reference_marginal_matches_bruteforce(self):
        sequences = self.make_synthetic_sequences()
        probs = [0.10, 0.20, 0.30, 0.25, 0.15]
        trie = self.m.build_byte_trie(sequences, mode="boundary")

        for prefix in (b"", b"a"):
            got = self.m.neural_byte_distribution(trie, probs, prefix)
            expected = self.m.bruteforce_neural_byte_distribution(sequences, probs, prefix, mode="boundary")
            self.assertEqual(set(got), set(expected))
            for byte_value, expected_prob in expected.items():
                self.assertAlmostEqual(got[byte_value], expected_prob, places=14)

    def test_neural_byte_distribution_normalizes(self):
        sequences = self.make_synthetic_sequences()
        probs = [0.05, 0.40, 0.10, 0.35, 0.10]
        trie = self.m.build_byte_trie(sequences, mode="boundary")

        for prefix in (b"", b"a"):
            dist = self.m.neural_byte_distribution(trie, probs, prefix)
            self.m.assert_distribution_normalized(dist)
            self.assertTrue(all(p >= 0.0 for p in dist.values()))

    def test_terminal_prefix_mass_is_excluded(self):
        sequences = [
            self.m.token_byte_sequences_from_piece(0, "a"),
            self.m.token_byte_sequences_from_piece(1, "ab"),
            self.m.token_byte_sequences_from_piece(2, "ac"),
        ]
        probs = [0.60, 0.10, 0.30]
        trie = self.m.build_byte_trie(sequences, mode="boundary")

        node = self.m.find_prefix_node(trie, b"a")
        self.assertIsNotNone(node)
        self.assertAlmostEqual(self.m.subtree_mass(node, probs), 1.0, places=14)
        self.assertAlmostEqual(self.m.terminal_mass(node, probs), 0.60, places=14)
        self.assertAlmostEqual(self.m.continuable_mass(node, probs), 0.40, places=14)

        dist = self.m.neural_byte_distribution(trie, probs, b"a")
        self.assertAlmostEqual(dist[ord("b")], 0.25, places=14)
        self.assertAlmostEqual(dist[ord("c")], 0.75, places=14)
        self.m.assert_distribution_normalized(dist)

    def test_sentencepiece_leading_space_modes(self):
        leading = self.m.token_byte_sequences_from_piece(0, "▁cat")
        plain = self.m.token_byte_sequences_from_piece(1, "dog")
        fallback_space = self.m.token_byte_sequences_from_piece(2, "<0x20>")
        marker_only = self.m.token_byte_sequences_from_piece(3, "▁")

        self.assertEqual(leading.base_bytes, b"cat")
        self.assertEqual(leading.after_boundary, b"cat")
        self.assertEqual(leading.after_non_boundary, b" cat")
        self.assertNotIn("▁".encode("utf-8"), (leading.base_bytes, leading.after_boundary))
        self.assertEqual(plain.after_boundary, b"dog")
        self.assertEqual(plain.after_non_boundary, b"dog")
        self.assertEqual(fallback_space.base_bytes, b" ")
        self.assertEqual(marker_only.after_boundary, b"")
        self.assertEqual(marker_only.after_non_boundary, b" ")

        probs = [0.20, 0.30, 0.40, 0.10]
        boundary_trie, non_boundary_trie = self.m.build_mode_tries([leading, plain, fallback_space, marker_only])
        boundary_root = self.m.neural_byte_distribution(boundary_trie, probs, b"")
        non_boundary_root = self.m.neural_byte_distribution(non_boundary_trie, probs, b"")

        self.assertIn(ord("c"), boundary_root)
        self.assertNotIn(ord("c"), non_boundary_root)
        self.assertIn(ord(" "), non_boundary_root)
        self.assertAlmostEqual(non_boundary_root[ord(" ")], (0.20 + 0.40 + 0.10) / 1.0, places=14)
        self.m.assert_distribution_normalized(boundary_root)
        self.m.assert_distribution_normalized(non_boundary_root)

    def test_zero_byte_special_tokens_do_not_break_root_distribution(self):
        sequences = [
            self.m.token_byte_sequences_from_piece(0, "<unk>", is_special=True),
            self.m.token_byte_sequences_from_piece(1, "a"),
            self.m.token_byte_sequences_from_piece(2, "b"),
        ]
        probs = [0.98, 0.01, 0.01]
        trie = self.m.build_byte_trie(sequences, mode="boundary")

        self.assertAlmostEqual(self.m.terminal_mass(trie, probs), 0.98, places=14)
        self.assertAlmostEqual(self.m.continuable_mass(trie, probs), 0.02, places=14)
        dist = self.m.neural_byte_distribution(trie, probs, b"")
        self.assertAlmostEqual(dist[ord("a")], 0.5, places=14)
        self.assertAlmostEqual(dist[ord("b")], 0.5, places=14)
        self.m.assert_distribution_normalized(dist)

    def test_optimized_marginal_matches_reference(self):
        sequences = [
            self.m.token_byte_sequences_from_piece(0, "a"),
            self.m.token_byte_sequences_from_piece(1, "ab"),
            self.m.token_byte_sequences_from_piece(2, "ac"),
            self.m.token_byte_sequences_from_piece(3, "b"),
            self.m.token_byte_sequences_from_piece(4, "<pad>", is_special=True),
        ]
        probs = [0.10, 0.20, 0.30, 0.25, 0.15]
        trie = self.m.build_byte_trie(sequences, mode="boundary")
        tables = self.m.build_optimized_trie_tables(trie)

        self.assertEqual(len(tables.token_order), len(sequences))
        for prefix in (b"", b"a"):
            expected = self.m.neural_byte_distribution(trie, probs, prefix)
            got = self.m.optimized_neural_byte_distribution(tables, probs, prefix)
            dense = self.m.optimized_neural_byte_distribution_dense(tables, probs, prefix)
            self.assertEqual(set(got), set(expected))
            self.assertEqual(len(dense), 256)
            self.assertAlmostEqual(sum(dense), 1.0, places=14)
            for byte_value, expected_prob in expected.items():
                self.assertAlmostEqual(got[byte_value], expected_prob, places=14)
                self.assertAlmostEqual(dense[byte_value], expected_prob, places=14)

        terminal_prefix = self.m.optimized_neural_byte_distribution(tables, probs, b"a")
        self.assertAlmostEqual(terminal_prefix[ord("b")], 0.40, places=14)
        self.assertAlmostEqual(terminal_prefix[ord("c")], 0.60, places=14)

    def test_vectorized_target_path_logprobs_match_reference(self):
        torch = self.m._optional_torch()
        if torch is None:
            self.skipTest("torch is required for vectorized target path extraction")

        sequences = [
            self.m.token_byte_sequences_from_piece(0, "a"),
            self.m.token_byte_sequences_from_piece(1, "ab"),
            self.m.token_byte_sequences_from_piece(2, "▁a"),
            self.m.token_byte_sequences_from_piece(3, "▁ab"),
            self.m.token_byte_sequences_from_piece(4, "b"),
            self.m.token_byte_sequences_from_piece(5, "<pad>", is_special=True),
        ]
        boundary_trie, non_boundary_trie = self.m.build_mode_tries(sequences)
        boundary_tables = self.m.build_optimized_trie_tables(boundary_trie)
        non_boundary_tables = self.m.build_optimized_trie_tables(non_boundary_trie)
        path_metadata = self.m.build_mode_token_path_metadata(
            sequences,
            {"boundary": boundary_trie, "non_boundary": non_boundary_trie},
            {"boundary": boundary_tables, "non_boundary": non_boundary_tables},
        )

        probs = torch.tensor(
            [
                [0.03, 0.11, 0.17, 0.19, 0.23, 0.27],
                [0.29, 0.07, 0.13, 0.31, 0.17, 0.03],
                [0.08, 0.22, 0.26, 0.10, 0.24, 0.10],
                [0.18, 0.09, 0.28, 0.14, 0.21, 0.10],
            ],
            dtype=torch.float64,
        )
        probs = probs / probs.sum(dim=1, keepdim=True)
        target_ids = torch.tensor([1, 3, 2, 5], dtype=torch.long)
        mode_flags = torch.tensor([0, 1, 1, 0], dtype=torch.long)

        records = self.m.vectorized_target_path_logprobs(
            probs,
            target_ids,
            mode_flags,
            {0: boundary_tables, 1: non_boundary_tables},
            path_metadata,
        )

        expected = []
        for row, target_id, mode_flag in zip(range(4), target_ids.tolist(), mode_flags.tolist()):
            mode = "boundary" if mode_flag == 0 else "non_boundary"
            emitted = sequences[target_id].bytes_for_mode(mode)
            prefix = b""
            tables = boundary_tables if mode == "boundary" else non_boundary_tables
            trie = boundary_trie if mode == "boundary" else non_boundary_trie
            row_probs = probs[row].tolist()
            for byte_offset, byte_value in enumerate(emitted):
                reference = self.m.neural_byte_distribution(trie, row_probs, prefix)
                optimized = self.m.optimized_neural_byte_distribution(tables, row_probs, prefix)
                self.assertAlmostEqual(reference[byte_value], optimized[byte_value], places=14)
                expected.append((row, byte_offset, byte_value, math.log(reference[byte_value])))
                prefix += bytes([byte_value])

        self.assertEqual(
            [(r.absolute_token_position, r.byte_offset_in_token, r.byte_value) for r in records],
            [(row, offset, byte_value) for row, offset, byte_value, _ in expected],
        )
        for record, (_, _, _, expected_logprob) in zip(records, expected):
            self.assertAlmostEqual(record.neural_logprob, expected_logprob, places=12)

    def test_ppmd_distribution_normalizes(self):
        model = self.m.PPMDByteModel(order=3)
        self.m.assert_dense_distribution_normalized(model.distribution())
        model.update_bytes(b"abracadabra")

        for history in (b"", b"a", b"bra", b"zzz"):
            dist = model.distribution(history)
            self.assertEqual(len(dist), 256)
            self.m.assert_dense_distribution_normalized(dist)
            self.assertTrue(all(p >= 0.0 for p in dist))

    def test_mixture_distribution_normalizes(self):
        model = self.m.PPMDByteModel(order=2)
        model.update_bytes(b"banana bandana")
        ppmd_dist = model.distribution(b"na")
        neural_dist = {ord("a"): 0.70, ord("n"): 0.20, ord(" "): 0.10}

        mixed = self.m.mixture_byte_distribution(
            neural_dist,
            ppmd_dist,
            ppmd_lambda=0.35,
            target_byte=ord("a"),
        )
        self.assertEqual(len(mixed.probs), 256)
        self.m.assert_dense_distribution_normalized(mixed.probs)
        self.assertAlmostEqual(mixed.target_prob, mixed.probs[ord("a")], places=14)
        self.assertAlmostEqual(mixed.target_logprob, math.log(mixed.target_prob), places=14)

    def test_score_before_update_ordering(self):
        model = self.m.PPMDByteModel(order=1)
        model.update_bytes(b"aa")
        old_prob = model.distribution()[ord("b")]

        score = self.m.score_ppmd_byte_then_update(model, ord("b"))
        self.assertAlmostEqual(score.probability, old_prob, places=14)
        self.assertAlmostEqual(score.logprob, math.log(old_prob), places=14)

        new_prob_for_old_context = model.distribution(b"a")[ord("b")]
        self.assertGreater(new_prob_for_old_context, old_prob)

    def test_cli_dry_run_artifact_paths(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "syntactically_bad_source.py"
            artifact = tmp / "final_model.int6.ptz"
            output = tmp / "plan.json"

            source.write_text("raise RuntimeError('dry-run must not import this file')\n", encoding="utf-8")
            artifact.write_bytes(b"tiny placeholder artifact")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dry-run",
                    "--source-python",
                    str(source),
                    "--artifact-path",
                    str(artifact),
                    "--output-json",
                    str(output),
                    "--subset-tokens",
                    "123",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output.exists())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["mode"], "dry_run")
            self.assertEqual(payload["source_python_path"], str(source.resolve()))
            self.assertEqual(payload["artifact_path"], str(artifact.resolve()))
            self.assertEqual(payload["artifact_size_bytes"], artifact.stat().st_size)
            self.assertEqual(payload["subset_tokens"], 123)
            self.assertFalse(payload["dry_run_imported_source"])
            self.assertIn("future_runpod_command_suggestions", payload)

    def test_shard_merge_preserves_order(self):
        shards = [
            [
                self.m.ByteLogprobRecord(absolute_token_position=2, byte_offset_in_token=0, byte_value=99, neural_logprob=-0.3),
                self.m.ByteLogprobRecord(absolute_token_position=4, byte_offset_in_token=0, byte_value=101, neural_logprob=-0.5),
            ],
            [
                self.m.ByteLogprobRecord(absolute_token_position=1, byte_offset_in_token=0, byte_value=98, neural_logprob=-0.2),
                self.m.ByteLogprobRecord(absolute_token_position=2, byte_offset_in_token=1, byte_value=100, neural_logprob=-0.4),
            ],
        ]

        merged = self.m.merge_shard_records(shards)
        self.assertEqual(
            [(r.absolute_token_position, r.byte_offset_in_token, r.byte_value) for r in merged],
            [(1, 0, 98), (2, 0, 99), (2, 1, 100), (4, 0, 101)],
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.m.merge_shard_records([[shards[0][0]], [shards[0][0]]])

        with self.assertRaisesRegex(ValueError, "out-of-order"):
            self.m.merge_shard_records([[shards[0][1], shards[0][0]]])

    def test_binary_shard_roundtrip_and_merge(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmpdir:
            tmp = Path(tmpdir)
            shard0 = [
                self.m.ByteLogprobRecord(absolute_token_position=2, byte_offset_in_token=0, byte_value=99, neural_logprob=-0.3),
                self.m.ByteLogprobRecord(absolute_token_position=4, byte_offset_in_token=0, byte_value=101, neural_logprob=-0.5),
            ]
            shard1 = [
                self.m.ByteLogprobRecord(absolute_token_position=1, byte_offset_in_token=0, byte_value=98, neural_logprob=-0.2),
                self.m.ByteLogprobRecord(absolute_token_position=2, byte_offset_in_token=1, byte_value=100, neural_logprob=-0.4),
            ]
            path0 = tmp / "rank0.npz"
            path1 = tmp / "rank1.npz"

            self.m.write_records_npz(path0, shard0)
            self.m.write_records_npz(path1, shard1)

            roundtrip = self.m.read_records_npz(path0)
            self.assertEqual(roundtrip, shard0)

            merged = self.m.merge_record_npz_shards([path0, path1])
            self.assertEqual(
                [(r.absolute_token_position, r.byte_offset_in_token, r.byte_value, r.neural_logprob) for r in merged],
                [(1, 0, 98, -0.2), (2, 0, 99, -0.3), (2, 1, 100, -0.4), (4, 0, 101, -0.5)],
            )

    def test_ppmd_stream_score_uses_neural_logprobs_and_updates_after_score(self):
        records = [
            self.m.ByteLogprobRecord(absolute_token_position=0, byte_offset_in_token=0, byte_value=ord("a"), neural_logprob=math.log(0.80)),
            self.m.ByteLogprobRecord(absolute_token_position=1, byte_offset_in_token=0, byte_value=ord("b"), neural_logprob=math.log(0.70)),
            self.m.ByteLogprobRecord(absolute_token_position=2, byte_offset_in_token=0, byte_value=ord("a"), neural_logprob=math.log(0.60)),
        ]
        model = self.m.PPMDByteModel(order=1)

        expected_ppm_probs = []
        expected_mix_probs = []
        expected_lambdas = []
        reference_model = self.m.PPMDByteModel(order=1)
        for record in records:
            ppmd_dist = reference_model.distribution()
            ppmd_prob = ppmd_dist[record.byte_value]
            lam = self.m.ppmd_prefix_lambda(
                reference_model,
                record.byte_value,
                base_lambda=0.25,
                lambda_hi=0.50,
                lambda_lo=0.10,
                conf_threshold=0.20,
            )
            nn_prob = math.exp(record.neural_logprob)
            expected_ppm_probs.append(ppmd_prob)
            expected_lambdas.append(lam)
            expected_mix_probs.append((1.0 - lam) * nn_prob + lam * ppmd_prob)
            reference_model.update(record.byte_value)

        summary = self.m.score_ppmd_stream(
            records,
            ppmd_order=1,
            ppmd_lambda=0.25,
            ppmd_lambda_hi=0.50,
            ppmd_lambda_lo=0.10,
            ppmd_conf_threshold=0.20,
            ppmd_confidence_gating=True,
        )

        self.assertEqual(summary.byte_count, 3)
        self.assertEqual(summary.ppmd_history, b"a")
        self.assertEqual(summary.lambdas, expected_lambdas)
        self.assertAlmostEqual(summary.ppm_nll, -sum(math.log(p) for p in expected_ppm_probs), places=14)
        self.assertAlmostEqual(summary.nn_nll, -sum(r.neural_logprob for r in records), places=14)
        self.assertAlmostEqual(summary.mix_nll, -sum(math.log(p) for p in expected_mix_probs), places=14)
        self.assertAlmostEqual(summary.mix_bpb, summary.mix_nll / math.log(2.0) / 3.0, places=14)
        self.assertAlmostEqual(summary.ppm_bpb, summary.ppm_nll / math.log(2.0) / 3.0, places=14)
        self.assertAlmostEqual(summary.nn_bpb, summary.nn_nll / math.log(2.0) / 3.0, places=14)

    def test_subset_byte_denominator_regression(self):
        constants = self.m.known_path_b_denominators()
        self.assertEqual(constants["full_validation_bytes"], 151078222)
        self.assertEqual(constants["first_8m_token_bytes"], 29365687)

        meta = self.m.build_output_schema_metadata(
            source_python_path=Path("results/exp_1876_ppmd/train_gpt_merged.py"),
            artifact_path=Path("results/exp_1876_ppmd/prod_8gpu_s42v2/final_model.int6.ptz"),
            artifact_size_bytes=None,
            subset_tokens=8000000,
            config=self.m.PathBEvalConfig(),
            mode="dry_run",
        )
        self.assertEqual(meta["denominator_constants"]["full_validation_bytes"], 151078222)
        self.assertEqual(meta["denominator_constants"]["first_8m_token_bytes"], 29365687)
        self.assertEqual(meta["subset_tokens"], 8000000)
        self.assertIn("token-trie marginalization", meta["normalizer_description"])
        self.assertIn("terminal mass exclusion", meta["normalizer_description"])

        fallback = self.m.byte_denominator_from_token_byte_lengths([3, 0, 2, 4], token_limit=3)
        self.assertEqual(fallback, 5)

    def test_eval_output_schema(self):
        config = self.m.PathBEvalConfig(subset_tokens=17, ppmd_order=5, ppmd_lambda=0.25)
        meta = self.m.build_output_schema_metadata(
            source_python_path=Path("/tmp/source.py"),
            artifact_path=Path("/tmp/artifact.ptz"),
            artifact_size_bytes=42,
            subset_tokens=config.subset_tokens,
            config=config,
            mode="dry_run",
        )

        required = {
            "path_b_version",
            "mode",
            "source_python_path",
            "artifact_path",
            "artifact_size_bytes",
            "subset_tokens",
            "denominator_constants",
            "normalizer_description",
            "distributed_merge_strategy",
            "future_runpod_command_suggestions",
            "ppmd",
            "mixture",
            "schema_version",
        }
        self.assertTrue(required.issubset(meta.keys()))
        self.assertEqual(meta["path_b_version"], self.m.PATH_B_VERSION)
        self.assertEqual(meta["ppmd"]["order"], 5)
        self.assertEqual(meta["mixture"]["ppmd_lambda"], 0.25)
        self.assertIn("absolute token position", meta["distributed_merge_strategy"])

        # The default config points at the real exp_1876 artifact; without torch
        # in this Python the executor must fail and the result must not claim BPB.
        result = self.m.run_explicit_eval(config)
        self.assertFalse(result.get("claim_ready", True))
        self.assertIsNone(result.get("mixture_bpb"))

    def test_explicit_eval_mode_rejects_unknown_eval_kind(self):
        with self.assertRaisesRegex(ValueError, "unknown eval_kind"):
            self.m.PathBEvalConfig(eval_kind="bogus")

        config = self.m.PathBEvalConfig(eval_kind="sliding")
        self.m.guard_explicit_eval_kind(config)

        ttt_config = self.m.PathBEvalConfig(eval_kind="ttt")
        with self.assertRaisesRegex(NotImplementedError, "TTT"):
            self.m.guard_explicit_eval_kind(ttt_config)

        parser = self.m.build_arg_parser()
        args = parser.parse_args(["--dry-run", "--eval-kind", "sliding"])
        self.assertEqual(self.m.config_from_args(args).eval_kind, "sliding")
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args(["--dry-run", "--eval-kind", "bogus"])


class PathBSlidingEvalHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_module()

    def test_plan_sliding_window_starts_matches_reference(self):
        # Mirror exp_1876 eval_val_sliding window planner for several configs.
        for total_tokens, seq_len, stride in [
            (200, 32, 8),
            (1000, 64, 16),
            (10_000, 128, 64),
        ]:
            context_size = seq_len - stride
            expected = [
                ws for ws in range(0, total_tokens, stride) if ws + context_size < total_tokens
            ]
            got = self.m.plan_sliding_window_starts(total_tokens, seq_len, stride)
            self.assertEqual(got, expected)

    def test_slice_window_starts_for_rank_partitions_evenly(self):
        windows = list(range(0, 1000, 8))
        all_assigned = []
        for rank in range(4):
            local = self.m.slice_window_starts_for_rank(windows, rank=rank, world_size=4)
            all_assigned.extend(local)
        self.assertEqual(all_assigned, windows)

    def test_rank_shard_filename_pattern(self):
        self.assertEqual(self.m.rank_shard_filename(0), "path_b_sliding_rank0.npz")
        self.assertEqual(self.m.rank_shard_filename(7), "path_b_sliding_rank7.npz")
        self.assertEqual(self.m.rank_accounting_filename(3), "path_b_sliding_accounting_rank3.json")

    def test_filter_records_by_subset_tokens(self):
        recs = [
            self.m.ByteLogprobRecord(absolute_token_position=p, byte_offset_in_token=0,
                                     byte_value=97, neural_logprob=-0.1)
            for p in (0, 5, 8, 12, 16)
        ]
        kept = self.m.filter_records_by_subset(recs, subset_tokens=10)
        self.assertEqual([r.absolute_token_position for r in kept], [0, 5, 8])
        self.assertEqual(self.m.filter_records_by_subset(recs, subset_tokens=None), recs)

    def test_emitted_token_byte_count_matches_eval_val_formula(self):
        # base_bytes_lut[t] + (has_leading_space[t] & ~is_boundary_token[prev])
        base = [3, 2, 0]
        has_space = [True, False, False]
        is_boundary = [True, False, True]
        # token 0 (with space), prev=1 (non-boundary): 3 + 1
        self.assertEqual(self.m.emitted_token_byte_count(0, 1, base, has_space, is_boundary), 4)
        # token 0 (with space), prev=0 (boundary by lut): 3 + 0
        self.assertEqual(self.m.emitted_token_byte_count(0, 0, base, has_space, is_boundary), 3)
        # zero-byte (special) token: 0 regardless
        self.assertEqual(self.m.emitted_token_byte_count(2, 1, base, has_space, is_boundary), 0)
        # prev=-1 (no previous): treat as boundary
        self.assertEqual(self.m.emitted_token_byte_count(0, -1, base, has_space, is_boundary), 3)

    def test_expected_denominator_for_eval(self):
        self.assertEqual(self.m.expected_denominator_for_eval(8_000_000, full_eval=False), 29_365_687)
        self.assertEqual(self.m.expected_denominator_for_eval(8_000_000, full_eval=True), 151_078_222)
        self.assertIsNone(self.m.expected_denominator_for_eval(7, full_eval=False))

    def test_build_merge_manifest_schema(self):
        entries = [
            {"rank": 0, "scored_tokens": 100, "scored_bytes": 350, "file_path": "x/rank0.npz"},
            {"rank": 1, "scored_tokens": 110, "scored_bytes": 380, "file_path": "x/rank1.npz", "sha256": "ab"},
        ]
        manifest = self.m.build_merge_manifest(entries, world_size=2)
        self.assertEqual(manifest["world_size"], 2)
        self.assertEqual(manifest["total_scored_tokens"], 210)
        self.assertEqual(manifest["total_scored_bytes"], 730)
        self.assertEqual(len(manifest["shards"]), 2)
        self.assertIn("schema_version", manifest)
        self.assertEqual(manifest["shards"][0]["rank"], 0)
        self.assertEqual(manifest["shards"][1].get("sha256"), "ab")

    def test_build_sliding_eval_result_claim_ready_true(self):
        config = self.m.PathBEvalConfig(eval_kind="sliding")

        class FakeSummary:
            byte_count = 100
            mix_bpb = 0.95
            ppm_bpb = 1.10
            nn_bpb = 1.05

        result = self.m.build_sliding_eval_result(
            config=config,
            source_module_path="/x/src.py",
            artifact_path="/x/art.ptz",
            artifact_size_bytes=42,
            rank=0,
            world_size=8,
            subset_tokens=8_000_000,
            full_eval=False,
            scored_token_count=8_000_000,
            scored_byte_count=29_365_687,
            zero_byte_token_count=12,
            runtime_seconds=600.5,
            summary=FakeSummary(),
            shard_manifest_path="/x/manifest.json",
            accounting_audit_path="/x/audit.json",
            warnings=[],
            error=None,
        )
        self.assertTrue(result["claim_ready"])
        self.assertEqual(result["expected_denominator"], 29_365_687)
        self.assertTrue(result["denominator_match"])
        self.assertEqual(result["mixture_bpb"], 0.95)
        self.assertEqual(result["neural_only_bpb"], 1.05)
        self.assertEqual(result["ppm_d_only_bpb"], 1.10)
        self.assertEqual(result["eval_kind"], "sliding")
        for required_field in (
            "schema_version", "path_b_version", "source_module_path", "artifact_path",
            "artifact_size_bytes", "eval_kind", "rank", "world_size", "subset_tokens",
            "scored_token_count", "scored_byte_count", "expected_denominator",
            "denominator_match", "denominator_formula", "zero_byte_token_count",
            "ppm_d_config", "lambda_gating_config", "shard_manifest_path",
            "accounting_audit_path", "runtime_seconds", "neural_only_bpb",
            "ppm_d_only_bpb", "mixture_bpb", "warnings", "claim_ready",
        ):
            self.assertIn(required_field, result, msg=required_field)

    def test_build_sliding_eval_result_claim_ready_false_on_denom_mismatch(self):
        config = self.m.PathBEvalConfig(eval_kind="sliding")

        class FakeSummary:
            byte_count = 100
            mix_bpb = 0.95
            ppm_bpb = 1.10
            nn_bpb = 1.05

        result = self.m.build_sliding_eval_result(
            config=config,
            source_module_path="/x/src.py",
            artifact_path="/x/art.ptz",
            artifact_size_bytes=42,
            rank=0,
            world_size=8,
            subset_tokens=8_000_000,
            full_eval=False,
            scored_token_count=8_000_000,
            scored_byte_count=29_000_000,  # mismatch
            zero_byte_token_count=12,
            runtime_seconds=600.5,
            summary=FakeSummary(),
            shard_manifest_path="/x/manifest.json",
            accounting_audit_path="/x/audit.json",
            warnings=[],
            error=None,
        )
        self.assertFalse(result["claim_ready"])
        self.assertFalse(result["denominator_match"])
        self.assertIsNone(result["mixture_bpb"])
        self.assertIsNone(result["neural_only_bpb"])
        self.assertIsNone(result["ppm_d_only_bpb"])

    def test_build_sliding_eval_result_claim_ready_false_on_error(self):
        config = self.m.PathBEvalConfig(eval_kind="sliding")
        result = self.m.build_sliding_eval_result(
            config=config,
            source_module_path="/x/src.py",
            artifact_path="/x/art.ptz",
            artifact_size_bytes=42,
            rank=0,
            world_size=1,
            subset_tokens=8_000_000,
            full_eval=False,
            scored_token_count=0,
            scored_byte_count=0,
            zero_byte_token_count=0,
            runtime_seconds=1.0,
            summary=None,
            shard_manifest_path=None,
            accounting_audit_path=None,
            warnings=["executor failed"],
            error="boom",
        )
        self.assertFalse(result["claim_ready"])
        self.assertIsNone(result["mixture_bpb"])
        self.assertEqual(result["error"], "boom")

    def test_run_explicit_eval_ttt_remains_blocked(self):
        cfg = self.m.PathBEvalConfig(eval_kind="ttt")
        with self.assertRaises(NotImplementedError):
            self.m.run_explicit_eval(cfg)

    def test_run_explicit_eval_sliding_writes_failed_metrics_when_executor_raises(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "src.py"
            artifact = tmp / "art.ptz"
            output = tmp / "result.json"
            source.write_text("# placeholder\n", encoding="utf-8")
            artifact.write_bytes(b"placeholder")

            cfg = self.m.PathBEvalConfig(
                source_python_path=source,
                artifact_path=artifact,
                output_json_path=output,
                eval_kind="sliding",
            )

            def stub_executor(config, *, output_json_path):
                raise RuntimeError("simulated executor failure")

            payload = self.m.run_explicit_eval(cfg, sliding_executor=stub_executor)
            self.assertFalse(payload["claim_ready"])
            self.assertIsNone(payload["mixture_bpb"])
            self.assertIn("simulated executor failure", payload.get("error", ""))
            self.assertTrue(output.exists())
            on_disk = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(on_disk["claim_ready"])

    def test_default_main_does_not_run_sliding_eval(self):
        # Sanity: dry-run is the default behavior; --eval flag is required.
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "src.py"
            artifact = tmp / "art.ptz"
            output = tmp / "plan.json"
            source.write_text("# placeholder\n", encoding="utf-8")
            artifact.write_bytes(b"placeholder")

            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--source-python", str(source),
                    "--artifact-path", str(artifact),
                    "--output-json", str(output),
                ],
                check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["mode"], "dry_run")


if __name__ == "__main__":
    unittest.main()