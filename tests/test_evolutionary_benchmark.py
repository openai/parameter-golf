from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import random

import numpy as np
import torch

from tools.evolutionary_benchmark import (
    _approx_quantile_threshold_from_tensors,
    approximate_state_distance,
    compute_committee_signals,
    EvoBenchmarkConfig,
    crossover_state,
    mutate_state,
    overlap_crossover,
    prepare_tokenized_enwik8_splits,
    recommend_committee_action,
    resolve_device,
    resolve_param_dtype,
    run_committee_adaptive,
    run_committee_compressibility,
    run_committee_schedule,
    run_crossover_viability,
    run_evolutionary_loop,
    run_recipe_evolution,
    run_vmap_throughput,
    select_distinct_parent_indices,
    tournament_select_index,
)


class EvolutionaryBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = resolve_device("cpu")
        self.param_dtype = resolve_param_dtype("fp32", self.device)
        self.cfg = EvoBenchmarkConfig(
            device="cpu",
            dtype="fp32",
            seed=123,
            model_dim=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            vocab_size=256,
            seq_len=16,
            stride=8,
            batch_size=2,
            base_lr=1e-3,
        )

    def _write_enwik8_fixture(self) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "enwik8"
        data = np.arange(8192, dtype=np.uint8)
        path.write_bytes(data.tobytes())
        return path

    def _write_text_fixture(self) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "enwik8"
        text = ("The quick brown fox jumps over the lazy dog. " * 400).encode("utf-8")
        path.write_bytes(text)
        return path

    def test_overlap_crossover_respects_overlap_zone(self) -> None:
        parent_a = {"w": torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)}
        parent_b = {"w": torch.tensor([0.0, 0.0, 10.0, 10.0], dtype=torch.float32)}

        child, stats = overlap_crossover(parent_a, parent_b, percentile=50.0, seed=7)

        self.assertAlmostEqual(stats["threshold"], 5.0, places=4)
        self.assertTrue(torch.equal(child["w"][:2], torch.tensor([0.0, 0.0], dtype=torch.float32)))
        self.assertTrue(bool(((child["w"][2:] == 0.0) | (child["w"][2:] == 10.0)).all().item()))
        self.assertAlmostEqual(stats["overlap_fraction"], 0.5, places=4)

    def test_delta_overlap_crossover_recombines_around_base(self) -> None:
        base = {"w": torch.tensor([5.0, 5.0, 5.0, 5.0], dtype=torch.float32)}
        parent_a = {"w": torch.tensor([6.0, 4.0, 8.0, 2.0], dtype=torch.float32)}
        parent_b = {"w": torch.tensor([4.0, 6.0, 2.0, 8.0], dtype=torch.float32)}

        child, stats = crossover_state(
            parent_a,
            parent_b,
            strategy="delta_overlap",
            percentile=50.0,
            seed=11,
            base_state=base,
        )

        self.assertIn("threshold", stats)
        self.assertEqual(tuple(child["w"].shape), (4,))
        self.assertTrue(bool((child["w"] >= 2.0).all().item()))
        self.assertTrue(bool((child["w"] <= 8.0).all().item()))

    def test_sign_consensus_averages_when_parents_agree_on_direction(self) -> None:
        parent_a = {"w": torch.tensor([2.0, -4.0, 3.0, -1.0], dtype=torch.float32)}
        parent_b = {"w": torch.tensor([6.0, -2.0, -7.0, 9.0], dtype=torch.float32)}

        child, stats = crossover_state(parent_a, parent_b, strategy="sign_consensus", percentile=50.0, seed=19)

        self.assertEqual(child["w"][0].item(), 4.0)
        self.assertEqual(child["w"][1].item(), -3.0)
        self.assertIn(child["w"][2].item(), {3.0, -7.0})
        self.assertIn(child["w"][3].item(), {-1.0, 9.0})
        self.assertAlmostEqual(stats["overlap_fraction"], 0.5, places=4)

    def test_delta_importance_prefers_larger_delta_magnitude_outside_overlap(self) -> None:
        base = {"w": torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)}
        parent_a = {"w": torch.tensor([1.0, 0.5, 4.0, 0.25], dtype=torch.float32)}
        parent_b = {"w": torch.tensor([3.0, 0.25, 1.0, 0.50], dtype=torch.float32)}

        child, stats = crossover_state(
            parent_a,
            parent_b,
            strategy="delta_importance",
            percentile=0.0,
            seed=23,
            base_state=base,
        )

        self.assertAlmostEqual(stats["threshold"], 0.25, places=6)
        self.assertTrue(torch.equal(child["w"], torch.tensor([3.0, 0.375, 4.0, 0.375], dtype=torch.float32)))

    def test_sampled_quantile_threshold_tracks_large_distribution(self) -> None:
        values = [
            torch.linspace(0.0, 1.0, 600, dtype=torch.float32),
            torch.linspace(1.0, 3.0, 600, dtype=torch.float32),
        ]
        exact = torch.quantile(torch.cat(values), 0.5).item()
        approx = _approx_quantile_threshold_from_tensors(values, 50.0, max_samples=100)

        self.assertAlmostEqual(approx, exact, delta=0.08)

    def test_delta_sparse_union_keeps_only_strong_parent_patches(self) -> None:
        base = {"w": torch.zeros((6,), dtype=torch.float32)}
        parent_a = {"w": torch.tensor([10.0, 9.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)}
        parent_b = {"w": torch.tensor([8.0, 0.0, 7.0, 0.0, 0.0, 0.0], dtype=torch.float32)}

        child, stats = crossover_state(
            parent_a,
            parent_b,
            strategy="delta_sparse_union",
            percentile=75.0,
            seed=31,
            base_state=base,
        )

        self.assertAlmostEqual(child["w"][0].item(), 9.0, places=6)
        self.assertAlmostEqual(child["w"][1].item(), 9.0, places=6)
        self.assertAlmostEqual(child["w"][2].item(), 7.0, places=6)
        self.assertTrue(torch.equal(child["w"][3:], torch.zeros((3,), dtype=torch.float32)))
        self.assertGreater(stats["overlap_fraction"], 0.0)

    def test_delta_sparse_consensus_requires_shared_strong_agreement(self) -> None:
        base = {"w": torch.zeros((6,), dtype=torch.float32)}
        parent_a = {"w": torch.tensor([10.0, 9.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)}
        parent_b = {"w": torch.tensor([8.0, 0.0, 7.0, 0.0, 0.0, 0.0], dtype=torch.float32)}

        child, stats = crossover_state(
            parent_a,
            parent_b,
            strategy="delta_sparse_consensus",
            percentile=75.0,
            seed=37,
            base_state=base,
        )

        self.assertAlmostEqual(child["w"][0].item(), 9.0, places=6)
        self.assertTrue(torch.equal(child["w"][1:], torch.zeros((5,), dtype=torch.float32)))
        self.assertGreater(stats["overlap_fraction"], 0.0)

    def test_layer_swap_keeps_whole_groups_together(self) -> None:
        parent_a = {
            "blocks.0.attn.c_q.weight": torch.ones((2, 2), dtype=torch.float32),
            "blocks.0.attn.c_k.weight": torch.ones((2, 2), dtype=torch.float32),
            "blocks.1.attn.c_q.weight": torch.full((2, 2), 2.0, dtype=torch.float32),
        }
        parent_b = {
            "blocks.0.attn.c_q.weight": torch.full((2, 2), 10.0, dtype=torch.float32),
            "blocks.0.attn.c_k.weight": torch.full((2, 2), 10.0, dtype=torch.float32),
            "blocks.1.attn.c_q.weight": torch.full((2, 2), 20.0, dtype=torch.float32),
        }

        child, _ = crossover_state(parent_a, parent_b, strategy="layer_swap", percentile=50.0, seed=5)

        self.assertTrue(torch.equal(child["blocks.0.attn.c_q.weight"], child["blocks.0.attn.c_k.weight"]))

    def test_mutate_state_respects_zero_fraction_and_changes_with_noise(self) -> None:
        state = {"w": torch.ones((16,), dtype=torch.float32)}

        unchanged = mutate_state(state, mutation_std=1e-2, mutation_fraction=0.0, seed=7)
        changed = mutate_state(state, mutation_std=1e-2, mutation_fraction=1.0, seed=7)

        self.assertTrue(torch.equal(unchanged["w"], state["w"]))
        self.assertFalse(torch.equal(changed["w"], state["w"]))

    def test_tournament_helpers_pick_best_and_force_distinct_parents(self) -> None:
        scores = [3.0, 2.0, 1.0, 0.0]
        rng = random.Random(5)

        best_idx = tournament_select_index(scores, tournament_size=4, rng=rng)
        pair = select_distinct_parent_indices(scores, tournament_size=4, rng=random.Random(5))

        self.assertEqual(best_idx, 3)
        self.assertEqual(pair[0], 3)
        self.assertNotEqual(pair[0], pair[1])

    def test_vmap_throughput_smoke(self) -> None:
        result = run_vmap_throughput(
            cfg=self.cfg,
            scales=(2, 4),
            noise_std=1e-3,
            warmup_repeats=0,
            timed_repeats=1,
            population_chunk_size=None,
            device=self.device,
            param_dtype=self.param_dtype,
        )

        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["status"], "ok")
        self.assertGreater(float(result["results"][0]["models_per_second"]), 0.0)

    def test_vmap_throughput_chunking_reports_num_chunks(self) -> None:
        result = run_vmap_throughput(
            cfg=self.cfg,
            scales=(5,),
            noise_std=1e-3,
            warmup_repeats=0,
            timed_repeats=1,
            population_chunk_size=2,
            device=self.device,
            param_dtype=self.param_dtype,
        )

        row = result["results"][0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["population_chunk_size"], 2)
        self.assertEqual(row["num_chunks"], 3)

    def test_crossover_viability_smoke(self) -> None:
        enwik8_path = self._write_enwik8_fixture()
        result = run_crossover_viability(
            cfg=self.cfg,
            enwik8_path=enwik8_path,
            copies=3,
            train_seconds=0.01,
            strategies=("weight_overlap", "delta_overlap", "layer_swap"),
            percentiles=(50.0,),
            eval_batches=2,
            pair_limit=2,
            ensemble_topks=(2,),
            member_train_mode="parallel_vmap",
            device=self.device,
            param_dtype=self.param_dtype,
        )

        self.assertEqual(len(result["members"]), 3)
        self.assertEqual(result["member_train_mode"], "parallel_vmap")
        self.assertEqual(result["member_ensemble_results"][0]["topk"], 2)
        trials = result["crossover"]["weight_overlap"]["50"]["trials"]
        self.assertEqual(len(trials), 2)
        self.assertIn("offspring_bpb", trials[0])
        self.assertIn("na", result["crossover"]["layer_swap"])

    def test_evolution_loop_smoke(self) -> None:
        enwik8_path = self._write_enwik8_fixture()
        result = run_evolutionary_loop(
            cfg=self.cfg,
            enwik8_path=enwik8_path,
            base_train_seconds=0.01,
            generations=2,
            population_size=4,
            tournament_size=2,
            crossover_strategy="delta_overlap",
            crossover_percentile=50.0,
            mutation_std=1e-3,
            mutation_fraction=0.1,
            eval_batches=2,
            ensemble_topks=(),
            device=self.device,
            param_dtype=self.param_dtype,
        )

        history = result["evolution"]["history"]
        self.assertEqual(len(history), 2)
        self.assertIn("best_bpb", history[0])
        self.assertEqual(result["evolution"]["crossover_strategy"], "delta_overlap")

    def test_evolution_loop_parent_copy_smoke(self) -> None:
        enwik8_path = self._write_enwik8_fixture()
        result = run_evolutionary_loop(
            cfg=self.cfg,
            enwik8_path=enwik8_path,
            base_train_seconds=0.01,
            generations=2,
            population_size=4,
            tournament_size=2,
            crossover_strategy="parent_copy",
            crossover_percentile=50.0,
            mutation_std=1e-3,
            mutation_fraction=0.1,
            eval_batches=2,
            ensemble_topks=(2,),
            device=self.device,
            param_dtype=self.param_dtype,
        )

        history = result["evolution"]["history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(result["evolution"]["crossover_strategy"], "parent_copy")
        self.assertEqual(result["evolution"]["ensemble_results"][0]["topk"], 2)

    def test_committee_schedule_smoke(self) -> None:
        enwik8_path = self._write_enwik8_fixture()
        result = run_committee_schedule(
            cfg=self.cfg,
            enwik8_path=enwik8_path,
            stage_copies=(2, 4),
            stage_train_seconds=(0.01, 0.005),
            ensemble_topks=(2, 4),
            eval_batches=2,
            spawn_noise_std=0.0,
            device=self.device,
            param_dtype=self.param_dtype,
        )

        self.assertEqual(result["member_train_mode"], "parallel_vmap_staged")
        self.assertEqual(result["committee_schedule"]["stage_signature"], "2x0.01->4x0.005")
        self.assertEqual(result["dataset"]["residency"], "cpu")
        self.assertEqual(len(result["members"]), 4)
        self.assertEqual(len(result["members"][0]["stages"]), 2)
        self.assertIn("archive_summary", result["members"][0])
        self.assertIn("replay_summary", result["members"][0])
        self.assertIsNotNone(result["members"][0]["stages"][1]["pre_stage_replay_eval"])
        self.assertEqual(result["member_ensemble_results"][-1]["topk"], 4)

    def test_committee_schedule_supports_prune_stages(self) -> None:
        enwik8_path = self._write_enwik8_fixture()
        result = run_committee_schedule(
            cfg=self.cfg,
            enwik8_path=enwik8_path,
            stage_copies=(2, 4, 2),
            stage_train_seconds=(0.01, 0.005, 0.005),
            ensemble_topks=(2,),
            eval_batches=2,
            spawn_noise_std=0.0,
            device=self.device,
            param_dtype=self.param_dtype,
        )

        self.assertEqual(result["committee_schedule"]["stage_signature"], "2x0.01->4x0.005->2x0.005")
        self.assertEqual(len(result["members"]), 2)
        self.assertEqual(len(result["members"][0]["stages"]), 3)
        self.assertTrue(any("prune2" == step for step in result["members"][0]["lineage"] if isinstance(step, str)))
        self.assertIn("best_stage_index", result["members"][0]["archive_summary"])

    def test_committee_signals_capture_branch_distance_and_widen_readiness(self) -> None:
        record_a = {
            "state": {"w": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)},
            "current_eval": {"bpb": 1.30},
            "replay_eval": {"bpb": 1.42},
            "stages": [{"archive_improved": True}],
        }
        record_b = {
            "state": {"w": torch.tensor([1.2, 0.8, 1.4], dtype=torch.float32)},
            "current_eval": {"bpb": 1.34},
            "replay_eval": {"bpb": 1.58},
            "stages": [{"archive_improved": False}],
        }
        signals = compute_committee_signals(
            records=[record_a, record_b],
            member_ensemble_results=[
                {"topk": 2, "val_bpb": 1.18, "test_bpb": 1.19},
            ],
            previous_best_member_bpb=1.36,
        )

        self.assertGreater(signals["pairwise_distance_mean"], 0.0)
        self.assertGreater(signals["pairwise_relative_distance_mean"], 0.0)
        self.assertAlmostEqual(signals["depth_gain"], 0.06, places=6)

        decision = recommend_committee_action(
            signals=signals,
            current_copies=2,
            min_copies=2,
            max_copies=8,
            widen_factor=2,
            narrow_factor=2,
            depth_gain_threshold=0.01,
            breadth_gain_threshold=0.05,
            archive_hit_rate_threshold=0.25,
            winner_concentration_threshold=0.85,
            replay_disagreement_threshold=0.02,
            pairwise_distance_threshold=0.01,
        )
        self.assertEqual(decision["action"], "widen")

    def test_committee_signals_block_widen_when_population_converged(self) -> None:
        near_identical_state = {"w": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)}
        signals = compute_committee_signals(
            records=[
                {
                    "state": near_identical_state,
                    "current_eval": {"bpb": 1.30},
                    "replay_eval": {"bpb": 1.48},
                    "stages": [{"archive_improved": False}],
                },
                {
                    "state": {"w": torch.tensor([1.00001, 1.0, 0.99999], dtype=torch.float32)},
                    "current_eval": {"bpb": 1.32},
                    "replay_eval": {"bpb": 1.54},
                    "stages": [{"archive_improved": False}],
                },
            ],
            member_ensemble_results=[
                {"topk": 2, "val_bpb": 1.20, "test_bpb": 1.21},
            ],
            previous_best_member_bpb=1.40,
        )
        decision = recommend_committee_action(
            signals=signals,
            current_copies=2,
            min_copies=2,
            max_copies=8,
            widen_factor=2,
            narrow_factor=2,
            depth_gain_threshold=0.20,
            breadth_gain_threshold=0.05,
            archive_hit_rate_threshold=0.25,
            winner_concentration_threshold=0.85,
            replay_disagreement_threshold=0.01,
            pairwise_distance_threshold=0.01,
        )
        self.assertEqual(decision["action"], "hold")
        self.assertEqual(decision["reason"], "ensemble_gain_but_population_converged")

    def test_committee_adaptive_smoke(self) -> None:
        enwik8_path = self._write_enwik8_fixture()
        result = run_committee_adaptive(
            cfg=self.cfg,
            enwik8_path=enwik8_path,
            initial_copies=2,
            round_train_seconds=(0.01, 0.005),
            ensemble_topks=(2, 4),
            eval_batches=2,
            spawn_noise_std=0.0,
            min_copies=2,
            max_copies=4,
            widen_factor=2,
            narrow_factor=2,
            depth_gain_threshold=0.01,
            breadth_gain_threshold=0.02,
            archive_hit_rate_threshold=0.25,
            winner_concentration_threshold=0.85,
            replay_disagreement_threshold=0.01,
            pairwise_distance_threshold=0.001,
            device=self.device,
            param_dtype=self.param_dtype,
        )

        self.assertEqual(result["member_train_mode"], "parallel_vmap_adaptive")
        self.assertEqual(result["committee_adaptive"]["round_signature"], "0.01s->0.005s")
        self.assertEqual(len(result["rounds"]), 2)
        self.assertIn("signals", result["rounds"][0])
        self.assertIn("decision", result["rounds"][0])

    def test_sentencepiece_tokenization_smoke(self) -> None:
        tokenizer_path = Path("data/tokenizers/fineweb_1024_bpe.model")
        if not tokenizer_path.exists():
            self.skipTest("sentencepiece tokenizer asset is not present")
        text_path = self._write_text_fixture()
        train_tokens, val_tokens, test_tokens, dataset_meta = prepare_tokenized_enwik8_splits(
            text_path,
            device=self.device,
            cache_on_device=False,
            tokenization_mode="sentencepiece",
            tokenizer_name="sp_bpe_1024",
            tokenizer_model_path=str(tokenizer_path),
        )

        self.assertGreater(train_tokens.numel(), 0)
        self.assertGreater(val_tokens.numel(), 0)
        self.assertGreater(test_tokens.numel(), 0)
        self.assertEqual(dataset_meta["tokenization_mode"], "sentencepiece")
        self.assertEqual(dataset_meta["tokenizer_vocab_size"], 1024)

    def test_recipe_evolution_smoke(self) -> None:
        tokenizer_path = Path("data/tokenizers/fineweb_1024_bpe.model")
        if not tokenizer_path.exists():
            self.skipTest("sentencepiece tokenizer asset is not present")
        enwik8_path = self._write_text_fixture()
        result = run_recipe_evolution(
            cfg=self.cfg,
            enwik8_path=enwik8_path,
            population_size=4,
            generations=2,
            tournament_size=2,
            train_seconds=0.01,
            eval_batches=2,
            mutation_rate=0.2,
            artifact_limit_mb=16.0,
            recipe_profile="compact",
            confirm_topk=1,
            confirm_train_seconds=0.005,
            device=self.device,
            param_dtype=self.param_dtype,
        )

        self.assertEqual(result["recipe_evolution"]["recipe_profile"], "compact")
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(len(result["population"]), 4)
        self.assertIsNotNone(result["best"])
        self.assertLessEqual(result["best"]["artifact_param_mb"], 16.0)
        self.assertEqual(result["confirm_results"][0]["rank_from_short_fitness"], 0)

    def test_committee_compressibility_smoke(self) -> None:
        enwik8_path = self._write_enwik8_fixture()
        result = run_committee_compressibility(
            cfg=self.cfg,
            enwik8_path=enwik8_path,
            stage_copies=(2, 4),
            stage_train_seconds=(0.01, 0.005),
            ensemble_topks=(2, 4),
            eval_batches=2,
            spawn_noise_std=0.0,
            artifact_limit_mb=16.0,
            delta_budget_fractions=(0.5, 1.0),
            basis_ranks=(1, 2),
            analysis_topk=16,
            device=self.device,
            param_dtype=self.param_dtype,
        )

        self.assertEqual(result["committee_compressibility"]["member_count"], 4)
        self.assertGreater(len(result["compression_trials"]), 0)
        self.assertGreater(len(result["best_by_budget"]), 0)
        self.assertIn("pairwise_l2_mean", result["delta_analysis"])
        self.assertLessEqual(result["committee_compressibility"]["base_artifact_mb"], 16.0)

    def test_default_vocab_size_matches_byte_level_enwik8(self) -> None:
        self.assertEqual(EvoBenchmarkConfig.vocab_size, 256)


if __name__ == "__main__":
    unittest.main()
