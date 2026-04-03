import random
import unittest

from tools.evolutionary_benchmark import (
    DeepFloorGenome,
    DeepFloorGeneSpace,
    canonicalize_deepfloor_genome,
    crossover_deepfloor_genomes,
    default_deepfloor_gene_space,
    deepfloor_genome_to_v3_config,
    mutate_deepfloor_genome,
    random_deepfloor_genome,
)
from spectral_flood_walk_v3 import V3Config


class DeepFloorEvolutionTests(unittest.TestCase):
    def test_random_genome_produces_valid_v3_config(self) -> None:
        space = default_deepfloor_gene_space("compact")
        rng = random.Random(42)
        genome = random_deepfloor_genome(space, rng=rng)
        cfg = deepfloor_genome_to_v3_config(genome)
        self.assertIsInstance(cfg, V3Config)
        self.assertGreater(cfg.recurrent_dim, 0)

    def test_crossover_produces_valid_genome(self) -> None:
        space = default_deepfloor_gene_space("compact")
        rng = random.Random(42)
        left = random_deepfloor_genome(space, rng=rng)
        right = random_deepfloor_genome(space, rng=rng)
        child = crossover_deepfloor_genomes(left, right, rng=rng)
        self.assertIsInstance(child, DeepFloorGenome)

    def test_mutation_can_change_genes(self) -> None:
        space = default_deepfloor_gene_space("compact")
        rng = random.Random(42)
        genome = random_deepfloor_genome(space, rng=rng)
        mutated = mutate_deepfloor_genome(genome, space, mutation_rate=1.0, rng=rng)
        self.assertNotEqual(genome, mutated)

    def test_gene_space_compact_has_stability_genes(self) -> None:
        space = default_deepfloor_gene_space("compact")
        self.assertTrue(len(space.contraction_targets) > 0)
        self.assertTrue(len(space.accumulator_decays) > 0)
        self.assertTrue(len(space.state_cores) > 0)
        self.assertTrue(len(space.hippo_delta_scales) > 0)
        self.assertTrue(len(space.hippo_ranks) > 0)
        self.assertTrue(len(space.norm_interval_ks) > 0)
        self.assertTrue(len(space.jacobian_lambdas) > 0)
        self.assertTrue(len(space.stochastic_round_ps) > 0)

    def test_floor_mode_canonicalizes_fused_only_state_core_genes(self) -> None:
        genome = DeepFloorGenome(
            recurrent_dim=32,
            num_distinct_blocks=1,
            view_count=1,
            view_combination="average",
            cross_token_mode="floor",
            block_has_residual=True,
            block_nonlinearity="gelu",
            recurrence_step_size=1.0,
            state_decay=1.0,
            contraction_target=0.99,
            train_recurrence_steps=8,
            eval_recurrence_steps=16,
            norm_interval_k=4,
            floor_min_interval=2,
            floor_max_interval=8,
            floor_threshold=0.05,
            kernel_feature_map="elu_plus_1",
            accumulator_decay=0.99,
            state_core="hippo_plus_lowrank",
            hippo_delta_scale=0.2,
            hippo_rank=4,
            quantization="ternary",
            jacobian_lambda=0.0,
            stochastic_round_p=0.0,
            base_lr=1e-3,
            weight_decay=0.0,
            seq_len=16,
            batch_size=2,
        )
        canonical = canonicalize_deepfloor_genome(genome)
        self.assertEqual(canonical.state_core, "scalar_decay")
        self.assertEqual(canonical.hippo_delta_scale, 0.0)
        self.assertEqual(canonical.hippo_rank, 1)

    def test_fused_mode_preserves_hippo_genes(self) -> None:
        genome = DeepFloorGenome(
            recurrent_dim=32,
            num_distinct_blocks=1,
            view_count=1,
            view_combination="average",
            cross_token_mode="fused",
            block_has_residual=True,
            block_nonlinearity="gelu",
            recurrence_step_size=1.0,
            state_decay=1.0,
            contraction_target=0.99,
            train_recurrence_steps=8,
            eval_recurrence_steps=16,
            norm_interval_k=4,
            floor_min_interval=2,
            floor_max_interval=8,
            floor_threshold=0.05,
            kernel_feature_map="elu_plus_1",
            accumulator_decay=0.99,
            state_core="hippo_plus_lowrank",
            hippo_delta_scale=0.2,
            hippo_rank=4,
            quantization="ternary",
            jacobian_lambda=0.0,
            stochastic_round_p=0.0,
            base_lr=1e-3,
            weight_decay=0.0,
            seq_len=16,
            batch_size=2,
        )
        cfg = deepfloor_genome_to_v3_config(genome)
        self.assertEqual(cfg.state_core, "hippo_plus_lowrank")
        self.assertEqual(cfg.hippo_delta_scale, 0.2)
        self.assertEqual(cfg.hippo_rank, 4)

    def test_evaluate_deepfloor_genome_returns_bpb(self) -> None:
        import tempfile
        from pathlib import Path
        import numpy as np
        import torch
        from tools.evolutionary_benchmark import evaluate_deepfloor_genome

        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "enwik8"
        data = np.arange(16384, dtype=np.uint8)
        path.write_bytes(data.tobytes())

        space = default_deepfloor_gene_space("compact")
        rng = random.Random(42)
        genome = random_deepfloor_genome(space, rng=rng)
        result = evaluate_deepfloor_genome(
            genome=genome,
            enwik8_path=path,
            train_steps=2,
            eval_batches=2,
            seed=42,
            device=torch.device("cpu"),
        )
        self.assertIn("val", result)
        self.assertIn("bpb", result["val"])
        self.assertGreater(result["val"]["bpb"], 0.0)


    def test_confirm_results_rerank_best(self) -> None:
        from tools.evolutionary_benchmark import run_deepfloor_recipe_evolution
        import tempfile
        from pathlib import Path
        import numpy as np
        import torch

        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "enwik8"
        data = np.arange(16384, dtype=np.uint8)
        path.write_bytes(data.tobytes())

        result = run_deepfloor_recipe_evolution(
            enwik8_path=path,
            population_size=3,
            generations=1,
            tournament_size=2,
            train_steps=2,
            eval_batches=2,
            mutation_rate=0.5,
            artifact_limit_mb=16.0,
            deepfloor_profile="compact",
            confirm_topk=2,
            confirm_train_steps=4,
            seed=42,
            device=torch.device("cpu"),
        )
        # best should come from confirm_results when confirmations ran
        self.assertIsNotNone(result["best"])
        if result["confirm_results"]:
            confirm_bpbs = [float(r["val"]["bpb"]) for r in result["confirm_results"]]
            best_bpb = float(result["best"]["val"]["bpb"])
            self.assertAlmostEqual(best_bpb, min(confirm_bpbs))


if __name__ == "__main__":
    unittest.main()
