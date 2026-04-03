import random
import unittest

from tools.evolutionary_benchmark import (
    DeepFloorGenome,
    DeepFloorGeneSpace,
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
        self.assertTrue(len(space.norm_interval_ks) > 0)
        self.assertTrue(len(space.jacobian_lambdas) > 0)
        self.assertTrue(len(space.stochastic_round_ps) > 0)

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


if __name__ == "__main__":
    unittest.main()
