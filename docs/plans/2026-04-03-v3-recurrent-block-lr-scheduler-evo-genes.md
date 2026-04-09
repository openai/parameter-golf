# V3 Recurrent Block, LR Scheduler, and Evolutionary Genes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the recurrent block to match the design doc's QKV+O budget, add a cosine LR scheduler, and thread the DeepFloor v3 stability genes into the evolutionary benchmark so recipe-evolution can search the v3 design space.

**Architecture:** Three independent changes. (1) Replace the gated MLP recurrent block with a QKV+O per-token attention block (~12K params at d=64 instead of ~20K). (2) Add cosine decay with warmup to the training loop. (3) Add a `DeepFloorGenome` dataclass, gene space, and a `deepfloor-recipe-evolution` subcommand to `tools/evolutionary_benchmark.py`.

**Tech Stack:** PyTorch, Python 3.14, unittest, argparse

---

### Task 1: Replace gated MLP recurrent block with QKV+O

**Files:**
- Modify: `spectral_flood_walk_v3.py:173-227` (DeepFloorRecurrentBlock)
- Modify: `tests/test_spectral_flood_walk_v3.py`

**Step 1: Write the failing test**

Add to `tests/test_spectral_flood_walk_v3.py`:

```python
def test_recurrent_block_param_count_matches_qkvo(self) -> None:
    cfg = self._base_config(cross_token_mode="floor")
    model = DeepFloorModel(cfg)
    block = model.blocks[0]
    # QKV+O at d=32: 4 × 32 × 32 = 4096 weight params + norm
    weight_params = sum(p.numel() for p in block.parameters() if p.dim() >= 2)
    self.assertEqual(weight_params, 4 * cfg.recurrent_dim * cfg.recurrent_dim)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_spectral_flood_walk_v3.SpectralFloodWalkV3Tests.test_recurrent_block_param_count_matches_qkvo -v`
Expected: FAIL (currently ~20K weight params from gated MLP, not 4×d²)

**Step 3: Rewrite DeepFloorRecurrentBlock**

Replace the `gate_proj`, `up_proj`, `down_proj` with `q_proj`, `k_proj`, `v_proj`, `o_proj` — all `nn.Linear(dim, dim, bias=False)`. The forward pass becomes:

```python
def forward(self, state, *, runtime=None, apply_quant_noise=True):
    normed = self.norm(state)
    runtime = runtime or self.prepare_runtime()
    q = F.linear(normed, runtime["q_weight"])
    k = F.linear(normed, runtime["k_weight"])
    v = F.linear(normed, runtime["v_weight"])
    # Per-token: q, k, v are all (batch, seq, dim)
    # No cross-token attention here — just per-token gated transform
    gate = torch.sigmoid(q)
    candidate = activation_fn(self.nonlinearity, k) * v
    update = F.linear(gate * candidate, runtime["o_weight"])
    if apply_quant_noise:
        update = stochastic_round_tensor(update, quantization=self.quantization, probability=self.stochastic_round_p)
    if self.has_residual:
        return self.state_decay * state + self.step_size * update
    return self.step_size * update
```

Update `prepare_runtime()` to clamp all four weights with `clamp_spectral_gain` using `contraction_target ** 0.25` (fourth root since four matrices compose).

Update `jacobian_proxy_penalty` to use the new forward signature (no change needed if it just calls `self.forward()`).

**Step 4: Run all v3 tests to verify they pass**

Run: `python3 -m unittest tests.test_spectral_flood_walk_v3 -v`
Expected: All tests PASS

**Step 5: Commit**

```
git add spectral_flood_walk_v3.py tests/test_spectral_flood_walk_v3.py
git commit -m "refactor: replace gated MLP recurrent block with QKV+O (4×d² params)"
```

---

### Task 2: Add cosine LR scheduler with warmup

**Files:**
- Modify: `spectral_flood_walk_v3.py:55-91` (V3Config — add fields)
- Modify: `spectral_flood_walk_v3.py:535-608` (train_and_evaluate — add scheduler)
- Modify: `spectral_flood_walk_v3.py:639-724` (argparse — add flags)
- Modify: `tests/test_spectral_flood_walk_v3.py`

**Step 1: Write the failing test**

Add to `tests/test_spectral_flood_walk_v3.py`:

```python
def test_cosine_lr_schedule_decays_during_training(self) -> None:
    cfg = self._base_config(cross_token_mode="floor")
    cfg.train_steps = 8
    cfg.warmup_steps = 2
    cfg.min_lr_scale = 0.1
    result = train_and_evaluate(cfg)
    history = result["train"]["history"]
    # After warmup, LR should decay
    self.assertIn("lr", history[-1])
    self.assertLess(history[-1]["lr"], history[2]["lr"])
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_spectral_flood_walk_v3.SpectralFloodWalkV3Tests.test_cosine_lr_schedule_decays_during_training -v`
Expected: FAIL (no `lr` in history, no `warmup_steps` field)

**Step 3: Add scheduler**

Add to `V3Config`:
```python
warmup_steps: int = 4
min_lr_scale: float = 0.10
```

In `train_and_evaluate`, after creating the optimizer, add:
```python
def lr_lambda(step: int) -> float:
    if step < cfg.warmup_steps:
        return max(float(step + 1) / max(cfg.warmup_steps, 1), 1e-6)
    progress = (step - cfg.warmup_steps) / max(cfg.train_steps - cfg.warmup_steps, 1)
    return cfg.min_lr_scale + 0.5 * (1.0 - cfg.min_lr_scale) * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

After `optimizer.step()`, add `scheduler.step()`. Add `"lr": float(scheduler.get_last_lr()[0])` to the record dict.

Add `--warmup-steps` and `--min-lr-scale` to the argparser and `config_from_args`.

**Step 4: Run all v3 tests to verify they pass**

Run: `python3 -m unittest tests.test_spectral_flood_walk_v3 -v`
Expected: All tests PASS

**Step 5: Commit**

```
git add spectral_flood_walk_v3.py tests/test_spectral_flood_walk_v3.py
git commit -m "feat: add cosine LR scheduler with warmup to v3 training loop"
```

---

### Task 3: Add DeepFloorGenome and gene space to evolutionary benchmark

**Files:**
- Modify: `tools/evolutionary_benchmark.py` (add genome, gene space, conversion, crossover, mutation, evaluation, subcommand)
- Create: `tests/test_deepfloor_evolution.py`

**Step 1: Write the failing test**

Create `tests/test_deepfloor_evolution.py`:

```python
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
        # With mutation_rate=1.0, at least one gene should differ
        self.assertNotEqual(genome, mutated)

    def test_gene_space_compact_has_stability_genes(self) -> None:
        space = default_deepfloor_gene_space("compact")
        self.assertTrue(len(space.contraction_targets) > 0)
        self.assertTrue(len(space.accumulator_decays) > 0)
        self.assertTrue(len(space.norm_interval_ks) > 0)
        self.assertTrue(len(space.jacobian_lambdas) > 0)
        self.assertTrue(len(space.stochastic_round_ps) > 0)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_deepfloor_evolution -v`
Expected: FAIL (ImportError — none of these exist yet)

**Step 3: Implement DeepFloorGenome and supporting functions**

Add to `tools/evolutionary_benchmark.py` after the existing `RecipeGenome` block (~line 117):

```python
@dataclass(frozen=True)
class DeepFloorGenome:
    recurrent_dim: int
    num_distinct_blocks: int
    view_count: int
    view_combination: str
    cross_token_mode: str
    block_has_residual: bool
    block_nonlinearity: str
    recurrence_step_size: float
    state_decay: float
    contraction_target: float
    train_recurrence_steps: int
    eval_recurrence_steps: int
    norm_interval_k: int
    floor_min_interval: int
    floor_max_interval: int
    floor_threshold: float
    kernel_feature_map: str
    accumulator_decay: float
    quantization: str
    jacobian_lambda: float
    stochastic_round_p: float
    base_lr: float
    weight_decay: float
    seq_len: int
    batch_size: int


@dataclass(frozen=True)
class DeepFloorGeneSpace:
    recurrent_dims: tuple[int, ...]
    num_distinct_blocks: tuple[int, ...]
    view_counts: tuple[int, ...]
    view_combinations: tuple[str, ...]
    cross_token_modes: tuple[str, ...]
    block_has_residuals: tuple[bool, ...]
    block_nonlinearities: tuple[str, ...]
    recurrence_step_sizes: tuple[float, ...]
    state_decays: tuple[float, ...]
    contraction_targets: tuple[float, ...]
    train_recurrence_steps: tuple[int, ...]
    eval_recurrence_steps: tuple[int, ...]
    norm_interval_ks: tuple[int, ...]
    floor_min_intervals: tuple[int, ...]
    floor_max_intervals: tuple[int, ...]
    floor_thresholds: tuple[float, ...]
    kernel_feature_maps: tuple[str, ...]
    accumulator_decays: tuple[float, ...]
    quantizations: tuple[str, ...]
    jacobian_lambdas: tuple[float, ...]
    stochastic_round_ps: tuple[float, ...]
    base_lrs: tuple[float, ...]
    weight_decays: tuple[float, ...]
    seq_lens: tuple[int, ...]
    batch_sizes: tuple[int, ...]
```

Then add the following functions:

```python
def default_deepfloor_gene_space(profile: str) -> DeepFloorGeneSpace:
    if profile == "compact":
        return DeepFloorGeneSpace(
            recurrent_dims=(32, 48, 64),
            num_distinct_blocks=(1, 2),
            view_counts=(1, 2),
            view_combinations=("average",),
            cross_token_modes=("floor", "fused"),
            block_has_residuals=(True,),
            block_nonlinearities=("gelu", "swish"),
            recurrence_step_sizes=(0.5, 1.0),
            state_decays=(0.99, 0.999, 1.0),
            contraction_targets=(0.95, 0.99, 0.995),
            train_recurrence_steps=(8, 16),
            eval_recurrence_steps=(16, 32),
            norm_interval_ks=(4, 8, 16),
            floor_min_intervals=(2, 4),
            floor_max_intervals=(8, 16),
            floor_thresholds=(0.02, 0.05, 0.1),
            kernel_feature_maps=("elu_plus_1", "identity"),
            accumulator_decays=(0.99, 0.999),
            quantizations=("ternary", "int4"),
            jacobian_lambdas=(0.0, 0.01, 0.05),
            stochastic_round_ps=(0.0, 0.5, 1.0),
            base_lrs=(1e-3, 2e-3),
            weight_decays=(0.0, 1e-2),
            seq_lens=(16, 32),
            batch_sizes=(2, 4),
        )
    if profile == "frontier":
        return DeepFloorGeneSpace(
            recurrent_dims=(48, 64, 96),
            num_distinct_blocks=(1, 2, 4),
            view_counts=(2, 4, 8),
            view_combinations=("average", "weighted", "project"),
            cross_token_modes=("floor", "fused"),
            block_has_residuals=(True,),
            block_nonlinearities=("gelu", "swish"),
            recurrence_step_sizes=(0.5, 0.75, 1.0),
            state_decays=(0.99, 0.995, 0.999, 1.0),
            contraction_targets=(0.95, 0.98, 0.99, 0.995),
            train_recurrence_steps=(16, 32, 64),
            eval_recurrence_steps=(64, 128, 256, 512),
            norm_interval_ks=(4, 8, 16, 32),
            floor_min_intervals=(2, 4, 8),
            floor_max_intervals=(16, 32, 64),
            floor_thresholds=(0.02, 0.05, 0.1),
            kernel_feature_maps=("elu_plus_1", "identity"),
            accumulator_decays=(0.99, 0.995, 0.999),
            quantizations=("ternary", "int4", "int6"),
            jacobian_lambdas=(0.0, 0.005, 0.01, 0.05),
            stochastic_round_ps=(0.0, 0.25, 0.5, 1.0),
            base_lrs=(1e-3, 2e-3, 3e-3),
            weight_decays=(0.0, 1e-2),
            seq_lens=(128, 256),
            batch_sizes=(4, 8),
        )
    raise ValueError(f"unsupported deepfloor profile: {profile}")


def random_deepfloor_genome(space: DeepFloorGeneSpace, *, rng: random.Random) -> DeepFloorGenome:
    return DeepFloorGenome(
        recurrent_dim=rng.choice(space.recurrent_dims),
        num_distinct_blocks=rng.choice(space.num_distinct_blocks),
        view_count=rng.choice(space.view_counts),
        view_combination=rng.choice(space.view_combinations),
        cross_token_mode=rng.choice(space.cross_token_modes),
        block_has_residual=rng.choice(space.block_has_residuals),
        block_nonlinearity=rng.choice(space.block_nonlinearities),
        recurrence_step_size=rng.choice(space.recurrence_step_sizes),
        state_decay=rng.choice(space.state_decays),
        contraction_target=rng.choice(space.contraction_targets),
        train_recurrence_steps=rng.choice(space.train_recurrence_steps),
        eval_recurrence_steps=rng.choice(space.eval_recurrence_steps),
        norm_interval_k=rng.choice(space.norm_interval_ks),
        floor_min_interval=rng.choice(space.floor_min_intervals),
        floor_max_interval=rng.choice(space.floor_max_intervals),
        floor_threshold=rng.choice(space.floor_thresholds),
        kernel_feature_map=rng.choice(space.kernel_feature_maps),
        accumulator_decay=rng.choice(space.accumulator_decays),
        quantization=rng.choice(space.quantizations),
        jacobian_lambda=rng.choice(space.jacobian_lambdas),
        stochastic_round_p=rng.choice(space.stochastic_round_ps),
        base_lr=rng.choice(space.base_lrs),
        weight_decay=rng.choice(space.weight_decays),
        seq_len=rng.choice(space.seq_lens),
        batch_size=rng.choice(space.batch_sizes),
    )


def crossover_deepfloor_genomes(left: DeepFloorGenome, right: DeepFloorGenome, *, rng: random.Random) -> DeepFloorGenome:
    left_dict = asdict(left)
    right_dict = asdict(right)
    child: dict[str, Any] = {}
    for key in left_dict:
        child[key] = left_dict[key] if rng.random() < 0.5 else right_dict[key]
    return DeepFloorGenome(**child)


def mutate_deepfloor_genome(genome: DeepFloorGenome, space: DeepFloorGeneSpace, *, mutation_rate: float, rng: random.Random) -> DeepFloorGenome:
    data = asdict(genome)
    gene_options: dict[str, tuple[Any, ...]] = {
        "recurrent_dim": space.recurrent_dims,
        "num_distinct_blocks": space.num_distinct_blocks,
        "view_count": space.view_counts,
        "view_combination": space.view_combinations,
        "cross_token_mode": space.cross_token_modes,
        "block_has_residual": space.block_has_residuals,
        "block_nonlinearity": space.block_nonlinearities,
        "recurrence_step_size": space.recurrence_step_sizes,
        "state_decay": space.state_decays,
        "contraction_target": space.contraction_targets,
        "train_recurrence_steps": space.train_recurrence_steps,
        "eval_recurrence_steps": space.eval_recurrence_steps,
        "norm_interval_k": space.norm_interval_ks,
        "floor_min_interval": space.floor_min_intervals,
        "floor_max_interval": space.floor_max_intervals,
        "floor_threshold": space.floor_thresholds,
        "kernel_feature_map": space.kernel_feature_maps,
        "accumulator_decay": space.accumulator_decays,
        "quantization": space.quantizations,
        "jacobian_lambda": space.jacobian_lambdas,
        "stochastic_round_p": space.stochastic_round_ps,
        "base_lr": space.base_lrs,
        "weight_decay": space.weight_decays,
        "seq_len": space.seq_lens,
        "batch_size": space.batch_sizes,
    }
    for key, choices in gene_options.items():
        if rng.random() < mutation_rate:
            data[key] = rng.choice(tuple(choices))
    return DeepFloorGenome(**data)


def deepfloor_genome_to_v3_config(genome: DeepFloorGenome) -> V3Config:
    return V3Config(
        enwik8_path="",
        recurrent_dim=genome.recurrent_dim,
        num_distinct_blocks=genome.num_distinct_blocks,
        view_count=genome.view_count,
        view_combination=genome.view_combination,
        cross_token_mode=genome.cross_token_mode,
        block_has_residual=genome.block_has_residual,
        block_nonlinearity=genome.block_nonlinearity,
        recurrence_step_size=genome.recurrence_step_size,
        state_decay=genome.state_decay,
        contraction_target=genome.contraction_target,
        train_recurrence_steps=genome.train_recurrence_steps,
        eval_recurrence_steps=genome.eval_recurrence_steps,
        norm_interval_k=genome.norm_interval_k,
        floor_min_interval=genome.floor_min_interval,
        floor_max_interval=genome.floor_max_interval,
        floor_threshold=genome.floor_threshold,
        kernel_feature_map=genome.kernel_feature_map,
        accumulator_decay=genome.accumulator_decay,
        quantization=genome.quantization,
        jacobian_lambda=genome.jacobian_lambda,
        stochastic_round_p=genome.stochastic_round_p,
        base_lr=genome.base_lr,
        weight_decay=genome.weight_decay,
        seq_len=genome.seq_len,
        stride=max(16, genome.seq_len // 4),
        batch_size=genome.batch_size,
    )
```

**Step 4: Run new tests to verify they pass**

Run: `python3 -m unittest tests.test_deepfloor_evolution -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```
git add tools/evolutionary_benchmark.py tests/test_deepfloor_evolution.py
git commit -m "feat: add DeepFloorGenome and gene space to evolutionary benchmark"
```

---

### Task 4: Add `deepfloor-recipe-evolution` subcommand

**Files:**
- Modify: `tools/evolutionary_benchmark.py` (add `evaluate_deepfloor_genome`, `run_deepfloor_recipe_evolution`, subcommand wiring)
- Modify: `tests/test_deepfloor_evolution.py`

**Step 1: Write the failing test**

Add to `tests/test_deepfloor_evolution.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_deepfloor_evolution.DeepFloorEvolutionTests.test_evaluate_deepfloor_genome_returns_bpb -v`
Expected: FAIL (ImportError — `evaluate_deepfloor_genome` doesn't exist)

**Step 3: Implement evaluation and subcommand**

Add `evaluate_deepfloor_genome` to `tools/evolutionary_benchmark.py`:

```python
def evaluate_deepfloor_genome(
    *,
    genome: DeepFloorGenome,
    enwik8_path: Path,
    train_steps: int,
    eval_batches: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    from spectral_flood_walk_v3 import DeepFloorModel, train_and_evaluate

    cfg = deepfloor_genome_to_v3_config(genome)
    cfg = replace(cfg,
        enwik8_path=str(enwik8_path),
        device=str(device),
        seed=seed,
        train_steps=train_steps,
        eval_batches=eval_batches,
    )
    result = train_and_evaluate(cfg)
    return {
        "genome": asdict(genome),
        "config": asdict(cfg),
        "artifact_estimated_mb": result["artifact"]["estimated_mb"],
        "train": result["train"],
        "val": result["val"],
        "test": result["test"],
    }
```

Note: `V3Config` is a `dataclass` not `frozen`, so use `dataclasses.replace` (imported as `replace` from `dataclasses` — already imported at top of file).

Add the `run_deepfloor_recipe_evolution` function following the same pattern as `run_recipe_evolution` but using `DeepFloorGenome`, `DeepFloorGeneSpace`, and `evaluate_deepfloor_genome`. Key differences:
- Artifact check uses `DeepFloorModel(cfg).estimate_artifact_bytes()` instead of `artifact_param_mb_for_cfg`
- Training is step-based (`train_steps`) rather than time-based (`train_seconds`)
- No `EvoBenchmarkConfig` base — the genome produces a complete `V3Config` directly

Add the argparse subcommand `deepfloor-recipe-evolution` with flags:
- `--enwik8-path` (required)
- `--population-size` (default 12)
- `--generations` (default 6)
- `--tournament-size` (default 3)
- `--train-steps` (default 16)
- `--eval-batches` (default 8)
- `--mutation-rate` (default 0.2)
- `--artifact-limit-mb` (default 16.0)
- `--deepfloor-profile` (choices: compact, frontier; default: compact)
- `--confirm-topk` (default 3)
- `--confirm-train-steps` (default 32)
- `--seed` (default 1337)
- `--device` (default auto)

**Step 4: Run all deepfloor evolution tests**

Run: `python3 -m unittest tests.test_deepfloor_evolution -v`
Expected: All tests PASS

**Step 5: Commit**

```
git add tools/evolutionary_benchmark.py tests/test_deepfloor_evolution.py
git commit -m "feat: add deepfloor-recipe-evolution subcommand to evolutionary benchmark"
```

---

### Task 5: Update design doc budget table

**Files:**
- Modify: `docs/plans/2026-04-03-deep-recurrence-multi-view-design.md`

**Step 1: Update the parameter budget table**

Change the recurrent block description from "~12K" to the actual QKV+O count: 4 × d² = 4 × 64² = 16,384. Update the "Notes" column to say "QKV+O per-token transform, applied thousands of times".

Also note the attention block (floor mode) adds another 16K for its QKV+O, and the fused mixer adds 4 × d² + the selection gate.

**Step 2: Commit**

```
git add docs/plans/2026-04-03-deep-recurrence-multi-view-design.md
git commit -m "docs: update budget table to match QKV+O recurrent block params"
```
