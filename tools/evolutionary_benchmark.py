#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import itertools
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.func import functional_call, grad_and_value, vmap

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover - optional dependency on some local envs
    spm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_flood_walk_v0 import get_cuda_memory_stats, maybe_reset_cuda_peak_memory, maybe_sync_cuda
from spectral_flood_walk_v2a import StrongTransformerLM, V2Config, batch_from_starts, build_lm_starts
from spectral_flood_walk_v3 import V3Config


@dataclass
class EvoBenchmarkConfig:
    device: str = "auto"
    dtype: str = "fp16"
    seed: int = 1337
    output_json: str | None = None

    model_dim: int = 512
    num_layers: int = 9
    num_heads: int = 8
    num_kv_heads: int = 4
    mlp_mult: int = 3
    tie_embeddings: bool = True
    tied_embed_init_std: float = 0.005
    rope_base: float = 10_000.0
    logit_softcap: float = 30.0
    qk_gain_init: float = 1.5
    spine_variant: str = "plain"
    xsa_last_n: int = 0
    vocab_size: int = 256
    tokenization_mode: str = "bytes"
    tokenizer_name: str | None = None
    tokenizer_model_path: str | None = None
    cache_dataset_on_device: bool = True
    hard_replay_topk: int = 4
    hard_replay_eval_batches: int = 2
    use_best_archive_for_spawn: bool = True

    seq_len: int = 256
    stride: int = 64
    batch_size: int = 8
    base_lr: float = 2e-3
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0


@dataclass
class TrainingOutcome:
    train_seconds_target: float
    train_seconds_actual: float
    steps_completed: int
    final_loss: float
    tokens_seen: int


@dataclass(frozen=True)
class CrossoverSpec:
    name: str
    requires_base: bool = False
    uses_percentile: bool = True


@dataclass(frozen=True)
class RecipeGenome:
    model_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    mlp_mult: int
    base_lr: float
    weight_decay: float
    qk_gain_init: float
    spine_gene: str
    tokenizer_gene: str
    seq_len: int
    batch_size: int


@dataclass(frozen=True)
class RecipeGeneSpace:
    model_dims: tuple[int, ...]
    num_layers: tuple[int, ...]
    num_heads: tuple[int, ...]
    num_kv_heads: tuple[int, ...]
    mlp_mults: tuple[int, ...]
    base_lrs: tuple[float, ...]
    weight_decays: tuple[float, ...]
    qk_gains: tuple[float, ...]
    spine_genes: tuple[str, ...]
    tokenizer_genes: tuple[str, ...]
    seq_lens: tuple[int, ...]
    batch_sizes: tuple[int, ...]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_param_dtype(name: str, device: torch.device) -> torch.dtype:
    requested = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]
    if device.type != "cuda":
        return torch.float32
    return requested


def maybe_autocast(device: torch.device, dtype: torch.dtype) -> Any:
    enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def vmap_safe_sdpa_context() -> Any:
    """Flash SDPA currently fails under vmap/functional_call on the H100 pod stack."""
    return sdpa_kernel(SDPBackend.MATH)


def maybe_cache_tokens_on_device(tokens: torch.Tensor, *, device: torch.device, enabled: bool) -> torch.Tensor:
    if enabled and device.type == "cuda" and tokens.device != device:
        return tokens.to(device=device, dtype=torch.long)
    return tokens


def load_tokenizer_specs(path: Path | None = None) -> dict[str, dict[str, Any]]:
    specs_path = path or (REPO_ROOT / "data" / "tokenizer_specs.json")
    if not specs_path.exists():
        return {}
    payload = json.loads(specs_path.read_text(encoding="utf-8"))
    tokenizers = payload.get("tokenizers", [])
    return {
        str(entry["name"]): entry
        for entry in tokenizers
        if isinstance(entry, dict) and entry.get("name")
    }


def resolve_tokenizer_model_path(
    *,
    tokenization_mode: str,
    tokenizer_name: str | None,
    tokenizer_model_path: str | None,
) -> tuple[str | None, Path | None]:
    if tokenization_mode == "bytes":
        return tokenizer_name, None
    if tokenization_mode != "sentencepiece":
        raise ValueError(f"unsupported tokenization mode: {tokenization_mode}")

    resolved_name = tokenizer_name
    resolved_path: Path | None = None
    if tokenizer_name:
        specs = load_tokenizer_specs()
        spec = specs.get(tokenizer_name)
        if spec is None:
            raise ValueError(f"unknown tokenizer name: {tokenizer_name}")
        model_path_value = spec.get("model_path")
        if model_path_value is None and tokenizer_model_path is None:
            raise ValueError(f"tokenizer spec {tokenizer_name} is missing model_path")
        if tokenizer_model_path is None:
            tokenizer_model_path = str(model_path_value)
    if tokenizer_model_path is None:
        raise ValueError("sentencepiece tokenization requires --tokenizer-model-path or --tokenizer-name")
    resolved_path = Path(tokenizer_model_path)
    if not resolved_path.is_absolute():
        resolved_path = (REPO_ROOT / resolved_path).resolve()
    if not resolved_path.exists():
        raise ValueError(f"missing tokenizer model: {resolved_path}")
    return resolved_name, resolved_path


def resolve_tokenizer_vocab_size(
    *,
    tokenization_mode: str,
    tokenizer_name: str | None,
    tokenizer_model_path: str | None,
) -> int | None:
    if tokenization_mode == "bytes":
        return None
    resolved_name, resolved_path = resolve_tokenizer_model_path(
        tokenization_mode=tokenization_mode,
        tokenizer_name=tokenizer_name,
        tokenizer_model_path=tokenizer_model_path,
    )
    if tokenization_mode == "sentencepiece":
        if spm is None:
            raise RuntimeError("sentencepiece is required for sentencepiece tokenization experiments")
        processor = spm.SentencePieceProcessor(model_file=str(resolved_path))
        vocab_size = int(processor.get_piece_size())
        if resolved_name:
            spec = load_tokenizer_specs().get(resolved_name)
            if spec is not None and spec.get("vocab_size") is not None:
                expected = int(spec["vocab_size"])
                if expected != vocab_size:
                    raise ValueError(
                        f"tokenizer spec vocab mismatch for {resolved_name}: spec={expected} model={vocab_size}"
                    )
        return vocab_size
    raise ValueError(f"unsupported tokenization mode: {tokenization_mode}")


def default_recipe_gene_space(profile: str) -> RecipeGeneSpace:
    if profile == "compact":
        return RecipeGeneSpace(
            model_dims=(32, 48),
            num_layers=(2, 3),
            num_heads=(4,),
            num_kv_heads=(2,),
            mlp_mults=(2, 3),
            base_lrs=(1e-3, 2e-3),
            weight_decays=(0.0, 1e-2),
            qk_gains=(1.5, 4.0),
            spine_genes=("plain", "xsa2"),
            tokenizer_genes=("bytes", "sp_bpe_1024"),
            seq_lens=(16, 24),
            batch_sizes=(2, 4),
        )
    if profile == "frontier":
        return RecipeGeneSpace(
            model_dims=(384, 512),
            num_layers=(6, 9),
            num_heads=(8,),
            num_kv_heads=(4,),
            mlp_mults=(3, 4),
            base_lrs=(1e-3, 2e-3, 3e-3),
            weight_decays=(1e-2, 8.5e-2),
            qk_gains=(1.5, 4.0),
            spine_genes=("plain", "xsa4"),
            tokenizer_genes=("bytes", "sp_bpe_1024"),
            seq_lens=(256, 384),
            batch_sizes=(8, 12),
        )
    raise ValueError(f"unsupported recipe profile: {profile}")


def decode_spine_gene(spine_gene: str) -> tuple[str, int]:
    if spine_gene == "plain":
        return "plain", 0
    if spine_gene.startswith("xsa"):
        return "xsa", int(spine_gene.removeprefix("xsa"))
    raise ValueError(f"unsupported spine gene: {spine_gene}")


def decode_tokenizer_gene(tokenizer_gene: str) -> tuple[str, str | None, str | None, int]:
    if tokenizer_gene == "bytes":
        return "bytes", None, None, 256
    specs = load_tokenizer_specs()
    spec = specs.get(tokenizer_gene)
    if spec is None:
        raise ValueError(f"unknown tokenizer gene: {tokenizer_gene}")
    kind = str(spec.get("kind", "sentencepiece"))
    if kind != "sentencepiece":
        raise ValueError(f"unsupported tokenizer kind for recipe gene {tokenizer_gene}: {kind}")
    model_path = spec.get("model_path")
    if model_path is None:
        raise ValueError(f"tokenizer spec {tokenizer_gene} is missing model_path")
    vocab_size = int(spec["vocab_size"])
    return "sentencepiece", tokenizer_gene, str(model_path), vocab_size


def recipe_stride_for_seq_len(seq_len: int) -> int:
    return max(16, seq_len // 4)


def recipe_genome_to_cfg(base_cfg: EvoBenchmarkConfig, genome: RecipeGenome) -> EvoBenchmarkConfig:
    spine_variant, xsa_last_n = decode_spine_gene(genome.spine_gene)
    tokenization_mode, tokenizer_name, tokenizer_model_path, vocab_size = decode_tokenizer_gene(genome.tokenizer_gene)
    return replace(
        base_cfg,
        model_dim=int(genome.model_dim),
        num_layers=int(genome.num_layers),
        num_heads=int(genome.num_heads),
        num_kv_heads=int(genome.num_kv_heads),
        mlp_mult=int(genome.mlp_mult),
        base_lr=float(genome.base_lr),
        weight_decay=float(genome.weight_decay),
        qk_gain_init=float(genome.qk_gain_init),
        spine_variant=spine_variant,
        xsa_last_n=int(xsa_last_n),
        tokenization_mode=tokenization_mode,
        tokenizer_name=tokenizer_name,
        tokenizer_model_path=tokenizer_model_path,
        vocab_size=int(vocab_size),
        seq_len=int(genome.seq_len),
        stride=recipe_stride_for_seq_len(int(genome.seq_len)),
        batch_size=int(genome.batch_size),
    )


def genome_to_dict(genome: RecipeGenome) -> dict[str, Any]:
    return asdict(genome)


def random_recipe_genome(space: RecipeGeneSpace, *, rng: random.Random) -> RecipeGenome:
    return RecipeGenome(
        model_dim=rng.choice(space.model_dims),
        num_layers=rng.choice(space.num_layers),
        num_heads=rng.choice(space.num_heads),
        num_kv_heads=rng.choice(space.num_kv_heads),
        mlp_mult=rng.choice(space.mlp_mults),
        base_lr=rng.choice(space.base_lrs),
        weight_decay=rng.choice(space.weight_decays),
        qk_gain_init=rng.choice(space.qk_gains),
        spine_gene=rng.choice(space.spine_genes),
        tokenizer_gene=rng.choice(space.tokenizer_genes),
        seq_len=rng.choice(space.seq_lens),
        batch_size=rng.choice(space.batch_sizes),
    )


def crossover_recipe_genomes(left: RecipeGenome, right: RecipeGenome, *, rng: random.Random) -> RecipeGenome:
    left_dict = asdict(left)
    right_dict = asdict(right)
    child: dict[str, Any] = {}
    for key in left_dict:
        child[key] = left_dict[key] if rng.random() < 0.5 else right_dict[key]
    return RecipeGenome(**child)


def mutate_recipe_genome(genome: RecipeGenome, space: RecipeGeneSpace, *, mutation_rate: float, rng: random.Random) -> RecipeGenome:
    data = asdict(genome)
    gene_options = {
        "model_dim": space.model_dims,
        "num_layers": space.num_layers,
        "num_heads": space.num_heads,
        "num_kv_heads": space.num_kv_heads,
        "mlp_mult": space.mlp_mults,
        "base_lr": space.base_lrs,
        "weight_decay": space.weight_decays,
        "qk_gain_init": space.qk_gains,
        "spine_gene": space.spine_genes,
        "tokenizer_gene": space.tokenizer_genes,
        "seq_len": space.seq_lens,
        "batch_size": space.batch_sizes,
    }
    for key, choices in gene_options.items():
        if rng.random() < mutation_rate:
            data[key] = rng.choice(tuple(choices))
    return RecipeGenome(**data)


# ---------------------------------------------------------------------------
# DeepFloor (v3) genome and gene space
# ---------------------------------------------------------------------------


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
    gene_options = {
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
    seq_len = int(genome.seq_len)
    return V3Config(
        enwik8_path="",
        stride=max(16, seq_len // 4),
        recurrent_dim=int(genome.recurrent_dim),
        num_distinct_blocks=int(genome.num_distinct_blocks),
        view_count=int(genome.view_count),
        view_combination=genome.view_combination,
        cross_token_mode=genome.cross_token_mode,
        block_has_residual=genome.block_has_residual,
        block_nonlinearity=genome.block_nonlinearity,
        recurrence_step_size=float(genome.recurrence_step_size),
        state_decay=float(genome.state_decay),
        contraction_target=float(genome.contraction_target),
        train_recurrence_steps=int(genome.train_recurrence_steps),
        eval_recurrence_steps=int(genome.eval_recurrence_steps),
        norm_interval_k=int(genome.norm_interval_k),
        floor_min_interval=int(genome.floor_min_interval),
        floor_max_interval=int(genome.floor_max_interval),
        floor_threshold=float(genome.floor_threshold),
        kernel_feature_map=genome.kernel_feature_map,
        accumulator_decay=float(genome.accumulator_decay),
        quantization=genome.quantization,
        jacobian_lambda=float(genome.jacobian_lambda),
        stochastic_round_p=float(genome.stochastic_round_p),
        base_lr=float(genome.base_lr),
        weight_decay=float(genome.weight_decay),
        seq_len=seq_len,
        batch_size=int(genome.batch_size),
    )


def summarize_replay_buffer(entries: list[dict[str, Any]]) -> dict[str, float]:
    if not entries:
        return {"size": 0.0, "max_loss": float("nan"), "mean_loss": float("nan"), "num_starts": 0.0}
    losses = [float(entry["loss"]) for entry in entries]
    num_starts = sum(len(entry["starts"]) for entry in entries)
    return {
        "size": float(len(entries)),
        "max_loss": float(max(losses)),
        "mean_loss": float(sum(losses) / len(losses)),
        "num_starts": float(num_starts),
    }


def update_replay_buffer(
    entries: list[dict[str, Any]],
    *,
    loss: float,
    starts: list[int],
    topk: int,
    stage_index: int | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    if topk <= 0:
        return []
    normalized_starts = [int(start) for start in starts]
    if not normalized_starts:
        return list(entries)
    merged: dict[tuple[int, ...], dict[str, Any]] = {
        tuple(int(start) for start in entry["starts"]): {
            "loss": float(entry["loss"]),
            "starts": [int(start) for start in entry["starts"]],
            "stage_index": None if entry.get("stage_index") is None else int(entry["stage_index"]),
            "seed": None if entry.get("seed") is None else int(entry["seed"]),
        }
        for entry in entries
    }
    key = tuple(normalized_starts)
    candidate = {
        "loss": float(loss),
        "starts": normalized_starts,
        "stage_index": None if stage_index is None else int(stage_index),
        "seed": None if seed is None else int(seed),
    }
    incumbent = merged.get(key)
    if incumbent is None or float(candidate["loss"]) > float(incumbent["loss"]):
        merged[key] = candidate
    ranked = sorted(merged.values(), key=lambda entry: float(entry["loss"]), reverse=True)
    return ranked[:topk]


def merge_replay_buffers(buffers: list[list[dict[str, Any]]], *, topk: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for buffer in buffers:
        for entry in buffer:
            merged = update_replay_buffer(
                merged,
                loss=float(entry["loss"]),
                starts=[int(start) for start in entry["starts"]],
                topk=topk,
                stage_index=entry.get("stage_index"),
                seed=entry.get("seed"),
            )
    return merged


def replay_eval_starts(entries: list[dict[str, Any]], *, batch_size: int, eval_batches: int) -> list[int]:
    needed = batch_size * eval_batches
    if needed <= 0:
        return []
    starts: list[int] = []
    seen: set[int] = set()
    for entry in sorted(entries, key=lambda row: float(row["loss"]), reverse=True):
        for start in entry["starts"]:
            normalized = int(start)
            if normalized in seen:
                continue
            starts.append(normalized)
            seen.add(normalized)
            if len(starts) >= needed:
                break
        if len(starts) >= needed:
            break
    remainder = len(starts) % batch_size
    if remainder != 0:
        starts = starts[: len(starts) - remainder]
    return starts


def build_model_cfg(cfg: EvoBenchmarkConfig) -> V2Config:
    return V2Config(
        seq_len=cfg.seq_len,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        train_steps=1,
        eval_batches=1,
        report_every=1,
        model_dim=cfg.model_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        mlp_mult=cfg.mlp_mult,
        tie_embeddings=cfg.tie_embeddings,
        tied_embed_init_std=cfg.tied_embed_init_std,
        rope_base=cfg.rope_base,
        logit_softcap=cfg.logit_softcap,
        qk_gain_init=cfg.qk_gain_init,
        spine_variant=cfg.spine_variant,
        xsa_last_n=cfg.xsa_last_n,
        base_lr=cfg.base_lr,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
    )


def build_model(cfg: EvoBenchmarkConfig, device: torch.device, dtype: torch.dtype) -> StrongTransformerLM:
    model_cfg = build_model_cfg(cfg)
    model = StrongTransformerLM(model_cfg, vocab_size=cfg.vocab_size).to(device)
    if device.type == "cuda":
        model = model.to(dtype=dtype)
    model.eval()
    return model


def artifact_param_mb_for_cfg(cfg: EvoBenchmarkConfig) -> float:
    model = build_model(cfg, torch.device("cpu"), torch.float32)
    return float(model.compact_bytes() * 2.0 / (1024.0 * 1024.0))


def parameter_state_dict(model: StrongTransformerLM) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters()}


def load_parameter_state(model: StrongTransformerLM, state: dict[str, torch.Tensor], device: torch.device) -> None:
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, source in state.items():
            target = named_params[name]
            target.copy_(source.to(device=device, dtype=target.dtype))


def load_enwik8_splits(
    path: Path,
    *,
    train_fraction: float = 0.90,
    val_fraction: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size == 0:
        raise ValueError(f"empty enwik8 file: {path}")
    tokens = torch.from_numpy(raw.astype(np.int64, copy=False))
    train_end = int(tokens.numel() * train_fraction)
    val_end = int(tokens.numel() * (train_fraction + val_fraction))
    train_tokens = tokens[:train_end]
    val_tokens = tokens[train_end:val_end]
    test_tokens = tokens[val_end:]
    if min(train_tokens.numel(), val_tokens.numel(), test_tokens.numel()) <= 0:
        raise ValueError("enwik8 split produced an empty partition")
    return train_tokens, val_tokens, test_tokens


def load_sentencepiece_enwik8_splits(
    path: Path,
    *,
    tokenizer_model_path: Path,
    train_fraction: float = 0.90,
    val_fraction: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if spm is None:
        raise RuntimeError("sentencepiece is required for sentencepiece tokenization experiments")
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"empty enwik8 file: {path}")
    text = raw.decode("utf-8", errors="replace")
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model_path))
    token_ids = processor.encode(text, out_type=int, enable_sampling=False)
    if not token_ids:
        raise ValueError(f"sentencepiece tokenizer produced no tokens for: {path}")
    tokens = torch.tensor(token_ids, dtype=torch.long)
    train_end = int(tokens.numel() * train_fraction)
    val_end = int(tokens.numel() * (train_fraction + val_fraction))
    train_tokens = tokens[:train_end]
    val_tokens = tokens[train_end:val_end]
    test_tokens = tokens[val_end:]
    if min(train_tokens.numel(), val_tokens.numel(), test_tokens.numel()) <= 0:
        raise ValueError("sentencepiece enwik8 split produced an empty partition")
    metadata = {
        "tokenization_mode": "sentencepiece",
        "tokenizer_model_path": str(tokenizer_model_path),
        "tokenizer_vocab_size": int(processor.get_piece_size()),
        "decoded_chars": int(len(text)),
        "raw_bytes": int(len(raw)),
    }
    return train_tokens, val_tokens, test_tokens, metadata


def prepare_enwik8_splits(
    path: Path,
    *,
    device: torch.device,
    cache_on_device: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    return prepare_tokenized_enwik8_splits(
        path,
        device=device,
        cache_on_device=cache_on_device,
        tokenization_mode="bytes",
        tokenizer_name=None,
        tokenizer_model_path=None,
    )


def prepare_tokenized_enwik8_splits(
    path: Path,
    *,
    device: torch.device,
    cache_on_device: bool,
    tokenization_mode: str,
    tokenizer_name: str | None,
    tokenizer_model_path: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if tokenization_mode == "bytes":
        train_tokens, val_tokens, test_tokens = load_enwik8_splits(path)
        metadata = {
            "tokenization_mode": "bytes",
            "tokenizer_name": None,
            "tokenizer_model_path": None,
            "tokenizer_vocab_size": 256,
            "raw_bytes": int(train_tokens.numel() + val_tokens.numel() + test_tokens.numel()),
        }
    elif tokenization_mode == "sentencepiece":
        resolved_name, resolved_path = resolve_tokenizer_model_path(
            tokenization_mode=tokenization_mode,
            tokenizer_name=tokenizer_name,
            tokenizer_model_path=tokenizer_model_path,
        )
        train_tokens, val_tokens, test_tokens, metadata = load_sentencepiece_enwik8_splits(
            path,
            tokenizer_model_path=resolved_path,
        )
        metadata["tokenizer_name"] = resolved_name
    else:
        raise ValueError(f"unsupported tokenization mode: {tokenization_mode}")

    cached = (
        maybe_cache_tokens_on_device(train_tokens, device=device, enabled=cache_on_device),
        maybe_cache_tokens_on_device(val_tokens, device=device, enabled=cache_on_device),
        maybe_cache_tokens_on_device(test_tokens, device=device, enabled=cache_on_device),
    )
    metadata["residency"] = cached[0].device.type
    return cached[0], cached[1], cached[2], metadata


def random_token_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    batch = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len + 1),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    return batch[:, :-1], batch[:, 1:]


def choose_eval_starts(
    starts: list[int],
    *,
    batch_size: int,
    eval_batches: int,
    seed: int,
) -> list[int]:
    needed = batch_size * eval_batches
    if not starts:
        return []
    if len(starts) <= needed:
        return starts[:needed]
    rng = random.Random(seed)
    return rng.sample(starts, needed)


def evaluate_bpb(
    model: StrongTransformerLM,
    *,
    tokens: torch.Tensor,
    eval_starts: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    param_dtype: torch.dtype,
) -> dict[str, float]:
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for idx in range(0, len(eval_starts), batch_size):
            batch_starts = eval_starts[idx : idx + batch_size]
            inputs, targets = batch_from_starts(tokens, batch_starts, seq_len, device)
            with maybe_autocast(device, param_dtype):
                logits = model(inputs)["logits"]
            assert logits is not None
            total_loss += float(
                F.cross_entropy(
                    logits.reshape(-1, model.vocab_size),
                    targets.reshape(-1),
                    reduction="sum",
                ).item()
            )
            total_tokens += int(targets.numel())
    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": float(mean_loss),
        "bpb": float(mean_loss / math.log(2.0)),
        "tokens": float(total_tokens),
    }


def train_for_seconds(
    model: StrongTransformerLM,
    *,
    train_tokens: torch.Tensor,
    train_starts: list[int],
    seq_len: int,
    batch_size: int,
    base_lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    train_seconds: float,
    seed: int,
    device: torch.device,
    param_dtype: torch.dtype,
    hard_replay_topk: int = 0,
) -> tuple[TrainingOutcome, list[dict[str, Any]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    start_time = time.perf_counter()
    steps = 0
    last_loss = float("nan")
    total_tokens = 0
    replay_entries: list[dict[str, Any]] = []
    model.train()
    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed >= train_seconds and steps > 0:
            break
        batch_starts = [train_starts[rng.randrange(len(train_starts))] for _ in range(batch_size)]
        inputs, targets = batch_from_starts(train_tokens, batch_starts, seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        with maybe_autocast(device, param_dtype):
            out = model(inputs, targets)
        loss = out["loss"]
        assert loss is not None
        loss.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        last_loss = float(loss.item())
        steps += 1
        total_tokens += int(inputs.numel())
        replay_entries = update_replay_buffer(
            replay_entries,
            loss=last_loss,
            starts=batch_starts,
            topk=hard_replay_topk,
            seed=seed,
        )
    maybe_sync_cuda(device)
    actual = time.perf_counter() - start_time
    model.eval()
    return (
        TrainingOutcome(
            train_seconds_target=float(train_seconds),
            train_seconds_actual=float(actual),
            steps_completed=steps,
            final_loss=float(last_loss),
            tokens_seen=int(total_tokens),
        ),
        replay_entries,
    )


def population_batch_from_starts(
    tokens: torch.Tensor,
    starts: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if starts.ndim != 2:
        raise ValueError(f"population_batch_from_starts expects [population, batch] starts, got shape={tuple(starts.shape)}")
    token_device = tokens.device
    flat_starts = starts.reshape(-1).to(device=token_device, dtype=torch.long)
    offsets = torch.arange(seq_len + 1, device=token_device, dtype=torch.long)
    windows = tokens[flat_starts.unsqueeze(1) + offsets.unsqueeze(0)]
    windows = windows.reshape(starts.shape[0], starts.shape[1], seq_len + 1).to(device=device, dtype=torch.long)
    return windows[:, :, :-1], windows[:, :, 1:]


def split_population_states(stacked_params: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    if not stacked_params:
        return []
    population_size = int(next(iter(stacked_params.values())).shape[0])
    states: list[dict[str, torch.Tensor]] = []
    for member_idx in range(population_size):
        states.append({name: value[member_idx].detach().cpu().clone() for name, value in stacked_params.items()})
    return states


def sample_population_batch_starts(
    train_starts: torch.Tensor,
    *,
    batch_size: int,
    generators: list[torch.Generator],
) -> torch.Tensor:
    return torch.stack(
        [
            train_starts[torch.randint(0, int(train_starts.numel()), (batch_size,), generator=generator, dtype=torch.int64)]
            for generator in generators
        ],
        dim=0,
    )


def population_grad_norms(grads: dict[str, torch.Tensor]) -> torch.Tensor:
    norms_sq: torch.Tensor | None = None
    for grad in grads.values():
        flat_sq = grad.float().reshape(grad.shape[0], -1).square().sum(dim=1)
        norms_sq = flat_sq if norms_sq is None else norms_sq + flat_sq
    if norms_sq is None:
        return torch.zeros((0,), dtype=torch.float32)
    return norms_sq.sqrt()


def clip_population_grads(
    grads: dict[str, torch.Tensor],
    *,
    max_norm: float,
) -> dict[str, torch.Tensor]:
    if max_norm <= 0.0 or not grads:
        return grads
    grad_norms = population_grad_norms(grads)
    if grad_norms.numel() == 0:
        return grads
    scales = torch.clamp(max_norm / grad_norms.clamp_min(1e-6), max=1.0)
    clipped: dict[str, torch.Tensor] = {}
    for name, grad in grads.items():
        view_shape = (grad.shape[0],) + (1,) * (grad.ndim - 1)
        clipped[name] = grad * scales.to(device=grad.device, dtype=grad.dtype).view(view_shape)
    return clipped


def train_population_for_seconds(
    model: StrongTransformerLM,
    *,
    states: list[dict[str, torch.Tensor]],
    train_tokens: torch.Tensor,
    train_starts: list[int],
    seq_len: int,
    batch_size: int,
    base_lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    train_seconds: float,
    seeds: list[int],
    device: torch.device,
    param_dtype: torch.dtype,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    hard_replay_topk: int = 0,
) -> tuple[list[dict[str, torch.Tensor]], list[TrainingOutcome], list[list[dict[str, Any]]]]:
    if not states:
        return [], []
    if len(states) != len(seeds):
        raise ValueError(f"len(states)={len(states)} != len(seeds)={len(seeds)}")

    stacked_params = stack_population_states(states, device=device, param_dtype=param_dtype)
    buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}
    optimizer_m = {name: torch.zeros_like(value, dtype=torch.float32) for name, value in stacked_params.items()}
    optimizer_v = {name: torch.zeros_like(value, dtype=torch.float32) for name, value in stacked_params.items()}
    train_starts_tensor = torch.tensor(train_starts, dtype=torch.int64)
    generators = []
    for seed in seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        generators.append(generator)

    train_tokens_device = train_tokens.to(device=device, dtype=torch.long) if device.type == "cuda" else train_tokens

    def single_loss(single_params: dict[str, torch.Tensor], input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        with maybe_autocast(device, param_dtype):
            out = functional_call(model, (single_params, buffers), (input_ids, target_ids))
        loss = out["loss"]
        assert loss is not None
        return loss.to(torch.float32)

    grad_and_loss = grad_and_value(single_loss)
    start_time = time.perf_counter()
    steps = 0
    last_losses = [float("nan")] * len(states)
    total_tokens = 0
    model.eval()
    replay_buffers: list[list[dict[str, Any]]] = [[] for _ in states]

    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed >= train_seconds and steps > 0:
            break
        batch_starts = sample_population_batch_starts(train_starts_tensor, batch_size=batch_size, generators=generators)
        inputs, targets = population_batch_from_starts(train_tokens_device, batch_starts, seq_len, device)
        with vmap_safe_sdpa_context():
            grads, losses = vmap(grad_and_loss, in_dims=(0, 0, 0))(stacked_params, inputs, targets)
        grads = clip_population_grads(grads, max_norm=grad_clip_norm)
        steps += 1
        bias_correction1 = 1.0 - beta1**steps
        bias_correction2 = 1.0 - beta2**steps
        with torch.no_grad():
            for name, param in stacked_params.items():
                grad_fp32 = grads[name].float()
                optimizer_m[name].mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                optimizer_v[name].mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)
                m_hat = optimizer_m[name] / bias_correction1
                v_hat = optimizer_v[name] / bias_correction2
                updated = param.float()
                if weight_decay > 0.0:
                    updated.mul_(1.0 - base_lr * weight_decay)
                updated.addcdiv_(m_hat, v_hat.sqrt().add(eps), value=-base_lr)
                param.copy_(updated.to(dtype=param.dtype))
        last_losses = [float(value) for value in losses.detach().cpu().tolist()]
        total_tokens += int(inputs.shape[1] * inputs.shape[2])
        for member_idx, member_loss in enumerate(last_losses):
            replay_buffers[member_idx] = update_replay_buffer(
                replay_buffers[member_idx],
                loss=member_loss,
                starts=[int(start) for start in batch_starts[member_idx].tolist()],
                topk=hard_replay_topk,
                seed=seeds[member_idx],
            )

    maybe_sync_cuda(device)
    actual = time.perf_counter() - start_time
    trained_states = split_population_states(stacked_params)
    train_infos = [
        TrainingOutcome(
            train_seconds_target=float(train_seconds),
            train_seconds_actual=float(actual),
            steps_completed=steps,
            final_loss=float(last_losses[idx]),
            tokens_seen=int(total_tokens),
        )
        for idx in range(len(trained_states))
    ]
    return trained_states, train_infos, replay_buffers


def clone_with_perturbation(
    state: dict[str, torch.Tensor],
    *,
    noise_std: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    cloned: dict[str, torch.Tensor] = {}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for name, value in state.items():
        if noise_std <= 0.0:
            cloned[name] = value.clone()
            continue
        noise = torch.randn(value.shape, generator=generator, dtype=torch.float32) * noise_std
        cloned[name] = (value.float() + noise).to(dtype=value.dtype)
    return cloned


def estimate_population_param_bytes(model: StrongTransformerLM, dtype: torch.dtype, population_size: int) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    return int(model.compact_bytes() * element_size * population_size)


def build_population_parameter_stack(
    model: StrongTransformerLM,
    *,
    population_size: int,
    noise_std: float,
    seed: int,
    device: torch.device,
    param_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    stacked: dict[str, torch.Tensor] = {}
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for name, param in model.named_parameters():
        base = param.detach().to(device=device, dtype=param_dtype)
        if noise_std > 0.0:
            noise = torch.randn((population_size, *base.shape), device=device, dtype=base.dtype, generator=generator)
            stacked[name] = base.unsqueeze(0) + noise_std * noise
        else:
            stacked[name] = base.unsqueeze(0).expand(population_size, *base.shape).clone()
    return stacked


def stack_population_states(
    states: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    param_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    if not states:
        return {}
    names = tuple(states[0].keys())
    stacked: dict[str, torch.Tensor] = {}
    for name in names:
        stacked[name] = torch.stack(
            [state[name].to(device=device, dtype=param_dtype) for state in states],
            dim=0,
        )
    return stacked


def iter_population_chunks(population_size: int, population_chunk_size: int | None) -> list[tuple[int, int]]:
    if population_size <= 0:
        return []
    chunk_size = population_size if population_chunk_size is None or population_chunk_size <= 0 else population_chunk_size
    chunk_size = min(chunk_size, population_size)
    chunks = []
    for start in range(0, population_size, chunk_size):
        chunks.append((start, min(chunk_size, population_size - start)))
    return chunks


def benchmark_population_vmap(
    model: StrongTransformerLM,
    *,
    population_size: int,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    noise_std: float,
    warmup_repeats: int,
    timed_repeats: int,
    population_chunk_size: int | None,
    seed: int,
    device: torch.device,
    param_dtype: torch.dtype,
) -> dict[str, Any]:
    model.eval()
    if device.type == "cuda":
        with torch.no_grad():
            _ = model(inputs)
        maybe_sync_cuda(device)
    buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}
    estimated_param_bytes = estimate_population_param_bytes(model, param_dtype, population_size)
    chunks = iter_population_chunks(population_size, population_chunk_size)
    max_chunk_size = max((chunk_size for _, chunk_size in chunks), default=population_size)
    estimated_chunk_param_bytes = estimate_population_param_bytes(model, param_dtype, max_chunk_size)
    start_mem = get_cuda_memory_stats(device) if device.type == "cuda" else None
    try:
        def population_loss(single_params: dict[str, torch.Tensor], input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
            out = functional_call(model, (single_params, buffers), (input_ids,))
            logits = out["logits"]
            assert logits is not None
            return F.cross_entropy(logits.reshape(-1, model.vocab_size), target_ids.reshape(-1))

        def evaluate_population_once() -> float:
            elapsed_s = 0.0
            for chunk_start, chunk_size in chunks:
                params = build_population_parameter_stack(
                    model,
                    population_size=chunk_size,
                    noise_std=noise_std,
                    seed=seed + chunk_start,
                    device=device,
                    param_dtype=param_dtype,
                )
                maybe_sync_cuda(device)
                if device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    with vmap_safe_sdpa_context():
                        _ = vmap(population_loss, in_dims=(0, None, None))(params, inputs, targets)
                    end_event.record()
                    maybe_sync_cuda(device)
                    elapsed_s += float(start_event.elapsed_time(end_event) / 1000.0)
                else:
                    start = time.perf_counter()
                    with vmap_safe_sdpa_context():
                        _ = vmap(population_loss, in_dims=(0, None, None))(params, inputs, targets)
                    elapsed_s += float(time.perf_counter() - start)
                del params
            return elapsed_s

        with torch.no_grad():
            for _ in range(warmup_repeats):
                _ = evaluate_population_once()

        elapsed_s = 0.0
        with torch.no_grad():
            for _ in range(timed_repeats):
                elapsed_s += evaluate_population_once()

        end_mem = get_cuda_memory_stats(device) if device.type == "cuda" else None
        models_per_second = float((population_size * timed_repeats) / max(elapsed_s, 1e-6))
        return {
            "population_size": int(population_size),
            "status": "ok",
            "batch_eval_s": float(elapsed_s / max(timed_repeats, 1)),
            "models_per_second": models_per_second,
            "estimated_population_param_mb": float(estimated_param_bytes / (1024.0 * 1024.0)),
            "population_chunk_size": int(max_chunk_size),
            "num_chunks": int(len(chunks)),
            "estimated_chunk_param_mb": float(estimated_chunk_param_bytes / (1024.0 * 1024.0)),
            "cuda_memory_before": start_mem,
            "cuda_memory_after": end_mem,
        }
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "population_size": int(population_size),
            "status": "oom",
            "error": str(exc),
            "estimated_population_param_mb": float(estimated_param_bytes / (1024.0 * 1024.0)),
            "population_chunk_size": int(max_chunk_size),
            "num_chunks": int(len(chunks)),
            "estimated_chunk_param_mb": float(estimated_chunk_param_bytes / (1024.0 * 1024.0)),
            "cuda_memory_before": start_mem,
        }


def flatten_distance_distribution(
    parent_a: dict[str, torch.Tensor],
    parent_b: dict[str, torch.Tensor],
) -> torch.Tensor:
    diffs = []
    for name in parent_a:
        diff = (parent_a[name].float() - parent_b[name].float()).abs().reshape(-1)
        diffs.append(diff)
    return torch.cat(diffs) if diffs else torch.zeros((0,), dtype=torch.float32)


def _quantile_threshold(values: torch.Tensor, percentile: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values, percentile / 100.0).item())


def _approx_quantile_threshold_from_tensors(
    flat_tensors: list[torch.Tensor],
    percentile: float,
    *,
    max_samples: int = 1_000_000,
) -> float:
    nonempty = [tensor.reshape(-1).float() for tensor in flat_tensors if tensor.numel() > 0]
    total_values = sum(int(tensor.numel()) for tensor in nonempty)
    if total_values == 0:
        return 0.0
    if total_values <= max_samples:
        return _quantile_threshold(torch.cat(nonempty), percentile)

    counts = np.array([int(tensor.numel()) for tensor in nonempty], dtype=np.int64)
    exact = max_samples * counts / max(total_values, 1)
    allocation = np.floor(exact).astype(np.int64)
    remaining = int(max_samples - allocation.sum())
    if remaining > 0:
        for index in np.argsort(-(exact - allocation))[:remaining]:
            allocation[index] += 1

    samples: list[torch.Tensor] = []
    for tensor, sample_count in zip(nonempty, allocation.tolist()):
        if sample_count <= 0:
            continue
        if int(tensor.numel()) <= sample_count:
            samples.append(tensor)
            continue
        positions = (
            (torch.arange(sample_count, dtype=torch.float64) + 0.5) * (float(tensor.numel()) / float(sample_count))
        ).floor()
        positions = positions.clamp_(0, int(tensor.numel()) - 1).to(torch.int64)
        samples.append(tensor.index_select(0, positions))
    return _quantile_threshold(torch.cat(samples), percentile)


def _layer_group_name(param_name: str) -> str:
    parts = param_name.split(".")
    if len(parts) >= 2 and parts[0] == "blocks":
        return ".".join(parts[:2])
    if parts[0] == "tok_emb":
        return "tok_emb"
    if parts[0] == "final_norm":
        return "final_norm"
    if parts[0] == "skip_weights":
        return "skip_weights"
    if parts[0] == "lm_head":
        return "lm_head"
    return parts[0]


def _choose_random_tensor(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    pick_left = bool(torch.rand((1,), generator=generator, dtype=torch.float32).item() < 0.5)
    chosen = left if pick_left else right
    return chosen.clone()


def _delta_magnitude_thresholds(
    parent_a: dict[str, torch.Tensor],
    parent_b: dict[str, torch.Tensor],
    *,
    base_state: dict[str, torch.Tensor],
    percentile: float,
) -> tuple[float, float]:
    mags_a = []
    mags_b = []
    for name in parent_a:
        base = base_state[name].float()
        mags_a.append((parent_a[name].float() - base).abs().reshape(-1))
        mags_b.append((parent_b[name].float() - base).abs().reshape(-1))
    return (
        _approx_quantile_threshold_from_tensors(mags_a, percentile),
        _approx_quantile_threshold_from_tensors(mags_b, percentile),
    )


def overlap_crossover(
    parent_a: dict[str, torch.Tensor],
    parent_b: dict[str, torch.Tensor],
    *,
    percentile: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    flat_diffs = []
    for name in parent_a:
        flat_diffs.append((parent_a[name].float() - parent_b[name].float()).abs().reshape(-1))
    if not flat_diffs:
        return {}, {"threshold": 0.0, "overlap_fraction": 0.0}
    threshold = _approx_quantile_threshold_from_tensors(flat_diffs, percentile)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    offspring: dict[str, torch.Tensor] = {}
    overlap_count = 0
    total_count = 0
    for name in parent_a:
        a = parent_a[name]
        b = parent_b[name]
        diff = (a.float() - b.float()).abs()
        overlap_mask = diff <= threshold
        pick_a = torch.rand(a.shape, generator=generator, dtype=torch.float32) < 0.5
        averaged = 0.5 * (a.float() + b.float())
        child = torch.where(overlap_mask, averaged, torch.where(pick_a, a.float(), b.float()))
        offspring[name] = child.to(dtype=a.dtype)
        overlap_count += int(overlap_mask.sum().item())
        total_count += int(overlap_mask.numel())
    return offspring, {
        "threshold": threshold,
        "overlap_fraction": float(overlap_count / max(total_count, 1)),
        "percentile": float(percentile),
    }


def delta_overlap_crossover(
    parent_a: dict[str, torch.Tensor],
    parent_b: dict[str, torch.Tensor],
    *,
    base_state: dict[str, torch.Tensor],
    percentile: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    diffs = []
    for name in parent_a:
        delta_a = parent_a[name].float() - base_state[name].float()
        delta_b = parent_b[name].float() - base_state[name].float()
        diffs.append((delta_a - delta_b).abs().reshape(-1))
    threshold = _approx_quantile_threshold_from_tensors(diffs, percentile)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    offspring: dict[str, torch.Tensor] = {}
    overlap_count = 0
    total_count = 0
    for name in parent_a:
        base = base_state[name].float()
        delta_a = parent_a[name].float() - base
        delta_b = parent_b[name].float() - base
        diff = (delta_a - delta_b).abs()
        overlap_mask = diff <= threshold
        pick_a = torch.rand(delta_a.shape, generator=generator, dtype=torch.float32) < 0.5
        child_delta = torch.where(overlap_mask, 0.5 * (delta_a + delta_b), torch.where(pick_a, delta_a, delta_b))
        offspring[name] = (base + child_delta).to(dtype=parent_a[name].dtype)
        overlap_count += int(overlap_mask.sum().item())
        total_count += int(overlap_mask.numel())
    return offspring, {
        "threshold": threshold,
        "overlap_fraction": float(overlap_count / max(total_count, 1)),
        "percentile": float(percentile),
    }


def sign_consensus_crossover(
    parent_a: dict[str, torch.Tensor],
    parent_b: dict[str, torch.Tensor],
    *,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    offspring: dict[str, torch.Tensor] = {}
    agree_count = 0
    total_count = 0
    for name in parent_a:
        a = parent_a[name].float()
        b = parent_b[name].float()
        sign_agree = torch.sign(a) == torch.sign(b)
        pick_a = torch.rand(a.shape, generator=generator, dtype=torch.float32) < 0.5
        child = torch.where(sign_agree, 0.5 * (a + b), torch.where(pick_a, a, b))
        offspring[name] = child.to(dtype=parent_a[name].dtype)
        agree_count += int(sign_agree.sum().item())
        total_count += int(sign_agree.numel())
    return offspring, {
        "threshold": 0.0,
        "overlap_fraction": float(agree_count / max(total_count, 1)),
        "percentile": -1.0,
    }


def delta_sign_consensus_crossover(
    parent_a: dict[str, torch.Tensor],
    parent_b: dict[str, torch.Tensor],
    *,
    base_state: dict[str, torch.Tensor],
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    offspring: dict[str, torch.Tensor] = {}
    agree_count = 0
    total_count = 0
    for name in parent_a:
        base = base_state[name].float()
        delta_a = parent_a[name].float() - base
        delta_b = parent_b[name].float() - base
        sign_agree = torch.sign(delta_a) == torch.sign(delta_b)
        pick_a = torch.rand(delta_a.shape, generator=generator, dtype=torch.float32) < 0.5
        child_delta = torch.where(sign_agree, 0.5 * (delta_a + delta_b), torch.where(pick_a, delta_a, delta_b))
        offspring[name] = (base + child_delta).to(dtype=parent_a[name].dtype)
        agree_count += int(sign_agree.sum().item())
        total_count += int(sign_agree.numel())
    return offspring, {
        "threshold": 0.0,
        "overlap_fraction": float(agree_count / max(total_count, 1)),
        "percentile": -1.0,
    }


def tensor_swap_crossover(
    parent_a: dict[str, torch.Tensor],
    parent_b: dict[str, torch.Tensor],
    *,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    offspring = {
        name: _choose_random_tensor(parent_a[name], parent_b[name], generator=generator)
        for name in parent_a
    }
    left_count = sum(
        1
        for name in parent_a
        if torch.equal(offspring[name], parent_a[name]) and not torch.equal(parent_a[name], parent_b[name])
    )
    return offspring, {
        "threshold": 0.0,
        "overlap_fraction": float(left_count / max(len(parent_a), 1)),
        "percentile": -1.0,
    }


def layer_swap_crossover(
    parent_a: dict[str, torch.Tensor],
    parent_b: dict[str, torch.Tensor],
    *,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    group_choice: dict[str, bool] = {}
    offspring: dict[str, torch.Tensor] = {}
    for name in parent_a:
        group = _layer_group_name(name)
        if group not in group_choice:
            group_choice[group] = bool(torch.rand((1,), generator=generator, dtype=torch.float32).item() < 0.5)
        offspring[name] = parent_a[name].clone() if group_choice[group] else parent_b[name].clone()
    left_groups = sum(1 for pick_left in group_choice.values() if pick_left)
    return offspring, {
        "threshold": 0.0,
        "overlap_fraction": float(left_groups / max(len(group_choice), 1)),
        "percentile": -1.0,
    }


def parent_copy_crossover(
    parent_a: dict[str, torch.Tensor],
    parent_b: dict[str, torch.Tensor],
    *,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    del parent_b, seed
    offspring = {name: value.clone() for name, value in parent_a.items()}
    return offspring, {
        "threshold": 0.0,
        "overlap_fraction": 1.0,
        "percentile": -1.0,
    }


def delta_importance_crossover(
    parent_a: dict[str, torch.Tensor],
    parent_b: dict[str, torch.Tensor],
    *,
    base_state: dict[str, torch.Tensor],
    percentile: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    diffs = []
    for name in parent_a:
        delta_a = parent_a[name].float() - base_state[name].float()
        delta_b = parent_b[name].float() - base_state[name].float()
        diffs.append((delta_a - delta_b).abs().reshape(-1))
    threshold = _approx_quantile_threshold_from_tensors(diffs, percentile)
    offspring: dict[str, torch.Tensor] = {}
    overlap_count = 0
    total_count = 0
    for name in parent_a:
        base = base_state[name].float()
        delta_a = parent_a[name].float() - base
        delta_b = parent_b[name].float() - base
        diff = (delta_a - delta_b).abs()
        overlap_mask = diff <= threshold
        choose_a = delta_a.abs() >= delta_b.abs()
        child_delta = torch.where(overlap_mask, 0.5 * (delta_a + delta_b), torch.where(choose_a, delta_a, delta_b))
        offspring[name] = (base + child_delta).to(dtype=parent_a[name].dtype)
        overlap_count += int(overlap_mask.sum().item())
        total_count += int(overlap_mask.numel())
    return offspring, {
        "threshold": threshold,
        "overlap_fraction": float(overlap_count / max(total_count, 1)),
        "percentile": float(percentile),
    }


def delta_sparse_union_crossover(
    parent_a: dict[str, torch.Tensor],
    parent_b: dict[str, torch.Tensor],
    *,
    base_state: dict[str, torch.Tensor],
    percentile: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    threshold_a, threshold_b = _delta_magnitude_thresholds(parent_a, parent_b, base_state=base_state, percentile=percentile)
    offspring: dict[str, torch.Tensor] = {}
    active_count = 0
    total_count = 0
    for name in parent_a:
        base = base_state[name].float()
        delta_a = parent_a[name].float() - base
        delta_b = parent_b[name].float() - base
        active_a = delta_a.abs() >= threshold_a
        active_b = delta_b.abs() >= threshold_b
        both_active = active_a & active_b
        sign_agree = torch.sign(delta_a) == torch.sign(delta_b)
        average_mask = both_active & sign_agree
        choose_a = delta_a.abs() >= delta_b.abs()
        child_delta = torch.where(
            average_mask,
            0.5 * (delta_a + delta_b),
            torch.where(
                active_a & ~active_b,
                delta_a,
                torch.where(
                    active_b & ~active_a,
                    delta_b,
                    torch.where(both_active, torch.where(choose_a, delta_a, delta_b), torch.zeros_like(delta_a)),
                ),
            ),
        )
        offspring[name] = (base + child_delta).to(dtype=parent_a[name].dtype)
        active_count += int((active_a | active_b).sum().item())
        total_count += int(active_a.numel())
    return offspring, {
        "threshold": max(threshold_a, threshold_b),
        "threshold_a": threshold_a,
        "threshold_b": threshold_b,
        "overlap_fraction": float(active_count / max(total_count, 1)),
        "percentile": float(percentile),
    }


def delta_sparse_consensus_crossover(
    parent_a: dict[str, torch.Tensor],
    parent_b: dict[str, torch.Tensor],
    *,
    base_state: dict[str, torch.Tensor],
    percentile: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    threshold_a, threshold_b = _delta_magnitude_thresholds(parent_a, parent_b, base_state=base_state, percentile=percentile)
    offspring: dict[str, torch.Tensor] = {}
    overlap_count = 0
    total_count = 0
    for name in parent_a:
        base = base_state[name].float()
        delta_a = parent_a[name].float() - base
        delta_b = parent_b[name].float() - base
        active_a = delta_a.abs() >= threshold_a
        active_b = delta_b.abs() >= threshold_b
        consensus_mask = active_a & active_b & (torch.sign(delta_a) == torch.sign(delta_b))
        child_delta = torch.where(consensus_mask, 0.5 * (delta_a + delta_b), torch.zeros_like(delta_a))
        offspring[name] = (base + child_delta).to(dtype=parent_a[name].dtype)
        overlap_count += int(consensus_mask.sum().item())
        total_count += int(consensus_mask.numel())
    return offspring, {
        "threshold": max(threshold_a, threshold_b),
        "threshold_a": threshold_a,
        "threshold_b": threshold_b,
        "overlap_fraction": float(overlap_count / max(total_count, 1)),
        "percentile": float(percentile),
    }


def crossover_specs() -> dict[str, CrossoverSpec]:
    return {
        "parent_copy": CrossoverSpec(name="parent_copy", requires_base=False, uses_percentile=False),
        "weight_overlap": CrossoverSpec(name="weight_overlap", requires_base=False, uses_percentile=True),
        "delta_overlap": CrossoverSpec(name="delta_overlap", requires_base=True, uses_percentile=True),
        "delta_sparse_union": CrossoverSpec(name="delta_sparse_union", requires_base=True, uses_percentile=True),
        "delta_sparse_consensus": CrossoverSpec(name="delta_sparse_consensus", requires_base=True, uses_percentile=True),
        "sign_consensus": CrossoverSpec(name="sign_consensus", requires_base=False, uses_percentile=False),
        "delta_sign_consensus": CrossoverSpec(name="delta_sign_consensus", requires_base=True, uses_percentile=False),
        "tensor_swap": CrossoverSpec(name="tensor_swap", requires_base=False, uses_percentile=False),
        "layer_swap": CrossoverSpec(name="layer_swap", requires_base=False, uses_percentile=False),
        "delta_importance": CrossoverSpec(name="delta_importance", requires_base=True, uses_percentile=True),
    }


def crossover_state(
    parent_a: dict[str, torch.Tensor],
    parent_b: dict[str, torch.Tensor],
    *,
    strategy: str,
    percentile: float,
    seed: int,
    base_state: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    specs = crossover_specs()
    if strategy not in specs:
        raise ValueError(f"unsupported crossover strategy: {strategy}")
    spec = specs[strategy]
    if spec.requires_base and base_state is None:
        raise ValueError(f"strategy {strategy} requires base_state")

    if strategy == "weight_overlap":
        return overlap_crossover(parent_a, parent_b, percentile=percentile, seed=seed)
    if strategy == "parent_copy":
        return parent_copy_crossover(parent_a, parent_b, seed=seed)
    if strategy == "delta_overlap":
        assert base_state is not None
        return delta_overlap_crossover(parent_a, parent_b, base_state=base_state, percentile=percentile, seed=seed)
    if strategy == "delta_sparse_union":
        assert base_state is not None
        return delta_sparse_union_crossover(parent_a, parent_b, base_state=base_state, percentile=percentile, seed=seed)
    if strategy == "delta_sparse_consensus":
        assert base_state is not None
        return delta_sparse_consensus_crossover(parent_a, parent_b, base_state=base_state, percentile=percentile, seed=seed)
    if strategy == "sign_consensus":
        return sign_consensus_crossover(parent_a, parent_b, seed=seed)
    if strategy == "delta_sign_consensus":
        assert base_state is not None
        return delta_sign_consensus_crossover(parent_a, parent_b, base_state=base_state, seed=seed)
    if strategy == "tensor_swap":
        return tensor_swap_crossover(parent_a, parent_b, seed=seed)
    if strategy == "layer_swap":
        return layer_swap_crossover(parent_a, parent_b, seed=seed)
    if strategy == "delta_importance":
        assert base_state is not None
        return delta_importance_crossover(parent_a, parent_b, base_state=base_state, percentile=percentile, seed=seed)
    raise AssertionError(f"unreachable strategy dispatch: {strategy}")


def mutate_state(
    state: dict[str, torch.Tensor],
    *,
    mutation_std: float,
    mutation_fraction: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    if mutation_std <= 0.0 or mutation_fraction <= 0.0:
        return {name: value.clone() for name, value in state.items()}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    mutated: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        noise = torch.randn(value.shape, generator=generator, dtype=torch.float32) * mutation_std
        if mutation_fraction < 1.0:
            mask = torch.rand(value.shape, generator=generator, dtype=torch.float32) < mutation_fraction
            noise = noise * mask
        mutated[name] = (value.float() + noise).to(dtype=value.dtype)
    return mutated


def state_bpb(
    state: dict[str, torch.Tensor],
    *,
    model: StrongTransformerLM,
    eval_tokens: torch.Tensor,
    eval_starts: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    param_dtype: torch.dtype,
) -> float:
    load_parameter_state(model, state, device)
    return float(
        evaluate_bpb(
            model,
            tokens=eval_tokens,
            eval_starts=eval_starts,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            param_dtype=param_dtype,
        )["bpb"]
    )


def population_eval_metrics(
    states: list[dict[str, torch.Tensor]],
    *,
    model: StrongTransformerLM,
    eval_tokens: torch.Tensor,
    eval_starts: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    param_dtype: torch.dtype,
    population_chunk_size: int | None = None,
) -> list[dict[str, float]]:
    if not states:
        return []
    buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}
    total_tokens = 0
    loss_sums = torch.zeros((len(states),), dtype=torch.float64)

    def population_loss_sum(single_params: dict[str, torch.Tensor], input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        out = functional_call(model, (single_params, buffers), (input_ids,))
        logits = out["logits"]
        assert logits is not None
        return F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            target_ids.reshape(-1),
            reduction="sum",
        )

    chunks = iter_population_chunks(len(states), population_chunk_size)
    model.eval()
    with torch.no_grad():
        for idx in range(0, len(eval_starts), batch_size):
            batch_starts = eval_starts[idx : idx + batch_size]
            inputs, targets = batch_from_starts(eval_tokens, batch_starts, seq_len, device)
            total_tokens += int(targets.numel())
            for start, chunk_size in chunks:
                params = stack_population_states(
                    states[start : start + chunk_size],
                    device=device,
                    param_dtype=param_dtype,
                )
                with maybe_autocast(device, param_dtype):
                    with vmap_safe_sdpa_context():
                        batch_losses = vmap(population_loss_sum, in_dims=(0, None, None))(params, inputs, targets)
                loss_sums[start : start + chunk_size] += batch_losses.detach().cpu().to(torch.float64)
    metrics = []
    for loss_sum in loss_sums.tolist():
        mean_loss = float(loss_sum / max(total_tokens, 1))
        metrics.append(
            {
                "loss": mean_loss,
                "bpb": float(mean_loss / math.log(2.0)),
                "tokens": float(total_tokens),
            }
        )
    return metrics


def population_bpbs(
    states: list[dict[str, torch.Tensor]],
    *,
    model: StrongTransformerLM,
    eval_tokens: torch.Tensor,
    eval_starts: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    param_dtype: torch.dtype,
    population_chunk_size: int | None = None,
) -> list[float]:
    return [
        float(metric["bpb"])
        for metric in population_eval_metrics(
            states,
            model=model,
            eval_tokens=eval_tokens,
            eval_starts=eval_starts,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            param_dtype=param_dtype,
            population_chunk_size=population_chunk_size,
        )
    ]


def population_replay_eval_metrics(
    records: list[dict[str, Any]],
    *,
    model: StrongTransformerLM,
    replay_tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    device: torch.device,
    param_dtype: torch.dtype,
) -> tuple[list[dict[str, float]], dict[str, Any] | None]:
    if not records or eval_batches <= 0:
        return [], None
    replay_pool = merge_replay_buffers(
        [list(record.get("replay", [])) for record in records],
        topk=max(eval_batches * max(batch_size, 1), 0),
    )
    replay_starts = replay_eval_starts(replay_pool, batch_size=batch_size, eval_batches=eval_batches)
    if not replay_starts:
        return [], None
    metrics = population_eval_metrics(
        [record["state"] for record in records],
        model=model,
        eval_tokens=replay_tokens,
        eval_starts=replay_starts,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        param_dtype=param_dtype,
    )
    return metrics, {
        "eval_batches": int(eval_batches),
        "num_starts": int(len(replay_starts)),
        "buffer": summarize_replay_buffer(replay_pool),
    }


def ensemble_bpb(
    states: list[dict[str, torch.Tensor]],
    *,
    model: StrongTransformerLM,
    eval_tokens: torch.Tensor,
    eval_starts: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    param_dtype: torch.dtype,
    population_chunk_size: int | None = None,
) -> float:
    if not states:
        raise ValueError("ensemble_bpb requires at least one state")
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}
    chunks = iter_population_chunks(len(states), population_chunk_size)

    def population_probs(single_params: dict[str, torch.Tensor], input_ids: torch.Tensor) -> torch.Tensor:
        out = functional_call(model, (single_params, buffers), (input_ids,))
        logits = out["logits"]
        assert logits is not None
        return torch.softmax(logits.float(), dim=-1)

    with torch.no_grad():
        for idx in range(0, len(eval_starts), batch_size):
            batch_starts = eval_starts[idx : idx + batch_size]
            inputs, targets = batch_from_starts(eval_tokens, batch_starts, seq_len, device)
            prob_sum = torch.zeros((inputs.shape[0], inputs.shape[1], model.vocab_size), device=device, dtype=torch.float32)
            for start, chunk_size in chunks:
                params = stack_population_states(
                    states[start : start + chunk_size],
                    device=device,
                    param_dtype=param_dtype,
                )
                with maybe_autocast(device, param_dtype):
                    with vmap_safe_sdpa_context():
                        chunk_probs = vmap(population_probs, in_dims=(0, None))(params, inputs)
                prob_sum.add_(chunk_probs.sum(dim=0).float())
            ensemble_log_probs = (prob_sum / float(len(states))).clamp_min(1e-8).log()
            total_loss += float(
                F.nll_loss(
                    ensemble_log_probs.reshape(-1, model.vocab_size),
                    targets.reshape(-1),
                    reduction="sum",
                ).item()
            )
            total_tokens += int(targets.numel())
    mean_loss = total_loss / max(total_tokens, 1)
    return float(mean_loss / math.log(2.0))


def build_member_record(
    *,
    member_id: int,
    state: dict[str, torch.Tensor],
    model: StrongTransformerLM,
    train_tokens: torch.Tensor,
    train_starts: list[int],
    eval_tokens: torch.Tensor,
    eval_starts: list[int],
    cfg: EvoBenchmarkConfig,
    train_seconds: float,
    seed: int,
    device: torch.device,
    param_dtype: torch.dtype,
) -> dict[str, Any]:
    load_parameter_state(model, state, device)
    train_info, replay_entries = train_for_seconds(
        model,
        train_tokens=train_tokens,
        train_starts=train_starts,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        base_lr=cfg.base_lr,
        weight_decay=cfg.weight_decay,
        grad_clip_norm=cfg.grad_clip_norm,
        train_seconds=train_seconds,
        seed=seed,
        device=device,
        param_dtype=param_dtype,
        hard_replay_topk=cfg.hard_replay_topk,
    )
    trained_state = parameter_state_dict(model)
    eval_info = evaluate_bpb(
        model,
        tokens=eval_tokens,
        eval_starts=eval_starts,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        device=device,
        param_dtype=param_dtype,
    )
    return {
        "member_id": int(member_id),
        "seed": int(seed),
        "train": asdict(train_info),
        "eval": eval_info,
        "current_eval": eval_info,
        "best_eval": eval_info,
        "archive_summary": {
            "best_stage_index": 0,
            "current_stage_index": 0,
        },
        "replay_summary": summarize_replay_buffer(replay_entries),
        "replay_eval": None,
        "state": trained_state,
        "best_state": trained_state,
        "replay": replay_entries,
    }


def build_member_records_parallel(
    *,
    copies: int,
    state: dict[str, torch.Tensor],
    model: StrongTransformerLM,
    train_tokens: torch.Tensor,
    train_starts: list[int],
    eval_tokens: torch.Tensor,
    eval_starts: list[int],
    cfg: EvoBenchmarkConfig,
    train_seconds: float,
    device: torch.device,
    param_dtype: torch.dtype,
) -> list[dict[str, Any]]:
    member_seeds = [cfg.seed + 101 + member_id for member_id in range(copies)]
    trained_states, train_infos, replay_buffers = train_population_for_seconds(
        model,
        states=[state] * copies,
        train_tokens=train_tokens,
        train_starts=train_starts,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        base_lr=cfg.base_lr,
        weight_decay=cfg.weight_decay,
        grad_clip_norm=cfg.grad_clip_norm,
        train_seconds=train_seconds,
        seeds=member_seeds,
        device=device,
        param_dtype=param_dtype,
        hard_replay_topk=cfg.hard_replay_topk,
    )
    eval_infos = population_eval_metrics(
        trained_states,
        model=model,
        eval_tokens=eval_tokens,
        eval_starts=eval_starts,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        device=device,
        param_dtype=param_dtype,
    )
    return [
        {
            "member_id": int(member_id),
            "seed": int(member_seeds[member_id]),
            "train": asdict(train_infos[member_id]),
            "eval": eval_infos[member_id],
            "current_eval": eval_infos[member_id],
            "best_eval": eval_infos[member_id],
            "archive_summary": {
                "best_stage_index": 0,
                "current_stage_index": 0,
            },
            "replay_summary": summarize_replay_buffer(replay_buffers[member_id]),
            "replay_eval": None,
            "state": trained_states[member_id],
            "best_state": trained_states[member_id],
            "replay": replay_buffers[member_id],
        }
        for member_id in range(copies)
    ]


def build_member_ensemble_results(
    *,
    trained_members: list[dict[str, Any]],
    ensemble_topks: tuple[int, ...],
    model: StrongTransformerLM,
    val_tokens: torch.Tensor,
    eval_starts: list[int],
    test_tokens: torch.Tensor,
    test_eval_starts: list[int],
    cfg: EvoBenchmarkConfig,
    device: torch.device,
    param_dtype: torch.dtype,
) -> list[dict[str, Any]]:
    member_ensemble_results = []
    requested_topks = sorted({topk for topk in ensemble_topks if topk > 0})
    if requested_topks:
        sorted_members = sorted(trained_members, key=lambda member: float(member["eval"]["bpb"]))
        for topk in requested_topks:
            actual_topk = min(topk, len(sorted_members))
            selected_states = [member["state"] for member in sorted_members[:actual_topk]]
            member_ensemble_results.append(
                {
                    "topk": int(actual_topk),
                    "val_bpb": ensemble_bpb(
                        selected_states,
                        model=model,
                        eval_tokens=val_tokens,
                        eval_starts=eval_starts,
                        batch_size=cfg.batch_size,
                        seq_len=cfg.seq_len,
                        device=device,
                        param_dtype=param_dtype,
                    ),
                    "test_bpb": ensemble_bpb(
                        selected_states,
                        model=model,
                        eval_tokens=test_tokens,
                        eval_starts=test_eval_starts,
                        batch_size=cfg.batch_size,
                        seq_len=cfg.seq_len,
                        device=device,
                        param_dtype=param_dtype,
                    ),
                }
            )
    return member_ensemble_results


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def std_or_none(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return float(math.sqrt(max(variance, 0.0)))


def build_signal_members(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for member_idx, record in enumerate(records):
        current_eval = record.get("current_eval")
        if current_eval is None:
            continue
        members.append(
            {
                "member_id": int(member_idx),
                "seed": int(record["stages"][-1]["seed"]) if record.get("stages") else int(member_idx),
                "eval": current_eval,
                "state": record["state"],
            }
        )
    return members


def approximate_state_distance(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    max_samples: int = 131_072,
) -> dict[str, float]:
    diffs = [(left[name].float() - right[name].float()).abs().reshape(-1) for name in left]
    scales = [0.5 * (left[name].float().abs() + right[name].float().abs()).reshape(-1) for name in left]
    nonempty_diffs = [tensor for tensor in diffs if tensor.numel() > 0]
    nonempty_scales = [tensor for tensor in scales if tensor.numel() > 0]
    total_values = sum(int(tensor.numel()) for tensor in nonempty_diffs)
    if total_values == 0:
        return {"mean_abs": 0.0, "relative_mean_abs": 0.0}

    if total_values <= max_samples:
        diff_sample = torch.cat(nonempty_diffs)
        scale_sample = torch.cat(nonempty_scales) if nonempty_scales else torch.zeros_like(diff_sample)
    else:
        counts = np.array([int(tensor.numel()) for tensor in nonempty_diffs], dtype=np.int64)
        exact = max_samples * counts / max(total_values, 1)
        allocation = np.floor(exact).astype(np.int64)
        remaining = int(max_samples - allocation.sum())
        if remaining > 0:
            for index in np.argsort(-(exact - allocation))[:remaining]:
                allocation[index] += 1
        diff_parts: list[torch.Tensor] = []
        scale_parts: list[torch.Tensor] = []
        for diff_tensor, scale_tensor, sample_count in zip(nonempty_diffs, nonempty_scales, allocation.tolist()):
            if sample_count <= 0:
                continue
            if int(diff_tensor.numel()) <= sample_count:
                diff_parts.append(diff_tensor)
                scale_parts.append(scale_tensor)
                continue
            positions = (
                (torch.arange(sample_count, dtype=torch.float64) + 0.5)
                * (float(diff_tensor.numel()) / float(sample_count))
            ).floor()
            positions = positions.clamp_(0, int(diff_tensor.numel()) - 1).to(torch.int64)
            diff_parts.append(diff_tensor.index_select(0, positions))
            scale_parts.append(scale_tensor.index_select(0, positions))
        diff_sample = torch.cat(diff_parts) if diff_parts else torch.zeros((0,), dtype=torch.float32)
        scale_sample = torch.cat(scale_parts) if scale_parts else torch.zeros_like(diff_sample)

    mean_abs = float(diff_sample.mean().item()) if diff_sample.numel() > 0 else 0.0
    mean_scale = float(scale_sample.mean().item()) if scale_sample.numel() > 0 else 0.0
    relative_mean_abs = float(mean_abs / max(mean_scale, 1e-8))
    return {
        "mean_abs": mean_abs,
        "relative_mean_abs": relative_mean_abs,
    }


def pairwise_state_distance_summary(
    records: list[dict[str, Any]],
    *,
    max_pairs: int = 256,
    max_samples_per_pair: int = 131_072,
) -> dict[str, float | int | None]:
    states = [record["state"] for record in records if record.get("state") is not None]
    if len(states) < 2:
        return {
            "pair_count": 0,
            "pairwise_distance_mean": None,
            "pairwise_distance_std": None,
            "pairwise_distance_min": None,
            "pairwise_distance_max": None,
            "pairwise_relative_distance_mean": None,
            "pairwise_relative_distance_std": None,
        }

    pair_indices = list(itertools.combinations(range(len(states)), 2))
    if len(pair_indices) > max_pairs:
        stride = max(1, len(pair_indices) // max_pairs)
        pair_indices = pair_indices[::stride][:max_pairs]

    mean_abs_values: list[float] = []
    relative_values: list[float] = []
    for left_idx, right_idx in pair_indices:
        distance = approximate_state_distance(
            states[left_idx],
            states[right_idx],
            max_samples=max_samples_per_pair,
        )
        mean_abs_values.append(float(distance["mean_abs"]))
        relative_values.append(float(distance["relative_mean_abs"]))

    return {
        "pair_count": int(len(pair_indices)),
        "pairwise_distance_mean": mean_or_none(mean_abs_values),
        "pairwise_distance_std": std_or_none(mean_abs_values),
        "pairwise_distance_min": None if not mean_abs_values else float(min(mean_abs_values)),
        "pairwise_distance_max": None if not mean_abs_values else float(max(mean_abs_values)),
        "pairwise_relative_distance_mean": mean_or_none(relative_values),
        "pairwise_relative_distance_std": std_or_none(relative_values),
    }


def compute_committee_signals(
    *,
    records: list[dict[str, Any]],
    member_ensemble_results: list[dict[str, Any]],
    previous_best_member_bpb: float | None,
) -> dict[str, Any]:
    current_bpbs = [float(record["current_eval"]["bpb"]) for record in records if record.get("current_eval") is not None]
    best_member_bpb = min(current_bpbs) if current_bpbs else None
    mean_member_bpb = mean_or_none(current_bpbs)
    member_bpb_std = std_or_none(current_bpbs)
    replay_bpbs = [
        float(record["replay_eval"]["bpb"])
        for record in records
        if isinstance(record.get("replay_eval"), dict) and record["replay_eval"].get("bpb") is not None
    ]
    replay_disagreement_std = std_or_none(replay_bpbs)
    archive_flags = [
        bool(record["stages"][-1].get("archive_improved"))
        for record in records
        if record.get("stages")
    ]
    archive_hit_rate = float(sum(1 for flag in archive_flags if flag) / len(archive_flags)) if archive_flags else None
    best_ensemble = min(member_ensemble_results, key=lambda row: row.get("val_bpb", float("inf")), default=None)
    top2_ensemble = next((row for row in sorted(member_ensemble_results, key=lambda row: int(row.get("topk", 0))) if int(row.get("topk", 0)) >= 2), None)
    ensemble_gain = None
    top2_gain = None
    winner_concentration = None
    if best_member_bpb is not None and best_ensemble is not None and best_ensemble.get("val_bpb") is not None:
        ensemble_gain = float(best_member_bpb - float(best_ensemble["val_bpb"]))
    if best_member_bpb is not None and top2_ensemble is not None and top2_ensemble.get("val_bpb") is not None:
        top2_gain = float(best_member_bpb - float(top2_ensemble["val_bpb"]))
    if ensemble_gain is not None and top2_gain is not None and ensemble_gain > 1e-9:
        winner_concentration = float(max(0.0, min(1.0, top2_gain / ensemble_gain)))
    depth_gain = None
    if best_member_bpb is not None and previous_best_member_bpb is not None:
        depth_gain = float(previous_best_member_bpb - best_member_bpb)
    distance_summary = pairwise_state_distance_summary(records)
    return {
        "population_size": int(len(records)),
        "best_member_bpb": best_member_bpb,
        "mean_member_bpb": mean_member_bpb,
        "member_bpb_std": member_bpb_std,
        "best_ensemble_topk": None if best_ensemble is None else int(best_ensemble.get("topk", 0)),
        "best_ensemble_bpb": None if best_ensemble is None else best_ensemble.get("val_bpb"),
        "ensemble_gain": ensemble_gain,
        "top2_ensemble_gain": top2_gain,
        "winner_concentration": winner_concentration,
        "archive_hit_rate": archive_hit_rate,
        "replay_disagreement_std": replay_disagreement_std,
        "depth_gain": depth_gain,
        **distance_summary,
    }


def recommend_committee_action(
    *,
    signals: dict[str, Any],
    current_copies: int,
    min_copies: int,
    max_copies: int,
    widen_factor: int,
    narrow_factor: int,
    depth_gain_threshold: float,
    breadth_gain_threshold: float,
    archive_hit_rate_threshold: float,
    winner_concentration_threshold: float,
    replay_disagreement_threshold: float,
    pairwise_distance_threshold: float,
) -> dict[str, Any]:
    next_copies = int(current_copies)
    action = "hold"
    reason = "signals_balanced"
    depth_gain = signals.get("depth_gain")
    ensemble_gain = signals.get("ensemble_gain")
    archive_hit_rate = signals.get("archive_hit_rate")
    winner_concentration = signals.get("winner_concentration")
    replay_disagreement = signals.get("replay_disagreement_std")
    pairwise_relative_distance = signals.get("pairwise_relative_distance_mean")

    if (
        ensemble_gain is not None
        and replay_disagreement is not None
        and ensemble_gain >= breadth_gain_threshold
        and replay_disagreement >= replay_disagreement_threshold
        and current_copies < max_copies
    ):
        if pairwise_relative_distance is not None and pairwise_relative_distance < pairwise_distance_threshold:
            action = "hold"
            next_copies = int(current_copies)
            reason = "ensemble_gain_but_population_converged"
        else:
            action = "widen"
            next_copies = min(max_copies, max(current_copies + 1, current_copies * max(widen_factor, 1)))
            reason = "ensemble_gain_and_disagreement"
    elif (
        winner_concentration is not None
        and archive_hit_rate is not None
        and winner_concentration >= winner_concentration_threshold
        and archive_hit_rate <= archive_hit_rate_threshold
        and current_copies > min_copies
    ):
        action = "narrow"
        next_copies = max(min_copies, max(1, current_copies // max(narrow_factor, 1)))
        reason = "winner_concentration_with_low_archive_hits"
    elif depth_gain is not None and depth_gain >= depth_gain_threshold:
        action = "stay_narrow"
        next_copies = max(min_copies, min(current_copies, max_copies))
        reason = "depth_still_paying"

    return {
        "action": action,
        "reason": reason,
        "next_copies": int(next_copies),
        "thresholds": {
            "depth_gain": float(depth_gain_threshold),
            "breadth_gain": float(breadth_gain_threshold),
            "archive_hit_rate": float(archive_hit_rate_threshold),
            "winner_concentration": float(winner_concentration_threshold),
            "replay_disagreement": float(replay_disagreement_threshold),
            "pairwise_distance": float(pairwise_distance_threshold),
        },
    }


def aggregate_stage_history(stage_history: list[dict[str, Any]]) -> dict[str, Any]:
    if not stage_history:
        return asdict(TrainingOutcome(0.0, 0.0, 0, float("nan"), 0))
    total_target = sum(float(stage["train"]["train_seconds_target"]) for stage in stage_history)
    total_actual = sum(float(stage["train"]["train_seconds_actual"]) for stage in stage_history)
    total_steps = sum(int(stage["train"]["steps_completed"]) for stage in stage_history)
    total_tokens = sum(int(stage["train"]["tokens_seen"]) for stage in stage_history)
    return asdict(
        TrainingOutcome(
            train_seconds_target=float(total_target),
            train_seconds_actual=float(total_actual),
            steps_completed=int(total_steps),
            final_loss=float(stage_history[-1]["train"]["final_loss"]),
            tokens_seen=int(total_tokens),
        )
    )


def expand_population_records(
    records: list[dict[str, Any]],
    *,
    target_size: int,
    spawn_noise_std: float,
    seed: int,
    use_best_archive_for_spawn: bool,
    hard_replay_topk: int,
) -> list[dict[str, Any]]:
    if target_size < len(records):
        raise ValueError(f"target_size={target_size} < current_size={len(records)}")
    if not records:
        return []
    expanded: list[dict[str, Any]] = []
    for idx in range(target_size):
        parent = records[idx % len(records)]
        parent_state = parent.get("best_state", parent["state"]) if use_best_archive_for_spawn else parent["state"]
        expanded.append(
            {
                "lineage": parent["lineage"] + [int(idx % len(records))],
                "state": clone_with_perturbation(parent_state, noise_std=spawn_noise_std, seed=seed + idx),
                "best_state": parent.get("best_state", parent["state"]),
                "best_eval": dict(parent.get("best_eval", parent.get("eval", {}))),
                "current_eval": dict(parent.get("current_eval", parent.get("eval", {}))),
                "archive_summary": dict(parent.get("archive_summary", {})),
                "replay": merge_replay_buffers([list(parent.get("replay", []))], topk=hard_replay_topk),
                "stages": [dict(stage) for stage in parent["stages"]],
            }
        )
    return expanded


def select_top_population_records(
    records: list[dict[str, Any]],
    *,
    target_size: int,
    model: StrongTransformerLM,
    eval_tokens: torch.Tensor,
    eval_starts: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    param_dtype: torch.dtype,
) -> list[dict[str, Any]]:
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    if target_size >= len(records):
        return list(records)
    eval_infos = population_eval_metrics(
        [record["state"] for record in records],
        model=model,
        eval_tokens=eval_tokens,
        eval_starts=eval_starts,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        param_dtype=param_dtype,
    )
    ranked_indices = sorted(range(len(records)), key=lambda idx: float(eval_infos[idx]["bpb"]))[:target_size]
    selected: list[dict[str, Any]] = []
    for rank, idx in enumerate(ranked_indices):
        record = records[idx]
        selected.append(
            {
                "lineage": record["lineage"] + [f"prune{target_size}", int(rank)],
                "state": record["state"],
                "best_state": record.get("best_state", record["state"]),
                "best_eval": dict(record.get("best_eval", record.get("eval", {}))),
                "current_eval": dict(record.get("current_eval", record.get("eval", {}))),
                "archive_summary": dict(record.get("archive_summary", {})),
                "replay": list(record.get("replay", [])),
                "stages": [dict(stage) for stage in record["stages"]],
            }
        )
    return selected


def build_state_layout(base_state: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    layout: list[dict[str, Any]] = []
    offset = 0
    for name, tensor in base_state.items():
        numel = int(tensor.numel())
        layout.append(
            {
                "name": name,
                "shape": tuple(tensor.shape),
                "dtype": tensor.dtype,
                "start": int(offset),
                "end": int(offset + numel),
            }
        )
        offset += numel
    return layout


def flatten_state_with_layout(
    state: dict[str, torch.Tensor],
    layout: list[dict[str, Any]],
) -> torch.Tensor:
    parts = [state[str(entry["name"])].reshape(-1).float() for entry in layout]
    return torch.cat(parts) if parts else torch.zeros((0,), dtype=torch.float32)


def unflatten_state_with_layout(
    flat: torch.Tensor,
    layout: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for entry in layout:
        name = str(entry["name"])
        start = int(entry["start"])
        end = int(entry["end"])
        shape = tuple(entry["shape"])
        dtype = entry["dtype"]
        state[name] = flat[start:end].reshape(shape).to(dtype=dtype).cpu()
    return state


def flatten_branch_deltas(
    base_state: dict[str, torch.Tensor],
    branch_states: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    layout = build_state_layout(base_state)
    base_flat = flatten_state_with_layout(base_state, layout)
    branch_flats = [flatten_state_with_layout(state, layout) for state in branch_states]
    if not branch_flats:
        return torch.zeros((0, 0), dtype=torch.float32), base_flat, layout
    delta_matrix = torch.stack([branch_flat - base_flat for branch_flat in branch_flats], dim=0)
    return delta_matrix, base_flat, layout


def pairwise_delta_metrics(delta_matrix: torch.Tensor, *, support_topk: int) -> dict[str, Any]:
    branch_count, width = delta_matrix.shape
    if branch_count < 2 or width == 0:
        return {
            "branch_count": int(branch_count),
            "num_params": int(width),
            "pairwise_l2_mean": None,
            "pairwise_l2_std": None,
            "pairwise_cosine_mean": None,
            "pairwise_cosine_std": None,
            "topk_support_jaccard_mean": None,
            "effective_rank_90": None,
            "singular_values": [],
        }

    l2_values: list[float] = []
    cosine_values: list[float] = []
    jaccards: list[float] = []
    topk = max(1, min(int(support_topk), width))
    top_supports = [
        set(torch.topk(delta.abs(), k=topk, largest=True).indices.tolist())
        for delta in delta_matrix
    ]
    for left_idx, right_idx in itertools.combinations(range(branch_count), 2):
        left = delta_matrix[left_idx]
        right = delta_matrix[right_idx]
        l2_values.append(float((left - right).norm().item()))
        denom = float(left.norm().item() * right.norm().item())
        cosine = 0.0 if denom <= 1e-8 else float(torch.dot(left, right).item() / denom)
        cosine_values.append(cosine)
        intersection = len(top_supports[left_idx] & top_supports[right_idx])
        union = len(top_supports[left_idx] | top_supports[right_idx])
        jaccards.append(float(intersection / max(union, 1)))

    singular_values = torch.linalg.svdvals(delta_matrix)
    energy = singular_values.square()
    energy_total = float(energy.sum().item())
    effective_rank_90 = None
    if energy_total > 0.0:
        cumulative = torch.cumsum(energy, dim=0) / energy_total
        effective_rank_90 = int(torch.searchsorted(cumulative, torch.tensor(0.90, dtype=cumulative.dtype)).item() + 1)

    return {
        "branch_count": int(branch_count),
        "num_params": int(width),
        "pairwise_l2_mean": mean_or_none(l2_values),
        "pairwise_l2_std": std_or_none(l2_values),
        "pairwise_cosine_mean": mean_or_none(cosine_values),
        "pairwise_cosine_std": std_or_none(cosine_values),
        "topk_support_jaccard_mean": mean_or_none(jaccards),
        "effective_rank_90": effective_rank_90,
        "singular_values": [float(value) for value in singular_values[: min(8, singular_values.numel())].tolist()],
    }


def reconstruct_branch_states(
    reconstructed_deltas: torch.Tensor,
    *,
    base_flat: torch.Tensor,
    layout: list[dict[str, Any]],
) -> list[dict[str, torch.Tensor]]:
    states: list[dict[str, torch.Tensor]] = []
    for branch_delta in reconstructed_deltas:
        states.append(unflatten_state_with_layout(base_flat + branch_delta, layout))
    return states


def compress_branch_sparse(
    delta_matrix: torch.Tensor,
    *,
    budget_bytes: int,
    value_bytes: int = 2,
    index_bytes: int = 4,
) -> tuple[torch.Tensor, dict[str, Any]]:
    branch_count, width = delta_matrix.shape
    if branch_count == 0 or width == 0 or budget_bytes <= 0:
        return torch.zeros_like(delta_matrix), {"bytes_used": 0, "support_per_branch": 0}
    bytes_per_entry = value_bytes + index_bytes
    support_per_branch = max(0, min(width, budget_bytes // max(branch_count * bytes_per_entry, 1)))
    if support_per_branch <= 0:
        return torch.zeros_like(delta_matrix), {"bytes_used": 0, "support_per_branch": 0}
    reconstructed = torch.zeros_like(delta_matrix)
    for branch_idx in range(branch_count):
        keep = torch.topk(delta_matrix[branch_idx].abs(), k=support_per_branch, largest=True).indices
        reconstructed[branch_idx, keep] = delta_matrix[branch_idx, keep]
    bytes_used = int(branch_count * support_per_branch * bytes_per_entry)
    return reconstructed, {
        "bytes_used": bytes_used,
        "support_per_branch": int(support_per_branch),
    }


def compress_shared_union_sparse(
    delta_matrix: torch.Tensor,
    *,
    budget_bytes: int,
    value_bytes: int = 2,
    index_bytes: int = 4,
) -> tuple[torch.Tensor, dict[str, Any]]:
    branch_count, width = delta_matrix.shape
    if branch_count == 0 or width == 0 or budget_bytes <= 0:
        return torch.zeros_like(delta_matrix), {"bytes_used": 0, "shared_support": 0}
    bytes_per_coordinate = index_bytes + branch_count * value_bytes
    shared_support = max(0, min(width, budget_bytes // max(bytes_per_coordinate, 1)))
    if shared_support <= 0:
        return torch.zeros_like(delta_matrix), {"bytes_used": 0, "shared_support": 0}
    scores = delta_matrix.abs().mean(dim=0)
    keep = torch.topk(scores, k=shared_support, largest=True).indices
    reconstructed = torch.zeros_like(delta_matrix)
    reconstructed[:, keep] = delta_matrix[:, keep]
    bytes_used = int(shared_support * bytes_per_coordinate)
    return reconstructed, {
        "bytes_used": bytes_used,
        "shared_support": int(shared_support),
    }


def compress_shared_basis(
    delta_matrix: torch.Tensor,
    *,
    budget_bytes: int,
    rank: int,
    value_bytes: int = 2,
    index_bytes: int = 4,
) -> tuple[torch.Tensor, dict[str, Any]] | None:
    branch_count, width = delta_matrix.shape
    if branch_count == 0 or width == 0 or budget_bytes <= 0 or rank <= 0:
        return None
    coeff_bytes = branch_count * rank * value_bytes
    remaining = budget_bytes - coeff_bytes
    bytes_per_coordinate = index_bytes + rank * value_bytes
    if remaining <= 0:
        return None
    shared_support = max(0, min(width, remaining // max(bytes_per_coordinate, 1)))
    if shared_support <= 0:
        return None
    scores = delta_matrix.abs().mean(dim=0)
    keep = torch.topk(scores, k=shared_support, largest=True).indices
    support_matrix = delta_matrix[:, keep]
    max_rank = min(rank, int(min(support_matrix.shape)))
    if max_rank <= 0:
        return None
    u, s, vh = torch.linalg.svd(support_matrix, full_matrices=False)
    reconstructed_support = (u[:, :max_rank] * s[:max_rank]) @ vh[:max_rank, :]
    reconstructed = torch.zeros_like(delta_matrix)
    reconstructed[:, keep] = reconstructed_support
    bytes_used = int(shared_support * bytes_per_coordinate + coeff_bytes)
    return reconstructed, {
        "bytes_used": bytes_used,
        "shared_support": int(shared_support),
        "rank": int(max_rank),
    }


def evaluate_compressed_committee(
    reconstructed_states: list[dict[str, torch.Tensor]],
    *,
    model: StrongTransformerLM,
    val_tokens: torch.Tensor,
    eval_starts: list[int],
    test_tokens: torch.Tensor,
    test_eval_starts: list[int],
    cfg: EvoBenchmarkConfig,
    device: torch.device,
    param_dtype: torch.dtype,
) -> dict[str, float]:
    val_bpb = ensemble_bpb(
        reconstructed_states,
        model=model,
        eval_tokens=val_tokens,
        eval_starts=eval_starts,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        device=device,
        param_dtype=param_dtype,
    )
    test_bpb = ensemble_bpb(
        reconstructed_states,
        model=model,
        eval_tokens=test_tokens,
        eval_starts=test_eval_starts,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        device=device,
        param_dtype=param_dtype,
    )
    return {"val_bpb": float(val_bpb), "test_bpb": float(test_bpb)}


def run_committee_schedule(
    *,
    cfg: EvoBenchmarkConfig,
    enwik8_path: Path,
    stage_copies: tuple[int, ...],
    stage_train_seconds: tuple[float, ...],
    ensemble_topks: tuple[int, ...],
    eval_batches: int,
    spawn_noise_std: float,
    include_states: bool = False,
    device: torch.device,
    param_dtype: torch.dtype,
) -> dict[str, Any]:
    if not stage_copies or not stage_train_seconds or len(stage_copies) != len(stage_train_seconds):
        raise ValueError("stage_copies and stage_train_seconds must be non-empty and the same length")
    if any(copies <= 0 for copies in stage_copies):
        raise ValueError("stage_copies must all be positive")

    train_tokens, val_tokens, test_tokens, dataset_meta = prepare_tokenized_enwik8_splits(
        enwik8_path,
        device=device,
        cache_on_device=cfg.cache_dataset_on_device,
        tokenization_mode=cfg.tokenization_mode,
        tokenizer_name=cfg.tokenizer_name,
        tokenizer_model_path=cfg.tokenizer_model_path,
    )
    train_starts = build_lm_starts(int(train_tokens.numel()), cfg.seq_len, cfg.stride)
    val_starts = build_lm_starts(int(val_tokens.numel()), cfg.seq_len, cfg.stride)
    test_starts = build_lm_starts(int(test_tokens.numel()), cfg.seq_len, cfg.stride)
    if not train_starts or not val_starts or not test_starts:
        raise ValueError("enwik8 split is too small for the requested seq_len/stride")

    eval_starts = choose_eval_starts(val_starts, batch_size=cfg.batch_size, eval_batches=eval_batches, seed=cfg.seed + 9000)
    test_eval_starts = choose_eval_starts(test_starts, batch_size=cfg.batch_size, eval_batches=eval_batches, seed=cfg.seed + 9001)
    model = build_model(cfg, device, param_dtype)
    base_init = parameter_state_dict(model)

    records = [
        {
            "lineage": [member_id],
            "state": base_init,
            "best_state": base_init,
            "best_eval": None,
            "current_eval": None,
            "archive_summary": {
                "best_stage_index": None,
                "current_stage_index": None,
            },
            "replay": [],
            "replay_eval": None,
            "stages": [],
        }
        for member_id in range(stage_copies[0])
    ]
    stage_signals: list[dict[str, Any]] = []
    previous_best_member_bpb: float | None = None

    for stage_idx, (copies, stage_seconds) in enumerate(zip(stage_copies, stage_train_seconds)):
        if stage_idx > 0:
            if copies > len(records):
                records = expand_population_records(
                    records,
                    target_size=copies,
                    spawn_noise_std=spawn_noise_std,
                    seed=cfg.seed + 50_000 + stage_idx * 1_000,
                    use_best_archive_for_spawn=cfg.use_best_archive_for_spawn,
                    hard_replay_topk=cfg.hard_replay_topk,
                )
            elif copies < len(records):
                records = select_top_population_records(
                    records,
                    target_size=copies,
                    model=model,
                    eval_tokens=val_tokens,
                    eval_starts=eval_starts,
                    batch_size=cfg.batch_size,
                    seq_len=cfg.seq_len,
                    device=device,
                    param_dtype=param_dtype,
                )
        replay_eval_infos, replay_eval_meta = population_replay_eval_metrics(
            records,
            model=model,
            replay_tokens=train_tokens,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            eval_batches=cfg.hard_replay_eval_batches,
            device=device,
            param_dtype=param_dtype,
        )
        states = [record["state"] for record in records]
        stage_seeds = [cfg.seed + 10_000 * (stage_idx + 1) + member_idx for member_idx in range(len(records))]
        trained_states, train_infos, replay_buffers = train_population_for_seconds(
            model,
            states=states,
            train_tokens=train_tokens,
            train_starts=train_starts,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
            base_lr=cfg.base_lr,
            weight_decay=cfg.weight_decay,
            grad_clip_norm=cfg.grad_clip_norm,
            train_seconds=stage_seconds,
            seeds=stage_seeds,
            device=device,
            param_dtype=param_dtype,
            hard_replay_topk=cfg.hard_replay_topk,
        )
        current_eval_infos = population_eval_metrics(
            trained_states,
            model=model,
            eval_tokens=val_tokens,
            eval_starts=eval_starts,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            device=device,
            param_dtype=param_dtype,
        )
        for member_idx, record in enumerate(records):
            record["state"] = trained_states[member_idx]
            record["current_eval"] = current_eval_infos[member_idx]
            record["replay"] = merge_replay_buffers(
                [list(record.get("replay", [])), replay_buffers[member_idx]],
                topk=cfg.hard_replay_topk,
            )
            record["replay_eval"] = replay_eval_infos[member_idx] if replay_eval_infos else None
            archive_improved = False
            best_eval = record.get("best_eval")
            current_eval = current_eval_infos[member_idx]
            if best_eval is None or float(current_eval["bpb"]) < float(best_eval["bpb"]):
                record["best_state"] = trained_states[member_idx]
                record["best_eval"] = current_eval
                record["archive_summary"] = {
                    "best_stage_index": int(stage_idx),
                    "current_stage_index": int(stage_idx),
                }
                archive_improved = True
            else:
                best_stage_index = record.get("archive_summary", {}).get("best_stage_index")
                record["archive_summary"] = {
                    "best_stage_index": int(stage_idx if best_stage_index is None else best_stage_index),
                    "current_stage_index": int(stage_idx),
                }
            record["stages"].append(
                {
                    "stage_index": int(stage_idx),
                    "stage_copies": int(copies),
                    "stage_train_seconds_target": float(stage_seconds),
                    "seed": int(stage_seeds[member_idx]),
                    "train": asdict(train_infos[member_idx]),
                    "current_eval": current_eval,
                    "pre_stage_replay_eval": replay_eval_infos[member_idx] if replay_eval_infos else None,
                    "replay_eval_meta": replay_eval_meta,
                    "replay_summary": summarize_replay_buffer(record["replay"]),
                    "archive_improved": bool(archive_improved),
                }
            )
        stage_signal_members = build_signal_members(records)
        stage_member_ensemble_results = build_member_ensemble_results(
            trained_members=stage_signal_members,
            ensemble_topks=ensemble_topks,
            model=model,
            val_tokens=val_tokens,
            eval_starts=eval_starts,
            test_tokens=test_tokens,
            test_eval_starts=test_eval_starts,
            cfg=cfg,
            device=device,
            param_dtype=param_dtype,
        )
        signals = compute_committee_signals(
            records=records,
            member_ensemble_results=stage_member_ensemble_results,
            previous_best_member_bpb=previous_best_member_bpb,
        )
        previous_best_member_bpb = signals.get("best_member_bpb")
        stage_signals.append(
            {
                "stage_index": int(stage_idx),
                "stage_copies": int(copies),
                "stage_train_seconds_target": float(stage_seconds),
                "signals": signals,
                "member_ensemble_results": stage_member_ensemble_results,
            }
        )
    trained_members = [
        {
            "member_id": int(member_idx),
            "seed": int(record["stages"][-1]["seed"]),
            "lineage": record["lineage"],
            "train": aggregate_stage_history(record["stages"]),
            "stages": record["stages"],
            "eval": record["best_eval"],
            "current_eval": record["current_eval"],
            "archive_summary": record["archive_summary"],
            "replay_summary": summarize_replay_buffer(record["replay"]),
            "replay_eval": record.get("replay_eval"),
            "state": record["best_state"],
            "best_state": record["best_state"],
            "replay": record["replay"],
        }
        for member_idx, record in enumerate(records)
    ]
    member_ensemble_results = build_member_ensemble_results(
        trained_members=trained_members,
        ensemble_topks=ensemble_topks,
        model=model,
        val_tokens=val_tokens,
        eval_starts=eval_starts,
        test_tokens=test_tokens,
        test_eval_starts=test_eval_starts,
        cfg=cfg,
        device=device,
        param_dtype=param_dtype,
    )
    return {
        "config": asdict(cfg),
        "dataset": {
            "enwik8_path": str(enwik8_path),
            "train_tokens": int(train_tokens.numel()),
            "val_tokens": int(val_tokens.numel()),
            "test_tokens": int(test_tokens.numel()),
            **dataset_meta,
        },
        "member_train_mode": "parallel_vmap_staged",
        "committee_schedule": {
            "stage_copies": [int(value) for value in stage_copies],
            "stage_train_seconds": [float(value) for value in stage_train_seconds],
            "stage_signature": "->".join(f"{copies}x{seconds:g}" for copies, seconds in zip(stage_copies, stage_train_seconds)),
            "spawn_noise_std": float(spawn_noise_std),
            "eval_batches": int(eval_batches),
            "hard_replay_topk": int(cfg.hard_replay_topk),
            "hard_replay_eval_batches": int(cfg.hard_replay_eval_batches),
            "use_best_archive_for_spawn": bool(cfg.use_best_archive_for_spawn),
        },
        "stage_signals": stage_signals,
        "members": [
            {
                "member_id": member["member_id"],
                "seed": member["seed"],
                "lineage": member["lineage"],
                "train": member["train"],
                "stages": member["stages"],
                "eval": member["eval"],
                "current_eval": member["current_eval"],
                "archive_summary": member["archive_summary"],
                "replay_summary": member["replay_summary"],
                "replay_eval": member["replay_eval"],
                **({"state": member["state"]} if include_states else {}),
            }
            for member in trained_members
        ],
        "member_ensemble_results": member_ensemble_results,
    }


def run_committee_compressibility(
    *,
    cfg: EvoBenchmarkConfig,
    enwik8_path: Path,
    stage_copies: tuple[int, ...],
    stage_train_seconds: tuple[float, ...],
    ensemble_topks: tuple[int, ...],
    eval_batches: int,
    spawn_noise_std: float,
    artifact_limit_mb: float,
    delta_budget_fractions: tuple[float, ...],
    basis_ranks: tuple[int, ...],
    analysis_topk: int,
    device: torch.device,
    param_dtype: torch.dtype,
) -> dict[str, Any]:
    committee_result = run_committee_schedule(
        cfg=cfg,
        enwik8_path=enwik8_path,
        stage_copies=stage_copies,
        stage_train_seconds=stage_train_seconds,
        ensemble_topks=ensemble_topks,
        eval_batches=eval_batches,
        spawn_noise_std=spawn_noise_std,
        include_states=True,
        device=device,
        param_dtype=param_dtype,
    )
    member_states = [member["state"] for member in committee_result["members"]]
    model = build_model(cfg, device, param_dtype)
    base_state = parameter_state_dict(model)
    base_artifact_mb = artifact_param_mb_for_cfg(cfg)
    available_delta_budget_mb = max(0.0, float(artifact_limit_mb) - float(base_artifact_mb))
    if available_delta_budget_mb <= 0.0:
        raise ValueError(
            f"artifact_limit_mb={artifact_limit_mb} leaves no room for deltas because base_artifact_mb={base_artifact_mb:.3f}"
        )

    train_tokens, val_tokens, test_tokens, dataset_meta = prepare_tokenized_enwik8_splits(
        enwik8_path,
        device=device,
        cache_on_device=cfg.cache_dataset_on_device,
        tokenization_mode=cfg.tokenization_mode,
        tokenizer_name=cfg.tokenizer_name,
        tokenizer_model_path=cfg.tokenizer_model_path,
    )
    val_starts = build_lm_starts(int(val_tokens.numel()), cfg.seq_len, cfg.stride)
    test_starts = build_lm_starts(int(test_tokens.numel()), cfg.seq_len, cfg.stride)
    eval_starts = choose_eval_starts(val_starts, batch_size=cfg.batch_size, eval_batches=eval_batches, seed=cfg.seed + 9000)
    test_eval_starts = choose_eval_starts(test_starts, batch_size=cfg.batch_size, eval_batches=eval_batches, seed=cfg.seed + 9001)

    delta_matrix, base_flat, layout = flatten_branch_deltas(base_state, member_states)
    delta_analysis = pairwise_delta_metrics(delta_matrix, support_topk=analysis_topk)
    budget_fractions = tuple(sorted({float(fraction) for fraction in delta_budget_fractions if float(fraction) > 0.0}))
    if not budget_fractions:
        budget_fractions = (1.0,)
    full_ensemble = evaluate_compressed_committee(
        member_states,
        model=model,
        val_tokens=val_tokens,
        eval_starts=eval_starts,
        test_tokens=test_tokens,
        test_eval_starts=test_eval_starts,
        cfg=cfg,
        device=device,
        param_dtype=param_dtype,
    )

    compression_trials: list[dict[str, Any]] = []
    for fraction in budget_fractions:
        budget_mb = available_delta_budget_mb * fraction
        budget_bytes = int(budget_mb * 1024.0 * 1024.0)

        sparse_deltas, sparse_meta = compress_branch_sparse(delta_matrix, budget_bytes=budget_bytes)
        sparse_states = reconstruct_branch_states(sparse_deltas, base_flat=base_flat, layout=layout)
        sparse_eval = evaluate_compressed_committee(
            sparse_states,
            model=model,
            val_tokens=val_tokens,
            eval_starts=eval_starts,
            test_tokens=test_tokens,
            test_eval_starts=test_eval_starts,
            cfg=cfg,
            device=device,
            param_dtype=param_dtype,
        )
        compression_trials.append(
            {
                "strategy": "branch_sparse",
                "budget_fraction": float(fraction),
                "budget_mb": float(budget_mb),
                "bytes_used": int(sparse_meta["bytes_used"]),
                "support": int(sparse_meta["support_per_branch"]),
                **sparse_eval,
                "retention_vs_full_ensemble": float(full_ensemble["val_bpb"] / max(sparse_eval["val_bpb"], 1e-8)),
            }
        )

        shared_sparse_deltas, shared_sparse_meta = compress_shared_union_sparse(delta_matrix, budget_bytes=budget_bytes)
        shared_sparse_states = reconstruct_branch_states(shared_sparse_deltas, base_flat=base_flat, layout=layout)
        shared_sparse_eval = evaluate_compressed_committee(
            shared_sparse_states,
            model=model,
            val_tokens=val_tokens,
            eval_starts=eval_starts,
            test_tokens=test_tokens,
            test_eval_starts=test_eval_starts,
            cfg=cfg,
            device=device,
            param_dtype=param_dtype,
        )
        compression_trials.append(
            {
                "strategy": "shared_union_sparse",
                "budget_fraction": float(fraction),
                "budget_mb": float(budget_mb),
                "bytes_used": int(shared_sparse_meta["bytes_used"]),
                "support": int(shared_sparse_meta["shared_support"]),
                **shared_sparse_eval,
                "retention_vs_full_ensemble": float(full_ensemble["val_bpb"] / max(shared_sparse_eval["val_bpb"], 1e-8)),
            }
        )

        for rank in basis_ranks:
            basis_result = compress_shared_basis(delta_matrix, budget_bytes=budget_bytes, rank=rank)
            if basis_result is None:
                continue
            basis_deltas, basis_meta = basis_result
            basis_states = reconstruct_branch_states(basis_deltas, base_flat=base_flat, layout=layout)
            basis_eval = evaluate_compressed_committee(
                basis_states,
                model=model,
                val_tokens=val_tokens,
                eval_starts=eval_starts,
                test_tokens=test_tokens,
                test_eval_starts=test_eval_starts,
                cfg=cfg,
                device=device,
                param_dtype=param_dtype,
            )
            compression_trials.append(
                {
                    "strategy": "shared_basis",
                    "rank": int(basis_meta["rank"]),
                    "budget_fraction": float(fraction),
                    "budget_mb": float(budget_mb),
                    "bytes_used": int(basis_meta["bytes_used"]),
                    "support": int(basis_meta["shared_support"]),
                    **basis_eval,
                    "retention_vs_full_ensemble": float(full_ensemble["val_bpb"] / max(basis_eval["val_bpb"], 1e-8)),
                }
            )

    best_by_budget: list[dict[str, Any]] = []
    for fraction in budget_fractions:
        candidates = [row for row in compression_trials if float(row["budget_fraction"]) == float(fraction)]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: float(row["val_bpb"]))
        best_by_budget.append(best)

    return {
        "config": asdict(cfg),
        "dataset": {
            "enwik8_path": str(enwik8_path),
            "train_tokens": int(train_tokens.numel()),
            "val_tokens": int(val_tokens.numel()),
            "test_tokens": int(test_tokens.numel()),
            **dataset_meta,
        },
        "committee_schedule": committee_result["committee_schedule"],
        "committee_compressibility": {
            "artifact_limit_mb": float(artifact_limit_mb),
            "base_artifact_mb": float(base_artifact_mb),
            "available_delta_budget_mb": float(available_delta_budget_mb),
            "delta_budget_fractions": [float(fraction) for fraction in budget_fractions],
            "basis_ranks": [int(rank) for rank in basis_ranks],
            "analysis_topk": int(analysis_topk),
            "full_committee_val_bpb": float(full_ensemble["val_bpb"]),
            "full_committee_test_bpb": float(full_ensemble["test_bpb"]),
            "member_count": int(len(member_states)),
        },
        "delta_analysis": delta_analysis,
        "compression_trials": compression_trials,
        "best_by_budget": best_by_budget,
    }


def run_committee_adaptive(
    *,
    cfg: EvoBenchmarkConfig,
    enwik8_path: Path,
    initial_copies: int,
    round_train_seconds: tuple[float, ...],
    ensemble_topks: tuple[int, ...],
    eval_batches: int,
    spawn_noise_std: float,
    min_copies: int,
    max_copies: int,
    widen_factor: int,
    narrow_factor: int,
    depth_gain_threshold: float,
    breadth_gain_threshold: float,
    archive_hit_rate_threshold: float,
    winner_concentration_threshold: float,
    replay_disagreement_threshold: float,
    pairwise_distance_threshold: float,
    device: torch.device,
    param_dtype: torch.dtype,
) -> dict[str, Any]:
    if initial_copies <= 0:
        raise ValueError("initial_copies must be positive")
    if not round_train_seconds:
        raise ValueError("round_train_seconds must be non-empty")
    train_tokens, val_tokens, test_tokens, dataset_meta = prepare_tokenized_enwik8_splits(
        enwik8_path,
        device=device,
        cache_on_device=cfg.cache_dataset_on_device,
        tokenization_mode=cfg.tokenization_mode,
        tokenizer_name=cfg.tokenizer_name,
        tokenizer_model_path=cfg.tokenizer_model_path,
    )
    train_starts = build_lm_starts(int(train_tokens.numel()), cfg.seq_len, cfg.stride)
    val_starts = build_lm_starts(int(val_tokens.numel()), cfg.seq_len, cfg.stride)
    test_starts = build_lm_starts(int(test_tokens.numel()), cfg.seq_len, cfg.stride)
    if not train_starts or not val_starts or not test_starts:
        raise ValueError("enwik8 split is too small for the requested seq_len/stride")

    eval_starts = choose_eval_starts(val_starts, batch_size=cfg.batch_size, eval_batches=eval_batches, seed=cfg.seed + 9000)
    test_eval_starts = choose_eval_starts(test_starts, batch_size=cfg.batch_size, eval_batches=eval_batches, seed=cfg.seed + 9001)
    model = build_model(cfg, device, param_dtype)
    base_init = parameter_state_dict(model)

    records = [
        {
            "lineage": [member_id],
            "state": base_init,
            "best_state": base_init,
            "best_eval": None,
            "current_eval": None,
            "archive_summary": {
                "best_stage_index": None,
                "current_stage_index": None,
            },
            "replay": [],
            "replay_eval": None,
            "stages": [],
        }
        for member_id in range(initial_copies)
    ]
    rounds: list[dict[str, Any]] = []
    previous_best_member_bpb: float | None = None
    target_copies = int(initial_copies)

    for round_idx, stage_seconds in enumerate(round_train_seconds):
        if round_idx > 0:
            if target_copies > len(records):
                records = expand_population_records(
                    records,
                    target_size=target_copies,
                    spawn_noise_std=spawn_noise_std,
                    seed=cfg.seed + 60_000 + round_idx * 1_000,
                    use_best_archive_for_spawn=cfg.use_best_archive_for_spawn,
                    hard_replay_topk=cfg.hard_replay_topk,
                )
            elif target_copies < len(records):
                records = select_top_population_records(
                    records,
                    target_size=target_copies,
                    model=model,
                    eval_tokens=val_tokens,
                    eval_starts=eval_starts,
                    batch_size=cfg.batch_size,
                    seq_len=cfg.seq_len,
                    device=device,
                    param_dtype=param_dtype,
                )

        replay_eval_infos, replay_eval_meta = population_replay_eval_metrics(
            records,
            model=model,
            replay_tokens=train_tokens,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            eval_batches=cfg.hard_replay_eval_batches,
            device=device,
            param_dtype=param_dtype,
        )
        states = [record["state"] for record in records]
        stage_seeds = [cfg.seed + 20_000 * (round_idx + 1) + member_idx for member_idx in range(len(records))]
        trained_states, train_infos, replay_buffers = train_population_for_seconds(
            model,
            states=states,
            train_tokens=train_tokens,
            train_starts=train_starts,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
            base_lr=cfg.base_lr,
            weight_decay=cfg.weight_decay,
            grad_clip_norm=cfg.grad_clip_norm,
            train_seconds=stage_seconds,
            seeds=stage_seeds,
            device=device,
            param_dtype=param_dtype,
            hard_replay_topk=cfg.hard_replay_topk,
        )
        current_eval_infos = population_eval_metrics(
            trained_states,
            model=model,
            eval_tokens=val_tokens,
            eval_starts=eval_starts,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            device=device,
            param_dtype=param_dtype,
        )
        for member_idx, record in enumerate(records):
            record["state"] = trained_states[member_idx]
            record["current_eval"] = current_eval_infos[member_idx]
            record["replay"] = merge_replay_buffers(
                [list(record.get("replay", [])), replay_buffers[member_idx]],
                topk=cfg.hard_replay_topk,
            )
            record["replay_eval"] = replay_eval_infos[member_idx] if replay_eval_infos else None
            archive_improved = False
            best_eval = record.get("best_eval")
            current_eval = current_eval_infos[member_idx]
            if best_eval is None or float(current_eval["bpb"]) < float(best_eval["bpb"]):
                record["best_state"] = trained_states[member_idx]
                record["best_eval"] = current_eval
                record["archive_summary"] = {
                    "best_stage_index": int(round_idx),
                    "current_stage_index": int(round_idx),
                }
                archive_improved = True
            else:
                best_stage_index = record.get("archive_summary", {}).get("best_stage_index")
                record["archive_summary"] = {
                    "best_stage_index": int(round_idx if best_stage_index is None else best_stage_index),
                    "current_stage_index": int(round_idx),
                }
            record["stages"].append(
                {
                    "stage_index": int(round_idx),
                    "stage_copies": int(len(records)),
                    "stage_train_seconds_target": float(stage_seconds),
                    "seed": int(stage_seeds[member_idx]),
                    "train": asdict(train_infos[member_idx]),
                    "current_eval": current_eval,
                    "pre_stage_replay_eval": replay_eval_infos[member_idx] if replay_eval_infos else None,
                    "replay_eval_meta": replay_eval_meta,
                    "replay_summary": summarize_replay_buffer(record["replay"]),
                    "archive_improved": bool(archive_improved),
                }
            )
        round_members = build_signal_members(records)
        round_ensemble_results = build_member_ensemble_results(
            trained_members=round_members,
            ensemble_topks=ensemble_topks,
            model=model,
            val_tokens=val_tokens,
            eval_starts=eval_starts,
            test_tokens=test_tokens,
            test_eval_starts=test_eval_starts,
            cfg=cfg,
            device=device,
            param_dtype=param_dtype,
        )
        signals = compute_committee_signals(
            records=records,
            member_ensemble_results=round_ensemble_results,
            previous_best_member_bpb=previous_best_member_bpb,
        )
        decision = recommend_committee_action(
            signals=signals,
            current_copies=len(records),
            min_copies=min_copies,
            max_copies=max_copies,
            widen_factor=widen_factor,
            narrow_factor=narrow_factor,
            depth_gain_threshold=depth_gain_threshold,
            breadth_gain_threshold=breadth_gain_threshold,
            archive_hit_rate_threshold=archive_hit_rate_threshold,
            winner_concentration_threshold=winner_concentration_threshold,
            replay_disagreement_threshold=replay_disagreement_threshold,
            pairwise_distance_threshold=pairwise_distance_threshold,
        )
        previous_best_member_bpb = signals.get("best_member_bpb")
        target_copies = int(decision["next_copies"])
        rounds.append(
            {
                "round_index": int(round_idx),
                "copies": int(len(records)),
                "train_seconds_target": float(stage_seconds),
                "signals": signals,
                "decision": decision,
                "member_ensemble_results": round_ensemble_results,
            }
        )

    trained_members = [
        {
            "member_id": int(member_idx),
            "seed": int(record["stages"][-1]["seed"]),
            "lineage": record["lineage"],
            "train": aggregate_stage_history(record["stages"]),
            "stages": record["stages"],
            "eval": record["best_eval"],
            "current_eval": record["current_eval"],
            "archive_summary": record["archive_summary"],
            "replay_summary": summarize_replay_buffer(record["replay"]),
            "replay_eval": record.get("replay_eval"),
            "state": record["best_state"],
            "best_state": record["best_state"],
            "replay": record["replay"],
        }
        for member_idx, record in enumerate(records)
    ]
    member_ensemble_results = build_member_ensemble_results(
        trained_members=trained_members,
        ensemble_topks=ensemble_topks,
        model=model,
        val_tokens=val_tokens,
        eval_starts=eval_starts,
        test_tokens=test_tokens,
        test_eval_starts=test_eval_starts,
        cfg=cfg,
        device=device,
        param_dtype=param_dtype,
    )
    return {
        "config": asdict(cfg),
        "dataset": {
            "enwik8_path": str(enwik8_path),
            "train_tokens": int(train_tokens.numel()),
            "val_tokens": int(val_tokens.numel()),
            "test_tokens": int(test_tokens.numel()),
            **dataset_meta,
        },
        "member_train_mode": "parallel_vmap_adaptive",
        "committee_adaptive": {
            "initial_copies": int(initial_copies),
            "round_train_seconds": [float(value) for value in round_train_seconds],
            "spawn_noise_std": float(spawn_noise_std),
            "eval_batches": int(eval_batches),
            "hard_replay_topk": int(cfg.hard_replay_topk),
            "hard_replay_eval_batches": int(cfg.hard_replay_eval_batches),
            "use_best_archive_for_spawn": bool(cfg.use_best_archive_for_spawn),
            "min_copies": int(min_copies),
            "max_copies": int(max_copies),
            "widen_factor": int(widen_factor),
            "narrow_factor": int(narrow_factor),
            "thresholds": {
                "depth_gain": float(depth_gain_threshold),
                "breadth_gain": float(breadth_gain_threshold),
                "archive_hit_rate": float(archive_hit_rate_threshold),
                "winner_concentration": float(winner_concentration_threshold),
                "replay_disagreement": float(replay_disagreement_threshold),
                "pairwise_distance": float(pairwise_distance_threshold),
            },
            "round_signature": "->".join(f"{seconds:g}s" for seconds in round_train_seconds),
        },
        "rounds": rounds,
        "members": [
            {
                "member_id": member["member_id"],
                "seed": member["seed"],
                "lineage": member["lineage"],
                "train": member["train"],
                "stages": member["stages"],
                "eval": member["eval"],
                "current_eval": member["current_eval"],
                "archive_summary": member["archive_summary"],
                "replay_summary": member["replay_summary"],
                "replay_eval": member["replay_eval"],
            }
            for member in trained_members
        ],
        "member_ensemble_results": member_ensemble_results,
    }


def ensure_valid_recipe_genome(
    genome: RecipeGenome,
    *,
    base_cfg: EvoBenchmarkConfig,
    space: RecipeGeneSpace,
    artifact_limit_mb: float,
    rng: random.Random,
    artifact_cache: dict[tuple[Any, ...], float],
) -> RecipeGenome:
    for _ in range(64):
        key = tuple(asdict(genome).items())
        artifact_mb = artifact_cache.get(key)
        if artifact_mb is None:
            artifact_mb = artifact_param_mb_for_cfg(recipe_genome_to_cfg(base_cfg, genome))
            artifact_cache[key] = artifact_mb
        if artifact_mb <= artifact_limit_mb:
            return genome
        genome = mutate_recipe_genome(genome, space, mutation_rate=0.5, rng=rng)
    while True:
        genome = random_recipe_genome(space, rng=rng)
        key = tuple(asdict(genome).items())
        artifact_mb = artifact_cache.get(key)
        if artifact_mb is None:
            artifact_mb = artifact_param_mb_for_cfg(recipe_genome_to_cfg(base_cfg, genome))
            artifact_cache[key] = artifact_mb
        if artifact_mb <= artifact_limit_mb:
            return genome


def evaluate_recipe_genome(
    *,
    genome: RecipeGenome,
    base_cfg: EvoBenchmarkConfig,
    enwik8_path: Path,
    train_seconds: float,
    eval_batches: int,
    seed: int,
    device: torch.device,
    param_dtype: torch.dtype,
    dataset_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]],
    starts_cache: dict[tuple[Any, ...], tuple[list[int], list[int], list[int], list[int], list[int]]],
    artifact_cache: dict[tuple[Any, ...], float],
) -> dict[str, Any]:
    cfg = recipe_genome_to_cfg(base_cfg, genome)
    dataset_key = (
        cfg.tokenization_mode,
        cfg.tokenizer_name,
        cfg.tokenizer_model_path,
        cfg.cache_dataset_on_device,
        device.type,
    )
    if dataset_key not in dataset_cache:
        dataset_cache[dataset_key] = prepare_tokenized_enwik8_splits(
            enwik8_path,
            device=device,
            cache_on_device=cfg.cache_dataset_on_device,
            tokenization_mode=cfg.tokenization_mode,
            tokenizer_name=cfg.tokenizer_name,
            tokenizer_model_path=cfg.tokenizer_model_path,
        )
    train_tokens, val_tokens, test_tokens, dataset_meta = dataset_cache[dataset_key]
    starts_key = dataset_key + (cfg.seq_len, cfg.stride, cfg.batch_size, eval_batches, seed)
    if starts_key not in starts_cache:
        train_starts = build_lm_starts(int(train_tokens.numel()), cfg.seq_len, cfg.stride)
        val_starts = build_lm_starts(int(val_tokens.numel()), cfg.seq_len, cfg.stride)
        test_starts = build_lm_starts(int(test_tokens.numel()), cfg.seq_len, cfg.stride)
        if not train_starts or not val_starts or not test_starts:
            raise ValueError("tokenized dataset split is too small for the requested recipe seq_len/stride")
        eval_starts = choose_eval_starts(val_starts, batch_size=cfg.batch_size, eval_batches=eval_batches, seed=seed + 9000)
        test_eval_starts = choose_eval_starts(test_starts, batch_size=cfg.batch_size, eval_batches=eval_batches, seed=seed + 9001)
        starts_cache[starts_key] = (train_starts, val_starts, test_starts, eval_starts, test_eval_starts)
    train_starts, _, _, eval_starts, test_eval_starts = starts_cache[starts_key]

    model = build_model(cfg, device, param_dtype)
    train_info, replay_entries = train_for_seconds(
        model,
        train_tokens=train_tokens,
        train_starts=train_starts,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        base_lr=cfg.base_lr,
        weight_decay=cfg.weight_decay,
        grad_clip_norm=cfg.grad_clip_norm,
        train_seconds=train_seconds,
        seed=seed,
        device=device,
        param_dtype=param_dtype,
        hard_replay_topk=cfg.hard_replay_topk,
    )
    val_info = evaluate_bpb(
        model,
        tokens=val_tokens,
        eval_starts=eval_starts,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        device=device,
        param_dtype=param_dtype,
    )
    test_info = evaluate_bpb(
        model,
        tokens=test_tokens,
        eval_starts=test_eval_starts,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        device=device,
        param_dtype=param_dtype,
    )
    artifact_key = tuple(asdict(genome).items())
    artifact_mb = artifact_cache.get(artifact_key)
    if artifact_mb is None:
        artifact_mb = artifact_param_mb_for_cfg(cfg)
        artifact_cache[artifact_key] = artifact_mb
    return {
        "genome": genome_to_dict(genome),
        "config": asdict(cfg),
        "artifact_param_mb": float(artifact_mb),
        "train": asdict(train_info),
        "val": val_info,
        "test": test_info,
        "replay_summary": summarize_replay_buffer(replay_entries),
        "dataset": {
            "train_tokens": int(train_tokens.numel()),
            "val_tokens": int(val_tokens.numel()),
            "test_tokens": int(test_tokens.numel()),
            **dataset_meta,
        },
    }


def run_recipe_evolution(
    *,
    cfg: EvoBenchmarkConfig,
    enwik8_path: Path,
    population_size: int,
    generations: int,
    tournament_size: int,
    train_seconds: float,
    eval_batches: int,
    mutation_rate: float,
    artifact_limit_mb: float,
    recipe_profile: str,
    confirm_topk: int,
    confirm_train_seconds: float,
    device: torch.device,
    param_dtype: torch.dtype,
) -> dict[str, Any]:
    if population_size <= 0 or generations <= 0:
        raise ValueError("population_size and generations must be positive")
    space = default_recipe_gene_space(recipe_profile)
    rng = random.Random(cfg.seed + 400_000)
    dataset_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]] = {}
    starts_cache: dict[tuple[Any, ...], tuple[list[int], list[int], list[int], list[int], list[int]]] = {}
    artifact_cache: dict[tuple[Any, ...], float] = {}

    population: list[RecipeGenome] = []
    while len(population) < population_size:
        genome = ensure_valid_recipe_genome(
            random_recipe_genome(space, rng=rng),
            base_cfg=cfg,
            space=space,
            artifact_limit_mb=artifact_limit_mb,
            rng=rng,
            artifact_cache=artifact_cache,
        )
        population.append(genome)

    history: list[dict[str, Any]] = []
    evaluated_population: list[dict[str, Any]] = []
    for generation in range(generations):
        evaluated_population = []
        for member_idx, genome in enumerate(population):
            evaluated_population.append(
                evaluate_recipe_genome(
                    genome=genome,
                    base_cfg=cfg,
                    enwik8_path=enwik8_path,
                    train_seconds=train_seconds,
                    eval_batches=eval_batches,
                    seed=cfg.seed + generation * 10_000 + member_idx,
                    device=device,
                    param_dtype=param_dtype,
                    dataset_cache=dataset_cache,
                    starts_cache=starts_cache,
                    artifact_cache=artifact_cache,
                )
            )
        evaluated_population.sort(key=lambda row: float(row["val"]["bpb"]))
        bpbs = [float(row["val"]["bpb"]) for row in evaluated_population]
        history.append(
            {
                "generation": int(generation),
                "best_bpb": float(min(bpbs)),
                "mean_bpb": float(sum(bpbs) / len(bpbs)),
                "worst_bpb": float(max(bpbs)),
                "best_recipe": evaluated_population[0]["genome"],
                "best_artifact_param_mb": float(evaluated_population[0]["artifact_param_mb"]),
            }
        )
        if generation == generations - 1:
            break

        next_population = [RecipeGenome(**evaluated_population[0]["genome"])]
        while len(next_population) < population_size:
            parent_left_idx, parent_right_idx = select_distinct_parent_indices(bpbs, tournament_size, rng)
            parent_left = RecipeGenome(**evaluated_population[parent_left_idx]["genome"])
            parent_right = RecipeGenome(**evaluated_population[parent_right_idx]["genome"])
            child = crossover_recipe_genomes(parent_left, parent_right, rng=rng)
            child = mutate_recipe_genome(child, space, mutation_rate=mutation_rate, rng=rng)
            child = ensure_valid_recipe_genome(
                child,
                base_cfg=cfg,
                space=space,
                artifact_limit_mb=artifact_limit_mb,
                rng=rng,
                artifact_cache=artifact_cache,
            )
            next_population.append(child)
        population = next_population

    confirm_results: list[dict[str, Any]] = []
    if confirm_topk > 0 and confirm_train_seconds > 0.0 and evaluated_population:
        for rank, row in enumerate(evaluated_population[: min(confirm_topk, len(evaluated_population))]):
            confirm_eval = evaluate_recipe_genome(
                genome=RecipeGenome(**row["genome"]),
                base_cfg=cfg,
                enwik8_path=enwik8_path,
                train_seconds=confirm_train_seconds,
                eval_batches=eval_batches,
                seed=cfg.seed + 700_000 + rank,
                device=device,
                param_dtype=param_dtype,
                dataset_cache=dataset_cache,
                starts_cache=starts_cache,
                artifact_cache=artifact_cache,
            )
            confirm_eval["rank_from_short_fitness"] = int(rank)
            confirm_results.append(confirm_eval)

    best_recipe = evaluated_population[0] if evaluated_population else None
    return {
        "config": asdict(cfg),
        "recipe_evolution": {
            "population_size": int(population_size),
            "generations": int(generations),
            "tournament_size": int(tournament_size),
            "train_seconds": float(train_seconds),
            "eval_batches": int(eval_batches),
            "mutation_rate": float(mutation_rate),
            "artifact_limit_mb": float(artifact_limit_mb),
            "recipe_profile": recipe_profile,
            "confirm_topk": int(confirm_topk),
            "confirm_train_seconds": float(confirm_train_seconds),
            "search_space": {
                "model_dims": list(space.model_dims),
                "num_layers": list(space.num_layers),
                "num_heads": list(space.num_heads),
                "num_kv_heads": list(space.num_kv_heads),
                "mlp_mults": list(space.mlp_mults),
                "base_lrs": list(space.base_lrs),
                "weight_decays": list(space.weight_decays),
                "qk_gains": list(space.qk_gains),
                "spine_genes": list(space.spine_genes),
                "tokenizer_genes": list(space.tokenizer_genes),
                "seq_lens": list(space.seq_lens),
                "batch_sizes": list(space.batch_sizes),
            },
        },
        "history": history,
        "population": [
            {
                "rank": int(rank),
                **row,
            }
            for rank, row in enumerate(evaluated_population)
        ],
        "best": best_recipe,
        "confirm_results": confirm_results,
    }


def summarize_crossover_trials(trials: list[dict[str, Any]]) -> dict[str, float]:
    if not trials:
        return {
            "num_trials": 0.0,
            "fraction_between_parents": 0.0,
            "fraction_collapse": 0.0,
            "fraction_improves_best_parent": 0.0,
        }
    between = sum(1 for trial in trials if trial["between_parents"])
    collapse = sum(1 for trial in trials if trial["collapsed"])
    improves = sum(1 for trial in trials if trial["improves_best_parent"])
    return {
        "num_trials": float(len(trials)),
        "fraction_between_parents": float(between / len(trials)),
        "fraction_collapse": float(collapse / len(trials)),
        "fraction_improves_best_parent": float(improves / len(trials)),
    }


def strategy_seed_offset(strategy: str) -> int:
    return sum((index + 1) * ord(ch) for index, ch in enumerate(strategy)) % 100_000


def run_crossover_viability(
    *,
    cfg: EvoBenchmarkConfig,
    enwik8_path: Path,
    copies: int,
    train_seconds: float,
    strategies: tuple[str, ...],
    percentiles: tuple[float, ...],
    eval_batches: int,
    pair_limit: int | None,
    ensemble_topks: tuple[int, ...],
    member_train_mode: str,
    device: torch.device,
    param_dtype: torch.dtype,
) -> dict[str, Any]:
    train_tokens, val_tokens, test_tokens, dataset_meta = prepare_tokenized_enwik8_splits(
        enwik8_path,
        device=device,
        cache_on_device=cfg.cache_dataset_on_device,
        tokenization_mode=cfg.tokenization_mode,
        tokenizer_name=cfg.tokenizer_name,
        tokenizer_model_path=cfg.tokenizer_model_path,
    )
    train_starts = build_lm_starts(int(train_tokens.numel()), cfg.seq_len, cfg.stride)
    val_starts = build_lm_starts(int(val_tokens.numel()), cfg.seq_len, cfg.stride)
    test_starts = build_lm_starts(int(test_tokens.numel()), cfg.seq_len, cfg.stride)
    if not train_starts or not val_starts or not test_starts:
        raise ValueError("enwik8 split is too small for the requested seq_len/stride")

    eval_starts = choose_eval_starts(val_starts, batch_size=cfg.batch_size, eval_batches=eval_batches, seed=cfg.seed + 9000)
    test_eval_starts = choose_eval_starts(test_starts, batch_size=cfg.batch_size, eval_batches=eval_batches, seed=cfg.seed + 9001)
    model = build_model(cfg, device, param_dtype)
    base_init = parameter_state_dict(model)

    if member_train_mode == "parallel_vmap":
        trained_members = build_member_records_parallel(
            copies=copies,
            state=base_init,
            model=model,
            train_tokens=train_tokens,
            train_starts=train_starts,
            eval_tokens=val_tokens,
            eval_starts=eval_starts,
            cfg=cfg,
            train_seconds=train_seconds,
            device=device,
            param_dtype=param_dtype,
        )
    else:
        trained_members = []
        for member_id in range(copies):
            member_seed = cfg.seed + 101 + member_id
            trained_members.append(
                build_member_record(
                    member_id=member_id,
                    state=base_init,
                    model=model,
                    train_tokens=train_tokens,
                    train_starts=train_starts,
                    eval_tokens=val_tokens,
                    eval_starts=eval_starts,
                    cfg=cfg,
                    train_seconds=train_seconds,
                    seed=member_seed,
                    device=device,
                    param_dtype=param_dtype,
                )
            )

    pair_indices = list(itertools.combinations(range(copies), 2))
    if pair_limit is not None:
        pair_indices = pair_indices[:pair_limit]

    member_ensemble_results = []
    requested_topks = sorted({topk for topk in ensemble_topks if topk > 0})
    if requested_topks:
        sorted_members = sorted(trained_members, key=lambda member: float(member["eval"]["bpb"]))
        for topk in requested_topks:
            actual_topk = min(topk, len(sorted_members))
            selected_states = [member["state"] for member in sorted_members[:actual_topk]]
            member_ensemble_results.append(
                {
                    "topk": int(actual_topk),
                    "val_bpb": ensemble_bpb(
                        selected_states,
                        model=model,
                        eval_tokens=val_tokens,
                        eval_starts=eval_starts,
                        batch_size=cfg.batch_size,
                        seq_len=cfg.seq_len,
                        device=device,
                        param_dtype=param_dtype,
                    ),
                    "test_bpb": ensemble_bpb(
                        selected_states,
                        model=model,
                        eval_tokens=test_tokens,
                        eval_starts=test_eval_starts,
                        batch_size=cfg.batch_size,
                        seq_len=cfg.seq_len,
                        device=device,
                        param_dtype=param_dtype,
                    ),
                }
            )

    strategy_results: dict[str, Any] = {}
    specs = crossover_specs()
    for strategy in strategies:
        spec = specs[strategy]
        active_percentiles = percentiles if spec.uses_percentile else (-1.0,)
        strategy_results[strategy] = {}
        seed_offset = strategy_seed_offset(strategy)
        for percentile in active_percentiles:
            trials: list[dict[str, Any]] = []
            for pair_idx, (left_idx, right_idx) in enumerate(pair_indices):
                left = trained_members[left_idx]
                right = trained_members[right_idx]
                child_state, crossover_stats = crossover_state(
                    left["state"],
                    right["state"],
                    strategy=strategy,
                    percentile=percentile,
                    seed=cfg.seed + 20_000 + seed_offset + pair_idx + int(max(percentile, 0.0)) * 100,
                    base_state=base_init,
                )
                child_bpb = state_bpb(
                    child_state,
                    model=model,
                    eval_tokens=val_tokens,
                    eval_starts=eval_starts,
                    batch_size=cfg.batch_size,
                    seq_len=cfg.seq_len,
                    device=device,
                    param_dtype=param_dtype,
                )
                parent_a_bpb = float(left["eval"]["bpb"])
                parent_b_bpb = float(right["eval"]["bpb"])
                lo = min(parent_a_bpb, parent_b_bpb)
                hi = max(parent_a_bpb, parent_b_bpb)
                trials.append(
                    {
                        "pair": [int(left_idx), int(right_idx)],
                        "strategy": strategy,
                        "percentile": float(percentile),
                        "parent_a_bpb": parent_a_bpb,
                        "parent_b_bpb": parent_b_bpb,
                        "offspring_bpb": float(child_bpb),
                        "between_parents": bool(lo <= child_bpb <= hi),
                        "collapsed": bool(child_bpb > hi),
                        "improves_best_parent": bool(child_bpb < lo),
                        "crossover": crossover_stats,
                    }
                )
            percentile_key = str(int(percentile)) if percentile >= 0.0 else "na"
            strategy_results[strategy][percentile_key] = {
                "summary": summarize_crossover_trials(trials),
                "trials": trials,
            }

    return {
        "config": asdict(cfg),
        "dataset": {
            "enwik8_path": str(enwik8_path),
            "train_tokens": int(train_tokens.numel()),
            "val_tokens": int(val_tokens.numel()),
            "test_tokens": int(test_tokens.numel()),
            **dataset_meta,
        },
        "member_train_mode": member_train_mode,
        "members": [
            {
                "member_id": member["member_id"],
                "seed": member["seed"],
                "train": member["train"],
                "eval": member["eval"],
            }
            for member in trained_members
        ],
        "member_ensemble_results": member_ensemble_results,
        "crossover": strategy_results,
    }


def tournament_select_index(scores: list[float], tournament_size: int, rng: random.Random) -> int:
    candidates = rng.sample(range(len(scores)), k=min(tournament_size, len(scores)))
    return min(candidates, key=lambda idx: scores[idx])


def select_distinct_parent_indices(scores: list[float], tournament_size: int, rng: random.Random) -> tuple[int, int]:
    left_idx = tournament_select_index(scores, tournament_size, rng)
    if len(scores) <= 1:
        return left_idx, left_idx

    right_idx = tournament_select_index(scores, tournament_size, rng)
    if right_idx != left_idx:
        return left_idx, right_idx

    remaining = [idx for idx in range(len(scores)) if idx != left_idx]
    fallback_idx = min(remaining, key=lambda idx: scores[idx])
    return left_idx, fallback_idx


def run_evolutionary_loop(
    *,
    cfg: EvoBenchmarkConfig,
    enwik8_path: Path,
    base_train_seconds: float,
    generations: int,
    population_size: int,
    tournament_size: int,
    crossover_strategy: str,
    crossover_percentile: float,
    mutation_std: float,
    mutation_fraction: float,
    eval_batches: int,
    ensemble_topks: tuple[int, ...],
    device: torch.device,
    param_dtype: torch.dtype,
) -> dict[str, Any]:
    train_tokens, val_tokens, test_tokens, dataset_meta = prepare_tokenized_enwik8_splits(
        enwik8_path,
        device=device,
        cache_on_device=cfg.cache_dataset_on_device,
        tokenization_mode=cfg.tokenization_mode,
        tokenizer_name=cfg.tokenizer_name,
        tokenizer_model_path=cfg.tokenizer_model_path,
    )
    train_starts = build_lm_starts(int(train_tokens.numel()), cfg.seq_len, cfg.stride)
    val_starts = build_lm_starts(int(val_tokens.numel()), cfg.seq_len, cfg.stride)
    test_starts = build_lm_starts(int(test_tokens.numel()), cfg.seq_len, cfg.stride)
    if not train_starts or not val_starts or not test_starts:
        raise ValueError("enwik8 split is too small for the requested seq_len/stride")

    val_eval_starts = choose_eval_starts(val_starts, batch_size=cfg.batch_size, eval_batches=eval_batches, seed=cfg.seed + 31_000)
    test_eval_starts = choose_eval_starts(test_starts, batch_size=cfg.batch_size, eval_batches=eval_batches, seed=cfg.seed + 32_000)

    model = build_model(cfg, device, param_dtype)
    base_init = parameter_state_dict(model)
    load_parameter_state(model, base_init, device)
    base_train, base_replay = train_for_seconds(
        model,
        train_tokens=train_tokens,
        train_starts=train_starts,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        base_lr=cfg.base_lr,
        weight_decay=cfg.weight_decay,
        grad_clip_norm=cfg.grad_clip_norm,
        train_seconds=base_train_seconds,
        seed=cfg.seed + 400,
        device=device,
        param_dtype=param_dtype,
        hard_replay_topk=cfg.hard_replay_topk,
    )
    base_state = parameter_state_dict(model)
    base_val = evaluate_bpb(
        model,
        tokens=val_tokens,
        eval_starts=val_eval_starts,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        device=device,
        param_dtype=param_dtype,
    )
    base_test = evaluate_bpb(
        model,
        tokens=test_tokens,
        eval_starts=test_eval_starts,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        device=device,
        param_dtype=param_dtype,
    )

    population = [
        mutate_state(
            base_state,
            mutation_std=mutation_std,
            mutation_fraction=mutation_fraction,
            seed=cfg.seed + 50_000 + member_idx,
        )
        for member_idx in range(population_size)
    ]
    rng = random.Random(cfg.seed + 60_000)
    history: list[dict[str, Any]] = []
    final_scored_population: list[tuple[dict[str, torch.Tensor], float]] = []

    for generation in range(generations):
        bpbs = population_bpbs(
            population,
            model=model,
            eval_tokens=val_tokens,
            eval_starts=val_eval_starts,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            device=device,
            param_dtype=param_dtype,
            population_chunk_size=population_size,
        )
        scored_population = list(zip(population, bpbs))
        scored_population.sort(key=lambda item: item[1])
        bpbs = [bpb for _, bpb in scored_population]
        history.append(
            {
                "generation": int(generation),
                "best_bpb": float(min(bpbs)),
                "mean_bpb": float(sum(bpbs) / len(bpbs)),
                "worst_bpb": float(max(bpbs)),
            }
        )
        elite_state = {name: value.clone() for name, value in scored_population[0][0].items()}
        next_population = [elite_state]
        while len(next_population) < population_size:
            if crossover_strategy == "parent_copy":
                parent_left_idx = tournament_select_index(bpbs, tournament_size, rng)
                parent_right_idx = parent_left_idx
            else:
                parent_left_idx, parent_right_idx = select_distinct_parent_indices(bpbs, tournament_size, rng)
            parent_left = scored_population[parent_left_idx][0]
            parent_right = scored_population[parent_right_idx][0]
            child_state, _ = crossover_state(
                parent_left,
                parent_right,
                strategy=crossover_strategy,
                percentile=crossover_percentile,
                seed=rng.randrange(1 << 30),
                base_state=base_state,
            )
            child_state = mutate_state(
                child_state,
                mutation_std=mutation_std,
                mutation_fraction=mutation_fraction,
                seed=rng.randrange(1 << 30),
            )
            next_population.append(child_state)
        population = next_population

    final_bpbs = population_bpbs(
        population,
        model=model,
        eval_tokens=val_tokens,
        eval_starts=val_eval_starts,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        device=device,
        param_dtype=param_dtype,
        population_chunk_size=population_size,
    )
    final_scored_population.extend(zip(population, final_bpbs))
    final_scored_population.sort(key=lambda item: item[1])
    final_best_state = final_scored_population[0][0]
    final_val_bpb = float(final_scored_population[0][1])
    final_test_bpb = state_bpb(
        final_best_state,
        model=model,
        eval_tokens=test_tokens,
        eval_starts=test_eval_starts,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        device=device,
        param_dtype=param_dtype,
    )
    ensemble_results = []
    requested_topks = sorted({topk for topk in ensemble_topks if topk > 0})
    for topk in requested_topks:
        actual_topk = min(topk, len(final_scored_population))
        selected_states = [state for state, _ in final_scored_population[:actual_topk]]
        ensemble_val_bpb = ensemble_bpb(
            selected_states,
            model=model,
            eval_tokens=val_tokens,
            eval_starts=val_eval_starts,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            device=device,
            param_dtype=param_dtype,
        )
        ensemble_test_bpb = ensemble_bpb(
            selected_states,
            model=model,
            eval_tokens=test_tokens,
            eval_starts=test_eval_starts,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            device=device,
            param_dtype=param_dtype,
        )
        ensemble_results.append(
            {
                "topk": int(actual_topk),
                "val_bpb": float(ensemble_val_bpb),
                "test_bpb": float(ensemble_test_bpb),
            }
        )
    return {
        "config": asdict(cfg),
        "dataset": {
            "enwik8_path": str(enwik8_path),
            "train_tokens": int(train_tokens.numel()),
            "val_tokens": int(val_tokens.numel()),
            "test_tokens": int(test_tokens.numel()),
            **dataset_meta,
        },
        "base": {
            "train": asdict(base_train),
            "val": base_val,
            "test": base_test,
            "replay_summary": summarize_replay_buffer(base_replay),
        },
        "evolution": {
            "population_size": int(population_size),
            "generations": int(generations),
            "tournament_size": int(tournament_size),
            "crossover_strategy": crossover_strategy,
            "crossover_percentile": float(crossover_percentile),
            "mutation_std": float(mutation_std),
            "mutation_fraction": float(mutation_fraction),
            "history": history,
            "final_best_val_bpb": float(final_val_bpb),
            "final_best_test_bpb": float(final_test_bpb),
            "ensemble_results": ensemble_results,
        },
    }


def run_vmap_throughput(
    *,
    cfg: EvoBenchmarkConfig,
    scales: tuple[int, ...],
    noise_std: float,
    warmup_repeats: int,
    timed_repeats: int,
    population_chunk_size: int | None,
    device: torch.device,
    param_dtype: torch.dtype,
) -> dict[str, Any]:
    model = build_model(cfg, device, param_dtype)
    inputs, targets = random_token_batch(
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        device=device,
        seed=cfg.seed + 700,
    )
    results = []
    maybe_reset_cuda_peak_memory(device)
    for population_size in scales:
        results.append(
            benchmark_population_vmap(
                model,
                population_size=population_size,
                inputs=inputs,
                targets=targets,
                noise_std=noise_std,
                warmup_repeats=warmup_repeats,
                timed_repeats=timed_repeats,
                population_chunk_size=population_chunk_size,
                seed=cfg.seed + 1000 + population_size,
                device=device,
                param_dtype=param_dtype,
            )
        )
    return {
        "config": asdict(cfg),
        "device": str(device),
        "param_dtype": str(param_dtype),
        "model_param_mb_estimate": float(
            model.compact_bytes() * torch.tensor([], dtype=param_dtype).element_size() / (1024.0 * 1024.0)
        ),
        "results": results,
    }


def parse_int_csv(spec: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in spec.split(",") if part.strip())


def parse_float_csv(spec: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in spec.split(",") if part.strip())


def parse_str_csv(spec: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in spec.split(",") if part.strip())


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default=EvoBenchmarkConfig.device)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default=EvoBenchmarkConfig.dtype)
    parser.add_argument("--seed", type=int, default=EvoBenchmarkConfig.seed)
    parser.add_argument("--output-json", default=EvoBenchmarkConfig.output_json)
    parser.add_argument("--model-dim", type=int, default=EvoBenchmarkConfig.model_dim)
    parser.add_argument("--num-layers", type=int, default=EvoBenchmarkConfig.num_layers)
    parser.add_argument("--num-heads", type=int, default=EvoBenchmarkConfig.num_heads)
    parser.add_argument("--num-kv-heads", type=int, default=EvoBenchmarkConfig.num_kv_heads)
    parser.add_argument("--mlp-mult", type=int, default=EvoBenchmarkConfig.mlp_mult)
    parser.add_argument("--seq-len", type=int, default=EvoBenchmarkConfig.seq_len)
    parser.add_argument("--stride", type=int, default=EvoBenchmarkConfig.stride)
    parser.add_argument("--batch-size", type=int, default=EvoBenchmarkConfig.batch_size)
    parser.add_argument("--base-lr", type=float, default=EvoBenchmarkConfig.base_lr)
    parser.add_argument("--weight-decay", type=float, default=EvoBenchmarkConfig.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--tokenization-mode", choices=("bytes", "sentencepiece"), default=EvoBenchmarkConfig.tokenization_mode)
    parser.add_argument("--tokenizer-name", default=EvoBenchmarkConfig.tokenizer_name)
    parser.add_argument("--tokenizer-model-path", default=EvoBenchmarkConfig.tokenizer_model_path)
    parser.add_argument("--qk-gain-init", type=float, default=EvoBenchmarkConfig.qk_gain_init)
    parser.add_argument("--spine-variant", choices=("plain", "xsa"), default=EvoBenchmarkConfig.spine_variant)
    parser.add_argument("--xsa-last-n", type=int, default=EvoBenchmarkConfig.xsa_last_n)
    parser.add_argument(
        "--cache-dataset-on-device",
        action=argparse.BooleanOptionalAction,
        default=EvoBenchmarkConfig.cache_dataset_on_device,
    )
    parser.add_argument("--hard-replay-topk", type=int, default=EvoBenchmarkConfig.hard_replay_topk)
    parser.add_argument("--hard-replay-eval-batches", type=int, default=EvoBenchmarkConfig.hard_replay_eval_batches)
    parser.add_argument(
        "--use-best-archive-for-spawn",
        action=argparse.BooleanOptionalAction,
        default=EvoBenchmarkConfig.use_best_archive_for_spawn,
    )


def config_from_args(args: argparse.Namespace) -> EvoBenchmarkConfig:
    tokenization_mode = getattr(args, "tokenization_mode", EvoBenchmarkConfig.tokenization_mode)
    tokenizer_name = getattr(args, "tokenizer_name", EvoBenchmarkConfig.tokenizer_name)
    tokenizer_model_path = getattr(args, "tokenizer_model_path", EvoBenchmarkConfig.tokenizer_model_path)
    if tokenization_mode == "bytes":
        if tokenizer_name is not None or tokenizer_model_path is not None:
            raise ValueError("tokenizer-name/tokenizer-model-path are only valid with --tokenization-mode sentencepiece")
        vocab_size = EvoBenchmarkConfig.vocab_size if args.vocab_size is None else int(args.vocab_size)
    else:
        resolved_vocab_size = resolve_tokenizer_vocab_size(
            tokenization_mode=tokenization_mode,
            tokenizer_name=tokenizer_name,
            tokenizer_model_path=tokenizer_model_path,
        )
        assert resolved_vocab_size is not None
        vocab_size = resolved_vocab_size if args.vocab_size is None else int(args.vocab_size)
        if vocab_size != resolved_vocab_size:
            raise ValueError(
                f"vocab_size={vocab_size} does not match tokenizer vocab_size={resolved_vocab_size} for {tokenization_mode}"
            )
    return EvoBenchmarkConfig(
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        output_json=args.output_json,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        vocab_size=vocab_size,
        tokenization_mode=tokenization_mode,
        tokenizer_name=tokenizer_name,
        tokenizer_model_path=tokenizer_model_path,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        qk_gain_init=args.qk_gain_init,
        spine_variant=args.spine_variant,
        xsa_last_n=args.xsa_last_n,
        cache_dataset_on_device=args.cache_dataset_on_device,
        hard_replay_topk=args.hard_replay_topk,
        hard_replay_eval_batches=args.hard_replay_eval_batches,
        use_best_archive_for_spawn=args.use_best_archive_for_spawn,
    )


def write_output(result: dict[str, Any], output_json: str | None) -> None:
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evolutionary optimization benchmark for small transformer language models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    vmap_parser = subparsers.add_parser("vmap-throughput", help="Benchmark torch.func/vmap throughput for a population of perturbed transformers")
    add_shared_args(vmap_parser)
    vmap_parser.add_argument("--population-scales", default="8,64,256,1024,4096,16384")
    vmap_parser.add_argument("--noise-std", type=float, default=1e-3)
    vmap_parser.add_argument("--warmup-repeats", type=int, default=2)
    vmap_parser.add_argument("--timed-repeats", type=int, default=5)
    vmap_parser.add_argument("--population-chunk-size", type=int, default=0)

    viability_parser = subparsers.add_parser("crossover-viability", help="Train several copies from the same init and test overlap crossover viability")
    add_shared_args(viability_parser)
    viability_parser.add_argument("--enwik8-path", type=Path, required=True)
    viability_parser.add_argument("--copies", type=int, default=8)
    viability_parser.add_argument("--train-seconds", type=float, default=120.0)
    viability_parser.add_argument(
        "--crossover-strategies",
        default="weight_overlap,delta_overlap,sign_consensus,delta_sign_consensus,tensor_swap,layer_swap,delta_importance",
    )
    viability_parser.add_argument("--percentiles", default="25,50,75")
    viability_parser.add_argument("--eval-batches", type=int, default=32)
    viability_parser.add_argument("--pair-limit", type=int, default=None)
    viability_parser.add_argument("--ensemble-topks", default="")
    viability_parser.add_argument("--member-train-mode", choices=("serial", "parallel_vmap"), default="serial")

    committee_parser = subparsers.add_parser("committee-schedule", help="Train a committee with staged depth-to-breadth expansion")
    add_shared_args(committee_parser)
    committee_parser.add_argument("--enwik8-path", type=Path, required=True)
    committee_parser.add_argument("--stage-copies", required=True)
    committee_parser.add_argument("--stage-train-seconds", required=True)
    committee_parser.add_argument("--eval-batches", type=int, default=32)
    committee_parser.add_argument("--ensemble-topks", default="")
    committee_parser.add_argument("--spawn-noise-std", type=float, default=0.0)

    compress_parser = subparsers.add_parser("committee-compressibility", help="Train a committee and test how much ensemble gain survives delta compression")
    add_shared_args(compress_parser)
    compress_parser.add_argument("--enwik8-path", type=Path, required=True)
    compress_parser.add_argument("--stage-copies", required=True)
    compress_parser.add_argument("--stage-train-seconds", required=True)
    compress_parser.add_argument("--eval-batches", type=int, default=32)
    compress_parser.add_argument("--ensemble-topks", default="")
    compress_parser.add_argument("--spawn-noise-std", type=float, default=0.0)
    compress_parser.add_argument("--artifact-limit-mb", type=float, default=16.0)
    compress_parser.add_argument("--delta-budget-fractions", default="0.25,0.5,1.0")
    compress_parser.add_argument("--basis-ranks", default="1,2,4")
    compress_parser.add_argument("--analysis-topk", type=int, default=8192)

    adaptive_parser = subparsers.add_parser("committee-adaptive", help="Train a committee with an adaptive widen/narrow controller")
    add_shared_args(adaptive_parser)
    adaptive_parser.add_argument("--enwik8-path", type=Path, required=True)
    adaptive_parser.add_argument("--initial-copies", type=int, default=2)
    adaptive_parser.add_argument("--round-train-seconds", required=True)
    adaptive_parser.add_argument("--eval-batches", type=int, default=32)
    adaptive_parser.add_argument("--ensemble-topks", default="")
    adaptive_parser.add_argument("--spawn-noise-std", type=float, default=0.0)
    adaptive_parser.add_argument("--min-copies", type=int, default=2)
    adaptive_parser.add_argument("--max-copies", type=int, default=16)
    adaptive_parser.add_argument("--widen-factor", type=int, default=2)
    adaptive_parser.add_argument("--narrow-factor", type=int, default=2)
    adaptive_parser.add_argument("--depth-gain-threshold", type=float, default=0.01)
    adaptive_parser.add_argument("--breadth-gain-threshold", type=float, default=0.02)
    adaptive_parser.add_argument("--archive-hit-rate-threshold", type=float, default=0.25)
    adaptive_parser.add_argument("--winner-concentration-threshold", type=float, default=0.85)
    adaptive_parser.add_argument("--replay-disagreement-threshold", type=float, default=0.02)
    adaptive_parser.add_argument("--pairwise-distance-threshold", type=float, default=0.01)

    recipe_parser = subparsers.add_parser("recipe-evolution", help="Evolve recipe genes instead of weights under a 16MB artifact budget")
    add_shared_args(recipe_parser)
    recipe_parser.add_argument("--enwik8-path", type=Path, required=True)
    recipe_parser.add_argument("--population-size", type=int, default=12)
    recipe_parser.add_argument("--generations", type=int, default=6)
    recipe_parser.add_argument("--tournament-size", type=int, default=3)
    recipe_parser.add_argument("--train-seconds", type=float, default=90.0)
    recipe_parser.add_argument("--eval-batches", type=int, default=8)
    recipe_parser.add_argument("--mutation-rate", type=float, default=0.2)
    recipe_parser.add_argument("--artifact-limit-mb", type=float, default=16.0)
    recipe_parser.add_argument("--recipe-profile", choices=("compact", "frontier"), default="frontier")
    recipe_parser.add_argument("--confirm-topk", type=int, default=3)
    recipe_parser.add_argument("--confirm-train-seconds", type=float, default=180.0)

    evo_parser = subparsers.add_parser("evolution-loop", help="Run an end-to-end evolutionary loop after a short gradient-based pretrain")
    add_shared_args(evo_parser)
    evo_parser.add_argument("--enwik8-path", type=Path, required=True)
    evo_parser.add_argument("--base-train-seconds", type=float, default=120.0)
    evo_parser.add_argument("--generations", type=int, default=8)
    evo_parser.add_argument("--population-size", type=int, default=16)
    evo_parser.add_argument("--tournament-size", type=int, default=4)
    evo_parser.add_argument("--crossover-strategy", default="delta_overlap")
    evo_parser.add_argument("--crossover-percentile", type=float, default=50.0)
    evo_parser.add_argument("--mutation-std", type=float, default=5e-4)
    evo_parser.add_argument("--mutation-fraction", type=float, default=0.05)
    evo_parser.add_argument("--eval-batches", type=int, default=32)
    evo_parser.add_argument("--ensemble-topks", default="")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    param_dtype = resolve_param_dtype(cfg.dtype, device)

    if args.command == "vmap-throughput":
        result = run_vmap_throughput(
            cfg=cfg,
            scales=parse_int_csv(args.population_scales),
            noise_std=args.noise_std,
            warmup_repeats=args.warmup_repeats,
            timed_repeats=args.timed_repeats,
            population_chunk_size=(args.population_chunk_size or None),
            device=device,
            param_dtype=param_dtype,
        )
        write_output(result, cfg.output_json)
        return

    if args.command == "crossover-viability":
        result = run_crossover_viability(
            cfg=cfg,
            enwik8_path=args.enwik8_path,
            copies=args.copies,
            train_seconds=args.train_seconds,
            strategies=parse_str_csv(args.crossover_strategies),
            percentiles=parse_float_csv(args.percentiles),
            eval_batches=args.eval_batches,
            pair_limit=args.pair_limit,
            ensemble_topks=parse_int_csv(args.ensemble_topks),
            member_train_mode=args.member_train_mode,
            device=device,
            param_dtype=param_dtype,
        )
        write_output(result, cfg.output_json)
        return

    if args.command == "committee-schedule":
        result = run_committee_schedule(
            cfg=cfg,
            enwik8_path=args.enwik8_path,
            stage_copies=parse_int_csv(args.stage_copies),
            stage_train_seconds=parse_float_csv(args.stage_train_seconds),
            ensemble_topks=parse_int_csv(args.ensemble_topks),
            eval_batches=args.eval_batches,
            spawn_noise_std=args.spawn_noise_std,
            device=device,
            param_dtype=param_dtype,
        )
        write_output(result, cfg.output_json)
        return

    if args.command == "committee-compressibility":
        result = run_committee_compressibility(
            cfg=cfg,
            enwik8_path=args.enwik8_path,
            stage_copies=parse_int_csv(args.stage_copies),
            stage_train_seconds=parse_float_csv(args.stage_train_seconds),
            ensemble_topks=parse_int_csv(args.ensemble_topks),
            eval_batches=args.eval_batches,
            spawn_noise_std=args.spawn_noise_std,
            artifact_limit_mb=args.artifact_limit_mb,
            delta_budget_fractions=parse_float_csv(args.delta_budget_fractions),
            basis_ranks=parse_int_csv(args.basis_ranks),
            analysis_topk=args.analysis_topk,
            device=device,
            param_dtype=param_dtype,
        )
        write_output(result, cfg.output_json)
        return

    if args.command == "committee-adaptive":
        result = run_committee_adaptive(
            cfg=cfg,
            enwik8_path=args.enwik8_path,
            initial_copies=args.initial_copies,
            round_train_seconds=parse_float_csv(args.round_train_seconds),
            ensemble_topks=parse_int_csv(args.ensemble_topks),
            eval_batches=args.eval_batches,
            spawn_noise_std=args.spawn_noise_std,
            min_copies=args.min_copies,
            max_copies=args.max_copies,
            widen_factor=args.widen_factor,
            narrow_factor=args.narrow_factor,
            depth_gain_threshold=args.depth_gain_threshold,
            breadth_gain_threshold=args.breadth_gain_threshold,
            archive_hit_rate_threshold=args.archive_hit_rate_threshold,
            winner_concentration_threshold=args.winner_concentration_threshold,
            replay_disagreement_threshold=args.replay_disagreement_threshold,
            pairwise_distance_threshold=args.pairwise_distance_threshold,
            device=device,
            param_dtype=param_dtype,
        )
        write_output(result, cfg.output_json)
        return

    if args.command == "recipe-evolution":
        result = run_recipe_evolution(
            cfg=cfg,
            enwik8_path=args.enwik8_path,
            population_size=args.population_size,
            generations=args.generations,
            tournament_size=args.tournament_size,
            train_seconds=args.train_seconds,
            eval_batches=args.eval_batches,
            mutation_rate=args.mutation_rate,
            artifact_limit_mb=args.artifact_limit_mb,
            recipe_profile=args.recipe_profile,
            confirm_topk=args.confirm_topk,
            confirm_train_seconds=args.confirm_train_seconds,
            device=device,
            param_dtype=param_dtype,
        )
        write_output(result, cfg.output_json)
        return

    if args.command == "evolution-loop":
        result = run_evolutionary_loop(
            cfg=cfg,
            enwik8_path=args.enwik8_path,
            base_train_seconds=args.base_train_seconds,
            generations=args.generations,
            population_size=args.population_size,
            tournament_size=args.tournament_size,
            crossover_strategy=args.crossover_strategy,
            crossover_percentile=args.crossover_percentile,
            mutation_std=args.mutation_std,
            mutation_fraction=args.mutation_fraction,
            eval_batches=args.eval_batches,
            ensemble_topks=parse_int_csv(args.ensemble_topks),
            device=device,
            param_dtype=param_dtype,
        )
        write_output(result, cfg.output_json)
        return

    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
