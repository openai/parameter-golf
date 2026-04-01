from __future__ import annotations

import argparse
import contextlib
import dataclasses
import glob
import hashlib
import io
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter


# ============================================================================
# CONSTANTS / ERRORS
# ============================================================================


RESULTS_SCHEMA_VERSION = "pgolf.results.v1"
ARTIFACT_MANIFEST_VERSION = "pgolf.artifact_manifest.v1"
CHECKPOINT_SCHEMA_VERSION = "pgolf.checkpoint.v1"
METRICS_STREAM_VERSION = "pgolf.metrics.v1"

EXIT_SUCCESS = 0
EXIT_INVALID_CONFIG = 2
EXIT_RUNTIME_ERROR = 3
EXIT_INTERRUPTED = 130

ALLOWED_ADAPTER_TARGETS = ("q", "k", "v", "attn_out", "mlp_in", "mlp_out")
HASH_EXCLUDED_KEYS = {
    "output_dir",
    "results_tsv_path",
    "metrics_jsonl_path",
    "tensorboard_log_dir",
    "run_name",
    "resume_from",
    "load_artifact_path",
}
RESUME_HASH_EXCLUDED_KEYS = HASH_EXCLUDED_KEYS | {
    "iterations",
    "max_wallclock_seconds",
    "checkpoint_every",
    "save_final_quantized",
    "verify_export_reload",
    "benchmark_only",
    "benchmark_train_steps",
    "benchmark_eval_repeats",
    "evaluate_only",
}


class ConfigError(ValueError):
    pass


class ResultsSchemaError(ValueError):
    pass


# ============================================================================
# CONFIG
# ============================================================================


@dataclass
class ModelConfig:
    vocab_size: int = 1024
    seq_len: int = 1024
    d_model: int = 768
    num_heads: int = 12
    num_kv_heads: int = 4
    mlp_mult: int = 2
    non_recurrent_mlp_hidden_bonus: int | None = None
    shared_mlp_hidden_bonus: int = 0
    rope_base: float = 10_000.0
    logit_softcap: float = 30.0
    tie_embeddings: bool = True
    emb_init_std: float = 0.02
    stem_layers: int = 1
    shared_layers: int = 3
    recurrence_loops: int = 3
    tail_layers: int = 1
    adapter_rank: int = 4
    adapter_alpha: float = 8.0
    adapter_targets: tuple[str, ...] = ("attn_out", "mlp_out")
    qk_gain_init: float = 1.0
    q_low_rank: int = 0
    final_tail_q_low_rank: int | None = None
    fake_quant_during_train: bool = True
    attn_fake_quant_during_train: bool | None = None
    fake_quant_start_step: int = 50
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    @property
    def effective_depth(self) -> int:
        return self.stem_layers + self.shared_layers * self.recurrence_loops + self.tail_layers


@dataclass
class OptimConfig:
    embed_lr: float = 3e-3
    head_lr: float = 3e-3
    matrix_lr: float = 2e-2
    scalar_lr: float = 3e-3
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    muon_momentum: float = 0.95
    muon_backend_steps: int = 5
    grad_clip_norm: float = 1.0
    warmup_steps: int = 20
    warmdown_steps: int = 100
    min_lr_scale: float = 0.1


@dataclass
class QuantConfig:
    keep_float_max_numel: int = 65_536
    keep_float_name_patterns: tuple[str, ...] = (
        "norm",
        "scale",
        "gain",
        "adapter",
        "lm_head",
    )
    scale_store_dtype: torch.dtype = torch.float16
    keep_float_store_dtype: torch.dtype = torch.float16
    low_bit_name_patterns: tuple[str, ...] = ()
    low_bit_bits: int = 8
    clip_percentile: float = 99.999
    zlib_level: int = 9


@dataclass
class TrainConfig:
    train_pattern: str = "../../../data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"
    val_pattern: str = "../../../data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"
    tokenizer_path: str | None = "../../../data/tokenizers/fineweb_1024_bpe.model"
    output_dir: str = "./out_autoresearch_parameter_golf"
    run_name: str | None = None
    results_tsv_path: str = "./results.tsv"
    metrics_jsonl_path: str | None = None
    tensorboard_log_dir: str | None = None
    seed: int = 1337
    deterministic: bool = True
    iterations: int = 20_000
    train_batch_tokens: int = 524_288
    val_batch_tokens: int = 524_288
    grad_accum_steps: int = 1
    train_seq_len_min: int | None = None
    train_seq_len_warmup_steps: int = 0
    log_every: int = 100
    val_every: int = 1_000
    eval_first_step: bool = True
    max_wallclock_seconds: float = 600.0
    checkpoint_every: int = 0
    resume_from: str | None = None
    use_compile: bool = False
    use_lawa: bool = True
    lawa_last_n_steps: int = 200
    save_final_quantized: bool = True
    verify_export_reload: bool = True
    evaluate_only: bool = False
    train_phase_only: bool = False
    load_artifact_path: str | None = None
    benchmark_only: bool = False
    benchmark_train_steps: int = 3
    benchmark_eval_repeats: int = 1
    counted_code_paths: tuple[str, ...] = ("train_gpt.py",)
    artifact_bundle_name: str = "submission_bundle"
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)

    @property
    def local_batch_tokens(self) -> int:
        return self.train_batch_tokens // max(1, self.grad_accum_steps)


@dataclass
class ExportStats:
    artifact_dir: str
    manifest_path: str
    model_blob_path: str
    compressed_model_bytes: int
    code_bytes: int
    artifact_bytes: int
    quant_payload_bytes: int
    reload_val_loss: float | None = None
    reload_val_bpb: float | None = None


@dataclass
class EvalStats:
    val_loss: float
    val_bpb: float | None
    token_count: int
    byte_count: int | None
    eval_seconds: float


@dataclass
class BenchmarkStats:
    train_tokens_per_second: float
    eval_tokens_per_second: float
    eval_seconds: float


@dataclass
class RunSummary:
    status: str
    mode: str
    run_id: str
    config_hash: str
    step: int
    num_steps: int
    train_loss: float | None
    val_loss: float | None
    val_bpb: float | None
    export: ExportStats | None
    benchmark: BenchmarkStats | None
    training_seconds: float
    total_seconds: float
    peak_vram_mb: float
    mfu_percent: float
    total_tokens: int
    total_tokens_M: float
    model_params: int
    num_params_M: float
    effective_depth: int
    checkpoint_path: str | None
    metrics_path: str | None
    tensorboard_log_dir: str | None
    results_path: str | None


@dataclass
class DistInfo:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int


@dataclass
class ResumeState:
    run_id: str
    config_hash: str
    next_step: int
    elapsed_training_seconds: float
    total_tokens: int
    last_train_loss: float | None
    last_val: EvalStats | None
    checkpoint_path: str


# ============================================================================
# JSON / HASH / FILE UTILS
# ============================================================================


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return value


def canonical_json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    return _jsonable(cfg)


def _maybe_parse_torch_dtype(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("torch."):
        return getattr(torch, value.removeprefix("torch."))
    return value


def train_config_from_dict(data: Mapping[str, Any]) -> TrainConfig:
    payload = dict(data)
    cfg = TrainConfig(**{k: v for k, v in payload.items() if k not in {"model", "optim", "quant"}})
    if "model" in payload:
        model_payload = dict(payload["model"])
        if "adapter_targets" in model_payload and isinstance(model_payload["adapter_targets"], list):
            model_payload["adapter_targets"] = tuple(model_payload["adapter_targets"])
        cfg.model = ModelConfig(**model_payload)
    if "optim" in payload:
        cfg.optim = OptimConfig(**dict(payload["optim"]))
    if "quant" in payload:
        quant_payload = dict(payload["quant"])
        for key in ("scale_store_dtype", "keep_float_store_dtype"):
            if key in quant_payload:
                quant_payload[key] = _maybe_parse_torch_dtype(quant_payload[key])
        if "keep_float_name_patterns" in quant_payload and isinstance(quant_payload["keep_float_name_patterns"], list):
            quant_payload["keep_float_name_patterns"] = tuple(quant_payload["keep_float_name_patterns"])
        if "low_bit_name_patterns" in quant_payload and isinstance(quant_payload["low_bit_name_patterns"], list):
            quant_payload["low_bit_name_patterns"] = tuple(quant_payload["low_bit_name_patterns"])
        cfg.quant = QuantConfig(**quant_payload)
    if isinstance(cfg.counted_code_paths, list):
        cfg.counted_code_paths = tuple(cfg.counted_code_paths)
    return cfg


def rebalance_shared_layers_vs_loops(model_cfg: ModelConfig) -> None:
    total_recurrent_applications = model_cfg.shared_layers * model_cfg.recurrence_loops
    if total_recurrent_applications < 8:
        return
    if model_cfg.shared_layers <= 0 or model_cfg.recurrence_loops < 4:
        return
    if model_cfg.recurrence_loops < model_cfg.shared_layers * 2:
        return
    target_loops = max(2, model_cfg.recurrence_loops // 2)
    if total_recurrent_applications % target_loops != 0:
        return
    target_shared_layers = total_recurrent_applications // target_loops
    if target_shared_layers <= model_cfg.shared_layers:
        return
    model_cfg.shared_layers = target_shared_layers
    model_cfg.recurrence_loops = target_loops


def expand_adapter_capacity(model_cfg: ModelConfig) -> None:
    if model_cfg.shared_layers < model_cfg.recurrence_loops:
        return
    if model_cfg.shared_layers * model_cfg.recurrence_loops < 8:
        return
    if model_cfg.adapter_rank != 4 or set(model_cfg.adapter_targets) != {"attn_out", "mlp_out"}:
        return
    model_cfg.adapter_rank = 8
    model_cfg.adapter_alpha = 16.0
    model_cfg.adapter_targets = ALLOWED_ADAPTER_TARGETS


def reallocate_one_shared_layer_into_tail(model_cfg: ModelConfig) -> None:
    if model_cfg.shared_layers != 4 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 1:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    total_depth = model_cfg.effective_depth
    model_cfg.shared_layers -= 1
    model_cfg.tail_layers = total_depth - model_cfg.stem_layers - model_cfg.shared_layers * model_cfg.recurrence_loops


def reallocate_second_shared_layer_into_tail(model_cfg: ModelConfig) -> None:
    if model_cfg.shared_layers != 3 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 3:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    total_depth = model_cfg.effective_depth
    model_cfg.shared_layers -= 1
    model_cfg.tail_layers = total_depth - model_cfg.stem_layers - model_cfg.shared_layers * model_cfg.recurrence_loops


def reallocate_third_shared_layer_into_tail(model_cfg: ModelConfig) -> None:
    if model_cfg.shared_layers != 2 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 5:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    total_depth = model_cfg.effective_depth
    model_cfg.shared_layers -= 1
    model_cfg.tail_layers = total_depth - model_cfg.stem_layers - model_cfg.shared_layers * model_cfg.recurrence_loops


def move_fake_quant_to_warmup_boundary_on_deep_tail(model_cfg: ModelConfig) -> None:
    if model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 7:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if model_cfg.fake_quant_start_step != 50:
        return
    model_cfg.fake_quant_start_step = 20


def widen_recurrent_mlp_on_deep_tail(model_cfg: ModelConfig) -> None:
    if model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 7:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != 0:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    model_cfg.shared_mlp_hidden_bonus = model_cfg.d_model // 4


def tighten_export_clip_on_accepted_deep_tail(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 7:
        return
    if model_cfg.shared_mlp_hidden_bonus != model_cfg.d_model // 4:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if not math.isclose(cfg.quant.clip_percentile, 99.999, rel_tol=0.0, abs_tol=1e-9):
        return
    cfg.quant.clip_percentile = 96.5


def modestly_widen_recurrent_mlp_on_wallclock_deep_tail(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 7:
        return
    if model_cfg.shared_mlp_hidden_bonus != model_cfg.d_model // 4:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    model_cfg.shared_mlp_hidden_bonus = (model_cfg.d_model * 7) // 16


def shift_accepted_deep_tail_stem_into_tail(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 1 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 7:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != (model_cfg.d_model * 7) // 16:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    total_depth = model_cfg.effective_depth
    model_cfg.stem_layers = 0
    model_cfg.tail_layers = total_depth - model_cfg.shared_layers * model_cfg.recurrence_loops


def retune_shifted_deep_tail_width_and_warmdown(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 8:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != (model_cfg.d_model * 7) // 16:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 100:
        return
    model_cfg.shared_mlp_hidden_bonus = (model_cfg.d_model * 3) // 8
    cfg.optim.warmdown_steps = 80


def use_int6_mlp_export_on_retuned_stemless_deep_tail(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 8:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != (model_cfg.d_model * 3) // 8:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_name_patterns:
        return
    if cfg.quant.low_bit_bits != 8:
        return
    cfg.quant.low_bit_bits = 6
    cfg.quant.low_bit_name_patterns = ("mlp.fc.weight", "mlp.proj.weight")


def trade_one_tail_block_for_true_3x_tail_mlp_on_int6_baseline(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 8:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != (model_cfg.d_model * 3) // 8:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus not in {None, model_cfg.d_model // 2}:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    model_cfg.non_recurrent_mlp_hidden_bonus = model_cfg.d_model
    model_cfg.tail_layers = 7


def trade_one_more_tail_block_for_3p5x_tail_mlp_on_int6_true_3x_line(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 7:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != (model_cfg.d_model * 3) // 8:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != model_cfg.d_model:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    model_cfg.non_recurrent_mlp_hidden_bonus = (model_cfg.d_model * 3) // 2
    model_cfg.tail_layers = 6


def trade_one_more_tail_block_for_4x_tail_mlp_on_int6_3p5x_line(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 6:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != (model_cfg.d_model * 3) // 8:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != (model_cfg.d_model * 3) // 2:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    model_cfg.non_recurrent_mlp_hidden_bonus = model_cfg.d_model * 2
    model_cfg.tail_layers = 5


def trade_one_more_tail_block_for_4p5x_tail_mlp_on_int6_4x_line(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 5:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != (model_cfg.d_model * 3) // 8:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != model_cfg.d_model * 2:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    model_cfg.non_recurrent_mlp_hidden_bonus = (model_cfg.d_model * 5) // 2
    model_cfg.tail_layers = 4


def widen_unique_tail_mlp_on_int6_4p5x_line(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 4:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != (model_cfg.d_model * 3) // 8:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != (model_cfg.d_model * 5) // 2:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    model_cfg.non_recurrent_mlp_hidden_bonus = model_cfg.d_model * 3


def widen_unique_tail_mlp_on_int6_5x_line(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 4:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != (model_cfg.d_model * 3) // 8:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != model_cfg.d_model * 3:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    model_cfg.non_recurrent_mlp_hidden_bonus = model_cfg.d_model * 4


def extend_context_on_compact_int6_6x_tail_line(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 4:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != (model_cfg.d_model * 3) // 8:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != model_cfg.d_model * 4:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if model_cfg.seq_len != 512:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    if cfg.grad_accum_steps != 4:
        return
    if cfg.train_batch_tokens != 131072 or cfg.val_batch_tokens != 131072:
        return
    model_cfg.seq_len = 640
    cfg.train_batch_tokens = 122_880
    cfg.val_batch_tokens = 122_880


def trade_one_tail_block_for_8x_tail_mlp_on_compact_seq640_line(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 4:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != (model_cfg.d_model * 3) // 8:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != model_cfg.d_model * 4:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if model_cfg.seq_len != 640:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    if cfg.grad_accum_steps != 4:
        return
    if cfg.train_batch_tokens != 122_880 or cfg.val_batch_tokens != 122_880:
        return
    model_cfg.tail_layers = 3
    model_cfg.non_recurrent_mlp_hidden_bonus = model_cfg.d_model * 6


def trade_one_more_tail_block_for_12x_tail_mlp_on_compact_seq640_line(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 3:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != (model_cfg.d_model * 3) // 8:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != model_cfg.d_model * 6:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if model_cfg.seq_len != 640:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    if cfg.grad_accum_steps != 4:
        return
    if cfg.train_batch_tokens != 122_880 or cfg.val_batch_tokens != 122_880:
        return
    model_cfg.tail_layers = 2
    model_cfg.non_recurrent_mlp_hidden_bonus = model_cfg.d_model * 10


def extend_context_on_compact_tail2_12x_line(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 2:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != (model_cfg.d_model * 3) // 8:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != model_cfg.d_model * 10:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if model_cfg.seq_len != 640:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    if cfg.grad_accum_steps != 4:
        return
    if cfg.train_batch_tokens != 122_880 or cfg.val_batch_tokens != 122_880:
        return
    model_cfg.seq_len = 768


def rebalance_compact_seq768_tail2_12x_line_into_tail3_8x(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 2 or model_cfg.tail_layers != 2:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != (model_cfg.d_model * 3) // 8:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != model_cfg.d_model * 10:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if model_cfg.seq_len != 768:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    if cfg.grad_accum_steps != 4:
        return
    if cfg.train_batch_tokens != 122_880 or cfg.val_batch_tokens != 122_880:
        return
    model_cfg.recurrence_loops = 1
    model_cfg.tail_layers = 3
    model_cfg.shared_mlp_hidden_bonus = 0
    model_cfg.non_recurrent_mlp_hidden_bonus = model_cfg.d_model * 6


def reallocate_low_rank_q_into_true_3x_carrier_on_recovered_compact_line(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 1 or model_cfg.tail_layers != 3:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != 0:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != model_cfg.d_model * 6:
        return
    if model_cfg.q_low_rank != 0:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if model_cfg.seq_len != 768:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    if cfg.grad_accum_steps != 4:
        return
    if cfg.train_batch_tokens != 122_880 or cfg.val_batch_tokens != 122_880:
        return
    model_cfg.q_low_rank = model_cfg.d_model // 4
    model_cfg.shared_mlp_hidden_bonus = model_cfg.d_model


def add_short_to_full_context_curriculum_on_low_rank_q_compact_line(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 1 or model_cfg.tail_layers != 3:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != model_cfg.d_model:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != model_cfg.d_model * 6:
        return
    if model_cfg.q_low_rank != model_cfg.d_model // 4:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if model_cfg.seq_len != 768:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    if cfg.grad_accum_steps != 4:
        return
    if cfg.train_batch_tokens != 122_880 or cfg.val_batch_tokens != 122_880:
        return
    if cfg.train_seq_len_min is not None or cfg.train_seq_len_warmup_steps != 0:
        return
    cfg.train_seq_len_min = 640
    cfg.train_seq_len_warmup_steps = 160


def disable_attention_fake_quant_on_warmed_low_rank_q_compact_line(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 1 or model_cfg.tail_layers != 3:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != model_cfg.d_model:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != model_cfg.d_model * 6:
        return
    if model_cfg.q_low_rank != model_cfg.d_model // 4:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if not model_cfg.fake_quant_during_train:
        return
    if model_cfg.attn_fake_quant_during_train is not None:
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if model_cfg.seq_len != 768:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    if cfg.grad_accum_steps != 4:
        return
    if cfg.train_batch_tokens != 122_880 or cfg.val_batch_tokens != 122_880:
        return
    if cfg.train_seq_len_min != 640 or cfg.train_seq_len_warmup_steps != 160:
        return
    model_cfg.attn_fake_quant_during_train = False


def shrink_global_batch_on_selective_qat_low_rank_q_compact_line(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 1 or model_cfg.tail_layers != 3:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != model_cfg.d_model:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != model_cfg.d_model * 6:
        return
    if model_cfg.q_low_rank != model_cfg.d_model // 4:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if not model_cfg.fake_quant_during_train or model_cfg.attn_fake_quant_during_train is not False:
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if model_cfg.seq_len != 768:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    if cfg.grad_accum_steps != 4:
        return
    if cfg.train_batch_tokens != 122_880 or cfg.val_batch_tokens != 122_880:
        return
    if cfg.train_seq_len_min != 640 or cfg.train_seq_len_warmup_steps != 160:
        return
    cfg.grad_accum_steps = 3
    cfg.train_batch_tokens = 92_160


def shrink_global_batch_further_on_selective_qat_low_rank_q_compact_line(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 1 or model_cfg.tail_layers != 3:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != model_cfg.d_model:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != model_cfg.d_model * 6:
        return
    if model_cfg.q_low_rank != model_cfg.d_model // 4:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if not model_cfg.fake_quant_during_train or model_cfg.attn_fake_quant_during_train is not False:
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if model_cfg.seq_len != 768:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    if cfg.grad_accum_steps != 3:
        return
    if cfg.train_batch_tokens != 92_160 or cfg.val_batch_tokens != 122_880:
        return
    if cfg.train_seq_len_min != 640 or cfg.train_seq_len_warmup_steps != 160:
        return
    cfg.grad_accum_steps = 2
    cfg.train_batch_tokens = 61_440


def delay_mlp_fake_quant_until_full_context_on_small_batch_selective_qat_line(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 1 or model_cfg.tail_layers != 3:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != model_cfg.d_model:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != model_cfg.d_model * 6:
        return
    if model_cfg.q_low_rank != model_cfg.d_model // 4:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if not model_cfg.fake_quant_during_train or model_cfg.attn_fake_quant_during_train is not False:
        return
    if model_cfg.fake_quant_start_step != 20:
        return
    if model_cfg.seq_len != 768:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    if cfg.grad_accum_steps != 2:
        return
    if cfg.train_batch_tokens != 61_440 or cfg.val_batch_tokens != 122_880:
        return
    if cfg.train_seq_len_min != 640 or cfg.train_seq_len_warmup_steps != 160:
        return
    model_cfg.fake_quant_start_step = cfg.train_seq_len_warmup_steps


def restore_full_rank_q_on_final_tail_of_late_qat_small_batch_line(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 1 or model_cfg.tail_layers != 3:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != model_cfg.d_model:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != model_cfg.d_model * 6:
        return
    if model_cfg.q_low_rank != model_cfg.d_model // 4:
        return
    if model_cfg.final_tail_q_low_rank is not None:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if not model_cfg.fake_quant_during_train or model_cfg.attn_fake_quant_during_train is not False:
        return
    if model_cfg.fake_quant_start_step != cfg.train_seq_len_warmup_steps:
        return
    if model_cfg.seq_len != 768:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    if cfg.grad_accum_steps != 2:
        return
    if cfg.train_batch_tokens != 61_440 or cfg.val_batch_tokens != 122_880:
        return
    if cfg.train_seq_len_min != 640 or cfg.train_seq_len_warmup_steps != 160:
        return
    model_cfg.final_tail_q_low_rank = 0


def keep_tied_embeddings_float_on_final_tail_q_small_batch_line(cfg: TrainConfig) -> None:
    model_cfg = cfg.model
    if not model_cfg.tie_embeddings:
        return
    if model_cfg.stem_layers != 0 or model_cfg.shared_layers != 1 or model_cfg.recurrence_loops != 1 or model_cfg.tail_layers != 3:
        return
    if model_cfg.mlp_mult != 2 or model_cfg.shared_mlp_hidden_bonus != model_cfg.d_model:
        return
    if model_cfg.non_recurrent_mlp_hidden_bonus != model_cfg.d_model * 6:
        return
    if model_cfg.q_low_rank != model_cfg.d_model // 4:
        return
    if model_cfg.final_tail_q_low_rank != 0:
        return
    if model_cfg.adapter_rank != 8 or tuple(model_cfg.adapter_targets) != ALLOWED_ADAPTER_TARGETS:
        return
    if not math.isclose(model_cfg.adapter_alpha, 16.0, rel_tol=0.0, abs_tol=1e-9):
        return
    if not model_cfg.fake_quant_during_train or model_cfg.attn_fake_quant_during_train is not False:
        return
    if model_cfg.fake_quant_start_step != cfg.train_seq_len_warmup_steps:
        return
    if model_cfg.seq_len != 768:
        return
    if not math.isclose(cfg.quant.clip_percentile, 96.5, rel_tol=0.0, abs_tol=1e-9):
        return
    if cfg.optim.warmdown_steps != 80:
        return
    if cfg.quant.low_bit_bits != 6:
        return
    if tuple(cfg.quant.low_bit_name_patterns) != ("mlp.fc.weight", "mlp.proj.weight"):
        return
    if cfg.grad_accum_steps != 2:
        return
    if cfg.train_batch_tokens != 61_440 or cfg.val_batch_tokens != 122_880:
        return
    if cfg.train_seq_len_min != 640 or cfg.train_seq_len_warmup_steps != 160:
        return
    if "tok_emb.weight" in cfg.quant.keep_float_name_patterns:
        return
    cfg.quant.keep_float_name_patterns = (*cfg.quant.keep_float_name_patterns, "tok_emb.weight")


def _dict_without_keys(data: Mapping[str, Any], keys: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key in keys:
            continue
        if isinstance(value, dict):
            out[key] = _dict_without_keys(value, set())
        else:
            out[key] = value
    return out


def config_hash(cfg: TrainConfig, *, for_resume: bool = False) -> str:
    exclude = RESUME_HASH_EXCLUDED_KEYS if for_resume else HASH_EXCLUDED_KEYS
    payload = _dict_without_keys(config_to_dict(cfg), exclude)
    return hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest()


def make_run_id(cfg: TrainConfig, cfg_hash: str) -> str:
    if cfg.run_name:
        return cfg.run_name
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{stamp}-{cfg_hash[:10]}"


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (repo_root() / p).resolve()


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json_dumps(payload) + "\n")
        handle.flush()
    return path


def resolve_run_output_path(output_dir: Path, configured_path: str | None, default_name: str) -> Path:
    if configured_path is None:
        return output_dir / default_name
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return output_dir / path


def resolve_metrics_jsonl_path(output_dir: Path, configured_path: str | None) -> Path:
    return resolve_run_output_path(output_dir, configured_path, "metrics.jsonl")


def resolve_tensorboard_log_dir(output_dir: Path, configured_path: str | None) -> Path:
    return resolve_run_output_path(output_dir, configured_path, "tensorboard")


def current_optimizer_lr(opt: torch.optim.Optimizer | None) -> float | None:
    if opt is None or not opt.param_groups:
        return None
    return float(opt.param_groups[0]["lr"])


def append_metrics_event(path: str | Path, payload: Mapping[str, Any]) -> Path:
    event = {
        "schema_version": METRICS_STREAM_VERSION,
        "event_time_unix": time.time(),
        **payload,
    }
    return append_jsonl(path, event)


def tb_step(step: int) -> int:
    return max(0, int(step))


def tb_add_scalar(writer: SummaryWriter | None, tag: str, value: float | int | None, step: int) -> None:
    if writer is None or value is None:
        return
    writer.add_scalar(tag, float(value), tb_step(step))


def tb_log_run_start(
    writer: SummaryWriter | None,
    *,
    run_id: str,
    mode: str,
    cfg_hash: str,
    cfg: TrainConfig,
    output_dir: Path,
) -> None:
    if writer is None:
        return
    writer.add_text("run/id", run_id, 0)
    writer.add_text("run/mode", mode, 0)
    writer.add_text("run/config_hash", cfg_hash, 0)
    writer.add_text("run/output_dir", str(output_dir), 0)
    writer.add_text("run/config_json", json.dumps(config_to_dict(cfg), indent=2, sort_keys=True), 0)
    writer.add_scalar("model/effective_depth", cfg.model.effective_depth, 0)
    writer.add_scalar("budget/max_wallclock_seconds", cfg.max_wallclock_seconds, 0)


def tb_log_train_event(
    writer: SummaryWriter | None,
    *,
    step: int,
    train_loss: float | None,
    step_seconds: float | None,
    elapsed_training_seconds: float,
    total_tokens: int,
    total_tokens_M: float,
    lr_scale_value: float,
    lr_now: Mapping[str, float | None],
) -> None:
    if writer is None:
        return
    tb_add_scalar(writer, "loss/train", train_loss, step)
    tb_add_scalar(writer, "perf/step_seconds", step_seconds, step)
    tb_add_scalar(writer, "perf/training_seconds", elapsed_training_seconds, step)
    tb_add_scalar(writer, "perf/total_tokens", total_tokens, step)
    tb_add_scalar(writer, "perf/total_tokens_M", total_tokens_M, step)
    tb_add_scalar(writer, "perf/tokens_per_second", total_tokens / max(elapsed_training_seconds, 1e-9), step)
    tb_add_scalar(writer, "lr/scale", lr_scale_value, step)
    for name, value in lr_now.items():
        tb_add_scalar(writer, f"lr/{name.removesuffix('_lr')}", value, step)


def tb_log_val_event(writer: SummaryWriter | None, *, phase: str, step: int, stats: EvalStats) -> None:
    if writer is None:
        return
    tb_add_scalar(writer, "loss/val", stats.val_loss, step)
    tb_add_scalar(writer, f"loss/val_{phase}", stats.val_loss, step)
    if stats.val_bpb is not None:
        tb_add_scalar(writer, "quality/val_bpb", stats.val_bpb, step)
        tb_add_scalar(writer, f"quality/val_bpb_{phase}", stats.val_bpb, step)
    tb_add_scalar(writer, "perf/eval_seconds", stats.eval_seconds, step)
    tb_add_scalar(writer, f"perf/eval_seconds_{phase}", stats.eval_seconds, step)


def tb_log_summary(
    writer: SummaryWriter | None,
    *,
    step: int,
    summary: RunSummary,
) -> None:
    if writer is None:
        return
    tb_add_scalar(writer, "summary/training_seconds", summary.training_seconds, step)
    tb_add_scalar(writer, "summary/total_seconds", summary.total_seconds, step)
    tb_add_scalar(writer, "summary/peak_vram_mb", summary.peak_vram_mb, step)
    tb_add_scalar(writer, "summary/total_tokens_M", summary.total_tokens_M, step)
    tb_add_scalar(writer, "summary/num_params_M", summary.num_params_M, step)
    tb_add_scalar(writer, "summary/depth", summary.effective_depth, step)
    tb_add_scalar(writer, "summary/val_bpb", summary.val_bpb, step)
    if summary.export is not None:
        tb_add_scalar(writer, "artifact/bytes", summary.export.artifact_bytes, step)
        tb_add_scalar(writer, "artifact/compressed_model_bytes", summary.export.compressed_model_bytes, step)
        tb_add_scalar(writer, "artifact/code_bytes", summary.export.code_bytes, step)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def tsv_header() -> str:
    return "\t".join(
        [
            "run_id",
            "config_hash",
            "status",
            "mode",
            "val_bpb",
            "training_seconds",
            "total_seconds",
            "peak_vram_mb",
            "mfu_percent",
            "total_tokens_M",
            "num_steps",
            "num_params_M",
            "depth",
            "artifact_bytes",
            "output_dir",
            "results_path",
        ]
    )


def append_results_tsv(path: str | Path, results: Mapping[str, Any]) -> Path:
    path = resolve_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        atomic_write_text(path, tsv_header() + "\n")
    row = "\t".join(
        [
            str(results["run_id"]),
            str(results["config_hash"]),
            str(results["status"]),
            str(results["mode"]),
            "null" if results["val_bpb"] is None else f"{float(results['val_bpb']):.9f}",
            f"{float(results['training_seconds']):.6f}",
            f"{float(results['total_seconds']):.6f}",
            f"{float(results['peak_vram_mb']):.3f}",
            f"{float(results['mfu_percent']):.3f}",
            f"{float(results['total_tokens_M']):.6f}",
            str(int(results["num_steps"])),
            f"{float(results['num_params_M']):.6f}",
            str(int(results["depth"])),
            str(int(results["artifact_bytes"])),
            str(results["output_dir"]),
            str(results["results_path"]),
        ]
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row + "\n")
    return path


def validate_results_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {
        "schema_version": str,
        "status": str,
        "mode": str,
        "run_id": str,
        "config_hash": str,
        "output_dir": str,
        "results_path": str,
        "training_seconds": (int, float),
        "total_seconds": (int, float),
        "peak_vram_mb": (int, float),
        "mfu_percent": (int, float),
        "total_tokens": int,
        "total_tokens_M": (int, float),
        "num_steps": int,
        "num_params": int,
        "num_params_M": (int, float),
        "depth": int,
        "artifact_bytes": int,
        "checkpoint_path": (str, type(None)),
        "val_loss": (int, float, type(None)),
        "val_bpb": (int, float, type(None)),
    }
    for key, expected in required.items():
        if key not in payload:
            raise ResultsSchemaError(f"results.json missing required key: {key}")
        if not isinstance(payload[key], expected):
            raise ResultsSchemaError(f"results.json key {key!r} has invalid type: {type(payload[key]).__name__}")
    if payload["schema_version"] != RESULTS_SCHEMA_VERSION:
        raise ResultsSchemaError(
            f"results.json schema_version mismatch: expected {RESULTS_SCHEMA_VERSION}, got {payload['schema_version']}"
        )
    if payload["status"] not in {"success", "failed"}:
        raise ResultsSchemaError(f"unsupported results status: {payload['status']}")
    if "tensorboard_log_dir" in payload and not isinstance(payload["tensorboard_log_dir"], (str, type(None))):
        raise ResultsSchemaError(f"results.json key 'tensorboard_log_dir' has invalid type: {type(payload['tensorboard_log_dir']).__name__}")
    return payload


def load_and_validate_results(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_results_payload(payload)


def metric_str(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


def emit_metric_lines(results: Mapping[str, Any]) -> None:
    print(f"val_bpb: {metric_str(results['val_bpb'])}")
    print(f"training_seconds: {metric_str(results['training_seconds'])}")
    print(f"total_seconds: {metric_str(results['total_seconds'])}")
    print(f"peak_vram_mb: {metric_str(results['peak_vram_mb'])}")
    print(f"mfu_percent: {metric_str(results['mfu_percent'])}")
    print(f"total_tokens_M: {metric_str(results['total_tokens_M'])}")
    print(f"num_steps: {metric_str(results['num_steps'])}")
    print(f"num_params_M: {metric_str(results['num_params_M'])}")
    print(f"depth: {metric_str(results['depth'])}")
    print(f"artifact_bytes: {metric_str(results['artifact_bytes'])}")


# ============================================================================
# DISTRIBUTED / DEVICE UTILS
# ============================================================================


def setup_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # pragma: no cover - older torch
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cuda"):
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
        if torch.cuda.is_available() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            print(
                "[warn] deterministic=True but CUBLAS_WORKSPACE_CONFIG is not set to :4096:8 or :16:8; "
                "GPU reproducibility may still drift",
                file=sys.stderr,
            )


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def init_distributed() -> DistInfo:
    rank_env = os.environ.get("RANK")
    world_env = os.environ.get("WORLD_SIZE")
    local_env = os.environ.get("LOCAL_RANK")
    if rank_env is None or world_env is None or local_env is None:
        return DistInfo(enabled=False, rank=0, world_size=1, local_rank=0)

    rank = int(rank_env)
    world_size = int(world_env)
    local_rank = int(local_env)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return DistInfo(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def cuda_device_index(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    if device.index is not None:
        return int(device.index)
    try:
        return int(torch.cuda.current_device())
    except Exception:
        return 0


def try_set_cuda_device(device: torch.device) -> None:
    index = cuda_device_index(device)
    if index is None:
        return
    try:
        torch.cuda.set_device(index)
    except Exception:
        pass


def get_device(dist_info: DistInfo) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda", dist_info.local_rank if dist_info.enabled else 0)
        try_set_cuda_device(device)
        return device
    return torch.device("cpu")


def barrier_if_needed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


@contextlib.contextmanager
def maybe_autocast(device: torch.device, enabled: bool = True) -> Iterator[None]:
    if enabled and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def peak_vram_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    index = cuda_device_index(device)
    if index is None:
        return 0.0
    try:
        return float(torch.cuda.max_memory_allocated(index) / (1024.0 * 1024.0))
    except Exception:
        return 0.0


def reset_peak_vram_stats(device: torch.device) -> None:
    index = cuda_device_index(device)
    if index is None:
        return
    try:
        try_set_cuda_device(device)
        torch.cuda.reset_peak_memory_stats(index)
    except Exception as exc:
        print(f"[warn] unable to reset CUDA peak memory stats: {exc}", file=sys.stderr)


# ============================================================================
# DATA SHARDS
# ============================================================================


SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_HEADER_INTS = 256


def write_data_shard(path: str | Path, tokens: np.ndarray | Tensor | list[int]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(tokens, Tensor):
        toks = tokens.detach().cpu().numpy().astype(np.uint16, copy=False)
    else:
        toks = np.asarray(tokens, dtype=np.uint16)
    header = np.zeros((SHARD_HEADER_INTS,), dtype="<i4")
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = toks.size
    with path.open("wb") as handle:
        header.tofile(handle)
        toks.astype("<u2", copy=False).tofile(handle)
    return path


def load_data_shard(path: str | Path) -> Tensor:
    path = Path(path)
    header = np.fromfile(path, dtype="<i4", count=SHARD_HEADER_INTS)
    if header.size != SHARD_HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    header_bytes = SHARD_HEADER_INTS * np.dtype("<i4").itemsize
    expected_size = header_bytes + num_tokens * np.dtype("<u2").itemsize
    if path.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {path}: expected {expected_size} bytes")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return torch.from_numpy(tokens.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            available = self.tokens.numel() - self.pos
            if available <= 0:
                self._advance_file()
                continue
            k = min(remaining, available)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

    def state_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "files": [str(path) for path in self.files],
            "file_idx": self.file_idx,
            "pos": self.pos,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        files = [Path(p) for p in state["files"]]
        if [str(p) for p in files] != [str(p) for p in self.files]:
            raise ConfigError("train shard file list changed across resume")
        self.file_idx = int(state["file_idx"])
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = int(state["pos"])


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        if local_tokens % seq_len != 0:
            raise ValueError(f"local_tokens={local_tokens} must be divisible by seq_len={seq_len}")
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device), y.to(self.device)

    def state_dict(self) -> dict[str, Any]:
        return {"stream": self.stream.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.stream.load_state_dict(state["stream"])


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(path) for path in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def make_smoke_shards(
    output_dir: str | Path,
    *,
    vocab_size: int = 32,
    train_tokens: int = 512,
    val_tokens: int = 128,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    pattern = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16)
    train = np.tile(pattern, train_tokens // len(pattern) + 1)[:train_tokens] % vocab_size
    val = np.tile(pattern, val_tokens // len(pattern) + 1)[:val_tokens] % vocab_size
    train_path = write_data_shard(output_dir / "train_000.bin", train)
    val_path = write_data_shard(output_dir / "val_000.bin", val)
    return train_path, val_path


# ============================================================================
# TOKENIZER-AGNOSTIC BPB SUPPORT
# ============================================================================


class PieceTokenizerLike(Protocol):
    def vocab_size(self) -> int: ...
    def is_control(self, token_id: int) -> bool: ...
    def is_unknown(self, token_id: int) -> bool: ...
    def is_unused(self, token_id: int) -> bool: ...
    def is_byte(self, token_id: int) -> bool: ...
    def id_to_piece(self, token_id: int) -> str: ...


def load_sentencepiece_model(path: str) -> PieceTokenizerLike:
    try:
        import sentencepiece as spm  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised in user environments
        raise ImportError(
            "sentencepiece is required to load a .model tokenizer. Install `sentencepiece` or pass tokenizer_path=None."
        ) from exc
    return spm.SentencePieceProcessor(model_file=path)


def build_piece_byte_luts(
    tokenizer: PieceTokenizerLike,
    vocab_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    tok_vocab = int(tokenizer.vocab_size())
    table_size = max(tok_vocab, vocab_size)
    base_bytes = np.zeros((table_size,), dtype=np.int16)
    has_leading_space = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(tok_vocab):
        if tokenizer.is_control(token_id) or tokenizer.is_unknown(token_id) or tokenizer.is_unused(token_id):
            continue
        is_boundary_token[token_id] = False
        if tokenizer.is_byte(token_id):
            base_bytes[token_id] = 1
            continue
        piece = tokenizer.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space[token_id] = True
            piece = piece[1:]
        base_bytes[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token, dtype=torch.bool, device=device),
    )


def load_validation_resources(
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    val_tokens = load_validation_tokens(cfg.val_pattern, cfg.model.seq_len) if cfg.val_pattern else None
    base_bytes_lut = has_leading_space_lut = is_boundary_token_lut = None
    if cfg.tokenizer_path:
        tokenizer = load_sentencepiece_model(cfg.tokenizer_path)
        if int(tokenizer.vocab_size()) != cfg.model.vocab_size:
            raise ValueError(
                f"vocab size mismatch: cfg has {cfg.model.vocab_size}, tokenizer has {int(tokenizer.vocab_size())}"
            )
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_piece_byte_luts(
            tokenizer,
            cfg.model.vocab_size,
            device,
        )
    return val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


# ============================================================================
# MODEL UTILS
# ============================================================================


def has_adapter(cfg: ModelConfig, target: str) -> bool:
    return cfg.adapter_rank > 0 and target in set(cfg.adapter_targets)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10_000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        return cos, sin


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    cos_full = torch.stack((cos, cos), dim=-1).flatten(-2)
    sin_full = torch.stack((sin, sin), dim=-1).flatten(-2)
    cos_full = cos_full.unsqueeze(0).unsqueeze(0)
    sin_full = sin_full.unsqueeze(0).unsqueeze(0)
    return (x * cos_full) + (rotate_half(x) * sin_full)


def per_row_fake_quant_ste(weight: Tensor, eps: float = 1e-8) -> Tensor:
    if weight.ndim != 2:
        scale = weight.detach().abs().amax().clamp_min(eps) / 127.0
        q = torch.clamp(torch.round(weight / scale), -127, 127)
        dq = q * scale
        return weight + (dq - weight).detach()
    scale = weight.detach().abs().amax(dim=1, keepdim=True).clamp_min(eps) / 127.0
    q = torch.clamp(torch.round(weight / scale), -127, 127)
    dq = q * scale
    return weight + (dq - weight).detach()


class FakeQuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        fake_quant_during_train: bool = True,
        fake_quant_start_step: int = 0,
        num_adapter_slots: int = 0,
        adapter_rank: int = 0,
        adapter_alpha: float = 1.0,
        zero_init: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fake_quant_during_train = fake_quant_during_train
        self.fake_quant_start_step = fake_quant_start_step
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long), persistent=False)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.has_adapter = num_adapter_slots > 0 and adapter_rank > 0
        self.num_adapter_slots = num_adapter_slots
        self.adapter_rank = adapter_rank
        self.adapter_alpha = adapter_alpha
        if self.has_adapter:
            self.adapter_A = nn.Parameter(torch.empty(num_adapter_slots, adapter_rank, in_features))
            self.adapter_B = nn.Parameter(torch.empty(num_adapter_slots, out_features, adapter_rank))
        else:
            self.register_parameter("adapter_A", None)
            self.register_parameter("adapter_B", None)
        self.zero_init = zero_init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.zero_init:
            nn.init.zeros_(self.weight)
        if self.bias is not None:
            bound = 1.0 / math.sqrt(max(1, self.in_features))
            nn.init.uniform_(self.bias, -bound, bound)
        if self.has_adapter:
            nn.init.normal_(self.adapter_A, mean=0.0, std=0.02)
            nn.init.zeros_(self.adapter_B)

    def set_global_step(self, step: int) -> None:
        self.global_step.fill_(int(step))

    def _effective_weight(self) -> Tensor:
        if self.training and self.fake_quant_during_train and int(self.global_step.item()) >= self.fake_quant_start_step:
            return per_row_fake_quant_ste(self.weight)
        return self.weight

    def forward(self, x: Tensor, slot: int | None = None) -> Tensor:
        y = F.linear(
            x,
            self._effective_weight().to(dtype=x.dtype),
            None if self.bias is None else self.bias.to(dtype=x.dtype),
        )
        if self.has_adapter and slot is not None:
            if slot < 0 or slot >= self.num_adapter_slots:
                raise IndexError(f"adapter slot {slot} out of range [0, {self.num_adapter_slots})")
            a = self.adapter_A[slot].to(dtype=x.dtype)
            b = self.adapter_B[slot].to(dtype=x.dtype)
            delta = F.linear(F.linear(x, a), b)
            y = y + delta * (self.adapter_alpha / self.adapter_rank)
        return y


class LowRankQProjection(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        fake_quant_during_train: bool,
        fake_quant_start_step: int,
        num_adapter_slots: int = 0,
        adapter_rank: int = 0,
        adapter_alpha: float = 1.0,
    ):
        super().__init__()
        self.rank = rank
        self.down_proj = FakeQuantLinear(
            in_features,
            rank,
            bias=False,
            fake_quant_during_train=fake_quant_during_train,
            fake_quant_start_step=fake_quant_start_step,
        )
        self.up_proj = FakeQuantLinear(
            rank,
            out_features,
            bias=False,
            fake_quant_during_train=fake_quant_during_train,
            fake_quant_start_step=fake_quant_start_step,
            num_adapter_slots=num_adapter_slots,
            adapter_rank=adapter_rank,
            adapter_alpha=adapter_alpha,
        )

    def set_global_step(self, step: int) -> None:
        self.down_proj.set_global_step(step)
        self.up_proj.set_global_step(step)

    def forward(self, x: Tensor, slot: int | None = None) -> Tensor:
        return self.up_proj(self.down_proj(x), slot=slot)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, num_adapter_slots: int = 0, q_low_rank: int | None = None):
        super().__init__()
        if cfg.d_model % cfg.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if cfg.num_heads % cfg.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.d_model = cfg.d_model
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.d_model // cfg.num_heads
        self.rope = RotaryEmbedding(self.head_dim, base=cfg.rope_base)
        resolved_q_low_rank = cfg.q_low_rank if q_low_rank is None else q_low_rank
        attn_fake_quant_during_train = (
            cfg.fake_quant_during_train if cfg.attn_fake_quant_during_train is None else cfg.attn_fake_quant_during_train
        )
        if resolved_q_low_rank > 0:
            self.q_proj = LowRankQProjection(
                cfg.d_model,
                cfg.d_model,
                rank=resolved_q_low_rank,
                fake_quant_during_train=attn_fake_quant_during_train,
                fake_quant_start_step=cfg.fake_quant_start_step,
                num_adapter_slots=num_adapter_slots if has_adapter(cfg, "q") else 0,
                adapter_rank=cfg.adapter_rank,
                adapter_alpha=cfg.adapter_alpha,
            )
        else:
            self.q_proj = FakeQuantLinear(
                cfg.d_model,
                cfg.d_model,
                bias=False,
                fake_quant_during_train=attn_fake_quant_during_train,
                fake_quant_start_step=cfg.fake_quant_start_step,
                num_adapter_slots=num_adapter_slots if has_adapter(cfg, "q") else 0,
                adapter_rank=cfg.adapter_rank,
                adapter_alpha=cfg.adapter_alpha,
            )
        self.k_proj = FakeQuantLinear(
            cfg.d_model,
            cfg.num_kv_heads * self.head_dim,
            bias=False,
            fake_quant_during_train=attn_fake_quant_during_train,
            fake_quant_start_step=cfg.fake_quant_start_step,
            num_adapter_slots=num_adapter_slots if has_adapter(cfg, "k") else 0,
            adapter_rank=cfg.adapter_rank,
            adapter_alpha=cfg.adapter_alpha,
        )
        self.v_proj = FakeQuantLinear(
            cfg.d_model,
            cfg.num_kv_heads * self.head_dim,
            bias=False,
            fake_quant_during_train=attn_fake_quant_during_train,
            fake_quant_start_step=cfg.fake_quant_start_step,
            num_adapter_slots=num_adapter_slots if has_adapter(cfg, "v") else 0,
            adapter_rank=cfg.adapter_rank,
            adapter_alpha=cfg.adapter_alpha,
        )
        self.out_proj = FakeQuantLinear(
            cfg.d_model,
            cfg.d_model,
            bias=False,
            fake_quant_during_train=attn_fake_quant_during_train,
            fake_quant_start_step=cfg.fake_quant_start_step,
            num_adapter_slots=num_adapter_slots if has_adapter(cfg, "attn_out") else 0,
            adapter_rank=cfg.adapter_rank,
            adapter_alpha=cfg.adapter_alpha,
            zero_init=True,
        )
        self.q_gain = nn.Parameter(torch.full((cfg.num_heads,), float(cfg.qk_gain_init)))
        self.dropout = nn.Dropout(cfg.resid_dropout)

    def set_global_step(self, step: int) -> None:
        self.q_proj.set_global_step(step)
        self.k_proj.set_global_step(step)
        self.v_proj.set_global_step(step)
        self.out_proj.set_global_step(step)

    def forward(self, x: Tensor, slot: int | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.q_proj(x, slot=slot).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x, slot=slot).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x, slot=slot).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        cos, sin = self.rope.get_cos_sin(seqlen, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype).view(1, -1, 1, 1)

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.dropout(self.out_proj(attn, slot=slot))


class ReLU2MLP(nn.Module):
    def __init__(self, cfg: ModelConfig, num_adapter_slots: int = 0, hidden_bonus: int = 0):
        super().__init__()
        hidden = cfg.d_model * cfg.mlp_mult + hidden_bonus
        self.fc = FakeQuantLinear(
            cfg.d_model,
            hidden,
            bias=False,
            fake_quant_during_train=cfg.fake_quant_during_train,
            fake_quant_start_step=cfg.fake_quant_start_step,
            num_adapter_slots=num_adapter_slots if has_adapter(cfg, "mlp_in") else 0,
            adapter_rank=cfg.adapter_rank,
            adapter_alpha=cfg.adapter_alpha,
        )
        self.proj = FakeQuantLinear(
            hidden,
            cfg.d_model,
            bias=False,
            fake_quant_during_train=cfg.fake_quant_during_train,
            fake_quant_start_step=cfg.fake_quant_start_step,
            num_adapter_slots=num_adapter_slots if has_adapter(cfg, "mlp_out") else 0,
            adapter_rank=cfg.adapter_rank,
            adapter_alpha=cfg.adapter_alpha,
            zero_init=True,
        )
        self.dropout = nn.Dropout(cfg.resid_dropout)

    def set_global_step(self, step: int) -> None:
        self.fc.set_global_step(step)
        self.proj.set_global_step(step)

    def forward(self, x: Tensor, slot: int | None = None) -> Tensor:
        x = torch.relu(self.fc(x, slot=slot))
        x = x.square()
        return self.dropout(self.proj(x, slot=slot))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        num_adapter_slots: int = 0,
        mlp_hidden_bonus: int = 0,
        q_low_rank: int | None = None,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(cfg, num_adapter_slots=num_adapter_slots, q_low_rank=q_low_rank)
        self.mlp = ReLU2MLP(cfg, num_adapter_slots=num_adapter_slots, hidden_bonus=mlp_hidden_bonus)
        scale_shape = (num_adapter_slots, cfg.d_model) if num_adapter_slots > 0 else (cfg.d_model,)
        self.attn_scale = nn.Parameter(torch.ones(scale_shape))
        self.mlp_scale = nn.Parameter(torch.ones(scale_shape))

    def set_global_step(self, step: int) -> None:
        self.attn.set_global_step(step)
        self.mlp.set_global_step(step)

    def residual_scale(self, scale: nn.Parameter, x: Tensor, slot: int | None) -> Tensor:
        if scale.ndim == 2:
            if slot is None:
                raise RuntimeError("per-loop residual scales require a loop slot")
            return scale[slot].to(dtype=x.dtype)
        return scale.to(dtype=x.dtype)

    def forward(self, x: Tensor, slot: int | None = None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), slot=slot) * self.residual_scale(self.attn_scale, x, slot)
        x = x + self.mlp(self.mlp_norm(x), slot=slot) * self.residual_scale(self.mlp_scale, x, slot)
        return x


class RecurrentGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        final_tail_q_low_rank = cfg.q_low_rank if cfg.final_tail_q_low_rank is None else cfg.final_tail_q_low_rank
        non_recurrent_hidden_bonus = (
            cfg.non_recurrent_mlp_hidden_bonus
            if cfg.non_recurrent_mlp_hidden_bonus is not None
            else cfg.d_model // 2
        )
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_norm = RMSNorm(cfg.d_model)
        self.stem = nn.ModuleList(
            [TransformerBlock(cfg, num_adapter_slots=0, mlp_hidden_bonus=non_recurrent_hidden_bonus) for _ in range(cfg.stem_layers)]
        )
        self.shared = nn.ModuleList(
            [
                TransformerBlock(
                    cfg,
                    num_adapter_slots=cfg.recurrence_loops,
                    mlp_hidden_bonus=cfg.shared_mlp_hidden_bonus,
                )
                for _ in range(cfg.shared_layers)
            ]
        )
        self.tail = nn.ModuleList(
            [
                TransformerBlock(
                    cfg,
                    num_adapter_slots=0,
                    mlp_hidden_bonus=non_recurrent_hidden_bonus,
                    q_low_rank=final_tail_q_low_rank if tail_idx == cfg.tail_layers - 1 else None,
                )
                for tail_idx in range(cfg.tail_layers)
            ]
        )
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = None if cfg.tie_embeddings else FakeQuantLinear(
            cfg.d_model,
            cfg.vocab_size,
            bias=False,
            fake_quant_during_train=cfg.fake_quant_during_train,
            fake_quant_start_step=cfg.fake_quant_start_step,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.cfg.emb_init_std)
        if self.lm_head is not None:
            nn.init.zeros_(self.lm_head.weight)

    def set_global_step(self, step: int) -> None:
        if self.lm_head is not None:
            self.lm_head.set_global_step(step)
        for block in self.stem:
            block.set_global_step(step)
        for block in self.shared:
            block.set_global_step(step)
        for block in self.tail:
            block.set_global_step(step)

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        x = self.emb_norm(self.tok_emb(input_ids))
        for block in self.stem:
            x = block(x, slot=None)
        for loop_idx in range(self.cfg.recurrence_loops):
            for block in self.shared:
                x = block(x, slot=loop_idx)
        for block in self.tail:
            x = block(x, slot=None)
        x = self.final_norm(x)
        if self.cfg.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head missing while tie_embeddings=False")
            logits = self.lm_head(x)
        logits = self.cfg.logit_softcap * torch.tanh(logits / self.cfg.logit_softcap)
        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="mean")
        return logits, loss


# ============================================================================
# OPTIMIZERS
# ============================================================================


def zeropower_via_newtonschulz5(g: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    x = x / (x.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.t()
    for _ in range(steps):
        a_mat = x @ x.t()
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    return x.t() if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        defaults = dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.ndim != 2:
                    raise ValueError("Muon is intended for matrix-shaped parameters only")
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(update, steps=backend_steps)
                update *= max(1.0, update.size(0) / max(1, update.size(1))) ** 0.5
                param.add_(update.to(dtype=param.dtype), alpha=-lr)
        return loss


def split_params_for_optim(model: RecurrentGPT) -> dict[str, list[nn.Parameter]]:
    buckets: dict[str, list[nn.Parameter]] = {"embed": [], "head": [], "matrix": [], "scalar": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == "tok_emb.weight":
            buckets["embed"].append(param)
        elif name.startswith("lm_head"):
            buckets["head"].append(param)
        elif param.ndim == 2:
            buckets["matrix"].append(param)
        else:
            buckets["scalar"].append(param)
    return buckets


@dataclass
class OptimBundle:
    embed: torch.optim.Optimizer
    head: torch.optim.Optimizer | None
    matrix: torch.optim.Optimizer
    scalar: torch.optim.Optimizer

    def all(self) -> list[torch.optim.Optimizer]:
        return [self.embed] + ([self.head] if self.head is not None else []) + [self.matrix, self.scalar]

    def state_dict(self) -> dict[str, Any]:
        return {
            "embed": self.embed.state_dict(),
            "head": None if self.head is None else self.head.state_dict(),
            "matrix": self.matrix.state_dict(),
            "scalar": self.scalar.state_dict(),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self.embed.load_state_dict(payload["embed"])
        if self.head is not None and payload["head"] is not None:
            self.head.load_state_dict(payload["head"])
        self.matrix.load_state_dict(payload["matrix"])
        self.scalar.load_state_dict(payload["scalar"])


def build_optimizers(model: RecurrentGPT, cfg: TrainConfig) -> OptimBundle:
    buckets = split_params_for_optim(model)
    optim_cfg = cfg.optim
    embed_opt = torch.optim.AdamW(
        [{"params": buckets["embed"], "lr": optim_cfg.embed_lr, "base_lr": optim_cfg.embed_lr}],
        betas=(optim_cfg.beta1, optim_cfg.beta2),
        eps=optim_cfg.adam_eps,
        weight_decay=optim_cfg.weight_decay,
    )
    head_opt = None
    if buckets["head"]:
        head_opt = torch.optim.AdamW(
            [{"params": buckets["head"], "lr": optim_cfg.head_lr, "base_lr": optim_cfg.head_lr}],
            betas=(optim_cfg.beta1, optim_cfg.beta2),
            eps=optim_cfg.adam_eps,
            weight_decay=optim_cfg.weight_decay,
        )
    matrix_opt = Muon(
        buckets["matrix"],
        lr=optim_cfg.matrix_lr,
        momentum=optim_cfg.muon_momentum,
        backend_steps=optim_cfg.muon_backend_steps,
    )
    for group in matrix_opt.param_groups:
        group["base_lr"] = optim_cfg.matrix_lr
    scalar_opt = torch.optim.AdamW(
        [{"params": buckets["scalar"], "lr": optim_cfg.scalar_lr, "base_lr": optim_cfg.scalar_lr}],
        betas=(optim_cfg.beta1, optim_cfg.beta2),
        eps=optim_cfg.adam_eps,
        weight_decay=optim_cfg.weight_decay,
    )
    return OptimBundle(embed=embed_opt, head=head_opt, matrix=matrix_opt, scalar=scalar_opt)


def lr_scale(step: int, total_steps: int, cfg: OptimConfig) -> float:
    if total_steps <= 1:
        return 1.0
    if step < cfg.warmup_steps:
        return max(1e-8, (step + 1) / max(1, cfg.warmup_steps))
    warmdown_start = max(cfg.warmup_steps, total_steps - cfg.warmdown_steps)
    if step >= warmdown_start:
        t = (step - warmdown_start) / max(1, total_steps - warmdown_start)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return cfg.min_lr_scale + (1.0 - cfg.min_lr_scale) * cosine
    return 1.0


def estimate_effective_total_steps(
    step: int,
    *,
    start_step: int,
    session_elapsed: float,
    target_iterations: int,
    max_wallclock_seconds: float,
    warmup_steps: int,
) -> int:
    if target_iterations <= 0 or max_wallclock_seconds <= 0.0 or session_elapsed <= 0.0:
        return target_iterations
    completed_steps = step - start_step
    # Wait for more steady-state training before estimating the 5-minute budget;
    # early startup overhead can otherwise bias the cosine schedule to decay too soon.
    calibration_steps = max(80, warmup_steps * 4)
    if completed_steps < calibration_steps:
        return target_iterations
    estimated_session_steps = max(
        completed_steps,
        int(round(completed_steps * max_wallclock_seconds / session_elapsed)),
    )
    return min(target_iterations, start_step + estimated_session_steps)


def set_optimizer_lrs(bundle: OptimBundle, step: int, total_steps: int, optim_cfg: OptimConfig) -> None:
    scale = lr_scale(step, total_steps, optim_cfg)
    for opt in bundle.all():
        for group in opt.param_groups:
            group["lr"] = float(group["base_lr"]) * scale


# ============================================================================
# QUANTIZATION / EXPORT
# ============================================================================


def tensor_nbytes(tensor: Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _store_passthrough_float(name: str, tensor: Tensor, cfg: QuantConfig, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if tensor.dtype != cfg.keep_float_store_dtype:
        passthrough_orig_dtypes[name] = str(tensor.dtype).removeprefix("torch.")
        return tensor.to(dtype=cfg.keep_float_store_dtype).contiguous()
    return tensor.contiguous()


def _keep_float_tensor(name: str, tensor: Tensor, cfg: QuantConfig, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in cfg.keep_float_name_patterns):
        return _store_passthrough_float(name, tensor, cfg, passthrough_orig_dtypes)
    if tensor.dtype in {torch.float32, torch.bfloat16}:
        return _store_passthrough_float(name, tensor, cfg, passthrough_orig_dtypes)
    return tensor


def quant_qmax_for_tensor(name: str, tensor: Tensor, cfg: QuantConfig) -> int:
    if cfg.low_bit_bits < 8 and tensor.ndim >= 2 and any(pattern in name for pattern in cfg.low_bit_name_patterns):
        return (1 << (cfg.low_bit_bits - 1)) - 1
    return 127


def quantize_float_tensor_export(name: str, tensor: Tensor, cfg: QuantConfig) -> tuple[Tensor, Tensor]:
    t32 = tensor.float()
    qmax = quant_qmax_for_tensor(name, t32, cfg)
    q_level = cfg.clip_percentile / 100.0
    if t32.ndim == 2:
        clip = torch.quantile(t32.abs(), q_level, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clip = clip.clamp_min(1.0 / qmax)
        clipped = torch.maximum(torch.minimum(t32, clip[:, None]), -clip[:, None])
        scale = (clip / qmax).clamp_min(1.0 / qmax)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -qmax, qmax).to(torch.int8).contiguous()
        return q, scale.to(dtype=cfg.scale_store_dtype).contiguous()
    clip_scalar = float(torch.quantile(t32.abs().flatten(), q_level).item()) if t32.numel() else 0.0
    clip_scalar = max(clip_scalar, 1.0 / qmax)
    scale = torch.tensor(clip_scalar / qmax, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_scalar, clip_scalar) / scale), -qmax, qmax).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor], cfg: QuantConfig) -> tuple[dict[str, Any], dict[str, int]]:
    quantized: dict[str, Tensor] = OrderedDict()
    scales: dict[str, Tensor] = OrderedDict()
    dtypes: dict[str, str] = OrderedDict()
    passthrough: dict[str, Tensor] = OrderedDict()
    qmeta: dict[str, dict[str, Any]] = OrderedDict()
    passthrough_orig_dtypes: dict[str, str] = OrderedDict()
    stats = {
        "param_count": 0,
        "num_tensors": 0,
        "num_float_tensors": 0,
        "num_nonfloat_tensors": 0,
        "baseline_tensor_bytes": 0,
        "quant_payload_bytes": 0,
    }
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["quant_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= cfg.keep_float_max_numel or any(pattern in name for pattern in cfg.keep_float_name_patterns):
            kept = _keep_float_tensor(name, t, cfg, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["quant_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, scale = quantize_float_tensor_export(name, t, cfg)
        quantized[name] = q
        scales[name] = scale
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        if scale.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        stats["quant_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(scale)
    payload: dict[str, Any] = {
        "__quant_format__": "rowwise_int8_lora_recurrent_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        payload["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        payload["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return payload, stats


def dequantize_state_dict_int8(payload: Mapping[str, Any]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = OrderedDict()
    qmeta = payload.get("qmeta", {})
    passthrough_orig_dtypes = payload.get("passthrough_orig_dtypes", {})
    for name, q in payload["quantized"].items():
        dtype = getattr(torch, payload["dtypes"][name])
        scale = payload["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            scale = scale.to(dtype=torch.float32)
            out[name] = (q.float() * scale.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(scale.item())).to(dtype=dtype).contiguous()
    for name, tensor in payload["passthrough"].items():
        out_tensor = tensor.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_tensor = out_tensor.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_tensor
    return out


def pack_quantized_payload(payload: Mapping[str, Any], cfg: QuantConfig) -> bytes:
    grouped: dict[str, Any] = {
        "__quant_format__": payload["__quant_format__"],
        "quantized": OrderedDict(payload["quantized"]),
        "scales": OrderedDict(payload["scales"]),
        "dtypes": OrderedDict(payload["dtypes"]),
        "passthrough": OrderedDict(payload["passthrough"]),
    }
    if "qmeta" in payload:
        grouped["qmeta"] = payload["qmeta"]
    if "passthrough_orig_dtypes" in payload:
        grouped["passthrough_orig_dtypes"] = payload["passthrough_orig_dtypes"]
    buffer = io.BytesIO()
    torch.save(grouped, buffer)
    return zlib.compress(buffer.getvalue(), level=cfg.zlib_level)


def unpack_quantized_payload(blob: bytes) -> dict[str, Any]:
    raw = zlib.decompress(blob)
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)


def resolve_counted_code_paths(cfg: TrainConfig) -> list[Path]:
    paths: list[Path] = []
    for item in cfg.counted_code_paths:
        path = resolve_repo_path(item)
        if not path.exists():
            raise ConfigError(f"counted code path does not exist: {path}")
        if not path.is_file():
            raise ConfigError(f"counted code path is not a file: {path}")
        paths.append(path)
    return paths


def export_quantized_artifact(
    cfg: TrainConfig,
    model: nn.Module,
    output_dir: Path,
    run_id: str,
    cfg_hash: str,
    *,
    val_stats: EvalStats | None = None,
) -> tuple[ExportStats, dict[str, Any]]:
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload, quant_stats = quantize_state_dict_int8(state_dict, cfg.quant)
    blob = pack_quantized_payload(payload, cfg.quant)

    artifact_dir = output_dir / cfg.artifact_bundle_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_blob_path = artifact_dir / "model_int8.zlib"
    atomic_write_bytes(model_blob_path, blob)

    code_dir = artifact_dir / "code"
    code_paths = resolve_counted_code_paths(cfg)
    file_entries: list[dict[str, Any]] = []
    code_bytes = 0
    for source_path in code_paths:
        try:
            rel_path = source_path.relative_to(repo_root())
        except ValueError:
            rel_path = Path(source_path.name)
        dest_path = code_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        file_bytes = dest_path.stat().st_size
        code_bytes += file_bytes
        file_entries.append(
            {
                "source_path": str(source_path),
                "artifact_relpath": str(dest_path.relative_to(artifact_dir)),
                "bytes": file_bytes,
                "sha256": sha256_file(dest_path),
            }
        )

    artifact_stats = ExportStats(
        artifact_dir=str(artifact_dir),
        manifest_path=str(artifact_dir / "manifest.json"),
        model_blob_path=str(model_blob_path),
        compressed_model_bytes=len(blob),
        code_bytes=code_bytes,
        artifact_bytes=code_bytes + len(blob),
        quant_payload_bytes=int(quant_stats["quant_payload_bytes"]),
    )

    manifest = {
        "schema_version": ARTIFACT_MANIFEST_VERSION,
        "run_id": run_id,
        "config_hash": cfg_hash,
        "model_config": _jsonable(cfg.model),
        "quant_config": _jsonable(cfg.quant),
        "counted_files": file_entries,
        "compressed_model": {
            "artifact_relpath": str(model_blob_path.relative_to(artifact_dir)),
            "bytes": len(blob),
            "sha256": sha256_bytes(blob),
        },
        "byte_counts": {
            "compressed_model_bytes": artifact_stats.compressed_model_bytes,
            "code_bytes": artifact_stats.code_bytes,
            "artifact_bytes": artifact_stats.artifact_bytes,
            "quant_payload_bytes": artifact_stats.quant_payload_bytes,
        },
        "quant_stats": quant_stats,
        "param_count": count_parameters(model),
        "effective_depth": cfg.model.effective_depth,
        "latest_val": None if val_stats is None else dataclasses.asdict(val_stats),
    }
    atomic_write_json(artifact_dir / "manifest.json", manifest)
    atomic_write_json(output_dir / "export_stats.json", {**manifest["byte_counts"], **quant_stats})
    return artifact_stats, manifest


def resolve_artifact_dir(path: str | Path) -> Path:
    artifact_path = Path(path)
    if artifact_path.is_dir():
        return artifact_path
    if artifact_path.name == "manifest.json":
        return artifact_path.parent
    if artifact_path.suffix == ".zlib":
        return artifact_path.parent
    raise ConfigError(f"Unable to resolve artifact bundle from path: {artifact_path}")


def load_artifact_bundle(path: str | Path) -> tuple[dict[str, Any], bytes, Path]:
    artifact_dir = resolve_artifact_dir(path)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise ConfigError(f"artifact manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_VERSION:
        raise ConfigError(f"unexpected artifact manifest schema: {manifest.get('schema_version')}")
    blob_rel = manifest["compressed_model"]["artifact_relpath"]
    blob_path = artifact_dir / blob_rel
    blob = blob_path.read_bytes()
    return manifest, blob, artifact_dir


def load_model_from_artifact(path: str | Path, device: torch.device) -> tuple[RecurrentGPT, dict[str, Any], Path]:
    manifest, blob, artifact_dir = load_artifact_bundle(path)
    model_payload = dict(manifest["model_config"])
    if "adapter_targets" in model_payload and isinstance(model_payload["adapter_targets"], list):
        model_payload["adapter_targets"] = tuple(model_payload["adapter_targets"])
    model = RecurrentGPT(ModelConfig(**model_payload)).to(device)
    payload = unpack_quantized_payload(blob)
    state_dict = dequantize_state_dict_int8(payload)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, manifest, artifact_dir


# ============================================================================
# EVALUATION
# ============================================================================


def eval_val(
    cfg: TrainConfig,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor | None = None,
    has_leading_space_lut: Tensor | None = None,
    is_boundary_token_lut: Tensor | None = None,
) -> EvalStats:
    start_time = time.time()
    local_batch_tokens = cfg.val_batch_tokens // max(1, world_size)
    if local_batch_tokens < cfg.model.seq_len:
        raise ValueError("val_batch_tokens must provide at least one sequence per rank")
    local_batch_seqs = local_batch_tokens // cfg.model.seq_len
    total_seqs = (val_tokens.numel() - 1) // cfg.model.seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * cfg.model.seq_len
            raw_end = batch_seq_end * cfg.model.seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, cfg.model.seq_len)
            y = local[1:].reshape(-1, cfg.model.seq_len)
            with maybe_autocast(device, enabled=True):
                _, batch_loss = model(x, y)
            if batch_loss is None:
                raise RuntimeError("model did not return a loss during evaluation")
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            if base_bytes_lut is not None and has_leading_space_lut is not None and is_boundary_token_lut is not None:
                prev_ids = x.reshape(-1)
                tgt_ids = y.reshape(-1)
                token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
                val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        if base_bytes_lut is not None:
            dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = float((val_loss_sum / val_token_count).item())
    val_bpb = None
    byte_count = None
    if base_bytes_lut is not None:
        bits_per_token = val_loss / math.log(2.0)
        tokens_per_byte = float((val_token_count / val_byte_count).item())
        val_bpb = bits_per_token * tokens_per_byte
        byte_count = int(round(float(val_byte_count.item())))
    if was_training:
        model.train()
    return EvalStats(
        val_loss=val_loss,
        val_bpb=val_bpb,
        token_count=int(round(float(val_token_count.item()))),
        byte_count=byte_count,
        eval_seconds=time.time() - start_time,
    )


# ============================================================================
# LINEAR WEIGHT AVERAGING
# ============================================================================


class LinearWeightAverager:
    def __init__(self):
        self.avg_state: dict[str, Tensor] | None = None
        self.count = 0

    def update(self, model: nn.Module) -> None:
        state = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
        if self.avg_state is None:
            self.avg_state = state
            self.count = 1
            return
        self.count += 1
        assert self.avg_state is not None
        alpha_old = (self.count - 1) / self.count
        alpha_new = 1.0 / self.count
        for key in self.avg_state:
            self.avg_state[key].mul_(alpha_old).add_(state[key], alpha=alpha_new)

    def load_into(self, model: nn.Module) -> None:
        if self.avg_state is None:
            return
        current = model.state_dict()
        merged = {key: self.avg_state[key].to(dtype=current[key].dtype) for key in current}
        model.load_state_dict(merged, strict=True)

    def state_dict(self) -> dict[str, Any]:
        return {"avg_state": self.avg_state, "count": self.count}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.avg_state = state["avg_state"]
        self.count = int(state["count"])


# ============================================================================
# CHECKPOINT / SUMMARY HELPERS
# ============================================================================


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def maybe_wrap_ddp(model: nn.Module, dist_info: DistInfo, device: torch.device) -> nn.Module:
    if not dist_info.enabled:
        return model
    if device.type == "cuda":
        return DDP(model, device_ids=[dist_info.local_rank], broadcast_buffers=False)
    return DDP(model)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def checkpoint_paths(output_dir: Path) -> tuple[Path, Path]:
    ckpt_dir = output_dir / "checkpoints"
    return ckpt_dir / "latest.pt", ckpt_dir / "final.pt"


def save_checkpoint(
    path: str | Path,
    cfg: TrainConfig,
    run_id: str,
    cfg_hash: str,
    raw_model: nn.Module,
    opt_bundle: OptimBundle,
    train_loader: DistributedTokenLoader,
    lawa: LinearWeightAverager | None,
    *,
    next_step: int,
    elapsed_training_seconds: float,
    total_tokens: int,
    last_train_loss: float | None,
    last_val: EvalStats | None,
) -> Path:
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "config_hash": cfg_hash,
        "resume_compat_hash": config_hash(cfg, for_resume=True),
        "cfg": config_to_dict(cfg),
        "next_step": next_step,
        "elapsed_training_seconds": elapsed_training_seconds,
        "total_tokens": total_tokens,
        "last_train_loss": last_train_loss,
        "last_val": None if last_val is None else dataclasses.asdict(last_val),
        "model_state": raw_model.state_dict(),
        "optim_state": opt_bundle.state_dict(),
        "train_loader_state": train_loader.state_dict(),
        "lawa_state": None if lawa is None else lawa.state_dict(),
        "rng_state": capture_rng_state(),
    }
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    return atomic_write_bytes(path, buffer.getvalue())


def load_checkpoint(
    path: str | Path,
    cfg: TrainConfig,
    raw_model: nn.Module,
    opt_bundle: OptimBundle,
    train_loader: DistributedTokenLoader,
    lawa: LinearWeightAverager | None,
) -> ResumeState:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ConfigError(f"unexpected checkpoint schema: {payload.get('schema_version')}")
    if payload.get("resume_compat_hash") != config_hash(cfg, for_resume=True):
        raise ConfigError("checkpoint config is incompatible with current run configuration")
    raw_model.load_state_dict(payload["model_state"], strict=True)
    opt_bundle.load_state_dict(payload["optim_state"])
    train_loader.load_state_dict(payload["train_loader_state"])
    if lawa is not None and payload["lawa_state"] is not None:
        lawa.load_state_dict(payload["lawa_state"])
    restore_rng_state(payload["rng_state"])
    last_val = payload.get("last_val")
    return ResumeState(
        run_id=str(payload["run_id"]),
        config_hash=str(payload["config_hash"]),
        next_step=int(payload["next_step"]),
        elapsed_training_seconds=float(payload["elapsed_training_seconds"]),
        total_tokens=int(payload["total_tokens"]),
        last_train_loss=payload["last_train_loss"],
        last_val=None if last_val is None else EvalStats(**last_val),
        checkpoint_path=str(path),
    )


def estimate_mfu_percent(_cfg: TrainConfig, _tokens_per_second: float, _num_params: int, _device: torch.device) -> float:
    # Reliable peak-FLOP data is not portable across user environments, so keep
    # this parseable and conservative rather than fabricating a noisy estimate.
    return 0.0


def summary_to_results(summary: RunSummary, cfg: TrainConfig, output_dir: Path, started_at: float, finished_at: float) -> dict[str, Any]:
    artifact_bytes = 0 if summary.export is None else summary.export.artifact_bytes
    payload = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "status": summary.status,
        "mode": summary.mode,
        "run_id": summary.run_id,
        "config_hash": summary.config_hash,
        "output_dir": str(output_dir),
        "results_path": "" if summary.results_path is None else summary.results_path,
        "config_path": str(output_dir / "config.json"),
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
        "training_seconds": summary.training_seconds,
        "total_seconds": summary.total_seconds,
        "peak_vram_mb": summary.peak_vram_mb,
        "mfu_percent": summary.mfu_percent,
        "total_tokens": summary.total_tokens,
        "total_tokens_M": summary.total_tokens_M,
        "num_steps": summary.num_steps,
        "num_params": summary.model_params,
        "num_params_M": summary.num_params_M,
        "depth": summary.effective_depth,
        "train_loss": summary.train_loss,
        "val_loss": summary.val_loss,
        "val_bpb": summary.val_bpb,
        "artifact_bytes": artifact_bytes,
        "artifact": None if summary.export is None else dataclasses.asdict(summary.export),
        "benchmark": None if summary.benchmark is None else dataclasses.asdict(summary.benchmark),
        "checkpoint_path": summary.checkpoint_path,
        "metrics_path": summary.metrics_path,
        "tensorboard_log_dir": summary.tensorboard_log_dir,
        "resume_from": cfg.resume_from,
    }
    return dict(validate_results_payload(payload))


def write_failure_outputs(
    cfg: TrainConfig,
    run_id: str,
    cfg_hash: str,
    started_at: float,
    exit_code: int,
    exc: BaseException,
) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    finished_at = time.time()
    metrics_path = resolve_metrics_jsonl_path(output_dir, cfg.metrics_jsonl_path)
    tensorboard_log_dir = resolve_tensorboard_log_dir(output_dir, cfg.tensorboard_log_dir)
    crash_payload = {
        "run_id": run_id,
        "config_hash": cfg_hash,
        "exit_code": exit_code,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
    }
    crash_path = atomic_write_json(output_dir / "crash.json", crash_payload)
    append_metrics_event(
        metrics_path,
        {
            "event": "crash",
            "mode": "eval" if cfg.evaluate_only else ("benchmark" if cfg.benchmark_only else "train"),
            "run_id": run_id,
            "config_hash": cfg_hash,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "exit_code": exit_code,
        },
    )
    failure_results = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "status": "failed",
        "mode": "eval" if cfg.evaluate_only else ("benchmark" if cfg.benchmark_only else "train"),
        "run_id": run_id,
        "config_hash": cfg_hash,
        "output_dir": str(output_dir),
        "results_path": str(output_dir / "results.json"),
        "config_path": str(output_dir / "config.json"),
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
        "training_seconds": 0.0,
        "total_seconds": finished_at - started_at,
        "peak_vram_mb": 0.0,
        "mfu_percent": 0.0,
        "total_tokens": 0,
        "total_tokens_M": 0.0,
        "num_steps": 0,
        "num_params": 0,
        "num_params_M": 0.0,
        "depth": cfg.model.effective_depth,
        "train_loss": None,
        "val_loss": None,
        "val_bpb": None,
        "artifact_bytes": 0,
        "artifact": None,
        "benchmark": None,
        "checkpoint_path": None,
        "metrics_path": str(metrics_path),
        "tensorboard_log_dir": str(tensorboard_log_dir),
        "resume_from": cfg.resume_from,
        "crash_path": str(crash_path),
        "exit_code": exit_code,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    atomic_write_json(output_dir / "results.json", failure_results)


# ============================================================================
# TRAIN / EVAL
# ============================================================================


def validate_config(cfg: TrainConfig) -> None:
    if cfg.model.vocab_size <= 0:
        raise ConfigError("vocab_size must be positive")
    if cfg.model.seq_len <= 0:
        raise ConfigError("seq_len must be positive")
    if cfg.model.non_recurrent_mlp_hidden_bonus is not None and cfg.model.non_recurrent_mlp_hidden_bonus < 0:
        raise ConfigError("non_recurrent_mlp_hidden_bonus must be >= 0 when set")
    if cfg.model.shared_mlp_hidden_bonus < 0:
        raise ConfigError("shared_mlp_hidden_bonus must be >= 0")
    if cfg.model.q_low_rank < 0:
        raise ConfigError("q_low_rank must be >= 0")
    if cfg.model.q_low_rank >= cfg.model.d_model:
        raise ConfigError("q_low_rank must be smaller than d_model")
    if cfg.model.final_tail_q_low_rank is not None and cfg.model.final_tail_q_low_rank < 0:
        raise ConfigError("final_tail_q_low_rank must be >= 0 when set")
    if cfg.model.final_tail_q_low_rank is not None and cfg.model.final_tail_q_low_rank >= cfg.model.d_model:
        raise ConfigError("final_tail_q_low_rank must be smaller than d_model when set")
    if cfg.grad_accum_steps <= 0:
        raise ConfigError("grad_accum_steps must be positive")
    if cfg.train_seq_len_min is not None and cfg.train_seq_len_min <= 0:
        raise ConfigError("train_seq_len_min must be positive when set")
    if cfg.train_seq_len_min is not None and cfg.train_seq_len_min > cfg.model.seq_len:
        raise ConfigError("train_seq_len_min must be <= seq_len")
    if cfg.train_seq_len_warmup_steps < 0:
        raise ConfigError("train_seq_len_warmup_steps must be >= 0")
    if cfg.iterations <= 0:
        raise ConfigError("iterations must be positive")
    if cfg.train_batch_tokens <= 0:
        raise ConfigError("train_batch_tokens must be positive")
    if cfg.train_batch_tokens % cfg.grad_accum_steps != 0:
        raise ConfigError("train_batch_tokens must be divisible by grad_accum_steps")
    if cfg.local_batch_tokens % cfg.model.seq_len != 0:
        raise ConfigError("train_batch_tokens / grad_accum_steps must be divisible by seq_len")
    if cfg.train_seq_len_min is not None and cfg.local_batch_tokens % cfg.train_seq_len_min != 0:
        raise ConfigError("train_batch_tokens / grad_accum_steps must be divisible by train_seq_len_min")
    if cfg.val_batch_tokens <= 0:
        raise ConfigError("val_batch_tokens must be positive")
    if cfg.val_every < 0:
        raise ConfigError("val_every must be >= 0")
    if cfg.quant.clip_percentile <= 0.0 or cfg.quant.clip_percentile > 100.0:
        raise ConfigError("quant.clip_percentile must be in (0, 100]")
    if cfg.quant.low_bit_bits < 2 or cfg.quant.low_bit_bits > 8:
        raise ConfigError("quant.low_bit_bits must be in [2, 8]")
    invalid_targets = sorted(set(cfg.model.adapter_targets) - set(ALLOWED_ADAPTER_TARGETS))
    if invalid_targets:
        raise ConfigError(f"invalid adapter_targets: {invalid_targets}")
    if not cfg.counted_code_paths:
        raise ConfigError("counted_code_paths must not be empty")
    if cfg.evaluate_only and not cfg.load_artifact_path:
        raise ConfigError("evaluate_only requires load_artifact_path")


def validate_runtime_config(cfg: TrainConfig, world_size: int) -> None:
    if cfg.train_batch_tokens % (world_size * cfg.grad_accum_steps) != 0:
        raise ConfigError("train_batch_tokens must be divisible by world_size * grad_accum_steps")
    local_tokens = cfg.train_batch_tokens // (world_size * cfg.grad_accum_steps)
    if local_tokens % cfg.model.seq_len != 0:
        raise ConfigError("per-rank train batch must be divisible by seq_len")
    if cfg.train_seq_len_min is not None and local_tokens % cfg.train_seq_len_min != 0:
        raise ConfigError("per-rank train batch must be divisible by train_seq_len_min")
    if cfg.val_batch_tokens % world_size != 0:
        raise ConfigError("val_batch_tokens must be divisible by world_size")
    if (cfg.val_batch_tokens // world_size) < cfg.model.seq_len:
        raise ConfigError("val_batch_tokens must provide at least one sequence per rank")


def benchmark_stats_from_run(summary: RunSummary, last_val: EvalStats | None) -> BenchmarkStats | None:
    if summary.num_steps <= 0:
        return None
    train_tps = summary.total_tokens / max(summary.training_seconds, 1e-9)
    eval_tps = 0.0
    eval_seconds = 0.0
    if last_val is not None and last_val.token_count > 0:
        eval_seconds = last_val.eval_seconds
        eval_tps = last_val.token_count / max(last_val.eval_seconds, 1e-9)
    return BenchmarkStats(
        train_tokens_per_second=train_tps,
        eval_tokens_per_second=eval_tps,
        eval_seconds=eval_seconds,
    )


def should_run_validation(step: int, target_iterations: int, val_every: int, eval_first_step: bool) -> bool:
    if target_iterations <= 0:
        return False
    if step == target_iterations - 1:
        return True
    if step == 0:
        return eval_first_step
    return val_every > 0 and step % val_every == 0


def should_run_post_loop_validation(has_validation: bool, completed_steps: int, last_step: int, last_val_step: int | None) -> bool:
    return has_validation and completed_steps > 0 and last_val_step != last_step


def train_time_validation_enabled(cfg: TrainConfig, has_validation: bool) -> bool:
    if not has_validation:
        return False
    return cfg.eval_first_step or cfg.val_every > 0


def train_seq_len_for_step(cfg: TrainConfig, step: int) -> int:
    if cfg.train_seq_len_min is None or cfg.train_seq_len_warmup_steps <= 0:
        return cfg.model.seq_len
    if step < cfg.train_seq_len_warmup_steps:
        return cfg.train_seq_len_min
    return cfg.model.seq_len


def train_one_run(cfg: TrainConfig) -> RunSummary:
    validate_config(cfg)
    started_at = time.time()
    cfg_hash = config_hash(cfg)
    run_id = make_run_id(cfg, cfg_hash)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "config.json", config_to_dict(cfg))
    metrics_path = resolve_metrics_jsonl_path(output_dir, cfg.metrics_jsonl_path)
    tensorboard_log_dir = resolve_tensorboard_log_dir(output_dir, cfg.tensorboard_log_dir)

    dist_info = init_distributed()
    session_summary: RunSummary | None = None
    writer: SummaryWriter | None = None
    try:
        device = get_device(dist_info)
        rank0 = dist_info.rank == 0
        validate_runtime_config(cfg, dist_info.world_size)
        setup_seed(cfg.seed + dist_info.rank, deterministic=cfg.deterministic)
        if device.type == "cuda":
            reset_peak_vram_stats(device)

        model = RecurrentGPT(cfg.model).to(device)
        if cfg.use_compile and hasattr(torch, "compile") and device.type == "cuda" and not cfg.evaluate_only:  # pragma: no cover - GPU feature
            model = torch.compile(model, dynamic=False)  # type: ignore[assignment]
        wrapped_model = maybe_wrap_ddp(model, dist_info, device)
        raw_model = unwrap_model(wrapped_model)

        train_loader = DistributedTokenLoader(cfg.train_pattern, dist_info.rank, dist_info.world_size, device)
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = load_validation_resources(cfg, device)

        opt_bundle = build_optimizers(raw_model, cfg)
        lawa = LinearWeightAverager() if cfg.use_lawa else None

        resume_state: ResumeState | None = None
        if cfg.resume_from:
            resume_state = load_checkpoint(cfg.resume_from, cfg, raw_model, opt_bundle, train_loader, lawa)
            run_id = resume_state.run_id
            cfg_hash = resume_state.config_hash

        if rank0:
            writer = SummaryWriter(
                log_dir=str(tensorboard_log_dir),
                purge_step=None if resume_state is None else resume_state.next_step,
                flush_secs=5,
            )
            tb_log_run_start(
                writer,
                run_id=run_id,
                mode="benchmark" if cfg.benchmark_only else "train",
                cfg_hash=cfg_hash,
                cfg=cfg,
                output_dir=output_dir,
            )

        if rank0:
            append_metrics_event(
                metrics_path,
                {
                    "event": "run_start",
                    "mode": "benchmark" if cfg.benchmark_only else "train",
                    "run_id": run_id,
                    "config_hash": cfg_hash,
                    "output_dir": str(output_dir),
                    "metrics_path": str(metrics_path),
                    "tensorboard_log_dir": str(tensorboard_log_dir),
                    "resume_from": cfg.resume_from,
                    "effective_depth": cfg.model.effective_depth,
                    "max_wallclock_seconds": cfg.max_wallclock_seconds,
                },
            )

        latest_ckpt_path, final_ckpt_path = checkpoint_paths(output_dir)
        training_start = time.time()
        training_elapsed_prior = 0.0 if resume_state is None else resume_state.elapsed_training_seconds
        total_tokens = 0 if resume_state is None else resume_state.total_tokens
        start_step = 0 if resume_state is None else resume_state.next_step
        completed_steps = start_step
        last_step = start_step - 1
        last_train_loss = None if resume_state is None else resume_state.last_train_loss
        last_val = None if resume_state is None else resume_state.last_val
        last_val_step: int | None = None

        target_iterations = cfg.iterations if not cfg.benchmark_only else min(cfg.iterations, cfg.benchmark_train_steps)
        schedule_total_steps = target_iterations
        for step in range(start_step, target_iterations):
            session_elapsed = time.time() - training_start
            if session_elapsed >= cfg.max_wallclock_seconds:
                if rank0:
                    print(f"[train] stopping early at step {step} due to max_wallclock_seconds={cfg.max_wallclock_seconds}")
                break

            if schedule_total_steps == target_iterations:
                schedule_total_steps = estimate_effective_total_steps(
                    step,
                    start_step=start_step,
                    session_elapsed=session_elapsed,
                    target_iterations=target_iterations,
                    max_wallclock_seconds=cfg.max_wallclock_seconds,
                    warmup_steps=cfg.optim.warmup_steps,
                )

            step_start = time.time()
            raw_model.set_global_step(step)
            set_optimizer_lrs(opt_bundle, step, schedule_total_steps, cfg.optim)
            for opt in opt_bundle.all():
                opt.zero_grad(set_to_none=True)

            train_seq_len = train_seq_len_for_step(cfg, step)
            total_loss = 0.0
            for _micro in range(cfg.grad_accum_steps):
                x, y = train_loader.next_batch(cfg.train_batch_tokens, train_seq_len, cfg.grad_accum_steps)
                with maybe_autocast(device, enabled=True):
                    _, loss = wrapped_model(x, y)
                if loss is None:
                    raise RuntimeError("model did not return a training loss")
                scaled_loss = loss / cfg.grad_accum_steps
                scaled_loss.backward()
                total_loss += float(loss.detach().item())

            if cfg.optim.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.optim.grad_clip_norm)
            for opt in opt_bundle.all():
                opt.step()

            last_train_loss = total_loss / cfg.grad_accum_steps
            total_tokens += cfg.train_batch_tokens
            completed_steps = step + 1
            last_step = step

            if lawa is not None and step >= max(0, schedule_total_steps - cfg.lawa_last_n_steps):
                lawa.update(raw_model)

            if rank0 and (step % cfg.log_every == 0 or step == target_iterations - 1):
                step_seconds = time.time() - step_start
                lr_now = {
                    "embed_lr": current_optimizer_lr(opt_bundle.embed),
                    "head_lr": current_optimizer_lr(opt_bundle.head),
                    "matrix_lr": current_optimizer_lr(opt_bundle.matrix),
                    "scalar_lr": current_optimizer_lr(opt_bundle.scalar),
                }
                print(
                    "[train] step={} loss={:.6f} depth={} step_seconds={:.4f}".format(
                        step,
                        last_train_loss,
                        cfg.model.effective_depth,
                        step_seconds,
                    )
                )
                append_metrics_event(
                    metrics_path,
                    {
                        "event": "train",
                        "mode": "benchmark" if cfg.benchmark_only else "train",
                        "run_id": run_id,
                        "config_hash": cfg_hash,
                        "step": step,
                        "num_steps": completed_steps,
                        "depth": cfg.model.effective_depth,
                        "train_loss": last_train_loss,
                        "train_seq_len": train_seq_len,
                        "step_seconds": step_seconds,
                        "elapsed_training_seconds": training_elapsed_prior + (time.time() - training_start),
                        "total_tokens": total_tokens,
                        "total_tokens_M": total_tokens / 1e6,
                        "lr_scale": lr_scale(step, schedule_total_steps, cfg.optim),
                        "lr_total_steps": schedule_total_steps,
                        **lr_now,
                    },
                )
                tb_log_train_event(
                    writer,
                    step=step,
                    train_loss=last_train_loss,
                    step_seconds=step_seconds,
                    elapsed_training_seconds=training_elapsed_prior + (time.time() - training_start),
                    total_tokens=total_tokens,
                    total_tokens_M=total_tokens / 1e6,
                    lr_scale_value=lr_scale(step, schedule_total_steps, cfg.optim),
                    lr_now=lr_now,
                )

            validation_enabled = train_time_validation_enabled(cfg, val_tokens is not None and not cfg.train_phase_only)

            if validation_enabled and should_run_validation(step, target_iterations, cfg.val_every, cfg.eval_first_step):
                barrier_if_needed()
                last_val = eval_val(
                    cfg=cfg,
                    model=wrapped_model,
                    rank=dist_info.rank,
                    world_size=dist_info.world_size,
                    device=device,
                    val_tokens=val_tokens,
                    base_bytes_lut=base_bytes_lut,
                    has_leading_space_lut=has_leading_space_lut,
                    is_boundary_token_lut=is_boundary_token_lut,
                )
                last_val_step = step
                if rank0:
                    if last_val.val_bpb is None:
                        print(f"[val] step={step} loss={last_val.val_loss:.6f}")
                    else:
                        print(f"[val] step={step} loss={last_val.val_loss:.6f} bpb={last_val.val_bpb:.6f}")
                    append_metrics_event(
                        metrics_path,
                        {
                            "event": "val",
                            "phase": "periodic",
                            "mode": "benchmark" if cfg.benchmark_only else "train",
                            "run_id": run_id,
                            "config_hash": cfg_hash,
                            "step": step,
                            "val_loss": last_val.val_loss,
                            "val_bpb": last_val.val_bpb,
                            "token_count": last_val.token_count,
                            "byte_count": last_val.byte_count,
                            "eval_seconds": last_val.eval_seconds,
                        },
                    )
                    tb_log_val_event(writer, phase="periodic", step=step, stats=last_val)

            if rank0 and cfg.checkpoint_every > 0 and completed_steps % cfg.checkpoint_every == 0:
                save_checkpoint(
                    latest_ckpt_path,
                    cfg,
                    run_id,
                    cfg_hash,
                    raw_model,
                    opt_bundle,
                    train_loader,
                    lawa,
                    next_step=completed_steps,
                    elapsed_training_seconds=training_elapsed_prior + (time.time() - training_start),
                    total_tokens=total_tokens,
                    last_train_loss=last_train_loss,
                    last_val=last_val,
                )

        validation_enabled = val_tokens is not None and not cfg.train_phase_only

        if should_run_post_loop_validation(validation_enabled, completed_steps, last_step, last_val_step):
            barrier_if_needed()
            last_val = eval_val(
                cfg=cfg,
                model=wrapped_model,
                rank=dist_info.rank,
                world_size=dist_info.world_size,
                device=device,
                val_tokens=val_tokens,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
            )
            last_val_step = last_step
            if rank0:
                if last_val.val_bpb is None:
                    print(f"[val:final] step={last_step} loss={last_val.val_loss:.6f}")
                else:
                    print(f"[val:final] step={last_step} loss={last_val.val_loss:.6f} bpb={last_val.val_bpb:.6f}")
                append_metrics_event(
                    metrics_path,
                    {
                        "event": "val",
                        "phase": "final",
                        "mode": "benchmark" if cfg.benchmark_only else "train",
                        "run_id": run_id,
                        "config_hash": cfg_hash,
                        "step": last_step,
                        "val_loss": last_val.val_loss,
                        "val_bpb": last_val.val_bpb,
                        "token_count": last_val.token_count,
                        "byte_count": last_val.byte_count,
                        "eval_seconds": last_val.eval_seconds,
                    },
                )
                tb_log_val_event(writer, phase="final", step=last_step, stats=last_val)

        if lawa is not None and lawa.count > 0:
            lawa.load_into(raw_model)
            if validation_enabled:
                last_val = eval_val(
                    cfg=cfg,
                    model=wrapped_model,
                    rank=dist_info.rank,
                    world_size=dist_info.world_size,
                    device=device,
                    val_tokens=val_tokens,
                    base_bytes_lut=base_bytes_lut,
                    has_leading_space_lut=has_leading_space_lut,
                    is_boundary_token_lut=is_boundary_token_lut,
                )
                if rank0:
                    if last_val.val_bpb is None:
                        print(f"[val:lawa] loss={last_val.val_loss:.6f}")
                    else:
                        print(f"[val:lawa] loss={last_val.val_loss:.6f} bpb={last_val.val_bpb:.6f}")
                    append_metrics_event(
                        metrics_path,
                        {
                            "event": "val",
                            "phase": "lawa",
                            "mode": "benchmark" if cfg.benchmark_only else "train",
                            "run_id": run_id,
                            "config_hash": cfg_hash,
                            "step": last_step,
                            "val_loss": last_val.val_loss,
                            "val_bpb": last_val.val_bpb,
                            "token_count": last_val.token_count,
                            "byte_count": last_val.byte_count,
                            "eval_seconds": last_val.eval_seconds,
                        },
                    )
                    tb_log_val_event(writer, phase="lawa", step=last_step, stats=last_val)

        training_seconds = training_elapsed_prior + (time.time() - training_start)
        num_params = count_parameters(raw_model)
        benchmark = None
        export_stats = None
        if cfg.save_final_quantized and rank0:
            export_stats, _ = export_quantized_artifact(
                cfg=cfg,
                model=raw_model,
                output_dir=output_dir,
                run_id=run_id,
                cfg_hash=cfg_hash,
                val_stats=last_val,
            )
            if cfg.verify_export_reload and validation_enabled:
                reloaded_model, _manifest, _artifact_dir = load_model_from_artifact(output_dir / cfg.artifact_bundle_name, device)
                reload_val = eval_val(
                    cfg=cfg,
                    model=reloaded_model,
                    rank=0,
                    world_size=1,
                    device=device,
                    val_tokens=val_tokens,
                    base_bytes_lut=base_bytes_lut,
                    has_leading_space_lut=has_leading_space_lut,
                    is_boundary_token_lut=is_boundary_token_lut,
                )
                export_stats.reload_val_loss = reload_val.val_loss
                export_stats.reload_val_bpb = reload_val.val_bpb
                atomic_write_json(
                    output_dir / "artifact_reload_eval.json",
                    {
                        "val_loss": reload_val.val_loss,
                        "val_bpb": reload_val.val_bpb,
                        "token_count": reload_val.token_count,
                        "eval_seconds": reload_val.eval_seconds,
                    },
                )

        checkpoint_path = None
        if rank0:
            save_checkpoint(
                latest_ckpt_path,
                cfg,
                run_id,
                cfg_hash,
                raw_model,
                opt_bundle,
                train_loader,
                lawa,
                next_step=completed_steps,
                elapsed_training_seconds=training_seconds,
                total_tokens=total_tokens,
                last_train_loss=last_train_loss,
                last_val=last_val,
            )
            save_checkpoint(
                final_ckpt_path,
                cfg,
                run_id,
                cfg_hash,
                raw_model,
                opt_bundle,
                train_loader,
                lawa,
                next_step=completed_steps,
                elapsed_training_seconds=training_seconds,
                total_tokens=total_tokens,
                last_train_loss=last_train_loss,
                last_val=last_val,
            )
            checkpoint_path = str(final_ckpt_path)

        session_summary = RunSummary(
            status="success",
            mode="benchmark" if cfg.benchmark_only else "train",
            run_id=run_id,
            config_hash=cfg_hash,
            step=last_step,
            num_steps=completed_steps,
            train_loss=last_train_loss,
            val_loss=None if last_val is None else last_val.val_loss,
            val_bpb=None if last_val is None else last_val.val_bpb,
            export=export_stats,
            benchmark=None,
            training_seconds=training_seconds,
            total_seconds=time.time() - started_at,
            peak_vram_mb=peak_vram_mb(device),
            mfu_percent=estimate_mfu_percent(cfg, total_tokens / max(training_seconds, 1e-9), num_params, device),
            total_tokens=total_tokens,
            total_tokens_M=total_tokens / 1e6,
            model_params=num_params,
            num_params_M=num_params / 1e6,
            effective_depth=cfg.model.effective_depth,
            checkpoint_path=checkpoint_path,
            metrics_path=str(metrics_path),
            tensorboard_log_dir=str(tensorboard_log_dir),
            results_path=str(output_dir / "results.json"),
        )
        session_summary.benchmark = benchmark_stats_from_run(session_summary, last_val)

        if rank0:
            results_payload = summary_to_results(session_summary, cfg, output_dir, started_at, time.time())
            atomic_write_json(output_dir / "results.json", results_payload)
            atomic_write_json(output_dir / "run_summary.json", dataclasses.asdict(session_summary))
            append_results_tsv(cfg.results_tsv_path, results_payload)
            append_metrics_event(
                metrics_path,
                {
                    "event": "summary",
                    "mode": session_summary.mode,
                    "run_id": run_id,
                    "config_hash": cfg_hash,
                    "status": session_summary.status,
                    "step": session_summary.step,
                    "num_steps": session_summary.num_steps,
                    "training_seconds": session_summary.training_seconds,
                    "total_seconds": session_summary.total_seconds,
                    "val_loss": session_summary.val_loss,
                    "val_bpb": session_summary.val_bpb,
                    "artifact_bytes": 0 if session_summary.export is None else session_summary.export.artifact_bytes,
                    "results_path": results_payload["results_path"],
                },
            )
            tb_log_summary(writer, step=session_summary.step, summary=session_summary)
            writer.flush()
            emit_metric_lines(results_payload)
        barrier_if_needed()
        return session_summary
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()


def evaluate_exported_artifact(cfg: TrainConfig) -> RunSummary:
    validate_config(cfg)
    if not cfg.load_artifact_path:
        raise ConfigError("evaluate_only requires load_artifact_path")
    started_at = time.time()
    cfg_hash = config_hash(cfg)
    run_id = make_run_id(cfg, cfg_hash)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "config.json", config_to_dict(cfg))
    metrics_path = resolve_metrics_jsonl_path(output_dir, cfg.metrics_jsonl_path)
    tensorboard_log_dir = resolve_tensorboard_log_dir(output_dir, cfg.tensorboard_log_dir)

    dist_info = init_distributed()
    writer: SummaryWriter | None = None
    try:
        device = get_device(dist_info)
        rank0 = dist_info.rank == 0
        validate_runtime_config(cfg, dist_info.world_size)
        setup_seed(cfg.seed + dist_info.rank, deterministic=cfg.deterministic)
        if device.type == "cuda":
            reset_peak_vram_stats(device)

        if rank0:
            writer = SummaryWriter(log_dir=str(tensorboard_log_dir), flush_secs=5)
            tb_log_run_start(
                writer,
                run_id=run_id,
                mode="eval",
                cfg_hash=cfg_hash,
                cfg=cfg,
                output_dir=output_dir,
            )

        if rank0:
            append_metrics_event(
                metrics_path,
                {
                    "event": "run_start",
                    "mode": "eval",
                    "run_id": run_id,
                    "config_hash": cfg_hash,
                    "output_dir": str(output_dir),
                    "metrics_path": str(metrics_path),
                    "tensorboard_log_dir": str(tensorboard_log_dir),
                    "load_artifact_path": cfg.load_artifact_path,
                    "effective_depth": cfg.model.effective_depth,
                },
            )

        model, manifest, _artifact_dir = load_model_from_artifact(cfg.load_artifact_path, device)
        wrapped_model = maybe_wrap_ddp(model, dist_info, device)
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = load_validation_resources(cfg, device)
        if val_tokens is None:
            raise ConfigError("evaluate_only requires val_pattern")
        val_stats = eval_val(
            cfg=cfg,
            model=wrapped_model,
            rank=dist_info.rank,
            world_size=dist_info.world_size,
            device=device,
            val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
        if rank0:
            append_metrics_event(
                metrics_path,
                {
                    "event": "val",
                    "phase": "eval",
                    "mode": "eval",
                    "run_id": run_id,
                    "config_hash": cfg_hash,
                    "step": -1,
                    "val_loss": val_stats.val_loss,
                    "val_bpb": val_stats.val_bpb,
                    "token_count": val_stats.token_count,
                    "byte_count": val_stats.byte_count,
                    "eval_seconds": val_stats.eval_seconds,
                },
            )
            tb_log_val_event(writer, phase="eval", step=-1, stats=val_stats)
        export_stats = ExportStats(
            artifact_dir=str(resolve_artifact_dir(cfg.load_artifact_path)),
            manifest_path=str(resolve_artifact_dir(cfg.load_artifact_path) / "manifest.json"),
            model_blob_path=str(resolve_artifact_dir(cfg.load_artifact_path) / manifest["compressed_model"]["artifact_relpath"]),
            compressed_model_bytes=int(manifest["byte_counts"]["compressed_model_bytes"]),
            code_bytes=int(manifest["byte_counts"]["code_bytes"]),
            artifact_bytes=int(manifest["byte_counts"]["artifact_bytes"]),
            quant_payload_bytes=int(manifest["byte_counts"]["quant_payload_bytes"]),
            reload_val_loss=val_stats.val_loss,
            reload_val_bpb=val_stats.val_bpb,
        )
        summary = RunSummary(
            status="success",
            mode="eval",
            run_id=run_id,
            config_hash=cfg_hash,
            step=-1,
            num_steps=0,
            train_loss=None,
            val_loss=val_stats.val_loss,
            val_bpb=val_stats.val_bpb,
            export=export_stats,
            benchmark=BenchmarkStats(
                train_tokens_per_second=0.0,
                eval_tokens_per_second=val_stats.token_count / max(val_stats.eval_seconds, 1e-9),
                eval_seconds=val_stats.eval_seconds,
            ),
            training_seconds=0.0,
            total_seconds=time.time() - started_at,
            peak_vram_mb=peak_vram_mb(device),
            mfu_percent=0.0,
            total_tokens=0,
            total_tokens_M=0.0,
            model_params=count_parameters(model),
            num_params_M=count_parameters(model) / 1e6,
            effective_depth=model.cfg.effective_depth,
            checkpoint_path=None,
            metrics_path=str(metrics_path),
            tensorboard_log_dir=str(tensorboard_log_dir),
            results_path=str(output_dir / "results.json"),
        )
        if rank0:
            results_payload = summary_to_results(summary, cfg, output_dir, started_at, time.time())
            atomic_write_json(output_dir / "results.json", results_payload)
            atomic_write_json(output_dir / "run_summary.json", dataclasses.asdict(summary))
            append_results_tsv(cfg.results_tsv_path, results_payload)
            append_metrics_event(
                metrics_path,
                {
                    "event": "summary",
                    "mode": "eval",
                    "run_id": run_id,
                    "config_hash": cfg_hash,
                    "status": summary.status,
                    "step": summary.step,
                    "num_steps": summary.num_steps,
                    "training_seconds": summary.training_seconds,
                    "total_seconds": summary.total_seconds,
                    "val_loss": summary.val_loss,
                    "val_bpb": summary.val_bpb,
                    "artifact_bytes": 0 if summary.export is None else summary.export.artifact_bytes,
                    "results_path": results_payload["results_path"],
                },
            )
            tb_log_summary(writer, step=0, summary=summary)
            writer.flush()
            emit_metric_lines(results_payload)
        barrier_if_needed()
        return summary
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()


# ============================================================================
# CLI
# ============================================================================


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Depth-recurrent GQA transformer with per-loop low-rank deltas and QAT.",
    )
    parser.add_argument("--config_json", type=str, default=None, help="Optional path to a JSON file matching TrainConfig.")
    parser.add_argument("--train_pattern", type=str, default=None)
    parser.add_argument("--val_pattern", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--results_tsv_path", type=str, default=None)
    parser.add_argument("--metrics_jsonl_path", type=str, default=None)
    parser.add_argument("--tensorboard_log_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--train_batch_tokens", type=int, default=None)
    parser.add_argument("--val_batch_tokens", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--val_every", type=int, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--max_wallclock_seconds", type=float, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--num_kv_heads", type=int, default=None)
    parser.add_argument("--q_low_rank", type=int, default=None)
    parser.add_argument("--stem_layers", type=int, default=None)
    parser.add_argument("--shared_layers", type=int, default=None)
    parser.add_argument("--recurrence_loops", type=int, default=None)
    parser.add_argument("--tail_layers", type=int, default=None)
    parser.add_argument("--adapter_rank", type=int, default=None)
    parser.add_argument("--adapter_alpha", type=float, default=None)
    parser.add_argument("--adapter_targets", type=str, default=None, help="Comma-separated subset of q,k,v,attn_out,mlp_in,mlp_out")
    parser.add_argument("--fake_quant_start_step", type=int, default=None)
    parser.add_argument("--clip_percentile", type=float, default=None)
    parser.add_argument("--embed_lr", type=float, default=None)
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--matrix_lr", type=float, default=None)
    parser.add_argument("--scalar_lr", type=float, default=None)
    parser.add_argument("--counted_code_paths", type=str, default=None, help="Comma-separated files to count into artifact_bytes")
    parser.add_argument("--load_artifact_path", type=str, default=None)
    parser.add_argument("--fake_quant_during_train", action="store_true")
    parser.add_argument("--no_fake_quant_during_train", action="store_true")
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--no_eval_first_step", action="store_true")
    parser.add_argument("--no_lawa", action="store_true")
    parser.add_argument("--no_save_final_quantized", action="store_true")
    parser.add_argument("--no_verify_export_reload", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark_train_steps", type=int, default=None)
    parser.add_argument("--benchmark_eval_repeats", type=int, default=None)
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--train_phase_only", action="store_true")
    return parser


def _parse_csv_tuple(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(parts)


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    if args.config_json:
        loaded = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        cfg = train_config_from_dict(loaded)

    for name in [
        "train_pattern",
        "val_pattern",
        "tokenizer_path",
        "output_dir",
        "run_name",
        "results_tsv_path",
        "metrics_jsonl_path",
        "tensorboard_log_dir",
        "seed",
        "iterations",
        "train_batch_tokens",
        "val_batch_tokens",
        "grad_accum_steps",
        "log_every",
        "val_every",
        "checkpoint_every",
        "resume_from",
        "max_wallclock_seconds",
        "load_artifact_path",
        "benchmark_train_steps",
        "benchmark_eval_repeats",
    ]:
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)

    for name in [
        "seq_len",
        "vocab_size",
        "d_model",
        "num_heads",
        "num_kv_heads",
        "q_low_rank",
        "stem_layers",
        "shared_layers",
        "recurrence_loops",
        "tail_layers",
        "adapter_rank",
        "adapter_alpha",
        "fake_quant_start_step",
    ]:
        value = getattr(args, name)
        if value is not None:
            setattr(cfg.model, name, value)

    for name in ["embed_lr", "head_lr", "matrix_lr", "scalar_lr"]:
        value = getattr(args, name)
        if value is not None:
            setattr(cfg.optim, name, value)

    if args.adapter_targets is not None:
        parsed_targets = _parse_csv_tuple(args.adapter_targets)
        if parsed_targets is None:
            raise ConfigError("adapter_targets parsing failed")
        cfg.model.adapter_targets = parsed_targets

    if args.clip_percentile is not None:
        cfg.quant.clip_percentile = args.clip_percentile
    if args.counted_code_paths is not None:
        parsed_paths = _parse_csv_tuple(args.counted_code_paths)
        if parsed_paths is None:
            raise ConfigError("counted_code_paths parsing failed")
        cfg.counted_code_paths = parsed_paths

    if args.fake_quant_during_train:
        cfg.model.fake_quant_during_train = True
    if args.no_fake_quant_during_train:
        cfg.model.fake_quant_during_train = False
    if args.use_compile:
        cfg.use_compile = True
    if args.no_eval_first_step:
        cfg.eval_first_step = False
    if args.no_lawa:
        cfg.use_lawa = False
    if args.no_save_final_quantized:
        cfg.save_final_quantized = False
    if args.no_verify_export_reload:
        cfg.verify_export_reload = False
    if args.benchmark:
        cfg.benchmark_only = True
    if args.evaluate_only:
        cfg.evaluate_only = True
    if args.train_phase_only:
        cfg.train_phase_only = True
    rebalance_shared_layers_vs_loops(cfg.model)
    expand_adapter_capacity(cfg.model)
    reallocate_one_shared_layer_into_tail(cfg.model)
    reallocate_second_shared_layer_into_tail(cfg.model)
    reallocate_third_shared_layer_into_tail(cfg.model)
    move_fake_quant_to_warmup_boundary_on_deep_tail(cfg.model)
    widen_recurrent_mlp_on_deep_tail(cfg.model)
    tighten_export_clip_on_accepted_deep_tail(cfg)
    modestly_widen_recurrent_mlp_on_wallclock_deep_tail(cfg)
    shift_accepted_deep_tail_stem_into_tail(cfg)
    retune_shifted_deep_tail_width_and_warmdown(cfg)
    use_int6_mlp_export_on_retuned_stemless_deep_tail(cfg)
    trade_one_tail_block_for_true_3x_tail_mlp_on_int6_baseline(cfg)
    trade_one_more_tail_block_for_3p5x_tail_mlp_on_int6_true_3x_line(cfg)
    trade_one_more_tail_block_for_4x_tail_mlp_on_int6_3p5x_line(cfg)
    trade_one_more_tail_block_for_4p5x_tail_mlp_on_int6_4x_line(cfg)
    widen_unique_tail_mlp_on_int6_4p5x_line(cfg)
    widen_unique_tail_mlp_on_int6_5x_line(cfg)
    extend_context_on_compact_int6_6x_tail_line(cfg)
    trade_one_tail_block_for_8x_tail_mlp_on_compact_seq640_line(cfg)
    trade_one_more_tail_block_for_12x_tail_mlp_on_compact_seq640_line(cfg)
    extend_context_on_compact_tail2_12x_line(cfg)
    rebalance_compact_seq768_tail2_12x_line_into_tail3_8x(cfg)
    reallocate_low_rank_q_into_true_3x_carrier_on_recovered_compact_line(cfg)
    add_short_to_full_context_curriculum_on_low_rank_q_compact_line(cfg)
    disable_attention_fake_quant_on_warmed_low_rank_q_compact_line(cfg)
    shrink_global_batch_on_selective_qat_low_rank_q_compact_line(cfg)
    shrink_global_batch_further_on_selective_qat_low_rank_q_compact_line(cfg)
    delay_mlp_fake_quant_until_full_context_on_small_batch_selective_qat_line(cfg)
    restore_full_rank_q_on_final_tail_of_late_qat_small_batch_line(cfg)
    keep_tied_embeddings_float_on_final_tail_q_small_batch_line(cfg)
    return cfg


def run_main(cfg: TrainConfig) -> RunSummary:
    return evaluate_exported_artifact(cfg) if cfg.evaluate_only else train_one_run(cfg)


def main() -> None:
    args = build_argparser().parse_args()
    cfg = config_from_args(args)
    started_at = time.time()
    cfg_hash = config_hash(cfg)
    run_id = make_run_id(cfg, cfg_hash)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "config.json", config_to_dict(cfg))
    try:
        summary = run_main(cfg)
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            print(json.dumps(dataclasses.asdict(summary), indent=2))
        raise SystemExit(EXIT_SUCCESS)
    except KeyboardInterrupt as exc:
        write_failure_outputs(cfg, run_id, cfg_hash, started_at, EXIT_INTERRUPTED, exc)
        print(f"interrupted; see {Path(cfg.output_dir) / 'crash.json'}", file=sys.stderr)
        raise SystemExit(EXIT_INTERRUPTED)
    except ConfigError as exc:
        write_failure_outputs(cfg, run_id, cfg_hash, started_at, EXIT_INVALID_CONFIG, exc)
        print(f"config error: {exc}; see {Path(cfg.output_dir) / 'crash.json'}", file=sys.stderr)
        raise SystemExit(EXIT_INVALID_CONFIG)
    except Exception as exc:
        write_failure_outputs(cfg, run_id, cfg_hash, started_at, EXIT_RUNTIME_ERROR, exc)
        print(f"runtime error: {exc}; see {Path(cfg.output_dir) / 'crash.json'}", file=sys.stderr)
        raise SystemExit(EXIT_RUNTIME_ERROR)


if __name__ == "__main__":
    main()
