"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 20000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    training_preset = os.environ.get("TRAINING_PRESET", "").strip().lower()
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    enable_torch_compile = bool(int(os.environ.get("ENABLE_TORCH_COMPILE", "1")))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 256))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 512))
    val_max_seqs = int(os.environ.get("VAL_MAX_SEQS", "0"))
    final_eval_max_seqs = int(os.environ.get("FINAL_EVAL_MAX_SEQS", "0"))
    submission_size_limit_bytes = int(os.environ.get("SUBMISSION_SIZE_LIMIT_BYTES", 16_000_000))
    label_smoothing = float(os.environ.get("LABEL_SMOOTHING", 0.0))
    z_loss_coeff = float(os.environ.get("Z_LOSS_COEFF", 0.0))
    row_norm_loss_coeff = float(os.environ.get("ROW_NORM_LOSS_COEFF", "0.0"))
    outlier_loss_coeff = float(os.environ.get("OUTLIER_LOSS_COEFF", "0.0"))
    outlier_loss_threshold = float(os.environ.get("OUTLIER_LOSS_THRESHOLD", "3.0"))
    logit_var_loss_coeff = float(os.environ.get("LOGIT_VAR_LOSS_COEFF", "0.0"))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.0))
    ema_start_step = int(os.environ.get("EMA_START_STEP", 0))
    force_fp16_tied_embed_export = bool(int(os.environ.get("FORCE_FP16_TIED_EMBED_EXPORT", "1")))
    int8_auto_keep_budget_bytes = int(os.environ.get("INT8_AUTO_KEEP_BUDGET_BYTES", "0"))
    int8_auto_keep_max_tensors = int(os.environ.get("INT8_AUTO_KEEP_MAX_TENSORS", "0"))
    int8_auto_keep_min_numel = int(os.environ.get("INT8_AUTO_KEEP_MIN_NUMEL", "65536"))
    int8_sensitivity_log_topk = int(os.environ.get("INT8_SENSITIVITY_LOG_TOPK", "8"))
    int8_auto_keep_row_budget_bytes = int(os.environ.get("INT8_AUTO_KEEP_ROW_BUDGET_BYTES", "0"))
    int8_auto_keep_row_group_size = int(os.environ.get("INT8_AUTO_KEEP_ROW_GROUP_SIZE", "32"))
    train_random_offset_tokens = int(os.environ.get("TRAIN_RANDOM_OFFSET_TOKENS", "0"))
    overtone_embed_init = bool(int(os.environ.get("OVERTONE_EMBED_INIT", "0")))
    overtone_embed_power = float(os.environ.get("OVERTONE_EMBED_POWER", "0.5"))
    overtone_embed_scale = float(os.environ.get("OVERTONE_EMBED_SCALE", "1.0"))
    resid_mix_phase_init = bool(int(os.environ.get("RESID_MIX_PHASE_INIT", "0")))
    resid_mix_phase_sharpness = float(os.environ.get("RESID_MIX_PHASE_SHARPNESS", "8.0"))
    use_alibi = bool(int(os.environ.get("USE_ALIBI", "0")))
    token_shift = bool(int(os.environ.get("TOKEN_SHIFT", "0")))
    use_xpos = bool(int(os.environ.get("USE_XPOS", "0")))
    xpos_scale_base = float(os.environ.get("XPOS_SCALE_BASE", "512.0"))
    use_residual_attention = bool(int(os.environ.get("USE_RESIDUAL_ATTENTION", "0")))
    residual_attention_gain_init = float(os.environ.get("RESIDUAL_ATTENTION_GAIN_INIT", "0.5"))
    memory_kv_slots = int(os.environ.get("MEMORY_KV_SLOTS", "0"))
    macaron_layout = bool(int(os.environ.get("MACARON_LAYOUT", "0")))
    use_relative_bias = bool(int(os.environ.get("USE_RELATIVE_BIAS", "0")))
    relative_bias_num_buckets = int(os.environ.get("RELATIVE_BIAS_NUM_BUCKETS", "32"))
    relative_bias_max_distance = int(os.environ.get("RELATIVE_BIAS_MAX_DISTANCE", "128"))
    retention_layers = int(os.environ.get("RETENTION_LAYERS", "0"))
    retention_decay_init = float(os.environ.get("RETENTION_DECAY_INIT", "1.5"))
    retention_output_gate = bool(int(os.environ.get("RETENTION_OUTPUT_GATE", "1")))
    moe_every_n_layers = int(os.environ.get("MOE_EVERY_N_LAYERS", "0"))
    moe_num_experts = int(os.environ.get("MOE_NUM_EXPERTS", "4"))
    moe_top_k = int(os.environ.get("MOE_TOP_K", "1"))
    moe_hidden = int(os.environ.get("MOE_HIDDEN", "0"))
    moe_aux_loss_coeff = float(os.environ.get("MOE_AUX_LOSS_COEFF", "0.0"))
    use_hyper_connections = bool(int(os.environ.get("USE_HYPER_CONNECTIONS", "0")))
    hyper_connection_layers = int(os.environ.get("HYPER_CONNECTION_LAYERS", "0"))
    hyper_connection_gate_init = float(os.environ.get("HYPER_CONNECTION_GATE_INIT", "-2.0"))
    distill_teacher_checkpoint = os.environ.get("DISTILL_TEACHER_CHECKPOINT", "").strip()
    distill_weight = float(os.environ.get("DISTILL_WEIGHT", "0.0"))
    distill_temperature = float(os.environ.get("DISTILL_TEMPERATURE", "1.0"))
    fake_quant_tail_steps = int(os.environ.get("FAKE_QUANT_TAIL_STEPS", "0"))
    fake_quant_bits = int(os.environ.get("FAKE_QUANT_BITS", "8"))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    mlp_hidden = int(os.environ.get("MLP_HIDDEN", "992"))
    num_register_tokens = int(os.environ.get("NUM_REGISTER_TOKENS", "0"))
    num_shared_layers = int(os.environ.get("NUM_SHARED_LAYERS", "0"))
    shared_layer_repeats = int(os.environ.get("SHARED_LAYER_REPEATS", "0"))
    shared_loop_gate_init = float(os.environ.get("SHARED_LOOP_GATE_INIT", "-2.0"))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.07))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.06))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.06))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", "0.0"))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))

    def __init__(self):
        self._apply_training_preset()

    def _apply_training_preset(self) -> None:
        if not self.training_preset:
            return
        presets: dict[str, dict[str, object]] = {
            "long_context_2048": {
                "train_seq_len": 2048,
                "eval_batch_seqs": 256,
            },
            "long_context_2048_standard_eval": {
                "train_seq_len": 2048,
                "eval_stride": 0,
                "eval_batch_seqs": 256,
            },
            "long_context_2048_sliding_eval": {
                "train_seq_len": 2048,
                "eval_stride": 64,
                "eval_batch_seqs": 256,
            },
            "top_recipe_10l": {
                "num_layers": 10,
                "mlp_hidden": 992,
                "warmdown_iters": 20000,
                "eval_stride": 64,
                "eval_batch_seqs": 512,
                "tied_embed_lr": 0.07,
                "matrix_lr": 0.06,
                "scalar_lr": 0.06,
                "muon_momentum": 0.95,
                "muon_momentum_warmup_start": 0.85,
                "muon_momentum_warmup_steps": 500,
                "grad_clip_norm": 1.0,
                "muon_weight_decay": 0.02,
                "overtone_embed_init": True,
                "resid_mix_phase_init": True,
                "force_fp16_tied_embed_export": True,
            },
            "top_recipe_10l_export_plus": {
                "num_layers": 10,
                "mlp_hidden": 992,
                "warmdown_iters": 20000,
                "eval_stride": 64,
                "eval_batch_seqs": 512,
                "tied_embed_lr": 0.07,
                "matrix_lr": 0.06,
                "scalar_lr": 0.06,
                "muon_momentum": 0.95,
                "muon_momentum_warmup_start": 0.85,
                "muon_momentum_warmup_steps": 500,
                "grad_clip_norm": 1.0,
                "muon_weight_decay": 0.02,
                "overtone_embed_init": True,
                "resid_mix_phase_init": True,
                "force_fp16_tied_embed_export": True,
                "fake_quant_tail_steps": 768,
                "fake_quant_bits": 8,
                "int8_auto_keep_budget_bytes": 32768,
                "int8_auto_keep_max_tensors": 2,
                "int8_auto_keep_row_budget_bytes": 98304,
                "int8_auto_keep_row_group_size": 32,
            },
        }
        if self.training_preset not in presets:
            valid = ", ".join(sorted(presets))
            raise ValueError(f"Unknown TRAINING_PRESET={self.training_preset!r}; expected one of: {valid}")
        env_overrides = {
            "train_seq_len": "TRAIN_SEQ_LEN",
            "eval_stride": "EVAL_STRIDE",
            "eval_batch_seqs": "EVAL_BATCH_SEQS",
            "num_layers": "NUM_LAYERS",
            "mlp_hidden": "MLP_HIDDEN",
            "warmdown_iters": "WARMDOWN_ITERS",
            "tied_embed_lr": "TIED_EMBED_LR",
            "matrix_lr": "MATRIX_LR",
            "scalar_lr": "SCALAR_LR",
            "muon_momentum": "MUON_MOMENTUM",
            "muon_momentum_warmup_start": "MUON_MOMENTUM_WARMUP_START",
            "muon_momentum_warmup_steps": "MUON_MOMENTUM_WARMUP_STEPS",
            "grad_clip_norm": "GRAD_CLIP_NORM",
            "muon_weight_decay": "MUON_WEIGHT_DECAY",
            "overtone_embed_init": "OVERTONE_EMBED_INIT",
            "resid_mix_phase_init": "RESID_MIX_PHASE_INIT",
            "force_fp16_tied_embed_export": "FORCE_FP16_TIED_EMBED_EXPORT",
            "fake_quant_tail_steps": "FAKE_QUANT_TAIL_STEPS",
            "fake_quant_bits": "FAKE_QUANT_BITS",
            "int8_auto_keep_budget_bytes": "INT8_AUTO_KEEP_BUDGET_BYTES",
            "int8_auto_keep_max_tensors": "INT8_AUTO_KEEP_MAX_TENSORS",
            "int8_auto_keep_row_budget_bytes": "INT8_AUTO_KEEP_ROW_BUDGET_BYTES",
            "int8_auto_keep_row_group_size": "INT8_AUTO_KEEP_ROW_GROUP_SIZE",
        }
        for attr, value in presets[self.training_preset].items():
            if env_overrides[attr] not in os.environ:
                setattr(self, attr, value)


TRACKED_CONFIG_ENV_VARS = [
    "DATA_PATH",
    "TOKENIZER_PATH",
    "RUN_ID",
    "SEED",
    "VAL_BATCH_SIZE",
    "VAL_LOSS_EVERY",
    "TRAIN_LOG_EVERY",
    "ITERATIONS",
    "WARMDOWN_ITERS",
    "WARMUP_STEPS",
    "TRAIN_BATCH_TOKENS",
    "TRAIN_SEQ_LEN",
    "TRAINING_PRESET",
    "MAX_WALLCLOCK_SECONDS",
    "QK_GAIN_INIT",
    "ENABLE_TORCH_COMPILE",
    "EVAL_STRIDE",
    "EVAL_BATCH_SEQS",
    "VAL_MAX_SEQS",
    "FINAL_EVAL_MAX_SEQS",
    "SUBMISSION_SIZE_LIMIT_BYTES",
    "LABEL_SMOOTHING",
    "Z_LOSS_COEFF",
    "ROW_NORM_LOSS_COEFF",
    "OUTLIER_LOSS_COEFF",
    "OUTLIER_LOSS_THRESHOLD",
    "LOGIT_VAR_LOSS_COEFF",
    "EMA_DECAY",
    "EMA_START_STEP",
    "FORCE_FP16_TIED_EMBED_EXPORT",
    "INT8_AUTO_KEEP_BUDGET_BYTES",
    "INT8_AUTO_KEEP_MAX_TENSORS",
    "INT8_AUTO_KEEP_MIN_NUMEL",
    "INT8_SENSITIVITY_LOG_TOPK",
    "INT8_AUTO_KEEP_ROW_BUDGET_BYTES",
    "INT8_AUTO_KEEP_ROW_GROUP_SIZE",
    "TRAIN_RANDOM_OFFSET_TOKENS",
    "OVERTONE_EMBED_INIT",
    "OVERTONE_EMBED_POWER",
    "OVERTONE_EMBED_SCALE",
    "RESID_MIX_PHASE_INIT",
    "RESID_MIX_PHASE_SHARPNESS",
    "USE_ALIBI",
    "TOKEN_SHIFT",
    "USE_XPOS",
    "XPOS_SCALE_BASE",
    "USE_RESIDUAL_ATTENTION",
    "RESIDUAL_ATTENTION_GAIN_INIT",
    "MEMORY_KV_SLOTS",
    "MACARON_LAYOUT",
    "USE_RELATIVE_BIAS",
    "RELATIVE_BIAS_NUM_BUCKETS",
    "RELATIVE_BIAS_MAX_DISTANCE",
    "RETENTION_LAYERS",
    "RETENTION_DECAY_INIT",
    "RETENTION_OUTPUT_GATE",
    "MOE_EVERY_N_LAYERS",
    "MOE_NUM_EXPERTS",
    "MOE_TOP_K",
    "MOE_HIDDEN",
    "MOE_AUX_LOSS_COEFF",
    "USE_HYPER_CONNECTIONS",
    "HYPER_CONNECTION_LAYERS",
    "HYPER_CONNECTION_GATE_INIT",
    "DISTILL_TEACHER_CHECKPOINT",
    "DISTILL_WEIGHT",
    "DISTILL_TEMPERATURE",
    "FAKE_QUANT_TAIL_STEPS",
    "FAKE_QUANT_BITS",
    "VOCAB_SIZE",
    "NUM_LAYERS",
    "NUM_KV_HEADS",
    "MODEL_DIM",
    "NUM_HEADS",
    "MLP_MULT",
    "MLP_HIDDEN",
    "NUM_REGISTER_TOKENS",
    "NUM_SHARED_LAYERS",
    "SHARED_LAYER_REPEATS",
    "SHARED_LOOP_GATE_INIT",
    "TIE_EMBEDDINGS",
    "ROPE_BASE",
    "LOGIT_SOFTCAP",
    "EMBED_LR",
    "HEAD_LR",
    "TIED_EMBED_LR",
    "TIED_EMBED_INIT_STD",
    "MATRIX_LR",
    "SCALAR_LR",
    "MUON_MOMENTUM",
    "MUON_WEIGHT_DECAY",
    "MUON_BACKEND_STEPS",
    "MUON_MOMENTUM_WARMUP_START",
    "MUON_MOMENTUM_WARMUP_STEPS",
    "BETA1",
    "BETA2",
    "ADAM_EPS",
    "GRAD_CLIP_NORM",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def split_csv_env(name: str) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def hyperparameters_to_dict(args: Hyperparameters) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in Hyperparameters.__dict__.items():
        if key.startswith("_") or callable(value):
            continue
        payload[key] = getattr(args, key)
    return payload


def experiment_metadata(args: Hyperparameters) -> dict[str, object]:
    env_overrides = {name: os.environ[name] for name in TRACKED_CONFIG_ENV_VARS if name in os.environ}
    explicit_changes = [line.strip() for line in os.environ.get("EXPERIMENT_CHANGES", "").splitlines() if line.strip()]
    if not explicit_changes:
        explicit_changes = [f"{name}={value}" for name, value in env_overrides.items() if name != "RUN_ID"]
    return {
        "title": os.environ.get("EXPERIMENT_TITLE", "").strip(),
        "baseline_run_id": os.environ.get("EXPERIMENT_BASELINE_RUN_ID", "").strip(),
        "parent_run_id": os.environ.get("EXPERIMENT_PARENT_RUN_ID", "").strip(),
        "hypothesis": os.environ.get("EXPERIMENT_HYPOTHESIS", "").strip(),
        "expected_effect": os.environ.get("EXPERIMENT_EXPECTED_EFFECT", "").strip(),
        "notes": os.environ.get("EXPERIMENT_NOTES", "").strip(),
        "tags": split_csv_env("EXPERIMENT_TAGS"),
        "explicit_changes": explicit_changes,
        "auto_env_overrides": env_overrides,
        "args": hyperparameters_to_dict(args),
    }


def write_json(path: str | os.PathLike[str], payload: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: str | os.PathLike[str], payload: dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")

# -----------------------------
# MUON OPTIMIZER
# -----------------------------
#
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        weight_decay: float = 0.0,
        nesterov: bool = True,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                weight_decay=weight_decay,
                nesterov=nesterov,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                if weight_decay > 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    max_seqs: int = 0,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    if max_seqs > 0:
        total_seqs = min(total_seqs, max_seqs)
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int,
    max_seqs: int = 0,
) -> tuple[float, float]:
    if batch_seqs <= 0:
        raise ValueError(f"EVAL_BATCH_SEQS must be positive for sliding eval, got {batch_seqs}")
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    if max_seqs > 0:
        total_tokens = min(total_tokens, max_seqs * seq_len)
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= stride]
    total_windows = len(window_starts)
    my_start = (total_windows * rank) // world_size
    my_end = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_start:my_end]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()
    with torch.inference_mode():
        for batch_idx in range(0, len(my_windows), batch_seqs):
            batch_windows = my_windows[batch_idx : batch_idx + batch_seqs]
            batch_size = len(batch_windows)
            x_batch = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=device)
            y_batch = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=device)
            window_lengths: list[int] = []

            for i, ws in enumerate(batch_windows):
                end = min(ws + seq_len, total_tokens)
                window_len = end - ws
                window_lengths.append(window_len)
                chunk = val_tokens[ws : end + 1].to(device=device, dtype=torch.int64, non_blocking=True)
                x_batch[i, :window_len] = chunk[:-1]
                y_batch[i, :window_len] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = base_model.forward_logits(x_batch)

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(batch_size, seq_len)

            for i, ws in enumerate(batch_windows):
                window_len = window_lengths[i]
                score_start = 0 if ws == 0 else window_len - stride
                scored_nll = nll[i, score_start:window_len].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(window_len - score_start)
                tgt_ids = y_batch[i, score_start:window_len]
                prev_ids = x_batch[i, score_start:window_len]
                token_bytes = base_bytes_lut[tgt_ids].to(torch.float64)
                token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
                byte_count += token_bytes.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = loss_sum / token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_NAME_PATTERNS = tuple(
    pattern for pattern in os.environ.get("INT8_KEEP_FLOAT_NAME_PATTERNS", "").split(",") if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
FORCE_FP16_TIED_EMBED_EXPORT = bool(int(os.environ.get("FORCE_FP16_TIED_EMBED_EXPORT", "1")))
LATE_K_FP16_LAYERS = int(os.environ.get("LATE_K_FP16_LAYERS", "2"))
LARGE_MATRIX_QUANT_BITS = int(os.environ.get("LARGE_MATRIX_QUANT_BITS", "6"))

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def should_keep_float_tensor(name: str, t: Tensor) -> bool:
    if FORCE_FP16_TIED_EMBED_EXPORT and name == "tok_emb.weight":
        return True
    if (
        LATE_K_FP16_LAYERS > 0
        and name.endswith("c_k.weight")
        and name.startswith("blocks.")
        and t.ndim == 2
    ):
        try:
            block_idx = int(name.split(".")[1])
            num_layers = Hyperparameters.num_layers
            if block_idx >= max(num_layers - LATE_K_FP16_LAYERS, 0):
                return True
        except (IndexError, ValueError):
            pass
    return t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL or any(pattern in name for pattern in INT8_KEEP_FLOAT_NAME_PATTERNS)

def quantize_float_tensor(t: Tensor, bits: int = 8) -> tuple[Tensor, Tensor]:
    max_val = 127 if bits == 8 else (2 ** (bits - 1)) - 1
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / float(max_val)).clamp_min(1.0 / float(max_val))
        q = torch.clamp(torch.round(clipped / scale[:, None]), -max_val, max_val).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / float(max_val) if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -max_val, max_val).to(torch.int8).contiguous()
    return q, scale


def quantize_tensor_row_groups(
    name: str,
    t: Tensor,
    row_group_size: int,
    passthrough_orig_dtypes: dict[str, str],
) -> list[dict[str, object]]:
    if t.ndim != 2 or row_group_size <= 0:
        return []
    groups: list[dict[str, object]] = []
    for row_start in range(0, t.shape[0], row_group_size):
        row_end = min(row_start + row_group_size, t.shape[0])
        chunk = t[row_start:row_end].contiguous()
        q, s = quantize_float_tensor(chunk)
        deq = dequantize_int8_tensor(q, s, t.dtype)
        diff = deq.float() - chunk.float()
        mse = float(diff.square().mean().item()) if diff.numel() else 0.0
        ref = float(chunk.float().square().mean().item()) if chunk.numel() else 0.0
        rel_mse = mse / max(ref, 1e-12)
        quant_bytes = tensor_nbytes(q) + tensor_nbytes(s)
        keep_bytes = tensor_nbytes(keep_float_tensor(name, chunk, passthrough_orig_dtypes))
        extra_bytes = max(keep_bytes - quant_bytes, 0)
        benefit = rel_mse * float(chunk.numel())
        groups.append(
            {
                "name": name,
                "row_start": row_start,
                "row_end": row_end,
                "extra_bytes": int(extra_bytes),
                "rel_mse": rel_mse,
                "benefit_per_extra_byte": benefit / max(extra_bytes, 1),
            }
        )
    return groups


def dequantize_int8_tensor(q: Tensor, s: Tensor, dtype: torch.dtype) -> Tensor:
    if s.ndim > 0:
        return (q.float() * s.to(dtype=torch.float32).view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
    return (q.float() * float(s.item())).to(dtype=dtype).contiguous()


def select_auto_keep_float_tensors(state_dict: dict[str, Tensor]) -> tuple[set[str], dict[str, object]]:
    budget_bytes = Hyperparameters.int8_auto_keep_budget_bytes
    max_tensors = Hyperparameters.int8_auto_keep_max_tensors
    min_numel = Hyperparameters.int8_auto_keep_min_numel
    log_topk = Hyperparameters.int8_sensitivity_log_topk
    enabled = budget_bytes > 0 or max_tensors > 0
    info: dict[str, object] = {
        "enabled": enabled,
        "budget_bytes": int(budget_bytes),
        "max_tensors": int(max_tensors),
        "min_numel": int(min_numel),
        "selected_names": [],
        "selected_count": 0,
        "selected_extra_bytes": 0,
        "top_candidates": [],
    }
    if not enabled:
        return set(), info

    candidates: list[dict[str, object]] = []
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        if not t.is_floating_point() or should_keep_float_tensor(name, t) or t.numel() < min_numel:
            continue
        q_bits = LARGE_MATRIX_QUANT_BITS if t.ndim == 2 else 8
        q, s = quantize_float_tensor(t, bits=q_bits)
        deq = dequantize_int8_tensor(q, s, t.dtype)
        diff = deq.float() - t.float()
        mse = float(diff.square().mean().item()) if diff.numel() else 0.0
        ref = float(t.float().square().mean().item()) if t.numel() else 0.0
        rel_mse = mse / max(ref, 1e-12)
        max_abs_diff = float(diff.abs().max().item()) if diff.numel() else 0.0
        quant_bytes = tensor_nbytes(q) + tensor_nbytes(s)
        keep_bytes = tensor_nbytes(keep_float_tensor(name, t, {}))
        extra_bytes = max(keep_bytes - quant_bytes, 0)
        benefit = rel_mse * float(t.numel())
        density = benefit / max(extra_bytes, 1)
        candidates.append(
            {
                "name": name,
                "numel": int(t.numel()),
                "keep_bytes": int(keep_bytes),
                "quant_bytes": int(quant_bytes),
                "extra_bytes": int(extra_bytes),
                "rel_mse": rel_mse,
                "max_abs_diff": max_abs_diff,
                "benefit": benefit,
                "benefit_per_extra_byte": density,
            }
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            float(item["benefit_per_extra_byte"]),
            float(item["benefit"]),
            float(item["rel_mse"]),
            int(item["numel"]),
        ),
        reverse=True,
    )
    selected_names: list[str] = []
    used_extra_bytes = 0
    for item in ranked:
        if max_tensors > 0 and len(selected_names) >= max_tensors:
            break
        extra_bytes = int(item["extra_bytes"])
        if budget_bytes > 0 and used_extra_bytes + extra_bytes > budget_bytes:
            continue
        selected_names.append(str(item["name"]))
        used_extra_bytes += extra_bytes

    info["selected_names"] = selected_names
    info["selected_count"] = len(selected_names)
    info["selected_extra_bytes"] = int(used_extra_bytes)
    info["top_candidates"] = [
        {
            "name": str(item["name"]),
            "numel": int(item["numel"]),
            "extra_bytes": int(item["extra_bytes"]),
            "rel_mse": float(item["rel_mse"]),
            "max_abs_diff": float(item["max_abs_diff"]),
            "benefit_per_extra_byte": float(item["benefit_per_extra_byte"]),
        }
        for item in ranked[: max(log_topk, 0)]
    ]
    return set(selected_names), info


def select_auto_keep_float_row_groups(
    state_dict: dict[str, Tensor], passthrough_orig_dtypes: dict[str, str]
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, object]]:
    budget_bytes = Hyperparameters.int8_auto_keep_row_budget_bytes
    group_size = Hyperparameters.int8_auto_keep_row_group_size
    enabled = budget_bytes > 0 and group_size > 0
    info: dict[str, object] = {
        "enabled": enabled,
        "budget_bytes": int(budget_bytes),
        "row_group_size": int(group_size),
        "selected_count": 0,
        "selected_extra_bytes": 0,
        "selected_groups": [],
    }
    if not enabled:
        return {}, info

    candidates: list[dict[str, object]] = []
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        if (
            not t.is_floating_point()
            or should_keep_float_tensor(name, t)
            or t.ndim != 2
            or t.shape[0] < group_size
        ):
            continue
        candidates.extend(quantize_tensor_row_groups(name, t, group_size, passthrough_orig_dtypes))

    ranked = sorted(
        candidates,
        key=lambda item: (float(item["benefit_per_extra_byte"]), float(item["rel_mse"])),
        reverse=True,
    )
    selected: dict[str, list[tuple[int, int]]] = {}
    used_bytes = 0
    seen: set[tuple[str, int]] = set()
    for item in ranked:
        key = (str(item["name"]), int(item["row_start"]))
        if key in seen:
            continue
        extra = int(item["extra_bytes"])
        if used_bytes + extra > budget_bytes:
            continue
        seen.add(key)
        used_bytes += extra
        selected.setdefault(str(item["name"]), []).append((int(item["row_start"]), int(item["row_end"])))
    for name in selected:
        selected[name].sort()
    info["selected_count"] = sum(len(v) for v in selected.values())
    info["selected_extra_bytes"] = int(used_bytes)
    info["selected_groups"] = [
        {"name": name, "row_start": row_start, "row_end": row_end}
        for name, groups in selected.items()
        for row_start, row_end in groups
    ]
    return selected, info


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    auto_keep_names, auto_keep_info = select_auto_keep_float_tensors(state_dict)
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    row_group_passthrough: dict[str, list[dict[str, object]]] = {}
    qmeta: dict[str, dict[str, object]] = {}
    row_keep_groups, row_keep_info = select_auto_keep_float_row_groups(state_dict, passthrough_orig_dtypes)
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "int8_payload_bytes",
            "auto_keep_count",
            "auto_keep_extra_bytes",
            "auto_keep_row_group_count",
            "auto_keep_row_group_extra_bytes",
        ),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if name in auto_keep_names:
            stats["auto_keep_count"] += 1
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        if should_keep_float_tensor(name, t):
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if name in row_keep_groups:
            entries: list[dict[str, object]] = []
            for row_start, row_end in row_keep_groups[name]:
                kept = keep_float_tensor(name, t[row_start:row_end].contiguous(), passthrough_orig_dtypes)
                entries.append({"row_start": row_start, "row_end": row_end, "value": kept})
                stats["auto_keep_row_group_count"] += 1
                stats["int8_payload_bytes"] += tensor_nbytes(kept)
            row_group_passthrough[name] = entries
            qmeta[name] = {
                "scheme": "per_row",
                "axis": 0,
                "row_passthrough": [(entry["row_start"], entry["row_end"]) for entry in entries],
            }
        if s.ndim > 0:
            qmeta.setdefault(name, {"scheme": "per_row", "axis": 0})
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    if row_group_passthrough:
        obj["row_group_passthrough"] = row_group_passthrough
    stats["auto_keep_extra_bytes"] = int(auto_keep_info["selected_extra_bytes"])
    stats["auto_keep_row_group_extra_bytes"] = int(row_keep_info["selected_extra_bytes"])
    return obj, stats, {**auto_keep_info, "row_groups": row_keep_info}

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    row_group_passthrough = obj.get("row_group_passthrough", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = dequantize_int8_tensor(q, s, dtype)
        else:
            out[name] = dequantize_int8_tensor(q, s, dtype)
        if name in row_group_passthrough:
            for entry in row_group_passthrough[name]:
                row_start = int(entry["row_start"])
                row_end = int(entry["row_end"])
                patch = entry["value"].detach().to("cpu").contiguous()
                orig_dtype = passthrough_orig_dtypes.get(name)
                if isinstance(orig_dtype, str):
                    patch = patch.to(dtype=getattr(torch, orig_dtype)).contiguous()
                out[name][row_start:row_end] = patch.to(dtype=out[name].dtype)
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str, random_offset_tokens: int = 0, seed: int = 1337):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.random_offset_tokens = max(int(random_offset_tokens), 0)
        self.rng = random.Random(seed)
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def _take_sequential(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

    def take(self, n: int) -> Tensor:
        if self.random_offset_tokens <= 0:
            return self._take_sequential(n)

        while True:
            max_start = self.tokens.numel() - n
            if max_start < 0:
                self._advance_file()
                continue
            start_low = min(self.pos, max_start)
            start_high = min(max_start, start_low + self.random_offset_tokens)
            start = self.rng.randint(start_low, start_high) if start_high > start_low else start_low
            self.pos = start + n
            return self.tokens[start : start + n]


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(
        self,
        pattern: str,
        rank: int,
        world_size: int,
        device: torch.device,
        random_offset_tokens: int = 0,
        seed: int = 1337,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern, random_offset_tokens=random_offset_tokens, seed=seed + rank)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


def fake_quantize_tensor(t: Tensor, bits: int) -> Tensor:
    if bits <= 0:
        return t
    t32 = t.float()
    qmax = float((1 << (bits - 1)) - 1)
    if qmax <= 0:
        return t
    clip_abs = t32.abs().amax()
    scale = (clip_abs / qmax).clamp_min(1.0 / qmax)
    return (torch.clamp(torch.round(t32 / scale), -qmax, qmax) * scale).to(dtype=t.dtype)


def row_norm_regularization(module: nn.Module) -> Tensor:
    penalty = None
    for param in module.parameters():
        if param.ndim != 2:
            continue
        rows = param.float()
        row_rms = rows.square().mean(dim=1).add_(1e-8).sqrt()
        log_row_rms = row_rms.log()
        value = (log_row_rms - log_row_rms.mean()).square().mean()
        penalty = value if penalty is None else penalty + value
    if penalty is None:
        return torch.zeros((), device=next(module.parameters()).device)
    return penalty


def outlier_regularization(module: nn.Module, threshold: float) -> Tensor:
    penalty = None
    clamped_threshold = max(float(threshold), 1e-4)
    for param in module.parameters():
        if param.ndim < 2:
            continue
        w = param.float()
        scale = w.square().mean().add_(1e-8).sqrt()
        excess = (w.abs() / scale - clamped_threshold).clamp_min(0.0)
        value = excess.square().mean()
        penalty = value if penalty is None else penalty + value
    if penalty is None:
        return torch.zeros((), device=next(module.parameters()).device)
    return penalty


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fake_quant_bits = 0

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        weight = self.weight
        if self.fake_quant_bits > 0:
            weight = fake_quantize_tensor(weight, self.fake_quant_bits)
        return F.linear(x, weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0, use_xpos: bool = False, xpos_scale_base: float = 512.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.use_xpos = use_xpos
        self.xpos_scale_base = xpos_scale_base
        xpos_scale = (torch.arange(0, dim, 2, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer("xpos_scale", xpos_scale.clamp_min(1e-4), persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
        self._scale_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor | None]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            if self.use_xpos:
                centered_t = (t - (seq_len // 2)).unsqueeze(-1) / max(self.xpos_scale_base, 1.0)
                self._scale_cached = self.xpos_scale.to(device).unsqueeze(0).pow(centered_t)[None, None, :, :]
            else:
                self._scale_cached = None
            self._seq_len_cached = seq_len
        scale = self._scale_cached.to(dtype=dtype) if self._scale_cached is not None else None
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype), scale


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, scale: Tensor | None = None, inverse_scale: bool = False) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    if scale is None:
        return rot
    full_scale = torch.cat((scale, scale), dim=-1)
    return rot / full_scale if inverse_scale else rot * full_scale


def _get_alibi_slopes(num_heads: int) -> Tensor:
    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio**i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(num_heads), dtype=torch.float32)
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    slopes = get_slopes_power_of_2(closest_power_of_2)
    extra = _get_alibi_slopes(2 * closest_power_of_2).tolist()[0::2][: num_heads - closest_power_of_2]
    return torch.tensor(slopes + extra, dtype=torch.float32)


def apply_token_shift(x: Tensor) -> Tensor:
    if x.size(1) <= 1 or x.size(2) < 2:
        return x
    half = x.size(-1) // 2
    shifted = torch.zeros_like(x)
    shifted[..., :half] = x[..., :half]
    shifted[:, 1:, half:] = x[:, :-1, half:]
    return shifted


def relative_position_bucket(relative_position: Tensor, num_buckets: int, max_distance: int) -> Tensor:
    n = torch.clamp(-relative_position, min=0)
    max_exact = max(num_buckets // 2, 1)
    is_small = n < max_exact
    large_pos = max_exact + (
        (torch.log(n.float() / max_exact + 1e-6) / math.log(max(max_distance / max_exact, 2.0))) * (num_buckets - max_exact)
    ).to(torch.long)
    large_pos = large_pos.clamp(max=num_buckets - 1)
    return torch.where(is_small, n.to(torch.long), large_pos)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        use_alibi: bool,
        use_xpos: bool,
        xpos_scale_base: float,
        use_residual_attention: bool,
        residual_attention_gain_init: float,
        memory_kv_slots: int,
        use_relative_bias: bool,
        relative_bias_num_buckets: int,
        relative_bias_max_distance: int,
        attention_variant: str = "attention",
        retention_decay_init: float = 1.5,
        retention_output_gate: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.use_alibi = use_alibi
        self.use_residual_attention = use_residual_attention
        self.memory_kv_slots = max(int(memory_kv_slots), 0)
        self.use_relative_bias = use_relative_bias
        self.relative_bias_num_buckets = relative_bias_num_buckets
        self.relative_bias_max_distance = relative_bias_max_distance
        self.attention_variant = attention_variant
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.residual_attention_gain = nn.Parameter(
            torch.full((num_heads,), residual_attention_gain_init, dtype=torch.float32)
        )
        self.retention_decay = nn.Parameter(torch.full((num_heads,), retention_decay_init, dtype=torch.float32))
        self.retention_gate = (
            nn.Parameter(torch.zeros(num_heads, self.head_dim, dtype=torch.float32)) if retention_output_gate else None
        )
        self.rotary = Rotary(self.head_dim, base=rope_base, use_xpos=use_xpos, xpos_scale_base=xpos_scale_base)
        self.register_buffer("alibi_slopes", _get_alibi_slopes(num_heads), persistent=False)
        self.relative_attention_bias = (
            nn.Parameter(torch.zeros(num_heads, relative_bias_num_buckets, dtype=torch.float32))
            if use_relative_bias
            else None
        )
        self.mem_k = (
            nn.Parameter(torch.randn(num_heads, self.memory_kv_slots, self.head_dim, dtype=torch.float32) * 0.02)
            if self.memory_kv_slots > 0
            else None
        )
        self.mem_v = (
            nn.Parameter(torch.randn(num_heads, self.memory_kv_slots, self.head_dim, dtype=torch.float32) * 0.02)
            if self.memory_kv_slots > 0
            else None
        )
        self._alibi_bias_cache: Tensor | None = None
        self._alibi_bias_seq_len = 0

    def _get_retention_bias(self, seq_len: int, total_k: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        mem = max(total_k - seq_len, 0)
        q_pos = torch.arange(seq_len, device=device)[:, None]
        k_pos = torch.arange(seq_len, device=device)[None, :]
        distance = (q_pos - k_pos).clamp_min(0).to(torch.float32)
        decay = F.softplus(self.retention_decay).to(device=device, dtype=torch.float32)[:, None, None]
        bias = -(decay * distance.unsqueeze(0))
        if mem > 0:
            mem_bias = torch.zeros((self.num_heads, seq_len, mem), device=device, dtype=torch.float32)
            bias = torch.cat((mem_bias, bias), dim=-1)
        return bias.unsqueeze(0).to(dtype=dtype)

    def _get_alibi_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if (
            self._alibi_bias_cache is None
            or self._alibi_bias_seq_len != seq_len
            or self._alibi_bias_cache.device != device
        ):
            pos = torch.arange(seq_len, device=device, dtype=torch.float32)
            rel = pos[None, :] - pos[:, None]
            rel = rel.clamp_min(0.0)
            bias = -self.alibi_slopes.to(device=device)[:, None, None] * rel[None, :, :]
            self._alibi_bias_cache = bias.unsqueeze(0)
            self._alibi_bias_seq_len = seq_len
        return self._alibi_bias_cache.to(dtype=dtype)

    def _get_relative_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tensor | None:
        if self.relative_attention_bias is None:
            return None
        q_pos = torch.arange(seq_len, device=device)[:, None]
        k_pos = torch.arange(seq_len, device=device)[None, :]
        rel = k_pos - q_pos
        buckets = relative_position_bucket(rel, self.relative_bias_num_buckets, self.relative_bias_max_distance)
        bias = self.relative_attention_bias[:, buckets]
        return bias.unsqueeze(0).to(dtype=dtype)

    def _manual_attention(
        self, q: Tensor, k: Tensor, v: Tensor, prev_attn_logits: Tensor | None, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        bsz, _, seqlen, _ = q.shape
        total_k = k.size(-2)
        scale = self.head_dim**-0.5
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        if self.attention_variant == "retention":
            logits = logits + self._get_retention_bias(seqlen, total_k, q.device, dtype)
        if self.use_alibi:
            alibi_bias = self._get_alibi_bias(seqlen, q.device, dtype)
            if self.memory_kv_slots > 0:
                mem_bias = torch.zeros((1, self.num_heads, seqlen, self.memory_kv_slots), device=q.device, dtype=dtype)
                alibi_bias = torch.cat((mem_bias, alibi_bias), dim=-1)
            logits = logits + alibi_bias
        rel_bias = self._get_relative_bias(seqlen, q.device, dtype)
        if rel_bias is not None:
            if self.memory_kv_slots > 0:
                mem_bias = torch.zeros((1, self.num_heads, seqlen, self.memory_kv_slots), device=q.device, dtype=dtype)
                rel_bias = torch.cat((mem_bias, rel_bias), dim=-1)
            logits = logits + rel_bias
        if prev_attn_logits is not None and self.use_residual_attention:
            gain = torch.sigmoid(self.residual_attention_gain).to(dtype=dtype)[None, :, None, None]
            logits = logits + gain * prev_attn_logits
        mem = self.memory_kv_slots
        causal = torch.ones((seqlen, seqlen), device=q.device, dtype=torch.bool).triu(1)
        if mem > 0:
            full_mask = torch.zeros((seqlen, total_k), device=q.device, dtype=torch.bool)
            full_mask[:, mem:] = causal
        else:
            full_mask = causal
        logits = logits.masked_fill(full_mask[None, None, :, :], float("-inf"))
        probs = torch.softmax(logits.float(), dim=-1).to(dtype=dtype)
        y = torch.matmul(probs, v)
        if self.attention_variant == "retention" and self.retention_gate is not None:
            gate = torch.sigmoid(self.retention_gate).to(dtype=dtype)[None, :, None, :]
            y = y * gate
        return y, logits.detach()

    def forward(self, x: Tensor, prev_attn_logits: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        if not self.use_alibi:
            cos, sin, scale = self.rotary(seqlen, x.device, q.dtype)
            q = apply_rotary_emb(q, cos, sin, scale=scale)
            k = apply_rotary_emb(k, cos, sin, scale=scale, inverse_scale=True)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        if self.memory_kv_slots > 0:
            mem_k = self.mem_k.to(dtype=q.dtype)[None, :, :, :].expand(bsz, -1, -1, -1)
            mem_v = self.mem_v.to(dtype=q.dtype)[None, :, :, :].expand(bsz, -1, -1, -1)
            k = torch.cat((mem_k, k), dim=-2)
            v = torch.cat((mem_v, v), dim=-2)
        next_attn_logits = None
        if self.use_residual_attention or self.memory_kv_slots > 0 or self.attention_variant == "retention":
            y, next_attn_logits = self._manual_attention(q, k, v, prev_attn_logits, q.dtype)
        else:
            attn_bias = self._get_alibi_bias(seqlen, x.device, q.dtype) if self.use_alibi else None
            rel_bias = self._get_relative_bias(seqlen, x.device, q.dtype)
            if rel_bias is not None:
                attn_bias = rel_bias if attn_bias is None else attn_bias + rel_bias
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                is_causal=True,
            )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y), next_attn_logits


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int, hidden_dim: int = 0):
        super().__init__()
        inner_dim = hidden_dim if hidden_dim > 0 else mlp_mult * dim
        self.fc = CastedLinear(dim, inner_dim, bias=False)
        self.proj = CastedLinear(inner_dim, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class MoEMLP(nn.Module):
    def __init__(self, dim: int, num_experts: int, top_k: int, hidden_dim: int):
        super().__init__()
        self.num_experts = max(int(num_experts), 1)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.gate = CastedLinear(dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(dim, 1, hidden_dim=hidden_dim) for _ in range(self.num_experts)])
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x: Tensor) -> Tensor:
        flat = x.reshape(-1, x.size(-1))
        gate_logits = self.gate(flat).float()
        gate_probs = torch.softmax(gate_logits, dim=-1)
        top_vals, top_idx = torch.topk(gate_probs, k=self.top_k, dim=-1)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        out = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            expert_out = expert(flat)
            expert_weight = (top_vals * (top_idx == expert_idx).to(top_vals.dtype)).sum(dim=-1, keepdim=True)
            out = out + expert_out * expert_weight.to(dtype=out.dtype)
        mean_probs = gate_probs.mean(dim=0)
        top1 = top_idx[:, 0]
        load = torch.bincount(top1, minlength=self.num_experts).to(dtype=mean_probs.dtype, device=mean_probs.device)
        load = load / max(int(top1.numel()), 1)
        self.last_aux_loss = self.num_experts * torch.sum(mean_probs * load)
        return out.view_as(x)


class Block(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        mlp_hidden: int,
        rope_base: float,
        qk_gain_init: float,
        use_alibi: bool,
        token_shift: bool,
        use_xpos: bool,
        xpos_scale_base: float,
        use_residual_attention: bool,
        residual_attention_gain_init: float,
        memory_kv_slots: int,
        use_relative_bias: bool,
        relative_bias_num_buckets: int,
        relative_bias_max_distance: int,
        macaron_layout: bool,
        attention_variant: str,
        retention_decay_init: float,
        retention_output_gate: bool,
        moe_every_n_layers: int,
        moe_num_experts: int,
        moe_top_k: int,
        moe_hidden: int,
    ):
        super().__init__()
        self.token_shift = token_shift
        self.macaron_layout = macaron_layout
        self.layer_idx = layer_idx
        self.attention_variant = attention_variant
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.pre_mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            use_alibi,
            use_xpos,
            xpos_scale_base,
            use_residual_attention,
            residual_attention_gain_init,
            memory_kv_slots,
            use_relative_bias,
            relative_bias_num_buckets,
            relative_bias_max_distance,
            attention_variant=attention_variant,
            retention_decay_init=retention_decay_init,
            retention_output_gate=retention_output_gate,
        )
        use_moe = moe_every_n_layers > 0 and ((layer_idx + 1) % moe_every_n_layers == 0)
        moe_inner_dim = moe_hidden if moe_hidden > 0 else (mlp_hidden if mlp_hidden > 0 else mlp_mult * dim)
        self.mlp = (
            MoEMLP(dim, moe_num_experts, moe_top_k, hidden_dim=moe_inner_dim) if use_moe else MLP(dim, mlp_mult, hidden_dim=mlp_hidden)
        )
        self.pre_mlp = MLP(dim, mlp_mult, hidden_dim=mlp_hidden) if macaron_layout else None
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.pre_mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, prev_attn_logits: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if self.pre_mlp is not None:
            pre_mlp_in = apply_token_shift(x) if self.token_shift else x
            x = x + 0.5 * self.pre_mlp_scale.to(dtype=x.dtype)[None, None, :] * self.pre_mlp(self.pre_mlp_norm(pre_mlp_in))
        x_in = apply_token_shift(x) if self.token_shift else x
        attn_out, next_attn_logits = self.attn(self.attn_norm(x_in), prev_attn_logits=prev_attn_logits)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_in = apply_token_shift(x) if self.token_shift else x
        mlp_scale = 0.5 if self.pre_mlp is not None else 1.0
        x = x + mlp_scale * self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(mlp_in))
        return x, next_attn_logits


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        mlp_hidden: int,
        num_register_tokens: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        overtone_embed_init: bool,
        overtone_embed_power: float,
        overtone_embed_scale: float,
        resid_mix_phase_init: bool,
        resid_mix_phase_sharpness: float,
        use_alibi: bool,
        token_shift: bool,
        use_xpos: bool,
        xpos_scale_base: float,
        num_shared_layers: int,
        shared_layer_repeats: int,
        shared_loop_gate_init: float,
        use_residual_attention: bool,
        residual_attention_gain_init: float,
        memory_kv_slots: int,
        use_relative_bias: bool,
        relative_bias_num_buckets: int,
        relative_bias_max_distance: int,
        macaron_layout: bool,
        retention_layers: int,
        retention_decay_init: float,
        retention_output_gate: bool,
        moe_every_n_layers: int,
        moe_num_experts: int,
        moe_top_k: int,
        moe_hidden: int,
        use_hyper_connections: bool,
        hyper_connection_layers: int,
        hyper_connection_gate_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.overtone_embed_init = overtone_embed_init
        self.overtone_embed_power = overtone_embed_power
        self.overtone_embed_scale = overtone_embed_scale
        self.resid_mix_phase_init = resid_mix_phase_init
        self.resid_mix_phase_sharpness = resid_mix_phase_sharpness
        self.use_alibi = use_alibi
        self.token_shift = token_shift
        self.use_xpos = use_xpos
        self.xpos_scale_base = xpos_scale_base
        self.use_residual_attention = use_residual_attention
        self.residual_attention_gain_init = residual_attention_gain_init
        self.memory_kv_slots = memory_kv_slots
        self.use_relative_bias = use_relative_bias
        self.relative_bias_num_buckets = relative_bias_num_buckets
        self.relative_bias_max_distance = relative_bias_max_distance
        self.macaron_layout = macaron_layout
        self.retention_layers = max(int(retention_layers), 0)
        self.retention_decay_init = retention_decay_init
        self.retention_output_gate = retention_output_gate
        self.moe_every_n_layers = max(int(moe_every_n_layers), 0)
        self.moe_num_experts = max(int(moe_num_experts), 1)
        self.moe_top_k = max(int(moe_top_k), 1)
        self.moe_hidden = max(int(moe_hidden), 0)
        self.use_hyper_connections = use_hyper_connections
        self.hyper_connection_layers = max(int(hyper_connection_layers), 0)
        self.hyper_connection_gate_init = hyper_connection_gate_init
        self.fake_quant_bits = 0
        self.num_register_tokens = max(int(num_register_tokens), 0)
        self.num_shared_layers = max(int(num_shared_layers), 0)
        self.shared_layer_repeats = max(int(shared_layer_repeats), 0)
        self.shared_loop_gate_init = shared_loop_gate_init
        retention_start = max(num_layers - self.retention_layers, 0)
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.register_tokens = (
            nn.Parameter(torch.zeros(self.num_register_tokens, model_dim, dtype=torch.float32))
            if self.num_register_tokens > 0
            else None
        )
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    i,
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    mlp_hidden,
                    rope_base,
                    qk_gain_init,
                    use_alibi,
                    token_shift,
                    use_xpos,
                    xpos_scale_base,
                    use_residual_attention,
                    residual_attention_gain_init,
                    memory_kv_slots,
                    use_relative_bias,
                    relative_bias_num_buckets,
                    relative_bias_max_distance,
                    macaron_layout,
                    attention_variant="retention" if i >= max(num_layers - self.retention_layers, 0) else "attention",
                    retention_decay_init=retention_decay_init,
                    retention_output_gate=retention_output_gate,
                    moe_every_n_layers=self.moe_every_n_layers,
                    moe_num_experts=self.moe_num_experts,
                    moe_top_k=self.moe_top_k,
                    moe_hidden=self.moe_hidden,
                )
                for i in range(num_layers)
            ]
        )
        self.shared_blocks = nn.ModuleList(
            [
                Block(
                    num_layers + i,
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    mlp_hidden,
                    rope_base,
                    qk_gain_init,
                    use_alibi,
                    token_shift,
                    use_xpos,
                    xpos_scale_base,
                    use_residual_attention,
                    residual_attention_gain_init,
                    memory_kv_slots,
                    use_relative_bias,
                    relative_bias_num_buckets,
                    relative_bias_max_distance,
                    macaron_layout,
                    attention_variant="attention",
                    retention_decay_init=retention_decay_init,
                    retention_output_gate=retention_output_gate,
                    moe_every_n_layers=self.moe_every_n_layers,
                    moe_num_experts=self.moe_num_experts,
                    moe_top_k=self.moe_top_k,
                    moe_hidden=self.moe_hidden,
                )
                for i in range(self.num_shared_layers)
            ]
        )
        self.shared_loop_gates = (
            nn.Parameter(torch.full((self.shared_layer_repeats, self.num_shared_layers, model_dim), shared_loop_gate_init))
            if self.num_shared_layers > 0 and self.shared_layer_repeats > 0
            else None
        )
        self.hyper_connection_gates = (
            nn.Parameter(
                torch.full((max(self.hyper_connection_layers, self.num_shared_layers), 2, model_dim), hyper_connection_gate_init, dtype=torch.float32)
            )
            if self.use_hyper_connections and max(self.hyper_connection_layers, self.num_shared_layers) > 0
            else None
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            if self.overtone_embed_init:
                self._init_overtone_embedding()
            else:
                nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
        if self.resid_mix_phase_init:
            self._init_resid_mix_phase_schedule()

    def _init_overtone_embedding(self) -> None:
        rows, cols = self.tok_emb.weight.shape
        with torch.no_grad():
            basis = torch.randn((rows, cols), dtype=torch.float32, device=self.tok_emb.weight.device)
            q, _ = torch.linalg.qr(basis, mode="reduced")
            ranks = torch.arange(1, cols + 1, dtype=torch.float32, device=q.device)
            spectrum = ranks.pow(-self.overtone_embed_power)
            spectrum = spectrum / spectrum.norm() * (cols**0.5) * self.tied_embed_init_std * self.overtone_embed_scale
            self.tok_emb.weight.copy_(q * spectrum.unsqueeze(0))

    def _init_resid_mix_phase_schedule(self) -> None:
        total = max(len(self.blocks) - 1, 1)
        with torch.no_grad():
            for idx, block in enumerate(self.blocks):
                depth = idx / total
                alpha = torch.sigmoid(
                    torch.tensor((depth - 0.5) * self.resid_mix_phase_sharpness, dtype=torch.float32)
                ).item()
                block.resid_mix[0].fill_(alpha)
                block.resid_mix[1].fill_(1.0 - alpha)

    def _prepend_register_tokens(self, x: Tensor) -> Tensor:
        if self.register_tokens is None:
            return x
        regs = self.register_tokens.to(dtype=x.dtype)[None, :, :].expand(x.size(0), -1, -1)
        return torch.cat((regs, x), dim=1)

    def set_fake_quant(self, bits: int) -> None:
        self.fake_quant_bits = max(int(bits), 0)
        for module in self.modules():
            if isinstance(module, CastedLinear):
                module.fake_quant_bits = self.fake_quant_bits

    def _embedding_weight(self) -> Tensor:
        weight = self.tok_emb.weight
        if self.fake_quant_bits > 0:
            weight = fake_quantize_tensor(weight, self.fake_quant_bits)
        return weight

    def _moe_aux_loss(self) -> Tensor:
        aux = None
        for module in self.modules():
            if isinstance(module, MoEMLP):
                value = module.last_aux_loss
                aux = value if aux is None else aux + value
        if aux is None:
            return self.tok_emb.weight.new_zeros(())
        return aux

    def _loss_from_logits(
        self,
        logits: Tensor,
        target_ids: Tensor,
        label_smoothing: float = 0.0,
        z_loss_coeff: float = 0.0,
        logit_var_loss_coeff: float = 0.0,
        moe_aux_loss_coeff: float = 0.0,
        teacher_logits: Tensor | None = None,
        distill_weight: float = 0.0,
        distill_temperature: float = 1.0,
    ) -> Tensor:
        flat_logits = logits.reshape(-1, logits.size(-1))
        targets = target_ids.reshape(-1)
        loss = F.cross_entropy(flat_logits.float(), targets, reduction="mean", label_smoothing=label_smoothing)
        if z_loss_coeff > 0.0:
            z_loss = torch.logsumexp(flat_logits.float(), dim=-1).square().mean()
            loss = loss + z_loss_coeff * z_loss
        if logit_var_loss_coeff > 0.0:
            # Keep per-token logit spread from exploding in short runs, which tends
            # to help the later quantized softcapped head behave more smoothly.
            logit_std = flat_logits.float().std(dim=-1)
            target_std = float(self.logit_softcap) / 4.0
            loss = loss + logit_var_loss_coeff * (logit_std - target_std).square().mean()
        if moe_aux_loss_coeff > 0.0:
            loss = loss + moe_aux_loss_coeff * self._moe_aux_loss()
        if teacher_logits is not None and distill_weight > 0.0:
            temp = max(distill_temperature, 1e-4)
            student_log_probs = F.log_softmax(flat_logits.float() / temp, dim=-1)
            teacher_probs = F.softmax(teacher_logits.reshape(-1, teacher_logits.size(-1)).float() / temp, dim=-1)
            kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            loss = loss + distill_weight * (temp * temp) * kl
        return loss

    def _apply_shared_tail(self, x: Tensor, x0: Tensor) -> Tensor:
        if self.shared_loop_gates is None:
            return x
        prev_attn_logits = None
        hyper_active = self.hyper_connection_gates is not None
        stream_aux = x
        for repeat_idx in range(self.shared_layer_repeats):
            for layer_idx, block in enumerate(self.shared_blocks):
                gate = torch.sigmoid(self.shared_loop_gates[repeat_idx, layer_idx]).to(dtype=x.dtype)[None, None, :]
                if hyper_active and layer_idx < self.hyper_connection_gates.size(0):
                    mix = torch.sigmoid(self.hyper_connection_gates[layer_idx]).to(dtype=x.dtype)
                    mixed = mix[0][None, None, :] * x + (1.0 - mix[0][None, None, :]) * stream_aux
                    block_out, prev_attn_logits = block(mixed, x0, prev_attn_logits=prev_attn_logits)
                    stream_aux = (1.0 - mix[1][None, None, :]) * stream_aux + mix[1][None, None, :] * block_out
                    x = (1.0 - gate) * x + gate * (0.5 * (block_out + stream_aux))
                else:
                    block_out, prev_attn_logits = block(x, x0, prev_attn_logits=prev_attn_logits)
                    x = (1.0 - gate) * x + gate * block_out
        return x

    def _forward_features(self, input_ids: Tensor) -> Tensor:
        x = F.embedding(input_ids, self._embedding_weight())
        x = self._prepend_register_tokens(x)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        prev_attn_logits = None

        for i in range(self.num_encoder_layers):
            x, prev_attn_logits = self.blocks[i](x, x0, prev_attn_logits=prev_attn_logits)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x, prev_attn_logits = self.blocks[self.num_encoder_layers + i](x, x0, prev_attn_logits=prev_attn_logits)
        x = self._apply_shared_tail(x, x0)

        x = self.final_norm(x)
        if self.num_register_tokens > 0:
            x = x[:, self.num_register_tokens :, :]
        return x

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        label_smoothing: float = 0.0,
        z_loss_coeff: float = 0.0,
        logit_var_loss_coeff: float = 0.0,
        moe_aux_loss_coeff: float = 0.0,
    ) -> Tensor:
        features = self._forward_features(input_ids)
        if self.tie_embeddings:
            logits_proj = F.linear(features, self._embedding_weight())
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(features)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return self._loss_from_logits(
            logits,
            target_ids,
            label_smoothing=label_smoothing,
            z_loss_coeff=z_loss_coeff,
            logit_var_loss_coeff=logit_var_loss_coeff,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
        )

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self._forward_features(input_ids)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self._embedding_weight())
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

def init_ema_state(module: nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().clone().float() for name, tensor in module.state_dict().items()}


@torch.no_grad()
def update_ema_state(ema_state: dict[str, Tensor], module: nn.Module, decay: float) -> None:
    one_minus_decay = 1.0 - decay
    for name, tensor in module.state_dict().items():
        ema_state[name].mul_(decay).add_(tensor.detach().float(), alpha=one_minus_decay)


def maybe_build_teacher(args: Hyperparameters, device: torch.device) -> nn.Module | None:
    if not args.distill_teacher_checkpoint or args.distill_weight <= 0.0:
        return None
    teacher = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        mlp_hidden=args.mlp_hidden,
        num_register_tokens=args.num_register_tokens,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        overtone_embed_init=args.overtone_embed_init,
        overtone_embed_power=args.overtone_embed_power,
        overtone_embed_scale=args.overtone_embed_scale,
        resid_mix_phase_init=args.resid_mix_phase_init,
        resid_mix_phase_sharpness=args.resid_mix_phase_sharpness,
        use_alibi=args.use_alibi,
        token_shift=args.token_shift,
        use_xpos=args.use_xpos,
        xpos_scale_base=args.xpos_scale_base,
        num_shared_layers=args.num_shared_layers,
        shared_layer_repeats=args.shared_layer_repeats,
        shared_loop_gate_init=args.shared_loop_gate_init,
        use_residual_attention=args.use_residual_attention,
        residual_attention_gain_init=args.residual_attention_gain_init,
        memory_kv_slots=args.memory_kv_slots,
        use_relative_bias=args.use_relative_bias,
        relative_bias_num_buckets=args.relative_bias_num_buckets,
        relative_bias_max_distance=args.relative_bias_max_distance,
        macaron_layout=args.macaron_layout,
        retention_layers=args.retention_layers,
        retention_decay_init=args.retention_decay_init,
        retention_output_gate=args.retention_output_gate,
        moe_every_n_layers=args.moe_every_n_layers,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_hidden=args.moe_hidden,
        use_hyper_connections=args.use_hyper_connections,
        hyper_connection_layers=args.hyper_connection_layers,
        hyper_connection_gate_init=args.hyper_connection_gate_init,
    ).to(device).bfloat16()
    state = torch.load(args.distill_teacher_checkpoint, map_location="cpu")
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def fake_quant_tail_active(args: Hyperparameters, step: int, elapsed_ms: float) -> bool:
    if args.fake_quant_tail_steps <= 0 or args.fake_quant_bits <= 0:
        return False
    if step >= max(args.iterations - args.fake_quant_tail_steps, 0):
        return True
    if args.max_wallclock_seconds > 0 and step > 0:
        step_ms = elapsed_ms / step
        remaining_ms = 1000.0 * args.max_wallclock_seconds - elapsed_ms
        return remaining_ms <= args.fake_quant_tail_steps * step_ms
    return False


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    if args.enable_torch_compile:
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if args.train_batch_tokens % (world_size * grad_accum_steps * args.train_seq_len) != 0:
        raise ValueError(
            "TRAIN_BATCH_TOKENS must be divisible by WORLD_SIZE * GRAD_ACCUM_STEPS * TRAIN_SEQ_LEN; "
            f"got TRAIN_BATCH_TOKENS={args.train_batch_tokens}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    run_dir = os.path.join("runs", args.run_id)
    run_manifest_path = os.path.join(run_dir, "manifest.json")
    run_metrics_path = os.path.join(run_dir, "metrics.jsonl")
    run_summary_path = os.path.join(run_dir, "summary.json")
    run_ledger_path = os.path.join("runs", "ledger.jsonl")
    raw_model_path = os.path.join(run_dir, "final_model.pt")
    quant_model_path = os.path.join(run_dir, "final_model.int8.ptz")
    logfile = None
    manifest_payload = experiment_metadata(args)
    if master_process:
        os.makedirs("runs", exist_ok=True)
        os.makedirs(run_dir, exist_ok=True)
        logfile = os.path.join(run_dir, "train.log")
        print(logfile)
        manifest_payload.update(
            {
                "schema_version": 1,
                "status": "started",
                "run_id": args.run_id,
                "started_at_utc": utc_now_iso(),
                "paths": {
                    "run_dir": run_dir,
                    "log": logfile,
                    "metrics_jsonl": run_metrics_path,
                    "summary_json": run_summary_path,
                    "raw_model": raw_model_path,
                    "quant_model": quant_model_path,
                },
            }
        )
        write_json(run_manifest_path, manifest_payload)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    def metric0(event: str, **payload: object) -> None:
        if not master_process:
            return
        append_jsonl(
            run_metrics_path,
            {
                "event": event,
                "run_id": args.run_id,
                "timestamp_utc": utc_now_iso(),
                **payload,
            },
        )

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"experiment_title:{manifest_payload['title'] or 'unspecified'}")
    log0(f"experiment_baseline_run_id:{manifest_payload['baseline_run_id'] or 'unspecified'}")
    log0(f"experiment_parent_run_id:{manifest_payload['parent_run_id'] or 'unspecified'}")
    log0(f"experiment_hypothesis:{manifest_payload['hypothesis'] or 'unspecified'}")
    log0(f"experiment_expected_effect:{manifest_payload['expected_effect'] or 'unspecified'}")
    for idx, change in enumerate(manifest_payload["explicit_changes"], start=1):
        log0(f"experiment_change_{idx}:{change}")
    if not manifest_payload["explicit_changes"]:
        log0("WARNING: experiment_changes:unspecified")
    metric0(
        "run_start",
        title=manifest_payload["title"],
        baseline_run_id=manifest_payload["baseline_run_id"],
        parent_run_id=manifest_payload["parent_run_id"],
        hypothesis=manifest_payload["hypothesis"],
        expected_effect=manifest_payload["expected_effect"],
        explicit_changes=manifest_payload["explicit_changes"],
        auto_env_overrides=manifest_payload["auto_env_overrides"],
    )
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
        args.data_path,
        args.tokenizer_path,
    )
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    if expected_train_files is None:
        log0(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}")
    elif actual_train_files < expected_train_files:
        log0(
            f"WARNING: train_loader:subset dataset:{dataset_name} "
            f"train_shards:{actual_train_files}/{expected_train_files} "
            f"new epochs will arrive sooner than the full dataset"
        )
    else:
        log0(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}/{expected_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    if master_process:
        manifest_payload["dataset"] = {
            "name": dataset_name,
            "actual_train_files": actual_train_files,
            "expected_train_files": expected_train_files,
            "tokenizer_path": args.tokenizer_path,
            "val_pattern": args.val_files,
            "val_tokens": int(val_tokens.numel() - 1),
        }
        write_json(run_manifest_path, manifest_payload)

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        mlp_hidden=args.mlp_hidden,
        num_register_tokens=args.num_register_tokens,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        overtone_embed_init=args.overtone_embed_init,
        overtone_embed_power=args.overtone_embed_power,
        overtone_embed_scale=args.overtone_embed_scale,
        resid_mix_phase_init=args.resid_mix_phase_init,
        resid_mix_phase_sharpness=args.resid_mix_phase_sharpness,
        use_alibi=args.use_alibi,
        token_shift=args.token_shift,
        use_xpos=args.use_xpos,
        xpos_scale_base=args.xpos_scale_base,
        num_shared_layers=args.num_shared_layers,
        shared_layer_repeats=args.shared_layer_repeats,
        shared_loop_gate_init=args.shared_loop_gate_init,
        use_residual_attention=args.use_residual_attention,
        residual_attention_gain_init=args.residual_attention_gain_init,
        memory_kv_slots=args.memory_kv_slots,
        use_relative_bias=args.use_relative_bias,
        relative_bias_num_buckets=args.relative_bias_num_buckets,
        relative_bias_max_distance=args.relative_bias_max_distance,
        macaron_layout=args.macaron_layout,
        retention_layers=args.retention_layers,
        retention_decay_init=args.retention_decay_init,
        retention_output_gate=args.retention_output_gate,
        moe_every_n_layers=args.moe_every_n_layers,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_hidden=args.moe_hidden,
        use_hyper_connections=args.use_hyper_connections,
        hyper_connection_layers=args.hyper_connection_layers,
        hyper_connection_gate_init=args.hyper_connection_gate_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = (
        torch.compile(base_model, dynamic=False, fullgraph=True)
        if args.enable_torch_compile
        else base_model
    )
    model: nn.Module = (
        DDP(
            compiled_model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        if distributed
        else compiled_model
    )
    ema_state = init_ema_state(base_model) if args.ema_decay > 0.0 else None
    teacher_model = maybe_build_teacher(args, device)
    if distributed and teacher_model is not None:
        if master_process:
            log0("WARNING: distillation disabled on distributed path; current implementation is local-only.")
        teacher_model = None

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    embed_params = [base_model.tok_emb.weight]
    if base_model.register_tokens is not None:
        embed_params.append(base_model.register_tokens)
    block_named_params = (
        [(f"blocks.{name}", p) for name, p in base_model.blocks.named_parameters()]
        + [(f"shared_blocks.{name}", p) for name, p in base_model.shared_blocks.named_parameters()]
    )
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if base_model.shared_loop_gates is not None:
        scalar_params.append(base_model.shared_loop_gates)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": embed_params, "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_weight_decay,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"torch_compile:{args.enable_torch_compile}")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"eval_mode:{'sliding' if 0 < args.eval_stride < args.train_seq_len else 'standard'} "
        f"eval_stride:{args.eval_stride} eval_batch_seqs:{args.eval_batch_seqs}"
    )
    log0(f"val_max_seqs:{args.val_max_seqs} final_eval_max_seqs:{args.final_eval_max_seqs}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"training_preset:{args.training_preset or 'none'}")
    log0(f"train_random_offset_tokens:{args.train_random_offset_tokens}")
    log0(f"num_register_tokens:{args.num_register_tokens}")
    if args.mlp_hidden > 0:
        log0(f"mlp_hidden_override:{args.mlp_hidden}")
    log0(
        f"regularization:label_smoothing:{args.label_smoothing} z_loss_coeff:{args.z_loss_coeff} "
        f"row_norm_loss_coeff:{args.row_norm_loss_coeff} "
        f"outlier_loss_coeff:{args.outlier_loss_coeff} outlier_loss_threshold:{args.outlier_loss_threshold} "
        f"logit_var_loss_coeff:{args.logit_var_loss_coeff} "
        f"ema_decay:{args.ema_decay} ema_start_step:{args.ema_start_step} "
        f"force_fp16_tied_embed_export:{args.force_fp16_tied_embed_export}"
    )
    log0(
        f"advanced_init:overtone_embed_init:{args.overtone_embed_init} "
        f"overtone_embed_power:{args.overtone_embed_power} "
        f"overtone_embed_scale:{args.overtone_embed_scale} "
        f"resid_mix_phase_init:{args.resid_mix_phase_init} "
        f"resid_mix_phase_sharpness:{args.resid_mix_phase_sharpness}"
    )
    log0(f"cheap_biases:use_alibi:{args.use_alibi} token_shift:{args.token_shift}")
    log0(
        f"context_ext:use_xpos:{args.use_xpos} xpos_scale_base:{args.xpos_scale_base} "
        f"shared_layers:{args.num_shared_layers} shared_repeats:{args.shared_layer_repeats} "
        f"shared_loop_gate_init:{args.shared_loop_gate_init}"
    )
    log0(
        f"attn_ext:residual_attention:{args.use_residual_attention} "
        f"residual_attention_gain_init:{args.residual_attention_gain_init} "
        f"memory_kv_slots:{args.memory_kv_slots} macaron_layout:{args.macaron_layout} "
        f"use_relative_bias:{args.use_relative_bias} "
        f"relative_bias_num_buckets:{args.relative_bias_num_buckets} "
        f"relative_bias_max_distance:{args.relative_bias_max_distance} "
        f"retention_layers:{args.retention_layers} retention_decay_init:{args.retention_decay_init} "
        f"retention_output_gate:{args.retention_output_gate}"
    )
    log0(
        f"capacity_ext:moe_every_n_layers:{args.moe_every_n_layers} "
        f"moe_num_experts:{args.moe_num_experts} moe_top_k:{args.moe_top_k} "
        f"moe_hidden:{args.moe_hidden} moe_aux_loss_coeff:{args.moe_aux_loss_coeff}"
    )
    log0(
        f"shared_tail_ext:use_hyper_connections:{args.use_hyper_connections} "
        f"hyper_connection_layers:{args.hyper_connection_layers} "
        f"hyper_connection_gate_init:{args.hyper_connection_gate_init}"
    )
    log0(
        f"distill:weight:{args.distill_weight} temperature:{args.distill_temperature} "
        f"teacher_checkpoint:{args.distill_teacher_checkpoint or 'none'}"
    )
    log0(f"fake_quant_tail:steps:{args.fake_quant_tail_steps} bits:{args.fake_quant_bits}")
    log0(
        f"quant_auto_keep:budget_bytes:{args.int8_auto_keep_budget_bytes} "
        f"max_tensors:{args.int8_auto_keep_max_tensors} "
        f"min_numel:{args.int8_auto_keep_min_numel} "
        f"log_topk:{args.int8_sensitivity_log_topk} "
        f"row_budget_bytes:{args.int8_auto_keep_row_budget_bytes} "
        f"row_group_size:{args.int8_auto_keep_row_group_size}"
    )
    log0(f"muon_weight_decay:{args.muon_weight_decay}")
    log0(f"seed:{args.seed}")
    metric0(
        "config",
        model_params=n_params,
        world_size=world_size,
        grad_accum_steps=grad_accum_steps,
        train_seq_len=args.train_seq_len,
        train_batch_tokens=args.train_batch_tokens,
        iterations=args.iterations,
        training_preset=args.training_preset,
        torch_compile=args.enable_torch_compile,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_register_tokens=args.num_register_tokens,
        val_max_seqs=args.val_max_seqs,
        final_eval_max_seqs=args.final_eval_max_seqs,
        int8_auto_keep_budget_bytes=args.int8_auto_keep_budget_bytes,
        int8_auto_keep_max_tensors=args.int8_auto_keep_max_tensors,
        train_random_offset_tokens=args.train_random_offset_tokens,
        row_norm_loss_coeff=args.row_norm_loss_coeff,
        outlier_loss_coeff=args.outlier_loss_coeff,
        outlier_loss_threshold=args.outlier_loss_threshold,
        logit_var_loss_coeff=args.logit_var_loss_coeff,
        muon_weight_decay=args.muon_weight_decay,
        overtone_embed_init=args.overtone_embed_init,
        resid_mix_phase_init=args.resid_mix_phase_init,
        use_alibi=args.use_alibi,
        token_shift=args.token_shift,
        use_xpos=args.use_xpos,
        num_shared_layers=args.num_shared_layers,
        shared_layer_repeats=args.shared_layer_repeats,
        int8_auto_keep_row_budget_bytes=args.int8_auto_keep_row_budget_bytes,
        use_residual_attention=args.use_residual_attention,
        memory_kv_slots=args.memory_kv_slots,
        macaron_layout=args.macaron_layout,
        use_relative_bias=args.use_relative_bias,
        retention_layers=args.retention_layers,
        moe_every_n_layers=args.moe_every_n_layers,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_aux_loss_coeff=args.moe_aux_loss_coeff,
        use_hyper_connections=args.use_hyper_connections,
        hyper_connection_layers=args.hyper_connection_layers,
        distill_weight=args.distill_weight,
        fake_quant_tail_steps=args.fake_quant_tail_steps,
    )

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(
        args.train_files,
        rank,
        world_size,
        device,
        random_offset_tokens=args.train_random_offset_tokens,
        seed=args.seed,
    )

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        base_model.set_fake_quant(0)
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(
                        x,
                        y,
                        args.label_smoothing,
                        args.z_loss_coeff,
                        args.logit_var_loss_coeff,
                        args.moe_aux_loss_coeff,
                    )
                    if args.row_norm_loss_coeff > 0.0:
                        warmup_loss = warmup_loss + args.row_norm_loss_coeff * row_norm_regularization(base_model)
                    if args.outlier_loss_coeff > 0.0:
                        warmup_loss = warmup_loss + args.outlier_loss_coeff * outlier_regularization(
                            base_model, args.outlier_loss_threshold
                        )
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
                metric0("warmup", step=warmup_step + 1, warmup_steps=args.warmup_steps)
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        base_model.set_fake_quant(0)
        train_loader = DistributedTokenLoader(
            args.train_files,
            rank,
            world_size,
            device,
            random_offset_tokens=args.train_random_offset_tokens,
            seed=args.seed,
        )

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                max_seqs=args.val_max_seqs,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            metric0(
                "val",
                step=step,
                iterations=args.iterations,
                val_loss=val_loss,
                val_bpb=val_bpb,
                train_time_ms=training_time_ms,
                step_avg_ms=training_time_ms / max(step, 1),
                max_seqs=args.val_max_seqs,
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        base_model.set_fake_quant(args.fake_quant_bits if fake_quant_tail_active(args, step, elapsed_ms) else 0)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                teacher_logits = None
                if teacher_model is not None:
                    with torch.inference_mode():
                        teacher_logits = teacher_model.forward_logits(x)
                if teacher_logits is None:
                    loss = model(
                        x,
                        y,
                        args.label_smoothing,
                        args.z_loss_coeff,
                        args.logit_var_loss_coeff,
                        args.moe_aux_loss_coeff,
                    )
                else:
                    student_logits = base_model.forward_logits(x)
                    loss = base_model._loss_from_logits(
                        student_logits,
                        y,
                        label_smoothing=args.label_smoothing,
                        z_loss_coeff=args.z_loss_coeff,
                        logit_var_loss_coeff=args.logit_var_loss_coeff,
                        moe_aux_loss_coeff=args.moe_aux_loss_coeff,
                        teacher_logits=teacher_logits,
                        distill_weight=args.distill_weight,
                        distill_temperature=args.distill_temperature,
                    )
                if args.row_norm_loss_coeff > 0.0:
                    loss = loss + args.row_norm_loss_coeff * row_norm_regularization(base_model)
                if args.outlier_loss_coeff > 0.0:
                    loss = loss + args.outlier_loss_coeff * outlier_regularization(
                        base_model, args.outlier_loss_threshold
                    )
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        if ema_state is not None and step + 1 >= args.ema_start_step:
            update_ema_state(ema_state, base_model, args.ema_decay)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
            metric0(
                "train",
                step=step,
                iterations=args.iterations,
                train_loss=float(train_loss.item()),
                train_time_ms=approx_training_time_ms,
                step_avg_ms=approx_training_time_ms / step,
                lr_scale=scale,
                muon_momentum=muon_momentum,
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    peak_allocated_mib = int(torch.cuda.max_memory_allocated() // 1024 // 1024)
    peak_reserved_mib = int(torch.cuda.max_memory_reserved() // 1024 // 1024)
    if ema_state is not None:
        base_model.load_state_dict(ema_state, strict=True)
        log0(f"final_weights_source:ema decay:{args.ema_decay} start_step:{args.ema_start_step}")
        metric0("ema_export", decay=args.ema_decay, start_step=args.ema_start_step)

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), raw_model_path)
        model_bytes = os.path.getsize(raw_model_path)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats, auto_keep_info = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    submission_over_limit = False
    raw_model_bytes = None
    quant_file_bytes = None
    total_submission_bytes = None
    if master_process:
        with open(quant_model_path, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(quant_model_path)
        code_bytes = len(code.encode("utf-8"))
        total_submission_bytes = quant_file_bytes + code_bytes
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        if auto_keep_info["enabled"]:
            log0(
                f"auto_keep:selected_count:{auto_keep_info['selected_count']} "
                f"selected_extra_bytes:{auto_keep_info['selected_extra_bytes']}"
            )
            for idx, candidate in enumerate(auto_keep_info["top_candidates"], start=1):
                log0(
                    "auto_keep_top_"
                    f"{idx}:name:{candidate['name']} rel_mse:{candidate['rel_mse']:.6e} "
                    f"extra_bytes:{candidate['extra_bytes']} "
                    f"benefit_per_extra_byte:{candidate['benefit_per_extra_byte']:.6e}"
                )
        log0(f"Total submission size int8+zlib: {total_submission_bytes} bytes")
        log0(f"Submission size limit: {args.submission_size_limit_bytes} bytes")
        submission_over_limit = total_submission_bytes > args.submission_size_limit_bytes
        metric0(
            "artifact",
            raw_model_bytes=model_bytes,
            quant_zlib_bytes=quant_file_bytes,
            quant_raw_bytes=quant_raw_bytes,
            code_bytes=code_bytes,
            total_submission_bytes=total_submission_bytes,
            submission_size_limit_bytes=args.submission_size_limit_bytes,
            over_limit=submission_over_limit,
            quant_stats={key: int(value) for key, value in quant_stats.items()},
            auto_keep_info=auto_keep_info,
        )
        raw_model_bytes = model_bytes

    if distributed:
        over_limit_tensor = torch.tensor(int(submission_over_limit), device=device)
        dist.all_reduce(over_limit_tensor, op=dist.ReduceOp.MAX)
        submission_over_limit = bool(over_limit_tensor.item())
    if submission_over_limit:
        raise RuntimeError(
            "Final int8+zlib artifact exceeds SUBMISSION_SIZE_LIMIT_BYTES; "
            "adjust model/code size or quantization settings."
        )

    if distributed:
        dist.barrier()
    with open(quant_model_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    final_eval_is_exact = True
    if 0 < args.eval_stride < args.train_seq_len:
        final_eval_is_exact = args.final_eval_max_seqs <= 0
        log0(
            f"final_eval_mode:sliding_window stride:{args.eval_stride} "
            f"batch_seqs:{args.eval_batch_seqs} max_seqs:{args.final_eval_max_seqs}"
        )
        q_val_loss, q_val_bpb = eval_val_sliding(
            args,
            base_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=args.eval_stride,
            batch_seqs=args.eval_batch_seqs,
            max_seqs=args.final_eval_max_seqs,
        )
    else:
        final_eval_is_exact = args.final_eval_max_seqs <= 0
        log0(f"final_eval_mode:standard max_seqs:{args.final_eval_max_seqs}")
        q_val_loss, q_val_bpb = eval_val(
            args,
            model,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            max_seqs=args.final_eval_max_seqs,
        )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    if final_eval_is_exact:
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    else:
        log0(f"final_int8_zlib_roundtrip_proxy val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    final_eval_ms = 1000.0 * (time.perf_counter() - t_qeval)
    metric0(
        "final_eval",
        val_loss=q_val_loss,
        val_bpb=q_val_bpb,
        eval_time_ms=final_eval_ms,
        exact=final_eval_is_exact,
        eval_mode="sliding_window" if 0 < args.eval_stride < args.train_seq_len else "standard",
    )

    if master_process:
        summary_payload = {
            "schema_version": 1,
            "status": "completed",
            "run_id": args.run_id,
            "completed_at_utc": utc_now_iso(),
            "title": manifest_payload["title"],
            "baseline_run_id": manifest_payload["baseline_run_id"],
            "parent_run_id": manifest_payload["parent_run_id"],
            "hypothesis": manifest_payload["hypothesis"],
            "expected_effect": manifest_payload["expected_effect"],
            "explicit_changes": manifest_payload["explicit_changes"],
            "tags": manifest_payload["tags"],
            "steps_completed": step,
            "target_iterations": args.iterations,
            "stopped_early": bool(stop_after_step is not None and step < args.iterations),
            "training_time_ms": training_time_ms,
            "peak_memory_allocated_mib": peak_allocated_mib,
            "peak_memory_reserved_mib": peak_reserved_mib,
            "artifact": {
                "raw_model_bytes": raw_model_bytes,
                "quant_zlib_bytes": quant_file_bytes,
                "quant_raw_bytes": quant_raw_bytes,
                "code_bytes": len(code.encode("utf-8")),
                "total_submission_bytes": total_submission_bytes,
                "submission_size_limit_bytes": args.submission_size_limit_bytes,
            },
            "quant_stats": {key: int(value) for key, value in quant_stats.items()},
            "auto_keep_info": auto_keep_info,
            "final_eval": {
                "mode": "sliding_window" if 0 < args.eval_stride < args.train_seq_len else "standard",
                "exact": final_eval_is_exact,
                "val_loss": q_val_loss,
                "val_bpb": q_val_bpb,
                "eval_time_ms": final_eval_ms,
            },
            "config": hyperparameters_to_dict(args),
        }
        write_json(run_summary_path, summary_payload)
        manifest_payload["status"] = "completed"
        manifest_payload["completed_at_utc"] = summary_payload["completed_at_utc"]
        write_json(run_manifest_path, manifest_payload)
        append_jsonl(run_ledger_path, summary_payload)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
