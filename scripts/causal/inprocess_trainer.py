"""In-process training runner for MLX compilation reuse.

Runs train_gpt_mlx.py training loop in-process, avoiding subprocess startup
and MLX warmup repetition. First run still needs warmup; subsequent same-
architecture runs benefit from cached Metal shaders.

Usage:
    from scripts.causal.inprocess_trainer import (
        create_shared_context,
        train_single_run,
        run_condition_inprocess,
    )

    ctx = create_shared_context(data_path, tokenizer_path, vocab_size, train_seq_len)
    result = train_single_run(ctx, env_overrides, seed=42, iterations=100)
"""
from __future__ import annotations

import gc
import os
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Ensure repo root is on sys.path for train_gpt_mlx import
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import train_gpt_mlx as tgm

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

# Track which architecture shapes have been warmed up in this process.
# Key: tuple of shape-affecting hyperparameters. Value: True.
_WARMED_ARCHITECTURES: set[tuple] = set()


def _arch_key(args) -> tuple:
    """Extract architecture-affecting params that determine Metal shader shapes."""
    return (
        args.vocab_size, args.num_layers, args.model_dim, args.num_heads,
        args.num_kv_heads, args.mlp_mult, args.train_seq_len,
        args.mlx_max_microbatch_tokens, args.grad_accum_steps,
    )


# ---------------------------------------------------------------------------
# Environment override context manager
# ---------------------------------------------------------------------------

@contextmanager
def _env_override(overrides: dict[str, str]):
    """Temporarily set os.environ keys, restore original values on exit."""
    old: dict[str, str | None] = {}
    for k in overrides:
        old[k] = os.environ.get(k)
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Shared context (loaded once per pipeline run)
# ---------------------------------------------------------------------------

@dataclass
class SharedTrainingContext:
    """Immutable resources loaded once and shared across training runs."""
    val_tokens: np.ndarray
    base_bytes_lut: np.ndarray
    has_leading_space_lut: np.ndarray
    is_boundary_token_lut: np.ndarray
    train_files_pattern: str
    dataset_name: str


def create_shared_context(
    data_path: str,
    tokenizer_path: str,
    vocab_size: int,
    train_seq_len: int,
) -> SharedTrainingContext:
    """Load tokenizer, validation data, and BPB lookup tables once."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    if int(sp.vocab_size()) != vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )

    dataset_name, _actual, _expected = tgm.validate_dataset_tokenizer_pair(
        data_path, tokenizer_path
    )

    val_pattern = f"{data_path}/fineweb_val_*.bin"
    val_tokens = tgm.load_validation_tokens(val_pattern, train_seq_len)

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        tgm.build_sentencepiece_luts(sp, vocab_size)
    )

    train_files_pattern = f"{data_path}/fineweb_train_*.bin"

    return SharedTrainingContext(
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        train_files_pattern=train_files_pattern,
        dataset_name=dataset_name,
    )


# ---------------------------------------------------------------------------
# Hyperparameters builder
# ---------------------------------------------------------------------------

# Mapping from env var name -> (attribute name, type converter).
# Hyperparameters class reads os.environ at class definition time (import),
# so we must apply overrides via setattr after construction.
_ENV_TO_ATTR: dict[str, tuple[str, Any]] = {
    "DATA_PATH": ("data_path", str),
    "TOKENIZER_PATH": ("tokenizer_path", str),
    "RUN_ID": ("run_id", str),
    "SEED": ("seed", int),
    "ITERATIONS": ("iterations", int),
    "VAL_LOSS_EVERY": ("val_loss_every", int),
    "VAL_BATCH_SIZE": ("val_batch_size", int),
    "TRAIN_LOG_EVERY": ("train_log_every", int),
    "TRAIN_BATCH_TOKENS": ("train_batch_tokens", int),
    "GRAD_ACCUM_STEPS": ("grad_accum_steps", int),
    "TRAIN_SEQ_LEN": ("train_seq_len", int),
    "MLX_MAX_MICROBATCH_TOKENS": ("mlx_max_microbatch_tokens", int),
    "MLX_EAGER_EVAL": ("mlx_eager_eval", lambda v: bool(int(v))),
    "WARMUP_STEPS": ("warmup_steps", int),
    "WARMDOWN_ITERS": ("warmdown_iters", int),
    "MAX_WALLCLOCK_SECONDS": ("max_wallclock_seconds", float),
    "VOCAB_SIZE": ("vocab_size", int),
    "NUM_LAYERS": ("num_layers", int),
    "MODEL_DIM": ("model_dim", int),
    "NUM_HEADS": ("num_heads", int),
    "NUM_KV_HEADS": ("num_kv_heads", int),
    "MLP_MULT": ("mlp_mult", int),
    "TIE_EMBEDDINGS": ("tie_embeddings", lambda v: bool(int(v))),
    "TIED_EMBED_INIT_STD": ("tied_embed_init_std", float),
    "LOGIT_CHUNK_TOKENS": ("logit_chunk_tokens", int),
    "LOGIT_SOFTCAP": ("logit_softcap", float),
    "ROPE_BASE": ("rope_base", float),
    "QK_GAIN_INIT": ("qk_gain_init", float),
    "BETA1": ("beta1", float),
    "BETA2": ("beta2", float),
    "ADAM_EPS": ("adam_eps", float),
    "TIED_EMBED_LR": ("tied_embed_lr", float),
    "MATRIX_LR": ("matrix_lr", float),
    "SCALAR_LR": ("scalar_lr", float),
    "MUON_MOMENTUM": ("muon_momentum", float),
    "MUON_BACKEND_STEPS": ("muon_backend_steps", int),
    "MUON_MOMENTUM_WARMUP_START": ("muon_momentum_warmup_start", float),
    "MUON_MOMENTUM_WARMUP_STEPS": ("muon_momentum_warmup_steps", int),
    "GRAD_CLIP_NORM": ("grad_clip_norm", float),
    "OUT_DIR": ("out_dir", str),
}


def _build_hyperparameters(
    env_overrides: dict[str, str],
    seed: int,
    iterations: int,
    val_loss_every: int = 0,
    warmup_steps: int = 1,
    warmdown_iters: int = 0,
) -> tgm.Hyperparameters:
    """Build Hyperparameters and apply env overrides via setattr.

    Hyperparameters reads os.environ at class definition (import) time,
    so env changes after import have no effect. We apply overrides
    directly to instance attributes instead.
    """
    args = tgm.Hyperparameters()

    # Apply env_overrides via the mapping
    for env_key, env_val in env_overrides.items():
        if env_key in _ENV_TO_ATTR:
            attr_name, converter = _ENV_TO_ATTR[env_key]
            setattr(args, attr_name, converter(env_val))

    # Per-run overrides (not part of the experiment config env)
    args.seed = seed
    args.iterations = iterations
    args.val_loss_every = val_loss_every
    args.warmup_steps = warmup_steps
    args.warmdown_iters = warmdown_iters
    args.run_id = f"inprocess_{uuid.uuid4().hex[:8]}"
    return args


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

_ORIGINAL_MLP_CALL = tgm.MLP.__call__

# Activation variants — monkey-patch MLP.__call__ before model construction
_ACTIVATION_VARIANTS = {
    "relu_sq": None,  # default, no patch needed
    "gelu": lambda self, x: self.proj(nn.gelu(self.fc(x))),
    "silu": lambda self, x: self.proj(nn.silu(self.fc(x))),
    "sin": lambda self, x: self.proj(mx.sin(self.fc(x))),
    "sin_sq": lambda self, x: self.proj(mx.sin(self.fc(x)) ** 2),
    # FAN excluded — doubles hidden dim, incompatible with existing proj shape.
    # To test FAN, use mlp_mult=1 (so sin+gelu concat = 2*dim, matching proj at mlp_mult=2).
}


def train_single_run(
    ctx: SharedTrainingContext,
    env_overrides: dict[str, str],
    seed: int,
    iterations: int,
    val_loss_every: int = 0,
    warmup_steps: int = 1,
    warmdown_iters: int = 0,
    screening_mode: bool = False,
    activation: str = "relu_sq",
    loss_variant: str = "standard",
    loss_config: dict | None = None,
) -> dict[str, Any]:
    """Run a single training session in-process, matching train_gpt_mlx.py main().

    Returns dict with keys: seed, val_bpb, val_loss, wall_time_s.
    """
    t_start = time.perf_counter()

    # --- Build hyperparameters ---
    args = _build_hyperparameters(
        env_overrides=env_overrides,
        seed=seed,
        iterations=iterations,
        val_loss_every=val_loss_every,
        warmup_steps=warmup_steps,
        warmdown_iters=warmdown_iters,
    )

    import logging as _log
    _logger = _log.getLogger("inprocess_trainer")

    # --- Seed PRNG ---
    mx.random.seed(args.seed)

    # --- Activation patch ---
    if activation != "relu_sq" and activation in _ACTIVATION_VARIANTS:
        tgm.MLP.__call__ = _ACTIVATION_VARIANTS[activation]
        _logger.info("seed=%d activation=%s (monkey-patched)", seed, activation)

    # --- Model + optimizer ---
    model = tgm.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
    )
    opt = tgm.SplitOptimizers(model, args)

    # --- Loss variant patch ---
    original_loss = None
    if loss_variant != "standard":
        from scripts.causal.loss_variants import patch_model_loss
        original_loss = patch_model_loss(model, loss_variant, loss_config)

    # --- Compiled functions ---
    compiled_loss = mx.compile(
        lambda x, y: model.loss(x, y),
        inputs=model.state,
        outputs=model.state,
    )
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state,
        outputs=model.state,
    )

    # --- Token loader ---
    train_loader = tgm.TokenLoader(
        ctx.train_files_pattern,
        dataset_name=ctx.dataset_name,
    )

    # --- Warmup phase (lines 963-996 of train_gpt_mlx.py) ---
    # Skip warmup if this architecture was already primed in a prior run.
    # Metal shaders are cached per tensor shape — no need to re-prime.
    arch = _arch_key(args)
    need_warmup = arch not in _WARMED_ARCHITECTURES and args.warmup_steps > 0
    if need_warmup:
        for _warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads = tgm.loss_and_grad_chunked(
                    args, train_loader, compiled_loss_and_grad
                )
                accum = tgm.accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum)
            mx.synchronize()

        # Prime the standalone eval graph
        val_batch_tokens = args.val_batch_size // args.grad_accum_steps
        warm_val_seqs = min(
            val_batch_tokens // args.train_seq_len,
            (ctx.val_tokens.size - 1) // args.train_seq_len,
        )
        warm_chunk = ctx.val_tokens[: warm_val_seqs * args.train_seq_len + 1]
        x_val = mx.array(
            warm_chunk[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32
        )
        y_val = mx.array(
            warm_chunk[1:].reshape(-1, args.train_seq_len), dtype=mx.int32
        )
        warm_val_loss = compiled_loss(x_val, y_val)
        mx.eval(warm_val_loss)
        mx.synchronize()

        # Recreate train_loader to reset data position (matching line 996)
        train_loader = tgm.TokenLoader(
            ctx.train_files_pattern,
            dataset_name=ctx.dataset_name,
        )
        _WARMED_ARCHITECTURES.add(arch)

    # --- Training loop (lines 998-1057 of train_gpt_mlx.py) ---
    train_time_ms = 0.0
    val_loss = float("nan")
    val_bpb = float("nan")
    step_losses: list[dict] = []
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations

        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            if not screening_mode:
                val_loss, val_bpb = tgm.eval_val(
                    args,
                    compiled_loss,
                    ctx.val_tokens,
                    ctx.base_bytes_lut,
                    ctx.has_leading_space_lut,
                    ctx.is_boundary_token_lut,
                )
                _logger.info(
                    "seed=%d step=%d/%d val_loss=%.4f val_bpb=%.4f train_time=%.0fms",
                    seed, step, args.iterations, val_loss, val_bpb, train_time_ms,
                )
            t0 = time.perf_counter()

        if last_step:
            if screening_mode:
                final_tl = step_losses[-1]["train_loss"] if step_losses else float("nan")
                val_loss = final_tl
                val_bpb = final_tl
                _logger.info("seed=%d SCREENING final train_loss=%.4f", seed, final_tl)
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))

        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = tgm.loss_and_grad_chunked(
                args, train_loader, compiled_loss_and_grad
            )
            accum = tgm.accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale
            if args.mlx_eager_eval:
                mx.eval(train_loss, accum)

        grads = tree_unflatten(list(accum.items()))
        grads = tgm.clip_grad_tree(grads, args.grad_clip_norm)
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.synchronize()

        # Capture per-step training loss (pre-update, after synchronize)
        step_losses.append({"step": step, "train_loss": float(train_loss.item())})

        if step % max(args.iterations // 5, 1) == 0:
            _logger.info("seed=%d step=%d/%d training...", seed, step, args.iterations)

        step += 1

    # --- Cleanup ---
    # Restore original loss variant before deleting model
    if original_loss is not None:
        from scripts.causal.loss_variants import restore_model_loss
        restore_model_loss(model, original_loss)

    # Restore original activation before deleting model
    tgm.MLP.__call__ = _ORIGINAL_MLP_CALL

    wall_time_s = time.perf_counter() - t_start
    del model, opt, train_loader, compiled_loss, compiled_loss_and_grad
    gc.collect()

    return {
        "seed": seed,
        "val_bpb": float(val_bpb),
        "val_loss": float(val_loss),
        "wall_time_s": round(wall_time_s, 2),
        "train_loss_final": step_losses[-1]["train_loss"] if step_losses else float("nan"),
        "step_losses": step_losses,
        "screening_mode": screening_mode,
        "loss_variant": loss_variant,
    }


# ---------------------------------------------------------------------------
# Run a full condition (all seeds) in-process
# ---------------------------------------------------------------------------

def run_condition_inprocess(
    ctx: SharedTrainingContext,
    cfg: dict,
    seeds: list[int],
    iterations: int = 100,
    val_loss_every: int = 0,
    warmup_steps: int = 1,
    warmdown_iters: int = 0,
    screening_mode: bool = False,
    loss_variant: str = "standard",
    loss_config: dict | None = None,
) -> list[dict]:
    """Run treatment/control condition across all seeds in-process.

    Returns list of result dicts (same schema as experiment_runner._run_single_seed).
    On OOM or other errors, returns a result dict with an "error" key for that seed.
    """
    env_overrides = cfg.get("env_overrides", {})
    results: list[dict] = []

    for seed in seeds:
        try:
            result = train_single_run(
                ctx=ctx,
                env_overrides=env_overrides,
                seed=seed,
                iterations=iterations,
                val_loss_every=val_loss_every,
                warmup_steps=warmup_steps,
                warmdown_iters=warmdown_iters,
                screening_mode=screening_mode,
                loss_variant=loss_variant,
                loss_config=loss_config,
            )
            results.append(result)
        except Exception as exc:
            # Broad catch for OOM fallback -- MLX Metal OOM may raise
            # RuntimeError, MemoryError, or framework-specific errors
            results.append({
                "seed": seed,
                "val_bpb": None,
                "val_loss": None,
                "wall_time_s": None,
                "error": f"{type(exc).__name__}: {exc}",
            })

    return results
