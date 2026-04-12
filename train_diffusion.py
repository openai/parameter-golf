#!/usr/bin/env python3
"""
Simple MLX masked diffusion language model for Parameter Golf week 3.

This script keeps the week-2 deterministic validation path while adding the
week-3 ablation surface:
- richer schedules and timestep sampling
- optional self-conditioning and loss reweighting
- x0 and xtminus1 parameterizations
- machine-readable logs plus best/last checkpoint saving
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from diffusion_config import Hyperparameters
from diffusion_eval import (
    deterministic_logits,
    evaluate_diffusion_model,
    format_metrics_for_log,
    load_checkpoint_into_model,
    metrics_json_line,
    null_self_condition_inputs,
    prepare_validation_state,
)
from diffusion_model import DiffusionTransformer
from diffusion_objectives import (
    args_mask_rates,
    build_loss_weights_np,
    corrupt_batch_np,
    sample_train_timesteps,
    training_loss_from_logits,
)
from validation_common import load_data_shard


def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def accumulate_flat_grads(
    accum: dict[str, mx.array] | None,
    grads_tree: dict,
    scale: float,
) -> dict[str, mx.array]:
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


class TokenStream:
    def __init__(
        self,
        pattern: str,
        train_shards: int = 0,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        import glob

        files = [Path(p) for p in sorted(glob.glob(pattern))]
        if train_shards > 0:
            files = files[:train_shards]
        self.files = files
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn is not None:
                self.log_fn(
                    f"WARNING: starting_epoch:{self.epoch} dataset:{self.dataset_name} train_shards:{len(self.files)}"
                )
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class TokenLoader:
    def __init__(
        self,
        pattern: str,
        train_shards: int = 0,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.stream = TokenStream(pattern, train_shards=train_shards, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> np.ndarray:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable)
        return chunk.reshape(-1, seq_len)


class SyntheticLoader:
    def __init__(self, tokens: np.ndarray):
        self.tokens = np.ascontiguousarray(tokens, dtype=np.int32)
        self.pos = 0

    def next_batch(self, batch_tokens: int, seq_len: int) -> np.ndarray:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        total = usable
        if self.pos + total > self.tokens.size:
            wrap = self.pos + total - self.tokens.size
            chunk = np.concatenate([self.tokens[self.pos :], self.tokens[:wrap]], axis=0)
            self.pos = wrap
        else:
            chunk = self.tokens[self.pos : self.pos + total]
            self.pos += total
        return chunk.reshape(-1, seq_len)


def clip_grad_tree(grads_tree: dict, max_norm: float) -> dict:
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = 0.0
    for grad in flat.values():
        g = np.asarray(grad.astype(mx.float32), dtype=np.float32)
        total_sq += float(np.sum(np.square(g), dtype=np.float64))
    if total_sq <= 0.0:
        return grads_tree
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


def build_synthetic_tokens(args: Hyperparameters) -> tuple[np.ndarray, np.ndarray]:
    usable_vocab = min(args.synthetic_vocab_size, args.vocab_size - 1)
    if usable_vocab < 4:
        raise ValueError("SYNTHETIC_VOCAB_SIZE must leave room for a mask token")
    pattern = (np.arange(args.synthetic_pattern_len, dtype=np.int32) % usable_vocab) + 1
    train_repeats = (args.synthetic_train_tokens + pattern.size - 1) // pattern.size
    val_repeats = (args.synthetic_val_tokens + pattern.size - 1) // pattern.size
    train_tokens = np.tile(pattern, train_repeats)[: args.synthetic_train_tokens]
    val_tokens = np.tile(pattern, val_repeats)[: args.synthetic_val_tokens]
    return train_tokens, val_tokens


def decode_ids(ids: list[int], sp: spm.SentencePieceProcessor | None, synthetic: bool) -> str:
    if synthetic or sp is None:
        return " ".join(str(int(x)) for x in ids)
    return sp.decode(ids)


def sample_text(
    model: DiffusionTransformer,
    compiled_logits,
    args: Hyperparameters,
    sp: spm.SentencePieceProcessor | None,
    mask_token_id: int,
    *,
    num_steps: int,
) -> str:
    from diffusion_objectives import mask_rate_for_t

    rng = np.random.default_rng(args.seed + 999)
    tokens = np.full((1, args.train_seq_len), mask_token_id, dtype=np.int32)
    fixed = np.zeros((1, args.train_seq_len), dtype=bool)
    if args.sample_prompt and sp is not None and not args.synthetic_data:
        prompt_ids = np.array(sp.encode(args.sample_prompt), dtype=np.int32)[: args.train_seq_len]
        tokens[0, : prompt_ids.size] = prompt_ids
        fixed[0, : prompt_ids.size] = True
    current = mx.array(tokens, dtype=mx.int32)
    empty_sc_ids, empty_sc_scale = null_self_condition_inputs(current)
    for step in range(num_steps, 0, -1):
        t = mx.array([min(step, args.num_diffusion_steps)], dtype=mx.int32)
        if args.self_conditioning:
            logits = deterministic_logits(
                compiled_logits,
                current,
                t,
                self_conditioning=True,
            )
        else:
            logits = compiled_logits(current, t, empty_sc_ids, empty_sc_scale).astype(mx.float32)
        logits = logits.astype(mx.float32) / args.sample_temperature
        sampled = mx.random.categorical(logits)
        mx.eval(sampled)
        sampled_np = np.asarray(sampled, dtype=np.int32)
        current_np = np.array(current, dtype=np.int32, copy=True)
        current_mask = (current_np == mask_token_id) & ~fixed
        if args.parameterization.lower().strip() == "xtminus1":
            current_np[current_mask] = sampled_np[current_mask]
        else:
            if step == 1:
                reveal = current_mask
            else:
                p_now = mask_rate_for_t(
                    np.array([step], dtype=np.int32),
                    num_diffusion_steps=args.num_diffusion_steps,
                    mask_schedule=args.mask_schedule,
                    min_mask_rate=args.min_mask_rate,
                    max_mask_rate=args.max_mask_rate,
                )[0]
                p_next = mask_rate_for_t(
                    np.array([step - 1], dtype=np.int32),
                    num_diffusion_steps=args.num_diffusion_steps,
                    mask_schedule=args.mask_schedule,
                    min_mask_rate=args.min_mask_rate,
                    max_mask_rate=args.max_mask_rate,
                )[0]
                keep_prob = 0.0 if p_now <= 0 else float(np.clip(p_next / p_now, 0.0, 1.0))
                keep_masked = rng.random(current_mask.shape) < keep_prob
                reveal = current_mask & ~keep_masked
            current_np[reveal] = sampled_np[reveal]
        current = mx.array(current_np, dtype=mx.int32)
    ids = np.asarray(current, dtype=np.int32)[0].tolist()
    if args.synthetic_data:
        return decode_ids(ids, None, synthetic=True)
    if sp is not None:
        ids = [tok for tok in ids if tok != mask_token_id]
    return decode_ids(ids, sp, synthetic=False)


def loss_and_grad_chunked(
    args: Hyperparameters,
    train_loader,
    rng: np.random.Generator,
    mask_token_id: int,
    mask_rates: np.ndarray,
    compiled_loss_and_grad,
    compiled_logits,
    train_timestep_offset: int,
) -> tuple[mx.array, dict, float, int]:
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    masked_fraction_sum = 0.0
    for chunk_tokens in chunk_sizes:
        clean_np = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        timesteps_np = sample_train_timesteps(
            clean_np.shape[0],
            args,
            rng,
            offset=train_timestep_offset,
        )
        train_timestep_offset += clean_np.shape[0]
        corrupted_np, timesteps_np, loss_mask_np, masked_fraction = corrupt_batch_np(
            clean_np,
            args,
            rng,
            mask_token_id,
            timesteps=timesteps_np,
            mask_rates=mask_rates,
        )
        loss_weights_np = build_loss_weights_np(
            loss_mask_np,
            timesteps_np,
            mask_rates,
            args.loss_reweighting,
            eps=args.loss_reweighting_eps,
        )
        x = mx.array(corrupted_np, dtype=mx.int32)
        y = mx.array(clean_np, dtype=mx.int32)
        t = mx.array(timesteps_np, dtype=mx.int32)
        m = mx.array(loss_mask_np, dtype=mx.float32)
        w = mx.array(loss_weights_np, dtype=mx.float32)
        sc_ids, sc_scale = build_training_self_condition(args, compiled_logits, x, t, rng)
        loss, grads = compiled_loss_and_grad(x, y, t, m, w, sc_ids, sc_scale)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
        masked_fraction_sum += masked_fraction * scale
        if args.mlx_eager_eval:
            mx.eval(loss_value, grad_accum)
    return loss_value, tree_unflatten(list(grad_accum.items())), masked_fraction_sum, train_timestep_offset


def build_training_self_condition(
    args: Hyperparameters,
    compiled_logits,
    input_ids: mx.array,
    timesteps: mx.array,
    rng: np.random.Generator,
) -> tuple[mx.array, mx.array]:
    empty_sc_ids, empty_sc_scale = null_self_condition_inputs(input_ids)
    if not args.self_conditioning:
        return empty_sc_ids, empty_sc_scale
    pre_logits = compiled_logits(input_ids, timesteps, empty_sc_ids, empty_sc_scale).astype(mx.float32)
    mx.eval(pre_logits)
    sc_ids = mx.argmax(pre_logits, axis=-1).astype(mx.int32)
    sc_scale = mx.array(0.0 if rng.random() < 0.5 else 1.0, dtype=mx.float32)
    return sc_ids, sc_scale


def save_model_state(model: DiffusionTransformer, out_path: Path) -> int:
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    mx.savez(str(out_path), **flat_state)
    return out_path.stat().st_size


def selected_checkpoint_metric(metrics, metric_name: str) -> float:
    value = getattr(metrics, metric_name, None)
    if value is None:
        raise ValueError(f"BEST_CHECKPOINT_METRIC={metric_name} is unavailable for the current validation settings")
    return float(value)


def write_checkpoint_manifest(
    manifest_path: Path,
    *,
    run_id: str,
    last_checkpoint: Path,
    best_checkpoint: Path | None,
    best_metric_name: str,
    best_metric_value: float | None,
    best_step: int | None,
) -> None:
    payload = {
        "run_id": run_id,
        "last_checkpoint": str(last_checkpoint),
        "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "best_step": best_step,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}_diffusion.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Running Python {sys.version}", console=False)
    log(f"Running MLX {mx.__version__}", console=False)
    log("=" * 100, console=False)

    mx.random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    mask_rates = args_mask_rates(args)
    mask_rates_mx = mx.array(mask_rates, dtype=mx.float32)

    sp, dataset_name, actual_train_files, expected_train_files, val_tokens, byte_luts, mask_token_id = prepare_validation_state(
        args,
        log_fn=log,
    )

    if args.synthetic_data:
        train_tokens, _ = build_synthetic_tokens(args)
        train_loader = SyntheticLoader(train_tokens)
        train_shard_msg = "synthetic"
    else:
        train_loader = TokenLoader(args.train_files, train_shards=args.train_shards, log_fn=log, dataset_name=dataset_name)
        train_shard_msg = args.train_shards if args.train_shards > 0 else actual_train_files

    model = DiffusionTransformer(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        mlp_mult=args.mlp_mult,
        num_diffusion_steps=args.num_diffusion_steps,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
    )
    if args.init_checkpoint:
        init_checkpoint_path = Path(args.init_checkpoint)
        if not init_checkpoint_path.is_file():
            raise FileNotFoundError(f"INIT_CHECKPOINT not found: {init_checkpoint_path}")
        load_checkpoint_into_model(model, init_checkpoint_path)
        mx.eval(model.state)
    optimizer = optim.Adam(
        learning_rate=args.learning_rate,
        betas=[args.beta1, args.beta2],
        eps=args.adam_eps,
        bias_correction=True,
    )

    compiled_logits = mx.compile(
        lambda x, t, sc_ids, sc_scale: model.logits(x, t, sc_ids, sc_scale),
        inputs=model.state,
        outputs=model.state,
    )
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(
            model,
            lambda x, y, t, m, w, sc_ids, sc_scale: training_loss_from_logits(
                model.logits(x, t, sc_ids, sc_scale),
                y,
                t,
                m,
                w,
                parameterization=args.parameterization,
                mask_rates=mask_rates_mx,
                mask_token_id=mask_token_id,
            ),
        ),
        inputs=model.state,
        outputs=model.state,
    )

    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"run_id:{args.run_id}")
    log(f"mode:{'synthetic' if args.synthetic_data else 'fineweb'} dataset:{dataset_name}")
    if sp is not None:
        log(f"tokenizer_path:{args.tokenizer_path}")
        log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.size - 1}")
    if expected_train_files is None:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}")
    elif isinstance(actual_train_files, int) and actual_train_files < expected_train_files:
        log(
            f"WARNING: train_loader:subset dataset:{dataset_name} "
            f"train_shards:{actual_train_files}/{expected_train_files} "
            f"new epochs will arrive sooner than the full dataset"
        )
    else:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}/{expected_train_files}")
    log(
        f"mask_token_id:{mask_token_id} mask_schedule:{args.mask_schedule} diffusion_steps:{args.num_diffusion_steps} "
        f"min_mask_rate:{args.min_mask_rate} max_mask_rate:{args.max_mask_rate} "
        f"val_metric:{args.val_metric} val_seed:{args.val_seed} parameterization:{args.parameterization} "
        f"train_timestep_sampling:{args.train_timestep_sampling} loss_reweighting:{args.loss_reweighting} "
        f"self_conditioning:{int(args.self_conditioning)}"
    )
    log(
        f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} dim:{args.model_dim} "
        f"heads:{args.num_heads} seq_len:{args.train_seq_len}"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} val_batch_tokens:{args.val_batch_tokens} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.1f} train_shards:{train_shard_msg}"
    )
    log(
        f"optimizer:adam lr:{args.learning_rate} wd:{args.weight_decay} "
        f"betas:({args.beta1},{args.beta2}) grad_clip_norm:{args.grad_clip_norm}"
    )
    if args.init_checkpoint:
        log(f"init_checkpoint:{args.init_checkpoint}")
    if args.early_stop_patience > 0:
        log(
            f"early_stop:enabled metric:{args.early_stop_metric} "
            f"patience:{args.early_stop_patience} min_delta:{args.early_stop_min_delta}"
        )
    else:
        log("early_stop:disabled")
    log(
        "config_json:"
        + json.dumps(
            {
                "run_id": args.run_id,
                "mask_schedule": args.mask_schedule,
                "num_diffusion_steps": args.num_diffusion_steps,
                "min_mask_rate": args.min_mask_rate,
                "max_mask_rate": args.max_mask_rate,
                "parameterization": args.parameterization,
                "train_timestep_sampling": args.train_timestep_sampling,
                "loss_reweighting": args.loss_reweighting,
                "self_conditioning": args.self_conditioning,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "beta1": args.beta1,
                "beta2": args.beta2,
                "grad_clip_norm": args.grad_clip_norm,
                "warmup_steps": args.warmup_steps,
                "sample_steps_list": args.sample_steps_list,
                "save_best_checkpoint": args.save_best_checkpoint,
                "best_checkpoint_metric": args.best_checkpoint_metric,
                "init_checkpoint": args.init_checkpoint,
                "early_stop_patience": args.early_stop_patience,
                "early_stop_metric": args.early_stop_metric,
                "early_stop_min_delta": args.early_stop_min_delta,
            },
            sort_keys=True,
        )
    )

    train_timestep_offset = 0
    if args.warmup_steps > 0:
        for warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            for _ in range(args.grad_accum_steps):
                loss, grads, _, train_timestep_offset = loss_and_grad_chunked(
                    args,
                    train_loader,
                    np_rng,
                    mask_token_id,
                    mask_rates,
                    compiled_loss_and_grad,
                    compiled_logits,
                    train_timestep_offset,
                )
                warmup_loss = warmup_loss + loss.astype(mx.float32) / args.grad_accum_steps
                accum = accumulate_flat_grads(accum, grads, 1.0 / args.grad_accum_steps)
            mx.eval(warmup_loss, accum)
            if args.warmup_steps <= 10 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        if args.synthetic_data:
            train_tokens, _ = build_synthetic_tokens(args)
            train_loader = SyntheticLoader(train_tokens)
        else:
            train_loader = TokenLoader(args.train_files, train_shards=args.train_shards, log_fn=log, dataset_name=dataset_name)

    t0 = time.perf_counter()
    step = 0
    stop_after_step: int | None = None
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    last_checkpoint_path = out_dir / f"{args.run_id}_diffusion_last_mlx.npz"
    legacy_final_path = out_dir / f"{args.run_id}_diffusion_mlx.npz"
    best_checkpoint_path = out_dir / f"{args.run_id}_diffusion_best_mlx.npz"
    manifest_path = out_dir / f"{args.run_id}_diffusion_manifest.json"
    best_metric_value: float | None = None
    best_metric_step: int | None = None
    saved_best_path: Path | None = None
    early_stop_best_metric: float | None = None
    early_stop_misses = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_run_val = False
        if last_step and args.val_at_end:
            should_run_val = True
        elif step == 0 and args.val_at_start:
            should_run_val = True
        elif step > 0 and args.val_loss_every > 0 and step % args.val_loss_every == 0:
            should_run_val = True
        if should_run_val:
            metrics = evaluate_diffusion_model(
                model,
                compiled_logits,
                val_tokens,
                args,
                mask_token_id,
                byte_luts,
                eval_phase="periodic",
                log_fn=log,
            )
            metric_parts = format_metrics_for_log(metrics).split()
            log(f"step:{step}/{args.iterations} {' '.join(part for part in metric_parts if not part.startswith('tokens:'))}")
            log(f"metrics_json:{metrics_json_line('periodic_eval', metrics, step=step, run_id=args.run_id)}")
            if args.save_best_checkpoint:
                metric_value = selected_checkpoint_metric(metrics, args.best_checkpoint_metric)
                if best_metric_value is None or metric_value < best_metric_value:
                    save_model_state(model, best_checkpoint_path)
                    best_metric_value = metric_value
                    best_metric_step = step
                    saved_best_path = best_checkpoint_path
                    write_checkpoint_manifest(
                        manifest_path,
                        run_id=args.run_id,
                        last_checkpoint=last_checkpoint_path,
                        best_checkpoint=saved_best_path,
                        best_metric_name=args.best_checkpoint_metric,
                        best_metric_value=best_metric_value,
                        best_step=best_metric_step,
                    )
                    log(
                        f"best_checkpoint_saved:step:{step} metric:{args.best_checkpoint_metric} "
                        f"value:{metric_value:.4f} path:{best_checkpoint_path}"
                    )
            if args.early_stop_patience > 0:
                early_metric_value = selected_checkpoint_metric(metrics, args.early_stop_metric)
                improved = (
                    early_stop_best_metric is None
                    or early_metric_value < (early_stop_best_metric - args.early_stop_min_delta)
                )
                if improved:
                    early_stop_best_metric = early_metric_value
                    early_stop_misses = 0
                    log(
                        f"early_stop_status:step:{step} metric:{args.early_stop_metric} "
                        f"value:{early_metric_value:.4f} best:{early_stop_best_metric:.4f} misses:0/{args.early_stop_patience}"
                    )
                else:
                    early_stop_misses += 1
                    log(
                        f"early_stop_status:step:{step} metric:{args.early_stop_metric} "
                        f"value:{early_metric_value:.4f} best:{early_stop_best_metric:.4f} "
                        f"misses:{early_stop_misses}/{args.early_stop_patience}"
                    )
                    if early_stop_misses >= args.early_stop_patience and stop_after_step is None:
                        stop_after_step = step
                        log(
                            f"early_stop_triggered:step:{step} metric:{args.early_stop_metric} "
                            f"best:{early_stop_best_metric:.4f} misses:{early_stop_misses}"
                        )
        if last_step:
            break

        step_t0 = time.perf_counter()
        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        train_masked_fraction = 0.0
        for _ in range(args.grad_accum_steps):
            loss, grads, masked_fraction, train_timestep_offset = loss_and_grad_chunked(
                args,
                train_loader,
                np_rng,
                mask_token_id,
                mask_rates,
                compiled_loss_and_grad,
                compiled_logits,
                train_timestep_offset,
            )
            train_loss = train_loss + loss.astype(mx.float32) / args.grad_accum_steps
            train_masked_fraction += masked_fraction / args.grad_accum_steps
            accum = accumulate_flat_grads(accum, grads, 1.0 / args.grad_accum_steps)
            if args.mlx_eager_eval:
                mx.eval(train_loss, accum)

        grads_tree = tree_unflatten(list(accum.items()))
        grads_tree = clip_grad_tree(grads_tree, args.grad_clip_norm)
        params = dict(tree_flatten(model.trainable_parameters()))
        grads = dict(tree_flatten(grads_tree))
        if args.weight_decay > 0:
            grads = {k: g + args.weight_decay * params[k] for k, g in grads.items()}
        updated = optimizer.apply_gradients(grads, params)
        model.update(tree_unflatten(list(updated.items())))
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        step += 1
        tok_s = args.train_batch_tokens / max(step_ms / 1000.0, 1e-9)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log(
                f"step:{step}/{args.iterations} train_loss:{float(train_loss.item()):.4f} "
                f"masked_frac:{train_masked_fraction:.4f} tok_s:{tok_s:.0f}"
            )
            log(
                "metrics_json:"
                + json.dumps(
                    {
                        "event": "train_step",
                        "run_id": args.run_id,
                        "step": step,
                        "iteration_target": args.iterations,
                        "train_loss": float(train_loss.item()),
                        "masked_frac": train_masked_fraction,
                        "tok_s": tok_s,
                    },
                    sort_keys=True,
                )
            )
        if args.sample_every > 0 and (step <= 3 or step % args.sample_every == 0):
            for num_steps in args.sample_steps_list:
                sample = sample_text(model, compiled_logits, args, sp, mask_token_id, num_steps=num_steps)
                log(f"sample_step:{step} denoise_steps:{num_steps} text:{sample[:400]}")
                log(
                    "sample_json:"
                    + json.dumps(
                        {
                            "event": "sample",
                            "run_id": args.run_id,
                            "step": step,
                            "denoise_steps": num_steps,
                            "text": sample[:400],
                        },
                        sort_keys=True,
                    )
                )
        if max_wallclock_ms is not None and stop_after_step is None:
            elapsed_ms = 1000.0 * (time.perf_counter() - t0)
            if elapsed_ms >= max_wallclock_ms:
                stop_after_step = step

    final_metrics = evaluate_diffusion_model(
        model,
        compiled_logits,
        val_tokens,
        args,
        mask_token_id,
        byte_luts,
        eval_phase="final",
        log_fn=log,
    )
    log(f"final_diffusion_eval {format_metrics_for_log(final_metrics)}")
    log(f"metrics_json:{metrics_json_line('final_diffusion_eval', final_metrics, step=step, run_id=args.run_id)}")
    if args.save_best_checkpoint:
        final_metric_value = selected_checkpoint_metric(final_metrics, args.best_checkpoint_metric)
        if best_metric_value is None or final_metric_value < best_metric_value:
            save_model_state(model, best_checkpoint_path)
            best_metric_value = final_metric_value
            best_metric_step = step
            saved_best_path = best_checkpoint_path
            log(
                f"best_checkpoint_saved:step:{step} metric:{args.best_checkpoint_metric} "
                f"value:{final_metric_value:.4f} path:{best_checkpoint_path}"
            )

    last_bytes = save_model_state(model, last_checkpoint_path)
    legacy_bytes = save_model_state(model, legacy_final_path)
    write_checkpoint_manifest(
        manifest_path,
        run_id=args.run_id,
        last_checkpoint=last_checkpoint_path,
        best_checkpoint=saved_best_path,
        best_metric_name=args.best_checkpoint_metric,
        best_metric_value=best_metric_value,
        best_step=best_metric_step,
    )
    log(f"saved_model:{last_checkpoint_path} bytes:{last_bytes}")
    log(f"saved_model:{legacy_final_path} bytes:{legacy_bytes}")
    log(f"saved_manifest:{manifest_path}")


if __name__ == "__main__":
    main()
