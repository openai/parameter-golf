from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sentencepiece as spm

from diffusion_config import Hyperparameters
from diffusion_objectives import (
    accumulate_elbo_from_nll,
    args_mask_rates,
    choose_mask_token_id,
    corruption_rng,
    corrupt_batch_np,
    stratified_timesteps,
    validate_elbo_mask_rates,
    validate_mask_token_id,
)
from validation_common import (
    build_sentencepiece_luts,
    count_batch_bytes,
    load_validation_tokens,
    trim_validation_tokens,
    validate_dataset_tokenizer_pair,
)


@dataclass
class DiffusionEvalMetrics:
    proxy_loss: float | None
    val_elbo_nats: float | None
    val_bits_per_token: float | None
    val_bpb: float | None
    token_count: int
    byte_count: float
    corruption_samples: int
    timestep_samples: int


def format_metrics_for_log(metrics: DiffusionEvalMetrics) -> str:
    parts: list[str] = []
    if metrics.proxy_loss is not None:
        parts.append(f"proxy_loss:{metrics.proxy_loss:.4f}")
    if metrics.val_elbo_nats is not None:
        parts.append(f"val_elbo_nats:{metrics.val_elbo_nats:.4f}")
        parts.append(f"val_bits_per_token:{metrics.val_bits_per_token:.4f}")
        if metrics.val_bpb is not None:
            parts.append(f"val_bpb:{metrics.val_bpb:.4f}")
    parts.append(f"tokens:{metrics.token_count}")
    parts.append(f"corruption_samples:{metrics.corruption_samples}")
    parts.append(f"timestep_samples:{metrics.timestep_samples}")
    return " ".join(parts)


def validation_metric_flags(args: Hyperparameters) -> tuple[bool, bool]:
    metric = args.val_metric.lower().strip()
    if metric == "proxy":
        return True, False
    if metric == "elbo":
        return False, True
    if metric == "both":
        return True, True
    raise ValueError(f"VAL_METRIC must be one of proxy|elbo|both, got {args.val_metric!r}")


def prepare_validation_state(
    args: Hyperparameters,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[spm.SentencePieceProcessor | None, str, int | str, int | None, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray] | None, int]:
    sp: spm.SentencePieceProcessor | None = None
    dataset_name = "synthetic"
    actual_train_files: int | str = "synthetic"
    expected_train_files: int | None = None
    byte_luts: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    if args.synthetic_data:
        pattern = (np.arange(args.synthetic_pattern_len, dtype=np.int32) % min(args.synthetic_vocab_size, args.vocab_size - 1)) + 1
        val_repeats = (args.synthetic_val_tokens + pattern.size) // pattern.size + 1
        val_tokens = np.tile(pattern, val_repeats)[: args.synthetic_val_tokens + 1]
        val_tokens = trim_validation_tokens(val_tokens, args.train_seq_len, max_tokens=args.val_max_tokens)
    else:
        if not args.tokenizer_path.endswith(".model"):
            raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
        sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
        if int(sp.vocab_size()) != args.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
            )
        dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
            args.data_path,
            args.tokenizer_path,
        )
        val_tokens = load_validation_tokens(args.val_files, args.train_seq_len, max_tokens=args.val_max_tokens)
        byte_luts = build_sentencepiece_luts(sp, args.vocab_size)

    mask_token_id = choose_mask_token_id(sp, args)
    validate_mask_token_id(sp, mask_token_id, synthetic_data=args.synthetic_data)
    if np.any(val_tokens[1:] == mask_token_id):
        raise ValueError(
            f"Validation targets contain MASK_TOKEN_ID={mask_token_id}. "
            "Choose a true non-data mask token before trusting diffusion evaluation."
        )
    if args.val_max_tokens > 0 and log_fn is not None:
        log_fn(
            f"WARNING: VAL_MAX_TOKENS={args.val_max_tokens} enables a validation subset. "
            "Use VAL_MAX_TOKENS=0 for challenge-aligned full-split numbers."
        )
    return sp, dataset_name, actual_train_files, expected_train_files, val_tokens, byte_luts, mask_token_id


def iter_validation_batches(
    val_tokens: np.ndarray,
    batch_tokens: int,
    seq_len: int,
):
    usable_batch = (batch_tokens // seq_len) * seq_len
    if usable_batch <= 0:
        raise ValueError(f"VAL_BATCH_TOKENS too small for TRAIN_SEQ_LEN={seq_len}")
    seqs_per_batch = usable_batch // seq_len
    total_seqs = (val_tokens.size - 1) // seq_len
    total_batches = max((total_seqs + seqs_per_batch - 1) // seqs_per_batch, 1)
    for batch_idx, seq_start in enumerate(range(0, total_seqs, seqs_per_batch), start=1):
        seq_end = min(seq_start + seqs_per_batch, total_seqs)
        raw_start = seq_start * seq_len
        raw_end = seq_end * seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        prev_ids = chunk[:-1].reshape(-1, seq_len)
        target_ids = chunk[1:].reshape(-1, seq_len)
        yield batch_idx, total_batches, prev_ids, target_ids


def clean_token_nll_from_logits(logits_np: np.ndarray, target_ids: np.ndarray, mask_token_id: int) -> np.ndarray:
    masked_logits = np.array(logits_np, dtype=np.float32, copy=True)
    masked_logits[..., mask_token_id] = -1.0e30
    max_logits = np.max(masked_logits, axis=-1, keepdims=True)
    logsumexp = max_logits + np.log(np.sum(np.exp(masked_logits - max_logits), axis=-1, keepdims=True))
    target_logits = np.take_along_axis(masked_logits, target_ids[..., None], axis=-1).squeeze(-1)
    return (logsumexp.squeeze(-1) - target_logits).astype(np.float32)


def evaluate_diffusion_model(
    model: Any,
    compiled_logits: Any,
    val_tokens: np.ndarray,
    args: Hyperparameters,
    mask_token_id: int,
    byte_luts: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    *,
    eval_phase: str,
    log_fn: Callable[[str], None] | None = None,
) -> DiffusionEvalMetrics:
    import mlx.core as mx

    proxy_enabled, elbo_enabled = validation_metric_flags(args)
    mask_rates = args_mask_rates(args)
    if elbo_enabled:
        validate_elbo_mask_rates(mask_rates)

    total_tokens = 0
    total_bytes = 0.0
    proxy_loss_sum = 0.0
    proxy_mask_count = 0.0
    elbo_nats_sum = 0.0

    timestep_samples = max(args.val_timestep_samples, 1)
    corruption_samples = 1 if eval_phase == "periodic" else max(args.val_corruption_samples, 1)

    for batch_idx, total_batches, prev_ids, target_ids in iter_validation_batches(
        val_tokens,
        args.val_batch_tokens,
        args.train_seq_len,
    ):
        batch_size = target_ids.shape[0]
        batch_tokens = int(target_ids.size)
        total_tokens += batch_tokens
        if byte_luts is not None:
            total_bytes += count_batch_bytes(prev_ids, target_ids, *byte_luts)

        for corruption_idx in range(corruption_samples):
            for timestep_sample_idx in range(timestep_samples):
                offset = (
                    args.val_seed
                    + (batch_idx - 1) * batch_size
                    + corruption_idx * batch_size * timestep_samples
                    + timestep_sample_idx * batch_size
                )
                timesteps = stratified_timesteps(
                    batch_size,
                    args.num_diffusion_steps,
                    offset=offset,
                )
                rng = corruption_rng(args.val_seed, batch_idx, corruption_idx, timestep_sample_idx)
                corrupted, _, loss_mask, _ = corrupt_batch_np(
                    target_ids,
                    args,
                    rng,
                    mask_token_id,
                    timesteps=timesteps,
                    mask_rates=mask_rates,
                    ensure_masked_token=False,
                )
                logits = compiled_logits(
                    mx.array(corrupted, dtype=mx.int32),
                    mx.array(timesteps, dtype=mx.int32),
                ).astype(mx.float32)
                mx.eval(logits)
                token_nll = clean_token_nll_from_logits(np.asarray(logits, dtype=np.float32), target_ids, mask_token_id)
                if proxy_enabled:
                    masked_sum = float(np.sum(loss_mask, dtype=np.float64))
                    proxy_loss_sum += float(np.sum(token_nll * loss_mask, dtype=np.float64))
                    proxy_mask_count += masked_sum
                if elbo_enabled:
                    elbo_nats_sum += accumulate_elbo_from_nll(
                        token_nll,
                        loss_mask,
                        timesteps,
                        mask_rates,
                        num_diffusion_steps=args.num_diffusion_steps,
                        timestep_samples=timestep_samples,
                        corruption_samples=corruption_samples,
                    )
        if log_fn is not None and total_batches > 1 and (
            batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0
        ):
            log_fn(f"val_progress:{batch_idx}/{total_batches} phase:{eval_phase}")

    proxy_loss = None
    if proxy_enabled:
        proxy_loss = proxy_loss_sum / max(proxy_mask_count, 1.0)

    val_elbo_nats = None
    val_bits_per_token = None
    val_bpb = None
    if elbo_enabled:
        val_elbo_nats = elbo_nats_sum / max(total_tokens, 1)
        val_bits_per_token = val_elbo_nats / math.log(2.0)
        if total_bytes > 0:
            val_bpb = val_bits_per_token * (float(total_tokens) / total_bytes)

    return DiffusionEvalMetrics(
        proxy_loss=proxy_loss,
        val_elbo_nats=val_elbo_nats,
        val_bits_per_token=val_bits_per_token,
        val_bpb=val_bpb,
        token_count=total_tokens,
        byte_count=total_bytes,
        corruption_samples=corruption_samples,
        timestep_samples=timestep_samples,
    )


def load_checkpoint_into_model(model: Any, checkpoint_path: Path) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    expected_keys = [k for k, _ in tree_flatten(model.state)]
    loaded_state = mx.load(str(checkpoint_path))
    loaded_keys = sorted(loaded_state.keys())
    missing = sorted(set(expected_keys) - set(loaded_keys))
    extra = sorted(set(loaded_keys) - set(expected_keys))
    if missing or extra:
        raise ValueError(
            f"Checkpoint keys mismatch for {checkpoint_path}. Missing={missing[:5]} extra={extra[:5]}"
        )
    restored = [(key, loaded_state[key]) for key in expected_keys]
    model.update(tree_unflatten(restored))


def make_model_from_args(args: Hyperparameters) -> Any:
    from diffusion_model import DiffusionTransformer

    return DiffusionTransformer(
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


def standalone_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved diffusion checkpoint on the validation split.")
    parser.add_argument("--checkpoint", required=True, help="Path to the saved .npz checkpoint")
    args_ns = parser.parse_args()

    import mlx.core as mx

    args = Hyperparameters()
    checkpoint_path = Path(args_ns.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_kind = "full" if args.val_max_tokens == 0 else f"subset_{args.val_max_tokens}"
    logfile = out_dir / f"{checkpoint_path.stem}_{eval_kind}_eval.txt"
    print(logfile)

    def log(msg: str) -> None:
        print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    log("=" * 100, )
    log(f"Running Python {sys.version}")
    log("=" * 100)

    sp, dataset_name, actual_train_files, expected_train_files, val_tokens, byte_luts, mask_token_id = prepare_validation_state(
        args,
        log_fn=log,
    )
    model = make_model_from_args(args)
    load_checkpoint_into_model(model, checkpoint_path)
    compiled_logits = mx.compile(lambda x, t: model.logits(x, t), inputs=model.state, outputs=model.state)

    log(f"checkpoint:{checkpoint_path}")
    log(f"mode:{'synthetic' if args.synthetic_data else 'fineweb'} dataset:{dataset_name}")
    if sp is not None:
        log(f"tokenizer_path:{args.tokenizer_path}")
        log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    if expected_train_files is None:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}")
    elif isinstance(actual_train_files, int) and actual_train_files < expected_train_files:
        log(
            f"WARNING: train_loader:subset dataset:{dataset_name} "
            f"train_shards:{actual_train_files}/{expected_train_files}"
        )
    else:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}/{expected_train_files}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.size - 1}")

    metrics = evaluate_diffusion_model(
        model,
        compiled_logits,
        val_tokens,
        args,
        mask_token_id,
        byte_luts,
        eval_phase="final",
        log_fn=log,
    )
    log(f"final_diffusion_eval {format_metrics_for_log(metrics)}")


if __name__ == "__main__":
    standalone_main()
