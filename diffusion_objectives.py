from __future__ import annotations

from typing import Any

import numpy as np
import sentencepiece as spm


MASK_RATE_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.15),
    (0.15, 0.4),
    (0.4, 0.7),
    (0.7, 1.0000001),
)


def build_mask_rates(
    num_diffusion_steps: int,
    mask_schedule: str,
    min_mask_rate: float,
    max_mask_rate: float,
) -> np.ndarray:
    if num_diffusion_steps <= 0:
        raise ValueError("NUM_DIFFUSION_STEPS must be positive")
    timesteps = np.arange(num_diffusion_steps + 1, dtype=np.int32)
    frac = timesteps.astype(np.float32) / float(num_diffusion_steps)
    if mask_schedule == "uniform":
        rates = np.zeros_like(frac)
        rates[1:] = 1.0
    elif mask_schedule == "linear":
        rates = frac
    elif mask_schedule == "cosine":
        rates = np.sin(0.5 * np.pi * frac) ** 2
    elif mask_schedule == "loglinear":
        rates = np.log1p(99.0 * frac) / np.log(100.0)
    else:
        raise ValueError(f"Unknown MASK_SCHEDULE={mask_schedule}")
    rates = np.asarray(rates, dtype=np.float32)
    if num_diffusion_steps >= 1:
        rates[1:] = np.clip(rates[1:], min_mask_rate, max_mask_rate)
    rates[0] = 0.0
    return rates


def mask_rate_for_t(
    timesteps: np.ndarray,
    *,
    num_diffusion_steps: int,
    mask_schedule: str,
    min_mask_rate: float,
    max_mask_rate: float,
) -> np.ndarray:
    mask_rates = build_mask_rates(num_diffusion_steps, mask_schedule, min_mask_rate, max_mask_rate)
    timesteps = np.asarray(timesteps, dtype=np.int32)
    if np.any(timesteps < 0) or np.any(timesteps > num_diffusion_steps):
        raise ValueError("Timesteps must lie in [0, NUM_DIFFUSION_STEPS]")
    return mask_rates[timesteps]


def args_mask_rates(args: Any) -> np.ndarray:
    return build_mask_rates(
        args.num_diffusion_steps,
        args.mask_schedule,
        args.min_mask_rate,
        args.max_mask_rate,
    )


def beta_schedule_from_mask_rates(mask_rates: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mask_rates = np.asarray(mask_rates, dtype=np.float64)
    betas = np.zeros_like(mask_rates)
    for t in range(1, mask_rates.size):
        denom = max(1.0 - float(mask_rates[t - 1]), eps)
        betas[t] = (float(mask_rates[t]) - float(mask_rates[t - 1])) / denom
    return betas.astype(np.float32)


def validate_elbo_mask_rates(mask_rates: np.ndarray, tol: float = 1e-6) -> None:
    mask_rates = np.asarray(mask_rates, dtype=np.float64)
    if abs(float(mask_rates[0])) > tol:
        raise ValueError(f"ELBO requires m_0=0, got {mask_rates[0]:.8f}")
    diffs = np.diff(mask_rates)
    if np.any(diffs < -tol):
        raise ValueError("ELBO requires non-decreasing cumulative mask rates")
    if abs(float(mask_rates[-1]) - 1.0) > tol:
        raise ValueError(
            "ELBO requires the final absorbing state to be fully masked; "
            f"got m_T={mask_rates[-1]:.8f}"
        )


def choose_mask_token_id(sp: spm.SentencePieceProcessor | None, args: Any) -> int:
    if args.mask_token_id >= 0:
        if args.mask_token_id >= args.vocab_size:
            raise ValueError(f"MASK_TOKEN_ID={args.mask_token_id} must be < VOCAB_SIZE={args.vocab_size}")
        return args.mask_token_id
    if sp is None:
        return args.vocab_size - 1
    if 0 <= sp.pad_id() < args.vocab_size and sp.pad_id() != sp.unk_id():
        return int(sp.pad_id())
    for token_id in range(args.vocab_size - 1, -1, -1):
        if sp.is_unused(token_id):
            return token_id
    for token_id in range(args.vocab_size):
        if sp.is_control(token_id) and token_id != sp.unk_id():
            return token_id
    raise ValueError(
        "Could not find a safe mask token in the tokenizer. Set MASK_TOKEN_ID to an unused/control/pad token."
    )


def validate_mask_token_id(
    sp: spm.SentencePieceProcessor | None,
    mask_token_id: int,
    *,
    synthetic_data: bool,
) -> None:
    if synthetic_data or sp is None:
        return
    if mask_token_id == sp.pad_id():
        return
    if sp.is_unused(mask_token_id):
        return
    if sp.is_control(mask_token_id) and mask_token_id != sp.unk_id():
        return
    raise ValueError(
        f"MASK_TOKEN_ID={mask_token_id} is not a safe non-data mask token. "
        "Use a pad/control/unused token for trustworthy diffusion evaluation."
    )


def posterior_clean_probability(mask_rates: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
    timesteps = np.asarray(timesteps, dtype=np.int32)
    m_t = mask_rates[timesteps]
    m_prev = mask_rates[timesteps - 1]
    out = np.zeros_like(m_t, dtype=np.float32)
    nonzero = m_t > 0
    out[nonzero] = ((m_t[nonzero] - m_prev[nonzero]) / m_t[nonzero]).astype(np.float32)
    return out


def posterior_mask_probability(mask_rates: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
    timesteps = np.asarray(timesteps, dtype=np.int32)
    m_t = mask_rates[timesteps]
    m_prev = mask_rates[timesteps - 1]
    out = np.zeros_like(m_t, dtype=np.float32)
    nonzero = m_t > 0
    out[nonzero] = (m_prev[nonzero] / m_t[nonzero]).astype(np.float32)
    return out


def stratified_timesteps(
    batch_size: int,
    num_diffusion_steps: int,
    *,
    offset: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_diffusion_steps <= 0:
        raise ValueError("num_diffusion_steps must be positive")
    return ((np.arange(batch_size, dtype=np.int32) + int(offset)) % num_diffusion_steps) + 1


def corruption_rng(seed: int, batch_idx: int, corruption_idx: int, timestep_sample_idx: int) -> np.random.Generator:
    seed_seq = np.random.SeedSequence([seed, batch_idx, corruption_idx, timestep_sample_idx])
    return np.random.default_rng(seed_seq)


def sample_train_timesteps(
    batch_size: int,
    args: Any,
    rng: np.random.Generator,
    *,
    offset: int,
) -> np.ndarray:
    mode = args.train_timestep_sampling.lower().strip()
    if mode == "random":
        return rng.integers(1, args.num_diffusion_steps + 1, size=(batch_size,), dtype=np.int32)
    if mode == "cyclic":
        return stratified_timesteps(batch_size, args.num_diffusion_steps, offset=offset)
    raise ValueError(f"Unknown TRAIN_TIMESTEP_SAMPLING={args.train_timestep_sampling}")


def corrupt_batch_np(
    clean_ids: np.ndarray,
    args: Any,
    rng: np.random.Generator,
    mask_token_id: int,
    *,
    timesteps: np.ndarray | None = None,
    mask_rates: np.ndarray | None = None,
    ensure_masked_token: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    batch, seq_len = clean_ids.shape
    if timesteps is None:
        timesteps = rng.integers(1, args.num_diffusion_steps + 1, size=(batch,), dtype=np.int32)
    else:
        timesteps = np.asarray(timesteps, dtype=np.int32)
        if timesteps.shape != (batch,):
            raise ValueError(f"timesteps must have shape ({batch},), got {timesteps.shape}")
    if mask_rates is None:
        mask_rates = args_mask_rates(args)
    per_row_rates = mask_rates[timesteps]
    mask = rng.random((batch, seq_len), dtype=np.float32) < per_row_rates[:, None]
    if ensure_masked_token:
        no_mask_rows = np.where(mask.sum(axis=1) == 0)[0]
        if no_mask_rows.size:
            cols = rng.integers(0, seq_len, size=(no_mask_rows.size,), dtype=np.int32)
            mask[no_mask_rows, cols] = True
    corrupted = np.asarray(clean_ids, dtype=np.int32).copy()
    corrupted[mask] = mask_token_id
    return corrupted, timesteps, mask.astype(np.float32), float(mask.mean())


def build_loss_weights_np(
    loss_mask: np.ndarray,
    timesteps: np.ndarray,
    mask_rates: np.ndarray,
    mode: str,
    *,
    eps: float,
) -> np.ndarray:
    weights = np.asarray(loss_mask, dtype=np.float32).copy()
    normalized_mode = mode.lower().strip()
    if normalized_mode == "none":
        return weights
    if normalized_mode != "inverse_mask_rate":
        raise ValueError(f"Unknown LOSS_REWEIGHTING={mode}")
    row_scale = 1.0 / np.maximum(mask_rates[np.asarray(timesteps, dtype=np.int32)], eps)
    weights *= row_scale[:, None].astype(np.float32)
    weight_sum = float(np.sum(weights, dtype=np.float64))
    mask_sum = float(np.sum(loss_mask, dtype=np.float64))
    if weight_sum > 0.0 and mask_sum > 0.0:
        weights *= mask_sum / weight_sum
    return weights.astype(np.float32)


def build_loss_weights_mx(
    loss_mask: Any,
    timesteps: Any,
    mask_rates: Any,
    mode: str,
    *,
    eps: float,
) -> Any:
    import mlx.core as mx

    weights = loss_mask.astype(mx.float32)
    normalized_mode = mode.lower().strip()
    if normalized_mode == "none":
        return weights
    if normalized_mode != "inverse_mask_rate":
        raise ValueError(f"Unknown LOSS_REWEIGHTING={mode}")
    row_scale = 1.0 / mx.maximum(mask_rates[timesteps], mx.array(eps, dtype=mx.float32))
    weights = weights * row_scale[:, None]
    weight_sum = mx.sum(weights)
    mask_sum = mx.sum(loss_mask.astype(mx.float32))
    return mx.where(weight_sum > 0, weights * (mask_sum / weight_sum), weights)


def xtminus1_clean_probability(mask_rates: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
    return posterior_clean_probability(mask_rates, timesteps)


def xtminus1_clean_probability_mx(mask_rates: Any, timesteps: Any) -> Any:
    import mlx.core as mx

    m_t = mask_rates[timesteps]
    m_prev = mask_rates[timesteps - 1]
    zeros = mx.zeros_like(m_t)
    return mx.where(m_t > 0, (m_t - m_prev) / m_t, zeros).astype(mx.float32)


def x0_token_nll_mx(logits: Any, target_ids: Any) -> Any:
    import mlx.core as mx

    log_probs = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32), axis=-1, keepdims=True)
    target_log_probs = mx.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    return (-target_log_probs).astype(mx.float32)


def xtminus1_token_kl_mx(
    logits: Any,
    target_ids: Any,
    timesteps: Any,
    mask_rates: Any,
    mask_token_id: int,
) -> Any:
    import mlx.core as mx

    log_probs = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32), axis=-1, keepdims=True)
    clean_log_probs = mx.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    mask_log_probs = log_probs[..., mask_token_id]
    alpha = xtminus1_clean_probability_mx(mask_rates, timesteps)[:, None]
    one_minus_alpha = mx.maximum(1.0 - alpha, mx.array(0.0, dtype=mx.float32))
    target_entropy = mx.where(alpha > 0, alpha * mx.log(alpha), mx.array(0.0, dtype=mx.float32))
    target_entropy = target_entropy + mx.where(
        one_minus_alpha > 0,
        one_minus_alpha * mx.log(one_minus_alpha),
        mx.array(0.0, dtype=mx.float32),
    )
    cross_entropy = -(alpha * clean_log_probs + one_minus_alpha * mask_log_probs)
    return (cross_entropy + target_entropy).astype(mx.float32)


def training_loss_from_logits(
    logits: Any,
    target_ids: Any,
    timesteps: Any,
    loss_mask: Any,
    loss_weights: Any,
    *,
    parameterization: str,
    mask_rates: Any,
    mask_token_id: int,
) -> Any:
    import mlx.core as mx

    normalized = parameterization.lower().strip()
    if normalized == "x0":
        token_loss = x0_token_nll_mx(logits, target_ids)
    elif normalized == "xtminus1":
        token_loss = xtminus1_token_kl_mx(logits, target_ids, timesteps, mask_rates, mask_token_id)
    else:
        raise ValueError(f"Unknown PARAMETERIZATION={parameterization}")
    weights = loss_mask.astype(mx.float32) * loss_weights.astype(mx.float32)
    return mx.sum(token_loss * weights) / mx.maximum(mx.sum(weights), mx.array(1.0, dtype=mx.float32))


def accumulate_elbo_from_kl(
    token_kl: np.ndarray,
    masked_positions: np.ndarray,
    *,
    num_diffusion_steps: int,
    timestep_samples: int,
    corruption_samples: int,
) -> float:
    scale = float(num_diffusion_steps) / float(max(timestep_samples * corruption_samples, 1))
    return float(np.sum(np.asarray(token_kl, dtype=np.float32) * np.asarray(masked_positions, dtype=np.float32), dtype=np.float64) * scale)


def xtminus1_token_kl_np(
    logits_np: np.ndarray,
    target_ids: np.ndarray,
    timesteps: np.ndarray,
    mask_rates: np.ndarray,
    mask_token_id: int,
) -> np.ndarray:
    masked_logits = np.asarray(logits_np, dtype=np.float32)
    max_logits = np.max(masked_logits, axis=-1, keepdims=True)
    logsumexp = max_logits + np.log(np.sum(np.exp(masked_logits - max_logits), axis=-1, keepdims=True))
    log_probs = masked_logits - logsumexp
    clean_log_probs = np.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    mask_log_probs = log_probs[..., mask_token_id]
    alpha = xtminus1_clean_probability(mask_rates, timesteps)[:, None].astype(np.float32)
    one_minus_alpha = np.maximum(1.0 - alpha, 0.0).astype(np.float32)
    target_entropy = np.where(alpha > 0, alpha * np.log(alpha), 0.0) + np.where(
        one_minus_alpha > 0,
        one_minus_alpha * np.log(one_minus_alpha),
        0.0,
    )
    cross_entropy = -(alpha * clean_log_probs + one_minus_alpha * mask_log_probs)
    return (cross_entropy + target_entropy).astype(np.float32)


def mask_rate_bucket_index(mask_rate: float) -> int:
    for idx, (lo, hi) in enumerate(MASK_RATE_BUCKETS):
        if lo <= mask_rate < hi:
            return idx
    return len(MASK_RATE_BUCKETS) - 1


def accumulate_elbo_from_nll(
    token_nll: np.ndarray,
    masked_positions: np.ndarray,
    timesteps: np.ndarray,
    mask_rates: np.ndarray,
    *,
    num_diffusion_steps: int,
    timestep_samples: int,
    corruption_samples: int,
) -> float:
    alpha = posterior_clean_probability(mask_rates, timesteps).astype(np.float32)
    masked_positions = np.asarray(masked_positions, dtype=np.float32)
    scale = float(num_diffusion_steps) / float(max(timestep_samples * corruption_samples, 1))
    return float(np.sum(np.asarray(token_nll, dtype=np.float32) * masked_positions * alpha[:, None], dtype=np.float64) * scale)
