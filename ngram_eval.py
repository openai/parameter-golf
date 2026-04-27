"""
N-gram Backoff Eval Cache for Parameter Golf.
Implements the breakthrough eval-time technique from PR #809 (0.295 BPB).

During evaluation, builds a growing N-gram cache from already-scored tokens.
Uses highest-order match with entropy-adaptive alpha blending to combine
N-gram predictions with model predictions.

Usage:
    Set NGRAM_EVAL=1 to enable during final evaluation.
    Set NGRAM_MAX_ORDER=9 for max N-gram order (default 9).
"""
from __future__ import annotations

import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class BackoffNgramMixer:
    """Backoff N-gram cache with entropy-adaptive blending.
    
    Maintains counts of N-gram patterns from scored tokens.
    For predictions, finds the highest-order matching context
    and builds a probability distribution from observed next-token counts.
    Blends with model probabilities using entropy-adaptive alpha.
    """

    def __init__(
        self,
        vocab_size: int,
        max_order: int = 9,
        alpha_min: float = 0.05,
        alpha_max: float = 0.70,
        entropy_center: float = 3.0,
        entropy_scale: float = 1.5,
    ):
        self.vocab_size = vocab_size
        self.max_order = max_order
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.entropy_center = entropy_center
        self.entropy_scale = entropy_scale
        # N-gram counts: order -> {context_tuple -> {next_token_id -> count}}
        self.tables: dict[int, dict[tuple[int, ...], dict[int, int]]] = {
            order: defaultdict(lambda: defaultdict(int))
            for order in range(2, max_order + 1)
        }
        self.history: list[int] = []

    def update(self, token_id: int) -> None:
        """Add a scored token to the history and update all N-gram tables."""
        self.history.append(token_id)
        n = len(self.history)
        for order in range(2, min(self.max_order + 1, n + 1)):
            context = tuple(self.history[n - order : n - 1])
            self.tables[order][context][token_id] += 1

    def update_batch(self, token_ids: list[int]) -> None:
        """Batch update the cache with multiple tokens."""
        for t in token_ids:
            self.update(t)

    def get_ngram_dist(self, context: list[int], device: torch.device) -> Tensor | None:
        """Get N-gram probability distribution using backoff strategy.
        
        Tries highest order first, backs off to lower orders.
        Returns tensor of shape [vocab_size] on device, or None if no match.
        """
        for order in range(self.max_order, 1, -1):
            if len(context) < order - 1:
                continue
            ctx_key = tuple(context[-(order - 1):])
            counts = self.tables[order].get(ctx_key)
            if counts is not None and len(counts) > 0:
                total = sum(counts.values())
                if total < 1:
                    continue
                probs = torch.zeros(self.vocab_size, device=device, dtype=torch.float32)
                for tok, cnt in counts.items():
                    probs[tok] = cnt / total
                return probs
        return None

    def compute_alpha(self, model_logprobs: Tensor) -> float:
        """Compute entropy-adaptive blending alpha from model log-probabilities.
        
        High model entropy -> trust N-gram more (higher alpha).
        Low model entropy -> trust model more (lower alpha).
        """
        probs = model_logprobs.exp()
        entropy = -(probs * model_logprobs).sum().item()
        # Clamp for numerical safety
        entropy = max(0.0, entropy)
        x = (entropy - self.entropy_center) * self.entropy_scale
        # Sigmoid
        if x > 20:
            sigmoid = 1.0
        elif x < -20:
            sigmoid = 0.0
        else:
            sigmoid = 1.0 / (1.0 + math.exp(-x))
        return self.alpha_min + (self.alpha_max - self.alpha_min) * sigmoid


def eval_val_ngram(
    args,
    base_model,
    model,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    max_order: int = 9,
    log_fn=None,
) -> tuple[float, float]:
    """N-gram enhanced validation evaluation.
    
    Processes validation tokens in chunks. For each chunk:
    1. Get model logits via forward pass
    2. For each token, look up N-gram prediction from cache
    3. Blend model + N-gram probabilities with entropy-adaptive alpha
    4. Compute cross-entropy from blended distribution
    5. Add scored tokens to cache for future predictions
    
    Returns (val_loss, val_bpb) like the standard eval_val.
    """
    if log_fn is None:
        log_fn = print

    mixer = BackoffNgramMixer(
        vocab_size=args.vocab_size,
        max_order=max_order,
    )

    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1

    # We need model to return logits instead of loss
    base_model._return_logits = True

    loss_sum = 0.0
    token_count = 0.0
    byte_count = 0.0
    ngram_hits = 0
    ngram_misses = 0

    model.eval()
    t_start = time.perf_counter()
    chunks_done = 0

    with torch.inference_mode():
        for start in range(0, total_tokens - seq_len + 1, seq_len):
            end = start + seq_len + 1
            if end > val_tokens.numel():
                break

            chunk = val_tokens[start:end].to(device=device, dtype=torch.int64)
            x = chunk[:-1].unsqueeze(0)  # [1, seq_len]
            y = chunk[1:]  # [seq_len]

            # Get model logits
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(x, y.unsqueeze(0))  # [1, seq_len, vocab_size]

            logits = logits.squeeze(0).float()  # [seq_len, vocab_size]
            model_logprobs = F.log_softmax(logits, dim=-1)

            # Context tokens for N-gram lookup (from history + current chunk)
            chunk_tokens = chunk.cpu().tolist()

            # Process each token in the chunk
            chunk_loss = 0.0
            for t in range(seq_len):
                target = y[t].item()
                token_logprobs = model_logprobs[t]  # [vocab_size]

                # Build context from history + current chunk tokens up to position t
                context = list(mixer.history) + chunk_tokens[:t + 1]

                # Try N-gram prediction
                ngram_probs = mixer.get_ngram_dist(context, device)

                if ngram_probs is not None:
                    ngram_hits += 1
                    alpha = mixer.compute_alpha(token_logprobs)
                    # Blend: p_final = (1-alpha) * p_model + alpha * p_ngram
                    model_probs = token_logprobs.exp()
                    blended = (1.0 - alpha) * model_probs + alpha * ngram_probs
                    blended = blended.clamp(min=1e-10)
                    token_loss = -torch.log(blended[target]).item()
                else:
                    ngram_misses += 1
                    token_loss = -token_logprobs[target].item()

                chunk_loss += token_loss

            loss_sum += chunk_loss
            token_count += seq_len

            # Byte counting
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            t_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            t_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            byte_count += t_bytes.to(torch.float64).sum().item()

            # Add all scored tokens to the N-gram cache
            mixer.update_batch(y.cpu().tolist())

            chunks_done += 1
            if chunks_done % 50 == 0:
                elapsed = time.perf_counter() - t_start
                current_bpb = (loss_sum / token_count) / math.log(2.0) * (token_count / max(byte_count, 1))
                hit_rate = ngram_hits / max(ngram_hits + ngram_misses, 1) * 100
                log_fn(
                    f"ngram_eval: chunk {chunks_done}, "
                    f"tokens {int(token_count)}/{total_tokens}, "
                    f"bpb_so_far {current_bpb:.4f}, "
                    f"hit_rate {hit_rate:.1f}%, "
                    f"elapsed {elapsed:.0f}s"
                )

    # Restore normal forward mode
    base_model._return_logits = False

    val_loss = loss_sum / token_count
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count / byte_count

    elapsed = time.perf_counter() - t_start
    hit_rate = ngram_hits / max(ngram_hits + ngram_misses, 1) * 100
    log_fn(
        f"ngram_eval: DONE, {chunks_done} chunks, {elapsed:.0f}s, "
        f"hit_rate {hit_rate:.1f}%, cache_size {len(mixer.history)}"
    )

    model.train()
    return float(val_loss), float(bits_per_token * tokens_per_byte)
