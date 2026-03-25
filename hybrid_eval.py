"""
Unified hybrid evaluation: blends neural model + FST + n-gram cache + match model
using a GLN context mixer (with entropy-adaptive fallback).

This replaces the eval_val_sliding function in train_gpt.py when HYBRID_EVAL=1.

Architecture:
  Neural Transformer (scored via standard cross-entropy)
       |
       v
  [neural logits] ----> entropy computation
       |                      |
       v                      v
  softmax(logits)     context features for GLN
       |
       +---> GLN Mixer <--- FST predictor (structural patterns)
       |         ^     <--- N-gram cache (order-8, Kneser-Ney)
       |         |     <--- Match model (LZ77-style)
       |         |
       |    mixed distribution
       v         |
  score token (NLL from mixed distribution)
       |
       v
  observe token into cache + match model (backward-looking)

All predictors are backward-looking: tokens are scored BEFORE being observed.
"""
import torch
import torch.nn.functional as F
import math
import time
import os
import sentencepiece as spm
from typing import Optional

try:
    import torch.distributed as dist
except ImportError:
    dist = None

from fst_predictor import WebTextFST
from ngram_cache import NgramCache
from match_model import MatchModel
from gln_mixer import GLNMixer, SimpleAlphaFallback


class UnifiedHybridPredictor:
    """Combines neural model with FST, n-gram cache, and match model
    using GLN context mixing."""

    def __init__(self, tokenizer_path: str, vocab_size: int = 1024):
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.vocab_size = vocab_size

        # Predictors
        self.fst = WebTextFST(self.sp)
        self.cache = NgramCache(max_order=8, vocab_size=vocab_size)
        self.match = MatchModel(vocab_size=vocab_size, max_order=32, min_match=3)

        # Mixer
        self.gln = GLNMixer(
            num_predictors=4,  # neural, fst, ngram, match
            num_context_bins=8,
            learning_rate=0.1,
            vocab_size=vocab_size,
        )
        self.fallback = SimpleAlphaFallback(vocab_size=vocab_size)

        # GLN warmup: use fallback for first N tokens
        self.gln_warmup_tokens = 500
        self.tokens_scored = 0

        # Decoded text buffer for FST
        self._text_buffer = ""
        self._token_buffer: list[int] = []

        # Statistics
        self.stats = {
            "total": 0,
            "fst_active": 0,
            "cache_active": 0,
            "match_active": 0,
            "gln_used": 0,
            "fallback_used": 0,
            "hybrid_nll_sum": 0.0,
            "neural_nll_sum": 0.0,
        }

    def _decode_tokens(self, token_ids: list[int]) -> str:
        """Decode token IDs to text for FST."""
        try:
            return self.sp.decode(token_ids)
        except Exception:
            return ""

    def blend_and_score(
        self,
        neural_logits: torch.Tensor,
        target_token: int,
        context_tokens: list[int],
    ) -> float:
        """Score a single token position using hybrid predictions.

        Args:
            neural_logits: [vocab_size] raw logits from the neural model.
            target_token: The true token at this position.
            context_tokens: Preceding token IDs.

        Returns:
            Negative log-likelihood (NLL) for this position using the
            blended distribution.
        """
        self.stats["total"] += 1

        # === Neural base distribution ===
        neural_probs = F.softmax(neural_logits.float().cpu(), dim=-1)

        # Compute neural entropy (normalized to [0, 1])
        ent = -(neural_probs * torch.log(neural_probs + 1e-10)).sum()
        max_ent = math.log(self.vocab_size)
        norm_entropy = min(1.0, (ent / max_ent).item())

        # Neural NLL for comparison
        neural_nll = -torch.log(neural_probs[target_token] + 1e-10).item()
        self.stats["neural_nll_sum"] += neural_nll

        # === Get predictions from all sources ===
        predictor_probs = [neural_probs, None, None, None]
        predictor_confs = [1.0, 0.0, 0.0, 0.0]

        # FST prediction
        if len(self._text_buffer) > 5:
            try:
                fst_probs, fst_conf = self.fst.predict(self._text_buffer)
                if fst_probs is not None and fst_conf > 0.2:
                    predictor_probs[1] = fst_probs
                    predictor_confs[1] = fst_conf
                    self.stats["fst_active"] += 1
            except Exception:
                pass

        # N-gram cache prediction
        if len(self._token_buffer) >= 2:
            cache_probs, cache_conf = self.cache.predict(self._token_buffer[-16:])
            if cache_probs is not None and cache_conf > 0.05:
                predictor_probs[2] = cache_probs
                predictor_confs[2] = cache_conf
                self.stats["cache_active"] += 1

        # Match model prediction
        if len(self._token_buffer) >= 3:
            match_probs, match_conf = self.match.predict(self._token_buffer[-64:])
            if match_probs is not None and match_conf > 0.1:
                predictor_probs[3] = match_probs
                predictor_confs[3] = match_conf
                self.stats["match_active"] += 1

        # === Mix predictions ===
        if self.tokens_scored >= self.gln_warmup_tokens:
            # Use GLN mixer
            mixed_probs = self.gln.mix(
                predictor_probs, predictor_confs, norm_entropy
            )
            self.stats["gln_used"] += 1
        else:
            # Use simple alpha fallback during warmup
            other_probs = predictor_probs[1:]
            other_confs = predictor_confs[1:]
            mixed_probs = self.fallback.mix(
                neural_probs, other_probs, other_confs, norm_entropy
            )
            self.stats["fallback_used"] += 1

        # === Compute NLL from mixed distribution ===
        mixed_nll = -torch.log(mixed_probs[target_token] + 1e-10).item()
        self.stats["hybrid_nll_sum"] += mixed_nll

        # === Update GLN weights (backward-looking: token is now scored) ===
        self.gln.update(
            predictor_probs, predictor_confs, norm_entropy, target_token
        )

        # === Observe the scored token into backward-looking predictors ===
        self._token_buffer.append(target_token)
        self.cache.observe([target_token])
        self.match.observe([target_token])

        # Update text buffer
        try:
            self._text_buffer = self._decode_tokens(
                self._token_buffer[-300:]
            )
        except Exception:
            pass

        self.tokens_scored += 1
        return mixed_nll

    def print_stats(self) -> None:
        """Print summary statistics."""
        t = max(self.stats["total"], 1)
        print(f"  Hybrid eval stats ({t} positions):")
        print(f"    FST active:   {self.stats['fst_active']:>7} ({100*self.stats['fst_active']/t:.1f}%)")
        print(f"    Cache active: {self.stats['cache_active']:>7} ({100*self.stats['cache_active']/t:.1f}%)")
        print(f"    Match active: {self.stats['match_active']:>7} ({100*self.stats['match_active']/t:.1f}%)")
        print(f"    GLN used:     {self.stats['gln_used']:>7} ({100*self.stats['gln_used']/t:.1f}%)")
        print(f"    Fallback:     {self.stats['fallback_used']:>7} ({100*self.stats['fallback_used']/t:.1f}%)")

        neural_avg = self.stats["neural_nll_sum"] / t
        hybrid_avg = self.stats["hybrid_nll_sum"] / t
        delta = hybrid_avg - neural_avg
        print(f"    Avg NLL: neural={neural_avg:.4f} hybrid={hybrid_avg:.4f} delta={delta:+.4f}")

        # GLN weight stats
        gln_stats = self.gln.get_stats()
        if gln_stats.get("total_updates", 0) > 0:
            print(f"    GLN updates: {gln_stats['total_updates']}")
            for key, val in gln_stats.items():
                if key.startswith("bin_"):
                    print(f"      {key}: {val}")

    def reset(self) -> None:
        """Reset all state for a new evaluation run."""
        self.cache.reset()
        self.match.reset()
        self.gln.reset()
        self._text_buffer = ""
        self._token_buffer = []
        self.tokens_scored = 0
        self.stats = {k: 0 if isinstance(v, int) else 0.0 for k, v in self.stats.items()}


def eval_val_sliding_hybrid(
    args,
    base_model: torch.nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
    log0=print,
) -> tuple[float, float]:
    """Sliding window evaluation with hybrid predictor blending.

    Same interface as eval_val_sliding but uses the UnifiedHybridPredictor
    to blend neural + FST + n-gram + match predictions.

    IMPORTANT: This processes windows sequentially (not batched) because
    the hybrid predictors need to observe tokens in order. However, the
    neural model forward pass is still batched for efficiency.

    Strategy: Run the neural model in batched sliding windows (standard),
    then post-process the logits with the hybrid predictor per-token.
    """
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1

    # Initialize hybrid predictor
    tokenizer_path = getattr(args, "tokenizer_path",
        os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"))
    vocab_size = getattr(args, "vocab_size", 1024)

    hybrid = UnifiedHybridPredictor(tokenizer_path, vocab_size)

    # Compute all window starts
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    neural_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()
    t0 = time.perf_counter()

    # Process in batches for neural forward pass, then score per-token with hybrid
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)

            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []

            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            # Neural forward pass (batched, efficient)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = base_model.forward_logits(x_batch)

            # Standard neural NLL (for comparison and byte counting)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            # Per-token hybrid scoring for the scored region
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)

                for pos in range(s, wlen):
                    target_tok = y_batch[i, pos].item()
                    token_logits = logits[i, pos]  # [vocab_size]

                    # Context: all tokens up to this position
                    ctx = x_batch[i, :pos + 1].cpu().tolist()

                    # Hybrid blend + score
                    hybrid_nll = hybrid.blend_and_score(
                        token_logits, target_tok, ctx
                    )

                    loss_sum += hybrid_nll
                    neural_loss_sum += nll[i, pos].to(torch.float64)
                    token_count += 1.0

                    # Byte counting (same as standard eval)
                    tgt = y_batch[i, pos:pos + 1]
                    prev = x_batch[i, pos:pos + 1]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

            # Progress logging
            if rank == 0 and bi % (batch_seqs * 10) == 0:
                elapsed = time.perf_counter() - t0
                if token_count.item() > 0:
                    curr_bpb = (loss_sum.item() / math.log(2.0)) * (token_count.item() / max(byte_count.item(), 1)) / max(token_count.item(), 1) * math.log(2.0)
                    # Simpler: just nll/log2 * tok/byte
                    avg_nll = loss_sum.item() / token_count.item()
                    neural_avg = neural_loss_sum.item() / token_count.item()
                    log0(f"  hybrid_sliding [{bi}/{len(my_windows)}] "
                         f"hybrid_nll={avg_nll:.4f} neural_nll={neural_avg:.4f} "
                         f"delta={avg_nll - neural_avg:+.4f} time={elapsed:.1f}s")

    # All-reduce across ranks
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(neural_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    neural_loss = (neural_loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    val_bpb = bits_per_token * tokens_per_byte

    neural_bpb = (neural_loss / math.log(2.0)) * tokens_per_byte

    if rank == 0:
        log0(f"hybrid_sliding:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f}")
        log0(f"  neural_only: val_loss={neural_loss:.6f} val_bpb={neural_bpb:.6f}")
        log0(f"  delta_bpb={val_bpb - neural_bpb:+.6f}")
        hybrid.print_stats()

    base_model.train()
    return val_loss, val_bpb
