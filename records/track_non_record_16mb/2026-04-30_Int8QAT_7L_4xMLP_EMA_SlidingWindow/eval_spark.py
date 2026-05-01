#!/usr/bin/env python3
"""Post-training evaluation for train_gpt_spark.py checkpoints.

Supports:
  - Chunked eval (standard, matches training script)
  - Sliding window eval (stride < seq_len for richer context)
  - Token-level N-gram backoff cache mixture

Usage:
  python eval_spark.py --checkpoint final_model_seed42.int8.ptz [--stride 64] [--alpha 0.2] [--max-order 5]
"""
from __future__ import annotations

import argparse
import copy
import glob
import io
import math
import os
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Import everything we need from the training script ─────────────────────
from train_gpt_spark import (
    GPT,
    Hyperparameters,
    RMSNorm,
    CastedLinear,
    Block,
    CausalSelfAttention,
    MLP,
    build_sentencepiece_luts,
    dequantize_state_dict_int8,
    dequantize_mixed_int6,
    load_data_shard,
)

# ── Token-level N-gram cache ────────────────────────────────────────────────

class TokenNGramCache:
    """Multi-order n-gram cache operating at the BPE token level.

    Optimised for throughput:
    - counts stored as int32 numpy arrays (not re-allocated per call)
    - pre-allocated float32 mixing buffer
    - stays on CPU; returns a numpy log-prob array
    """

    def __init__(self, vocab_size: int = 1024, max_order: int = 5,
                 smoothing: float = 1e-6):
        self.vocab_size = vocab_size
        self.max_order = max_order
        self.smoothing = smoothing
        self.history: list[int] = []
        # counts[n][context_tuple] → int32 array of length vocab_size
        V = vocab_size
        self.counts: list[dict] = [
            defaultdict(lambda: np.zeros(V, dtype=np.int32))
            for _ in range(max_order + 1)   # index 0 unused
        ]
        # reusable buffers (avoid per-call allocation)
        self._probs   = np.empty(V, dtype=np.float64)
        self._ngram_p = np.empty(V, dtype=np.float64)
        self._log_mix = np.empty(V, dtype=np.float64)
        self._uniform = np.full(V, 1.0 / V, dtype=np.float64)

    def update(self, token: int) -> None:
        self.history.append(token)
        for n in range(1, self.max_order + 1):
            if len(self.history) >= n:
                ctx = tuple(self.history[-n:-1])   # last n-1 tokens (context)
                self.counts[n][ctx][token] += 1

    def score(self, neural_logits_np: np.ndarray, target: int, alpha: float) -> float:
        """Score target token; returns NLL in nats (fast, CPU-only).

        neural_logits_np: float32/64 numpy array of shape [vocab_size], raw logits.
        Returns: NLL in nats.
        """
        V = self.vocab_size
        H = len(self.history)

        # --- n-gram probs (reuse self._probs buffer) ---
        np.copyto(self._probs, self._uniform)
        inv_ord1 = 1.0 / (self.max_order + 1)
        for n in range(1, self.max_order + 1):
            if H < n - 1:
                continue
            ctx = tuple(self.history[-(n - 1):]) if n > 1 else ()
            cnt = self.counts[n].get(ctx)
            if cnt is None:
                continue
            cnt_sum = cnt.sum()
            if cnt_sum == 0:
                continue
            total = cnt_sum + self.smoothing * V
            np.add(cnt, self.smoothing, out=self._ngram_p)
            self._ngram_p /= total
            w = n * inv_ord1
            self._probs *= (1.0 - w)
            self._probs += w * self._ngram_p

        # --- neural probs (softmax on CPU, subtract max for stability) ---
        logits = neural_logits_np.astype(np.float64, copy=False)
        logits = logits - logits.max()
        exp_l = np.exp(logits)
        neural_probs = exp_l / exp_l.sum()

        # --- mix ---
        mixed = (1.0 - alpha) * neural_probs + alpha * self._probs
        nll = -math.log(max(mixed[target], 1e-40))
        return nll

    # keep mix_with_neural for compatibility (used in tests)
    def mix_with_neural(self, neural_logits: torch.Tensor, alpha: float) -> torch.Tensor:
        ngram_probs = torch.from_numpy(self._probs.copy()).float()
        neural_probs = torch.softmax(neural_logits.float(), dim=-1)
        mixed = (1.0 - alpha) * neural_probs + alpha * ngram_probs
        return torch.log(mixed.clamp_min(1e-40))

    def reset(self) -> None:
        self.history.clear()
        for n in range(1, self.max_order + 1):
            self.counts[n].clear()


# ── Model loading ───────────────────────────────────────────────────────────

def load_gpt_from_int8(ckpt_path: str, device: torch.device) -> tuple[GPT, Hyperparameters]:
    """Load an int8 checkpoint and reconstruct the float GPT model."""
    with open(ckpt_path, "rb") as f:
        blob = f.read()

    # Detect format: int6+lzma vs int8+zlib
    import lzma
    try:
        raw = lzma.decompress(blob)
        quant_state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        is_int6 = True
    except Exception:
        raw = zlib.decompress(blob)
        quant_state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        is_int6 = False

    args = Hyperparameters()
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    )

    if is_int6:
        # Need a template state dict for dtype info
        template_sd = model.state_dict()
        sd = dequantize_mixed_int6(quant_state["w"], quant_state["m"], template_sd)
    else:
        sd = dequantize_state_dict_int8(quant_state)

    model.load_state_dict(sd, strict=True)
    model.eval()
    model.to(device)
    return model, args


def get_logits(model: GPT, input_ids: Tensor) -> Tensor:
    """Run forward pass and return raw logits (not loss)."""
    x = model.tok_emb(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    skips: list[Tensor] = []
    for i in range(model.num_encoder_layers):
        x = model.blocks[i](x, x0)
        skips.append(x)
    for i in range(model.num_decoder_layers):
        if skips:
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        x = model.blocks[model.num_encoder_layers + i](x, x0)
    x = model.final_norm(x)  # [B, T, D]
    if model.tie_embeddings:
        logits_proj = F.linear(x, model.tok_emb.weight)
    else:
        logits_proj = model.lm_head(x)
    logits = model.logit_softcap * torch.tanh(logits_proj / model.logit_softcap)
    return logits.float()  # [B, T, V]


# ── Validation data helpers ─────────────────────────────────────────────────

def load_val_tokens(args: Hyperparameters) -> Tensor:
    files = sorted(glob.glob(args.val_files))
    if not files:
        raise FileNotFoundError(f"No val files: {args.val_files}")
    tokens = torch.cat([load_data_shard(Path(p)) for p in files]).contiguous()
    usable = ((tokens.numel() - 1) // args.train_seq_len) * args.train_seq_len
    return tokens[: usable + 1]


def load_tokenizer_luts(args: Hyperparameters, device: torch.device):
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    return build_sentencepiece_luts(sp, args.vocab_size, device)


# ── Chunked eval (matches training script, optionally with n-gram) ──────────

def chunked_eval(
    model: GPT,
    val_tokens: Tensor,
    args: Hyperparameters,
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    ngram_alpha: float = 0.0,
    ngram_max_order: int = 5,
) -> float:
    """Standard chunked BPB eval, optionally mixing with n-gram cache."""
    seq_len = args.train_seq_len
    tokens = val_tokens.to(device=device, dtype=torch.int64)
    total_seqs = (tokens.numel() - 1) // seq_len

    use_ngram = ngram_alpha > 0.0
    cache = TokenNGramCache(args.vocab_size, ngram_max_order) if use_ngram else None

    total_nll_bits = 0.0
    total_bytes = 0.0

    model.eval()
    with torch.inference_mode():
        for s in range(total_seqs):
            raw_start = s * seq_len
            raw_end = raw_start + seq_len + 1
            chunk = tokens[raw_start:raw_end]
            x = chunk[:-1].unsqueeze(0)   # [1, T]
            y = chunk[1:]                  # [T]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = get_logits(model, x)[0]  # [T, V]

            if use_ngram:
                # Seed cache with x[0] (= tokens[raw_start]) before scoring
                if s == 0:
                    cache.update(x[0, 0].item())
                # Move logits to CPU numpy once per chunk
                logits_np = logits.cpu().numpy()  # [T, V]
                inv_ln2 = 1.0 / math.log(2)
                for t in range(seq_len):
                    target_tok = y[t].item()
                    nats = cache.score(logits_np[t], target_tok, ngram_alpha)
                    total_nll_bits += nats * inv_ln2
                    cache.update(target_tok)
            else:
                loss_each = F.cross_entropy(logits, y, reduction="none")
                total_nll_bits += (loss_each / math.log(2)).sum().item()

            prev_ids = x[0]
            tgt_ids = y
            tb = base_bytes_lut[tgt_ids].to(dtype=torch.float32)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.float32)
            total_bytes += tb.sum().item()

    bpb = total_nll_bits / total_bytes
    return bpb


# ── Sliding window eval ─────────────────────────────────────────────────────

def sliding_window_eval(
    model: GPT,
    val_tokens: Tensor,
    args: Hyperparameters,
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int = 64,
    ngram_alpha: float = 0.0,
    ngram_max_order: int = 5,
    neural_batch: int = 64,
) -> float:
    """Sliding window BPB eval: every scored token has full seq_len context.

    Without n-gram: fully batched on GPU (fast).
    With n-gram: batch neural passes, sequential n-gram scoring.
    """
    seq_len = args.train_seq_len
    tokens = val_tokens.to(device=device, dtype=torch.int64)
    T = tokens.numel()
    V = args.vocab_size

    use_ngram = ngram_alpha > 0.0
    cache = TokenNGramCache(V, ngram_max_order) if use_ngram else None
    abs_seen_up_to = -1

    total_nll_bits = 0.0
    total_bytes = 0.0
    inv_ln2 = 1.0 / math.log(2)

    positions = list(range(0, T - seq_len, stride))
    n_windows = len(positions)
    score_start = seq_len - stride  # constant for full windows
    log_every = max(1, n_windows // 20)

    model.eval()
    t_start = time.time()

    with torch.inference_mode():
        for batch_idx in range(0, n_windows, neural_batch):
            batch_pos = positions[batch_idx : batch_idx + neural_batch]
            B = len(batch_pos)

            # Build input batch [B, seq_len] using advanced indexing (fast)
            idx_base = torch.tensor(batch_pos, device=device).unsqueeze(1)  # [B, 1]
            offsets  = torch.arange(seq_len, device=device).unsqueeze(0)   # [1, seq_len]
            xs = tokens[idx_base + offsets]  # [B, seq_len]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_batch = get_logits(model, xs)  # [B, seq_len, V]

            if not use_ngram:
                # ── Fast fully-batched path ──────────────────────────────────
                # Score only the last `stride` positions of each window
                sl  = logits_batch[:, score_start:, :]            # [B, stride, V]
                tgt = xs[:, score_start + 1:]                      # [B, stride] input shift
                # targets are the NEXT tokens: tokens[pos+score_start+1..pos+seq_len]
                tgt = tokens[(idx_base + score_start + 1) + offsets[:, :stride]]  # [B, stride]
                loss_each = F.cross_entropy(
                    sl.reshape(-1, V), tgt.reshape(-1), reduction="none"
                ).reshape(B, stride)
                total_nll_bits += (loss_each * inv_ln2).sum().item()

                # Byte counts [B, stride]
                prev_ids = tokens[(idx_base + score_start) + offsets[:, :stride]]     # [B, stride]
                tgt_ids  = tokens[(idx_base + score_start + 1) + offsets[:, :stride]] # [B, stride]
                tb = base_bytes_lut[tgt_ids].float()
                tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).float()
                total_bytes += tb.sum().item()

            else:
                # ── N-gram path: sequential per-window ──────────────────────
                logits_np = logits_batch.cpu().numpy()  # [B, seq_len, V]

                for bi, pos in enumerate(batch_pos):
                    al = min(seq_len, T - pos - 1)  # actual_len
                    if al <= 0:
                        continue
                    ss = max(al - stride, 0)  # score_start for this window

                    log = logits_np[bi]
                    for local_t in range(ss, al):
                        abs_context = pos + local_t
                        abs_target  = pos + local_t + 1

                        if abs_context > abs_seen_up_to:
                            for t in range(abs_seen_up_to + 1, abs_context + 1):
                                cache.update(tokens[t].item())
                            abs_seen_up_to = abs_context

                        target_tok = tokens[abs_target].item()
                        nats = cache.score(log[local_t], target_tok, ngram_alpha)
                        total_nll_bits += nats * inv_ln2
                        cache.update(target_tok)
                        abs_seen_up_to = abs_target

                    # Byte counts (still sequential but fast tensor ops)
                    prev_ids = tokens[pos + ss : pos + al].long()
                    tgt_ids  = tokens[pos + ss + 1 : pos + al + 1].long()
                    tb = base_bytes_lut[tgt_ids].float()
                    tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).float()
                    total_bytes += tb.sum().item()

            # Progress
            done = batch_idx + B
            if done % log_every < B:
                elapsed = time.time() - t_start
                pct = done / n_windows * 100
                eta = elapsed / max(done, 1) * (n_windows - done)
                bpb_so_far = total_nll_bits / max(total_bytes, 1)
                print(f"  sliding {pct:5.1f}% ({done:,}/{n_windows:,})  "
                      f"bpb={bpb_so_far:.4f}  eta={eta/60:.1f}min", flush=True)

    return total_nll_bits / total_bytes


# ── TTT sliding window eval ─────────────────────────────────────────────────

def ttt_sliding_window_eval(
    model: GPT,
    val_tokens: Tensor,
    args: Hyperparameters,
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int = 64,
    ngram_alpha: float = 0.0,
    ngram_max_order: int = 5,
    ttt_lr: float = 1e-3,
    ttt_steps: int = 1,
    ttt_reset: bool = False,
    ttt_train_window: int = 512,
) -> float:
    """Score-first TTT + sliding window eval.

    Before scoring each stride, runs ttt_steps SGD steps on the preceding
    ttt_train_window tokens using only: last transformer block, final_norm,
    and lm_head (if untied). No future tokens are seen — fully causal.
    """
    seq_len = args.train_seq_len
    tokens = val_tokens.to(device=device, dtype=torch.int64)
    T = tokens.numel()
    V = args.vocab_size

    use_ngram = ngram_alpha > 0.0
    cache = TokenNGramCache(V, ngram_max_order) if use_ngram else None
    abs_seen_up_to = -1

    base_state = copy.deepcopy(model.state_dict())

    # Params to adapt: last block + final_norm + lm_head (if untied)
    ttt_params = (
        list(model.blocks[-1].parameters()) +
        list(model.final_norm.parameters()) +
        (list(model.lm_head.parameters()) if model.lm_head is not None else [])
    )

    total_nll_bits = 0.0
    total_bytes = 0.0
    inv_ln2 = 1.0 / math.log(2)

    positions = list(range(0, T - seq_len, stride))
    n_windows = len(positions)
    score_start = seq_len - stride
    log_every = max(1, n_windows // 20)

    t_start = time.time()

    for win_idx, pos in enumerate(positions):
        # Adapt on history before the scored region
        history_end = pos + score_start
        if history_end >= 2:
            train_start = max(0, history_end - ttt_train_window)
            hist_seq = tokens[train_start:history_end].unsqueeze(0)  # [1, L]
            if hist_seq.size(1) >= 2:
                model.train()
                opt = torch.optim.SGD(ttt_params, lr=ttt_lr)
                for _ in range(ttt_steps):
                    opt.zero_grad()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = get_logits(model, hist_seq[:, :-1])
                    loss = F.cross_entropy(
                        logits.reshape(-1, V),
                        hist_seq[:, 1:].reshape(-1),
                    )
                    loss.backward()
                    opt.step()

        # Score the stride with adapted model
        al = min(seq_len, T - pos - 1)
        if al <= 0:
            if ttt_reset:
                model.load_state_dict(base_state)
            continue
        ss = max(al - stride, 0)

        model.eval()
        x_window = tokens[pos : pos + al].unsqueeze(0)  # [1, al]
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                log_window = get_logits(model, x_window)[0]  # [al, V]

        if use_ngram:
            log_np = log_window.cpu().numpy()
            for local_t in range(ss, al):
                abs_context = pos + local_t
                abs_target = pos + local_t + 1
                if abs_context > abs_seen_up_to:
                    for t in range(abs_seen_up_to + 1, abs_context + 1):
                        cache.update(tokens[t].item())
                    abs_seen_up_to = abs_context
                target_tok = tokens[abs_target].item()
                nats = cache.score(log_np[local_t], target_tok, ngram_alpha)
                total_nll_bits += nats * inv_ln2
                cache.update(target_tok)
                abs_seen_up_to = abs_target
        else:
            tgt = tokens[pos + ss + 1 : pos + al + 1]
            sl = log_window[ss:al]
            loss_each = F.cross_entropy(sl, tgt, reduction="none")
            total_nll_bits += (loss_each * inv_ln2).sum().item()

        prev_ids = tokens[pos + ss : pos + al].long()
        tgt_ids = tokens[pos + ss + 1 : pos + al + 1].long()
        tb = base_bytes_lut[tgt_ids].float()
        tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).float()
        total_bytes += tb.sum().item()

        if ttt_reset:
            model.load_state_dict(base_state)

        if (win_idx + 1) % log_every == 0:
            elapsed = time.time() - t_start
            pct = (win_idx + 1) / n_windows * 100
            eta = elapsed / max(win_idx + 1, 1) * (n_windows - win_idx - 1)
            bpb_so_far = total_nll_bits / max(total_bytes, 1)
            print(f"  ttt {pct:5.1f}% ({win_idx+1:,}/{n_windows:,})  "
                  f"bpb={bpb_so_far:.4f}  eta={eta/60:.1f}min", flush=True)

    model.load_state_dict(base_state)
    return total_nll_bits / total_bytes


# ── Alpha sweep ─────────────────────────────────────────────────────────────

def alpha_sweep(
    model: GPT,
    val_tokens: Tensor,
    args: Hyperparameters,
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    alphas: list[float],
    max_orders: list[int],
    stride: int = 64,
    out_path: str = "logs/alpha_sweep_summary.txt",
    max_sweep_tokens: int = 0,
) -> tuple[float, float, float]:
    """Sweep (alpha, max_order) and report best BPB. Returns (best_alpha, best_order, best_bpb).

    max_sweep_tokens: if > 0, truncate val_tokens for the sweep to save time.
    The truncated set is large enough for reliable alpha estimation.
    """
    if max_sweep_tokens > 0 and val_tokens.numel() > max_sweep_tokens:
        sweep_tokens = val_tokens[:max_sweep_tokens + 1]
        print(f"  [sweep] using {max_sweep_tokens:,} tokens (of {val_tokens.numel():,}) for speed")
    else:
        sweep_tokens = val_tokens

    results = []
    for max_order in max_orders:
        for alpha in alphas:
            t0 = time.time()
            bpb = sliding_window_eval(
                model, sweep_tokens, args, device,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                stride=stride,
                ngram_alpha=alpha,
                ngram_max_order=max_order,
            )
            elapsed = time.time() - t0
            results.append((alpha, max_order, bpb, elapsed))
            print(f"  alpha={alpha:.2f} max_order={max_order}  BPB={bpb:.6f}  ({elapsed:.0f}s)")

    results.sort(key=lambda r: r[2])
    best_alpha, best_order, best_bpb, _ = results[0]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("alpha\tmax_order\tBPB\ttime_s\n")
        for alpha, max_order, bpb, t in results:
            f.write(f"{alpha:.2f}\t{max_order}\t{bpb:.6f}\t{t:.0f}\n")
        f.write(f"\nBEST: alpha={best_alpha:.2f} max_order={best_order} BPB={best_bpb:.6f}\n")
    print(f"\nSweep written to {out_path}")
    print(f"BEST: alpha={best_alpha:.2f}, max_order={best_order}, BPB={best_bpb:.6f}")

    return best_alpha, best_order, best_bpb


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stride", type=int, default=0,
                   help="Sliding window stride (0 = chunked only)")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="N-gram cache weight (0 = disabled)")
    p.add_argument("--max-order", type=int, default=5)
    p.add_argument("--sweep", action="store_true",
                   help="Run full alpha/order sweep after main eval")
    p.add_argument("--sweep-out", type=str, default="logs/alpha_sweep_summary.txt")
    p.add_argument("--sweep-tokens", type=int, default=2_000_000,
                   help="Max tokens used per sweep run (0=full val set)")
    # TTT options
    p.add_argument("--ttt", action="store_true",
                   help="Run sliding+ngram+TTT eval (score-first test-time training)")
    p.add_argument("--ttt-lr", type=float, default=1e-3)
    p.add_argument("--ttt-steps", type=int, default=1)
    p.add_argument("--ttt-reset", action="store_true",
                   help="Reset weights to base after each chunk (default: accumulate)")
    p.add_argument("--ttt-train-window", type=int, default=512,
                   help="Max history tokens for TTT adaptation per chunk")
    p.add_argument("--ttt-max-tokens", type=int, default=0,
                   help="Truncate val set to this many tokens for TTT (0=full). "
                        "Use 2000000 for a fast ~5min estimate on H100.")
    return p.parse_args()


def main():
    args_cli = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args_cli.checkpoint}")

    model, hp = load_gpt_from_int8(args_cli.checkpoint, device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    print("Loading validation tokens...")
    val_tokens = load_val_tokens(hp)
    print(f"Val tokens: {val_tokens.numel():,}")

    print("Loading tokenizer LUTs...")
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = load_tokenizer_luts(hp, device)

    # Step 2: chunked eval with n-gram (full val set)
    t0 = time.time()
    bpb_chunked = chunked_eval(
        model, val_tokens, hp, device,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        ngram_alpha=args_cli.alpha,
        ngram_max_order=args_cli.max_order,
    )
    t2 = time.time() - t0
    print(f"\n[Step 2] chunked+ngram(a={args_cli.alpha},ord={args_cli.max_order}) BPB={bpb_chunked:.6f}  ({t2:.0f}s)")

    bpb_sliding = None
    t3 = 0.0
    if args_cli.stride > 0:
        # Step 3: sliding window eval with n-gram (full val set)
        t0 = time.time()
        bpb_sliding = sliding_window_eval(
            model, val_tokens, hp, device,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args_cli.stride,
            ngram_alpha=args_cli.alpha,
            ngram_max_order=args_cli.max_order,
        )
        t3 = time.time() - t0
        delta = bpb_sliding - bpb_chunked
        print(f"[Step 3] sliding(stride={args_cli.stride})+ngram  BPB={bpb_sliding:.6f}  delta={delta:+.6f}  ({t3:.0f}s)")

    bpb_ttt = None
    t4 = 0.0
    if args_cli.ttt:
        stride_ttt = args_cli.stride if args_cli.stride > 0 else 64
        ttt_tokens = val_tokens
        ttt_label = "full"
        if args_cli.ttt_max_tokens > 0 and val_tokens.numel() > args_cli.ttt_max_tokens:
            ttt_tokens = val_tokens[: args_cli.ttt_max_tokens + 1]
            ttt_label = f"{args_cli.ttt_max_tokens//1_000_000}M"
        print(f"\n[Step TTT] sliding(stride={stride_ttt})+ngram+TTT "
              f"(lr={args_cli.ttt_lr}, steps={args_cli.ttt_steps}, "
              f"reset={args_cli.ttt_reset}, window={args_cli.ttt_train_window}, "
              f"tokens={ttt_label})...")
        t0 = time.time()
        bpb_ttt = ttt_sliding_window_eval(
            model, ttt_tokens, hp, device,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=stride_ttt,
            ngram_alpha=args_cli.alpha,
            ngram_max_order=args_cli.max_order,
            ttt_lr=args_cli.ttt_lr,
            ttt_steps=args_cli.ttt_steps,
            ttt_reset=args_cli.ttt_reset,
            ttt_train_window=args_cli.ttt_train_window,
        )
        t4 = time.time() - t0
        base = bpb_sliding if bpb_sliding is not None else bpb_chunked
        print(f"[Step TTT] sliding+ngram+TTT [{ttt_label}]  BPB={bpb_ttt:.6f}  "
              f"delta={bpb_ttt - base:+.6f}  ({t4:.0f}s)")

    best_a = args_cli.alpha
    best_o = args_cli.max_order
    best_sweep_bpb = bpb_sliding if bpb_sliding is not None else bpb_chunked

    if args_cli.sweep:
        print(f"\n[Step 4] Alpha sweep (sliding window, stride=64, up to {args_cli.sweep_tokens:,} tokens)...")
        alphas = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
        max_orders = [5, 7, 9]
        best_a, best_o, best_sweep_bpb = alpha_sweep(
            model, val_tokens, hp, device,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            alphas=alphas,
            max_orders=max_orders,
            stride=64,
            out_path=args_cli.sweep_out,
            max_sweep_tokens=args_cli.sweep_tokens,
        )

    # Artifact size check
    ckpt_bytes = os.path.getsize(args_cli.checkpoint)
    print(f"\n[Step 5] Checkpoint compressed size: {ckpt_bytes:,} bytes ({ckpt_bytes/1e6:.2f} MB)")
    if ckpt_bytes > 16_000_000:
        print("  WARNING: exceeds 16MB limit!")
    else:
        print("  OK: under 16MB limit")

    # Determine best BPB across all variants
    candidates = [bpb_chunked]
    if bpb_sliding is not None:
        candidates.append(bpb_sliding)
    if args_cli.sweep:
        candidates.append(best_sweep_bpb)
    if bpb_ttt is not None:
        candidates.append(bpb_ttt)
    best_bpb = min(candidates)

    # Summary
    print(f"\n{'='*60}")
    print(f"EVAL SUMMARY")
    print(f"  Checkpoint:          {args_cli.checkpoint}")
    print(f"  Step 2 BPB (chunked+ngram): {bpb_chunked:.6f}")
    if bpb_sliding is not None:
        print(f"  Step 3 BPB (sliding+ngram): {bpb_sliding:.6f}  (delta={bpb_sliding-bpb_chunked:+.6f})")
    if args_cli.sweep:
        print(f"  Best sweep:          alpha={best_a:.2f}, order={best_o}, BPB={best_sweep_bpb:.6f}")
    if bpb_ttt is not None:
        print(f"  TTT BPB (slide+ngram+TTT): {bpb_ttt:.6f}  (delta={bpb_ttt-bpb_chunked:+.6f})")
    print(f"  BEST BPB (all variants): {best_bpb:.6f}")
    print(f"  Artifact size:       {ckpt_bytes/1e6:.2f} MB")
    print(f"{'='*60}")

    return {
        "bpb_chunked_ngram": bpb_chunked,
        "bpb_sliding_ngram": bpb_sliding,
        "bpb_ttt": bpb_ttt,
        "best_bpb": best_bpb,
        "best_alpha": best_a,
        "best_order": best_o,
        "artifact_mb": ckpt_bytes / 1e6,
    }


if __name__ == "__main__":
    main()
