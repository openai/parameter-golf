#!/usr/bin/env python3
"""
Multi-Epoch SGD Test-Time Training (TTT) Evaluation
====================================================
Implements the PR #461 approach: SGD with momentum, 3 epochs per 32K chunk,
cosine LR decay across chunks, freeze first 2 blocks.

Expected improvement: -0.0165 BPB over baseline sliding window eval.

PROTOCOL (legal score-first TTT):
  For each 32K-token chunk:
    1. SCORE: model.eval(), inference_mode, sliding window eval, record NLL
    2. TRAIN: model.train(), SGD(lr=0.002, momentum=0.9), 3 epochs on chunk
    3. Cosine LR decay: lr = base_lr * 0.5 * (1 + cos(pi * chunk_idx / (num_chunks - 1)))
  Freeze first 2 blocks during TTT training.
  Gradient clipping = 1.0.

CRITICAL: SGD TTT does NOT work with full GPTQ (+0.03 BPB regression).
It works with simple int6 per-row quantization (quantize -> dequantize -> float).

USAGE:
  # On H100 with trained model:
  python3 sgd_ttt_eval.py --model-path final_model.int6.ptz

  # With simple int6 (default, recommended for TTT):
  python3 sgd_ttt_eval.py --model-path final_model.int6.ptz --quant simple_int6

  # Skip TTT (baseline sliding window only):
  python3 sgd_ttt_eval.py --model-path final_model.int6.ptz --no-ttt
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
import io
import math
import os
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Import model architecture from Clark's train_gpt (record_train_gpt.py)
# ---------------------------------------------------------------------------
# We import lazily to allow the script to be placed anywhere.
# Set REPO_DIR to the directory containing record_train_gpt.py if needed.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = os.environ.get("REPO_DIR", str(_SCRIPT_DIR))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

# Try importing from record_train_gpt first, fall back to train_gpt
try:
    import record_train_gpt as tg
except ImportError:
    import train_gpt as tg

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SGD TTT Eval (PR #461)")
    p.add_argument("--model-path", type=str, default="final_model.int6.ptz",
                   help="Path to quantized model (.ptz) or raw checkpoint (.pt)")
    p.add_argument("--data-dir", type=str, default=os.environ.get("DATA_DIR", "./data/"),
                   help="Data directory containing datasets/ and tokenizers/")
    p.add_argument("--quant", type=str, default="simple_int6",
                   choices=["simple_int6", "gptq", "none"],
                   help="Quantization mode: simple_int6 (default, best for TTT), "
                        "gptq (worse with TTT), none (raw .pt checkpoint)")

    # TTT hyperparameters (PR #461 defaults)
    p.add_argument("--no-ttt", action="store_true", help="Skip TTT, baseline eval only")
    p.add_argument("--ttt-lr", type=float, default=0.002, help="Base TTT learning rate")
    p.add_argument("--ttt-momentum", type=float, default=0.9, help="SGD momentum")
    p.add_argument("--ttt-epochs", type=int, default=3, help="Epochs per chunk")
    p.add_argument("--ttt-chunk-size", type=int, default=32768,
                   help="Chunk size in tokens (32K)")
    p.add_argument("--ttt-freeze-blocks", type=int, default=2,
                   help="Number of initial blocks to freeze during TTT")
    p.add_argument("--ttt-grad-clip", type=float, default=1.0, help="Gradient clip norm")
    p.add_argument("--ttt-batch-seqs", type=int, default=32,
                   help="Batch size (sequences) for TTT training")

    # Eval parameters
    p.add_argument("--eval-stride", type=int, default=64,
                   help="Sliding window stride for scoring")
    p.add_argument("--eval-seq-len", type=int, default=2048,
                   help="Sequence length for evaluation")
    p.add_argument("--eval-batch-seqs", type=int, default=32,
                   help="Batch size (sequences) for sliding window scoring")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(args: argparse.Namespace, h: tg.Hyperparameters) -> nn.Module:
    """Load model from quantized or raw checkpoint, return in bfloat16."""
    model_path = Path(args.model_path)
    if not model_path.exists():
        # Try in data dir
        alt = Path(args.data_dir) / args.model_path
        if alt.exists():
            model_path = alt
        else:
            raise FileNotFoundError(
                f"Model not found at {args.model_path} or {alt}")

    model = tg.GPT(h).to(DEVICE).bfloat16()

    # Regenerate frozen FC layers if selective freeze is enabled
    if hasattr(tg, '_apply_selective_freeze'):
        tg._apply_selective_freeze(model)
    if hasattr(tg, 'restore_fp32_params'):
        tg.restore_fp32_params(model)

    if args.quant == "none" or model_path.suffix == ".pt":
        # Raw checkpoint
        print(f"Loading raw checkpoint from {model_path}")
        state = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state, strict=False)
    else:
        # Quantized checkpoint (.ptz) - decompress and dequantize
        print(f"Loading quantized model from {model_path}")
        with open(model_path, "rb") as f:
            quant_blob = f.read()

        # Decompress
        compressor = getattr(h, 'compressor', 'brotli')
        raw_bytes = tg._decompress(quant_blob, compressor)
        quant_state = torch.load(io.BytesIO(raw_bytes), map_location="cpu")

        # Dequantize
        sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        deq_state = tg.dequantize_mixed_int6(
            quant_state["w"], quant_state["m"], sd_cpu)

        strict = os.environ.get("SELECTIVE_FREEZE", "0") not in ("1", "true")
        model.load_state_dict(deq_state, strict=strict)
        print(f"  Dequantized from int6 to bfloat16")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    return model


# ---------------------------------------------------------------------------
# Byte counting LUTs (for BPB computation)
# ---------------------------------------------------------------------------
def load_byte_luts(h: tg.Hyperparameters, device: torch.device):
    """Load SentencePiece tokenizer and build byte-counting lookup tables."""
    sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
    base_lut, space_lut, boundary_lut = tg.build_sentencepiece_luts(
        sp, h.vocab_size, device)
    return base_lut, space_lut, boundary_lut


def count_bytes(prev_ids: Tensor, tgt_ids: Tensor,
                base_lut: Tensor, space_lut: Tensor,
                boundary_lut: Tensor) -> float:
    """Count UTF-8 bytes for BPB computation using SentencePiece LUTs."""
    tb = base_lut[tgt_ids].to(torch.float64)
    tb += (space_lut[tgt_ids] & ~boundary_lut[prev_ids]).to(torch.float64)
    return tb.sum().item()


# ---------------------------------------------------------------------------
# Sliding window scoring (score phase of TTT)
# ---------------------------------------------------------------------------
def sliding_window_score(
    model: nn.Module,
    tokens: Tensor,
    seq_len: int,
    stride: int,
    vocab_size: int,
    batch_seqs: int,
    base_lut: Tensor,
    space_lut: Tensor,
    boundary_lut: Tensor,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Score tokens using sliding window evaluation.
    Each token is scored with maximum context.
    Returns (total_nll, total_tokens_scored, total_bytes).
    """
    total_tokens = tokens.numel() - 1
    context_size = seq_len - stride

    # Generate window start positions
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if ws + context_size < total_tokens]

    total_nll = 0.0
    total_scored = 0
    total_bytes = 0.0

    with torch.inference_mode():
        for bi in range(0, len(window_starts), batch_seqs):
            batch_ws = window_starts[bi:bi + batch_seqs]
            bsz = len(batch_ws)

            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []

            for i, ws in enumerate(batch_ws):
                we = min(ws + seq_len, total_tokens)
                wlen = we - ws
                wlens.append(wlen)
                chunk = tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x_batch)

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                # First window scores from position 0; others from context_size
                s = 0 if ws == 0 else context_size
                scored_nll = nll[i, s:wlen].to(torch.float64)
                total_nll += scored_nll.sum().item()
                total_scored += wlen - s

                # Byte counting
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                total_bytes += count_bytes(prev, tgt, base_lut, space_lut, boundary_lut)

    return total_nll, total_scored, total_bytes


# ---------------------------------------------------------------------------
# Chunk-aware sliding window scoring
# ---------------------------------------------------------------------------
def assign_windows_to_chunks(
    total_tokens: int,
    seq_len: int,
    stride: int,
    chunk_size: int,
) -> list[list[int]]:
    """
    Assign each sliding window to the chunk containing its scored region.
    A window's scored region starts at max(ws, context_size) for ws > 0.
    We assign based on where the scored tokens fall.
    Returns a list of lists: chunk_windows[chunk_idx] = [window_start, ...].
    """
    context_size = seq_len - stride
    n_chunks = math.ceil(total_tokens / chunk_size)

    # Pre-compute chunk boundaries
    chunk_windows: list[list[int]] = [[] for _ in range(n_chunks)]

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if ws + context_size < total_tokens]

    for ws in window_starts:
        # The scored region of this window
        scored_start = ws if ws == 0 else ws + context_size
        # Assign to the chunk containing scored_start
        chunk_idx = min(scored_start // chunk_size, n_chunks - 1)
        chunk_windows[chunk_idx].append(ws)

    return chunk_windows


def score_chunk_windows(
    model: nn.Module,
    tokens: Tensor,
    window_starts: list[int],
    seq_len: int,
    stride: int,
    vocab_size: int,
    batch_seqs: int,
    base_lut: Tensor,
    space_lut: Tensor,
    boundary_lut: Tensor,
    device: torch.device,
) -> tuple[float, int, float]:
    """
    Score a specific set of sliding windows. Returns (nll_sum, tokens_scored, bytes).
    """
    if not window_starts:
        return 0.0, 0, 0.0

    total_tokens = tokens.numel() - 1
    context_size = seq_len - stride

    nll_sum = 0.0
    tok_scored = 0
    byte_sum = 0.0

    with torch.inference_mode():
        for bi in range(0, len(window_starts), batch_seqs):
            batch_ws = window_starts[bi:bi + batch_seqs]
            bsz = len(batch_ws)

            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []

            for i, ws in enumerate(batch_ws):
                we = min(ws + seq_len, total_tokens)
                wlen = we - ws
                wlens.append(wlen)
                chunk = tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x_batch)

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else context_size
                scored_nll = nll[i, s:wlen].to(torch.float64)
                nll_sum += scored_nll.sum().item()
                tok_scored += wlen - s

                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                byte_sum += count_bytes(prev, tgt, base_lut, space_lut, boundary_lut)

    return nll_sum, tok_scored, byte_sum


# ---------------------------------------------------------------------------
# TTT training on a chunk
# ---------------------------------------------------------------------------
def train_on_chunk(
    model: nn.Module,
    tokens: Tensor,
    chunk_start: int,
    chunk_end: int,
    ttt_params: list[nn.Parameter],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    grad_clip: float,
    batch_seqs: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> float:
    """
    Train on a chunk for multiple epochs using SGD.
    Splits chunk into seq_len-sized sequences, batched.
    Returns average training loss of final epoch.
    """
    model.train()

    # Extract chunk tokens
    chunk_tokens = tokens[chunk_start:min(chunk_end + 1, tokens.numel())].to(
        dtype=torch.int64, device=device)
    chunk_len = chunk_tokens.numel() - 1  # -1 for target offset

    if chunk_len < 2:
        return 0.0

    # Split chunk into sequences of seq_len
    n_seqs = chunk_len // seq_len
    if n_seqs == 0:
        # Chunk smaller than seq_len: use what we have
        n_seqs = 1
        actual_len = chunk_len
    else:
        actual_len = seq_len

    # Build x, y tensors for the chunk
    x_all = []
    y_all = []
    for si in range(n_seqs):
        start = si * seq_len
        end = start + actual_len
        if end + 1 > chunk_tokens.numel():
            break
        x_all.append(chunk_tokens[start:end])
        y_all.append(chunk_tokens[start + 1:end + 1])

    if not x_all:
        return 0.0

    x_all = torch.stack(x_all)  # (n_seqs, seq_len)
    y_all = torch.stack(y_all)  # (n_seqs, seq_len)
    n_seqs = x_all.shape[0]

    last_epoch_loss = 0.0
    for epoch in range(epochs):
        # Shuffle sequence order each epoch
        perm = torch.randperm(n_seqs)
        epoch_loss = 0.0
        epoch_tokens = 0

        for bi in range(0, n_seqs, batch_seqs):
            batch_idx = perm[bi:bi + batch_seqs]
            x_batch = x_all[batch_idx]
            y_batch = y_all[batch_idx]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x_batch)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="mean",
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ttt_params, grad_clip)
            optimizer.step()

            batch_tok = y_batch.numel()
            epoch_loss += loss.item() * batch_tok
            epoch_tokens += batch_tok

        if epoch_tokens > 0:
            last_epoch_loss = epoch_loss / epoch_tokens

    return last_epoch_loss


# ---------------------------------------------------------------------------
# Main SGD TTT loop
# ---------------------------------------------------------------------------
def sgd_ttt_eval(
    model: nn.Module,
    val_tokens: Tensor,
    args: argparse.Namespace,
    h: tg.Hyperparameters,
    base_lut: Tensor,
    space_lut: Tensor,
    boundary_lut: Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """
    Full SGD TTT evaluation following PR #461.

    Algorithm:
      1. Assign sliding windows to 32K-token chunks based on scored region
      2. For each chunk:
         a. SCORE: model.eval(), inference_mode, score the chunk's windows
         b. TRAIN: model.train(), SGD update, 3 epochs
         c. Apply cosine LR decay
      3. Freeze first 2 blocks throughout

    Returns (final_loss, final_bpb).
    """
    t0 = time.time()

    # --- Setup: freeze first N blocks ---
    freeze_n = args.ttt_freeze_blocks
    frozen_params = set()
    for i in range(min(freeze_n, len(model.blocks))):
        for p in model.blocks[i].parameters():
            p.requires_grad = False
            frozen_params.add(id(p))
    print(f"  Froze first {freeze_n} blocks")

    # Collect trainable parameters
    ttt_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in ttt_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} parameters")

    # --- Setup: SGD optimizer ---
    optimizer = torch.optim.SGD(
        ttt_params,
        lr=args.ttt_lr,
        momentum=args.ttt_momentum,
    )

    # --- Assign windows to chunks ---
    total_tokens = val_tokens.numel() - 1
    chunk_size = args.ttt_chunk_size
    n_chunks = math.ceil(total_tokens / chunk_size)

    chunk_windows = assign_windows_to_chunks(
        total_tokens, args.eval_seq_len, args.eval_stride, chunk_size)

    # Sanity check: count total windows
    total_windows = sum(len(cw) for cw in chunk_windows)
    print(f"  Chunks: {n_chunks}, Total windows: {total_windows}, "
          f"Chunk size: {chunk_size:,} tokens")

    # --- Main loop ---
    overall_nll = 0.0
    overall_scored = 0
    overall_bytes = 0.0
    current_lr = args.ttt_lr

    for ci in range(n_chunks):
        chunk_start = ci * chunk_size
        chunk_end = min((ci + 1) * chunk_size, total_tokens)
        windows = chunk_windows[ci]

        # === STEP 1: SCORE (eval mode, inference_mode) ===
        model.eval()
        chunk_nll, chunk_scored, chunk_bytes = score_chunk_windows(
            model, val_tokens, windows,
            args.eval_seq_len, args.eval_stride, h.vocab_size,
            args.eval_batch_seqs,
            base_lut, space_lut, boundary_lut, device,
        )

        overall_nll += chunk_nll
        overall_scored += chunk_scored
        overall_bytes += chunk_bytes

        # === STEP 2: TRAIN on scored chunk (AFTER scoring) ===
        # Skip training on the last chunk (no future tokens to benefit)
        if ci < n_chunks - 1:
            # Cosine LR decay across chunks
            if n_chunks > 1:
                cos_decay = 0.5 * (1.0 + math.cos(math.pi * ci / (n_chunks - 1)))
            else:
                cos_decay = 1.0
            current_lr = args.ttt_lr * cos_decay

            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            train_loss = train_on_chunk(
                model, val_tokens,
                chunk_start, chunk_end,
                ttt_params, optimizer,
                args.ttt_epochs, args.ttt_grad_clip,
                args.ttt_batch_seqs, args.eval_seq_len,
                h.vocab_size, device,
            )

        # --- Progress logging ---
        if (ci + 1) % 5 == 0 or ci == 0 or ci == n_chunks - 1:
            if overall_scored > 0 and overall_bytes > 0:
                running_loss = overall_nll / overall_scored
                running_bpb = (running_loss / math.log(2)) * (overall_scored / overall_bytes)
                lr_str = f" lr={current_lr:.6f}" if ci < n_chunks - 1 else ""
                print(f"  TTT [{ci+1}/{n_chunks}] loss={running_loss:.6f} "
                      f"bpb={running_bpb:.6f} scored={overall_scored:,}{lr_str} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    # --- Unfreeze (restore requires_grad) ---
    for i in range(min(freeze_n, len(model.blocks))):
        for p in model.blocks[i].parameters():
            p.requires_grad = True

    # --- Final BPB ---
    if overall_scored == 0 or overall_bytes == 0:
        raise RuntimeError("No tokens scored during TTT eval")

    final_loss = overall_nll / overall_scored
    final_bpb = (final_loss / math.log(2)) * (overall_scored / overall_bytes)
    elapsed = time.time() - t0
    print(f"\n  TTT complete: loss={final_loss:.6f} bpb={final_bpb:.6f} "
          f"tokens={overall_scored:,} bytes={overall_bytes:.0f} time={elapsed:.0f}s")

    return final_loss, final_bpb


# ---------------------------------------------------------------------------
# Baseline eval (no TTT, just sliding window)
# ---------------------------------------------------------------------------
def baseline_sliding_window_eval(
    model: nn.Module,
    val_tokens: Tensor,
    args: argparse.Namespace,
    h: tg.Hyperparameters,
    base_lut: Tensor,
    space_lut: Tensor,
    boundary_lut: Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """Baseline sliding window eval without TTT."""
    print("\n=== Baseline Sliding Window Eval (no TTT) ===")
    t0 = time.time()

    model.eval()
    total_nll, total_scored, total_bytes = sliding_window_score(
        model, val_tokens,
        args.eval_seq_len, args.eval_stride, h.vocab_size,
        args.eval_batch_seqs,
        base_lut, space_lut, boundary_lut, device,
    )

    if total_scored == 0 or total_bytes == 0:
        raise RuntimeError("No tokens scored in baseline eval")

    loss = total_nll / total_scored
    bpb = (loss / math.log(2)) * (total_scored / total_bytes)
    elapsed = time.time() - t0
    print(f"  Baseline: loss={loss:.6f} bpb={bpb:.6f} "
          f"tokens={total_scored:,} bytes={total_bytes:.0f} time={elapsed:.0f}s")

    return loss, bpb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print(f"SGD TTT Eval (PR #461 approach)")
    print(f"  Device: {DEVICE}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model: {args.model_path}")
    print(f"  Quant: {args.quant}")

    if not args.no_ttt:
        print(f"  TTT: lr={args.ttt_lr} momentum={args.ttt_momentum} "
              f"epochs={args.ttt_epochs} chunk={args.ttt_chunk_size}")
        print(f"  TTT: freeze_blocks={args.ttt_freeze_blocks} "
              f"grad_clip={args.ttt_grad_clip} batch_seqs={args.ttt_batch_seqs}")
    print(f"  Eval: stride={args.eval_stride} seq_len={args.eval_seq_len} "
          f"batch_seqs={args.eval_batch_seqs}")
    print()

    # --- Build hyperparameters ---
    h = tg.Hyperparameters()
    # Override data dir if specified
    if args.data_dir:
        h.data_dir = args.data_dir
        h.datasets_dir = os.path.join(args.data_dir, 'datasets',
                                       f'fineweb10B_sp{h.vocab_size}')
        h.val_files = os.path.join(h.datasets_dir, 'fineweb_val_*.bin')
        h.tokenizer_path = os.path.join(args.data_dir, 'tokenizers',
                                         f'fineweb_{h.vocab_size}_bpe.model')
    h.eval_stride = args.eval_stride
    h.eval_seq_len = args.eval_seq_len

    # --- Load tokenizer + byte LUTs ---
    print("Loading tokenizer and byte LUTs...")
    base_lut, space_lut, boundary_lut = load_byte_luts(h, torch.device(DEVICE))

    # --- Load validation tokens ---
    print("Loading validation tokens...")
    val_tokens = tg.load_validation_tokens(h.val_files, h.eval_seq_len)
    print(f"  Validation tokens: {val_tokens.numel():,}")

    # --- Load model ---
    print("Loading model...")
    model = load_model(args, h)

    # --- Baseline eval ---
    baseline_loss, baseline_bpb = baseline_sliding_window_eval(
        model, val_tokens, args, h,
        base_lut, space_lut, boundary_lut, torch.device(DEVICE),
    )

    if args.no_ttt:
        print(f"\n{'='*60}")
        print(f"RESULTS (baseline only)")
        print(f"{'='*60}")
        print(f"  Baseline BPB: {baseline_bpb:.6f}")
        return

    # --- SGD TTT eval ---
    print(f"\n=== SGD TTT Eval (PR #461) ===")
    print(f"  SGD(lr={args.ttt_lr}, momentum={args.ttt_momentum})")
    print(f"  {args.ttt_epochs} epochs/chunk, {args.ttt_chunk_size:,} tokens/chunk")
    print(f"  Freeze first {args.ttt_freeze_blocks} blocks, grad_clip={args.ttt_grad_clip}")

    ttt_loss, ttt_bpb = sgd_ttt_eval(
        model, val_tokens, args, h,
        base_lut, space_lut, boundary_lut, torch.device(DEVICE),
    )

    # --- Results ---
    improvement_bpb = ttt_bpb - baseline_bpb
    improvement_pct = improvement_bpb / baseline_bpb * 100

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Baseline:  loss={baseline_loss:.6f}  bpb={baseline_bpb:.6f}")
    print(f"  SGD TTT:   loss={ttt_loss:.6f}  bpb={ttt_bpb:.6f}")
    print(f"  Delta BPB: {improvement_bpb:+.6f} ({improvement_pct:+.2f}%)")
    print(f"  Expected:  -0.0165 BPB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
