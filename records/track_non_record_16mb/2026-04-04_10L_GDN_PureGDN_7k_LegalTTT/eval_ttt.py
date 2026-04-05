#!/usr/bin/env python3
"""Standalone legal score-first TTT evaluation for GDN Hybrid models.

Loads a quantized artifact (.int6.ptz) or raw checkpoint (.pt),
runs legal score-first TTT (PR #461 recipe), and reports BPB.

Legal TTT: For each chunk of validation tokens, FIRST score all windows in
that chunk (inference_mode), THEN train on those already-scored tokens.
Every token is scored BEFORE any model update that could use its information.

Environment variables:
    ARTIFACT_PATH:    Path to .int6.ptz or .pt checkpoint
    ARCH_MODE:        Architecture config key (default: A)
    DATA_PATH:        Path to dataset dir
    TOKENIZER_PATH:   Path to sentencepiece model
    EVAL_SEQ_LEN:     Sequence length for eval (default: 1024)
    EVAL_STRIDE:      Sliding window stride (default: 64)

    TTT_ENABLED:      1 to run TTT, 0 to run plain sliding-window only (default: 1)
    TTT_LR:           SGD learning rate (default: 0.002)
    TTT_EPOCHS:       Training epochs per chunk (default: 3)
    TTT_CHUNK_TOKENS: Chunk size in tokens (default: 32768)
    TTT_FREEZE_BLOCKS: Number of early blocks to freeze (default: 2)
    TTT_MOMENTUM:     SGD momentum (default: 0.9)
    TTT_BATCH_SEQS:   Batch size in sequences for TTT training (default: 32)
    TTT_GRAD_CLIP:    Grad norm clipping (default: 1.0)
"""
from __future__ import annotations

import glob
import io
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
import zstandard
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from architectures import HybridGDN, CastedLinear
from configs import get_config


# ─── Config ──────────────────────────────────────────────────────────────────

class EvalConfig:
    arch_mode = os.environ.get("ARCH_MODE", "A")
    artifact_path = os.environ.get("ARTIFACT_PATH", "")
    data_path = os.environ.get("DATA_PATH", "../data/datasets/fineweb10B_sp1024")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "../data/tokenizers/fineweb_1024_bpe.model")
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 1024))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 128))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # TTT
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 2))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype=np.uint32, count=256)
    assert header[0] == 20240520, f"Bad magic: {header[0]}"
    assert header[1] in (1, 7), f"Bad version: {header[1]}"
    ntok = int(header[2])
    return torch.from_numpy(np.fromfile(file, dtype=np.uint16, offset=256 * 4)[:ntok].astype(np.int64))


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = sorted(glob.glob(pattern))
    assert files, f"No validation files matching: {pattern}"
    parts = [load_data_shard(Path(f)) for f in files]
    combined = torch.cat(parts)
    return combined[:((combined.numel() - 1) // seq_len) * seq_len + 1]


def build_sentencepiece_luts(sp, vocab_size, device):
    base_bytes = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    has_space = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    is_boundary = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        raw = piece.encode("utf-8")
        base_bytes[i] = len(raw)
        if piece.startswith("\u2581"):
            has_space[i] = True
            base_bytes[i] = len(piece[1:].encode("utf-8")) + 1
        if sp.is_control(i) or sp.is_unknown(i):
            is_boundary[i] = True
    return base_bytes, has_space, is_boundary


# ─── Quantization helpers (for loading .int6.ptz artifacts) ──────────────────

CONTROL_PATTERNS = (
    "resid_mix", "q_gain", "smear", "skip_weight", "attn_scale", "mlp_scale",
)


def dequantize_mixed(result: dict[str, Tensor], meta: dict[str, object],
                     template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info == "passthrough":
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out


# ─── Sliding Window Eval (no TTT) ───────────────────────────────────────────

def eval_val_sliding(
    model: nn.Module, val_tokens: Tensor,
    base_bytes_lut: Tensor, has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    device: torch.device, seq_len: int = 1024, stride: int = 64,
    batch_seqs: int = 128,
) -> tuple[float, float]:
    """Standard sliding window evaluation (no TTT)."""
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for bi in range(0, len(window_starts), batch_seqs):
            batch_ws = window_starts[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x_batch)

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1), reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb


# ─── Legal Score-First TTT ───────────────────────────────────────────────────

def eval_val_sliding_ttt(
    model: nn.Module, val_tokens: Tensor,
    base_bytes_lut: Tensor, has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    device: torch.device, cfg: EvalConfig,
) -> tuple[float, float]:
    """Legal score-first TTT (PR #461 recipe).

    For each chunk of validation data:
      Phase 1: Score all sliding windows in this chunk (inference_mode).
      Phase 2: Train on this chunk's tokens (already scored = legal).
    Every token is graded BEFORE any update that could leverage it.
    """
    seq_len = cfg.eval_seq_len
    stride = cfg.eval_stride
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = cfg.ttt_chunk_tokens

    # Pre-compute all window starts
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]

    # Assign each window to a chunk based on the first token it scores
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    print(f"ttt_sliding: chunks={num_chunks} chunk_tokens={ttt_chunk} "
          f"total_windows={len(window_starts)} stride={stride}")
    print(f"ttt_sliding: lr={cfg.ttt_lr} epochs={cfg.ttt_epochs} "
          f"freeze_blocks={cfg.ttt_freeze_blocks} momentum={cfg.ttt_momentum}")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Freeze first N blocks
    frozen_block_ids = set(range(min(cfg.ttt_freeze_blocks, len(model.blocks))))
    ttt_params = []
    for name, p in model.named_parameters():
        freeze = False
        for bi in frozen_block_ids:
            if f"blocks.{bi}." in name:
                freeze = True
                break
        if freeze:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            ttt_params.append(p)

    n_unfrozen = sum(p.numel() for p in ttt_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"ttt_sliding: unfrozen={n_unfrozen:,} frozen={n_frozen:,}")

    optimizer = torch.optim.SGD(ttt_params, lr=cfg.ttt_lr, momentum=cfg.ttt_momentum)
    t0 = time.perf_counter()

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)

        # --- Phase 1: SCORE this chunk's windows (inference_mode) ---
        model.eval()
        with torch.inference_mode():
            for bi in range(0, len(windows), cfg.eval_batch_seqs):
                batch_ws = windows[bi:bi + cfg.eval_batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens: list[int] = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk_tok = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model.forward_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1), reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt, prev = y_batch[i, s:wlen], x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        # --- Phase 2: TRAIN on this chunk (already scored = legal) ---
        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and cfg.ttt_epochs > 0:
            model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = cfg.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cos_lr
                for _ep in range(cfg.ttt_epochs):
                    for bs in range(0, chunk_seqs, cfg.ttt_batch_seqs):
                        be = min(bs + cfg.ttt_batch_seqs, chunk_seqs)
                        start_tok = chunk_start + bs * seq_len
                        end_tok = chunk_start + be * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = model(x, y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(ttt_params, cfg.ttt_grad_clip)
                        optimizer.step()

        if ci % 10 == 0 or ci == num_chunks - 1:
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
            print(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())

    # Restore all params to trainable
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()

    print(f"ttt_sliding: done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
          f"elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_model(artifact_path: str, config: dict, vocab_size: int, device: torch.device) -> nn.Module:
    """Load model from .int6.ptz artifact or raw .pt checkpoint."""
    model = HybridGDN(config, vocab_size).to(device).bfloat16()
    # CastedLinear and 1D params in fp32
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    for name, p in model.named_parameters():
        if p.ndim <= 1:
            p.data = p.data.float()

    if artifact_path.endswith(".ptz"):
        # Quantized artifact: decompress + dequantize
        print(f"Loading quantized artifact: {artifact_path}")
        with open(artifact_path, "rb") as f:
            blob = f.read()
        print(f"  Artifact size: {len(blob):,} bytes ({len(blob)/1024/1024:.2f} MB)")
        raw = zstandard.ZstdDecompressor().decompress(blob)
        quant_state = torch.load(io.BytesIO(raw), map_location="cpu")
        template_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        deq_state = dequantize_mixed(quant_state["w"], quant_state["m"], template_sd)
        model.load_state_dict(deq_state, strict=True)
    elif artifact_path.endswith(".pt"):
        print(f"Loading raw checkpoint: {artifact_path}")
        ckpt = torch.load(artifact_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            print(f"  Checkpoint step: {ckpt.get('step', '?')}, val_bpb: {ckpt.get('val_bpb', '?')}")
        else:
            # Bare state dict
            model.load_state_dict(ckpt, strict=True)
    else:
        raise ValueError(f"Unknown artifact format: {artifact_path}")

    return model


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    cfg = EvalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*80}")
    print(f"GDN Hybrid — Legal TTT Evaluation")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Arch: {cfg.arch_mode}")
    print(f"Artifact: {cfg.artifact_path}")
    print(f"TTT enabled: {cfg.ttt_enabled}")
    print(f"Eval seq_len: {cfg.eval_seq_len}, stride: {cfg.eval_stride}")

    if not cfg.artifact_path:
        print("ERROR: Set ARTIFACT_PATH to a .int6.ptz or .pt file")
        sys.exit(1)
    if not os.path.exists(cfg.artifact_path):
        print(f"ERROR: Artifact not found: {cfg.artifact_path}")
        sys.exit(1)

    # Load model
    config = get_config(cfg.arch_mode)
    model = load_model(cfg.artifact_path, config, cfg.vocab_size, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Load tokenizer and build BPB LUTs
    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, cfg.vocab_size, device)

    # Load validation tokens
    val_tokens = load_validation_tokens(cfg.val_files, cfg.eval_seq_len)
    print(f"Validation tokens: {val_tokens.numel():,}")

    # ─── Baseline: Plain sliding window eval (no TTT) ────────────────────
    print(f"\n{'='*60}")
    print("Phase 1: Plain sliding-window eval (no TTT)")
    print(f"{'='*60}")
    t_base = time.perf_counter()
    base_loss, base_bpb = eval_val_sliding(
        model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        device, seq_len=cfg.eval_seq_len, stride=cfg.eval_stride,
        batch_seqs=cfg.eval_batch_seqs,
    )
    base_elapsed = time.perf_counter() - t_base
    print(f"baseline val_loss={base_loss:.6f} val_bpb={base_bpb:.6f} time={base_elapsed:.1f}s")
    print(f"baseline_exact val_loss={base_loss:.8f} val_bpb={base_bpb:.8f}")

    # ─── Legal TTT ───────────────────────────────────────────────────────
    if cfg.ttt_enabled:
        print(f"\n{'='*60}")
        print("Phase 2: Legal score-first TTT")
        print(f"{'='*60}")

        # Reload model fresh — TTT modifies weights
        model_ttt = load_model(cfg.artifact_path, config, cfg.vocab_size, device)

        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_sliding_ttt(
            model_ttt, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            device, cfg,
        )
        ttt_elapsed = time.perf_counter() - t_ttt
        print(f"legal_ttt val_loss={ttt_loss:.6f} val_bpb={ttt_bpb:.6f} time={ttt_elapsed:.1f}s")
        print(f"legal_ttt_exact val_loss={ttt_loss:.8f} val_bpb={ttt_bpb:.8f}")

        delta = ttt_bpb - base_bpb
        print(f"\nTTT improvement: {delta:+.6f} BPB ({delta:.4f})")

    # ─── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"SUMMARY — {config['arch_name']}")
    print(f"  Baseline BPB:  {base_bpb:.6f}")
    if cfg.ttt_enabled:
        print(f"  Legal TTT BPB: {ttt_bpb:.6f}  (delta: {delta:+.6f})")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
