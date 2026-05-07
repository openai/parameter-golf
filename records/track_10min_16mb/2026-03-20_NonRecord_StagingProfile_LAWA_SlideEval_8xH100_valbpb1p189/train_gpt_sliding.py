"""Sliding-window eval helpers for train_gpt.py — ported from PR60 record."""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


def gpt_forward_logits(model: nn.Module, input_ids: Tensor) -> Tensor:
    """Standalone logits-only forward (no LoRA, no loss). Works with DDP-wrapped models."""
    m = model.module if hasattr(model, "module") else model
    x = m.tok_emb(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    skips: list[Tensor] = []
    for i in range(m.num_encoder_layers):
        x = m.blocks[i](x, x0)
        skips.append(x)
    for i in range(m.num_decoder_layers):
        if skips:
            x = x + m.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        x = m.blocks[m.num_encoder_layers + i](x, x0)
    x = m.final_norm(x)
    if m.tie_embeddings:
        logits = F.linear(x, m.tok_emb.weight.to(x.dtype))
    else:
        logits = m.lm_head(x)
    return m.logit_softcap * torch.tanh(logits / m.logit_softcap)


def eval_val_sliding(
    logits_fn,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    seq_len: int,
    stride: int,
    eval_batch_seqs: int = 256,
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with near-full context."""
    total = val_tokens.numel() - 1

    # Build windows: (start_pos, score_offset)
    windows: list[tuple[int, int]] = []
    p = 0
    while p + seq_len <= total:
        s = 0 if p == 0 else (seq_len - stride)
        windows.append((p, s))
        p += stride

    # Distribute across ranks
    n = len(windows)
    per_rank = (n + world_size - 1) // world_size
    my_start = rank * per_rank
    my_end = min(my_start + per_rank, n)
    my_windows = windows[my_start:my_end]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    with torch.inference_mode():
        for i in range(0, len(my_windows), eval_batch_seqs):
            batch = my_windows[i : i + eval_batch_seqs]
            bs = len(batch)

            # Pad to eval_batch_seqs to avoid recompilation
            x_list = [val_tokens[w : w + seq_len] for w, _ in batch]
            y_list = [val_tokens[w + 1 : w + seq_len + 1] for w, _ in batch]
            pad = eval_batch_seqs - bs
            if pad > 0:
                x_list.extend([x_list[-1]] * pad)
                y_list.extend([y_list[-1]] * pad)

            x = torch.stack(x_list).to(device=device, dtype=torch.int64)
            y = torch.stack(y_list).to(device=device, dtype=torch.int64)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = logits_fn(x)

            for b in range(bs):
                s = batch[b][1]
                scored_logits = logits[b, s:]
                scored_targets = y[b, s:]

                loss = F.cross_entropy(scored_logits.float(), scored_targets, reduction="sum")
                loss_sum += loss.to(torch.float64)
                ns = scored_targets.numel()
                tok_count += ns

                prev = x[b, s : s + ns]
                tgt = scored_targets
                tb = base_bytes_lut[tgt].to(torch.int16)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.int16)
                byte_count += tb.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / tok_count).item()
    bpb = val_loss / math.log(2.0) * (tok_count.item() / byte_count.item())
    return val_loss, bpb


def eval_sliding_roundtrip(
    base_model: nn.Module,
    args,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """Orchestrator: compile forward_logits, warmup, run sliding eval, return (loss, bpb)."""
    seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    stride = args.eval_stride

    def _logits_fn(input_ids: Tensor) -> Tensor:
        return gpt_forward_logits(base_model, input_ids)

    compiled_fn = torch.compile(_logits_fn)
    # Warmup compile with a dummy batch matching eval_val_sliding's padded batch size
    eval_batch_seqs = 256
    dummy = torch.zeros(eval_batch_seqs, seq_len, dtype=torch.int64, device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        compiled_fn(dummy)

    return eval_val_sliding(
        compiled_fn, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        seq_len=seq_len, stride=stride,
    )
