"""Recurrent-aware TTT (test-time training) wrapper.

Preserves the legal score-first protocol from the current best record
while adding recurrence-safe adaptation regimes:

  1. tail_only              – only tail blocks adapt
  2. tail_plus_stem         – stem + tail, core frozen
  3. all_unique_layers      – stem + core + tail (core at full LR)
  4. all_layers             – same as all_unique (alias)
  5. all_layers_with_recurrent_lr_scale – core at reduced LR
"""
from __future__ import annotations

import math
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from model_recurrent_bestbase import RecurrentGPT
from train_utils_recurrent import Hyperparameters


def _select_ttt_params(
    model: RecurrentGPT,
    regime: str,
    recurrent_lr_scale: float = 0.1,
    base_lr: float = 0.002,
) -> list[dict]:
    """Return param groups for TTT, honouring the chosen regime."""
    stem_params = list(model.stem_blocks.parameters())
    core_params = list(model.core_blocks.parameters())
    tail_params = list(model.tail_blocks.parameters())

    # Non-block params: embeddings, norms, skip weights, bigram, VE, LM head
    other_params = [p for n, p in model.named_parameters()
                    if not any(tag in n for tag in
                               ("stem_blocks.", "core_blocks.", "tail_blocks.",
                                "qo_bank", "kv_bank", "mlp_up_bank",
                                "mlp_down_bank"))]
    bank_params = [model.qo_bank, model.kv_bank,
                   model.mlp_up_bank, model.mlp_down_bank]

    if regime == "tail_only":
        return [{"params": tail_params + other_params, "lr": base_lr}]

    if regime == "tail_plus_stem":
        return [{"params": stem_params + tail_params + other_params,
                 "lr": base_lr}]

    if regime in ("all_unique_layers", "all_layers"):
        return [{"params": (stem_params + core_params + tail_params
                            + other_params + bank_params),
                 "lr": base_lr}]

    if regime == "all_layers_with_recurrent_lr_scale":
        groups = [
            {"params": stem_params + tail_params + other_params,
             "lr": base_lr},
            {"params": core_params, "lr": base_lr * recurrent_lr_scale},
            {"params": bank_params, "lr": base_lr * recurrent_lr_scale},
        ]
        return groups

    raise ValueError(f"Unknown TTT regime: {regime}")


def eval_val_sliding_ttt(
    args: Hyperparameters,
    base_model: RecurrentGPT,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    log0=print,
    feedback_fn=None,
    stabilizer=None,
    ttt_regime: str = "tail_only",
    ttt_recurrent_lr_scale: float = 0.1,
) -> tuple[float, float]:
    """Legal score-first TTT with recurrence-aware param selection."""
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride
                     or ws == 0]

    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    log0(f"ttt_sliding:start chunks={num_chunks} chunk_tokens={ttt_chunk} "
         f"total_windows={len(window_starts)} stride={stride} "
         f"regime={ttt_regime} ttt_lr={args.ttt_lr} "
         f"ttt_epochs={args.ttt_epochs}")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Build TTT param groups
    param_groups = _select_ttt_params(
        base_model, ttt_regime,
        recurrent_lr_scale=ttt_recurrent_lr_scale,
        base_lr=args.ttt_lr)
    ttt_params = []
    for pg in param_groups:
        ttt_params.extend(pg["params"])

    # Freeze everything first, then unfreeze TTT params
    for p in base_model.parameters():
        p.requires_grad_(False)
    for p in ttt_params:
        p.requires_grad_(True)

    log0(f"ttt_sliding:params unfrozen={sum(p.numel() for p in ttt_params)} "
         f"frozen={sum(p.numel() for p in base_model.parameters() if not p.requires_grad)}")

    optimizer = torch.optim.SGD(param_groups, lr=args.ttt_lr,
                                momentum=args.ttt_momentum)
    t0 = time.perf_counter()

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)

        # --- Phase 1: SCORE (inference_mode) ---
        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]

        base_model.eval()
        with torch.inference_mode():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64,
                                      device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64,
                                      device=device)
                wlens: list[int] = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk_tok = val_tokens[ws:end + 1].to(
                        dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(
                        x_batch, feedback_fn=feedback_fn,
                        stabilizer=stabilizer)
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
                    tb += (has_leading_space_lut[tgt]
                           & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        # --- Phase 2: TRAIN ---
        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and args.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = (args.ttt_lr * 0.5
                          * (1.0 + math.cos(
                              math.pi * ci / max(num_chunks - 1, 1))))
                for pg in optimizer.param_groups:
                    scale = pg.get("lr", args.ttt_lr) / args.ttt_lr
                    pg["lr"] = cos_lr * scale

                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s

                for _ep in range(args.ttt_epochs):
                    for bs in range(0, my_chunk_seqs, args.ttt_batch_seqs):
                        be = min(bs + args.ttt_batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = (chunk_start
                                   + (my_seq_s + be) * seq_len + 1)
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(
                            device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda",
                                            dtype=torch.bfloat16):
                            loss = base_model(
                                x, y, feedback_fn=feedback_fn,
                                stabilizer=stabilizer)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(
                                        p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(
                            ttt_params, args.ttt_grad_clip)
                        optimizer.step()

        if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = (rl / math.log(2.0)
                    * (token_count.item() / max(byte_count.item(), 1))
                    if token_count.item() > 0 else 0.0)
            log0(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} "
                 f"time={elapsed:.1f}s")

    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    val_bpb = (val_loss / math.log(2.0)
               * (token_count.item() / byte_count.item()))

    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()

    log0(f"ttt_sliding:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
         f"elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb
