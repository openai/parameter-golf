"""
Eval-only script for testing TTT variants on an existing quantized model.

Loads final_model.int6.ptz, runs TTT with configurable param selection
and optional entropy-adaptive epochs, reports BPB.

Env vars (beyond standard Hyperparameters):
  TTT_PARAM_MODE:  all | late_blocks | loop_blocks | control_only
                   | no_embeddings | norm_scale_qgain_skip   (default: all)
  TTT_ADAPTIVE:          0 or 1                               (default: 0)
  TTT_ADAPTIVE_EASY_NLL: NLL below which chunk is "easy"      (default: 2.5)
  TTT_ADAPTIVE_HARD_NLL: NLL above which chunk is "hard"      (default: 3.0)
  TTT_ADAPTIVE_EASY_EP:  epochs for easy chunks               (default: 1)
  TTT_ADAPTIVE_HARD_EP:  epochs for hard chunks               (default: 5)

Usage:
  TTT_PARAM_MODE=no_embeddings TTT_LR=0.005 TTT_EPOCHS=3 \
    torchrun --standalone --nproc_per_node=8 eval_ttt_variants.py
"""
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from train_pr1493 import (
    Hyperparameters,
    ValidationData,
    GPT,
    deserialize,
    _loss_bpb,
    log,
    set_logging_hparams,
    CONTROL_TENSOR_NAME_PATTERNS,
)

# ---------------------------------------------------------------------------
# Param-mode selection
# ---------------------------------------------------------------------------

def get_ttt_params(model, mode, h):
    """Return the list of parameters to update during TTT."""

    all_params = {id(p): (name, p) for name, p in model.named_parameters()}

    if mode == "all":
        return list(model.parameters())

    if mode == "no_embeddings":
        return [p for n, p in model.named_parameters() if "tok_emb" not in n]

    if mode == "late_blocks":
        params = []
        for i in range(7, h.num_layers):
            params.extend(model.blocks[i].parameters())
        if model.skip_weights is not None:
            params.append(model.skip_weights)
        if model.skip_gates is not None:
            params.append(model.skip_gates)
        return params

    if mode == "loop_blocks":
        params = []
        for i in range(h.loop_start, h.loop_end + 1):
            params.extend(model.blocks[i].parameters())
        return params

    if mode == "control_only":
        params = []
        for block in model.blocks:
            params.extend([block.attn_scale, block.mlp_scale,
                           block.resid_mix, block.attn.q_gain])
        if model.skip_weights is not None:
            params.append(model.skip_weights)
        if model.skip_gates is not None:
            params.append(model.skip_gates)
        return params

    if mode == "norm_scale_qgain_skip":
        params = []
        for block in model.blocks:
            params.extend([block.attn_scale, block.mlp_scale,
                           block.attn.q_gain])
        if model.skip_weights is not None:
            params.append(model.skip_weights)
        if model.skip_gates is not None:
            params.append(model.skip_gates)
        return params

    if mode == "late_blocks_6":
        params = []
        for i in range(6, h.num_layers):
            params.extend(model.blocks[i].parameters())
        if model.skip_weights is not None:
            params.append(model.skip_weights)
        if model.skip_gates is not None:
            params.append(model.skip_gates)
        return params

    if mode == "mlp_proj_only":
        # ByteDance In-Place TTT: only update MLP down-projection weights
        # blocks.*.mlp.proj.weight — maps hidden activations back to residual
        params = []
        for block in model.blocks:
            params.append(block.mlp.proj.weight)
        return params

    raise ValueError(f"Unknown TTT_PARAM_MODE: {mode!r}")


def _count_params(params):
    return sum(p.numel() for p in params)


# ---------------------------------------------------------------------------
# TTT eval with variant support
# ---------------------------------------------------------------------------

def eval_val_ttt_variant(h, device, val_data, base_model, param_mode,
                         adaptive, adaptive_easy_nll, adaptive_hard_nll,
                         adaptive_easy_ep, adaptive_hard_ep,
                         batch_seqs=32):
    rank = h.rank
    world_size = h.world_size
    seq_len = h.eval_seq_len
    stride = h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    ttt_chunk = h.ttt_chunk_tokens
    context_size = seq_len - stride

    # Build window→chunk mapping (identical to PR #1493)
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if ws + context_size < total_tokens]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        wlen = min(ws + seq_len, total_tokens) - ws
        s = 0 if ws == 0 else context_size
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    # Select params
    ttt_params = get_ttt_params(base_model, param_mode, h)
    ttt_param_ids = {id(p) for p in ttt_params}
    total_model_params = sum(p.numel() for p in base_model.parameters())
    ttt_param_count = _count_params(ttt_params)

    log(f"ttt_variant: mode={param_mode} params={ttt_param_count}/{total_model_params} "
        f"({100*ttt_param_count/total_model_params:.1f}%) "
        f"lr={h.ttt_lr} epochs={h.ttt_epochs} momentum={h.ttt_momentum} "
        f"adaptive={adaptive}")
    if adaptive:
        log(f"ttt_adaptive: easy_nll<{adaptive_easy_nll} → {adaptive_easy_ep}ep, "
            f"hard_nll>{adaptive_hard_nll} → {adaptive_hard_ep}ep, "
            f"normal → {h.ttt_epochs}ep")

    # Freeze non-selected params, enable selected
    for p in base_model.parameters():
        p.requires_grad_(id(p) in ttt_param_ids)

    compiled_logits = torch.compile(base_model.forward_logits,
                                    dynamic=False, fullgraph=True)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    optimizer = torch.optim.SGD(ttt_params, lr=h.ttt_lr,
                                momentum=h.ttt_momentum)

    adaptive_epoch_counts = []

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue

        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
        my_s = len(windows) * rank // world_size
        my_e = len(windows) * (rank + 1) // world_size
        my_windows = windows[my_s:my_e]

        # Track per-chunk loss for adaptive epochs
        pre_loss = loss_sum.clone()
        pre_tokens = token_count.clone()

        # ---- Score phase (no grad, compiled forward) ----
        base_model.eval()
        with torch.no_grad():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64,
                                      device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64,
                                      device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    we = min(ws + seq_len, total_tokens)
                    wlen = we - ws
                    wlens.append(wlen)
                    chunk_tok = val_data.val_tokens[ws:we + 1].to(
                        dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = compiled_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1), reduction="none"
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else context_size
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                    tb += (val_data.has_leading_space_lut[tgt]
                           & ~val_data.is_boundary_token_lut[prev]).to(
                               torch.float64)
                    byte_count += tb.sum()

        # ---- Decide epochs for adaptation ----
        is_last_chunk = (ci == num_chunks - 1)
        epochs = h.ttt_epochs

        if adaptive and not is_last_chunk:
            chunk_loss_delta = loss_sum - pre_loss
            chunk_token_delta = token_count - pre_tokens
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(chunk_loss_delta, op=dist.ReduceOp.SUM)
                dist.all_reduce(chunk_token_delta, op=dist.ReduceOp.SUM)
            chunk_nll = (chunk_loss_delta / chunk_token_delta).item()
            if chunk_nll < adaptive_easy_nll:
                epochs = adaptive_easy_ep
            elif chunk_nll > adaptive_hard_nll:
                epochs = adaptive_hard_ep
            adaptive_epoch_counts.append(epochs)

        # ---- Adapt phase (SGD on selected params) ----
        if not is_last_chunk and epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = h.ttt_lr * 0.5 * (
                    1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg["lr"] = cos_lr
                my_seq_s = chunk_seqs * rank // world_size
                my_seq_e = chunk_seqs * (rank + 1) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(epochs):
                    for bs in range(0, my_chunk_seqs, batch_seqs):
                        be = min(bs + batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = (chunk_start
                                   + (my_seq_s + be) * seq_len + 1)
                        if end_tok > val_data.val_tokens.numel():
                            continue
                        local = val_data.val_tokens[start_tok:end_tok].to(
                            device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda",
                                            dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad,
                                                    op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                        optimizer.step()

    # ---- Global reduce ----
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    # Restore all params
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()

    if adaptive and adaptive_epoch_counts and rank == 0:
        from collections import Counter
        counts = Counter(adaptive_epoch_counts)
        log(f"ttt_adaptive_stats: {dict(sorted(counts.items()))}")

    return _loss_bpb(loss_sum, token_count, byte_count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.optimize_ddp = False

    h = Hyperparameters()
    set_logging_hparams(h)

    param_mode = os.environ.get("TTT_PARAM_MODE", "all")
    adaptive = bool(int(os.environ.get("TTT_ADAPTIVE", "0")))
    adaptive_easy_nll = float(os.environ.get("TTT_ADAPTIVE_EASY_NLL", "2.5"))
    adaptive_hard_nll = float(os.environ.get("TTT_ADAPTIVE_HARD_NLL", "3.0"))
    adaptive_easy_ep = int(os.environ.get("TTT_ADAPTIVE_EASY_EP", "1"))
    adaptive_hard_ep = int(os.environ.get("TTT_ADAPTIVE_HARD_EP", "5"))

    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        log(f"=== TTT Variant Eval ===")
        log(f"param_mode={param_mode} lr={h.ttt_lr} epochs={h.ttt_epochs} "
            f"momentum={h.ttt_momentum} chunk={h.ttt_chunk_tokens}")
        if adaptive:
            log(f"adaptive: easy<{adaptive_easy_nll}→{adaptive_easy_ep}ep "
                f"hard>{adaptive_hard_nll}→{adaptive_hard_ep}ep")

    val_data = ValidationData(h, device)
    log(f"val_tokens: {val_data.val_tokens.numel() - 1}")

    # Load quantized model fresh from disk
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    val_loss, val_bpb = eval_val_ttt_variant(
        h, device, val_data, eval_model, param_mode,
        adaptive, adaptive_easy_nll, adaptive_hard_nll,
        adaptive_easy_ep, adaptive_hard_ep,
    )

    torch.cuda.synchronize()
    elapsed_ms = 1e3 * (time.perf_counter() - t0)
    log(f"RESULT: mode={param_mode} lr={h.ttt_lr} epochs={h.ttt_epochs} "
        f"val_bpb={val_bpb:.8f} eval_time={elapsed_ms:.0f}ms")

    # Cleanup
    del eval_model
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
