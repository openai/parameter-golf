#!/usr/bin/env python3
"""Fast eval-only TTT iteration script.
Loads a pre-saved quantized model and runs various TTT strategies.
No training — iterate on TTT approaches in minutes instead of hours.

Usage:
  TTT_STRATEGY=lora_sliding \
    torchrun --standalone --nproc_per_node=8 eval_ttt.py

Strategies:
  baseline     — sliding window only (no TTT)
  sgd_sliding  — current SGD TTT (chunk-major, sliding window)
  lora_sliding — LoRA TTT (chunk-major, sliding window, Adam)
  lora_perdoc  — per-document LoRA TTT (PR#548 style)
  ensemble     — Bayesian ensemble TTT (multiple trajectories)
  routed       — routed multi-adapter TTT (4 experts)
"""
import os, sys, math, time, copy
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import sentencepiece as spm
from torch import Tensor, nn

# Import everything from the submission script
sys.path.insert(0, str(Path(__file__).parent))
from train_gpt import (
    Hyperparameters, GPT, CastedLinear, restore_low_dim_params_to_fp32,
    load_validation_tokens, build_sentencepiece_luts,
    dequantize_mixed_int6, eval_val_sliding, eval_val_sliding_ttt,
    eval_val_ttt_lora, BatchedLinearLoRA, BatchedTTTLoRA,
    _find_docs, _reset_ttt_optimizer, _compute_chunk_window,
    CONTROL_TENSOR_NAME_PATTERNS,
)
from flash_attn_interface import flash_attn_func as flash_attn_3_func


# ── Non-batched LoRA for chunk-major TTT ──

class SimpleLoRA(nn.Module):
    """Non-batched LoRA: same adapter applied to all batch elements."""
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(F.linear(x, self.A), self.B)

    def reset(self):
        nn.init.normal_(self.A, std=1.0 / math.sqrt(self.A.shape[1]))
        nn.init.zeros_(self.B)


class SimpleTTTLoRA:
    """Non-batched LoRA container for chunk-major sliding window TTT."""
    def __init__(self, model, rank: int, device):
        dim = model.tok_emb.embedding_dim
        vocab = model.tok_emb.num_embeddings
        self.q_loras = nn.ModuleList()
        self.v_loras = nn.ModuleList()
        for block in model.blocks:
            q_out = block.attn.c_q.weight.shape[0]
            v_out = block.attn.c_v.weight.shape[0]
            self.q_loras.append(SimpleLoRA(dim, q_out, rank).to(device))
            self.v_loras.append(SimpleLoRA(dim, v_out, rank).to(device))
        self._lm_head = SimpleLoRA(dim, vocab, rank).to(device)

    def lm_head_lora(self, x: Tensor) -> Tensor:
        return self._lm_head(x)

    def parameters(self):
        params = []
        for m in list(self.q_loras) + list(self.v_loras) + [self._lm_head]:
            params.extend(m.parameters())
        return params

    def reset(self):
        for m in list(self.q_loras) + list(self.v_loras) + [self._lm_head]:
            m.reset()


def eval_val_sliding_lora_ttt(
    args, base_model, rank, world_size, device, val_tokens,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride=64, batch_seqs=32, log_fn=None,
    lora_rank=8, lora_lr=0.01, ttt_epochs=3, chunk_tokens=32768,
    ttt_momentum=0.9, ttt_grad_clip=1.0,
) -> tuple[float, float]:
    """Sliding window eval + LoRA TTT (chunk-major).
    Same structure as eval_val_sliding_ttt but uses LoRA+Adam instead of SGD on full weights.
    Score-first legal: score each chunk with sliding windows, then train LoRA on it."""
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1

    # Pre-compute all window starts
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]

    # Assign windows to chunks
    num_chunks = (total_tokens + chunk_tokens - 1) // chunk_tokens
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // chunk_tokens, num_chunks - 1)
        chunk_windows[ci].append(ws)

    if log_fn:
        log_fn(f"lora_sliding_ttt:start chunks={num_chunks} chunk_tokens={chunk_tokens} "
               f"total_windows={len(window_starts)} stride={stride} "
               f"lora_rank={lora_rank} lora_lr={lora_lr} ttt_epochs={ttt_epochs}")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Freeze base model, create LoRA
    for p in base_model.parameters():
        p.requires_grad_(False)

    lora = SimpleTTTLoRA(base_model, lora_rank, device)
    optimizer = torch.optim.Adam(lora.parameters(), lr=lora_lr, betas=(0.9, 0.95), eps=1e-10)
    t0 = time.perf_counter()

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * chunk_tokens
        chunk_end = min((ci + 1) * chunk_tokens, total_tokens)

        # --- Phase 1: SCORE this chunk's windows with LoRA ---
        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]

        base_model.eval()
        with torch.no_grad():
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
                    chunk_tok = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    nll = base_model(x_batch, y_batch, lora=lora)  # [bsz, seq_len]
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

        # --- Phase 2: TRAIN LoRA on this chunk (already scored = legal) ---
        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = lora_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cos_lr
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                train_batch = min(32, my_chunk_seqs)
                for _ep in range(ttt_epochs):
                    for bs in range(0, my_chunk_seqs, train_batch):
                        be = min(bs + train_batch, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = base_model(x, y, lora=lora).mean()
                        loss.backward()
                        if world_size > 1:
                            for p in lora.parameters():
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(lora.parameters(), ttt_grad_clip)
                        optimizer.step()

        if log_fn and (ci % 10 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rbpb = float((loss_sum / math.log(2.0)) / byte_count) if byte_count > 0 else 0.0
            log_fn(f"  lora_ttt_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    if log_fn:
        log_fn(f"lora_sliding_ttt:done val_bpb={val_bpb:.6f} time={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb


# ── Ensemble TTT: multiple trajectories with mean aggregation ──

def eval_val_ensemble_ttt(
    args, base_model, rank, world_size, device, val_tokens,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride=64, batch_seqs=32, log_fn=None,
    n_paths=3, lora_rank=4, base_lr=0.01, lr_spread=3.0,
    ttt_epochs=3, chunk_tokens=32768,
) -> tuple[float, float]:
    """Ensemble TTT: run N LoRA trajectories with different LR, mean-aggregate logprobs.
    Each trajectory is a SimpleTTTLoRA with a different learning rate.
    Score-first legal on each trajectory independently."""
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1

    # Pre-compute window starts
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    num_chunks = (total_tokens + chunk_tokens - 1) // chunk_tokens
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // chunk_tokens, num_chunks - 1)
        chunk_windows[ci].append(ws)

    # Create N LoRA trajectories with different learning rates
    lrs = [base_lr * (lr_spread ** (i / max(n_paths - 1, 1) - 0.5)) for i in range(n_paths)]
    if log_fn:
        log_fn(f"ensemble_ttt:start paths={n_paths} lrs={[f'{lr:.4f}' for lr in lrs]} "
               f"rank={lora_rank} epochs={ttt_epochs}")

    for p in base_model.parameters():
        p.requires_grad_(False)

    loras = [SimpleTTTLoRA(base_model, lora_rank, device) for _ in range(n_paths)]
    optimizers = [torch.optim.Adam(l.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-10)
                  for l, lr in zip(loras, lrs)]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    t0 = time.perf_counter()

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * chunk_tokens
        chunk_end = min((ci + 1) * chunk_tokens, total_tokens)

        # --- SCORE: mean logprobs across all trajectories ---
        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]

        base_model.eval()
        with torch.no_grad():
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
                    chunk_tok = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]

                # Get NLL from each trajectory and average
                all_nll = []
                for lora in loras:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        nll = base_model(x_batch, y_batch, lora=lora)
                    all_nll.append(nll)
                # Mean of NLL across trajectories (approximation to log-mean-exp)
                mean_nll = torch.stack(all_nll).mean(dim=0)

                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    scored_nll = mean_nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt, prev = y_batch[i, s:wlen], x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        # --- TRAIN each trajectory independently ---
        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                train_batch = min(32, my_chunk_seqs)

                for path_idx, (lora, opt) in enumerate(zip(loras, optimizers)):
                    cos_lr = lrs[path_idx] * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                    for pg in opt.param_groups:
                        pg['lr'] = cos_lr
                    for _ep in range(ttt_epochs):
                        for bs in range(0, my_chunk_seqs, train_batch):
                            be = min(bs + train_batch, my_chunk_seqs)
                            actual_bs = my_seq_s + bs
                            start_tok = chunk_start + actual_bs * seq_len
                            end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                            if end_tok > val_tokens.numel():
                                continue
                            local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                            x = local[:-1].reshape(-1, seq_len)
                            y = local[1:].reshape(-1, seq_len)
                            opt.zero_grad(set_to_none=True)
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                loss = base_model(x, y, lora=lora).mean()
                            loss.backward()
                            if world_size > 1:
                                for p in lora.parameters():
                                    if p.grad is not None:
                                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                            torch.nn.utils.clip_grad_norm_(lora.parameters(), 1.0)
                            opt.step()

        if log_fn and (ci % 10 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rbpb = float((loss_sum / math.log(2.0)) / byte_count) if byte_count > 0 else 0.0
            log_fn(f"  ensemble_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    if log_fn:
        log_fn(f"ensemble_ttt:done val_bpb={val_bpb:.6f} paths={n_paths} time={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb


def main():
    args = Hyperparameters()
    strategy = os.environ.get("TTT_STRATEGY", "lora_sliding")

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, local_rank = 0, 1, 0
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def log0(msg):
        if master:
            print(msg)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    model_path = os.environ.get("MODEL_PATH", "final_int6_model.pt")
    log0(f"Loading quantized model from {model_path}")
    saved = torch.load(model_path, map_location="cpu", weights_only=True)

    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init, bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim, xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)

    deq_state = dequantize_mixed_int6(saved["quantized"], saved["meta"], eval_model.state_dict())
    eval_model.load_state_dict(deq_state, strict=True)
    CastedLinear._qat_enabled = False
    CastedLinear._soft_tau = 1000.0
    log0(f"Model loaded: {sum(p.numel() for p in eval_model.parameters())} params")

    # Baseline sliding window
    log0("=" * 60)
    log0("BASELINE: Sliding window eval (stride=64)")
    t_slide = time.perf_counter()
    sw_loss, sw_bpb = eval_val_sliding(
        args, eval_model, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=64, batch_seqs=32,
    )
    log0(f"sliding_window val_bpb:{sw_bpb:.6f} time:{time.perf_counter()-t_slide:.1f}s")

    if strategy == "baseline":
        log0("Done (baseline only)")
    elif strategy == "sgd_sliding":
        log0("=" * 60)
        log0("SGD sliding TTT (current approach)")
        if distributed:
            dist.barrier()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_sliding_ttt(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=64, batch_seqs=32, log_fn=log0,
        )
        log0(f"sgd_sliding val_bpb:{ttt_bpb:.6f} delta:{ttt_bpb - sw_bpb:+.6f} "
             f"time:{time.perf_counter()-t_ttt:.1f}s")

    elif strategy == "lora_sliding":
        log0("=" * 60)
        lora_rank = int(os.environ.get("LORA_RANK", "8"))
        lora_lr = float(os.environ.get("LORA_LR", "0.01"))
        ttt_epochs = int(os.environ.get("LORA_EPOCHS", "3"))
        chunk_tokens = int(os.environ.get("LORA_CHUNK", "32768"))
        log0(f"LoRA sliding TTT: rank={lora_rank} lr={lora_lr} epochs={ttt_epochs} chunk={chunk_tokens}")
        if distributed:
            dist.barrier()
        torch._dynamo.reset()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_sliding_lora_ttt(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=64, batch_seqs=32, log_fn=log0,
            lora_rank=lora_rank, lora_lr=lora_lr, ttt_epochs=ttt_epochs,
            chunk_tokens=chunk_tokens,
        )
        log0(f"lora_sliding val_bpb:{ttt_bpb:.6f} delta:{ttt_bpb - sw_bpb:+.6f} "
             f"time:{time.perf_counter()-t_ttt:.1f}s")

    elif strategy == "lora_perdoc":
        log0("=" * 60)
        log0("Per-document LoRA TTT (PR#548 style)")
        if distributed:
            dist.barrier()
        torch._dynamo.reset()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_ttt_lora(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            log_fn=log0,
        )
        log0(f"lora_perdoc val_bpb:{ttt_bpb:.6f} delta:{ttt_bpb - sw_bpb:+.6f} "
             f"time:{time.perf_counter()-t_ttt:.1f}s")

    elif strategy == "ensemble":
        log0("=" * 60)
        n_paths = int(os.environ.get("ENSEMBLE_PATHS", "3"))
        lora_rank = int(os.environ.get("LORA_RANK", "4"))
        base_lr = float(os.environ.get("LORA_LR", "0.01"))
        lr_spread = float(os.environ.get("LR_SPREAD", "3.0"))
        ttt_epochs = int(os.environ.get("LORA_EPOCHS", "3"))
        chunk_tokens = int(os.environ.get("LORA_CHUNK", "32768"))
        log0(f"Ensemble TTT: paths={n_paths} rank={lora_rank} base_lr={base_lr} "
             f"spread={lr_spread} epochs={ttt_epochs}")
        if distributed:
            dist.barrier()
        torch._dynamo.reset()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_ensemble_ttt(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=64, batch_seqs=32, log_fn=log0,
            n_paths=n_paths, lora_rank=lora_rank, base_lr=base_lr,
            lr_spread=lr_spread, ttt_epochs=ttt_epochs, chunk_tokens=chunk_tokens,
        )
        log0(f"ensemble val_bpb:{ttt_bpb:.6f} delta:{ttt_bpb - sw_bpb:+.6f} "
             f"time:{time.perf_counter()-t_ttt:.1f}s")

    else:
        log0(f"Unknown strategy: {strategy}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
