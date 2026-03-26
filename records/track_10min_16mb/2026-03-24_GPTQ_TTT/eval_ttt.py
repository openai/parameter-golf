"""Score-first TTT evaluation for PR#414 base model.

Algorithm (legal per competition rules):
  For each chunk of validation tokens:
    1. SCORE the chunk using global sliding window (with context from prev tokens)
    2. TRAIN on the chunk (cumulative adaptation, no weight reset)
    3. Last chunk: score only, never trained on

CRITICAL: Scoring uses GLOBAL positions (not per-chunk) so tokens at chunk
boundaries get full context from previous tokens. Only tokens WITHIN the
current chunk contribute to the loss sum.
"""
from __future__ import annotations

import copy
import glob
import io
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except ImportError:
    from flash_attn import flash_attn_func as flash_attn_3_func

sys.path.insert(0, str(Path(__file__).parent))
from train_pr414 import (
    Hyperparameters,
    GPT,
    CastedLinear,
    build_sentencepiece_luts,
    load_validation_tokens,
    eval_val_sliding,
    restore_low_dim_params_to_fp32,
    dequantize_mixed_int6,
    CONTROL_TENSOR_NAME_PATTERNS,
)

# ── TTT Hyperparameters ─────────────────────────────────────────────────────
TTT_LR = float(os.environ.get("TTT_LR", "0.002"))
TTT_MOMENTUM = float(os.environ.get("TTT_MOMENTUM", "0.9"))
TTT_EPOCHS = int(os.environ.get("TTT_EPOCHS", "3"))
TTT_CHUNK_TOKENS = int(os.environ.get("TTT_CHUNK_TOKENS", "32768"))
TTT_GRAD_CLIP = float(os.environ.get("TTT_GRAD_CLIP", "1.0"))
TTT_FREEZE_BLOCKS = int(os.environ.get("TTT_FREEZE_BLOCKS", "2"))
TTT_FREEZE_EMBEDDINGS = bool(int(os.environ.get("TTT_FREEZE_EMBEDDINGS", "1")))
TTT_EVAL_STRIDE = int(os.environ.get("TTT_EVAL_STRIDE", "64"))
TTT_BATCH_SEQS = int(os.environ.get("TTT_BATCH_SEQS", "64"))
TTT_OPTIMIZER = os.environ.get("TTT_OPTIMIZER", "sgd")
TTT_MAX_CHUNKS = int(os.environ.get("TTT_MAX_CHUNKS", "200"))  # 0 = unlimited
TTT_SKIP_BASELINE = bool(int(os.environ.get("TTT_SKIP_BASELINE", "1")))
TTT_LR_SCHEDULE = os.environ.get("TTT_LR_SCHEDULE", "cosine")  # "cosine" or "constant"

# Temperature sweep values
TEMPS = [0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.05]


def eval_ttt_score_first(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float, float]:
    """Score-first TTT with global sliding window context."""
    seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    chunk_tokens = TTT_CHUNK_TOKENS
    stride = TTT_EVAL_STRIDE
    batch_seqs = TTT_BATCH_SEQS
    total_val_tokens = val_tokens.numel() - 1

    total_possible_chunks = (total_val_tokens + chunk_tokens - 1) // chunk_tokens
    num_chunks = total_possible_chunks
    if TTT_MAX_CHUNKS > 0:
        num_chunks = min(num_chunks, TTT_MAX_CHUNKS)

    # ── Setup trainable params ───────────────────────────────────────────
    num_blocks = len(model.blocks)
    for p in model.parameters():
        p.requires_grad_(False)
    ttt_params = []
    for name, p in model.named_parameters():
        # Skip embeddings
        if any(k in name for k in ("tok_emb", "bigram", "ve_shared")):
            continue
        # Unfreeze last blocks
        is_unfrozen_block = False
        for bi in range(TTT_FREEZE_BLOCKS, num_blocks):
            if f"blocks.{bi}." in name:
                is_unfrozen_block = True
                break
        # Also unfreeze norms, scales, final_norm, skip_weights, smear
        is_auxiliary = any(k in name for k in ("final_norm", "skip_weights", "smear"))
        if is_unfrozen_block or is_auxiliary:
            p.requires_grad_(True)
            ttt_params.append(p)

    num_ttt_params = sum(p.numel() for p in ttt_params)
    if rank == 0:
        print(f"TTT: {num_chunks}/{total_possible_chunks} chunks of {chunk_tokens} tokens")
        print(f"TTT: lr={TTT_LR}, schedule={TTT_LR_SCHEDULE}, epochs={TTT_EPOCHS}, stride={stride}, opt={TTT_OPTIMIZER}")
        print(f"TTT: freeze_blocks={TTT_FREEZE_BLOCKS}/{num_blocks}, freeze_embed={TTT_FREEZE_EMBEDDINGS}")
        print(f"TTT: {num_ttt_params:,} trainable / {sum(p.numel() for p in model.parameters()):,} total")

    if TTT_OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(ttt_params, lr=TTT_LR, weight_decay=0.0, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.SGD(ttt_params, lr=TTT_LR, momentum=TTT_MOMENTUM)

    # Accumulators
    total_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    total_token_count = torch.zeros((), device=device, dtype=torch.float64)
    total_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Per-temperature loss accumulators (token/byte counts are shared)
    temp_loss_sums = {T: torch.zeros((), device=device, dtype=torch.float64) for T in TEMPS}

    # Compile forward_logits for scoring
    compiled_logits = torch.compile(model.forward_logits, dynamic=False)

    t_start = time.perf_counter()

    TTT_TIME_LIMIT = float(os.environ.get("TTT_TIME_LIMIT", "0"))  # 0 = no limit
    actual_chunks_processed = 0
    chunk_timings = []  # per-chunk wall clock times for budget analysis
    current_lr = TTT_LR  # initialize for logging

    for ci in range(num_chunks):
        chunk_t0 = time.perf_counter()
        # Time guard: stop early if over budget
        if TTT_TIME_LIMIT > 0 and (time.perf_counter() - t_start) > TTT_TIME_LIMIT:
            if rank == 0:
                print(f"TTT: time limit {TTT_TIME_LIMIT:.0f}s reached at chunk {ci}, stopping")
            break

        chunk_start = ci * chunk_tokens
        chunk_end = min(chunk_start + chunk_tokens, total_val_tokens)

        # ── Phase 1: SCORE using global sliding windows ──────────────────
        # Generate windows that cover tokens in [chunk_start, chunk_end)
        # but extend backwards up to seq_len for context
        # Windows: start positions such that scored tokens fall in this chunk
        # A window starting at ws scores tokens in [max(ws, ws + seq_len - stride) .. ws + seq_len)
        # We need windows where at least some scored tokens are in [chunk_start, chunk_end)

        # Simple approach: iterate global windows that overlap with this chunk's
        # scored region. The scored region of window starting at ws is:
        #   first_scored = ws (if ws == 0) else ws + seq_len - stride
        #   last_scored = min(ws + seq_len, total_val_tokens) - 1
        # We need: first_scored < chunk_end AND last_scored >= chunk_start

        # Generate all candidate window starts
        # Start from max(0, chunk_start - seq_len + stride) to ensure context
        first_ws = max(0, chunk_start - seq_len + stride)
        # Align to stride
        first_ws = (first_ws // stride) * stride
        last_ws = chunk_end  # windows starting past chunk_end can't score chunk tokens

        window_starts = []
        for ws in range(first_ws, last_ws, stride):
            wend = min(ws + seq_len, total_val_tokens)
            if wend - ws < 1:
                continue
            # Scored range for this window
            if ws == 0:
                s_start_global = 0
            else:
                s_start_global = ws + seq_len - stride
                if s_start_global > wend:
                    s_start_global = ws  # fallback for short windows
            s_end_global = wend
            # Check overlap with chunk
            if s_start_global < chunk_end and s_end_global > chunk_start:
                window_starts.append(ws)

        total_windows = len(window_starts)
        my_s = (total_windows * rank) // world_size
        my_e = (total_windows * (rank + 1)) // world_size
        my_windows = window_starts[my_s:my_e]

        model.eval()
        with torch.inference_mode():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_val_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk_data = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_data[:-1]
                    y_batch[i, :wlen] = chunk_data[1:]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = compiled_logits(x_batch)

                logits_flat = logits.reshape(-1, logits.size(-1)).float()
                y_flat = y_batch.reshape(-1)

                # Compute NLL at T=1.0 (original)
                nll = F.cross_entropy(logits_flat, y_flat, reduction="none").reshape(bsz, seq_len)

                # Compute NLL at each temperature
                temp_nlls = {}
                for T in TEMPS:
                    if T == 1.0:
                        temp_nlls[T] = nll
                    else:
                        temp_nlls[T] = F.cross_entropy(
                            logits_flat / T, y_flat, reduction="none"
                        ).reshape(bsz, seq_len)

                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    # Scored range within this window
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    # Global positions of scored tokens
                    global_s = ws + s
                    global_e = ws + wlen
                    # Only count tokens that fall within current chunk
                    effective_s = max(s, chunk_start - ws)
                    effective_e = min(wlen, chunk_end - ws)
                    # Also respect the stride-based scoring rule
                    effective_s = max(effective_s, s)
                    if effective_e <= effective_s:
                        continue

                    scored_nll = nll[i, effective_s:effective_e].to(torch.float64)
                    total_loss_sum += scored_nll.sum()
                    total_token_count += float(effective_e - effective_s)
                    tgt = y_batch[i, effective_s:effective_e]
                    prev = x_batch[i, effective_s:effective_e]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    total_byte_count += tb.sum()

                    # Accumulate per-temperature losses
                    for T in TEMPS:
                        scored_nll_t = temp_nlls[T][i, effective_s:effective_e].to(torch.float64)
                        temp_loss_sums[T] += scored_nll_t.sum()

        # ── Phase 2: TRAIN on this chunk ─────────────────────────────────
        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and TTT_EPOCHS > 0:
            model.train()
            if TTT_LR_SCHEDULE == "constant":
                current_lr = TTT_LR
            else:  # cosine
                current_lr = TTT_LR * 0.5 * (1.0 + math.cos(math.pi * ci / max(total_possible_chunks - 1, 1)))
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

            chunk_train = val_tokens[chunk_start:chunk_end + 1].to(dtype=torch.int64, device=device)
            num_seqs = (chunk_end - chunk_start) // seq_len
            if num_seqs >= 1:
                usable = num_seqs * seq_len
                x_all = chunk_train[:usable].reshape(num_seqs, seq_len)
                y_all = chunk_train[1:usable + 1].reshape(num_seqs, seq_len)

                for _ep in range(TTT_EPOCHS):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(x_all, y_all)
                    loss.backward()
                    if TTT_GRAD_CLIP > 0:
                        torch.nn.utils.clip_grad_norm_(ttt_params, TTT_GRAD_CLIP)
                    optimizer.step()

        actual_chunks_processed = ci + 1
        chunk_elapsed = time.perf_counter() - chunk_t0
        chunk_timings.append(chunk_elapsed)
        elapsed = time.perf_counter() - t_start
        if rank == 0 and (ci % 50 == 0 or ci == num_chunks - 1):
            partial_bpb = 0.0
            if total_token_count.item() > 0:
                partial_loss = (total_loss_sum / total_token_count).item()
                partial_bpt = partial_loss / math.log(2.0)
                partial_tpb = total_token_count.item() / max(total_byte_count.item(), 1.0)
                partial_bpb = partial_bpt * partial_tpb
            lr_now = current_lr if not is_last_chunk else 0.0
            avg_chunk_s = sum(chunk_timings) / len(chunk_timings)
            print(f"TTT chunk {ci + 1}/{num_chunks}: partial_bpb={partial_bpb:.4f} "
                  f"lr={lr_now:.6f} elapsed={elapsed:.1f}s avg_chunk={avg_chunk_s:.3f}s")

    # Aggregate across ranks
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_byte_count, op=dist.ReduceOp.SUM)
        for T in TEMPS:
            dist.all_reduce(temp_loss_sums[T], op=dist.ReduceOp.SUM)

    val_loss = (total_loss_sum / total_token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = total_token_count.item() / total_byte_count.item()
    ttt_bpb = bits_per_token * tokens_per_byte
    total_time = time.perf_counter() - t_start

    # Compute per-temperature bpb
    temp_bpb = {}
    for T in TEMPS:
        t_loss = (temp_loss_sums[T] / total_token_count).item()
        t_bpt = t_loss / math.log(2.0)
        temp_bpb[T] = t_bpt * tokens_per_byte

    return ttt_bpb, total_time, float(actual_chunks_processed), chunk_timings, temp_bpb


def make_model(args, device):
    """Create a fresh GPT model for evaluation."""
    m = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims,
        ln_scale=args.ln_scale, dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for mod in m.modules():
        if isinstance(mod, CastedLinear):
            mod.float()
    restore_low_dim_params_to_fp32(m)
    return m


def main() -> None:
    args = Hyperparameters()
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_tokens = load_validation_tokens(args.val_files, max(args.train_seq_len, effective_eval_seq_len))
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    if master_process:
        print(f"val tokens: {val_tokens.numel() - 1}")

    model_path = os.environ.get("MODEL_PATH", "final_model.int6.ptz")
    if master_process:
        print(f"Loading model from {model_path}")

    with open(model_path, "rb") as f:
        quant_blob = f.read()
    if _COMPRESSOR == "zstd":
        quant_raw = zstandard.ZstdDecompressor().decompress(quant_blob)
    else:
        quant_raw = zlib.decompress(quant_blob)
    quant_state = torch.load(io.BytesIO(quant_raw), map_location="cpu")

    CastedLinear._qat_enabled = False
    template_model = make_model(args, torch.device("cpu"))
    template_sd = {k: v.cpu() for k, v in template_model.state_dict().items()}
    del template_model

    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], template_sd)

    # ── Baseline eval ────────────────────────────────────────────────────
    if not TTT_SKIP_BASELINE:
        eval_model = make_model(args, device)
        eval_model.load_state_dict(deq_state, strict=True)

        if master_process:
            print(f"\n{'='*60}")
            print(f"BASELINE EVAL (sliding window, stride={args.eval_stride})")
            print(f"{'='*60}")

        torch.cuda.synchronize()
        t_base = time.perf_counter()
        base_val_loss, base_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, eval_seq_len=effective_eval_seq_len,
        )
        torch.cuda.synchronize()
        base_time = time.perf_counter() - t_base
        if master_process:
            print(f"Baseline bpb: {base_val_bpb:.6f} time: {base_time:.1f}s")

        del eval_model
        torch.cuda.empty_cache()
    else:
        base_val_bpb = 1.1269  # known from T015 (PR#414 A800 baseline)
        base_time = 0.0
        if master_process:
            print(f"\nSkipping baseline eval (known: {base_val_bpb})")

    # ── TTT eval ─────────────────────────────────────────────────────────
    ttt_model = make_model(args, device)
    ttt_model.load_state_dict(deq_state, strict=True)

    if master_process:
        print(f"\n{'='*60}")
        print(f"TTT EVAL (score-first)")
        print(f"{'='*60}")

    torch.cuda.synchronize()
    ttt_bpb, ttt_time, num_chunks, chunk_timings, temp_bpb = eval_ttt_score_first(
        args, ttt_model, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()

    if master_process:
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Baseline sliding-window bpb: {base_val_bpb:.6f}")
        print(f"TTT score-first bpb:         {ttt_bpb:.6f}")
        print(f"Delta (TTT - baseline):      {ttt_bpb - base_val_bpb:.6f}")
        print(f"Baseline eval time:          {base_time:.1f}s")
        print(f"TTT eval time:               {ttt_time:.1f}s")
        print(f"Chunks processed:            {int(num_chunks)}")
        print(f"Peak GPU memory:             {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
        print(f"LR schedule:                 {TTT_LR_SCHEDULE}")
        if chunk_timings:
            avg_ct = sum(chunk_timings) / len(chunk_timings)
            h100_factor = 1.5  # A800→H100 speedup estimate
            h100_chunk_s = avg_ct / h100_factor
            h100_600s_chunks = int(600.0 / h100_chunk_s)
            print(f"Avg chunk time (A800):       {avg_ct:.3f}s")
            print(f"Est chunk time (H100):       {h100_chunk_s:.3f}s")
            print(f"Est max chunks in 600s H100: {h100_600s_chunks}")

        # ── Temperature sweep results ─────────────────────────────────────
        bpb_at_1 = temp_bpb.get(1.0, ttt_bpb)
        best_T = min(temp_bpb, key=temp_bpb.get)
        best_bpb = temp_bpb[best_T]

        print(f"\n{'='*60}")
        print(f"TEMPERATURE SWEEP RESULTS")
        print(f"{'='*60}")
        print(f"{'T':>6s}  {'bpb':>10s}  {'delta_vs_T1':>12s}  {'delta_vs_base':>14s}")
        print(f"{'-'*6}  {'-'*10}  {'-'*12}  {'-'*14}")
        for T in sorted(temp_bpb.keys()):
            bpb_t = temp_bpb[T]
            delta_t1 = bpb_t - bpb_at_1
            delta_base = bpb_t - base_val_bpb
            marker = " <-- best" if T == best_T else ""
            print(f"{T:6.2f}  {bpb_t:10.6f}  {delta_t1:+12.6f}  {delta_base:+14.6f}{marker}")
        print(f"\nBest temperature: T={best_T:.2f}, bpb={best_bpb:.6f}")
        print(f"Improvement vs T=1.0: {bpb_at_1 - best_bpb:.6f}")
        print(f"Improvement vs baseline: {base_val_bpb - best_bpb:.6f}")

        # Check kill criteria
        improvement = bpb_at_1 - best_bpb
        if improvement < 0.001:
            print(f"\n⚠ KILL: Temperature gain {improvement:.6f} < 0.001 — not worth pursuing")
        if best_T > 0.995:
            print(f"\n⚠ FALSIFIED: Optimal T={best_T:.2f} > 0.995 — model is already well-calibrated")

        results = {
            "baseline_bpb": base_val_bpb,
            "ttt_bpb": ttt_bpb,
            "delta_bpb": ttt_bpb - base_val_bpb,
            "baseline_time_s": base_time,
            "ttt_time_s": ttt_time,
            "total_time_s": base_time + ttt_time,
            "num_chunks": int(num_chunks),
            "ttt_lr": TTT_LR,
            "ttt_epochs": TTT_EPOCHS,
            "ttt_chunk_tokens": TTT_CHUNK_TOKENS,
            "ttt_freeze_blocks": TTT_FREEZE_BLOCKS,
            "ttt_freeze_embeddings": TTT_FREEZE_EMBEDDINGS,
            "ttt_grad_clip": TTT_GRAD_CLIP,
            "ttt_optimizer": TTT_OPTIMIZER,
            "ttt_eval_stride": TTT_EVAL_STRIDE,
            "ttt_max_chunks": TTT_MAX_CHUNKS,
            "ttt_skip_baseline": TTT_SKIP_BASELINE,
            "ttt_lr_schedule": TTT_LR_SCHEDULE,
            "peak_gpu_mib": torch.cuda.max_memory_allocated() // 1024 // 1024,
            "avg_chunk_time_s": sum(chunk_timings) / max(len(chunk_timings), 1),
            "chunk_timings_s": chunk_timings,
            "total_possible_chunks": int((val_tokens.numel() - 2 + TTT_CHUNK_TOKENS - 1) // TTT_CHUNK_TOKENS),
            "temp_sweep": {str(T): bpb_t for T, bpb_t in sorted(temp_bpb.items())},
            "best_temperature": best_T,
            "best_temp_bpb": best_bpb,
            "temp_gain_vs_t1": float(bpb_at_1 - best_bpb),
        }
        with open("ttt_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to ttt_results.json")

        import shutil
        records_dir = Path("records/ttt_eval")
        records_dir.mkdir(parents=True, exist_ok=True)
        ts_name = f"ttt_temp_sweep_{int(time.time())}.json"
        shutil.copy("ttt_results.json", records_dir / ts_name)
        print(f"Results copied to records/ttt_eval/{ts_name}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
