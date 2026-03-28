"""MC Dropout eval: run K forward passes with dropout ON, average softmax probs, compute BPB.

Each softmax distribution sums to 1.0. The average of K such distributions also sums to 1.0.
No normalization needed — this is a convex combination of valid probability distributions.

Usage (single GPU):
    python eval.py --checkpoint final_model.pt --K 16

Usage (distributed, 8 GPU):
    torchrun --nproc_per_node=8 eval.py --checkpoint final_model.pt --K 16
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

# Import model + helpers from our training script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (
    CastedLinear,
    GPT,
    Hyperparameters,
    build_sentencepiece_luts,
    eval_val,
    load_validation_tokens,
    restore_low_dim_params_to_fp32,
)


def mc_dropout_eval(
    model: GPT,
    args: Hyperparameters,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    K: int,
) -> tuple[float, float]:
    """Run K forward passes with dropout ON, average probs, compute BPB."""
    seq_len = args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Keep dropout ON
    model.train()

    with torch.no_grad():
        for bs in range(seq_start, seq_end, batch_seqs):
            be = min(bs + batch_seqs, seq_end)
            raw_start = bs * seq_len
            raw_end = be * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            n_tokens = y.numel()

            # Run K forward passes and average softmax probs
            avg_probs = torch.zeros(n_tokens, args.vocab_size, device=device, dtype=torch.float32)
            for k in range(K):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    logits = model(x)  # [B*T, V]
                avg_probs += F.softmax(logits.float(), dim=-1)
            avg_probs /= K

            # Sanity check: probs should sum to ~1.0
            prob_sums = avg_probs.sum(dim=-1)
            assert prob_sums.min() > 0.99 and prob_sums.max() < 1.01, \
                f"Probability sums out of range: min={prob_sums.min():.6f} max={prob_sums.max():.6f}"

            # Cross-entropy from averaged probs: -log(p[target])
            targets = y.reshape(-1)
            target_probs = avg_probs[torch.arange(n_tokens, device=device), targets]
            batch_loss = -torch.log(target_probs.clamp(min=1e-10)).sum()

            val_loss_sum += batch_loss.to(torch.float64)
            val_token_count += n_tokens

            # BPB byte counting
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = (val_loss_sum / val_token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    val_bpb = bits_per_token * tokens_per_byte

    return float(val_loss), float(val_bpb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to final_model.pt")
    parser.add_argument("--K", type=int, default=16, help="Number of MC forward passes")
    cli = parser.parse_args()

    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    # Match training's SDP backend settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    grad_accum_steps = 8 // world_size

    # Load tokenizer + val data
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Build model matching training's precision setup
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
        dropout_rate=args.dropout_rate,
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)

    # Load checkpoint
    state = torch.load(cli.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    if master:
        print(f"Loaded checkpoint: {cli.checkpoint}")
        print(f"MC Dropout K={cli.K}, dropout_rate={args.dropout_rate}")

    # Compile model (matches training)
    compiled_model = torch.compile(model, dynamic=False, fullgraph=True)

    # Baseline: use training's exact eval_val (dropout OFF via model.eval())
    t0 = time.perf_counter()
    bl, bb = eval_val(
        args, compiled_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    if master:
        print(f"Baseline (dropout OFF): val_loss={bl:.6f} val_bpb={bb:.6f} time={time.perf_counter()-t0:.1f}s")

    # MC Dropout eval (dropout ON)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    mc_loss, mc_bpb = mc_dropout_eval(
        compiled_model, args, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, cli.K,
    )
    torch.cuda.synchronize()
    if master:
        print(f"MC Dropout (K={cli.K}): val_loss={mc_loss:.6f} val_bpb={mc_bpb:.6f} time={time.perf_counter()-t0:.1f}s")
        delta = mc_bpb - bb
        print(f"Delta BPB (MC - baseline): {delta:+.6f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
