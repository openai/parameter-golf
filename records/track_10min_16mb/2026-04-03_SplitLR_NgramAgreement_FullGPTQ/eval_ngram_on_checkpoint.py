#!/usr/bin/env python3
"""Evaluate n-gram agreement on a saved int6 checkpoint."""
from __future__ import annotations
import io
import os
import sys
import time

import brotli
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.pr1179_baseline.train_gpt import (
    GPT,
    CastedLinear,
    Hyperparameters,
    _byte_unshuffle,
    _unbank_state_dict,
    _rebank_state_dict,
    build_sentencepiece_luts,
    dequantize_mixed_int6,
    load_validation_tokens,
    restore_low_dim_params_to_fp32,
)
from online_best_agree_eval import eval_val_sliding_online_best_agree


def main():
    args = Hyperparameters()
    args.bigram_dim = int(os.environ.get("BIGRAM_DIM", "160"))

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

    def log0(msg, console=True):
        if master and console:
            print(msg, flush=True)

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Load int6 checkpoint
    ptz_path = os.environ.get("CHECKPOINT", "final_model.int6.ptz")
    log0(f"Loading checkpoint: {ptz_path}")
    with open(ptz_path, "rb") as f:
        quant_blob = f.read()
    quant_state = torch.load(
        io.BytesIO(_byte_unshuffle(brotli.decompress(quant_blob))),
        map_location="cpu",
    )

    # Build model
    eval_model = GPT(
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
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        neg_slope=args.negative_slope,
    ).to(device).bfloat16()
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)

    # Dequantize and load weights
    template_sd = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}
    unbanked_template = _unbank_state_dict(template_sd, args.num_layers)
    deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_template)
    deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, template_sd)
    eval_model.load_state_dict(deq_state, strict=True)
    eval_model.eval()

    log0(f"Model loaded, running n-gram agreement eval...")
    t0 = time.perf_counter()
    _, best_bpb, timings = eval_val_sliding_online_best_agree(
        args=args,
        base_model=eval_model,
        rank=rank,
        world_size=world_size,
        device=device,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        stride=args.eval_stride,
        batch_seqs=32,
        eval_seq_len=args.train_seq_len,
        log0=log0,
    )
    elapsed = time.perf_counter() - t0
    log0(f"n-gram agreement BPB: {best_bpb:.8f} (elapsed: {elapsed:.1f}s)")
    log0(f"LLM-only BPB: {timings['llm_bpb']:.8f}")
    log0(f"Gain: {timings['gain_bpb']:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
