#!/usr/bin/env python3
"""Fast TTT eval script — loads pre-saved quantized model, runs TTT eval only.
Skips the 10-minute training phase for rapid iteration on TTT parameters.

Usage:
  # First: run full training to save model
  RUN_ID=base ... torchrun --nproc=8 train_submission.py

  # Then: fast TTT eval iterations
  TTT_LR=0.001 TTT_EPOCHS=3 ... torchrun --nproc=8 eval_ttt.py
"""
import os, sys, math, time, glob
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
import sentencepiece as spm

# Import everything from the submission script
sys.path.insert(0, str(Path(__file__).parent))
from train_submission import (
    Hyperparameters, GPT, CastedLinear, restore_low_dim_params_to_fp32,
    load_validation_tokens, build_sentencepiece_luts,
    dequantize_mixed_int6, eval_val_sliding, eval_ttt_perdoc,
    BatchedTTTLoRA, BatchedLinearLoRA, BOS_ID,
)

def main():
    args = Hyperparameters()
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
    master_process = rank == 0

    def log0(msg):
        if master_process:
            print(msg)

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Load saved quantized model
    model_path = os.environ.get("MODEL_PATH", "final_int6_model.pt")
    log0(f"Loading quantized model from {model_path}")
    saved = torch.load(model_path, map_location="cpu", weights_only=True)

    # Create fresh eval model and load dequantized weights
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
    log0(f"Model loaded: {sum(p.numel() for p in eval_model.parameters())} params")

    CastedLinear._qat_enabled = False
    CastedLinear._soft_tau = 1000.0

    grad_accum_steps = 8 // world_size

    # Run TTT eval
    if args.ttt_enabled:
        if distributed:
            dist.barrier()
        for block in eval_model.blocks:
            block.attn.rotary._cos_cached = None
            block.attn.rotary._sin_cached = None
            block.attn.rotary._seq_len_cached = 0
        log0(f"ttt:start lr={args.ttt_lr} epochs={args.ttt_epochs} rank={args.ttt_lora_rank}")
        t_ttt = time.perf_counter()
        ttt_val_loss, ttt_val_bpb = eval_ttt_perdoc(
            args, eval_model, device, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            rank=rank, world_size=world_size, log_fn=log0,
        )
        log0(f"ttt:elapsed={time.perf_counter() - t_ttt:.1f}s")
        log0(f"final_ttt val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f}")
        log0(f"final_ttt_exact val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f}")
        if distributed:
            dist.barrier()

    # Also run standard sliding window for comparison
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
    t_slide = time.perf_counter()
    sw_val_loss, sw_val_bpb = eval_val_sliding(
        args, eval_model, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=64, batch_seqs=32,
    )
    log0(f"sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} time:{time.perf_counter()-t_slide:.1f}s")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
