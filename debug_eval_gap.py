#!/usr/bin/env python3
"""
Deep debug: compare parameter values between saved checkpoint and fresh model.
Also test eval_val with the model INSIDE the training context.

Usage:
    MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 python debug_eval_gap.py
"""
import io
import zlib
import torch
import torch.nn.functional as F
from torch import Tensor

from train_gpt import (
    CastedLinear,
    GPT,
    Hyperparameters,
    build_sentencepiece_luts,
    dequantize_state_dict_int8,
    eval_val,
    load_validation_tokens,
    restore_low_dim_params_to_fp32,
)


def main():
    device = torch.device("cuda")
    args = Hyperparameters()

    # Load raw state dict from disk
    print("=== Loading raw checkpoint ===")
    raw_state = torch.load("final_model.pt", map_location="cpu")

    # Print param stats
    for name in sorted(raw_state.keys())[:5]:
        t = raw_state[name]
        print(f"  {name}: shape={t.shape} dtype={t.dtype} mean={t.float().mean():.6f} std={t.float().std():.6f}")

    # Create model A (standard way)
    print("\n=== Creating model (standard init) ===")
    model_a = GPT(
        vocab_size=args.vocab_size, num_unique_blocks=args.num_unique_blocks,
        num_loops=args.num_loops, model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for m in model_a.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model_a)

    # Check model param dtypes BEFORE load
    print("  Model param dtypes (before load):")
    for name, p in list(model_a.named_parameters())[:5]:
        print(f"    {name}: dtype={p.dtype}")

    # Load state dict
    model_a.load_state_dict(raw_state, strict=True)
    model_a.to(device)

    # Check model param dtypes AFTER load
    print("  Model param dtypes (after load):")
    for name, p in list(model_a.named_parameters())[:5]:
        print(f"    {name}: dtype={p.dtype} device={p.device}")

    # Compare state dicts element-wise
    print("\n=== Comparing loaded params vs checkpoint ===")
    loaded_state = model_a.state_dict()
    max_diff = 0.0
    for name in sorted(raw_state.keys()):
        orig = raw_state[name].float()
        loaded = loaded_state[name].float().cpu()
        diff = (orig - loaded).abs().max().item()
        if diff > 1e-6:
            print(f"  MISMATCH {name}: max_diff={diff:.8f} orig_dtype={raw_state[name].dtype} loaded_dtype={loaded_state[name].dtype}")
        max_diff = max(max_diff, diff)
    print(f"  Max diff across all params: {max_diff:.8f}")

    # Quick single-batch test
    print("\n=== Single batch forward test ===")
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    luts = build_sentencepiece_luts(sp, args.vocab_size, device)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)

    model_a.eval()
    with torch.inference_mode():
        chunk = val_tokens[:args.train_seq_len + 1].to(device=device, dtype=torch.int64)
        x = chunk[:-1].reshape(1, args.train_seq_len)
        y = chunk[1:].reshape(1, args.train_seq_len)

        # Test WITHOUT autocast
        loss_nocast = model_a(x, y)
        print(f"  No autocast: loss={loss_nocast.item():.4f}")

        # Test WITH autocast
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss_cast = model_a(x, y)
        print(f"  With autocast: loss={loss_cast.item():.4f}")

    # Full eval_val
    print("\n=== Full eval_val ===")
    vl, vb = eval_val(args, model_a, 0, 1, device, 8, val_tokens, *luts)
    print(f"  val_loss={vl:.4f} val_bpb={vb:.4f}")

    # NOW: also try eval_val with CUDA settings matching training
    print("\n=== eval_val with training CUDA settings ===")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    vl2, vb2 = eval_val(args, model_a, 0, 1, device, 8, val_tokens, *luts)
    print(f"  val_loss={vl2:.4f} val_bpb={vb2:.4f}")


if __name__ == "__main__":
    main()
