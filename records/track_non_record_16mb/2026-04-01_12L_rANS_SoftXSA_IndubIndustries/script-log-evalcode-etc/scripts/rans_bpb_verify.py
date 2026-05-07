"""
rANS BPB Verification Script
Proves rANS roundtrip produces bit-identical eval results.

Path A: quantize → dequantize directly (never compressed)
Path B: quantize → rANS encode → write disk → read disk → rANS decode → dequantize

Both paths eval with EVAL_STRIDE=0. val_loss and BPB must match to 10 decimal places.
"""
import os, sys, math, io, torch, numpy as np
import torch.nn.functional as F

# Force single-GPU, no compile, no distributed
os.environ.setdefault("TORCH_COMPILE", "0")
os.environ.setdefault("EVAL_STRIDE", "0")
os.environ.setdefault("VAL_LOSS_EVERY", "0")
os.environ.setdefault("ITERATIONS", "1")
os.environ.setdefault("WARMUP_STEPS", "0")
os.environ.setdefault("SWA_ENABLED", "0")
os.environ.setdefault("MAX_WALLCLOCK_SECONDS", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import (
    Hyperparameters, GPT, mixed_quantize_int6, dequantize_mixed_int6,
    encode_artifact_rans, decode_artifact_rans,
    build_sentencepiece_luts, load_validation_tokens,
    eval_val, restore_low_dim_params_to_fp32, CastedLinear,
)
import sentencepiece as spm

def main():
    args = Hyperparameters()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    # --- Build model ---
    print(f"Building model: {args.num_layers}L dim{args.model_dim} mlp{args.mlp_mult} kv{args.num_kv_heads}")
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
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        mlp_widths=args.mlp_widths if args.mlp_widths else None,
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)

    # --- Load val data + LUTs ---
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    if args.val_token_limit > 0 and val_tokens.numel() - 1 > args.val_token_limit:
        val_tokens = val_tokens[:args.val_token_limit + 1]
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    print(f"Val tokens: {val_tokens.numel() - 1}")

    # --- Get a real state dict (random init is fine, we just need both paths identical) ---
    sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # --- Magnitude pruning (same as serialize path) ---
    with torch.no_grad():
        for name, param in sd_cpu.items():
            if param.ndim == 2 and param.numel() > 65536:
                threshold = torch.quantile(param.abs().float().flatten(), 0.03)
                mask = param.abs() < threshold
                param.masked_fill_(mask, 0.0)

    # --- Quantize ---
    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn", "bigram"})
    print(f"Quantized: {len(quant_result)} tensors")

    # =============================================
    # PATH A: Never compressed — dequantize directly
    # =============================================
    print("\n=== PATH A: Direct dequantize (never compressed) ===")
    deq_state_a = dequantize_mixed_int6(quant_result, quant_meta, sd_cpu)
    model.load_state_dict(deq_state_a, strict=True)
    model.to(device)
    # grad_accum_steps=1 for single GPU eval
    val_loss_a, val_bpb_a = eval_val(
        args, model, rank=0, world_size=1, device=device,
        grad_accum_steps=1, val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )
    print(f"PATH A val_loss: {val_loss_a:.10f}")
    print(f"PATH A val_bpb:  {val_bpb_a:.10f}")

    # Fence: ensure all GPU work from Path A is complete before loading Path B weights
    torch.cuda.synchronize()

    # =============================================
    # PATH B: rANS encode → disk → read → decode → dequantize
    # =============================================
    print("\n=== PATH B: rANS disk roundtrip ===")
    blob = encode_artifact_rans(quant_result, quant_meta)
    artifact_path = "/tmp/rans_verify_artifact.ptz"
    with open(artifact_path, "wb") as f:
        f.write(blob)
    rans_size = os.path.getsize(artifact_path)
    print(f"rANS artifact written: {rans_size} bytes ({rans_size/1e6:.2f} MB)")

    with open(artifact_path, "rb") as f:
        blob_disk = f.read()
    quant_result_rt, quant_meta_rt = decode_artifact_rans(blob_disk)

    deq_state_b = dequantize_mixed_int6(quant_result_rt, quant_meta_rt, sd_cpu)
    model.load_state_dict(deq_state_b, strict=True)
    model.to(device)
    val_loss_b, val_bpb_b = eval_val(
        args, model, rank=0, world_size=1, device=device,
        grad_accum_steps=1, val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )
    print(f"PATH B val_loss: {val_loss_b:.10f}")
    print(f"PATH B val_bpb:  {val_bpb_b:.10f}")

    # =============================================
    # VERDICT
    # =============================================
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    print(f"PATH A val_loss: {val_loss_a:.10f}")
    print(f"PATH B val_loss: {val_loss_b:.10f}")
    print(f"val_loss match:  {val_loss_a == val_loss_b}")
    print(f"val_loss delta:  {abs(val_loss_a - val_loss_b):.15e}")
    print()
    print(f"PATH A val_bpb:  {val_bpb_a:.10f}")
    print(f"PATH B val_bpb:  {val_bpb_b:.10f}")
    print(f"val_bpb match:   {val_bpb_a == val_bpb_b}")
    print(f"val_bpb delta:   {abs(val_bpb_a - val_bpb_b):.15e}")
    print()
    print(f"rANS artifact:   {rans_size} bytes ({rans_size/1e6:.2f} MB)")
    print()

    if val_loss_a == val_loss_b and val_bpb_a == val_bpb_b:
        print("VERDICT: PASS — rANS compression is bit-identical. Ship it.")
    else:
        print("VERDICT: FAIL — values differ. Investigate before shipping.")

    # Cleanup
    os.remove(artifact_path)

if __name__ == "__main__":
    main()
