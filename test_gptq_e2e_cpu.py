#!/usr/bin/env python3
"""End-to-end CPU integration test for GPTQ in train_gpt_v2.

Tests the FULL pipeline: build model → train (few steps) → GPTQ quantize → save → load → eval.
Uses synthetic data to avoid loading 62M val tokens on CPU.
This catches integration bugs that unit tests miss.
"""
import sys
import time
import io
import lzma
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/tmp/pgolf-repo")
from train_gpt_v2 import (
    HybridGPT,
    CastedLinear,
    RMSNorm,
    Muon,
    SmearGate,
    BigramHashEmbedding,
    restore_low_dim_params_to_fp32,
    quantize_state_dict_int6,
    dequantize_state_dict_int6,
    generate_autoregressive_calib,
    collect_hessians_from_tokens,
    CONTROL_TENSOR_NAME_PATTERNS,
)


def main():
    print("\n" + "=" * 60)
    print("  GPTQ E2E CPU INTEGRATION TEST")
    print("=" * 60 + "\n")

    device = torch.device("cpu")
    torch.manual_seed(1337)

    # =========================================================================
    # STEP 1: Build model (same as Phase 2 in train_gpt_v2.py)
    # =========================================================================
    print("STEP 1: Building model...")
    vocab_size = 256  # big enough for tok_emb quantization
    num_layers = 3
    model_dim = 256  # big enough for weight quantization (numel > 65536)
    num_heads = 4
    num_kv_heads = 4
    mlp_mult = 3     # matches real config
    seq_len = 64
    xsa_last_n = 2    # XSA on last 2 layers
    smear_gate = True
    bigram_hash_vocab = 512
    bigram_hash_dim = 64

    bigram_table = torch.randn(vocab_size, vocab_size, dtype=torch.float16) * 0.1

    model = HybridGPT(
        vocab_size=vocab_size,
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        bigram_table=bigram_table,
        bigram_scale_mode="0",
        xsa_last_n=xsa_last_n,
        smear_gate=smear_gate,
        bigram_hash_vocab=bigram_hash_vocab,
        bigram_hash_dim=bigram_hash_dim,
    ).to(device)

    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)

    n_params = sum(p.numel() for p in model.parameters())
    # Verify features are wired
    assert model.smear is not None, "SmearGate not created"
    assert model.bigram_hash is not None, "BigramHash not created"
    xsa_count = sum(1 for b in model.blocks if b.attn.use_xsa)
    assert xsa_count == xsa_last_n, f"XSA on {xsa_count} layers, expected {xsa_last_n}"

    print(f"  Model: {num_layers}L × {model_dim}d, {n_params:,} params")
    print(f"  Features: XSA(last {xsa_last_n}), SmearGate, BigramHash(vocab={bigram_hash_vocab})")
    print(f"  ✓ Model built\n")

    # =========================================================================
    # STEP 2: Train for a few steps
    # =========================================================================
    print("STEP 2: Training (5 steps)...")

    # Simple optimizer setup (skip Muon for CPU speed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(1, 6):
        model.train()
        x = torch.randint(0, vocab_size, (4, seq_len))
        y = torch.randint(0, vocab_size, (4, seq_len))
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  step {step}: loss={loss.item():.4f}")

    print(f"  ✓ Training complete\n")

    # =========================================================================
    # STEP 3: Pre-quant eval
    # =========================================================================
    print("STEP 3: Pre-quant eval...")
    model.eval()
    with torch.inference_mode():
        x_eval = torch.randint(0, vocab_size, (8, seq_len))
        y_eval = torch.randint(0, vocab_size, (8, seq_len))
        pre_quant_loss = model(x_eval, y_eval).item()
    print(f"  Pre-quant loss: {pre_quant_loss:.4f}")
    print(f"  ✓ Pre-quant eval done\n")

    # =========================================================================
    # STEP 4: GPTQ — generate AR calib, collect Hessians
    # =========================================================================
    print("STEP 4: GPTQ Hessian collection...")
    t0 = time.perf_counter()

    # AR calibration generation
    print("  Generating AR calibration data (4 seqs × 32 tokens)...")
    ar_tokens = generate_autoregressive_calib(
        model, device, num_seqs=4, seq_len=seq_len,
        vocab_size=vocab_size, temperature=0.8, batch_size=2, seed=42,
    )
    t_gen = time.perf_counter() - t0
    print(f"  Generated {len(ar_tokens)} sequences in {t_gen:.1f}s")

    # Verify AR tokens are valid
    for i, seq in enumerate(ar_tokens):
        assert seq.shape == (1, seq_len), f"AR seq {i} shape {seq.shape} != (1, {seq_len})"
        assert seq.min() >= 0 and seq.max() < vocab_size, f"AR seq {i} has OOB tokens"
    print(f"  AR tokens valid: shape OK, range [0, {vocab_size})")

    # Collect Hessians
    print("  Collecting Hessians...")
    t_hess = time.perf_counter()
    hessians = collect_hessians_from_tokens(model, ar_tokens, device)
    t_hess = time.perf_counter() - t_hess
    print(f"  Collected {len(hessians)} Hessians in {t_hess:.1f}s:")
    for name, H in sorted(hessians.items()):
        diag = torch.diag(H)
        print(f"    {name:40s} shape={tuple(H.shape)} diag=[{diag.min().item():.4f}, {diag.max().item():.4f}]")
        assert diag.min().item() > 0, f"Non-positive diagonal in {name}!"
    print(f"  ✓ Hessian collection done\n")

    # =========================================================================
    # STEP 5: Quantize with GPTQ
    # =========================================================================
    print("STEP 5: Quantizing (GPTQ + percentile comparison)...")

    full_sd = model.state_dict()
    bigram_table_save = full_sd.pop("bigram_table").cpu().half()
    bigram_scale_save = full_sd.pop("bigram_scale").cpu().float()
    neural_sd = {k: v.detach().cpu() for k, v in full_sd.items()}

    # Quantize WITH GPTQ
    t_gptq = time.perf_counter()
    quant_gptq, meta_gptq = quantize_state_dict_int6(neural_sd, hessians=hessians)
    t_gptq = time.perf_counter() - t_gptq

    # Quantize WITHOUT GPTQ (percentile only)
    t_pct = time.perf_counter()
    quant_pct, meta_pct = quantize_state_dict_int6(neural_sd, hessians=None)
    t_pct = time.perf_counter() - t_pct

    print(f"  GPTQ quantize: {t_gptq:.1f}s")
    print(f"  Percentile quantize: {t_pct:.1f}s")

    # Compare MSE per layer
    restored_gptq = dequantize_state_dict_int6(quant_gptq, meta_gptq)
    restored_pct = dequantize_state_dict_int6(quant_pct, meta_pct)

    print("\n  Per-layer MSE comparison:")
    gptq_wins = 0
    pct_wins = 0
    for name in sorted(neural_sd.keys()):
        orig = neural_sd[name].float()
        r_gptq = restored_gptq[name].float()
        r_pct = restored_pct[name].float()
        mse_gptq = (orig - r_gptq).pow(2).mean().item()
        mse_pct = (orig - r_pct).pow(2).mean().item()
        is_control = any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)
        if is_control or orig.numel() <= 65536:
            tag = "passthrough"
        elif mse_gptq < mse_pct:
            tag = "GPTQ WINS"
            gptq_wins += 1
        else:
            tag = "pct wins"
            pct_wins += 1
        if not is_control and orig.numel() > 65536:
            print(f"    {name:40s} pct={mse_pct:.2e} gptq={mse_gptq:.2e} [{tag}]")

    print(f"\n  Score: GPTQ={gptq_wins}, Percentile={pct_wins}")
    print(f"  (GPTQ winning on real trained weights is the goal — wins may be partial on tiny CPU model)")
    print(f"  ✓ Quantization done\n")

    # =========================================================================
    # STEP 6: Save artifact (same format as train_gpt_v2.py Phase 3)
    # =========================================================================
    print("STEP 6: Saving artifact...")

    artifact = {
        "bigram_table": bigram_table_save,
        "bigram_scale": bigram_scale_save,
        "neural_weights": quant_gptq,
        "neural_meta": meta_gptq,
        "config": {
            "vocab_size": vocab_size,
            "num_layers": num_layers,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "mlp_mult": mlp_mult,
            "tie_embeddings": True,
            "tied_embed_init_std": 0.005,
            "logit_softcap": 30.0,
            "rope_base": 10000.0,
            "qk_gain_init": 1.5,
            "bigram_scale_mode": "0",
            "xsa_last_n": xsa_last_n,
            "smear_gate": smear_gate,
            "bigram_hash_vocab": bigram_hash_vocab,
            "bigram_hash_dim": bigram_hash_dim,
        },
    }
    buf = io.BytesIO()
    torch.save(artifact, buf)
    raw_bytes = buf.getvalue()
    compressed = lzma.compress(raw_bytes, preset=9)
    artifact_bytes = len(compressed)
    print(f"  Raw: {len(raw_bytes):,} bytes")
    print(f"  Compressed: {artifact_bytes:,} bytes (ratio: {len(raw_bytes)/artifact_bytes:.1f}x)")
    print(f"  ✓ Artifact saved\n")

    # =========================================================================
    # STEP 7: Load artifact + reconstruct model (same as train_gpt_v2.py roundtrip)
    # =========================================================================
    print("STEP 7: Loading artifact + reconstructing model...")

    loaded = torch.load(io.BytesIO(lzma.decompress(compressed)), map_location="cpu")
    loaded_bigram = loaded["bigram_table"].to(device)
    loaded_scale = loaded["bigram_scale"].to(device)
    loaded_neural_sd = dequantize_state_dict_int6(loaded["neural_weights"], loaded["neural_meta"])

    eval_model = HybridGPT(
        bigram_table=loaded_bigram, **loaded["config"]
    ).to(device)
    for module in eval_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(
        {**loaded_neural_sd, "bigram_table": loaded_bigram, "bigram_scale": loaded_scale},
        strict=True,
    )
    print(f"  Loaded config: {loaded['config']}")
    print(f"  ✓ Model reconstructed (strict=True)\n")

    # =========================================================================
    # STEP 8: Post-quant eval (same data as pre-quant)
    # =========================================================================
    print("STEP 8: Post-quant eval...")
    eval_model.eval()
    with torch.inference_mode():
        post_quant_loss = eval_model(x_eval, y_eval).item()

    degradation = post_quant_loss - pre_quant_loss
    print(f"  Pre-quant loss:  {pre_quant_loss:.4f}")
    print(f"  Post-quant loss: {post_quant_loss:.4f}")
    print(f"  Degradation:     {degradation:+.4f}")
    assert post_quant_loss < 100.0, f"Post-quant loss exploded: {post_quant_loss}"
    print(f"  ✓ Post-quant eval done\n")

    # =========================================================================
    # STEP 9: Compare GPTQ vs percentile post-quant
    # =========================================================================
    print("STEP 9: GPTQ vs percentile post-quant comparison...")

    # Load percentile version
    loaded_pct_sd = dequantize_state_dict_int6(quant_pct, meta_pct)
    eval_model_pct = HybridGPT(
        bigram_table=loaded_bigram, **loaded["config"]
    ).to(device)
    for module in eval_model_pct.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(eval_model_pct)
    eval_model_pct.load_state_dict(
        {**loaded_pct_sd, "bigram_table": loaded_bigram, "bigram_scale": loaded_scale},
        strict=True,
    )
    eval_model_pct.eval()
    with torch.inference_mode():
        pct_quant_loss = eval_model_pct(x_eval, y_eval).item()

    print(f"  Pre-quant loss:       {pre_quant_loss:.4f}")
    print(f"  Percentile post-quant: {pct_quant_loss:.4f} (degrad: {pct_quant_loss - pre_quant_loss:+.4f})")
    print(f"  GPTQ post-quant:      {post_quant_loss:.4f} (degrad: {post_quant_loss - pre_quant_loss:+.4f})")
    if post_quant_loss < pct_quant_loss:
        print(f"  → GPTQ wins by {pct_quant_loss - post_quant_loss:.4f}")
    else:
        print(f"  → Percentile wins by {post_quant_loss - pct_quant_loss:.4f}")
        print(f"    (Expected on tiny model — GPTQ shines with larger models + real data)")
    print(f"  ✓ Comparison done\n")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 60)
    print("  E2E INTEGRATION TEST: ALL STEPS PASSED")
    print("=" * 60)
    print(f"  Pipeline: build → train(5 steps) → AR calib → Hessians → GPTQ → save → load → eval")
    print(f"  Pre-quant loss:  {pre_quant_loss:.4f}")
    print(f"  Post-quant GPTQ: {post_quant_loss:.4f} (degrad: {degradation:+.4f})")
    print(f"  Post-quant pct:  {pct_quant_loss:.4f} (degrad: {pct_quant_loss - pre_quant_loss:+.4f})")
    print(f"  Artifact: {artifact_bytes:,} bytes compressed")
    print(f"  Hessians: {len(hessians)} layers collected")
    print(f"  GPTQ layer wins: {gptq_wins}/{gptq_wins + pct_wins}")
    print("=" * 60)


if __name__ == "__main__":
    t_start = time.perf_counter()
    main()
    print(f"\n  Total time: {time.perf_counter() - t_start:.1f}s\n")
