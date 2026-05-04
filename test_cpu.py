#!/usr/bin/env python3
"""CPU test suite for train_gpt_1218_slot.py -- run before any GPU spend.

Validates: import, hyperparams, model creation, forward pass, code size,
quantization roundtrip, quant MSE, weight distribution, parallel residuals,
and mixed INT5/INT6 quantization.

Usage:
    python test_cpu.py                  # run all tests (default config)
    python test_cpu.py --parallel 6     # test with parallel_start_layer=6
"""

import argparse
import io
import math
import os
import sys
import time
from pathlib import Path

# Force CPU mode and prevent CUDA errors on CPU-only machines
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    return condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=0, help="parallel_start_layer")
    parser.add_argument("--vocab", type=int, default=4096, help="vocab_size")
    parser.add_argument("--layers", type=int, default=11, help="num_layers")
    parser.add_argument("--dim", type=int, default=512, help="model_dim")
    args = parser.parse_args()

    # Set env vars BEFORE import (Hyperparameters reads at class-definition time)
    os.environ["VOCAB_SIZE"] = str(args.vocab)
    os.environ["NUM_LAYERS"] = str(args.layers)
    os.environ["MODEL_DIM"] = str(args.dim)
    os.environ["PARALLEL_START_LAYER"] = str(args.parallel)

    print("=" * 60)
    print("CPU Test Suite for train_gpt_1218_slot.py")
    print("=" * 60)

    # ---- Test 1: Import ----
    print("\n1. Import test")
    t0 = time.perf_counter()
    try:
        import train_gpt_1218_slot as T
        check("import", True, f"{time.perf_counter() - t0:.2f}s")
    except Exception as e:
        check("import", False, str(e))
        sys.exit(1)

    # ---- Test 2: Hyperparameters ----
    print("\n2. Hyperparameter validation")
    h = T.Hyperparameters()
    check("vocab_size", h.vocab_size == args.vocab, f"{h.vocab_size}")
    check("num_layers", h.num_layers == args.layers, f"{h.num_layers}")
    check("model_dim", h.model_dim == args.dim, f"{h.model_dim}")
    check("parallel_start_layer", h.parallel_start_layer == args.parallel, f"{h.parallel_start_layer}")
    check("logit_softcap > 0", h.logit_softcap > 0, f"{h.logit_softcap}")
    check("ttt hyperparams exist", hasattr(h, 'ttt_lr') and hasattr(h, 'ttt_epochs'))

    # ---- Test 3: Model creation ----
    print("\n3. Model creation")
    t0 = time.perf_counter()
    try:
        model = T.GPT(h).float()
        elapsed = time.perf_counter() - t0
        num_params = sum(p.numel() for p in model.parameters())
        check("model_create", True, f"{num_params:,} params in {elapsed:.2f}s")
    except Exception as e:
        check("model_create", False, str(e))
        sys.exit(1)

    # Check parallel residual structure
    if args.parallel > 0:
        print("\n3b. Parallel residuals structure")
        has_parallel = any(b.parallel for b in model.blocks)
        check("parallel_blocks_exist", has_parallel)
        check("lane_merge_exists", model.lane_merge is not None)
        n_par = sum(1 for b in model.blocks if b.parallel)
        n_seq = sum(1 for b in model.blocks if not b.parallel)
        check("parallel_count", n_par == h.num_layers - args.parallel,
              f"{n_par} parallel, {n_seq} sequential")

    # ---- Test 4: Forward pass ----
    print("\n4. Forward pass (CPU)")
    bsz, seq = 2, 64
    x = torch.randint(0, h.vocab_size, (bsz, seq))
    y = torch.randint(0, h.vocab_size, (bsz, seq))
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            loss = model(x, y)
        elapsed = time.perf_counter() - t0
        check("forward_loss_finite", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
        check("forward_time", elapsed < 30.0, f"{elapsed:.2f}s")
    except Exception as e:
        check("forward_pass", False, str(e))

    # Also test forward_hidden + compute_logits path
    try:
        with torch.no_grad():
            hidden = model.forward_hidden(x)
            logits = model.compute_logits(hidden)
        check("forward_hidden_shape", hidden.shape == (bsz, seq, h.model_dim),
              f"{hidden.shape}")
        check("logits_shape", logits.shape == (bsz, seq, h.vocab_size),
              f"{logits.shape}")
    except Exception as e:
        check("forward_hidden", False, str(e))

    # ---- Test 5: Code size ----
    print("\n5. Code size")
    code = Path("train_gpt_1218_slot.py").read_text(encoding="utf-8")
    code_bytes = len(code.encode("utf-8"))
    check("code_under_100KB", code_bytes < 100_000, f"{code_bytes:,} bytes")

    # ---- Test 6: INT6 quantization roundtrip ----
    print("\n6. INT6 quantization roundtrip")
    T.set_logging_hparams(h)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    quant_result, quant_meta, deq_sd = None, None, None
    try:
        quant_result, quant_meta = T.mixed_quantize_int6(
            sd, {"mlp", "attn"},
            num_layers=h.num_layers, n_int6_mlp_layers=h.n_int6_mlp_layers)
        deq_sd = T.dequantize_mixed_int6(quant_result, quant_meta, sd)
        check("quant_roundtrip_keys", set(deq_sd.keys()) == set(sd.keys()),
              f"orig={len(sd)} deq={len(deq_sd)}")
    except Exception as e:
        check("quant_roundtrip", False, str(e))

    # ---- Test 7: Quant MSE ----
    print("\n7. Quantization MSE")
    if deq_sd is not None:
        total_mse = 0.0
        total_els = 0
        for name in sd:
            if name in deq_sd and sd[name].is_floating_point():
                diff = (sd[name].float() - deq_sd[name].float())
                total_mse += diff.pow(2).sum().item()
                total_els += sd[name].numel()
        avg_mse = total_mse / max(total_els, 1)
        check("quant_mse_low", avg_mse < 0.01, f"MSE={avg_mse:.6f}")

    # ---- Test 8: Mixed INT5/INT6 layer assignment ----
    print("\n8. Mixed INT5/INT6 quantization")
    if quant_meta is not None:
        int5_layers = [n for n, m in quant_meta.items() if isinstance(m, dict) and m.get("type") == "int5"]
        int6_layers = [n for n, m in quant_meta.items() if isinstance(m, dict) and "int6" in str(m.get("type", ""))]
        check("int5_layers_exist", len(int5_layers) > 0 or h.n_int6_mlp_layers >= h.num_layers,
              f"{len(int5_layers)} INT5 layers")
        check("int6_layers_exist", len(int6_layers) > 0, f"{len(int6_layers)} INT6 layers")
        if int5_layers:
            print(f"    INT5 middle MLP: {sorted(int5_layers)[:4]}...")
        if int6_layers:
            print(f"    INT6 edge/attn:  {sorted(int6_layers)[:4]}...")

    # ---- Test 9: Weight distribution ----
    print("\n9. Weight distribution")
    # Skip params that are intentionally constant or zero-init
    skip_patterns = {"skip_weights", "skip_gates", "attn_scale", "mlp_scale", "lane_merge",
                     "route", "ve_layer_scales", "resid_mix", ".proj.weight"}
    for name, p in model.named_parameters():
        if p.numel() > 1000 and not any(sp in name for sp in skip_patterns):
            std = p.detach().float().std().item()
            if std == 0.0:
                check(f"nonzero_std:{name}", False, "std=0.0 (dead weights)")
                break
            if std > 100.0:
                check(f"reasonable_std:{name}", False, f"std={std:.4f} (exploded)")
                break
    else:
        check("weight_distribution", True, "all weights have reasonable std")

    # ---- Test 10: Compression estimate ----
    print("\n10. Compression estimate")
    quant_buf = io.BytesIO()
    if quant_result is not None:
        torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
        raw_bytes = len(quant_buf.getvalue())
        # Estimate brotli ratio ~0.45 for quantized weights (INT5 compresses better)
        est_compressed = int(raw_bytes * 0.45)
        est_total = est_compressed + code_bytes
        check("estimated_under_16MB", est_total < 16_777_216,
              f"~{est_total:,} bytes ({est_total/1024/1024:.2f} MB)")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
