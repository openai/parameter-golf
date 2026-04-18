#!/usr/bin/env python3
"""Spec 001: Hessian-SDClip lambda sweep.

Loads a spec-000 checkpoint, applies EMA, computes Hessians ONCE (saved to disk
for reuse across sessions), then loops over lambdas in `lambdas.txt`, quantizing
and evaluating for each. Idempotent: skips any lambda with existing JSON.
"""
import argparse
import io
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/workspace/parameter-golf")
from train_gpt_sota import (
    GPT,
    Hyperparameters,
    ShuffledSequenceLoader,
    ValidationData,
    _compress,
    _decompress,
    collect_hessians,
    dequantize_mixed,
    eval_val,
    gptq_mixed_quantize,
    log,
    restore_fp32_params,
    set_logging_hparams,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--run_dir", required=True)
    args = p.parse_args()

    random.seed(1337); np.random.seed(1337); torch.manual_seed(1337); torch.cuda.manual_seed_all(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    h = Hyperparameters()
    set_logging_hparams(h)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    os.makedirs(args.run_dir, exist_ok=True)

    # 1. Load checkpoint
    log(f"loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    log(f"checkpoint step: {ckpt.get('step')}")

    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    base_model.load_state_dict(ckpt["model_state_dict"])

    # Apply EMA weights so we quantize the same post-EMA state as spec-000's pipeline
    if ckpt.get("ema_state"):
        current = base_model.state_dict()
        ema = {k: v.to(device=device, dtype=current[k].dtype) for k, v in ckpt["ema_state"].items()}
        base_model.load_state_dict(ema, strict=True)
        log("applied EMA weights from checkpoint")
    else:
        log("WARN: checkpoint has no EMA state — quantizing raw weights (will NOT match spec-000 baseline)")

    # Activate recurrence if model uses it (matches inference behavior)
    if h.num_loops > 0:
        base_model.looping_active = True
        log(f"recurrence active: encoder={base_model.encoder_indices} decoder={base_model.decoder_indices}")

    # 2. Hessian: load cached OR compute once
    hessian_path = os.path.join(args.run_dir, "hessians.pt")
    if os.path.exists(hessian_path):
        hessians = torch.load(hessian_path, map_location="cpu", weights_only=False)
        log(f"RELOADED Hessians from {hessian_path} ({len(hessians)} keys)")
    else:
        log("computing Hessians (one-time, ~3-5 min)...")
        calib_loader = ShuffledSequenceLoader(h, device)
        t0 = time.perf_counter()
        hessians = collect_hessians(base_model, calib_loader, h, device, n_calibration_batches=h.gptq_calibration_batches)
        dt = time.perf_counter() - t0
        log(f"Hessians collected in {dt:.1f}s — saving to {hessian_path}")
        torch.save(hessians, hessian_path)
        log(f"Hessians SAVED ({len(hessians)} keys)")

    # 3. Prepare template sd (for dequantize) + val data
    sd_cpu = {k: v.detach().cpu() for (k, v) in base_model.state_dict().items()}
    val_data = ValidationData(h, device)

    # 4. Lambdas
    lambdas_path = os.path.join(args.run_dir, "lambdas.txt")
    with open(lambdas_path) as f:
        lambdas = [float(line.strip()) for line in f if line.strip() and not line.strip().startswith("#")]
    log(f"lambdas to process: {lambdas}")

    for lam in lambdas:
        tag = f"{lam:.2f}"
        jpath = os.path.join(args.run_dir, f"lambda_{tag}.json")
        if os.path.exists(jpath):
            log(f"λ={tag}: already done (json exists), skipping")
            continue

        log(f"λ={tag}: starting")
        h.hessian_clip_lambda = lam
        t0 = time.perf_counter()

        # Quantize (uses cached hessians + current lambda)
        quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, h)

        # Serialize to .ptz matching serialize()'s format
        buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta}, buf)
        quant_raw = buf.getvalue()
        quant_blob = _compress(quant_raw, h.compressor)
        ptz_path = os.path.join(args.run_dir, f"lambda_{tag}.ptz")
        with open(ptz_path, "wb") as fout:
            fout.write(quant_blob)
        size = len(quant_blob)

        # Eval: dequantize inline (avoid fixed-path deserialize)
        deq_state = dequantize_mixed(quant_result, quant_meta, sd_cpu)
        eval_model = GPT(h).to(device).bfloat16()
        restore_fp32_params(eval_model)
        eval_model.load_state_dict(deq_state, strict=True)
        if h.num_loops > 0:
            eval_model.looping_active = True
        compiled = torch.compile(eval_model, dynamic=False, fullgraph=True)
        vloss, vbpb = eval_val(h, device, val_data, compiled)
        dt = time.perf_counter() - t0

        result = {
            "lambda": lam,
            "val_bpb_quantized": float(vbpb),
            "val_loss_quantized": float(vloss),
            "artifact_size_bytes": size,
            "elapsed_sec": dt,
        }
        with open(jpath, "w") as f:
            json.dump(result, f, indent=2)
        log(f"λ={tag}: val_bpb={vbpb:.6f} val_loss={vloss:.6f} size={size} bytes ({dt:.1f}s)")

        del eval_model, compiled, deq_state
        torch._dynamo.reset()
        torch.cuda.empty_cache()

    log("sweep.py done")


if __name__ == "__main__":
    main()
