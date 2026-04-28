"""
Mixed-precision sensitivity scan.

For each quantizable matrix, test int7 (clip_range=63) while keeping
everything else at int6 (clip_range=31). Measure:
- extra compressed bytes vs baseline (all int6)
- BPB improvement vs baseline
- efficiency = bpb_gain / extra_bytes

Uses code packing size estimate (~16,600 bytes) to determine available budget.
"""
import io
import os
import sys
import time

import torch
import torch.distributed as dist

from train_pr1493 import (
    Hyperparameters, GPT, ValidationData, ShuffledSequenceLoader,
    collect_hessians, gptq_quantize_weight, gptq_mixed_quantize,
    dequantize_mixed, _compress, _decompress,
    restore_fp32_params, eval_val_sliding, set_logging_hparams, log,
    CastedLinear, classify_param,
)

PACKED_CODE_SIZE = 16_600
CAP = 16_000_000


def quantize_mixed_precision(sd_cpu, hessians, h, promoted_layers=None, promote_bits=7):
    """Quantize with some layers at higher precision."""
    if promoted_layers is None:
        promoted_layers = set()
    result = {}
    meta = {}
    for name, tensor in sd_cpu.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = 'passthrough (float16)'
            continue
        is_embed = 'tok_emb' in name
        if is_embed:
            cs = h.embed_clip_sigmas
            bits = h.embed_bits
        elif name in promoted_layers:
            cs = h.matrix_clip_sigmas
            bits = promote_bits
        else:
            cs = h.matrix_clip_sigmas
            bits = h.matrix_bits
        clip_range = 2 ** (bits - 1) - 1
        q, s = gptq_quantize_weight(t, hessians[name], clip_sigmas=cs, clip_range=clip_range)
        result[name + '.q'] = q
        result[name + '.scale'] = s
        meta[name] = f"gptq (int{bits})"
    return result, meta


def compress_and_measure(quant_result, quant_meta, compressor='brotli'):
    """Compress and return size."""
    quant_buf = io.BytesIO()
    torch.save({'w': quant_result, 'm': quant_meta}, quant_buf)
    quant_blob = _compress(quant_buf.getvalue(), compressor)
    return quant_blob, len(quant_blob)


def main():
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend='nccl', device_id=device)
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    h = Hyperparameters()
    set_logging_hparams(h)

    log(f"Packed code size estimate: {PACKED_CODE_SIZE} bytes")
    log(f"Cap: {CAP:,} bytes")
    log(f"Available for model: {CAP - PACKED_CODE_SIZE:,} bytes")

    # Load model
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    sd_fp32 = torch.load(h.model_path, map_location='cpu', weights_only=True)
    base_model.load_state_dict(sd_fp32)
    sd_cpu = {k: v.detach().cpu() for k, v in sd_fp32.items()}
    log(f"Loaded {h.model_path}")

    # Identify quantizable matrices
    quant_matrices = []
    for name, tensor in sd_cpu.items():
        if tensor.is_floating_point() and tensor.numel() > 65536 and 'tok_emb' not in name:
            quant_matrices.append(name)
    log(f"Quantizable matrices: {len(quant_matrices)}")
    for name in quant_matrices:
        log(f"  {name}: {sd_cpu[name].shape} ({sd_cpu[name].numel():,} params)")

    # Collect Hessians
    log("\nCollecting Hessians...")
    t0 = time.perf_counter()
    calib_loader = ShuffledSequenceLoader(h, device)
    hessians = collect_hessians(base_model, calib_loader, h, device,
                                n_calibration_batches=h.gptq_calibration_batches)
    log(f"Hessians collected in {time.perf_counter() - t0:.1f}s")

    # Phase 1: Baseline (all int6)
    log("\n===== PHASE 1: BASELINE (all int6) =====")
    baseline_result, baseline_meta = quantize_mixed_precision(sd_cpu, hessians, h,
                                                              promoted_layers=set())
    baseline_blob, baseline_model_bytes = compress_and_measure(baseline_result, baseline_meta, h.compressor)
    baseline_total = baseline_model_bytes + PACKED_CODE_SIZE
    log(f"Baseline: model={baseline_model_bytes:,} total={baseline_total:,} margin={CAP - baseline_total:+,}")

    # Eval baseline
    val_data = ValidationData(h, device)
    quant_state = torch.load(io.BytesIO(_decompress(baseline_blob, h.compressor)), map_location='cpu')
    deq_state = dequantize_mixed(quant_state['w'], quant_state['m'], sd_cpu)
    eval_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    if h.num_loops > 0:
        eval_model.looping_active = True
    torch._dynamo.reset()
    _, baseline_bpb = eval_val_sliding(h, device, val_data, eval_model)
    log(f"Baseline BPB (sliding): {baseline_bpb:.8f}")
    del eval_model
    torch.cuda.empty_cache()

    # Phase 2: Per-matrix sensitivity scan (int6 -> int7)
    log("\n===== PHASE 2: SENSITIVITY SCAN (int6 -> int7) =====")
    log(f"{'matrix':<40} {'params':>10} {'extra_bytes':>12} {'bpb':>12} {'gain':>10} {'eff':>12}")
    log("-" * 100)

    results = []
    for name in quant_matrices:
        promoted = {name}
        trial_result, trial_meta = quantize_mixed_precision(sd_cpu, hessians, h,
                                                            promoted_layers=promoted,
                                                            promote_bits=7)
        _, trial_model_bytes = compress_and_measure(trial_result, trial_meta, h.compressor)
        trial_total = trial_model_bytes + PACKED_CODE_SIZE
        extra_bytes = trial_model_bytes - baseline_model_bytes

        if trial_total > CAP:
            log(f"{name:<40} {sd_cpu[name].numel():>10,} {extra_bytes:>+12,} {'OVER':>12} {'--':>10} {'--':>12}")
            continue

        # Eval
        trial_blob_raw = io.BytesIO()
        torch.save({'w': trial_result, 'm': trial_meta}, trial_blob_raw)
        trial_blob = _compress(trial_blob_raw.getvalue(), h.compressor)
        quant_state = torch.load(io.BytesIO(_decompress(trial_blob, h.compressor)), map_location='cpu')
        deq_state = dequantize_mixed(quant_state['w'], quant_state['m'], sd_cpu)

        eval_model = GPT(h).to(device).bfloat16()
        restore_fp32_params(eval_model)
        eval_model.load_state_dict(deq_state, strict=True)
        if h.num_loops > 0:
            eval_model.looping_active = True
        torch._dynamo.reset()
        _, trial_bpb = eval_val_sliding(h, device, val_data, eval_model)

        gain = baseline_bpb - trial_bpb
        eff = gain / max(extra_bytes, 1) * 1e6  # gain per MB
        log(f"{name:<40} {sd_cpu[name].numel():>10,} {extra_bytes:>+12,} {trial_bpb:>12.8f} {gain:>+10.6f} {eff:>12.2f}")
        results.append((name, sd_cpu[name].numel(), extra_bytes, trial_bpb, gain, eff))

        del eval_model
        torch.cuda.empty_cache()

    # Summary
    log("\n===== SUMMARY (sorted by efficiency) =====")
    log(f"{'matrix':<40} {'extra_bytes':>12} {'gain':>10} {'eff (gain/MB)':>14}")
    log("-" * 80)
    results.sort(key=lambda r: -r[5])  # sort by efficiency descending
    for name, params, extra_bytes, trial_bpb, gain, eff in results:
        marker = "+" if gain > 0 else " "
        log(f"{name:<40} {extra_bytes:>+12,} {gain:>+10.6f} {eff:>14.2f}")

    # Budget allocation
    budget = CAP - baseline_total
    log(f"\nAvailable budget: {budget:,} bytes")
    log("\nOptimal allocation (greedy by efficiency):")
    spent = 0
    total_gain = 0.0
    selected = []
    for name, params, extra_bytes, trial_bpb, gain, eff in results:
        if gain <= 0:
            continue
        if spent + extra_bytes <= budget:
            spent += extra_bytes
            total_gain += gain
            selected.append(name)
            log(f"  SELECT: {name} +{extra_bytes:,} bytes, gain={gain:+.6f}")

    log(f"\nTotal: {len(selected)} matrices promoted, {spent:,} extra bytes, {total_gain:+.6f} BPB gain")
    log(f"Estimated final BPB: {baseline_bpb - total_gain:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
