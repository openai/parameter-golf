"""
Post-hoc quantization sweep: find optimal (matrix_clip_sigmas, embed_clip_sigmas)
for PR #1493's trained model.

Phase 1: size-only sweep — quantize+compress all combos, log sizes
Phase 2: eval combos that fit under 16,000,000 bytes
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
    classify_param, CastedLinear,
)

CAP = 16_000_000

MATRIX_KS = [11.75, 12.0, 12.25, 12.5, 12.85, 13.25, 13.75]
EMBED_KS  = [16, 18, 20, 22, 24]

def quantize_with_sigmas(sd_cpu, hessians, h, matrix_k, embed_k):
    import collections, re
    result = {}
    meta = {}
    for name, tensor in sd_cpu.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = 'passthrough (float16)'
            continue
        is_embed = 'tok_emb' in name
        cs = embed_k if is_embed else matrix_k
        bits = h.embed_bits if is_embed else h.matrix_bits
        clip_range = 2 ** (bits - 1) - 1
        q, s = gptq_quantize_weight(t, hessians[name], clip_sigmas=cs, clip_range=clip_range)
        result[name + '.q'] = q
        result[name + '.scale'] = s
        meta[name] = f"gptq (int{bits})"
    return result, meta


def compress_quantized(quant_result, quant_meta, compressor='brotli'):
    quant_buf = io.BytesIO()
    torch.save({'w': quant_result, 'm': quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = _compress(quant_raw, compressor)
    return quant_blob


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
    code_bytes = len(open(__file__.replace('requant_sweep.py', 'train_pr1493.py'),
                          encoding='utf-8').read().encode('utf-8'))
    log(f"Code size: {code_bytes} bytes")

    # Load model
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    sd_fp32 = torch.load(h.model_path, map_location='cpu', weights_only=True)
    base_model.load_state_dict(sd_fp32)
    sd_cpu = {k: v.detach().cpu() for k, v in sd_fp32.items()}
    log(f"Loaded {h.model_path}")

    # Collect Hessians once
    log("Collecting Hessians...")
    t0 = time.perf_counter()
    calib_loader = ShuffledSequenceLoader(h, device)
    hessians = collect_hessians(base_model, calib_loader, h, device,
                                n_calibration_batches=h.gptq_calibration_batches)
    log(f"Hessians collected in {time.perf_counter() - t0:.1f}s")

    # Phase 1: size sweep
    log("\n===== PHASE 1: SIZE SWEEP =====")
    log(f"{'matrix_k':>10} {'embed_k':>10} {'model_bytes':>12} {'total_bytes':>12} {'margin':>10} {'legal':>6}")
    log("-" * 70)

    legal_combos = []
    for matrix_k in MATRIX_KS:
        for embed_k in EMBED_KS:
            quant_result, quant_meta = quantize_with_sigmas(sd_cpu, hessians, h, matrix_k, embed_k)
            quant_blob = compress_quantized(quant_result, quant_meta, h.compressor)
            model_bytes = len(quant_blob)
            total_bytes = model_bytes + code_bytes
            margin = CAP - total_bytes
            legal = total_bytes <= CAP
            marker = "OK" if legal else "OVER"
            log(f"{matrix_k:>10.2f} {embed_k:>10.1f} {model_bytes:>12,} {total_bytes:>12,} {margin:>+10,} {marker:>6}")
            if legal:
                legal_combos.append((matrix_k, embed_k, model_bytes, total_bytes, quant_blob))

    log(f"\n{len(legal_combos)} / {len(MATRIX_KS) * len(EMBED_KS)} combos fit under {CAP:,} bytes")

    if not legal_combos:
        log("No legal combos found. Done.")
        if distributed:
            dist.destroy_process_group()
        return

    # Phase 2: eval legal combos with sliding window
    log("\n===== PHASE 2: SLIDING EVAL =====")
    val_data = ValidationData(h, device)

    results = []
    for matrix_k, embed_k, model_bytes, total_bytes, quant_blob in legal_combos:
        label = f"m{matrix_k}_e{embed_k}"
        log(f"\n--- Evaluating {label} ({total_bytes:,} bytes) ---")

        # Deserialize from blob
        quant_state = torch.load(
            io.BytesIO(_decompress(quant_blob, h.compressor)),
            map_location='cpu'
        )
        deq_state = dequantize_mixed(quant_state['w'], quant_state['m'], sd_cpu)

        eval_model = GPT(h).to(device).bfloat16()
        restore_fp32_params(eval_model)
        eval_model.load_state_dict(deq_state, strict=True)
        if h.num_loops > 0:
            eval_model.looping_active = True

        torch._dynamo.reset()
        t0 = time.perf_counter()
        val_loss, val_bpb = eval_val_sliding(h, device, val_data, eval_model)
        elapsed = time.perf_counter() - t0

        log(f"RESULT: matrix_k={matrix_k} embed_k={embed_k} val_bpb={val_bpb:.8f} "
            f"total_bytes={total_bytes} margin={CAP - total_bytes:+d} eval_time={elapsed:.1f}s")
        results.append((matrix_k, embed_k, val_bpb, total_bytes))

        del eval_model
        torch.cuda.empty_cache()

    # Summary
    log("\n===== SUMMARY =====")
    log(f"{'matrix_k':>10} {'embed_k':>10} {'val_bpb':>12} {'total_bytes':>12} {'margin':>10}")
    log("-" * 60)
    results.sort(key=lambda r: r[2])
    for matrix_k, embed_k, val_bpb, total_bytes in results:
        log(f"{matrix_k:>10.2f} {embed_k:>10.1f} {val_bpb:>12.8f} {total_bytes:>12,} {CAP - total_bytes:>+10,}")

    best = results[0]
    log(f"\nBest: matrix_k={best[0]} embed_k={best[1]} val_bpb={best[2]:.8f} "
        f"({CAP - best[3]:+,} bytes margin)")
    log(f"Baseline (12.85/20.0): compare against logs")

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
