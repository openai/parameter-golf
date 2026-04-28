"""
Hessian-Aware SDClip sweep.

Standard SDClip:  scale[row] = k * std(row) / clip_range
Hessian-aware:    scale[row] = k * std(row) * adj[row] / clip_range
  where adj modulates per-row scale based on column importance from diag(H).

Sweeps HESSIAN_CLIP_LAMBDA in [0.00, 0.05, 0.10, 0.15, 0.20]
with matrix_clip_sigmas=12.85, embed_clip_sigmas=20.0 (optimal quant).

Code size assumes lzma+base85 packing (~16,600 bytes).
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

LAMBDAS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
MATRIX_K = 12.85
EMBED_K = 20.0


def gptq_quantize_weight_hessian_aware(w, H_orig, clip_sigmas=12.85, clip_range=63,
                                        block_size=128, hessian_lambda=0.0):
    """GPTQ quantization with Hessian-aware per-row scale adjustment."""
    W_orig = w.float().clone()
    rows, cols = W_orig.shape
    H = H_orig.float().clone()

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * H.diag().mean()
    H.diagonal().add_(damp)

    # Hessian-aware row scale adjustment (before permutation)
    row_std = W_orig.std(dim=1)

    if hessian_lambda > 0:
        diagH = torch.diag(H).clamp_min(1e-8)
        col_importance = diagH / diagH.mean()
        row_importance = (W_orig.abs() * col_importance[None, :]).mean(dim=1)
        row_importance = row_importance / row_importance.mean()
        adj = 1.0 + hessian_lambda * (row_importance - 1.0)
        s = (clip_sigmas * row_std * adj / clip_range).clamp_min(1e-10).to(torch.float16)
    else:
        s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)

    sf = s.float()

    # Standard GPTQ quantization with the adjusted scales
    perm = torch.argsort(torch.diag(H), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()

    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros(rows, i2 - i1)

        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)

        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]

    return Q[:, invperm], s


def quantize_with_hessian_clip(sd_cpu, hessians, h, matrix_k, embed_k, hessian_lambda):
    """Quantize all weights with Hessian-aware clip for matrix weights."""
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
        # Only apply hessian-aware clip to non-embedding weights
        lam = 0.0 if is_embed else hessian_lambda
        q, s = gptq_quantize_weight_hessian_aware(
            t, hessians[name], clip_sigmas=cs, clip_range=clip_range,
            hessian_lambda=lam
        )
        result[name + '.q'] = q
        result[name + '.scale'] = s
        meta[name] = f"gptq (int{bits})"
    return result, meta


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
    log(f"matrix_clip_sigmas={MATRIX_K}, embed_clip_sigmas={EMBED_K}")

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

    # Phase 1: baseline (lambda=0, should match PR #1493)
    log("\n===== PHASE 1: SIZE + BASELINE =====")
    log(f"{'lambda':>8} {'model_bytes':>12} {'total_bytes':>12} {'margin':>10}")
    log("-" * 50)

    val_data = ValidationData(h, device)
    results = []

    for lam in LAMBDAS:
        label = f"lam{lam:.2f}"
        quant_result, quant_meta = quantize_with_hessian_clip(
            sd_cpu, hessians, h, MATRIX_K, EMBED_K, lam
        )
        quant_buf = io.BytesIO()
        torch.save({'w': quant_result, 'm': quant_meta}, quant_buf)
        quant_blob = _compress(quant_buf.getvalue(), h.compressor)
        model_bytes = len(quant_blob)
        total_bytes = model_bytes + PACKED_CODE_SIZE
        margin = CAP - total_bytes
        legal = "OK" if total_bytes <= CAP else "OVER"
        log(f"{lam:>8.2f} {model_bytes:>12,} {total_bytes:>12,} {margin:>+10,}  {legal}")

        if total_bytes > CAP:
            log(f"  {label}: OVER cap, skipping eval")
            continue

        # Eval
        log(f"\n--- Evaluating {label} ---")
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

        log(f"RESULT: lambda={lam:.2f} val_bpb={val_bpb:.8f} "
            f"total_bytes={total_bytes} margin={margin:+d} eval_time={elapsed:.1f}s")
        results.append((lam, val_bpb, total_bytes))

        del eval_model
        torch.cuda.empty_cache()

    # Summary
    log("\n===== SUMMARY =====")
    log(f"{'lambda':>8} {'val_bpb':>12} {'total_bytes':>12} {'margin':>10}")
    log("-" * 50)
    results.sort(key=lambda r: r[1])
    for lam, val_bpb, total_bytes in results:
        log(f"{lam:>8.2f} {val_bpb:>12.8f} {total_bytes:>12,} {CAP - total_bytes:>+10,}")

    if results:
        best = results[0]
        log(f"\nBest: lambda={best[0]:.2f} val_bpb={best[1]:.8f} ({CAP - best[2]:+,} margin)")

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
