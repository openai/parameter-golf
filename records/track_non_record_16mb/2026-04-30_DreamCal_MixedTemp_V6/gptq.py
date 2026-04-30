"""
Full GPTQ quantization with Cholesky error compensation.
Based on arxiv:2210.17323. Key difference from GPTQ-lite:
GPTQ uses the Hessian (H = X^T X / n) to optimally distribute quantization
error to unquantized columns, minimizing output reconstruction error.

For the competition: calibration uses training data (not val), damp=0.005.

Usage:
  python gptq.py --model best_model_v6.pt --bits 4 --calib-seqs 128
  python gptq.py --model best_model_v6.pt --bits 6 --calib-seqs 128
  python gptq.py --model best_model_v6.pt --mixed --calib-seqs 128  # int4 MLP + int6 attn
"""
import os, sys, math, time, argparse, io, lzma
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

# ── GPTQ Core Algorithm ──

def compute_hessian(X: Tensor) -> Tensor:
    """Compute Hessian H = X^T X / n.
    X: (n_samples, d_in) — layer input activations.
    Returns: (d_in, d_in) symmetric PSD matrix."""
    n = X.shape[0]
    H = (X.float().T @ X.float()) / n
    return H


def gptq_quantize_weight(W: Tensor, H: Tensor, bits: int = 6,
                          block_size: int = 128, damp: float = 0.005,
                          clip_search: bool = True) -> tuple[Tensor, Tensor]:
    """GPTQ quantization of a single weight matrix with Cholesky error compensation.

    Args:
        W: (out_features, in_features) weight matrix
        H: (in_features, in_features) Hessian
        bits: quantization bits (4 or 6)
        block_size: column block size for blocked GPTQ
        damp: damping factor for Hessian stability
        clip_search: if True, search for optimal clip percentile per row

    Returns:
        Q: (out_features, in_features) quantized weights as int8
        scale: (out_features,) per-row fp16 scales
    """
    max_val = (1 << (bits - 1)) - 1  # 7 for int4, 31 for int6

    W = W.float().clone()
    n_rows, n_cols = W.shape

    # Add damping for numerical stability
    diag_mean = torch.diag(H).mean()
    H = H + damp * diag_mean * torch.eye(n_cols, device=H.device, dtype=H.dtype)

    # Compute per-row scales (with optional clip search)
    if clip_search:
        clip_percentiles = [0.995, 0.999, 0.9995, 0.9999, 1.0] if bits <= 4 else [0.999, 0.9995, 0.9999, 0.99999, 1.0]
        best_scale = None
        best_mse = torch.full((n_rows,), float('inf'))

        for pct in clip_percentiles:
            if pct < 1.0:
                clip_val = torch.quantile(W.abs(), pct, dim=1).clamp(min=1e-12)
            else:
                clip_val = W.abs().amax(dim=1).clamp(min=1e-12)

            s = (clip_val / max_val).to(torch.float16)
            q_try = torch.round(W / s.float()[:, None]).clamp(-max_val, max_val)
            recon = q_try * s.float()[:, None]
            mse = (W - recon).pow(2).mean(dim=1)

            improved = mse < best_mse
            if improved.any():
                if best_scale is None:
                    best_scale = s.clone()
                else:
                    best_scale[improved] = s[improved]
                best_mse[improved] = mse[improved]
        scale = best_scale
    else:
        row_max = W.abs().amax(dim=1).clamp(min=1e-12)
        scale = (row_max / max_val).to(torch.float16)

    # Cholesky decomposition of H for stable error compensation
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except RuntimeError:
        # Fallback: add more damping
        H_safe = H + 0.01 * diag_mean * torch.eye(n_cols, device=H.device, dtype=H.dtype)
        L = torch.linalg.cholesky(H_safe)
        H_inv = torch.cholesky_inverse(L)

    # Blocked GPTQ: process columns in blocks
    # Reference: Frantar et al. 2022, Algorithm 1
    # Key: error compensation SUBTRACTS, and Err stores SCALED error (err/d)
    Q = torch.zeros_like(W, dtype=torch.int8)
    Err = torch.zeros(n_rows, block_size, device=W.device, dtype=torch.float32)

    for col_start in range(0, n_cols, block_size):
        col_end = min(col_start + block_size, n_cols)
        bs = col_end - col_start

        # Get block of H_inv for this column block
        H_inv_block = H_inv[col_start:col_end, col_start:col_end]

        W_block = W[:, col_start:col_end].clone()
        Err[:, :bs] = 0

        for j in range(bs):
            col = col_start + j
            w_col = W_block[:, j]
            s = scale.float()

            # Quantize
            q_col = torch.round(w_col / s).clamp(-max_val, max_val)
            Q[:, col] = q_col.to(torch.int8)

            # Scaled error: δ_j = (w_j - q_j) / H_inv[j,j]
            h_diag = H_inv_block[j, j].clamp(min=1e-8)
            raw_err = (w_col - q_col * s)
            scaled_err = raw_err / h_diag

            # Compensate remaining columns in this block (SUBTRACT)
            if j + 1 < bs:
                W_block[:, j+1:bs] -= scaled_err[:, None] * H_inv_block[j, j+1:bs][None, :]

            # Store SCALED error for inter-block update
            Err[:, j] = scaled_err

        # After processing the block, compensate all remaining columns (SUBTRACT)
        if col_end < n_cols:
            H_inv_cross = H_inv[col_start:col_end, col_end:]
            W[:, col_end:] -= Err[:, :bs] @ H_inv_cross

    return Q, scale


# ── Calibration Data Collection ──

def collect_calibration_data(model, data_tokens, n_seqs=128, seq_len=1024, device='cpu'):
    """Run model forward on calibration data and collect per-layer input activations.

    Returns: dict mapping layer_name -> (n_seqs*seq_len, d_in) activations
    """
    activations = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            x = input[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.size(-1))  # (B*T, d)
            if name not in activations:
                activations[name] = []
            activations[name].append(x.cpu())
        return hook_fn

    # Register hooks on all linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.numel() > 65536:
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    n_tokens = data_tokens.numel()

    with torch.no_grad():
        for i in range(min(n_seqs, n_tokens // seq_len)):
            start = i * seq_len
            x = data_tokens[start:start + seq_len].long().unsqueeze(0).to(device)
            model(x)

            if (i + 1) % 32 == 0:
                print(f'  Calibration: {i+1}/{n_seqs} seqs', flush=True)

    for h in hooks:
        h.remove()

    # Concatenate activations
    for name in activations:
        activations[name] = torch.cat(activations[name], dim=0)
        print(f'  {name}: {activations[name].shape}', flush=True)

    return activations


def gptq_quantize_model(model_state_dict, activations, bits=6, mixed=False,
                         block_size=128, damp=0.005):
    """Quantize all large weight matrices using GPTQ.

    If mixed=True: MLP weights (fc, proj) get int4, attention gets int6.
    """
    result = {
        '__format__': f'gptq_{"mixed" if mixed else f"int{bits}"}_v1',
        'quantized': {},
        'scales': {},
        'bits_per_layer': {},
        'passthrough': {},
    }

    for name, tensor in model_state_dict.items():
        t = tensor.detach().cpu()

        if t.numel() <= 65536 or not t.is_floating_point():
            if t.is_floating_point():
                result['passthrough'][name] = t.to(torch.float16)
            else:
                result['passthrough'][name] = t
            continue

        if t.ndim != 2:
            result['passthrough'][name] = t.to(torch.float16)
            continue

        # Determine bits for this layer
        is_mlp = any(k in name for k in ('fc.weight', 'proj.weight'))
        layer_bits = 4 if (mixed and is_mlp) else bits

        # Find matching activation
        # Weight name like "blocks.0.fc.weight" -> activation key "blocks.0.fc"
        act_key = name.rsplit('.weight', 1)[0] if name.endswith('.weight') else name

        if act_key in activations:
            H = compute_hessian(activations[act_key])
            print(f'  GPTQ {name}: {t.shape} -> int{layer_bits} (with Hessian {H.shape})', flush=True)
            q, s = gptq_quantize_weight(t, H, bits=layer_bits, block_size=block_size, damp=damp)
        else:
            # No calibration data — fall back to GPTQ-lite (clip search only)
            print(f'  Clip  {name}: {t.shape} -> int{layer_bits} (no Hessian)', flush=True)
            max_val = 7 if layer_bits == 4 else 31
            row_max = t.float().abs().amax(dim=1).clamp(min=1e-12)
            s = (row_max / max_val).to(torch.float16)
            q = torch.round(t.float() / s.float()[:, None]).clamp(-max_val, max_val).to(torch.int8)

        result['quantized'][name] = q
        result['scales'][name] = s
        result['bits_per_layer'][name] = layer_bits

    return result


def dequantize_gptq_model(quant_dict):
    """Dequantize GPTQ model back to float."""
    state = {}
    for name, t in quant_dict['passthrough'].items():
        state[name] = t.float() if t.is_floating_point() else t
    for name, q in quant_dict['quantized'].items():
        s = quant_dict['scales'][name]
        state[name] = q.float() * s.float()[:, None]
    return state


def compress_artifact(quant_dict):
    """LZMA compress."""
    buf = io.BytesIO()
    torch.save(quant_dict, buf)
    raw = buf.getvalue()
    return lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='best_model_v6.pt')
    parser.add_argument('--bits', type=int, default=6, choices=[4, 6])
    parser.add_argument('--mixed', action='store_true', help='int4 MLP + int6 attention')
    parser.add_argument('--calib-seqs', type=int, default=128)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--damp', type=float, default=0.005)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    print(f'Loading {args.model}...', flush=True)
    state = torch.load(args.model, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'emb.weight' not in state:
        state = state.get('ema_state_dict', state.get('model_state_dict', state))

    n_params = sum(v.numel() for v in state.values())
    print(f'Params: {n_params:,}', flush=True)

    # For now, skip calibration (requires full model class)
    # TODO: integrate with eval_slot_v4.py model class for calibration
    print('\nNote: Full GPTQ requires calibration data (model forward pass).', flush=True)
    print('Using GPTQ-lite (clip search) as fallback. Run with model class for full GPTQ.', flush=True)

    # Quantize without Hessian (GPTQ-lite fallback)
    bits = args.bits
    if args.mixed:
        print(f'\nMixed quantization: int4 MLP + int6 attention', flush=True)
    else:
        print(f'\nUniform int{bits} quantization', flush=True)

    activations = {}  # Empty — will use clip search fallback
    t0 = time.time()
    quant = gptq_quantize_model(state, activations, bits=bits, mixed=args.mixed,
                                 block_size=args.block_size, damp=args.damp)
    print(f'Quantized in {time.time()-t0:.1f}s', flush=True)

    # Compress
    compressed = compress_artifact(quant)
    code_est = 50000
    ngram_est = 800000  # 500K bucket table
    total = len(compressed) + code_est + ngram_est
    headroom = (16e6 - total) / 1e3
    fits = total < 16e6
    print(f'\nLZMA: {len(compressed)/1e6:.3f} MB')
    print(f'Total (model + code + ngram): {total/1e6:.3f} MB')
    print(f'{"OK" if fits else "OVER"} (headroom: {headroom:.0f} KB)')

    artifact_path = args.model.replace('.pt', f'.gptq{"_mixed" if args.mixed else f"_{bits}bit"}.lzma')
    with open(artifact_path, 'wb') as f:
        f.write(compressed)
    print(f'Artifact: {artifact_path}', flush=True)

    if args.eval:
        print('\nRoundtrip evaluation...', flush=True)
        rt_state = dequantize_gptq_model(quant)
        total_mse = 0
        total_n = 0
        for name in state:
            if name in rt_state:
                orig = state[name].float()
                recon = rt_state[name].float()
                mse = (orig - recon).pow(2).mean().item()
                total_mse += mse * orig.numel()
                total_n += orig.numel()
        print(f'Weighted avg MSE: {total_mse/total_n:.6e}', flush=True)
