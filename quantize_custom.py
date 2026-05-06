"""
Custom bit-packing quantization for fitting larger models in 16MB.
Supports int4, int5, int6, int7, int8 with per-row scales.
Bit-packs weights into uint8 arrays for maximum compression.

Usage:
  python quantize_custom.py best_model_8B.pt [--bits 5] [--attn-bits 6] [--mlp-bits 5]
"""
import os, sys, io, zlib, math, time, argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor

def pack_bits(values: np.ndarray, bits: int) -> np.ndarray:
    """Pack integer values (each using `bits` bits) into uint8 array."""
    # values should be unsigned integers in [0, 2^bits - 1]
    total_bits = len(values) * bits
    packed = np.zeros((total_bits + 7) // 8, dtype=np.uint8)
    bit_pos = 0
    for v in values:
        for b in range(bits):
            if v & (1 << b):
                packed[bit_pos // 8] |= (1 << (bit_pos % 8))
            bit_pos += 1
    return packed

def unpack_bits(packed: np.ndarray, bits: int, count: int) -> np.ndarray:
    """Unpack uint8 array back to integer values."""
    values = np.zeros(count, dtype=np.int32)
    bit_pos = 0
    for i in range(count):
        v = 0
        for b in range(bits):
            if packed[bit_pos // 8] & (1 << (bit_pos % 8)):
                v |= (1 << b)
            bit_pos += 1
        values[i] = v
    return values

def pack_bits_fast(values: np.ndarray, bits: int) -> np.ndarray:
    """Vectorized bit packing - much faster than loop version."""
    n = len(values)
    total_bits = n * bits
    packed_len = (total_bits + 7) // 8
    packed = np.zeros(packed_len, dtype=np.uint8)

    for b in range(bits):
        # Extract bit b from all values
        bit_mask = ((values >> b) & 1).astype(np.uint8)
        # Each value's bit b goes to position (i*bits + b) in the bitstream
        bit_positions = np.arange(n, dtype=np.int64) * bits + b
        byte_idx = bit_positions >> 3
        bit_idx = (bit_positions & 7).astype(np.uint8)
        np.add.at(packed, byte_idx, bit_mask << bit_idx)

    return packed

def unpack_bits_fast(packed: np.ndarray, bits: int, count: int) -> np.ndarray:
    """Vectorized bit unpacking."""
    values = np.zeros(count, dtype=np.int32)

    for b in range(bits):
        bit_positions = np.arange(count, dtype=np.int64) * bits + b
        byte_idx = bit_positions >> 3
        bit_idx = (bit_positions & 7).astype(np.uint8)
        bit_vals = (packed[byte_idx] >> bit_idx) & 1
        values |= (bit_vals.astype(np.int32) << b)

    return values

def quantize_tensor(t: Tensor, bits: int) -> tuple[bytes, Tensor, tuple]:
    """Quantize a float tensor to N-bit integers with per-row scales.
    Returns packed bytes, scales tensor, and metadata."""
    t32 = t.float()
    shape = t32.shape

    max_val = (1 << (bits - 1)) - 1  # e.g., 15 for int5, 127 for int8

    if t32.ndim == 2:
        # Per-row quantization
        row_max = t32.abs().amax(dim=1).clamp(min=1e-8)
        scale = row_max / max_val
        q = torch.round(t32 / scale[:, None]).clamp(-max_val, max_val).to(torch.int32)
        # Shift to unsigned: [-max_val, max_val] -> [0, 2*max_val]
        q_unsigned = (q + max_val).numpy().astype(np.int32).flatten()
        packed = pack_bits_fast(q_unsigned, bits)
        return packed.tobytes(), scale.to(torch.float16), shape
    else:
        # Per-tensor quantization
        t_max = t32.abs().max().item()
        scale = torch.tensor(t_max / max_val if t_max > 0 else 1.0, dtype=torch.float16)
        q = torch.round(t32 / scale).clamp(-max_val, max_val).to(torch.int32)
        q_unsigned = (q + max_val).numpy().astype(np.int32).flatten()
        packed = pack_bits_fast(q_unsigned, bits)
        return packed.tobytes(), scale, shape

def dequantize_tensor(packed_bytes: bytes, scale: Tensor, shape: tuple, bits: int) -> Tensor:
    """Dequantize packed N-bit integers back to float tensor."""
    max_val = (1 << (bits - 1)) - 1
    count = 1
    for s in shape:
        count *= s

    packed = np.frombuffer(packed_bytes, dtype=np.uint8)
    q_unsigned = unpack_bits_fast(packed, bits, count)
    q = torch.from_numpy(q_unsigned.astype(np.int32)) - max_val
    q = q.reshape(shape).float()

    if len(shape) == 2:
        return q * scale.float()[:, None]
    else:
        return q * scale.float()

def quantize_model(state_dict: dict, attn_bits: int = 6, mlp_bits: int = 5,
                   other_bits: int = 8, embed_bits: int = 8) -> dict:
    """Quantize a model state dict with mixed precision per component type."""
    result = {
        '__format__': 'custom_mixed_bitpack_v1',
        'tensors': {},
        'metadata': {}
    }

    total_bytes = 0

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu()

        # Determine bit width based on tensor name/role
        if t.numel() <= 65536:  # Small tensors: keep as fp16
            result['tensors'][name] = {'type': 'fp16', 'data': t.to(torch.float16)}
            total_bytes += t.numel() * 2
            continue

        if not t.is_floating_point():
            result['tensors'][name] = {'type': 'passthrough', 'data': t}
            total_bytes += t.numel() * t.element_size()
            continue

        # Choose bits based on component
        if 'emb.weight' in name:
            bits = embed_bits
        elif any(k in name for k in ['fc.weight', 'proj.weight']):
            bits = mlp_bits  # MLP weights
        elif any(k in name for k in ['q.weight', 'k.weight', 'v.weight', 'o.weight']):
            bits = attn_bits  # Attention weights
        else:
            bits = other_bits

        packed, scale, shape = quantize_tensor(t, bits)
        result['tensors'][name] = {
            'type': 'packed',
            'bits': bits,
            'packed': packed,
            'scale': scale,
            'shape': shape
        }
        total_bytes += len(packed) + scale.numel() * 2  # packed + fp16 scales

    result['metadata']['total_bytes'] = total_bytes
    return result

def dequantize_model(quant_dict: dict) -> dict:
    """Dequantize a packed model back to float state dict."""
    state = {}
    for name, info in quant_dict['tensors'].items():
        if info['type'] == 'fp16':
            state[name] = info['data'].float()
        elif info['type'] == 'passthrough':
            state[name] = info['data']
        elif info['type'] == 'packed':
            state[name] = dequantize_tensor(
                info['packed'], info['scale'], info['shape'], info['bits']
            )
    return state

def measure_artifact_size(quant_dict: dict) -> tuple[int, int]:
    """Serialize and compress the quantized model, return (raw_bytes, compressed_bytes)."""
    buf = io.BytesIO()
    torch.save(quant_dict, buf)
    raw = buf.getvalue()
    compressed = zlib.compress(raw, level=9)
    return len(raw), len(compressed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', default='best_model_8B.pt', nargs='?')
    parser.add_argument('--attn-bits', type=int, default=6)
    parser.add_argument('--mlp-bits', type=int, default=5)
    parser.add_argument('--embed-bits', type=int, default=8)
    parser.add_argument('--other-bits', type=int, default=8)
    parser.add_argument('--eval', action='store_true', help='Evaluate roundtrip quality')
    args = parser.parse_args()

    print(f'Loading {args.model_path}...', flush=True)
    state = torch.load(args.model_path, map_location='cpu')
    n_params = sum(v.numel() for v in state.values())

    n_blocks = sum(1 for k in state if k.startswith('blocks.') and '.n1.' in k)
    fc_key = [k for k in state if 'fc.weight' in k][0]
    mlp_mult = state[fc_key].shape[0] // 512
    print(f'Model: {n_blocks}L {mlp_mult}xMLP, {n_params:,} params')
    print(f'Quantization: attn={args.attn_bits}b, mlp={args.mlp_bits}b, embed={args.embed_bits}b')

    t0 = time.time()
    quant = quantize_model(state, args.attn_bits, args.mlp_bits, args.other_bits, args.embed_bits)
    print(f'Quantized in {time.time()-t0:.1f}s')
    print(f'Estimated payload: {quant["metadata"]["total_bytes"]/1e6:.2f} MB')

    raw_bytes, compressed_bytes = measure_artifact_size(quant)
    code_est = 60000
    total = compressed_bytes + code_est
    print(f'Serialized raw: {raw_bytes/1e6:.2f} MB')
    print(f'Compressed (zlib-9): {compressed_bytes/1e6:.2f} MB')
    print(f'Total w/ code: {total/1e6:.2f} MB')
    print(f'Under 16MB: {total < 16e6} (headroom: {(16e6-total)/1e6:.2f} MB)')

    # Compute average bits per param
    avg_bits = quant["metadata"]["total_bytes"] * 8 / n_params
    print(f'Average bits/param: {avg_bits:.2f}')

    if args.eval:
        print('\nRoundtrip evaluation...')
        deq = dequantize_model(quant)

        # Compute per-tensor MSE
        total_mse = 0
        total_params = 0
        for name in state:
            if name in deq:
                orig = state[name].float()
                recon = deq[name].float()
                mse = (orig - recon).pow(2).mean().item()
                if orig.numel() > 1000:
                    print(f'  {name}: MSE={mse:.6e}, shape={tuple(orig.shape)}')
                total_mse += mse * orig.numel()
                total_params += orig.numel()

        print(f'\nWeighted avg MSE: {total_mse/total_params:.6e}')

        # Save roundtrip model
        rt_path = args.model_path.replace('.pt', f'_rt_{args.mlp_bits}b.pt')
        torch.save(deq, rt_path)
        print(f'Roundtrip model saved: {rt_path}')
        print(f'Run: python sliding_window_eval.py {rt_path}')
