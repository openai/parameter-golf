"""
Adaptive Mixed-Precision Quantizer for Parameter Golf

Post-training symmetric quantization with:
1. Activation entropy sensitivity measurement — ranks layers by importance
2. Symmetric per-group quantization — zero-point fixed at 0 (Muon enforces symmetric weights)
3. Knapsack bit allocation — greedy marginal-gain over 66 unbanked slices within 16MB
4. Variable group sizes — smaller groups for sensitive Q/K, larger for compressible MLP

NO Hadamard rotation — Muon's orthogonal structure already provides incoherence.
NO asymmetric zero-point — Muon + weight decay keeps weights centered at zero.
NO QAT threshold change — keep battle-tested 0.15.

Validated by: Claude Opus, Grok, Gemini Deep Research, TTS Swarm (10 missions)
Key papers: Q-BERT, LieQ, NVFP4, CLASE-Quant, "Beyond Outliers" (arXiv:2509.23500)

Plugs into SOTA unbank→quantize→compress→decompress→dequantize→rebank flow.
Replaces mixed_quantize_int6() and dequantize_mixed_int6() in train_gpt.py.
"""

import io
import math
import time
import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional
import itertools


# ── Component Classification ────────────────────────────────────────────

def classify_slice(name: str, num_layers: int) -> Tuple[str, int]:
    """Classify an unbanked weight slice by component type and layer index.

    After _unbank_state_dict(), names look like:
      blocks.{i}.attn.c_q.weight, blocks.{i}.attn.c_k.weight,
      blocks.{i}.attn.c_v.weight, blocks.{i}.attn.proj.weight,
      blocks.{i}.mlp.fc.weight, blocks.{i}.mlp.proj.weight,
      tok_emb.weight, bigram.embed.weight, etc.
    """
    layer_idx = -1
    parts = name.split('.')
    for p in parts:
        if p.isdigit():
            layer_idx = int(p)
            break

    if 'tok_emb' in name:
        return 'embed', -1
    if 'bigram' in name and 'embed' in name:
        return 'bigram_embed', -1
    if 'c_q.weight' in name:
        return 'q', layer_idx
    if 'c_k.weight' in name:
        return 'k', layer_idx
    if 'c_v.weight' in name:
        return 'v', layer_idx
    if 'proj.weight' in name and 'mlp' not in name:
        return 'out', layer_idx
    if 'mlp.fc.weight' in name:
        return 'mlp_up', layer_idx
    if 'mlp.proj.weight' in name:
        return 'mlp_down', layer_idx
    return 'other', layer_idx


def get_group_size(component: str, layer_idx: int, num_layers: int) -> int:
    """Variable group size based on component sensitivity.

    Q/K: small groups (32) — softmax cascade makes these most sensitive.
    V/Out: medium groups (64).
    MLP down: medium (64) — error accumulator.
    MLP up sensitive layers: medium (64).
    MLP up middle layers: large (128) — most compressible, minimize overhead.
    """
    sensitive = {0, 1, num_layers - 1}
    if component in ('q', 'k'):
        return 32
    if component in ('v', 'out'):
        return 64
    if component == 'mlp_down':
        return 64
    if component == 'mlp_up':
        return 64 if layer_idx in sensitive else 128
    return 64


def sensitivity_boost(component: str, layer_idx: int, num_layers: int) -> float:
    """Boost factor for known-sensitive components.

    Based on converged findings from Opus, Grok, Gemini, Q-BERT, CLASE-Quant:
    - Layers 0, 1, last: boundary layers, most sensitive (representation mapping)
    - Middle layers: abstract/redundant, skip-connected, compressible
    - Q/K: softmax cascade amplifies errors exponentially
    - MLP down: accumulates quantization error from up-projection
    - V/Out: less sensitive
    - MLP up: most compressible (especially middle layers)
    """
    boost = 1.0

    sensitive = {0, 1, num_layers - 1}
    if layer_idx in sensitive:
        boost *= 2.0
    elif 2 <= layer_idx <= num_layers - 2:
        boost *= 0.5

    if component in ('q', 'k'):
        boost *= 1.5
    elif component == 'mlp_down':
        boost *= 1.2
    elif component in ('v', 'out'):
        boost *= 0.8
    elif component == 'mlp_up':
        boost *= 0.6

    return boost


# ── Activation Entropy ──────────────────────────────────────────────────

def measure_activation_entropy(
    model: torch.nn.Module,
    calibration_tokens: Tensor,
    device: torch.device,
    seq_len: int = 2048,
) -> Dict[str, float]:
    """Single forward pass → activation entropy per module. <5 seconds on H100."""
    entropies = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            if out is None or not isinstance(out, Tensor):
                return
            flat = out.detach().float().reshape(-1)
            if flat.numel() > 100_000:
                flat = flat[torch.randint(0, flat.numel(), (100_000,))]
            hist = torch.histc(flat, bins=256)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            entropies[name] = -(hist * torch.log2(hist)).sum().item()
        return hook_fn

    for name, module in model.named_modules():
        if any(k in name for k in ['blocks.', 'mlp', 'attn', 'tok_emb']):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.inference_mode():
        n_seqs = min(32, calibration_tokens.numel() // seq_len)
        if n_seqs > 0:
            x = calibration_tokens[:n_seqs * seq_len].reshape(n_seqs, seq_len)
            x = x.to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model.forward_logits(x[:min(8, n_seqs)])

    for h in hooks:
        h.remove()
    return entropies


# ── Symmetric Per-Group Quantization ─────────────────────────────────────

def quantize_symmetric_group(
    weight: Tensor,
    bits: int,
    group_size: int = 64,
) -> Tuple[Tensor, Tensor]:
    """Symmetric per-group quantization. Zero-point fixed at 0.

    Muon + weight decay keeps weights centered at zero (confirmed by Opus/Grok/Gemini).
    No zero-point → no cross-term penalty → fast GEMM on Tensor Cores.

    Returns (quantized_int8, scales_fp16).
    """
    w = weight.float()
    clip_range = (1 << (bits - 1)) - 1  # e.g., int6 → 31, int4 → 7, int3 → 3

    if w.ndim != 2:
        amax = w.abs().max().clamp_min(1e-8)
        scale = (amax / clip_range).to(torch.float16)
        q = torch.clamp(torch.round(w / scale.float()), -clip_range, clip_range).to(torch.int8)
        return q, scale

    out_dim, in_dim = w.shape
    padded_in = ((in_dim + group_size - 1) // group_size) * group_size
    if padded_in != in_dim:
        w_padded = torch.zeros(out_dim, padded_in, dtype=w.dtype, device=w.device)
        w_padded[:, :in_dim] = w
    else:
        w_padded = w

    n_groups = padded_in // group_size
    w_grouped = w_padded.reshape(out_dim, n_groups, group_size)

    # Per-group symmetric scale: max absolute value
    g_amax = w_grouped.abs().amax(dim=-1).clamp_min(1e-8)
    scale = (g_amax / clip_range).to(torch.float16)

    q = torch.clamp(
        torch.round(w_grouped / scale.float().unsqueeze(-1)),
        -clip_range, clip_range
    ).to(torch.int8)

    q = q.reshape(out_dim, padded_in)[:, :in_dim].contiguous()
    return q, scale


def dequantize_symmetric_group(
    q: Tensor, scale: Tensor,
    group_size: int = 64,
    orig_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Dequantize symmetric per-group: W ≈ q * scale. No zero-point."""
    if q.ndim != 2:
        return (q.float() * scale.float()).to(orig_dtype)

    out_dim, in_dim = q.shape
    padded_in = ((in_dim + group_size - 1) // group_size) * group_size
    n_groups = padded_in // group_size

    if padded_in != in_dim:
        q_padded = torch.zeros(out_dim, padded_in, dtype=q.dtype, device=q.device)
        q_padded[:, :in_dim] = q
    else:
        q_padded = q

    w = q_padded.reshape(out_dim, n_groups, group_size).float() * scale.float().unsqueeze(-1)
    return w.reshape(out_dim, padded_in)[:, :in_dim].to(orig_dtype).contiguous()


# ── Reconstruction Error ─────────────────────────────────────────────────

def reconstruction_mse(weight: Tensor, bits: int, group_size: int) -> float:
    """MSE between original and round-tripped weight at given bit-width."""
    q, s = quantize_symmetric_group(weight.float(), bits, group_size)
    recon = dequantize_symmetric_group(q, s, group_size, torch.float32)
    if recon.shape != weight.shape:
        slices = tuple(slice(0, d) for d in weight.shape)
        recon = recon[slices]
    return (weight.float() - recon).pow(2).mean().item()


# ── Size Calculation ─────────────────────────────────────────────────────

def size_bytes(num_params: int, bits: int, group_size: int) -> int:
    """Storage: quantized weights (int8 containers) + per-group fp16 scales."""
    weight_bytes = (num_params * bits + 7) // 8
    n_groups = max(1, num_params // group_size)
    scale_bytes = n_groups * 2  # fp16
    return weight_bytes + scale_bytes


# ── Knapsack Bit Allocator ───────────────────────────────────────────────

def solve_allocation(
    slices: List[Dict],
    budget_bytes: int,
    bit_options: List[int] = [3, 4, 5, 6, 8],
    fp16_names: set = None,
) -> Dict[str, int]:
    """Greedy marginal-gain knapsack over 66 unbanked weight slices.

    Start at minimum bits. Repeatedly upgrade the slice with highest
    error-reduction per extra byte until budget is exhausted.
    """
    fp16_names = fp16_names or set()
    allocation = {}

    # FP16 passthrough
    fp16_cost = 0
    allocatable = []
    for info in slices:
        if info['name'] in fp16_names:
            fp16_cost += info['num_params'] * 2
            allocation[info['name']] = 16
        else:
            allocatable.append(info)

    remaining = budget_bytes - fp16_cost
    if remaining <= 0:
        for info in allocatable:
            allocation[info['name']] = min(bit_options)
        return allocation

    # Start everything at minimum bits
    current = {info['name']: min(bit_options) for info in allocatable}
    current_cost = sum(
        size_bytes(info['num_params'], min(bit_options), info['group_size'])
        for info in allocatable
    )

    # Greedy upgrade loop
    while True:
        best_gain = 0.0
        best_name = None
        best_bits = None
        best_extra = 0

        for info in allocatable:
            cur_b = current[info['name']]
            idx = bit_options.index(cur_b)
            if idx >= len(bit_options) - 1:
                continue

            new_b = bit_options[idx + 1]
            extra = (
                size_bytes(info['num_params'], new_b, info['group_size']) -
                size_bytes(info['num_params'], cur_b, info['group_size'])
            )
            if current_cost + extra > remaining:
                continue

            reduction = info['werrors'][cur_b] - info['werrors'][new_b]
            gain = reduction / max(extra, 1)

            if gain > best_gain:
                best_gain = gain
                best_name = info['name']
                best_bits = new_b
                best_extra = extra

        if best_name is None or best_gain <= 0:
            break

        current[best_name] = best_bits
        current_cost += best_extra

    allocation.update(current)
    return allocation


# ── Integration: Drop-in replacement for mixed_quantize_int6 ─────────────

# These are imported from the main training script
CONTROL_TENSOR_NAME_PATTERNS = ()  # Will be set from train_gpt.py


def adaptive_quantize(
    state_dict: Dict[str, Tensor],
    model: torch.nn.Module,
    calibration_tokens: Tensor,
    device: torch.device,
    num_layers: int = 11,
    budget_bytes: int = 16_000_000,
    code_bytes: int = 20_000,
    bit_options: List[int] = [3, 4, 5, 6, 8],
    log_fn=print,
) -> Tuple[Dict[str, Tensor], Dict]:
    """
    Drop-in replacement for mixed_quantize_int6().

    Call after _unbank_state_dict() with the unbanked state dict.
    Returns (result, meta) in the same format expected by the serialization code.
    """
    t0 = time.perf_counter()
    effective_budget = budget_bytes - code_bytes
    fp16_names = {"tok_emb.weight"}

    log_fn(f"[adaptive] Budget: {effective_budget:,} bytes")
    log_fn(f"[adaptive] Params: {sum(t.numel() for t in state_dict.values()):,}")

    # Step 1: Activation entropy
    log_fn("[adaptive] Step 1: Activation entropy (single forward pass)...")
    t1 = time.perf_counter()
    entropies = measure_activation_entropy(model, calibration_tokens, device)
    max_ent = max(entropies.values()) if entropies else 1.0
    log_fn(f"  {len(entropies)} modules measured in {time.perf_counter()-t1:.1f}s")

    # Step 2: Build slice info with sensitivity + reconstruction errors
    log_fn("[adaptive] Step 2: Per-slice reconstruction errors...")
    t2 = time.perf_counter()
    slice_info = []

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu()
        if not t.is_floating_point():
            continue
        if t.numel() <= 65536:
            continue

        component, layer_idx = classify_slice(name, num_layers)
        gs = get_group_size(component, layer_idx, num_layers)

        if name in fp16_names:
            slice_info.append({
                'name': name, 'num_params': t.numel(), 'component': component,
                'layer_idx': layer_idx, 'group_size': gs,
                'sensitivity': 1.0, 'werrors': {},
            })
            continue

        # Entropy-based sensitivity
        base_sens = 1.0
        if layer_idx >= 0:
            for ent_name, ent_val in entropies.items():
                if f'.{layer_idx}.' in ent_name or f'blocks.{layer_idx}' in ent_name:
                    base_sens = ent_val / max_ent
                    break

        sens = base_sens * sensitivity_boost(component, layer_idx, num_layers)

        # Reconstruction error at each bit-width
        errors = {}
        werrors = {}
        for bits in bit_options:
            err = reconstruction_mse(t, bits, gs)
            errors[bits] = err
            werrors[bits] = sens * err

        slice_info.append({
            'name': name, 'num_params': t.numel(), 'component': component,
            'layer_idx': layer_idx, 'group_size': gs,
            'sensitivity': sens, 'errors': errors, 'werrors': werrors,
        })

    log_fn(f"  {len(slice_info)} slices in {time.perf_counter()-t2:.1f}s")

    # Step 3: Knapsack
    log_fn("[adaptive] Step 3: Knapsack allocation...")
    t3 = time.perf_counter()
    allocation = solve_allocation(slice_info, effective_budget, bit_options, fp16_names)
    log_fn(f"  Solved in {time.perf_counter()-t3:.3f}s")

    # Print allocation table
    total_size = 0
    for info in slice_info:
        bits = allocation.get(info['name'], 6)
        sz = info['num_params'] * 2 if bits == 16 else size_bytes(info['num_params'], bits, info['group_size'])
        total_size += sz
        li = f"L{info['layer_idx']:2d}" if info['layer_idx'] >= 0 else "   "
        log_fn(f"  {info['name']:42s} {li} {info['component']:10s} → {bits:2d}b g{info['group_size']:3d}  {sz:>9,}B  s={info['sensitivity']:.2f}")

    log_fn(f"\n  Total: {total_size:,} / {effective_budget:,} ({100*total_size/effective_budget:.1f}%)")

    # Step 4: Apply quantization
    log_fn("[adaptive] Step 4: Quantizing...")
    t4 = time.perf_counter()

    result = {}
    meta = {}

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()

        # Non-float or tiny → passthrough
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue

        # Control tensors → float32 passthrough
        if CONTROL_TENSOR_NAME_PATTERNS and any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue

        bits = allocation.get(name, 6)

        # FP16 passthrough
        if bits == 16 or name in fp16_names:
            result[name] = t.to(torch.float16).contiguous()
            meta[name] = "passthrough_fp16"
            continue

        # Symmetric per-group quantization
        info = next((s for s in slice_info if s['name'] == name), None)
        gs = info['group_size'] if info else 64

        q, s = quantize_symmetric_group(t, bits, gs)
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = {
            "type": f"symmetric_int{bits}",
            "bits": bits,
            "group_size": gs,
            "orig_dtype": str(t.dtype).replace("torch.", ""),
        }

    log_fn(f"  Done in {time.perf_counter()-t4:.1f}s")
    log_fn(f"[adaptive] Total: {time.perf_counter()-t0:.1f}s")

    return result, meta


def dequantize_adaptive(
    result: Dict[str, Tensor],
    meta: Dict,
    template_sd: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Dequantize adaptive state dict. Drop-in replacement for dequantize_mixed_int6."""
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue

        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype != orig.dtype:
                t = t.to(orig.dtype)
            out[name] = t
            continue

        if isinstance(info, dict) and "bits" in info:
            q = result[name + ".q"]
            s = result[name + ".scale"]
            gs = info["group_size"]
            orig_dtype = getattr(torch, info.get("orig_dtype", "bfloat16"))
            out[name] = dequantize_symmetric_group(q, s, gs, orig_dtype)
            continue

        if name in result:
            out[name] = result[name]

    return out
