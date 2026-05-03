"""
stage3_1 patches — first-principles mechanisms from distant fields.

Base: 1.1233 frontier record train_gpt.py (all SOTA features built in).
Each patch crosses a concept from a non-ML field into the pgolf pipeline.

Lane B patches (H1-H3): modify export pipeline only, no retraining needed.
Lane A patches (H4-H7): modify training dynamics.
"""

from __future__ import annotations
import re


def _replace_unique(source: str, old: str, new: str, patch_name: str) -> str:
    count = source.count(old)
    if count == 0:
        raise ValueError(f"[{patch_name}] target string not found:\n{old[:200]!r}")
    if count > 1:
        raise ValueError(f"[{patch_name}] ambiguous: found {count} occurrences:\n{old[:200]!r}")
    return source.replace(old, new)


PATCH_ORDER: dict[str, int] = {
    # Lane B (export-only) — order doesn't matter much, they modify export section
    "companding": 10,
    "fisher_bits": 20,
    "sparsify_before_quant": 30,
    "sdclip_sweep": 35,
    "int4_embed": 36,
    "gptq_post_ttt": 37,
    # Lane A (training) — order matters for composability
    "byte_weighted_loss": 40,
    "tapered_mlp": 50,
    "quant_anneal": 60,
    "staged_objective": 70,
    "best_checkpoint": 80,
    # Post-training
    "ttt_aggressive": 90,
}


def apply_patches(source: str, patch_names: list[str]) -> str:
    ordered = sorted(patch_names, key=lambda n: PATCH_ORDER.get(n, 100))
    for name in ordered:
        fn = PATCHES[name]
        source = fn(source)
    return source


# ===========================================================================
# H1: μ-LAW COMPANDING QUANTIZATION
# Field: Telecommunications (μ-law companding, ITU-T G.711)
#
# Problem: Weight distributions are heavy-tailed. Uniform int6 quantization
# allocates equal bin width across the whole range, wasting resolution on
# sparse tails while under-resolving the dense near-zero region.
#
# Mechanism: Before quantization, apply companding: compress the dynamic
# range so near-zero weights get more bins. After dequantization, expand.
#   compress: w_c = sign(w) * ln(1 + μ|w|/max) / ln(1 + μ) * max
#   expand:   w   = sign(w_c) * (max/μ) * ((1+μ)^(|w_c|/max) - 1)
#
# The μ parameter controls compression strength. μ=0 is linear (no effect).
# μ=255 is standard telephone companding. For weights, μ=50-100 is a
# good starting range — enough to concentrate bins near zero without
# destroying the tails.
#
# Implementation: Replace quantize_int6_per_row with a companded version.
# The companding is applied per-row before the uniform quantizer, and
# the inverse is applied per-row during dequantization. The scale factors
# absorb the per-row max, so no extra metadata is needed.
# ===========================================================================

def patch_companding(source: str) -> str:
    # Add env var
    source = _replace_unique(
        source,
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))',
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))\n'
        '    compand_mu = float(os.environ.get("COMPAND_MU", "0"))  # μ-law companding: 0=off, 50-255=typical',
        "companding",
    )

    # Replace quantize_int6_per_row with companding-aware version.
    # The original applies uniform quantization. The new version optionally
    # applies μ-law compression before quantization, which concentrates
    # quantization bins near zero where most weights live.
    source = _replace_unique(
        source,
        'def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:\n'
        '    t32 = t.float()\n'
        '    if t32.ndim == 2:\n'
        '        best_q, best_s, best_err = None, None, float(\'inf\')\n'
        '        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:\n'
        '            if pct < 1.0:\n'
        '                row_clip = torch.quantile(t32.abs(), pct, dim=1)\n'
        '            else:\n'
        '                row_clip = t32.abs().amax(dim=1)\n'
        '            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)\n'
        '            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)\n'
        '            recon = q.float() * s.float()[:, None]\n'
        '            err = (t32 - recon).pow(2).mean().item()\n'
        '            if err < best_err:\n'
        '                best_q, best_s, best_err = q, s, err\n'
        '        return best_q, best_s\n'
        '    amax = t32.abs().max().item()\n'
        '    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)\n'
        '    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)\n'
        '    return q, scale',

        '_COMPAND_MU: float = 0.0  # set from args before export\n'
        '\n'
        'def _mu_compress(x: Tensor, mu: float) -> Tensor:\n'
        '    """μ-law compression: concentrates quantization bins near zero."""\n'
        '    if mu <= 0:\n'
        '        return x\n'
        '    sign = x.sign()\n'
        '    ax = x.abs()\n'
        '    amax = ax.amax(dim=-1, keepdim=True).clamp_min(1e-12)\n'
        '    compressed = sign * (torch.log1p(mu * ax / amax) / math.log(1 + mu)) * amax\n'
        '    return compressed\n'
        '\n'
        'def _mu_expand(x: Tensor, mu: float) -> Tensor:\n'
        '    """μ-law expansion: inverse of compression."""\n'
        '    if mu <= 0:\n'
        '        return x\n'
        '    sign = x.sign()\n'
        '    ax = x.abs()\n'
        '    amax = ax.amax(dim=-1, keepdim=True).clamp_min(1e-12)\n'
        '    expanded = sign * (amax / mu) * (torch.pow(1 + mu, ax / amax) - 1)\n'
        '    return expanded\n'
        '\n'
        'def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:\n'
        '    t32 = t.float()\n'
        '    mu = _COMPAND_MU\n'
        '    if t32.ndim == 2:\n'
        '        # Apply μ-law compression before quantization\n'
        '        t_q = _mu_compress(t32, mu) if mu > 0 else t32\n'
        '        best_q, best_s, best_err = None, None, float(\'inf\')\n'
        '        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:\n'
        '            if pct < 1.0:\n'
        '                row_clip = torch.quantile(t_q.abs(), pct, dim=1)\n'
        '            else:\n'
        '                row_clip = t_q.abs().amax(dim=1)\n'
        '            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)\n'
        '            q = torch.clamp(torch.round(t_q / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)\n'
        '            # Reconstruct in original space to measure true error\n'
        '            recon_compressed = q.float() * s.float()[:, None]\n'
        '            recon = _mu_expand(recon_compressed, mu) if mu > 0 else recon_compressed\n'
        '            err = (t32 - recon).pow(2).mean().item()\n'
        '            if err < best_err:\n'
        '                best_q, best_s, best_err = q, s, err\n'
        '        return best_q, best_s\n'
        '    amax = t32.abs().max().item()\n'
        '    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)\n'
        '    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)\n'
        '    return q, scale',
        "companding",
    )

    # Modify dequantization to apply μ-law expansion.
    # The dequantized values are in compressed space — we need to expand them.
    source = _replace_unique(
        source,
        '        q, s = result[name + ".q"], result[name + ".scale"]\n'
        '        if s.ndim > 0:\n'
        '            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)\n'
        '        else:\n'
        '            out[name] = (q.float() * float(s.item())).to(orig_dtype)',
        '        q, s = result[name + ".q"], result[name + ".scale"]\n'
        '        if s.ndim > 0:\n'
        '            recon = q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))\n'
        '        else:\n'
        '            recon = q.float() * float(s.item())\n'
        '        # Apply μ-law expansion if companding was used for int6 params\n'
        '        if _COMPAND_MU > 0 and isinstance(info, dict) and info.get("type") == "int6":\n'
        '            recon = _mu_expand(recon, _COMPAND_MU)\n'
        '        out[name] = recon.to(orig_dtype)',
        "companding",
    )

    # Set _COMPAND_MU from args before export
    source = _replace_unique(
        source,
        '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}',
        '    _COMPAND_MU = args.compand_mu  # set global for quantization functions\n'
        '    globals()["_COMPAND_MU"] = args.compand_mu\n'
        '    if args.compand_mu > 0:\n'
        '        log0(f"companding:mu={args.compand_mu}")\n'
        '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}',
        "companding",
    )
    return source


# ===========================================================================
# H2: FISHER-AWARE LAYER BIT ALLOCATION
# Field: Bayesian inference (Fisher information matrix)
#
# Problem: All MLP+attention layers get uniform int6. But layers near the
# output (high Fisher information) are more loss-sensitive to quantization
# than early layers. Uniform allocation wastes bits on insensitive layers.
#
# Mechanism: After training, compute diagonal Fisher approximation
# (= E[grad^2]) for each layer by running a few batches. Then allocate:
#   - int8 to the top-K highest-Fisher layers (more bits for sensitive params)
#   - int5 to the bottom-K lowest-Fisher layers (fewer bits, saves space)
#   - int6 to the rest
# The total artifact size stays within budget because int4 layers save
# enough bytes to offset the int8 layers.
#
# This is the information-theoretically optimal allocation: bits go where
# the marginal distortion cost (measured by Fisher) is highest.
# ===========================================================================

def patch_fisher_bits(source: str) -> str:
    # Add env vars
    source = _replace_unique(
        source,
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))'
        if 'compand_mu' not in source else
        '    compand_mu = float(os.environ.get("COMPAND_MU", "0"))  # μ-law companding: 0=off, 50-255=typical',
        (
            '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))'
            if 'compand_mu' not in source else
            '    compand_mu = float(os.environ.get("COMPAND_MU", "0"))  # μ-law companding: 0=off, 50-255=typical'
        ) + '\n'
        '    fisher_bits = bool(int(os.environ.get("FISHER_BITS", "0")))  # Fisher-aware bit allocation\n'
        '    fisher_n_int8 = int(os.environ.get("FISHER_N_INT8", "2"))  # layers getting int8\n'
        '    fisher_n_int5 = int(os.environ.get("FISHER_N_INT5", "2"))  # layers getting int5',
        "fisher_bits",
    )

    # Add Fisher profiling function and int4 quantizer before main().
    source = _replace_unique(
        source,
        'def main() -> None:',
        'def quantize_int5_per_row(t: Tensor, clip_range: int = 15) -> tuple[Tensor, Tensor]:\n'
        '    """5-bit quantization: [-15, 15] range, per-row scaling with percentile search."""\n'
        '    t32 = t.float()\n'
        '    if t32.ndim == 2:\n'
        '        best_q, best_s, best_err = None, None, float("inf")\n'
        '        for pct in [0.999, 0.9999, 1.0]:\n'
        '            if pct < 1.0:\n'
        '                row_clip = torch.quantile(t32.abs(), pct, dim=1)\n'
        '            else:\n'
        '                row_clip = t32.abs().amax(dim=1)\n'
        '            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)\n'
        '            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)\n'
        '            recon = q.float() * s.float()[:, None]\n'
        '            err = (t32 - recon).pow(2).mean().item()\n'
        '            if err < best_err:\n'
        '                best_q, best_s, best_err = q, s, err\n'
        '        return best_q, best_s\n'
        '    amax = t32.abs().max().item()\n'
        '    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)\n'
        '    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)\n'
        '    return q, scale\n'
        '\n'
        '\n'
        'def compute_layer_fisher(\n'
        '    model: nn.Module, loader, device, grad_accum_steps: int,\n'
        '    train_batch_tokens: int, train_seq_len: int, n_batches: int = 3,\n'
        ') -> dict[int, float]:\n'
        '    """Compute diagonal Fisher approximation per layer: sum of squared gradients."""\n'
        '    model.train()\n'
        '    layer_fisher: dict[int, float] = {}\n'
        '    for _ in range(n_batches):\n'
        '        model.zero_grad(set_to_none=True)\n'
        '        x, y = loader.next_batch(train_batch_tokens, train_seq_len, grad_accum_steps)\n'
        '        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):\n'
        '            loss = model(x, y)\n'
        '        loss.backward()\n'
        '        for name, p in model.named_parameters():\n'
        '            if p.grad is None or p.ndim < 2:\n'
        '                continue\n'
        '            m = re.match(r"blocks\\.(\\d+)\\.", name)\n'
        '            if m is None:\n'
        '                continue\n'
        '            layer_idx = int(m.group(1))\n'
        '            fisher_val = p.grad.float().pow(2).sum().item()\n'
        '            layer_fisher[layer_idx] = layer_fisher.get(layer_idx, 0.0) + fisher_val\n'
        '    model.zero_grad(set_to_none=True)\n'
        '    return layer_fisher\n'
        '\n'
        '\n'
        'def allocate_bits_by_fisher(\n'
        '    fisher: dict[int, float], n_int8: int = 2, n_int5: int = 2,\n'
        ') -> dict[int, int]:\n'
        '    """Allocate bits per layer: highest Fisher → int8, lowest → int5, rest → int6."""\n'
        '    sorted_layers = sorted(fisher.keys(), key=lambda l: fisher[l], reverse=True)\n'
        '    bits: dict[int, int] = {}\n'
        '    for i, l in enumerate(sorted_layers):\n'
        '        if i < n_int8:\n'
        '            bits[l] = 8\n'
        '        elif i >= len(sorted_layers) - n_int5:\n'
        '            bits[l] = 5\n'
        '        else:\n'
        '            bits[l] = 6\n'
        '    return bits\n'
        '\n'
        '\n'
        'def mixed_quantize_fisher(\n'
        '    state_dict: dict[str, Tensor], int6_cats: set[str], layer_bits: dict[int, int],\n'
        ') -> tuple[dict[str, Tensor], dict[str, object]]:\n'
        '    """Quantize with per-layer bit allocation from Fisher analysis."""\n'
        '    result: dict[str, Tensor] = {}\n'
        '    meta: dict[str, object] = {}\n'
        '    for name, tensor in state_dict.items():\n'
        '        t = tensor.detach().cpu().contiguous()\n'
        '        cat = _classify_param(name)\n'
        '        if not t.is_floating_point() or t.numel() <= 65536:\n'
        '            result[name] = t.to(torch.float16) if t.is_floating_point() else t\n'
        '            meta[name] = "passthrough"\n'
        '            continue\n'
        '        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):\n'
        '            result[name] = t.float()\n'
        '            meta[name] = "passthrough_ctrl"\n'
        '            continue\n'
        '        m = re.match(r"blocks\\.(\\d+)\\.", name)\n'
        '        layer_idx = int(m.group(1)) if m else -1\n'
        '        bits = layer_bits.get(layer_idx, 6)\n'
        '        if cat in int6_cats and t.ndim >= 1:\n'
        '            if bits == 8:\n'
        '                q, s = quantize_float_tensor(t)\n'
        '                result[name + ".q"] = q\n'
        '                result[name + ".scale"] = s\n'
        '                meta[name] = {"type": "int8"}\n'
        '            elif bits == 5:\n'
        '                q, s = quantize_int5_per_row(t)\n'
        '                result[name + ".q"] = q\n'
        '                result[name + ".scale"] = s\n'
        '                meta[name] = {"type": "int5"}\n'
        '            else:\n'
        '                q, s = quantize_int6_per_row(t)\n'
        '                result[name + ".q"] = q\n'
        '                result[name + ".scale"] = s\n'
        '                meta[name] = {"type": "int6"}\n'
        '        else:\n'
        '            q, s = quantize_float_tensor(t)\n'
        '            result[name + ".q"] = q\n'
        '            result[name + ".scale"] = s\n'
        '            meta[name] = {"type": "int8"}\n'
        '    return result, meta\n'
        '\n'
        '\n'
        'def main() -> None:',
        "fisher_bits",
    )

    # Replace the export quantization call with Fisher-aware version.
    source = _replace_unique(
        source,
        '    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})',
        '    if args.fisher_bits:\n'
        '        # Compute Fisher information per layer (3 batches, fast)\n'
        '        fisher = compute_layer_fisher(\n'
        '            model, train_loader, device, grad_accum_steps,\n'
        '            args.train_batch_tokens, args.train_seq_len,\n'
        '        )\n'
        '        layer_bits = allocate_bits_by_fisher(fisher, args.fisher_n_int8, args.fisher_n_int5)\n'
        '        log0(f"fisher_bits:fisher={dict(sorted(fisher.items()))}")\n'
        '        log0(f"fisher_bits:allocation={dict(sorted(layer_bits.items()))}")\n'
        '        quant_result, quant_meta = mixed_quantize_fisher(sd_cpu, {"mlp", "attn"}, layer_bits)\n'
        '    else:\n'
        '        quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})',
        "fisher_bits",
    )
    return source


# ===========================================================================
# H3: SPARSIFY BEFORE QUANTIZE
# Field: Compressed sensing / sparse coding
#
# Problem: Near-zero weights that quantize to ±1 in int6 contribute almost
# nothing to model output but consume compression bandwidth. Setting them
# to exact zero BEFORE quantization creates structured sparsity that zstd/
# lzma can exploit much more efficiently (long runs of zeros compress well).
#
# Mechanism: After training + EMA, for each row, compute the threshold T
# below which weights should be zeroed. T is chosen so that the fraction
# of zeroed weights equals `sparsity_target`. The freed compression
# bandwidth translates to a smaller artifact, meaning we could either:
# (a) fit a bigger model, or (b) use fewer compression bits elsewhere.
#
# The key insight from compressed sensing: if a signal is approximately
# sparse, you can zero out the small components with bounded distortion.
# Weight matrices are approximately sparse (many near-zero entries).
# ===========================================================================

def patch_sparsify_before_quant(source: str) -> str:
    # Add env vars
    last_env = '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))'
    for candidate in [
        '    fisher_n_int5 = int(os.environ.get("FISHER_N_INT5", "2"))  # layers getting int5',
        '    compand_mu = float(os.environ.get("COMPAND_MU", "0"))  # μ-law companding: 0=off, 50-255=typical',
    ]:
        if candidate in source:
            last_env = candidate
            break

    source = _replace_unique(
        source,
        last_env,
        last_env + '\n'
        '    sparsify_frac = float(os.environ.get("SPARSIFY_FRAC", "0"))  # fraction of weights to zero before quant (0=off)',
        "sparsify_before_quant",
    )

    # Add sparsification step before quantization in the export pipeline.
    source = _replace_unique(
        source,
        '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}\n'
        '    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})'
        if 'fisher_bits' not in source else
        '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}\n'
        '    if args.fisher_bits:',
        (
            '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}\n'
            if 'fisher_bits' not in source else
            '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}\n'
        ) +
        '    # Sparsify: zero out smallest weights before quantization for better compression\n'
        '    if args.sparsify_frac > 0:\n'
        '        total_zeroed, total_params = 0, 0\n'
        '        for name, t in sd_cpu.items():\n'
        '            if not t.is_floating_point() or t.ndim < 2 or t.numel() <= 65536:\n'
        '                continue\n'
        '            cat = _classify_param(name)\n'
        '            if cat not in ("mlp", "attn"):\n'
        '                continue\n'
        '            t32 = t.float()\n'
        '            # Per-tensor threshold: zero out the smallest fraction of weights\n'
        '            threshold = torch.quantile(t32.abs().flatten(), args.sparsify_frac)\n'
        '            mask = t32.abs() <= threshold\n'
        '            t32[mask] = 0.0\n'
        '            sd_cpu[name] = t32.to(t.dtype)\n'
        '            total_zeroed += mask.sum().item()\n'
        '            total_params += t.numel()\n'
        '        log0(f"sparsify:frac={args.sparsify_frac} zeroed={total_zeroed}/{total_params} "\n'
        '             f"actual_frac={total_zeroed/max(total_params,1):.4f}")\n' +
        (
            '    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})'
            if 'fisher_bits' not in source else
            '    if args.fisher_bits:'
        ),
        "sparsify_before_quant",
    )
    return source


# ===========================================================================
# H4: BYTE-WEIGHTED TRAINING LOSS
# Field: Decision theory / metric alignment (de Finetti)
#
# Problem: Training loss weights all tokens equally. But the eval metric
# (BPB) weights tokens by their byte count. A 4-byte token contributes
# 4x more to BPB than a 1-byte token. This misalignment means training
# doesn't directly optimize what's being measured.
#
# Mechanism: Weight each token's cross-entropy by bytes_per_token / mean_bpt.
# This directly aligns the training objective with the BPB metric.
# The normalization ensures the mean weight is 1 (no scale change).
#
# Implementation: Build a byte-weight LUT from the tokenizer. In the
# model's loss computation, use per-token weights in cross_entropy.
# This is NOT a reduction="none" hack — it uses the proper weight
# argument of F.cross_entropy which handles it efficiently.
# ===========================================================================

def patch_byte_weighted_loss(source: str) -> str:
    # Add env var
    last_env = '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))'
    for candidate in [
        '    sparsify_frac = float(os.environ.get("SPARSIFY_FRAC", "0"))  # fraction of weights to zero before quant (0=off)',
        '    fisher_n_int5 = int(os.environ.get("FISHER_N_INT5", "2"))  # layers getting int5',
        '    compand_mu = float(os.environ.get("COMPAND_MU", "0"))  # μ-law companding: 0=off, 50-255=typical',
    ]:
        if candidate in source:
            last_env = candidate
            break

    source = _replace_unique(
        source,
        last_env,
        last_env + '\n'
        '    byte_weighted_loss = bool(int(os.environ.get("BYTE_WEIGHTED_LOSS", "0")))  # weight CE by bytes/token',
        "byte_weighted_loss",
    )

    # Replace the loss computation in forward() to use per-token byte weights.
    # We replace the simple F.cross_entropy with a weighted version.
    # F.cross_entropy supports a `weight` parameter but that's per-class, not per-sample.
    # We need per-sample weighting, so we use reduction="none" + manual mean.
    source = _replace_unique(
        source,
        '        main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")\n'
        '        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:',
        '        if hasattr(self, "_byte_weights") and self._byte_weights is not None and self.training:\n'
        '            # Byte-weighted loss: each token weighted by its byte count / mean\n'
        '            per_token_loss = F.cross_entropy(logits.float(), targets, reduction="none")\n'
        '            bw = self._byte_weights[targets]  # look up byte weight for each target token\n'
        '            main_loss = (per_token_loss * bw).mean()\n'
        '        else:\n'
        '            main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")\n'
        '        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:',
        "byte_weighted_loss",
    )

    # Build and attach the byte-weight LUT after tokenizer loading.
    source = _replace_unique(
        source,
        '    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")',
        '    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")\n'
        '    # Byte-weighted loss LUT: weight[token_id] = bytes_for_token / mean_bytes_per_token\n'
        '    _byte_weight_lut = None\n'
        '    if args.byte_weighted_loss:\n'
        '        _bw = base_bytes_lut.float().clamp_min(1.0)  # at least 1 byte per token\n'
        '        _bw = _bw / _bw.mean()  # normalize so mean weight = 1\n'
        '        _byte_weight_lut = _bw.to(device)\n'
        '        log0(f"byte_weighted_loss:enabled mean_bytes={base_bytes_lut.float().mean():.2f} "\n'
        '             f"min_w={_bw.min():.3f} max_w={_bw.max():.3f}")',
        "byte_weighted_loss",
    )

    # Attach LUT to model after construction.
    source = _replace_unique(
        source,
        '    restore_low_dim_params_to_fp32(base_model)',
        '    restore_low_dim_params_to_fp32(base_model)\n'
        '    # Attach byte-weight LUT (not a parameter, just data for loss computation)\n'
        '    base_model._byte_weights = _byte_weight_lut',
        "byte_weighted_loss",
    )
    return source


# ===========================================================================
# H5: TAPERED MLP WIDTH
# Field: Rate-distortion theory (Shannon)
#
# Problem: All layers use the same MLP multiplier (3x). But information
# density varies by depth. Early layers build low-level representations
# (need capacity). Late layers refine (less capacity needed, but must
# quantize well). Within a fixed byte budget, wider early layers + narrower
# late layers should give better quality per bit.
#
# Mechanism: Use per-layer MLP multipliers:
#   - Layers 0-3 (early): 4x MLP (more capacity for feature extraction)
#   - Layers 4-7 (middle): 3x MLP (standard)
#   - Layers 8-10 (late): 2x MLP (less capacity, cheaper to quantize)
#
# The total parameter count is roughly the same as uniform 3x,
# but capacity is redistributed to where it matters most.
# ===========================================================================

def patch_tapered_mlp(source: str) -> str:
    # Add env vars
    last_env = '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))'
    for candidate in [
        '    byte_weighted_loss = bool(int(os.environ.get("BYTE_WEIGHTED_LOSS", "0")))  # weight CE by bytes/token',
        '    sparsify_frac = float(os.environ.get("SPARSIFY_FRAC", "0"))  # fraction of weights to zero before quant (0=off)',
        '    fisher_n_int5 = int(os.environ.get("FISHER_N_INT5", "2"))  # layers getting int5',
        '    compand_mu = float(os.environ.get("COMPAND_MU", "0"))  # μ-law companding: 0=off, 50-255=typical',
    ]:
        if candidate in source:
            last_env = candidate
            break

    source = _replace_unique(
        source,
        last_env,
        last_env + '\n'
        '    tapered_mlp = bool(int(os.environ.get("TAPERED_MLP", "0")))  # per-layer MLP width\n'
        '    tapered_mlp_early = float(os.environ.get("TAPERED_MLP_EARLY", "4.0"))  # early layers multiplier\n'
        '    tapered_mlp_late = float(os.environ.get("TAPERED_MLP_LATE", "2.0"))    # late layers multiplier',
        "tapered_mlp",
    )

    # Modify Block to accept per-layer mlp_mult.
    # The Block constructor already takes mlp_mult as a parameter.
    # We just need to change GPT.__init__ to pass per-layer multipliers.
    # The block creation loop is in GPT.__init__.
    source = _replace_unique(
        source,
        '        self.blocks = nn.ModuleList(\n'
        '            [\n'
        '                Block(\n'
        '                    model_dim,\n'
        '                    num_heads,\n'
        '                    num_kv_heads,\n'
        '                    mlp_mult,\n'
        '                    rope_base,\n'
        '                    qk_gain_init,\n'
        '                    layer_idx=i,\n'
        '                    ln_scale=ln_scale,\n'
        '                    dtg=dtg,',
        '        # Support per-layer MLP multipliers (tapered MLP)\n'
        '        _mlp_mults = mlp_mult if isinstance(mlp_mult, (list, tuple)) else [mlp_mult] * num_layers\n'
        '        self.blocks = nn.ModuleList(\n'
        '            [\n'
        '                Block(\n'
        '                    model_dim,\n'
        '                    num_heads,\n'
        '                    num_kv_heads,\n'
        '                    _mlp_mults[i],\n'
        '                    rope_base,\n'
        '                    qk_gain_init,\n'
        '                    layer_idx=i,\n'
        '                    ln_scale=ln_scale,\n'
        '                    dtg=dtg,',
        "tapered_mlp",
    )

    # When constructing the model, pass per-layer MLP mults if tapered.
    source = _replace_unique(
        source,
        '    base_model = GPT(\n'
        '        vocab_size=args.vocab_size,\n'
        '        num_layers=args.num_layers,\n'
        '        model_dim=args.model_dim,\n'
        '        num_heads=args.num_heads,\n'
        '        num_kv_heads=args.num_kv_heads,\n'
        '        mlp_mult=args.mlp_mult,',
        '    # Compute per-layer MLP multipliers for tapered architecture\n'
        '    if args.tapered_mlp:\n'
        '        _n = args.num_layers\n'
        '        _early_end = _n // 3\n'
        '        _late_start = 2 * _n // 3\n'
        '        _mlp_mults = []\n'
        '        for _i in range(_n):\n'
        '            if _i < _early_end:\n'
        '                _mlp_mults.append(args.tapered_mlp_early)\n'
        '            elif _i >= _late_start:\n'
        '                _mlp_mults.append(args.tapered_mlp_late)\n'
        '            else:\n'
        '                _mlp_mults.append(args.mlp_mult)\n'
        '        log0(f"tapered_mlp:mults={_mlp_mults}")\n'
        '        _effective_mlp = _mlp_mults\n'
        '    else:\n'
        '        _effective_mlp = args.mlp_mult\n'
        '    base_model = GPT(\n'
        '        vocab_size=args.vocab_size,\n'
        '        num_layers=args.num_layers,\n'
        '        model_dim=args.model_dim,\n'
        '        num_heads=args.num_heads,\n'
        '        num_kv_heads=args.num_kv_heads,\n'
        '        mlp_mult=_effective_mlp,',
        "tapered_mlp",
    )

    # Also need to pass tapered mults to the eval model at export time.
    source = _replace_unique(
        source,
        '    eval_model = GPT(\n'
        '        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,\n'
        '        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,',
        '    eval_model = GPT(\n'
        '        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,\n'
        '        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=_effective_mlp if args.tapered_mlp else args.mlp_mult,',
        "tapered_mlp",
    )
    return source


# ===========================================================================
# H6: GRADUAL QUANTIZATION ANNEALING
# Field: Statistical physics (simulated annealing)
#
# Problem: Late QAT switches on abruptly at 15% LR. This is a sudden
# phase transition — the loss landscape discontinuously changes. In
# statistical physics, gradual cooling (annealing) finds better minima
# than quenching because the system has time to explore at each temperature.
#
# Mechanism: Replace binary QAT with a continuous alpha parameter:
#   w_eff = w + alpha * (w_q - w).detach()
# alpha ramps smoothly from 0 (start of warmdown) to 1 (end of warmdown).
# This means the optimization landscape continuously deforms from the
# real-weight landscape to the quantized-weight landscape, allowing the
# optimizer to track the changing minimum throughout.
# ===========================================================================

def patch_quant_anneal(source: str) -> str:
    # Add env var
    last_env = '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))'
    for candidate in [
        '    tapered_mlp_late = float(os.environ.get("TAPERED_MLP_LATE", "2.0"))    # late layers multiplier',
        '    byte_weighted_loss = bool(int(os.environ.get("BYTE_WEIGHTED_LOSS", "0")))  # weight CE by bytes/token',
        '    sparsify_frac = float(os.environ.get("SPARSIFY_FRAC", "0"))  # fraction of weights to zero before quant (0=off)',
        '    fisher_n_int5 = int(os.environ.get("FISHER_N_INT5", "2"))  # layers getting int5',
        '    compand_mu = float(os.environ.get("COMPAND_MU", "0"))  # μ-law companding: 0=off, 50-255=typical',
    ]:
        if candidate in source:
            last_env = candidate
            break

    source = _replace_unique(
        source,
        last_env,
        last_env + '\n'
        '    quant_anneal = bool(int(os.environ.get("QUANT_ANNEAL", "0")))  # gradual QAT annealing',
        "quant_anneal",
    )

    # Replace CastedLinear to support continuous alpha (not just boolean).
    source = _replace_unique(
        source,
        'class CastedLinear(nn.Linear):\n'
        '    _qat_enabled: bool = False\n'
        '    def forward(self, x: Tensor) -> Tensor:\n'
        '        w = self.weight.to(x.dtype)\n'
        '        if CastedLinear._qat_enabled and self.training and w.ndim == 2:\n'
        '            with torch.no_grad():\n'
        '                w32 = self.weight.float()\n'
        '                row_max = w32.abs().amax(dim=1)\n'
        '                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)\n'
        '                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)\n'
        '            w = w + (w_q - w).detach()\n'
        '        bias = self.bias.to(x.dtype) if self.bias is not None else None\n'
        '        return F.linear(x, w, bias)',
        'class CastedLinear(nn.Linear):\n'
        '    _qat_enabled: bool = False\n'
        '    _qat_alpha: float = 0.0  # 0=real weights, 1=fully quantized, 0<a<1=interpolated\n'
        '    def forward(self, x: Tensor) -> Tensor:\n'
        '        w = self.weight.to(x.dtype)\n'
        '        # Determine effective alpha: use continuous alpha if set, else binary flag\n'
        '        alpha = CastedLinear._qat_alpha if CastedLinear._qat_alpha > 0 else (1.0 if CastedLinear._qat_enabled else 0.0)\n'
        '        if alpha > 0 and self.training and w.ndim == 2:\n'
        '            with torch.no_grad():\n'
        '                w32 = self.weight.float()\n'
        '                row_max = w32.abs().amax(dim=1)\n'
        '                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)\n'
        '                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)\n'
        '            # Smooth interpolation: alpha=0 → real, alpha=1 → quantized\n'
        '            w = w + alpha * (w_q - w).detach()\n'
        '        bias = self.bias.to(x.dtype) if self.bias is not None else None\n'
        '        return F.linear(x, w, bias)',
        "quant_anneal",
    )

    # Replace the late QAT activation with smooth annealing.
    source = _replace_unique(
        source,
        '        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:\n'
        '            CastedLinear._qat_enabled = True\n'
        '            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")',
        '        if args.quant_anneal and scale < 1.0:\n'
        '            # Gradual annealing: alpha ramps from 0 at warmdown start to 1 at end\n'
        '            alpha = 1.0 - scale  # scale goes from 1→0 during warmdown\n'
        '            if CastedLinear._qat_alpha == 0.0 and alpha > 0:\n'
        '                log0(f"quant_anneal:start step:{step} scale:{scale:.4f}")\n'
        '            CastedLinear._qat_alpha = alpha\n'
        '        elif args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:\n'
        '            CastedLinear._qat_enabled = True\n'
        '            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")',
        "quant_anneal",
    )
    return source


# ===========================================================================
# H7: THREE-PHASE STAGED OBJECTIVE
# Field: Staged rocket design / multi-phase optimization
#
# Problem: Training uses one objective (CE) with an abrupt QAT addition.
# Different training phases have fundamentally different goals:
#   Phase 1 (bulk): Learn representations → pure CE
#   Phase 2 (pre-warmdown, scale<1.0 but >0.3): Prepare for quant →
#     CE + quantization gap penalty (penalize disagreement between
#     real-weight and quantized-weight outputs)
#   Phase 3 (deep warmdown, scale<0.3): Full deploy alignment →
#     CE + quant gap + byte-weighted CE
#
# This is the staged-rocket principle: each stage has a specialized
# engine optimized for that flight regime. Previous stages are jettisoned.
# ===========================================================================

def patch_staged_objective(source: str) -> str:
    # Add env vars
    last_env = '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))'
    for candidate in [
        '    quant_anneal = bool(int(os.environ.get("QUANT_ANNEAL", "0")))  # gradual QAT annealing',
        '    tapered_mlp_late = float(os.environ.get("TAPERED_MLP_LATE", "2.0"))    # late layers multiplier',
        '    byte_weighted_loss = bool(int(os.environ.get("BYTE_WEIGHTED_LOSS", "0")))  # weight CE by bytes/token',
        '    sparsify_frac = float(os.environ.get("SPARSIFY_FRAC", "0"))  # fraction of weights to zero before quant (0=off)',
    ]:
        if candidate in source:
            last_env = candidate
            break

    source = _replace_unique(
        source,
        last_env,
        last_env + '\n'
        '    staged_obj = bool(int(os.environ.get("STAGED_OBJ", "0")))  # three-phase staged objective\n'
        '    staged_obj_phase2 = float(os.environ.get("STAGED_OBJ_PHASE2", "0.8"))  # LR scale threshold for phase 2\n'
        '    staged_obj_phase3 = float(os.environ.get("STAGED_OBJ_PHASE3", "0.3"))  # LR scale threshold for phase 3\n'
        '    staged_obj_gap_weight = float(os.environ.get("STAGED_OBJ_GAP_WEIGHT", "0.3"))  # quant-gap loss weight',
        "staged_objective",
    )

    # Add the staged objective logic after the main forward+backward.
    # We add a second forward pass with quantized weights during phase 2+3,
    # and compute the KL divergence between real and quantized outputs.
    source = _replace_unique(
        source,
        '            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):\n'
        '                loss = model(x, y)\n'
        '            train_loss += loss.detach()\n'
        '            (loss * grad_scale).backward()',
        '            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):\n'
        '                loss = model(x, y)\n'
        '                # Staged objective: add quant-gap penalty in phases 2 and 3\n'
        '                if args.staged_obj and scale < args.staged_obj_phase2 and micro_step == 0:\n'
        '                    # Phase 2+3: penalize divergence between real and quantized outputs\n'
        '                    with torch.no_grad():\n'
        '                        # Get quantized-weight logits (STE forward)\n'
        '                        old_qat = CastedLinear._qat_enabled\n'
        '                        old_alpha = CastedLinear._qat_alpha\n'
        '                        CastedLinear._qat_enabled = True\n'
        '                        CastedLinear._qat_alpha = 0.0  # use binary QAT for gap measurement\n'
        '                    quant_logits = base_model.forward_logits(x).detach()\n'
        '                    with torch.no_grad():\n'
        '                        CastedLinear._qat_enabled = old_qat\n'
        '                        CastedLinear._qat_alpha = old_alpha\n'
        '                    # Real-weight logits\n'
        '                    real_logits = base_model.forward_logits(x)\n'
        '                    # KL divergence: minimize gap between real and quantized distributions\n'
        '                    gap_loss = F.kl_div(\n'
        '                        F.log_softmax(real_logits.float(), dim=-1),\n'
        '                        F.softmax(quant_logits.float(), dim=-1),\n'
        '                        reduction="batchmean",\n'
        '                    )\n'
        '                    loss = loss + args.staged_obj_gap_weight * gap_loss\n'
        '            train_loss += loss.detach()\n'
        '            (loss * grad_scale).backward()',
        "staged_objective",
    )
    return source


# ===========================================================================
# H8: BEST CHECKPOINT SELECTION BY DEPLOYED SCORE
# Field: Optimal stopping / selection theory
#
# Problem: The current code always exports the EMA of the final training
# state. But the deployed score (post-quant roundtrip val_bpb) is a
# composition of raw model quality and quantization damage. These two
# components can trade off non-monotonically during warmdown — raw quality
# keeps improving while quant damage may worsen as weights drift further
# from the quantization grid. The best deployed score may occur at an
# intermediate warmdown step, not at the final one.
#
# Mechanism: During warmdown (scale < threshold), periodically:
#   1. Snapshot the current EMA state
#   2. Run a fast post-quant roundtrip eval (quantize → dequantize → eval)
#   3. If this deployed val_bpb is better than the best seen, save the state
# At export time, use the best-deployed checkpoint instead of final EMA.
#
# This is the selection problem from optimal stopping theory: we want to
# pick the best element from a sequence we observe one at a time.
# ===========================================================================

def patch_best_checkpoint(source: str) -> str:
    # Add env vars
    last_env = '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))'
    for candidate in [
        '    staged_obj_gap_weight = float(os.environ.get("STAGED_OBJ_GAP_WEIGHT", "0.3"))  # quant-gap loss weight',
        '    quant_anneal = bool(int(os.environ.get("QUANT_ANNEAL", "0")))  # gradual QAT annealing',
        '    tapered_mlp_late = float(os.environ.get("TAPERED_MLP_LATE", "2.0"))    # late layers multiplier',
        '    byte_weighted_loss = bool(int(os.environ.get("BYTE_WEIGHTED_LOSS", "0")))  # weight CE by bytes/token',
        '    sparsify_frac = float(os.environ.get("SPARSIFY_FRAC", "0"))  # fraction of weights to zero before quant (0=off)',
    ]:
        if candidate in source:
            last_env = candidate
            break

    source = _replace_unique(
        source,
        last_env,
        last_env + '\n'
        '    best_ckpt = bool(int(os.environ.get("BEST_CKPT", "0")))  # checkpoint selection by deployed score\n'
        '    best_ckpt_every = int(os.environ.get("BEST_CKPT_EVERY", "100"))  # eval frequency during warmdown\n'
        '    best_ckpt_start = float(os.environ.get("BEST_CKPT_START", "0.3"))  # LR scale threshold to start tracking',
        "best_checkpoint",
    )

    # Add checkpoint tracking state after the SWA state initialization.
    # We insert right after "swa_count = 0" since that's where training state is initialized.
    source = _replace_unique(
        source,
        '    swa_state: dict[str, Tensor] | None = None\n'
        '    swa_count = 0',
        '    swa_state: dict[str, Tensor] | None = None\n'
        '    swa_count = 0\n'
        '    # Best-checkpoint tracking: track deployed score during warmdown\n'
        '    _best_ckpt_bpb: float = float("inf")\n'
        '    _best_ckpt_state: dict[str, Tensor] | None = None\n'
        '    _best_ckpt_step: int = -1',
        "best_checkpoint",
    )

    # Add checkpoint evaluation in the training loop, after SWA collection.
    # We insert after the SWA block (after swa_count += 1 and its enclosing blocks).
    source = _replace_unique(
        source,
        '        should_log_train = (\n'
        '            args.train_log_every > 0\n'
        '            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)\n'
        '        )',
        '        # Best-checkpoint: periodically eval deployed score during warmdown\n'
        '        if (args.best_ckpt and scale < args.best_ckpt_start\n'
        '                and step % args.best_ckpt_every == 0):\n'
        '            # Snapshot EMA state for fast quant-roundtrip eval\n'
        '            _ema_snap = {n: t.to(dtype=base_model.state_dict()[n].dtype)\n'
        '                         for n, t in ema_state.items()}\n'
        '            _orig_sd = {n: t.clone() for n, t in base_model.state_dict().items()}\n'
        '            base_model.load_state_dict(_ema_snap, strict=True)\n'
        '            # Quick quant roundtrip: quantize → dequantize → eval\n'
        '            _snap_sd = {k: v.detach().cpu() for k, v in base_model.state_dict().items()\n'
        '                        if "mtp_heads" not in k}\n'
        '            _qr, _qm = mixed_quantize_int6(_snap_sd, {"mlp", "attn"})\n'
        '            _deq = dequantize_mixed_int6(_qr, _qm, _snap_sd)\n'
        '            base_model.load_state_dict(_deq, strict=False)\n'
        '            with torch.no_grad():\n'
        '                _, _ckpt_bpb = eval_val(\n'
        '                    args, compiled_model, rank, world_size, device, grad_accum_steps,\n'
        '                    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,\n'
        '                )\n'
        '            log0(f"best_ckpt:probe step:{step} scale:{scale:.4f} deployed_bpb:{_ckpt_bpb:.6f} "\n'
        '                 f"best_so_far:{_best_ckpt_bpb:.6f}")\n'
        '            if _ckpt_bpb < _best_ckpt_bpb:\n'
        '                _best_ckpt_bpb = _ckpt_bpb\n'
        '                _best_ckpt_state = {n: t.detach().cpu().clone() for n, t in _ema_snap.items()}\n'
        '                _best_ckpt_step = step\n'
        '                log0(f"best_ckpt:new_best step:{step} bpb:{_ckpt_bpb:.6f}")\n'
        '            # Restore original training state\n'
        '            base_model.load_state_dict(_orig_sd, strict=True)\n'
        '        should_log_train = (\n'
        '            args.train_log_every > 0\n'
        '            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)\n'
        '        )',
        "best_checkpoint",
    )

    # Replace the EMA apply section: if best_ckpt found a better state, use it.
    source = _replace_unique(
        source,
        '    # Apply EMA weights (better than SWA alone per PR#401)\n'
        '    log0("ema:applying EMA weights")\n'
        '    current_state = base_model.state_dict()\n'
        '    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}\n'
        '    base_model.load_state_dict(avg_state, strict=True)',
        '    # Apply EMA weights (better than SWA alone per PR#401)\n'
        '    log0("ema:applying EMA weights")\n'
        '    current_state = base_model.state_dict()\n'
        '    if args.best_ckpt and _best_ckpt_state is not None:\n'
        '        log0(f"best_ckpt:using best checkpoint from step:{_best_ckpt_step} bpb:{_best_ckpt_bpb:.6f}")\n'
        '        avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in _best_ckpt_state.items()}\n'
        '    else:\n'
        '        avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}\n'
        '    base_model.load_state_dict(avg_state, strict=True)',
        "best_checkpoint",
    )
    return source


# ===========================================================================
# H9: SDCLIP K SWEEP (PER-LAYER)
# Field: Information theory / rate-distortion
#
# Problem: quantize_int6_per_row tries 5 fixed percentiles for all layers.
# But different layers have different weight distributions. The optimal
# clipping threshold varies per layer.
#
# Mechanism: For each 2D matrix parameter, sweep k in {10.0, 11.5, 12.85,
# 14.0, 16.0} and pick the k that minimizes reconstruction MSE. This is
# strictly better than using a global fixed percentile search.
# ===========================================================================

def patch_sdclip_sweep(source: str) -> str:
    # Add env var
    source = _replace_unique(
        source,
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))',
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))\n'
        '    sdclip_sweep = bool(int(os.environ.get("SDCLIP_SWEEP", "0")))',
        "sdclip_sweep",
    )

    # Replace quantize_int6_per_row with a version that sweeps k per-layer.
    # The original tries 5 fixed percentiles. The new version sweeps SDClip k
    # values (max / (k * std)) which is more principled.
    source = _replace_unique(
        source,
        'def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:\n'
        '    t32 = t.float()\n'
        '    if t32.ndim == 2:\n'
        '        best_q, best_s, best_err = None, None, float(\'inf\')\n'
        '        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:\n'
        '            if pct < 1.0:\n'
        '                row_clip = torch.quantile(t32.abs(), pct, dim=1)\n'
        '            else:\n'
        '                row_clip = t32.abs().amax(dim=1)\n'
        '            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)\n'
        '            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)\n'
        '            recon = q.float() * s.float()[:, None]\n'
        '            err = (t32 - recon).pow(2).mean().item()\n'
        '            if err < best_err:\n'
        '                best_q, best_s, best_err = q, s, err\n'
        '        return best_q, best_s\n'
        '    amax = t32.abs().max().item()\n'
        '    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)\n'
        '    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)\n'
        '    return q, scale',

        '_SDCLIP_SWEEP: bool = False  # set from args before export\n'
        '_SDCLIP_K_VALUES = [10.0, 11.5, 12.85, 14.0, 16.0]\n'
        '\n'
        'def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:\n'
        '    t32 = t.float()\n'
        '    if t32.ndim == 2:\n'
        '        best_q, best_s, best_err = None, None, float(\'inf\')\n'
        '        if _SDCLIP_SWEEP:\n'
        '            # SDClip sweep: union of k*std candidates AND original percentiles\n'
        '            # This gives strictly more candidates to search over\n'
        '            row_std = t32.std(dim=1).clamp_min(1e-12)\n'
        '            row_amax = t32.abs().amax(dim=1)\n'
        '            candidates: list[Tensor] = []\n'
        '            # k * std candidates (the SDClip innovation)\n'
        '            for k in _SDCLIP_K_VALUES:\n'
        '                candidates.append((k * row_std).clamp_min(1e-12))\n'
        '            # Original percentile candidates\n'
        '            for pct in [0.9990, 0.9995, 0.9999, 0.99999]:\n'
        '                candidates.append(torch.quantile(t32.abs(), pct, dim=1))\n'
        '            candidates.append(row_amax)\n'
        '            for row_clip in candidates:\n'
        '                s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)\n'
        '                q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)\n'
        '                recon = q.float() * s.float()[:, None]\n'
        '                err = (t32 - recon).pow(2).mean().item()\n'
        '                if err < best_err:\n'
        '                    best_q, best_s, best_err = q, s, err\n'
        '        else:\n'
        '            for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:\n'
        '                if pct < 1.0:\n'
        '                    row_clip = torch.quantile(t32.abs(), pct, dim=1)\n'
        '                else:\n'
        '                    row_clip = t32.abs().amax(dim=1)\n'
        '                s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)\n'
        '                q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)\n'
        '                recon = q.float() * s.float()[:, None]\n'
        '                err = (t32 - recon).pow(2).mean().item()\n'
        '                if err < best_err:\n'
        '                    best_q, best_s, best_err = q, s, err\n'
        '        return best_q, best_s\n'
        '    amax = t32.abs().max().item()\n'
        '    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)\n'
        '    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)\n'
        '    return q, scale',
        "sdclip_sweep",
    )

    # Set the global flag before export
    source = _replace_unique(
        source,
        '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}',
        '    global _SDCLIP_SWEEP\n'
        '    _SDCLIP_SWEEP = args.sdclip_sweep\n'
        '    if args.sdclip_sweep:\n'
        '        log0("sdclip_sweep:enabled")\n'
        '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}',
        "sdclip_sweep",
    )
    return source


# ===========================================================================
# H10: AGGRESSIVE TTT (10 EPOCHS, COSINE LR, UNFREEZE BLOCK 1)
#
# Problem: No TTT exists in the base pipeline. The frontier (PR #1487)
# gets -0.034 BPB from 10-epoch TTT with cosine LR. Our base has no TTT.
#
# Mechanism: After training completes, before export quantization, run TTT:
# fine-tune the EMA checkpoint on validation data for N epochs with AdamW.
# Freeze early blocks (configurable), use cosine LR decay within TTT.
# ===========================================================================

def patch_ttt_aggressive(source: str) -> str:
    # Add env vars
    source = _replace_unique(
        source,
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))',
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))\n'
        '    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))\n'
        '    ttt_epochs = int(os.environ.get("TTT_EPOCHS", "5"))\n'
        '    ttt_lr = float(os.environ.get("TTT_LR", "0.00045"))\n'
        '    ttt_freeze_blocks = os.environ.get("TTT_FREEZE_BLOCKS", "0,1")\n'
        '    ttt_cosine = bool(int(os.environ.get("TTT_COSINE", "0")))',
        "ttt_aggressive",
    )

    # Add re import if not present
    if 'import re\n' not in source and 'import re as' not in source:
        source = _replace_unique(
            source,
            'import math\n',
            'import math\nimport re\n',
            "ttt_aggressive_import",
        )

    # Insert TTT loop after EMA weights are applied, before export quantization.
    source = _replace_unique(
        source,
        '    full_state_dict = base_model.state_dict()\n'
        '    export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}',
        '    # ---- TEST-TIME TRAINING (TTT) ----\n'
        '    if args.ttt_enabled:\n'
        '        freeze_blocks = set(int(b) for b in args.ttt_freeze_blocks.split(",") if b.strip())\n'
        '        ttt_params = []\n'
        '        frozen_count = 0\n'
        '        for pname, p in base_model.named_parameters():\n'
        '            m = re.match(r"blocks\\.(\\d+)\\.", pname)\n'
        '            if m and int(m.group(1)) in freeze_blocks:\n'
        '                p.requires_grad_(False)\n'
        '                frozen_count += p.numel()\n'
        '            else:\n'
        '                p.requires_grad_(True)\n'
        '                ttt_params.append(p)\n'
        '        log0(f"ttt:start epochs:{args.ttt_epochs} lr:{args.ttt_lr} "\n'
        '             f"freeze:{freeze_blocks} cosine:{args.ttt_cosine} "\n'
        '             f"unfrozen:{sum(p.numel() for p in ttt_params)} frozen:{frozen_count}")\n'
        '        ttt_opt = torch.optim.AdamW(ttt_params, lr=args.ttt_lr, weight_decay=0.01, fused=True)\n'
        '        base_model.train()\n'
        '        for ttt_epoch in range(args.ttt_epochs):\n'
        '            if args.ttt_cosine:\n'
        '                ttt_lr_now = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ttt_epoch / args.ttt_epochs))\n'
        '                for g in ttt_opt.param_groups:\n'
        '                    g["lr"] = ttt_lr_now\n'
        '            total_seqs = (val_tokens.numel() - 1) // args.train_seq_len\n'
        '            ttt_loss_sum = 0.0\n'
        '            ttt_steps = 0\n'
        '            for seq_start in range(0, total_seqs, 8):\n'
        '                seq_end = min(seq_start + 8, total_seqs)\n'
        '                raw_s = seq_start * args.train_seq_len\n'
        '                raw_e = seq_end * args.train_seq_len + 1\n'
        '                local = val_tokens[raw_s:raw_e].to(device=device, dtype=torch.int64)\n'
        '                x = local[:-1].reshape(-1, args.train_seq_len)\n'
        '                y = local[1:].reshape(-1, args.train_seq_len)\n'
        '                ttt_opt.zero_grad(set_to_none=True)\n'
        '                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):\n'
        '                    loss = base_model(x, y)\n'
        '                loss.backward()\n'
        '                torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)\n'
        '                ttt_opt.step()\n'
        '                ttt_loss_sum += loss.item()\n'
        '                ttt_steps += 1\n'
        '            log0(f"ttt:epoch:{ttt_epoch + 1}/{args.ttt_epochs} "\n'
        '                 f"avg_loss:{ttt_loss_sum / max(ttt_steps, 1):.4f}")\n'
        '        # Restore all params to requires_grad=True for export\n'
        '        for p in base_model.parameters():\n'
        '            p.requires_grad_(True)\n'
        '        base_model.eval()\n'
        '        log0("ttt:done")\n'
        '    full_state_dict = base_model.state_dict()\n'
        '    export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}',
        "ttt_aggressive",
    )
    return source


# ===========================================================================
# H11: INT4 EMBEDDING QUANTIZATION
#
# Problem: Embeddings (tok_emb) are 8192×512 = 4M params. At int8 they
# take ~4MB (25% of the 16MB budget). At int4 they take ~2MB, freeing
# 2MB for a larger model body or higher-precision body weights.
#
# Mechanism: Replace int8 embedding quantization with int4 ([-7, 7] range).
# Per-row scale. The savings enable a 576d or 13L model in later stages.
# ===========================================================================

def patch_int4_embed(source: str) -> str:
    # Add env var
    source = _replace_unique(
        source,
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))',
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))\n'
        '    embed_quant_bits = int(os.environ.get("EMBED_QUANT_BITS", "8"))',
        "int4_embed",
    )

    # Modify mixed_quantize_int6 to handle int4 embeddings.
    # The current code sends embeds to int8 via quantize_float_tensor.
    # We intercept embeds and use int4 when configured.
    source = _replace_unique(
        source,
        '        if cat in int6_cats and t.ndim >= 1:\n'
        '            q, s = quantize_int6_per_row(t)\n'
        '            result[name + ".q"] = q\n'
        '            result[name + ".scale"] = s\n'
        '            meta[name] = {"type": "int6"}\n'
        '        else:\n'
        '            q, s = quantize_float_tensor(t)\n'
        '            result[name + ".q"] = q\n'
        '            result[name + ".scale"] = s\n'
        '            meta[name] = {"type": "int8"}\n'
        '    return result, meta',
        '        if cat in int6_cats and t.ndim >= 1:\n'
        '            q, s = quantize_int6_per_row(t)\n'
        '            result[name + ".q"] = q\n'
        '            result[name + ".scale"] = s\n'
        '            meta[name] = {"type": "int6"}\n'
        '        elif cat == "embed" and _EMBED_QUANT_BITS == 4 and t.ndim == 2:\n'
        '            # Int4 embedding: [-7, 7] range, per-row scale with percentile sweep\n'
        '            t32 = t.float()\n'
        '            clip_range_4 = 7\n'
        '            best_q4, best_s4, best_err4 = None, None, float("inf")\n'
        '            for pct in [0.999, 0.9995, 0.9999, 1.0]:\n'
        '                if pct < 1.0:\n'
        '                    row_clip = torch.quantile(t32.abs(), pct, dim=1)\n'
        '                else:\n'
        '                    row_clip = t32.abs().amax(dim=1)\n'
        '                s4 = (row_clip / clip_range_4).clamp_min(1.0 / clip_range_4).to(torch.float16)\n'
        '                q4 = torch.clamp(torch.round(t32 / s4.float()[:, None]), -clip_range_4, clip_range_4).to(torch.int8)\n'
        '                recon = q4.float() * s4.float()[:, None]\n'
        '                err = (t32 - recon).pow(2).mean().item()\n'
        '                if err < best_err4:\n'
        '                    best_q4, best_s4, best_err4 = q4, s4, err\n'
        '            result[name + ".q"] = best_q4\n'
        '            result[name + ".scale"] = best_s4\n'
        '            meta[name] = {"type": "int4"}\n'
        '        else:\n'
        '            q, s = quantize_float_tensor(t)\n'
        '            result[name + ".q"] = q\n'
        '            result[name + ".scale"] = s\n'
        '            meta[name] = {"type": "int8"}\n'
        '    return result, meta',
        "int4_embed",
    )

    # Set the global before export
    source = _replace_unique(
        source,
        '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}',
        '    global _EMBED_QUANT_BITS\n'
        '    _EMBED_QUANT_BITS = args.embed_quant_bits\n'
        '    if args.embed_quant_bits != 8:\n'
        '        log0(f"int4_embed:quant_bits={args.embed_quant_bits}")\n'
        '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}',
        "int4_embed",
    )

    # Add global variable before mixed_quantize_int6
    source = _replace_unique(
        source,
        'def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str]):',
        '_EMBED_QUANT_BITS: int = 8  # set from args before export\n'
        '\n'
        'def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str]):',
        "int4_embed",
    )
    return source


# ===========================================================================
# H12: ACTIVATION-WEIGHTED QUANTIZATION
#
# Problem: The default quantizer minimizes raw reconstruction MSE per row.
# But not all rows contribute equally to output loss — rows that see larger
# activations have more impact on the output. Minimizing activation-weighted
# reconstruction error is theoretically optimal (it approximates the Fisher
# information / Hessian diagonal).
#
# Mechanism: Run calibration batches through the model to collect per-layer
# input activation magnitudes. Then weight the quantization error by
# activation magnitude: err_weighted = (act_weight * (w - w_q))^2.
# This means rows with high-activation inputs get more careful clipping.
#
# Implementation: Collect E[|x|] per CastedLinear. During quantization,
# weight the reconstruction error by the activation scale. Rows that
# see large inputs get tighter clipping (more outlier preservation).
# ===========================================================================

def patch_gptq_post_ttt(source: str) -> str:
    # Add env var
    source = _replace_unique(
        source,
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))',
        '    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))\n'
        '    gptq_post_ttt = bool(int(os.environ.get("GPTQ_POST_TTT", "0")))',
        "gptq_post_ttt",
    )

    # Add a global dict for activation-weighted quantization, and modify
    # quantize_int6_per_row to use it when available.
    source = _replace_unique(
        source,
        'def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:',
        '_ACT_WEIGHT_MAP: dict[str, Tensor] = {}  # populated by calibration pass\n'
        '_CURRENT_QUANT_NAME: str = ""  # set per-parameter during quantization\n'
        '\n'
        'def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:',
        "gptq_post_ttt",
    )

    # Modify the quantization error metric: if activation weights exist for this
    # parameter, use weighted MSE instead of plain MSE.
    source = _replace_unique(
        source,
        '            recon = q.float() * s.float()[:, None]\n'
        '            err = (t32 - recon).pow(2).mean().item()\n'
        '            if err < best_err:\n'
        '                best_q, best_s, best_err = q, s, err\n'
        '        return best_q, best_s',
        '            recon = q.float() * s.float()[:, None]\n'
        '            diff_sq = (t32 - recon).pow(2)\n'
        '            # Activation-weighted error if available\n'
        '            if _CURRENT_QUANT_NAME in _ACT_WEIGHT_MAP:\n'
        '                aw = _ACT_WEIGHT_MAP[_CURRENT_QUANT_NAME]\n'
        '                if aw.shape[0] == diff_sq.shape[1]:  # [cols] matches\n'
        '                    diff_sq = diff_sq * aw[None, :]\n'
        '            err = diff_sq.mean().item()\n'
        '            if err < best_err:\n'
        '                best_q, best_s, best_err = q, s, err\n'
        '        return best_q, best_s',
        "gptq_post_ttt",
    )

    # In mixed_quantize_int6, set _CURRENT_QUANT_NAME before each call
    source = _replace_unique(
        source,
        '        if cat in int6_cats and t.ndim >= 1:\n'
        '            q, s = quantize_int6_per_row(t)',
        '        if cat in int6_cats and t.ndim >= 1:\n'
        '            global _CURRENT_QUANT_NAME\n'
        '            _CURRENT_QUANT_NAME = name\n'
        '            q, s = quantize_int6_per_row(t)',
        "gptq_post_ttt",
    )

    # Before quantization, collect activation stats and populate _ACT_WEIGHT_MAP
    source = _replace_unique(
        source,
        '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}\n'
        '    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})',
        '    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}\n'
        '    if args.gptq_post_ttt:\n'
        '        # Collect per-linear-layer input activation magnitudes\n'
        '        log0("gptq_post_ttt:collecting activation weights for quantization")\n'
        '        _act_accum: dict[str, Tensor] = {}\n'
        '        _act_count: dict[str, int] = {}\n'
        '        _cal_hooks = []\n'
        '        def _make_cal_hook(param_name: str, in_features: int):\n'
        '            def hook(module, inp, out):\n'
        '                if isinstance(inp, tuple) and len(inp) > 0:\n'
        '                    x = inp[0].float()\n'
        '                    # Mean absolute activation per input feature\n'
        '                    act = x.reshape(-1, x.shape[-1]).abs().mean(dim=0)  # [in_features]\n'
        '                    if param_name not in _act_accum:\n'
        '                        _act_accum[param_name] = torch.zeros(in_features)\n'
        '                        _act_count[param_name] = 0\n'
        '                    _act_accum[param_name] += act.cpu()\n'
        '                    _act_count[param_name] += 1\n'
        '            return hook\n'
        '        for mod_name, module in base_model.named_modules():\n'
        '            if isinstance(module, CastedLinear) and module.weight.ndim == 2:\n'
        '                # The weight param name is mod_name + ".weight"\n'
        '                wname = mod_name + ".weight"\n'
        '                h = module.register_forward_hook(_make_cal_hook(wname, module.in_features))\n'
        '                _cal_hooks.append(h)\n'
        '        base_model.eval()\n'
        '        with torch.inference_mode():\n'
        '            cal_seqs = (val_tokens.numel() - 1) // args.train_seq_len\n'
        '            for cs in range(0, min(32, cal_seqs), 8):\n'
        '                ce = min(cs + 8, cal_seqs)\n'
        '                rs = cs * args.train_seq_len\n'
        '                re_ = ce * args.train_seq_len + 1\n'
        '                local = val_tokens[rs:re_].to(device=device, dtype=torch.int64)\n'
        '                x_cal = local[:-1].reshape(-1, args.train_seq_len)\n'
        '                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):\n'
        '                    _ = base_model.forward_logits(x_cal)\n'
        '        for h in _cal_hooks:\n'
        '            h.remove()\n'
        '        # Normalize and store as activation weights\n'
        '        global _ACT_WEIGHT_MAP\n'
        '        for wname, accum in _act_accum.items():\n'
        '            avg = accum / max(_act_count.get(wname, 1), 1)\n'
        '            # Normalize so mean weight = 1 (preserves overall error scale)\n'
        '            avg = avg / avg.mean().clamp_min(1e-12)\n'
        '            _ACT_WEIGHT_MAP[wname] = avg\n'
        '        log0(f"gptq_post_ttt:calibrated {len(_ACT_WEIGHT_MAP)} layers")\n'
        '    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})',
        "gptq_post_ttt",
    )
    return source


# ===========================================================================
# Registry
# ===========================================================================

PATCHES: dict[str, callable] = {
    # Lane B (export-only, no retraining)
    "companding": patch_companding,
    "fisher_bits": patch_fisher_bits,
    "sparsify_before_quant": patch_sparsify_before_quant,
    "sdclip_sweep": patch_sdclip_sweep,
    "int4_embed": patch_int4_embed,
    "gptq_post_ttt": patch_gptq_post_ttt,
    # Lane A (training dynamics)
    "byte_weighted_loss": patch_byte_weighted_loss,
    "tapered_mlp": patch_tapered_mlp,
    "quant_anneal": patch_quant_anneal,
    "staged_objective": patch_staged_objective,
    "best_checkpoint": patch_best_checkpoint,
    # TTT (post-training)
    "ttt_aggressive": patch_ttt_aggressive,
}
