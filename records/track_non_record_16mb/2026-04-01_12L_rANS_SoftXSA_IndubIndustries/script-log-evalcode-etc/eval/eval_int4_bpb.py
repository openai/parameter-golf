"""
Int4 MLP BPB Evaluation — The Go/No-Go Test

Loads a checkpoint, quantizes MLP weights to int4 (vs int5 baseline),
dequantizes, and runs full BPB eval on the validation set.

This determines whether the dequant-matmul Triton kernel has a reason
to exist in the submission.

Usage:
    # Using the 2000-step quick baseline checkpoint:
    python eval_int4_bpb.py checkpoints/quick_baseline/step_2000.pt

    # Or the overnight run's periodic checkpoint (if available):
    python eval_int4_bpb.py checkpoints/overnight_baseline/step_4000.pt

    # Override eval settings:
    EVAL_STRIDE=0 python eval_int4_bpb.py <checkpoint>   # standard eval (faster)
    EVAL_STRIDE=64 python eval_int4_bpb.py <checkpoint>   # sliding window (slower, more accurate)

Requires GPU for the forward pass. If the overnight run is using the GPU,
either wait for it to finish or use a checkpoint it already saved.

NOTE: This script imports from train_gpt.py. Run from the repo root.
"""

from __future__ import annotations

import copy
import io
import math
import os
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import zstandard
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False

# Import model and eval infrastructure from train_gpt.py
from train_gpt import (
    Hyperparameters,
    GPT,
    CastedLinear,
    build_sentencepiece_luts,
    load_validation_tokens,
    eval_val,
    eval_val_sliding,
    mixed_quantize_int6,
    dequantize_mixed_int6,
    quantize_intN_per_row,
    restore_low_dim_params_to_fp32,
    _classify_param,
)


# ─── Block-diagonal Hadamard for BlkHad+int4 test ───

def _hadamard_po2(n, seed=42):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    H = H / np.sqrt(n)
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=n)
    return H * signs[np.newaxis, :]

def _block_diagonal_hadamard(n, block_size=512, seed=42):
    assert n % block_size == 0
    H_full = np.zeros((n, n), dtype=np.float64)
    for b in range(n // block_size):
        Hb = _hadamard_po2(block_size, seed=seed + b)
        i0, i1 = b * block_size, (b + 1) * block_size
        H_full[i0:i1, i0:i1] = Hb
    return H_full


def quantize_dequant_roundtrip(
    state_dict: dict[str, Tensor],
    mlp_clip: int = 15,
    label: str = "int5",
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, object]]:
    """Quantize and immediately dequantize a state dict.

    Uses the same mixed_quantize_int6 pipeline as training, but allows
    overriding the MLP clip range to test int4 (clip=7) vs int5 (clip=15).

    Returns (dequantized_state_dict, quant_result, quant_meta) so callers
    can measure compressed artifact size from quant_result.
    """
    import train_gpt

    def patched_mixed_quantize(sd, int6_cats):
        """mixed_quantize_int6 with configurable MLP clip range."""
        result = {}
        meta = {}
        for name, tensor in sd.items():
            t = tensor.detach().cpu().contiguous()
            cat = train_gpt._classify_param(name)

            if not t.is_floating_point() or t.numel() <= 8192:
                result[name] = t.to(torch.float16) if t.is_floating_point() else t
                meta[name] = "passthrough"
                continue
            if any(p in name for p in train_gpt.CONTROL_TENSOR_NAME_PATTERNS):
                result[name] = t.float()
                meta[name] = "passthrough_ctrl"
                continue
            if any(p in name for p in train_gpt.FP16_KEEP_NAME_PATTERNS):
                result[name] = t.to(dtype=torch.float16).contiguous()
                meta[name] = "passthrough_fp16"
                continue
            if cat in int6_cats and t.ndim >= 1:
                if cat == "mlp":
                    clip = mlp_clip  # THIS IS THE ONLY CHANGE
                else:
                    clip = 31  # int6 for attention (unchanged)
                q, s = quantize_intN_per_row(t, clip_range=clip)
                n_bits = 4 if clip <= 7 else (5 if clip <= 15 else 6)
                result[name + ".q"] = q
                result[name + ".scale"] = s
                meta[name] = {"type": f"int{n_bits}"}
            else:
                q, s = train_gpt.quantize_float_tensor(t)
                result[name + ".q"] = q
                result[name + ".scale"] = s
                meta[name] = {"type": "int8"}
        return result, meta

    quant_result, quant_meta = patched_mixed_quantize(state_dict, {"mlp", "attn", "bigram"})
    deq_state = dequantize_mixed_int6(quant_result, quant_meta, state_dict)

    return deq_state, quant_result, quant_meta


def measure_compressed_size(quant_result: dict[str, Tensor], quant_meta: dict[str, object]) -> int:
    """Compress a quantized artifact with zstd-22 and return byte count."""
    buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, buf)
    raw = buf.getvalue()
    if _HAS_ZSTD:
        compressed = zstandard.ZstdCompressor(level=22).compress(raw)
    else:
        import zlib
        compressed = zlib.compress(raw, 9)
    return len(compressed)


def run_eval(
    args: Hyperparameters,
    model: GPT,
    state_dict: dict[str, Tensor],
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    label: str = "",
    use_sliding: bool = False,
) -> tuple[float, float]:
    """Load state dict into model and run eval."""
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    if use_sliding:
        stride = int(os.environ.get("EVAL_STRIDE", "64"))
        batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", "32"))
        val_loss, val_bpb = eval_val_sliding(
            args, model, 0, 1, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=stride, batch_seqs=batch_seqs,
        )
    else:
        # eval_val expects grad_accum_steps to compute local batch size:
        # local_batch_tokens = val_batch_size // (world_size * grad_accum_steps)
        # With world_size=1 and grad_accum_steps=8: 524288 / 8 = 65536 tokens
        # = 32 sequences per batch. Safe for single-GPU eval.
        val_loss, val_bpb = eval_val(
            args, model, 0, 1, device, 8,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )

    # eval_val sets model.train() internally — restore eval mode so
    # the model stays in a consistent state between successive evals
    model.eval()

    print(f"  [{label:12s}]  val_loss={val_loss:.6f}  val_bpb={val_bpb:.6f}")
    return val_loss, val_bpb


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_int4_bpb.py <checkpoint_path>")
        print("  Set EVAL_STRIDE=64 for sliding window (slower, more accurate)")
        print("  Set EVAL_STRIDE=0 for standard eval (faster)")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    use_sliding = int(os.environ.get("EVAL_STRIDE", "0")) > 0
    skip_blkhad = int(os.environ.get("SKIP_BLKHAD", "0")) > 0
    eval_mode = "sliding_window" if use_sliding else "standard"

    print(f"\n{'='*70}")
    print(f"INT4 MLP BPB EVALUATION")
    print(f"{'='*70}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Eval mode:  {eval_mode}")
    print(f"  Compressor: {'zstd-22' if _HAS_ZSTD else 'zlib-9'}")
    print()

    # Setup
    args = Hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load tokenizer and validation data
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    print(f"  Validation tokens: {val_tokens.numel() - 1:,}")

    # Build model
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        raw_sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("blocks.") for k in ckpt):
        raw_sd = ckpt
    else:
        print(f"  Checkpoint keys: {list(ckpt.keys())[:10]}")
        raise ValueError("Can't find state_dict in checkpoint")

    # Load into model to get proper dtypes, then extract
    model.load_state_dict(raw_sd, strict=True)
    model.bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model)
    base_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Apply magnitude pruning (same as training script)
    with torch.no_grad():
        for name, param_tensor in base_sd.items():
            if param_tensor.ndim == 2 and param_tensor.numel() > 65536:
                threshold = torch.quantile(param_tensor.abs().float().flatten(), 0.03)
                mask = param_tensor.abs() < threshold
                base_sd[name] = param_tensor.masked_fill(mask, 0.0)

    print()
    print("  Running evaluations...")
    print()

    # --- Config A: int5 MLP (current baseline) ---
    t0 = time.perf_counter()
    sd_int5, qr_int5, qm_int5 = quantize_dequant_roundtrip(base_sd, mlp_clip=15, label="int5")
    _, bpb_int5 = run_eval(
        args, model, sd_int5, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        label="int5 MLP", use_sliding=use_sliding,
    )
    t_int5 = time.perf_counter() - t0
    size_int5 = measure_compressed_size(qr_int5, qm_int5)

    # --- Config B: int4 MLP ---
    t0 = time.perf_counter()
    sd_int4, qr_int4, qm_int4 = quantize_dequant_roundtrip(base_sd, mlp_clip=7, label="int4")
    _, bpb_int4 = run_eval(
        args, model, sd_int4, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        label="int4 MLP", use_sliding=use_sliding,
    )
    t_int4 = time.perf_counter() - t0
    size_int4 = measure_compressed_size(qr_int4, qm_int4)

    # --- Config C: int4 fc only, int5 proj (compromise) ---
    import train_gpt
    mixed_result = {}
    mixed_meta = {}
    for name, tensor in base_sd.items():
        t = tensor.detach().cpu().contiguous()
        cat = train_gpt._classify_param(name)
        if not t.is_floating_point() or t.numel() <= 8192:
            mixed_result[name] = t.to(torch.float16) if t.is_floating_point() else t
            mixed_meta[name] = "passthrough"
            continue
        if any(p in name for p in train_gpt.CONTROL_TENSOR_NAME_PATTERNS):
            mixed_result[name] = t.float()
            mixed_meta[name] = "passthrough_ctrl"
            continue
        if any(p in name for p in train_gpt.FP16_KEEP_NAME_PATTERNS):
            mixed_result[name] = t.to(dtype=torch.float16).contiguous()
            mixed_meta[name] = "passthrough_fp16"
            continue
        if cat == "mlp" and ".fc." in name and t.ndim >= 1:
            # int4 for MLP up-projection only
            q, s = quantize_intN_per_row(t, clip_range=7)
            mixed_result[name + ".q"] = q
            mixed_result[name + ".scale"] = s
            mixed_meta[name] = {"type": "int4"}
        elif cat in {"mlp", "attn", "bigram"} and t.ndim >= 1:
            clip = 15 if cat == "mlp" else 31
            q, s = quantize_intN_per_row(t, clip_range=clip)
            n_bits = 5 if clip <= 15 else 6
            mixed_result[name + ".q"] = q
            mixed_result[name + ".scale"] = s
            mixed_meta[name] = {"type": f"int{n_bits}"}
        else:
            q, s = train_gpt.quantize_float_tensor(t)
            mixed_result[name + ".q"] = q
            mixed_result[name + ".scale"] = s
            mixed_meta[name] = {"type": "int8"}

    sd_mixed_deq = dequantize_mixed_int6(mixed_result, mixed_meta, base_sd)
    _, bpb_mixed = run_eval(
        args, model, sd_mixed_deq, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        label="int4fc+int5proj", use_sliding=use_sliding,
    )
    size_mixed = measure_compressed_size(mixed_result, mixed_meta)

    # --- Config D: BlkHad+int4 on MLP proj only, int5 on MLP fc, int6 on attn ---
    # This is the Option D test: block-diagonal Hadamard rotation + int4 on proj
    if skip_blkhad:
        print(f"\n  Skipping Config D (BlkHad+int4) — SKIP_BLKHAD=1")
        bpb_blkhad = None
    else:
        pass  # fall through to existing code

    if not skip_blkhad:
        print(f"\n  Building block-diagonal 3×512 Hadamard for MLP proj...")
        n_cols = args.model_dim * int(args.mlp_mult)  # 512 * 3 = 1536
        H_np = _block_diagonal_hadamard(n_cols, block_size=512, seed=42)
        H_t = torch.from_numpy(H_np).float()
        orth_err = float((H_t @ H_t.T - torch.eye(n_cols)).abs().max())
        print(f"  Hadamard orthogonality err: {orth_err:.2e}")

        # Start from int5 roundtrip state dict, then replace MLP proj with BlkHad+int4
        sd_blkhad = {}
        for name, tensor in base_sd.items():
            t = tensor.detach().cpu().contiguous()
            cat = _classify_param(name)
            if cat == 'mlp' and '.proj.' in name and t.ndim == 2:
                w = t.float()
                w_rot = w @ H_t
                row_max = w_rot.abs().amax(dim=1)
                scale = (row_max / 7.0).clamp_min(1e-12).to(torch.float16)
                q = torch.clamp(torch.round(w_rot / scale.float()[:, None]), -8, 7).to(torch.int8)
                recon_rot = q.float() * scale.float()[:, None]
                recon = recon_rot @ H_t.T
                sd_blkhad[name] = recon.to(t.dtype)
            else:
                sd_blkhad[name] = t

        sd_blkhad_final = {}
        for name in base_sd:
            cat = _classify_param(name)
            if cat == 'mlp' and '.proj.' in name and base_sd[name].ndim == 2:
                sd_blkhad_final[name] = sd_blkhad[name]
            else:
                sd_blkhad_final[name] = sd_int5[name]

        _, bpb_blkhad = run_eval(
            args, model, sd_blkhad_final, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            label="blkhad+int4", use_sliding=use_sliding,
        )

    # Results
    delta_int4 = bpb_int4 - bpb_int5
    delta_mixed = bpb_mixed - bpb_int5
    savings_int4 = size_int5 - size_int4
    savings_mixed = size_int5 - size_mixed

    print(f"\n{'='*70}")
    print(f"RESULTS (all sizes measured with {'zstd-22' if _HAS_ZSTD else 'zlib-9'})")
    print(f"{'='*70}")
    print(f"  {'Config':22s}  {'BPB':>10s}  {'Delta':>10s}  {'Artifact':>12s}  {'Savings':>12s}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}")
    print(f"  {'int5 MLP (current)':22s}  {bpb_int5:10.6f}  {'baseline':>10s}  {size_int5:>10,} B  {'baseline':>12s}")
    print(f"  {'int4 all MLP':22s}  {bpb_int4:10.6f}  {delta_int4:+10.6f}  {size_int4:>10,} B  {savings_int4:>+10,} B")
    print(f"  {'int4 fc + int5 proj':22s}  {bpb_mixed:10.6f}  {delta_mixed:+10.6f}  {size_mixed:>10,} B  {savings_mixed:>+10,} B")
    if not skip_blkhad:
        delta_blkhad = bpb_blkhad - bpb_int5
        print(f"  {'BlkHad+int4 proj':22s}  {bpb_blkhad:10.6f}  {delta_blkhad:+10.6f}  {'(see RES-028)':>12s}  {'~468 KB':>12s}")

    print(f"\n  Eval times: int5={t_int5:.1f}s, int4={t_int4:.1f}s")

    # Decision logic
    print(f"\n{'='*70}")
    print(f"DECISION")
    print(f"{'='*70}")

    if not skip_blkhad:
        print(f"\n  === OPTION D TEST (BlkHad+int4 on MLP proj) ===")
        print(f"  BPB delta vs int5 roundtrip: {delta_blkhad:+.6f}")
        if abs(delta_blkhad) < 0.005:
            print(f"  >>> VERDICT: ACCEPTABLE (delta < 0.005). Option D lives! <<<")
            print(f"      Ship BlkHad+int4 on MLP proj as-is. Saves ~468KB.")
        elif abs(delta_blkhad) < 0.010:
            print(f"  VERDICT: MARGINAL (0.005 <= delta < 0.010). Check CPU noise control results.")
        else:
            print(f"  VERDICT: DEAD (delta >= 0.010). Option D killed by BPB cost.")

    # Plain int4 decisions
    print(f"\n  === PLAIN INT4 (no Hadamard) ===")
    if abs(delta_int4) < 0.003:
        print(f"  >> INT4 ALL MLP: GO")
        print(f"     BPB degradation {delta_int4:+.6f} is below 0.003 threshold.")
        print(f"     Saves {savings_int4:,} bytes.")
    elif abs(delta_mixed) < 0.003:
        print(f"  >> INT4 FC ONLY: GO (compromise)")
        print(f"     Full int4 too costly ({delta_int4:+.6f}), but fc-only ({delta_mixed:+.6f}) is viable.")
        print(f"     Saves {savings_mixed:,} bytes.")
    elif abs(delta_int4) < 0.010:
        print(f"  >> INT4 ALL MLP: MAYBE")
        print(f"     BPB degradation {delta_int4:+.6f} is between 0.003-0.010.")
        print(f"     Need: 11th layer BPB gain > {abs(delta_int4):.6f}")
    else:
        print(f"  >> INT4: NO-GO")
        print(f"     BPB degradation {delta_int4:+.6f} exceeds 0.010.")


if __name__ == "__main__":
    main()
