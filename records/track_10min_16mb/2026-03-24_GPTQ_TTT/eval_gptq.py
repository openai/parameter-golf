"""GPTQ post-training quantization for PR414+LeakyReLU² base model.

Implements Hessian-guided column-wise int6 quantization (GPTQ algorithm):
1. Load FP32 model from final_model.pt
2. Run calibration data through model to collect per-layer Hessians (H = X^T X)
3. Apply GPTQ: column-wise int6 quantization with block-128 error compensation
4. Save GPTQ-quantized model and eval with sliding window

Reference: PR#578 (256-sample calibration, block-128, Cholesky-factored error propagation)
"""
from __future__ import annotations

import copy
import glob
import io
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except ImportError:
    from flash_attn import flash_attn_func as flash_attn_3_func

sys.path.insert(0, str(Path(__file__).parent))
from train_leakyrelu2_pr414 import (
    Hyperparameters,
    GPT,
    CastedLinear,
    build_sentencepiece_luts,
    load_validation_tokens,
    load_data_shard,
    eval_val,
    eval_val_sliding,
    restore_low_dim_params_to_fp32,
    mixed_quantize_int6,
    dequantize_mixed_int6,
    quantize_int6_per_row,
    _classify_param,
    CONTROL_TENSOR_NAME_PATTERNS,
)

# ── GPTQ Hyperparameters ─────────────────────────────────────────────────────
GPTQ_NSAMPLES = int(os.environ.get("GPTQ_NSAMPLES", "256"))
GPTQ_BLOCK_SIZE = int(os.environ.get("GPTQ_BLOCK_SIZE", "128"))
GPTQ_DAMP_PCT = float(os.environ.get("GPTQ_DAMP_PCT", "0.01"))
GPTQ_CLIP_RANGE = int(os.environ.get("GPTQ_CLIP_RANGE", "31"))  # int6: ±31
GPTQ_PERCDAMP = float(os.environ.get("GPTQ_PERCDAMP", "0.01"))


# ── GPTQ Core Algorithm ──────────────────────────────────────────────────────

def gptq_quantize_weight(
    W: Tensor,  # [out_features, in_features], float32
    H: Tensor,  # [in_features, in_features], Hessian = X^T X, float32
    block_size: int = 128,
    clip_range: int = 31,
    percdamp: float = 0.01,
) -> tuple[Tensor, Tensor, float, float]:
    """Apply GPTQ algorithm to quantize weight matrix to int6.

    Returns: (quantized_int8, scale_fp16, elem_mse, hessian_weighted_mse)

    NOTE: GPTQ minimizes Hessian-weighted error, NOT element-wise MSE.
    Element-wise MSE will typically be ~1.1-1.3x worse than naive quantization.
    This is EXPECTED and correct — the Hessian-weighted error (which correlates
    with bpb) should be ~5-10% better.
    """
    W = W.clone().float()
    nrow, ncol = W.shape

    H = H.float()

    # Dampening
    damp = percdamp * torch.diag(H).mean()
    diag_idx = torch.arange(ncol, device=H.device)
    H[diag_idx, diag_idx] += damp

    # Find optimal per-row scale using same approach as naive quantizer
    # (search over percentiles to find best scale)
    W_orig = W.clone()
    best_scale = None
    best_scale_err = float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(W_orig.abs(), pct, dim=1)
        else:
            row_clip = W_orig.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        # Quick estimate: naive quantize and check error
        q = torch.clamp(torch.round(W_orig / s.float()[:, None]), -clip_range, clip_range)
        recon = q * s.float()[:, None]
        err = (W_orig - recon).pow(2).mean().item()
        if err < best_scale_err:
            best_scale = s
            best_scale_err = err

    scale = best_scale.float()  # [nrow]

    # Cholesky factorization of H
    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
    except RuntimeError:
        # Fallback: add more dampening
        H[diag_idx, diag_idx] += damp * 10
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
        except RuntimeError:
            # Last resort: pseudo-inverse
            Hinv = torch.linalg.pinv(H)

    Q = torch.zeros_like(W)
    Err = torch.zeros_like(W)

    # Process columns in blocks
    for col_start in range(0, ncol, block_size):
        col_end = min(col_start + block_size, ncol)
        block_cols = col_end - col_start

        # Get the block's Hessian inverse
        Hinv_block = Hinv[col_start:col_end, col_start:col_end]

        for j in range(col_start, col_end):
            w_col = W[:, j]  # [nrow]
            d = Hinv[j, j]

            # Quantize this column
            q_col = torch.clamp(
                torch.round(w_col / scale),
                -clip_range, clip_range
            )
            Q[:, j] = q_col

            # Quantization error
            err = (w_col - q_col * scale) / d
            Err[:, j] = err

            # Compensate remaining columns in this block
            if j + 1 < col_end:
                W[:, j + 1:col_end] -= err[:, None] * Hinv[j, j + 1:col_end][None, :]

        # Compensate remaining columns outside this block
        if col_end < ncol:
            W[:, col_end:] -= Err[:, col_start:col_end] @ Hinv[col_start:col_end, col_end:]

    Q_int8 = Q.to(torch.int8)

    # ── Post-GPTQ: Recompute per-row scale via least-squares ───────────
    # Q values are fixed (GPTQ-optimized integer assignments).
    # Find the scale that minimizes ||W_orig - scale * Q||² per row:
    #   scale_i = dot(W_orig[i], Q[i]) / dot(Q[i], Q[i])
    Q_float = Q_int8.float()
    dot_wq = (W_orig * Q_float).sum(dim=1)       # [nrow]
    dot_qq = (Q_float * Q_float).sum(dim=1)       # [nrow]
    ls_scale = torch.where(dot_qq > 0, dot_wq / dot_qq, scale)
    ls_scale = ls_scale.clamp_min(1.0 / clip_range)

    # Per-row: pick whichever scale gives lower element-wise MSE
    recon_orig = Q_float * scale[:, None]
    recon_ls = Q_float * ls_scale[:, None]
    mse_orig_per_row = (W_orig - recon_orig).pow(2).mean(dim=1)
    mse_ls_per_row = (W_orig - recon_ls).pow(2).mean(dim=1)
    use_ls = mse_ls_per_row < mse_orig_per_row
    final_scale = torch.where(use_ls, ls_scale, scale)
    scale_fp16 = final_scale.to(torch.float16)

    # Compute element-wise MSE
    recon = Q_float * scale_fp16.float()[:, None]
    elem_mse = (W_orig - recon).pow(2).mean().item()

    # Compute Hessian-weighted error (the metric GPTQ actually optimizes)
    # hw_err = tr((W-Q*s)^T H (W-Q*s)) / nrow
    H_raw = H.clone()  # H already has dampening added
    diff = W_orig - recon  # [nrow, ncol]
    hw_mse = (diff @ H_raw * diff).sum().item() / nrow

    return Q_int8, scale_fp16, elem_mse, hw_mse


# ── Hessian Collection ────────────────────────────────────────────────────────

class HessianCollector:
    """Hook-based collector for per-layer input Hessians H = X^T X."""

    def __init__(self):
        self.hessians: dict[str, Tensor] = {}
        self.nsamples: dict[str, int] = {}
        self.hooks = []

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            inp = input[0].detach().clone()
            if inp.ndim == 3:
                inp = inp.reshape(-1, inp.shape[-1])
            inp = inp.float()
            H = inp.T @ inp  # [dim, dim]
            if name not in self.hessians:
                self.hessians[name] = H.cpu()
                self.nsamples[name] = inp.shape[0]
            else:
                self.hessians[name] = self.hessians[name] + H.cpu()
                self.nsamples[name] += inp.shape[0]
        return hook_fn

    def register(self, model: nn.Module):
        """Register hooks on all CastedLinear modules."""
        for name, module in model.named_modules():
            if isinstance(module, CastedLinear):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def normalize(self):
        """Average Hessians by number of samples."""
        for name in list(self.hessians.keys()):
            self.hessians[name] = self.hessians[name].clone() / self.nsamples[name]


def make_model(args, device):
    """Create a fresh GPT model for evaluation."""
    m = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims,
        ln_scale=args.ln_scale, dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for mod in m.modules():
        if isinstance(mod, CastedLinear):
            mod.float()
    restore_low_dim_params_to_fp32(m)
    return m


def collect_calibration_data(args, device, nsamples: int = 256, seq_len: int = 2048):
    """Load calibration sequences from training data."""
    train_files = sorted(glob.glob(args.train_files))
    if not train_files:
        raise FileNotFoundError(f"No training files found: {args.train_files}")

    all_tokens = []
    for f in train_files:
        tokens = load_data_shard(Path(f))
        all_tokens.append(tokens)
        total = sum(t.numel() for t in all_tokens)
        if total >= nsamples * seq_len + 1:
            break

    all_tokens = torch.cat(all_tokens)
    sequences = []
    for i in range(nsamples):
        start = i * seq_len
        end = start + seq_len + 1
        if end > all_tokens.numel():
            break
        sequences.append(all_tokens[start:end])

    return sequences


def collect_hessians(model, sequences, device, batch_size=8):
    """Run calibration sequences through model and collect per-layer Hessians."""
    collector = HessianCollector()
    collector.register(model)

    model.eval()
    with torch.inference_mode():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            # Stack into batch
            x_list = [s[:-1] for s in batch_seqs]
            y_list = [s[1:] for s in batch_seqs]
            x = torch.stack(x_list).to(device=device, dtype=torch.int64)
            y = torch.stack(y_list).to(device=device, dtype=torch.int64)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                _ = model(x, y)

            if (i // batch_size + 1) % 4 == 0:
                print(f"  Calibration: {min(i + batch_size, len(sequences))}/{len(sequences)} sequences")

    collector.remove_hooks()
    collector.normalize()

    return collector.hessians


def gptq_quantize_state_dict(
    state_dict: dict[str, Tensor],
    hessians: dict[str, Tensor],
    int6_cats: set[str],
    block_size: int = 128,
    clip_range: int = 31,
    percdamp: float = 0.01,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    """Apply GPTQ quantization to state dict, replacing naive int6 for layers with Hessians.

    For layers without Hessians (embeddings, small tensors), falls back to naive quantization.
    """
    num_layers_total = max(
        (int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")),
        default=0,
    ) + 1

    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}

    gptq_stats = {"gptq_layers": 0, "naive_layers": 0, "passthrough_layers": 0}
    gptq_errors = {}
    gptq_hw_errors = {}
    naive_errors = {}
    naive_hw_errors = {}

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)

        # Passthrough: non-float or small tensors
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            gptq_stats["passthrough_layers"] += 1
            continue

        # Passthrough: control tensors
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            gptq_stats["passthrough_layers"] += 1
            continue

        if cat in int6_cats and t.ndim >= 1:
            # Find matching Hessian
            # Module name for the layer containing this weight
            # e.g., "blocks.0.attn.c_q.weight" -> module name "blocks.0.attn.c_q"
            layer_module_name = name.rsplit(".weight", 1)[0] if name.endswith(".weight") else None

            hessian = hessians.get(layer_module_name) if layer_module_name else None

            if hessian is not None and t.ndim == 2:
                # GPTQ quantization
                # First compute naive error for comparison
                q_naive, s_naive = quantize_int6_per_row(t)
                naive_recon = q_naive.float() * s_naive.float()[:, None]
                naive_mse = (t.float() - naive_recon).pow(2).mean().item()
                # Naive Hessian-weighted error
                H_raw = hessian.cpu().float()
                diff_naive = t.float() - naive_recon
                naive_hw = (diff_naive @ H_raw * diff_naive).sum().item() / t.shape[0]

                q_gptq, s_gptq, gptq_mse, gptq_hw = gptq_quantize_weight(
                    t, hessian.cpu(), block_size, clip_range, percdamp
                )

                result[name + ".q"] = q_gptq
                result[name + ".scale"] = s_gptq
                meta[name] = {"type": "int6"}
                gptq_stats["gptq_layers"] += 1
                gptq_errors[name] = gptq_mse
                gptq_hw_errors[name] = gptq_hw
                naive_errors[name] = naive_mse
                naive_hw_errors[name] = naive_hw

                ratio_elem = gptq_mse / naive_mse if naive_mse > 0 else 1.0
                ratio_hw = gptq_hw / naive_hw if naive_hw > 0 else 1.0
                print(f"  GPTQ {name}: elemMSE {ratio_elem:.3f}x | hessMSE {ratio_hw:.3f}x")
            else:
                # Fall back to naive int6
                q, s = quantize_int6_per_row(t)
                result[name + ".q"] = q
                result[name + ".scale"] = s
                meta[name] = {"type": "int6"}
                gptq_stats["naive_layers"] += 1
        else:
            # int8 quantization (same as original)
            from train_leakyrelu2_pr414 import quantize_float_tensor
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
            gptq_stats["naive_layers"] += 1

    print(f"\nGPTQ stats: {gptq_stats}")
    if gptq_errors:
        total_gptq_elem = sum(gptq_errors.values())
        total_naive_elem = sum(naive_errors.values())
        total_gptq_hw = sum(gptq_hw_errors.values())
        total_naive_hw = sum(naive_hw_errors.values())
        print(f"Element-wise MSE: naive={total_naive_elem:.6e}, gptq={total_gptq_elem:.6e}, ratio={total_gptq_elem/total_naive_elem:.4f}")
        print(f"Hessian-W  MSE:   naive={total_naive_hw:.6e}, gptq={total_gptq_hw:.6e}, ratio={total_gptq_hw/total_naive_hw:.4f}")
        print(f"NOTE: GPTQ minimizes Hessian-weighted error. Element-wise MSE increase is EXPECTED.")

    return result, meta


SKIP_BASELINE = bool(int(os.environ.get("SKIP_BASELINE", "1")))
KNOWN_BASELINE_BPB = float(os.environ.get("KNOWN_BASELINE_BPB", "1.1243"))


def main() -> None:
    print("GPTQ eval starting...", flush=True)
    args = Hyperparameters()
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    print("Loading tokenizer...", flush=True)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    print("Loading validation tokens...", flush=True)
    val_tokens = load_validation_tokens(args.val_files, max(args.train_seq_len, effective_eval_seq_len))
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    if master_process:
        print(f"val tokens: {val_tokens.numel() - 1}")
        print(f"GPTQ config: nsamples={GPTQ_NSAMPLES}, block_size={GPTQ_BLOCK_SIZE}, "
              f"percdamp={GPTQ_PERCDAMP}, clip_range={GPTQ_CLIP_RANGE}")

    # ── Step 1: Load FP32 model ───────────────────────────────────────────
    fp32_model_path = os.environ.get("MODEL_PATH", "final_model.pt")
    if master_process:
        print(f"\nLoading FP32 model from {fp32_model_path}", flush=True)

    CastedLinear._qat_enabled = False
    model = make_model(args, device)

    fp32_sd = torch.load(fp32_model_path, map_location="cpu")
    model.load_state_dict(fp32_sd, strict=True)

    if master_process:
        print(f"Model loaded: {sum(p.numel() for p in model.parameters())} params", flush=True)

    # ── Step 2: Baseline eval (skip if known) ─────────────────────────────
    # Build template state dict (needed for dequantization later)
    template_model = make_model(args, torch.device("cpu"))
    template_sd = {k: v.cpu() for k, v in template_model.state_dict().items()}
    del template_model

    if SKIP_BASELINE:
        base_val_bpb = KNOWN_BASELINE_BPB
        base_time = 0.0
        if master_process:
            print(f"\nSkipping baseline eval (known: {base_val_bpb})", flush=True)
    else:
        if master_process:
            print(f"\n{'='*60}")
            print(f"BASELINE EVAL (naive int6, sliding window stride={args.eval_stride})")
            print(f"{'='*60}", flush=True)

        naive_int6_path = os.environ.get("NAIVE_MODEL_PATH", "final_model.int6.ptz")
        with open(naive_int6_path, "rb") as f:
            quant_blob = f.read()
        if _COMPRESSOR == "zstd":
            quant_raw = zstandard.ZstdDecompressor().decompress(quant_blob)
        else:
            quant_raw = zlib.decompress(quant_blob)
        quant_state = torch.load(io.BytesIO(quant_raw), map_location="cpu")

        naive_deq = dequantize_mixed_int6(quant_state["w"], quant_state["m"], template_sd)

        naive_model = make_model(args, device)
        naive_model.load_state_dict(naive_deq, strict=True)

        torch.cuda.synchronize()
        t_base = time.perf_counter()
        base_val_loss, base_val_bpb = eval_val_sliding(
            args, naive_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, eval_seq_len=effective_eval_seq_len,
        )
        torch.cuda.synchronize()
        base_time = time.perf_counter() - t_base
        if master_process:
            print(f"Naive int6 bpb: {base_val_bpb:.6f} (time: {base_time:.1f}s)")

        del naive_model
        torch.cuda.empty_cache()

    # ── Step 3: Collect Hessians ──────────────────────────────────────────
    if master_process:
        print(f"\n{'='*60}")
        print(f"COLLECTING HESSIANS ({GPTQ_NSAMPLES} calibration sequences)")
        print(f"{'='*60}", flush=True)

    torch.cuda.synchronize()
    t_cal = time.perf_counter()

    cal_sequences = collect_calibration_data(args, device, GPTQ_NSAMPLES, args.train_seq_len)
    if master_process:
        print(f"Loaded {len(cal_sequences)} calibration sequences ({args.train_seq_len} tokens each)", flush=True)

    hessians = collect_hessians(model, cal_sequences, device, batch_size=16)

    torch.cuda.synchronize()
    cal_time = time.perf_counter() - t_cal
    if master_process:
        print(f"Hessian collection time: {cal_time:.1f}s")
        print(f"Collected Hessians for {len(hessians)} layers:")
        for name, H in sorted(hessians.items()):
            diag_mean = H.diag().mean().item()
            print(f"  {name}: shape={tuple(H.shape)}, diag_mean={diag_mean:.4e}")
        sys.stdout.flush()

    del model, cal_sequences
    torch.cuda.empty_cache()

    # ── Step 4: GPTQ quantization ─────────────────────────────────────────
    if master_process:
        print(f"\n{'='*60}")
        print(f"GPTQ QUANTIZATION")
        print(f"{'='*60}", flush=True)

    t_gptq = time.perf_counter()

    sd_cpu = {k: v.detach().cpu() for k, v in fp32_sd.items()}
    gptq_result, gptq_meta = gptq_quantize_state_dict(
        sd_cpu, hessians, {"mlp", "attn"},
        block_size=GPTQ_BLOCK_SIZE,
        clip_range=GPTQ_CLIP_RANGE,
        percdamp=GPTQ_PERCDAMP,
    )

    gptq_time = time.perf_counter() - t_gptq
    if master_process:
        print(f"GPTQ quantization time: {gptq_time:.1f}s", flush=True)

    # ── Step 5: Save and eval GPTQ model ──────────────────────────────────
    if master_process:
        print(f"\n{'='*60}")
        print(f"GPTQ EVAL (sliding window stride={args.eval_stride})")
        print(f"{'='*60}", flush=True)

    # Dequantize GPTQ result
    gptq_deq = dequantize_mixed_int6(gptq_result, gptq_meta, template_sd)

    gptq_model = make_model(args, device)
    gptq_model.load_state_dict(gptq_deq, strict=True)

    torch.cuda.synchronize()
    t_gptq_eval = time.perf_counter()
    gptq_val_loss, gptq_val_bpb = eval_val_sliding(
        args, gptq_model, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=args.eval_stride, eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    gptq_eval_time = time.perf_counter() - t_gptq_eval
    if master_process:
        print(f"GPTQ int6 bpb: {gptq_val_bpb:.6f} (time: {gptq_eval_time:.1f}s)", flush=True)

    # ── Save GPTQ quantized model ────────────────────────────────────────
    gptq_file_bytes = 0
    if master_process:
        quant_buf = io.BytesIO()
        torch.save({"w": gptq_result, "m": gptq_meta}, quant_buf)
        quant_raw = quant_buf.getvalue()
        if _COMPRESSOR == "zstd":
            # Use LDM (long-distance matching) for better cross-layer compression
            params = zstandard.ZstdCompressionParameters.from_level(
                21, enable_ldm=True, window_log=25,
            )
            quant_blob = zstandard.ZstdCompressor(compression_params=params).compress(quant_raw)
        else:
            quant_blob = zlib.compress(quant_raw, 9)

        with open("final_model.gptq.int6.ptz", "wb") as f:
            f.write(quant_blob)
        gptq_file_bytes = len(quant_blob)
        print(f"\nGPTQ model saved: {gptq_file_bytes} bytes")

    # ── Results ───────────────────────────────────────────────────────────
    if master_process:
        delta = base_val_bpb - gptq_val_bpb
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Naive int6 sliding-window bpb:  {base_val_bpb:.6f}")
        print(f"GPTQ int6 sliding-window bpb:   {gptq_val_bpb:.6f}")
        print(f"Delta (naive - GPTQ):           {delta:+.6f}")
        print(f"Improvement:                    {delta:.6f}")
        print(f"")
        print(f"Calibration time:               {cal_time:.1f}s")
        print(f"GPTQ quantization time:         {gptq_time:.1f}s")
        print(f"Baseline eval time:             {base_time:.1f}s")
        print(f"GPTQ eval time:                 {gptq_eval_time:.1f}s")
        print(f"Peak GPU memory:                {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

        # Kill criterion check
        if delta < 0.001:
            print(f"\n!! KILL: GPTQ improvement {delta:.6f} < 0.001 -- not worth the complexity")
        else:
            print(f"\n++ PASS: GPTQ improvement {delta:.6f} >= 0.001 -- worth pursuing")

        results = {
            "naive_bpb": base_val_bpb,
            "gptq_bpb": gptq_val_bpb,
            "delta_bpb": delta,
            "calibration_time_s": cal_time,
            "gptq_time_s": gptq_time,
            "baseline_eval_time_s": base_time,
            "gptq_eval_time_s": gptq_eval_time,
            "gptq_nsamples": GPTQ_NSAMPLES,
            "gptq_block_size": GPTQ_BLOCK_SIZE,
            "gptq_percdamp": GPTQ_PERCDAMP,
            "gptq_clip_range": GPTQ_CLIP_RANGE,
            "gptq_file_bytes": gptq_file_bytes,
            "peak_gpu_mib": torch.cuda.max_memory_allocated() // 1024 // 1024,
            "kill_criterion_met": delta < 0.001,
        }
        with open("gptq_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to gptq_results.json")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
