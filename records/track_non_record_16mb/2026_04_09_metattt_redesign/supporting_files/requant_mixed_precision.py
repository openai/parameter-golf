#!/usr/bin/env python3
"""Re-quantize final_model.pt with MIXED int6/int7 precision (no retrain).

Given a saved float checkpoint, this script:
  1. Generates the same autoregressive GPTQ calibration tokens train_gpt.py uses.
  2. Collects per-linear hessians from the calibration tokens.
  3. Runs mixed_quantize_int6 with a per-tensor precision override: any tensor
     whose name matches one of INT7_PATTERNS is quantized with clip_range=63
     (7-bit range) instead of clip_range=31 (6-bit range). Everything else
     stays int6 — identical to the canonical path.
  4. Selective-prune to fit the TARGET_MB byte budget (LZMA-9 compressed).
  5. Saves as OUTPUT_PTZ_PATH.
  6. Loads back, dequantizes, and runs:
       a. Baseline eval_val → "final_mixed_roundtrip val_bpb"
       b. eval_val_sliding_ttt → "legal_ttt val_bpb"

The dequant path uses `dequantize_mixed_int6` unchanged — its math is `q*scale`
regardless of whether the value range was clip_range=31 or 63, so there is no
special-casing needed on the read side. We just tag the meta with "int7"
instead of "int6" so the precision is visible in logs / introspection.

ENV vars
--------
  MODEL_PT_PATH       path to float final_model.pt (default: ./final_model.pt)
  OUTPUT_PTZ_PATH     where to write the new artifact (default: ./final_model.mixed.ptz)
  INT7_PATTERNS       comma-separated substrings; any unbanked-tensor name matching
                      ANY pattern gets clip_range=63 treatment
                      (default: "blocks.0.,blocks.10.,mlp.proj")
  SKIP_TTT            if "1", skip the TTT eval (baseline only)
  TTT_QAT             1/0 for CastedLinear._qat_enabled during TTT adapt (default 1)
  TTT_EPOCHS          override TTT_EPOCHS env used by the inherited Hyperparameters
  TARGET_MB           override target size budget (default: 15.9)
  TRAIN_GPT_DIR       path to the folder containing train_gpt.py to import from
                      (default: the folder this script lives in)
"""
from __future__ import annotations

import io
import lzma
import math
import os
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Import train_gpt.py from a configurable directory so we can pin to the exp106
# version even when launched from /workspace/parameter-golf.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_GPT_DIR = Path(os.environ.get("TRAIN_GPT_DIR", str(SCRIPT_DIR))).resolve()
if not (TRAIN_GPT_DIR / "train_gpt.py").exists():
    raise FileNotFoundError(
        f"train_gpt.py not found in TRAIN_GPT_DIR={TRAIN_GPT_DIR}. "
        "Set TRAIN_GPT_DIR to the folder containing the exp106 train_gpt.py."
    )
sys.path.insert(0, str(TRAIN_GPT_DIR))

import sentencepiece as spm  # noqa: E402
from train_gpt import (  # noqa: E402
    GPT,
    CastedLinear,
    Hyperparameters,
    _HessianGPT,
    CONTROL_TENSOR_NAME_PATTERNS,
    _classify_param,
    _rebank_state_dict,
    _unbank_state_dict,
    build_sentencepiece_luts,
    collect_hessians_from_tokens,
    dequantize_mixed_int6,
    eval_val,
    eval_val_sliding_ttt,
    generate_autoregressive_calib,
    load_validation_tokens,
    quantize_float_tensor,
    quantize_int6_gptq,
    quantize_int6_per_row,
    restore_low_dim_params_to_fp32,
)
import train_gpt as _tg  # noqa: E402


def _log(msg: str, *args, **kwargs) -> None:
    print(msg, flush=True)


def _parse_int7_patterns() -> list[str]:
    raw = os.environ.get("INT7_PATTERNS", "blocks.0.,blocks.10.,mlp.proj")
    return [p.strip() for p in raw.split(",") if p.strip()]


def _is_int7_promoted(name: str, patterns: list[str]) -> bool:
    return any(p in name for p in patterns)


def mixed_quantize_with_int7(
    state_dict: dict[str, torch.Tensor],
    int6_cats: set[str],
    hessians: dict[str, torch.Tensor] | None,
    int7_patterns: list[str],
) -> tuple[dict[str, torch.Tensor], dict[str, object], dict[str, str]]:
    """Variant of train_gpt.mixed_quantize_int6 that promotes named tensors to int7.

    Returns (result, meta, per_tensor_precision) where per_tensor_precision
    is a {name: "int6"|"int7"|"int8"|"passthrough"|"passthrough_ctrl"} dict
    used for logging / debugging.
    """
    result: dict[str, torch.Tensor] = {}
    meta: dict[str, object] = {}
    precision: dict[str, str] = {}

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)

        # Small / non-float → passthrough.
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            precision[name] = "passthrough"
            continue

        # Control tensors → passthrough_ctrl (float32).
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            precision[name] = "passthrough_ctrl"
            continue

        if cat in int6_cats and t.ndim >= 1:
            # Choose precision
            if _is_int7_promoted(name, int7_patterns):
                cr = 63
                type_name = "int7"
            else:
                cr = 31
                type_name = "int6"

            H = hessians.get(name) if hessians else None
            if H is not None:
                q, s = quantize_int6_gptq(t, hessian=H, clip_range=cr)
            else:
                q, s = quantize_int6_per_row(t, clip_range=cr)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": type_name}
            precision[name] = type_name
        else:
            # int8 for embed / other 2D tensors
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
            precision[name] = "int8"

    return result, meta, precision


def selective_prune_to_budget(
    quant_result: dict[str, torch.Tensor],
    quant_meta: dict[str, object],
    target_mb: float,
    code_bytes_est: int,
    log0=_log,
) -> dict[str, torch.Tensor]:
    """Mirror of the selective +/-1 pruning pass in train_gpt.py:2294-2337.

    Only prunes entries tagged as {"type": "int6"} (not int7 or int8). Walks
    the +/-1 candidates in order of smallest projected reconstruction error
    first. Returns the (possibly-pruned) quant_result.
    """
    ones_info: list[tuple[str, int, float]] = []
    for name, info in quant_meta.items():
        if not (isinstance(info, dict) and info.get("type") == "int6"):
            continue
        qk, sk = name + ".q", name + ".scale"
        if qk not in quant_result or sk not in quant_result:
            continue
        q, s = quant_result[qk], quant_result[sk]
        if s.ndim > 0:
            ones_mask = (q.abs() == 1)
            if ones_mask.any():
                row_idx = torch.arange(q.shape[0]).unsqueeze(1).expand_as(q)[ones_mask]
                flat_idx = torch.arange(q.numel()).reshape(q.shape)[ones_mask]
                errors = s.float()[row_idx].pow(2)
                for fi, err in zip(flat_idx.tolist(), errors.tolist()):
                    ones_info.append((qk, fi, err))
    if not ones_info:
        return quant_result
    ones_info.sort(key=lambda x: x[2])

    def _try_prune(n: int) -> tuple[int, dict[str, torch.Tensor]]:
        tmp = {k: v.clone() for k, v in quant_result.items()}
        for i in range(min(n, len(ones_info))):
            tmp[ones_info[i][0]].view(-1)[ones_info[i][1]] = 0
        buf = io.BytesIO()
        torch.save({"w": tmp, "m": quant_meta}, buf)
        return len(lzma.compress(buf.getvalue(), preset=9)) + code_bytes_est, tmp

    no_sz, _ = _try_prune(0)
    target_bytes = int(target_mb * 1024 * 1024)
    log0(
        f"selective_prune: {len(ones_info)} +/-1 candidates, "
        f"unpruned={no_sz/(1024*1024):.2f}MB target={target_mb}MB"
    )
    if no_sz <= target_bytes:
        log0("selective_prune: already fits, no pruning needed")
        return quant_result
    full_sz, _ = _try_prune(len(ones_info))
    log0(f"selective_prune: full +/-1 prune={full_sz/(1024*1024):.2f}MB")
    if full_sz > target_bytes:
        log0("selective_prune: even full prune not enough, applying all")
        _, result = _try_prune(len(ones_info))
        return result
    lo, hi = 0, len(ones_info)
    while lo < hi:
        mid = (lo + hi) // 2
        sz, _ = _try_prune(mid)
        if sz <= target_bytes:
            hi = mid
        else:
            lo = mid + 1
    log0(
        f"selective_prune: pruning {lo}/{len(ones_info)} +/-1 values "
        f"({100*lo/len(ones_info):.1f}%) to fit {target_mb}MB"
    )
    _, result = _try_prune(lo)
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for requant (hessian collection + quantize)")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    args = Hyperparameters()
    model_pt = Path(os.environ.get("MODEL_PT_PATH", "./final_model.pt")).expanduser().resolve()
    output_ptz = Path(os.environ.get("OUTPUT_PTZ_PATH", "./final_model.mixed.ptz")).expanduser().resolve()
    int7_patterns = _parse_int7_patterns()
    target_mb = float(os.environ.get("TARGET_MB", "15.9"))
    skip_ttt = os.environ.get("SKIP_TTT", "0") == "1"
    ttt_qat = bool(int(os.environ.get("TTT_QAT", "1")))

    _log(f"=== requant_mixed_precision: starting ===")
    _log(f"imported train_gpt from: {Path(_tg.__file__).resolve()}")
    _log(f"MODEL_PT_PATH:  {model_pt}")
    _log(f"OUTPUT_PTZ_PATH: {output_ptz}")
    _log(f"INT7_PATTERNS:  {int7_patterns}")
    _log(f"TARGET_MB:      {target_mb}")
    _log(f"SKIP_TTT:       {skip_ttt}")
    _log(f"TTT_QAT:        {ttt_qat}")
    if not model_pt.exists():
        raise FileNotFoundError(f"MODEL_PT_PATH does not exist: {model_pt}")

    # --- Tokenizer + val data + LUTs (needed for eval_val / TTT later) ---
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer "
            f"vocab_size={int(sp.vocab_size())}"
        )
    effective_eval_seq_len = (
        args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    )
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        build_sentencepiece_luts(sp, args.vocab_size, device)
    )
    _log(f"val_tokens:{val_tokens.numel() - 1}")

    # --- Construct base_model (GPT) and load the float checkpoint ---
    # QAT off for the requant / calibration pipeline (matches train_gpt.py:1970 init).
    CastedLinear._qat_enabled = False
    base_model = GPT(
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
        mtp_num_heads=0,
        mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        gated_attention=args.gated_attention,
        value_residual=args.value_residual,
    ).to(device).bfloat16()
    base_model._has_leading_space = has_leading_space_lut
    base_model.qo_bank.data = base_model.qo_bank.data.float()
    base_model.kv_bank.data = base_model.kv_bank.data.float()
    base_model.mlp_up_bank.data = base_model.mlp_up_bank.data.float()
    base_model.mlp_down_bank.data = base_model.mlp_down_bank.data.float()
    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(base_model)

    _log(f"loading float checkpoint from {model_pt}")
    sd_float = torch.load(model_pt, map_location="cpu")
    missing, unexpected = base_model.load_state_dict(sd_float, strict=False)
    # meta_sgd_* are allowed to be missing (they're init'd to 1.0 and don't affect
    # the forward; they were filtered out of export_sd in train_gpt.py).
    unexpected_filtered = [k for k in missing if not k.startswith("meta_sgd_")]
    if unexpected_filtered:
        _log(f"WARN missing keys in base_model load: {unexpected_filtered}")
    if unexpected:
        _log(f"WARN unexpected keys in base_model load: {unexpected}")

    # --- Unbank sd for hessian model + quantize input ---
    sd_cpu = {k: v.detach().cpu() for k, v in sd_float.items()}
    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)
    _log(f"unbanked_sd: {len(unbanked_sd)} keys")

    # --- Build hessian_model and load unbanked weights ---
    _log("building hessian_model (_HessianGPT) and loading unbanked weights...")
    hessian_model = _HessianGPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    hessian_model._has_leading_space = has_leading_space_lut
    for m in hessian_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(hessian_model)
    hessian_model.load_state_dict(
        {k: v.to(device) for k, v in unbanked_sd.items() if k in hessian_model.state_dict()},
        strict=False,
    )

    # --- Generate AR calibration data with base_model, collect hessians ---
    _log("generating autoregressive calibration data (64 seqs x 2048 tokens, temp=0.8)...")
    t_gen = time.perf_counter()
    base_model.eval()
    ar_tokens = generate_autoregressive_calib(
        base_model, device,
        num_seqs=64, seq_len=args.train_seq_len,
        vocab_size=args.vocab_size, temperature=0.8, batch_size=8, seed=args.seed,
    )
    _log(f"generated {len(ar_tokens)} sequences in {time.perf_counter()-t_gen:.1f}s")

    _log("collecting hessians from AR tokens...")
    t_h = time.perf_counter()
    hessians = collect_hessians_from_tokens(hessian_model, ar_tokens, device)
    _log(f"collected hessians for {len(hessians)} layers in {time.perf_counter()-t_h:.1f}s")
    del ar_tokens, hessian_model
    torch.cuda.empty_cache()

    # --- Quantize with mixed int6/int7 precision ---
    _log("quantizing with mixed int6/int7 precision...")
    t_q = time.perf_counter()
    quant_result, quant_meta, precision = mixed_quantize_with_int7(
        unbanked_sd, {"mlp", "attn"}, hessians=hessians, int7_patterns=int7_patterns
    )
    _log(f"quantize done in {time.perf_counter()-t_q:.1f}s")

    # Log which tensors got promoted
    promoted = sorted([k for k, p in precision.items() if p == "int7"])
    int6_count = sum(1 for p in precision.values() if p == "int6")
    int8_count = sum(1 for p in precision.values() if p == "int8")
    _log(f"precision breakdown: int7={len(promoted)} int6={int6_count} int8={int8_count} "
         f"passthrough={sum(1 for p in precision.values() if 'passthrough' in p)}")
    _log("int7 promoted tensors:")
    for name in promoted:
        t = unbanked_sd[name]
        _log(f"  {name:<45s} shape={tuple(t.shape)}  numel={t.numel()}")

    # --- Read current code size for selective_prune budget calculation ---
    code = (TRAIN_GPT_DIR / "train_gpt.py").read_text(encoding="utf-8")
    code_bytes_est = len(code.encode("utf-8"))
    _log(f"code_bytes_est: {code_bytes_est}")

    # --- Selective prune to fit budget ---
    quant_result = selective_prune_to_budget(
        quant_result, quant_meta, target_mb, code_bytes_est
    )

    # --- Save as LZMA-compressed artifact ---
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = lzma.compress(quant_raw, preset=9)
    with open(output_ptz, "wb") as f:
        f.write(quant_blob)
    artifact_bytes = len(quant_blob)
    total_bytes = artifact_bytes + code_bytes_est
    _log(f"Serialized mixed int6/int7+lzma: {artifact_bytes} bytes "
         f"({artifact_bytes/(1024*1024):.3f} MB)")
    _log(f"Total submission size (ptz + code): {total_bytes} bytes "
         f"({total_bytes/(1024*1024):.3f} MB)")
    budget_bytes = 16 * 1024 * 1024
    _log(f"Headroom to 16 MB: {budget_bytes - total_bytes} bytes "
         f"({(budget_bytes - total_bytes)/1024:.1f} KB)")
    if total_bytes > budget_bytes:
        _log("WARN: exceeds 16 MB budget!")

    # --- Round-trip: read back, dequantize ---
    _log("round-tripping: reloading ptz from disk and dequantizing...")
    with open(output_ptz, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob_disk)), map_location="cpu")
    deq_unbanked = dequantize_mixed_int6(
        quant_state["w"], quant_state["m"], unbanked_sd
    )
    deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)
    # Re-inject meta_sgd_* (they're excluded from export, but eval_model needs them)
    for k in ("meta_sgd_qo", "meta_sgd_kv", "meta_sgd_up", "meta_sgd_down"):
        if k not in deq_state and hasattr(base_model, k):
            deq_state[k] = getattr(base_model, k).detach().cpu().clone()

    # --- Construct a fresh eval_model and load the dequantized state ---
    CastedLinear._qat_enabled = ttt_qat
    _log(f"CastedLinear._qat_enabled for eval/TTT: {CastedLinear._qat_enabled}")
    eval_model = GPT(
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
        mtp_num_heads=0,
        mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        gated_attention=args.gated_attention,
        value_residual=args.value_residual,
    ).to(device).bfloat16()
    eval_model._has_leading_space = has_leading_space_lut
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    missing2, unexpected2 = eval_model.load_state_dict(deq_state, strict=False)
    if missing2:
        _log(f"WARN missing keys in eval_model load: {missing2}")
    if unexpected2:
        _log(f"WARN unexpected keys in eval_model load: {unexpected2}")

    # --- Baseline eval (compiled) ---
    _log("running baseline eval_val on dequantized mixed-precision model...")
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    base_loss, base_bpb = eval_val(
        args, compiled_eval, rank=0, world_size=1, device=device,
        grad_accum_steps=1,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    baseline_ms = 1000.0 * (time.perf_counter() - t0)
    _log(f"final_mixed_roundtrip val_loss:{base_loss:.4f} val_bpb:{base_bpb:.4f} "
         f"eval_time:{baseline_ms:.0f}ms")
    _log(f"final_mixed_roundtrip_exact val_loss:{base_loss:.8f} val_bpb:{base_bpb:.8f}")

    if skip_ttt or not args.ttt_enabled or args.eval_stride <= 0:
        _log("SKIP_TTT set or TTT disabled; stopping after baseline.")
        return

    # --- TTT eval ---
    # Reset model weights to the dequantized starting point before TTT
    eval_model.load_state_dict(deq_state, strict=False)
    _log("=" * 60)
    _log("STARTING TTT (Test-Time Training) on mixed-precision model")
    _log("=" * 60)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ttt_loss, ttt_bpb = eval_val_sliding_ttt(
        args, eval_model, rank=0, world_size=1, device=device,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        stride=args.eval_stride,
        log0=_log,
    )
    torch.cuda.synchronize()
    ttt_ms = 1000.0 * (time.perf_counter() - t0)
    _log(f"mixed_legal_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
         f"eval_time:{ttt_ms:.0f}ms")
    _log(f"mixed_legal_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")

    delta = base_bpb - ttt_bpb
    _log("")
    _log("=" * 60)
    _log("MIXED-PRECISION REQUANT + TTT SUMMARY")
    _log("=" * 60)
    _log(f"float_pt_path:      {model_pt}")
    _log(f"mixed_ptz_path:     {output_ptz}")
    _log(f"artifact_bytes:     {artifact_bytes}")
    _log(f"int7_promoted:      {len(promoted)} tensors")
    _log(f"baseline_bpb:       {base_bpb:.6f}")
    _log(f"ttt_bpb:            {ttt_bpb:.6f}")
    _log(f"delta_bpb:          {delta:+.6f}  (positive = TTT helped)")
    _log(f"baseline_time_ms:   {baseline_ms:.0f}")
    _log(f"ttt_time_ms:        {ttt_ms:.0f}")


if __name__ == "__main__":
    main()
