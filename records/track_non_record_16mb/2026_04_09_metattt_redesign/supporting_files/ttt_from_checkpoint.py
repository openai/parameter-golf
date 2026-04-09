#!/usr/bin/env python3
"""Run TTT on an already-saved model artifact.

Use this to re-evaluate a trained model under different TTT hyperparameters
without retraining. Supports both the full-precision float checkpoint
(`final_model.pt`) and the int6+LZMA artifact (`final_model.int6.ptz`).

Usage
-----
    # float checkpoint, default TTT knobs:
    MODEL_PATH=/workspace/parameter-golf/final_model.pt \
        python3 ttt_from_checkpoint.py

    # int6 artifact (the real competition submission), overrides some knobs:
    MODEL_PATH=/workspace/parameter-golf/final_model.int6.ptz \
        TTT_LR=0.006 TTT_EPOCHS=5 \
        python3 ttt_from_checkpoint.py

Environment variables
---------------------
All `TTT_*`, `META_TTT_*`, `EVAL_*`, `VAL_*`, and architecture env vars
understood by `train_gpt.py`'s Hyperparameters dataclass are respected here,
so you can feed the same run.sh-style environment block in and it just works.

Only additional var:
    MODEL_PATH       path to either final_model.pt or final_model.int6.ptz
                     (default: ./final_model.pt)

Outputs
-------
    baseline val_loss / val_bpb              — fresh model, no TTT
    TTT      val_loss / val_bpb              — same model after TTT
    delta    bpb_baseline - bpb_ttt          — positive = TTT helped

The script imports directly from train_gpt.py in the same directory, so it
stays byte-faithful to whatever version of GPT / eval_val / eval_val_sliding_ttt
was used during training.
"""
from __future__ import annotations

import io
import lzma
import os
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Import train_gpt.py. By default we import the sibling file next to this
# script, but TRAIN_GPT_DIR lets the caller point us at a specific version
# (e.g. the exp106 version in records/phase3/... even when running the
# script from /workspace/parameter-golf). This is important because the
# repo root may contain a DIFFERENT train_gpt.py without exp106's meta_sgd_*
# params — importing the wrong one will silently mismatch GPT.__init__
# and break strict state_dict load.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_GPT_DIR = Path(os.environ.get("TRAIN_GPT_DIR", str(SCRIPT_DIR))).resolve()
if not (TRAIN_GPT_DIR / "train_gpt.py").exists():
    raise FileNotFoundError(
        f"train_gpt.py not found in TRAIN_GPT_DIR={TRAIN_GPT_DIR}. "
        "Set TRAIN_GPT_DIR to the folder containing the exp106 train_gpt.py."
    )
# Put TRAIN_GPT_DIR FIRST on sys.path so it beats any other train_gpt.py
# that might be visible on the default Python path.
sys.path.insert(0, str(TRAIN_GPT_DIR))

import sentencepiece as spm  # noqa: E402
from train_gpt import (  # noqa: E402
    GPT,
    CastedLinear,
    Hyperparameters,
    build_sentencepiece_luts,
    dequantize_mixed_int6,
    eval_val,
    eval_val_sliding_ttt,
    load_validation_tokens,
    restore_low_dim_params_to_fp32,
    _rebank_state_dict,
    _unbank_state_dict,
)
import train_gpt as _train_gpt  # noqa: E402
_log_module_path = Path(_train_gpt.__file__).resolve()


def _log(msg: str, *args, **kwargs) -> None:
    """Logging shim matching train_gpt.py's log0 signature (accepts optional
    console= and flush= kwargs; we ignore them and always flush to stdout)."""
    print(msg, flush=True)


def _resolve_model_path() -> Path:
    # Default to the int6+lzma artifact because that's what the canonical
    # train_gpt.py main() eval path uses (train_gpt.py:2349-2396): round-trips
    # final_model.int6.ptz → dequantize_mixed_int6 → eval_model.load_state_dict
    # → eval_val_sliding_ttt. The float final_model.pt is only an intermediate
    # debugging artifact / GPTQ calib source in the canonical flow; it is NOT
    # what "legal_ttt" is measured on. Set MODEL_PATH explicitly to .pt for the
    # non-canonical float-path TTT.
    env_path = os.environ.get("MODEL_PATH", "./final_model.int6.ptz")
    p = Path(env_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {p}")
    return p


def _load_state_dict(
    path: Path,
    fresh_model: GPT,
    num_layers: int,
) -> dict[str, torch.Tensor]:
    """Load a state_dict from either a .pt or .int6.ptz artifact.

    For .pt: plain torch.load, no post-processing.
    For .int6.ptz: mirror the exact dequant path in train_gpt.py main()
    (lines 2349-2352) — LZMA decompress, torch.load bytes, build an
    unbanked template from the fresh_model, dequantize, then rebank.
    """
    name = path.name
    if name.endswith(".pt"):
        _log(f"loading float checkpoint from {path}")
        sd = torch.load(path, map_location="cpu")
        return sd

    if name.endswith(".int6.ptz"):
        _log(f"loading int6+lzma checkpoint from {path}")
        with open(path, "rb") as f:
            blob = f.read()
        quant_state = torch.load(io.BytesIO(lzma.decompress(blob)), map_location="cpu")
        # Build the unbanked template from a fresh GPT's cpu state_dict, dropping
        # train-only params that were filtered out during export (meta_sgd_* and
        # mtp_heads). This mirrors the export → sd_cpu → _unbank_state_dict path.
        raw_sd = {
            k: v.detach().cpu()
            for k, v in fresh_model.state_dict().items()
            if not (k.startswith("meta_sgd_") or "mtp_heads" in k)
        }
        unbanked_sd = _unbank_state_dict(raw_sd, num_layers)
        deq_unbanked = dequantize_mixed_int6(
            quant_state["w"], quant_state["m"], unbanked_sd
        )
        deq_state = _rebank_state_dict(deq_unbanked, num_layers, raw_sd)
        return deq_state

    raise ValueError(
        f"Unsupported model file extension on {path}. "
        "Expected .pt or .int6.ptz."
    )


def _inject_train_only_params(
    sd: dict[str, torch.Tensor], fresh_model: GPT
) -> dict[str, torch.Tensor]:
    """Re-inject meta_sgd_* scales that were filtered out of final_model.pt
    or final_model.int6.ptz. These are train-time-only params (only used
    in meta_ttt_step's inner-SGD update) so they never influence the eval
    forward pass — but the GPT module has them as nn.Parameters, so
    strict=True load requires them present. Source from the fresh_model
    (init value = 1.0 everywhere) since we don't have the learned values
    at inference time.
    """
    fresh_sd = fresh_model.state_dict()
    for k in ("meta_sgd_qo", "meta_sgd_kv", "meta_sgd_up", "meta_sgd_down"):
        if k not in sd and k in fresh_sd:
            sd[k] = fresh_sd[k].detach().cpu().clone()
    return sd


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for TTT eval")
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

    _log(f"=== ttt_from_checkpoint: starting ===")
    _log(f"imported train_gpt from: {_log_module_path}")
    _log(f"eval_seq_len:{args.eval_seq_len} train_seq_len:{args.train_seq_len}")
    _log(f"ttt_enabled:{args.ttt_enabled} ttt_lr:{args.ttt_lr} "
         f"ttt_epochs:{args.ttt_epochs} ttt_chunk_tokens:{args.ttt_chunk_tokens} "
         f"ttt_freeze_blocks:{args.ttt_freeze_blocks} ttt_momentum:{args.ttt_momentum}")
    _log(f"eval_stride:{args.eval_stride} eval_batch_seqs:{args.eval_batch_seqs}")

    # --- Tokenizer + val data + LUTs ---
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(
            f"Only SentencePiece .model tokenizers supported, got {args.tokenizer_path}"
        )
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

    # --- Construct a fresh GPT with the same config as training ---
    # QAT flag: in the canonical train_gpt.py main() flow, late_qat flips
    # CastedLinear._qat_enabled → True during warmdown (around step 5110 for
    # exp106) and it stays True through the eval phase. eval_model inherits
    # True because the class-level flag is never reset. To replicate the
    # canonical eval+TTT numerics exactly we must set it True here too.
    # Override with TTT_QAT=0 env var to run the non-QAT path (for A/B tests).
    CastedLinear._qat_enabled = bool(int(os.environ.get("TTT_QAT", "1")))
    _log(f"CastedLinear._qat_enabled: {CastedLinear._qat_enabled}")
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
        # Inference/TTT: no MTP heads, matches the eval_model construction
        # in train_gpt.py main() at line 2358.
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
    model._has_leading_space = has_leading_space_lut
    model.qo_bank.data = model.qo_bank.data.float()
    model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model)

    # --- Load weights ---
    model_path = _resolve_model_path()
    sd = _load_state_dict(model_path, model, args.num_layers)
    sd = _inject_train_only_params(sd, model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        _log(f"WARN: missing keys in state_dict: {missing}")
    if unexpected:
        _log(f"WARN: unexpected keys in state_dict: {unexpected}")
    if not missing and not unexpected:
        _log("state_dict loaded cleanly (no missing/unexpected keys)")

    # --- Baseline val_bpb (no TTT) ---
    # Pass a compile wrapper to mirror train_gpt.py's eval_val invocation.
    compiled_eval = torch.compile(model, dynamic=False, fullgraph=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    base_loss, base_bpb = eval_val(
        args,
        compiled_eval,
        rank=0,
        world_size=1,
        device=device,
        grad_accum_steps=1,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    baseline_ms = 1000.0 * (time.perf_counter() - t0)
    _log(
        f"baseline val_loss:{base_loss:.4f} val_bpb:{base_bpb:.4f} "
        f"eval_time:{baseline_ms:.0f}ms"
    )
    _log(f"baseline_exact val_loss:{base_loss:.8f} val_bpb:{base_bpb:.8f}")

    if not args.ttt_enabled or args.eval_stride <= 0:
        _log("TTT disabled (ttt_enabled=0 or eval_stride<=0); stopping after baseline.")
        return

    # --- TTT eval ---
    # eval_val_sliding_ttt mutates model weights via SGD during the inner loop,
    # so reload the state_dict to the original starting point first.
    model.load_state_dict(sd, strict=False)

    _log("=" * 60)
    _log("STARTING TTT (Test-Time Training)")
    _log("=" * 60)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ttt_loss, ttt_bpb = eval_val_sliding_ttt(
        args,
        model,
        rank=0,
        world_size=1,
        device=device,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        stride=args.eval_stride,
        log0=_log,
    )
    torch.cuda.synchronize()
    ttt_ms = 1000.0 * (time.perf_counter() - t0)
    _log(
        f"ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
        f"eval_time:{ttt_ms:.0f}ms"
    )
    _log(f"ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")

    # --- Summary ---
    delta_bpb = base_bpb - ttt_bpb
    _log("")
    _log("=" * 60)
    _log("TTT SUMMARY")
    _log("=" * 60)
    _log(f"model:           {model_path}")
    _log(f"baseline_bpb:    {base_bpb:.6f}")
    _log(f"ttt_bpb:         {ttt_bpb:.6f}")
    _log(f"delta_bpb:       {delta_bpb:+.6f}  (positive = TTT helped)")
    _log(f"baseline_time_ms:{baseline_ms:.0f}")
    _log(f"ttt_time_ms:     {ttt_ms:.0f}")


if __name__ == "__main__":
    main()
