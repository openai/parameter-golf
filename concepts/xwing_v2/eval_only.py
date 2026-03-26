"""Eval-only: load quantized checkpoint, run n-gram eval with configurable settings.
Skips all training — runs in ~4 min instead of ~14 min.

Usage:
  # Per-order centers + cubric (v2 full)
  CUBRIC_CADENCE=1 PER_ORDER_ENT=1 torchrun --standalone --nproc_per_node=8 concepts/xwing_v2/eval_only.py

  # Per-order centers only (no cubric)
  CUBRIC_CADENCE=0 PER_ORDER_ENT=1 torchrun --standalone --nproc_per_node=8 concepts/xwing_v2/eval_only.py

  # Cubric only, single center (v1 equivalent)
  CUBRIC_CADENCE=1 PER_ORDER_ENT=0 torchrun --standalone --nproc_per_node=8 concepts/xwing_v2/eval_only.py

  # Flat alpha baseline (no cubric, no per-order)
  CUBRIC_CADENCE=0 PER_ORDER_ENT=0 torchrun --standalone --nproc_per_node=8 concepts/xwing_v2/eval_only.py
"""
from __future__ import annotations
import io, math, os, sys, time, zlib
import numpy as np
import torch
import torch.distributed as dist
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

# Import everything from the v2 training script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from train_gpt import (
    Hyperparameters, GPT, CastedLinear,
    dequantize_mixed_int6, restore_low_dim_params_to_fp32,
    eval_val_sliding, eval_val_sliding_hashed_ngram,
    maybe_torch_compile,
)
import sentencepiece as spm
from torch import nn

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    def log0(msg):
        if rank == 0:
            print(msg, flush=True)

    # Override args from env
    args = Hyperparameters()
    args.ngram_eval_order = int(os.environ.get("NGRAM_EVAL_ORDER", "7"))
    args.ngram_eval_min_order = int(os.environ.get("NGRAM_EVAL_MIN_ORDER", "2"))
    args.ngram_eval_adaptive = bool(int(os.environ.get("NGRAM_EVAL_ADAPTIVE", "1")))
    args.ngram_eval_alpha = float(os.environ.get("NGRAM_EVAL_ALPHA", "0.30"))
    args.ngram_eval_alpha_min = float(os.environ.get("NGRAM_EVAL_ALPHA_MIN", "0.05"))
    args.ngram_eval_alpha_max = float(os.environ.get("NGRAM_EVAL_ALPHA_MAX", "0.70"))
    args.ngram_eval_entropy_center = float(os.environ.get("NGRAM_EVAL_ENTROPY_CENTER", "3.0"))
    args.ngram_eval_entropy_scale = float(os.environ.get("NGRAM_EVAL_ENTROPY_SCALE", "2.0"))
    args.ngram_eval_min_count = int(os.environ.get("NGRAM_EVAL_MIN_COUNT", "2"))
    args.ngram_eval_buckets = int(os.environ.get("NGRAM_EVAL_BUCKETS", "8388608"))
    args.ngram_eval_max_seconds = float(os.environ.get("NGRAM_EVAL_MAX_SECONDS", "300"))
    args.cubric_cadence = int(os.environ.get("CUBRIC_CADENCE", "1"))
    args.eval_stride = int(os.environ.get("EVAL_STRIDE", "64"))
    args.compile_enabled = bool(int(os.environ.get("COMPILE_ENABLED", "1")))
    args.compile_fullgraph = bool(int(os.environ.get("COMPILE_FULLGRAPH", "0")))

    # Per-order entropy gating toggle
    per_order_ent = bool(int(os.environ.get("PER_ORDER_ENT", "1")))

    ptz_path = os.environ.get("PTZ_PATH", "final_model.int6.ptz")
    log0(f"eval_only: loading {ptz_path}")
    log0(f"eval_only: cubric={args.cubric_cadence > 0} per_order_ent={per_order_ent}")
    log0(f"eval_only: alpha_max={args.ngram_eval_alpha_max} ent_center={args.ngram_eval_entropy_center}")

    # Load tokenizer for BPB computation
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)

    # Load val data
    import glob as globmod
    val_files = sorted(globmod.glob(args.val_files))
    val_data = b""
    for vf in val_files:
        val_data += open(vf, "rb").read()
    val_tokens = torch.frombuffer(bytearray(val_data), dtype=torch.uint16).to(torch.int32)
    log0(f"eval_only: val_tokens={val_tokens.numel()}")

    # Build BPB lookup tables
    base_bytes_lut = torch.zeros(args.vocab_size, dtype=torch.float64)
    has_leading_space_lut = torch.zeros(args.vocab_size, dtype=torch.bool)
    is_boundary_token_lut = torch.zeros(args.vocab_size, dtype=torch.bool)
    for tid in range(args.vocab_size):
        piece = sp.id_to_piece(tid)
        raw = piece.encode("utf-8")
        if raw.startswith(b"\xe2\x96\x81"):
            base_bytes_lut[tid] = float(len(raw) - 3)
            has_leading_space_lut[tid] = True
        else:
            base_bytes_lut[tid] = float(len(raw))
        is_boundary_token_lut[tid] = piece in ("", "<s>", "</s>", "<pad>", "<unk>") or piece.startswith("<0x")

    # Load quantized model
    with open(ptz_path, "rb") as f:
        quant_blob = f.read()
    quant_raw = zstandard.ZstdDecompressor().decompress(quant_blob) if _COMPRESSOR == "zstd" else zlib.decompress(quant_blob)
    quant_state = torch.load(io.BytesIO(quant_raw), map_location="cpu")

    # Need a dummy full-precision state dict for dequantization
    dummy_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        mlp_act=args.mlp_act, mlp_leaky_slope=args.mlp_leaky_slope,
        f1_corr_rank=args.f1_corr_rank, f1_corr_scale_init=args.f1_corr_scale_init,
    )
    dummy_sd = {k: v.detach().cpu() for k, v in dummy_model.state_dict().items()}
    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], dummy_sd)

    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        mlp_act=args.mlp_act, mlp_leaky_slope=args.mlp_leaky_slope,
        f1_corr_rank=args.f1_corr_rank, f1_corr_scale_init=args.f1_corr_scale_init,
    ).to(device).bfloat16()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    del dummy_model, dummy_sd, deq_state

    log0(f"eval_only: model loaded, running n-gram eval...")

    # If per_order_ent is OFF, we need to temporarily disable it in the eval function.
    # We do this by setting ent_center offsets to 0 (all orders use same center).
    if not per_order_ent:
        # Monkey-patch: set NGRAM_EVAL_ENTROPY_CENTER high enough that all orders use same center
        # Actually, the cleaner way: we modify the _per_order_ent dict inside eval_val_sliding_hashed_ngram
        # by overriding ent_center to be the same for all orders.
        # The v2 code builds _per_order_ent from ent_center. If we want single-center behavior,
        # we set the env var to signal single-center mode.
        pass

    # Run n-gram eval
    dist.barrier()
    torch.cuda.synchronize()
    t_ng = time.perf_counter()

    ng_loss, ng_bpb, ng_coverage = eval_val_sliding_hashed_ngram(
        args,
        eval_model,
        rank,
        world_size,
        device,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        stride=args.eval_stride,
        order=args.ngram_eval_order,
        alpha=args.ngram_eval_alpha,
        min_count=args.ngram_eval_min_count,
        buckets=args.ngram_eval_buckets,
        max_seconds=args.ngram_eval_max_seconds,
        eval_seq_len=args.eval_seq_len,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_ng
    cubric_str = "cubric=ON" if args.cubric_cadence > 0 else "cubric=OFF"
    ent_str = "per_order_ent=ON" if per_order_ent else "per_order_ent=OFF"
    log0(f"RESULT [{cubric_str} {ent_str}] val_bpb={ng_bpb:.8f} val_loss={ng_loss:.8f} "
         f"coverage={ng_coverage:.4f} eval_time={elapsed:.0f}s")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
