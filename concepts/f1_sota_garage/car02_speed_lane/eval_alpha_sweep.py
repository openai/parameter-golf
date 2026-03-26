"""Eval-only alpha sweep — loads quantized checkpoint, runs n-gram eval at multiple alphas."""
import io, math, os, sys, time, zlib
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

# Import everything from the training script
sys.path.insert(0, os.path.dirname(__file__))
from train_gpt import (
    Hyperparameters, GPT, CastedLinear, Rotary,
    dequantize_mixed_int6, restore_low_dim_params_to_fp32,
    eval_val_sliding_hashed_ngram, maybe_torch_compile,
    load_validation_tokens, build_sentencepiece_luts,
)

def main():
    args = Hyperparameters()
    args.ngram_eval_order = int(os.environ.get("NGRAM_EVAL_ORDER", 5))
    args.ngram_eval_min_count = int(os.environ.get("NGRAM_EVAL_MIN_COUNT", 2))
    args.ngram_eval_buckets = int(os.environ.get("NGRAM_EVAL_BUCKETS", 4194304))
    args.ngram_eval_max_seconds = float(os.environ.get("NGRAM_EVAL_MAX_SECONDS", 300))
    args.mlp_act = os.environ.get("MLP_ACT", "leaky_relu_sq").lower()
    args.mlp_leaky_slope = float(os.environ.get("MLP_LEAKY_SLOPE", 0.5))
    args.xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
    args.bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 1536))
    args.rope_dims = int(os.environ.get("ROPE_DIMS", 24))
    args.compile_enabled = bool(int(os.environ.get("COMPILE_ENABLED", "1")))
    args.compile_fullgraph = bool(int(os.environ.get("COMPILE_FULLGRAPH", "0")))

    ptz_path = os.environ.get("PTZ_PATH", "final_model.int6.ptz")
    alphas_str = os.environ.get("ALPHAS", "0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    alphas = [float(a) for a in alphas_str.split(",")]

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"Loading {ptz_path}...", flush=True)

    with open(ptz_path, "rb") as f:
        blob = f.read()
    raw = zstandard.ZstdDecompressor().decompress(blob) if _COMPRESSOR == "zstd" else zlib.decompress(blob)
    quant_state = torch.load(io.BytesIO(raw), map_location="cpu")

    # Need a dummy full state dict for dequantization
    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        dtg=args.dtg_enabled, ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        mlp_act=args.mlp_act, mlp_leaky_slope=args.mlp_leaky_slope,
        f1_corr_rank=args.f1_corr_rank, f1_corr_scale_init=args.f1_corr_scale_init,
    ).to(device).bfloat16()

    sd_cpu = {k: v.cpu() for k, v in eval_model.state_dict().items()}
    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    eval_model.eval()

    if rank == 0:
        print(f"Model loaded. Running alpha sweep: {alphas}", flush=True)

    # Load val data
    import sentencepiece as spm
    val_tokens = load_validation_tokens(args.val_files, args.eval_seq_len)
    sp = spm.SentencePieceProcessor(args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, args.vocab_size, device)

    for alpha in alphas:
        if distributed:
            dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        ng_loss, ng_bpb, ng_coverage = eval_val_sliding_hashed_ngram(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
            order=args.ngram_eval_order,
            alpha=alpha,
            min_count=args.ngram_eval_min_count,
            buckets=args.ngram_eval_buckets,
            max_seconds=args.ngram_eval_max_seconds,
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if rank == 0:
            tag = "FULL" if ng_coverage >= 0.999 else f"partial({ng_coverage:.2%})"
            print(f"alpha={alpha:.2f}  bpb={ng_bpb:.6f}  coverage={tag}  time={elapsed:.0f}s", flush=True)

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
