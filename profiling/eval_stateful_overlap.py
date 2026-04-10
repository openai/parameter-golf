"""
Standalone stateful-overlap eval for an existing INT6 checkpoint.

Supports both single-GPU and multi-GPU (torchrun) to match training eval.

Usage:
    # 1 GPU:
    CHECKPOINT=final_model.int6.ptz python3 profiling/eval_stateful_overlap.py
    # 8 GPU (matches training eval):
    CHECKPOINT=final_model.int6.ptz torchrun --nproc_per_node=8 profiling/eval_stateful_overlap.py
"""

import os, sys, io, lzma, glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for k, v in {
    "FP16_INPROJ_ROWS": "0", "WARMDOWN_ITERS": "2600", "WARMDOWN_SHAPE": "linear",
    "MUON_EQ_R": "1", "LATE_QAT_THRESHOLD": "0.15", "USE_GPTQ": "1",
    "QUANT_BITS": "6", "USE_LZMA": "1", "EVAL_TEMP": "0.9",
    "WEIGHT_DECAY": "0.04", "MUON_MOMENTUM": "0.99", "MATRIX_LR": "0.025",
    "EVAL_OVERLAP": "1024",
}.items():
    os.environ.setdefault(k, v)

import torch
import torch.distributed as dist
import sentencepiece as spm

from train_mamba3_hybrid import (
    Hyperparameters, GPT,
    build_sentencepiece_luts, load_validation_tokens,
    dequantize_state_dict_int8, eval_val_stateful_overlap,
)

# --- Distributed setup ---
distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
if distributed:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    rank, world_size, local_rank = 0, 1, 0
    device = torch.device("cuda")

def log0(msg):
    if rank == 0:
        print(msg)

args = Hyperparameters()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
log0(f"Tokenizer: {args.tokenizer_path}  vocab={int(sp.vocab_size())}")

val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
log0(f"Val tokens: {val_tokens.numel():,}")

base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
    sp, args.vocab_size, device
)

base_model = GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap,
    use_smeargate=args.use_smeargate, use_bigram_hash=args.use_bigram_hash,
    bigram_buckets=args.bigram_buckets, bigram_hash_dim=args.bigram_hash_dim,
    use_ortho_init=args.use_ortho_init,
    mamba3_d_state=args.mamba3_d_state, mamba3_expand=args.mamba3_expand,
    mamba3_headdim=args.mamba3_headdim, mamba3_chunk_size=args.mamba3_chunk_size,
    mamba3_ngroups=args.mamba3_ngroups, mamba3_rope_fraction=args.mamba3_rope_fraction,
    mamba3_outproj_norm=args.mamba3_outproj_norm,
    num_attn_layers=args.num_attn_layers, num_heads=args.num_heads,
    num_kv_heads=args.num_kv_heads, rope_base=args.rope_base,
    qk_gain_init=args.qk_gain_init,
    ve_enabled=args.ve_enabled, ve_dim=args.ve_dim,
).to(device).bfloat16()
log0(f"Built model: {sum(p.numel() for p in base_model.parameters())/1e6:.1f}M params")

ckpt_path = os.environ.get("CHECKPOINT", f"final_model.int{args.quant_bits}.ptz")
if not os.path.exists(ckpt_path):
    candidates = sorted(glob.glob("final_model*.ptz"))
    raise FileNotFoundError(f"CHECKPOINT={ckpt_path} not found. Available: {candidates}")
log0(f"Loading checkpoint: {ckpt_path}  ({os.path.getsize(ckpt_path):,} bytes)")
with open(ckpt_path, "rb") as f:
    blob = f.read()
decompressed = lzma.decompress(blob) if args.use_lzma else blob
state = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=False)
base_model.load_state_dict(dequantize_state_dict_int8(state), strict=True)
base_model.eval()

log0(f"Running stateful-overlap eval: overlap={args.eval_overlap}, "
     f"seq_len={args.train_seq_len}, eval_temp={args.eval_temp}, "
     f"rank={rank}, world_size={world_size}")

val_loss, val_bpb = eval_val_stateful_overlap(
    args, base_model, rank=rank, world_size=world_size, device=device,
    val_tokens=val_tokens, base_bytes_lut=base_bytes_lut,
    has_leading_space_lut=has_leading_space_lut,
    is_boundary_token_lut=is_boundary_token_lut,
)

log0(f"\nFINAL  stateful_overlap  val_loss={val_loss:.6f}  val_bpb={val_bpb:.6f}")
log0(f"       baseline Run 4c sliding = 1.1501 bpb")
log0(f"       delta vs Run 4c = {1000.0 * (val_bpb - 1.1501):+.1f} mBPB")

if distributed:
    dist.destroy_process_group()
