"""
Cheap validation: per-position cross-entropy on a trained checkpoint.

Question: does the model show a "cold-start cost" inside a single 4K window —
are tokens 0..~100 noticeably worse than tokens 1000+?  If yes, training with
SSM state carry would eliminate that per-window cold-start penalty and is
worth implementing.  If no, state-carry training is a weaker bet.

This runs on 1 GPU in ~30s.  It does NOT need the pod's full training setup
beyond `final_model*.ptz` and the val shard.

Usage (on pod, 1 GPU):
    CHECKPOINT=final_model_gptq_seed1337.ptz python3 profiling/per_position_loss.py
    CHECKPOINT=final_model.int6.ptz BSZ=8 NUM_BATCHES=64 python3 profiling/per_position_loss.py
"""

import os, sys, io, lzma, math, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import torch
import torch.nn.functional as F

# Match the env that produced Run 4c so the constructed model has the same shape
# as the checkpoint.  Override CHECKPOINT / BSZ / NUM_BATCHES via shell env.
for k, v in {
    "FP16_INPROJ_ROWS": "0", "WARMDOWN_ITERS": "2600", "WARMDOWN_SHAPE": "linear",
    "MUON_EQ_R": "1", "LATE_QAT_THRESHOLD": "0.15", "USE_GPTQ": "1",
    "QUANT_BITS": "6", "USE_LZMA": "1", "EVAL_TEMP": "0.9",
    "WEIGHT_DECAY": "0.04", "MUON_MOMENTUM": "0.99", "MATRIX_LR": "0.025",
}.items():
    os.environ.setdefault(k, v)

from train_mamba3_hybrid import (
    Hyperparameters, GPT, load_data_shard, dequantize_state_dict_int8,
)

args = Hyperparameters()
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- Build model (must match checkpoint shape) ---
model = GPT(
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
print(f"Built model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# --- Load the quantized checkpoint ---
ckpt_path = os.environ.get("CHECKPOINT", "final_model_gptq_seed1337.ptz")
if not os.path.exists(ckpt_path):
    candidates = sorted(glob.glob("final_model*.ptz"))
    raise FileNotFoundError(
        f"CHECKPOINT={ckpt_path} not found. Available: {candidates}"
    )
print(f"Loading checkpoint: {ckpt_path}")
with open(ckpt_path, "rb") as f:
    blob = f.read()
decompressed = lzma.decompress(blob)
state = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=False)
model.load_state_dict(dequantize_state_dict_int8(state), strict=True)
model.eval()

# --- Load val shard(s) ---
val_files = sorted(glob.glob(args.val_files))
if not val_files:
    raise FileNotFoundError(f"No val shards matched: {args.val_files}")
print(f"Loading val shard: {val_files[0]}")
val_tokens = load_data_shard(Path(val_files[0]))
print(f"Val tokens available: {val_tokens.numel():,}")

# --- Configure sweep ---
seq_len = args.train_seq_len  # 4096
bsz = int(os.environ.get("BSZ", "8"))
num_batches = int(os.environ.get("NUM_BATCHES", "64"))
total_windows = num_batches * bsz

max_start = val_tokens.numel() - 1 - seq_len
if max_start <= 0:
    raise RuntimeError(f"Val shard too small: {val_tokens.numel()} tokens")
# Non-overlapping windows spread across the shard
stride = max(1, max_start // max(1, total_windows))
starts = [min(i * stride, max_start) for i in range(total_windows)]
print(f"Windows: {total_windows} (bsz={bsz} x num_batches={num_batches}), "
      f"seq_len={seq_len}, stride between window starts={stride}")

# --- Per-position accumulator ---
pos_loss_sum = torch.zeros(seq_len, device=device, dtype=torch.float64)
pos_count = 0

with torch.inference_mode():
    for bi in range(num_batches):
        batch_starts = starts[bi * bsz : (bi + 1) * bsz]
        xs, ys = [], []
        for s in batch_starts:
            chunk = val_tokens[s : s + seq_len + 1].to(dtype=torch.int64)
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        x = torch.stack(xs).to(device=device, non_blocking=True)
        y = torch.stack(ys).to(device=device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward_logits(x)
        logits = logits.float()
        if args.eval_temp != 1.0:
            logits = logits / args.eval_temp
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="none",
        ).reshape(bsz, seq_len)
        pos_loss_sum += nll.sum(dim=0).to(torch.float64)
        pos_count += bsz
        if (bi + 1) % 10 == 0:
            print(f"  batch {bi+1}/{num_batches}")

# --- Report ---
pos_loss_nats = (pos_loss_sum / pos_count).cpu()
pos_bits = pos_loss_nats / math.log(2.0)  # bits/token (not bits/byte)

def region_mean(lo: int, hi: int) -> float:
    return pos_bits[lo:hi].mean().item()

baseline_region = (1000, 2000)
baseline_bits = region_mean(*baseline_region)

print(f"\nTotal scored windows: {pos_count}")
print(f"Baseline region [{baseline_region[0]},{baseline_region[1]}): "
      f"{baseline_bits:.4f} bits/token\n")

print("Per-position mean loss:")
print(f"{'pos':>6}  {'nats':>8}  {'bits/tok':>9}  {'Δ vs [1000,2000) (mBPT)':>26}")
for p in [0, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 4095]:
    if p < seq_len:
        delta = 1000.0 * (pos_bits[p].item() - baseline_bits)
        print(f"{p:>6d}  {pos_loss_nats[p].item():>8.4f}  "
              f"{pos_bits[p].item():>9.4f}  {delta:>+26.1f}")

print("\nRegion means (bits/token):")
print(f"{'range':>16}  {'bits/tok':>9}  {'Δ vs [1000,2000) (mBPT)':>26}")
for lo, hi in [
    (0, 50), (50, 100), (100, 250), (250, 500), (500, 1000),
    (1000, 2000), (2000, 3000), (3000, 4096),
]:
    m = region_mean(lo, hi)
    delta = 1000.0 * (m - baseline_bits)
    print(f"{f'[{lo:4d},{hi:4d})':>16}  {m:>9.4f}  {delta:>+26.1f}")

cold_cost = region_mean(0, 50) - region_mean(1000, 2000)
print(f"\nCold-start cost (region [0,50) minus [1000,2000)): "
      f"{1000.0*cold_cost:+.1f} mBPT")
if cold_cost > 0.005:
    print("=> Clear cold-start cost.  Training with SSM state carry is worth "
          "pursuing: warm state from the previous window would eliminate this "
          "per-window penalty.")
elif cold_cost > 0.002:
    print("=> Moderate cold-start cost.  State-carry training plausibly helps, "
          "but gain is bounded by how much of this is SSM-state vs "
          "attention-context driven.")
else:
    print("=> No meaningful cold-start cost.  The model does not currently "
          "undervalue early-window positions.  Training with state carry is a "
          "weaker bet; look elsewhere for the throughput/quality win.")
