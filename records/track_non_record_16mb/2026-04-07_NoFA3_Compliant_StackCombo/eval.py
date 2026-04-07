#!/usr/bin/env -S python3 -u
"""
SLOT eval with DDP support for 8xH100.
Splits windows across GPUs, each GPU runs SLOT independently, then all_reduce results.

Usage:
  # 8xH100
  torchrun --nproc_per_node=8 eval_slot_ddp.py

  # 1xH100 (for testing)
  python eval_slot_ddp.py

Config via environment variables:
  SLOT_STEPS=96  STRIDE=96  BATCH_SIZE=32
  MODEL_PATH=/workspace/models/model_bf16.pt
  DATA_DIR=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024
  TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
  VOCAB_SIZE=1024  DIM=512  LAYERS=11
"""
import sys, os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import sentencepiece as spm

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------
SLOT_STEPS = int(os.environ.get("SLOT_STEPS", "96"))
SLOT_LR = float(os.environ.get("SLOT_LR", "0.012"))
SLOT_LR_MIN = float(os.environ.get("SLOT_LR_MIN", "0.001"))
STRIDE = int(os.environ.get("STRIDE", "96"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/models/model_bf16.pt")
DATA_DIR = os.environ.get("DATA_DIR", "/workspace/parameter-golf/data/datasets/fineweb10B_sp1024")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model")
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", "1024"))
DIM = int(os.environ.get("DIM", "512"))
LAYERS = int(os.environ.get("LAYERS", "11"))
HEADS = int(os.environ.get("HEADS", "8"))
KV_HEADS = int(os.environ.get("KV_HEADS", "4"))
MLP_MULT = int(os.environ.get("MLP_MULT", "3"))
BIGRAM_DIM = int(os.environ.get("BIGRAM_DIM", "128"))
BIGRAM_BUCKETS = int(os.environ.get("BIGRAM_BUCKETS", "2048"))
XSA_LAST_N = int(os.environ.get("XSA_LAST_N", "4"))
XSA_ALL = int(os.environ.get("XSA_ALL", "0"))  # 1 = override XSA_LAST_N to LAYERS
EVAL_TEMPERATURE = float(os.environ.get("EVAL_TEMPERATURE", "1.0"))
SEQ = 1024
softcap = 30.0

# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

if world_size > 1:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
rank = dist.get_rank() if dist.is_initialized() else 0
is_main = rank == 0

def print0(*args, **kwargs):
    if is_main:
        print(*args, **kwargs, flush=True)

print0(f"{'='*70}")
print0(f"  SLOT-{SLOT_STEPS} DDP eval (stride={STRIDE}, batch={BATCH_SIZE})")
print0(f"  Model: {MODEL_PATH}")
print0(f"  Vocab: {VOCAB_SIZE}, Dim: {DIM}, Layers: {LAYERS}")
print0(f"  BigramHash: {BIGRAM_BUCKETS} buckets x {BIGRAM_DIM} dim")
print0(f"  XSA: {'ALL '+str(LAYERS)+' layers' if XSA_ALL else 'last '+str(XSA_LAST_N)+' layers'}")
print0(f"  Eval temperature: {EVAL_TEMPERATURE}")
print0(f"  World size: {world_size}")
print0(f"{'='*70}")

# ---------------------------------------------------------------------------
# Load model using training script
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(MODEL_PATH) or '/workspace')
sys.path.insert(0, '/workspace')
import train_retro_v4096 as trmod

trmod.VOCAB_SIZE = VOCAB_SIZE
trmod.MODEL_DIM = DIM
trmod.NUM_LAYERS = LAYERS
trmod.NUM_HEADS = HEADS
trmod.NUM_KV_HEADS = KV_HEADS
trmod.MLP_MULT = MLP_MULT
trmod.BIGRAM_DIM = BIGRAM_DIM
trmod.BIGRAM_BUCKETS = BIGRAM_BUCKETS
trmod.XSA_LAST_N = LAYERS if XSA_ALL else XSA_LAST_N
trmod.USE_FA3 = False
trmod.SEQ_LEN = SEQ

model = trmod.GPTv2().to(device)
state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(state)
model.eval()
proj_w = model.tok_emb.weight.detach().float()
n_params = sum(p.numel() for p in model.parameters())
print0(f"Params: {n_params:,}")

# ---------------------------------------------------------------------------
# Tokenizer + BPB LUTs
# ---------------------------------------------------------------------------
sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
bb, hs, ib = trmod.build_sentencepiece_luts(sp, VOCAB_SIZE)

# ---------------------------------------------------------------------------
# Val data
# ---------------------------------------------------------------------------
import glob
val_files = sorted(glob.glob(f"{DATA_DIR}/fineweb_val_*.bin"))
header = np.fromfile(val_files[0], dtype="<i4", count=256)
tokens = np.fromfile(val_files[0], dtype="<u2",
                     count=int(header[2]), offset=256 * 4).astype(np.int64)
tokens_t = torch.tensor(tokens, dtype=torch.long)
print0(f"Val: {len(tokens):,} tokens")

# ---------------------------------------------------------------------------
# Build windows + split across ranks
# ---------------------------------------------------------------------------
total_tok = len(tokens) - 1
all_windows = [ws for ws in range(0, total_tok, STRIDE)
               if min(ws + SEQ, total_tok) - ws >= 1]
n_total = len(all_windows)

# Each rank gets a contiguous chunk of windows
per_rank = (n_total + world_size - 1) // world_size
my_start = rank * per_rank
my_end = min(my_start + per_rank, n_total)
my_windows = all_windows[my_start:my_end]
my_batches = (len(my_windows) + BATCH_SIZE - 1) // BATCH_SIZE

print0(f"Total windows: {n_total:,} | Per rank: {len(my_windows):,} | Batches: {my_batches:,}")

# Precompute cosine LR
lr_schedule = [SLOT_LR_MIN + 0.5 * (SLOT_LR - SLOT_LR_MIN) * (1 + math.cos(math.pi * si / SLOT_STEPS))
               for si in range(SLOT_STEPS)]

# ---------------------------------------------------------------------------
# SLOT eval
# ---------------------------------------------------------------------------
slot_nll = 0.0
slot_scored = 0
slot_bytes = 0.0
t0 = time.time()

for bi in range(my_batches):
    bs_start = bi * BATCH_SIZE
    bs_end = min(bs_start + BATCH_SIZE, len(my_windows))
    cur = bs_end - bs_start

    # Build batch
    x_batch = torch.zeros(cur, SEQ, dtype=torch.long)
    y_batch = torch.zeros(cur, SEQ, dtype=torch.long)
    mask_batch = torch.zeros(cur, SEQ)
    wlens = []
    starts = []

    for i in range(cur):
        ws = my_windows[bs_start + i]
        wend = min(ws + SEQ, total_tok)
        wlen = wend - ws
        s = 0 if ws == 0 else max(wlen - STRIDE, 0)
        x_batch[i, :wlen] = tokens_t[ws:wend]
        y_batch[i, :wlen] = tokens_t[ws + 1:wend + 1]
        mask_batch[i, s:wlen] = 1.0
        wlens.append(wlen)
        starts.append(s)

    xb = x_batch.to(device)
    yb = y_batch.to(device)
    mb = mask_batch.to(device)

    # Frozen forward
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        hidden = model.forward_hidden(xb, is_causal=True)
    hf = hidden.detach().float()

    # SLOT params
    delta = torch.zeros(cur, 1, DIM, device=device, requires_grad=True)
    lbias = torch.zeros(cur, 1, VOCAB_SIZE, device=device, requires_grad=True)
    opt = torch.optim.AdamW([delta, lbias], lr=SLOT_LR, weight_decay=1e-8, eps=1e-5)
    vc = mb.sum(dim=1, keepdim=True).clamp(min=1.0)

    # Optimize
    for si in range(SLOT_STEPS):
        for pg in opt.param_groups:
            pg['lr'] = lr_schedule[si]
        opt.zero_grad()
        h = hf + delta
        lp = F.linear(h, proj_w) + lbias
        lg = softcap * torch.tanh(lp / softcap)
        if EVAL_TEMPERATURE != 1.0:
            lg = lg / EVAL_TEMPERATURE
        nll = F.cross_entropy(lg.reshape(-1, VOCAB_SIZE), yb.reshape(-1),
                              reduction='none').reshape(cur, SEQ)
        loss = ((nll * mb).sum(dim=1) / vc.squeeze(1)).mean()
        loss.backward()
        opt.step()

    # Score
    with torch.no_grad():
        h = hf + delta
        lp = F.linear(h, proj_w) + lbias
        lg = softcap * torch.tanh(lp / softcap)
        if EVAL_TEMPERATURE != 1.0:
            lg = lg / EVAL_TEMPERATURE
        nll_final = F.cross_entropy(lg.reshape(-1, VOCAB_SIZE), yb.reshape(-1),
                                    reduction='none').reshape(cur, SEQ)
        slot_nll += (nll_final * mb).sum().item()
        slot_scored += int(mb.sum().item())

    # Byte counting (vectorized per window)
    for i in range(cur):
        ws = my_windows[bs_start + i]
        wlen = wlens[i]
        s = starts[i]
        yn = y_batch[i, s:wlen]
        xn = x_batch[i, s:wlen]
        b = bb[yn.numpy()].copy().astype(np.float64)
        if len(b) > 1:
            b[1:] += (hs[yn[1:].numpy()] & ~ib[xn[1:].numpy()]).astype(np.float64)
        if s > 0:
            prev = int(x_batch[i, s])
        elif ws > 0:
            prev = int(tokens[ws - 1])
        else:
            prev = -1
        if prev >= 0 and hs[int(yn[0])] and not ib[prev]:
            b[0] += 1.0
        b = np.maximum(b, 1.0)
        slot_bytes += b.sum()

    if is_main and ((bi + 1) % 200 == 0 or bi == my_batches - 1):
        elapsed = time.time() - t0
        bpb = (slot_nll / slot_scored) / math.log(2) * (slot_scored / slot_bytes)
        eta = elapsed / (bi + 1) * (my_batches - bi - 1)
        print0(f"  rank0 batch {bi+1}/{my_batches} | BPB:{bpb:.4f} | scored:{slot_scored:,} | {elapsed:.0f}s (ETA {eta:.0f}s)")

# ---------------------------------------------------------------------------
# All-reduce across ranks
# ---------------------------------------------------------------------------
if dist.is_initialized():
    nll_t = torch.tensor([slot_nll], dtype=torch.float64, device=device)
    scored_t = torch.tensor([float(slot_scored)], dtype=torch.float64, device=device)
    bytes_t = torch.tensor([slot_bytes], dtype=torch.float64, device=device)
    dist.all_reduce(nll_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(scored_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(bytes_t, op=dist.ReduceOp.SUM)
    slot_nll = nll_t.item()
    slot_scored = int(scored_t.item())
    slot_bytes = bytes_t.item()

elapsed = time.time() - t0
bpb = (slot_nll / slot_scored) / math.log(2) * (slot_scored / slot_bytes)

print0(f"\n{'='*70}")
print0(f"  SLOT-{SLOT_STEPS} BPB (full val set): {bpb:.4f}")
print0(f"  Tokens scored: {slot_scored:,}")
print0(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
print0(f"  World size: {world_size}")
print0(f"{'='*70}")

if dist.is_initialized():
    dist.destroy_process_group()
