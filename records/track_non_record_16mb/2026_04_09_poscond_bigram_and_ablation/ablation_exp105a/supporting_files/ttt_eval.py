"""Standalone TTT eval with SGD optimizations on an already-quantized exp101 model."""
import sys, os, glob, math, time, io, lzma
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from pathlib import Path

# Add the exp101 code to path
sys.path.insert(0, "/workspace/parameter-golf/records/track_10min_16mb/exp101_poscond-bigram-trigram_from_exp95")
os.environ.setdefault("POS_CONDITIONAL_BIGRAM", "1")
os.environ.setdefault("TRIGRAM", "1")
os.environ["BIGRAM_VOCAB_SIZE"] = "4096"
os.environ["BIGRAM_DIM"] = "64"
os.environ["VE_LAYERS"] = "7,8,9,10"
os.environ["VE_ENABLED"] = "1"
os.environ["ROPE_DIMS"] = "16"
os.environ["LN_SCALE"] = "1"
os.environ["XSA_LAST_N"] = "11"
os.environ["NUM_LAYERS"] = "11"

from train_gpt import (
    GPT, CastedLinear, Rotary, Hyperparameters,
    build_sentencepiece_luts, load_validation_tokens,
    _unbank_state_dict, _rebank_state_dict,
    dequantize_mixed_int6, restore_low_dim_params_to_fp32,
)
import sentencepiece as spm

device = torch.device("cuda")
args = Hyperparameters()

# Load tokenizer and val data
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

# Load quantized model
print("Loading quantized model...")
with open("/workspace/parameter-golf/final_model.int6.ptz", "rb") as f:
    quant_blob = f.read()
quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob)), map_location="cpu")

# Load raw model to get template state dict for rebanking
raw_sd = torch.load("/workspace/parameter-golf/final_model.pt", map_location="cpu")

# Dequantize
unbanked_sd = _unbank_state_dict({k: v.detach().cpu() for k, v in raw_sd.items()}, args.num_layers)
deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, raw_sd)

# Build model
print("Building model...")
model = GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    mtp_num_heads=0, mtp_loss_weight=0.0,
    bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
    ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
).to(device).bfloat16()
model.qo_bank.data = model.qo_bank.data.float()
model.kv_bank.data = model.kv_bank.data.float()
model.mlp_up_bank.data = model.mlp_up_bank.data.float()
model.mlp_down_bank.data = model.mlp_down_bank.data.float()
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()
restore_low_dim_params_to_fp32(model)
model.load_state_dict(deq_state, strict=True)
model._has_leading_space = has_leading_space_lut

print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

# --- TTT with optimized SGD ---
seq_len = args.train_seq_len
total_tokens = val_tokens.numel() - 1
stride = 64

# === TUNED HYPERPARAMS ===
ttt_lr = 0.002          # [1] higher than 0.001 — old cosine peak was 0.001, now flat
ttt_epochs = 3          # keep 3 (4 risks overfitting per chunk with SGD)
ttt_chunk = 65536       # [2] larger chunks — more data per adaptation, less overfitting
ttt_freeze_blocks = 2
ttt_momentum = 0.9
ttt_nesterov = True     # [3] Nesterov look-ahead — faster convergence, free
ttt_wd = 0.001          # [4] small weight decay — regularizes per-chunk adaptation
ttt_grad_clip = 1.0
eval_batch = 128
train_batch = 16

window_starts = [ws for ws in range(0, total_tokens, stride)
                 if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
chunk_windows = [[] for _ in range(num_chunks)]
for ws in window_starts:
    end = min(ws + seq_len, total_tokens)
    wlen = end - ws
    s = 0 if ws == 0 else max(wlen - stride, 0)
    scored_start = ws + s
    ci = min(scored_start // ttt_chunk, num_chunks - 1)
    chunk_windows[ci].append(ws)

# Freeze first N blocks
frozen_ids = set(range(ttt_freeze_blocks))
ttt_params = []
for name, p in model.named_parameters():
    freeze = any(f"blocks.{bi}." in name for bi in frozen_ids)
    if freeze:
        p.requires_grad_(False)
    else:
        p.requires_grad_(True)
        ttt_params.append(p)

unfrozen_n = sum(p.numel() for p in ttt_params)
frozen_n = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"TTT: SGD lr={ttt_lr} momentum={ttt_momentum} nesterov={ttt_nesterov} "
      f"wd={ttt_wd} epochs={ttt_epochs} chunks={num_chunks} chunk_tokens={ttt_chunk}")
print(f"TTT: unfrozen={unfrozen_n:,} frozen={frozen_n:,}")

# [1,3,4] SGD with Nesterov + weight decay
optimizer = torch.optim.SGD(ttt_params, lr=ttt_lr, momentum=ttt_momentum,
                             nesterov=ttt_nesterov, weight_decay=ttt_wd)

loss_sum = torch.zeros((), device=device, dtype=torch.float64)
token_count = torch.zeros((), device=device, dtype=torch.float64)
byte_count = torch.zeros((), device=device, dtype=torch.float64)
t0 = time.perf_counter()

for ci in range(num_chunks):
    windows = chunk_windows[ci]
    if not windows:
        continue
    chunk_start = ci * ttt_chunk
    chunk_end = min((ci + 1) * ttt_chunk, total_tokens)

    # Phase 1: SCORE (evaluate before training — legal TTT)
    model.eval()
    with torch.inference_mode():
        for bi in range(0, len(windows), eval_batch):
            batch_ws = windows[bi:bi + eval_batch]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk_tok = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk_tok[:-1]
                y_batch[i, :wlen] = chunk_tok[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1), reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt, prev = y_batch[i, s:wlen], x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

    # Phase 2: TRAIN with SGD
    is_last = (ci == num_chunks - 1)
    if not is_last and ttt_epochs > 0:
        model.train()
        chunk_seqs = (chunk_end - chunk_start) // seq_len
        if chunk_seqs > 0:
            # [5] Flat LR — each chunk is independent data,
            #     cosine across chunks starved late chunks (lr→0)
            for pg in optimizer.param_groups:
                pg['lr'] = ttt_lr

            # [6] Reset momentum buffers between chunks — stale momentum
            #     from chunk N is noise for chunk N+1's different data
            for p in ttt_params:
                state = optimizer.state.get(p, {})
                if 'momentum_buffer' in state:
                    state['momentum_buffer'].zero_()

            for _ep in range(ttt_epochs):
                for bs in range(0, chunk_seqs, train_batch):
                    be = min(bs + train_batch, chunk_seqs)
                    start_tok = chunk_start + bs * seq_len
                    end_tok = chunk_start + be * seq_len + 1
                    if end_tok > val_tokens.numel():
                        continue
                    local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                    x = local[:-1].reshape(-1, seq_len)
                    y = local[1:].reshape(-1, seq_len)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(x, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ttt_params, ttt_grad_clip)
                    optimizer.step()

    if ci % 100 == 0 or ci == num_chunks - 1:
        elapsed = time.perf_counter() - t0
        rl = loss_sum.item() / max(token_count.item(), 1)
        rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1))
        pct = (ci + 1) / num_chunks * 100
        eta = (elapsed / max(ci + 1, 1)) * (num_chunks - ci - 1)
        print(f"  chunk {ci+1}/{num_chunks} ({pct:.1f}%) bpb={rbpb:.6f} ETA={eta:.0f}s")

val_loss = (loss_sum / token_count).item()
val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
print(f"\nFINAL TTT (SGD nesterov, flat LR={ttt_lr}): val_loss={val_loss:.6f} val_bpb={val_bpb:.6f}")

for p in model.parameters():
    p.requires_grad_(True)
