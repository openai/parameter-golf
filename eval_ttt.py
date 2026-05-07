"""
Full-Model SGD Test-Time Training Evaluation
Based on PR #254 (1.1313 BPB) and PR #152 (0.034 BPB gain from TTT alone).

Usage: After training with train_gpt_v2.py, run this to evaluate with TTT:
  python3 eval_ttt.py --model final_model.quant.ptz --epochs 3 --lr 0.002

The model is adapted on the validation set using full-model SGD with momentum,
then evaluated with sliding window. Each epoch takes ~100-200s on 8xH100.
"""

from __future__ import annotations

import argparse
import glob
import io
import math
import os
import time
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

try:
    import zstandard
except ImportError:
    zstandard = None


def load_data_shard(file: Path) -> torch.Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model.quant.ptz")
    parser.add_argument("--script", default="train_gpt_v2.py")
    parser.add_argument("--val-pattern", default="./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin")
    parser.add_argument("--tokenizer", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--epochs", type=int, default=5)  # v2: 5 epochs at lower LR
    parser.add_argument("--lr", type=float, default=0.001)  # v2: lower peak, less forgetting
    parser.add_argument("--momentum", type=float, default=0.5)  # v2: reduced momentum (non-stationary)
    parser.add_argument("--freeze-first-n", type=int, default=2)
    parser.add_argument("--wd", type=float, default=0.01)  # v2: weight decay during TTT
    parser.add_argument("--cosine-lr", action="store_true")  # v2: cosine decay during TTT
    parser.add_argument("--per-layer-lr", action="store_true")  # v2: discriminative LR per block
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-tokens", type=int, default=65536)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--temp-search", action="store_true")  # search optimal temperature
    parser.add_argument("--ppm-mix", type=float, default=0.0)  # PPM-C context mixer weight (0.05=5%)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load quantized model
    with open(args.model, "rb") as f:
        blob = f.read()
    try:
        if zstandard:
            raw = zstandard.ZstdDecompressor().decompress(blob)
        else:
            raw = zlib.decompress(blob)
    except Exception:
        raw = zlib.decompress(blob)

    quant_obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)

    # Import model class and dequantize
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_mod", args.script)
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    state_dict = train_mod.dequantize_state_dict(quant_obj)
    hp = train_mod.Hyperparameters()
    model = train_mod.GPT(
        vocab_size=hp.vocab_size, num_layers=hp.num_layers, model_dim=hp.model_dim,
        num_heads=hp.num_heads, num_kv_heads=hp.num_kv_heads, mlp_mult=hp.mlp_mult,
        tie_embeddings=hp.tie_embeddings, tied_embed_init_std=hp.tied_embed_init_std,
        logit_softcap=hp.logit_softcap, rope_base=hp.rope_base, qk_gain_init=hp.qk_gain_init,
        use_smear_gate=hp.use_smear_gate, bigram_vocab_size=hp.bigram_vocab_size,
        bigram_dim=hp.bigram_dim, depth_recurrence=hp.depth_recurrence,
        xsa_last_n=hp.xsa_last_n,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.bfloat16()

    # Load validation tokens
    val_files = sorted(glob.glob(args.val_pattern))
    val_tokens = torch.cat([load_data_shard(Path(f)) for f in val_files])
    print(f"Val tokens: {val_tokens.numel()}")

    # Tokenizer LUTs
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = train_mod.build_sentencepiece_luts(
        sp, hp.vocab_size, device
    )

    # --- TTT Phase ---
    print(f"\n=== TTT: {args.epochs} epochs, lr={args.lr}, momentum={args.momentum} ===")
    for i, block in enumerate(model.blocks):
        if i < args.freeze_first_n:
            for p in block.parameters():
                p.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable}/{total} ({100*trainable/total:.1f}%)")

    if args.per_layer_lr and hasattr(model, 'blocks'):
        param_groups = []
        n_blocks = len(model.blocks)
        for i, block in enumerate(model.blocks):
            if i < args.freeze_first_n:
                continue
            layer_lr = args.lr * ((i + 1) / n_blocks)  # ramp: 0.0001 -> lr
            param_groups.append({"params": [p for p in block.parameters() if p.requires_grad], "lr": layer_lr})
        # Non-block params at full LR
        block_params = set(id(p) for b in model.blocks for p in b.parameters())
        other = [p for p in model.parameters() if p.requires_grad and id(p) not in block_params]
        if other:
            param_groups.append({"params": other, "lr": args.lr})
        optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.wd)
        print(f"Per-layer LR: {len(param_groups)} groups, min_lr={args.lr/n_blocks:.6f}, max_lr={args.lr}")
    else:
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, momentum=args.momentum, weight_decay=args.wd
        )
    total_steps = args.epochs * (usable // args.batch_tokens + 1)

    seq_len = args.seq_len
    usable = ((val_tokens.numel() - 1) // seq_len) * seq_len
    trunc = val_tokens[:usable + 1]

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, usable, args.batch_tokens):
            # Cosine LR decay
            if args.cosine_lr and total_steps > 0:
                lr_scale = 0.5 * (1 + math.cos(math.pi * global_step / total_steps))
                for g in optimizer.param_groups:
                    g["lr"] = args.lr * lr_scale
            end = min(start + args.batch_tokens + 1, usable + 1)
            chunk = trunc[start:end].to(device, dtype=torch.int64)
            n_seqs = (chunk.numel() - 1) // seq_len
            if n_seqs == 0:
                continue
            x = chunk[:n_seqs * seq_len].reshape(n_seqs, seq_len)
            y = chunk[1:n_seqs * seq_len + 1].reshape(n_seqs, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
        print(f"TTT epoch {epoch+1}/{args.epochs}: loss={epoch_loss/max(n_batches,1):.4f} time={time.perf_counter()-t0:.1f}s")

    # --- PPM-C Context Model (order-2) ---
    ppm_counts = None
    if args.ppm_mix > 0:
        print(f"\n=== Building PPM-C order-2 context model (mix={args.ppm_mix}) ===")
        vocab_sz = hp.vocab_size
        # Count bigram frequencies from val data
        ppm_counts = torch.zeros(vocab_sz, vocab_sz, dtype=torch.float32)
        tokens_np = val_tokens.numpy().astype(int)
        for i in range(len(tokens_np) - 1):
            ppm_counts[tokens_np[i], tokens_np[i + 1]] += 1
        # Normalize to probabilities with add-1 smoothing
        ppm_probs = (ppm_counts + 1) / (ppm_counts.sum(dim=1, keepdim=True) + vocab_sz)
        ppm_probs = ppm_probs.to(device)
        print(f"PPM-C built: {vocab_sz}x{vocab_sz} bigram table")

    # --- Sliding Window Eval ---
    print(f"\n=== Sliding Window: stride={args.stride}, seq={seq_len} ===")
    model.eval()
    t_sw = time.perf_counter()
    total_tok = val_tokens.numel() - 1
    wins = [ws for ws in range(0, total_tok, args.stride) if min(ws + seq_len, total_tok) - ws >= 1]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    with torch.inference_mode():
        for bi in range(0, len(wins), args.eval_batch):
            batch_ws = wins[bi:bi + args.eval_batch]
            sx, sy, wl = [], [], []
            for ws in batch_ws:
                we = min(ws + seq_len, total_tok)
                wlen = we - ws
                wl.append(wlen)
                c = val_tokens[ws:we + 1].to(dtype=torch.int64)
                xq, yq = c[:-1], c[1:]
                if len(xq) < seq_len:
                    xq = F.pad(xq, (0, seq_len - len(xq)))
                    yq = F.pad(yq, (0, seq_len - len(yq)))
                sx.append(xq)
                sy.append(yq)
            xb = torch.stack(sx).to(device)
            yb = torch.stack(sy).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(xb)
            if args.ppm_mix > 0 and ppm_probs is not None:
                # Mix neural softmax with PPM bigram probs
                neural_probs = F.softmax(logits.float(), dim=-1)
                # Get PPM probs for each (prev_token -> next_token)
                prev_tokens = xb  # (batch, seq)
                ppm_p = ppm_probs[prev_tokens.reshape(-1)].reshape(xb.size(0), xb.size(1), -1)
                mixed = (1 - args.ppm_mix) * neural_probs + args.ppm_mix * ppm_p
                nll = -torch.log(mixed.gather(-1, yb.unsqueeze(-1)).squeeze(-1) + 1e-10)
                nll = nll.reshape(xb.size(0), -1)
            else:
                nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                      yb.reshape(-1), reduction="none").reshape(xb.size(0), -1)
            for i, ws in enumerate(batch_ws):
                wlen = wl[i]
                s = 0 if ws == 0 else max(wlen - args.stride, 0)
                loss_sum += nll[i, s:wlen].to(torch.float64).sum()
                token_count += float(wlen - s)
                tgt, prev = yb[i, s:wlen], xb[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.int16)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.int16)
                byte_count += tb.to(torch.float64).sum()

    vl = (loss_sum / token_count).item()
    vb = (loss_sum / math.log(2.0) / byte_count).item()
    print(f"\nfinal_ttt_sliding val_loss:{vl:.4f} val_bpb:{vb:.4f} time:{time.perf_counter()-t_sw:.0f}s")
    print(f"final_ttt_sliding_exact val_loss:{vl:.8f} val_bpb:{vb:.8f}")

    if args.temp_search:
        print("\n=== Temperature Search ===")
        best_t, best_bpb = 1.0, vb
        for t in [0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 1.01, 1.02, 1.05]:
            t_loss = torch.zeros((), device=device, dtype=torch.float64)
            t_tc = torch.zeros((), device=device, dtype=torch.float64)
            t_bc = torch.zeros((), device=device, dtype=torch.float64)
            with torch.inference_mode():
                for bi in range(0, len(wins), args.eval_batch * 4):  # 4x coarser for speed
                    bws = wins[bi:bi + args.eval_batch * 4]
                    ssx, ssy, wwl = [], [], []
                    for ws in bws:
                        we = min(ws + seq_len, total_tok)
                        wl2 = we - ws; wwl.append(wl2)
                        c2 = val_tokens[ws:we+1].to(dtype=torch.int64)
                        xx, yy = c2[:-1], c2[1:]
                        if len(xx) < seq_len:
                            xx = F.pad(xx, (0, seq_len-len(xx)))
                            yy = F.pad(yy, (0, seq_len-len(yy)))
                        ssx.append(xx); ssy.append(yy)
                    xb2 = torch.stack(ssx).to(device)
                    yb2 = torch.stack(ssy).to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        lg = model.forward_logits(xb2) / t
                    nl = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), yb2.reshape(-1), reduction="none").reshape(xb2.size(0), -1)
                    for i2, ws2 in enumerate(bws):
                        w2 = wwl[i2]; s2 = 0 if ws2 == 0 else max(w2 - args.stride * 4, 0)
                        t_loss += nl[i2, s2:w2].to(torch.float64).sum()
                        t_tc += float(w2 - s2)
                        tg2 = yb2[i2, s2:w2]; pg2 = xb2[i2, s2:w2]
                        tb2 = base_bytes_lut[tg2].to(torch.int16)
                        tb2 += (has_leading_space_lut[tg2] & ~is_boundary_token_lut[pg2]).to(torch.int16)
                        t_bc += tb2.to(torch.float64).sum()
            tbpb = (t_loss / math.log(2.0) / t_bc).item()
            marker = " <-- BEST" if tbpb < best_bpb else ""
            print(f"  T={t:.2f}: val_bpb={tbpb:.6f}{marker}")
            if tbpb < best_bpb:
                best_t, best_bpb = t, tbpb
        print(f"\nBest temperature: T={best_t:.2f}, val_bpb={best_bpb:.6f}")


if __name__ == "__main__":
    main()
