#!/usr/bin/env python3
"""
Legal Score-First TTT Eval for Clark's Model
==============================================
Loads a trained Clark model, adds LoRA adapters to Q and V,
runs strict score-first TTT, reports BPB.

PROTOCOL (100% legal, same as PR #549 approved by valerio-oai):
  For each chunk:
    1. SCORE: forward pass, compute loss (eval mode, no grad)
    2. Record loss for BPB calculation
    3. TRAIN: gradient update on scored chunk (AFTER scoring)
    4. NEXT: use updated model for next chunk

USAGE on H100 (after Clark's train_gpt.py has trained a model):
  python3 clark_ttt_eval.py

Requires Clark's train_gpt.py in the same directory (as module).
Loads model checkpoint from final_model.pt or trains briefly for testing.
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/workspace/repo')

import os, time, math, copy
os.chdir('/workspace/repo')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Legal Score-First TTT Eval — {time.strftime('%H:%M:%S')}")

# ============================================================
# Load Clark's code as module
# ============================================================
import train_gpt as tg

# ============================================================
# LoRA wrapper
# ============================================================
class LoRAWrapper(nn.Module):
    """Wraps a CastedLinear/Linear with LoRA. Only A and B are trainable."""
    def __init__(self, base_linear, rank=8):
        super().__init__()
        self.base = base_linear
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.scale = 1.0 / rank
        device = next(base_linear.parameters()).device
        self.lora_A = nn.Parameter(torch.randn(in_f, rank, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f, device=device))
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A @ self.lora_B) * self.scale

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):
        return self.base.weight


def add_lora(model, rank=8):
    """Add LoRA to Q and V projections in all attention blocks.
    Freeze all base params. Returns list of LoRA parameters."""
    for p in model.parameters():
        p.requires_grad = False

    lora_params = []
    for block in model.blocks:
        attn = block.attn
        # Wrap c_q
        lora_q = LoRAWrapper(attn.c_q, rank=rank)
        attn.c_q = lora_q
        lora_params.extend([lora_q.lora_A, lora_q.lora_B])
        # Wrap c_v
        lora_v = LoRAWrapper(attn.c_v, rank=rank)
        attn.c_v = lora_v
        lora_params.extend([lora_v.lora_A, lora_v.lora_B])

    n_lora = sum(p.numel() for p in lora_params)
    print(f"  LoRA: rank={rank}, {n_lora:,} params on Q,V in {len(model.blocks)} layers")
    return lora_params


# ============================================================
# Score-First TTT
# ============================================================
def score_first_ttt(model, val_tokens, lora_params, h,
                     chunk_size=2048, epochs=3, lr=0.001,
                     byte_luts=None):
    """Strict score-first TTT. Score chunk → record loss → train on it → next chunk."""
    optimizer = torch.optim.AdamW(lora_params, lr=lr, betas=(0.9, 0.95))

    n_tokens = val_tokens.numel()
    n_chunks = (n_tokens - 1) // chunk_size
    vocab_size = h.vocab_size

    total_nll = 0.0
    total_scored = 0
    total_bytes = 0.0
    t0 = time.time()

    for c in range(n_chunks):
        start = c * chunk_size
        end = min(start + chunk_size + 1, n_tokens)
        chunk = val_tokens[start:end].to(device=DEVICE, dtype=torch.long)
        if len(chunk) < 2:
            continue

        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)
        n_tok = y.numel()

        # === STEP 1: SCORE (eval mode, no gradients) ===
        model.eval()
        with torch.no_grad():
            with torch.autocast("cuda", torch.bfloat16):
                logits = model.forward_logits(x)
            loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), y.reshape(-1))

        total_nll += loss.item() * n_tok
        total_scored += n_tok

        # Byte counting for BPB
        if byte_luts is not None:
            base_lut, space_lut, boundary_lut = byte_luts
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            tb = base_lut[tgt_ids].to(torch.int16)
            tb += (space_lut[tgt_ids] & ~boundary_lut[prev_ids]).to(torch.int16)
            total_bytes += tb.float().sum().item()

        # === STEP 2: TRAIN on scored chunk (AFTER scoring) ===
        if c < n_chunks - 1:
            model.train()
            for ep in range(epochs):
                with torch.autocast("cuda", torch.bfloat16):
                    logits = model.forward_logits(x)
                train_loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), y.reshape(-1))
                optimizer.zero_grad()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()

        # Progress
        if (c + 1) % 50 == 0 or c == n_chunks - 1:
            avg_loss = total_nll / total_scored
            if total_bytes > 0:
                bpb = (avg_loss / math.log(2)) * (total_scored / total_bytes)
                print(f"  TTT [{c+1}/{n_chunks}] loss={avg_loss:.4f} bpb={bpb:.4f} ({time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"  TTT [{c+1}/{n_chunks}] loss={avg_loss:.4f} ({time.time()-t0:.0f}s)", flush=True)

    avg_loss = total_nll / total_scored
    if total_bytes > 0:
        bpb = (avg_loss / math.log(2)) * (total_scored / total_bytes)
    else:
        bpb = avg_loss / math.log(2)

    return avg_loss, bpb, time.time() - t0


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("\n=== Building model ===")
    h = tg.Hyperparameters()

    # Load tokenizer + byte LUTs
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
    byte_luts = tg.build_sentencepiece_luts(sp, h.vocab_size, torch.device(DEVICE))

    # Load validation tokens — h.val_files is a glob pattern STRING
    val_tokens = tg.load_validation_tokens(h.val_files, h.eval_seq_len)
    print(f"Val tokens: {val_tokens.numel():,}")

    # Build model
    model = tg.GPT(h).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # Load checkpoint if available, else quick train
    ckpt_path = Path("final_model.pt")
    if ckpt_path.exists():
        print(f"Loading checkpoint from {ckpt_path}...")
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state, strict=False)
        print("Checkpoint loaded")
    else:
        print("\n=== No checkpoint — quick training (200 steps) ===")
        train_files = sorted(Path(h.datasets_dir).glob("fineweb_train_*.bin"))
        if not train_files:
            print("ERROR: No training data found")
            sys.exit(1)
        train_shard = tg.load_data_shard(train_files[0])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=h.muon_wd)
        model.train()
        for step in range(200):
            start_idx = step * h.train_seq_len * 8
            if start_idx + h.train_seq_len * 8 + 1 > train_shard.numel():
                start_idx = 0
            chunk = train_shard[start_idx:start_idx + h.train_seq_len * 8 + 1].to(DEVICE, torch.long)
            x = chunk[:-1].reshape(-1, h.train_seq_len)[:8]
            y = chunk[1:].reshape(-1, h.train_seq_len)[:8]
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(x, y)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 100 == 0:
                print(f"  Step {step}: loss={loss.item():.4f}")

    # === Eval WITHOUT TTT ===
    print("\n=== Eval WITHOUT TTT ===")
    model.eval()
    n_eval = min(500000, val_tokens.numel() - 1)
    chunk_size = h.eval_seq_len
    n_chunks = n_eval // chunk_size

    base_lut, space_lut, boundary_lut = byte_luts
    total_nll = 0.0; total_tok = 0; total_bytes = 0.0

    with torch.no_grad():
        for c in range(n_chunks):
            s = c * chunk_size
            chunk = val_tokens[s:s + chunk_size + 1].to(DEVICE, torch.long)
            x = chunk[:-1].unsqueeze(0)
            y = chunk[1:].unsqueeze(0)
            with torch.autocast("cuda", torch.bfloat16):
                logits = model.forward_logits(x)
            loss = F.cross_entropy(logits.float().reshape(-1, h.vocab_size), y.reshape(-1))
            total_nll += loss.item() * y.numel()
            total_tok += y.numel()
            tb = base_lut[y.reshape(-1)].to(torch.int16)
            tb += (space_lut[y.reshape(-1)] & ~boundary_lut[x.reshape(-1)]).to(torch.int16)
            total_bytes += tb.float().sum().item()

    pre_loss = total_nll / total_tok
    pre_bpb = (pre_loss / math.log(2)) * (total_tok / total_bytes)
    print(f"Pre-TTT: loss={pre_loss:.4f} bpb={pre_bpb:.4f} ({total_tok:,} tokens)")

    # === Add LoRA + Run TTT ===
    print("\n=== Score-First TTT (LoRA rank=8) ===")
    ttt_model = copy.deepcopy(model)
    lora_params = add_lora(ttt_model, rank=8)

    ttt_loss, ttt_bpb, ttt_time = score_first_ttt(
        ttt_model, val_tokens[:n_eval + 1], lora_params, h,
        chunk_size=chunk_size, epochs=3, lr=0.001,
        byte_luts=byte_luts
    )

    # === Results ===
    improvement = (ttt_bpb - pre_bpb) / pre_bpb * 100
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Pre-TTT:  loss={pre_loss:.4f} bpb={pre_bpb:.4f}")
    print(f"Post-TTT: loss={ttt_loss:.4f} bpb={ttt_bpb:.4f}")
    print(f"Change:   {improvement:+.2f}%")
    print(f"TTT time: {ttt_time:.0f}s")
    print(f"Tokens:   {total_tok:,}")
