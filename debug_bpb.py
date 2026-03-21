"""Debug: compare sliding window vs baseline BPB for the same data region."""
import torch
import torch.nn.functional as F
import math
from train_gpt import GPT, Hyperparameters, dequantize_state_dict_int8, build_sentencepiece_luts, load_validation_tokens
from eval_competition import load_model
import sentencepiece as spm
import os

args = Hyperparameters()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = os.environ.get("CHECKPOINT", "final_model.int6.ptz")

# Load model
model = load_model(args, checkpoint, device)
model.eval()

# Load val tokens
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
luts = build_sentencepiece_luts(sp, args.vocab_size, device)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = luts
val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)

n_tokens = val_tokens.numel()
seq_len = 1024

print(f"val_tokens: {n_tokens:,}")
print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

# Test: score EXACTLY the same tokens (positions 1025-1088 in the data) 
# using two different windows:
# 1. Baseline window 1: start=1024, score position 0 (val_tokens[1025])
# 2. Sliding window: start=64, score position 960 (also val_tokens[1025])

with torch.inference_mode():
    # === Method 1: Baseline (start=1024, score ALL) ===
    start_bl = 1024
    chunk_bl = val_tokens[start_bl:start_bl + seq_len + 1].to(device=device, dtype=torch.int64)
    x_bl = chunk_bl[:-1].reshape(1, seq_len)
    y_bl = chunk_bl[1:]
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        logits_bl = model.forward_logits(x_bl)
    
    # Score first 64 positions (val_tokens[1025:1089])
    ce_bl = F.cross_entropy(logits_bl[:64].float(), y_bl[:64].to(device), reduction="none")
    
    # Byte count
    prev_bl = x_bl.reshape(-1)[:64]
    tgt_bl = y_bl[:64].to(device)
    bytes_bl = base_bytes_lut[tgt_bl].to(torch.int16)
    bytes_bl += (has_leading_space_lut[tgt_bl] & ~is_boundary_token_lut[prev_bl]).to(torch.int16)
    
    print(f"\n=== Baseline window (start={start_bl}) ===")
    print(f"  Targets: val_tokens[{start_bl+1}:{start_bl+65}]")
    print(f"  Context for pos 0: {0} tokens")
    print(f"  CE first 64 tokens: {ce_bl.sum().item():.4f} (mean: {ce_bl.mean().item():.4f})")
    print(f"  Bytes: {bytes_bl.sum().item()}")
    bl_bpb = (ce_bl.sum().item() / math.log(2)) * (64 / bytes_bl.to(torch.float64).sum().item())
    print(f"  BPB (64 tokens): {bl_bpb:.4f}")
    
    # === Method 2: Sliding window (start=64, score positions 960-1023) ===
    start_sw = 64
    chunk_sw = val_tokens[start_sw:start_sw + seq_len + 1].to(device=device, dtype=torch.int64)
    x_sw = chunk_sw[:-1].reshape(1, seq_len)
    y_sw = chunk_sw[1:]
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        logits_sw = model.forward_logits(x_sw)
    
    score_start = 960
    ce_sw = F.cross_entropy(logits_sw[score_start:].float(), y_sw[score_start:].to(device), reduction="none")
    
    # Byte count  
    prev_sw = x_sw.reshape(-1)[score_start:]
    tgt_sw = y_sw[score_start:].to(device)
    bytes_sw = base_bytes_lut[tgt_sw].to(torch.int16)
    bytes_sw += (has_leading_space_lut[tgt_sw] & ~is_boundary_token_lut[prev_sw]).to(torch.int16)
    
    print(f"\n=== Sliding window (start={start_sw}, score_start={score_start}) ===")
    print(f"  Targets: val_tokens[{start_sw + score_start + 1}:{start_sw + seq_len + 1}]")
    print(f"  Context for scored pos: {score_start} tokens")
    print(f"  CE 64 tokens: {ce_sw.sum().item():.4f} (mean: {ce_sw.mean().item():.4f})")
    print(f"  Bytes: {bytes_sw.sum().item()}")
    sw_bpb = (ce_sw.sum().item() / math.log(2)) * (64 / bytes_sw.to(torch.float64).sum().item())
    print(f"  BPB (64 tokens): {sw_bpb:.4f}")
    
    # Verify same targets
    same = torch.all(y_bl[:64] == y_sw[score_start:]).item()
    print(f"\n  Same targets? {same}")
    print(f"  Same prev_ids? {torch.all(prev_bl == prev_sw).item()}")
    print(f"  Same bytes? {bytes_bl.sum().item() == bytes_sw.sum().item()}")
    print(f"  CE ratio (sw/bl): {ce_sw.sum().item() / ce_bl.sum().item():.4f}")
    
    # === Now test: full 100 windows comparison ===
    print("\n\n=== Full comparison: 100 baseline vs 100 sliding windows ===")
    
    # Baseline: stride=1024, first 100 windows
    total_loss_bl = 0.0
    total_tokens_bl = 0
    total_bytes_bl = 0.0
    for w in range(100):
        start = w * 1024
        end = start + seq_len + 1
        if end > n_tokens: break
        chunk = val_tokens[start:end].to(device=device, dtype=torch.int64)
        x = chunk[:-1].reshape(1, seq_len)
        y = chunk[1:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits = model.forward_logits(x)
        ce = F.cross_entropy(logits.float(), y.to(device), reduction="sum")
        total_loss_bl += ce.item()
        total_tokens_bl += 1024
        prev = x.reshape(-1)
        tgt = y.to(device)
        tb = base_bytes_lut[tgt].to(torch.int16) + (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.int16)
        total_bytes_bl += tb.to(torch.float64).sum().item()
    
    bl_avg_loss = total_loss_bl / total_tokens_bl
    bl_bpb100 = (bl_avg_loss / math.log(2)) * (total_tokens_bl / total_bytes_bl)
    print(f"  Baseline 100 windows: loss={bl_avg_loss:.4f} bpb={bl_bpb100:.4f} tokens={total_tokens_bl} bytes={total_bytes_bl:.0f}")
    
    # Sliding: stride=64, covering same data range
    total_loss_sw = 0.0
    total_tokens_sw = 0
    total_bytes_sw = 0.0
    n_data = 100 * 1024  # same data range
    stride = 64
    starts = list(range(0, n_data, stride))
    for w, start in enumerate(starts):
        end = start + seq_len + 1
        if end > n_tokens: break
        chunk = val_tokens[start:end].to(device=device, dtype=torch.int64)
        x = chunk[:-1].reshape(1, seq_len)
        y = chunk[1:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits = model.forward_logits(x)
        ss = 0 if start == 0 else seq_len - stride
        ce = F.cross_entropy(logits[ss:].float(), y[ss:].to(device), reduction="sum")
        n = seq_len - ss
        total_loss_sw += ce.item()
        total_tokens_sw += n
        prev = x.reshape(-1)[ss:]
        tgt = y[ss:].to(device)
        tb = base_bytes_lut[tgt].to(torch.int16) + (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.int16)
        total_bytes_sw += tb.to(torch.float64).sum().item()
    
    sw_avg_loss = total_loss_sw / total_tokens_sw
    sw_bpb100 = (sw_avg_loss / math.log(2)) * (total_tokens_sw / total_bytes_sw)
    print(f"  Sliding {len(starts)} windows: loss={sw_avg_loss:.4f} bpb={sw_bpb100:.4f} tokens={total_tokens_sw} bytes={total_bytes_sw:.0f}")
    print(f"  Token/byte ratio: baseline={total_tokens_bl/total_bytes_bl:.4f} sliding={total_tokens_sw/total_bytes_sw:.4f}")
    print(f"  BPB delta: {sw_bpb100 - bl_bpb100:+.4f}")
