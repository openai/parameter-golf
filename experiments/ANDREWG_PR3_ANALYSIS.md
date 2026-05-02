# andrewgcodes/parameter-golf PR#3 Analysis — val_bpb=1.1570

## Key Findings

### 1. RoPE base 200K (vs default 10K) — NOVEL, HIGH PRIORITY
- Standard RoPE base is 10,000. They use 200,000.
- Higher base = lower-frequency positional encoding = smoother position interpolation
- This likely helps sliding window eval: tokens at positions 960-1024 (end of window)
  get smoother attention patterns with higher RoPE base
- We haven't tried this at all. Could be worth 0.002-0.005 BPB.
- **EXPERIMENT: Try ROPE_BASE=200000 on our best config (060)**

### 2. Warmdown 14K (wallclock-based) — interesting but different approach
- They use WARMDOWN_ITERS=14000 but it's wallclock-based, so it adapts
- With ~7300 steps, this means essentially ALL training is warmdown
- We tried WD=6000 (exp062) and it hurt. But their full-warmdown approach is even more extreme.
- Lower priority since we already tested aggressive warmdown.

### 3. matrix_lr=0.025 (vs our 0.02) — minor tweak
- 25% higher LR. Could help or hurt.
- Lower priority.

### 4. 11-layer config got 1.1448 BPP — over budget at 18.6MB
- Same artifact size problem as us!
- They solved it by going to 9 layers (1.1570) which fits at 15.35MB
- Confirms: the model quality vs artifact size tradeoff is the key constraint

### 5. batch_tokens=393K (vs our 786K with seq2048)
- Smaller batch, same seq_len used in actual run (2048)
- Our 055 used 786K batch with seq2048 — maybe 393K is better?
- But PR135 uses 786K and gets good results. Not clear this matters.

## Action Items
1. **HIGH**: Try ROPE_BASE=200000 on our best config
2. **LOW**: Try matrix_lr=0.025
3. **LOW**: Try extreme warmdown (14K+)

## Script saved at: /tmp/andrewg_train_gpt.py
