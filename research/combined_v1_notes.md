# combined_v1.py — Merge Notes

**File:** `parameter-golf/combined_v1.py`  
**Lines:** 1409 (limit: 1500) ✅  
**Syntax:** compiles cleanly ✅  
**Base:** PR #281 (1.1374 bpb — strongest leaderboard score)

---

## What Was Merged

### From PR #281 (base — everything kept)
- U-Net skip connections with learned per-dim weights (`skip_weights`)
- SmearGate: sigmoid-gated token smoothing before transformer blocks
- BigramHashEmbedding: hash-based bigram context (upgraded to 10240 per #295)
- OrthoInit: orthogonal init for weight matrices, scaled output projections
- SVD embedding init for tied embeddings
- SWA every 50 steps during warmdown (aggressive, `swa_start_frac=0.4`)
- NTK-aware RoPE for long-context generalization
- Flash Attention 3 support (falls back to `F.scaled_dot_product_attention`)
- Muon optimizer with Newton-Schulz backend + weight decay
- Full warmup-then-restore priming cycle
- Sliding window eval (stride=64, batched for GPU efficiency)
- Wallclock cap synchronization across DDP ranks

### From PR #295 — surgically added
1. **QAT with STE** (`CastedLinear.qat_levels` per-instance):
   - MLP weights: int5 quantization (clip=15) during forward pass, STE gradient
   - Attention weights: int6 quantization (clip=31) during forward pass, STE gradient
   - Enabled by `QAT_ENABLED=1` (default on)
   - Per-instance `qat_levels` (not class-level) to avoid interference with eval
2. **Backout** (`backout_lambda` parameter):
   - Learned scalar multiplying mid-point residual and subtracting from output
   - Init at 0.2, tunable via `BACKOUT_INIT`
   - Included in CONTROL_TENSOR_NAME_PATTERNS (fp32, not quantized)
3. **BigramHash(10240)**: upgraded bucket size from 4096 → 10240 (more unique bigram slots)
4. **zstd-22**: mandatory (hard import, fails fast if missing). Replaces zlib from #281.
5. **Magnitude pruning**: 8% by default (`PRUNE_FRAC=0.08`) applied pre-quantization

### From PR #302 — surgically added
1. **Online Causal TTT with decay prior** (`eval_val_ttt`):
   - Replaces #281's full-weight SGD TTT (3 epochs over full val set)
   - Interleaves forward-score with backward-adapt, one window at a time
   - Coarser TTT stride (`max(eval_stride, seq_len//4)`) keeps it within 10-min budget
   - After each SGD update: `p += λ(p₀ - p)` prevents drift
   - Only adapts MLP weights in last N blocks (`TTT_EVAL_ADAPT_LAST_N=3`)
   - Params restored to original after TTT eval
2. **Reptile meta-learning** (last 10% of training):
   - Active when `elapsed_frac >= meta_ttt_start_frac` (default 0.90)
   - Inner loop: K SGD steps on training batches (simulating TTT)
   - Reptile interpolation: `θ ← θ₀ + ε(θ_inner - θ₀)` (default ε=0.3)
   - Only modifies MLP params in last N blocks (same set as eval TTT)
   - Wallclock-aware: aborts inner loop if cap approaching

---

## What Was Left Out

### From #295 (not included)
- **XSA (Exclusive Self Attention)**: #302 has XSA too, but adding it to #281's attention module would break flash_attn_3 compatibility and require significant restructuring. Skipped to stay under 1500 lines and avoid destabilizing #281's working attention path.
- **Sliding window eval variant from #295**: kept #281's version (functionally equivalent, slightly different batching logic)

### From #302 (not included)
- **Pre-Q/K RMSNorm**: #302 applies an extra `F.rms_norm` before c_q/c_k projections. This conflicts with #281's placement of RMSNorm (which applies it after reshape, on the head dimension). Merging would require restructuring CausalSelfAttention and retuning QK gain. Left out.
- **Simple eval (non-sliding)**: #302 uses `eval_val_simple` during training, then `eval_val_ttt` at the end. Combined_v1 uses #281's `eval_val` during training (equivalent) and `eval_val_ttt` + `eval_val_sliding` at the end.
- **Safety checkpoint** (`ckpt_every`): omitted to save lines; not critical for submission.

---

## Key Design Decisions

1. **QAT per-instance not per-class**: #281's `CastedLinear` uses a class-level `_qat_enabled` bool. #295 uses `_qat_levels` as a class variable. Combined uses `qat_levels` as a per-instance `int` (set in `__init__`) — this is safer because eval/inference won't accidentally quantize, and future uses can set it per-layer without interference.

2. **Backout index**: PR #295's backout captures `x` at layer `num_layers // 2`. Combined keeps this but integrates it cleanly into `_run_layers` alongside U-Net skips from #281.

3. **zstd is required**: Following #302, we hard-import `zstandard` at the top. If it's missing the script fails immediately with a clear error rather than silently falling back to zlib (which produces a larger artifact that may exceed 16MB).

4. **Reptile + normal step**: After Reptile inner loop, the combined script does an additional normal training step (grad accumulation + apply_optimizers). This matches #302's pattern and ensures the optimizer momentum buffers stay current.

---

## Hyperparameter Summary (env vars)

| Variable | Default | Note |
|---|---|---|
| `QAT_ENABLED` | 1 | QAT with STE (int5 MLP / int6 attn) |
| `BACKOUT_ENABLED` | 1 | Learned residual subtraction |
| `BACKOUT_INIT` | 0.2 | Initial backout scale |
| `BIGRAM_VOCAB_SIZE` | 10240 | #295 larger bucket |
| `PRUNE_FRAC` | 0.08 | 8% magnitude pruning |
| `TTT_EVAL_LR` | 2e-3 | Online TTT learning rate |
| `TTT_EVAL_DECAY` | 0.02 | Decay prior λ |
| `TTT_EVAL_ADAPT_LAST_N` | 3 | Last N blocks for TTT |
| `META_TTT_ENABLED` | 1 | Reptile meta-learning |
| `META_TTT_START_FRAC` | 0.90 | Start Reptile at 90% training |
| `META_TTT_EPSILON` | 0.3 | Reptile interpolation step |
| `META_TTT_INNER_STEPS` | 1 | Inner loop iterations |
| `META_TTT_INNER_LR` | 2e-3 | Inner loop LR |
| `SWA_EVERY` | 50 | SWA collection interval (#281 aggressive) |
| `WARMDOWN_ITERS` | 1200 | LR warmdown steps |
| `EVAL_STRIDE` | 64 | Sliding window stride |
