# Masterplan: Hybrid Mamba-Attention (arch1/mamba-hybrid)

## Objective

Replace most attention layers with Mamba-2 selective state-space layers to fit **18 total layers** (15 Mamba + 3 GQA Attention) in the same 16MB budget that currently holds 11 transformer layers. Target BPB: **1.095-1.115** (vs SOTA 1.1147).

## SOTA Baseline

- **File:** `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py` (2135 lines)
- **BPB:** 1.11473 (3-seed mean), steps ~6922, step_avg ~86.7ms
- **Architecture:** 11L, 512d, 8Q/4KV GQA, 3x MLP, U-Net skip, XSA-all, BigramHash 3072x112
- **Quantization:** Int6 GPTQ + selective ±1 pruning + LZMA9

## Implementation Phases

### Phase 1: Smoke Test (Day 1, ~4 hours)

**Goal:** Confirm Mamba is viable on H100 before committing.

1. Copy SOTA baseline to working file
2. Add minimal `MambaBlock` class (sequential scan, pure PyTorch)
3. Replace 1 attention layer with 1 Mamba layer
4. Profile step time on 1xH100
5. **Go/No-Go:** Step time must be ≤100ms. If >150ms, abort this architecture.

### Phase 2: Core Implementation (Days 2-3)

**Goal:** Full hybrid model training end-to-end.

1. Implement complete `MambaBlock` with d_state=32, expand=1.5, d_conv=4
2. Build hybrid GPT: 15 Mamba + 3 GQA Attention layers
3. Modify `GPT.__init__` — banks sized for attention layers only, `mamba_blocks` ModuleList
4. Modify `GPT.forward` — dispatch loop (mamba vs attention)
5. Modify `GPT.forward_logits` — same dispatch
6. Set up Muon/Adam optimizer split for Mamba params
7. Verify end-to-end training on 1xH100

### Phase 3: Optimization & Kernels (Days 3-4)

**Goal:** Step time ≤85ms for maximum training steps.

1. Install `mamba-ssm` package, test CUDA/Triton selective scan kernels
2. If mamba-ssm doesn't work: write custom Triton kernel or use chunked parallel scan
3. Profile and optimize: fuse operations, minimize memory overhead
4. **Target:** ≤85ms/step → ~7000 steps in 600s

### Phase 4: Full Stack (Days 5-6)

**Goal:** Complete training pipeline with all SOTA techniques.

1. Modify `_unbank_state_dict` / `_rebank_state_dict` for Mamba params
2. Build `_HessianGPT` variant with Mamba layers for GPTQ Hessian collection
3. Extend `_classify_param` with `"mamba"` category
4. Apply GPTQ int6 quantization to Mamba weight matrices
5. Verify artifact fits ≤16MB after LZMA compression
6. Add EMA(0.997) + SWA(50) + Late QAT for Mamba layers

### Phase 5: Ablation & Tuning (Days 5-6, parallel)

**Goal:** Find optimal hybrid configuration.

| Experiment | Variable | Values |
|-----------|----------|--------|
| A1.1 | All-Mamba vs hybrid | 18M+0A vs 15M+3A |
| A1.2 | Number of attention layers | 1, 2, 3, 5 |
| A1.3 | Mamba d_state | 16, 32, 64 |
| A1.4 | Mamba expand | 1.0, 1.5, 2.0 |
| A1.5 | Attention position | bottom, top, interleaved |
| A1.6 | Total layers | 12, 15, 18, 20 |

All ablations: single seed (42), 1xH100, 10min each.

### Phase 6: Final Evaluation (Day 7)

**Goal:** 3-seed validation for record submission.

1. Run with seeds {42, 1337, 2025} on 8xH100
2. Collect: val_bpb, val_loss, artifact_bytes, step_avg_ms, total_steps
3. Welch's t-test vs SOTA (1.11473 ± 0.00035)
4. **Submission criteria:** mean BPB ≤ 1.1097, p < 0.01

## Key Files to Modify

```
train_gpt.py (copy from SOTA baseline)
├── Hyperparameters: +mamba_layers, mamba_d_state, mamba_d_conv, mamba_expand, num_layers=18
├── +NEW: class MambaBlock (~120 lines after line 678)
├── GPT.__init__: banks for n_attn only; mamba_blocks ModuleList
├── GPT.forward/forward_logits: dispatch loop
├── Optimizer setup: mamba_matrix_params → Muon, mamba_scalar_params → Adam
├── _unbank/_rebank_state_dict: handle mamba params
├── _HessianGPT: add mamba variant
├── _classify_param: add "mamba"
├── mixed_quantize_int6: include "mamba" in int6_cats
└── requirements.txt: +mamba-ssm, +causal-conv1d
```

## Parameter Budget (Option B: d_state=32, expand=1.5)

| Component | Params | Est. LZMA bytes |
|-----------|--------|----------------|
| 15x MambaBlock (~1.27M each) | 19.0M | ~10.7M |
| 3x GQA Attention + MLP | 7.9M | ~3.9M |
| Embedding + BigramHash + controls | 0.9M | ~0.7M |
| **Total** | **~27.8M** | **~15.3M + code ≈ 15.4M** |

## Go/No-Go Decision Points

| Checkpoint | Criteria | Action if Fail |
|-----------|----------|----------------|
| Day 1 step time | ≤100ms/step on 1xH100 | Abort arch1 entirely |
| Day 3 training curve | Loss decreasing smoothly by step 1000 | Debug init/LR; reduce Mamba count |
| Day 5 pre-quant BPB | ≤1.120 on 8xH100 | Reduce Mamba layers; widen model |
| Day 7 post-quant 3-seed | mean ≤1.115 | Not a record; evaluate cross-pollination with TTT |

## Dependencies

```
pip install mamba-ssm>=2.2.0 causal-conv1d>=1.4.0
```

Fallback: Pure PyTorch sequential scan if CUDA kernels fail.

## Risk Summary

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| CUDA kernel incompatibility | 20% | Fatal | PyTorch fallback; custom Triton |
| Step time too slow | 30% | Fatal | Fewer/narrower Mamba layers |
| SSM quality < attention at 512d | 25% | Major | Keep 3+ attention layers |
| Training instability | 15% | Major | Separate LR; careful A_log/D init |
