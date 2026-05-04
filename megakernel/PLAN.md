# Mega-Kernel: 3-Day Sprint Plan

## Competition Context (as of 2026-05-04)
- Current SOTA: PR #2135 at **1.05651 BPB** (PR #2130 + GPTQ_CALIBRATION_BATCHES=32)
- Our non-record: PR #2129 at **1.05874 BPB**
- PR #1138 attempted whole-model megakernel — currently 646ms/step vs 120ms/step (5× SLOWER)
- PR #2155 tried Mamba3 SSM hybrid — non-record

## What "Mega-Kernel" Means Here
NOT fusing the entire model. TARGETED fusion of the most memory-bandwidth-limited operations.

## Root Cause Analysis: What's Slow?
Per Block.forward (PR #1855 base):
```
# Attention path (5 SEPARATE kernel launches per layer):
x_normed = attn_norm(x_in)          # RMSNorm  [512-dim] — UNFUSED
x_normed = x_normed * scale          # scalar mul — UNFUSED
q = F.linear(x_normed, q_w)         # [M,512]→[M,512] — UNFUSED
k = F.linear(x_normed, k_w)         # [M,512]→[M,256] — UNFUSED
v = F.linear(x_normed, v_w)         # [M,512]→[M,256] — UNFUSED
# ... q/k head-norm, RoPE, FA3 ...
out = F.linear(y, out_w)            # [M,512]→[M,512] — UNFUSED

# MLP path (2 SEPARATE kernel launches per layer):
mlp_in = mlp_norm(x_out)            # RMSNorm [512-dim] — UNFUSED
mlp_in = mlp_in * scale             # scalar mul — UNFUSED
# then FusedMLP (up→LeakyReLU²→down) — ALREADY FUSED
```

## Memory Bandwidth Math
Per GPU (8-GPU DDP): ~73K tokens per GPU per step
x tensor per GPU: 73K × 512 × 2 bytes = ~75MB

Current cost per forward pass (per GPU):
- 11 layers × 2 norm+linear pairs = 22 pairs
- Each pair: 1 RMSNorm read (75MB) + 1 write (75MB) + 3 linear reads (75MB each) = 375MB
- Total: 22 × 375MB = 8.25GB HBM traffic just for norm intermediates

After fusion (fused RMSNorm + linear):
- Each pair: 2 reads of x (150MB) + no write of normed intermediate
- Total: 22 × 150MB = 3.3GB HBM traffic
- **SAVINGS: 4.95GB per forward pass**

At H100 HBM bandwidth 3.35 TB/s: **saves ~1.5ms per forward pass**
With backward (2× forward): **saves ~3ms per step total**
At current 84ms/step: **saves ~3.5% → ~250+ more steps in 600s**

Conservative estimate (60% cache hit rate): ~150 more steps → ~0.001-0.002 BPB

## The Three Mega-Kernels

### Kernel 1: fused_rmsnorm_mlp (Day 1 — extends existing kernel)
**Purpose**: Add RMSNorm to the existing `linear_leaky_relu_square_kernel` in PR #1855
```
Before: mlp_norm(x)*scale → FusedMLP(up_w, down_w)  [3 kernel launches]
After:  FusedRMSNormMLP(x, up_w, down_w, scale)       [1 kernel launch]
```
**Risk**: Low (extends working code)
**Implementation**: Add pre-pass to compute per-row RMS within the existing TMA matmul kernel
**Files**: `kernel1_rmsnorm_mlp.py` (standalone test), then integrate into train_gpt.py

### Kernel 2: fused_rmsnorm_qkv (Day 2 — new kernel)
**Purpose**: Fuse pre-attention RMSNorm + scale + 3-way QKV linear projection
```
Before: attn_norm(x)*scale → [q_proj, k_proj, v_proj]  [5 kernel launches]
After:  FusedRMSNormQKV(x, q_w, k_w, v_w, scale)       [1 kernel launch]
```
**Risk**: Medium (new kernel, larger output)
**Key challenge**: Q=512-out, K=256-out, V=256-out — different sizes need careful tiling
**Files**: `kernel2_rmsnorm_qkv.py`

### Kernel 3: fused_head_norm_rope (Day 2-3 — if time permits)
**Purpose**: Fuse q/k head-dimension RMSNorm + RoPE application
```
Before: rms_norm(q, 64) → apply_rotary_emb(q)  [2 kernel launches × 2 (q+k)]
After:  FusedHeadNormRoPE(q, k, cos, sin)       [1 kernel launch]
```
**Risk**: Medium (RoPE requires trig operations in kernel)

## Full Submission Stack (Day 3)
Base: PR #2130 stack (1.05670 BPB) which includes:
- Token-only n-gram tilt (PR #1514, `TOKEN_ORDER=16, BOOST=2.625`)
- AsymLogit Rescale (`ASYM_LOGIT_RESCALE=1`)
- `MATRIX_LR=0.028, LQER_ASYM_GROUP=32, TTT_LORA_LR=8e-5`
- `GPTQ_CALIBRATION_BATCHES=32` (from PR #2135)

**OUR ADD**: Mega-kernel fusions → more steps in 600s → better pre-quant BPB

## Compliance Checklist (MANDATORY before any compute spend)
- [ ] No external downloads in train_gpt.py during eval
- [ ] All code in single train_gpt.py file
- [ ] Model produces normalized probability distribution BEFORE seeing target
- [ ] TTT is score-first (evaluate chunk THEN train on it)
- [ ] Artifact ≤ 16,000,000 bytes
- [ ] Training ≤ 600 seconds wallclock
- [ ] Eval ≤ 600 seconds wallclock
- [ ] Mega-kernels are pure compute optimizations — no statistical model changes — ALWAYS COMPLIANT

## Key Design Decisions
1. **Use TensorDescriptor (TMA)** — H100-specific hardware feature, already used in PR #1855
2. **Two-pass RMSNorm within kernel** — Pass 1 computes sum(x²) per row, Pass 2 does normalized matmul
3. **Save inv_rms for backward** — needed for RMSNorm gradient computation
4. **Persistent kernel pattern** — same NUM_SMS approach as existing kernel

## Files Created
- `PLAN.md` — this file
- `pr1855_train_gpt.py` — the PR #1855 base code (downloaded)
- `kernel1_rmsnorm_mlp.py` — Kernel 1 standalone implementation + tests
- `kernel2_rmsnorm_qkv.py` — Kernel 2 standalone implementation + tests
- `train_gpt_mega.py` — Full integrated submission (Day 3)
