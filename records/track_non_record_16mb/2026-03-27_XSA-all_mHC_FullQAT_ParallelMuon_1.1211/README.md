# Non-record: XSA-all + mHC + Full QAT + Parallel Muon

**val_bpb: 1.1211** (seed 1337) | 15.95 MB | 8xH100 SXM, 600s train + 482s eval

## Method

Three changes on top of PR #549's LeakyReLU_LegalTTT_ParallelMuon stack:

| Change | Impact | Extra params |
|--------|--------|-------------|
| **XSA on all 11 layers** | Cross-sequence attention from layer 0 instead of last 4. Forces cross-position mixing earlier. | 0 |
| **Manifold-constrained Hyper-Connections (mHC)** | Learnable alpha/beta residual mixing per block with norm constraint (alpha^2 + beta^2 = 2). Each layer learns how much of its own output vs input to keep. | 22 (2 per layer) |
| **Full-training QAT** | Int6 fake quantization from step 1 (LATE_QAT_THRESHOLD=1.0) instead of late QAT. Model learns quantization robustness throughout training. | 0 |

## Results (seed 1337)

| Metric | Value |
|--------|-------|
| Training steps | 6558 |
| Step avg | 91.5 ms |
| Post-EMA val_bpb | 1.1389 |
| Int6 roundtrip | 1.1463 |
| Sliding window (stride=64) | 1.1229 |
| **Legal TTT** | **1.1211** |
| Artifact size | 15.95 MB |

## Reproduction

```bash
RUN_ID=mhc_xsa_s1337 SEED=1337 \
TTT_ENABLED=1 QAT_ENABLED=1 LATE_QAT_THRESHOLD=1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

XSA_LAST_N defaults to 11 in this script. No other env vars needed.

## Environment

- torch 2.11+cu126, flash-attn-3, cuda-toolkit 13.0.2
- RunPod 8xH100 SXM 80GB, driver 580+
