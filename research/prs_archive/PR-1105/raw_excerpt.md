# PR 1105 — Fused MLP (Triton+CUTLASS EVT) + SP4608 + Fast Causal N-Gram

**Author:** Abay Bektursun
**Branch date:** 2026-04-02
**Claimed BPB:** 1.0962 (3-seed mean, sliding eval)
**Artifact size:** ~15.79 MB
**Seeds:** 42, 314, 1337
**Hardware:** 8×H100 SXM, 600s

## Files retrieved
- `README.md`
- `records__track_10min_16mb__2026-03-29_FusedMLP_Brotli_Memmap__README.md`
- `records__track_10min_16mb__2026-03-29_FusedMLP_Brotli_Memmap__train_gpt.py`
- `records__track_10min_16mb__2026-03-29_FusedMLP_Brotli_Memmap__submission.json`

## Claimed changes (from README, verbatim)
"## Changes vs our #1019

### 1. Fused MLP Kernels: Triton TMA Forward + CUTLASS EVT Backward
Forward (Triton TMA): Fuses `F.linear(x, up_w) → LeakyReLU(0.5) → square` into a single kernel. The 302MB intermediate never touches HBM.
Backward (CUTLASS EVT): Fuses `(go @ down_w.T) * act_grad` into a single CUTLASS 3.x kernel via Epilogue Visitor Tree.

### 2. Fast Causal N-Gram Tilt & Subword Certainty (~0.0025 BPB, ~295× speedup)
Replaces the old eval-time n-gram mixing path with a fast, legal, single-pass causal n-gram tilt system. The n-gram is no longer treated as a second language model. Instead, it acts as a sparse auxiliary memory that proposes a hinted token from the strict prefix, while the neural model remains the full normalized distribution.

### 3. Brotli-11 Compression (replaces LZMA-9) — −581 KB (−5.9%)
### 4. Memmap Multi-Shard Data Pipeline + GPU Prefetch
### 5. MLP 3.5× (1536 → 1792 hidden dim) — +2.88M params, −0.003 BPB
### 6. LR Floor (0.05) — ~0.001 BPB
### 7. Vocab 4608 — −0.0007 BPB, freed space for all-int6"
