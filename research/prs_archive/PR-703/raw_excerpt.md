# PR 703 — Full GPTQ + LeakyReLU² + MiLe + Cache+Backout + 8-bit Muon

**Author:** not stated in record README (PR is a derivative of PR549)
**Claimed BPB:** 1.1171 (3-seed mean, std 0.0003)
**Artifact size:** ~15.95 MB (15,901,230 / 15,962,990 / 15,994,746)
**Seeds:** 2025, 1337, 2024

## Files retrieved
- `records__track_10min_16mb__2026-03-23_FullGPTQ_LeakyReLU_ParallelMuon__README.md`
- `records__track_10min_16mb__2026-03-23_FullGPTQ_LeakyReLU_ParallelMuon__submission.json`
- `records__track_10min_16mb__2026-03-23_FullGPTQ_LeakyReLU_ParallelMuon__train_gpt.py`
- `README.md` (root)

## Environment variables (from run command in README)
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 SEED=1337

## Claimed changes (from README, verbatim)
> Full Hessian GPTQ: Standard GPTQ-lite searches for the best per-row clip percentile — a greedy row-wise optimization. Full GPTQ uses second-order information (the Hessian H = X^T X) to compensate for quantization error across columns. 1) Hessian collection: 256 calibration batches through a non-banked model replica. 2) Column reordering (actorder). 3) Cholesky error compensation. 4) Per-row scale search. Based on IST-DASLab/gptq.

> MiLe Decay (Minimum-Entropy Loss Reweighting): Entropy-based per-token loss reweighting that downweights "easy" tokens (low entropy) and upweights "hard" tokens (high entropy). `weights = (1 - exp(-entropy))^gamma` with `gamma=1.1`, clamped at `min=0.1`. Gradually introduced during warmdown.

> Cache + Backout: Caches the residual stream at layer 7 and uses it for late-layer attention context. Layers 8-10 attend to the cached layer-7 state instead of the current residual. After the final layer, subtracts `backout_lambda * x_cache` (lambda=0.1, learned).

> 8-bit Blockwise Muon Momentum: Quantizes Muon optimizer momentum buffers to int8 with blockwise scaling (block_size=256). Each block stored as int8 + one fp32 scale factor. ~4× memory reduction. Negligible impact on training quality.

> No TTT needed — Full GPTQ alone beats all prior TTT-based submissions. GPTQ improves post-quantization BPB by 0.0216 vs pre-quantization.
