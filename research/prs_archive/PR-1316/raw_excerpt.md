# PR 1316 — Full-Depth MLP Megakernel + Fused Attention Preprocessing

**Author:** Adarsh Reddy Balanolla (AR6420)
**Claimed BPB:** 1.13096275 (1 seed — seed 1337; val_loss 1.90957818)
**Artifact size:** 15,597,863 bytes (~15.6 MB); peak memory 15,686 MiB
**Seeds:** 1337 only (additional seeds blocked by compute budget)
**Track:** non_record_16mb
**Steps:** 4,917 at 122.0 ms/step
**Base PR:** 1019

## Files retrieved
- `records__track_non_record_16mb__2026-04-03_MegakernelFusion_TileEngine__README.md`
- `records__track_non_record_16mb__2026-04-03_MegakernelFusion_TileEngine__submission.json`
- `records__track_non_record_16mb__2026-04-03_MegakernelFusion_TileEngine__train_gpt.py`

## Environment variables (from README run command)
BIGRAM_VOCAB_SIZE=3072, BIGRAM_DIM=112, WARMDOWN_ITERS=4000, TARGET_MB=15.9, SEED=1337

## Claimed changes (from README, verbatim)

> Full-depth MLP megakernel: 5 operations (RMSNorm -> gate projection -> LeakyReLU^2 -> down projection -> residual) fused into 1 Triton kernel. The 1536-dim intermediate is never written to HBM — processed via tiled register accumulation in BLOCK_K=64 chunks. Deeper fusion than PR #1072.

> Attention preprocessing fusion: QK RMSNorm + partial RoPE + q_gain fused into 2 Triton kernels, down from 6+. 41% memory reduction (1562 MiB vs 2656 MiB on RTX 5070 Ti).

> Near-perfect numerical accuracy: MLP cos_sim=0.99998, attention Q/K cos_sim=0.99999. H100 autotune: BLOCK_M=32, BLOCK_K=64, nw=8.

> What Didn't Work: Step time 15% slower on consumer GPU (451.9ms vs 392.7ms); 41% slower on H100 (122ms vs 86.7ms); fully fused attention preprocessing blocked by Triton block-tensor model limitation on RoPE half-dim splitting.

> Same as PR #1019 base: 11 layers (512d, 8 GQA / 4 KV), MLP 3x (1536) LeakyReLU(0.5)^2, XSA all 11, BigramHash 3072 x 112, partial RoPE 16/64, VE128 layers 9-10, EMA(0.997)+Tight SWA(50), full Hessian GPTQ int6 AR self-gen, LZMA preset=9, Parallel Muon + Parameter Banking.
