# PR 198 — 11-Layer Int6 + WD=0.04 + SWA + FA3

**Author:** Jack Princz (jfprincz)
**Branch created:** 2026-03-20
**Claimed BPB:** 1.13178 (sliding window stride=64, seed 1337); mean 1.1326 across 3 seeds
**Artifact size:** 15,689,380 bytes (15.7 MB, int6+zstd-22)
**Seeds:** 1337, 42, 2025

## Files retrieved
- `records__track_10min_16mb__2026-03-20_11L_Int6_MLP3x_WD04_SmearBigram2k_1.1318__README.md`
- `records__track_10min_16mb__2026-03-20_11L_Int6_MLP3x_WD04_SmearBigram2k_1.1318__submission.json`
- `records__track_10min_16mb__2026-03-20_11L_Int6_MLP3x_WD04_SmearBigram2k_1.1318__train_gpt.py`

## Note
The task description labeled PR #198 as "XSA origin, 11L EfficientPartialXSA, unnir, 1.1307" but the PR's actual content is jfprincz's 11L Int6+WD+SWA record (1.1318 / dir "2026-03-20_11L_Int6_MLP3x_WD04_SmearBigram2k_1.1318"). The EfficientPartialXSA record by Vadim Borisov ("unnir") is a separate commit (a81f85b) not in PR #198.

## Claimed changes (from README, verbatim)

> ### What's new
> 1. 11 transformer layers (was 9). Two extra layers add 4.4M parameters. The main driver of the BPB gain. Fits under 16 MB thanks to int6 compression headroom.
> 2. Weight decay 0.04. Applied to both Muon (decoupled WD on matrix params) and AdamW (on embeddings/scalars). Shrinks weight magnitudes, improving int6 quantization tolerance and zstd compression ratio.
> 3. Stochastic Weight Averaging. Collects ~8 checkpoints during warmdown (when LR scale < 0.5, every 200 steps) and averages them before quantization.
> 4. Sliding window stride=64 (was 256). Each scored token now has nearly full 2048-token context. ~0.002 BPB gain over stride=256.
> 5. Bigram vocab 2048 (was 4096). Halved the bigram hash table to save ~300 KB artifact space with <0.001 BPB cost.
> 6. Tuned LRs. matrix_lr=0.025, scalar_lr=0.025, tied_embed_lr=0.035.
> ### Carried from PR #164
> - Orthogonal + muP-scaled init on all large matrices
> - 3x MLP (hidden=1536), relu² activation
> - Int6 mixed quantization + zstd-22 (int6 on MLP+attention, int8 on embeddings)
> - SmearGate, Bigram Hash Embedding, FlashAttention 3
> - Sequence length 2048 with NTK-aware RoPE
> - Muon optimizer, momentum 0.99 with warmup, warmdown 3000 iters, grad clip 0.3

Pre-quant 1.1432; int6 roundtrip 1.1543; 7412 steps at 81ms/step on 8xH100.
