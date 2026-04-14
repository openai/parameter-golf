# PR 549 — LeakyReLU² + Legal Score-First TTT + Parallel Muon

**Author:** abaybektursun
**Claimed BPB:** 1.1194 (3-seed mean, std 0.0006)
**Artifact size:** ~15.95 MB (15,977,386 / 15,876,510 / 15,990,006 per seed)
**Seeds:** 1337, 42, 2025

## Files retrieved
- `records__track_10min_16mb__2026-03-23_LeakyReLU_LegalTTT_ParallelMuon__README.md`
- `records__track_10min_16mb__2026-03-23_LeakyReLU_LegalTTT_ParallelMuon__submission.json`
- `records__track_10min_16mb__2026-03-23_LeakyReLU_LegalTTT_ParallelMuon__train_gpt.py`

## Environment variables (from run command in README)
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 SEED=1337

## Claimed changes (from README, verbatim)
> LeakyReLU(0.5)² activation (-0.003 BPB vs relu²) + legal score-first TTT (PR #461 recipe, 3ep SGD, all blocks unfrozen) + BigramHash(1536) + Parameter Banking + Parallel Muon (PR #399). Built on PR #414 stack.

> One-line activation change that delivers -0.003 BPB: `x = F.leaky_relu(self.fc(x), negative_slope=0.5).square()`. LeakyReLU with slope 0.5 preserves negative gradient flow through the MLP.

> Legal TTT Protocol (PR #461): Val tokens split into 1,893 non-overlapping 32K-token chunks. For each chunk: SCORE under `torch.inference_mode()`, then TRAIN with SGD(lr=0.002, momentum=0.9), 3 epochs, all blocks unfrozen, cosine LR decay, grad clip 1.0.

> Parameter Banking + Parallel Muon (PR #399): 4 contiguous 3D nn.Parameter banks replace 66 separate nn.Linear weights. Batched Newton-Schulz via torch.bmm. DDP removed for banks; async reduce-scatter → local NS → async all-gather. 83.3ms/step vs ~85ms baseline.
