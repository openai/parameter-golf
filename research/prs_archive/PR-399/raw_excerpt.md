# PR 399 — Parallel Muon + Parameter Banking

**Author:** abaybektursun
**Claimed BPB:** val_loss 1.1247 (mean of 3 seeds; per-seed int6 sliding val_bpb: 1.1241 / 1.1253 / 1.1247, std 0.0006)
**Artifact size:** 15,830,960 bytes (seed 1337); ~15.8 MB mean
**Seeds:** 1337, 42, 2025

## Files retrieved
- `records__track_10min_16mb__2026-03-22_ParallelMuon_ParameterBanking_82ms__README.md`
- `records__track_10min_16mb__2026-03-22_ParallelMuon_ParameterBanking_82ms__submission.json`
- `records__track_10min_16mb__2026-03-22_ParallelMuon_ParameterBanking_82ms__train_gpt.py`

## Environment variables (from run script in README)

```
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1
MUON_WD=0.04 ADAM_WD=0.04
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64
SEED=1337
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Claimed changes (from README, verbatim)

> Systems optimization: 81.87 ms/step, mean val_bpb 1.1247 (3 seeds), all artifacts under 16 MB. Pure training speed optimization. Model architecture and hyperparameters are unchanged — only the optimizer and weight storage layout are modified.
>
> Two optimizer optimizations replacing 66 sequential individual Newton-Schulz calls with 4 batched operations:
>
> 1. Parameter Banking. 3D nn.Parameter banks replace 66 separate nn.Linear weights: qo_bank (22, 512, 512) — Q + Out projections; kv_bank (22, 256, 512) — K + V projections; mlp_up_bank (11, 1536, 512) — MLP up; mlp_down_bank (11, 512, 1536) — MLP down. Forward: F.linear(x, bank[layer_idx]). Compiled forward+backward verified identical: 72.33ms vs 72.59ms. Standard Newton-Schulz (a=3.4445, b=-4.7750, c=2.0315) batched over banks via torch.bmm.
>
> 2. Parallel Muon (arXiv:2511.07464). DDP removed for bank params. Post-backward communication scheduled explicitly: (1) launch async reduce_scatter for all banks (biggest first); (2) all_reduce + Adam step on small params (while bank RS is in-flight); (3) wait for RS, local batched NS on each GPU's shard, async all_gather.
>
> Why DDP doesn't work with banking: Bank gradients aggregate across all 11 layers → available only at end of backward → zero DDP overlap (+4ms regression). Removing DDP for banks and scheduling communication explicitly restores full overlap.
>
> Per-seed table: 1337 → 81.86ms, 7331 steps, 1.1241, 15,830,960 bytes. 42 → 81.88ms, 7328 steps, 1.1253, 15,819,728 bytes. 2025 → 81.86ms, 7330 steps, 1.1247, 15,796,052 bytes. Mean 81.87ms, 7330 steps, 1.1247 (std 0.0006), ~15.8 MB.
