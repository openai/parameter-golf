# PR 1285 — MuonEq-R + Depth Recurrence + WD=0.090 + All-Int6 GPTQ

**Author:** dexhunter
**Claimed BPB:** 1.0912 (3-seed mean, std 0.0009; seeds 42, 0, 1337; per-seed 1.09057, 1.09084, 1.09230)
**Artifact size:** ~15.96 MB (15,962,993 bytes mean); code 21,396 bytes
**Seeds:** 42, 0, 1337
**Track:** 10min_16mb
**Val_loss:** 2.51064 nats (3-seed mean)
**Base PR:** 1218

## Files retrieved
- `records__track_10min_16mb__2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6__README.md`
- `records__track_10min_16mb__2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6__submission.json`
- `records__track_10min_16mb__2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6__train_gpt.py`

## Environment variables (from README run command)
NCCL_NET=Socket, DATA_DIR=./data, SEED=42|0|1337, MIXED_QUANT=1, N_INT6_LAYERS=66, MUON_WD=0.090, EMBED_WD=0.090, RECUR_LAYERS=4,5

## Claimed changes (from README, verbatim)

> Changes from PR #1218: Optimizer Muon -> MuonEq-R (row-norm before NS5); depth recurrence None -> Layers 4,5 repeated; weight decay 0.085 -> 0.090; mixed quantization No -> All int6 (66/66 layers).

> The critical insight: higher weight decay (0.090 vs 0.085) produces smaller weights that compress 5% better under brotli-11, creating enough artifact headroom to keep ALL 66 layers at int6 precision (vs 60-61 int6 in previous PRs).

> 1. WD=0.090 — Higher WD reduces weight magnitudes, improving brotli-11 compression by ~5%. Creates ~280K bytes of artifact headroom. 2. All-Int6 GPTQ — clip_range=31. 3. MuonEq-R — Row-normalizes gradient matrices before Newton-Schulz orthogonalization. Zero-byte cost, ~0.001 BPB improvement. 4. Depth Recurrence — Layers 4,5 repeated with fully shared MLP (zero extra params). ~0.003 BPB improvement.

> Architecture: 11 layers + 2 virtual (depth rec on 4,5), d_model=512, MLP 4x (2048), 8 heads / 4 KV heads, 4096 SP BPE vocab, BigramHash(2816x160), sigmoid-gated skips + soft-round QAT, EMA 0.997. Train 590s, eval ~83s, 8xH100 SXM. No TTT, no SLOT.
