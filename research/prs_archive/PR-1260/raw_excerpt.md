# PR 1260 — Record: MuonEq-R + Depth Recurrence + Mixed Int5/Int6 GPTQ

**Author:** dexhunter (per batch metadata)
**Claimed BPB:** 1.0929 (3-seed mean, std 0.0009) — sliding BPB
**Artifact size:** 15,958,277 bytes mean (max 15,981,324)
**Seeds:** 1337, 42, 0
**Hardware:** 8xH100 80GB SXM, PyTorch 2.9.1+cu128, 590s train + ~83s eval

## Files retrieved
- `records__track_10min_16mb__2026-04-02_MuonEqR_DepthRecurrence_MixedQuant__README.md`
- `records__track_10min_16mb__2026-04-02_MuonEqR_DepthRecurrence_MixedQuant__submission.json`
- `records__track_10min_16mb__2026-04-02_MuonEqR_DepthRecurrence_MixedQuant__train_gpt.py`

## Run command (from README)
```
NCCL_NET=Socket DATA_DIR=./data SEED=1337 MIXED_QUANT=1 N_INT6_LAYERS=60 RECUR_LAYERS=4,5 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Claimed changes (from README, verbatim)

Built on PR #1218 (clarkkev, 1.09785).

Three additions vs #1218:
1. **MuonEq-R** — Row-normalizes gradient matrices before Newton-Schulz orthogonalization in Muon. Zero-byte cost, ~0.001 BPB improvement.
2. **Depth Recurrence** — Layers 4 and 5 repeated once after initial forward pass (virtual layers 12-13 on 11 physical). MLP weights fully shared during recurrence (REPEAT_UNTIE_MLP=none), zero extra params. Activated at step 3000 with 20-step linear warmup. ~0.003 BPB improvement.
3. **Mixed Int5/Int6 GPTQ** — Hessian-based sensitivity ranking; 60 most sensitive layers keep int6 (clip_range=31), 6 least sensitive get int5 (clip_range=15). Combined with full GPTQ and brotli-11.

Everything else carried from PR #1218: 4096 SP BPE, 4.0x MLP, WD 0.085, Full Hessian GPTQ, XSA-all-11, BigramHash(2816x160), sigmoid-gated skip, soft-round QAT, split-LR, brotli-11 + byte shuffle, EMA (0.997).

Rule compliance: No TTT, no SLOT.
