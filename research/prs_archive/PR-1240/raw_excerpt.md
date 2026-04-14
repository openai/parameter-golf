# PR 1240 — Full GPTQ + Score-First TTT + SLOT (record) AND SLOT Causal Dependence Analysis (non-record)

**Author:** Andrew Baggio (2026-04-01)
**Claimed BPB (record attempt):** 1.1064 (3-seed mean, std=0.0004) on seeds 1337, 42, 7
**Hardware:** 8xH100 SXM, eval ~557s

## Files retrieved
- `records__track_10min_16mb__2026-04-01_FullGPTQ_ScoreFirstTTT_SLOT_8xH100__README.md`
- `records__track_10min_16mb__2026-04-01_FullGPTQ_ScoreFirstTTT_SLOT_8xH100__train_gpt.py`
- `records__track_10min_16mb__2026-04-01_SLOT_Causal_Dependence_Analysis__README.md`
- `records__track_10min_16mb__2026-04-01_SLOT_Causal_Dependence_Analysis__prove_slot_causal_violation.py`

## Run command (from record README)
```
SEED=1337 TTT_ENABLED=1 TTT_EPOCHS=3 TTT_LR=0.002 TTT_CHUNK_TOKENS=65536 SLOT_ENABLED=1 SLOT_STEPS=8 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Claimed changes (from READMEs, verbatim)

Record PR combines:
- Full Hessian GPTQ (from PR #1019)
- Score-first chunked TTT: 65K-token chunks, 3 epochs, lr=0.002, first 2 blocks frozen (-0.003 BPB, ~302s)
- SLOT delta optimization: 8 AdamW steps, lr=0.005, per-batch delta reset (-0.010 BPB, ~255s)
- Architecture: PR #1184 stack (11L LeakyReLU(0.5)^2, d=512, 4 KV GQA, MLP 3x, BigramHash(2816,112), SmearGate, XSA4, Partial RoPE, EMA, SWA, Late QAT, OrthoInit, VE128)

SLOT Causal Dependence Analysis companion document:
> SLOT optimizes a delta vector using target tokens, then scores those same targets with the optimized delta. This means the prediction at position t depends on tokens beyond x_1..x_{t-1} — a causal dependence violation.

Empirical proof via minimal 2-layer random-weight transformer:
- Shared delta: Max NLL change from future token flip 0.2557, self-prediction advantage +0.2382, 240/240 cross-position violations (100%).
- Per-sample delta + logit bias (PR #1229): 0.7744, +0.7255, 240/240 (100%).
- Author flagged his own submission (PR #1209) and requested organizer ruling from @0hq @valerio-oai.
