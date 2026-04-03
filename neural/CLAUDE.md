# Neural Track — Agent Protocol

## You are in: NEURAL SOTA (Rascal lineage)
Goal: beat leaderboard #1. Score measured by sliding-window BPB. Lower is better.

## Current leader
```
cat neural/LEADER.md
```
Hash-verified source: `vault/train_gpt_rascal_sota_REAL.py`
SHA256: `0ec1f462ab39fd601b18f2b086f6283a0c8db3d2a9780a92dfb206ec46e067cb`
Run baseline: `bash scripts/sota_now.sh`

## Leg structure
```
neural/YYYY-MM-DD_name/
  hypothesis.md   ← what ONE thing changed, and why
  train_gpt.py    ← copy from leader, then modify
  gate.sh         ← 1-GPU 2000-step gate
  run.sh          ← 8×H100 full run (only after gate passes)
  gate_seed444.log / run results after runs complete
```

## Rules specific to this track
- Source of truth for training code: `vault/train_gpt_rascal_sota_REAL.py`
- Do NOT use `records/track_10min_16mb/2026-03-30_Rascal_8xH100/train_gpt.py`
  as a base — it is a stripped post-hoc copy, not what ran.
- SKIP_GPTQ=1 is the baseline lane. Do not change this without an explicit hypothesis.
- BIGRAM_DIM=128, XSA_LAST_N=11, ROPE_DIMS=16 are the locked architecture params.
- Compile: enabled=1, fullgraph=1. Do not disable without a reason.

## Promotion gate
Beat `1.10986874` BPB on seed 444
→ confirm on seed 300
→ update `neural/LEADER.md`
→ update `vault/README.md` with new hash

## Never
- Modify vault/ files
- Run without committing and pushing the script first
- Change more than one variable vs the parent leg
- Use seed 1337
