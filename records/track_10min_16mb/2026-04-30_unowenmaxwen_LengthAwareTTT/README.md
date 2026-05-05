# Single-seed validation: LQER SparseAttnGate BOS-fixed phased TTT stack

**Status:** non-record / record-support package. This folder contains one full from-scratch validated run for seed 42. It should **not** be submitted as a new SOTA record unless seeds 0 and 1234 are also run and the three-seed mean satisfies the current SOTA significance rule.

**Validated seed-42 result:** `val_bpb=1.06018067`, `val_loss=2.32006828`, total artifact size `15,898,824` bytes, TTT eval time `497.6s`.

## Results

| Seed | Steps | Train wallclock logged | Pre-quant val_bpb | Post-quant val_bpb | Post-TTT val_bpb | Artifact bytes | Eval time |
|------|-------|------------------------|-------------------|--------------------|------------------|----------------|-----------|
| 42 | 4,872 | 592.059s | 1.06427086 | 1.07281079 | **1.06018067** | 15,898,824 | 497.6s |

The current validated SOTA folder reports a three-seed mean of `1.06107587` BPB. This seed-42 run alone is not enough to claim a record because the challenge requires enough logs to show a statistically significant improvement of at least `0.005` nats.

## Why this is a non-record submission

This package is intended as a single-seed validation and record-support artifact, not a SOTA claim. It documents that the 2026-04-27 LQER SparseAttnGate BOS-fixed phased TTT stack can be reproduced for seed 42 using the public CaseOps Hugging Face export while staying under the 16 MB artifact cap and the 10-minute training/evaluation limits.

The result is interesting as a reproducibility checkpoint and negative/partial result: seed 42 alone reaches `1.06018067` BPB, but the package intentionally does not claim statistical significance. A SOTA submission would require additional full-seed logs, such as seeds 0 and 1234, and evidence that the three-seed result satisfies the challenge significance rule.

## Compliance Notes

- Artifact size is below the decimal 16 MB cap: `15,898,824 < 16,000,000`.
- TTT evaluation time is below 600s: `497.6s`.
- Training loop wallclock is below 600s: `592.059s`.
- No tokenizer or dataset changes were made.
- CaseOps data was downloaded from the public Hugging Face export with `cached_challenge_fineweb.py`; raw-doc rebuilding was not used.
- TTT-only A/B sweeps were exploration only and are intentionally excluded from the submission files.

## Reproduction

Run from this folder after downloading CaseOps data:

```bash
MATCHED_FINEWEB_REPO_ID=romeerp/parameter-golf-caseops-v1 \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets \
python3 cached_challenge_fineweb.py \
  --variant sp8192_lossless_caps_caseops_v1_reserved \
  --train-shards 80

SEED=42 RUN_ID=seed42_repro ARTIFACT_DIR=artifacts/seed42_repro \
VOCAB_SIZE=8192 \
DATA_PATH=./datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
GLOBAL_TTT_MOMENTUM=0.9 WARMDOWN_FRAC=0.85 BETA2=0.99 \
TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 GPTQ_RESERVE_SECONDS=8.0 GPTQ_CALIBRATION_BATCHES=16 \
VAL_LOSS_EVERY=0 \
GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 SMEAR_GATE_ENABLED=1 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 \
LQER_TOP_K=3 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed42.log
```

## File Notes

- `train_gpt.py` is the executable training/evaluation script.
- `cached_challenge_fineweb.py` downloads the public CaseOps shard export and tokenizer.
- `train_seed42.log` is the full validated run log produced in this session.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` is the tokenizer model used by the run.

## Attribution

This is a reproduction/continuation of the 2026-04-27 LQER SparseAttnGate BOS-fixed stack. The implementation lineage includes PR #1797, PR #1787, PR #1736, PR #1729, PR #1667, PR #1626, PR #1610, PR #1586, PR #1530, PR #1344, PR #493, PR #478, PR #315, and PR #289.
