# Record candidate: StageB v2 Scalar-Control Quant + LQER Top-1 + CaseOps Phased TTT

**val_bpb: 1.06099764** (seed 42 single run) | **15,995,233 bytes** | 8xH100 SXM | TTT eval

This is a conservative under-cap StageB v2 run. Under a single-run ordering its seed-42 BPB is below the accepted PR #1855 leaderboard row mean of 1.06107587, but it is not a better 3-seed submission and is not better than PR #1855's best individual seed. Multi-seed confirmation was run after the fact; the 3-seed mean is not claimed as a top-2 result because seed 999 was weaker.

## Results

| Seed | Pre-quant val_bpb | Post-quant val_bpb | Post-TTT val_bpb | Artifact bytes | TTT eval timer | Process wall |
|------|-------------------|--------------------|------------------|----------------|----------------|--------------|
| 42   | 1.06467457 | 1.07393374 | **1.06099764** | 15,995,233 | 524.630s | 713.87s |
| 314  | 1.06457879 | 1.07419498 | **1.06100570** | 15,995,151 | 517.960s | 708.90s |
| 999  | 1.06634471 | 1.07587634 | **1.06247161** | 15,986,471 | 454.766s | 581.81s |
| Mean | | | **1.06149165** | | | |

Seed-999 TTT-only rescue attempts improved the 3-seed mean slightly but did not clear the accepted top-2 mean target:

| Variant | Seed 999 val_bpb | 3-seed mean | TTT eval timer | Process wall |
|---------|------------------|-------------|----------------|--------------|
| rank96 p2000 beta999 wd1 | 1.06244678 | 1.06148337 | 527.605s | 711.25s |
| rank128 p2000 beta999 wd1 | 1.06243525 | 1.06147953 | 554.796s | 734.67s |

## Key Changes Versus Accepted #1855 Lineage

- Brotli-only self-contained compression path; no `lrzip`, `apt-get`, or external binary runtime dependency.
- Scalar/control quantization enabled for model scales, skip gates, residual mixes, and related scalar tensors.
- `LQER_TOP_K=1`, `EMBED_CLIP_SIGMAS=15.0`, `WARMDOWN_FRAC=0.82`, `QK_GAIN_INIT=5.125`, and `SPARSE_ATTN_GATE_SCALE=0.75`.
- Phased score-first LoRA TTT with rank 80, prefix 2500 docs, `TTT_BETA2=0.99`, and `TTT_WEIGHT_DECAY=0.5`.
- `NGRAM_MIX_ALPHA=0`; no byte PPM, no casefold-only path, and no validation-time n-gram cache.

## Compliance Notes

- Artifact size: all scored artifacts are under the decimal 16,000,000-byte cap.
- Training timing caveat: the train loop stopped at the 600s wallclock cap on 8xH100, but seed 42 artifact production including serialization/GPTQ logged `1095.421s`. If reviewers count full artifact production as train time, this candidate is not a clean official record.
- TTT: phased score-first LoRA TTT scores before update, uses a normalized softmax distribution, and does not use byte PPM or an n-gram cache.
- Eval timing caveat: the script-reported TTT eval timers are under 600s, but the shell `real` times for seed 42, seed 314, and the seed999 rescue variants exceed 600s because compile/warmup and process overhead are included. This should be disclosed in the PR.
- Dependency caveat: the Runpod research image needed `brotli` and `python-minifier` installed for the experiment. Final submission should include `requirements.txt` or otherwise ensure those Python dependencies are explicit.

## Reproduction

```bash
DATA_PATH=./datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 \
NGRAM_MIX_ALPHA=0 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_SHARDS=80 \
NPROC_PER_NODE=8 \
SEED=42 \
WARMDOWN_FRAC=0.82 \
QK_GAIN_INIT=5.125 \
EMBED_CLIP_SIGMAS=15.0 \
MLP_CLIP_SIGMAS=11.75 \
MATRIX_CLIP_SIGMAS=12.65 \
ATTN_CLIP_SIGMAS=13.0 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_TOP_K=1 \
SCALAR_QUANT_ENABLED=1 GATED_ATTN_QUANT_GATE=1 \
SMEAR_GATE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_SCALE=0.75 \
TTT_ENABLED=1 TTT_EVAL_ONLY=0 TTT_LORA_RANK=80 PHASED_TTT_PREFIX_DOCS=2500 \
TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_CHUNK_SIZE=48 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Files

- `train_gpt.py` - submission training/eval script used for this candidate.
- `submission.json` - structured metadata.
- `train_seed42.log`, `ttt_seed42.log`, `ttt_seed314.log`, `ttt_seed999.log` - evidence logs.
- `runpod_terminal_summary.json` - terminal Runpod/Flywheel summary for the seed confirmation and rescue attempts.
- `requirements.txt` - explicit Python dependencies observed on the Runpod template.
- `lossless_caps.py`, `prepare_caseops_data.py`, and `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` - CaseOps support files matching the accepted #1855 packaging pattern.

## Attribution

This is accepted-family work, not a copy of a pending open PR. It builds on the merged/accepted CaseOps + phased TTT lineage, especially PR #1855's accepted stack, while removing the lrzip final dependency and retuning scalar-control quantization and TTT settings. PufferLib was inspected during the final run, but no PufferLib code was copied into this submission.
