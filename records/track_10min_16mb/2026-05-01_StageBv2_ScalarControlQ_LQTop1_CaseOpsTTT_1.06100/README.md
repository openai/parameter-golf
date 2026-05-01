# Record candidate: StageB v2 Scalar-Control Quant + LQER Top-1 + CaseOps Phased TTT

**val_bpb: 1.06095913** (seed 42 single run) | **15,995,233 counted submission bytes** | 8xH100 SXM | TTT eval

This is a conservative under-cap StageB v2 run. Under a single-run ordering its final seed-42 BPB is below the accepted PR #1855 leaderboard row mean of 1.06107587, but it is not claimed as a better 3-seed submission. Multi-seed confirmation was run after the fact; neither the official-reference seed set nor the auxiliary seed set is claimed as a top-2 3-seed result.

## Results

| Seed | Pre-quant val_bpb | Post-quant val_bpb | Post-TTT val_bpb | Counted bytes | TTT eval timer | Process wall |
|------|-------------------|--------------------|------------------|----------------|----------------|--------------|
| 42 final rank128 p2000 | 1.06467457 | 1.07393374 | **1.06095913** | 15,995,233 | 546.432s | 733.28s |
| 42 original rank80 p2500 | 1.06467457 | 1.07393374 | 1.06099764 | 15,995,233 | 524.630s | 713.87s |
| 0    | 1.06542461 | 1.07509476 | **1.06162560** | 15,992,241 | 526.704s | 712.18s |
| 1234 | 1.06581926 | 1.07530838 | **1.06220754** | 15,994,552 | 461.001s | 584.04s |
| Mean, official-reference seeds 42/0/1234 with final seed42 | | | **1.06159742** | | | |

Auxiliary forced-seed confirmation used before the official-reference seed0/seed1234 run:

| Seed | Pre-quant val_bpb | Post-quant val_bpb | Post-TTT val_bpb | Artifact bytes | TTT eval timer | Process wall |
|------|-------------------|--------------------|------------------|----------------|----------------|--------------|
| 314  | 1.06457879 | 1.07419498 | **1.06100570** | 15,995,151 | 517.960s | 708.90s |
| 999  | 1.06634471 | 1.07587634 | **1.06247161** | 15,986,471 | 454.766s | 581.81s |
| Mean, final seed42/314/999 | | | **1.06147881** | | | |

Seed-999 TTT-only rescue attempts improved the auxiliary 3-seed mean slightly but did not clear the accepted top-2 mean target:

| Variant | Seed 999 val_bpb | 3-seed mean | TTT eval timer | Process wall |
|---------|------------------|-------------|----------------|--------------|
| rank96 p2000 beta999 wd1 | 1.06244678 | 1.06147054 | 527.605s | 711.25s |
| rank128 p2000 beta999 wd1 | 1.06243525 | 1.06146669 | 554.796s | 734.67s |

The final seed-42 rank128/p2000 rerun improves the best available mixed-profile mean to `1.06146669`, still above the accepted top-2 target of about `1.06141`.

## Key Changes Versus Accepted #1855 Lineage

- Brotli-only self-contained compression path; no `lrzip`, `apt-get`, or external compression binary runtime dependency.
- Scalar/control quantization enabled for model scales, skip gates, residual mixes, and related scalar tensors.
- `LQER_TOP_K=1`, `EMBED_CLIP_SIGMAS=15.0`, `WARMDOWN_FRAC=0.82`, `QK_GAIN_INIT=5.125`, and `SPARSE_ATTN_GATE_SCALE=0.75`.
- Phased score-first LoRA TTT. The training queue used rank 80, prefix 2500 docs, `TTT_BETA2=0.99`, and `TTT_WEIGHT_DECAY=0.5`; the final seed-42 score uses eval-only rank 128, prefix 2000 docs, `TTT_BETA2=0.999`, and `TTT_WEIGHT_DECAY=1.0` on the same under-cap artifact.
- `NGRAM_MIX_ALPHA=0`; no byte PPM, no casefold-only path, and no validation-time n-gram cache.

## Compliance Notes

- Artifact size: all scored runs are under the decimal 16,000,000-byte cap. The final score log reports `total_submission_bytes=15,995,233`; the compressed model file itself is smaller, with the difference accounting for counted code bytes.
- Training timing caveat: the train loop stopped at the 600s wallclock cap on 8xH100, but artifact production including serialization/GPTQ logged over 600s for seed 42 (`1095.421s`), seed 0 (`1100.329s`), and seed 1234 (`821.455s`). If reviewers count full artifact production as train time, this candidate is not a clean official record.
- TTT: phased score-first LoRA TTT scores each batch before its LoRA update, uses a normalized softmax distribution, and does not use byte PPM or an n-gram cache.
- TTT ordering caveat: this scored script keeps a length-sorted global TTT work queue and phase-prefix global updates. A stricter original-validation-order variant was prepared locally but not rerun on 8xH100, so this candidate should not be described as having verified strict-order TTT.
- Eval timing caveat: the script-reported TTT eval timers are under 600s, including the final seed42 rank128 timer at 546.432s, but the shell `real` times for seed 42, seed 0, seed 314, and the seed999 rescue variants exceed 600s because compile/warmup and process overhead are included. This should be disclosed in the PR.
- Dependency caveat: the Runpod research image needed `brotli`, `python-minifier`, and FlashAttention 3's `flash_attn_interface` installed for the experiment. The artifact-size helper shells out through `pyminify`, provided by `python-minifier`, so `requirements.txt` is included explicitly.

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

The final seed42 score was then measured with the saved artifact and:

```bash
ARTIFACT_DIR=./artifacts \
DATA_PATH=./datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 NGRAM_MIX_ALPHA=0 \
SCALAR_QUANT_ENABLED=1 GATED_ATTN_QUANT_GATE=1 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_TOP_K=1 \
EMBED_CLIP_SIGMAS=15.0 MATRIX_CLIP_SIGMAS=12.65 MLP_CLIP_SIGMAS=11.75 ATTN_CLIP_SIGMAS=13.0 \
SMEAR_GATE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_SCALE=0.75 \
TTT_ENABLED=1 TTT_EVAL_ONLY=1 TTT_LORA_RANK=128 PHASED_TTT_PREFIX_DOCS=2000 \
TTT_BETA2=0.999 TTT_WEIGHT_DECAY=1.0 TTT_CHUNK_SIZE=48 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The campaign-level run ID was `runpod_8xh100_stageb_v2_seed42_rank128_final_20260502`; the internal eval log run ID was `stageb_v2_seed42_ttt_rank128_p2000_beta999_wd1_final`.

## Files

- `train_gpt.py` - submission training/eval script used for this candidate.
- `submission.json` - structured metadata.
- `train_seed42.log`, `train_seed0.log`, `train_seed1234.log`, `ttt_seed42.log`, `ttt_seed42_rank128_final.log`, `ttt_seed0.log`, `ttt_seed1234.log`, `ttt_seed314.log`, `ttt_seed999.log` - evidence logs.
- `runpod_seed42_rank128_final_summary.jsonl` - final eval-only score summary.
- `runpod_terminal_summary.json` - terminal Runpod/Flywheel summary for the final seed42 pass plus seed314/seed999 confirmation and rescue attempts.
- `runpod_seed0_1234_summary.json` - terminal Runpod/Flywheel summary for the official-reference seed0/seed1234 confirmation.
- `requirements.txt` - explicit Python dependencies observed on the Runpod template.
- `lossless_caps.py`, `prepare_caseops_data.py`, and `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` - CaseOps support files matching the accepted #1855 packaging pattern.

## Attribution

This is accepted-family work, not a copy of a pending open PR. It builds on the merged/accepted CaseOps + phased TTT lineage, especially PR #1855's accepted stack, while removing the lrzip final dependency and retuning scalar-control quantization and TTT settings. PufferLib was inspected during the final run, but no PufferLib code was copied into this submission.
