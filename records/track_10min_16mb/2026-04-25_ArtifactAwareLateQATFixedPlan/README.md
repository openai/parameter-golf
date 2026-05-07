# Artifact-Aware LateQAT Fixed Bit Plan

Single-run result: **val_bpb = 1.1919522011** using exact compiled roundtrip evaluation after mixed-precision artifact compression.

This is intended as an interesting, reproducible non-record submission rather than a current leaderboard SOTA claim. It satisfies the 16MB artifact cap and the 10-minute 8xH100 budget in the recorded run.

## Result

| Metric | Value |
|---|---:|
| final mixed-quant roundtrip `val_bpb` | 1.1919522011 |
| final mixed-quant roundtrip `val_loss` | 2.0125613628 |
| prequant `val_bpb` | 1.1802617551 |
| quant gap | 0.0116904461 |
| artifact bytes | 15,415,044 |
| compact `train_gpt.py` bytes | 79,879 |
| compact artifact+script bytes | 15,494,923 |
| decimal 16MB headroom | 505,077 |
| review helper `train_gpt_human.py` bytes | 5,222 |
| artifact+both Python files bytes | 15,500,145 |
| conservative 16MB headroom including helper | 499,855 |
| artifact+entire record folder bytes | 15,525,003 |
| conservative 16MB headroom including entire folder | 474,997 |
| original managed-run total submission bytes | 15,716,141 |
| train stage | 518.13s |
| quant/compiler stage | 17.12s |
| final 8xH100 eval stage | 53.38s |
| full measured pipeline | 588.64s |

The run log is included as `train.log`. The submitted `train_gpt.py` is a compact lzma/base85 wrapper, matching the style of recent record submissions. It contains the pipeline, local helper modules, and fixed bit plan; at runtime it materializes those files inside this record directory before launching the same staged train/quant/eval pipeline. `train_gpt_human.py` is a short review helper that documents the runtime layout and can list or extract the exact embedded sources from `train_gpt.py` without executing the training pipeline. No project code outside this record folder is required at evaluation time.

## What Is Different

The submission separates model training from artifact compilation, but keeps both inside the recorded 10-minute end-to-end pipeline:

1. Train a dense blessed 9-layer model for 8000 steps.
2. During late warmdown, enable fake-quant only for tensors that the final fixed bit plan will quantize.
3. Use the same fixed bit plan for the artifact compiler.
4. Compile the dense checkpoint into a `budgeted_v2` mixed-precision lzma artifact with GPTQ refinement.
5. Re-evaluate the exact artifact roundtrip with compiled 8xH100 eval.

The main differentiator is the artifact-aware late-QAT stage. It is not full-training blanket QAT and not a post-hoc PTQ-only run: the final warmdown sees the exact per-tensor bit plan used by the artifact compiler.

## Architecture

- 9 transformer blocks
- 512 model dimension
- 8 attention heads, 8 KV heads
- MLP hidden size 1536
- tied 1024-token SentencePiece embeddings
- BigramHash vocabulary 10240, dim 128
- SmearGate and scalar control tensors
- MuonW for matrix parameters, AdamW for embeddings/scalars

## Training And Artifact Settings

- `TRAIN_BATCH_TOKENS=524288`
- `ITERATIONS=8000`
- `WARMUP_STEPS=5`
- `WARMDOWN_MODE=hybrid`
- `WARMDOWN_ITERS=3700`
- `WARMDOWN_CURVE=cosine`
- `WARMDOWN_MIN_LR_SCALE=0.05`
- `LATE_QAT=1`
- `QAT_THRESHOLD=0.4`
- `QAT_BIT_PLAN_PATH=final_bit_plan.json`
- `SWA_ENABLED=0`
- `PACKING_SCHEME=interleaved`
- `COMPRESSOR=lzma`
- `CALIBRATION_SOURCE=train_tokens`
- `CALIBRATION_TOKENS=65536`
- `GPTQ_ENABLED=1`
- `GPTQ_DAMPING=0.1`
- `GPTQ_BLOCK_SIZE=128`

## How To Run

From this record directory, after the standard Parameter Golf dataset has been prepared under the repo root:

```bash
python3 train_gpt.py
```

The script runs the same staged pipeline used for the submitted log: train, fixed-plan artifact compilation, and final exact roundtrip eval. It expects the standard cached FineWeb paths at:

```text
../../../data/datasets/fineweb10B_sp1024
../../../data/tokenizers/fineweb_1024_bpe.model
```

Override `DATA_PATH` and `TOKENIZER_PATH` if your dataset is elsewhere.

## Reproducibility Note

This folder contains one completed 8xH100 run. The challenge recommends multiple seeds for statistical claims; before presenting this as more than a non-record submission, rerun at least two additional seeds or exact reproductions and report the variance.

## Included Files

- `README.md`
- `submission.json`
- `requirements.txt`
- `train.log`
- `train_gpt.py`
- `train_gpt_human.py` short review helper for inspecting the embedded sources
