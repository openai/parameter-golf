# Non-record: PR #1855 + Loss-Gated Score-First TTT (4xH100)

**val_bpb: 1.08838847** | **15,916,403 bytes** | **4xH100 SXM** | **seed 42**

This is a non-record submission. It satisfies the 16MB artifact limit and uses the normal 600s training timer, but it was run on 4xH100 rather than 8xH100 and its full TTT evaluation took 901.3s, so it is not a strict 10min_16mb leaderboard claim.

The goal is to document a small eval-time idea on top of PR #1855 and provide a reproducible above-baseline result. The new contribution is loss-gated score-first TTT: each validation chunk is scored first, then the already-scored per-document chunk losses are used to weight that same chunk's subsequent LoRA adaptation objective.

## Result

| Seed | GPUs | Steps | Pre-quant val_bpb | Quantized val_bpb | Final TTT val_bpb | Artifact bytes | TTT eval |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 4xH100 SXM | 2,532 | 1.09511199 | 1.10197870 | **1.08838847** | 15,916,403 | 901.3s |

The 4x run improves over the quantized non-TTT artifact by 0.01359023 BPB during phased TTT:

```text
diagnostic quantized val_bpb:1.10197870
quantized_ttt_phased val_bpb:1.08838847
```

This is above the naive baseline listed in the main leaderboard, but below the inherited PR #1855 8xH100 result. The gap is expected: this run completed 2,532 training steps, while PR #1855 reports about 4,931 steps on 8xH100.

## What Changed

PR #1855 uses phased score-first TTT where every document chunk contributes equally to the per-document LoRA update. This variant keeps the score-first rule but changes the post-score adaptation weight:

- warmup chunks use multiplier 1.0
- chunks with loss below the running EMA by `ADAPTIVE_TTT_MARGIN_NATS` use multiplier 0.5
- chunks near the running EMA use multiplier 1.0
- chunks above the running EMA use multiplier 1.25

The score is accumulated before the multiplier is computed for the adaptation step, so no future validation tokens or unscored losses are used to improve the score for the current chunk.

New knobs:

```bash
ADAPTIVE_TTT_ENABLED=1
ADAPTIVE_TTT_EMA=0.02
ADAPTIVE_TTT_WARMUP_CHUNKS=32
ADAPTIVE_TTT_MARGIN_NATS=0.03
ADAPTIVE_TTT_LOW_MULT=0.5
ADAPTIVE_TTT_MID_MULT=1.0
ADAPTIVE_TTT_HIGH_MULT=1.25
```

Set `ADAPTIVE_TTT_ENABLED=0` to recover the inherited PR #1855 TTT path from this script.

## Run Details

| | |
|---|---|
| Hardware | 4x NVIDIA H100 80GB HBM3 SXM on RunPod, US-MO-1 |
| Image | `runpod/parameter-golf:latest` |
| PyTorch | 2.9.1+cu128 |
| CUDA | 12.8 runtime image |
| FlashAttention | FA3 cu128/torch291 wheel (`flash_attn_interface`) |
| Triton | 3.5.1 |
| Python deps | `sentencepiece==0.2.1`, `brotli`, `python-minifier` |
| System deps | `lrzip` 0.651 for per-group compression |
| Dataset | CaseOps SP8192 FineWeb shards, 80 train shards + fixed validation shard + byte sidecar |
| Training timer | `MAX_WALLCLOCK_SECONDS=600`, `GPTQ_RESERVE_SECONDS=8.0` |
| Final training stop | step 2,532, `train_time: 590894ms` |
| Peak training memory | 41,897 MiB allocated, 47,176 MiB reserved |
| TTT setup | 3 phases, prefix docs 833 / 1666 / 2500, LoRA rank 80 |

## Reproduction Command

Run from this record folder after preparing the CaseOps SP8192 dataset and installing requirements.

```bash
apt-get update && apt-get install -y lrzip
pip install -r requirements.txt
pip install --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

DATA_DIR=./data \
VOCAB_SIZE=8192 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
ARTIFACT_DIR=./runs/loss_gated_ttt_seed42_4xh100 \
CASEOPS_ENABLED=1 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
GLOBAL_TTT_MOMENTUM=0.9 WARMDOWN_FRAC=0.85 BETA2=0.99 \
TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
GPTQ_RESERVE_SECONDS=8.0 GPTQ_CALIBRATION_BATCHES=16 VAL_LOSS_EVERY=0 \
GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 \
SMEAR_GATE_ENABLED=1 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
ADAPTIVE_TTT_ENABLED=1 \
ADAPTIVE_TTT_EMA=0.02 ADAPTIVE_TTT_WARMUP_CHUNKS=32 ADAPTIVE_TTT_MARGIN_NATS=0.03 \
ADAPTIVE_TTT_LOW_MULT=0.5 ADAPTIVE_TTT_MID_MULT=1.0 ADAPTIVE_TTT_HIGH_MULT=1.25 \
SEED=42 RUN_ID=loss_gated_ttt_seed42_4xh100 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

For a strict leaderboard reproduction attempt, use 8xH100 and `--nproc_per_node=8`. This folder reports only the 4xH100 non-record run above.

## Files

- `train_gpt.py` - full training/eval script with the adaptive TTT change.
- `train_seed42_4xh100.log` - tee log from the 4xH100 run.
- `loss_gated_ttt_seed42_4xh100.txt` - full internal log file for the same run.
- `submission.json` - structured metadata.
- `requirements.txt` - Python dependencies plus FA3/lrzip notes.
- `lossless_caps.py` and `prepare_caseops_data.py` - CaseOps dataset/tokenizer support.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` - SentencePiece tokenizer used by the run.

## Attribution

This submission is intentionally built on prior public work. The architecture, tokenizer, compression path, quantization path, and most hyperparameters are inherited from PR #1855 and its lineage. The new contribution in this folder is the adaptive loss-gated weighting of legal score-first TTT updates.

Most direct base:

- [PR #1855](https://github.com/openai/parameter-golf/pull/1855) by @codemath3000 - BOS-fixed SmearGate + LQER + SparseAttnGate + 9-hparam stack, including per-group compression and the final 8xH100 3-seed record this work builds on.

Important lineage credited by PR #1855 and reused here:

- [PR #1851](https://github.com/openai/parameter-golf/pull/1851) by @aquariouseworkman.
- [PR #1797](https://github.com/openai/parameter-golf/pull/1797) by @dexhunter - SmearGate + LQER asymmetric rank-4 on the PR #1787 base.
- [PR #1787](https://github.com/openai/parameter-golf/pull/1787) by @nprime06 - Polar Express Newton-Schulz, MIN_LR, sparse attention gate, fused softcapped CE.
- [PR #1736](https://github.com/openai/parameter-golf/pull/1736) - CaseOps + gated attention quant gate + looped layers + phased TTT integration.
- [PR #1729](https://github.com/openai/parameter-golf/pull/1729) by @romeerp - CaseOps tokenizer and validation byte sidecar infrastructure.
- [PR #1667](https://github.com/openai/parameter-golf/pull/1667) by @MarioPaerle - SmearGate and Attention Output Gate.
- [PR #1626](https://github.com/openai/parameter-golf/pull/1626) by @dexhunter - multi-phase global SGD phased TTT.
- [PR #1610](https://github.com/openai/parameter-golf/pull/1610) - phased TTT lineage.
- [PR #1586](https://github.com/openai/parameter-golf/pull/1586) - adaptive GPTQ clipping and int7 embeddings.
- [PR #1530](https://github.com/openai/parameter-golf/pull/1530) by @samacqua - varlen attention, fused LeakyReLU-square MLP, parallel residuals, doc-based LoRA TTT.
- [PR #1394](https://github.com/openai/parameter-golf/pull/1394) by @clarkkev - SP8192/GPTQ/depth-recurrence lineage.
- [PR #1344](https://github.com/openai/parameter-golf/pull/1344) - Polar-Express Newton-Schulz coefficients and recurrence.
- [PR #549](https://github.com/openai/parameter-golf/pull/549) by @abaybektursun - early legal TTT + parallel Muon stack.
- [PR #493](https://github.com/openai/parameter-golf/pull/493) - LeakyReLU-square activation.
- [PR #478](https://github.com/openai/parameter-golf/pull/478) by @gowtham0992 - XSA-all.
- [PR #315](https://github.com/openai/parameter-golf/pull/315) - partial RoPE and LN scale.
- [PR #289](https://github.com/openai/parameter-golf/pull/289) - U-Net skip connections.
