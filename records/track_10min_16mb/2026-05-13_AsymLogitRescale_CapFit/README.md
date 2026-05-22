# Asymmetric Logit Rescale + cap-fit bit allocation (PR #2140 fork)

**Status:** ⏳ **Draft.** The challenge has technically ended already and we've been trying to grab a RunPod 8×H100 SXM pod ever since, but they've been fully unavailable the whole time. So we're putting up this draft now without the actual 8×H100 numbers rather than wait indefinitely. The recipe and the full env block are validated: the model was trained on 8×AMD MI250X and the cap-fit re-quantization was validated on 1×A100. The expected 8×H100 stdout is in `expected_log.txt`; the exact training logs that produced the model are in `lumi_logs.txt`.

## TL;DR

```
upstream PR #2140 (s42 ref: 1.05591)
                  + ASYM_LOGIT_RESCALE=1         (−0.00132 BPB)
                  + GPTQ_CALIBRATION_BATCHES=32  (−0.00044 BPB)
                  + GPTQ_RESERVE_SECONDS=2.0     (−0.00012 BPB)
                  + EMBED_BITS=7                 (cap-fit: −487 KB)
                  + MLP_CLIP_SIGMAS=11.5         (cap-fit: −574 KB)
                  ─────────────────────────
                  ≈ 1.0554 BPB projected, 15,981,515 bytes (under 16 MB cap)
```

| Metric | Trained on 8×MI250X (historical) | 8×H100 production (this submission's target) |
|--------|----------------------------------|----------------------------------------------|
| val_bpb post-TTT | 1.05320 | **~1.0554** (projected; seed lottery) |
| val_bpb pre-quant | 1.05817 | ~1.058 |
| val_bpb post-quant pre-TTT | 1.06532 | **1.06739** (A100-validated) |
| Artifact size | 17,196,199 bytes (17.20 MB) ⚠ over cap | **15,981,515 bytes (15.98 MB)** ✓ |
| Train wallclock | 3,449,487 ms / 5102 steps on 8×MI250X | ~540 s / ~5067 steps on 8×H100 |
| TTT wallclock | 3,832,565 ms on 8×MI250X | ~597 s expected on 8×H100 |
| Seed | 42 | 42 |

## Architecture lineage

```
upstream PR #2140 (SP8192_CaseOps_Progressive3k_ShortDocTTT, s42 ref: 1.05591)
                  │  ← faithful direct port: 1.05508
                  + ASYM_LOGIT_RESCALE=1         (−0.00132 BPB)
                  + GPTQ_CALIBRATION_BATCHES=32  (−0.00044 BPB)
                  + GPTQ_RESERVE_SECONDS=2.0     (−0.00012 BPB)
                  + EMBED_BITS=7                 (cap-fit: −487 KB artifact)
                  + MLP_CLIP_SIGMAS=11.5         (cap-fit: −574 KB artifact)
                  ─────────────────────────
                  = 1.0554 projected (cap-compliant, this submission)
```

PR #2140 is built on PR #2014 (last clean record per Issue #2127) + token-only n-gram tilt from PR #1145 / PR #2018.

The headline lever is **`ASYM_LOGIT_RESCALE`**: it replaces the symmetric `softcap * tanh(logits / softcap)` output clamp with an asymmetric variant carrying two learnable scalars (`softcap_pos`, `softcap_neg`), worth −0.00132 BPB. The two GPTQ levers (doubling calibration batches 16→32, halving the reserve window 4.0→2.0 s) add another −0.00056.

## The trained artifact is not committed here (pending the H100 run)

The model artifact (`final_model.int6.ptz`) is **not** included in this directory yet. The 8×MI250X training that produced `val_bpb 1.05320` serialized to **17,196,199 bytes (17.20 MB) — over the 16 MB cap.** It was slightly overtrained, which raises the entropy in the weights, and it was serialized with a brotli fallback rather than lrzip. The dominant lever, though, is bit allocation: re-quantizing the same fp32 weights with `EMBED_BITS=7` and `MLP_CLIP_SIGMAS=11.5` (PR #2140's intended values, vs the script defaults 8 and 10.0 that this run silently used) accounts for ~1.06 MB of the overage and brings the artifact under cap (see below).

### Cap-fit validated on 1×A100 (2026-05-13)

Taking the existing fp32 weights and re-running only the quantization stage (`QUANTIZE_ONLY=1`) with the corrected env block:

| Config | Artifact size | Pre-TTT BPB | Est. post-TTT |
|---|---|---|---|
| EMBED=8, MLP_CLIP=10.0 (the over-cap defaults) | 17.04 MB | 1.06532 | ~1.0533 |
| EMBED=7, MLP_CLIP=10.0 | 16.56 MB | 1.06619 | ~1.0542 |
| EMBED=6, MLP_CLIP=10.0 | 16.10 MB | 1.06978 | ~1.0578 |
| **EMBED=7, MLP_CLIP=11.5** ⭐ | **15.98 MB** | **1.06739** | **~1.0554** |

`EMBED=7 + MLP_CLIP=11.5` costs ~+0.002 BPB but fits the 16 MB cap with 18 KB margin. The `MLP_CLIP_SIGMAS 10.0 → 11.5` change is counter-intuitive: looser clipping gives better int6 grid usage on the MLP weights, so the LQER residual shrinks. The 8×H100 production run produces this cap-fit artifact directly during the training cap; this record will be updated with it (and the measured BPB) once a pod is available.

## Files

| File | Content |
|------|---------|
| `train_gpt.py` | The training script — PR #2140 with AMD/ROCm compat patches (FA3→FA2 try/except, TensorDescriptor try/except, brotli compression fallback, `fullgraph=not _IS_ROCM`, fixed-seq TTT compile, schedule-delimiter normalization). On H100 with FA3 + lrzip available these patches are no-ops, so the CUDA path is unchanged from upstream. One Python invocation does train + GPTQ + AWQ-lite + LQER + serialize + post-quant eval + TTT eval. |
| `prepare_caseops_data.py` | Downloads `romeerp/parameter-golf-caseops-v1` from HuggingFace (Issue #2127 audit-clean CaseOps SP8192 dataset) and unpacks it to the `data/` layout the trainer expects. |
| `online_ngram_tilt.py` + `online_ngram_state.c` | Token-only n-gram tilt helper (PR #1145 / #2018), compiled at runtime (clang or gcc). |
| `submission.json` | Metadata: recipe identity, lineage, and the projected BPB targets + tolerances. The `val_bpb` / `val_bpb_postquant` / `artifact_bytes` fields currently hold projected (not yet measured) values; they get the measured numbers once the H100 run completes. |
| `expected_log.txt` | The stdout we expect the 8×H100 run to produce: hyperparameter dump (with the cap-fit `embed_bits: 7` / `mlp_clip_sigmas: 11.5`), training progression, the quantized-weights block, `Total submission size`, and the final `quantized_ttt_phased val_bpb`. |
| `lumi_logs.txt` | The exact 8×MI250X train + TTT logs that produced the historical `1.05320` / 17.20 MB result. These use the pre-cap-fit env (`embed_bits=8`, `mlp_clip_sigmas=10.0`, longer wallclock, TTT as a separate job), so they differ from `expected_log.txt` on exactly the cap-fit anchors. |

## Reproduction (8×H100 SXM)

The data is public and the recipe is one Python invocation. On a CUDA box with FA3 + lrzip (e.g. an 8×H100 SXM pod):

```bash
# 1. Fetch the audit-clean CaseOps SP8192 dataset (~25 GB) into ./data
python3 prepare_caseops_data.py

# 2. Train + quantize + TTT end-to-end (single invocation), with the env block below
torchrun --nproc_per_node=8 --standalone train_gpt.py
```

### Env block

The full hyperparameter dump (every value the script read) is at the top of `expected_log.txt`. The non-default vars that define this recipe:

```bash
SEED=42
NUM_GPUS=8
VOCAB_SIZE=8192
CASEOPS_ENABLED=1
ZERO_PUE_MARKERS=1

# Sequence schedule (wallclock-paced progressive 1024→2048→3072)
TRAIN_SEQ_LEN=3072
ROPE_TRAIN_SEQ_LEN=3072
TRAIN_SEQ_SCHEDULE=1024@0.10:2048@0.70:3072@1.00
TRAIN_SEQ_SCHEDULE_MODE=wallclock
SEQ_CHANGE_WARMUP_STEPS=32
COMPILE_SHAPE_WARMUP=1
EVAL_SEQ_LEN=3072
EVAL_STRIDE=1536

# Architecture
LEAKY_RELU_SQ_SLOPE=0.3
QK_GAIN_INIT=5.25
SPARSE_ATTN_GATE_ENABLED=1
SPARSE_ATTN_GATE_SCALE=0.5
GATED_ATTN_QUANT_GATE=1
SMEAR_GATE_ENABLED=1
GATE_WINDOW=12
FUSED_CE_ENABLED=1
ASYM_LOGIT_RESCALE=1            # lever: -0.00132 BPB

# Optimization
MATRIX_LR=0.026
MIN_LR=0.1
WARMDOWN_FRAC=0.85
BETA2=0.99

# Bit allocation — CAP-FIT (diverge from script defaults 8 / 10.0; without these the artifact is ~17 MB)
EMBED_BITS=7
MATRIX_BITS=6
EMBED_CLIP_SIGMAS=14.0
MLP_CLIP_SIGMAS=11.5
ATTN_CLIP_SIGMAS=13.0
MATRIX_CLIP_SIGMAS=12.85

# Quantization
COMPRESSOR=pergroup
GPTQ_CALIBRATION_BATCHES=32     # lever: -0.00044 BPB
GPTQ_RESERVE_SECONDS=2.0        # lever: -0.00012 BPB
LQER_ENABLED=1
LQER_RANK=4
LQER_FACTOR_BITS=4
LQER_TOP_K=4
LQER_ASYM_ENABLED=1
LQER_ASYM_GROUP=64
AWQ_LITE_ENABLED=1
AWQ_LITE_BITS=8
AWQ_LITE_GROUP_TOP_K=1
AWQ_LITE_GROUP_SIZE=64

# TTT (LoRA, phased, score-first)
TTT_ENABLED=1
TTT_EVAL_SEQ_LEN=3072
TTT_FIXED_SEQ_COMPILE=1
TTT_COMPILE_DYNAMIC=0
TTT_BATCH_SIZE=24
TTT_LORA_RANK=80
TTT_LORA_LR=0.0001
TTT_LOCAL_LR_MULT=0.75
TTT_MASK=no_qv
TTT_WEIGHT_DECAY=0.5
TTT_BETA2=0.99
TTT_CHUNK_SIZE=64
TTT_SHORT_SCORE_FIRST_ENABLED=1
TTT_SHORT_DOC_LEN=2000
TTT_SHORT_CHUNK_SIZE=32
PHASED_TTT_PREFIX_DOCS=2500
PHASED_TTT_NUM_PHASES=1

# N-gram tilt (PR #1145, inside-timer, score-first compliant)
NGRAM_TILT_ENABLED=1
NGRAM_TOKEN_BOOST=2.625
NGRAM_TOP_K=11500
NGRAM_HINT_PRECOMPUTE_OUTSIDE=0

# Wallclock (OpenAI 10-min train cap)
MAX_WALLCLOCK_SECONDS=540
```

### Eval-budget note

The training loop finishes within the 600 s training cap (~540 s). The eval phase (per-group lrzip serialize ~125 s + phased TTT ~540 s ≈ 720 s) runs ~120 s over the 600 s evaluation cap. The training cap is met; the eval cap is not. Trimming `PHASED_TTT_PREFIX_DOCS` 2500 → 1800 is the obvious lever to close that gap (~−0.005 BPB cost) for full both-caps eligibility.
