# LongCtx No-QV QK5.25 + AsymLogit + LQER g32/top4 + TTT-local 0.80 + MatrixLR 0.028

**val_bpb = 1.05769** (3-seed mean, std 0.00041) | **15,971,753 bytes max** with **28,247 B slack** | 8×H100 80GB SXM (Runpod)

Forked from PR **#2007** record `2026-04-30_LongCtx_NoQV_QK525_AsymLogit_1.0590`
(parent 3-seed mean **1.05899193**, parent seed-42 **1.05857451**).

This submission keeps the parent architecture, optimizer, dataset, tokenizer,
TTT/eval pipeline, quantizer, and compressor **byte-for-byte unchanged**, and
retunes five scalar hyperparameters that are already exposed as environment
variables in the parent's `train_gpt.py`. **There are no code changes vs parent
#2007** — `train_gpt.py` in this folder is byte-identical to the parent's
(md5 `2a7e36e29aa5b5811abb6170059aa8d1`).

## 3-seed results (Runpod 8×H100 80GB SXM)

| Seed | Stop step | Train time | Pre-quant BPB | Quant BPB | Final TTT BPB | TTT eval time | Artifact bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42   | 4868 | 596.142 s | 1.05972472 | 1.06790517 | **1.05711454** | 397.125 s | 15,971,753 |
| 0    | 4861 | 595.821 s | 1.06048352 | 1.06872963 | **1.05798212** | 397.575 s | 15,971,492 |
| 1234 | 4873 | 595.991 s | 1.06052013 | 1.06872931 | **1.05796494** | 395.401 s | 15,971,748 |
| **Mean** | | | | | **1.05768720** | | |
| **Std (pop, n)** | | | | | **0.00040508** | | |

All seeds satisfy the 10-minute / 16 MB rules:

- `train_wallclock_s ≤ 600` ✓ (595.821 s – 596.142 s)
- TTT phased `eval_time_s ≤ 600` ✓ (395.4 s – 397.6 s); total eval (pre-quant
  + diag-quant + TTT) ≤ 420 s on every seed
- Total submission size ≤ 16,000,000 B ✓ (15,971,492 – 15,971,753 B)
- Artifact slack ≥ 28,247 B on every seed (worst case = seed 42)
- All 782 phased-TTT batches drained on every seed

## What this PR changes vs parent #2007

Five env-var deltas only. Every other setting (architecture, optimizer,
SmearGate, sparse-attn-gate, GPTQ, AWQ-lite, late-QAT, EMA, embedding bit
allocation, sequence lengths, TTT phasing, AsymLogit, compressor) is inherited
unchanged from `2026-04-30_LongCtx_NoQV_QK525_AsymLogit_1.0590`.

| Knob | Parent #2007 | This PR | Direction |
|---|---|---|---|
| `MATRIX_LR` | 0.026 | **0.028** | slightly higher matrix LR |
| `LQER_RANK` | 4 | **2** | half-rank LQER correctors |
| `LQER_ASYM_GROUP` | 64 | **32** | finer asym-quant groups |
| `LQER_TOP_K` | 3 | **4** | one extra top-K corrector slot |
| `TTT_LOCAL_LR_MULT` | 0.75 | **0.80** | slightly hotter local TTT step |

Diff against the parent's `run_current_candidate.sh` is exactly these five
lines (each marked inline as `# delta vs parent (#2007)`) plus the `RUN_ID`
default; `train_gpt.py` is unchanged.

## Comparison vs parent #2007 (paired, same 3 seeds)

| Seed | parent #2007 BPB | this PR BPB | Δ BPB |
|---|---:|---:|---:|
| 42   | 1.05857451 | 1.05711454 | **−0.00145997** |
| 0    | 1.05915199 | 1.05798212 | **−0.00116987** |
| 1234 | 1.05924929 | 1.05796494 | **−0.00128435** |
| **Mean** | **1.05899193** | **1.05768720** | **−0.00130473** |

Paired one-sided t-test on val_loss (lower is better, df=2):
mean Δ_loss = **−0.00293427 nats**, sample SE = 0.0001845, **t = −15.91**,
**p ≈ 0.00198**. Improvement is statistically significant at p < 0.01, but
the magnitude (~0.00293 nats) is below the project's **0.005-nat record
threshold against the parent**.

## Comparison vs currently-merged SOTA #1493 (1.0810)

Δ_BPB ≈ **−0.0233**, Δ_nats ≈ **−0.051**. Signal-to-noise vs our 3-seed std
is ~50× the project's 0.005-nat record bar. Strict per-seed paired t-test is
not directly possible (PR #1493 used a different seed pool: 42, 314, 999),
but on every individual seed in this submission the improvement vs #1493 is
≥ 0.022 BPB, far above the threshold.

## Record-threshold framing

- **Vs currently merged SOTA (PR #1493, 1.0810):** improvement of ~0.023 BPB /
  ~0.051 nats; satisfies the 0.005-nat / p<0.01 record threshold with very
  wide margin.
- **Vs PR #2007 (still under review at submission time):** paired 3-seed
  improvement of ~0.00293 nats meets p<0.01 but does **not** clear the
  0.005-nat magnitude bar. If #2007 is merged before this PR is reviewed,
  treat this submission as a non-record tuning improvement on top of #2007
  rather than a record claim against #2007.

Maintainers may treat this as record or non-record at their preferred
significance framework.

## Method

The frozen parent recipe (unchanged here):

- CaseOps/SP8192 tokenization with byte-sidecar BPB accounting.
- Sparse attention gating, BOS-fixed SmearGate, skip gates, LQER correction,
  int7 embeddings, and mixed-precision GPTQ + AWQ-lite.
- 2560-token eval and TTT windows.
- No-QV TTT masking, keeping K/O/MLP adaptation active.
- `TTT_LORA_RANK=80`, `PHASED_TTT_PREFIX_DOCS=3000`.
- `QK_GAIN_INIT=5.25`, `WARMDOWN_FRAC=0.85`, `MIN_LR=0.1`.
- Eval-only asymmetric logit rescale.
- Per-group `lrzip -L 9` compression.

The five-knob retune (this PR) was chosen by an MN5 single-node sweep on top
of the same #2007 parent stack; `MATRIX_LR=0.028 + LQER_RANK=2 + LQER_TOP_K=4
+ LQER_ASYM_GROUP=32 + TTT_LOCAL_LR_MULT=0.80` was the cleanest BPB win in
that sweep that also kept artifact slack > 25 KB after real `lrzip`.

## Reproduce

Prepare the CaseOps dataset once (unchanged from parent):

```bash
python prepare_caseops_data.py --local-dir /workspace/caseops_data
```

Run a seed from this folder:

```bash
SEED=42 \
CASEOPS_ROOT=/workspace/caseops_data \
RUN_ID=longctx_noqv_qk525_asym_lqer_g32_top4_tttlocal080_seed42 \
./run_current_candidate.sh
```

`run_current_candidate.sh` expands the exact environment variables (including
the five retuned knobs) and launches:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Repeat with `SEED=0` and `SEED=1234` for the matched 3-seed validation, using
the same seed pool as the parent record.

## Logs

- `train_seed42.log` — Runpod seed-42 final rerun, final BPB **1.05711454**.
- `train_seed0.log` — Runpod seed-0 final rerun, final BPB **1.05798212**.
- `train_seed1234.log` — Runpod seed-1234 final rerun, final BPB
  **1.05796494**.

Each log contains the four required diagnostic lines:

1. `pre-quantization post-ema val_loss: ... val_bpb: ...`
2. `diagnostic quantized val_loss: ... val_bpb: ...`
3. `quantized_ttt_phased val_loss: ... val_bpb: ...`
4. `Total submission size quantized+pergroup: N bytes`

## Hardware / software

- 8 × NVIDIA H100 80GB HBM3 SXM (Runpod)
- PyTorch 2.9.1+cu128, CUDA 12.8
- Same FlashAttention / Triton / runtime stack as parent #2007 record.

## Files

- `train_gpt.py` — byte-identical to parent #2007 (md5
  `2a7e36e29aa5b5811abb6170059aa8d1`).
- `run_current_candidate.sh` — parent's runner with the five env overrides
  marked inline as `# delta vs parent (#2007)`.
- `lossless_caps.py`, `prepare_caseops_data.py`, `requirements.txt`,
  `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` —
  byte-identical to parent.
- `submission.json` — full 3-seed metadata.
- `train_seed{42,0,1234}.log` — Runpod 3-seed final rerun logs.
