# GDN Hybrid Attention: Long-Context Performance in the Parameter-Constrained Regime

**Track:** Non-Record Submission (Unlimited Compute)  
**Date:** 2026-04-04  
**Hardware:** 1× H100 SXM, 600 seconds per run  

---

## Summary

 Test whether replacing most full-attention layers with Gated Delta Net (GDN) linear-recurrent layers — the design pattern introduced in Olmo Hybrid (Merrill et al., 2026) — benefits a parameter-constrained model at long context lengths. The short answer: **there is a clear crossover between 8k and 16k context**. Below the crossover, full attention is better; above it, the hybrid wins by a widening margin (about **0.035** bpb better at 16k and **0.235** bpb better at 32k on `final_int8_zlib_roundtrip`, 2-seed averages) within an identical 600-second wall-clock budget on a single H100.

---

## Motivation

Olmo Hybrid (AI2, Merrill et al., 2026) interleaves Gated DeltaNet (GDN) linear-recurrent layers with full-attention layers at a 3:1 ratio (GDN:attention), replacing the sliding-window attention layers from Olmo 3. While the paper's primary motivation is greater expressivity, a practical consequence at scale is subquadratic cost: most layers avoid O(L²) attention while a few full-attention layers (every 4th) preserve global context mixing. A natural question for Parameter Golf is whether the same trade-off applies in the parameter-constrained, fixed-wall-clock setting. The answer is non-obvious because:

1. At short context, GDN's conv + recurrent kernel overhead is not offset by attention savings — FlashAttention is already extremely fast at moderate sequence lengths.
2. At long context, the hybrid completes more optimizer steps in the same wall-clock time, which means more training signal in addition to better per-step loss.
3. Both effects are parameter-regime-specific: with only ~17M parameters and aggressive quantization, the model cannot absorb arbitrary context — there is a natural limit to how much long-context training helps.

---

## Architecture

### Baseline (`train_gpt.py`)
- 9 layers, width 512, 8Q/4KV heads (GQA), MLP 2×
- Every layer: causal scaled dot-product attention (FlashAttention path)
- RoPE, RMSNorm, residual scaling, skip connections (U-Net style)
- **17,059,912 parameters**

### Hybrid GDN (`train_gpt_gdn.py`)
- 9 layers, width 448, 8Q/4KV GQA where attention is used, MLP 2×
- **7 layers** use Gated Delta Net mixer (`chunk_gated_delta_rule`): causal depthwise conv on Q/K/V, per-head gates and decay
- **2 layers** (indices 3 and 7) keep full causal attention
- GDN shape: `head_dim_ratio=0.75`, `expand_v=2.0`, `conv_size=4`
- Same RoPE, RMSNorm, residual/skip structure as baseline
- Width reduced 512→448 to keep parameter count comparable
- **17,424,332 parameters** (~2% more than baseline)

This is the default configuration of `train_gpt_gdn.py` with no environment overrides.

---

## Experimental Setup

All runs use the same optimizer recipe adapted from the `TrainingOptSeq4096` record submission, with schedule parameters rescaled to match each model's step budget within 600 seconds in 1 H100:

```
TIED_EMBED_LR=0.030  MATRIX_LR=0.020  SCALAR_LR=0.020
MUON_MOMENTUM=0.99   MUON_MOMENTUM_WARMUP_START=0.92
WARMUP_STEPS=20
VAL_LOSS_EVERY=50    TRAIN_LOG_EVERY=10
MAX_WALLCLOCK_SECONDS=600
```

Batch size and schedule are set per context length:

| Seq Len | `TRAIN_BATCH_TOKENS` | `MUON_MOMENTUM_WARMUP_STEPS` | `WARMDOWN_ITERS` |
|---------|----------------------|------------------------------|------------------|
| 8192 | 393216 | 190 (baseline) / 160 (hybrid) | 381 / 320 |
| 16384 | 393216 | 118 / 104 | 236 / 208 |
| 32768 | 524288 | 50 / 79 | 100 / 158 |

The hybrid schedule is recalculated from measured step times to preserve the same warmup/warmdown ratios (~0.18 / ~0.36 of total steps) used in the 4k record.

Evaluation: standard `final_int8_zlib_roundtrip` — int8 quantized weights, zlib-compressed, round-trip loaded for evaluation. Score = `val_bpb` on the full FineWeb validation split.

---

## Results

### Step time and step budget

Measured steady-state step time (mean of steps 6–10):

| Seq Len | Baseline (ms/step) | Hybrid (ms/step) | Speedup | Steps in 600s (baseline) | Steps in 600s (hybrid) |
|---------|-------------------|-----------------|---------|--------------------------|------------------------|
| 8192 | 594 | 757 | **0.79×** | 1011 | 789 |
| 16384 | 952 | 847 | **1.12×** | 629 | 712 |
| 32768 | 2202 | 1361 | **1.62×** | 271 | 440 |

The crossover in step time occurs between 8k and 16k context, consistent with the `compute_experiment.md` probe runs. At 32k, the hybrid completes **62% more optimizer steps** in the same wall-clock.

### `final_int8_zlib_roundtrip` val_bpb

All figures below are from the logged `final_int8_zlib_roundtrip` line (int8+zlib weights, round-trip reload, full FineWeb val). For configs where the capture appended a second run in one file, the **last** `final_int8_zlib_roundtrip` in that log is used for the baseline. Hybrid 16k and 32k each have two seed runs; the table reports **2-seed averages** with individual results below.

| Seq Len | Baseline val_bpb | Hybrid val_bpb | Δ (hybrid − baseline) | Winner |
|---------|------------------|----------------|----------------------|--------|
| 8192 | 1.3507 | 1.3810 | +0.030 | Baseline |
| 16384 | 1.4353 | 1.3999† | **−0.035** | **Hybrid** |
| 32768 | 1.7059 | 1.4709‡ | **−0.235** | **Hybrid** |

† Hybrid 16k, 2-seed mean. Seed **1341**: 1.3970, seed **1337**: 1.4029.
‡ Hybrid 32k, 2-seed mean. Default seed: 1.4703, seed **42**: 1.4716.

At seq **4096**, `final_int8_zlib_roundtrip` reports baseline **1.3385** vs hybrid **1.3608** — baseline still ahead; use only as qualitative context next to the 8k–32k grid above.

On `final_int8_zlib_roundtrip`, the hybrid at 32k (1.4709 mean) is still a bit **worse** than the baseline at **16k** (1.4353). What the hybrid *does* avoid is the baseline’s blow-up at 32k (**1.7059** vs **1.4709**): long context is far cheaper for the hybrid, so 32k training stays competitive while full attention does not.

### Val loss learning curves at 32k

Numbers are **in-training validation** `val_bpb` (same dtype as training), not the post-int8 `final_int8_zlib_roundtrip` eval.

| Step | Baseline val_bpb | Hybrid val_bpb |
|------|-----------------|----------------|
| 50 | 2.5255 | 2.1231 |
| 100 | 2.1579 | 1.9097 |
| 150 | 1.9480 | 1.8040 |
| 200 | 1.8220 | 1.6803 |
| 250 | 1.7190 | 1.5973 |
| 271 | 1.7035 *(end)* | ≈1.5927 *(interp)* |
| 300 | — | 1.5459 |
| 350 | — | 1.5071 |
| 400 | — | 1.4805 |
| 440 | — | 1.4694 *(end)* |

The hybrid is ahead from the first val checkpoint and the gap grows: at the baseline's final step (271), the hybrid is already at ≈1.5927 (interpolated; nearest val checkpoints are 1.5973 at step 250 and 1.5459 at step 300) and continues improving for another 169 steps.

---

## Analysis

### Why the hybrid is slower at short context

At 8k and below, FlashAttention is efficient enough that the GDN kernel overhead (causal conv1d + `chunk_gated_delta_rule`) is not compensated by the attention savings. The hybrid is slower per step (757 vs 594 ms at 8k). Notably, the hybrid's per-step loss is actually *better* than the baseline at every matched checkpoint (e.g., 1.3802 vs ~1.3855 at step 789), but the 28% fewer optimizer steps (789 vs 1011) more than erases this advantage, resulting in a worse final score.

### Why the hybrid wins at long context

Two effects compound at 16k+:

1. **Step time advantage.** 7 of 9 layers use O(L) GDN recurrence; only 2 remain O(L²) attention. At 32k this yields 62% more optimizer steps in the same 600-second budget.

2. **Effective context utilization.** GDN layers compress history into a fixed-size state regardless of sequence length. At only 17M parameters, this bottleneck may actually help — a model this small may not effectively exploit full O(L²) all-to-all attention over 32k tokens.

### The crossover

The performance crossover (hybrid starts winning) falls between 8k and 16k context. The step-time crossover (hybrid is faster per step) falls in the same interval. This is not a coincidence: since the hybrid has better per-step loss at *all* context lengths tested, the only variable that determines whether it wins overall is the step budget — which flips in the hybrid's favor once it becomes faster per step, governed by O(L) vs O(L²) scaling.

For a fixed 600-second budget, the hybrid should be preferred for any context length ≥ 16k.

---

## Reproducing These Results

### Setup

From the repo root, install dependencies:

```bash
pip install -r records/track_non_record_16mb/2026-04-04_GDN_Hybrid_LongContext/requirements.txt
```

The key extra dependency vs the standard baseline is `flash-linear-attention` (the `fla` package), which provides `chunk_gated_delta_rule` and the causal conv1d kernel used by the GDN layers.

Download the dataset if not already present:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

### Running the hybrid

All scripts are included in this folder. From this submission's directory:

```bash
# Single context length
bash run_hybrid_long_context_single_h100.sh {8192|16384|32768}

# All three context lengths sequentially
bash run_hybrid_long_context_all_single_h100.sh
```

The scripts use the `train_gpt_gdn.py` included in this folder and expect to be run from the **repo root** (so that `data/` paths resolve correctly):

```bash
cd /path/to/parameter-golf
bash records/track_non_record_16mb/2026-04-04_GDN_Hybrid_LongContext/run_hybrid_long_context_all_single_h100.sh
```

Environment variables can be used to override defaults:

```bash
# Custom run ID or seed
RUN_ID=my_run SEED=42 bash run_hybrid_long_context_single_h100.sh 32768
```



### Running the baseline for comparison

The baseline runs used the existing repo-root script with identical optimizer settings:

```bash
bash run_baseline_long_context_single_h100.sh {8192|16384|32768}
```

### Log files

Training logs for all runs reported here are in `logs_experiment/`:

| File | Description |
|------|-------------|
| `logs_experiment/baseline_1xh100_600s_seq8192.txt` | Full-attention baseline, 8k context |
| `logs_experiment/baseline_1xh100_600s_seq16384.txt` | Full-attention baseline, 16k context |
| `logs_experiment/baseline_1xh100_600s_seq32768.txt` | Full-attention baseline, 32k context |
| `logs_experiment/hybrid_gdn_1xh100_600s_seq8192.txt` | GDN hybrid, 8k context |
| `logs_experiment/hybrid_gdn_1xh100_600s_seq16384_seed1337.txt` | GDN hybrid, 16k context, seed 1337 |
| `logs_experiment/hybrid_gdn_1xh100_600s_seq16384_seed1341.txt` | GDN hybrid, 16k context, seed 1341 (default) |
| `logs_experiment/hybrid_gdn_1xh100_600s_seq32768.txt` | GDN hybrid, 32k context, default seed |
| `logs_experiment/hybrid_gdn_1xh100_600s_seq32768_s42.txt` | GDN hybrid, 32k context, seed 42 |

Each log file contains the full script source followed by the training output (step logs, val checkpoints, and the final `final_int8_zlib_roundtrip` eval line).
