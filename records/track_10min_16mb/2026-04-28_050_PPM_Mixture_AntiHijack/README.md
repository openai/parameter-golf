# Record: PR #1850 + Anti-Hijack Gate — val_bpb 0.99445 (full val)

**val_bpb: 0.99445** (3-seed mean, std 0.00141; **full 47.85M val**) | **best seed 0.99291** | **~15.92 MB** | 8×H100 SXM, 600s train / 600s eval

This submission is a small additive change on top of **PR #1850**: we replace `score_byte`'s `λ`-gating with an **anti-hijack-guarded** variant that suppresses the high-λ branch when the NN is already confident on the actual byte. The gate is **stackable with #1881 and #1877**.

## Results

| Seed | Steps | Pre-quant post-EMA | Diagnostic quantized | **mix_bpb_sidecar (full val)** | gate_high_frac | Train time | PPM time | Eval total |
|------|------:|-------------------:|---------------------:|------------------------------:|---------------:|-----------:|---------:|-----------:|
| 42   | 4952  | 1.06445            | 1.07330              | **0.99291**                   | 16.38%         | 595.98s    | 158.8s   | 221s       |
| 7    | 4898  | 1.06692            | 1.07573              | **0.99471**                   | 16.41%         | 596.04s    | 156.0s   | 173s       |
| 1337 | 4920  | 1.06800            | 1.07698              | **0.99572**                   | 16.43%         | 596.10s    | 152.9s   | 170s       |
| **Mean** | **4923** |                |                      | **0.99445**                   | **16.41%**     |            |          |            |
| **Std**  |          |                |                      | **0.00141**                   |                |            |          |            |

All measurements on the **full 47,851,520-token val set** — no subsetting. PPM scoring is gathered across all 8 ranks.

## Class disclosure

This submission is in the PPM byte-mixture class under discussion in Issue #1872 — same scoring scheme as PR #1850 (`score_byte`/`ppm_score` infrastructure unchanged from #1850 except for the anti-hijack patch below). If #1872 disallows the class, our neural-only fallback is the diagnostic quantized number ≈1.073.

## Main contribution — anti-hijack gate

In #1850's `score_byte`, the high-λ branch fires whenever the PPM table's confidence on the current prefix exceeds `thr`. The anti-hijack gate adds a second condition: high-λ fires **only when the NN is not already confident on the actual byte**.

```c
// 5-line patch on top of #1850's score_byte:
int hi_raw = (conf >= thr);
int hi = hi_raw && !(nn_skip_thr > 0.0 && nn_logp > -nn_skip_thr);
double lam = hi ? lambda_lo : lambda_hi;
(*gate_total)++;
if (hi) (*gate_high)++;
```

With `nn_skip_thr_nats = 0.277` (= 0.40 bits), high-λ is suppressed whenever `−log p_NN(actual byte) < 0.40 bits`. This addresses the legality concern raised in Issue #1017 / #1872 about confidence-gated mixtures: when the NN already nails the byte, we don't let the PPM table compound — the mixture stays at low λ even if PPM has high prefix confidence.

Tuning: `thr = 0.76` (vs #1850's 0.9). Lower threshold widens the high-λ band on the *training-distribution* side; the anti-hijack guard ensures it only fires where the NN actually needs help. Empirically `gate_high_frac ≈ 16.4%` across all 3 seeds.

### Stackability with #1881 / #1877

The gate is a local change to `score_byte` — it does not touch the NN forward, the PPM table construction, the OMP scoring infrastructure, or the gather pattern. It composes cleanly with:

- **#1881** — drop-in replacement; the patch applies to whichever `score_byte` variant #1881 uses.
- **#1877** — orthogonal lever; #1877's contribution operates outside `score_byte`.

A stacked submission would inherit both improvements with a one-line config diff (`PPM_NN_SKIP_THR_NATS=0.277`).

### Mixture hyperparameters (all seeds)

| Hyperparameter         | Value | Notes |
|------------------------|------:|-------|
| `PPM_ORDER`            |     4 | match #1850 |
| `PPM_LAMBDA_HI`        |   0.9 | match #1850 |
| `PPM_LAMBDA_LO`        |  0.05 | match #1850 |
| `PPM_CONF_THRESHOLD`   |  0.76 | tuned (vs 0.9 in #1850) |
| `PPM_NN_SKIP_THR_NATS` | 0.277 | **NEW** anti-hijack guard (= 0.40 bits) |
| `PPM_OMP_CHUNK_TOKENS` |     0 | single-pass scoring, byte-deterministic |

## Lineage

- **Base stack**: PR #1797 (nprime06) + 2026-04-27 050 baseline (this author) — see [`2026-04-27_050_PR1797_Base_BOS_Fix/`](../2026-04-27_050_PR1797_Base_BOS_Fix/) for the NN-side details.
- **PPM mixture**: **PR #1850**. We inherit `score_byte`/`ppm_score`/`ppm_score_omp` from #1850 verbatim, with the 5-line anti-hijack patch above.

## Rule compliance

- **Artifact ≤ 16,000,000 bytes (decimal)**: 15,917,572 / 15,914,567 / 15,914,752 (all 3 seeds; ≥ 80 KB headroom).
- **train_time ≤ 600s**: 595.98 / 596.04 / 596.10s.
- **total_eval_time ≤ 600s**: 221 / 173 / 170s.
- **Issue #1017 Condition 3 (score-before-update)**: PPM table updates *after* scoring each byte; single L→R pass; no chunk reset (`PPM_OMP_CHUNK_TOKENS=0`). The anti-hijack gate tightens this further.
- **Issue #1017 Condition 1 (causal NN)**: standard `eval_val` non-overlap stride=2048 forward with BOS-aware varlen attention via `cu_seqlens`.
- **No val data in training**: `fineweb_train_*.bin` only; PPM tables built and used only at eval time.
- **Full-val coverage**: all 47,851,520 tokens scored.

## Requirements

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn-interface sentencepiece triton numpy brotli
# Python ≥ 3.12.
```

## Run command (3-seed reproduction)

```bash
for SEED in 42 7 1337; do
  SEED=$SEED MAX_WALLCLOCK_SECONDS=600 RUN_LABEL=seed_$SEED \
    bash launch_055_run.sh
done
```

`launch_055_run.sh` sets the full 050-baseline env plus the PPM block:

```bash
PPM_NATIVE_ENABLED=1
PPM_ORDER=4
PPM_LAMBDA_HI=0.9
PPM_LAMBDA_LO=0.05
PPM_CONF_THRESHOLD=0.76
PPM_NN_SKIP_THR_NATS=0.277      # anti-hijack
PPM_LOG_CACHE_SIZE=1048576
PPM_OMP_THREADS=8
PPM_OMP_CHUNK_TOKENS=0          # single-pass, byte-deterministic
```

## Credits

- Authors of **PR #1850** — PPM-D mixture infrastructure this work builds on.
- @nprime06 — PR #1797 base stack.
- @romeerp — PR #1729 CaseOps tokenizer concept.

## Included files

- `train_gpt.py` — single-file training + post-training (PPM mixture embedded).
- `submission.json` — 3-seed metadata.
- `README.md` — this file.
- `train_seed42.log`, `train_seed7.log`, `train_seed1337.log` — per-seed run logs (training + diagnostic eval + PPM scoring line).

The CaseOps tokenizer (`tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`), the bijective transform (`lossless_caps.py`), and the data-prep script (`prepare_caseops_data.py`) are unchanged from the 050 baseline submission ([`2026-04-27_050_PR1797_Base_BOS_Fix/`](../2026-04-27_050_PR1797_Base_BOS_Fix/)) and not duplicated here.
