# Record: PR #1908 reproduction with compliant 600s wallclock — val_bpb 1.06044 (3-seed mean)

**val_bpb: 1.06044** (3-seed mean, std 0.00091) | **15,950,342 bytes max** | 8×H100 SXM | full TTT eval

**Improvement over PR #1908 (current candidate at 1.06081):** **−0.00037 BPB**, all three seeds compliant under the 600s training cap.

## Why this PR exists

PR #1908 (romeerp) introduced an activation-aware GPTQ mixed-precision base on top of PR #1855 and reported a 3-seed mean val_bpb of `1.06081076`. Their seed-42 training run, however, consumed `601153 ms` of training wallclock — `1153 ms` over the 600,000 ms (10-minute) cap that defines the `track_10min_16mb` track. This submission re-runs the same recipe with the same quantization knobs, removing the over-cap path so every seed completes strictly under 600 s.

The PR #1908 author flagged the issue themselves:

> I personally don't want to have to re-run these three seeds, so this should be open for anyone who wants to claim a new record if they can re-run it under the 600s wallclock on a better GPU setup …
> — @romeerp on PR #1908

This submission takes them up on it.

## Results

| Seed | Stop step | Train wallclock (ms) | Pre-quant BPB | Quantized BPB | **Post-TTT BPB** | Artifact bytes |
|------|-----------|---------------------:|--------------:|--------------:|----------------:|---------------:|
| 42   | 4,960 | 599,521 | 1.06363 | 1.07254 | **1.05938494** | 15,943,518 |
| 0    | 4,942 | 599,665 | 1.06532 | 1.07407 | **1.06101359** | 15,945,548 |
| 1234 | 4,943 | 599,676 | 1.06501 | 1.07478 | **1.06092004** | 15,950,342 |
| **Mean** | **4,948** | **599,621** |               |               | **1.06043952** | **15,946,469** |

3-seed std: 0.00091 BPB. All three seeds clear both the 600,000 ms training cap and the 16,000,000 byte artifact cap.

### Comparison with PR #1908

| Metric | PR #1908 | This submission | Δ |
|---|---:|---:|---:|
| 3-seed mean post-TTT val_bpb | 1.06081076 | **1.06043952** | **−0.00037124** |
| Seed 42 post-TTT val_bpb | 1.05957221 | 1.05938494 | −0.00019 |
| Seed 0 post-TTT val_bpb | 1.06127329 | 1.06101359 | −0.00026 |
| Seed 1234 post-TTT val_bpb | 1.06158679 | 1.06092004 | −0.00067 |
| Seed 42 train wallclock | **601,153 ms** (over cap) | **599,521 ms** | −1,632 |
| Seed 0 train wallclock | matched-step | 599,665 ms | — |
| Seed 1234 train wallclock | matched-step | 599,676 ms | — |
| Max artifact bytes | 15,996,559 | 15,950,342 | −46,217 |

## What changed vs PR #1908

**Code:** `train_gpt.py` is byte-identical to the PR #1908 submission (commit `291d3abd` on `romeerp/parameter-golf:codex/awq-stepmatched`). All architectural and quantization logic — including the activation-aware GPTQ mixed-precision path, the LQER asymmetric int4 correction, and the per-group lrzip-zpaq compression — is unchanged.

**Run config:** the only difference is the wallclock control path.

| | PR #1908 | This submission |
|---|---|---|
| `MAX_WALLCLOCK_SECONDS` | 0 | 600 |
| `FORCE_STOP_STEP` | 4945 (forces step count regardless of wallclock) | unset (organic 600 s cap) |

PR #1908 used `FORCE_STOP_STEP=4945` to step-match against PR #1855's stopping steps; with that flag set, `train_gpt.py` ignores the wallclock cap. On the GPU instance available to that submission, hitting 4945 steps required 601,153 ms — over the cap.

This submission removes `FORCE_STOP_STEP` and lets training stop organically at the 600,000 ms wallclock cap. On the GPU instance used here (8×H100 80GB SXM, RunPod community cloud), that yields stop steps of 4960 / 4942 / 4943 — within ~15 steps of PR #1908's targets — at 599,521 / 599,665 / 599,676 ms.

**Quantization knobs** are unchanged from PR #1908 baseline:

| | PR #1908 | This submission |
|---|---|---|
| `AWQ_LITE_ENABLED` | 1 | 1 |
| `AWQ_LITE_BITS` | 8 | 8 |
| `AWQ_LITE_GROUP_TOP_K` | 1 | 1 |
| `AWQ_LITE_GROUP_SIZE` | 64 | 64 |
| `LQER_ENABLED` | 1 | 1 |
| `LQER_RANK` | 4 | 4 |
| `LQER_TOP_K` | 3 | 3 |
| `LQER_GAIN_SELECT` | 0 | 0 |
| All other PR #1855 hparams | unchanged | unchanged |

## Reproducing

Same dataset and tokenizer as PR #1908:

- dataset: `romeerp/parameter-golf-caseops-v1` (HuggingFace)
- variant: `sp8192_lossless_caps_caseops_v1_reserved`

```bash
DATA_DIR=./data \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
VOCAB_SIZE=8192 CASEOPS_ENABLED=1 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
GLOBAL_TTT_MOMENTUM=0.9 WARMDOWN_FRAC=0.85 BETA2=0.99 \
TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 VAL_LOSS_EVERY=0 \
GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 \
SMEAR_GATE_ENABLED=1 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 AWQ_LITE_GROUP_TOP_K=1 AWQ_LITE_GROUP_SIZE=64 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

(Replace `SEED=42` with `SEED=0` and `SEED=1234` for the other two seeds.)

System dependencies: PyTorch 2.9.1+cu128, CUDA 12.8, 8×H100 80GB SXM, `flash_attn_3` (separate install — see `requirements.txt`), `lrzip` system binary (`apt-get install lrzip`).

## Files

- `train_gpt.py` — verbatim copy of PR #1908's `train_gpt.py` (commit `291d3abd`, ~3,998 lines). Configurable via env vars; the env-var values listed above reproduce these results.
- `submission.json` — structured per-seed metadata + compliance attestation.
- `requirements.txt` — minimal Python deps.
- `train_seed42.log`, `train_seed0.log`, `train_seed1234.log` — full per-seed run logs.

## Credits

This submission stands entirely on the work of:

- **PR #1908** (@romeerp) — activation-aware GPTQ mixed-precision base; this submission is a compliance-fixed reproduction of that work, taken with the author's explicit invitation in [PR #1908 comment](https://github.com/openai/parameter-golf/pull/1908).
- **PR #1855** (@codemath3000) — full architectural stack with BOS-fixed SmearGate and 9-hparam greedy stack.
- **PR #1797** (@dexhunter) — Smear Gate + LQER asymmetric int4 correction.
- **PR #1787** (@nprime06) — Polar-Express Newton-Schulz, sparse attention gate, MIN_LR floor, fused softcapped CE.
- **PR #1736** — CaseOps + GatedAttn + QuantGate + Loop4-5 + PhasedTTT integration.
- **PR #1729** (@romeerp) — sp8192 lossless caps caseops v1 reserved tokenizer.
- And the rest of the PR #1855 lineage as listed in PR #1908's README.

The contribution of this submission is narrow: **demonstrating that the PR #1908 stack achieves its quality strictly within the 600-second training cap** when run on stock GPU instances with organic wallclock control, eliminating PR #1908's known compliance overshoot.
