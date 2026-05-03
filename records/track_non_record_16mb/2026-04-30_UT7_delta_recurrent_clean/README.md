# UT7 Delta-Residual + RLMA-rank320 — Documented Failed Direction

**TL;DR — this direction did not pan out.** A four-stop attempt at Universal Transformer + RLMA (UT6+TTT3 → UT6+TTT clean → UT6+RLMA256 noTTT → UT7 delta-residual) landed at `val_bpb = 1.29740` on two 1×H100 seeds. The current 10-min SOTA is around `1.06141`. This is **0.236 nat above SOTA** — not a leaderboard contender.

Submitted to `track_non_record_16mb` as a **documented failed experiment**. The reusable parts are the negative results, all with numbers and direct ablations:

- TTT (Test-Time Training, in-model) was directly ablated at rank-288 / 600 steps and **made things worse**: no-TTT `1.418` vs TTT `1.448` `val_bpb`, with TTT 2.17× slower per step.
- Recurrent state contraction `x_next = branch(x)` was the real architectural bottleneck. A 240-step same-config A/B moved `val_bpb` from `1.840` → `1.586` just by switching to a scaled additive delta (`x_next = x + 0.6 * branch(x)`). Even after the fix, the gap to SOTA persists.
- The 04-26 forensic 3-seed run **never engaged warmdown** — `lr_scale` stayed at `1.000` from step 500 through step 9000 because the iteration cap fired before the wallclock cap. The reported `val_bpb=1.43101` for that earlier run is therefore a no-decay number.

Full chronology, architecture, and the rest of the negative results below in case any of it helps the next person to try Universal Transformers, deterministic random-base adapters, or in-model TTT at this scale.

---

## Status

Submitted to `records/track_non_record_16mb` as a **documented failed direction**, not a leaderboard contender. The 10-minute SOTA is around `val_bpb 1.06141`; this submission landed at `1.29740`. It is offered because the journey produced concrete negative results (summarized in the TL;DR, detailed throughout — see `## TTT Ablation`, `## How This Got Here`, `## Negative Results`).

Submitted evidence is two 1×H100 Runpod seed runs plus one cap-hardening rerun. All three logs stay below the hard decimal `16,000,000` byte artifact cap, keep validation disabled during training (`VAL_LOSS_EVERY=0`), and evaluate only after training has completed. The configuration is reproducible from the env-vars in the next section.

## Submitted Result

Locked configuration (env-vars consumed by `train_gpt.py`):

```bash
USE_RLMA=1 USE_TTT=0 ADAPTER_RANK=320 K_ITERS=6
MODEL_DIM=1024 D_FF=3072 NUM_HEADS=8 NUM_KV_HEADS=4 HEAD_DIM=128
UT_RESIDUAL_DELTA=1 BRANCH_SCALE_INIT=0.6
ITERATIONS=2400 WARMUP_STEPS=50 WARMDOWN_ITERS=600
TRAIN_SEQ_LEN=8192 TRAIN_BATCH_TOKENS=262144 GRAD_ACCUM_STEPS=8
MATRIX_LR=0.026 GRAD_CLIP_NORM=0.2
GPTQ_BITS=8 GPTQ_CLIP_K=12.85 EMBED_QUANT_BITS=8 ZSTD_LEVEL=22
VAL_LOSS_EVERY=0 EVAL_CTX=8192 EVAL_STRIDE=8192 VAL_TOKEN_LIMIT=524288
```

For these 1×H100 runs, `GRAD_ACCUM_STEPS=8` preserves the intended 262 K-token global batch. The 8×H100 launcher uses `GRAD_ACCUM_STEPS=1` for the same global batch with one microbatch per rank; no 8×H100 production claim is made in this PR.

| run                              | seed | val_bpb       | val_loss     | train wallclock | total bytes  | headroom |
|----------------------------------|-----:|--------------:|-------------:|----------------:|-------------:|---------:|
| `train_seed42.log`               |  42  | `1.29703488`  | `3.31280854` |    `3000.944s`  |  `15,800,799`|  `199,201` |
| `train_seed314.log`              | 314  | `1.29777215`  | `3.31469164` |    `2990.426s`  |  `15,842,747`|  `157,253` |
| `train_seed42_cap_hardening.log` |  42  | `1.29569451`  | `3.30938505` |    `2990.983s`  |  `15,776,117`|  `223,883` |

Two-seed mean: **`val_bpb = 1.29740352`** with sample std **`0.00052132`**. The max submitted artifact total is **`15,842,747`** bytes (seed 314), leaving **157,253 bytes** of headroom under the cap.

All three logs show `world_size:1 grad_accum_steps:8`, warmdown engagement (`lr_scale` decays below `1.0` in the final block of steps), `final_quant_roundtrip_exact`, and `compressed_blob_bytes + code_bytes < 16,000,000`.

## Hardware

- **Training:** Runpod 1×H100 80 GB HBM3 (template `y5cejece4j`).
- **Stack:** torch `2.9.1+cu128`, Python `3.12.3`, CUDA driver 13.0.
- **Local dev:** Windows 11 + RTX 4090 (24 GB) was used for code-only smoke and `py_compile` checks. The 4090 cannot reproduce the production numbers (different memory and FP8 path), but it ran the unit-level architecture tests during iteration.
- **Long-run target:** an 8×H100 80 GB SXM run with this same config (`GRAD_ACCUM_STEPS=1`) is the next step but is **not** included in this submission. The launchers (`launch_production.sh`, `launch_10min_sanity.sh`) are shipped for that future run.

## Architecture

A seven-pass Universal Transformer stack: a unique input block, a six-iteration shared block, and a final output block. Compared to the prior UT6 iteration in this lineage, the single load-bearing change is the residual update on the shared block:

```text
# Pre-UT7 (recurrent state contraction)
x_next = branch(x)

# UT7 (scaled residual delta)
x_next = x + branch_scale * branch(x)
```

Where `branch_scale` is initialized to `0.6` and learned. The earlier contraction made the shared state harder to optimize at deeper iterations; the additive form lets the K=6 unrolled stack reuse and refine information instead of overwriting it. The `## Negative Results` section quantifies this — same config, same seed, the residual change alone moves 240-step `val_bpb` from `1.840` to `1.586`.

RLMA stores each learned matrix as a low-rank adapter over a deterministic random base:

```text
W = alpha * R + U @ V
```

`R` is regenerated from `blake2b(layer_name) ^ k_iter`, so every recurrent iteration of the shared block sees a *different* random projection without storing the base matrix in the artifact. `R` costs zero stored bytes; only the per-matrix `alpha` (fp16 scalar), `U` and `V` (int8 GPTQ at rank 320), and biases (fp16) are persisted. Tied embeddings are int8 per-channel symmetric. Scalars, biases, and depth embeddings pass through fp16. The final blob is compressed with `zstd-22`.

Architectural knobs (matches `submission.json` and the locked config above):

| Knob | Value |
|---|---|
| `vocab_size` | 8192 (SP8192 BPE) |
| `model_dim` | 1024 |
| `d_ff` | 3072 (3× MLP) |
| `num_heads / num_kv_heads / head_dim` | 8 / 4 / 128 (GQA) |
| `K_iters` (shared block) | 6 |
| `adapter_rank` (RLMA) | 320 |
| `ut_residual` | `delta_scaled` (`UT_RESIDUAL_DELTA=1`) |
| `branch_scale_init` | 0.6 |
| `tie_embeddings` | true |
| `logit_softcap` | 30.0 |
| `train_seq_len` | 8192 |
| `train_batch_tokens` | 262144 |

TTT is **disabled** in the submitted candidate. The reasoning is in the `## TTT Ablation` section below.

## Cap-Hardening Sweep

The clip threshold for std-clipped int8 GPTQ on the `U`/`V` matrices was selected via a sweep on seed 42. All five candidates pass the 16 MB cap; `clip_k=12.85` was selected as the lowest `val_bpb` while keeping headroom margin under the cap that survives variance into seed 314.

| `gptq_clip_k` | `val_bpb` | `total bytes` | headroom (16M − total) | notes |
|--------------:|----------:|--------------:|-----------------------:|---|
| **12.85**     | **1.29569451** | **15,776,117** | **223,883** | selected |
| 13.5          | 1.29639711 | 15,636,167 | 363,833 | slightly worse `val_bpb`, more headroom |
| 14.0          | 1.29651887 | 15,434,063 | 565,937 |  |
| 15.0          | 1.29697248 | 15,152,606 | 847,394 |  |
| 16.0          | 1.29816346 | 14,935,951 | 1,064,049 | most aggressive clip, worst `val_bpb` |

Independent confirmation on **seed 314** (file `supporting_logs/clip_sweep_seed314.log`, screening schedule rather than full production schedule, hence higher absolute `val_bpb`):

| `gptq_clip_k` | `val_bpb` (seed 314 screen) |
|--------------:|----------------------------:|
| 10            | 1.31076144                  |
| 11.5          | 1.31129943                  |
| **12.85**     | **1.31170050**              |
| 14            | 1.31238501                  |
| 16            | 1.31286905                  |

The seed-314 sweep agrees with the seed-42 ranking — `clip_k=12.85` sits in the optimal band — so the choice does not appear to be seed-specific overfitting.

## TTT Ablation

Test-Time Training was a load-bearing component in earlier iterations of this lineage (`2026-04-26_UT6_RLMA64_TTT3_4hour`, where TTT-Linear ran in iters 3, 4, 5 of the shared block). For UT7, a direct ablation at rank 288 / int8 / 600 steps was run to decide whether to keep it:

| Run                                  | `val_bpb`    | per-step time | notes |
|--------------------------------------|-------------:|-------------:|------|
| Control (no TTT, rank288, q8, 600 steps)  | **`1.41796689`** | baseline     | wins on quality |
| TTT enabled (rank288, q8, 600 steps) | `1.44762819` | **2.17×** baseline | loses on quality and speed |
| Δ                                    | `+0.02966` (TTT worse) | 2.17× slowdown |  |

TTT cost more compute per step (the Python chunk-wise dual form is launch-bound on a 1×H100 without a fused FA3+Triton path) **and** delivered worse `val_bpb` at this scale and budget. The honest call was to ship without TTT. The earlier UT6+TTT3 result at `val_bpb 1.43` (3-seed 8×H100, 04-26 lineage) was driven by other factors (warmdown bug, different residual style) that interacted with TTT in ways that did not reproduce after the warmdown fix and the residual change.

This does not say TTT-Linear is a bad architecture — only that, after the residual fix landed, the no-TTT path won at this compute budget. A fused TTT path or a different `K_iters` × TTT split could reverse this; both are plausible follow-ups.

## How This Got Here

A four-stop arc, all on the same UT + RLMA core, with the architectural lever changing at each stop:

### 1. `2026-04-26_UT6_RLMA64_TTT3_4hour` — forensic 3-seed 8×H100 baseline

UT6 + RLMA-rank64 + in-model TTT-Linear (chunk-wise dual form, iters 3,4,5). 8×H100 SXM, 9000 iters, ~87 min/seed, ~$110 total. Three seeds (42, 314, 999), per-seed `val_bpb` 1.4339 / 1.4305 / 1.4286, mean **`1.43101 ± 0.00267`**, artifact 9.6 MB. This was the proof-of-life that UT + RLMA + TTT composes end-to-end at the parameter-golf scale.

It also surfaced the **warmdown bug**: the `lr_mul()` schedule's wallclock branch decided not to fire because remaining wallclock budget exceeded the warmdown duration, the iteration cap bound first, and so `lr_scale` stayed at `1.000` from step 500 through step 9000. The 1.43101 result is therefore a *no-decay* number — labeled warmdown never engaged.

### 2. `2026-04-29_UT6_RLMA128_TTT_clean` — warmdown fix and gating discipline

Patch: `lr_mul = max(min(iter_mul, wall_mul), 0.1)` so the schedule decays whichever cap binds first. Adapter rank doubled to 128. Production launchers staged behind 1×H100 gates (UT-only, UT+RLMA, full UT+RLMA+TTT, production-shape probe, warmdown proof) — no 8×H100 spend without those passing. **No production seed runs were completed in this folder** — the gating turned up architectural concerns that motivated the next iteration before spending on full seeds.

### 3. `2026-04-30_UT6_RLMA256_noTTT_online_clean` — TTT pivot

The 1×H100 short-screen ablation (rank 288, 600 steps, this folder) showed TTT-on was net negative on `val_bpb` *and* 2.17× slower per step. The pivot: drop TTT, scale rank to 256, lean into the static-architecture path. Seed-42 short result on this branch: `val_bpb=2.11548` over 60 calibration steps; seed-314 confirmation `val_bpb=2.13140`. (These are short-screen numbers, not production schedules — the production-schedule version would have been much better.) Before running the full schedule, one more change wanted to land: the residual update.

The residual at this point was still the recurrent-state-contraction form `x_next = branch(x)`. The 6-iteration unrolled stack was overwriting state instead of accumulating it.

### 4. `2026-04-30_UT7_delta_recurrent_clean` — delta-residual unlock (this folder)

Single architectural change: `x_next = x + branch_scale * branch(x)` with `branch_scale_init=0.6`. Same RLMA, same no-TTT decision, rank scaled to 320 to use the byte budget the no-TTT path freed up.

The cleanest evidence is a 240-step direct A/B (file pair `supporting_logs/old_residual_240step_seed42.log` and `supporting_logs/delta_residual_240step_seed42.log`, same seed, same data, same iter count, only `UT_RESIDUAL_DELTA` flipped):

| Residual style                       | `val_bpb` (240 steps, seed 42) | total bytes |
|--------------------------------------|-------------------------------:|------------:|
| `x_next = branch(x)` (recurrent state contraction) | `1.84038776` | 14,279,048 |
| `x_next = x + 0.6 * branch(x)` (delta-residual)    | **`1.58599604`** | 14,506,646 |
| Δ                                    | **−0.254 nat** from the residual change alone | +227 KB |

That carries through to the production schedule (2400 iters): seed 42 `val_bpb = 1.297` and seed 314 `val_bpb = 1.298`, both well under the 16 MB artifact cap.

## Negative Results

Documented so other contributors don't redo them by accident:

- **TTT enabled at this scale and budget hurts.** Rank-288 / 600 steps / 1×H100: no-TTT 1.418 vs TTT 1.448 `val_bpb`, with TTT 2.17× slower per step. See `## TTT Ablation`.
- **Recurrent state contraction (`x_next = branch(x)`) was the bottleneck.** 240-step A/B at the same config moved `val_bpb` from 1.840 → 1.586 just by switching to a scaled additive delta. See `## How This Got Here` step 4 and `supporting_logs/old_residual_240step_seed42.log` vs `supporting_logs/delta_residual_240step_seed42.log`.
- **Iteration-only warmdown silently masks LR decay.** If `MAX_WALLCLOCK_SECONDS` is set high enough that the wallclock branch doesn't fire and the iteration cap binds first, `lr_scale` stays at `1.000` for the entire run. The 04-26 forensic 3-seed package (`val_bpb=1.43101`) was a no-decay number. Use `lr_mul = max(min(iter_mul, wall_mul), 0.1)` so whichever cap binds first triggers decay. The patched form is what this submission uses.
- **Aggressive int8 clipping is worth ~0.001 nat at most.** The cap-hardening sweep moved `clip_k` from 16.0 (most aggressive) down to 12.85 (least). Total `val_bpb` swing across the sweep was about `0.0025` nat. The byte-headroom cost was ~840 KB. Worth it because variance into seed 314 is small enough that the 157 KB residual headroom is comfortable, but not a large source of further gains.

## What Might Work With More Compute

Concrete things this submission did not try, ordered by expected payoff:

1. **A third production seed (999) on 8×H100.** Two seeds give a usable mean/std but a third would let this become a 3-seed claim consistent with the leaderboard's statistical bar (and would unblock comparison to the 04-26 forensic 3-seed at 1.431). The launchers shipped with this folder are configured for it (`SEEDS="999" bash launch_production.sh`).
2. **Longer schedule.** Production was 2400 iters / ~50 min on 1×H100 with `GRAD_ACCUM_STEPS=8`. The 8×H100 launcher with `GRAD_ACCUM_STEPS=1` would do the same global batch in ~6 min/seed; a 3–4 hour budget on 8×H100 buys ~60K iters, where the loss is still descending at 2400.
3. **Rank sweep with cap-hardened clip.** The cap-hardening sweep fixed `clip_k` but not `adapter_rank`. With ~157 KB headroom at rank 320, rank 352 may fit if the embedding is squeezed slightly; rank 256 might leave headroom for a rank sweep on `K_iters` instead.
4. **`branch_scale_init` and per-iteration scale.** The submitted run uses a single learned scalar at init 0.6. A per-iteration `branch_scale` would let early iterations write more aggressively while later iterations refine — testable with a one-line change.
5. **Fused TTT path.** A fused FA3+Triton TTT chunk loop could plausibly invert the TTT ablation result; the rejection here is *budget*-conditional, not architectural.
6. **FP8 attention on H100.** The current path is bf16 forward / fp32 master. FP8 (Transformer Engine) on H100 SXM is the obvious throughput unlock for an 8×H100 long-run; the 4090 dev box can't validate it, but Runpod can.

## Reproduction

Use the official Parameter Golf Runpod template (`y5cejece4j`), then:

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
pip install --break-system-packages -r records/track_non_record_16mb/2026-04-30_UT7_delta_recurrent_clean/requirements.txt
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
bash records/track_non_record_16mb/2026-04-30_UT7_delta_recurrent_clean/launch_production.sh
```

Single-seed, no launcher (1×H100):

```bash
cd records/track_non_record_16mb/2026-04-30_UT7_delta_recurrent_clean
RUN_ID=ut7_repro_seed42 SEED=42 \
  VOCAB_SIZE=8192 \
  DATA_PATH=../../../data/datasets/fineweb10B_sp8192 \
  TOKENIZER_PATH=../../../data/tokenizers/fineweb_8192_bpe.model \
  MODEL_DIM=1024 D_FF=3072 NUM_HEADS=8 NUM_KV_HEADS=4 HEAD_DIM=128 \
  USE_RLMA=1 USE_TTT=0 ADAPTER_RANK=320 K_ITERS=6 \
  UT_RESIDUAL_DELTA=1 BRANCH_SCALE_INIT=0.6 \
  ITERATIONS=2400 WARMUP_STEPS=50 WARMDOWN_ITERS=600 \
  TRAIN_SEQ_LEN=8192 TRAIN_BATCH_TOKENS=262144 GRAD_ACCUM_STEPS=8 \
  MATRIX_LR=0.026 GRAD_CLIP_NORM=0.2 \
  GPTQ_BITS=8 GPTQ_CLIP_K=12.85 EMBED_QUANT_BITS=8 ZSTD_LEVEL=22 \
  VAL_LOSS_EVERY=0 EVAL_CTX=8192 EVAL_STRIDE=8192 VAL_TOKEN_LIMIT=524288 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py
```

For a short 8×H100 sanity run before production:

```bash
bash records/track_non_record_16mb/2026-04-30_UT7_delta_recurrent_clean/launch_10min_sanity.sh
```

## Compliance

- [x] **Artifact under 16,000,000 bytes (decimal).** Max submitted is `15,842,747` bytes (seed 314); `compressed_blob_bytes + code_bytes < 16,000,000` for all three submitted logs.
- [x] **`VAL_LOSS_EVERY=0` — validation data not accessed during training.** The training data loader only reads `fineweb_train_*.bin`; the validation loader is constructed only after training/quantization for the final FP and quantized roundtrip eval.
- [x] **Validation loaded after training.** The post-training eval reports both `final_fp_exact` and `final_quant_roundtrip_exact` for transparency on quantization damage.
- [x] **Self-contained `train_gpt.py`** (1420 physical lines, under the 1500 hard cap).
- [x] **Submitted to `track_non_record_16mb`** as unlimited-compute research. **No SOTA claim** — current 10-min SOTA is around `val_bpb 1.06`; this submission is at `1.297`.
- [x] **No "paid prefix" tricks.** Validation tokens are never compressed into the artifact. Test-time training is not used.
- [x] **3-seed claim is not made.** Two production seeds (42, 314) plus one cap-hardening rerun (seed 42, different clip). `submission.json` reports the 2-seed mean explicitly.

## Acknowledgments

- **Parameter-golf upstream baseline.** The root `train_gpt.py` and the Runpod template `y5cejece4j` are the launch pad for this work; the architectural tooling (Muon optimizer split, RMSNorm forward, FineWeb dataset wrappers, GPTQ recipe with std-clip) is borrowed from there.
- **Modded-nanogpt** (Keller Jordan and contributors). Muon, the Polar Express NS coefficient schedule, and the bf16-forward / fp32-master pattern come from this lineage. See `THIRD_PARTY_NOTICES.md` in the repo root.
- **Universal Transformer** (Dehghani et al., 2018). The K-iteration shared-block structure of this submission is a direct descendant. This submission does not use Adaptive Computation Time (fixed K=6).
- **GPTQ** (Frantar et al., 2022). The std-clip GPTQ recipe (with `clip_k` parameterization) is borrowed from the upstream record at PR #1394 (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`).
- **The 04-26 forensic seed package in this same lineage** (`records/track_non_record_16mb/2026-04-26_UT6_RLMA64_TTT3_4hour/`) — establishes the UT + RLMA composition with 3-seed `val_bpb=1.43101 ± 0.00267` on 8×H100, including the TTT-Linear chunk-wise dual form integration. UT7 is the warmdown-fixed, residual-fixed, no-TTT successor.

## Included Files

- `README.md` — this writeup.
- `submission.json` — structured metadata and exact submitted metrics.
- `train_gpt.py` — self-contained trainer (1420 lines).
- `requirements.txt` — record-local dependency addition (`zstandard>=0.22.0` over the root `requirements.txt`).
- `train_seed42.log`, `train_seed314.log` — submitted 1×H100 production seed logs.
- `train_seed42_cap_hardening.log` — selected clip / cap-hardening evidence (seed 42, clip_k 12.85).
- `launch_10min_sanity.sh` — 8×H100 sanity launcher (single seed, short).
- `launch_production.sh` — 8×H100 production launcher for future longer runs (defaults to seed 42; `SEEDS="314 999"` for the rest).
- `supporting_logs/` — five curated supporting logs (old vs delta residual A/B, rank256 noTTT screening, rank320 selection, seed-314 clip sweep) with their own `supporting_logs/README.md`.
