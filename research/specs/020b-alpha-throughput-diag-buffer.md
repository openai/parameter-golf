# Spec 020b — Throughput diagnostic on buffer-α (4×H100 cheap test)

**Slug:** `alpha-throughput-diag-buffer`
**Created:** 2026-04-21
**Status:** READY
**Links to:** `research/specs/020-alpha-throughput-diag.md` (literal-α diagnostic), `research/specs/021-recur-alpha-buffer.md` (buffer-α full pipeline)

## Why 020b

Spec 021 proposes `register_buffer`-α as a fix for 019/019b's dip pattern. Before committing a full 8×H100 pipeline run (~$10), we want a **cheap dip-pattern validation** on 4×H100 (~$3-4) that answers one question:

> **Does buffer-α show the same post-val dip pattern as 019/019b, or is it dip-free like 017?**

If dip-free → 021's hypothesis holds → launch 021 full pipeline with confidence. If dips like 019b → buffer's "not-a-constant" theory is wrong, 021's full run is a waste, pivot to 020's diagnostic.

4×H100 is chosen because:
- Better availability than 8×H100 right now
- Half the step count (~2400 vs ~4700) is still plenty to see dips — dip *pattern* doesn't require full 4800 steps
- Post-TTT val_bpb from 4×H100 is NOT comparable to #1736, but we don't need it for this test

## Hypothesis

Buffer-α has the same graph structure as 017's tensor-α (Dynamo treats both as runtime tensor inputs, not compile-time constants). Therefore:
- Post-activation steady-state: dip-free (like 017, unlike 019/019b)
- Post-val (after train→eval→train mode switch): no cold-path penalty, no dip cluster

Test: run val mid-training (`VAL_LOSS_EVERY=1500` on 4×H100) and check whether intervals 1500-1700 show a dip cluster.

## Baseline (for dip-pattern comparison only)

Dip intervals observed on 8×H100 (post-activation):
- 017 (tensor α, manual): **0 dips**
- 019b (literal α, manual-algebraic): **7 dips** incl. cluster at 4100-4200 (right after val at step 4000)

4×H100 is a different regime. What we're comparing is the *dip pattern shape*, not absolute throughput.

## Accept criteria

Primary: dip count in the 020b run's `diag_steps.csv` post-activation (step > ~515 on 4×H100).

| buffer-α dip count | Interpretation | Next action |
|---|---|---|
| 0-1 | Matches 017 pattern; hypothesis confirmed | Launch spec 021 full 8×H100 run (high confidence) |
| 2-4 | Ambiguous; scattered noise but no cluster | Launch 021 anyway (still reasonable bet); flag result |
| 5+ with post-val cluster | Dips like 019b | Buffer-α doesn't fix the vulnerability; spec 021 is not the right fix; pivot to 020 full diagnostic |

Secondary: the post-val window (steps 1500-1700) specifically. If that window is dip-free, we have a strong answer to the train↔eval recompile hypothesis independent of total dip count.

## Config diff

| var | value | note |
|---|---|---|
| `RECUR_ALPHA_ENABLED` | `1` | buffer-α path |
| `ENABLE_LOOPING_AT` | `0.17` | activation ~step 515 on 4×H100 |
| `VAL_LOSS_EVERY` | `1500` | mid-run val fires once at step 1500; test post-val dip |
| `THROUGHPUT_DIAG` | `1` | enables `diag_steps.csv` output |
| `TTT_ENABLED` | `0` | skip TTT (saves ~$3); not needed for this test |
| `TRAIN_LOG_EVERY` | `100` | standard |

All other 019b/021 flags unchanged.

## Code changes

**Branch:** `exp/alpha-throughput-diag-buffer` forking from `cb5cd78` (spec 021 buffer-α).
**Commit:** `3cfc372` — buffer-α + instrumentation + cuda.synchronize fix.
- `cb5cd78` — buffer-α (from 021)
- `2483bb5` — 020's instrumentation cherry-picked
- `3cfc372` — cuda.synchronize() before elapsed_time() (fix for -1.0 readings)

No new code. This branch is the intersection:
- Buffer-α from 021 (cb5cd78): `register_buffer` at 017 endpoint values
- Instrumentation from 020 (9bb1b01): per-step fwd/bwd/opt μs, dataloader μs, allocator stats, CSV output when `THROUGHPUT_DIAG=1`

## Hardware ladder

**4×H100 JP** — this is the whole point (cheap, available).

If 4×H100 also unavailable, fall back to 2×H100 — **but** the full 11L/512d model may OOM on 2×H100 DDP (per 016b). If OOM, abort 020b and go straight to 021 on 8×H100 when it becomes available.

8×H100 acceptable as a bonus (just delay 021 and use its slot), but 020b is intentionally scoped to 4×H100 for cost.

## Seed plan

Single seed 42.

## Inputs

- Data: CaseOps dataset, JP mount at `/runpod/data/...`
- Tokenizer: `fineweb_8192_bpe.model` bundled
- Hotstart: none

## Run protocol

```bash
cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git checkout 2483bb5

mkdir -p /runpod/runs/020b-alpha-throughput-diag-buffer/seed_42
mkdir -p /runpod/.torch_inductor_cache_020b

# nvidia-smi sidecar (optional for 020b — dip pattern can be read from diag_steps.csv alone)
nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /runpod/runs/020b-alpha-throughput-diag-buffer/seed_42/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/020b-alpha-throughput-diag-buffer/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/runpod/.torch_inductor_cache_020b \
CASEOPS_ENABLED=1 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
ENABLE_LOOPING_AT=0.17 \
VAL_LOSS_EVERY=1500 \
TTT_ENABLED=0 THROUGHPUT_DIAG=1 \
TRAIN_LOG_EVERY=100 \
SEED=42 \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /runpod/runs/020b-alpha-throughput-diag-buffer/seed_42/train.log 2>&1

kill $NVSMI_PID
```

## Checkpoints / artifacts to emit

- `train.log` — every-100-step tok/s
- `diag_steps.csv` — per-step diagnostics (~2400 rows at 4×H100)
- `diag_nvsmi.csv` — per-GPU per-second (~2400 rows)
- `final_model.pt` — post-EMA
- `final.json` — summary

No GPTQ/TTT artifacts (`TTT_ENABLED=0` skips TTT; GPTQ still runs but we don't care about the result).

## Stop-early criteria

- NaN/inf in loss → halt
- Compile > 5 min → halt (something wrong with fresh cache)
- If `layer_loop_enabled_at_step` > 700 → halt (activation didn't fire early as expected)

## Cost estimate

| item | cost |
|---|---|
| 4×H100 JP × ~12 min (training + GPTQ + instrumentation, no TTT) | ~$3 |
| **020b total** | **~$3-4** |

## Analysis on completion (zero cost)

Reuse the playbook from `research/ideas/020-diag-analysis-plan.md`. Two key queries:

1. **Steady-state dips:** `pandas.read_csv('diag_steps.csv')`; count intervals with `step_time_ms > 1.15 × median` in the post-activation regime (step > 515).
2. **Post-val cluster:** slice steps 1500-1700; compute mean step_time_ms. If > 1.15 × steady-state mean, post-val dip is present.

## Decision tree

- **Dip-free + no post-val cluster** → 021's hypothesis confirmed → launch 021 full 8×H100 pipeline
- **Dip-free steady state + post-val cluster still present** → buffer fixes the "steady dips" but not the val-recompile dip → investigate more before 021 (maybe the val-triggered recompile is unavoidable with any attribute on the module)
- **Dips throughout** → buffer doesn't fix it → pivot to spec 020's full diagnostic to find the real cause

## Open questions for interview (execution)

1. **4×H100 OOM check.** 11L/512d on 4 GPUs should fit (8× fits at 40 GiB peak, so 4× doubles to ~80 GiB — tight but should be under H100's 80 GB cap with DDP). If OOM on 4×H100, can't run 020b there; escalate to user.
2. **Inductor cache is fresh.** New branch, no cache reuse. Compile ~1-2 min expected.
3. **VAL_LOSS_EVERY=1500 interaction.** Verify val fires at step 1500 (not deferred or skipped) by grepping `val_loss:` in train.log after the run.
