# Spec 020 — Alpha-throughput diagnostic (instrumented 019b rerun)

**Slug:** `alpha-throughput-diag`
**Created:** 2026-04-21
**Status:** READY (code changes in progress)
**Links to:** `research/ideas/throughput-alpha-proxy-gap.md`

## Hypothesis

The ~3% post-loop-activation throughput tax on constant-α runs is driven by sporadic "dip intervals" where tok/s collapses by 15-30% for 1-2 intervals. These dips happen silently — no existing logged event explains them. Running 019b with heavy instrumentation will reveal the mechanism: which phase slows (forward/backward/optimizer), whether allocator stats shift, whether GPU clocks/power change, whether the dip is a single slow step or many, whether dataloader stalls.

## Baseline

019b original run: `runs/019b-recur-alpha-manual-constant-full/seed_42/`. Commit `9517a3b`. Dip intervals at steps 2900, 3000, 3100, 3500, 3600, 4100, 4200 (loop activated at step 2143).

**This spec uses EARLY ACTIVATION** (`ENABLE_LOOPING_AT=0.17` → activation at ~step 1040 instead of ~2143). Rationale: gives ~3660 post-activation steps vs ~2570 in a normal run (40% more diagnostic data), plus enables an activation-offset diagnostic:

- If dips happen at the **same absolute step indices** (2900, 3500, 4100 still) → step-indexed mechanism (LR schedule, optimizer state, step-count-keyed event).
- If dips happen at **constant offset from activation** (e.g. ~750 steps after wherever activation fired) → time-since-activation mechanism (thermal equilibration, kernel warmup cycle, cache/allocator state).

Cross-reference the original 019b activation-at-2143 run as the "normal activation" arm.

## Accept criteria

Per run we want, at minimum, for every step (~4700 steps):
- Wall-time (timestamp, step_time_ms)
- Allocator stats (active_bytes, reserved_bytes, num_alloc_retries, num_device_alloc)
- Forward/backward/optimizer μs (CUDA events)
- Dataloader batch μs

Plus a background CSV of per-GPU temp/sm_clock/power/util every 1s.

Decision criterion — do dip steps reproduce across the two pods?
| outcome | interpretation | next step |
|---|---|---|
| Dip steps **match** (>50% overlap) | **Deterministic/internal** | Per-phase timing reveals which phase; spec a targeted fix |
| Dip steps **don't match** (<20% overlap) | **External/contention** | nvidia-smi correlation reveals thermal/power/interconnect; fix at infra level |
| Mixed | Both contribute | Combine both fix paths |

## Code changes

**Branch:** `exp/alpha-throughput-diag` forking from `9517a3b` (019b).
**Worktree:** `worktrees/alpha-throughput-diag/`
**Commit:** `85d502a` — 020 instrumentation + cuda.synchronize() fix before elapsed_time.
- `9bb1b01` — initial instrumentation (78 LOC)
- `85d502a` — fix: add `torch.cuda.synchronize()` before `elapsed_time()` reads (elapsed_time's implicit sync was empirically insufficient; pods were getting -1.0 on all per-phase μs columns)

**Diff scope (all behind env flags; no-op when flags unset):**

1. **`TTT_ENABLED=0`** (already exists as env gate): skips TTT phase in `train_and_eval`. EMA + GPTQ still run (cheap), TTT does not (~500s saved). No code change.
2. **`THROUGHPUT_DIAG=1`** env: enables per-step logging + phase timing + dataloader timing. ~60 LOC.
3. **`ENABLE_LOOPING_AT=0.17`**: early activation (~step 1040). Already an existing env gate, no code change.

Per-step instrumentation (inside training loop, gated on `THROUGHPUT_DIAG`):
- Per-step wall-time: `timestamp_iso, step, step_time_ms`
- Allocator stats snapshot: `active_bytes, reserved_bytes, num_alloc_retries, num_device_alloc`
- Forward/backward/optimizer μs (CUDA event pairs wrapped around each phase in `step_fn`)
- Dataloader batch fetch μs (time from `train_loader.next_batch(...)` call to return)
- NVTX ranges: `forward`, `backward`, `optimizer_step`, `dataloader_next`

Separate `diag_*.csv` output files in `ARTIFACT_DIR` to avoid spamming `train.log`:
- `diag_steps.csv`: per-step walltime + phase timing + allocator stats + dataloader
- `diag_nvsmi.csv`: nvidia-smi output (written by sidecar script)

## Hardware ladder

**Skip mini (user decision, 2026-04-21):** 78 LOC of standard APIs (torch CUDA events, memory_stats, csv stdlib, NVTX) — risk surface is small. First diagnostic pod serves as de-facto smoke; if it fails before loop activation, we fix and rerun before pod 2.

Two 8×H100 JP pods, sequential (not parallel — avoids provider race conditions on pod creation). Same seed, same commit. Natural 596s wallclock cap.

## Seed plan

Seed 42 on both runs. Same seed intentionally — we want identical training trajectory so any dip-step difference is attributable to the pod, not to seed variance.

## Inputs

Standard 019b inputs:
- Data: CaseOps dataset, JP mount at `/runpod/data/...`
- Tokenizer: `fineweb_8192_bpe.model` bundled
- Hotstart: none

## Run protocol

Each pod:

```bash
cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git checkout <exp/alpha-throughput-diag commit>

mkdir -p /runpod/runs/020-alpha-throughput-diag/pod_$POD_NAME
mkdir -p /runpod/.torch_inductor_cache_020
rm -rf /runpod/.torch_inductor_cache_020/*

# Launch nvidia-smi poller in background
nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /runpod/runs/020-alpha-throughput-diag/pod_$POD_NAME/diag_nvsmi.csv &
NVSMI_PID=$!

# Launch training
NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/020-alpha-throughput-diag/pod_$POD_NAME \
TORCHINDUCTOR_CACHE_DIR=/runpod/.torch_inductor_cache_020 \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
TTT_ENABLED=0 THROUGHPUT_DIAG=1 \
ENABLE_LOOPING_AT=0.17 \
TRAIN_LOG_EVERY=100 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /runpod/runs/020-alpha-throughput-diag/pod_$POD_NAME/train.log 2>&1

kill $NVSMI_PID
```

Each pod produces: `train.log`, `diag_steps.csv`, `diag_nvsmi.csv`, `final_model.pt` (post-EMA, no TTT), `final.json`.

## Checkpoints / artifacts to emit

- `train.log` — standard training log (every 100 steps)
- `diag_steps.csv` — per-step diagnostics (~4700 rows)
- `diag_nvsmi.csv` — per-GPU per-second data (~40K rows)
- `final_model.pt` — post-EMA checkpoint (for potential reuse)
- `final.json` — summary (step count, end-of-run tok/s, pre-quant val_bpb, post-GPTQ val_bpb, no post-TTT)

## Stop-early criteria

- NaN/inf loss → halt
- Instrumentation overhead > 5% of 019b tok/s (e.g. proxy-compare first 200 steps: if Run with `THROUGHPUT_DIAG=1` runs >5% slower than 019b's first-200-step rate from its log, kill and simplify instrumentation)

## Cost estimate

| item | cost |
|---|---|
| Pod 1: 8×H100 JP × ~12 min (training + EMA + GPTQ, no TTT) | ~$4 |
| Pod 2: same | ~$4 |
| **020 total** | **~$8–10** |

Significantly cheaper than two full-pipeline runs because TTT is skipped.

## Extra artifacts

Post-run analysis script (research session, zero cost):
- Load both runs' `diag_steps.csv`, find all dip intervals (tok/s < 90% of median)
- Cross-reference: do dip steps overlap? What does `nvsmi.csv` show at dip timestamps?
- Per-phase attribution: for each dip, which phase μs spiked?

## Open questions for interview (execution)

1. **Instrumentation overhead sanity check.** The 50 LOC of timing adds ~10 Python-side microseconds per step. For a 120ms step that's <0.01% — negligible. But confirm empirically: run first 200 steps, compare tok/s to 019b's original first-200 rate (~8.0M). If >5% slower, something is wrong in the instrumentation.
2. **Pod availability.** Two 8×H100 JP pods needed. If only one available, run first one, stop, then re-provision. If the same physical node gets assigned twice (unlikely with pool rotation), note in eval.
3. **nvidia-smi poller death.** If `nvidia-smi -l 1` dies mid-run (rare but happens), the CSV truncates. Worth a simple supervisor loop: `while kill -0 $TRAIN_PID; do nvidia-smi ...; sleep 1; done`.
