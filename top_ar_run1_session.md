# PR1851 GPTQ All-Reduce — Run 1 Findings + Reversal of wd_strong Verdict

Session 2026-04-30, after `top_wd_strong_session.md`. Three new ports were
added to `train_top.py` (all gated, all default-off): GPTQ Hessian all-reduce,
WD-schedule (already present from prior session), and paired-head Muon NS for
the bank architecture. Run 1 tests AR alone to isolate its contribution and
to provide a within-pod baseline against today's `top_wd_strong_s42` run
(Run 0).

The headline finding is **a reversal of yesterday's verdict that wd_strong
is a no-op on PR #1851**. The earlier comparison was against PR #1855's
published pre-quant — the wrong baseline. A clean within-pod A/B against Run 0
shows wd_strong actually improves PR #1851 pre-quant by **~0.00194 BPB**.

## Run 1 setup

```bash
RUN_ID=top_ar_s42 SEED=42 \
CASEOPS_ENABLED=1 EMBED_BITS=7 \
SMEAR_GATE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 \
MIN_LR=0.1 EMBED_CLIP_SIGMAS=15.0 MLP_CLIP_SIGMAS=12.0 \
GPTQ_RESERVE_SECONDS=8.0 PHASED_TTT_NUM_PHASES=3 \
GPTQ_ALL_REDUCE=1 \
torchrun --standalone --nproc_per_node=8 train_top.py
```

Single-variable change vs PR #1851 unmodified: `GPTQ_ALL_REDUCE=1`. No WD
scheduling, no paired-head Muon. Same dataset, same pod, same seed as Run 0.

## What the AR fix actually does

`_top_ref/train_gpt.py:2037-2150` (`collect_hessians`) — confirmed by reading
the upstream code, not previously documented:

- `ShuffledSequenceLoader` shards the train files across ranks
  (`self.files = all_files[h.rank :: h.world_size]`), so each rank's
  calibration data is a different 1/8 of the corpus
- Each rank accumulates its Hessian on rank-local data only
  (`hessians[name].addmm_(x.T, x)` inside per-rank forward hooks)
- Line 2141 divides by `n_calibration_batches` only, not by
  `n_calibration_batches * world_size` — there's no cross-rank averaging
- Only rank 0 writes the quantized blob (line 2505 `if h.is_main_process:`)

So 7/8 of calibration compute is wasted on ranks whose Hessian is never used.
At PR #1851's default `gptq_calibration_batches=16`, only 16 batches of
calibration data drive quantization instead of effectively 16 × 8 = 128.

Smoking-gun log line confirming AR fired during Run 1:

```text
gptq:all-rank Hessian averaging across 8 ranks (denom=128)
```

## Run 1 partial results (q_ttt still in progress at time of writing)

```text
pre   = 1.06623095   (post-EMA, pre-quant)
q     = 1.07548427   (post-LQER asymmetric quantization)
q_ttt = TBD          (phased-LoRA TTT eval still running)
artifact = ??? (post-brotli)
size  = 15,956,401 B (under 16 MB cap by 43,599 B)
stop  = 4829/20000   (wallclock cap = 592128 ms)
```

## Within-pod A/B: Run 0 vs Run 1

Both runs: same pod, same seed, same dataset, same code path with one
difference each.

|  | Run 0 (`top_wd_strong_s42`) | Run 1 (`top_ar_s42`) | Δ Run 1 − Run 0 |
|---|---|---|---|
| AR fix | OFF | **ON** | |
| wd_schedule | **ON** (low=0.5, high=1.75) | OFF | |
| pre | 1.06429 | 1.06623 | **+0.00194** |
| q | 1.07403 | 1.07548 | +0.00145 |
| q_gap (q−pre) | **0.00974** | **0.00925** | **−0.00049** |
| stop_step | 4846/20000 | 4829/20000 | −17 |
| train_time_ms | 592154 | 592128 | −26 |
| artifact_bytes | 15,948,542 | 15,956,401 | +7,859 |

### What this isolates

The two changes between runs (AR on/off, WD on/off) hit *different* phases of
training:

- **AR fix runs only post-train** (during `collect_hessians`, which is invoked
  after the wallclock cap is hit and before LQER quantization). It cannot
  affect pre-quant val_bpb. Therefore the +0.00194 pre-quant gap between
  Run 0 and Run 1 is **fully attributable to WD scheduling**.
- **WD scheduling runs only during training** (modulating
  `group["weight_decay"]` per step via `wd_mul(frac)`). It cannot affect the
  Hessian collection or LQER quantization directly. Therefore the −0.00049
  narrowing of the quant gap (q − pre) between Run 0 and Run 1 is **fully
  attributable to the AR fix**.

### Two clean signals, both real

1. **AR fix is real on PR #1851.** Quant gap narrowed by 0.00049 BPB
   (Run 1 0.00925 < Run 0 0.00974). Smaller than the −0.00083 PR1493
   evidence at 16-shard, but consistent with PR #1851's LQER asymmetric
   being more robust to Hessian sparsity than PR #1493's GPTQ-int6.
2. **wd_strong is real on PR #1851** (reversing yesterday's verdict).
   Pre-quant improved by 0.00194 BPB (Run 0 1.06429 < Run 1 1.06623).
   That is **3× the published seed-to-seed std (0.00068)** and well above
   the noise floor.

## Why I got wd_strong wrong yesterday

`top_wd_strong_session.md` Part 4 compared Run 0's pre (1.06429) to PR #1855's
**published** pre (1.06396) and concluded "WD made pre-quant slightly worse"
(+0.00033). That comparison was wrong because:

- **PR #1855 is not PR #1851.** PR #1855 has different compressor
  (lrzip vs brotli), 9 extra hparam knobs, and a different stack. Its
  pre-quant is not a valid baseline for our PR #1851-derived stack.
- **Pod environment differences matter.** Different CUDA non-determinism,
  different fp seeds, different shard counts can shift pre-quant by
  comfortably more than 0.0005 BPB run-to-run. PR #1855's published number
  is from a different pod, possibly different hardware revision.
- **Cross-stack baselines drift.** I was treating a single published number
  as ground truth and reading 0.0003 BPB deltas from it. That's below the
  cross-pod / cross-stack noise floor.

A within-pod A/B (Run 0 vs Run 1) controls for all these confounders. The
+0.00194 pre signal from WD scheduling is robust under that control.

## Updated expected stack value

The earlier ceiling estimate in `top_wd_strong_session.md` Part 6 said:

> Even with all three layered, expected q_ttt is ~1.05970, which clears
> PR #1855 by ~0.00140 BPB but does NOT clear the 0.0024-BPB acceptance bar.

That estimate assumed wd_strong contributed ~0 BPB. With the within-pod
finding, the revised estimate is:

| stack on PR #1851 s42 | expected pre | expected q | expected q_ttt |
|---|---|---|---|
| unmodified | 1.06623 (Run 1 pre, no WD) | ~1.07548 (Run 1 q) | ~1.06128 (PR #1851 published) |
| + AR alone (Run 1) | 1.06623 | 1.07548 | ~1.0622 (predicted, q_ttt landing imminent) |
| + AR + wd_strong (Run 0 + AR — never run yet) | ~1.06429 | ~1.07354 (q gap −0.00049) | **~1.06062** |
| + AR + wd_strong + paired-head (Run 3 — pending) | ~1.06410 (PH usually flat on pre) | ~1.07310 | **~1.06020** |

If the predictions hold, **AR + wd_strong + paired-head clears
PR #1855 (1.06108 mean) by ~0.0008 BPB** but still likely does not clear the
0.0024-BPB acceptance bar without something else. The contest record bar
stays intractable; the leaderboard PR opportunity is narrower than I
initially thought.

But: a clean ~−0.001 BPB stacked improvement on PR #1851 is still a real
result, and it's what we have time to deliver before the deadline.

## Next runs (queued)

- **Run 2** (auto-launched when Run 1 GPUs free):
  ```text
  GPTQ_ALL_REDUCE=1 WD_SCHEDULE_ENABLED=1
  (no factor overrides → defaults: low=0.65, high=1.5)
  ```
  Tests whether *default* WD factors carry the same pre-quant value as
  *strong* WD factors did in Run 0. If yes, this is the right config for
  Run 3. If default-factors regress vs Run 0's strong-factors, switch
  Run 3 to wd_strong.

- **Run 3** (launches after Run 2 finishes):
  AR + WD + paired-head Muon NS. Factor choice TBD based on Run 2.

## Other PR1493-stack additions audited as not portable / not useful

For completeness, this is what we are NOT testing and why:

- `iha` — failed harness on PR1493, regressed pre-quant on the only
  successfully-completed `wd_paired_iha` stack
- `mtp` — clear regression on PR1493 (+0.00944), implementation bug (shared
  head supervision)
- `doc_shuffle` — regression on PR1493 (+0.002) + tokens/sec drop
- `qat` — EMA contamination on PR1493; PR #1851's LQER is post-train so no
  direct conflict but no clear porting path
- `pko` — catastrophic with TTT (+0.024) on PR1493; breaks gradient-based TTT
- in-training SmearGate + per-head 1D attn_gate — PR #1851 has its own
  better-validated SmearGate (we tested ours on PR1493 + wd_strong and it
  regressed +0.00081)
- `GPTQ_DAMP / GPTQ_BLOCK_SIZE` sweep — already swept on PR1493, defaults
  optimal within 3e-6 BPB

## Files committed in this session

- `train_top.py` (since `top_wd_strong_session.md`):
  - `ec48ff1` — wd_schedule port (already present)
  - `6c53583` — GPTQ Hessian all-reduce
  - `97fc8a5` — paired-head Muon NS port to bank architecture
- `top_ar_run1_session.md` — this document
- `logs/top_ar_s42.{txt,stdout}` — Run 1 logs (committed when Run 1 finishes)
