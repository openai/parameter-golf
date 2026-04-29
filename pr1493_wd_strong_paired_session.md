# PR1493 wd_strong + paired-head Muon — Session, 2026-04-29

This document records the run that stacks the **stronger WD schedule** on top of `wd_paired` (i.e., `WD_SCHED_LOW_FACTOR=0.50`, `WD_SCHED_HIGH_FACTOR=1.75` in addition to `WD_SCHEDULE_ENABLED=1` and `PAIRED_HEAD_MUON_ENABLED=1`). Run executed locally on 8×H100, seed 42, on commit `49d1068b…` (HEAD at launch time, code blob `1e4f7b4…`).

This document also records a **filesystem rollback event** that hit `.git/` and a subset of working-tree files at 2026-04-29 10:45:24 UTC, *after* the wd_strong_paired training run had already completed. The recovery procedure is documented at the end.

## Why this run

After `wd_paired` (1.07974) closed −0.00055 BPB on top of `wd` alone but still left us 0.0004–0.0007 BPB short of the leaderboard acceptance margin, the natural question was: does a **more aggressive WD ramp** help when stacked with paired-head Muon? Specifically the `pr1493_stacking_plan.md` candidate:

```text
WD_SCHEDULE_ENABLED=1 PAIRED_HEAD_MUON_ENABLED=1 \
WD_SCHED_LOW_FACTOR=0.50 \
WD_SCHED_HIGH_FACTOR=1.75
```

The default WD schedule:
- 0.0 → 0.40: hold at 1× (default `WD_SCHED_HOLD_FRAC=0.40`).
- 0.40 → 0.85: ramp from 1× to `WD_SCHED_LOW_FACTOR` (default 0.65).
- 0.85 → 1.0: ramp from low → `WD_SCHED_HIGH_FACTOR` (default 1.5).

Strong variant: `low=0.50` (more aggressive ramp-down, more capacity in mid-training), `high=1.75` (stronger end-of-training contraction).

## What we ran

```bash
RUN_ID=pr1493_wd_strong_paired_s42 \
SEED=42 \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5 \
WD_SCHEDULE_ENABLED=1 PAIRED_HEAD_MUON_ENABLED=1 \
WD_SCHED_LOW_FACTOR=0.50 WD_SCHED_HIGH_FACTOR=1.75 \
./safe_launch.sh torchrun --standalone --nproc_per_node=8 train_pr1493.py
```

Local 8×H100, torch 2.9.1+cu128, flash-attn-3 cu128_torch291. Same dataset as `wd_paired`: `kevclark/parameter-golf` SP8192, 128 train shards (24 GB), 1 val shard.

## Pre-launch verification

`safe_launch.sh` self-tested at 2026-04-29 10:24:21 with:

```text
[safe_launch] OK ts=20260429T102421Z
  head=49d1068b880e83c42e70f2664ef27030c42b9267
  md5=968e5ab744772b096a8f9b521656019d
  blob=1e4f7b4391f9a82b0ca7f735bbbb0db6eea8e8ad
  symbols=6
```

Same train_pr1493.py blob as `wd_paired` — content unchanged, only env vars differ between the two runs.

## Live evidence (smoking-gun signals)

From `logs/pr1493_wd_strong_paired_s42.txt` and `/proc/<pid>/environ` of all 8 worker PIDs at runtime:

```text
paired_head_muon_enabled: True
wd_schedule_enabled: True
wd_sched_low_factor: 0.5     # ← strong (default would be 0.65)
wd_sched_high_factor: 1.75   # ← strong (default would be 1.5)
wd_sched_hold_frac: 0.4
wd_sched_ramp_frac: 0.85
train_shards: 128
muon:paired-head NS enabled for q/k matrices tagged=22
```

Verified mid-run via `tr '\0' '\n' < /proc/<pid>/environ`: every rank carried `PAIRED_HEAD_MUON_ENABLED=1`, `WD_SCHEDULE_ENABLED=1`, `WD_SCHED_LOW_FACTOR=0.50`, `WD_SCHED_HIGH_FACTOR=1.75`, `RUN_ID=pr1493_wd_strong_paired_s42`, `SEED=42`, `TTT_LR=0.007`, `TTT_EPOCHS=5`, `QK_GAIN_INIT=5.25`. All 8 GPUs ran at 98–100% util, 40 GB allocated each — same throughput shape as `wd_paired`.

Train cap hit at step **4602/20000** (wallclock 588124 ms; comparable to `wd_paired`'s 4596 — within 0.15 % step drift, as expected from matmul nondeterminism).

## Final results

```text
pre   = 1.08573153   (post-EMA, pre-GPTQ)
q     = 1.09873717   (post-GPTQ, no sliding, no TTT)
q_sw  = 1.08194261   (post-GPTQ, sliding window)
q_ttt = 1.07970528   (primary metric)
size  = 16,030,578 B (30,578 over the 16 MB cap; +654 vs wd_paired)
stop  = 4602/20000   (wallclock cap = 588124 ms)
```

### Comparison vs all comparators

| run | pre | q | q_sw | q_ttt | Δq_ttt vs raw |
|---|---|---|---|---|---|
| PR1493 raw, seed 42 | 1.08757 | 1.10014 | 1.08329 | 1.08103 | — |
| tuned TTT (lr=.007, ep=5) | 1.08757 | 1.10014 | 1.08329 | 1.08079 | −0.00024 |
| `wd` alone | 1.08650 | 1.09951 | 1.08269 | 1.08029 | −0.00074 |
| **wd_paired** (default WD) | 1.08610 | 1.09891 | 1.08209 | **1.07974** | −0.00129 |
| **wd_strong_paired** (this run) | **1.08573** | **1.09874** | **1.08194** | **1.07971** | **−0.00132** |

### Per-stage delta vs `wd_paired`

| stage | wd_paired | wd_strong_paired | Δ | above 0.0002 noise floor? |
|---|---|---|---|---|
| pre | 1.08610 | 1.08573 | −0.00037 | yes (small real win) |
| q | 1.09891 | 1.09874 | −0.00017 | no (at noise floor) |
| q_sw | 1.08209 | 1.08194 | −0.00015 | no (at noise floor) |
| **q_ttt** | **1.07974** | **1.07971** | **−0.00003** | **no — essentially zero** |

Stronger WD adds a small, real **pre-quant** gain (−0.00037) but that gain is **completely absorbed by the time TTT runs**. Net effect on the primary metric (q_ttt) is **0.00003**, well below the 0.0002 noise threshold. On a single seed this looks indistinguishable from `wd_paired`.

### Quant gap (`q − pre`)

- baseline: 0.01257
- wd alone: 0.01301
- wd_paired: 0.01281
- wd_strong_paired: 0.01301

Stronger WD slightly **widens** the quant gap vs `wd_paired` (back to roughly the level of `wd` alone), even though pre-quant got better. Plausible reading: the stronger end-of-training spike in WD compresses weights into a regime that GPTQ doesn't quantize as cleanly as the milder schedule.

### Verdict

**Not worth stacking.** `wd_strong_paired` does not beat `wd_paired` on q_ttt at single-seed resolution, and the pre-quant edge is not preserved through GPTQ + TTT. Default WD + paired-head Muon remains the right baseline for further stacking experiments. Move on to `wd_paired_iha` next.

## The 10:45 FS rollback event

At 2026-04-29 10:45:24 UTC — **after** the wd_strong_paired training run completed at 10:43 — the host filesystem rolled `.git/` back to a snapshot taken at the start of the original prior session (yesterday's clone + this morning's `pull --ff-only`).

### Detection

After the run finished, post-run integrity check showed:

```text
HEAD: 74dc7028a06a0f52e2ce23a925ef24404e93ca1b   ← rolled back
expected: 49d1068b880e83c42e70f2664ef27030c42b9267
```

Triggered alarm. Diagnostic forensics:

```text
.git/HEAD       Birth: 2026-04-29 10:45:24 UTC      ← reborn after run finished
.git/refs/heads/shikhar  Modify: 2026-04-29 09:03:49 UTC, Birth: 10:45:24 UTC
git reflog: only entries from yesterday's clone + 09:03:49 pull --ff-only
git rev-parse origin/shikhar: 74dc702 (stale)
git ls-remote origin shikhar: 49d1068b... (truth on GitHub)
git cat-file -p 49d1068: still present as loose object
train_pr1493.py blob: 1e4f7b4... (unchanged throughout)
```

So:
- `.git/` directory was **reborn** at 10:45:24 from a snapshot taken before commit `49d1068`.
- The loose object database survived (commit `49d1068` is recoverable locally).
- Local refs (`refs/heads/shikhar`, `refs/remotes/origin/shikhar`) were both rolled back.
- GitHub remote was **not affected** — `49d1068` is still there.
- Workspace files (training file, log files, the just-finished `pr1493_wd_strong_paired_s42.txt`) were **not in the snapshot scope** — they were preserved.
- BUT: `logs/pr1493_wd_paired_s42.txt` (the **real** wd_paired log committed in `49d1068`) was **overwritten** by the **prior session's bogus version** (mtime back to 2026-04-29 07:44, content showed `train_shards: 127`, `wd_sched_high_factor: 1.5` defaults, q_ttt 1.08008 — exactly the bogus number we'd already proven to be a no-op flag).
- 20 ghost log files from the prior session reappeared in `logs/`.

### A separate stale-monitor scare

A stale `tail -F` monitor (id `bsy9pzhuv`, started during the wd_paired run) re-emitted a flood of "new" lines from `logs/pr1493_wd_paired_s42.txt` at the moment the file was swapped out under it. It looked like a new training run had launched with default WD values, but `ps -ef`, `nvidia-smi`, and shard count all confirmed nothing was actually running. The monitor was just re-reading the rolled-back file.

### Recovery

```bash
# 1. belt-and-suspenders backups (the wd_strong_paired evidence is untracked,
#    but I wanted a /tmp copy in case anything else went sideways)
cp -p logs/pr1493_wd_strong_paired_s42.{txt,stdout} /tmp/parameter-golf-backup/
cp -p safe_launch.sh /tmp/parameter-golf-backup/safe_launch.sh.bumped

# 2. fetch the real remote state
git fetch origin
# → "74dc702..49d1068  shikhar -> origin/shikhar"

# 3. align local to remote — restores the REAL wd_paired log from blob
git reset --hard origin/shikhar
# → "HEAD is now at 49d1068"

# 4. (safe_launch.sh on disk now is the user's preferred version, not my
#    bumped-pin version — kept as is.)
```

Post-recovery verification:

```text
HEAD: 49d1068b880e83c42e70f2664ef27030c42b9267
real wd_paired log: q_ttt 1.07974373, tagged=22, train_shards: 128  ✓ (restored)
wd_strong_paired log: q_ttt 1.07970528, tagged=22, low=0.5, high=1.75  ✓ (untouched)
md5 of wd_strong_paired log: byte-identical to /tmp backup  ✓
```

No results lost. wd_paired evidence recovered from git blob; wd_strong_paired evidence preserved as-is (it was untracked at the time of the rollback, and `git reset --hard` does not touch untracked files).

## Learnings

1. **The host's filesystem is snapshot-based and `.git/` is in scope of those snapshots.** A rollback can revert your local refs (and *some* working-tree files) while leaving GitHub and most other files alone. Always verify:
   - `git rev-parse HEAD` matches what you committed
   - `git rev-parse origin/shikhar` matches `git ls-remote origin shikhar`
   - File mtimes for "ghost" files that shouldn't exist are a red flag
2. **Stale `tail -F` monitors will emit "new" content when a file is swapped under them.** This can look like a new training run starting. Always cross-check with `ps -ef`, `nvidia-smi`, and shard count before trusting the flood.
3. **Push commits to GitHub immediately.** The remote is the single durable source of truth on this host. Anything not on GitHub can vanish on the next snapshot.
4. **Prefer the smoking-gun signal `tagged=22` and the explicit `wd_sched_*` values in the hyperparameter dump over inferring from env vars.** They survive monitor noise and prove what the *running model* actually saw.
5. **wd_strong on top of wd_paired is a no-op at q_ttt.** Pre-quant gain doesn't transfer through GPTQ + TTT. The plan-doc estimate that "stronger WD only matters if default WD still looks good" was right — default WD looked good, but stronger WD didn't extend the gain.
6. **Wallclock-cap step is reproducible to ~10 steps run-to-run** at this seed/hardware. wd_paired stopped at 4596; wd_strong_paired at 4602; the bogus-flag prior session at 4596 too. If a run stops at a wildly different step, that's a signal something is off (e.g., a missing shard caused fewer effective tokens/step).

## Files committed in this session

- `pr1493_wd_strong_paired_session.md` — this document.
- `pr1493_priority_results.md` — appended row 7 (wd_strong_paired).
- `logs/pr1493_wd_strong_paired_s42.txt` — real run log.
- `logs/pr1493_wd_strong_paired_s42.stdout` — torchrun stdout for the run.
- `safe_launch.sh` — user-modified guard wrapper (symbol-presence check with auto-recovery from local backup or `origin/shikhar`).

The 20 ghost prior-session log files restored by the FS rollback are **not** committed — they're forensics, not part of this session's work. `train_pr1493.py.74dc702` and `train_pr1493.py.bak.74dc702` (both byte-identical to HEAD's `train_pr1493.py`) are also not committed; the safe_launch.sh script will recover from `origin/shikhar` if either is missing on a future fresh clone.

## Next stack candidate

`wd_paired_iha` — fixed IHA fold path is now end-to-end testable since brotli is installed and the prior IHA `KeyError` was a separate harness bug fixed in commit `74dc702`. If IHA closes another 0.0004–0.0008 BPB on top of `wd_paired`, we'd cross the leaderboard acceptance bar.
