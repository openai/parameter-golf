# Spec 020 post-run analysis plan

**Purpose:** when the two diagnostic pods complete, this is the playbook for extracting and interpreting the outputs. Pre-written so no thinking needed at eval time.

**Inputs:** for each pod, `runs/020-alpha-throughput-diag/pod_<name>/`:
- `train.log` — standard training log (every 100 steps)
- `diag_steps.csv` — per-step diagnostics, ~5000 rows (main rank 0 only)
- `diag_nvsmi.csv` — per-GPU per-second, ~4800 rows (sidecar)
- `final_model.pt`, `final.json`

Also compare against `runs/019b-recur-alpha-manual-constant-full/seed_42/train.log` (normal-activation arm).

## `diag_steps.csv` schema

| column | units | source | what it tells us |
|---|---|---|---|
| `step` | int | loop counter | which training step |
| `wallclock_s` | s (perf_counter) | `time.perf_counter()` | align with nvsmi timestamps |
| `step_time_ms` | ms | delta from prev row | actual per-step wall-time |
| `fwd_us` | μs | CUDA event | forward-pass GPU time |
| `bwd_us` | μs | CUDA event | backward-pass GPU time |
| `opt_us` | μs | CUDA event | optimizer.step (Muon + Adam) |
| `dataloader_us` | μs | perf_counter | next_batch() wall time |
| `active_bytes` | bytes | memory_stats | live allocations |
| `reserved_bytes` | bytes | memory_stats | cache-pool size |
| `num_alloc_retries` | int (cum) | memory_stats | cache-pool exhaustion events |
| `num_device_alloc` | int (cum) | memory_stats | cudaMalloc calls |
| `num_device_free` | int (cum) | memory_stats | cudaFree calls |

## `diag_nvsmi.csv` schema (per-GPU per-second)

| column | what it tells us |
|---|---|
| `timestamp` | align with diag_steps |
| `index` | GPU 0-7 |
| `temperature.gpu` (°C) | thermal throttle at ~82°C sustained |
| `clocks.current.sm` (MHz) | SM clock; drop below ~1980 = throttle |
| `power.draw` (W) | power-cap hit shows as pinning to ~700W |
| `utilization.gpu` (%) | starved vs compute-saturated |
| `memory.used` (MiB) | cross-check allocator stats |

## Six findings to extract

### Finding 1 — Is a dip 1 slow step or many?
```python
import pandas as pd
s = pd.read_csv("pod1/diag_steps.csv")
slow = s.nlargest(30, "step_time_ms")
print(slow[["step","step_time_ms"]])
```
- Contiguous clusters of 3-5 slow steps → multi-step stall
- Scattered single outliers → single-step events

### Finding 2 — Which phase slows during a dip?
```python
median = s.step_time_ms.median()
dips = s[s.step_time_ms > 1.15 * median]
baseline = s[s.step_time_ms < 1.05 * median]
for col in ["fwd_us","bwd_us","opt_us","dataloader_us"]:
    print(f"{col}: baseline {baseline[col].median():.0f}μs, dips {dips[col].median():.0f}μs, "
          f"ratio {dips[col].median()/baseline[col].median():.2f}x")
```
Which phase's ratio blows up > 1.3× is the culprit:
- **fwd blows up** → compute slow → constant-folded kernel cold path, CUDA graph recapture
- **bwd blows up** → gradient / AllReduce collectives
- **opt blows up** → Muon backend / param rotation / optimizer state
- **dataloader blows up** → I/O / data pipeline stall (unlikely given HBM bandwidth, but rules it out)

### Finding 3 — Allocator correlation
```python
s["retry_delta"] = s.num_alloc_retries.diff()
dip_retries = s[(s.step_time_ms > 1.15*median) & (s.retry_delta > 0)]
```
- If dips coincide with retry deltas > 0 → fragmentation / allocator reorg is cause
- If retry delta flat throughout → allocator is fine, look elsewhere

### Finding 4 — Thermal throttling check
```python
n = pd.read_csv("pod1/diag_nvsmi.csv")
import matplotlib.pyplot as plt
for gpu_idx in range(8):
    g = n[n["index"] == gpu_idx].sort_values("timestamp")
    plt.plot(g.timestamp, g["clocks.current.sm"], label=f"gpu{gpu_idx}")
plt.legend(); plt.ylabel("SM clock MHz"); plt.savefig("sm_clocks.png")
```
- SM clock drops during dip intervals → thermal or power-cap. Check temp column.
- Flat SM clock across dip times → not thermal.

### Finding 5 — Reproducibility (pod 1 vs pod 2)
```python
p1 = pd.read_csv("pod1/diag_steps.csv")
p2 = pd.read_csv("pod2/diag_steps.csv")
m1 = p1.step_time_ms.median(); m2 = p2.step_time_ms.median()
dips1 = set(p1[p1.step_time_ms > 1.15*m1].step)
dips2 = set(p2[p2.step_time_ms > 1.15*m2].step)
overlap = len(dips1 & dips2) / max(len(dips1), len(dips2), 1)
print(f"dip-step overlap: {overlap:.2%}")
```
- `>50%` overlap → **deterministic/internal** (LR schedule, optimizer state, unlogged periodic event). Fix in software.
- `<20%` overlap → **external/contention** (NCCL latency, noisy neighbor). Fix at infra.
- Middle → both contribute.

### Finding 6 — Activation-offset test (early-activation payoff)
Compare pod 1 (activation at ~step 1040) vs 019b original (activation at step 2143):
```python
early = pd.read_csv("pod1/diag_steps.csv")
# 019b original logs every 100 steps, so derive "slow intervals" from its train.log
# Known dip step indices in 019b (from idea file): 2900, 3000, 3100, 3500, 3600, 4100, 4200
orig_dips_abs = {2900, 3000, 3100, 3500, 3600, 4100, 4200}
orig_activation = 2143
early_activation = 1040  # approx; check pod1 train.log for exact

# Hypothesis A: same absolute step indices
absolute_match = orig_dips_abs & set(early[early.step_time_ms > 1.15*early.step_time_ms.median()].step)
# Hypothesis B: same offset from activation
orig_offsets = {d - orig_activation for d in orig_dips_abs}
expected_early_dips = {early_activation + off for off in orig_offsets}
offset_match = expected_early_dips & set(early[early.step_time_ms > 1.15*early.step_time_ms.median()].step)

print(f"absolute-step match: {len(absolute_match)}")
print(f"offset-from-activation match: {len(offset_match)}")
```
- More absolute-step matches → step-indexed cause (absolute step count drives dips)
- More offset matches → time/step-since-activation drives dips (kernel warmup, thermal steady-state)

## Cross-CSV alignment

Both CSVs share `wallclock_s` (perf_counter in steps, absolute timestamp in nvsmi). Merge with tolerance:
```python
# Anchor: the first step's perf_counter == some absolute wall time.
# Log the absolute time at train start to final.json, then subtract.
# Simpler: match on adjacent ranges — for each dip step, grab nvsmi rows within ±2s of its wallclock_s.
```

## Quick summary table we want to produce at eval time

| axis | answer |
|---|---|
| dips are | 1-step single / multi-step cluster |
| slow phase | fwd / bwd / opt / dataloader |
| allocator | implicated / flat |
| SM clock | throttling / stable |
| temp | throttle zone / safe |
| reproducibility | deterministic / contention |
| step-indexed vs offset | absolute / offset |
| → implied mechanism | (derived from above) |

## Expected outcomes → next spec

- **fwd slow + deterministic + step-indexed** → Inductor recompile or LR-schedule-triggered event. Fix: pin schedule or bypass recompile. **Targeted fix spec.**
- **bwd slow + contention** → NCCL latency. Fix: launch-latency tolerant kernels (e.g. tensor-α since empirically less dip-prone). **Switch 019b recipe back toward 017.**
- **opt slow + deterministic** → Muon backend rotations misaligned with constant-folded kernel boundaries. Fix: relax constant folding on α (go back to tensor α). **Kill constant-α path.**
- **thermal** → pod-level issue; accept, select better pods, or submit on faster regions.
- **inconclusive** → one more rerun with a different diagnostic axis (e.g. activation at frac=0.5 for a third offset point).
