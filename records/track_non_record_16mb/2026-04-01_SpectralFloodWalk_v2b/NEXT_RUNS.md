# Next Runs

This is the shortest useful `v2b` pod plan.

The goal is not "run everything."

The goal is:

1. confirm the wrapper stack works on the pod
2. test whether read-gating helps
3. only then spend more eval FLOPs on maintenance

The latest pod sweep says:

- `gate4_nomaint` is the best gate-only branch so far
- the first maintenance-heavy run raised FLOPs a lot but hurt `delta_online`
- replay maintenance now defaults to pure replay sharpening with a per-slot replay depth of `2`

## Default Start

From `/workspace/parameter-golf/records/track_non_record_16mb/2026-04-01_SpectralFloodWalk_v2b`:

```bash
./runpod_preflight.sh
./runpod_queue_parallel_core.sh
```

That gives the first useful answer with less babysitting:

- pod health
- immediate-read baseline on one GPU
- `gate2` on one GPU
- `gate4` on one GPU

## First Decision

After the core queue finishes, summarize:

```bash
python3 ../../../tools/summarize_v2b_runs.py runs/*
```

Check these fields:

- `delta_online`
- `online_bpb`
- `total_gflop`
- `readable_slots`
- `readable_fraction`

Interpretation:

- if `gate2_nomaint` beats `baseline_memread1_nomaint`, the early-chunk tax is probably real
- if `gate2_nomaint` is flat or worse but `gate4_nomaint` improves, the system wants a longer warm-up
- if both gated runs are flat or worse, pause the compute-heavy maintenance branch and revisit the memory update/read design first

## Current Best Branch

Right now the best follow-up is not more maintenance by default. It is seed confirmation on `gate4`:

```bash
./runpod_queue_parallel_gate4_seeds.sh
python3 ../../../tools/summarize_v2b_runs.py runs/*
```

If you want the sequential version instead:

```bash
SFW_PROFILE_SCRIPT=runpod_gate4.sh ./runpod_three_seeds.sh
python3 ../../../tools/summarize_v2b_runs.py runs/*
```

## If Gate4 Still Wins

Only then try the compute-heavier follow-up that matches the winning gate:

```bash
./runpod_flop_push_gate4.sh
python3 ../../../tools/summarize_v2b_runs.py runs/*
```

This is the first "light it up" profile on top of the current best gate:

- `memory_min_read_count=4`
- `maintenance_passes=2`
- `maintenance_mode=replay`
- `maintenance_metric=loss`
- `maintenance_use_grad=false`
- `maintenance_replay_depth=2`
- `maintenance_max_slots=128`

What we want:

- `total_gflop` rises materially
- `delta_online` improves or at least holds while the memory stats stay healthy

What we do **not** want:

- much higher `total_gflop`
- flat or worse `delta_online`
- near-zero `readable_slots` or `readable_fraction`

That combination means we are burning eval compute without growing a better effective model.

## If Flop Push Looks Good

Only if `runpod_flop_push_gate4.sh` is promising, run:

```bash
./runpod_hits_gate4.sh
python3 ../../../tools/summarize_v2b_runs.py runs/*
```

This tests whether maintenance compute should prioritize:

- hard cases via `loss`
- high-read slots via `hits`

If `hits` wins, that is a good sign that the model is learning where the useful memory actually lives.

If the loss-prioritized replay branch is close but noisy, the first follow-up knobs worth sweeping are:

- `maintenance_replay_depth=3`
- `maintenance_use_grad=true` with `maintenance_grad_mix=0.25`

## Short Version

Use this exact order unless there is a strong reason not to:

1. `./runpod_preflight.sh`
2. `./runpod_queue_parallel_core.sh`
3. summarize
4. `./runpod_queue_parallel_gate4_seeds.sh`
5. summarize
6. only if still justified: `./runpod_flop_push_gate4.sh`
7. only if that helps: `./runpod_hits_gate4.sh`

## Success Signal

The outcome we are hoping to see is:

- read-gating improves the online delta
- maintenance-heavy runs increase `total_gflop`
- the higher-compute run also improves `delta_online`

That would be the first clean evidence that more runtime memory work is not just using hardware, but actually making the model better as it reads.
