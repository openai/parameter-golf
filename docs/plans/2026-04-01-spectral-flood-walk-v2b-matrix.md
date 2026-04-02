# Spectral Flood Walk v2b Initial Test Matrix

## Purpose

This matrix is designed to answer `v2b` questions in the right order while keeping the total run count small enough to execute quickly on exploratory hardware.

The sequence is intentional:

1. verify that persistent hidden memory has any online signal at all
2. verify that read-gating reduces the early-chunk tax
3. spend extra eval FLOPs on memory maintenance and see whether it buys better late-stream performance
4. only then probe larger capacity or stronger reads

## Design Rule

Do **not** run a large Cartesian sweep first.

The core question is not:

> what is the best random combination of memory knobs?

The core question is:

> which mechanism is actually helping?

So the first matrix is staged rather than exhaustive.

## Stages

### Stage 0 — Sanity

- `baseline_memread1_nomaint`

This establishes the raw online hidden-memory line with immediate reads and zero maintenance compute.

### Stage 1 — Read Gate

- `gate2_nomaint`
- `gate4_nomaint`

These runs test whether delaying reads helps by making the first chunk more sacrificial and the later stream cleaner.

Interpretation:

- if `gate2` beats immediate reads, the early-chunk tax is real
- if `gate4` also helps, more aggressive warm-up may be justified
- if both gates hurt, the system needs faster initial usefulness rather than more protection

### Stage 2 — Maintenance FLOPs

- `gate2_maint1_slots64`
- `gate2_maint2_slots64`
- `gate2_maint2_slots128`
- `gate2_maint2_slots128_hits`
- `gate2_maint2_slots128_nograd`

These runs spend progressively more eval compute on touched-slot refinement.

Interpretation:

- if higher maintenance improves `delta_online`, extra eval FLOPs are being converted into useful model growth
- if `hits` beats `counts`, the read path is identifying the right slots to refine
- if `nograd` collapses, the EMA signal is doing real work rather than maintenance being generic smoothing

### Stage 3 — Capacity / Read Strength

- `gate2_maint2_slots128_table96k`
- `gate2_maint2_slots128_readscale125`

These are later because there is little value in widening capacity or amplifying reads before we know the mechanism is healthy.

## Commands

Generate the curated matrix table:

```bash
python3 tools/generate_v2b_matrix.py
```

Generate shell commands:

```bash
python3 tools/generate_v2b_matrix.py --format shell --python-bin python3.11 --output-dir runs
```

Summarize finished runs:

```bash
python3 tools/summarize_v2b_runs.py runs/*
```

## Metrics To Watch

Primary:

- `eval_delta_online_bpb`
- `eval_online_persistent_hidden.val_bpb`

Mechanism health:

- `active_slots_mean`
- `readable_slots_mean`
- `delta_norm_mean`
- `persistent_memory.readable_fraction`

Compute pressure:

- `memory_lookup_flops_estimate`
- `memory_update_flops_estimate`
- `memory_maintenance_flops_estimate`
- `memory_total_flops_estimate`

## What We Want To Learn

The strongest outcome is:

- gated persistent memory beats ungated memory
- added maintenance FLOPs further improve the online delta
- the compute-heavy runs show meaningfully higher `memory_total_flops_estimate` without flatlining quality

That would be the first evidence that the coprocessor framing is working:

> more runtime memory work is making the model better as it reads.
