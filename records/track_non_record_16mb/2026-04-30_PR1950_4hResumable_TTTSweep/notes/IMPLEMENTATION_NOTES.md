# Implementation Notes: 4h Resumable Long-Train + TTT Sweep

## Resumable Checkpoints

### Design Decisions

1. **Rank-local saves**: Each rank writes its own `.pt` file. This avoids gather/scatter
   overhead and is more robust to NCCL failures during saves.

2. **Manifest-driven**: Rank 0 writes `resume_manifest.json` which lists all rank files,
   step, hparam fingerprint. Resume starts by reading this manifest.

3. **Atomic saves**: Write to `*.tmp` then `os.replace()` for atomicity. No partial files.

4. **Compatibility validation**: On resume, checks world_size + 7 architecture params
   (num_layers, model_dim, num_heads, num_kv_heads, vocab_size, mlp_mult, num_loops).
   Warns on tokenizer/data path changes.

5. **Keep-last cleanup**: Rank 0 removes old checkpoints beyond RESUME_KEEP_LAST (default 3).

### DocumentPackingLoader State

The loader has async prefetch (next shard + next batch). On state_dict():
- Drains pending `_next_batch` future (discards result; the prefetch used stale cursor)
- Records `current_shard_idx` and `cursor` position
- On load: cancels pending futures, reloads shard at saved index, restores cursor

This means resumed training starts from the correct byte position in the data stream.
The next batch after resume will be the same as would have been served without interruption.

### Muon Shard Mom State

Muon's `_bank_meta` contains rank-local `shard_mom` buffers that accumulate momentum
from reduce-scattered gradients. These MUST be saved per-rank to resume correctly,
as they depend on the rank's gradient shard.

## TTT/LoRA Sweep

### Isolation Strategy

Each variant runs as a separate `torchrun` invocation with:
- `TTT_EVAL_ONLY=1` — skips training/GPTQ entirely
- `LOAD_QUANTIZED_MODEL_PATH=<artifact>` — points at the shared final artifact
- Unique `ARTIFACT_DIR` and `TTT_EVAL_OUTPUT_JSON` per variant
- Process isolation prevents state contamination between variants

### Variant Design Rationale

- **v0**: Exact PR #1979 control for baseline comparison
- **v1–v2**: Tests rank/alpha scaling independently from LR
- **v3**: Tests local batch/chunk size (more tokens per TTT step)
- **v4**: Tests global TTT intensity (epochs, chunk size, warmup)
- **v5**: Tests prefix coverage (how much data TTT adapts on)
- **v6**: Tests phase granularity (diminishing returns expected)

### Fixed Parameters

These are held constant to isolate the variable effects:
- TTT_WEIGHT_DECAY=1.0 (strong regularization, established in PR #1767)
- TTT_BETA1=0 (no momentum for TTT optimizer)
- TTT_BETA2=0.999
- TTT_OPTIMIZER=adam
- TTT_WARM_START_A=1 (alpha-scaling warm-start)
- GLOBAL_TTT_LR=0.001

## Safety Analysis

### No Impact on Record-Track Behavior

All new code is gated behind:
- `RESUME_ENABLED=1` (default: off)
- `NON_RECORD_LONGTRAIN=1` (already required for longtrain)
- `TTT_EVAL_ONLY=1` (skips training entirely)
- `LOAD_QUANTIZED_MODEL_PATH` (optional override)
- `TTT_EVAL_OUTPUT_JSON` (optional output path)

When none of these are set, behavior is identical to PR #1950.

### Artifact Size Unchanged

LoRA/TTT parameters exist only in GPU RAM during evaluation.
They are never serialized to the artifact. The 16 MB cap is unchanged.

### No Validation Data Leakage

Score-first TTT (established in PR #461) only trains on tokens that have
already been scored/graded. No future validation tokens are accessed.
