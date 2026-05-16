# Non-record submission: Rust/CUDA record-shaped systems stack

This is a non-record systems submission, not a leaderboard record claim.

## Summary

This PR adds and documents a Rust/CUDA Parameter Golf stack that can execute the real record-shaped 8xH100 training surface:

```text
world_size             = 8
seq_len                = 2048
global_batch_tokens     = 786432
local_batch_sequences   = 48 per rank
attention_backend       = cuDNN frontend BF16 SDPA
optimizer target        = sharded Parallel Muon
```

The best clean measured record-shaped proxy result is:

```text
timing_measured_ms_per_step = 256.787
```

This is not competitive with the current leaderboard target of roughly <=130 ms/step, and I am not claiming a full-validation BPB.

## Architecture overview

The implementation is a Rust workspace under `parameter-golf-rs/` with explicit crates for the training runtime, model execution, CUDA kernels, optimizer, quantization/export, data loading, and eval:

```text
pg-train    record-shaped runner, 8-GPU orchestration, timing/audit logs
pg-model    GPU transformer runtime, BF16 activation bridges, frontier config
pg-kernels  CUDA/cuDNN kernels, SDPA bridge, fused pointwise/output helpers
pg-optim    GPU Muon / PolarNS / sharded Parallel Muon support
pg-core     CUDA/NCCL tensor wrappers and collective APIs
pg-data     shard-backed token streams and record-shaped batch loading
pg-quant    int6/int7/LQER export and code+model byte accounting
pg-eval     legal score-first GPU LoRA/phased TTT eval path
```

The current record-shaped path uses true batched `[B=48, T=2048]` execution per H100, cuDNN frontend BF16 SDPA, prepacked BF16 Q/K/V freshness checks, BOS-safe SmearGate, chunked BF16 output CE cache, and sharded Parallel Muon scaffolding. The remaining gap is primarily the backward/communication tail, not basic record semantics.

For a longer architectural walkthrough, I added `ARCHITECTURE_BLOG.md` in the submission folder. It covers the runtime design, measurement ledger, negative results, and remaining production cuts in more detail than the PR body.

## What this contributes

I am submitting this because the Rust stack reached the real record-shaped 8xH100 workload and the measurements are now specific enough to be useful, even though it is not ready to claim a record.

The main result is the systems delta: the first valid record-shaped run took about 91 seconds per step; the best clean run is now 256.8 ms/step. The report also keeps the negative results in the open, because they changed the engineering direction:

- Tiled CE removed persistent logits but lost time to repeated output GEMMs.
- The BF16 attention backward tail was not faster until the downstream BF16 QKV-gradient path is complete.
- Compact u16 input upload did not help because H2D was already below 1 ms/step.
- Fast TF32 was slower on this workload.

## What is included

```text
records/track_non_record_16mb/2026-04-30_RustCudaSystems/
  README.md
  TECHNICAL_REPORT.md
  ARCHITECTURE_BLOG.md
  PR_BODY.md
  submission.json
  specs/frontier_1855_merged_target.toml
  scripts/exact_modal_commands.sh
  logs/
  artifacts/artifact_budget.json
```

The short submission summary is in `README.md`; the detailed engineering writeup and roadmap are in `TECHNICAL_REPORT.md`.

## Validation

Local validation passed:

```text
cargo check -q --features cuda -p pg-train -p pg-eval -p pg-data -p pg-kernels
cargo test -q -p pg-data
cargo test -q --features cuda -p pg-eval
cargo test -q --features cuda -p pg-train
python3 -m py_compile deploy/run_detached.py deploy/build_submission.py
```

## Remaining blockers

The remaining blockers for a real leaderboard submission are:

- Complete BF16 backward activation graph.
- Add bucketed backward/NCCL overlap.
- Replace chunked BF16 CE cache with production fused projection + softcapped CE/backward.
- Implement a fully GPU-resident sampler.
- Prove full legal distributed eval/TTT under 600 seconds.
- Produce final artifact/code-byte proof.
- Run 3 full validation seeds.
