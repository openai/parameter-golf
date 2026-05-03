# Rust/CUDA Record-Shaped Systems Stack

**Submission type:** Non-record systems submission  
**Author:** Cedric Haddad  
**Date:** 2026-04-30  
**Claim:** This is not a leaderboard record claim. It is a Rust/CUDA systems submission documenting a record-shaped 8xH100 training runtime, measured bottlenecks, negative results, and the remaining work required to turn it into a competitive leaderboard record.

## Summary

This submission ports the Parameter Golf training/eval stack toward a Rust/CUDA implementation and measures it on the real record-shaped workload:

```text
world_size              = 8
seq_len                 = 2048
global_batch_tokens      = 786432
local_batch_sequences    = 48 per rank
attention_backend        = cuDNN frontend BF16 SDPA
distributed_optimizer    = sharded Parallel Muon scaffold
frontier target family   = SP8192 / CaseOps / SparseAttnGate / SmearGate / LQER / phased TTT
```

The best clean H100 record-shaped proxy measurement is:

```text
timing_measured_ms_per_step = 256.787 ms
target_ms_per_step          = 130.000 ms
remaining throughput gap    = about 2.0x
```

The stack is therefore a serious systems prototype, but not a top leaderboard submission. It does not include a full validation BPB, 3-seed result, final artifact byte proof, or train/eval wallclock proof under the official leaderboard path.

## Best Measured Run

The current best clean record-shaped run is `v86_throughput_clean`.

```text
mode                         = RecordShapedProxy
backend                      = cuda-distributed
world_size                   = 8
seq_len                      = 2048
global_batch_tokens           = 786432
local_microbatches_per_step   = 48
steps_completed              = 8
timing_steps                 = 6
timing_measured_ms_per_step  = 256.787
train_loss_source            = disabled_for_record_shaped_timing
distributed_sync             = true
attention_backend            = CudnnSdpaBf16
distributed_optimizer_backend = ShardedParallelMuon
microbatch_serial_loop       = false
```

Stage-level timing from the most comparable profiled run showed the remaining bottleneck is no longer full logits or host input copies. It is the backward graph and optimizer tail:

```text
backward_ms_per_step      ~= 231-236 ms
bank_update_ms_per_step   ~= 16-23 ms depending run/profile
output_ce_ms_per_step     ~= 8 ms on the chunked BF16 cache path
h2d_ms_per_step           < 1 ms on the clean path
```

## What Works

- Real record-shaped batch arithmetic is now explicit and logged.
- The old fake "FlashAttention" naming was removed from the production path.
- cuDNN frontend BF16 SDPA is wired for forward/backward.
- The CUDA record path uses true local `B=48, T=2048` execution rather than a 48-iteration serial full-model loop.
- Prepacked BF16 attention has freshness checks for the fused Q/K/RoPE/Gain producer.
- SmearGate has BOS/document-boundary masking and legality tests.
- Persistent full F32 logits are disabled in the record-shaped fast path.
- A chunked BF16 output CE cache avoids tiled-CE GEMM recompute and avoids a full local-batch logits allocation.
- Sharded Parallel Muon has reduce-scatter / local-shard update / all-gather scaffolding.
- Distributed GPU LoRA/phased TTT has score-before-update audit hooks and grouped packed LoRA gradient all-reduce.
- Code+model byte budget checks exist and fail closed when used in record/eval mode.

## What We Tried

| Cut | Outcome |
|---|---:|
| Original full record run | `91,218 ms/step`, only 7 steps in 638.5s. This proved the old path was executing the wrong H100 program. |
| cuDNN BF16 SDPA and true record-shaped batching | Closed the catastrophic gap down to the 250-300 ms/step range. |
| `v84_default_fast_no_stage` | `258.880 ms/step`. |
| `v85_fast_tf32` | `275.210 ms/step`, regression. Not promoted. |
| `v86_throughput_clean` | `256.787 ms/step`, current best clean floor. |
| `v87_u16_shift` compact u16 upload | `267.259 ms/step`, regression. Left as opt-in only. |
| Tiled output CE | Regressed output stage in earlier H100 logs because it repeated output GEMMs. Not promoted. |
| Chunked BF16 CE cache | Kept as bridge. Removes persistent full logits without repeated tile GEMMs. |
| BF16 attention backward tail | Implemented but regressed in A/B. Disabled until downstream BF16 QKV gradient path is complete. |
| Fast TF32 | Regressed. Disabled. |

## Remaining Blockers

These are the blockers that prevent claiming a leaderboard record:

- **Step time:** current best is ~256.8 ms/step; target is <=130 ms/step.
- **BF16 backward tail:** cuDNN can return BF16 dQ/dK/dV, but the downstream BF16 QK/RoPE/Gain and QKV gradient path is not complete enough to be faster.
- **Bucketed backward/NCCL overlap:** bank communication still launches after full backward.
- **Production fused projection+CE:** chunked BF16 CE cache is a bridge, not the final no-cache fused output projection + softcapped CE/backward kernel.
- **GPU-resident sampler:** current best path still samples on host and copies input/target each step, although H2D is currently not the dominant bottleneck.
- **Full leaderboard eval:** full validation distributed GPU LoRA/phased TTT has not been proven under 600s.
- **Final artifact proof:** code+model byte accounting exists, but no final leaderboard artifact has been produced from the current Rust path.
- **3-seed score:** no full-validation 3-seed BPB exists.

## Why This Is Submitted as Non-Record

The official leaderboard top score is around 1.061 BPB. This Rust stack has not produced a full validation BPB. It also does not yet meet the step-time required to execute ~4,600-4,900 record steps in 600 seconds.

The useful contribution is the systems work: a measured Rust/CUDA record-shaped training path, a concrete bottleneck ledger, and negative results that identify which cuts did and did not move the H100 runtime.

## Files

```text
README.md                     - submission summary
TECHNICAL_REPORT.md           - detailed engineering report and roadmap
ARCHITECTURE_BLOG.md          - long-form architecture writeup
submission.json               - machine-readable metadata
specs/frontier_1855_merged_target.toml
scripts/exact_modal_commands.sh
logs/v86_record_shaped_clean.log
logs/v87_u16_shift_regression.log
logs/modal_connectivity_failure.log
logs/local_validation.log
artifacts/artifact_budget.json
```

## Reproduction

See `scripts/exact_modal_commands.sh` for the exact commands used for the record-shaped proxy measurements and local validation.

The implementation lives in `parameter-golf-rs/` in this PR. This record folder intentionally does not claim to be a minimal leaderboard package.
