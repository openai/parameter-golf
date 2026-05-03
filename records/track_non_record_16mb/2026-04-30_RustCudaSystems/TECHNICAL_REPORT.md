# Technical Report: Rust/CUDA Record-Shaped Parameter Golf Runtime

## Executive Status

The Rust/CUDA stack has moved from a correctness prototype to a measurable record-shaped H100 runtime. The original real record run was effectively unusable:

```text
steps_completed        = 7
wallclock_seconds      = 638.528
ms_per_step            = 91,218.335
global_batch_tokens    = 786,432
```

The current best clean record-shaped proxy result is:

```text
timing_measured_ms_per_step = 256.787
world_size                  = 8
seq_len                     = 2048
global_batch_tokens          = 786432
local_batch_sequences        = 48 per rank
```

This is roughly a 355x improvement over the first real record-shaped run, but still about 2x short of the <=130 ms/step target needed for a top leaderboard submission.

This submission is therefore intentionally non-record. It documents the Rust/CUDA systems work, the measured negative results, and the remaining production cuts.

## Target Surface

The final leaderboard target we used for engineering was the late-frontier SP8192 stack:

```text
SP8192 tokenizer family
CaseOps byte sidecar
11 layers, 512 model dim, 8 heads, 4 KV heads
seq_len = 2048
global_batch_tokens = 786432
cuDNN BF16 SDPA
SparseAttnGate
BOS-safe SmearGate
Polar Express / sharded Parallel Muon target
int6 matrices, int7 embeddings
LQER asymmetric rank-4 correction target
GPU LoRA phased score-first TTT target
```

The explicit spec used for record-shaped benchmarking is included at `specs/frontier_1855_merged_target.toml`.

## Major Systems Milestones

### 1. Record-shaped benchmarking

The runner now distinguishes smoke/proxy runs from record-shaped runs. The record-shaped audit logs:

```text
record_shape
seq_len
global_batch_tokens
world_size
local_batch
attention_backend
optimizer_backend
microbatch_serial_loop
artifact byte fields
frontier gaps
```

This prevented the earlier mistake of treating small-sequence proxy speed as evidence of record-speed readiness.

### 2. Real attention path

The old scalar F32 SDPA path was renamed and demoted. The production record-shaped path uses cuDNN frontend BF16 SDPA. This was the main reason the stack moved out of the 91-second-per-step regime.

Remaining issue: the BF16 attention backward tail is not yet fully profitable because downstream QK/RoPE/Gain and QKV gradient accumulation are still partly F32.

### 3. True B=48 execution

The CUDA record-shaped path folds the local batch into real tensor dimensions:

```text
input_ids    [B*T]
hidden       [B*T, D]
qkv          [B, T, H, Dh]
attention    [B, T, H, Dh]
GEMM M       = 98,304
```

This avoids the catastrophic 48x serial full-model loop.

### 4. Prepacked BF16 QKV safety

The prepacked attention path now requires a fresh fused Q/K/RoPE/Gain producer and logs:

```text
prepacked_bf16_qkv_freshness_checked = true
cudnn_prepacked_bf16_qk_fresh_producer = true
```

This prevents stale or uninitialized BF16 Q/K/V buffers from silently entering cuDNN attention.

### 5. SmearGate legality

SmearGate is BOS/document-boundary masked. The audit logs:

```text
smeargate_bos_doc_mask = true
smear_gate_boundary_token_id = 1
```

This addresses the known leakage risk where previous-token mixing can cross packed document boundaries.

### 6. Output path

The persistent full F32 logits tensor is no longer used in the fast record-shaped path. A chunked BF16 CE cache is used instead:

```text
materializes_full_logits      = false
materializes_full_bf16_logits = false
chunked_bf16_output_ce_cache  = true
output_ce_chunk_tokens        = 8192
```

This is not the final production cut. It is a bridge that avoids both persistent full logits and the measured-worse tiled CE repeated-GEMM path.

### 7. Distributed optimizer

The distributed optimizer target is sharded Parallel Muon:

```text
reduce-scatter bank grads
local shard update
all-gather updated params
BF16 bank grad wire path
BF16 shadow all-gather path
```

Current status: scaffolding and proof logs exist, but bucketed backward/NCCL overlap is not implemented. Communication still launches after full backward.

### 8. Legal GPU TTT

GPU LoRA phased TTT is implemented with score-before-update audit hooks. Distributed eval has grouped packed LoRA gradient all-reduce.

Current status: this is not full-validation proven under 600 seconds.

## Run Ledger

| Run | Result | Decision |
|---|---:|---|
| Initial real record | 91,218.335 ms/step | Proved the original path was structurally wrong. |
| v84 default fast no-stage | 258.880 ms/step | Good baseline. |
| v85 fast TF32 | 275.210 ms/step | Regression. Disabled. |
| v86 throughput clean | 256.787 ms/step | Current best clean floor. |
| v87 shifted u16 upload | 267.259 ms/step | Regression. Kept opt-in only. |
| v88/v90 graph/export probes | No result | Modal server connection failed before execution. |

## Negative Results

### Tiled output CE

Tiled output CE was the wrong cut. It removed persistent logits but repeated output projection GEMMs and worsened the output stage in prior H100 logs. The final required cut is a real fused output projection + softcapped CE/backward path.

### BF16 attention backward tail

The BF16 tail is implemented, but enabling it regressed timing in A/B. The likely reason is that the saved BF16 dQ/dK/dV path still has enough downstream conversion and QKV-gradient overhead that it does not pay for itself.

### Compact u16 batch upload

The compact upload path reduced host-transfer size but regressed end-to-end step time:

```text
v86 clean floor       = 256.787 ms/step
v87 shifted u16 path  = 267.259 ms/step
```

H2D was already below 1 ms/step, so the extra device construction work did not pay off.

### Fast TF32

Fast TF32 was slower than the existing GEMM compute mode in this workload:

```text
v85 fast TF32 = 275.210 ms/step
```

It remains disabled.

## Remaining Roadmap

The next production cuts are ordered by expected impact:

1. Complete the BF16 backward activation graph so cuDNN dQ/dK/dV stays BF16 through QK/RoPE/Gain and QKV projection backward.
2. Implement bucketed backward/NCCL overlap with a comm stream and per-bucket readiness events.
3. Replace the chunked BF16 CE cache with a production fused output projection + softcapped CE/backward kernel.
4. Move sampling fully onto GPU or into a CUDA-graphable pinned/device ring.
5. Prove full legal distributed eval/TTT under 600 seconds.
6. Produce final artifact byte proof with code bytes + compressed model bytes under 16,000,000.
7. Run 3 full seeds and report mean/std BPB.

## Why No Leaderboard Claim

The current stack has no full validation BPB, no full legal TTT eval proof, no 3-seed result, and does not meet the <=130 ms/step target. Submitting it as a record would be misleading.

The value of this submission is the systems result: a Rust/CUDA implementation that reaches the real record-shaped H100 workload, documents the remaining gap quantitatively, and leaves concrete engineering cuts for future work.

