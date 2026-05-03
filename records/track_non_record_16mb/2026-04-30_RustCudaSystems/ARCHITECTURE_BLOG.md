# Building a Rust/CUDA Parameter Golf Runtime Under the 10-Minute Constraint

This is the longer architecture writeup for my non-record submission. The short version is in `README.md`; this document is meant for readers who want to understand what was actually built, what moved the H100 runtime, and what still blocks a real leaderboard attempt.

I am not claiming a SOTA result here. The final measured record-shaped runtime is still too slow. The reason I think this is still worth submitting is that the project crossed the hard systems boundary: it now runs the real record-shaped workload in Rust/CUDA and produces useful measurements on the exact parts of the stack that matter.

## The Starting Point

The early Rust stack had a few good pieces but was not yet executing the real competition program:

- GPU forward parity existed, but training was not fully GPU-native.
- NCCL was originally just a rank/world-size placeholder.
- Record mode had local safety caps and did not exercise the full wallclock budget.
- Several frontier architecture variants existed in the search space but were not executable.
- Quantization export was not fully driven by the spec.
- The attention path was named like FlashAttention but was still a scalar F32 causal SDPA kernel.

That produced a misleading situation: small proxy runs could look plausible, while real record-shaped runs collapsed.

The first valid 8xH100 record-shaped run made the problem unambiguous:

```text
steps_completed        = 7
wallclock_seconds      = 638.528
ms_per_step            = 91,218.335
global_batch_tokens    = 786,432
seq_len                = 2,048
local_batch_per_rank   = 48 sequences
```

That was not a tuning problem. It meant the runtime was still executing the wrong computational graph for H100s.

## The Real Target Shape

For the late-frontier SP8192 target, the important training shape is:

```text
world_size             = 8
global_batch_tokens     = 786,432
seq_len                = 2,048
local_tokens_per_rank  = 98,304
local_batch_per_rank   = 48 sequences
model_dim              = 512
heads                  = 8
kv_heads               = 4
head_dim               = 64
```

The target tensor layout is:

```text
input_ids       [B, T]
targets         [B, T]
hidden          [B*T, D]
Q/K/V           [B, T, H, Dh]
attention out   [B, T, H, Dh]
linear inputs   [B*T, D]
```

The most important early correction was to make this shape explicit in logs and fail-closed checks. A proxy run with small sequence length is not a performance result for this problem.

## Workspace Architecture

The implementation is split into Rust crates so each subsystem has a clean owner:

```text
pg-train    record-shaped runner, distributed orchestration, timing/audit logs
pg-model    GPU transformer runtime, activation buffers, frontier architecture
pg-kernels  CUDA kernels and cuDNN frontend SDPA bridge
pg-optim    GPU Muon / PolarNS / sharded Parallel Muon support
pg-core     CUDA tensor wrappers and NCCL collectives
pg-data     shard-backed token streams and record-shaped loaders
pg-quant    int6/int7/LQER export and byte-budget accounting
pg-eval     legal GPU LoRA/phased TTT eval path
pg-bench    parity and profiling entrypoints
```

The final runtime is still not small enough to be a leaderboard package by itself, but that was not the point of this submission. The dev stack is a systems vehicle: make correctness visible, make the H100 bottlenecks measurable, and keep enough structure that the remaining work can be attacked directly.

## Record-Shaped Audit

The runner now emits a machine-readable `record_audit_json` before training. It records:

```text
record_shape
seq_len
global_batch_tokens
world_size
local_batch
attention_backend
model_compute_dtype
distributed_optimizer_backend
microbatch_serial_loop
output loss backend
SmearGate boundary masking
TTT settings
artifact byte fields
frontier readiness gaps
```

This became one of the most useful pieces of the stack. Instead of asking "is the run fast?", I can ask "is this run even measuring the final workload?"

## Attention: From Scalar SDPA to cuDNN BF16 SDPA

The original CUDA attention path was a scalar F32 causal attention kernel. At `T=2048`, that is a non-starter. Each GPU has to process 48 local sequences per step, and scalar O(T^2) attention dominates immediately.

The replacement path uses cuDNN frontend BF16 scaled dot-product attention. The current record-shaped path has:

```text
attention_backend                 = CudnnSdpaBf16
cudnn_saved_bf16_attention         = true
cudnn_prepacked_bf16_attention     = true
prepacked_bf16_qkv_freshness_checked = true
```

The prepacked path matters because Q/K/V are produced by a fused QK/RoPE/Gain path. It is not safe to simply pass BF16 buffers to cuDNN and hope they are fresh. The runtime now checks that the prepacked buffers were produced by the fused producer for the current step, layer, and shape.

This was the single biggest systems win. It moved the stack from the 91-second/step regime into the 250-300 ms/step regime.

## Batch Execution: No More 48x Serial Full-Model Loop

The repaired record semantics imply 48 local sequences per rank:

```text
786,432 / (8 * 2,048) = 48
```

A naive implementation would run:

```text
for sequence in 0..48:
    forward full model
    backward full model
```

That is not competitive. The current CUDA path folds the local batch into real batched tensors and runs GEMMs over `M = B*T = 98,304`. The audit reports:

```text
microbatch_serial_loop = false
gemm_m_dimension       = 98304
attention_batch_dimension = 48
```

This is why the current results are meaningful even though they are not yet fast enough.

## Output Projection and Cross Entropy

The original forward path materialized full logits:

```text
[tokens, vocab] = [98,304, 8,192]
```

That is over 805M logits per rank. In F32, just the logits tensor is about 3.2GB. The fast path no longer persists that full tensor.

The current bridge is a chunked BF16 output CE cache:

```text
materializes_full_logits      = false
materializes_full_bf16_logits = false
chunked_bf16_output_ce_cache  = true
output_ce_chunk_tokens        = 8192
```

I also tried tiled CE. That was the wrong cut: it removed persistent logits but repeated output projection GEMMs, and the measured output stage got worse. The final production cut should be a real fused output projection + softcapped CE/backward kernel, not repeated GEMM tiling.

## Distributed Optimizer

The sharded optimizer target is:

```text
reduce-scatter bank gradients
update local parameter shard with Muon / PolarNS
all-gather updated parameters
```

The current runtime has the sharded Parallel Muon scaffolding and BF16 bank gradient wire path:

```text
distributed_optimizer_backend = ShardedParallelMuon
sharded_parallel_muon_reduce_scatter = true
sharded_parallel_muon_local_shard_update = true
sharded_parallel_muon_all_gather = true
sharded_parallel_muon_bank_grad_wire_dtype = bf16
```

The remaining distributed blocker is overlap. Today, full backward finishes first, then NCCL runs, then optimizer runs. The production design needs bucketed reduce-scatter fired as gradients become ready, using a comm stream and per-bucket events.

## Legal TTT and Eval

The eval path targets score-first GPU LoRA/phased TTT:

```text
score validation chunk
assert LoRA state did not mutate during scoring
only then update LoRA state using already-scored tokens
```

The distributed eval path now has grouped packed LoRA gradient all-reduce and audit logs:

```text
lora_grad_packed_all_reduce = true
lora_grad_grouped_all_reduce = true
score_first = true
future_token_access = false
```

This is implemented, but not full-validation proven under 600 seconds. That is why this submission does not claim leaderboard BPB.

## SmearGate Legality

SmearGate mixes previous-token state. In a packed stream, that can leak across document boundaries if the previous token belongs to a different document.

The runtime now requires a boundary token and logs:

```text
smeargate_bos_doc_mask = true
smear_gate_boundary_token_id = 1
```

The intended rule is:

```text
allow previous-token mixing only when current token is not BOS / not a new document
```

This is necessary for any final score to be defensible.

## Quantization and Artifact Budget

The frontier target spec includes:

```text
matrix_bits = 6
embed_bits = 7
attn_gate_bits = 8
LQER rank = 4
LQER top_k = 3
compression = pergroup
```

The repo now has code+model budget accounting instead of model-only accounting. That distinction matters because official submissions count code bytes plus compressed model bytes.

This submission does not include a final leaderboard artifact. The artifact proof attempts were blocked by Modal connectivity failures before deadline. The record folder includes `artifacts/artifact_budget.json` with that status explicitly marked rather than pretending a final artifact exists.

## Measurement Ledger

The best measurements I have are:

| Run | Result | Interpretation |
|---|---:|---|
| First valid real record run | 91,218.335 ms/step | Structurally wrong H100 graph. |
| v84 default fast no-stage | 258.880 ms/step | Good baseline after major systems cuts. |
| v85 fast TF32 | 275.210 ms/step | Regression. Disabled. |
| v86 throughput clean | 256.787 ms/step | Current best clean floor. |
| v87 compact u16 upload | 267.259 ms/step | Regression. Disabled by default. |
| v88/v90 export/graph probes | No result | Modal server connection failure. |

The current best is about 355x faster than the first valid record-shaped run, but still about 2x slower than the record target.

## What Did Not Work

### Tiled CE

This sounded right because it avoids full logits. In practice, it repeated the output projection work and made the output stage worse. The next cut has to fuse projection and CE more deeply.

### BF16 attention backward tail

cuDNN can return BF16 dQ/dK/dV, but the rest of the backward tail must stay BF16 for that to help. Partial BF16 tail regressed because the downstream QKV-gradient path still had enough conversion and packing overhead.

### Compact u16 upload

I expected this to help by halving input-transfer bytes and constructing shifted targets on device. It regressed:

```text
v86 clean floor      = 256.787 ms/step
v87 compact upload   = 267.259 ms/step
```

The reason is simple: H2D was already under 1 ms/step, so the added device work did not pay for itself.

### Fast TF32

Fast TF32 also regressed. The existing BF16 tensor-op GEMM path is better for this workload.

## Remaining Gap

The current system is blocked by:

1. BF16 backward activation graph: keep dQ/dK/dV and QKV gradient flow BF16 end to end.
2. Bucketed backward/NCCL overlap: launch reduce-scatter as gradient buckets become ready.
3. Production fused output projection + softcapped CE/backward.
4. Fully GPU-resident or CUDA-graphable sampler.
5. Full legal distributed eval/TTT under 600 seconds.
6. Final artifact proof under code bytes + compressed model bytes < 16,000,000.
7. 3-seed full-validation BPB.

## Final Takeaway

The Rust stack is not a leaderboard record yet. But it now runs the real record-shaped workload, exposes the right audit fields, and has enough measurements to separate useful kernel work from attractive but losing optimizations.

That is the contribution: not a score, but a concrete systems map of what it takes to move a Rust/CUDA Parameter Golf runtime from correctness parity toward record-grade H100 execution.
