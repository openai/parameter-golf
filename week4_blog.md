# Week 4: From a 91-Second Record Step to a Rust/CUDA Systems Submission

**Parameter Golf Competition Blog — Cedric Haddad**

---

Last week ended with the Rust stack proving that GPU forward, backward, optimizer, distributed smoke, export, and eval could all execute on H100s. This week was about the harder question: could that path execute the **real record-shaped workload** fast enough to matter?

The short answer is: not yet. The longer answer is much more useful.

The stack moved from a real record run at **91.2 seconds per optimizer step** to a record-shaped 8xH100 systems runtime at **256.8 ms/step**. That is a roughly **355x improvement** over the first honest record-shaped run. It is still about **2x slower** than the 120-130 ms/step range needed for a top leaderboard submission.

Because of that, the final PR is intentionally a **non-record systems submission**, not a leaderboard claim. The submission documents the Rust/CUDA runtime, the H100 measurements, the negative results, and the remaining production cuts needed to turn the stack into a true record candidate.

## The Core Insight: Honest Record Shape Changed Everything

The biggest change this week was not one kernel. It was making the benchmark impossible to misread.

Earlier proxy runs were fast because they were not exercising the real competition shape. The real record shape is:

| Field | Value |
|---|---:|
| World size | `8` |
| Sequence length | `2,048` |
| Global batch tokens | `786,432` |
| Local batch sequences per rank | `48` |
| Local tokens per rank | `98,304` |

That means each H100 needs to process 48 sequences of length 2048 per optimizer step. A small-sequence proxy can hide structural problems. A record-shaped proxy cannot.

The first honest record run exposed the problem clearly:

| Metric | Value |
|---|---:|
| Steps completed | `7` |
| Wallclock | `638.528 s` |
| Mean step time | `91,218.335 ms/step` |
| Global batch tokens | `786,432` |
| Tokens seen | `5,505,024` |

That run was operationally valuable because it proved the fixed batch-token semantics were real. It was also obviously not competitive. A top record attempt needs roughly 4,600-4,900 useful optimizer steps in 600 seconds, not seven.

So the week became a systems exercise: replace the computational graph that produced 91-second steps with the graph an H100 actually wants to run.

## Experiment 1: Record-Shaped Audit Logs

The first cut was adding a record-shaped mode and audit fields that describe the actual training surface.

The runner now emits the key facts needed to judge whether a run is meaningful:

- `seq_len`
- `global_batch_tokens`
- `world_size`
- `local_batch`
- `local_microbatches_per_step`
- `attention_backend`
- `distributed_optimizer_backend`
- `microbatch_serial_loop`
- `materializes_full_logits`
- `smeargate_bos_doc_mask`
- `prepacked_bf16_qkv_freshness_checked`
- `frontier_record_ready`
- `leaderboard_algorithm_ready`

This is basic engineering hygiene, but it matters. It prevents a 64-token proxy from being confused with the real 2048-token record workload.

The current best clean run is labeled `v86_throughput_clean`:

| Metric | Value |
|---|---:|
| Mode | `RecordShapedProxy` |
| Backend | `cuda-distributed` |
| World size | `8` |
| Seq len | `2,048` |
| Global batch tokens | `786,432` |
| Local microbatches per step | `48` |
| Steps completed | `8` |
| Timing steps | `6` |
| Measured step time | `256.787 ms/step` |
| Distributed sync | `true` |
| Attention backend | `CudnnSdpaBf16` |
| Distributed optimizer | `ShardedParallelMuon` |
| Microbatch serial loop | `false` |
| Frontier record ready | `false` |

This is the number I trust most right now. It is not a leaderboard score. It is a systems timing result on the real shape.

## Experiment 2: Replacing the Attention Path

The original record path was dominated by attention. At `T=2048`, a naive attention implementation is exactly the wrong program to run on H100s. It scales quadratically in sequence length and spends too much time in scalar/F32 work.

This week the production record-shaped path moved to **cuDNN frontend BF16 SDPA**.

That change was the biggest reason the system left the 91-second regime. It also forced stricter naming. The old "FlashAttention" label was misleading because the implementation was not real FlashAttention. The stack now distinguishes the scalar F32 SDPA path from the cuDNN BF16 SDPA path.

The current attention state is:

| Item | Status |
|---|---|
| Scalar F32 SDPA | Demoted from record-shaped fast path |
| cuDNN BF16 SDPA forward/backward | Implemented |
| Prepacked BF16 Q/K/V path | Implemented |
| Q/K/V freshness checks | Implemented |
| BF16 attention backward tail | Implemented but not yet profitable |

The important safety fix was the prepacked Q/K/V freshness contract. Prepacked attention is only valid if the fused Q/K/RoPE/Gain producer wrote the BF16 buffers for the current step, layer, and shape. The audit now records:

```text
prepacked_bf16_qkv_freshness_checked = true
cudnn_prepacked_bf16_qk_fresh_producer = true
```

This avoids a dangerous class of bugs where cuDNN could consume stale BF16 Q/K/V buffers.

## Experiment 3: True Local B=48 Execution

The next structural issue was local batching.

The record shape implies:

```text
global_batch_tokens = 786,432
world_size          = 8
seq_len             = 2,048

local_tokens_per_rank = 98,304
local_batch_sequences = 48
```

If the code runs:

```text
for microbatch in 0..48:
    forward full model
    backward full model
```

then the H100 is paying for 48 serial model passes. That is not the target graph.

The CUDA record-shaped path now folds the local batch into actual tensor dimensions:

| Tensor | Shape |
|---|---|
| Input ids | `[B*T]` |
| Hidden | `[B*T, D]` |
| Q/K/V | `[B, T, H, Dh]` |
| Attention | `[B, T, H, Dh]` |
| GEMM M dimension | `98,304` |

The audit reports:

```text
microbatch_serial_loop = false
```

This was the second major structural fix. The remaining gap is no longer a catastrophic "wrong shape" problem. It is now concentrated in the backward graph, optimizer tail, and output/loss path.

## Experiment 4: Output Projection and Cross Entropy

The output path was one of the most important cuts this week.

At record shape, full logits are huge:

```text
tokens = 48 * 2048 = 98,304
vocab  = 8,192

logits = 805,306,368 elements
```

That is about 3.22 GB per rank in F32. Keeping a persistent `[tokens, vocab]` logits tensor is not acceptable for a final record engine.

The fast path now disables persistent full F32 logits and uses a chunked BF16 CE cache:

```text
materializes_full_logits      = false
materializes_full_bf16_logits = false
chunked_bf16_output_ce_cache  = true
output_ce_chunk_tokens        = 8192
```

This is an improvement, but it is not the final answer. I tried tiled output CE, but that was the wrong cut because it repeated output projection GEMMs and made the output stage worse. The correct production cut is still a real fused output projection + softcapped CE/backward path that avoids both persistent full logits and repeated GEMM passes.

The current bridge is good enough to keep output CE from dominating the step. It is not good enough to be called final.

## Experiment 5: Distributed Optimizer and Communication

The distributed path now targets sharded Parallel Muon. The intended update is:

```text
reduce-scatter bank gradients
local shard update
all-gather updated parameters
```

That is the right shape for a distributed optimizer because it avoids every rank redundantly updating the full parameter bank.

The current run uses:

```text
distributed_optimizer_backend = ShardedParallelMuon
distributed_sync = true
```

The implementation has proof hooks for:

- reduce-scatter bank gradients
- local shard update
- all-gather updated parameters
- BF16 bank gradient wire path
- BF16 shadow all-gather path

The main missing systems cut is overlap. Right now, the structure is still effectively:

```text
full backward
then communication
then optimizer
```

The next production version needs bucketed reduce-scatter during backward:

```text
layer bucket gradients ready
comm stream starts reduce-scatter
main stream continues earlier-layer backward
optimizer waits only for the relevant bucket
```

That is one of the remaining paths to getting from roughly 256 ms/step toward the 130 ms/step target.

## Experiment 6: Legal SmearGate and TTT Hooks

This week also closed a legality hazard around SmearGate.

SmearGate mixes information from neighboring tokens. If implemented naively on packed documents, the BOS token of document `B` can depend on the final token of document `A`. That would be a document-boundary leak.

The GPU path now applies a BOS/document-boundary mask. The audit reports:

```text
smeargate_bos_doc_mask = true
smear_gate_boundary_token_id = 1
```

The TTT path also moved in the right direction. GPU LoRA/phased TTT has score-before-update audit hooks, and distributed eval has grouped packed LoRA gradient all-reduce. That said, the full validation eval has not been proven under 600 seconds. So this remains a systems feature, not an official score claim.

For a final leaderboard submission, the required invariant is:

```text
score validation chunk
then update LoRA/TTT state using only that scored chunk
never update using future validation tokens
```

The code now has the right audit direction. It still needs a full validation proof.

## Experiment 7: Negative Results

Several aggressive cuts regressed. These were useful because they narrowed the remaining work.

| Cut | Result | Decision |
|---|---:|---|
| Fast TF32 | `275.210 ms/step` | Regression; disabled |
| Shifted u16 compact upload | `267.259 ms/step` | Regression; opt-in only |
| Tiled output CE | Slower output stage | Not promoted |
| BF16 attention backward tail | Regressed in A/B | Disabled until downstream BF16 gradient path is complete |

The shifted u16 result was especially instructive. It reduced host-transfer size but made the step slower:

| Run | Step time |
|---|---:|
| `v86_throughput_clean` | `256.787 ms/step` |
| `v87_u16_shift` | `267.259 ms/step` |

The v87 H2D time was only:

```text
timing_cuda_h2d_ms_per_step = 0.409
```

So host input transfer is not currently the bottleneck. It still needs to be cleaned up for CUDA graph capture and production determinism, but it is not where the next 100 ms is hiding.

## Experiment 8: Final Submission Package

Because the stack did not hit the leaderboard gates, I packaged it as a non-record systems submission.

The PR is:

```text
https://github.com/openai/parameter-golf/pull/2002
```

The record folder is:

```text
records/track_non_record_16mb/2026-04-30_RustCudaSystems/
```

It includes:

```text
README.md
TECHNICAL_REPORT.md
ARCHITECTURE_BLOG.md
PR_BODY.md
submission.json
logs/local_validation.log
logs/modal_connectivity_failure.log
logs/v86_record_shaped_clean.log
logs/v87_u16_shift_regression.log
scripts/exact_modal_commands.sh
specs/frontier_1855_merged_target.toml
artifacts/artifact_budget.json
```

This was the right submission type. A leaderboard record would need:

- full validation BPB
- train time under 600 seconds
- eval time under 600 seconds
- code bytes + compressed model bytes under 16,000,000
- score-first TTT audit over the full validation set
- 3-seed mean and standard deviation

The Rust stack does not have those proofs yet.

## Metric Definitions

`record-shaped proxy` is a systems timing mode that uses the real record dimensions but does not claim to produce a leaderboard artifact or score.

`ms/step` is measured optimizer-step time:

```text
1000 * measured_wallclock_seconds / measured_steps
```

For `v86`, this was computed over six timing steps after setup.

`global tokens/sec` is:

```text
global_batch_tokens / seconds_per_step
```

At `256.787 ms/step`, the current stack is around:

```text
786,432 / 0.256787 ~= 3.06M global tokens/sec
```

At the 130 ms target, it would be:

```text
786,432 / 0.130 ~= 6.05M global tokens/sec
```

`microbatch_serial_loop=false` means the local batch is not being processed as 48 serial full-model passes.

`frontier_record_ready=false` means the stack is not yet allowed to claim leaderboard readiness. In the current run, that is correct.

`leaderboard_algorithm_ready=true` means the spec surface has the intended late-frontier features wired well enough to execute in the systems harness. It does not mean the model has reached leaderboard BPB.

`total counted bytes` means code bytes plus compressed model bytes. This is the only byte accounting that matters for the official 16MB rule.

## Review Findings Status

Several earlier review findings were addressed during the week.

| Finding | Week 4 status |
|---|---|
| Training was CPU-only | Closed for the CUDA record-shaped path. GPU backward and GPU grad buffers exist. |
| NCCL was a placeholder | Closed for the systems prototype. Collectives are wired, but overlap is still missing. |
| Record mode was capped below real semantics | Closed for record semantics. Record-shaped proxy intentionally runs a short fixed timing window. |
| Finalist variants were not executable | Mostly closed. The frontier 1855-like spec is executable in the record-shaped path. |
| `QuantSpec` was not driving export | Improved, but not fully closed. The spec has matrix/embed/gate/LQER fields; final artifact proof is missing. |
| Prepacked BF16 attention could read stale buffers | Addressed with producer/freshness checks. |
| Full logits were materialized | Addressed in the fast path with chunked BF16 CE cache; production fused CE remains. |
| BF16 attention bridged back to F32 | Improved, but the full activation/backward graph is still partly F32. |
| SmearGate needed document-boundary audit | Addressed with BOS/document-boundary masking. |

The important point is that the remaining blockers are now much narrower than they were at the start of the week.

## Challenges and Roadblocks

The biggest technical challenge was that several obvious optimizations were not actually wins.

Tiled CE sounded right because full logits are expensive. But the implementation repeated output projection GEMMs, so it moved cost rather than eliminating it. The lesson is that the final CE path needs to fuse projection and loss semantics, not just tile the existing computation.

The BF16 backward tail had a similar problem. Returning BF16 from cuDNN attention is only useful if the downstream QK/RoPE/Gain backward and QKV projection backward consume that representation efficiently. If the path converts back and forth, BF16 becomes overhead instead of speed.

The final operational blocker was Modal availability. The final artifact/export/eval proof attempts failed with:

```text
Could not connect to the Modal server.
```

Because of that, there is no final H100 artifact export, full eval, or 3-seed validation proof in the package.

## Evidence of Progress

This week produced a concrete systems result:

| Milestone | Status |
|---|---|
| Real record-shaped workload identified | Done |
| Record-shaped audit fields | Done |
| cuDNN BF16 SDPA path | Done |
| True B=48 local batch execution | Done |
| Prepacked BF16 Q/K/V safety checks | Done |
| BOS-safe SmearGate | Done |
| Persistent full F32 logits removed from fast path | Done |
| Sharded Parallel Muon scaffold | Done |
| GPU LoRA/phased TTT audit hooks | Done |
| Code+model budget infrastructure | Partial |
| Full validation leaderboard proof | Not done |
| 3-seed result | Not done |

The best single summary is:

```text
initial honest record run: 91,218.335 ms/step
best clean record-shaped run: 256.787 ms/step
target for top submission: <=130 ms/step
```

That means the Rust stack is no longer fundamentally executing the wrong program. It is now a serious systems candidate with a roughly 2x remaining throughput gap.

## Plans for Next Week

First, finish the BF16 backward activation graph. cuDNN can produce BF16 attention gradients, but the rest of the backward tail needs to consume them without redundant conversion.

Second, implement bucketed backward/NCCL overlap. Communication needs to start as soon as gradient buckets are ready, not after the entire backward pass finishes.

Third, replace the chunked BF16 CE cache with a real fused output projection + softcapped CE/backward kernel. The current bridge is useful, but the final path should not store full logits and should not repeat output GEMMs.

Fourth, move sampling into a GPU-resident or CUDA-graphable path. H2D is not the current bottleneck, but the final record engine needs a static schedule.

Fifth, run full legal distributed eval with score-before-update TTT audit logs. Without that, there is no leaderboard score.

Finally, produce a real submission package only after the stack clears the gates:

```text
record-shaped step time <= 130 ms
train wallclock <= 600 s
full eval wallclock <= 600 s
code bytes + compressed model bytes < 16,000,000
3-seed full validation BPB reported
```
