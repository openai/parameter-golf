# Week 3: From Forward Parity to a Real GPU Training Pipeline

**Parameter Golf Competition Blog — Cedric Haddad**

---

Last week ended with the first hard correctness gate: the Rust GPU forward pass matched the CPU reference. This week was about turning that from a forward-only milestone into a train/eval pipeline that can actually run on H100s.

The headline result is that the Rust stack has now moved from “GPU forward works” to **GPU forward, backward, optimizer, distributed smoke, export, and eval all execute on Modal H100s**. It is still not leaderboard-quality in BPB, but the failure mode is much more useful now: the core systems path is alive, measurable, and no longer just CPU/GPU fallback scaffolding.

## The Core Insight: Correctness Gates Changed the Kind of Problem

At the beginning of the week, the important question was still correctness. The forward pass had passed parity, but a training stack needs much more than matching logits. It needs gradients, optimizer updates, distributed synchronization, artifact export, and legal evaluation.

The H100 runs this week cleared several of those gates.

| Gate | Result |
|---|---:|
| Baseline forward parity | `max_abs_diff = 0.000002`, `parity_ok` |
| Frontier forward parity | `max_abs_diff = 0.000002`, `parity_ok` |
| Baseline one-step GPU backward parity | `gpu_backward_status = ok` |
| Frontier one-step GPU backward parity | `gpu_backward_status = ok` |
| CUDA-single proxy train/export | success |
| 8-GPU distributed smoke | success |
| 8-GPU distributed proxy train/export | success |
| 8-GPU record-mode train/export | success, but too slow |
| GPU LoRA/phased TTT eval | success after one layout fix |

This changed the project from a debugging exercise into an optimization exercise. The most important remaining gap is not that the Rust stack cannot run the current frontier ideas. It can now run recurrence, split parallel residuals, AttnOutGate, Flash-style attention, GPU Muon, sharded distributed updates, and GPU LoRA/phased TTT. The gap is that the current record run is much too slow and the measured BPB is still far from leaderboard quality.

## Experiment 1: Kernel and Forward Parity

I started by rerunning the low-level parity tests on H100. The custom kernels matched the CPU references exactly under the current harness:

| Kernel family | Max abs diff |
|---|---:|
| embedding gather / backward | `0.000000` |
| bigram hash / backward | `0.000000` |
| RMSNorm / backward | `0.000000` |
| SmearGate / backward | `0.000000` |
| residual mix / add-scale / backward | `0.000000` |
| QK norm / backward | `0.000000` |
| partial RoPE / backward | `0.000000` |
| q-gain / backward | `0.000000` |
| causal attention / backward | `0.000000` |
| XSA / backward | `0.000000` |
| GEMM linear / backward | `0.000000` |

The model-level forward parity results were:

| Spec | Backend | Tokens | Max abs diff | Mean abs diff | Status |
|---|---|---:|---:|---:|---|
| `baseline_sp8192` | `cuda-single` | `16` | `0.000002` | `0.000000` | `parity_ok` |
| `frontier_sp8192_target` | `cuda-single` | `16` | `0.000002` | `0.000000` | `parity_ok` |

The second row matters. `frontier_sp8192_target` includes the newer frontier architecture features that the baseline does not: recurrence, split parallel residuals, AttnOutGate, Flash-style attention selection, and GPU LoRA/phased TTT as the eval backend. Passing forward parity there means the new architecture path is executable, not just described in the spec.

## Experiment 2: One-Step Backward Parity

The next gate was one-step parity. This compares CPU and GPU losses and selected gradients on the same deterministic batch.

| Spec | Loss | GPU loss abs diff | Max grad abs diff | Max grad rel diff | Status |
|---|---:|---:|---:|---:|---|
| `baseline_sp8192` | `9.008008` | `1.907349e-6` | `1.907349e-6` | `2.379503e-5` | `ok` |
| `frontier_sp8192_target` | `9.021423` | `1.907349e-6` | `1.907349e-6` | `3.580978e-5` | `ok` |

The gradient checks covered representative parameter groups:

- token embedding
- Q/O bank
- K/V bank
- MLP up bank
- MLP down bank
- SmearGate
- q-gain

This is not yet a proof that every long training run will be stable, but it is the correct gate before trusting training metrics. The important result is that the GPU backward path is now numerically aligned with the CPU reference on both the baseline and frontier variants.

## Experiment 3: CUDA-Single Training and Export

The `cuda-single` smoke run on the frontier spec completed successfully:

| Metric | Value |
|---|---:|
| Steps | `4` |
| Mean step time | `75.799 ms/step` |
| Train loss | `20.964806` |
| Train data source | shard-backed data |
| BPB byte source | tokenizer vocab |
| Bank update backend | `gpu_muon_ns5` |

The proxy run then exercised the train/export path:

| Metric | Value |
|---|---:|
| Steps | `32` |
| Mean step time | `53.196 ms/step` |
| Wallclock | `1.702 s` |
| Tokens seen | `2,048` |
| Train loss | `17.148842` |
| Proxy BPB | `7.132411` |
| Artifact bytes | `4,916,759` |
| Code bytes | `4,645,336` |
| Total counted bytes | `9,562,095` |
| Artifact budget OK | `true` |

This is a major improvement over the earlier fallback-heavy proxy runs, which were taking more than two minutes per step. The comparison is not apples-to-apples in sequence length or batch size, so I am not treating it as a final speedup number. But directionally, it confirms that moving the optimizer and backward path onto the GPU changes the order of magnitude of the system.

One piece of feedback from last week was that I should time the forward pass and compare CPU versus CUDA. I still do not want to overclaim here. The numbers above are **end-to-end training step times**, not isolated forward latency. The forward-only benchmark still needs to be added. The correct measurement should report CPU forward latency, CUDA forward latency, tokens/sec, and p50/p95 after warmup on the same batch.

## Experiment 4: GPU LoRA/Phased TTT Eval

The first eval run exposed a useful bug. The GPU LoRA path produced a `[tokens, model_dim]` delta, while the Q tensor is represented as `[tokens, heads, head_dim]`. Those layouts contain the same number of elements and use the same contiguous memory order, but the tensor shape check correctly rejected the add:

`ShapeMismatch { expected: [2048, 8, 64], got: [2048, 512] }`

The fix was to reshape the LoRA delta view before adding it to Q. After that, eval completed on H100.

For the CUDA-single proxy artifact:

| Metric | Value |
|---|---:|
| Eval tokens | `16,384` |
| Eval loss | `27.382767` |
| Eval BPB kind | tokenizer vocab |
| Eval BPB | `10.672893` |
| Total counted bytes | `7,820,295` |
| Artifact budget OK | `true` |

For the 8-GPU distributed proxy artifact:

| Metric | Value |
|---|---:|
| Eval tokens | `16,384` |
| Eval loss | `17.187910` |
| Eval BPB kind | tokenizer vocab |
| Eval BPB | `6.699276` |
| Total counted bytes | `10,569,576` |
| Artifact budget OK | `true` |

For the 8-GPU record artifact:

| Metric | Value |
|---|---:|
| Eval tokens | `16,384` |
| Eval loss | `11.669149` |
| Eval BPB kind | tokenizer vocab |
| Eval BPB | `4.548247` |
| Total counted bytes | `7,080,720` |
| Artifact budget OK | `true` |

These BPB values are not competitive yet. They are also not official leaderboard BPB because they use a 16k-token validation slice, not the full validation run. Their value is operational: the artifact can be exported, loaded, legally scored before adaptation, adapted with GPU LoRA/phased TTT, and byte-counted under the submission budget.

## Experiment 5: 8-GPU Distributed Training

The 8-GPU smoke run passed:

| Metric | Value |
|---|---:|
| Backend | `cuda-distributed` |
| World size | `8` |
| Steps | `4` |
| Mean step time | `312.661 ms/step` |
| Global batch tokens | `128` |
| Distributed sync | `true` |
| Distributed optimizer | `ShardedParallelMuon` |
| Bank update backend | `nccl_reduce_scatter_all_gather_parallel_muon_ns5` |
| Frontier record ready flag | `true` |

The 8-GPU proxy run also passed:

| Metric | Value |
|---|---:|
| Steps | `32` |
| Mean step time | `132.554 ms/step` |
| Wallclock | `4.242 s` |
| Global batch tokens | `512` |
| Tokens seen | `16,384` |
| Train loss | `15.071462` |
| Proxy BPB | `5.834740` |
| Artifact bytes | `7,666,040` |
| Code bytes | `4,649,464` |
| Total counted bytes | `12,315,504` |
| Artifact budget OK | `true` |

This is the first run where the stack exercised the intended distributed shape: 8 GPUs, shard-backed data, tokenizer byte accounting, sharded Parallel Muon, and export under the full code-plus-artifact budget.

The full 8-GPU record-mode run also completed:

| Metric | Value |
|---|---:|
| Steps | `7` |
| Mean step time | `91,218.335 ms/step` |
| Wallclock | `638.528 s` |
| Seq len | `2,048` |
| Global batch tokens | `786,432` |
| Local microbatches per step | `48` |
| Tokens seen | `5,505,024` |
| Train loss | `13.435371` |
| Distributed sync | `true` |
| Artifact bytes | `4,177,184` |
| Code bytes | `4,649,464` |
| Total counted bytes | `8,826,648` |
| Artifact budget OK | `true` |

This is operationally important but not competitive. The run did use the intended record semantics: 8 GPUs, `seq_len=2048`, `global_batch_tokens=786432`, real shard data, tokenizer byte accounting, sharded Parallel Muon, and a budget-valid artifact. But `91.2 s/step` means the system only completed seven optimizer steps. A top leaderboard submission needs far more useful updates inside the 600-second window.

## Metric Definitions

I am being explicit about metric definitions because several numbers this week look similar but mean different things.

`max_abs_diff` is the maximum absolute elementwise difference between CPU and GPU outputs or gradients:

`max_i |cpu_i - gpu_i|`

`mean_abs_diff` is the average absolute elementwise difference:

`(1/N) * sum_i |cpu_i - gpu_i|`

`ms/step` is end-to-end training time per optimizer step:

`1000 * wallclock_seconds / steps_completed`

It includes forward, backward, optimizer, synchronization, data movement, and export only if export happens inside the measured loop. It is not forward-only latency.

`proxy BPB` is a diagnostic estimate from short runs. It is useful for catching obviously broken pipelines, but it is not the official leaderboard metric.

`eval BPB` in this post is tokenizer-backed BPB on a 16k-token validation slice. It is closer to the official metric than proxy BPB, but still not final because the leaderboard requires the full legal eval path under the competition constraints.

`total counted bytes` is compressed model artifact bytes plus the measured binary/code bytes used by the Rust path. This is stricter than model-only artifact size and better reflects the actual 16MB budget.

## Challenges and Roadblocks

The main challenge this week was no longer basic CUDA correctness. It was making the validation stack honest enough that performance and quality numbers mean something.

One example was the LoRA shape mismatch. A less strict tensor layer might have silently added incompatible views or forced an unnecessary copy. The explicit shape check caught the bug at the boundary where it happened. The fix was small, but the lesson was important: GPU eval-time adaptation is complicated enough that shape/layout invariants need to be enforced aggressively.

The second challenge is that short proxy BPB is still a weak quality signal. The distributed proxy looks better than the single-GPU proxy, but both are trained on tiny token counts compared with a real record attempt. These runs prove the system path, not leaderboard competitiveness.

The third challenge is Modal orchestration. Short jobs are now easy to run and inspect. Long record-mode jobs need stronger detached result handling, ideally writing structured JSON results to a persistent volume rather than relying on stdout and client heartbeats.

## Evidence of Progress

This week produced several concrete milestones:

- Frontier forward parity passes on H100.
- Frontier one-step backward parity passes on H100.
- CUDA-single training runs with GPU Muon/NS5 bank updates.
- 8-GPU distributed training runs with `distributed_sync=true`.
- A full 8-GPU record-mode run completes and exports a budget-valid artifact.
- Sharded Parallel Muon is selected in the distributed path.
- GPU LoRA/phased TTT eval runs after fixing the Q-layout mismatch.
- Artifacts are exported, reloaded, evaluated, and checked against code-plus-model byte budget.

The important negative result is also clear: the current BPB is not close to the leaderboard. The stack is now operational, but the record path completes only seven steps in roughly the full time budget. It needs much higher throughput and real quality tuning before it can be considered a submission candidate.

## Plans for Next Week

The next work should be narrower, not broader.

First, I need an isolated forward timing benchmark to answer the grader feedback directly:

1. fixed weights
2. fixed batch
3. CPU forward warmup and timed loop
4. CUDA forward warmup and timed loop
5. mean, p50, p95, and tokens/sec for both

Second, the record path needs to become fast enough to matter. The first record-mode run proves the semantics, but `91.2 s/step` is not viable. The next optimization target is to identify whether the dominant cost is attention at `seq_len=2048`, local microbatch looping, sharded Muon communication, or repeated synchronization around eval/export bookkeeping.

Third, I need to move from pipeline validation to quality. The high-priority ablations are the ones most aligned with the current competition frontier:

- QK gain around `5.25`
- AttnOutGate width around `24`
- LoRA TTT rank/alpha around `128/144`
- phased TTT schedule
- GPTQ calibration and clipping settings

The main lesson from Week 3 is that the Rust path is no longer just a proposal. It now has real H100 evidence across forward, backward, distributed training, export, and eval. The next question is not whether the stack can run. It is whether it can train enough useful tokens in 600 seconds to produce a competitive BPB.
