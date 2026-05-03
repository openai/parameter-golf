# Week 2: Forward Parity, Metric Discipline, and Finding the Real Bug

**Parameter Golf Competition Blog — Cedric Haddad**

---

This week was about solidifying the rust stack. The main milestone was not speed, and it was not multi-GPU training. It was **forward-pass parity**: proving that the GPU implementation produces the same logits as the CPU reference model on the same weights and the same inputs.

That gate now passes.

On the H100, the final result for the baseline `SP8192` forward path was:

- `max_abs_diff = 0.000002`
- parity threshold: `1e-3`
- status: `parity_ok`

The most important insight from this week is that the main source of error was **not** one of the custom CUDA kernels. It was a **layout mismatch in the GEMM path**. Once we isolated the problem carefully, the fix was small. Getting to that point required much better metric definitions and much more disciplined validation than I used in Week 1.

## The Core Insight and difficulties: The Hard Part Was Not CUDA Math, It Was Convention Matching

At the start of the week, the GPU model executed end-to-end on the H100 but failed parity badly:

- `max_abs_diff = 0.398665`
- `mean_abs_diff = 0.134476`

Those numbers told us the implementation was not just slightly numerically different. Something structural was wrong.

The key debugging move was to stop treating the model as a black box and instead test each GPU operator against its CPU reference on deterministic inputs. I added a kernel-level parity harness and measured every major custom op separately.

The result was surprisingly clean:

| Kernel / op | Max abs diff |
|---|---:|
| `embedding_gather` | `0.000000` |
| `bigram_hash` | `0.000000` |
| `rms_norm` | `0.000000` |
| `smear_gate` | `0.000000` |
| `residual_mix` | `0.000000` |
| `residual_add_scale` | `0.000000` |
| `qk_norm` | `0.000000` |
| `partial_rope` | `0.000000` |
| `q_gain` | `0.000000` |
| naive causal attention | `0.000000` |
| `xsa` | `0.000000` |
| `gemm_linear` | `0.114000` |

This was the turning point of the week. Every custom kernel matched perfectly. The entire model-level mismatch was downstream of one thing: **the linear projection convention used by cuBLAS**.

The CPU reference uses the semantics

`Y = X @ W^T`

with:

- `X` stored row-major as `[m, k]`
- `W` stored row-major as `[n, k]`
- `Y` stored row-major as `[m, n]`

My initial GPU path did not preserve that exact convention. After fixing the GEMM wrapper to match the CPU interpretation exactly, the full-model parity error collapsed from `0.398665` to `0.000002`.

That is a strong result for the Rust stack. It means the CUDA implementation is not “approximately right.” It is now reproducing the CPU reference to within a tolerance that is well below the required gate.

## What We Built This Week

This week’s work had two parallel goals:

1. make the architecture easier to iterate on under the 600-second training constraint
2. prove that the current GPU forward pass is correct

The first part was infrastructure in the positive sense: not deployment plumbing, but **research iteration infrastructure**.

I added a spec-driven control plane with:

- `ModelSpec`
- `TrainSpec`
- `QuantSpec`
- `EvalSpec`
- `RunSpec`
- `ExecutionPlan`

The point of this design is to keep the architecture fluid at the **variant** level, while freezing concrete tensor layouts and execution choices before runtime. In other words, the code can still compare bounded families of models, but the hot path does not interpret architecture decisions inside the inner loop.

I also defined a bounded variant family rather than a fully generic graph system. That was an important design correction. For this competition, the goal is not maximal framework generality; it is fast iteration on a small set of competitive candidates.

On top of that, I brought the GPU forward path much closer to the CPU reference. The forward pass now includes the full baseline sequence of operations:

- token embedding
- BigramHash path
- initial RMSNorm
- SmearGate
- residual mixing
- Q/K/V projections
- value embedding injection
- QK-norm
- partial RoPE
- q-gain
- naive causal attention for parity
- XSA
- output projection
- MLP up / activation / down
- U-Net skip connections
- final RMSNorm
- tied output logits

This matters because it means the parity result is not just for a partial model skeleton. It is for the actual baseline forward assembly.

## Metric Definitions

One piece of feedback on Week 1 was completely fair: I used terms like “attention MSE” without defining them clearly enough. That made it harder to interpret what the experiments actually meant. I want this week’s blog to be more precise.

Here are the metrics I am using now:

### 1. `max_abs_diff`

For parity checks, this is:

`max_i |cpu_i - gpu_i|`

where `cpu_i` and `gpu_i` are matching output elements, usually logits.

This is the primary correctness gate because it catches even one badly wrong value.

### 2. `mean_abs_diff`

For parity checks, this is:

`(1/N) * sum_i |cpu_i - gpu_i|`

This is a useful secondary metric because it tells me whether the error is spread across the output or concentrated in a few locations.

### 3. Reconstruction MSE

When I talk about quantization experiments, MSE means mean squared error between an original weight tensor and its reconstructed quantized approximation:

`(1/N) * sum_i (w_i - ŵ_i)^2`

This is a weight-space metric, not a task metric. It is useful for screening compression schemes quickly, but it does **not** directly prove lower BPB. That was the ambiguity in Week 1, and I should have stated it more explicitly.

### 4. BPB

BPB is **bits per byte**, the competition’s main evaluation metric. Lower is better. Ultimately, this is the metric that matters for ranking submissions. Reconstruction error and parity metrics are only proxy metrics or engineering gates on the way there.

## Clarifying the Pruning Discussion From Week 1

Another fair criticism was that my description of pruning was too vague. Here is the concrete version.

The pruning experiments from Week 1 used **structured 2:4 sparsity**, which means:

- weights are partitioned into groups of 4
- within each group, only the 2 weights with largest magnitude are kept
- the other 2 are set to zero

This is important because 2:4 is not just a compression trick. It is a hardware-aligned sparsity pattern supported efficiently on NVIDIA GPUs.

The comparison I reported last week was:

- **Path A:** prune first, then quantize
- **Path B:** quantize first, then prune

The evaluation metric there was reconstruction error on the resulting weight tensor. Path A consistently won, which supports the idea that the order of lossy transforms matters. But again, that was a **weight reconstruction** result, not yet a final BPB result on trained models.

That distinction is exactly why this week’s parity work matters so much. Before running broader sweeps on the Rust stack, I need confidence that model behavior differences come from the architecture or quantization choices I am testing, not from silent implementation bugs.

## Challenges and Roadblocks

The central challenge this week was not “getting CUDA to run.” That part is mostly solved. The real challenge was **establishing trust in the system**.

There are at least three ways a new training stack can fail:

- it can fail to execute
- it can execute but compute the wrong thing
- it can compute the right thing, but only under a narrow configuration

This week lived in the second category. The GPU model ran, but it disagreed with the CPU model. That is a dangerous state because it is easy to misread as progress. You can benchmark it, profile it, even start training with it, but none of those results are meaningful if the math is off.

The main roadblock was therefore methodological: how to localize a large model-level discrepancy quickly. The answer was to improve the validation hierarchy:

- first, a spec-level execution plan
- then kernel-by-kernel parity
- then full forward parity

That hierarchy worked. It turned an opaque model mismatch into a single concrete culprit.

This is the broader lesson of Week 2: in systems-heavy ML work, debugging speed depends less on cleverness than on having the right layers of instrumentation.

## Evidence of Progress

I think the progress this week is meaningful for two reasons.

First, it is a real milestone, not just code volume. The Rust stack now clears the forward-parity gate on the GPU. That was the hard stop before moving to backward pass work.

Second, the work compounds. The same harnesses that found the GEMM bug will be reusable for:

- backward-pass operator checks
- one-step training parity
- regression testing when I change architecture variants
- quantization validation after export

In other words, the output of this week is not just “the bug is fixed.” It is also a better process for catching the next bug faster.

## Plans for Next Week

The next priority is **one-step parity**.

The immediate goal is:

- run one deterministic training step on CPU and GPU
- compare loss
- compare a selected set of gradients
- expand from those key gradients to full backward parity

I want to stage that work in the same way as forward parity:

1. validate individual backward operators where possible
2. build a one-step parity harness for CPU vs GPU
3. compare output, tied embedding, and one block’s Q/K/V/O and MLP gradients first
4. only then move to full backward assembly

If that passes, the next gate is a short single-GPU training run where the GPU loss curve tracks the CPU curve over a few dozen to a few hundred steps.

Only after those gates are green does it make sense to spend real time on:

- multi-GPU training
- performance tuning
- quantization/export sweeps
- architecture comparisons under the 600-second budget

That ordering is much clearer to me now than it was in Week 1.

The GPU model was close enough to run, but still wrong. By tightening the metric definitions, improving the validation harnesses, and isolating the discrepancy to the GEMM path, I was able to convert “it runs” into “it matches.”

That feels like the real beginning of the Rust submission effort.
