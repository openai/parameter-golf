# BPB, Compression, and Formal Methods

## Purpose

This note tightens three parts of the project:

1. the mathematical meaning of `val_bpb`
2. the correct optimization framing for the `<16 MB` artifact constraint
3. whether invariant testing and formal verification are worth using here

The goal is not abstract rigor for its own sake. The goal is to reduce the probability of wasting leaderboard attempts on metric bugs, accounting mistakes, or export logic that is "probably right" but not defensible.

## 1. What `val_bpb` Actually Is

The right mental model is:

- the validation metric is an operational code length
- the model plus evaluator define a predictive code
- `val_bpb` is the normalized number of coded bits per raw validation byte

The repo computes:

`val_loss = loss_sum / token_count`

`val_bpb = (loss_sum / ln 2) / byte_count`

equivalently:

`val_bpb = val_loss / ln 2 * (token_count / byte_count)`

This matches the standard source-coding view:

- ideal arithmetic coding gives code length close to `sum_i -log_2 p(t_i | t_<i)`
- the repo stores token loss in natural logs, then divides by `ln 2`
- the final normalization is by raw byte count, not token count

So `val_bpb` is not just correlated with compression. It is a compression length under the evaluator's coding semantics.

## 2. Cross-Entropy / KL View Of BPB

Let:

- `q` be the true validation-stream distribution under the chosen evaluator semantics
- `p` be the model distribution

Then expected cross-entropy in bits per token is:

`H(q, p) = H(q) + D_KL(q || p)`

and expected bits per byte are:

`E[val_bpb(p)] = H(q, p) / E_q[bytes_per_token]`

Therefore:

`E[val_bpb(p)] = H(q) / E_q[bytes_per_token] + D_KL(q || p) / E_q[bytes_per_token]`

Interpretation:

- `H(q) / E_q[bytes_per_token]` is the irreducible entropy floor
- `D_KL(q || p) / E_q[bytes_per_token]` is the model's excess codelength per byte
- every real BPB improvement is a reduction in excess codelength

This gives the correct standard for evaluating ideas:

- better tokenization only matters if it improves the effective coding distribution or byte-normalized context use
- better evaluation only matters if it legally reduces cross-entropy per byte
- better training only matters if it survives export and helps final coded length

## 3. Acceptance Margin Math

The challenge acceptance rule is stated in natural-log units, but the public leaderboard talks in BPB.

Because the repo metric is:

`val_bpb = val_loss / ln 2 * (token_count / byte_count)`

the conversion depends on the actual `token_count / byte_count` ratio of the scored validation corpus and evaluator.

So:

`Delta_BPB = Delta_nats_per_token / ln 2 * (token_count / byte_count)`

This is why the practical frontier conversion is about `0.003 BPB` rather than `0.0072 BPB`: the byte normalization matters.

Operational rule:

- compute record margin from raw `loss_sum`, `token_count`, and `byte_count`
- do not trust rounded `val_bpb` table entries for acceptance math

## 4. Metric Lock Is Mandatory

The metric is only meaningful if the following stay fixed:

- validation shard set
- tokenizer model
- byte LUT logic
- boundary-token semantics
- accumulation order

In this repo, byte accounting is not trivial token length lookup. It depends on:

- `base_bytes_lut[target_token]`
- leading-space handling
- previous-token boundary status

That means metric bugs can come from:

- changing tokenization assets
- changing BOS/boundary interpretation
- averaging local BPB values instead of accumulating global raw sums
- silently altering evaluator partitioning semantics

The metric kernel must therefore be treated like frozen infrastructure.

## 5. There Are Two Compression Problems Here

People keep saying "compression" as if there is one optimization problem. There are two.

### 5.1 Validation-Stream Compression

This is the leaderboard objective:

`minimize code_length(validation | model, evaluator) / raw_validation_bytes`

This is lossless predictive compression.

### 5.2 Model-Artifact Compression

This is a feasibility constraint:

`compress model + metadata + code to < 16,000,000 bytes while preserving predictive quality`

This is lossy model compression plus lossless packaging.

The first problem decides rank.
The second problem decides whether the run is submittable.

## 6. Artifact Compression Must Be Framed As Rate-Distortion

Let tensor groups be `g = 1, ..., G`.

For each group, choose a representation `z_g` with:

- serialized byte cost `R_g(z_g)`
- induced score damage `D_g(z_g)`

Then the constrained problem is:

`minimize sum_g D_g(z_g)`

subject to:

`sum_g R_g(z_g) < 16,000,000`

The Lagrangian form is:

`minimize sum_g [D_g(z_g) + lambda * R_g(z_g)]`

At a good operating point:

`-Delta D_g / Delta R_g ~= lambda`

Interpretation:

- spend bytes where marginal score damage drops fastest
- cut bytes where score damage rises slowest
- stop using layer folklore as a substitute for marginal analysis

## 7. Real Byte Cost Means Post-Serialization Byte Cost

This repo does not submit "nominal bits per weight." It submits a serialized object that is then compressed again with `zlib`.

So:

`actual_bytes != param_count * nominal_bitwidth / 8`

because actual size depends on:

- quantized value entropy
- repeated patterns and runs
- scales and metadata
- container overhead
- downstream `zlib` behavior

Therefore the only trustworthy byte-cost measurement is:

- bytes of the actual exported artifact
- or delta-bytes from swapping one tensor group at a time in the actual export path

This is a major reason naive mixed-precision heuristics can be misleading.

## 8. Second-Order Math Is The Right Proxy, But Still A Proxy

Search-backed quantization literature supports a consistent hierarchy:

- OBC / OBQ: quadratic local loss with OBS-style updates
- GPTQ: approximate second-order quantization for GPT-like models
- HAWQ-V2: Hessian trace as a sensitivity metric for mixed precision
- APTQ: Hessian-trace plus attention-aware sensitivity for LLM mixed precision

That implies the right ranking of evidence for this project:

1. best: direct final-score damage under the real export and evaluator
2. next best: post-quant eval damage under a matched evaluator
3. fallback: Hessian-aware or output-reconstruction sensitivity
4. weak fallback: weight magnitude or hand-wavy layer importance

So yes, second-order methods are mathematically justified here.
No, they are not a substitute for actual leaderboard-metric measurement.

## 9. What Can Invariant Testing Do Here?

Yes, invariant testing is highly applicable here.

In fact, it is one of the highest-value things we can add, because the challenge has a narrow deterministic core:

- metric accumulation
- byte accounting
- document/window partitioning
- distributed aggregation
- quantize/dequantize/export roundtrip logic

These are exactly the kinds of places where property-based testing works well.

There is currently no test harness in this repo snapshot, which increases the value of adding one.

### 9.1 Best Targets For Invariant Testing

#### Metric Accumulation Invariants

- global `loss_sum` equals the sum of partition `loss_sum`s
- global `byte_count` equals the sum of partition `byte_count`s
- `val_bpb` recomputed from raw sums matches printed `val_bpb`
- re-batching or re-chunking does not change final raw sums

#### Coverage Invariants

- every target token is scored exactly once
- no target token is skipped
- no target token is double-counted
- doc-isolated evaluation never crosses a boundary

#### Boundary Invariants

- first scored token after a boundary never inherits leading-space logic from the previous document
- per-document token counts sum to the full validation total

#### Quantizer / Export Invariants

- quantized tensors stay in `[-127, 127]`
- scales are strictly positive
- dequantized tensors preserve original keys, shapes, and dtypes
- passthrough tensors round-trip exactly, modulo intentional storage dtype changes
- actual submission size equals measured artifact bytes plus code bytes

#### Distributed Invariants

- rank partitions are disjoint and exhaustive
- all-reduced raw sums equal the single-process reference on the same data

### 9.2 Why Hypothesis Fits

Hypothesis' property-based testing model is a strong fit because we can express:

- "for all synthetic token streams with random boundaries"
- "for all random tensor shapes and magnitudes"
- "for all valid chunk/stride/window settings"

and then check invariants instead of point examples.

Its stateful testing support is especially useful for:

- window schedulers
- cache reset behavior
- export / load / re-export sequences
- distributed partition bookkeeping models

## 10. What Can Formal Verification Do Here?

Yes, but only on the right scope.

Formal verification is useful here for:

- proving combinatorial coverage properties
- proving integer accounting identities
- proving small pure helper functions preserve invariants
- checking bounded symbolic equivalence of two schedulers on all small inputs

Formal verification is not a good primary tool here for:

- full transformer training dynamics
- end-to-end score superiority
- full GPU floating-point execution of the model
- proving the exported quantized model stays within a useful BPB bound on the real validation set

The searched literature on quantized-network verification shows that formal verification is possible for quantized nets, but mostly in settings aimed at robustness/equivalence questions and at much smaller or more structured verification workloads than this leaderboard stack.

The floating-point verification literature also warns that verification becomes unsound if floating-point effects are abstracted incorrectly. That matters here because:

- evaluation uses floating-point model inference
- quantization uses floating-point quantiles, clipping, and scaling
- GPU behavior and kernel fusion are not the right first target for proof obligations

So the right conclusion is:

- use formal methods on the deterministic accounting shell around the model
- do not try to formally verify the whole ML system

## 11. The Correct Hybrid Assurance Strategy

The right approach is not "formal verification everywhere."

It is:

1. unit tests for obvious local behavior
2. property-based invariant tests for combinatorial edge cases
3. differential tests against simpler reference implementations
4. bounded symbolic verification for pure helper logic
5. ordinary statistical experiments for model quality

That gives much better return than trying to prove end-to-end ML correctness.

## 12. Modular Verification Architecture

The verification story should be modular, not monolithic.

The key design principle is:

- isolate the deterministic accounting shell from the model runtime
- verify each shell component with the cheapest method that gives strong guarantees
- compose guarantees using explicit assume/guarantee contracts

### 12.1 Proposed Module Split

Recommended filesystem layout:

- `core/metric_core.py`
- `core/schedule_core.py`
- `core/partition_core.py`
- `core/quant_core.py`
- `core/artifact_core.py`
- `adapters/runtime_eval.py`
- `tests/`

The `core/` modules should be as pure as possible.
The `adapters/` layer is where PyTorch, CUDA, distributed collectives, and file I/O remain.

### 12.2 Module Responsibilities

#### `core/metric_core.py`

Own:

- token-to-byte accounting
- raw accumulator updates
- exact BPB reconstruction from raw sums

Candidate interfaces:

```python
def token_byte_count(prev_ids, tgt_ids, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut) -> int
def accumulate_loss_bytes(losses, prev_ids, tgt_ids, luts...) -> tuple[float, int, int]
def compute_val_bpb(loss_sum: float, byte_count: int) -> float
```

#### `core/schedule_core.py`

Own:

- document boundary handling
- score interval generation
- context interval generation
- stride / window / chunk legality

Candidate interfaces:

```python
def find_docs(tokens) -> list[tuple[int, int]]
def score_windows(doc_len: int, chunk_size: int, eval_seq_len: int) -> list[Window]
def validate_window_plan(windows: list[Window], doc_len: int) -> None
```

#### `core/partition_core.py`

Own:

- rank partition arithmetic
- disjointness and exhaustiveness rules
- mapping corpus work to processes

Candidate interfaces:

```python
def partition_range(total: int, rank: int, world_size: int) -> tuple[int, int]
def partition_docs(num_docs: int, rank: int, world_size: int) -> tuple[int, int]
```

#### `core/quant_core.py`

Own:

- tensor-wise quantization math
- scale generation
- metadata packing for quantized tensors
- dequantization shape/dtype restoration

Candidate interfaces:

```python
def quantize_float_tensor(t) -> tuple[q, scale]
def quantize_state_dict_int8(state_dict) -> tuple[obj, stats]
def dequantize_state_dict_int8(obj) -> state_dict
```

#### `core/artifact_core.py`

Own:

- artifact byte accounting
- package-size calculations
- submission-size composition

Candidate interfaces:

```python
def serialized_num_bytes(blob: bytes) -> int
def submission_size(model_bytes: int, code_bytes: int) -> int
def export_size_report(quant_stats, quant_raw_bytes, quant_file_bytes, code_bytes) -> dict
```

#### `adapters/runtime_eval.py`

Own:

- model forward passes
- distributed reductions
- file I/O
- actual export/load path

This module should be intentionally thin.
It should delegate arithmetic and bookkeeping to `core/`.

### 12.3 Pure vs Impure Boundary

The modular split matters because the verification methods depend on purity.

Good proof/test targets:

- integer accounting
- interval logic
- metadata transforms
- deterministic tensor-local quantization logic

Bad proof/test targets:

- GPU kernels
- distributed runtime details beyond coarse differential tests
- `zlib` internals
- whole-model floating-point behavior

### 12.4 Assume / Guarantee Contracts

Each module should advertise what it assumes and what it guarantees.

#### `schedule_core`

Assumes:

- valid document boundaries
- legal configuration parameters

Guarantees:

- score intervals are disjoint
- score intervals cover the target set exactly once
- context intervals stay within allowed bounds
- no interval crosses a document boundary in doc-isolated mode

#### `metric_core`

Assumes:

- the scheduler produced legal, non-overlapping scored regions
- inputs use the intended tokenizer/LUT semantics

Guarantees:

- raw sums are additive across partitions
- reconstructed `val_bpb` matches the raw-sum formula exactly
- byte counts match the token/space/boundary semantics

#### `partition_core`

Assumes:

- valid `rank`, `world_size`, and total-work inputs

Guarantees:

- partitions are disjoint
- partitions are exhaustive
- concatenating partition outputs reproduces the serial result

#### `quant_core`

Assumes:

- input tensors and dtypes are valid

Guarantees:

- quantized values respect the target numeric range
- scales are positive
- metadata is sufficient for dequantization
- key sets, shapes, and intended dtypes survive roundtrip

#### `artifact_core`

Assumes:

- export buffers and metadata are real outputs from the chosen export path

Guarantees:

- reported sizes match actual serialized byte counts
- total submission size is computed consistently

### 12.5 Why This Composition Is Strong

This structure gives a clean proof/testing stack:

- if `schedule_core` proves exact legal coverage
- and `metric_core` proves exact additive accounting
- and `partition_core` proves exact partition composition

then the full evaluator accounting shell is trustworthy modulo the runtime adapter and model forward values.

That is a meaningful guarantee.

It is also the highest-value guarantee available for this challenge.

## 13. What To Verify In This Repo Specifically

The best concrete extraction targets from `train_gpt.py` are:

- `eval_val`
- `_accumulate_bpb`
- `quantize_float_tensor`
- `quantize_state_dict_int8`
- `dequantize_state_dict_int8`

These should be split across the modules above.

If document-isolated sliding eval is added, then also extract:

- boundary detection helpers
- window generation / score-range helpers
- cache reset scheduling logic

The right migration path is:

1. leave `train_gpt.py` as the orchestration script
2. move deterministic helper logic into `core/`
3. keep only runtime orchestration in `adapters/runtime_eval.py`

## 14. Recommended Invariants

### 14.1 Metric Invariants

- `loss_sum >= 0`
- `byte_count > 0` on non-empty scored inputs
- `token_count > 0` on non-empty scored inputs
- `val_bpb == (loss_sum / ln 2) / byte_count`
- concatenating two independently scored disjoint chunks preserves additivity of raw sums

### 14.2 Scheduler Invariants

- score intervals are disjoint
- score intervals cover the intended target range exactly
- context interval always contains the scored interval
- context length never exceeds configured max length
- boundary resets happen exactly at document boundaries

### 14.3 Partition Invariants

- rank partitions are disjoint
- rank partitions are exhaustive
- partition union equals the full serial range
- empty-work edge cases are handled explicitly

### 14.4 Quantizer Invariants

- output `q.dtype == int8`
- `q.min() >= -127`, `q.max() <= 127`
- every scale is positive
- for unclipped elements, dequantization error is at most roughly half a quantization step
- key sets are preserved through quantize/dequantize roundtrip

### 14.5 Artifact Invariants

- serialized artifact byte count matches what is logged
- dequantized state dict loads into the original module shape-wise
- export followed by load followed by export is stable in format and metadata

## 15. What Is Worth Formally Proving

The best proof targets are the ones with finite symbolic structure.

### 15.1 Good Formal Targets

- score-range coverage for a scheduler
- no-overlap / no-gap properties on token intervals
- exact byte-count formulas on abstract token/space/boundary sequences
- disjointness and exhaustiveness of rank partitions
- preservation of key sets and tensor shape metadata through export helpers

### 15.2 Bad Formal Targets

- the full training loop
- CUDA kernel behavior
- `zlib` compression ratio claims
- final leaderboard win probability
- full-network floating-point equivalence after quantization

Formal verification of quantized neural networks exists in the literature, but that does not mean it is the right optimization for this repo. Here it is mostly too expensive, too indirect, and too far from the actual failure modes we care about.

## 16. Tooling Recommendation

### 16.1 Add Property-Based Testing

Add dev dependencies:

- `pytest`
- `hypothesis`

Use them for:

- random token-stream generators with boundary markers
- random tensor generators for quantization checks
- stateful testing for window schedulers and export roundtrips

### 16.2 Add Lightweight Symbolic Checking

Use:

- `CrossHair` for assert-based contracts on pure Python helpers
- `Z3` for small integer / interval / partition proofs when the property is cleaner in SMT than in Python

This only becomes worthwhile after the logic is refactored into small pure functions.

### 16.3 Keep Proof Scope Narrow

Use formal methods on:

- integer arithmetic
- interval arithmetic
- small data-structure invariants

Do not try to encode:

- PyTorch GPU execution
- quantile-based floating-point numerics of the full export path
- end-to-end transformer correctness

### 16.4 Verification-Method Mapping

Use the weakest tool that can convincingly establish the property.

- `metric_core`: unit tests + Hypothesis + differential tests
- `schedule_core`: Hypothesis + CrossHair/Z3
- `partition_core`: Hypothesis + Z3
- `quant_core`: unit tests + Hypothesis
- `artifact_core`: unit tests + differential tests against real serialized outputs
- `runtime_adapter`: integration tests only

## 17. Step-By-Step Rollout

### Phase 0: Extract Verifiable Kernels

Refactor helper logic out of `train_gpt.py` into `core/` modules:

- token byte accounting
- boundary-aware score interval generation
- document partitioning
- rank partitioning
- quantizer metadata packing / unpacking

Without this step, the testing and formal methods story stays weak.

### Phase 1: Build Deterministic Reference Implementations

Write slow reference versions for:

- byte counting
- score coverage
- document-isolated scheduler behavior
- quantize/dequantize shape and metadata handling

These become the oracle for differential tests.

### Phase 2: Add Property-Based Invariant Tests

Use Hypothesis to generate:

- token streams with random document boundaries
- random chunk/stride/length settings
- random float tensors with edge-case magnitudes

Check the invariants listed above.

This is likely the highest-ROI reliability work in the whole formal-methods stack.

### Phase 3: Add Stateful Tests

Use Hypothesis state machines for:

- scheduler progression across windows
- cache reset and document transitions
- export -> load -> export sequences

This catches sequencing bugs that example-based tests usually miss.

### Phase 4: Add Bounded Symbolic Verification

Run CrossHair or Z3 on:

- interval coverage and no-overlap properties
- rank partition arithmetic
- boundary-aware byte-count identities

Keep the domains small and exact.

### Phase 5: Add Differential Runtime Checks

For a small set of frozen toy inputs:

- compare single-process vs distributed raw sums
- compare baseline evaluator vs alternate scheduler on cases where they should agree
- compare export roundtrip loads against pre-export shape/dtype expectations

This bridges the pure proofs to the real runtime path.

### Phase 6: Use Statistical Validation For The Rest

Once invariants and accounting are locked:

- use ordinary repeated-run statistics for score changes
- do not attempt to prove leaderboard improvements formally

That part is an empirical question, not a proof obligation.

## 18. Final Recommendation

Yes, we should use invariant testing here.

Yes, we should use limited formal verification here.

Yes, the right implementation shape is modular verification with explicit module contracts.

But the correct target is the metric/export/scheduler shell around the model, not the full model itself.

The highest-value stack for this repo is:

- property-based tests for metric and export invariants
- stateful tests for schedulers and roundtrips
- differential tests against simple references
- symbolic checking for small pure helper functions

Trying to formally verify the full transformer or full quantized inference path would be the wrong use of effort for this challenge.

## Sources

Search-backed references used for the reasoning in this note:

- Shannon, *A Mathematical Theory of Communication*:
  https://www.mpi.nl/publications/item2383162/mathematical-theory-communication

- Delétang et al., *Language Modeling Is Compression*:
  https://arxiv.org/abs/2309.10668

- Moffat, Neal, Witten, *Arithmetic Coding Revisited*:
  https://researchcommons.waikato.ac.nz/items/ef7c7d25-0857-448f-a02e-0895747df2bc

- Frantar et al., *Optimal Brain Compression*:
  https://arxiv.org/abs/2208.11580

- Frantar et al., *GPTQ*:
  https://arxiv.org/abs/2210.17323

- Dong et al., *HAWQ-V2*:
  https://arxiv.org/abs/1911.03852

- Guan et al., *APTQ*:
  https://arxiv.org/abs/2402.14866

- Huang et al., *Towards Efficient Verification of Quantized Neural Networks*:
  https://arxiv.org/abs/2312.12679

- Jia and Rinard, *Exploiting Verified Neural Networks via Floating Point Numerical Error*:
  https://arxiv.org/abs/2003.03021

- Hypothesis documentation:
  https://hypothesis.readthedocs.io/en/latest/reference/api.html

- CrossHair documentation:
  https://crosshair.readthedocs.io/en/stable/kinds_of_contracts.html

- Z3 Guide:
  https://microsoft.github.io/z3guide/

 Try the distillation approach, try Universal Transformer, write your own torch.save, try random linear maps

  - Staying custom is the right direction.
      - PyTorch’s own docs say torch.save is an uncompressed ZIP64
        archive with pickled metadata and separate storage files, which
        matches why our wins over it were small.
      - safetensors is the cleaner precedent: tiny header, contiguous
        byte buffer, no holes, packed tensors. Its docs explicitly say
        header space is tiny compared with tensor data.
      - So: do not go back to real torch.save. Keep parameter-golf/core/
        artifact_core.py-style packed artifacts.
  - The next quick artifact benchmark should be packed + zstd, not more
    header work.
      - The official zstd repo says it targets zlib-like use cases with
        better compression ratios.
      - Since packages are allowed, zstandard is fair game if listed in
        requirements.
      - I would benchmark:
          - packed + zlib
          - packed + zstd
      - I would not spend time on zstd dictionaries: the official docs
        say dictionary gains are mostly effective in the first few KB,
        while our artifact is multi-MB.
  - The most plausible serializer-side prefilter is bitshuffle, not more
    scale coding.
      - The official Bitshuffle repo says it is specifically for typed
        binary data and can improve compression when adjacent values are
        correlated or only a subset of bits are exercised.
      - This is tailored to our case:
          - scales are too small to matter
          - payload is numeric arrays
          - int8/fp16 sections dominate
      - Important nuance:
          - byte-shuffle is useless for 1-byte int8
          - bitshuffle is the relevant filter for int8
          - fp16 sections can use byte-shuffle or bitshuffle
      - So the right experiment is blockwise bitshuffle + zstd on the
        quantized payload section, not global metadata tweaks.
  - The bigger lever is still payload quality, not payload packaging.
      - QuaRot shows rotations can remove outliers without changing
        full-precision outputs, making quantization easier.
      - SpinQuant says not all rotations are equal, and learned
        rotations can beat random ones.
      - Tailored to this challenge, my inference is:
          - prefer structured or implicit rotations first, like
            Hadamard/sign/permutation-style transforms
          - avoid storing dense learned rotation matrices unless they
            clearly pay for themselves, because artifact bytes matter
            here
      - This fits your current results: once the container is “good
        enough,” the next real gain should come from making the
        quantized payload easier to compress accurately, not from
        shrinking JSON or scales.
  - If artifact-side work goes beyond rotations, the next real family is
    codebook/vector quantization.
      - VPTQ explicitly frames extreme compression as vectors compressed
        into indices plus lookup tables.
      - That is a real artifact-compression direction, but it is much
        more invasive than the current path.
      - I would treat it as a later branch, not the next step.

      What I would do next, in order:

  1. Keep packed_zlib as the current default baseline.
  2. Add a codec backend and benchmark packed + zstd on the current
     stronger checkpoint.
  3. Add an optional prefilter backend:
      - none
      - bitshuffle for int8 payload
      - maybe byte_shuffle/bitshuffle for fp16 passthrough blocks
  4. Keep invariants strict:
      - deterministic bytes
      - exact roundtrip
      - no overlapping offsets / no holes
      - strict load_state_dict
      - same eval helper comparison on the same checkpoint
  5. After that, move to structured rotation experiments on the dominant
     payload tensors.
  6. Only if that stalls, consider a more invasive codebook/VQ branch.

  - keep the custom artifact path
  - keep packed_zlib as the baseline container
  - stop spending time on scale codecs and generic save-format tweaks
  - keep tok_emb.weight as the current mixed-precision exception
  - treat the next real frontier as low-byte quantization
    preconditioning on the remaining dominant quantized tensors

  The search-backed conclusion is:

  - the only remaining packaging experiments worth a short pass are
    packed + zstd and maybe packed + bitshuffle + zstd
  - after that, the next serious branch should be structured rotations/
    preconditioning, starting with Hadamard/sign/permutation-style
    transforms
  - do not start with dense learned rotations, because their metadata
    cost fights the artifact budget directly

  The plan also gives:

  - exact experiment phases
  - pass/fail gates
  - transform and artifact invariants
  - what not to do next

  Main sources used:

  - PyTorch serialization docs:
    https://docs.pytorch.org/docs/stable/notes/serialization.html
  - torch.save docs:
    https://docs.pytorch.org/docs/stable/generated/torch.save.html
  - Safetensors docs: https://huggingface.co/docs/safetensors/en/index
  - Zstandard repo/docs: https://github.com/facebook/zstd
  - Bitshuffle repo: https://github.com/kiyo-masui/bitshuffle
  - QuaRot: https://arxiv.org/abs/2404.00456
  - SpinQuant: https://arxiv.org/abs/2405.16406
  - QuIP#: https://arxiv.org/abs/2402.04396
  - QTIP repo: https://github.com/Cornell-RelaxML/qtip
  - fast-hadamard-transform:
    https://github.com/Dao-AILab/fast-hadamard-transform