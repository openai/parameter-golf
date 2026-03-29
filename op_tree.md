# Parameter Golf Optimization Tree

This is the working mental model for the challenge.

You are not optimizing only a model. You are optimizing a pipeline:

`training recipe + architecture + tokenizer + evaluation method + artifact compression`

The final judged object is the submitted artifact after roundtrip reload and evaluation.

## 1. Top-Level Objective

Maximize:

- final roundtrip quality on FineWeb validation
- measured as `val_bpb`

Subject to:

- total artifact size `< 16,000,000` bytes
- training runtime `< 10 minutes` on `8xH100` for record runs
- evaluation runtime `< 10 minutes` on `8xH100`
- no invalid validation leakage

---

## 2. Main Optimization Branches

### A. Capability

Question:
- how good is the model before compression?

Main levers:
- architecture
- tokenizer / vocabulary
- model depth / width / head layout
- recurrence / parameter sharing
- loss head design
- optimizer and LR schedule
- data order / sequence length plan

Primary effect:
- lowers pre-compression validation loss

Main risk:
- you build a strong model that compresses badly or trains too slowly

### B. Compressibility

Question:
- how well does the trained model survive artifact compression?

Main levers:
- quantization policy
- mixed precision by tensor family
- QAT / export-aware warmdown
- clipping / scaling
- serialization layout
- entropy coding / zlib friendliness
- keeping some tensors in float

Primary effect:
- reduces artifact bytes while preserving score

Main risk:
- the model looks good before export and collapses after roundtrip

### C. Evaluation Score Boosts

Question:
- how much extra `val_bpb` can we gain at evaluation time legally?

Main levers:
- sliding window eval
- longer eval context
- legal score-first TTT
- LoRA-style chunkwise updates
- context reuse

Primary effect:
- improves final judged score without needing a better training model

Main risk:
- evaluation becomes too slow or illegal

### D. Runtime / Systems Efficiency

Question:
- how much useful learning or evaluation work can we fit inside the wallclock?

Main levers:
- FlashAttention-friendly design
- compile/warmup strategy
- sequence length schedule
- kernel-friendly tensor shapes
- avoiding slow exotic ops
- keeping update/eval loops coarse enough for GPU utilization

Primary effect:
- more useful steps inside the same 10-minute cap

Main risk:
- a theoretically better idea loses because it drops off the fast path

---

## 3. Interaction Map

These branches are not independent.

### Capability -> Compressibility

- wider or deeper models may gain accuracy but become harder to quantize
- recurrence can save bytes but may amplify quantization error
- tokenizer changes can improve `bpb` but increase embedding cost

### Capability -> Runtime

- better architecture may be slower per step
- long context may help quality but reduce total steps
- exotic blocks may break optimized attention kernels

### Compressibility -> Capability

- export-aware training can improve post-roundtrip quality
- low-bit-friendly architectures may outperform nominally stronger fragile ones

### Evaluation -> Runtime

- sliding eval and TTT can improve score
- they also spend evaluation wallclock

### Evaluation -> Legality

- score-first adaptation is allowed
- training on future unevaluated validation tokens is not allowed

---

## 4. What We Are Really Optimizing

A useful decomposition is:

### Stage 1: Train a strong base model

Optimize:
- architecture
- schedule
- tokenizer
- training throughput

Metric:
- pre-export `val_bpb`

### Stage 2: Make the model survive export

Optimize:
- quantization scheme
- mixed precision allocation
- serialization format
- export-aware tuning

Metric:
- post-roundtrip `val_bpb`
- artifact bytes

### Stage 3: Improve judged score at eval time

Optimize:
- legal eval tricks
- context handling
- tiny adaptation methods

Metric:
- final judged `val_bpb`
- eval wallclock

---

## 5. Practical Decision Tree

### Step 1

Ask:
- is the current bottleneck model quality, artifact size, or runtime?

If model quality is weak:
- work on architecture, schedule, tokenizer, or depth/width allocation

If artifact size is weak:
- work on quantization, mixed export, embedding budget, or serializer layout

If eval score is weak:
- work on sliding eval, longer context, or legal adaptation

If runtime is weak:
- work on sequence schedule, warmup, kernel-friendly design, or simpler blocks

### Step 2

Ask:
- does the idea help before export, after export, or only during eval?

If before export only:
- it is incomplete

If after export:
- it is more challenge-aligned

If only during eval:
- check legality and eval wallclock immediately

### Step 3

Ask:
- is the idea first-order or second-order?

First-order ideas:
- mixed export
- schedule tuning
- embedding budget changes
- H100-friendly architecture

Second-order ideas:
- learned fast weights
- full native BitNet path
- fancy vector quantizers with metadata-heavy decoders

Prototype first-order ideas first.

---

## 6. Current Best Working Heuristic

For this repo, the current best heuristic is:

1. build a strong dense baseline that stays on the fast path
2. improve artifact compression in a sensitivity-aware way
3. add only small legal eval-time gains
4. delay moonshots until the boring path is strong

In short:

`capability x compressibility x legal eval gain x runtime efficiency`

not:

`capability alone`

---

## 7. Immediate Experiment Mapping

### Baseline

Use for:
- reference capability
- reference artifact size
- reference roundtrip loss

### Experiment 01: Mixed Export

Targets:
- compressibility
- post-roundtrip robustness

### Experiment 02: Factored Embeddings

Targets:
- capability-per-byte
- embedding budget efficiency

### Future Optuna Use

Best for:
- local proxy sweeps on a few continuous or small categorical knobs

Good Optuna surfaces:
- clip percentiles
- warmdown iters
- factorized embed dim
- learning-rate splits
- small export thresholds

Bad Optuna surfaces:
- giant architecture spaces
- tokenizer redesign
- unrestricted seed brute force
- full leaderboard-budget search
