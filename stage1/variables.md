# Stage 1 Variables

## Task Snapshot

- Target: Parameter Golf 10-minute track
- Primary metric: final round-tripped `val_bpb`
- Secondary metrics:
  - pre-quant `val_bpb`
  - step time under wallclock
  - total effective steps reached before `600s`
  - total artifact bytes
  - pre/post-quant degradation
- Baseline score: `1.22436570` round-tripped `val_bpb`
- Baseline holdout / longer-run anchor: `1.20737944` in the 4-hour non-record run
- Hardware:
  - live final target: `8xH100`
  - current local ranking context: A100 experiments are already running
  - early experiments per upstream note: smaller shard prefixes, cheaper hardware, `1xH100` when possible
- Framework / language: PyTorch + CUDA in `train_gpt.py`
- Evaluation command: final path is the normal `torchrun --standalone --nproc_per_node=8 train_gpt.py`
- Evaluator noise / variance: unknown enough that early runs are ranking signals, not proof
- Dominant workload shape:
  - 9-layer, 512-dim GPT
  - 1024 vocab baseline
  - tied embeddings
  - `TRAIN_BATCH_TOKENS=524288`
  - `TRAIN_SEQ_LEN=1024`
- Hard constraints:
  - training under `10 minutes`
  - total artifact under `16,000,000` bytes
  - no evaluation-time training-data access unless paid for in bytes
- Soft preferences:
  - compact code diffs
  - mechanisms that AlphaEvolve can mutate further
  - changes that preserve reproducibility and simple submission structure

## Immutable Constraints

- Semantic invariants that must never break:
  - causal language modeling on the published FineWeb export
  - BPB calculation remains correct
  - artifact must round-trip and evaluate correctly
- Numerical tolerances:
  - no unstable NaNs / divergence
  - no fake wins from mismatched tokenizer or byte accounting
- API / ABI constraints:
  - submission must still run from a self-contained `train_gpt.py`
- Memory limits:
  - practical GPU memory is not the main constraint on H100, but memory-heavy ideas that slow kernels still matter
- Latency limits:
  - effective per-step time is a first-class constraint
- Compile-time limits:
  - warmup + `torch.compile` overhead is part of the actual 10-minute budget
- Output-format or evaluator-format constraints:
  - final score is computed after quantization and zlib compression

## Optimization Questions To Resolve Up Front

These are the questions the Stage 1 slate should answer.

1. Is the main bottleneck parameter allocation or optimizer bootstrap?
2. Is the win surface dominated by more effective steps, or by better learning per step?
3. Is the current post-quantization loss mostly a quantizer problem or a model-shape problem?
4. Is `TRAIN_SEQ_LEN=1024` helping enough to justify its throughput cost?
5. Are tied embeddings still optimal once tokenizer and vocab become editable?
6. Are the float-preserved control tensors worth their byte cost?
7. Is Muon compute itself too expensive relative to the quality it buys in a 600-second regime?
8. Does the frontier live in architecture, compression, or tokenizer space rather than optimizer space?
9. How much of the current public frontier comes from evaluation-time context rather than training?
10. Which public record families transfer cleanly from A100 scouting to `8xH100`?

## Bayesian Learning Strategy

We should treat each mechanism family as a latent variable with a prior probability of producing a track-legal gain.

### Families

- architecture / parameter allocation
- optimizer bootstrap and schedule
- throughput / token economics
- compression / artifact
- tokenizer / vocabulary
- risky structural compression-first moves

### Posterior Update Signals

Each experiment should be judged on the vector:

- `delta_final_bpb`
- `delta_prequant_bpb`
- `delta_step_avg`
- `delta_steps_before_cap`
- `delta_artifact_bytes`
- `stability`

### Update Rules

- Promote a family when it improves final `val_bpb` without breaking wallclock or bytes.
- Downweight a family if it improves pre-quant loss but loses after quantization.
- Retire a family after:
  - one fatal signal: unstable training, artifact blow-up, or obvious throughput collapse
  - or two independent negative aligned runs
- Prefer the next experiment that maximizes expected information gain over unresolved top-level questions, not just expected score.

### Stage 1 Decision Policy

- early runs are for posterior sharpening,
- final 8xH100 runs are only for families that still have posterior mass after cheap falsification,
- do not let easy retunes consume all portfolio slots if they teach less than a structural scout.

## Known Bottlenecks

- suspected bottleneck: wallclock-limited step budget
  - evidence: baseline stops at `13780/20000` because of the 600s cap
  - confidence: high

- suspected bottleneck: artifact headroom is nearly exhausted
  - evidence: baseline leaves only about `136 KB` slack
  - confidence: high

- suspected bottleneck: quantization quality matters more as the model gets better
  - evidence: the 4-hour run improves pre-quant strongly, but its post-quant gap is much larger than the 10-minute baseline gap
  - confidence: high

- suspected bottleneck: embeddings and tokenizer economics dominate small-model parameter budgets
  - evidence: tied embeddings are used and vocab is currently fixed at 1024
  - confidence: medium

## Unknowns That Need Evidence

- unknown: whether architecture reallocation beats optimizer tuning under the 10-minute cap
  - why it matters: this changes the whole search hierarchy
  - cheapest way to resolve it: one strong architecture scout vs one strong optimizer scout on aligned short runs

- unknown: whether shorter sequence length wins by allowing more steps
  - why it matters: this can dominate the entire short-horizon objective
  - cheapest way to resolve it: a matched `TRAIN_SEQ_LEN` sweep while holding artifact fixed

- unknown: whether quantization is leaving easy wins on the table
  - why it matters: final score is round-tripped, not raw
  - cheapest way to resolve it: shrink passthrough set and alter clipping/scaling scheme

- unknown: whether tokenizer changes are worth the complexity
  - why it matters: vocabulary changes alter both BPB and embedding budget
  - cheapest way to resolve it: one vocab-size scout before full tokenizer-family branching

## Ignored Variables Worth Questioning

- variable humans usually ignore: the byte cost of control tensors that remain float-preserved
  - why it might matter: small vector controls are intentionally kept in float, and the artifact budget is almost full
  - possible mechanism: replace per-dimension controls with cheaper per-block controls
  - cheapest code-level test: compress or simplify `attn_scale`, `mlp_scale`, `resid_mix`, or `skip_weights`

- variable humans usually ignore: the pre/post-quantization gap
  - why it might matter: a model that wins before export can still lose after export
  - possible mechanism: choose architectures and quantizers that are intrinsically more compressible
  - cheapest code-level test: compare quantizer changes against a matched model baseline

## What Is Allowed To Change

- files:
  - `pgolf/parameter-golf/train_gpt.py`
  - `pgolf/parameter-golf/data/tokenizer_specs.json`
  - `pgolf/parameter-golf/data/download_hf_docs_and_tokenize.py`
- regions / evolve blocks:
  - model architecture
  - optimizer split and update rules
  - quantization path
  - schedule logic
  - tokenizer/data export path
- launch config:
  - batch size
  - sequence length
  - vocabulary size
  - model width/depth/head counts
- evaluator knobs:
  - only when still track-legal and semantically aligned

## What Must Not Change

- externally visible behavior:
  - valid language-model training and evaluation
- interfaces:
  - submission must remain reproducible from the record folder
- benchmarking protocol:
  - fixed validation split semantics
- reference outputs:
  - BPB computation must remain correct

## Benchmark Slices

| Slice | Why it matters | Current baseline | Notes |
| --- | --- | --- | --- |
| Cheap scout | Early family elimination | not yet run | use small shard prefix or cheaper hardware |
| Aligned short run | Real Stage 1 promotion signal | baseline `1.2244` | must preserve final metric semantics |
| Pre-quant vs post-quant | Detect fake model-only wins | baseline gap is modest, 4-hour gap is larger | this is a core slice |
| Throughput-limited slice | Tests steps-before-cap sensitivity | `43.54 ms/step` baseline | small slowdowns can erase gains |
| Tokenizer/vocab holdout | High-complexity but high-upside surface | only `sp1024` exists now | needs careful validation |

## Space Coverage Check

| Area | Coverage | Notes |
| --- | --- | --- |
| Algorithmic structure | lightly explored | architecture hypotheses exist but not yet executed |
| Memory / data movement | lightly explored | mostly via throughput and quantization proxies |
| Parallelism / scheduling | lightly explored | sequence length and Muon compute are the main handles |
| Control flow | untouched | little explicit branching structure today |
| Numerical strategy | lightly explored | quantizer and optimizer denominator surfaces are open |
| Specialization / dispatch | untouched | little specialization in current CUDA path |
| Build / compiler tactics | lightly explored | warmup and compile overhead matter but are not yet targeted directly |

## Attack Surface Matrix

| Surface ID | Region | Bottleneck | Change Class | Mechanism | Observability | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S01 | model depth/width | parameter allocation | structural | reallocate params under same byte budget | high | unexplored | likely high upside |
| S02 | KV heads | attention state size | structural | more aggressive GQA / MQA | high | unexplored | direct parameter + compute lever |
| S03 | control tensors | byte overhead + compressibility | structural | cheaper control parameterization | medium | unexplored | many miss this |
| S04 | Muon compute | step time vs quality | schedule | fewer or adaptive orthogonalization steps | high | unexplored | speed-quality trade |
| S05 | sequence length | throughput vs context | policy | change tokens-per-step geometry | high | unexplored | likely decisive in 600s |
| S06 | quantizer passthrough | artifact bytes | policy | lower keep-float budget | high | unexplored | direct byte lever |
| S07 | quantizer clipping/scales | post-quant loss | numerical | alter clipping percentile / scheme | high | unexplored | quality lever |
| S08 | embedding optimizer | early bootstrap | schedule | denominator / epsilon / beta rules | medium | unexplored | embeddings matter a lot here |
| S09 | tokenizer/vocab | BPB + embedding burden | structural | change token inventory | medium | unexplored | high upside, higher risk |
| S10 | layer sharing | parameter reuse | structural | ALBERT-like tying / recurrence | medium | unexplored | abstract transfer from small-model design |

## Evidence Notes

- profiler observations: none yet, but step budget data already says throughput is central
- evaluator caveats: final score is round-tripped, so pre-quant wins can mislead
- suspected fake-win patterns:
  - faster training that loses after export
  - lower train loss that does not improve `val_bpb`
  - larger code or float passthrough sets that silently break the artifact budget
- places where a static knob may need to become an adaptive mechanism:
  - Muon backend steps
  - sequence length or batch geometry
  - quantizer clipping
  - per-group optimizer rules
