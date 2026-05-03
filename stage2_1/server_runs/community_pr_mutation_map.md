# Community PR Mutation Map

This document translates the public PR survey into a mutation plan for `pgolf/parameter-golf` specifically.

The question is not just "does the technique look strong in public records?"

It is:

- can we mutate it in this codebase quickly,
- does the current root `train_gpt.py` already expose it,
- is there an existing record implementation we can port,
- and what observable should decide whether it survives?

References:

- [community_pr_survey.md]( nanoevolve/pgolf/parameter-golf/stage2/community_pr_survey.md)
- [train_gpt.py]( nanoevolve/pgolf/parameter-golf/train_gpt.py)
- [experiments.md]( nanoevolve/pgolf/parameter-golf/stage2/experiments.md)

## Read This First

The current root script already exposes some important mutation hooks:

- model size and shape: `VOCAB_SIZE`, `NUM_LAYERS`, `MODEL_DIM`, `MLP_MULT` in [train_gpt.py#L63]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L63)
- optimizer knobs: `MATRIX_LR`, `MUON_MOMENTUM`, warmup, `BETA1`, `BETA2`, `ADAM_EPS`, `GRAD_CLIP_NORM` in [train_gpt.py#L73]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L73)
- export knobs: int8 quantizer, keep-float rules, per-row scales in [train_gpt.py#L287]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L287)
- LoRA TTT is already implemented in [train_gpt.py#L746]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L746)

The current root script still does **not** expose several public-record surfaces directly:

- sliding-window eval
- int6 or mixed-bit export
- zstd export
- fp16 embedding passthrough as a first-class export rule
- late-K passthrough
- SmearGate
- BigramHash
- SWA
- QAT
- MTP

Those are still fair mutation targets. They just need patches, not env-only sweeps.

## Technique Map

### Training Dynamics

`NorMuon`
- Impact expectation: medium, around the survey's `0.005-0.01` range.
- Current status: not present in root `train_gpt.py`; current Muon is a simple momentum + Newton-Schulz path in [train_gpt.py#L119]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L119).
- Mutation class: medium patch.
- Why it is worth testing: public recurrence across multiple strong records makes this a real family, not a one-off.
- Validate on: lower post-quant gap or better final round-tripped BPB at similar throughput.
- Kill if: step time rises materially and final round-tripped BPB does not improve.

`Muon Weight Decay`
- Impact expectation: medium-high.
- Current status: not present in root Muon; root Muon update is pure `p.add_(g, alpha=-lr)` in [train_gpt.py#L170]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L170).
- Mutation class: easy-medium patch.
- Best mutation shape: add Muon-only weight decay with schedule and compare `0.02`, `0.03`, `0.04`.
- Validate on: smaller post-quant gap and stable or better final BPB.
- Kill if: pre-quant improves but quantized score does not, or training loses too much capacity.

`SWA`
- Impact expectation: small-medium, likely more about export robustness than raw pre-quant loss.
- Current status: absent.
- Mutation class: easy patch.
- Best mutation shape: save checkpoints every `50-200` steps during warmdown, average weights before export only.
- Validate on: better round-tripped BPB than the same trained checkpoint without SWA.
- Kill if: final score is flat and bytes or runtime overhead are annoying.

`FlashAttention 3`
- Impact expectation: mostly throughput.
- Current status: root uses `scaled_dot_product_attention` with flash backend enabled in [train_gpt.py#L604]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L604) and [train_gpt.py#L994]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L994).
- Mutation class: probably low priority in root.
- Reason: the public gain may already be mostly captured by PyTorch SDPA on H100.
- Validate on: lower `ms/step` and more steps reached before wallclock cap.
- Kill if: throughput is unchanged.

`OrthoInit + muP`
- Impact expectation: small-medium.
- Current status: root init is minimal and does not do orthogonal init or muP-style output scaling in [train_gpt.py#L706]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L706).
- Mutation class: easy-medium patch.
- Why it is attractive: cheap to implement and low artifact risk.
- Validate on: better early curve and better final score with no byte penalty.
- Kill if: early gain disappears by export time.

`MTP`
- Impact expectation: small but real if the auxiliary signal helps in the short training budget.
- Current status: absent.
- Mutation class: medium patch.
- Why it is attractive: training-only improvement, no artifact cost if excluded from export.
- Validate on: better pre-quant learning curve and equal or better post-quant final BPB.
- Kill if: training speed or code complexity grows too much for a tiny gain.

`Grad Clip 0.3`
- Impact expectation: small, but very cheap.
- Current status: already exposed as `GRAD_CLIP_NORM` in [train_gpt.py#L87]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L87) and applied in [train_gpt.py#L1262]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L1262).
- Mutation class: env-only.
- Validate on: stabler training and better final score on longer-seq or deeper branches.
- Kill if: no curve change.

`Momentum warmup`
- Impact expectation: already known-live.
- Current status: already exposed in [train_gpt.py#L82]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L82) and used in [train_gpt.py#L1253]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L1253).
- Mutation class: env-only.
- Best use: treat as a tuning surface, not a new family.

`Higher LR`
- Impact expectation: small unless paired with another family.
- Current status: already exposed through optimizer env vars.
- Mutation class: env-only.
- Best use: child mutations on top of a stronger structural/export stack.

### Architecture

`10-11 layers`
- Impact expectation: medium-high if export budget is recovered elsewhere.
- Current status: already mutable via `NUM_LAYERS` in [train_gpt.py#L64]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L64).
- Mutation class: env-only, but only meaningful on an export-aware branch.
- Validate on: better final BPB despite fewer steps.
- Kill if: naked depth still loses after export improvements are present.

`SmearGate`
- Impact expectation: medium.
- Current status: absent.
- Mutation class: medium patch.
- Why it is attractive: tiny parameter cost, repeated appearance in top public stacks.
- Validate on: better final BPB at nearly unchanged bytes and throughput.
- Kill if: effect vanishes without BigramHash.

`BigramHash`
- Impact expectation: medium.
- Current status: absent, but the data tooling already knows about `recommended_bigram_vocab_size`, which suggests a clean path exists for adding it.
- Mutation class: medium patch.
- Why it is attractive: strong public recurrence and plausible complement to tied embeddings.
- Validate on: better BPB with acceptable parameter/byte cost.
- Kill if: artifact bloat cancels the score gain.

`U-Net skip connections`
- Impact expectation: already partly absorbed.
- Current status: root model already has encoder/decoder skip reuse via `skip_weights` in [train_gpt.py#L683]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L683) and [train_gpt.py#L727]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L727).
- Mutation class: low priority as a new family.
- Better mutation: change skip initialization or skip weighting policy rather than "add U-Net".

`2048/4096 vocab`
- Impact expectation: mixed, as the survey says.
- Current status: root supports `VOCAB_SIZE`, but a real run also needs the matching tokenizer + dataset shards.
- Mutation class: env-plus-data branch.
- Why it is worth keeping alive: vocab is one of the biggest parameter-budget levers.
- Validate on: better post-quant BPB after matching bytes and train budget.
- Kill if: tokenizer changes dominate attribution and do not beat export/eval improvements.

`Depth recurrence`
- Impact expectation: unclear but potentially large if depth is the real bottleneck.
- Current status: absent.
- Mutation class: medium-high patch.
- Best use: later wildcard, not early slot.

`SwiGLU`
- Impact expectation: low prior because the survey marks it neutral.
- Current status: root uses `relu^2` in [train_gpt.py#L616]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L616).
- Mutation class: easy patch.
- Best use: only as a cheap control if an MLP-family branch is already open.

### Export / Quantization

`STE Int6 QAT`
- Impact expectation: medium if export is currently the bottleneck.
- Current status: absent in root; root export is post-training int8 only in [train_gpt.py#L349]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L349).
- Mutation class: medium patch.
- Why it is attractive: direct attack on post-quant gap.
- Validate on: quantized score improves more than pre-quant score moves.
- Kill if: training slows or destabilizes with no export win.

`Mixed int5/int6`
- Impact expectation: medium-high, mostly because it can fund depth.
- Current status: absent in root, but conceptually adjacent to the record script's int6 path.
- Mutation class: medium-high patch.
- Best use: on a record-like export branch, not on the clean trunk first.
- Validate on: enough byte savings to buy a better architecture without worsening final BPB.
- Kill if: extra complexity does not buy a better architecture frontier point.

`zstd-22`
- Impact expectation: small-medium but direct on artifact bytes.
- Current status: root uses `zlib.compress(..., level=9)` in [train_gpt.py#L1312]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L1312).
- Mutation class: easy-medium patch.
- Why it is attractive: cheap export-only move.
- Validate on: smaller artifact at equal round-tripped score.
- Kill if: decompression/runtime complications outweigh byte savings.

`fp16 tied embedding`
- Impact expectation: high prior because the survey calls it standard.
- Current status: not first-class in root, though keep-float logic exists in [train_gpt.py#L320]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L320).
- Mutation class: easy patch.
- Best mutation shape: explicit name-based passthrough for embedding matrix.
- Validate on: smaller post-quant gap at acceptable bytes.
- Kill if: bytes become the binding constraint.

`Late-K fp16 passthrough`
- Impact expectation: small-medium.
- Current status: absent in root, but already implemented in the public record branch.
- Mutation class: easy-medium port.
- Why it is attractive: concrete implementation precedent already exists in the repo's record snapshot.
- Validate on: improved round-tripped score with small byte cost.
- Kill if: byte cost is too high for the gain.

### Eval Policy

`Sliding window stride=64`
- Impact expectation: very high on the final benchmark score.
- Current status: absent in root, but present in the public record branch.
- Mutation class: easy port.
- Why it matters: this is the single biggest survey line item by estimated gain.
- Validate on: large final BPB lift on the same checkpoint.
- Kill if: challenge rules or operational cost make it unusable, not because it lacks effect.

`LoRA TTT`
- Impact expectation: already live.
- Current status: already implemented and used in final eval in [train_gpt.py#L848]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L848) and [train_gpt.py#L1353]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L1353).
- Mutation class: env-only and tuning.
- Best use: tune rank, LR, chunk size, and eval window. Do not treat it as a missing family.

`Doc-isolated eval`
- Impact expectation: medium.
- Current status: standard eval is still continuous fixed-chunk in [train_gpt.py#L226]( nanoevolve/pgolf/parameter-golf/train_gpt.py#L226); document structure is only used in the TTT path.
- Mutation class: medium patch.
- Why it is interesting: it is orthogonal to training and export and could combine with sliding eval.
- Validate on: benchmark score improves on the same checkpoint under a rule-clean implementation.
- Kill if: effect disappears when replayed carefully.

## What Is Already Strongest For Next Stage

These are the most actionable next mutations, balancing impact and implementation cost.

### Tier 1: Immediate

- sliding-window eval port with `stride=64`
- Muon weight decay
- fp16 tied-embedding passthrough
- late-K fp16 passthrough
- zstd export
- grad clip sweep around `0.3`

### Tier 2: Strong Medium Patches

- SWA during warmdown
- NorMuon
- OrthoInit + muP-style output scaling
- STE int6 QAT
- SmearGate

### Tier 3: Structural Branches

- BigramHash
- mixed int5/int6 export
- 10-layer stack on top of recovered byte budget
- MTP
- doc-isolated eval

## Recommended 8-Run Pack

If the goal is to spend the next stage efficiently, this is the clean pack:

1. exact record-like replay control
2. `stride=64` eval-only port
3. `MuonWD=0.02`
4. fp16 tied embedding export
5. late-K fp16 passthrough
6. zstd export
7. SWA on top of the best export-aware branch
8. NorMuon on top of the best training/export branch

That pack tests:

- the biggest eval lever,
- the cheapest export levers,
- the most repeated training-dynamics lever,
- and one real optimizer upgrade.

## Interpretation Rules

- If `stride=64` wins big again, eval remains a first-order optimization lane and should stay separate from training attribution.
- If export-only changes move score more than training changes, the frontier is still export-bound.
- If MuonWD or NorMuon win only after export is improved, they are helper mutations, not solo winners.
- If SWA helps only after quant/export improvements, that means it is smoothing for quant robustness rather than general training quality.
- If SmearGate or BigramHash are tested, compare them against a byte-matched control, not a raw parameter-matched control.
