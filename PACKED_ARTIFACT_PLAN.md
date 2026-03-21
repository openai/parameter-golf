# Packed Artifact Plan

## Goal

Convert export bytes into leaderboard score under the actual challenge objective:

- keep `T_train <= 600s`
- keep `T_eval <= 600s`
- keep `artifact_bytes < 16,000,000`
- minimize final post-quant, post-eval `val_bpb`

This document ranks a few speculative ideas and turns the artifact work into a concrete execution plan with pass/fail gates.

## Idea Ranking

### 1. Submission-specific packed artifact

Why it is promising:

- It attacks `Delta_quant` directly.
- It can free bytes for higher-value tensors without retraining.
- It is low-risk relative to architecture changes.
- It is fully aligned with the math in `RIGOROUS_MATH_APPROACH.md`: improve `Q(theta)` without changing evaluator semantics.

Pass gate:

- compressed artifact bytes improve by at least `100 KB` at equal dequantized weights, or
- the packed format enables a precision change that improves final `val_bpb` with artifact size still under cap

Fail gate:

- no meaningful byte win over `torch.save + zlib`, or
- load/dequant invariants fail, or
- eval/runtime cost becomes fragile

### 2. Random linear maps for quantization preconditioning

Interpret this narrowly:

- fixed rotations
- Hadamard-like transforms
- orthogonal preconditioning before quantization

Why it is promising:

- It may reduce quantization damage without paying many artifact bytes.
- It is an export-side method, so it sits in the right part of the stack.

Pass gate:

- post-quant `val_bpb` improves by at least `0.001` at matched evaluator totals and matched artifact size

Fail gate:

- requires too much metadata
- increases eval complexity too much
- improves dense math but not final post-quant score

### 3. Universal Transformer / recurrent depth sharing

Why it is promising:

- This challenge should reward parameter reuse.
- Weight tying across depth could move the quality-per-byte frontier.

Why it is risky:

- It is a macro-architecture bet.
- It requires retraining and likely retuning the whole recipe.
- It is slower to validate than export-side work.

Pass gate:

- 5090 proxy says it fits the `8xH100` train budget
- artifact remains under cap
- dense and post-quant score are both competitive

Fail gate:

- training throughput falls too far
- quantization damage rises
- repeated-depth sharing hurts optimization more than it saves bytes

### 4. Distillation

Why it ranks last for the record track:

- It consumes training budget directly.
- It risks challenge-spirit scrutiny if it effectively imports extra compute through a teacher.
- It is better suited to non-record or unlimited-compute exploration.

Pass gate:

- only worth keeping if it gives a very large and reproducible gain, clearly within the spirit of the rules

Fail gate:

- any ambiguous dependence on extra compute or external capability
- any gain that is smaller than what export/eval work can buy more safely

## Policy On External Libraries

The repo FAQ allows imported packages so long as they do not violate the rules on evaluation, compute, code size, or otherwise sneak in unfair capability.

Operationally:

- using `flash-attn`, `zstandard`, or a faster serializer is allowed in principle
- but the imported package does not excuse extra eval compute, larger effective artifact semantics, or irreproducible setup
- every dependency used by a record submission should be explicit in `requirements.txt` inside the record folder and explained in the README

For the artifact path, this means:

- start with stdlib + existing stack first
- only bring in a codec like `zstandard` if it wins on the actual score/runtime/size objective

## Submission-Specific Packed Artifact

### Design Objective

Replace generic `torch.save(quant_obj) + zlib` with a deterministic serializer specialized for the quantized export object:

- `quantized` int8 tensors
- `scales`
- `passthrough` tensors
- quantization metadata

This should improve one or both of:

1. raw serialized bytes before compression
2. compressibility after outer compression

### Non-Negotiable Invariants

The packed format must satisfy:

1. deterministic bytes for identical inputs
2. exact roundtrip of the quantized object
3. identical dequantized state dict after load
4. identical `load_state_dict(..., strict=True)` success behavior
5. no evaluator metric drift caused by serialization

These are the first tests to keep green before any byte optimization.

### Format V1

Current implementation target:

- magic: `PGQ1`
- fixed header with version and metadata length
- compact JSON metadata
- a single concatenated payload section
- tensor entries carrying:
  - name
  - section: `quantized`, `scales`, or `passthrough`
  - stored dtype
  - shape
  - payload offset
  - payload byte length
  - logical float dtype for quantized tensors

Outer compression remains a separate stage.

### Why This First Version Matters

The first version is not expected to produce the final byte win by itself.

Its purpose is to create a stable artifact substrate so we can test byte-saving ideas safely:

- tensor ordering for better compression locality
- more compact metadata
- scale compression
- mixed-precision group packing
- codec swaps

Without a deterministic packer, those experiments are noisy and hard to audit.

## Experiment Order

### Phase A. Serializer Parity

Goal:

- prove `packed_zlib` is functionally identical to `torchsave_zlib`

Checks:

- artifact roundtrip tests pass
- dequantized tensors match the baseline export exactly
- final post-quant `val_bpb` matches within floating-point noise

Decision:

- if parity fails, stop here and fix correctness first

### Phase B. Raw Byte Audit

Measure for the same `quant_obj`:

- `payload_bytes`
- `raw_serialized_bytes`
- `compressed_bytes`
- load/decompress time

Decision:

- keep the packed format only if it is at least not worse on compressed bytes or load time

### Phase C. Compression-Locality Experiments

Low-risk experiments:

1. reorder tensors by section then name
2. cluster tensors with similar statistics
3. isolate fp16 passthrough tensors into contiguous blocks

Pass gate:

- `compressed_bytes` improves without any metric change

### Phase D. Scale Compression

Most likely meaningful artifact lever after parity:

1. fp16 row scales
2. blockwise shared scales
3. log-domain or integer-coded scales

Pass gate:

- compressed artifact improves materially
- final post-quant `val_bpb` does not regress, or improves because saved bytes are reallocated elsewhere

### Phase E. Precision Reallocation

Treat freed bytes as a knapsack budget.

Candidate high-value targets:

1. tied embedding
2. first and last layer tensors
3. attention output projections
4. other sensitive tensors found by measured ablation

Pass gate:

- final post-quant `val_bpb` improves under the hard artifact cap

## Immediate Next Experiments

1. Compare `QUANT_ARTIFACT_FORMAT=torchsave_zlib` vs `packed_zlib` on the same smoke checkpoint.
2. Record:
   - `raw_serialized_bytes`
   - `compressed_bytes`
   - final post-quant `val_bpb`
   - eval/load time
3. If packed parity holds, start with scale-format experiments before any architecture work.

## Current Measured Status

The first artifact phase is complete enough to narrow the search surface.

Measured on the current smoke-scale checkpoint:

- `torchsave_zlib`: `4,925,143` bytes
- `packed_zlib` with raw scales: `4,914,172` bytes
- `packed_zlib` with `log_u8` scales: `4,914,034` bytes

Implications:

- the submission-specific packer is a real improvement, but only a small one
- scale compression is effectively irrelevant on this checkpoint
- the dominant bottleneck is the quantized payload itself, not metadata or scale storage

Section audit:

- `quantized`: `17,039,360` raw payload bytes, `4,899,741` standalone zlib bytes
- `passthrough`: `82,208` raw payload bytes, `9,392` standalone zlib bytes
- `scales`: `57,344` raw payload bytes, `81` standalone zlib bytes

That section breakdown is decisive: do not spend more time on generic save-format tweaks or scale codecs unless a later checkpoint proves otherwise.

## Mixed-Precision Findings

Single-tensor fp16 passthrough measurements on the same checkpoint:

| Candidate | Post-quant `val_bpb` | Artifact bytes | Verdict |
| --- | ---: | ---: | --- |
| baseline packed artifact | `4.1081269968` | `4,914,172` | reference |
| `tok_emb.weight` | `4.1064685447` | `5,598,496` | clear win |
| `blocks.0.mlp.proj.weight` | `4.1081241614` | `5,401,326` | noise |
| `blocks.0.mlp.fc.weight` | `4.1081269968` | `5,428,443` | no gain |
| `blocks.8.mlp.fc.weight` | `4.1081269968` | `5,429,969` | no gain |

Interpretation:

- `tok_emb.weight` is the only mixed-precision knob with a meaningful score improvement so far
- the checked MLP tensors are not worth more artifact budget
- because embeddings are tied in the default model, `tok_emb.weight` is also the output-head-sensitive candidate
- small logit-path tensors like norms are already kept in float by the existing passthrough rules, so the next large candidate set should stay tightly focused

Stronger-checkpoint confirmation on a 2,000-step rerun of the same 9x512/seq256 recipe:

- helper baseline with packed artifact: `1.9572997503 BPB`, `14,346,367` bytes
- helper baseline + `tok_emb.weight` fp16: `1.9566958943 BPB`, `14,677,339` bytes

That is still a win:

- `Delta BPB = -0.00060386`
- `Delta artifact bytes = +330,972`

The gain is smaller than on the smoke checkpoint, but it survives on a more serious model state and still leaves the artifact under the `16 MB` cap.

## Updated Decision

For the current branch:

1. keep `packed_zlib` as the default artifact format
2. keep `PACKED_SCALE_CODEC=raw`
3. promote `tok_emb.weight` as the current best mixed-precision candidate
4. rerun `tok_emb.weight` on a stronger checkpoint before baking it into the main quantization path
5. stop testing large MLP weights unless a later checkpoint changes the ranking
6. if another artifact-side idea is needed after that, move to quantization preconditioning or rotations on the dominant payload tensors

## Next Candidate Surface

After the stronger-checkpoint rerun, only continue artifact-side candidate testing for tensors that are plausibly embedding-adjacent or logits-sensitive.

In practice that means:

- keep testing `tok_emb.weight`
- do not expand the search to bulk MLP tensors again
- prefer preconditioning/rotation experiments over more generic mixed-precision sweeps once the embedding result is confirmed

## Recommendation

For the current record-track branch:

1. packed artifact
2. quantization-oriented random linear maps / rotations
3. Universal Transformer branch
4. distillation

That order is not about novelty. It is about expected leaderboard value per unit of engineering and verification risk.
