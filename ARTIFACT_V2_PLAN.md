# Artifact V2 Plan

## Goal

Turn the current artifact findings into the next concrete branch:

- keep `T_train <= 600s`
- keep `T_eval <= 600s`
- keep `artifact_bytes < 16,000,000`
- minimize final post-quant, post-eval `val_bpb`

This document starts from the measured facts already established in this repo and then narrows the next artifact work to the parts that still have a realistic chance of moving score.

## Measured Starting Point

These points are already established by experiment, not speculation:

- `packed_zlib` beats `torchsave_zlib`, but only modestly.
- scale compression is negligible on our current checkpoints.
- the bottleneck is the quantized payload section, not metadata or scales.
- `tok_emb.weight` is the only mixed-precision keep that survived a stronger-checkpoint rerun.
- large MLP keeps tested so far were not worth their bytes.

Current policy implied by the measurements:

1. keep the custom packed artifact path
2. stop spending time on generic serializer tweaks
3. stop spending time on scale codecs
4. keep `tok_emb.weight` in float passthrough
5. focus next on the still-quantized payload tensors

## Why We Should Keep Writing Our Own Artifact Format

The search-backed answer is yes: keep the custom artifact path.

Why:

- PyTorch documents that `torch.save` uses an uncompressed ZIP64 archive with pickled metadata and one file per storage, plus alignment padding. That is a good general checkpoint format, not an artifact format specialized for our compressed submission object.
- `safetensors` is the closest precedent for the design direction we want: simple header, contiguous tensor payload, no pickle semantics, and fast direct loading.

So the correct framing is:

- we are not trying to "beat `torch.save`" in a generic sense
- we are building a submission-specific tensor container for our quantized object

That direction is already validated by the repo measurements, so V2 should build on it rather than revisit format basics.

## Search-Backed Conclusions For V2

### 1. Codec work is still worth one short pass

Official `zstd` guidance says it targets zlib-like scenarios with better compression ratios, and its decompression speed stays high across settings.

Implication for us:

- `packed + zstd` is worth benchmarking
- `packed + zlib` remains the baseline

But `zstd` dictionaries are not the next thing to do. The official docs say dictionary gains are mostly effective in the first few KB. Our artifact is multi-MB, and our bottleneck is a giant tensor payload, not small-record startup overhead.

Decision:

- benchmark `zstd`
- do not spend time on dictionary training unless later evidence strongly contradicts this

### 2. Prefilters are more promising than more metadata work

`bitshuffle` is explicitly designed for typed binary data and is particularly relevant when:

- adjacent values are correlated
- only a subset of bits are exercised
- you want both compression ratio and performance

That matches our artifact better than more scale coding does.

Implication:

- `bitshuffle` is the only serializer-adjacent prefilter worth a real benchmark now
- it should be applied only where it makes sense

Important nuance:

- byte shuffle is not useful for `int8` payload because the element size is one byte
- bitshuffle is the relevant filter for `int8`
- `fp16` passthrough blocks can plausibly benefit from byte- or bit-level shuffling

Decision:

- if we do one more packaging experiment, it should be blockwise `bitshuffle + zstd` on the quantized payload section

### 3. The real next frontier is quantization preconditioning / rotations

The best search-backed match for our current findings is not "better save format", but weight-space preconditioning before quantization.

Relevant results from the literature:

- QuaRot shows that rotations can remove outliers and make quantization easier.
- SpinQuant shows that rotations are a principled way to help quantization, but random rotations vary a lot and learned rotations can do better.
- QuIP# is especially relevant to our case because it is weight-only PTQ and uses randomized Hadamard incoherence processing specifically to make weights easier to quantize.
- QTIP extends that incoherence-processing line further, but is a bigger algorithmic jump than we need for the next branch.

The key repo-specific inference is:

- for this challenge, we should start with low-byte or zero-byte preconditioning methods
- we should not start with dense learned rotation matrices, because their metadata cost would directly compete with artifact budget

## Tailored Recommendation For This Repo

### Keep this baseline fixed

Baseline for V2 experiments:

- custom packed artifact container
- `packed_zstd` as current default baseline
- `tok_emb.weight` kept in float passthrough
- no more large MLP float-keep sweeps by default

### The next packaging branch

Only two packaging experiments remain justified:

1. `packed_zstd`
2. `packed_bitshuffle_zstd`

Everything else in the serializer family is lower priority than payload-quality work.

### The next payload-quality branch

Start with low-byte weight preconditioning on the dominant quantized tensors.

That means:

- leave `tok_emb.weight` alone because it is already float passthrough
- target the largest still-quantized matrices
- prefer transforms that need no stored dense metadata

Best first candidates:

1. deterministic Hadamard-based preconditioning
2. randomized Hadamard with a tiny seed or deterministic name-derived seed
3. sign/permutation transforms with negligible metadata

Not recommended as the first step:

- per-tensor dense learned rotations
- broad learned rotation checkpoints
- full architecture-level QuaRot/SpinQuant surgery

The reason is simple: our current load path dequantizes into dense model weights before evaluation. We want the preconditioner to help quantization error without becoming a byte problem of its own.

## Practical Design For Preconditioning

### Interface

The transform layer should be optional and explicit:

- `preconditioner = none | hadamard | hadamard_sign | hadamard_perm_sign`
- metadata per tensor should be:
  - nothing, if deterministic from shape and name
  - or a tiny seed / permutation id

### Where it fits

For a matrix `W`, the offline flow should look like:

1. choose a structured transform `P`
2. map to a preconditioned space
3. quantize in that space
4. store quantized coefficients plus tiny transform metadata
5. during load, dequantize and invert the transform to recover the dense approximation loaded into the model

This keeps runtime semantics simple:

- no graph surgery at eval time
- no dependency on special kernels
- all changes stay inside the export/load shell

### Why this is the right first variant

It is the closest analogue to weight-only incoherence processing in QuIP# while fitting our current repo architecture.

It also keeps formal verification tractable:

- transform invertibility is easy to test
- roundtrip structure is easy to test
- model load semantics stay unchanged

## Candidate Surface

Given the current repo measurements, candidate testing should stay narrow.

Allowed first-wave candidates:

- the largest still-quantized attention and MLP matrices
- grouped by actual payload contribution
- one tensor at a time first

Not worth retesting now:

- bulk MLP float keeps
- more scale variants
- more generic archive rewrites

Because embeddings are tied and `tok_emb.weight` is already retained, the next large tensors to test are the biggest remaining quantized matrices, but under transform-based quantization rather than direct fp16 passthrough.

## Invariants

Artifact V2 must keep the same strict shell as V1.

### Container invariants

1. deterministic bytes for identical quantized inputs
2. exact container roundtrip
3. exact tensor names, shapes, and dtypes on deserialize
4. strict `load_state_dict` success

### Transform invariants

1. transform metadata is deterministic
2. inverse(transform(x)) reconstructs the original tensor numerically before quantization
3. dequantize + inverse-transform yields finite tensors with original shapes
4. no hidden dependency on runtime random state

### Metric invariants

1. evaluator totals remain matched for A/B comparisons
2. comparisons use the same checkpoint and same eval helper
3. any gain must be reported with both `val_bpb` and compressed bytes

## Experiment Order

### Phase 0. Lock the baseline

Baseline:

- `packed_zstd`
- `tok_emb.weight` float passthrough
- current stronger checkpoint helper workflow

Output to record:

- compressed bytes
- raw serialized bytes
- `val_bpb`
- eval time

### Phase 1. Codec sanity pass

Benchmark:

1. `packed_zlib`
2. `packed_zstd`

Pass gate:

- `zstd` improves bytes enough to matter, with acceptable load/eval cost

Measured result on the stronger checkpoint:

- `packed_zlib`: `14,677,339` bytes, `~794 ms` serialize, `~57 ms` deserialize
- `packed_zstd`: `14,566,076` bytes, `~455 ms` serialize, `~18 ms` deserialize
- dequantized state after load matched exactly

Decision:

- promote `packed_zstd` to the default baseline
- do not spend more time on codec-only work unless a later branch changes the payload statistics materially

### Phase 2. Prefilter sanity pass

Benchmark:

1. `packed_zstd`
2. `packed_bitshuffle_zstd`

Apply the prefilter only to sections where it is appropriate:

- quantized `int8` payload
- optionally fp16 passthrough blocks

Pass gate:

- material byte win over plain `packed_zstd`

Fail gate:

- negligible byte change
- complexity increase without score leverage

### Phase 3. Zero-byte / low-byte preconditioning

Implement and test:

1. deterministic Hadamard preconditioning
2. randomized Hadamard with tiny seed
3. sign/permutation variants

Evaluation protocol:

- same stronger checkpoint
- one tensor or one tensor family at a time
- compare against the exact same helper baseline

Pass gate:

- post-quant `val_bpb` improves by at least `0.0005` with no artifact-cap issue

Strong pass:

- `>= 0.0010` BPB gain under matched helper conditions

Fail gate:

- no measurable BPB gain
- metadata overhead eats the benefit
- load/eval complexity becomes fragile

### Phase 4. Only then consider learned rotations

Learned rotations are allowed only if:

- zero-byte structured rotations show real promise
- we can keep metadata tiny or shared
- the resulting artifact still has a clear byte/score advantage

This is deliberately not the starting point.

## What Not To Do

Do not spend more time now on:

- scale codecs
- generic `torch.save` variations
- zstd dictionary training
- more wide float-keep sweeps over MLP tensors
- dense learned rotations as the first move

## Recommended Immediate Next Step

The best next experiment sequence is:

1. add `packed_zstd`
2. if that is not a clear win, stop codec work
3. add a small preconditioner interface
4. try deterministic Hadamard-style preconditioning on the largest remaining quantized tensors
5. evaluate with the same helper on the stronger checkpoint

If that branch works, continue with structured rotation variants.
If it does not, artifact-side work should likely move toward a deeper quantization family such as QuIP#/QTIP-style codebook methods rather than more save-format work.

## Sources

- PyTorch serialization docs: https://docs.pytorch.org/docs/stable/notes/serialization.html
- `torch.save` docs: https://docs.pytorch.org/docs/stable/generated/torch.save.html
- Safetensors docs: https://huggingface.co/docs/safetensors/en/index
- Zstandard reference implementation and docs: https://github.com/facebook/zstd
- Bitshuffle docs/repo: https://github.com/kiyo-masui/bitshuffle
- QuaRot paper: https://arxiv.org/abs/2404.00456
- SpinQuant paper: https://arxiv.org/abs/2405.16406
- SpinQuant repo: https://github.com/facebookresearch/SpinQuant
- QuIP# paper: https://arxiv.org/abs/2402.04396
- QuIP# repo: https://github.com/Cornell-RelaxML/quip-sharp
- QTIP repo: https://github.com/Cornell-RelaxML/qtip
- fast-hadamard-transform repo: https://github.com/Dao-AILab/fast-hadamard-transform
