# Non-Record Draft: EchoHarp + Random Kitchen Sink

**Status:** late-minute non-record research submission with a small 1xH100 RunPod probe. Not leaderboard-comparable.

This entry takes a deliberately different, slightly whimsical direction: **a tiny neural model playing a classic-compressor harp**. Instead of adding another hand-tuned transformer stack, it asks whether procedural context echoes and deterministic random features can buy compression behavior without storing much learned capacity.

## Strategy Context

This follows the project strategy notes from the `Parameter golf` ChatGPT project:

- Preserve outsider hypotheses before trying to get fully "caught up."
- Look for hidden assumptions in strong submissions instead of only admiring their details.
- Cross two levers rather than maximizing one in isolation.
- Ship a weird but defensible experiment with a clear ablation story.

The hidden assumption targeted here is: **all useful prediction machinery must look like learned transformer weights**. Classic compressors lean hard on caches, repetition, and local continuations. Random-feature methods lean on procedural bases. This submission grafts both instincts onto a small LM in a way that is easy to ablate.

## Idea

The model keeps the clean official baseline backbone and adds three small pieces:

1. **EchoHarp procedural prior**
   - A tiny causal compressor modifies logits using only `input_ids`, never future targets.
   - `copy_scale`: boosts the current token as the next-token candidate.
   - `unigram_scale`: boosts recently seen tokens with a `1/sqrt(lag)` decay.
   - `motif1_scale`: if the current token appeared earlier, boosts the token that followed that earlier occurrence.
   - `motif2_scale`: if the last 2-token motif appeared earlier, boosts the continuation that followed it.
   - `motif3_scale`: if the last 3-token motif appeared earlier, boosts the continuation that followed it.
   - This is PPM-ish/kNN-ish in spirit, but tiny: five learned scalar "strings" and no stored n-gram table.

2. **Deterministic random kitchen sink adapter**
   - A frozen random matrix maps the final hidden state from 512 dimensions into `RANDOM_FEATURES=192` random ReLU-squared features.
   - The random matrix is regenerated from `RANDOM_FEATURE_SEED=20260430`, registered as `persistent=False`, and is therefore not saved in the model artifact.
   - Only the learned down-projection back to 512 dimensions and a scalar gate are stored.
   - This is a cheap way to spend compute on a nonparametric feature bank while spending artifact bytes only on the adapter.

3. **Learned logit prior**
   - A 1024-value learned vocabulary prior is added before logit softcapping.
   - This is intentionally simple: if the challenge metric is compression, a tiny unigram-ish prior should be allowed to learn the first-order entropy shape rather than forcing the tied embedding head to represent it indirectly.

The bolder thesis is that the model should use the transformer for semantic/contextual abstraction, EchoHarp for compression-shaped repetition, and random kitchen sinks for cheap nonlinear correction. It is closer in spirit to PPM, kNN-LM, kernel machines, and old-school adaptive compressors than to the current SOTA meta-stack.

Approximate parameter/artifact cost:

- Stored adapter down-projection: `512 * 192 = 98,304` parameters.
- Stored adapter gate: `1` parameter.
- Stored logit prior: `1,024` parameters.
- Stored EchoHarp strings: `5` scalar parameters.
- Unstored random feature matrix: `192 * 512 = 98,304` deterministic buffer parameters regenerated from seed.

So the submission spends roughly 100k stored parameters to get a 192-feature nonlinear procedural lift plus a variable-order motif cache with only five learned knobs. That is the golf move.

## Why It Might Move The Field Forward

Most Parameter Golf progress has come from making a small transformer act more like a better transformer: stronger tokenization, better quantization, smarter TTT, fused kernels, and careful hparams. EchoHarp tests a different decomposition:

- Let the transformer learn what procedural compressors are bad at.
- Let a tiny causal compressor handle the boring but important repeat/motif statistics.
- Let procedural random features add nonlinear correction without storing the feature basis.

If this works even modestly, the useful follow-up is not this exact implementation. The follow-up is a family of **neural + procedural compression hybrids** where learned weights are reserved for abstraction and byte-free algorithms shoulder predictable local structure.

## Local Toy Evidence

Local CPU tests cannot estimate FineWeb BPB, but they can test whether the mechanism is pointed in the right direction.

On a synthetic repeated-phrase corpus with zero neural logits, training only EchoHarp scalars gave:

| Prior strings trained | Cross-entropy |
|---|---:|
| Uniform logits | 3.4657 |
| Copy + recency | 3.0845 |
| Motif-1 only | 2.5580 |
| Motif-1 + motif-2 + motif-3 | 2.2795 |
| All five strings | 2.2706 |

The useful signal is that motif continuations dominate simple copy/recency. That is exactly the field-moving hypothesis: repeated symbolic structure may be better handled by a tiny causal compressor than by spending transformer parameters relearning the same local motif machinery.

Correctness spot check: for context `[1,2,3,4,1,2,3,0]` with only `motif3_scale=1`, the second `[1,2,3]` boosts token `4` by `0.5`, matching the earlier continuation after the same 3-token motif with the intended `1/sqrt(lag)` decay.

## RunPod Probe Evidence

This was run on a single RunPod H100 SXM using the official template image, `sp1024`, one train shard, `ITERATIONS=50`, `TRAIN_BATCH_TOKENS=65536`, `VAL_BATCH_SIZE=262144`, and the full cached validation shard (`62,021,632` tokens). This is **not** an official 8xH100 leaderboard run, but it is enough to tell whether the implementation survives real data, real CUDA, serialization, quantization, and roundtrip validation.

| Run | Final roundtrip BPB | Pre-quant BPB | Train time | Notes |
|---|---:|---:|---:|---|
| Official baseline, matched tiny run | 2.538749 | 2.5373 | 10.4s | Root `train_gpt.py` |
| EchoHarp only, `RANDOM_FEATURES=0` | 2.539374 | 2.5379 | 11.1s | Motif prior learns, but does not beat baseline here |
| EchoHarp + random kitchen sink | 2.540498 | 2.5397 | 11.7s | Submitted default probe, under 16MB |

The honest read: the current whimsical stack is runnable and compact, but this tiny probe does **not** show a win. The interesting signal is mechanistic rather than scoreboard-shaped: after 50 steps, EchoHarp learned `copy_scale=-0.331`, `unigram_scale=0.768`, `motif1_scale=1.049`, `motif2_scale=1.040`, and `motif3_scale=1.001`. In other words, the optimizer discovered the compressor-like motif strings, but the present form does not yet convert that behavior into better BPB.

That negative result is still useful. It suggests the next rabbit hole is not "make the kitchen sink bigger," but rather: keep the causal motif harp, reduce or regularize the random adapter, and test whether the motif prior becomes helpful later in training or when grafted onto a stronger tokenizer/TTT stack.

## Why This Is Non-Record

The current 10-minute leaderboard frontier is around **1.0611 BPB**, with heavily optimized tokenization, GPTQ/LQER compression, phased TTT, fused kernels, and multi-seed evidence. This entry does not claim to beat that.

The goal is to submit a compact, readable, runnable experiment that explores a genuinely different axis:

- Can a procedural, causal variable-order motif prior help BPB enough to justify its compute?
- Can deterministic random maps provide useful expressive capacity without paying model bytes?
- Does a learned vocabulary prior help compression under the 16 MB artifact cap?
- Is this worth combining later with the stronger SP8192/CaseOps/TTT stacks?

## Reproduction

From this folder inside the repository:

```bash
RUN_ID=random_feature_adapter_seed42 \
SEED=42 \
VAL_LOSS_EVERY=0 \
RANDOM_FEATURES=192 \
RANDOM_FEATURE_SCALE=0.35 \
RANDOM_FEATURE_SEED=20260430 \
ECHO_WINDOW=64 \
ECHO_COPY_INIT=0.0 \
ECHO_UNIGRAM_INIT=0.0 \
ECHO_MOTIF1_INIT=0.0 \
ECHO_MOTIF2_INIT=0.0 \
ECHO_MOTIF3_INIT=0.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a quick 1-GPU smoke run on the RunPod template:

```bash
RUN_ID=random_feature_adapter_smoke \
SEED=42 \
ITERATIONS=20 \
MAX_WALLCLOCK_SECONDS=0 \
VAL_LOSS_EVERY=0 \
TRAIN_BATCH_TOKENS=65536 \
VAL_BATCH_SIZE=65536 \
RANDOM_FEATURES=192 \
ECHO_WINDOW=16 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

The script resolves default data paths relative to the repository root, so it should run from inside this record folder after the standard `data/cached_challenge_fineweb.py --variant sp1024` download.

## Local Checks

Local desktop environment check:

```bash
python3 -m py_compile records/track_non_record_16mb/2026-04-30_EchoHarp_RandomKitchenSink/train_gpt.py
```

This passed on 2026-04-30 Pacific time. The RunPod probe above passed on the official template image with PyTorch 2.9.1 + CUDA 12.8 on a single H100.

## Files

- `train_gpt.py`: official clean baseline plus EchoHarp, the random kitchen sink adapter, and a learned logit prior.
- `submission.json`: metadata for the draft submission, with score fields intentionally null until a GPU run is completed.
- `requirements.txt`: same Python package baseline as the official repo.
- `train.log`: generated 1xH100 RunPod probe log for the submitted EchoHarp + random kitchen sink configuration.
- `local_static_check.log`: local compile, autograd, and mechanism-check provenance.
