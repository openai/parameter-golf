# Non-record: Last-Day Frontier Transfer Autopsy

**Track:** non-record / methodology
**Author:** Himanshu Dongre
**Date:** 2026-05-01
**Leaderboard claim:** none

This is a non-record submission documenting a final-day attempt to improve the
late Parameter Golf frontier under the exact 10-minute training, 10-minute eval,
and 16 MB artifact constraints.

The submission does **not** claim a new leaderboard result.  Its purpose is to
leave behind a compact, reproducible record of what was tested, what failed, why
it failed, and which lines of research still look promising.  I hope this is
useful to future small-model/compression work, and to reviewers trying to
separate real frontier gains from denominator bugs, eval-time leakage, and
non-transferable knobs.

## Executive Summary

I attempted to transfer several plausible orthogonal improvements onto the
clean late frontier around PR #2018:

- widening the sparse attention gate reader (`GATE_WINDOW=32`),
- a stricter q-aware token-only n-gram tilt path,
- preserving PR #2018's native in-timer token-only n-gram path,
- a tiny `BigramHashEmbedding` branch with Path-A-v3-style small-tensor routing,
- and several earlier local-only probes: CrossWS tokenizer, Memento/copy memory,
  online bias adaptation, context horizon extension, and artifact packing.

The decisive lesson is:

> The current frontier is limited first by the trained neural model.  If
> pre-quant BPB is not competitive, quantization and TTT do not rescue the run.

Every final-day architecture branch that touched the #2018 neural model
regressed pre-quant BPB badly on the weak seed `1337`.

| Run | Base | Change | Seed | Pre-quant BPB | Quant BPB | Final BPB | Train time | Eval time | Artifact | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| PR #2018 reference | #2018 | none | 1337 | **1.05124428** | 1.05990331 | **1.04826351** | 596.167s | 465.480s | 15,992,746 | target |
| Plan A | #2018 | Gate32 + q-aware token-only tilt | 1337 | 1.06385301 | 1.07199665 | 1.06057508 | 595.974s | 515.283s | 15,972,854 | no-go |
| Plan B | #2018 | Gate32 + native n-gram | 1337 | 1.06434971 | not run | not run | 596.116s | not run | not run | killed at pre-quant |
| Plan C | #2018 | native n-gram + BigramHash 512x4 + Path-A-v3 small | 1337 | 1.06471733 | not retained | not retained | 596.111s | not run | not retained | killed at pre-quant |

The evidence says that `GATE_WINDOW=32` is not a harmless widening on this
stack, and that a very small BigramHash branch does not recover the lost
training quality.  The final run was stopped at the pre-quant gate because it
was already about `+0.0135 BPB` worse than the #2018 reference pre-quant.

## Why This Is Submitted

The last few days of the competition produced many strong-looking PRs, some of
which were later disputed on byte accounting, eval timing, or validation
adaptation.  This submission tries to do the opposite: document the negative
path cleanly, including cost-aware stop decisions.

The final stretch used roughly **$150 of personal RunPod spend** after grant
credits were exhausted.  I include that only as compute-accounting context:
when a run is clearly outside the pre-quant target band, stopping is part of the
methodology, not a lack of ambition.

## Included Evidence

```text
records/track_non_record_16mb/2026-05-01_LastDay_Frontier_Transfer_Autopsy/
|-- README.md
|-- PR_BODY.md
|-- submission.json
|-- logs/
|   |-- planA_2018_qaware_gate32_seed1337.log
|   |-- planA_metrics_and_decision.jsonish
|   |-- planB_2018_native_gate32_seed1337_prequant_killed.log
|   |-- planB_metrics_and_decision.jsonish
|   `-- planC_terminal_observation.md
`-- scripts/
    |-- pod_move39_gate_scout.sh
    |-- patch_qaware_ngram_2018.py
    |-- patch_bigramhash_frontier.py
    `-- patch_path_a_v3_small_frontier.py
```

The final Plan C pod connection died before the full remote folder could be
copied.  I therefore include only the terminal-observed pre-quant line and do
not treat that run as a complete logged artifact.

## Final-Day Runs

### Plan A: q-aware token-only n-gram + Gate32

Command shape:

```bash
QAWARE_NGRAM_PATCH=1 NGRAM_QAWARE_DYNAMIC=1 BASE_STACK=2018 \
  bash /workspace/pod_move39_gate_scout.sh \
  run_split 1337 32 12 planA_2018_qaware_gate32
```

Key log lines:

```text
caseops_enabled: True
gate_window: 32
smear_gate_window: 12
ngram_tilt_enabled: True
ngram_qaware_dynamic: True
val_tokens: 47851520
stopping_early: wallclock_cap train_time: 595974ms step: 4871/20000
diagnostic pre-quantization post-ema val_loss:2.32830764 val_bpb:1.06385301
diagnostic quantized val_loss:2.34613048 val_bpb:1.07199665
ngram_tilt:hints total=47851520 gated=628130 token_gate=628130 within_gate=0 word_gate=0 agree2plus=0
quantized_ttt_phased val_loss:2.32093140 val_bpb:1.06057508 eval_time:515283ms
total_eval_time:515.3s
```

Constraints passed:

- artifact: `15,972,854` bytes,
- train: `595.974s`,
- eval: `515.283s`,
- CaseOps byte sidecar active,
- n-gram hint construction inside the eval timer,
- token-only path (`within_gate=0`, `word_gate=0`).

But the run was not competitive.  The most important number is not the final
BPB; it is the pre-quant regression from #2018's seed-1337 `1.05124428` to
`1.06385301`.

### Plan B: native #2018 n-gram + Gate32

After Plan A, I tested whether the q-aware patch was the problem by removing it
and preserving #2018's native in-timer token-only n-gram path.

Command shape:

```bash
BASE_STACK=2018 \
  bash /workspace/pod_move39_gate_scout.sh \
  run_split 1337 32 12 planB_2018_native_ngram_gate32
```

Key log lines:

```text
caseops_enabled: True
gate_window: 32
smear_gate_window: 12
ngram_tilt_enabled: True
val_tokens: 47851520
stopping_early: wallclock_cap train_time: 596116ms step: 4827/20000
diagnostic pre-quantization post-ema val_loss:2.32939469 val_bpb:1.06434971
```

This failed the early pre-quant gate and was killed before paying for full
quantized TTT eval.

This result isolates the main failure: **Gate32 itself did not transfer to
#2018**.  The stricter q-aware n-gram patch was not the main cause.

### Plan C: exact #2018 gates + BigramHash 512x4 + Path A v3 small

The final paid branch removed Gate32 and tested the most orthogonal training
feature available from my earlier work: a tiny causal BigramHash input feature.

Command shape:

```bash
BIGRAMHASH_PATCH=1 PATH_A_V3_SMALL=1 \
BIGRAM_VOCAB_SIZE=512 BIGRAM_DIM=4 BIGRAM_BITS=6 \
BASE_STACK=2018 \
  bash /workspace/pod_move39_gate_scout.sh \
  run_split 1337 12 12 planC_2018_native_bigram512d4_patha
```

Terminal-observed key lines:

```text
bigram_bits: 6
bigram_dim: 4
bigram_vocab_size: 512
path_a_v3_small: True
gate_window: 12
smear_gate_window: 12
model_params:35949858
stopping_early: wallclock_cap train_time: 596111ms step: 4837/20000
diagnostic pre-quantization post-ema val_loss:2.33019926 val_bpb:1.06471733
```

This was also killed at the pre-quant gate.  The pod disconnected before I could
copy the full remote output folder, so this is recorded as a terminal-observed
negative result rather than a complete artifact.

## Earlier Local/Smoke Findings

### A40 structural smoke tests

Before the final 8xH100 run, I used a cheaper A40 pod for structural, non-score
smokes.  The useful signal was compile/serialization health, not BPB.

| Candidate | Params | Smoke artifact bytes | Structural verdict |
|---|---:|---:|---|
| Gate32 baseline | 35,947,453 | 15,870,213 | compiles/runs |
| Gate40 | 35,948,165 | 15,871,819 | compiles/runs |
| Gate48 | 35,948,877 | 15,875,774 | compiles but artifact-risky |
| Gate32 + Bigram 512x8 | 35,955,646 | 15,857,770 | compiles/runs |
| Gate32 + Bigram 512x4 | 35,951,550 | 15,872,856 | compiles/runs |
| Gate32 + Bigram 1024x4 | 35,953,598 | 15,864,121 | compiles/runs |

The A40 tests proved that the Bigram and Path-A-v3 patches were mechanically
viable.  They did **not** predict 8xH100 BPB transfer.

### Cross-whitespace tokenizer

Cross-whitespace SentencePiece (`split_by_whitespace=False`) was the strongest
local tokenizer idea.

On a 10 MB train-proxy slice decoded from the official SP1024 train shard:

| tokenizer | tokens | tokens/byte | ratio |
|---|---:|---:|---:|
| default SP8192 training | 2,880,110 | 0.26126 | 1.00000 |
| cross-whitespace SP8192 | 2,731,553 | 0.24778 | **0.94842** |

The same effect was stable on val-derived 2.4 MB, 10 MB, and 25 MB samples
(`0.9466-0.9483`).  Byte-denominator invariants passed when counting
byte-fallback pieces correctly.

Why it did not become the final submission:

- it requires a full raw-docs export/tokenizer/data-sidecar pipeline,
- the deadline window was too tight,
- a legal tokenizer PR must be extremely careful about raw-byte sidecars and
  validation split provenance.

I still think this was the best longer-horizon novel direction.

### Memento / runtime copy memory

The first Memento probe looked promising, but a sliding-window bug was found.
After fixing prefix-depth accounting, the apparent gain collapsed.

Representative fixed result:

| Gate | old buggy delta | fixed 512/256 | fixed 1024/512 |
|---|---:|---:|---:|
| L_min=5 | +0.01118 nats/tok | -0.00063 | -0.00466 |
| L_min=6 | +0.00971 nats/tok | +0.00137 | -0.00178 |
| L_min=8 | +0.00566 nats/tok | +0.00126 | -0.00054 |

With proper prefix depth, the neural model was already highly confident on
copy-hit events, so the memory overlay had little upside and still paid miss
penalties.  Verdict: no-go as currently designed.

### Context horizon

Local context-horizon audit on the same 52k-token slice:

| seq_len : stride | median prefix depth | BPB proxy |
|---:|---:|---:|
| 512 : 256 | 383 | 1.1231 |
| 1024 : 512 | 764 | 1.1118 |
| 2048 : 1024 | 1524 | **1.1078** |
| 4096 : 2048 | 3012 | 1.1135 |

Longer is not automatically better.  RoPE/context mismatch makes 4096 worse on
this lineage, and 4096 plus TTT is not budget-safe.

### Online scalar/bias adaptation

A local score-first online adaptation probe found only tiny gains:

| family | best test delta | BPB-equivalent |
|---|---:|---:|
| unigram Dirichlet mixture | -0.00028 nats/tok | -0.00011 |
| logit-bias SGD | near zero | near zero |
| online temperature | +0.00088 nats/tok | +0.00034 |

Verdict: real but too small to rescue a frontier run.

### Artifact savings

Semantic-preserving artifact savings were limited:

| lever | measured direction |
|---|---:|
| compact custom packer | about 10 KB possible |
| code wrapper tweaks | sub-KB |
| int6 bit-packing | regressed by MBs after compression |
| compressor swap | brotli/lrzip already near best for this stack |

This matters because many attractive architectural additions are artifact-bound.
Compression alone was not a hidden 100 KB lever.

## Legality Notes

For the runs actually executed here:

| Condition | Status |
|---|---|
| C1 strict causal dependence | yes; normal causal transformer and prefix-only n-gram state |
| C2 normalized full-vocab distribution | yes; neural softmax, closed-form n-gram tilt renormalization where used |
| C3 score before update | yes; inherited score-first phased TTT |
| C4 single pass | yes; no token rescored after adaptation |
| CaseOps byte accounting | yes for Plan A/B; `caseops_enabled=True` and sidecar path active |
| Eval timing | yes for Plan A; n-gram hint precompute inside timer |

I intentionally did not build on:

- validation PreQuantTTT,
- byte-level PPM scores not normalized over the official SP token vocabulary,
- suspicious CaseOps byte-denominator accounting,
- or any approach that scores validation tokens after adapting on those same
  tokens.

## Lessons

### 1. Pre-quant BPB is the first kill gate

On the frontier, TTT is mostly recovering quantization damage and adding a
small legal adaptation gain.  It does not rescue a trained model that is already
`+0.013 BPB` worse pre-quant.

### 2. Gate variants are not interchangeable

Public evidence for a wider attention-output gate did not transfer to #2018's
specific sparse-attention/gated-XSA stack.  The same extra reader dimensions
that look harmless in one gate mechanism can destabilize another.

### 3. Mechanical viability is not BPB transfer

The A40 smoke tests correctly predicted that BigramHash and Path-A-v3 would
compile, serialize, and fit in small form.  They did not predict competitive
training dynamics on 8xH100.

### 4. Eval-time memory is mostly already priced in

By the time the model has 2048-2560 context plus TTT and token-only n-gram tilt,
simple runtime copy/memory overlays tend to fire where the neural model is
already confident.

### 5. Tokenizers remain the most interesting longer-horizon lever

Cross-whitespace SP8192 gave a stable ~5.15% token-count reduction on train
proxy data.  It was not operationally finishable in the final hour, but it is
the most promising unmerged idea from this work.

## Compute and Reproducibility Note

I spent the final phase trying to validate ideas under real 8xH100 conditions
rather than rely on weak local proxies.  After grant credits were exhausted, I
used roughly `$150` of personal RunPod spend across the last pushes.  This is
included for reproducibility/cost context only: the main point is that the
negative result was earned under the same constraints as a real record attempt.

## Conclusion

This PR is a non-record, but it is not empty.

It documents:

- a failed transfer of Gate32 to #2018,
- a failed transfer of tiny BigramHash to #2018,
- a corrected no-go for Memento/copy memory,
- a validated but unfinished CrossWS tokenizer direction,
- and a concrete stop rule for future final-day frontier chasing:

> If the run is not pre-quant competitive, stop before quantization and TTT.

That rule saved additional spend in the final hour, and it is the clearest
lesson I would carry into future compression competitions.
