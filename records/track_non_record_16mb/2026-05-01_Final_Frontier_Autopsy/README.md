# Non-record: Final Frontier Autopsy

**Track:** non-record / methodology
**Author:** Himanshu Dongre
**Date:** 2026-05-01
**Leaderboard claim:** none

This is a non-record submission.  It records my final attempt to improve the
late PR #2018 frontier, the logs behind that attempt, and the stop rule that
fell out of it.

The short version: every serious branch failed before quantization.  On this
frontier, the trained model had to be competitive before GPTQ and TTT.  Mine
was not.

## Result Summary

| Run | Base | Change | Seed | Pre-quant BPB | Quant BPB | Final BPB | Train time | Eval time | Artifact | Result |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| PR #2018 reference | #2018 | none | 1337 | **1.05124428** | 1.05990331 | **1.04826351** | 596.167s | 465.480s | 15,992,746 | target |
| Plan A | #2018 | Gate32 + q-aware token-only tilt | 1337 | 1.06385301 | 1.07199665 | 1.06057508 | 595.974s | 515.283s | 15,972,854 | no-go |
| Plan B | #2018 | Gate32 + native n-gram | 1337 | 1.06434971 | not run | not run | 596.116s | not run | not run | stopped at pre-quant |
| Plan C | #2018 | native n-gram + BigramHash 512x4 + Path-A-v3 small | 1337 | 1.06471733 | not retained | not retained | 596.111s | not run | not retained | stopped at pre-quant |

The useful observation is where the runs failed.  The gap was already around
`+0.013 BPB` before quantization.  That was too large for quantization choices
or score-first TTT to rescue.

```text
Frontier stop rule from these runs:
If a branch is about +0.01 BPB worse before quantization on the same seed,
stop unless the branch introduces a proven legal eval mechanism of that size.
```

## Files

```text
records/track_non_record_16mb/2026-05-01_Final_Frontier_Autopsy/
|-- README.md
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

Plan C is terminal-observed only.  The pod connection dropped before I could
copy the full remote output folder, so I do not treat it as a complete logged
artifact.

## Plan A: Gate32 + q-aware token-only n-gram tilt

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

Budget checks passed:

- artifact: `15,972,854` bytes,
- train: `595.974s`,
- eval: `515.283s`.

The n-gram path was token-only and timed inside eval.  The failure was not
timing or artifact size.  The trained model was weak before quantization.

## Plan B: Gate32 + native PR #2018 n-gram

Plan B removed the q-aware patch to test whether the stricter n-gram logic was
responsible for the regression.

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

That result isolated the main issue.  Gate32 itself did not transfer to this
stack.  The q-aware n-gram patch was not the cause of the pre-quant regression.

## Plan C: exact #2018 gates + tiny BigramHash

The last branch removed Gate32 and tested a small causal input feature from my
earlier work.

Command shape:

```bash
BIGRAMHASH_PATCH=1 PATH_A_V3_SMALL=1 \
BIGRAM_VOCAB_SIZE=512 BIGRAM_DIM=4 BIGRAM_BITS=6 \
BASE_STACK=2018 \
  bash /workspace/pod_move39_gate_scout.sh \
  run_split 1337 12 12 planC_2018_native_bigram512d4_patha
```

Terminal-observed lines:

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

This was also stopped at pre-quant.  A tiny BigramHash branch did not recover
training quality on the #2018 frontier.

## Interpretation

The final runs support three narrow conclusions.

### 1. Gate32 did not transfer here

Gate widening had public evidence on nearby stacks, but it damaged this one.
The failure appeared before quantization and before TTT, so this was a training
dynamics problem rather than an eval-time issue.

### 2. The q-aware n-gram patch was not the root cause

Plan B removed that patch and still produced an even worse pre-quant result.

### 3. A tiny BigramHash branch was too small or mismatched

BigramHash helped my earlier PR #1716 in a different base.  The 512x4 version
tested here did not transfer to #2018.

## Compliance Notes

This is a non-record package, but the executed Plan A run still satisfies the
ordinary record constraints:

| Check | Plan A |
|---|---|
| Artifact under 16,000,000 bytes | yes, 15,972,854 |
| Train under 600s | yes, 595.974s |
| Eval under 600s | yes, 515.283s |
| Token-only n-gram path | yes |
| In-timer n-gram preprocessing | yes |
| Score-first TTT inherited | yes |
| Leaderboard claim | none |

## Closing Note

This folder is intentionally narrow.  It is not a competition-wide synthesis.
It is the evidence package for one failed final transfer attempt.

The broader research notes are in PR #2111.
