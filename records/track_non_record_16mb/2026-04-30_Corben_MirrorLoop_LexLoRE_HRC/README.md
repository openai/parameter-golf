# MirrorLoop HRC + LexLoRE

**Track:** non-record / art submission  
**Author:** Corben Sorenson ([@corbensorenson](https://github.com/corbensorenson))  
**Status:** 1xH100 evidence now, intended 8xH100 follow-up if capacity becomes available before review.

## Summary

This submission explores a deliberately nonstandard small language model shape:

- **MirrorLoop HRC:** a mirrored input/output shell around a recurrent middle:
  `012 | 34567 | 34567 | 210`.
- **LexLoRE:** token-conditioned low-rank lexical expert residual adapters at
  the input and loop-entry sites.
- **Train-time quantization from step 0:** the model is trained through the same
  q8 quantized forward path used by the final artifact, including embeddings.
- **Factored tied embeddings:** the token interface is widened without paying
  the full dense `vocab x dim` cost.
- **LQER:** low-rank quantization error repair is applied at export.
- **One attention-capable core-entry block:** the recurrent core is otherwise
  MLP-only for speed.

The goal is not to clone the accepted leaderboard transformer stack. It is to
test whether a mirrored recurrent core plus lexical low-rank steering can form a
compact, auditable architecture family under the 16MB artifact constraint.

## Architecture

The core idea is to treat the transformer as an hourglass-like route rather
than a flat list of layers. The first three blocks form a token-facing entry
tail. Five semantic blocks then run as a recurrent middle. The exit path mirrors
the entry tail:

```text
entry      loop pass 1       loop pass 2       mirrored exit
0 1 2  ->  3 4 5 6 7   ->    3 4 5 6 7   ->   2 1 0
```

Only block `3` keeps attention inside the loop. Blocks `4,5,6,7` are MLP-only
in the recurrent core, which is a speed/quality tradeoff that worked better
than fully-attentional recurrence in local and H100 sweeps. The route is
conditioned with pass embeddings, loop index information, and small recurrent
injection parameters so repeated blocks can learn different roles on different
passes.

LexLoRE is the token-facing control surface. It uses token-conditioned low-rank
expert residuals with 16 experts and rank 2, shared across the selected sites
with site-specific bias/scale. In the best current spine it is active at:

```text
input, loop_first
```

This means lexical experts advise both the initial read-in and the first entry
to the recurrent semantic middle.

The submission intentionally trains on the quantized forward path from the
first optimizer step:

```text
TRAIN_QUANT_FORWARD=1
TRAIN_QUANT_EMBEDDINGS=1
QUANT_WEIGHT_BITS=8
VOCAB_MOE_TRAIN_QUANT_BITS=8
```

That is important for the architecture: recurrence amplifies export-time
quantization mismatch, so the model should learn under the precision regime it
will actually be scored under. LQER is then used only as an export-time repair
for the largest residual quantization errors.

## Best Current Result

The best legal H100 result available before this PR was a 1xH100, 10-minute
wall-clock run:

| Candidate | BPB | Steps | Step speed | Total bytes |
|---|---:|---:|---:|---:|
| `h100_batch32k_d704e832_w2200_q8_coreattn1_lqer10t20_vocabmoe_qk55` | `1.35692129` | `5018` | `119.57 ms` | `15,658,145` |

This is below the decimal 16MB cap. It is **not** claimed as a record
submission, and it has **not** yet been reproduced on the official 8xH100
configuration. It is submitted here as a non-record/art lane result so the
architecture and negative/positive findings are visible before the challenge
deadline.

The raw RunPod pod used for the strongest 1xH100 queue became unavailable after
the wallet ran out of funds, so `train.log` contains the preserved result notes
from the project audit rather than the full raw stdout. If 8xH100 capacity
becomes available, this PR should be updated with the raw 8x logs.

## What We Learned

The strongest result came from a 32k-token 1xH100 batch. A 24k row reached a
slightly better BPB, but exceeded the decimal 16MB cap. A 16k row produced many
more optimizer steps but worse BPB and a larger artifact. The useful signal is
that this model has a real batch/update-rate sweet spot:

| Batch row | BPB | Steps | Artifact | Read |
|---|---:|---:|---:|---|
| 32k | `1.35692129` | `5018` | `15,658,145` | best legal result |
| 24k | `1.35552525` | `6235` | `16,292,969` | better raw score, over cap |
| 16k | `1.36157540` | `8295` | `16,741,661` | more steps alone lost |

Higher recurrence was also not automatically better. A 32k/r3 probe was legal
but worse (`1.37374423` BPB), so the current best branch stays at r2 and spends
bytes on the token interface and quantization repair instead of more repeated
depth.

## Reproduction

This record folder includes `train_gpt.py` and the small `ternary_golf` helper
package imported by the trainer. The exact 1xH100 command is in
`run_1xh100_best.sh`.

Expected data/tokenizer:

- `DATA_PATH`: CaseOps/SP8192 lossless dataset directory.
- `TOKENIZER_PATH`: matching CaseOps/SP8192 SentencePiece model.

Run:

```bash
bash run_1xh100_best.sh
```

For an 8xH100 follow-up, keep the architecture constants and change only the
distributed launch/batch schedule. The project repository contains a prepared
`final8x` runner for that paid test.

## Planned 8xH100 Update

The intended 8x update is not to broaden the search blindly. It should test the
same architecture family under official-shaped distributed wall-clock runs:

1. Preserve the best 24k-per-rank rhythm across 8 GPUs.
2. Test a 32k-per-rank middle point.
3. Test an official-style 524k global batch.
4. Keep the r3 loop-index row as a single architecture sanity check.

The PR should be updated with raw 8x logs only if an actual 8xH100 pod becomes
available. Partial-H100 runs are intentionally not being substituted for the
official-shaped result.

## Notes On Validity

This submission is intentionally conservative about claims:

- It is under `track_non_record_16mb`.
- It does not claim an 8xH100 official score.
- It does not claim SOTA.
- It reports the current best legal 1xH100 evidence and the intended 8x path.
- It keeps the architecture self-contained and auditable.

## Why This Is Interesting

Most ingredients have relatives in prior work: recurrence, adapters,
quantization-aware training, factored embeddings, and low-rank repair. The art
piece is the combination and the failure analysis:

1. A mirrored IO shell that returns through the same semantic ladder it entered.
2. A looped middle that spends compute without spending many new parameters.
3. Lexical low-rank experts that steer both token read-in and recurrent entry.
4. Training on the quantized forward path from the first step rather than
   quantizing only after training.
5. Treating batch/update rate as part of the architecture search rather than a
   separate systems detail.

This is not presented as “recurrence beats the leaderboard transformer stack.”
It is a compact record of what a mirrored recurrent/lexical-expert family can
do, where it failed, and which pieces seemed to matter.
