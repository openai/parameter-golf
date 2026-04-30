# MirrorLoop HRC + LexLoRE

**Track:** non-record / art submission  
**Author:** Corben Sorenson ([@corbensorenson](https://github.com/corbensorenson))  
**Status:** non-record/art submission with 1xH100 scouts and a limited,
self-funded 8xH100 follow-up. No leaderboard record is claimed.

## Executive Summary

This is an exploratory 16MB language-model submission built around a mirrored
recurrent transformer route rather than a standard stack of unique layers.

The main experiment is:

```text
0 1 2  |  3 4 5 6 7  |  3 4 5 6 7  |  2 1 0
entry     recurrent middle, pass 1     recurrent middle, pass 2     mirrored exit
```

In the project notes we called this shape **HRC**. In reviewer-facing terms,
HRC means an **hourglass recurrent circuit**:

- **Hourglass:** the model enters through a token-facing input tail, spends most
  of its compute in a compact middle, then exits through a mirrored output tail.
- **Recurrent:** the middle blocks are reused for additional virtual depth.
- **Circuit:** the route is explicitly scheduled; it is not just "more layers".

The second main idea is **LexLoRE**, short for **lexical low-rank experts**.
This is implemented in the code under the older flag name `VOCAB_MOE_*`.
LexLoRE adds token-conditioned low-rank residual adapters at selected points in
the route. In the best legal 1xH100 run it is active at the model input and at
the first entrance into the recurrent middle:

```text
VOCAB_MOE_LAYERS=input,loop_first
```

This submission is not presented as a state-of-the-art leaderboard record. It is
intended to be an auditable non-record/art submission showing a novel architecture
lane, the best score we obtained under limited 1xH100 testing, and the negative
results that shaped the design.

The broader experiment repository, including project notes, matrix runners,
H100 launch tooling, and design ledgers, is public here:

```text
https://github.com/corbensorenson/parameter-golf-experiments
```

## Best Preserved Result

The best preserved legal run at PR-open time was a 10-minute 1xH100 RunPod
scout:

| Candidate | BPB | Steps | Step speed | Total artifact bytes |
|---|---:|---:|---:|---:|
| `h100_batch32k_d704e832_w2200_q8_coreattn1_lqer10t20_vocabmoe_qk55` | `1.35692129` | `5018` | `119.57 ms` | `15,658,145` |

This artifact is under the decimal 16,000,000 byte cap by 341,855 bytes.

After the PR was opened, I was able to self-fund roughly one hour of 8xH100
RunPod time. I had applied for compute support but had not received any email
response by the deadline window, so the 8x results below are what I could test
within that narrow paid window.

The best completed under-cap 8xH100 row preserved so far is:

| Candidate | Final export BPB | Train-time val BPB | Steps | Step speed | Total artifact bytes |
|---|---:|---:|---:|---:|---:|
| `final8x_legal_196k_r2_d704e768_w2200_wd02_lqer6t12_vocabmoe_qk55` | `1.35496419` | `1.3191` | `6658` | `90.13 ms` | `15,989,749` |

This row is under the decimal cap by only 10,251 bytes. It should still be read as
non-record/art evidence, not as a fully tuned official-record attempt: the 8x
search budget was approximately one hour total, not a multi-seed or
grant-supported sweep.

Important limitations:

- This is a 1xH100 result, not an official 8xH100 result.
- The 8xH100 follow-up was self-funded and limited to about one hour of wall
  time; no grant response had arrived by the time this was run.
- The raw RunPod stdout from the strongest queue was lost when the pod was
  interrupted after the wallet ran out of funds.
- `train.log` therefore contains the preserved audit note for the best row, not
  a full raw training transcript.
- The submission is intentionally placed in `track_non_record_16mb`.

Nothing here relies on hidden training data, validation-set training, network
access during evaluation, or a target-conditioned decoding trick. The claim is
only that this architecture family produced the reported 1xH100 scout result in
our local/RunPod experiments.

## Architecture In Plain Terms

### 1. Mirrored IO Tail

The model does not run a flat sequence such as `0,1,2,...,N`. Instead, it enters
through three token-facing blocks and exits by reusing those same blocks in
reverse order:

```text
entry: 0 1 2
exit:  2 1 0
```

The motivation is parameter efficiency. The same blocks learn both read-in and
write-out roles, while small route and pass embeddings tell them where they are
being used.

### 2. Looped Middle

The semantic middle uses five physical blocks, then runs them twice:

```text
3 4 5 6 7  |  3 4 5 6 7
```

This creates more virtual depth without adding another full set of parameters.
In this best run, `r=2` was better than `r=3`. More recurrence was not
automatically better; it slowed training and gave a worse 1xH100 score in the
rows we tested.

### 3. One Attention Block In The Core

Only the first middle block keeps attention enabled:

```text
core attention block: 3
MLP-only recurrent blocks: 4, 5, 6, 7
```

This was a speed/quality compromise. Fully attentional recurrence was expensive.
An all-MLP middle was faster but less expressive. A single attention-capable
entry block gave the recurrent core one token-mixing point while keeping the
repeated middle relatively cheap.

### 4. LexLoRE: Token-Conditioned Low-Rank Experts

LexLoRE is the experiment originally named `VocabMoE` in the code. It is not a
large sparse MoE with separate full transformer experts. It is a small bank of
token-conditioned low-rank residual adapters:

```text
VOCAB_MOE_ENABLED=1
VOCAB_MOE_EXPERTS=16
VOCAB_MOE_RANK=2
VOCAB_MOE_LAYERS=input,loop_first
```

Each token gets a lightweight lexical correction vector. The same expert bank is
reused at multiple route sites with site-specific scale and bias terms.

The architectural hypothesis is that a very small model benefits from cheap
lexical specialization at the token interface and at the recurrent-core entry,
without paying for a much larger vocabulary embedding or a full MoE layer.

### 5. Factored Tied Embeddings

The best row uses:

```text
MODEL_DIM=704
FACTORED_EMBED_DIM=832
```

The embedding/output interface is factored and tied so that the model can spend
capacity on the token interface without paying the full dense `vocab_size x
model_dim` cost twice. This mattered because the submission is trying to stay
near the 16MB artifact limit while still having enough lexical capacity.

### 6. Train-Time Quantized Forward

The model trains through the quantized forward path from the beginning:

```text
TRAIN_QUANT_FORWARD=1
TRAIN_QUANT_EMBEDDINGS=1
QUANT_WEIGHT_BITS=8
VOCAB_MOE_TRAIN_QUANT_BITS=8
```

This is different from training a full-precision model and quantizing only at
export. The reason is that recurrence can amplify quantization mismatch. If the
scored artifact will use quantized weights, the recurrent blocks should learn
under that precision regime during training.

For this best 16MB row, the chosen precision target is q8. Earlier sub-4MB
experiments used more aggressive q4/ternary paths, but those models lost too
much quality for this 16MB art submission goal.

### 7. LQER Export Repair

`LQER` means low-rank quantization error repair. It stores a small low-rank
correction for the largest export-time quantization errors:

```text
LQER_RANK=10
LQER_TOP_K=20
```

LQER is not extra training data or evaluation-time adaptation. It is part of the
self-contained model artifact.

### 8. QK Gain

The attention block uses a larger query/key gain:

```text
QK_GAIN_INIT=5.5
```

This was borrowed from the broader Parameter Golf leaderboard trend, where
larger QK gain often improved small transformer behavior. Here it is applied to
the single attention-capable core-entry block.

## Exact Best-Run Configuration

The reproduction script is `run_1xh100_best.sh`. The important settings are:

```bash
MODEL_FAMILY=hrc
MODEL_DIM=704
NUM_UNIQUE_BLOCKS=8
EFFECTIVE_DEPTH=16
FACTORED_EMBED_DIM=832

HRC_DEPTH_SCHEDULE_MODE=transition_recursive_cycle
HRC_RECURSIVE_CORE_START=3
HRC_ROUTE_REPEATS=2
HRC_MLP_ONLY_BLOCKS=4,5,6,7
HRC_LOOP_INDEX_ENABLED=1
HRC_PASS_EMBED_ENABLED=1
HRC_PASS_EMBED_MODE=block_peer
HRC_PASS_ROLE_MODE=phase5
HRC_DEPTH_ADAPTER_TIE_MODE=block
HRC_RECUR_INJECT_ENABLED=1

TRAIN_BATCH_TOKENS=32768
TRAIN_SEQ_LEN=1024
WARMDOWN_ITERS=2200
MUON_WEIGHT_DECAY=0.0

TRAIN_QUANT_FORWARD=1
TRAIN_QUANT_EMBEDDINGS=1
QUANT_WEIGHT_BITS=8
MODEL_CODEC=lzma

VOCAB_MOE_ENABLED=1
VOCAB_MOE_LAYERS=input,loop_first
VOCAB_MOE_EXPERTS=16
VOCAB_MOE_RANK=2
VOCAB_MOE_TRAIN_QUANT_BITS=8

QK_GAIN_INIT=5.5
LQER_RANK=10
LQER_TOP_K=20
SUBMISSION_SIZE_CAP_BYTES=16000000
```

## Implementation Map

The code uses environment flags rather than a separate config file. The main
implementation points in `train_gpt.py` are:

- HRC route construction: `build_hrc_route_metadata` and
  `build_hrc_route_package`.
- LexLoRE / VocabMoE adapter: `VocabMoELite`.
- Train-time quantized forward assignment: `configure_train_quant_forward`.
- Factored tied embedding interface: `GPT.__init__`, controlled by
  `FACTORED_EMBED_DIM`.
- Export-time LQER payload: `pack_lqer_residual` and the final model export
  path.

The old project flag names are kept in the script for reproducibility. In this
README, `LexLoRE` refers to the feature controlled by `VOCAB_MOE_*`, and
`MirrorLoop HRC` refers to `MODEL_FAMILY=hrc` plus the `HRC_*` route flags.

## What We Tried And What Changed The Design

The useful signal from the project was not one magic setting. It was a sequence
of narrowing experiments.

| Experiment | Result | What it taught us |
|---|---:|---|
| Sub-4MB ternary/q4 family | Much faster, but far worse BPB | Extremely small artifacts were interesting but too capacity-limited for the main result. |
| Wider 16MB q8 family | Better quality | The architecture needed to spend bytes on width and embeddings once the target became quality. |
| Recurrent middle with `r=3` | `1.37374423` BPB on the comparable 32k legal row | More repeated compute was not automatically better; `r=2` was the best preserved legal setting. |
| 24k batch row | `1.35552525` BPB but `16,292,969` bytes | It was the best raw BPB, but not legal under the decimal cap. |
| 16k batch row | `1.36157540` BPB and over cap | More steps alone did not win; batch/update shape mattered. |
| 32k batch row | `1.35692129` BPB and `15,658,145` bytes | Best preserved legal 1xH100 row. |

This is why the submitted row keeps `r=2`, q8 training, a 32k token batch, a
single attention-capable core-entry block, and LQER r10/t20.

## Relation To Known Parameter Golf Ideas

This submission intentionally uses some ideas that are common in strong
Parameter Golf entries:

- quantized model artifacts,
- larger token vocabulary / token-facing capacity,
- depth recurrence,
- tied or factored embeddings,
- QK gain,
- low-rank quantization repair,
- artifact-size-aware hyperparameter tuning.

The novel part is the way these ideas are arranged:

- the same IO blocks are used on both entry and mirrored exit;
- the recurrent middle is an explicit route, not just extra layers;
- the lexical expert adapters are small low-rank residuals rather than full MoE
  experts;
- the lexical adapters are placed at the token interface and the loop entry;
- the quantized forward path is used from step 0 so the recurrent route learns
  under the same precision regime used by the artifact.

In short: this is not a copy of the standard SP8192 leaderboard transformer. It
is a compact mirrored-recurrent route with lexical low-rank steering.

## Reproduction

This record folder includes:

- `train_gpt.py`: the self-contained training script for the submission.
- `ternary_golf/`: helper layers used by the trainer.
- `run_1xh100_best.sh`: exact 1xH100 command for the best preserved row.
- `submission.json`: metadata for the non-record submission.
- `train.log`: preserved result note and caveats.
- `logs/8xh100_runpod_final8x_20260430_185628/`: live 8xH100 RunPod log
  snapshot copied off the pod during the first official-shaped 8x attempt.
- `logs/8xh100_runpod_final8x_20260430_185628_completed1/`: completed first
  8xH100 row, showing the e832 row used the cluster well but exceeded the
  decimal artifact cap after export.
- `logs/8xh100_runpod_legalfallback_20260430_191032_completed1/`: completed
  first legal-size e768 fallback row from the one-hour 8xH100 window.
- `logs/8xh100_runpod_legalfallback_20260430_191032_completed2/`: completed
  first two legal-size e768 fallback rows, including the current best
  under-cap 8xH100 row.

Expected inputs:

- `DATA_PATH`: the lossless CaseOps/SP8192 dataset directory used in the
  experiments.
- `TOKENIZER_PATH`: the matching CaseOps/SP8192 SentencePiece tokenizer.

Run:

```bash
bash run_1xh100_best.sh
```

The script is written for the RunPod Parameter Golf image and a single H100.
An official-shaped 8xH100 follow-up would use the same architecture constants
with a distributed launch and a re-tuned global batch schedule.

## 8xH100 Log Snapshot

After this PR was opened, an 8xH100 RunPod became available. The first
official-shaped matrix was launched with the same no-fetch bundle described in
the project notes. A live snapshot of that run is included under:

```text
logs/8xh100_runpod_final8x_20260430_185628/
```

That snapshot includes:

- the full live runner log at the time it was copied off the pod;
- the candidate plan for the five-row 8x matrix;
- the first candidate's per-run log snapshot;
- a `snapshot-status.txt` file with GPU utilization and timestamp.

It is intentionally labeled as a snapshot because the matrix was still running
when those files were first preserved. Completed 8x results should be added as
another update if the pod finishes before the submission can no longer be
edited.

The first completed 8x row is preserved under:

```text
logs/8xh100_runpod_final8x_20260430_185628_completed1/
```

That result was:

| Candidate | Final export BPB | Train-time val BPB | Steps | Step avg | Artifact bytes |
|---|---:|---:|---:|---:|---:|
| `final8x_196k_r2_d704e832_w2200_wd02_lqer8t16_vocabmoe_qk55` | `1.35704747` | `1.3174` | `6628` | `90.54 ms` | `16,413,081` |

This row is useful evidence but not a legal under-cap result. It exceeded the
decimal cap by `413,081` bytes. The important signal is that 8xH100 training
used the GPUs well and reduced train-time validation BPB, but the export gap and
compressed artifact size remained the binding constraint. The follow-up queue
therefore moved to `FACTORED_EMBED_DIM=768` legalizer rows instead of continuing
larger LQER/e832 variants.

The first completed under-cap 8x legalizer row is preserved under:

```text
logs/8xh100_runpod_legalfallback_20260430_191032_completed1/
```

That result was:

| Candidate | Final export BPB | Train-time val BPB | Steps | Step avg | Artifact bytes |
|---|---:|---:|---:|---:|---:|
| `final8x_legal_196k_r2_d704e768_w2200_wd02_lqer8t16_vocabmoe_qk55` | `1.35536174` | `1.3158` | `6655` | `90.17 ms` | `15,803,789` |

This is the best completed under-cap 8xH100 evidence currently included in this
submission. It came from the self-funded one-hour 8x window described above.

The second completed legalizer row improved the final export score and nearly
filled the artifact cap. It is preserved under:

```text
logs/8xh100_runpod_legalfallback_20260430_191032_completed2/
```

| Candidate | Final export BPB | Train-time val BPB | Steps | Step avg | Artifact bytes |
|---|---:|---:|---:|---:|---:|
| `final8x_legal_196k_r2_d704e768_w2200_wd02_lqer6t12_vocabmoe_qk55` | `1.35496419` | `1.3191` | `6658` | `90.13 ms` | `15,989,749` |

This is the best completed under-cap 8xH100 evidence currently included in this
submission.

## Validity And Caveats

This submission is deliberately explicit about what is and is not being claimed.

Claimed:

- a self-contained non-record/art submission;
- a best preserved legal 1xH100 scout result of `1.35692129` BPB;
- a best preserved legal 8xH100 one-hour-window result of `1.35496419` BPB;
- an artifact-size estimate of `15,658,145` bytes for that row;
- a novel mirrored-recurrent / lexical-low-rank architecture family.

Not claimed:

- official 8xH100 record eligibility;
- state-of-the-art leaderboard performance;
- statistical significance over multiple seeds;
- a full raw stdout log for the best 1xH100 run.

The raw log limitation is included because the RunPod pod became unavailable
after a billing interruption. The result was preserved in the project audit
notes before this PR was prepared, but the full terminal transcript was not.

## Why Submit This As Non-Record Art

The competition encouraged unusual approaches, including non-record submissions.
This architecture is submitted in that spirit. It did not beat the current
leaderboard, and it has not yet been run on 8xH100, but it does provide a
working, auditable example of a different route through the design space:

```text
mirrored IO reuse + recurrent middle + lexical low-rank steering + train-time quantization
```

The most important takeaway is negative as well as positive: simply adding more
recurrence or more steps did not solve the problem. The best row came from a
specific balance between batch size, q8 train-time quantization, one attention
mixing point, factored embeddings, and low-rank export repair.
