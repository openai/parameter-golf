# Non-Record: SP8192 + MuonEq-R + Loop@0.42 + RECUR_AB + QAT-lite + Compact Artifact

`val_bpb = 1.09960971` | `15,974,435` bytes | `8xH100 SXM`

## Summary

This submission packages the strongest **fully under-cap, under-time, rule-compliant** branch that came out of an April 2026 SP8192 research cycle focused on legal recurrence-native improvements.

The main research question was:

- can a legal score-first TTT stack still improve meaningfully through recurrence-side changes, without tokenizer tricks, SLOT, or other review-risky evaluation semantics?

The answer from this branch was yes, with the best signal coming from:

- `MuonEq-R`
- wallclock-aware depth recurrence activated at `ENABLE_LOOPING_AT=0.42`
- learned recurrent alpha/beta blending (`RECUR_AB`)
- late-stage `QAT-lite` on sensitive late `q/k` projections
- compact artifact engineering to bring the branch under the 16MB cap

This is a **non-record** submission because it does not beat the current SOTA, but it is intended as a clean, reproducible, legal, and novel implementation that documents the research path and preserves a working under-cap artifact.

## Final Result

Single seed: `1337`

| Stage | BPB | Notes |
|---|---:|---|
| Raw pre-quant | `1.1046` | final validation before GPTQ |
| Quantized | `1.1336` | GPTQ + Brotli |
| **TTT** | **`1.09960971`** | final score-first TTT result |

Artifact breakdown:

| Item | Bytes |
|---|---:|
| Quantized model + Brotli | `15,949,492` |
| Code | `24,943` |
| **Total** | **`15,974,435`** |

Timing:

| Phase | Time |
|---|---:|
| Train | `599.092s` |
| Quantized eval | `2.548s` |
| TTT eval | `544.199s` |

## Novel Techniques Used

The most important ideas in this submission are:

- **RECUR_AB recurrent blending**  
  Repeated loop visits do not simply reuse the same shared block output. Instead, each repeated recurrent site learns a small alpha/beta blend over the current activation and its cached recurrent state. This was the clearest recurrence-native gain beyond the plain looping baseline.

- **Wallclock-aware recurrence activation at `0.42`**  
  The branch uses a late recurrence curriculum keyed to training progress rather than running the repeated core from the beginning. The `0.42` activation point consistently beat earlier recurrence-on schedules in local ablations.

- **Targeted late `QAT-lite` on `q/k` projections**  
  The branch applies lightweight quantization-aware regularization only where it helped most in this architecture: late-layer `q/k` weights that mattered for the final compressed artifact.

- **Compact artifact engineering**  
  The final legal run combines small-control-tensor compression, compact GPTQ scale storage, and an LZMA code wrapper so the same 512d branch can fit under the hard `16,000,000` byte limit without shrinking the core model.

## Research Highlights

This branch is the result of a much broader local search. The most relevant outcomes were:

- `Loop@0.42` beat earlier recurrence activation schedules such as `0.35` and `0.40`.
- `RECUR_AB` beat both the plain recurrence stack and the earlier `RecurAlpha` variant.
- `XSA` produced mixed signals and did not survive the final ablations.
- broad `HQClip` gave the best quality-only result, but artifact size exploded and it was not practical for submission.
- `RECUR_LORA` and `AWQ-lite` did not survive quality/size tradeoff testing.
- simple compressor swaps such as `COMPRESSOR=lzma` were worse than Brotli on this artifact.
- small model shrink variants like `MODEL_DIM=496` got under cap but sacrificed too much BPB.

The final submission therefore keeps only the changes that were both:

- clearly legal under the current evaluation rules
- still useful after accounting for train time, eval time, and artifact size

## Why This Submission Is Interesting

Even though the final score is not record-level, this run is useful because it shows a complete end-to-end recipe for:

- legal score-first TTT
- recurrence-native architectural experimentation
- quantization-aware training on a looped model
- artifact-side engineering to convert an over-cap branch into an under-cap one

In other words, this PR is intended to show **signs of life for legal recurrence-side novelty**, not just another retuning of the merged baseline.

## Exact Reproduction

From a fresh pod:

```bash
cd /workspace
git clone -b submission/recurab-042-nonrecord https://github.com/ChideraIbe123/parameter-golf.git
cd parameter-golf

python3 -m pip install --upgrade pip
pip install numpy sentencepiece huggingface-hub brotli

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 10
```

Run the submission script:

```bash
SEED=1337 \
MUON_EQR=1 \
EMA_DECAY=0 \
ENABLE_LOOPING_AT=0.42 \
MAX_WALLCLOCK_SECONDS=599.0 \
RECUR_ALPHA_ENABLED=0 \
RECUR_AB_ENABLED=1 \
RECUR_A_INIT=1.0 \
RECUR_B_INIT=0.0 \
QAT_LITE_ENABLED=1 \
QAT_LITE_START_FRAC=0.55 \
QAT_LITE_EVERY=4 \
QAT_LITE_LAMBDA=0.02 \
QAT_LITE_BITS=6 \
QAT_LITE_CLIP_SIGMAS=12.85 \
QAT_LITE_LAYER_START=7 \
QAT_LITE_TARGETS=qk \
QAT_LITE_PENALTY=mse \
QAT_LITE_DEPTH_POWER=0.0 \
COMPRESSOR=brotli \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
torchrun --standalone --nproc_per_node=8 \
records/track_non_record_16mb/2026-04-27_SP8192_MuonEqR_Loop042_RecurAB_QATLite/train_gpt.py
```

## Compliance Notes

This submission is intended to satisfy Issue `#1017`:

- causal left-to-right dependence
- full normalized softmax distribution
- score-before-update TTT ordering
- single left-to-right pass with no rescoring

This is a **non-record** submission because:

- it is a single-seed result
- it does not beat the current record stack

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
