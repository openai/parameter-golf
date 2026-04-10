# Shallow Blue: Exact Memory Probe Under BOS-Reset Eval

## Summary

This folder packages the actual Shallow Blue submission surface, not just the
backbone.

The submission keeps the standard `9L/512d` SP-1024 backbone and adds a
document-local exact-memory scorer at evaluation time:

- `NGRAM` expert: exact causal `3`-gram with `top_k=3`, `min_support=2`
- `EXACT_REPEAT` expert: bounded exact local repeat, match length `4-8`
- `NGRAM_SAFE_PROBE` deploy lane: a compact probe routes between a safe n-gram
  floor and a boosted fixed-alpha mix
- evaluation regime: BOS-delimited documents, `window=1024`, `stride=1024`

The key result is that the exact-memory mechanism survives the stricter
non-overlap BOS-reset regime and becomes more valuable there because the
backbone loses overlapping-window help while document-local exact memory keeps
the same within-document history.

## What This Folder Runs

`train_gpt.py` still trains the backbone under the normal `10`-minute
competition wallclock cap. After export it does two evaluations on the
quantized roundtrip artifact:

1. plain backbone roundtrip validation
2. folder-local Shallow Blue deploy evaluation on the exact same quantized model

The final competition-facing score from this folder is the
`final_shallow_blue_probe_exact` line printed by `train_gpt.py`.

The plain `final_int8_zlib_roundtrip_exact` line is kept as a reference-only
backbone control. It is not the submission score for this method, because the
method being submitted is the backbone plus the legal Shallow Blue evaluation-time
mechanism. The official rules explicitly allow custom evaluation as long as it
is self-contained, does not access forbidden data, and stays under the
evaluation time limit.

## Frozen Deploy Lane

The packaged evaluator is intentionally narrow:

- no static precomputed tables
- no research overlays
- no tilt surface
- no higher-order sweep
- no adaptive policy search at runtime

It freezes the proved lane:

- `NGRAM 3-3`
- `min_support=2`
- `EXACT_REPEAT 4-8`
- probe artifact: `shallow_blue_probe.json`
- fixed `alpha=0.30`
- BOS-reset non-overlap scoring

This is the lane that already cleared the BOS-reset ship gate.

## Reference Gate Result

The current packaged default was rechecked with an artifact-backed alpha sweep
on `2026-04-08`, using the same quantized checkpoint and deploy policy on the
full `50k` validation documents under the BOS-reset non-overlap regime:

- plain roundtrip artifact control: `val_loss=2.07145173`, `val_bpb=1.22683039`
- baseline BPB: `1.20567221`
- `NGRAM_SAFE` BPB: `1.20346535`
- `NGRAM_SAFE_PROBE` BPB: `1.19948112`
- probe delta vs baseline: `-0.00619109`
- bits saved: `934994.00`

By document length:

- `<512`: slight drag
- `512-2047`: slight gain
- `>=2048`: strong gain (`ΔBPB=-0.01192902`)

That is the scientific reason this submission exists: the exact-memory probe is
not a generic booster everywhere; it is a strong long-document mechanism.

## Size Budget

This folder is built to stay under the `16MB` artifact limit with:

- compressed int8+zlib model artifact
- `train_gpt.py`
- `shallow_blue_submission_eval.py`
- `shallow_blue_probe_runtime.py`
- `shallow_blue_probe.json`

The latest artifact-backed rerun stayed under the `16MB` limit at
`15.933MB` total:

- compressed model artifact: `15,823,116` bytes
- counted code: `92,939` bytes
- probe artifact: `16,982` bytes

## Run

From the repo root:

```bash
RUN_ID=shallow_blue_record_submission_20260407 MAX_WALLCLOCK_SECONDS=595 NPROC_PER_NODE=8 bash scripts/run_shallow_blue_record_submission.sh
```

After the run:

```bash
python3 scripts/print_shallow_blue_record_submission_summary.py
```

## Included Files

- `train_gpt.py`: backbone trainer plus final Shallow Blue deploy evaluation
- `shallow_blue_submission_eval.py`: folder-local BOS-reset deploy scorer
- `shallow_blue_probe_runtime.py`: probe artifact loader and live feature builder
- `shallow_blue_probe.json`: frozen two-level uplift probe artifact
- `train.log`: backbone training reference log used during packaging
- `submission.json`: metadata for the current packaged lane
- `requirements.txt`
