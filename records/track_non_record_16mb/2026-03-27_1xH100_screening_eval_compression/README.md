This folder captures a log-backed `1xH100` screening bundle for non-record submission.

It is not a leaderboard claim and is not intended to satisfy the main `8xH100 / 10-minute` track. The goal of this bundle is to document the March 26 screening matrix that we used to decide where additional compute should go next: quantization-friendly training and evaluation strategy, not just larger dense models.

## Canonical Result

The canonical metric in `submission.json` is `B0`, the checked dense baseline screen:

- post-quant `val_bpb`: `1.35375147`
- pre-quant `val_bpb`: `1.3521`
- wallclock: `600.189s`
- eval time: `11.070s`
- total artifact size: `12,783,154` bytes

This is a `1xH100` result and is included here as a screening anchor, not as a competitive submission against the public `8xH100` leaderboard.

## Screening Matrix

All rows below are backed by raw logs included in this folder.

| ID | Family | Params | Pre-quant BPB | Post-quant BPB | Quant gap | Bytes total | Key point |
|---|---|---:|---:|---:|---:|---:|---|
| `B0` | dense baseline | 17,059,912 | 1.3521 | 1.3538 | +0.0017 | 12,783,154 | reference dense baseline |
| `Q1` | fp16-embed family | 16,765,000 | 1.3525 | 1.3569 | +0.0044 | 11,574,110 | smaller artifact, near-baseline post-quant |
| `Q3` | fp16-embed family | 16,765,000 | 1.4463 | 1.5543 | +0.1080 | 8,726,877 | aggressive quant-friendly tuning overshot badly |
| `C1` | 10L mixed-precision family | 18,897,488 | 1.3408 | 1.3704 | +0.0296 | 10,744,314 | best pre-quant result in this screen, but quantization erased the gain |
| `C2` | 10L mixed-precision family | 18,897,488 | 1.3872 | 1.6530 | +0.2658 | 8,162,470 | compression got much smaller, but quality collapsed |

## Main Finding

The dominant screening result from this matrix is that `pre-quant quality` and `post-quant quality` diverge sharply once we push capacity or compression too hard.

- `C1` beat `B0` before quantization (`1.3408` vs `1.3521`) but lost after roundtrip (`1.3704` vs `1.3538`).
- `Q1` stayed close to the baseline while producing a meaningfully smaller artifact.
- `Q3` and `C2` show that making the model easier to compress is not enough if the resulting weight distribution is too fragile after quantization.

The practical takeaway is that the next compute should go to:

1. evaluation strategy, especially rerunning the missing-log sliding-window candidate
2. quantization-aware schedule tuning and compression-aware training
3. only then broader `8xH100` validation on the best 2-3 candidates

## What Is Deliberately Excluded

We observed a later `1xH100` summary showing a faster matched baseline around `1.3411` and a stride-64 sliding-window eval around `1.3075`, but the raw logs for that later pod were not preserved. Those results motivated the next experiment plan, but they are intentionally excluded from this public bundle.

## Script Provenance

The exact March 26 source snapshots were not archived separately for every screen. The included scripts are the surviving script families that match the logged experiments:

- `train_gpt.py`: nearest surviving dense-baseline-family snapshot, copied from `records/track_10min_16mb/2026-03-17_NaiveBaseline/`
- `train_gpt_fp16_embed.py`: matching fp16-embedding family snapshot, copied from `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
- `train_gpt_10l_mixed_precision.py`: matching 10-layer mixed-precision family snapshot, copied from `records/track_10min_16mb/2026-03-19_10L_MixedPrecision/`

These scripts are included to document the screening families and keep the bundle self-contained. The authoritative evidence for this folder is the raw logs.

## Included Files

- `README.md`
- `submission.json`
- `B0.log`, `Q1.log`, `Q3.log`, `C1.log`, `C2.log`
- `train_gpt.py`
- `train_gpt_fp16_embed.py`
- `train_gpt_10l_mixed_precision.py`
