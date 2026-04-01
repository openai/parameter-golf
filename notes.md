# Parameter Golf Notes

## Repo Rules

### Exact 16 MB Artifact Definition

- The hard cap is **decimal** `16,000,000` bytes, not `16 MiB`.
- The counted artifact is:
  - `len(train_gpt.py in UTF-8 bytes)`
  - plus the **compressed model bytes**
- The README is explicit that counted code should live in `train_gpt.py`.
- No network calls, dataset access, or external downloads are allowed during evaluation.
- Evaluation itself also has a separate runtime cap: under 10 minutes on `8xH100`.

Source:
- `README.md`, FAQ section and submission process.

## Exact Eval Metric

From `train_gpt.py`:

1. Validation loss is average token cross-entropy in **natural log units**:
   - `val_loss = sum(token_ce) / num_val_tokens`
2. Convert to bits per token:
   - `bits_per_token = val_loss / ln(2)`
3. Compute tokenizer-aware bytes represented by each target token:
   - base UTF-8 byte length of the SentencePiece token
   - plus one extra byte for a leading space when the piece starts with `▁` and the previous token is not a boundary token
4. Convert to tokens per byte:
   - `tokens_per_byte = num_val_tokens / num_val_bytes`
5. Final metric:
   - `val_bpb = bits_per_token * tokens_per_byte`

Important implementation detail:
- Baseline validation uses **non-overlapping** chunks of length `TRAIN_SEQ_LEN` (or `EVAL_SEQ_LEN` in some PRs).
- Many strong PRs improve score further with **sliding-window eval**, but the underlying `val_bpb` formula above stays the same.

Relevant code:
- `train_gpt.py`: `build_sentencepiece_luts`, `load_validation_tokens`, `eval_val`

## Baseline Architecture

Baseline config from README, baseline record, and root `train_gpt.py`:

- Decoder-only GPT
- `VOCAB_SIZE=1024`
- `NUM_LAYERS=9`
- `MODEL_DIM=512`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4` (GQA)
- `MLP_MULT=2` so FFN hidden size is `1024`
- tied input/output embeddings
- RoPE attention
- RMSNorm
- `relu^2` MLP
- zero-initialized attention and MLP output projections
- encoder/decoder-style skip reuse:
  - first half of blocks store skips
  - second half re-inject them in reverse order with learned skip weights

Exact baseline parameter count:

- `17,059,912` parameters

Baseline record metrics:

- Pre-quant `val_bpb`: `1.2172`
- Post-quant roundtrip `val_bpb`: `1.22436570`
- Artifact size: `15,863,489` bytes

Sources:
- `README.md`
- `records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md`
- `records/track_10min_16mb/2026-03-17_NaiveBaseline/train.log`

## Leaderboard Snapshot Seen

### Committed README Leaderboard

- `1.2060` — `2048 seq length`
- `1.2197` — `fp16 Embed`
- `1.2244` — `Naive Baseline`
- notable non-record:
  - `1.2074` — `4-Hour Baseline`

### Live PR State Pulled From GitHub API

README is already stale relative to open PRs. Strongest current public PRs I found are:

- `#106` — `1.15824056`
  - title: `record: 1.158`
  - branch indicates it is an exact-export follow-up to PR #88
- `#88` — `1.16050360`
  - `Int6 MLP3x + MTP + Sliding Window Eval`
- `#99` — `1.16050360`
  - `Int6 MLP3x + Late-K Passthrough + SlidingWindow`
- `#102` — `1.1618`
  - `Int6 MLP3x + Tuned LR + SmearGate + SlidingWindow`
- `#89` — `1.1622`
  - `NorMuon + int6 STE + SWA + sliding window`

Notes:

- I excluded obviously non-standard or special-case titles like the `val-only` PRs when summarizing the strongest standard submissions.
- The main repo leaderboard has not yet caught up to the live PR frontier.

## What Top PRs Already Tried

Across the strongest public submissions, the field is converging on a pretty clear recipe:

- **fp16 tied embedding passthrough**
  - embedding/head quantization is highly sensitive
- **mixed low-bit export**
  - especially `int6` for large matrices
- **artifact compression tricks**
  - mostly `zstd`-based in the top PRs
- **uniform 9x512 body with wider FFN**
  - especially `MLP 3x` instead of baseline `2x`
- **sliding-window eval**
  - commonly stride `64` or `512`
- **optimizer retuning**
  - lower LR, higher Muon momentum, longer warmdown
- **long-context training**
  - `TRAIN_SEQ_LEN=2048` and `4096`
- **MTP auxiliary loss**
  - small extra head, often excluded from exported artifact
- **late-stage quantization-awareness**
  - QAT or fake quant / STE
- **selective preservation of sensitive layers**
  - e.g. fp16 late `K` projections
- **SWA / EMA / NorMuon**
  - modest training-dynamics improvements
- **small architectural extras**
  - e.g. SmearGate

## Negative Results / Approaches Mentioned As Weak

From PR READMEs:

- depth recurrence / looped transformers often had:
  - worse throughput
  - larger quantization gaps
  - too few effective steps in 10 minutes
- SwiGLU often improved per-step quality but hurt total steps enough to lose overall
- MoE was too slow at this scale
- aggressive pruning often compounded quantization error
- per-group quantization sometimes bloated artifact size
- generic QAT helped less than hoped unless very carefully targeted

## Main Gap I See

Top public PRs are overwhelmingly using the same **uniform 9x512 backbone** and competing on:

- export format
- eval strategy
- minor training-dynamics tweaks

I did **not** find a top-5 PR using systematic **layer-wise / top-heavy parameter allocation** in the OpenELM sense, where later layers get wider FFNs than early layers at the same total parameter budget.

That looks like the cleanest open lane:

- theoretically motivated
- different from the current top cluster
- compatible with the proven low-bit + sliding-eval infrastructure
