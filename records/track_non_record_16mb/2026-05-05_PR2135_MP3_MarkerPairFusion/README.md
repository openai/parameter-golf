# Non-record: PR #2135 stack + MP3 marker-pair fusion

Post-deadline note documenting that the MP3 marker-pair fusion from our prior PR #2109 composes with the May 1 frontier stack (PR #2135). Filed under `track_non_record_16mb` for reference only — not eligible for chronological standing.

## Numbers

3-seed mean of (42, 0, 314), phased TTT, on canonical CaseOps SP8192 (val byte sum identical to canonical within 0.015 %, see `submission.json` note):

| Seed | Train | Pre-quant | Quant | **Post-TTT** | Eval s | Artifact |
|-----:|------:|----------:|------:|-------------:|-------:|---------:|
| 42   | 599.6 s | 1.05887 | 1.06741 | **1.05577** | 473.6 | 15,946,755 |
| 0    | 599.6 s | 1.05858 | 1.06706 | **1.05534** | 501.9 | 15,936,375 |
| 314  | 599.5 s | 1.05862 | 1.06726 | **1.05564** | 460.1 | 15,939,487 |
| **Mean** | | **1.05869** | **1.06724** | **1.05559** (std 0.00018) | | max 15,946,755 |

For reference, PR #2135's reported 3-seed mean is 1.05651 (std 0.00036); a matched-environment vanilla rerun in our setup produced 1.05669 (std 0.00041), so most of the apparent gap is attributable to the fusion lever rather than environment drift.

## What changed vs PR #2135

Single addition: the three 2-grams `[▁,TITLE]` / `[▁,ALLCAPS]` / `[▁,CAPNEXT]` are fused into single alias donor tokens (donors 8/9/10, byte-fallback IDs that occur 0× in the CaseOps corpus). Tokenizer is unchanged; the transform is a stream-edit on already-tokenized shards. Each alias row gets a norm-matched warm-init `0.4·E[▁] + 0.6·E[marker]` once at training start. PR #2135's training/eval code path is otherwise identical (`SmearGate` dampening branch is present in the patched `train_gpt.py` but hard-disabled via `alias_dampening_active=False`).

Patch size: +45 lines vs upstream PR #2135 `train_gpt.py`. Patched md5 `c90dba41e5ce9586871a05e94e6e7445`.

> Note on the log line `marker_pair:smear_dampening prev_alias_scale=0.5 num_pairs=3`: this is a config dump of `ALIAS_PREV_SMEAR_SCALE`'s default value (the env var is unused in this submission). The actual SmearGate forward path remains byte-identical to PR #2135 vanilla because `alias_dampening_active=False` is hard-coded in `train_model()` (line 3994). Verifiable by reading the patched source.

## Reproduction

Self-contained pipeline from the official FineWeb-10B doc stream (no third-party data mirrors):

```bash
# 1. Fetch the canonical FineWeb-10B doc stream (~45 GiB) from the official
#    willdepueoai/parameter-golf HF dataset.
python3 download_docs.py

# 2. Apply CaseOps tokenization (canonical caseops_v1_reserved).
python3 prepare_caseops_data.py \
    --docs ./data/datasets/docs_selected.jsonl \
    --out  ./data \
    --sp   ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model

# 3. Apply MP3 marker-pair stream-edit on the already-tokenized shards.
python3 prepare_marker_pair_v3.py

# 4. 3-seed run (builds libonline_ngram_state.so on first invocation).
bash run_3seed.sh
```

`run_3seed.sh` adds only `MARKER_PAIR_MODE=1`, `MARKER_PAIR_W_SPACE=0.4`, `MARKER_PAIR_W_TITLE=0.6` and `DATA_PATH=...marker_pair_v3` to PR #2135's repro env block; all other env vars are inherited from PR #2135 (`SAG_SCALE=0.5`, `NUM_PHASES=1`, `EVAL_SEQ_LEN=2560`, `GPTQ_CALIBRATION_BATCHES=32`, etc.).

## Compliance

- C1 strict causal dependence: standard sliding-window scoring with cu_seqlens packed-doc handling. PR #2135 BOS-fixed SmearGate inherited.
- C2 full normalized softmax over SP8192 vocab.
- C3 score-before-update: phased TTT (NUM_PHASES=1, prefix_docs=2500) inherited from PR #2135.
- C4 single left-to-right pass.
- No SLOT, no logit bias beyond the inherited PR #2135 token-only n-gram tilt, no pre-quant TTT on val data, no PPM mixture.
- Full validation set; val byte sum identical to canonical caseops_v1_reserved within 0.015 % (151,546,741 vs 151,568,749).
- Artifact ≤ 16,000,000 bytes (max 15,946,755), train ≤ 600 s strict, eval ≤ 600 s.
