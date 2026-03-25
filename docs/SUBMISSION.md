# Non-record PR checklist (Parameter Golf + BESE)

Use this when submitting to [openai/parameter-golf](https://github.com/openai/parameter-golf) on the **non-record** track with the BESE+BPE tokenizer.

## Before you open the PR

1. **Fork** the upstream repo on GitHub (if you have not already).
2. **Branch** from `main` with a descriptive name, e.g. `bese-bpe-tokenizer`.
3. **Copy or submodule** this repo’s `tokenizer/` into your fork (or document `BESE_TOKENIZER_ROOT`).
4. **Data:** Re-export FineWeb shards with `scripts/export_shards.py` using the **same** tokenizer JSON you pass to training.
5. **BPB sanity:** Run byte checks (see [integration_guide.md](integration_guide.md)) on a sample of validation text.

## Files to include

| Artifact | Purpose |
|----------|---------|
| `tokenizer/bese_constants.py`, `bese_tokenizer.py`, `bese_bpe_tokenizer.py` | BESE implementation |
| `scripts/export_shards.py`, `scripts/train_bpe_jsonl.py` | Reproducible data + tokenizer training |
| `integration/train_gpt_bese.py` | Training entrypoint with `.json` tokenizer support |
| `integration/README.md` | How to set `TOKENIZER_PATH`, `VOCAB_SIZE`, `BESE_TOKENIZER_ROOT` |

You may instead **patch** upstream `train_gpt.py` with the same tokenizer-loading logic as `integration/train_gpt_bese.py` (keep under 1500 lines).

## PR description (suggested sections)

1. **Summary:** BESE two-layer tokenizer (38 base + BPE merges) to shrink embedding table and fund extra layers.
2. **Motivation:** Parameter savings vs SP1024 (~295 KB Int6 embeddings at ~250 merges).
3. **BPB:** Explain bytes-per-token LUT; confirm validation matches UTF-8 byte counts.
4. **Training:** Dataset suffix, `VOCAB_SIZE`, merge count, hardware (e.g. 8×H100).
5. **Results:** Best val_bpb (and comparison to baseline if available).
6. **License:** MIT (per challenge).

## Review expectations

OpenAI notes that **tokenizer changes get extra scrutiny**. Keep the byte accounting transparent and reproducible.
