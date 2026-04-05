## What this is

An audit of PR #1184's Scylla tokenizer byte accounting. I ran their exact
code with corrected `candidate.meta.npz` and proper val data. The result:
**1.1289 BPB, not 0.9485.**

The sub-1.0 claim was a measurement error.

## The bug

PR #1184's `candidate.meta.npz` has 27 byte-fallback tokens (IDs 75-101)
with `base_bytes=3` instead of 1. These tokens represent single raw bytes
but are counted as 3 bytes each. This overcounts the byte denominator in
the BPP formula, making the score look ~4% better than it actually is.

This was originally flagged by @dexhunter on PR #1143 (the earlier Scylla
submission, which was closed for exactly this reason). PR #1184 reuses the
same buggy `candidate.meta.npz`.

## My test

I ran PR #1184's **exact, unmodified `train_gpt.py`** with every env var
matching their README:

```bash
VOCAB_SIZE=998 XSA_LAST_N=11 USE_GPTQ=1 GPTQ_RESERVE_MS=9000
BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=112 TTT_ENABLED=0
```

The only change: a corrected `candidate.meta.npz` that fixes the 27
byte-fallback tokens from `base_bytes=3` to `base_bytes=1`. Everything else
is identical — same architecture, same optimizer, same GPTQ, same data.

I also retokenized the val shard directly from `docs_selected.jsonl` (no
SP1024 roundtrip) using the official split: shuffle with seed 1337, last
50K docs = val. This produced 62.6M val tokens, close to their 62.4M.

## Results

| | PR #1184 (buggy meta) | Me (corrected meta) |
|---|:---:|:---:|
| Val tokens | 62,363,648 | 62,609,408 |
| Val NLL | 1.928 | 1.916 |
| **Sliding BPB** | **0.9491** | **1.1289** |
| Train shards | 194 | 207 |
| Code | Their exact train_gpt.py | Same |

The NLL is nearly identical (1.928 vs 1.916 — my model is actually slightly
*better*). The entire 0.18 BPP gap comes from different byte accounting.

## Decomposing the gap

I decomposed the BPB formula `(NLL / ln2) × (tokens / bytes)` to isolate
what's driving the difference:

| Factor | BPB impact |
|--------|:---------:|
| Model quality (NLL difference) | +0.010 |
| **Byte accounting difference** | **+0.133** |
| Val text/token boundary differences | +0.037 |
| **Total** | **+0.180** |

**93% of the gap is byte accounting, not model quality.** The Scylla
tokenizer doesn't make the model predict better, the buggy meta just
makes the denominator bigger, which makes BPP smaller.

## What this means

With corrected accounting, the Scylla stack lands at ~1.13 BPB. This is
essentially the same as the SP1024 stack at ~1.11-1.12. The tokenizer
itself provides no meaningful advantage.

I'd like to flag PR #1184 for review.

## Corrected files included

- `correct_meta.npz` — fixes only the 27 byte-fallback tokens, leaves
  everything else unchanged (has_leading_space=0, is_boundary=0)
- `retokenize_proper.py` — retokenizes from raw `docs_selected.jsonl`
  with proper train/val split (shuffle seed 1337, last 50K = val)

## Reproducing

```bash
# Create corrected meta (only byte-fallback fix)
python3 -c "
import numpy as np
orig = np.load('candidate.meta.npz')
bb = orig['base_bytes'].copy()
for i in range(75, 102): bb[i] = 1
np.savez('correct_meta.npz', **{k: orig[k] for k in orig if k != 'base_bytes'}, base_bytes=bb)
"

# Retokenize from raw docs (proper split)
python3 retokenize_proper.py

# Train (PR #1184's exact code + corrected meta)
SEED=1337 VOCAB_SIZE=998 XSA_LAST_N=11 USE_GPTQ=1 GPTQ_RESERVE_MS=9000 \
BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=112 TTT_ENABLED=0 \
DATA_PATH=./fineweb_scylla TOKENIZER_PATH=./candidate.vocab \
TOKENIZER_META_PATH=./correct_meta.npz \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Request for review

@0hq @valerio-oai PR #1184 should be re-evaluated with corrected byte
accounting before being merged.