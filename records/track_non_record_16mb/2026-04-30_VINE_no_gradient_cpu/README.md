# VINE — Non-Record Submission

**Author:** Raychell Langan, NEXICOG Ltd
**Contact:** raychell.langan@gmail.com

---

## What this is

A non-record entry. A geometric AI predictor that compresses byte-level English text without using a transformer, without backpropagation, without gradient descent, and without a loss function. There are zero learned parameters. The FineWeb input fills statistical n-gram counts but is not "training data" in the usual sense — there is no training loop.

This is a minimal byte-prediction adaptation of the VINE architecture.

---

## Result

**val_bpb: 2.1728** on a 500,000-byte slice of FineWeb val shard 0 (offset 0).

Across five disjoint 50,000-byte slices, the same configuration ranges 1.85–2.41 BPB; mean 2.20, stdev 0.22. A single 50 KB slice is dominated by slice variance, not model behaviour, so the 500 KB number is the one to read.

End-to-end wall-clock on an AMD Ryzen 7 5700X3D 8-Core Processor: ~31 minutes (PPM build ~11 min + eval ~19 min).


---

## Properties

- No transformer
- No gradient descent, no backpropagation, no loss function
- Zero learned parameters; PPM byte-counts plus fixed structural priors
- CPU only; verified to run on BBC BASIC (1981) with integer-only arithmetic
- Deterministic — same input bytes give bit-identical output

---

## Why this is non-record

VINE has zero learned parameters and uses no gradient descent. The leaderboard's 10-minute / 16 MB envelope assumes a training loop that optimises weights against a loss; this submission has neither, so it sits in the non-record category.

The 500 KB headline `val_bpb: 2.1728` is what the system reaches using only embedded structural priors and PPM byte counts ingested from a 20-shard primer.

---

## Reproduction

`train_gpt.py` is self-contained — all data is base64-zlib-inlined, no separate artefacts. To reproduce the headline number, point it at a cached FineWeb sp1024 dataset and run it:

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
python train_gpt.py
```

It globs `fineweb_train_*.bin` and `fineweb_val_*.bin` from `DATA_PATH`, decodes through the SentencePiece model at `TOKENIZER_PATH`, builds, and prints `val_bpb`. End-to-end ~31 minutes on the CPU above.

Defaults match the headline run: 20 train shards, 2 M tokens per shard, PPM order 5, eval slice 500,000 bytes from offset 0 of val shard 0. Override via env vars if you want a different protocol:

| Variable | Default | Effect |
|---|---|---|
| `MAX_TRAIN_SHARDS` | 20 | Number of train shards to ingest |
| `MAX_TOKENS_PER_SHARD` | 2_000_000 | Token cap per shard |
| `VINE_MAX_ORDER` | 5 | PPM max n-gram order |
| `EVAL_BYTES` | 500_000 | Eval slice size |
| `EVAL_OFFSET` | 0 | Eval slice start offset within val |

Dependencies: `numpy`, `sentencepiece`. The reference baseline already requires both.

---

## Author

Raychell Langan — NEXICOG Ltd, Hampshire UK
[vineai.net/research/portfolio](https://vineai.net/research/portfolio/) · raychell.langan@gmail.com

See also: transformer decoupling — the mechanical cause of catastrophic forgetting and jailbreak susceptibility — at [vineai.net/research/portfolio](https://vineai.net/research/portfolio/) under the *Transformer Decoupling* bucket.
