# Non-Record Submission: BigramHash Upper Enrichment on 1xH100

This is a non-record submission documenting a focused sweep around a simple idea: spend the spare 16 MB artifact budget on a learned hashed local-memory table, then improve the hash address with cheap tokenizer-derived bits.

The best completed run uses an additive BigramHash table with 12,288 buckets, injected at transformer layer 1, with a one-bit `starts-with-uppercase` hash enrichment. It reached **1.31321205 val_bpb** after int8+zlib roundtrip on **1xH100** in a 600 second wall-clock run.

This is not a leaderboard-record claim. It was not run on 8xH100 SXM, it is a single-seed result, and it does not beat the current 8xH100 baseline. It is included as an honest non-record contribution because the sweep gives a clear signal about where this family of ideas helped and where it did not.

## Key Result

| Run | Hardware | Seed | Steps | Pre-quant val_bpb | Post-quant val_bpb | Total bytes |
|-----|----------|------|-------|-------------------|--------------------|-------------|
| BigramHash 12288 + `HASH_ENRICH=upper` | 1xH100 80GB | 1337 | 1430 | 1.3122 | 1.31321205 | 15,343,280 |

Compared with the local 1xH100 stock baseline from the same experiment log, this improves post-quant validation BPB from `1.3448` to `1.3132`, a gain of about `0.0316` BPB under the same 600 second single-GPU constraint.

## What Changed

The starting point was the stock 9 layer, 512 dim, 1024 vocab baseline. The best run adds:

- `USE_BIGRAM_HASH=1`
- `BIGRAM_TABLE_SIZE=12288`
- `BIGRAM_INJECT_LAYER=1`
- `HASH_ENRICH=upper`

The BigramHash path computes a learned residual feature from the previous and current token ids. `HASH_ENRICH=upper` folds a token-local uppercase flag into the hash key. This keeps the same hot-path module shape but partitions collisions in a way that helped more than simply using a larger table or adding extra feature modules.

## Sweep Summary

The main sweep is in `results.tsv`. The important pattern was:

- Plain 12,288-bucket BigramHash reached `1.3197` post-quant BPB.
- `HASH_ENRICH=space` improved that to `1.3159`.
- `HASH_ENRICH=upper` improved it further to `1.3132`.
- More complicated variants were worse: `space_upper`, `new_word`, count initialization, later reinjection, skip-bigram, trigram hashing, and higher LR all regressed.

That suggests the useful contribution was not "more n-gram machinery everywhere"; it was a very small, cheap improvement to the hash address.

## Reproduction

From this record folder, point the script at the repo-level data and tokenizer:

```bash
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
RUN_ID=bigram_upper_seed1337 \
SEED=1337 \
USE_BIGRAM_HASH=1 \
BIGRAM_TABLE_SIZE=12288 \
BIGRAM_INJECT_LAYER=1 \
HASH_ENRICH=upper \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

The submitted `train.log` is the complete captured output from the canonical run, including the script snapshot, environment, training progress, size accounting, and final int8+zlib roundtrip validation.

## Included Files

- `train_gpt.py`: exact training script snapshot used for the BigramHash sweep.
- `train.log`: canonical 1xH100 run for `HASH_ENRICH=upper`.
- `results.tsv`: compact table of the completed local sweep.
- `submission.json`: metadata for the non-record submission.
- `requirements.txt`: Python package requirements matching the repo baseline.

## Hardware and Submission Note

An 8 GPU Runpod attempt was prepared on 2026-04-30 using 8x RTX PRO 6000, but the instance was reclaimed before a completed training result was available. No score from that interrupted run is claimed here.
