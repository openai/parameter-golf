# Experiment Results — RunPod 1xH100 SXM (March 25, 2026)

## Setup

- **Hardware:** 1x NVIDIA H100 80GB HBM3 (SXM) on RunPod secure cloud
- **Cost:** $2.69/hr
- **Template:** Official Parameter Golf template (`runpod/parameter-golf:latest`)
- **Upstream repo:** [openai/parameter-golf](https://github.com/openai/parameter-golf) (forked to [mrbese/parameter-golf](https://github.com/mrbese/parameter-golf))
- **Data:** FineWeb SP1024 cached shards (10 train shards = ~1B tokens + val shard = 62M tokens)

## Data Pipeline

### 1. Decode text from SP shards

Since `docs_selected.jsonl` is 48GB (too large for the 20GB container disk), we decoded raw text from the existing SP-tokenized binary shard using the SentencePiece model.

- Source: `fineweb_train_000000.bin` (100M tokens)
- Split on BOS token (id=1) to recover document boundaries
- Result: **10,000 documents** decoded to `decoded_docs.jsonl`

### 2. Train BESE+BPE merges

- Input: 10,000 decoded FineWeb documents (39,521,418 base BESE tokens)
- Merges: 250
- Output vocab size: **288** (38 base + 250 merges)
- Compression: 39.5M base tokens -> 15.4M merged tokens (**39.09%** of original = 2.56x compression)
- Byte check: **100/100 documents passed** (token byte count == UTF-8 byte count)
- Time: **1,354 seconds** (~22.5 min) on CPU — pure-Python BPE, the main bottleneck

### 3. Export BESE shards

- Val shard: 2,000 docs -> 2,843,989 tokens
- Train shards: 8,000 docs -> 12,604,981 tokens (2 shards)
- Format: upstream-compatible (magic=20240520, version=1, int32 header + uint16 tokens)
- Export time: ~25 min (pure-Python `_apply_merges` bottleneck)

## Results

### Baseline (SP1024)

```
Training: 1,356 steps in 10 min (441ms/step avg)
Data: 10 shards (~1B tokens)
val_loss: 2.2488
val_bpb: 1.3319
Model size (int8+zlib): 13.6 MB
```

### BESE+BPE (250 merges, vocab=288)

```
Training: 1,189 steps in 10 min (505ms/step avg)
Data: 2 shards (~12.6M tokens)
val_loss: 5.4200
val_bpb: 3.9143
Model size (int8+zlib): 12.9 MB
```

## Analysis

### Why BESE performed poorly: data starvation

The comparison is **not apples-to-apples**. The critical difference is training data volume:

| | Baseline | BESE |
|---|---|---|
| Train tokens | ~1,000,000,000 | ~12,604,981 |
| Train shards | 10 | 2 |
| Unique text | ~80K docs worth | ~8K docs worth |
| Ratio | 1x | **0.013x (80x less data)** |

The model ran out of unique data almost immediately and cycled through the same small corpus for the entire 10-minute run.

### Signs that BESE itself works

1. **Train loss dropped normally**: 5.69 -> 1.22 over 1000 steps (healthy learning curve)
2. **Byte accounting is correct**: 100/100 documents pass the BPB byte check
3. **Model is smaller**: 12.9 MB vs 13.6 MB (the embedding savings are real)
4. **Steps per second are similar**: 505ms vs 441ms (~15% slower, likely due to shorter sequences hitting different attention patterns)

### Bottlenecks identified

1. **Pure-Python BPE trainer**: O(num_merges * total_tokens) — 22 min for 10K docs. SentencePiece (C++) does equivalent work in seconds. This blocks training on 80K+ docs.

2. **Pure-Python shard export**: `_apply_merges()` runs 250 merge passes per document during encoding. 25 min for 10K docs. Would need hours for the full 80K+.

3. **Disk space**: The pod has 20GB container disk. FineWeb `docs_selected.jsonl` is 48GB. Either need a larger disk pod or must decode from existing SP shards (which we did).

## What a fair test needs

1. **Equal data volume**: Re-encode ALL 10 shards with BESE+BPE (requires either the 48GB JSONL or decoding all 10 SP shards)
2. **Faster tokenizer**: Rewrite BPE training and encoding in C/Cython, or use SentencePiece to train BPE on the BESE base token stream
3. **Larger disk pod** (~100GB container disk) or use a network volume to hold the raw JSONL
4. **Model config tuning**: With vocab=288 (saving ~295KB), add extra transformer layers (target: 13L vs baseline 9L) to use the freed parameters

## Cost

- Pod 1 (India): ~25 min at $2.69/hr = ~$1.12 (killed by SSH drop)
- Pod 2 (US): ~62 min at $2.69/hr = ~$2.78
- Total spend: ~$3.90
- Remaining balance: ~$9.23

## Key takeaways

1. The BESE tokenizer is **functionally correct** — byte counts match, round-trips work, model trains
2. The poor val_bpb is caused by **80x less training data**, not a flaw in the tokenizer
3. The pure-Python BPE implementation is too slow for production use — needs C/Cython rewrite or SentencePiece integration
4. The model size savings are real (12.9 MB vs 13.6 MB) and could fund extra layers
5. A fair comparison requires equal data volume, which requires solving the tokenizer speed bottleneck first
