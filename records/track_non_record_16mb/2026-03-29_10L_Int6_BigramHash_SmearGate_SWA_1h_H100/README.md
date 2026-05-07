# Parameter Golf v3 — 1.15 BPB in 16MB

My submission for [OpenAI's Parameter Golf challenge](https://github.com/openai/parameter-golf): best language model that fits in **16MB** (compressed), trained in under 10 minutes on 8×H100s, evaluated by bits-per-byte (BPB) on FineWeb.

**Final result: ~1.15 BPB** (sliding window eval)

> ⚠️ **Honest disclaimer**: This entry does not strictly satisfy the competition constraints. The compressed model is **16.1 MB** (limit: 16 MB), and training used **1× H100 for 2 hours** (~120 H100-minutes) vs the allowed 8× H100 for 10 minutes (80 H100-minutes). This is more of a personal exploration of the techniques than a valid leaderboard submission.

---

## Results

| Metric | Value |
|---|---|
| Standard BPB | 1.186 |
| Sliding window BPB | ~1.151 |
| Compressed size | 16.1 MB (zstd-22) — just over the 16 MB limit |
| Training time | 2 hours |
| Hardware | 1× H100 (OpenAI API credits) |

Leaderboard context at submission time:
```
#1 thwu1:    1.1428
#2 Raahil:   1.1458
#3 aruniyer: 1.1502
Ours:        ~1.15  (non-compliant — 16.1 MB, 1×H100 2h)
```

---

## Architecture

**GPTv3** — a 10-layer transformer with 512 hidden dim, built around several techniques borrowed from top leaderboard entries and a couple of original additions.

### Key techniques

**BigramHash Embedding** — hashes `(prev_token, curr_token)` pairs into 10,240 buckets using a simple XOR hash, projecting to a 128-dim bigram embedding that is added to the standard token embedding. Initialized to zero (no-op at start), the model learns to use bigram context where it helps. Cheap (~1.3M params) and effective.

**SmearGate** — a learned per-dimension sigmoid gate that blends each token's embedding with the previous token's embedding. ~512 parameters total. Initialized to zero (no blending), so it only activates if it helps. Essentially a soft learned 1-gram lookahead within the embedding.

**Int6 QAT (Quantization-Aware Training)** — fake-quantizes weights to 6-bit range `[-32, 31]` during the forward pass using a straight-through estimator (STE), so gradients flow normally. Applied from step 0. At export, weights are actually quantized to int8 containers (6-bit range), which zstd can compress aggressively. Tied embeddings and small/control tensors are kept in fp16.

**zstd-22 compression** — replaces zlib. At level 22 (max), zstd compresses int6 weights significantly better than zlib-9 due to the restricted value range `[-32, 31]`.

**3× MLP width** — `mlp_mult=3`, wider FFN. Consistent with top leaderboard entries.

**SWA (Stochastic Weight Averaging)** — collects model snapshots during the final ~600 steps (when `lr_mul < 0.2`) every 50 steps, averages them. Gives a small but reliable BPB improvement at no extra training cost.

**Sliding window eval** — recovers ~0.02 BPB over standard fixed-window eval by using stride-64 overlapping windows, giving the model more context on average.

**Other standard ingredients**: GQA (8 heads, 4 KV heads), RoPE, RMSNorm on Q/K, logit softcapping (30.0), orthogonal weight init (OrthoInit), skip connections between encoder and decoder halves, Muon optimizer with weight decay for matrix parameters, Adam for embeddings and scalars.

---

## Model config

```python
vocab_size       = 1024      # SentencePiece BPE
model_dim        = 512
num_layers       = 10
num_heads        = 8
num_kv_heads     = 4
mlp_mult         = 3
bigram_vocab     = 10240
bigram_dim       = 128
rope_base        = 10000.0
logit_softcap    = 30.0
```

---

## Training

```
Dataset:    FineWeb 10B (sp1024 tokenizer, 1024-vocab SentencePiece BPE)
Seq length: 1024
Batch:      524,288 tokens/step (grad_accum=8)
Steps:      20,000 (wall-clock capped at 2h)
Warmup:     20 steps
Warmdown:   1,200 steps (cosine-style LR decay)
Optimizer:  Muon (matrix params) + Adam (embeddings, scalars)
WD:         0.04 (Muon)
```

---

## Compression pipeline

1. Train with Int6 QAT active from step 0
2. Apply SWA averaging over final checkpoints
3. Quantize all large weight matrices to int6 per-row (`abs_max / 31` scale, stored as int8)
4. Keep tied embeddings and control parameters as fp16
5. Serialize with `torch.save` → compress with `zstd` level 22
6. Verify roundtrip by decompressing and re-evaluating

---

## Reproduce

```bash
pip install sentencepiece huggingface-hub tqdm zstandard torch

# Download FineWeb data
cd /workspace/parameter-golf
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Run notebook
jupyter nbconvert --to notebook --execute parameter_golf_v3.ipynb
```

Requires 1× H100 (or equivalent). Training runs ~2 hours.

---

## What didn't work / wasn't tried

- **Depth recurrence / weight tying across layers** — explored in earlier versions, dropped for stability
- **Mediator token attention** — an earlier architectural idea; didn't fit cleanly within the 16MB budget at this scale
- **NF4 / FP4 quantization** — int6 + zstd-22 was already sufficient to stay under 16MB with good quality

---

## Hardware note

Trained on a **single H100 GPU** using OpenAI API credits, for **2 hours** (~120 H100-minutes). The competition spec was 8×H100s for 10 minutes (80 H100-minutes); I used 1.5× the allowed compute. Combined with the 16.1 MB model size, this is not a valid competition entry — just a personal experiment with the techniques.

---

## License

MIT
