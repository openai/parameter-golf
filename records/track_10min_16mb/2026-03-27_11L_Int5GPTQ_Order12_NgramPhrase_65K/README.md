# Record: 0.0881 BPB — 11L Int5 GPTQ + Order-12 N-gram + Phrase Cache + 65K Chunks

**Order-12 backoff n-gram cache + long phrase cache + entropy-adaptive alpha + temperature sharpening on a fully-trained 11-layer base model**

**val_bpb: 0.0881** (3-seed mean, std 0.0004) | **~13.0 MB** artifact | 8xH100 SXM, 600s train + <600s eval

## Results (3 seeds, 8xH100 SXM)

| Seed | val_bpb | Pre-quant BPB | Steps | ms/step | Eval time | Artifact |
|------|---------|---------------|-------|---------|-----------|----------|
| 1337 | 0.08855 | 1.1383 | 7,167 | 83.7 | 568s | 13,018,464 |
| 42 | 0.08793 | 1.1385 | 7,171 | 83.7 | 572s | 12,992,096 |
| 2024 | 0.08777 | 1.1391 | 7,172 | 83.7 | 563s | 13,032,072 |
| **Mean** | **0.08808** | **1.1386** | **7,170** | **83.7** | **568** | **13,014,211** |
| **Std** | **0.00041** | **0.0004** | **3** | **0.0** | **5** | **20,194** |

## Architecture

- 11 transformer layers, dim=512, GQA 8Q/4KV, head_dim=64
- MLP 3.0x expansion (1536 hidden) with LeakyReLU(0.5) squared
- BigramHash(1536, dim=128), XSA-4, Partial RoPE 16/64, LN Scale
- VE128 on layers 9-10, SmearGate, Logit softcap 30.0
- EMA decay 0.997, SWA (15 checkpoints in final warmdown)
- Parallel Muon + Parameter Banking (~84ms/step)
- ~27M parameters, tied embeddings

## Quantization

Full Hessian GPTQ int5 with activation-order column permutation and Cholesky error compensation. Deliberate int5 over int6: accepts ~0.02 BPB quantization penalty to reclaim ~1MB artifact headroom (13.0 MB vs ~15.9 MB at int6). LZMA preset 9 extreme compression.

## Eval Techniques (Single-Pass, Score-First)

### Order-12 N-gram Backoff Cache
- Orders 2-12, highest-order-first backoff with early exit
- 4M hash buckets per order, XOR-of-primes hashing
- Entropy-adaptive alpha with per-order thresholds and multipliers
- min_count=1, alpha range [0.05, 0.95]

### Long Phrase Cache
- 7 probe lengths: [64, 56, 48, 36, 28, 20, 16]
- XOR-of-products rolling hash, 4M buckets
- Alpha 0.90–0.99 based on match length

### Temperature Sharpening
- T=0.85 applied to logits before softmax

### 65K Chunk Size
Default 131K-token chunks from PR #913 exceed the 600s eval budget on models beyond a few layers, as forward pass cost scales with model depth. Reducing to 65K resolves this while providing warmer cache through 2x more frequent updates. Empirically, 65K chunks also complete faster in total (568s) than 131K (606s) or 140K (613s) on the same model — contrary to the assumption that fewer chunks reduce eval time.

### Score-First Protocol
Cache starts empty. Each 65K-token chunk: score all windows first, then update cache. Strictly backward-looking.

## Setup and Run

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git pgolf
cd pgolf
pip install --break-system-packages -r requirements.txt zstandard
python data/cached_challenge_fineweb.py --variant sp1024

SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Discussion: On n-gram cache dominance and the diminishing role of neural models.

This submission applies all 295 lines of PR #913's eval stack ("Cache Is All You Need") onto a fully-trained 11-layer 512d base model (pre-quant 1.14 BPB) — a 0.64 BPB improvement over #913's original 2-layer 128d model (pre-quant 1.78 BPB, 500K parameters, 622KB artifact). The base model represents a 54x parameter increase, Full Hessian GPTQ int5 quantization, aggressive SWA, and Parallel Muon optimization.

One practical adaptation was required: #913's default 131K-token chunk size exceeds the 600s eval budget on any model beyond a few layers, as forward pass cost scales with model depth. Reducing chunk size to 65K resolves this while providing warmer cache through more frequent updates — a necessary change for applying cache eval techniques to models built with any meaningful training investment.

The core finding: a 0.64 BPB gap in pre-quantization model quality (1.78 vs 1.14) — representing substantial differences in architecture depth, parameter count, training compute, quantization strategy, and optimization techniques — collapses to <0.001 BPB after cache application. The n-gram cache handles approximately 97% of token predictions through pure frequency statistics. The neural model contributes meaningfully only on the narrow residual of tokens with no cache match. Beyond order-10 n-gram caching with sufficient training data, marginal returns from neural model innovation approach zero.

This suggests the current leaderboard measures n-gram engineering quality, not language model quality. The competition's meta incentivizes cache engineering over model innovation — a dynamic where a 500K-parameter model performs equivalently to one 54x its size. This entry serves as an empirical demonstration of this limitation.

## Compliance

- [x] 3 seeds on 8xH100 SXM
- [x] All seeds train ≤600s
- [x] All seeds eval ≤600s (max 572s)
- [x] Artifact ≤16,000,000 bytes (~13.0 MB)
- [x] No validation data during training
- [x] Single-pass, score-first, backward-looking
- [x] No multi-pass rescoring
- [x] No TTT, no learned gate — pure cache eval
- [x] Reproducible single script

## Credits

Eval approach: PR #913 ("Cache Is All You Need", @RoyiRa). Base model architecture: PR #549 (@abaybektursun). Architecture foundation: PR #414 (@signalrush). Chunk size optimization concept: PR #840 (@quietsmile).
