# Backoff N-gram Cache + LeakyReLU(0.9)² + Distributed Pre-fill

**val_bpb: 0.6678** (seed 1337, additional seeds pending)

## Approach

This submission combines a strong neural base with a backward-looking n-gram eval cache that provides ~0.47 BPB improvement at zero artifact cost.

### Key Techniques

1. **Multi-order backoff n-gram cache (orders 2-7)**: During sliding window evaluation, a hash-table-based n-gram cache is built from already-scored tokens. For each new token, the cache looks up the highest-order n-gram match (7-gram first, backing off to 2-gram) and mixes the empirical n-gram probability with the model's softmax output.

2. **Entropy-adaptive alpha mixing**: `alpha = 0.05 + 0.55 * sigmoid(2.0 * (H - 4.0))` where H is the model's per-token entropy. When the model is uncertain (high entropy), we trust the n-gram cache more. When confident, we trust the model. This avoids oracle selection (which was ruled illegal in #659).

3. **Distributed cache pre-fill**: On multi-GPU eval, each rank pre-populates its n-gram tables with ALL tokens preceding its assigned window range. This makes multi-GPU results match single-GPU (no cold-start cache on ranks 1-7). Pre-fill for rank 7 takes ~68s, well within eval budget.

4. **LeakyReLU(0.9)²**: Replaces standard relu² with leaky_relu(slope=0.9)², allowing gradient flow through negative activations. ~0.013 BPB improvement over relu² per community ablations.

5. **Score-first legality**: Every token is scored under `torch.inference_mode()` BEFORE being added to the n-gram cache. The alpha depends only on the model's own entropy, never on ground truth. No pre-eval TTT.

### Neural Base

11L transformer, 512d, 8H/4KV GQA, MLP 3x, SmearGate + BigramHash(2048), Muon optimizer (momentum 0.99, warmup from 0.92 over 1500 steps), seq2048, warmdown 3000 iters.

### Results

| Eval Method | val_loss | val_bpb |
|-------------|----------|---------|
| Non-overlapping (post-quant) | 1.9576 | 1.1594 |
| Sliding window (stride=64) | 1.9199 | 1.1371 |
| **N-gram cache (orders 2-7)** | 1.1275 | **0.6678** |

- Training: 7189 steps in 600s on 8xH100 SXM (~83ms/step)
- Artifact: 8,622,077 bytes (well under 16MB)
- N-gram eval time: 200s (including 68s pre-fill for rank 7)

### Ablation

The n-gram cache provides the dominant improvement:
- Neural only (sliding window): 1.1371 BPB
- N-gram cache improvement: -0.4693 BPB

### Config

```bash
NCCL_IB_DISABLE=1 RUN_ID=ngram_v1_8x \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 RELU_SLOPE=0.9 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
NGRAM_CACHE=1 NGRAM_ENTROPY=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### N-gram Cache Implementation

- 6 hash tables (orders 2-7), 4M buckets each, uint32 counts (~192MB CPU RAM)
- XOR hashing with position-dependent primes
- Highest-order-first backoff: 7-gram match takes priority
- min_count=2 threshold to avoid noisy predictions
- All operations on CPU via numpy — zero GPU overhead
- `np.add.at` for correct duplicate handling in batch updates
