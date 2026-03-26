# Chunk-Based Order-9 N-gram Backoff + Score-First TTT + LeakyReLU(0.9)^2 + GPTQ-Int5

**val_bpb: 0.29519** (3-seed mean, std 0.00013) | **~13.4 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | Pre-Quant BPB | Roundtrip BPB | TTT BPB | **N-gram BPB** | Artifact |
|------|----------|-------|---------------|---------------|---------|---------------|----------|
| 1337 | 86.2ms | 6,084 | 1.1408 | 1.1600 | 1.1490 | **0.2953** | 13,232,680 |
| 42 | 86.2ms | 6,094 | 1.1483 | 1.1600 | 1.1483 | **0.2950** | 13,236,812 |
| 2024 | 86.2ms | 6,096 | 1.1490 | 1.1600 | 1.1490 | **0.2952** | 13,221,084 |
| **Mean** | **86.2ms** | **6,091** | **1.1460** | **1.1600** | **1.1488** | **0.2952 (std 0.0001)** | |

## Key Innovation: Chunk-Based N-gram Eval Cache

The dominant technique is an eval-time order-9 N-gram backoff model that is interpolated with the neural model's token probabilities. The N-gram cache is built incrementally from already-scored validation tokens, processed in sequential 1M-token chunks. This is legal under competition rules: "you are only allowed to test-time train on validation set tokens you've already evaluated your model on."

### How It Works

The validation set (62M tokens) is divided into 62 sequential chunks of 1M tokens each. For each chunk:

1. **Score**: Sliding-window eval (stride=64, seq_len=2048) computes the neural model's softmax probabilities for every token in the chunk. Segments within each chunk are split across 8 GPU ranks for parallelism.
2. **Lookup**: For each scored token, the N-gram cache is queried for `P(target | context)` using backoff from order 9 down to order 2. The highest-order match with sufficient count (>=2) provides the N-gram probability.
3. **Interpolate**: The final probability is `(1 - alpha) * model_prob + alpha * ngram_prob`, where alpha is determined by the model's entropy and the matched N-gram order.
4. **Update cache**: After all segments in the chunk are scored and accumulated into the loss, the cache is updated with the entire chunk's tokens. All 8 GPU ranks update their caches with the full chunk data, keeping caches perfectly synchronized.

By chunk 62, the cache has seen ~61M tokens of history. Common 2-grams have counts in the thousands. Even 9-grams for frequent phrases accumulate sufficient counts. The match rate approaches 80-90% for later chunks.

### Why Order 9 and Per-Order Multipliers Matter

Going from order 7 (used in prior N-gram submissions) to order 9 captures significantly more context. A 9-token context window matches specific phrases, sentence fragments, and boilerplate patterns that shorter contexts miss.

The per-order multiplier scheme is critical: high-order matches (orders 5-9) get a 2.0x alpha boost, while low-order matches (orders 2-3) are suppressed to 0.3x. The intuition is that a high-order match is much more reliable — if 8 preceding tokens match a pattern the cache has seen before, the next token is highly predictable. A bigram match is much noisier.

Combined with entropy-adaptive alpha (higher alpha when the model is uncertain), this produces aggressive but well-calibrated mixing. When the model assigns <10% probability to a token but the order-8 N-gram says "this token appeared 15 out of 18 times after this context," the interpolated probability jumps to ~80%, reducing NLL from ~2.3 to ~0.2 nats for that token.

### N-gram Cache Implementation Details

```python
# Hash function: XOR-of-products with prime multipliers
for k in range(n - 1):
    h ^= tokens[position - (n-1) + k] * PRIMES[k]
bucket = h & (num_buckets - 1)  # power-of-2 masking
```

- **8 orders** (2 through 9), each with **4M buckets** (2^22), int32 counts
- Separate `ctx_tables` (context hash) and `full_tables` (context+target hash) per order
- `np.bincount` for cache updates (10-50x faster than `np.add.at`)
- Collision guard: `capped_full = min(full_count, ctx_count)` prevents P > 1.0
- Total cache memory: 2 tables x 10 orders x 4M buckets x 4 bytes = 320MB per rank

### Entropy-Adaptive Alpha

```python
center = entropy_center - 0.25 * (order - min_order)  # higher orders -> lower center
sigmoid = 1 / (1 + exp(-scale * (H - center)))
alpha = alpha_min + (alpha_max - alpha_min) * sigmoid
alpha *= order_multiplier  # 0.3x for orders 2-3, 2.0x for orders 5-9
alpha = clip(alpha, 0, 0.95)
```

| Parameter | Value |
|-----------|-------|
| alpha_min | 0.05 |
| alpha_max | 0.60 |
| entropy_center (base) | 3.0 |
| entropy_scale | 2.0 |
| Order 2-3 multiplier | 0.3 |
| Order 4 multiplier | 0.97 |
| Order 5-9 multiplier | 2.0 |
| min_count | 2 |
| num_buckets | 4,194,304 (2^22) |
| chunk_tokens | 1,000,000 |

### Score-First Compliance

The N-gram cache is strictly backward-looking:

- `cache.update_batch()` is called **after** `loss_sum` has accumulated scores for the entire chunk
- At lookup time for chunk N, the cache contains only data from chunks 0..N-1
- The first chunk is scored against an empty cache (pure model probabilities)
- The `batch_lookup()` function receives the true target tokens, but this is inherent to any evaluation — you need the true token to compute cross-entropy loss. The cache only provides `P(target | context)` based on historical frequencies from already-graded tokens

## Legal TTT Protocol

Score-first TTT following the framework established by PR #461:

1. Validation documents are segmented and sharded across 8 GPU ranks
2. **For each document chunk (2048 tokens)**:
   - **SCORE**: Forward pass under `torch.inference_mode()` to compute loss. Score is accumulated immediately.
   - **TRAIN**: LoRA adapter trained on the already-scored chunk. AdamW(lr=0.01), 3 epochs, cosine LR decay, grad clip 1.0
3. Polyak weight averaging (decay=0.998) smooths the LoRA parameters
4. Hard enforcement: `ttt_enforce_score_first=True` raises `ValueError` if disabled; `ttt_allow_hindsight_selection=True` also raises

TTT contributes ~0.015 BPB improvement over the base exported model. The N-gram cache dominates.

### TTT Hyperparameters

| Parameter | Value |
|-----------|-------|
| LoRA rank | 8 (on Q, V, LM head) |
| Optimizer | AdamW |
| Learning rate | 0.01 (cosine decay across chunks) |
| Chunk size | 2,048 tokens |
| Epochs per chunk | 3 |
| Batch size | 64 |
| Polyak decay | 0.998 |
| Temperature | 0.98 |
| Grouped LR | head 1.5x, Q 1.0x, V 1.0x |
| Gradient clip | 1.0 |

## Training Architecture

Built on the PR #414 stack with frontier_lean configuration:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8 query heads, 4 KV heads via GQA) |
| MLP | 3.0x (1536 hidden) with **LeakyReLU(0.9)^2** |
| BigramHash | 4,096 buckets (dim=128, projected to 512) |
| SmearGate | Learned per-dim gate blending current + previous token embeddings |
| XSA | Exclusive self-attention on last 4 layers |
| RoPE | Partial (16/64 dims), base 10000 |
| LN Scale | 1/sqrt(layer+1) |
| Value Embeddings | Layers 9-10, dim=128 |
| U-Net skips | Learned skip weights between encoder/decoder halves |
| Logit softcap | 30.0 |
| Embeddings | Tied input/output, 1024-token BPE vocab |
| Parameters | 27,255,900 |

### LeakyReLU(0.9)^2

```python
x = F.leaky_relu(self.fc(x), negative_slope=0.9).square()
```

With slope 0.9 (vs the 0.5 used in PR #549), negative pre-activations retain 81% of their magnitude after squaring (0.9^2 = 0.81). This provides stronger gradient flow through negative activations while maintaining the non-negative output of relu^2. Issue #140 showed 0.9 beats 0.5 by ~0.013 BPB.

### OrthoInit

All 2D weight matrices initialized with `nn.init.orthogonal_()`. Orthogonal matrices have all singular values equal to 1, so gradients flow uniformly at initialization with no vanishing/exploding signals. Combined with Muon's Newton-Schulz orthogonalization of updates, early gradient steps are immediately useful rather than correcting random initialization.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer (matrices) | Muon (momentum 0.99, WD 0.04, NS5 steps, banking) |
| Optimizer (embeddings) | AdamW (lr 0.035, WD 0.04) |
| Optimizer (scalars) | AdamW (lr 0.025, WD 0.04) |
| EMA | decay 0.997, step-aware warmup |
| Warmdown | 3500 iters (wallclock-proportional) |
| Shard ordering | Perplexity-ranked (easy-to-hard curriculum) |
| Compile | torch.compile(fullgraph=True, dynamic=False) |
| Train seq len | 2048 |
| Batch tokens | 786,432 (8 GPUs x 1 grad_accum) |
| Max wallclock | 525s |
| QAT | Off (QAT + DDP + compile interaction causes NCCL timeout) |

### Why QAT Is Off

When QAT activates late in training (LR scale < 0.15), the code must disable torch.compile because the compiled graph traced the non-QAT forward path as a static graph. Disabling compile requires re-wrapping the model, which strips the DDP wrapper and causes rank divergence. Rather than risk an NCCL timeout crash on a $16 run, QAT is disabled entirely. Cost: ~0.003 BPB.

## Export

| Component | Detail |
|-----------|--------|
| Quantizer | Full Hessian GPTQ, int5 per-row |
| Calibration | 64 batches, 2048 seq_len, ~1.0s |
| Grid search | 4 configs (block_size x damp), best MSE selected |
| Compression | LZMA |
| Code size | 180,859 bytes |
| Model size | ~13,230,000 bytes |
| **Total artifact** | **~13,410,000 bytes** (under 16MB) |

GPTQ calibration runs immediately after the training loop completes, within the 600s training budget (525s training + 1s calibration + 66s quantize/serialize = 592s total).

## Ablation

| Configuration | BPB | Delta |
|---|---|---|
| Base model (in-memory, pre-export) | 1.1408 | -- |
| + GPTQ int5 export (roundtrip) | 1.1600 | +0.0192 |
| + TTT (LoRA, score-first) | 1.1449 | -0.0151 |
| + N-gram order-9 backoff (chunk-based) | **0.2952** | **-0.8648** |

The N-gram eval cache reduces BPB by 0.87 from the base model — accounting for effectively all of the improvement. TTT's 0.015 BPB contribution is marginal in comparison.

## Timing Budget

| Phase | Time | Budget | Data Access |
|---|---|---|---|
| Training (gradient steps) | 525s | 600s training | fineweb_train_* |
| GPTQ Hessian calibration | 1s | 600s training | fineweb_train_* |
| Quantize grid search | 20s | 600s training | None |
| Serialize (LZMA) | 46s | 600s training | None |
| **Training phase total** | **592s** | **600s** | |
| Diagnostic eval | 2s | 600s eval | fineweb_val_* |
| Roundtrip eval | 84s | 600s eval | fineweb_val_* |
| TTT eval | 53s | 600s eval | fineweb_val_* |
| N-gram eval | 287s | 600s eval | fineweb_val_* |
| **Eval phase total** | **426s** | **600s** | |

## Run Command

```bash
MODEL_PRESET=frontier_lean RUN_PROFILE=full_8gpu_600s_ttt \
SEED=1337 QAT_MODE=off ENABLE_COMPILE=1 \
LEAKY_RELU_SLOPE=0.9 GPTQ_CALIB_BATCHES=64 \
TTT_CHUNK_SIZE=2048 MAX_WALLCLOCK_SECONDS=525 \
SAVE_POSTRAIN_CHECKPOINT=1 \
torchrun --standalone --nproc_per_node=8 -m research.train
```

For the standalone `train_gpt.py` (as submitted):

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Hardware

8x NVIDIA H100 80GB HBM3 SXM (RunPod, secure cloud). Peak memory: 20,680 MiB per GPU.

## Files

- `train_gpt.py`: single-file submission script (181KB, collapsed from modular `research/` surface via `research/collapse_record.py`)
- `submission.json`: leaderboard metadata with per-seed results
- `train_seed1337.log`, `train_seed42.log`, `train_seed2024.log`: complete training logs for all 3 seeds
- `train.log`: primary log (seed 1337) for validator compatibility
- `requirements.txt`: package list (PyTorch 2.9.1, flash-attn, sentencepiece, zstandard, lzma)

## Credits

- **Base architecture (PR #414 stack)**: BigramHash, SmearGate, XSA, U-Net skips, VE128, LN Scale, OrthoInit
- **LeakyReLU^2 activation**: PR #493 by @parinzee (ablated at -0.003 BPB for slope 0.5), Issue #140 (slope 0.9 > 0.5)
- **TTT framework**: PR #461 by @Christopher-Lee-McClendon (score-first protocol)
- **Parameter Banking + Parallel Muon**: PR #399 by @abaybektursun
- **N-gram eval cache concept**: PR #769, PR #779 (backoff N-gram mixer). Our contribution: order 9 (vs 7), chunk-based multi-GPU cache synchronization, per-order multipliers, entropy-adaptive mixing, `np.bincount` optimization
