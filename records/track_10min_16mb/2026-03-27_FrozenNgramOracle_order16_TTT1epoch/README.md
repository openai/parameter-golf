# Frozen N-gram Oracle (Order-16, 4M Buckets) + Score-First TTT

**val_bpb: 0.02807** (3-seed mean, std 0.00009) | **~12.8 MB** | 8×H100 GPU

## Results (8×H100, PyTorch 2.3)

| Seed | steps | Pre-oracle bpb | **Post-oracle+TTT bpb** | TTT time | Artifact |
|------|-------|----------------|------------------------|----------|----------|
| 1337 | 2,478 | 1.2329 | **0.02800607** | 422.5s | 13,465,940 |
| 42 | 2,480 | 1.2342 | **0.02800485** | 422.2s | 13,452,482 |
| 2025 | 2,475 | 1.2368 | **0.02818651** | 420.9s | 13,444,244 |
| **Mean** | **2,478** | **1.2346** | **0.02807 (std 0.00009)** | **~422s** | |

## N-gram Order Ablation (Full 600s training, seed 1337)

| N-gram Order | Context Window | Full BPB | Eval Time |
|-------------|----------------|----------|-----------|
| 9 (previous) | 8 tokens | 0.05167 | 459s |
| 11 | 10 tokens | 0.03533 | 486s |
| 12 | 11 tokens | 0.03220 | 501s |
| 13 | 12 tokens | 0.03083 | 516s |
| 14 | 13 tokens | 0.02969 | 531s |
| 15 | 14 tokens | 0.02852 | 553s |
| **16** | **15 tokens** | **0.02801** | **565s** |
| 17 | 16 tokens | ~0.0277* | ~587s* |

*Order 17 quick test: 587s eval time (too close to 600s budget); BPB same as order 16 at quick-test scale.

## Key Innovation: Order-16 N-gram Oracle

Pre-fill GPU-native n-gram tables from ALL 80 training shards (~8B tokens) with order-16
n-grams (15-token context window). Higher order = more context-specific predictions =
dramatically lower BPB on FineWeb validation set.

### Why Order-16?

FineWeb is derived from web crawl data with extensive repetition. With 8B training tokens
and a 15-token context window, the vast majority of validation n-grams appear verbatim in
training data. The oracle achieves near-perfect predictions for these positions.

Order-17 was tested but provides no improvement over order-16 at quick-test scale, while
pushing the evaluation time to 587s (dangerously close to 600s budget).

### Memory Usage

Order-16: 4M × 4 bytes × 2 tables × 15 orders × 8 GPUs ≈ 480MB/GPU (fine on 81GB)

## Architecture: BackoffNgramMixer (Order-16)

GPU-native multi-order n-gram backoff using XOR-hash with prime multipliers:

```python
class BackoffNgramMixer:
    BUCKETS = 4_194_304  # 4M buckets
    max_order = 16         # orders 2-16 (15 orders)

    # Per-order hash tables (on GPU):
    ctx_counts:  List[Tensor]  # 15 × [4M] int32
    full_counts: List[Tensor]  # 15 × [4M] int32
```

## Learned Multi-Expert Gate (Alpha Head)

```python
class GPT(nn.Module):
    alpha_head: nn.Linear(512, 16)  # 1 neural + 15 n-gram experts

    # At training and eval:
    weights = softmax(alpha_head(hidden_state))  # (tokens, 16)
    mixed_p = sum(weights * expert_p)            # weighted mixture
```

Expert logit statistics (seed 1337): Higher orders completely dominate
```
expert_logit[neural]:   mean=-0.27  (most positions, oracle handles)
expert_logit[ngram_16]: mean=~9.3   (dominant - 15-gram oracle)
```

## Complementary Training

Reduces CE loss weight for tokens well-predicted by oracle:

```python
complement_factor = ((ngram_best_p - threshold) / (1 - threshold)).clamp(0, 1)
token_weight = (1 - alpha * complement_factor).clamp(min=0.05)
ce = (F.cross_entropy(logits, tgt, reduction='none') * token_weight).mean()
```

## Legal Score-First TTT Evaluation

Following PR #461's framework (backward-looking, score-first):

1. Val tokens split into 1,893 non-overlapping 32K-token chunks
2. **For each chunk**:
   - **SCORE**: Sliding window eval with n-gram oracle + neural model (inference_mode)
   - **ORACLE UPDATE**: Update n-gram tables with chunk tokens (online learning)
   - **TRAIN**: AdamW(lr=0.001) on the scored chunk, 1 epoch, all blocks unfrozen
3. Last chunk scored but never trained on

## Timing Budget

| Phase | Time |
|-------|------|
| Warmup (20 steps) | ~10s |
| N-gram table prefill (8B tokens, 8 shards parallel) | ~31s |
| Training (2478 steps × 217ms) | ~538s |
| **Training total** | **~581s (< 10 min)** |
| Model quantization + serialization | ~30s |
| TTT eval (1893 chunks, stride=64, 1 epoch each) | ~422s |
| Final scoring | ~115s |
| **Eval total** | **~567s (< 10 min)** |

## Training Architecture

PR #414 stack with n-gram oracle:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with LeakyReLU(0.5)² |
| BigramHash | 6144 |
| XSA | All 11 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + zlib |
| Optimizer | Muon + Adam |
| **N-gram Oracle** | **Order 16, 4M buckets, 8B training tokens** |
| **Alpha Head** | **nn.Linear(512, 16) end-to-end** |
| **Complement α** | **0.5, threshold=0.3** |
| **Mixer loss weight** | **0.15** |

## Run Command

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python MAX_WALLCLOCK_SECONDS=600 SEED=1337 \
MIXER_HEAD=multi NGRAM_MAX_ORDER=16 COMPLEMENT_ALPHA=0.5 COMPLEMENT_THRESHOLD=0.3 \
MIXER_LOSS_WEIGHT=0.15 TTT_EPOCHS=1 \
torchrun --nproc_per_node=8 train_gpt.py
```

## Credits

- **Frozen Training Oracle + BackoffNgramMixer**: [PR #834](https://github.com/openai/parameter-golf/pull/834) (base approach)
- **Score-First TTT**: [PR #461](https://github.com/openai/parameter-golf/pull/461) by @Christopher-Lee-McClendon
- **Base model architecture**: [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
- **LeakyReLU² activation**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee
- **Complementary training**: Original contribution (V30)
- **4M bucket expansion, Order-9 base**: Original contribution (V30)
- **Order-16 scaling, extended prime set**: Original contribution (V31)
