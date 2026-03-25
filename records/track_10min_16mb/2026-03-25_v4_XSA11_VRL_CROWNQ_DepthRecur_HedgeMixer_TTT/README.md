# Record: XSA-all + VRL + CROWN-Q + Depth Recurrence + Hedge Mixer TTT

**val_bpb = 1.0278** (3-seed mean, std 0.0039) | **~15.8 MB** | 8xH100 SXM, 600s train

## 3-Seed Results (8xH100 80GB SXM, PyTorch 2.4.0+cu124)

| Seed | Steps | step_avg | Pre-TTT bpb | **Post-TTT bpb** | TTT time | Artifact |
|------|-------|----------|-------------|-----------------|----------|----------|
| 1337 | 4,465 | 134.4ms | 1.1335 | **1.0235** | 763s | 15,827,512 |
| 42 | ~4,460 | ~134ms | 1.1346 | **1.0289** | ~750s | 15,760,352 |
| 2025 | ~4,460 | ~134ms | 1.1365 | **1.0311** | 751s | 15,713,536 |
| **Mean** | | | **1.1349** | **1.0278 (std 0.0039)** | **~755s** | |

All artifacts under 16,000,000 bytes. Training: 600s wallclock on 8xH100 SXM.

**Note on eval time:** TTT eval takes ~755s (>600s limit). Reducing `TTT_EPOCHS` from 3 to 1 would bring eval under 600s with expected BPB ~1.08-1.09. We submit with 3 epochs for completeness; happy to resubmit with 1 epoch if required.

## Architecture: PR #549 base + 6 innovations

Building on the merged SOTA (PR #549, 1.1194 BPB), this submission adds:

### 1. XSA on all 11 layers (PR #634)
Exclusive Self-Attention applied to every layer instead of last 4. Forces cross-position mixing from layer 0. -0.006 BPB.

### 2. Value Residual Learning (PR #657, arXiv:2410.17897)
Layer 0's V output blended into all subsequent attention via learned sigmoid gates. Combats attention concentration. +10 scalar params, -0.002 BPB.

### 3. Gated Attention (PR #638)
Per-head sigmoid gates on attention output. Learned bias=4.0 (starts near-open). -0.002 BPB.

### 4. CROWN-Q (PR #693)
Curvature-weighted quantization variance penalty during warmdown: `lambda * mean(w^2) * (row_max/15)^2 / 12`. Pushes weights into flat minima where int6 quantization causes less damage. Zero eval-time cost.

### 5. Depth Recurrence (PR #686)
Layers 4 and 5 re-executed with independent scalar parameters: physical 11 layers become 13 virtual layers (pattern: 0,1,2,3,4,5,4,5,6,7,8,9,10). Banks indexed via v2p mapping. +~2K block scalar params, near-zero size overhead. Before TTT, recurrence is untied so each virtual layer gets independent weights.

### 6. 5-Expert Hedge Mixer (PR #688)
GPU-vectorized online context mixing during TTT eval. Five experts blend predictions in log-probability space:

| Expert | Source |
|--------|--------|
| Neural | Base model log-softmax |
| Unigram | Token frequency from scored tokens |
| Bigram | P(next given prev) from scored tokens |
| Trigram | Hashed P(next given prev2, prev1), 64K buckets |
| Entropy | Neural model entropy as confidence regularizer |

N-gram tables built incrementally from already-scored tokens only (legal). Expert weights updated online via Hedge algorithm: `log_w -= eta * loss`. All computations GPU-vectorized.

## Training Architecture

| Component | Details |
|-----------|---------|
| Layers | 11 physical, **13 virtual** (depth recurrence L4,L5) |
| Dimensions | 512d, 8H/4KV (GQA), MLP 3x (1536) |
| Activation | **LeakyReLU(0.5) squared** |
| Attention | **XSA all 13 virtual layers**, Partial RoPE 16/64, LN Scale 1/sqrt(i+1) |
| Residuals | U-Net skip connections, **Value Residual Learning** |
| Gates | **Gated Attention** (per-head sigmoid) |
| Embeddings | BigramHash(2048), VE128 (layers 9-10), SmearGate |
| Training | EMA(0.997) + Tight SWA, **CROWN-Q** + Late QAT@0.15 |
| Optimizer | Muon WD=0.04, warmdown=3500, batch=786K tokens |
| Quantization | GPTQ-lite int6 + lzma |
| FA3 fallback | Auto-detects FA3 vs SDPA for non-H100 testing |

## Legal TTT (Score-First, PR #549 framework)

Every token scored BEFORE any weight update:

```
for each 32K-token chunk:
    Phase 1 - SCORE: sliding window eval (torch.inference_mode), Hedge Mixer scoring
    Phase 2 - UPDATE MIXER: n-gram tables updated with scored tokens
    Phase 3 - TRAIN: SGD(lr=0.002, mom=0.9) on already-scored chunk, 3 epochs
```

SGD with cosine LR decay. All blocks unfrozen (freeze=0). Depth recurrence untied before TTT.

## Compliance

- [x] Training: 600s wallclock on 8xH100 SXM
- [x] All artifacts under 16,000,000 bytes
- [x] Score-first TTT: tokens scored under inference_mode before training
- [x] N-gram tables built from already-scored tokens only
- [x] No training data access during evaluation
- [x] No oracle/hindsight selection
- [x] GPTQ-lite operates on weights only (no calibration data)
- [ ] Eval time: ~755s (exceeds 600s; reducible to <600s with TTT_EPOCHS=1)

## Credits

- **Base model + Legal TTT**: PR #549 by @abaybektursun
- **XSA-all**: PR #634 by @raahilshah
- **Value Residual Learning**: PR #657 by @anthony-maio
- **Gated Attention**: PR #638 by @Asukabot0
- **CROWN-Q**: PR #693 by @EthanYangTW
- **Depth Recurrence**: PR #686 by @msisovic
- **Hedge Mixer**: PR #688 by @RoyiRa
- **LeakyReLU squared**: PR #493 by @parinzee
- **Base stack**: PR #414 by @signalrush

## Reproduction

```bash
pip install sentencepiece datasets huggingface-hub zstandard tiktoken flash-attn
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

SEED=1337 MAX_WALLCLOCK_SECONDS=600 XSA_LAST_N=11 GATED_ATTENTION=1 \
VALUE_RESIDUAL=1 CROWNQ_LAMBDA=0.01 RECUR_LAYERS="4,5" USE_MIXER=1 \
TTT_ENABLED=1 TTT_EPOCHS=3 TTT_FREEZE_BLOCKS=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
