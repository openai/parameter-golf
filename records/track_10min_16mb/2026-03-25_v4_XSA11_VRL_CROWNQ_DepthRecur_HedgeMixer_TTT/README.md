# Record: XSA-all + VRL + CROWN-Q + Depth Recurrence + Hedge Mixer TTT

**val_bpb = 1.0222** (3-seed mean, std 0.0067) | **<16 MB** | 8xH100 SXM | 600s train, 507s eval

## 3-Seed Results (8xH100 80GB SXM, PyTorch 2.4.0+cu124)

| Seed | Steps | step_avg | Pre-TTT bpb | **Post-TTT bpb** | TTT time | Artifact |
|------|-------|----------|-------------|-----------------|----------|----------|
| 1337 | 4,473 | 134.2ms | 1.1336 | **1.0201** | 507s | 15,857,972 |
| 42 | 4,452 | 134.8ms | 1.1339 | **1.0165** | 508s | 15,846,228 |
| 2025 | 4,451 | 134.8ms | 1.1369 | **1.0299** | 507s | 15,669,888 |
| **Mean** | | | **1.1348** | **1.0222 (std 0.0067)** | **507s** | |

All artifacts under 16,000,000 bytes. Training: 600s. Eval (TTT + sliding): 507s. Both within limits.

## Architecture: PR #549 base + 6 additions

### 1. XSA on all layers (PR #634)
Exclusive Self-Attention on all 13 virtual layers (11 physical + 2 recurred). -0.006 BPB vs XSA-last-4.

### 2. Value Residual Learning (PR #657, arXiv:2410.17897)
Layer 0's V output blended into subsequent attention via learned sigmoid gates. +10 params.

### 3. Gated Attention (PR #638)
Per-head sigmoid gates on attention output.

### 4. CROWN-Q (PR #693)
Curvature-weighted quantization penalty during warmdown: `lambda * mean(w^2) * (row_max/15)^2 / 12`. Pushes weights into flat minima for better int6 quantization. Zero eval cost.

### 5. Depth Recurrence (PR #686)
Layers 4,5 re-executed: 11 physical layers become 13 virtual (pattern: 0,1,2,3,4,5,4,5,6,7,8,9,10). Banks indexed via v2p mapping. Untied before TTT.

### 6. 5-Expert Hedge Mixer (PR #688)
GPU-vectorized online context mixing during TTT eval:

| Expert | Source |
|--------|--------|
| Neural | Base model log-softmax |
| Unigram | Token frequency from scored tokens |
| Bigram | P(next | prev) from scored tokens |
| Trigram | Hashed 64K-bucket trigram table |
| Entropy | Neural entropy as confidence weight |

Weights updated via Hedge algorithm. All n-gram tables from already-scored tokens only.

### Training Stack

11L physical (13 virtual), 512d, 8H/4KV GQA, MLP 3x LeakyReLU(0.5)^2, SmearGate, BigramHash(2048), VE128, EMA(0.997) + SWA, GPTQ-lite int6 + lzma, Muon WD=0.04, warmdown=3500.

### Legal Score-First TTT (1 epoch)

```
for each 32K-token chunk:
  Phase 1: SCORE under torch.inference_mode() + Hedge Mixer scoring
  Phase 2: UPDATE mixer n-gram tables with scored tokens
  Phase 3: TRAIN SGD(lr=0.002, mom=0.9) on scored chunk, 1 epoch, all blocks unfrozen
```

## Compliance

- [x] Training: 600s wallclock on 8xH100 SXM
- [x] Eval (TTT): 507s on 8xH100 SXM
- [x] All artifacts under 16,000,000 bytes
- [x] Score-first TTT: tokens scored under inference_mode before training
- [x] N-gram tables from already-scored tokens only
- [x] No training data access during evaluation
- [x] No oracle/hindsight selection
- [x] GPTQ-lite: no calibration data

## Reproduction

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All defaults match submitted results. No env vars needed.

## Credits

PR #549 (@abaybektursun), #634 (@raahilshah), #657 (@anthony-maio), #638 (@Asukabot0), #693 (@EthanYangTW), #686 (@msisovic), #688 (@RoyiRa), #493 (@parinzee), #414 (@signalrush)
