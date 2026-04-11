# 11L GEPA 30k Steps Pure Int6 Legal TTT

## Result: 1.09197 BPB (non-record track)

### Architecture
- 11 transformer layers, 512-dim, 8 heads (4 KV), MLP=1536
- Value Embedding (VE_DIM=128) on layers 9,10
- BigramHash(2048, dim=128), Partial RoPE (16 dims)
- ReLU² MLP, U-Net skip connections, SmearGate
- 27M parameters

### Training
- **30000 steps** on 4×A100-40GB (~4.17 hours)
- 786K tokens/step = 23.59B tokens total
- Muon optimizer: LR=0.025 (matrix), 0.035 (tied embed), decoder 2× mult
- Muon momentum: 0.92→0.99 warmup over 1500 steps
- Weight decay: 0.04, Gradient clip: 0.3
- EMA decay 0.997
- **Warmdown: 18000 steps (60%)** — key insight: longer warmdown reduces quant gap
- Warmup: 20 steps

### Quantization
- Pure int6 per-row + zstd-22 compression
- GPTQ-lite with 15-percentile clip search
- QUANT_EMBED=1 (int6 per-row for embeddings)
- **Artifact: 14,057,451 bytes (13.40 MB)**
- **Quant gap: 0.0224** (float 1.1043 → quant 1.1267)

### Test-Time Training (Legal)
- SGD with momentum=0.9, lr=0.002, 10 epochs per chunk
- 32768 tokens/chunk, freeze first 2 blocks
- Gradient clip: 1.0
- Cosine LR decay across chunks with 50-chunk warmup
- **TTT gain: -0.035** (quant 1.1267 → TTT 1.0920)

### Training Trajectory
| Step | val_bpb | Phase |
|------|---------|-------|
| 500 | 1.3944 | Warmup |
| 5000 | 1.2315 | Peak LR |
| 10000 | 1.2177 | Peak LR plateau |
| 12000 | 1.2178 | Warmdown start |
| 15000 | 1.2021 | Early warmdown |
| 20000 | 1.1828 | Mid warmdown |
| 25000 | 1.1561 | Deep warmdown |
| 27000 | 1.1397 | Acceleration |
| 29000 | 1.1167 | Rapid convergence |
| 30000 | **1.1043** | **Final** |

### Key Insights
1. **60% warmdown ratio** reduces quantization gap from 0.027 → 0.022 (5 mBPB)
2. **Peak-LR plateau** at ~1.217 reached by step ~9000 — longer peak LR has diminishing returns
3. **Final 5000 steps** of warmdown produce largest BPP decline (−0.052 from step 25k→30k)
4. **SGD TTT** more stable than AdamW TTT for this architecture
5. Scaling from 25k→30k steps: -0.0024 BPP improvement

### Scaling Law (observed)
| Steps | Float base | TTT final | Δ from prev |
|-------|-----------|-----------|-------------|
| 9000 | 1.1353 | 1.1157 | — |
| 12000 | 1.1268 | 1.1079 | -0.008 |
| 15000 | 1.1217 | 1.1035 | -0.004 |
| 20000 | 1.1153 | 1.0983 | -0.005 |
| 25000 | 1.1088 | 1.0944 | -0.004 |
| **30000** | **1.1043** | **1.0920** | **−0.002** |

---

## Acknowledgments

This submission builds on techniques introduced by many contributors to the parameter-golf community:

- **signalrush** (PR #414): GPTQ-lite clip search and EMA — the quantization backbone of this submission
- **jfprincz** (PR #315): Partial RoPE (16/64 dims) and layerwise LN scale
- **jfprincz** (PR #287): XSA on last 4 layers, EMA replacing SWA, MLP 3× expansion
- **unnir** (PR #265): Efficient Partial XSA concept
- **raahilshah** (PR #162): SmearGate, BigramHash embeddings, OrthoInit, Muon weight decay
- **aruniyer** (PR #86): Int6 quantization with STE QAT
- **samacqua**: LoRA-based test-time training concept
- **abaybektursun** (PR #549): LeakyReLU² activation exploration
- **OpenAI**: Baseline architecture, Muon optimizer, and competition infrastructure
