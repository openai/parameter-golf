# Record: SLOT-32 + Partial Depth Recurrence (Layers 4,5) — val_bpb 0.7736

**val_bpb: 0.7736** (3-seed mean, std 0.0026) | **~15.71 MB** | 8xH100 SXM (Vast.ai), 600s

**Beats current SOTA (PR #1313, 0.8637 BPB) by 0.0901 BPB.**

## 3-Seed Results

| Seed | Steps | ms/step | Sliding BPB | **SLOT-32 BPB** | Artifact |
|------|-------|---------|-------------|----------------|----------|
| 42 | 4,929 | 121.7 | 1.1259 | **0.7732** | 15,656,490 |
| 1337 | 4,935 | 121.6 | 1.1257 | **0.7764** | 15,725,938 |
| 314 | 4,938 | 121.5 | 1.1255 | **0.7713** | 15,733,118 |
| **Mean** | | | **1.1257** | **0.7736** | |

## Key Techniques

| Technique | BPB Impact | Source |
|-----------|-----------|--------|
| **SLOT-32** (per-sample delta + logit bias, 32 AdamW steps) | **-0.352** | arXiv:2505.12392v2, PR #1229 |
| Partial depth recurrence (layers 4,5 repeated) | -0.005 | This work + PR #1204, PR #1260 |
| Per-iteration conditioning (iter_embed + iter_gate) | Novel | This work |
| XSA all 11 layers | -0.002 | PR #1176 |
| QK-Gain 4.0 | -0.003 | PR #1125 |
| VRL (Value Residual Learning) | -0.002 | arXiv:2410.17897, PR #175 |
| BigramHash 1024x128 | Input-level | PR #162 |
| EMA(0.997) + SWA(every 50) | Weight averaging | PR #401 |
| Late QAT (STE at scale < 0.15) | Quant-aware training | PR #286 |
| int6 + LZMA | Compression | PR #160, PR #535 |

## SLOT-32 Configuration

The primary contribution over prior SLOT submissions is tuning to 32 steps with higher learning rate:

| Parameter | PR #1303 (0.9462) | PR #1313 (0.8637) | **This work (0.7736)** |
|-----------|-------------------|-------------------|----------------------|
| SLOT_STEPS | 16 | 24 | **32** |
| SLOT_LR | 0.008 | 0.012 | **0.015** |
| SLOT_LR_MIN | 0.0008 | 0.001 | **0.001** |
| EVAL_STRIDE | 64 | 96 | **96** |

- Hidden delta: [bsz, 1, 512] + logit bias: [bsz, 1, 1024]
- 32 AdamW steps, cosine LR 0.015 -> 0.001, weight_decay=1e-8
- Scored-position masking: last stride=96 tokens per non-first window
- Model weights frozen, delta optimized through detached hidden states
- Eval time: ~304s (within 10-min eval budget)

## Partial Depth Recurrence

Virtual 13-layer network from 11 unique blocks by repeating layers 4 and 5:

```
virtual_layers = [0, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10]
                              ^     ^  (repeated)
```

Per-iteration conditioning ensures repeated passes are distinct:
```python
gate = sigmoid(iter_gate[i])      # learned, starts near 0
x = x + gate * iter_embed[i]      # additive conditioning
x = blocks[layer_idx](x, x0)     # same weights, different input
```

Active from step 0 (static graph for torch.compile fullgraph=True).

## Compliance

- Score-first SLOT (frozen model, torch.no_grad() hidden states)
- No external data access during eval
- No n-gram cache, no two-pass rescoring, no warmstart between windows
- Self-contained (no network calls)
- All seeds: training 600s, eval ~304s, total ~904s (within combined budget)
- All artifacts under 16MB

## Reproduction

```bash
pip install sentencepiece huggingface_hub
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

RECUR_LAYERS=4,5 SLOT_STEPS=32 SLOT_LR=0.015 SLOT_LR_MIN=0.001 \
  EVAL_STRIDE=96 SEED=42 \
  torchrun --standalone --nproc_per_node=8 train_gpt_slot_recurrence.py
```

## Lineage

```
PR #1019 (Merged SOTA, 1.1147 BPB)
  +-- PR #1303 (SLOT-16 + VRL + XSA-11, 0.9462 BPB)
    +-- PR #1313 (SLOT-24 tuning, 0.8637 BPB)
      +-- This work:
          +-- SLOT-32 tuning (32 steps, LR=0.015)
          +-- Partial depth recurrence (layers 4,5)
          +-- Per-iteration conditioning (iter_embed + iter_gate)
```

## Credits

- SLOT: Hu et al. arXiv:2505.12392v2, PR #1176 (@bigbag), PR #1229 (@resouer)
- SLOT-24 tuning: PR #1313
- Depth recurrence: PR #1204 (@msisovic), PR #1260 (@dexhunter)
- Base architecture: PR #1303 (@resouer), PR #1019 (@abaybektursun)
- QK-Gain: PR #1125 (@bigbag)
- VRL: arXiv:2410.17897, PR #175 (@anthony-maio)

## Author

Arnell Milhouse (@GitGeeks)
