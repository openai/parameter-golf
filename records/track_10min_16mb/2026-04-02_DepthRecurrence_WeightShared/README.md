# Record: SLOT-24 + Partial Depth Recurrence (Layers 4,5) + XSA-11

**val_bpb: 0.8648** (3-seed mean, std 0.0014) | **~15.72 MB** | 8xH100 SXM (Vast.ai), 600s

## 3-Seed Results

| Seed | Steps | ms/step | Sliding BPB | **SLOT BPB** | Artifact |
|------|-------|---------|-------------|-------------|----------|
| 1337 | 4,932 | 121.7 | 1.1251 | **0.8664** | 15,726,046 |
| 42 | 4,930 | 121.7 | 1.1260 | **0.8637** | 15,667,502 |
| 314 | 4,935 | 121.6 | 1.1254 | **0.8643** | 15,745,654 |
| **Mean** | | | **1.1255** | **0.8648** | |

Merged SOTA (PR #1019): **1.1147 BPB**. Delta: **-0.2499 BPB**.

## Technique Summary

Built on PR #1303 SLOT + VRL + LeakyReLU2 + XSA-11 base, adding partial depth recurrence:

| Technique | Contribution | Source |
|-----------|-------------|--------|
| SLOT-24 (per-sample delta + logit bias) | -0.175 BPB | arXiv:2505.12392v2, PR #1229 |
| Partial depth recurrence (layers 4,5) | Novel combination | This work + PR #1204, PR #1260 |
| Per-iteration conditioning (iter_embed + iter_gate) | Novel | This work |
| XSA all 11 layers | -0.002 BPB | PR #1176 |
| QK-Gain 4.0 | -0.003 BPB | PR #1125 |
| VRL (Value Residual Learning) | -0.002 BPB | arXiv:2410.17897, PR #175 |
| BigramHash 1024x128 | Input-level | PR #162 |
| EMA(0.997) + SWA(every 50) | Weight averaging | PR #401 |
| Late QAT (STE at scale < 0.15) | Quant-aware training | PR #286 |
| int6 + LZMA | Compression | PR #160, PR #535 |

## Key Innovation: Partial Depth Recurrence with Per-Iteration Conditioning

Standard transformers use unique weights per layer. We share layers 4 and 5, repeating them once to create a virtual 13-layer network from 11 unique blocks:

```
virtual_layers = [0, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10]
                              ^     ^  (repeated)
```

Per-iteration conditioning ensures repeated passes are distinct:
```python
# On second pass through shared layer:
gate = sigmoid(iter_gate[i])      # learned, starts near 0
x = x + gate * iter_embed[i]      # additive conditioning
x = blocks[layer_idx](x, x0)     # same weights, different input
```

Recurrence is active from step 0 (static computation graph for torch.compile fullgraph=True).

## SLOT-24 Configuration

- Hidden delta: [bsz, 1, 512] + logit bias: [bsz, 1, 1024]
- 24 AdamW steps, cosine LR 0.012 -> 0.001, weight_decay=1e-8
- Scored-position masking: last stride=96 tokens per non-first window
- Model weights frozen, delta optimized through detached hidden states
- Eval time: ~245s (well within 10-min eval budget)

## Compliance

- Score-first SLOT (frozen model, torch.no_grad() hidden states)
- No external data access during eval
- No n-gram cache, no two-pass rescoring
- Self-contained (no network calls)
- All seeds within time (600s train + ~350s eval) and size (< 16MB) budgets

## Reproduction

```bash
pip install sentencepiece huggingface_hub
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
RECUR_LAYERS=4,5 SLOT_STEPS=24 SLOT_LR=0.012 SLOT_LR_MIN=0.001 EVAL_STRIDE=96 SEED=42 \
  torchrun --standalone --nproc_per_node=8 train_gpt_slot_recurrence.py
```

## Lineage

```
PR #1303 (SLOT + QK-Gain 4.0 + XSA-11, 0.9462 BPB)
    +-- This work adds:
        +-- Partial depth recurrence (layers 4,5 repeated)
        +-- Per-iteration conditioning (iter_embed + iter_gate)
        +-- SLOT-24 tuning (from PR #1313)
```

## Credits

- SLOT: Hu et al. arXiv:2505.12392v2, PR #1176 (@bigbag), PR #1229 (@resouer)
- Depth recurrence: PR #1204 (@msisovic), PR #1260 (@dexhunter)
- Base architecture: PR #1303 (@resouer), PR #1019 (@abaybektursun)
- QK-Gain: PR #1125 (@bigbag)
- VRL: arXiv:2410.17897, PR #175 (@anthony-maio)

## Author

Arnell Milhouse (@GitGeeks)
