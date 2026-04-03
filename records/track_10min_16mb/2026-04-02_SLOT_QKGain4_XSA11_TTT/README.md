# SLOT + QK-Gain 4.0 + XSA-11 + TTT

Record submission integrating four proven post-March-25 techniques onto the VRL + LeakyReLU2 base (PR #175).

## Architecture

11L, 512d, 8H/4KV GQA, LeakyReLU(0.5)^2 MLP 3x, VRL, VE128, BigramHash(2048), XSA on all 11 layers, Partial RoPE 16/64, LN Scale, SmearGate, U-Net skips, EMA(0.997) + Tight SWA, Late QAT, int6 + lzma, FA3 Hopper, Muon WD=0.04.

## New Techniques (over PR #175 base)

| Technique | Source | Expected Impact |
|-----------|--------|-----------------|
| QK-Gain 4.0 | PR #1125 (45-experiment sweep) | -0.006 BPB |
| XSA all 11 layers | PR #1176 | -0.002 BPB |
| SLOT (per-sample delta + logit bias) | PR #1229 (arXiv:2505.12392v2) | -0.021 to -0.060 BPB |

### SLOT Details

Scored-position Learned Output Tuning optimizes a per-sequence additive delta vector [bsz, 1, 512] at the last hidden layer plus a per-sequence logit bias [bsz, 1, vocab], using frozen model weights. Optimization is 16 AdamW steps with cosine LR 0.008 -> 0.0008. Only scored positions (last stride=64 tokens per non-first window) contribute to the SLOT loss, aligning optimization with the eval metric.

## Reproduction

```bash
QK_GAIN_INIT=4.0 XSA_LAST_N=11 SLOT_ENABLED=1 SLOT_STEPS=16 SLOT_LR=0.008 \
  SEED=1337 DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- Base model: PR #175 (anthony-maio)
- SLOT mechanism: Hu et al. arXiv:2505.12392v2, PR #1176 (@bigbag), PR #1229 (@resouer)
- QK-Gain 4.0: PR #1125 (45-experiment sweep)
- VRL: ResFormer (arXiv:2410.17897)
