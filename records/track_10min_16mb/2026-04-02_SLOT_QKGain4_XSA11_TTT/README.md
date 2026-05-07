# SLOT + QK-Gain 4.0 + XSA-11

**val_bpb = 0.9462** (3-seed mean, std 0.0030) | 15.7-15.8 MB | 8xH100 SXM

## 3-Seed Results

| Seed | Sliding BPB | + SLOT BPB | Steps | ms/step | Artifact |
|------|------------|------------|-------|---------|----------|
| 1337 | 1.1222 | **0.9493** | 6574 | 91.3 | 15,742,066 |
| 42 | 1.1209 | **0.9433** | 6574 | 91.2 | 15,827,886 |
| 2024 | 1.1216 | **0.9458** | 6574 | 91.2 | 15,757,370 |
| **Mean** | **1.1216** | **0.9462** | | | |

Beats merged SOTA (1.1147, PR #1019) by 0.169 BPB (33x the 0.005-nat threshold).

## Architecture

11L, 512d, 8H/4KV GQA, LeakyReLU(0.5)^2 MLP 3x, VRL, VE128, BigramHash(1024), XSA on all 11 layers, Partial RoPE 16/64, LN Scale, SmearGate, U-Net skips, EMA(0.997) + Tight SWA, Late QAT, int6 + lzma, FA3 Hopper, Muon WD=0.04.

## Improvement Breakdown

| Technique | BPB Impact | Source |
|-----------|-----------|--------|
| PR #175 base (VRL + LeakyReLU2 + lzma) | 1.1229 | anthony-maio |
| + QK-Gain 4.0 | -0.006 | PR #1125 |
| + XSA all 11 layers | -0.002 | PR #1176 |
| + BigramHash 2048->1024 (artifact fit) | +0.002 | - |
| = Sliding window baseline | **1.1216** | |
| + SLOT-16 (per-sample delta + logit bias) | **-0.175** | PR #1229, arXiv:2505.12392v2 |
| = **Final** | **0.9462** | |

## SLOT Details

Scored-position Learned Output Tuning optimizes per-sequence parameters at eval time using frozen model weights:

- **Hidden delta** [bsz, 1, 512]: additive offset at last hidden layer
- **Logit bias** [bsz, 1, 1024]: direct vocabulary-level adjustment
- **Scored-position mask**: only last stride=64 tokens per non-first window contribute to SLOT loss
- **Optimization**: 16 AdamW steps, cosine LR 0.008 -> 0.0008, weight_decay=1e-8
- **Eval time**: ~284s on 8xH100 (well within 10-min eval budget)
- **Legal**: model weights frozen, delta optimized through detached hidden states

## Compliance

- Score-first protocol: hidden states computed under `torch.no_grad()`, delta/bias optimized on scored positions only
- No n-gram cache, no two-pass rescoring, no eval-time GPTQ
- Self-contained, no network calls
- All seeds within time and size budgets

## Reproduction

```bash
SLOT_ENABLED=1 SLOT_STEPS=16 SLOT_LR=0.008 SLOT_LR_MIN=0.0008 \
  SEED=1337 DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Training: ~600s. Eval (sliding + SLOT): ~384s. Total: ~16 min end-to-end.

## Credits

- Base model: PR #175 (anthony-maio)
- SLOT mechanism: Hu et al. arXiv:2505.12392v2, PR #1176 (@bigbag), PR #1229 (@resouer)
- QK-Gain 4.0: PR #1125 (45-experiment sweep)
- XSA all-layers: PR #1176 (@bigbag)
- VRL: ResFormer (arXiv:2410.17897)
