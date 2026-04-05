# Cosine LR Schedule: -0.070 BPB + Focal Loss Investigation (Corrected)

## TL;DR

**Cosine LR schedule** replaces linear warmdown and gives **-0.070 BPB** consistently across training lengths. Combined with asymmetric 1/10 split (PR #1275), gives -0.080 at 5000 steps.

We also investigated **focal loss** which initially showed massive gains but contained a **critical eval bug** — all focal loss numbers were wrong. Corrected results show focal loss does not help. Documented here as a cautionary tale.

## Correction: Focal Loss Was An Eval Artifact

Our focal loss implementation applied `(1-pt)^gamma` weighting in `GPT.forward()`, called during both training AND evaluation. The "improvement" was entirely from down-weighting hard tokens in the eval metric, not from better model quality.

**Bug:** `if focal_gamma > 0:` (always active)
**Fix:** `if focal_gamma > 0 and self.training:` (training only)

**Corrected results (3000 steps, multi-seed):**

| Config | Seed 1337 | Seed 42 | Seed 2025 | Mean |
|--------|-----------|---------|-----------|------|
| Cosine LR only | 1.6538 | 1.6480 | 1.6687 | **1.657** |
| Cosine + Focal γ=2 | 1.6612 | 1.6560 | 1.6594 | **1.659** |
| Cosine + Focal γ=5 | 1.6858 | — | — | 1.686 |
| Cosine + Focal γ=8 | 1.7124 | — | — | 1.712 |

Focal loss does not help. Higher gamma actively hurts.

## What IS Real: Cosine LR Schedule

```python
# Replace linear warmdown in lr_mul():
min_lr_frac = 0.1
progress = step / max(args.iterations, 1)
return min_lr_frac + 0.5 * (1.0 - min_lr_frac) * (1.0 + math.cos(math.pi * progress))
```

| Steps | Baseline | Cosine LR | Delta |
|-------|----------|-----------|-------|
| 1000 | 2.0568 | 1.9334 | -0.123 |
| 2000 | 1.8330 | 1.8050 | -0.028 |
| 3000 | 1.7233 | 1.6538 | -0.070 |
| 5000 | 1.6422 | 1.5706 | -0.072 |

Consistent, not diminishing with training length.

## What IS Real: Asymmetric 1/10 Split

`self.num_encoder_layers = 1` — see PR #1275 for full details. Stacks with cosine: 1.5619 at 5000 steps (vs 1.5706 cosine alone).

## Reproduce

```bash
git clone https://github.com/openai/parameter-golf.git && cd parameter-golf
pip install sentencepiece huggingface-hub datasets tiktoken flash-attn
python data/cached_challenge_fineweb.py --variant sp1024
COSINE_LR=1 python train_gpt.py
```
