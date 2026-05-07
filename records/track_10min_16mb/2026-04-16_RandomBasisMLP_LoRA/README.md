# Random-Basis MLPs + LoRA

This PR does **not** attempt to beat the 1.0985 SOTA and is not claiming to.

### TL;DR of the partial seed-42 run

| Seed | Steps | Stopped | Pre-quant val_bpb | Post-quant val_bpb | Artifact |
|------|-------|---------|-------------------|---------------------|----------|
| 42   | 2564 / 20000 | wallclock_cap | **1.25536** | _not captured¹_ | _not captured¹_ |

¹ The first training pass crashed at `serialize(...)` because the pod's image lacked `brotli` and then promptly ran out of Runpod credits.
The partial training log itself is attached as [`train_seed42_partial.log`](train_seed42_partial.log). 
With a (re-)deployed pod it is straightforward to produce a complete pre-/post-quant + artifact-size line as the training loop itself runs
cleanly end-to-end.

## Why it's interesting


Compression is not the same as entropy-coding. Most approaches are just competing to pack weights tighter into int6 to squeeze information density. However, random matrices don't work that way: they have maximum entropy, so entropy coders can't touch them. But a 4-byte seed can regenerate an arbitrarily large matrix from scratch. The 16MB artifact never says anything about how the process works. So instead of compressing weights, this PR stores seeds.

Rahimi & Recht (2007) showed that a ReLU²-activated Gaussian random projection is a universal nonlinear feature map: it can, in principle, represent any function. You get all that capacity for free, just from the seed. LoRA (Hu et al., 2021) then provides the small learnable layer on top; a rank-16 correction and the per-hidden gate that take those random features and push them toward directions that are actually useful for the task.

The per-hidden diagonal gate is doing the most work. A gate of 0 kills a feature entirely (same logic as Lottery Ticket pruning); a gate > 1 amplifies it. This also explains the hidden width choice. Everyone else is paying int6 cost per element, so wider hidden layers are expensive. My random hidden layer costs nothing but the seed. That's why 4x the hidden width of SOTA (mlp_mult=16 vs 4) is now affordable.


## Artifact math

At `dim=512`, `mlp_mult=16`, `lora_rank=16`, `num_layers=11`:

| Component                  | Baseline (mlp_mult=4)          | This submission (mlp_mult=16)                          |
|----------------------------|--------------------------------|--------------------------------------------------------|
| MLP weights stored         | 11 × (2 M params) int6 ≈ 17 MB | 11 × 0 params ≈ 4 bytes total (one seed)               |
| LoRA A (16, 512) × 2       | —                              | 16 K params × 11 (small -> fp16 passthrough) ≈ 0.35 MB |
| LoRA B (8192, 16) × 2      | —                              | 262 K params × 11 int6 ≈ 2.4 MB                        |
| Per-hidden gate (8192,)    | —                              | 8 K params × 11 fp32 passthrough ≈ 0.35 MB             |
| Attention, embed, norms    | ~8 MB                          | ~8 MB                                                  |
| Code                       | 68 KB                          | 74 KB                                                  |
| **Total (pre-compression)** | ~25 MB -> ~16 MB after brotli  | ~11 MB -> ~8–10 MB after brotli                        |

We net ~4–6 MB of headroom even while running a 4× wider hidden
dimension. That headroom is available to spend on more layers, a bigger
vocab (SP8192/SP12288), or a longer training schedule.

## Hyperparameters

```bash
# Defaults in Hyperparameters — can be overridden via env vars
MLP_MULT=16
LORA_RANK=16
RANDOM_BASIS_ENABLED=1
RANDOM_BASIS_SEED=0xD15EA5E   # deterministic per-layer via base_seed + 1009*layer_idx
NUM_LAYERS=11
MODEL_DIM=512
VOCAB_SIZE=4096                # same tokenizer path as PR #1218 backbone
```

Training/optimizer hyperparameters (Muon, warmdown, WD, EMA, GPTQ) are inherited unchanged from the forked backbone
(`records/track_10min_16mb/2026-04-01_Vocab4096_MLPMult4_WD085/`) to keep the ablation clean the only change is the MLP module.

## Command(s)

```bash
RUN_ID=rbmlp_seed42 \
DATA_PATH=./data/datasets/fineweb10B_sp4096/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
SEED=42 VOCAB_SIZE=4096 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-16_RandomBasisMLP_LoRA/train_gpt.py
```


## Attribution

- Backbone forked from
  [`records/track_10min_16mb/2026-04-01_Vocab4096_MLPMult4_WD085`](../2026-04-01_Vocab4096_MLPMult4_WD085/)
  (Kevin Clark, PR #1218). 
- The LoRA pattern is a direct adaptation of the one in
  [`records/track_10min_16mb/2026-03-17_LoRA_TTT`](../2026-03-17_LoRA_TTT/)
  (samacqua).
- Random-features math: Rahimi & Recht, "Random Features for Large-Scale Kernel Machines", NeurIPS 2007. [nips.cc](https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html) 
- Low-rank adaptation: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022. [arxiv.org(https://arxiv.org/abs/2106.09685) 
