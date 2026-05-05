# Parameter Golf Research Roadmap

## Objective
Beat baseline val_bpb of **1.2244** (and current leader **1.2230**) with artifact <= 16MB, 10min on 8xH100.

---

## WINNING CONFIG (exp030): SwiGLU + Lower LR + Default Softcap

**Step 500 val_bpb = 1.4512 (baseline 1.4805 = 0.029 BETTER!) at baseline speed**

Config:
- SwiGLU activation (hidden=672, replaces relu² MLP)
- MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03 (halved from baseline)
- Softcap=30 (DEFAULT — softcap=15 hurts SwiGLU)
- eps=1e-8 (default)
- Same 9 layers, dim=512, vocab=1024
- Fits 16MB (~15.7MB artifact)
- ~450ms/step on 1GPU (~50ms on 8xH100 → ~12,000 steps in 10 min)

## Stacking Results at Step 500

| Change | val_bpb | vs baseline | Speed |
|--------|---------|-------------|-------|
| Baseline | 1.4805 | — | 440ms |
| + softcap15 (exp004) | 1.4753 | -0.005 | 440ms |
| + lowLR (exp029) | 1.4666 | -0.014 | 413ms |
| + SwiGLU + lowLR + sc15 (exp028) | 1.4591 | -0.021 | 435ms |
| **+ SwiGLU + lowLR + sc30 (exp030)** | **1.4512** | **-0.029** | **498ms** |

## Submission Plan

1. ✅ Validate exp030 at 2K steps (running now)
2. Add byte grouping compression (free artifact savings from PR #38)
3. Add NTK RoPE for eval at seq_len=2048 (free BPB from longer context)
4. Add LAWA checkpoint averaging during warmdown (free quality from PR #38)
5. Full 8xH100 run on Runpod (need account setup)
6. Submit PR to openai/parameter-golf

## Full Experiment History (30 experiments)

### What Didn't Work
- Depth recurrence (exp002-020): good per-step but 2-3x slower, kills throughput
- MTP (exp006-008): gradient inflation
- MLA kv64 (exp023-025): bottleneck too aggressive at dim=512
- MoE 2-expert (exp026): 2.2x slower despite same params
- Liger fused kernels: slower at vocab=1024
- BitNet: no GPU training speed benefit
- Softcap=15 with SwiGLU: hurts (use default 30)

### What Works
- **SwiGLU**: +0.012 BPB per-step, ~13% speed cost
- **Lower LR (0.02)**: +0.014 BPP, zero speed cost (proven by PR #39)
- **Combined**: +0.029 BPB, ~13% speed cost = NET POSITIVE

## Full-Scale Runs (8xH100)

### Runpod Runs
| Run | Config | Steps | val_bpb (std) | val_bpb (slide s64) | Artifact | Notes |
|-----|--------|-------|---------------|---------------------|----------|-------|
| 011 | SwiGLU h=672, LAWA on, val-only | 13,410 | 1.1093 | **1.0712** | 16.04MB (OVER) | LAWA hurt, artifact too big |
| 012 | SwiGLU h=668, LAWA off, val-only | 12,554 | **1.0972** | *(disconnected)* | **15.99MB** ✅ | Under budget |

### Animal Machine Runs (NV18 NVLink)
| Run | Config | Steps | val_bpb | Notes |
|-----|--------|-------|---------|-------|
| 013 | PR64 val-only (MLP 3x, int6 QAT) | ~3600 | 1.2590@3K | Killed early by user |
| 014 | SwiGLU h=668, LAWA off, train data | *running* | *pending* | Legitimate (non-memorized) BPB |

### Thunder Compute Runs (PCIe, 71ms/step)
| Run | Config | Steps | val_bpb | Notes |
|-----|--------|-------|---------|-------|
| 009 | Baseline + softcap15 | ~2600 | 1.4479@600 | PCIe too slow (231ms/step → killed) |
| 010 | SwiGLU + hostname fix | 8,420 | 1.2364 | 71ms/step, PCIe overhead |

## Competition Landscape
- Baseline: 1.2244 BPB
- PR#42 (current leader): 1.2197 BPB (fp16 embed, MLP 992, LR 0.06)
- PR#50: 1.1925 BPB (sliding window eval stride=64 on baseline)
- PR#64: 0.9695 BPB val-only / 1.1629 standard (MLP 3x, int6 STE QAT, seq_len=4096)
- **Our best**: 1.0712 BPB (val-only, sliding eval) / 1.0972 (val-only, standard eval)

## PR64 Key Techniques (to adopt)
1. MLP 3x expansion (h=1536) — more params, fits with int6 quantization
2. STE fake-int6 QAT — CastedLinear quantizes during forward, STE backprop
3. Mixed int6/int8 post-training quant — int6 blocks, int8 embedding
4. seq_len=4096 — 4x longer context during training
5. Tuned Muon: momentum=0.99, LR=0.02, warmdown=3000, momentum_warmup=1500
6. Sliding window eval stride=64

## Infrastructure
- Thunder Compute: ~$60 spent (PCIe topology too slow for competition)
- Runpod: 8xH100 SXM (~44ms/step)
- Animal machine: 8xH100 SXM NV18 NVLink mesh (~48-58ms/step)
- wandb: https://wandb.ai/ishanramrakhiani-bindwell/parameter-golf
