# Parameter Golf Competition - Project Guide

## What This Is
OpenAI Parameter Golf: train the best LM in 16MB, 10 min on 8xH100. Metric: val_bpb (lower = better).
Baseline: 1.2244 BPB. Our target: < 1.15 BPB. **WE ARE THE CURRENT LEADER.**

## Current Best Config (proven on 8xH100) — PR #114
```
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 MLP_HIDDEN=1536
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3
EVAL_SEQ_LEN=2048 EVAL_STRIDE=256
+ int6 quantization (per-row, ±31 range) for all weight matrices
+ FP16 tied embedding + Late-K passthrough (last 2 layers c_k)
```
On 8xH100 (RunPod): **1.1574 BPB** (baseline: 1.2244). 21.8M params in 15.98MB.

## ENVIRONMENT CRITICAL NOTE
- **Official eval environment**: Python 3.12 + PyTorch 2.9.1+cu128 (RunPod template)
- Our 1.1574 was produced on py3.10 + torch 2.6.0 (83ms/step, 7199 steps)
- On the official env: same config gives ~1.1632 (100ms/step, 5967 steps)
- The gap is purely from step throughput — model quality is identical
- **All future optimization must target the official 100ms/step environment**
- OpenAI re-runs submissions on their hardware — optimize for THEIR environment

## Competition Landscape (2026-03-20 evening)
| PR | BPB | Approach |
|----|-----|----------|
| #162 | 1.1483 | Int6+MLP3x+SmearGate+BigramHash+OrthoInit+SWA+MuonWD |
| #164 | 1.1538 | OrthoInit+Int6+MLP3x+SmearGate+BigramHash+FA3 |
| #173 | 1.1532 | Int6+MLP3x+FA3+NorMuon |
| **Ours (#114)** | **1.1574** | Int6+MLP3x+FP16embed+LateK+slide+train@2048 |
| #88 | 1.1605 | int6 STE QAT + MLP 3x + zstd + sliding window |

**Gap to leader: 0.009 BPP.** They use OrthoInit, SmearGate, BigramHash, SWA, MuonWD on top of int6+MLP3x.

## Techniques We Need to Add (from PR #162)
1. OrthoInit — orthogonal init for weight matrices, output proj scaled by 1/sqrt(2*layers)
2. SmearGate — learned sigmoid gate blending each token with previous (~512 params)
3. BigramHash — 4096-bucket hash embedding for token pairs (128→512 dim, ~524K params)
4. SWA — average checkpoints every 50 steps during last 50% of warmdown
5. MuonWD — weight decay 0.02 on Muon params, 0.01 on Adam params
6. Zstd-22 compression instead of zlib (needs pip install zstandard)

## Proven Wins (ranked by impact, 8xH100)
1. Sliding window eval stride=256 (+0.013 BPB — every token gets 1792+ context)
2. TRAIN_SEQ_LEN=2048 with LR=0.02, momentum=0.99 (+0.020 vs stock 1024)
3. Int6 quantization + MLP 3x (hidden=1536, 21.8M params in 16MB)
4. TRAIN_BATCH_TOKENS=786432 (+0.002 vs 524K default)
5. GRAD_CLIP_NORM=0.3 (+0.002 — specifically helps long-sequence training)
6. FP16 tied embedding + Late-K passthrough (last 2 layers c_k in fp16)
7. MUON_MOMENTUM_WARMUP_STEPS=1500 (optimal, 500 and 2000 both worse)

## Proven Losses (DO NOT RE-TEST)
- ALBERT embedding factorization (torch.compile overhead kills 32% of steps)
- NorMuon optimizer (110ms/step, throughput death on our hardware)
- MTP / multi-token prediction (86ms/step, aux head too expensive)
- SwiGLU at iso-params (narrower hidden doesn't beat ReLU²)
- WD=20000 with low LR (doesn't stack — low LR already smooths weights)
- NUM_KV_HEADS=2 (capacity loss not worth param savings)
- Int6 step=4 rounding with LR=0.06 (catastrophic — needs proper per-row ±31)
- Depth recurrence (abandoned by all teams)
- EMA weights (any decay)
- eval@2048 on well-trained 8xH100 models (NTK distortion)
- train@4096 vs train@2048 (identical with sliding window, 2048 gets more steps)
- Batch=1M (too few steps)
- WD=30000 (decays too fast)
- QK_GAIN_INIT=1.7 (no improvement over 1.5)

## Key Insight: Step Throughput Is King
On a 10-min budget, per-step overhead >10% is a net loss. Only MLP 3x (77% overhead, 28% more params) has ever compensated. NorMuon, MTP, ALBERT all failed this test.

**NEW REALITY**: On the official env (100ms/step), we get ~6000 steps instead of 7200. Techniques that were marginal before might now tip negative. Re-validate anything that adds per-step cost.

## Key Interaction Effects
- WD=20000 helps with high LR (0.06) but hurts with low LR (0.02)
- MUON_BACKEND_STEPS=5 beats 7 at WD=20000 but 7 beats 5 at WD=2400
- GRAD_CLIP matters more for long sequences than short sequences

## Cloud Compute
- RunPod pods with H100 SXM GPUs
- **Official template**: Python 3.12 + PyTorch 2.9.1+cu128
- SP1024 data: `/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/`
- SP4096 data: `/workspace/parameter-golf/data/datasets/fineweb10B_sp4096/`
- DISK: Pod has 50GB. Clean /tmp/torchinductor_root periodically.

## Our PRs
- #61: 1.2154 BPB (warmdown-quantization discovery, merged pending leaderboard)
- #96: 1.1764 BPB (sliding window + long-context training)
- #114: 1.1574 BPB (int6 + MLP 3x + selective precision) — **current leader**
