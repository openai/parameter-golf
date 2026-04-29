# 11L Int6 + Tuned LR + FP16 Embed

## Summary
11-layer transformer with int6 quantization export, FP16 embedding passthrough,
and tuned Muon optimizer schedule. Non-record submission focused on systematic
exploration of architecture, quantization, and hyperparameter choices.

## Key Changes from Baseline
- **11 transformer layers** (vs 9 baseline) — more depth, funded by int6 compression
- **Int6 quantization export** (QUANT_MAX=31) — narrower value range compresses better with zlib
- **FP16 embedding passthrough** — keeps tied embedding in fp16 to eliminate quantization tax
- **Tuned LR schedule** — lower matrix_lr (0.025), higher Muon momentum (0.99), longer warmdown (3000)
- **MLP_HIDDEN override** — added env var for fine-grained MLP width control

## Architecture
- 11 transformer blocks, 512 model dim, 8 attention heads, 4 KV heads
- GQA attention with RoPE, ReLU² MLP (2x expansion)
- Tied embeddings with 1024 BPE vocabulary

## Run Command
NUM_LAYERS=11 
WARMDOWN_ITERS=3000 
MATRIX_LR=0.025 
SCALAR_LR=0.025 
TIED_EMBED_LR=0.035 
MUON_MOMENTUM=0.99 
MUON_MOMENTUM_WARMUP_START=0.92 
MUON_MOMENTUM_WARMUP_STEPS=1500 
QUANT_MAX=31 
torchrun --standalone --nproc_per_node=8 train_gpt.py
## Results (1 * RTX4090, local validation)
| Metric | Value |
|---|---|
| val_bpb (post-quant) | 1.30660277 |
| val_bpb (pre-quant) | 1.3067 |
| Quantization tax | 0.0001 bpb |
| Artifact size | 14,987,861 bytes (14.99 MB) |
| Steps completed | 10,000 |
| Step time | 149.89ms |
| Training time | ~25 minutes |
| Batch tokens | 65,536 |

Note: this is a local run on a 4090 with 1/8th the batch size of the official 8XH100 setup. 
official run could not be completed due to some nagging pytorch version compatibility issues.
expected score on H100 would be significantly better due to a larger and more training steps within the 10 min wall clock limit. 

## Local Experiment Ladder
Run 1: Baseline 9L                    → 1.3323 bpb (local, 10K steps)
Run 2: + Tuned LR                     → 1.3201 bpb (Δ = -0.0122)
Run 3: + FP16 embed (over budget)     → 1.3174 bpb
Run 4: 11L + int6 export              → 1.3066 bpb (Δ = -0.0257, best)
Run 5: 11L + QAT from step 0          → 1.3169 bpb (QAT hurt — see notes)
## Architecture Ablation (5000 steps each, local)
Config A: 9L MLP_HIDDEN=992   → 1.3465 bpb, 15.90 MB (fits int8)
Config B: 11L MLP_MULT=2      → 1.3328 bpb, 19.55 MB (needs int6)
Config C: 9L MLP_MULT=3       → 1.3312 bpb, 20.36 MB (needs int6)
Config D: 11L MLP_MULT=3      → 1.3209 bpb, 24.67 MB (needs int6)
## Things Tried That Didn't Work
- **QAT from step 0**: Reduced quantization tax to 0.0001 bpb but hurt overall
  model quality by 0.01 bpb. Net negative. The optimal approach (FP first, then
  QAT during warmdown) was blocked by torch.compile(fullgraph=True) ignoring
  runtime variable toggles. A two-stage training script would fix this.
- **FP16 embed with baseline MLP**: Artifact exceeded 16MB. Required MLP_HIDDEN
  override to shrink MLP and make room.

## Code Changes
1. FP16 embed export in quantize_state_dict_int8 (~line 378)
2. MLP_HIDDEN env var override in MLP.__init__ (line 615)
3. QUANT_MAX env var in quantize_float_tensor (line 322, 347-348)
4. QAT infrastructure: fake_quantize function + maybe_quantize dispatch (line 94-107)
5. QAT toggle in CastedLinear.forward (line 538)
6. Conditional torch.compile skip when QAT enabled (line 869)

## Files
- train_gpt.py — modified training script
- train.log — 8xH100 log
- submission.json — leaderboard metadata
