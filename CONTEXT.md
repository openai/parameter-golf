# Parameter Golf Competition Context

## What This Is
OpenAI's Parameter Golf competition: train the best language model that fits in a 16MB artifact and trains in 10 minutes on 8xH100 SXM. Metric is val_bpb (bits per byte) on FineWeb validation set. Lower is better. Runs March 18 to April 30, 2026. Repo: https://github.com/openai/parameter-golf

## Current Status (as of 2026-03-27)
- **Our PRs**:
  - PR #838: **1.1215 BPB** (pure neural + TTT, unquestionably legal) — OPEN
  - PR #893: **0.1310 BPB** (two-pass order-12 n-gram, legality disputed) — OPEN
  - PR #635, #754, #864, #865: closed (superseded)
- **Our fork**: https://github.com/aryanbhosale/parameter-golf
  - Branch `submission/11l-parallel-muon-lnscale-ttt` → PR #838 (neural)
  - Branch `submission/twopass-ngram-0.1310` → PR #893 (n-gram)
- **Official merged SOTA**: 1.1194 BPB (PR #549, @abaybektursun)
- **N-gram frontier** (legality disputed): 0.0804 BPB (PR #933, @haikosys)
- **Our gap to SOTA**: 0.002 BPB (neural), beaten by 4x (n-gram)

## Leaderboard Landscape (2026-03-27)
### Pure Neural (unquestionably legal)
| Rank | BPB | Author | PR |
|------|-----|--------|-----|
| 1 | 1.1194 | @abaybektursun | #549 (merged SOTA) |
| 2 | **1.1215** | **@aryanbhosale** | **#838** |
| 3 | 1.1228 | @signalrush | merged |

### N-gram Cache (legal per original ruling, but hash inflation + two-pass contested)
| BPB | Technique | Status |
|-----|-----------|--------|
| 0.0804 | CacheMoney (full-rescore + online alpha) | Legality TBD |
| 0.0887 | Cache Is All You Need (622KB artifact!) | Hash inflation concerns |
| 0.0935 | BROADSIDE (full-rescore all chunks) | Two-pass = likely illegal |
| **0.1310** | **Ours (two-pass order-12)** | Two-pass = likely illegal |
| 0.0280 | Frozen N-gram Oracle (pre-filled from 8B train tokens) | Likely illegal |

## Our Best Architecture (PR #838, 26.8M params, 1.1215 BPB)
- 11 layers, dim=512, 8 heads, 4 KV heads (GQA), MLP 3.0x (hidden=1536)
- **Parallel Muon**: 4 contiguous 3D parameter banks (qo_bank, kv_bank, mlp_up_bank, mlp_down_bank), batched Newton-Schulz via torch.bmm, 3-phase async reduce-scatter → Adam → NS5+all-gather. No DDP.
- LeakyReLU(0.5)^2 activation
- SmearGate + BigramHash(1536, dim=128), NO TrigramHash
- Value Residual (ResFormer): cache V from layer 0, blend via learned lambda
- Gated Attention: nn.Linear(dim, num_heads, bias=True), zeros weight, bias=4.0
- XSA on last 4 layers (XSA-all-11 causes 7x slowdown with torch.compile)
- Partial RoPE: 16/64 head dims
- U-Net skip connections, OrthoInit, tied FP16 embeddings, logit softcap 30.0
- Flash Attention 3 (`flash_attn_interface`), torch.compile(fullgraph=True)
- ~89ms/step on 8xH100, ~6750 steps in 600s

## Our Training Config
- Muon lr=0.025, WD=0.04, momentum 0.92→0.99 over 1500 steps, Newton-Schulz 5 steps
- Adam for embeddings (lr=0.035), scalars (lr=0.025), betas=(0.9, 0.95)
- EMA decay=0.997, SWA every 50 steps when scale < 0.2
- Batch 786K tokens (8xH100), seq_len=2048
- Warmdown 3500 iters (wallclock-based)
- Late QAT via STE (final 15% of wallclock time)
- Gradient clipping 0.3

## Our Quantization
- Int6 uniform (range [-31,31]) with GPTQ-lite (5-percentile clip search per row)
- FP16 passthrough for tok_emb.weight
- zstd-22 compression
- Unbank → quantize → rebank for parameter banking compatibility
- Artifact size: ~15.8 MB (under 16MB limit)

## Our Eval
- Sliding window, stride=64, seq=2048
- Legal score-first TTT: SGD momentum=0.9, lr=0.002, 3 epochs, all blocks unfrozen, 32K-token chunks, cosine LR decay
- N-gram cache (PR #893 only): order 2-12, 4M buckets, 256K chunks, entropy-adaptive alpha, two-pass rescore

## Key Results (3-seed means, 8xH100)
| Config | EMA | Quantized | TTT | N-gram |
|--------|-----|-----------|-----|--------|
| Old (PR #635, MLP 3.5x, no Parallel Muon) | 1.1194 | 1.1283 | 1.1253 | — |
| + Parallel Muon + FA3 (MLP 3.0x, XSA4) | 1.1194 | 1.1283 | 1.1253 | — |
| + LN Scale + bigram (PR #838) | 1.1160 | 1.1235 | **1.1215** | — |
| + N-gram single-pass | 1.1190 | 1.1274 | — | 0.2841 |
| + N-gram two-pass order-12 (PR #893) | 1.1190 | 1.1269 | — | **0.1310** |

## Setup Required on New Pod
```bash
cd /workspace && rm -rf parameter-golf && git clone https://github.com/aryanbhosale/parameter-golf.git && cd parameter-golf
# For neural submission:
git checkout submission/11l-parallel-muon-lnscale-ttt
# For n-gram submission:
git checkout submission/twopass-ngram-0.1310
pip install --break-system-packages zstandard sentencepiece huggingface-hub
python data/cached_challenge_fineweb.py
cp records/track_10min_16mb/2026-03-25_11L_ParallelMuon_MLP3x_TTT/train_gpt.py .
# OR for n-gram:
cp records/track_10min_16mb/2026-03-26_11L_ParallelMuon_NgramBackoff/train_gpt.py .
```

## 8xH100 Run Commands
```bash
# Neural + TTT (PR #838):
for SEED in 1337 42 2024; do
  SEED=$SEED MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 USE_EMA=1 EMA_DECAY=0.997 \
  SWA_ENABLED=1 SWA_EVERY=50 WARMDOWN_ITERS=3500 TRAIN_BATCH_TOKENS=786432 \
  USE_TTT=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
  TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 \
  torchrun --nproc_per_node=8 --standalone train_gpt.py > run_seed${SEED}.log 2>&1
done

# N-gram (PR #893):
for SEED in 1337 42 2024; do
  SEED=$SEED MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 USE_EMA=1 EMA_DECAY=0.997 \
  SWA_ENABLED=1 SWA_EVERY=50 WARMDOWN_ITERS=3500 TRAIN_BATCH_TOKENS=786432 \
  USE_TTT=0 NGRAM_EVAL=1 NGRAM_CHUNK_TOKENS=256000 \
  torchrun --nproc_per_node=8 --standalone train_gpt.py > run_seed${SEED}.log 2>&1
done
```

## 1xH100 Commands
```bash
# Quick experiment (no EMA/SWA, fast eval):
PYTHONUNBUFFERED=1 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=256 USE_EMA=0 SWA_ENABLED=0 \
WARMDOWN_ITERS=300 TRAIN_BATCH_TOKENS=524288 python -u train_gpt.py > run.log 2>&1
# TTT on 1xH100 DIVERGES with reference params! Use conservative:
TTT_LR=0.0005 TTT_EPOCHS=1 TTT_FREEZE_BLOCKS=2
```

## Local Files
- `/Users/aryan.bhosale/Desktop/param/parameter-golf/` — the repo
- `/Users/aryan.bhosale/Desktop/param/h100_runs/` — all run logs and artifacts:
  - `h100_runs/` — early 1xH100 experiments
  - `h100_runs/pmuon_1xh100/` — Parallel Muon 1xH100 test
  - `h100_runs/8xh100_submission/` — first 8xH100 run (1.1253 BPB)
  - `h100_runs/8xh100_final/` — 8xH100 with LN Scale (1.1215 BPB)
  - `h100_runs/8xh100_ngram/` — n-gram single-pass (0.2841 BPB)
  - `h100_runs/8xh100_twopass/` — n-gram two-pass (0.1310 BPB)

## What We Tried and What Happened

### WORKED (kept)
| Change | Impact | Notes |
|--------|--------|-------|
| Parallel Muon | -0.008 BPB raw, 89ms/step | Ported from PR #549. Parameter banking + batched NS5. No DDP. |
| Flash Attention 3 | 193ms→89ms/step | `flash_attn_interface` import, [B,T,H,D] in/out |
| lr=0.025 (from 0.03) | -0.002 BPB | Tuned for 8xH100 step count |
| Drop TrigramHash | -0.002 BPB + saves params | Community consensus: trigram doesn't help |
| Reduce bigram 10240→1536 | ~neutral BPP, -1MB artifact | Matches merged SOTA |
| LN Scale 1/sqrt(layer+1) | -0.003 BPB | BUT pushes artifact over 16MB with VE |
| QAT clamp [-32,31] (amax-aligned) | ~neutral | Matches actual int6 range |
| TTT rewrite (score-first) | -0.002 BPB | SGD momentum=0.9, cosine LR, all blocks unfrozen |
| EMA on GPU (not CPU) | faster steps | Remove .cpu() in EMA update |
| N-gram backoff cache | 1.12→0.28 BPB | Order 2-9, 65K chunks, entropy-adaptive alpha |
| Two-pass rescore | 0.28→0.13 BPP | Rescore cold chunks with full cache |
| Order 12 (from 9) | -0.01 BPP | 3 more hash primes, higher order catch longer patterns |

### FAILED (reverted/dropped)
| Change | Result | Why |
|--------|--------|-----|
| Full Hessian GPTQ | +0.16 BPB degradation | Conflicts with QAT. QAT trains weights for round-to-nearest; GPTQ error compensation fights this. Merged SOTA also uses GPTQ-lite, not full GPTQ. |
| LZMA compression | +5MB larger artifact | LZMA is WORSE than zstd-22 for our int6 quantized tensors |
| Selective magnitude pruning (10%) | +0.14 BPB degradation | Too aggressive with QAT-trained weights |
| Selective magnitude pruning (3%) | +0.18 BPB degradation | Still too aggressive with QAT |
| XSA on all 11 layers | 7x slowdown (89ms→623ms) | Breaks torch.compile fullgraph optimization somehow |
| VE128 (Value Embeddings) | -0.004 BPB but +300KB artifact | Pushes over 16MB limit. VE64 also too large. |
| LN Scale + VE + bigram | Best EMA (1.114) but 16.3MB | Over budget. Can't fit all three. |
| MLP 3.5x | 17% more FLOPs, fewer steps | MLP 3.0x is better at 8xH100 step count |
| Orthogonal Residuals (arXiv:2505.11881) | +0.05 BPB | Tested by community. Doesn't help at small model scale. |

## Legality Rules (from Issues #677, #402)
### LEGAL
- Score-first TTT (score chunk under inference_mode, then train on scored tokens)
- Single-pass backward-looking n-gram cache (per original ruling)
- GPTQ-lite at any time (no calibration data needed)
- Full GPTQ within 600s training window (uses training data)
- Late QAT, EMA, SWA, all architectural changes

### ILLEGAL
- Multi-epoch TTT where final-epoch score is reported
- GPTQ calibration using training data DURING evaluation (after 600s train)
- Oracle/min(NLL) selection across multiple passes
- Peeking at true tokens to choose predictor

### DISPUTED / UNDER REVIEW
- **Two-pass N-gram rescoring**: Issue #677 says "violates causality" — pass 2 rescores token #100 using cache with tokens #101-62M. Multiple community members call it "obviously invalid." Our PR #893 uses this.
- **N-gram hash collision inflation**: @Eppie proved hash ratio is NOT a real conditional probability. With 256M buckets (collision-free), n-gram gives 1.11 BPB (same as baseline). Apparent sub-0.3 BPB scores are partially artifacts of hash collisions.
- **Training-data n-gram oracles**: Pre-filling from 8B FineWeb training tokens (PR #924, 0.028 BPB). Carries hundreds of MB of training state not serialized in artifact.
- **Auxiliary eval-time state cap**: Proposed 32MB cap on eval-time state would constrain n-gram caches. Not yet official.

## GitHub Auth
- `arbyte77` (Alaan work account): default for git operations, `git config user.email aryan.bhosale@alaanpay.com`
- `aryanbhosale` (personal): used for PRs and issue comments. Switch with `gh auth switch --user aryanbhosale`
- For commits that show as `aryanbhosale` on GitHub, use email `36108149+aryanbhosale@users.noreply.github.com`
- Fork: aryanbhosale/parameter-golf, upstream: openai/parameter-golf
- IMPORTANT: `unset GITHUB_TOKEN` before `gh` commands to avoid auth conflicts

## Key Competition Rules
- Artifact (code + compressed model) <= 16,000,000 bytes
- Training: 10 min on 8xH100 SXM
- Evaluation: 10 min on 8xH100 SXM (separate budget)
- New SOTA records need >= 0.005 nats improvement, p < 0.01, minimum 3 seeds
- TTT must be score-first (score chunk BEFORE training on it)
- No training on validation data during training phase
- Tokenizer files don't count toward 16MB
- Submission must be in `/records/track_10min_16mb/` with README.md, submission.json, train_gpt.py, and 3-seed logs

## Remaining Gap to SOTA (1.1194)
Our EMA (1.116) beats SOTA's pre-TTT (1.122) by 0.006. But quantization adds 0.008 BPP degradation (SOTA has near-zero). This eats the advantage. TTT recovers only 0.002.

To close the 0.002 gap:
1. **Reduce quant degradation** — PR #634 achieves 0.002 quant degradation without QAT by using full GPTQ within training time. This requires disabling QAT entirely. We tried this but GPTQ conflicted with our QAT-trained weights. The path: train WITHOUT QAT, then apply GPTQ within the 600s window.
2. **Fit LN Scale + VE under 16MB** — use int5 for MLP weights to free size budget, or remove bigram to make room for VE.
3. **Better TTT** — tune SGD lr, epochs, chunk size for better recovery.

## Anti-Patterns (NEVER try these)
MoE, int4, 12L at seq2048, SwiGLU alone, SmearGate without OrthoInit, SWA with bf16, EMA+SWA combined naively, sequential curriculum, higher embed LR 0.08, NCCL_IB_DISABLE=1, NorMuon, MUD optimizer, MTP, NUM_KV_HEADS=1, seq_len=4096, ternary/BitNet, per-window TTT, focal loss for TTT, early QAT on short runs, step-based LR schedule, document isolation in eval, EMA decay=0.999, EMA without XSA, cautious WD (breaks torch.compile), full Hessian GPTQ with QAT (conflicts), magnitude pruning with QAT (degrades), LZMA for our payload (worse than zstd), XSA on all 11 layers (breaks torch.compile speed), orthogonal residuals (hurts at this scale)
