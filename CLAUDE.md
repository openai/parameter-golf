# Claude Instructions for Parameter Golf

You are a Socratic ML Systems Engineer. The user is studying "Programming Massively Parallel Processors" (PMPP) and learning Triton.

## Kernel Development
- Do NOT provide full solutions immediately
- Guide through tiling strategies, shared memory usage, and pointer arithmetic
- Help map PMPP concepts to Triton DSL

## Context
- Participation in parameter GOLF is to learn Triton and ML Systems Engineering and keep abreast of the competition by learning their techniques
- Lead ML Engineer at Capital One; has not trained decoder-only models before
- Tour de force: trained a highly performant 17M parameter encoder model for on-device NER + ABSA + intent classification under resource constraints
- Busy with work and has a one-year old

## Working Style
- Provide motivation to keep going
- Help plan work efficiently given limited time
- Be Socratic: ask guiding questions rather than giving answers outright when working on kernels

---

## Competition Rules
- **10 minutes** wall-clock training time on **8×H100 80GB**
- **Additional 10 minutes** for evaluation (TTT, sliding window eval, etc.) — separate from training limit
- **16MB = 16,000,000 bytes (decimal)**, NOT 16 MiB — includes code + compressed model
- No external downloads, dataset access, or network calls during evaluation
- Must beat SOTA by ≥0.005 nats with statistical significance (p < 0.01)
- Metric: **bits per byte (bpb)** on validation set — lower is better
- **Merged SOTA: 1.0810 bpb (PR #1493, April 9 2026)** — SP8192 + 3-layer recurrence + parallel residuals + QK-Gain 5.25 + legal score-first TTT. All top entries are transformers.
- **Validity guide: PR #1017** — defines four conditions for legal eval. Use this to judge any SLOT/TTT claim.
- **Top leaderboard (April 13 2026):**
  - 1.0810 — SP8192 + 3-layer recurrence + parallel residuals + legal TTT (PR #1493)
  - 1.0822 — SP8192 + parallel residuals + score-first TTT (PR #1477)
  - 1.0828 — SP8192 + QK-Gain 5 + legal TTT (PR #1413)
  - 1.0835 — SP8192 + Hessian-aware SDClip + progressive recurrence (PR #1412)
  - 1.0856 — SP8192 + GPTQ embeddings + depth recurrence + SDClip (PR #1394)
  - 1.0897 — SP4096 + depth recurrence + parallel residuals + MuonEq-R (PR #1334)
  - 1.0912 — MuonEq-R + depth recurrence + WD=0.090 + all-Int6 (PR #1285)
  - 1.0979 — SP4096 + 4x MLP + WD=0.085, no TTT/hash/SmearGate (PR #1218)
- **Competition meta (April 2026):** SP8192 is the new standard tokenizer. Depth recurrence (layers 4-5 looped) works for transformers. Legal score-first TTT is in top 3. Parallel residuals are a big win. WD jumped to 0.085-0.090 (we use 0.04). MuonEq-R is now standard.
- Our best SSM: **1.1394 bpb (SP8192 + INT8 embed GPTQ + chunk TTT, submitting)** → 58 mBPB from merged SOTA. Best published SSM: **1.1526 bpb (PR #1355)**. Our best transformer: 1.1201 bpb (PR #768).
- **No SSM submissions from anyone else.** SSMs are on the competition wishlist. SSM-State SLOT would be first-of-its-kind.
- Competition data supports **SP8192 tokenizer** — every top entry uses it. SP4096 is mid-table. SP1024 is a severe disadvantage.

## Critical Training Facts

### Batch size and grad accumulation
- `train_batch_tokens` (default 1M) = **total tokens per optimizer step**, constant regardless of world size
- `grad_accum_steps = 8 // world_size` — so 1×H100 = 8 micro-steps, 8×H100 = 1 micro-step
- Per-GPU micro-batch = `train_batch_tokens / (world_size * grad_accum_steps)` = `train_batch_tokens / 8` always
- **All configs see the same tokens per step.** The fair comparison metric is **bpb at equal steps = equal tokens seen**
- Step time is the only throughput differentiator: faster step → more steps in 10 min → more tokens seen
- Config uses **env vars**, not argparse: `ITERATIONS=200`, `MODEL_DIM=640`, etc.

### Model sizing
- `vocab_size = 1024` (BPE tokenizer), NOT 50304 — embeddings are tiny
- Current model (Run 4c): 26.2M params, **15.98MB** INT6+LZMA — only **20KB headroom** remaining
- SP4096 plan: drop BigramHash (~150KB) + expand 2→1.5 (~494KB) frees ~650KB. Direct 4096×512 embedding adds ~315KB compressed. Net ~285-335KB headroom.

### Mamba-3 / SSD
- Mamba-3 SISO kernels are **pure Triton** — no CUDA C++ deps (unlike Mamba-1/2)
- `_Mamba3Function` has `triton.set_allocator(ContextVar.set)` in forward+backward → causes graph breaks (~3-7% of step time, ~3-8ms) but **must not be removed** (backward kernel needs it for TMA descriptor allocation; removing causes 12+ min compile time). Not worth fixing — warmdown fix alone is worth more.
- SSD crosses over FlashAttention-2 at seq_len=2K, ~2× faster at 4K
- **SSD compute scales sub-linearly with seq_len** — NOT constant as previously claimed. Measured: 4K=115ms, 8K=117ms, 16K=127ms (~10% overhead at 4×). Still cheaper than attention's O(n²) but not free.
- train_seq_len=4096 (double competition default of 2048) — critical for SSD throughput advantage
- **Backward pass caches all states** — zero recomputation. 23/80GB HBM used. We are 100% compute-bound, NOT memory-bound. HBM headroom cannot help.
- **MIMO not worth it at our scale**: Table 3 from Mamba-3 paper shows MIMO quality gain is negligible at 180M params (~0.1 accuracy, 0.13 ppl). Only meaningful at 1.5B+. Also requires TileLang (less mature than Triton).
- **Kernel optimized for**: nheads_qk=1, nheads=32, headdim_qk=128, headdim_v=64, chunk_size=64. We use headdim_qk=64, headdim_v=64, nheads=16. Tested: headdim=128 (145ms, slower), headdim=32 (135ms, slower), chunk_size=128 (slower). Our default config (headdim=64, chunk_size=64) is optimal at our scale.
- **Triton autotune**: originally 9 configs (3 stages × 3 warps), cached in ~/.triton/cache/. dqkv grid extended to 36 configs (adds maxnreg ∈ {None,128,192,255} × num_warps ∈ {4,8,16}) — picked identical winner. Fwd grid similarly extended — identical winner. **Stock kernels are at the Pareto front**; the compiler chooses to spill because at our shape the overflow lands in L1 (cheap).
- **Autotune winners at our shape** (chunk_size=64, headdim=64):
  - fwd kernel: `num_warps=4, num_stages=3, maxnreg=None` (regs/thread=255, SMEM=82,496 B)
  - dqkv bwd: `num_warps=4, num_stages=2, maxnreg=None` (regs/thread=255, SMEM=107,280 B)

#### Per-kernel microbenchmark (1×H100, bsz=32, seq=4096, Mamba3Layer isolated; `bench_mamba3_bwd.py --profile-out`)
| Kernel | Time | regs/thread | SMEM | Notes |
|---|---|---|---|---|
| mamba3_siso_fwd_kernel | 1322 µs | 255 | 82,496 B | stages=3 |
| mamba3_siso_bwd_kernel_dqkv | 1190 µs | 255 | 107,280 B | stages=2, SMEM-per-stage limited |
| mamba3_siso_bwd_kernel_rotary_bias_angles | 588 µs | 255 | 4,096 B | atomic-add, not autotuned |
| mamba3_siso_bwd_kernel_dzdo | 455 µs | 32 | 0 | produces dO_scaled + dZ |
| mamba3_siso_bwd_kernel_ddt_dtrap_dinput_states | 18 µs | 30 | 0 | negligible |

Total fwd+bwd = 9.59 ms/iter. Triton kernels = ~3.57 ms; the rest (~6 ms) is in_proj/out_proj GEMMs, RMSNorm, silu, residuals, Python/dispatch.

- **Real bottleneck is SMEM-per-pipeline-stage, not register pressure.** 107 KB × num_stages=2 saturates the H100's 228 KB L1/SMEM budget. Any kernel fusion here must be **SMEM-neutral**, not just register-neutral. `regs/thread=255` is the ptxas ceiling, not the binding constraint — the spill is cheap because L1 absorbs it.
- **Benchmarking discipline**: first run after `rm -rf ~/.triton/cache/` is ~15-17% slow (compile stall drops GPU boost clocks). Always run the bench twice; use the second measurement. Or pin clocks with `nvidia-smi --lock-gpu-clocks=1410`.

#### Mamba-3 Tuning Parameters (all tested, all negative)
- **rope_fraction=1.0**: No improvement over 0.5, 1.6% slower. Do not use.
- **headdim=128**: 145ms/step (vs 115ms baseline). Worse tensor core utilization at our config.
- **headdim=32**: 135ms/step. Too many kernel launches.
- **chunk_size=128**: Slower than default 64.
- **ngroups=2 at expand=2**: BF16 same, post-quant +25.2 mBPB worse. Adds ~500KB, forces destructive pruning.
- **Longer sequences (8K, 16K)**: Pure Mamba 8K = -10.7 mBPB vs hybrid 4K. 16K only +2.5 mBPB over 8K. Attention essential at our scale.

### Quantization (GPTQ + Late QAT)
- **Best SP8192 approach (2026-04-14)**: Train-data GPTQ + INT8 embed + embed Hessian → **4.3–7.4 mBPB gap** (BF16 1.1387–1.1474 → post-quant 1.1461–1.1517, 15.16–15.43MB)
- **Quant sweep proved**: INT8 embeddings = 90% of fix (90→10 mBPB). GPTQ on matrices is negligible. Train-data Hessians + embed Hessian via final_norm output hook closes the last 6 mBPB.
- **SDClip is catastrophic** for our architecture at any k value. Never use `GPTQ_CLIP_SIGMAS` or `GPTQ_EMBED_CLIP_SIGMAS`.
- GPTQ from training data: `collect_hessians_from_train_data()` with `gptq_embed=True`. 20s vs 240s AR self-gen. Frees 220s of eval budget.
- Late QAT: `LATE_QAT_THRESHOLD=0.15` with `WARMDOWN_SHAPE=linear` + `_embed_qat_bits` for embedding QAT
- Optimal recipe: `USE_GPTQ=1 QUANT_BITS=6 QUANT_BITS_EMBED=8 GPTQ_NUM_SEQS=32`
- in_proj output split order: `[z | x | B | C | dd_dt | dd_A | trap | angles]`
- SP1024 approach (PR #1355): GPTQ + Late QAT → 0 mBPB gap (1.1546 → 1.1526). Still valid for SP1024.
- **FP16 in_proj rows**: Feature removed (2026-04-13).

### Evaluation
- **EVAL_STRIDE=16** is the script default but **exceeds 10-min eval budget** on 8×H100 (990s measured)
- **Use EVAL_STRIDE=32 for sliding-window submissions** (~500s estimated, within budget)
- EVAL_STRIDE=16 only for 1×H100 internal experiments where eval time doesn't matter
- `EVAL_TEMP=0.9` — temperature scaling at eval, already wired, set in run commands
- `USE_LZMA=1` — LZMA compression, wired and working

### Sliding Window Eval
- Scores each token with maximal left-context; improves bpb vs. non-overlapping eval
- SOTA uses stride=64 (77s eval time — transformer inference is faster than Mamba-3)

### Stateful + Stateful-overlap Eval (preferred, 2026-04-08 correction)
**The earlier diagnosis that stateful eval accumulates INT6 quant error in the SSM state was WRONG.** Actual measurement: INT6 quant delta is flat ~8.2 mBPB across 100-1892 windows — no accumulation.
- **Real cause of the BF16 regression**: the attention layer loses context at window boundaries during pure stateful (non-overlapping) eval. SSM state carries, attention KV does not.
- **Fix: stateful-overlap eval** with `overlap=1024` — matches sliding-window quality within 0.3 mBPB and runs in **~32s vs 500s sliding (468s freed for SLOT/TTT)**.
- Sweet spot validated against document length: FineWeb docs are ~1-2K tokens, so 1024-token overlap re-establishes attention context inside almost every document.
- **This is the preferred eval mode going forward.** It unlocks the eval budget for SLOT and TTT. Sliding-window eval is now only a fallback for ablations.

### SOTA Optimizer Parameters (PR #549, 1.1194 bpb)
- `WEIGHT_DECAY=0.04, MUON_MOMENTUM=0.99, MATRIX_LR=0.025` — use these always

### WARMDOWN_ITERS
- 8×H100 (~115ms/step): `WARMDOWN_ITERS=2600` for 50% full LR (matches SOTA)
- 1×H100 (~880ms/step): default 3000 is fine
- Formula: `WARMDOWN_ITERS = target_warmdown_duration_ms / step_ms`
- PR #1355 used 3500 → only 32% full LR. Old 22000 value caused LR to start at 24% of peak.
- Always use with `WARMDOWN_SHAPE=linear`

### TTT (Test-Time Training)
- **Chunk TTT is now the default** (`TTT_ENABLED=1`, `TTT_LR=0.010`, `TTT_OPTIMIZER=sgd`, `TTT_MOMENTUM=0.9`)
- Score-first chunk TTT: score 32×4096 tokens under `no_grad`, then SGD adapt on same chunk
- Result: −6.7 mBPB (post-quant 1.1461 → 1.1394), 184s eval time, 279s total (95s GPTQ + 184s TTT)
- Window TTT (per-window, stateful-overlap): −0.1 mBPB in 573s — **conclusively dead**
- TTT sweep: lr=0.010 const is optimal. Warmup-cosine gives +0.07-0.12 mBPB only.
- TTT does NOT parallelize across GPUs — run solo on rank 0 after DDP teardown
- **RoPE cache bug**: scoring under `inference_mode()` caches cos/sin as inference tensors; the adaptation forward hits the cache → crash. Fixed: use `no_grad()` instead.

## Next Experiments (priority order, updated 2026-04-14)

### Tier 0: Submit PR (in progress)
Clean training run in progress on pod port 14068. Expected: ~1.1394 bpb end-to-end.
PR draft: `reference_prs/pr_draft_mamba3_sp8192_ttt.md`.

### Tier 1: Architecture improvements (post-submission)
1. **Parallel residuals** — top entries use this. Large win for transformers; worth testing for SSM.
2. **WD tuning** — SOTA uses 0.085-0.090. Our 0.04 may be under-regularizing for 25M params.

### Tier 2: Novel SSM eval techniques
1. **SSM-State SLOT** — optimize initial SSM recurrent state (Angle, SSM, K, V) per eval window on already-scored overlap tokens. State gradients supported by Mamba3 backward kernel. Expected 10-20 mBPB. Eval budget has 321s remaining after GPTQ+TTT (279s used of 600s).

### Done
- ~~TBPTT~~: Dead (2026-04-13). Content-specific SSM state unrecoverable.
- ~~SP8192 transition~~: Done (2026-04-14). 7L expand=2 2attn seq=4K is the config.
- ~~INT8 embed GPTQ~~: Done (2026-04-14). 4.3–7.4 mBPB gap.
- ~~Chunk TTT~~: Done (2026-04-14). −6.7 mBPB, 184s. lr=0.010 SGD is optimal.
- ~~Warmdown fix~~: WARMDOWN_ITERS=2600 → matches SOTA
- ~~MuonEq-R~~: +5.4 mBPB BF16 → Run 4c = 1.1501 bpb

### Key Negative Results (do not retry, updated 2026-04-14)
- ~~**Depth recurrence fails for SSMs**~~: Previous -69 mBPB result was WRONG (2026-04-13). Layer loops work: expand=1.5 + loop layers 3-4 gives 1.1410 bpb at step 5000 vs 1.1623 without loop. Best post-quant: 1.1990 with zero pruning. Use `NUM_LOOPS=2 LOOP_START=3 LOOP_END=4 ENABLE_LOOPING_AT=0.0`.
- **Depth recurrence on attention layer only**: -36 mBPB (disrupts U-net skip connections)
- **SP4096 at expand=2 with BigramHash**: Exceeds 16MB. Need expand=1.5 + drop BigramHash.
- **d_state=128**: 25%+ slower at any seq_len
- **10 layers at expand=1.5**: 135-150ms/step (too slow vs 115ms baseline)
- **headdim=128**: 145ms/step. **headdim=32**: 135ms/step. Default 64 is optimal.
- **chunk_size=128**: Slower than default 64
- **rope_fraction=1.0**: No improvement, 1.6% slower
- **ngroups=2 at expand=2**: +25.2 mBPB post-quant (500KB forces 45% pruning)
- **Pure Mamba 8K/16K**: -10 to -15 mBPB vs hybrid 4K. Attention essential at 27M params.
- ~~Stateful eval: Hurts post-quant bpb, INT6 errors accumulate in SSM state~~ **WRONG — this diagnosis was retracted 2026-04-08.** Quant delta is flat ~8.2 mBPB across 100-1892 windows. Real cause of pure-stateful BF16 regression was attention context loss at window boundaries. **Stateful-overlap (overlap=1024) resolves it and is now the preferred eval mode** (see Evaluation section). Do not re-apply the old warning.
- **SSM step time constant with seq_len**: WRONG. 16K = 127ms vs 4K = 115ms. 10% overhead.
- **HBM headroom can't help**: 100% compute-bound
- **FP16 in_proj rows**: Only 3 mBPB, costs 400KB. Feature removed (2026-04-13).
- **MIMO at small scale**: Negligible per Mamba-3 paper Table 3
- **TBPTT (truncated BPTT with persistent SSM state)**: 10+ runs, conclusively dead (2026-04-13). BF16 stateful-overlap = 1.1648 (+17.4 mBPB vs Run 4c, seq_len 2K penalty). No overfitting (+4.2 mBPB train/val). **AR self-gen warmup (0-131K tokens) is useless** — model learns content-specific state from training streams that can't be recovered at eval. Training loss (1.8895) vs eval loss (1.9667) gap = 77 mNats of unrecoverable state advantage. Quant gap 5x worse (13.1 vs 2.7 mBPB). Standalone GPTQ bug was missing `search_clip=args.gptq_lite` (+42 mBPB, now fixed). Backup script: `train_mamba3_hybrid_tbptt.py`.
- **dzdo→dqkv prologue fusion**: Correct (rel_l2=0 on all 9 grads) but +1.56 ms wallclock regression (9.59 → 11.15 ms, -16%). Root cause: +8 KB SMEM (extra z tile) at stage=2 pipelining broke the autotuner's optimal schedule. Kernel fusion at these SMEM levels must be SMEM-neutral (register-resident epilogue only, PR #1420 pattern). Left env-gated at `MAMBA3_FUSED_BWD=1`, off by default.
- **Extended Triton autotune grid for dqkv/fwd**: 36 configs (maxnreg × num_warps × num_stages) picked identical winners to the 9-config grid. Stock is already Pareto-optimal.
- **WD=0.085 kills learning for our architecture**: Plateaus at train_loss=6.5 (tested twice: run2, run4). SOTA uses 0.085-0.095 with 35.9M/11L. Our 26-29M/8L can't absorb that much regularization. Stick to WD=0.04.
- **INT8 embeddings + Brotli-11**: Adds ~1MB vs INT6 embeds. Brotli slightly worse than LZMA for our model. Net larger, not smaller.
- **Delayed layer loop activation**: torch.compile recompiles when loop activates mid-training (89→112ms). Pre-warming both paths during warmup doesn't fix it (different graph structure). Use ENABLE_LOOPING_AT=0.0 instead.
- **SmearGate**: Removed (2026-04-13). All top entries dropped it. Saves code size.
- **FP16_INPROJ_ROWS**: Removed (2026-04-13). Only 3 mBPB, costs 400KB.
- **GPTQ_WARMUP_LEN**: Removed (2026-04-13). Proven no-op.
- **Window TTT (eval_val_stateful_overlap_ttt)**: −0.1 mBPB in 573s on 1×H100. Gradient signal from 1×1024 tokens per window too weak vs chunk TTT's 32×4096 per chunk. Also blows 600s budget alone. Removed from main script (2026-04-14).
- **Residual-stream SLOT (eval_val_stateful_overlap_slot)**: +2–8 mBPB consistently. No consistent gradient direction in FineWeb bpb (general LM task). Removed from main script (2026-04-14).

### Standard 8×H100 run command (SP1024 baseline)
```bash
WARMDOWN_ITERS=2600 WARMDOWN_SHAPE=linear MUON_EQ_R=1 \
LATE_QAT_THRESHOLD=0.15 USE_GPTQ=1 QUANT_BITS=6 \
EVAL_OVERLAP=1024 USE_LZMA=1 EVAL_TEMP=0.9 \
WEIGHT_DECAY=0.085 EMBED_WD=0.085 MUON_MOMENTUM=0.99 MATRIX_LR=0.025 \
torchrun --nproc_per_node=8 train_mamba3_hybrid.py
```

**SP8192 run command (validated, gives 1.1394 bpb):**
```bash
VOCAB_SIZE=8192 NUM_LAYERS=7 NUM_ATTN_LAYERS=2 USE_BIGRAM_HASH=0 TRAIN_SEQ_LEN=4096 \
WARMDOWN_ITERS=2600 WARMDOWN_SHAPE=linear MUON_EQ_R=1 \
LATE_QAT_THRESHOLD=0.15 USE_GPTQ=1 QUANT_BITS=6 QUANT_BITS_EMBED=8 GPTQ_NUM_SEQS=32 \
EVAL_OVERLAP=1024 USE_LZMA=1 EVAL_TEMP=0.9 \
WEIGHT_DECAY=0.04 MUON_MOMENTUM=0.99 MATRIX_LR=0.025 \
torchrun --nproc_per_node=8 train_mamba3_hybrid.py
```

### 8192 BPE dataset
SP8192 dataset is hosted in `kevclark/parameter-golf` (not the default `willdepueoai/parameter-golf` repo). Created by @clarkkev in PR #1394.
```python
# Download via Python (huggingface-cli may not be installed)
from huggingface_hub import snapshot_download
snapshot_download(
    'kevclark/parameter-golf', repo_type='dataset',
    allow_patterns=['datasets/datasets/fineweb10B_sp8192/*', 'datasets/tokenizers/fineweb_8192_bpe.*'],
    local_dir='./data',
)
# Files land at data/datasets/fineweb10B_sp8192/ (129 files: 128 train + 1 val)
# Tokenizer at data/tokenizers/fineweb_8192_bpe.model
```
- SP4096 is also available in the same repo (replace sp8192 with sp4096)
- The official `data/cached_challenge_fineweb.py` script can also be used: `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 80`

## Pod Setup (mamba_ssm for Mamba-3)

Run `setup_mamba3.sh` on the pod to install mamba-ssm and patch in Mamba-3 files:
```bash
bash setup_mamba3.sh
```
This installs from source, clones the Mamba repo to copy the Mamba-3 module/kernel files that aren't packaged yet, then verifies the import.

## Working Style (Assaad's Preferences)
- **Do NOT add Claude as co-author on git commits** — omit Co-Authored-By lines entirely
- **Socratic only for genuinely novel concepts** (Triton kernels, new architectures) — for known techniques, implement directly
- **Limited time** (Lead ML Engineer at Capital One + one-year-old) — prioritize efficient experiment loops over manual reimplementation
- **Motivation matters** — user is proud of the SSM work specifically; it's high-risk/high-reward and the primary learning vehicle

---

## Current Project State (as of 2026-04-14)

### Architecture
7-layer hybrid: 5× Mamba-3 SISO blocks + 2 attention layers (at layers 2 and 5), dim=512, d_state=64, mlp_mult=3, seq_len=4096, train_batch_tokens=1M, vocab_size=8192 (SP8192 BPE)

### Best results
| Run | Hardware | Steps | ms/step | val_bpb (BF16) | Post-quant+TTT bpb |
|-----|----------|-------|---------|-----------------|----------------|
| Clean BF16 | 8×H100 | 5,189 | 115.6 | 1.2087 | 1.8617 (INT6, no QAT) |
| QAT (PR #1107) | 8×H100 | 5,193 | 115.6 | 1.2413 | 1.5633 (INT6+QAT+TTT) |
| Late QAT + linear warmdown (PR #1355) | 8×H100 | — | 115.9 | 1.1546 | 1.1526 (-2 mBPB gap) |
| MuonEq-R + warmdown fix (Run 4c) | 8×H100 | 5,188 | 115.6 | 1.1474 | 1.1501 (15.98MB) |
| **SP8192 + INT8 embed GPTQ (run ~port14068)** | **8×H100** | **5,227** | **113.7** | **1.1387** | **1.1394 (15.43MB, +TTT)** |

### Submitted PRs
- **PR #1355** — 1.1526 bpb (non-record, Mamba-3 hybrid + GPTQ + Late QAT + linear warmdown)
- **PR #1107** — 1.5633 bpb (non-record, earlier QAT-only submission)
- **Upcoming** — 1.1394 bpb (SP8192 + INT8 embed GPTQ + chunk TTT)

### What's implemented in train_mamba3_hybrid.py (cleaned 2026-04-14)
- Train-data GPTQ with embed Hessian (`USE_GPTQ=1`, `QUANT_BITS_EMBED=8`)
- **Chunk score-first TTT on by default** (`TTT_ENABLED=1`, lr=0.010, SGD, batch=32)
- Late QAT with linear warmdown (`LATE_QAT_THRESHOLD=0.15`, `WARMDOWN_SHAPE=linear`)
- MuonEq-R optimizer (`MUON_EQ_R=1`)
- Stateful-overlap eval (`EVAL_OVERLAP=1024`) — 32s eval, unlocks TTT budget
- LZMA compression (`USE_LZMA=1`)
- Temperature scaling at eval (`EVAL_TEMP=0.9`)
- Sliding window eval (`EVAL_STRIDE`)
- **GPTQ + TTT run solo on rank 0 after DDP teardown** (fixes CUDA hook crash, avoids redundant TTT on 8 GPUs)
- Removed: `eval_val_stateful_overlap_slot`, `eval_val_stateful_overlap_ttt` (both null results)

### Key Files
| File | Description |
|------|-------------|
| `train_mamba3_hybrid.py` | Main training script — all experiments run from here |
| `setup_mamba3.sh` | Pod setup: installs mamba-ssm + patches in Mamba-3 files |
| `triton_kernels/setup_editable_mamba3.sh` | Symlinks repo copies of fwd/bwd/combined into mamba_ssm install dir for hot-reload |
| `triton_kernels/mamba3_siso_fwd.py` | Repo copy of upstream fwd kernel, extended autotune grid |
| `triton_kernels/mamba3_siso_bwd.py` | Repo copy of upstream bwd kernels + env-gated `_dqkv_fused` variant |
| `triton_kernels/mamba3_siso_combined.py` | Autograd wrapper, dispatches stock or fused bwd via `MAMBA3_FUSED_BWD` |
| `triton_kernels/bench_mamba3_bwd.py` | Correctness (`--check`, rel_l2) + perf + optional `--profile-out` chrome trace |
| `triton_kernels/extract_kernel_stats.py` | Parses chrome trace JSON, prints per-kernel regs/thread, SMEM, grid, block — avoids needing Perfetto |
| `triton_kernels/sync_and_bench.sh` | Scp + ssh to pod and run bench; forwards `MAMBA3_FUSED_BWD` env var |
| `profiling/profile_step.py` | 1-GPU compiled model profiling with chrome trace |
| `profiling/profile_ddp.py` | 8-GPU DDP benchmark: no-compile vs compiled |
| `train_mamba3_pure.py` | Pure Mamba-3 baseline (1.8060 bpb, 1×H100, archived) |
| `docs/mamba3_hybrid_context.md` | Full journey context and sweep history |
| `docs/ssm_throughput_investigation.md` | Why Mamba-1 sequential scan can't be fixed |
