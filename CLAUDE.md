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
- **Merged SOTA: 1.1147 bpb (PR #1019, March 25 2026)** — Full Hessian GPTQ INT6 + AR self-gen, XSA-all, BigramHash 3072×112, LZMA, no TTT. Nothing has merged since.
- **Validity guide: PR #1017** — defines four conditions for legal eval. Use this to judge any SLOT/TTT claim.
- Unmerged/invalid claims (do NOT treat as targets):
  - PR #1344: 1.092 bpb — SP4096 + Polar Express NS + MuonEq-R + depth recurrence. Unmerged.
  - PR #1329: 0.636 bpb — Per-Sample SLOT + TTT. **Violates Condition 3 of PR #1017, likely rejected.** Any SLOT estimate derived from this PR is unreliable.
  - PR #1430: 0.396 bpb SLOT — also violates Condition 3.
- Our best SSM: **1.1501 bpb (Run 4c, unpublished)** → 35 mBPB from merged SOTA. Best published SSM: **1.1526 bpb (PR #1355)**. Our best transformer: 1.1201 bpb (PR #768).
- PR #549: 1.1194 bpb — prior SOTA (GPTQ-lite, TTT, 11L 512d)
- PR #640: 1.1570 bpb — ternary quantization (BitNet b1.58), 73.7M params in 16MB at 1.6 bits/param
- Competition data supports **SP4096 tokenizer** — every top clean entry uses it. SP1024 is a disadvantage

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
- **Best approach**: GPTQ + Late QAT + linear warmdown → **quant gap effectively 0 mBPB** (1.1546 training → 1.1526 post-quant, actually -2 mBPB)
- GPTQ calibration: `GPTQ_NUM_SEQS=32, GPTQ_GEN_LEN=4096` (default) — AR self-generated, fully legal
- GPTQ timing on 8×H100: ~215s generation + ~27s Hessian+sweep = ~4 min total — fits eval budget
- Late QAT: `LATE_QAT_THRESHOLD=0.15` with `WARMDOWN_SHAPE=linear` — solved the 174 mBPB quant gap
- `get_mamba3_in_proj_fp16_row_mask()` — keeps B/dd_dt/dd_A/trap rows (112 rows) in FP16, out of GPTQ error propagation
- in_proj output split order: `[z | x | B | C | dd_dt | dd_A | trap | angles]`
- INT6 (`QUANT_BITS=6`) + LZMA (`USE_LZMA=1`). Model size: 15.78MB (within 16MB)
- **FP16 in_proj rows**: Only 3 mBPB difference, not worth the 400KB cost

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
- `TTT_ENABLED=0` — disabled for GPTQ runs (GPTQ replaces the benefit, eval budget is tight)
- TTT does NOT parallelize across GPUs — all 8 GPUs run same sequential loop redundantly

## Next Experiments (priority order)

### Tier 1: SP4096 + Direct Embedding (NEXT)
1. **SP4096 + direct embed + expand=1.5 + drop BigramHash**. Expected 12-30 mBPB. See plan file for details.
2. **Weight decay sweep**: WD=0.06 or 0.09 with MATRIX_LR co-tuning. Quick config change.

### Tier 2: Valid SLOT + TTT (PR #1017 compliant) on stateful-overlap scaffold
3. **Valid causal SLOT** — optimize a per-sample `[bsz,1,dim]` residual-stream delta on already-scored context tokens only, apply to unseen tokens. Must satisfy all four conditions in PR #1017. **Expected: 15-30 mBPB** (informed guess, not measured). The prior 50-150 mBPB estimate came from PR #1329 which violates Condition 3 — invalid, do NOT use as a target. Capacity-regularization argument: SLOT fits 512 params on 4K tokens per sample (vs TTT's 26M params on the same), so it converges cleanly in the time budget where TTT cannot.
4. **Score-first TTT improvements** — cosine LR decay, freeze most blocks, gradient clipping. Notebook TTT gave ~5 mBPB; tuned version expected **10-20 mBPB**.
5. **Stateful-overlap budget**: both (3) and (4) ride on the stateful-overlap scaffold which frees ~468s of eval time vs sliding-window.

### Done (Tier 1 complete)
- ~~Warmdown fix~~: WARMDOWN_ITERS=2600 → +0.5 mBPB alone (negligible without MuonEq-R)
- ~~MuonEq-R~~: +5.4 mBPB BF16, +2.5 mBPB post-quant → **Run 4c = 1.1501 bpb**

### Key Negative Results (do not retry)
- **Depth recurrence fails for SSMs**: -69 mBPB regression (SSM state discarded on repeat)
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
- **FP16 in_proj rows**: Only 3 mBPB, costs 400KB. Always use `FP16_INPROJ_ROWS=0`
- **MIMO at small scale**: Negligible per Mamba-3 paper Table 3
- **dzdo→dqkv prologue fusion**: Correct (rel_l2=0 on all 9 grads) but +1.56 ms wallclock regression (9.59 → 11.15 ms, -16%). Root cause: +8 KB SMEM (extra z tile) at stage=2 pipelining broke the autotuner's optimal schedule. Kernel fusion at these SMEM levels must be SMEM-neutral (register-resident epilogue only, PR #1420 pattern). Left env-gated at `MAMBA3_FUSED_BWD=1`, off by default.
- **Extended Triton autotune grid for dqkv/fwd**: 36 configs (maxnreg × num_warps × num_stages) picked identical winners to the 9-config grid. Stock is already Pareto-optimal.

### Standard 8×H100 run command
```bash
FP16_INPROJ_ROWS=0 WARMDOWN_ITERS=2600 WARMDOWN_SHAPE=linear MUON_EQ_R=1 \
LATE_QAT_THRESHOLD=0.15 USE_GPTQ=1 QUANT_BITS=6 \
EVAL_STRIDE=32 USE_LZMA=1 EVAL_TEMP=0.9 \
WEIGHT_DECAY=0.04 MUON_MOMENTUM=0.99 MATRIX_LR=0.025 \
torchrun --nproc_per_node=8 train_mamba3_hybrid.py
```

**SP4096 run command (next experiment):**
```bash
VOCAB_SIZE=4096 MAMBA3_EXPAND=1.5 USE_BIGRAM_HASH=0 \
FP16_INPROJ_ROWS=0 WARMDOWN_ITERS=2600 WARMDOWN_SHAPE=linear MUON_EQ_R=1 \
LATE_QAT_THRESHOLD=0.15 USE_GPTQ=1 QUANT_BITS=6 \
EVAL_STRIDE=32 USE_LZMA=1 EVAL_TEMP=0.9 \
WEIGHT_DECAY=0.04 MUON_MOMENTUM=0.99 MATRIX_LR=0.025 \
torchrun --nproc_per_node=8 train_mamba3_hybrid.py
```

### 8192 BPE dataset
- `huggingface-cli download sproos/parameter-golf-tokenizers --include 'datasets/fineweb10B_sp8192/*' --local-dir ./data --repo-type dataset`
- Tokenizer: `records/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/fineweb_8192_bpe.model`
- NOT in the default competition HF repo

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

## Current Project State (as of 2026-04-09)

### Architecture
8-layer hybrid: 7× Mamba-3 SISO blocks + 1 attention layer at layer 4, dim=512, d_state=64, mlp_mult=3, seq_len=4096, train_batch_tokens=1M

### Best results
| Run | Hardware | Steps | ms/step | val_bpb (BF16) | Post-quant bpb |
|-----|----------|-------|---------|-----------------|----------------|
| Clean BF16 | 8×H100 | 5,189 | 115.6 | 1.2087 | 1.8617 (INT6, no QAT) |
| QAT (PR #1107) | 8×H100 | 5,193 | 115.6 | 1.2413 | 1.5633 (INT6+QAT+TTT) |
| **Best: Late QAT + linear warmdown (PR #1355)** | **8×H100** | — | **115.9** | **1.1546** | **1.1526 (-2 mBPB gap)** |
| **Best: MuonEq-R + warmdown fix (Run 4c)** | **8×H100** | **5,188** | **115.6** | **1.1474** | **1.1501 (15.98MB)** |

### Submitted PRs
- **PR #1355** — 1.1526 bpb (non-record, Mamba-3 hybrid + GPTQ + Late QAT + linear warmdown)
- **PR #1107** — 1.5633 bpb (non-record, earlier QAT-only submission)

### What's implemented in train_mamba3_hybrid.py
- Full Hessian GPTQ with AR self-gen calibration (`USE_GPTQ=1`)
- Late QAT with linear warmdown (`LATE_QAT_THRESHOLD=0.15`, `WARMDOWN_SHAPE=linear`)
- Mixed-precision in_proj: `get_mamba3_in_proj_fp16_row_mask()` protects B/dd_dt/dd_A/trap rows
- MuonEq-R optimizer (`MUON_EQ_R=1`, +5.4 mBPB BF16, +2.5 mBPB post-quant)
- Depth recurrence (`DEPTH_RECURRENCE=1`, coded but harmful for SSMs — do not use)
- Stateful + stateful-overlap eval (`STATEFUL_EVAL=1`, `STATEFUL_OVERLAP=1024`) — **preferred eval mode**; ~32s eval vs 500s sliding, frees 468s for SLOT/TTT
- Mamba-3 param env vars (`MAMBA3_EXPAND`, `MAMBA3_ROPE_FRACTION`, `MAMBA3_NGROUPS`, `MAMBA3_OUTPROJ_NORM`)
- LZMA compression (`USE_LZMA=1`) — wired in compress+decompress paths
- Temperature scaling at eval (`EVAL_TEMP`)
- Sliding window eval with configurable stride (`EVAL_STRIDE`)
- TTT (disabled by default)

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
