# Experiment Log — A100 Session (2026-03-20)

## Environment
- **Hardware**: TACC Lonestar6, 1x NVIDIA A100-PCIE-40GB (via SLURM job 3019533 on c301-004)
- **Python**: 3.12.11 with torch 2.9.1+cu128
- **Run command**: `ssh c301-004 "cd ~/parameter-golf && LD_LIBRARY_PATH=/opt/apps/python/3.12.11/lib:$LD_LIBRARY_PATH /opt/apps/python/3.12.11/bin/python3 train_exp.py"`
- **Data**: 4 train shards (0,2,3,4 — 800M tokens total), 1 val shard (~62M tokens). Shard 1 failed (disk quota).
- **Branch**: `submission/depth-recurrence-layer-sharing`
- **GPU node**: c301-004 (SLURM interactive job, gpu-a100-dev partition)

## Script: `train_exp.py`

Based on the **WarmdownQuantization record** (int6 quant, FP16 tied embed, sliding window eval, aggressive warmdown). Built by copying that record's `train_gpt.py` and making targeted edits to add:

- **Layer sharing** (`NUM_UNIQUE_LAYERS`): Cycle N unique blocks over M virtual layers. Block forward accepts optional ext_attn_scale/ext_mlp_scale/ext_resid_mix.
- **Per-layer scales** (`PER_LAYER_SCALES`): Each virtual depth gets its own attn/mlp/resid modulation tensors. Tiny param cost (~27K for 9 layers at 512d).
- **BigramHash** (`BIGRAM_HASH`): Hash-based bigram features added to token embeddings before RMSNorm. Uses `(prev * vocab_size + curr) % table_size` hash. Separate embed (hash_dim) -> project (model_dim) architecture. **NOTE**: PR #162's SOTA uses XOR hash instead: `XOR(36313*t, 27191*t_prev) % (table_size-1)` with 128-dim embed, learned scale=0.05, zero-init. Our impl differs — should update.
- **SmearGate** (`SMEAR_GATE`): Per-dim sigmoid gate blending x with x_prev. Our impl matches PR #162 almost exactly. Applied once per block (PR #162 applies once after embed). **NOTE**: PR #162 places SmearGate AFTER RMSNorm, before blocks — not per-block.
- **SWA** (`USE_SWA`): Stochastic weight averaging — accumulates running mean of model params during warmdown. Loads SWA weights before serialization.
- **Muon weight decay** (`MUON_WEIGHT_DECAY`): Decoupled `p.mul_(1 - lr*wd)` in Muon optimizer step.
- **OrthoInit** (`ORTHO_INIT`): Orthogonal init for linear layers. Casts to float32 first (bfloat16 QR decomposition not supported).
- **zstd compression** (`USE_ZSTD`): Uses `zstandard` library at configurable level. Decompression also handled.
- **Configurable quant bits** (`QUANT_BITS`): Passed through to `quantize_state_dict_int8(quant_bits=...)`.

### Known Bugs / Differences from Competition SOTA
1. **BigramHash hash function** — ours uses simple modular arithmetic; SOTA uses XOR with coprime multipliers. Theirs distributes collisions better.
2. **BigramHash hash_dim** — ours defaults to 32; SOTA uses 128. Ours is underpowered.
3. **BigramHash learned scale** — SOTA has `self.scale = nn.Parameter(torch.tensor(0.05))` to gate the contribution. We don't have this.
4. **BigramHash init** — SOTA zero-inits both embed and proj. We normal-init embed (std=0.02) and zero-init proj.
5. **SmearGate placement** — ours is per-block (inside `smear_gates` ModuleList). SOTA applies it once, after embed+RMSNorm, before all blocks. Per-block is more expensive and may interact poorly with layer sharing.
6. **Sliding window eval did NOT trigger** in Experiment 7 — the process completed without printing sliding window results. Likely the eval_val_sliding function wasn't called, or the SSH pipe (`| tail -30`) lost output. Need to investigate.

---

## Results Summary

| # | Config | Params | Artifact | Steps | ms/step | BPB (post-quant) | Notes |
|---|--------|--------|----------|-------|---------|-------------------|-------|
| 0 | Smoke test (3 shared, 9 virt, MLP 3x, 100 steps) | 7.6M | 3.6MB | 100 | 135ms | 2.1806 | Validates script works |
| 1 | **9 unique, 512d, MLP 3x (baseline)** | 21.8M | 13.5MB | 1494 | 160ms | **1.4417** | Clean reference, 4-min run |
| 2 | 3 shared x 9 virtual, 512d, MLP 3x | 7.6M | 5.4MB | 1500 | 136ms | 1.5320 | Layer sharing: 0.09 BPB worse |
| 3 | 3 shared x 12 virtual (GPU contention)* | 7.6M | 4.1MB | 540 | 445ms | 1.6521 | Unreliable — ran concurrently with #4 |
| 4 | 9 unique + BigramHash (GPU contention)* | 21.9M | 11.6MB | 731 | 328ms | 1.5432 | Unreliable — ran concurrently with #3 |
| 5 | **9 unique + BigramHash + SmearGate** | 21.9M | 13.3MB | 1422 | 169ms | **1.4384** | Best short run. +0.003 BPB over baseline |
| 6 | 10L + BigramHash + SmearGate + SWA + OrthoInit | ~24M | 14.3MB | 1257 | 191ms | 1.4493 | Too many features, too few steps |
| 7 | 10L + full 10-min + WARMDOWN_ITERS=20000 | ~24M | 8.7MB | 3169 | 189ms | **1.5381** | Huge quant penalty (0.16 BPB). See analysis below. |

*Experiments 3-4 ran concurrently on same GPU — results unreliable due to GPU contention.

---

## Session 2: Bug Fixes + A100 x3 Parallel Experiments (2026-03-20)

**Hardware**: c301-001, 3x A100-PCIE-40GB (SLURM job 3020340).
**Key fixes applied**: BigramHash XOR hash (128-dim, zero-init, learned scale 0.05), SmearGate single-gate placement after embed+RMSNorm.

| # | Config | Params | Artifact | Steps | ms/step | BPB (post-quant) | BPB (sliding window) | Notes |
|---|--------|--------|----------|-------|---------|-------------------|---------------------|-------|
| A | 9L + fixed BigramHash + SmearGate (zlib) | 22.4M | 12.8MB | 3595 | 167ms | 1.3442 | — | BigramHash/SmearGate fix = +0.094 |
| B | 10L + SWA(bug) + zstd | 24.7M | 12.3MB | 3239 | 185ms | 1.6226 | — | SWA started at step 2, destroyed model |
| C | 9L + higher LR + SWA + zstd | 22.4M | 12.3MB | 3578 | 166ms | 1.4033 | — | Higher LR hurt |
| D | 10L clean + zstd | 24.7M | 13.3MB | 3251 | 185ms | 1.3456 | — | Clean 10L baseline |
| E | 10L + SWA(0.5) + zstd | 24.7M | 13.2MB | 3251 | 185ms | 1.3566 | — | SWA hurt by +0.011 |
| F | 9L + SWA(0.5) + zstd | 22.4M | 12.0MB | 3645 | 165ms | 1.3491 | — | SWA hurt by +0.005 |
| G | 9L + zstd (no SWA) | 22.4M | 12.5MB | 3617 | 166ms | 1.3431 | **1.3260** | **Best legal result!** |
| H | 10L + FTLE-lite + eval recurrence | 24.7M | 18.1MB | 3265 | 184ms | 1.3418 | — | Over 16MB limit. Eval recurrence=3.47 BPB (useless) |
| I | 10L + long warmdown (6500) | 24.7M | 11.7MB | 3272 | 183ms | 1.3517 | 1.3358 | Better compression, matches 8xH100 LR schedule |

**Key findings from Session 2:**
1. BigramHash/SmearGate fixes gave **0.094 BPB improvement** (1.3442 vs 1.4384)
2. **SWA consistently hurts** at SWA_START_FRAC=0.5 — adds 0.005-0.011 BPB penalty
3. 10L slightly better than 9L pre-quant but fewer steps on A100 (184 vs 167 ms/step)
4. **Higher LR (0.10/0.06) hurts** vs default (0.05/0.04)
5. zstd-22 gives 5-8% smaller artifacts than zlib-9 with no quality impact
6. TRAIN_BATCH_TOKENS must be 65536 for 1xA100 (524K default gives 880ms/step!)
7. **Sliding window eval (stride=1024, seq_len=2048) gives 0.017 BPB boost** (477s eval time, within 10-min budget)
8. **Eval-time extra recurrence is useless** on non-shared models — repeating decoder blocks gives 3.47 BPB (random noise)
9. **FTLE-lite mixed precision** improves BPB but increases artifact size (18MB > 16MB). Bug: only detects gradient for BigramHash embed, not block weights
10. **WARMDOWN_ITERS=6500** matches 8xH100 LR schedule better (LR starts at ~50% of peak). Gives best compression (11.7MB) but worse quant penalty
11. **Best competition-legal result: 1.3260 BPB** (9L, zstd-22, sliding window stride=1024)

### Round 3 Results (Session 2 continued)

| # | Config | Steps | Post-quant BPB | Sliding BPB | Artifact | Notes |
|---|--------|-------|----------------|-------------|----------|-------|
| G | 9L + zstd (no SWA) | 3617 | 1.3431 | **1.3260** | 12.5MB | **Best legal w/ sliding** |
| H | 10L + FTLE-lite | 3265 | 1.3418 | — | 18.1MB | Over 16MB! Eval recurrence = 3.47 (useless) |
| I | 10L + long WD (6500) | 3272 | 1.3517 | 1.3358 | 11.7MB | Better compression, matches H100 LR schedule |

---

## Competition Intelligence (gathered 2026-03-20)

### Current Leaderboard State
- **Best merged:** 1.1428 BPB (PR #180: 10L Int5-MLP + BigramHash(10240) + SWA + WD=0.04)
- **Best clean open:** 1.1318 BPB (PR #198: 11L Int6 + WD=0.04 + SWA + stride=64)
- **Paid prefix exploit:** 1.02-1.05 BPB (stores val tokens in artifact — controversial)

### Key Techniques We're Missing
1. **WD=0.04** (we use 0.02) — competition found 0.04 is better for both quality and artifact size
2. **11 layers** — PR #198 uses 11L (vs our 9-10L)
3. **SWA every 50 steps during warmdown** (not continuous averaging — explains why our SWA hurt!)
4. **BigramHash table 10240** (PR #180) vs our 4096
5. **RoPE base 50000** (not default 10000)
6. **Stride-64 sliding window** with batched processing (32 windows) — ~172s on 8xH100
7. **Low-Rank Q factorization** (PR #215) — Q has extreme condition numbers, 25% param savings
8. **muP scaling** — output projections scaled by 1/sqrt(2*num_layers)
9. **Smaller batch wins** — 524K beats 786K (more gradient updates in fixed time)
10. **Stride-OGD** (PR #241) — online gradient descent on vocab bias during eval, zero artifact cost

### Priority Fixes for Next Session
1. **MUON_WEIGHT_DECAY=0.04** (instant improvement)
2. **NUM_LAYERS=11** (go deeper)
3. **BIGRAM_TABLE_SIZE=10240** (larger hash table)
4. **ROPE_BASE=50000** (better position encoding)
5. **Fix SWA** to use periodic checkpointing (every N steps) not continuous
6. **Batch sliding window** (process 32 windows at once for stride=64)

### Low-Rank Q Factorization (added session 2)
PR #215 found Q projection matrices have condition numbers >100M — Q is naturally low-rank.
Factoring Q as `x @ W_down(512→r) @ W_up(r→512)` with r=192 saves:
- 2.6% total params (590K fewer)
- ~22% step time on H100 (smaller matmuls)
- ~28% more training steps in 10 min
Enable with `Q_RANK=192`. Default is 0 (full rank).

### Recommended 8xH100 Submission Config
```bash
torchrun --standalone --nproc_per_node=8 train_exp.py \
  NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
  MUON_WEIGHT_DECAY=0.04 QUANT_BITS=6 GRAD_CLIP_NORM=1.0 \
  BIGRAM_HASH=1 BIGRAM_TABLE_SIZE=10240 BIGRAM_HASH_DIM=128 SMEAR_GATE=1 \
  WARMDOWN_ITERS=20000 EVAL_STRIDE=64 EVAL_SEQ_LEN=2048 \
  TIED_EMBED_LR=0.05 MATRIX_LR=0.04 SCALAR_LR=0.04 \
  USE_ZSTD=1 ZSTD_LEVEL=22 USE_SWA=0 ROPE_BASE=50000 Q_RANK=192
```

---

## Detailed Analysis

### Experiment 7: Full 10-min Run (the problematic one)

**Config**: 10L, 512d, MLP 3x, BigramHash, SmearGate, WARMDOWN_ITERS=20000, EVAL_STRIDE=64, 65K batch, 10-min wallclock.

**Training trajectory**:
```
step:0     val_bpb: 4.1100  (init)
step:2000  val_bpb: 1.4364  (pre-quant, good)
step:3169  val_bpb: 1.3775  (pre-quant at wallclock cap, great!)
```

**Post-quant**: 1.5381 BPB — a **0.16 BPB quantization penalty**.

**Why so bad?** The aggressive warmdown (WARMDOWN_ITERS=20000) with wallclock-based scheduling means:
- `warmdown_ms = 20000 * 189ms = 3,780,000ms` (63 minutes)
- `remaining_ms / warmdown_ms` at step 0 = `600000 / 3780000 = 0.159`
- LR starts at only **16% of peak** and decays to near-zero

This is WAY too aggressive for a 10-min 1xA100 run. On 8xH100 the model trains for ~10,500 steps at ~57ms/step, so warmdown_ms = 20000*57 = 1.14M ms. LR at step 0 would be 600000/1140000 = 0.53 — much more reasonable.

**Fix**: For 1xA100 testing, use `WARMDOWN_ITERS=3000` (matches the actual step count). The WARMDOWN_ITERS=20000 setting is specifically tuned for 8xH100's faster step rate.

**Sliding window eval**: Did NOT run. The process log shows it stopped after the standard roundtrip eval. Possible causes: (a) SSH pipe lost output, (b) the sliding window eval crashed silently, (c) process was killed. Needs investigation.

**Artifact size**: 8.7MB — excellent compression. The aggressive warmdown DID help compression even if it hurt pre-quant quality. On 8xH100 with proper LR schedule, this would be ideal.

### Layer Sharing: Why It Lost

At 512d with MLP 3x, 9 unique blocks have 21.8M params and the artifact is 13.5MB with int6+zlib. This FITS in the 16MB limit. So the artifact savings from sharing (5.4MB vs 13.5MB) don't unlock anything meaningful — there's no need for smaller artifact.

Meanwhile, 9 unique blocks learn more diverse features than 3 shared blocks. Each block specializes for its depth position. With sharing, blocks must serve 3 different positions, limiting specialization.

The previous Apple Silicon result (sharing nearly equal to baseline at 256d) doesn't hold because:
1. At 256d, fewer params per block → less specialization opportunity → sharing cost is lower
2. At 512d, each block has enough capacity to specialize → sharing cost is higher
3. Apple Silicon was limited to ~6K steps with 100M tokens; A100 is faster per step

**Verdict**: Layer sharing is an interesting research direction but NOT competitive for this challenge at 512d. Abandon it for submission.

### BigramHash + SmearGate: Small but Real Win

Experiment 5 (BigramHash + SmearGate) vs Experiment 1 (baseline):
- **1.4384 vs 1.4417** = +0.0033 BPB improvement
- 1422 vs 1494 steps (5% fewer due to overhead)
- Even with fewer steps, BigramHash+SmearGate wins

This is only with our suboptimal implementation. With the PR #162 XOR hash, 128-dim embed, and learned scale, the improvement should be larger (~0.005 BPB per competition data).

---

## Competition Research: SmearGate & BigramHash from PRs #102, #135, #162

(From research agent's findings on the actual GitHub PRs)

### SmearGate (PR #162 implementation)
```python
class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev
```
- 512 params. Applied ONCE after embed+RMSNorm, before all transformer blocks.
- Our implementation is nearly identical but we apply per-block (more expensive, possibly worse).

### BigramHash (PR #162 implementation)
```python
class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size, bigram_dim, model_dim):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)  # 4096 x 128
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False)  # 128 x 512
        nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens):
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
```
- ~524K params (4096x128 embed + 128x512 proj + 1 scale)
- XOR hash with coprime multipliers for better distribution
- Zero-initialized everything + learned scale starting at 0.05
- Added to embeddings BEFORE RMSNorm

### Forward pass order in PR #162:
```python
x = tok_emb(input_ids) + bigram(input_ids)  # embed + bigram
x = rms_norm(x)                              # normalize
x = smear(x)                                 # SmearGate
x0 = x                                       # anchor
# ... transformer blocks ...
```

---

## What To Do Next (for a continuing agent)

### Immediate Priority: Fix `train_exp.py` to match PR #162
1. **Update BigramHash** — switch to XOR hash, 128-dim, zero-init, learned scale 0.05
2. **Fix SmearGate placement** — apply once after embed+RMSNorm, not per-block
3. **Fix WARMDOWN_ITERS** — use 3000 for A100 testing, 20000 for 8xH100 submission
4. **Debug sliding window eval** — figure out why it didn't trigger in Exp 7

### Experiments to Run
1. **Fixed BigramHash + SmearGate** vs baseline (validate improvement is larger)
2. **zstd-22 vs zlib-9** compression comparison (artifact size)
3. **SWA with late start** (SWA_START_FRAC=0.9, only average last 10% of training)
4. **10L vs 9L** at matched step count (need WARMDOWN_ITERS=3000 for fair A100 comparison)
5. **Full run with 4 shards** (800M tokens) to test data scaling

### For 8xH100 Submission
Best config (estimated):
```bash
torchrun --standalone --nproc_per_node=8 train_exp.py \
  NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
  MUON_WEIGHT_DECAY=0.02 QUANT_BITS=6 GRAD_CLIP_NORM=1.0 \
  BIGRAM_HASH=1 BIGRAM_TABLE_SIZE=4096 BIGRAM_HASH_DIM=128 SMEAR_GATE=1 \
  WARMDOWN_ITERS=20000 EVAL_STRIDE=64 EVAL_SEQ_LEN=1024 \
  TIED_EMBED_LR=0.10 MATRIX_LR=0.04 SCALAR_LR=0.04 \
  USE_ZSTD=1 ZSTD_LEVEL=22 USE_SWA=1 SWA_START_FRAC=0.9
```

### Files on this branch
- `train_exp.py` — Main experimental script (all features, ~1350 lines)
- `train_gpt_submission.py` — Previous session's CUDA script with layer sharing
- `train_gpt_mlx_exp.py` — Previous session's MLX script with all features
- `make_mini_shards.py` — Data subset creator for local testing
- `EXPERIMENTS.md` — Strategy doc from previous (Apple Silicon) session
- `NOTES.md` — Dev notes from previous session
- `EXPERIMENT_LOG.md` — THIS FILE (A100 session progress)

### Data Location
- Train shards: `data/datasets/fineweb10B_sp1024/fineweb_train_00000{0,2,3,4}.bin` (shard 1 missing — disk quota)
- Val shard: `data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin`
- Tokenizer: `data/tokenizers/fineweb_1024_bpe.model`
- Logs: `logs/` directory on GPU node (not committed)

### How to Run on This System
```bash
# SSH to GPU node (must have active SLURM job)
ssh c301-004

# Set up environment
cd ~/parameter-golf
export LD_LIBRARY_PATH=/opt/apps/python/3.12.11/lib:$LD_LIBRARY_PATH
PY=/opt/apps/python/3.12.11/bin/python3

# Quick test (2 min)
ITERATIONS=1000 VAL_LOSS_EVERY=1000 TRAIN_LOG_EVERY=200 MAX_WALLCLOCK_SECONDS=120 \
TRAIN_BATCH_TOKENS=65536 WARMUP_STEPS=5 WARMDOWN_ITERS=300 \
NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
MUON_WEIGHT_DECAY=0.02 QUANT_BITS=6 GRAD_CLIP_NORM=1.0 \
RUN_ID=quick_test $PY train_exp.py

# Full 10-min run (use WARMDOWN_ITERS=3000 for A100, 20000 for 8xH100)
ITERATIONS=20000 VAL_LOSS_EVERY=2000 TRAIN_LOG_EVERY=500 MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=65536 WARMUP_STEPS=10 WARMDOWN_ITERS=3000 \
NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
MUON_WEIGHT_DECAY=0.02 QUANT_BITS=6 GRAD_CLIP_NORM=1.0 \
BIGRAM_HASH=1 BIGRAM_TABLE_SIZE=4096 BIGRAM_HASH_DIM=32 SMEAR_GATE=1 \
EVAL_STRIDE=64 EVAL_SEQ_LEN=1024 \
RUN_ID=full_run $PY train_exp.py
```
