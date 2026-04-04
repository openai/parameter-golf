# Parameter Golf Autoresearch

This is an autonomous research agent for the OpenAI Parameter Golf challenge.

## Security Constraints

**CRITICAL: You MUST obey these at all times.**

1. **File access boundary:** NEVER read, write, or execute ANY file outside of `/media/Datacenter_storage/winston_001a/openai/`. This includes home directories, system files, `/tmp` (except for torch cache), and any other path. Before ANY file operation, verify the path starts with `/media/Datacenter_storage/winston_001a/openai/`.
2. **No network access** except for pre-cached HuggingFace models (GPT-2 is already downloaded).
3. **No destructive operations** on files outside the `autoresearch/` directory. You may read (but not write) files in the parent `parameter-golf/` directory.

## Setup

**CRITICAL ENVIRONMENT RULES:**
- **Data is already downloaded and ready.** Do NOT run `prepare.py --setup`.
- **Do NOT use `uv`.** It is not available and not needed.
- **Python venv is at `../.venv/`.** Always activate with `source ../.venv/bin/activate` before running Python.
- **To run training:** `source ../.venv/bin/activate && bash run.sh` or directly modify run.sh (which already activates the venv).
- **To run Python scripts:** `source ../.venv/bin/activate && python train.py`
- **Data location:** `../data/datasets/fineweb10B_sp1024/` (10 training shards + validation)
- **Tokenizer:** `../data/tokenizers/fineweb_1024_bpe.model`

To set up a new experiment session:

1. **Verify data exists**: `ls ../data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l` (should show 10)
2. **Check GPU availability**: Run `nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader`
3. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`).
4. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
4. **Read the in-scope files**:
   - `program.md` — this file, your instructions.
   - `prepare.py` — fixed constants, data loading, tokenizer, evaluation. **Do NOT modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
5. **Read the research context files** (read-only):
   - `../experiments/adapter_options_analysis.md` — analysis of embedding adapter approaches
   - `../experiments/pretrained_embedding_init_diagram.md` — how pretrained init works
   - `../gpt2_pca_embeddings_512.pt` — pre-extracted GPT-2 embeddings (PCA'd to 512-dim, [1024, 512])
6. **Initialize results.tsv**: Create `results.tsv` with just the header row.
7. **Initialize progress.md**: Create `progress.md` with a header and status section.
8. **Confirm and go**: Start experimenting.

## Challenge Context

**Goal:** Train the best language model fitting in a **16MB artifact** (code + compressed model ≤ 16,000,000 bytes), evaluated by **val_bpb** (bits per byte) on FineWeb validation. Lower is better.

**Current SOTA (2026-03-23):** 1.1194 BPB — LeakyReLU(0.5)² + Legal TTT + Parallel Muon
**Our best:** 1.3143 BPB — 11L, 512dim, SiLU, int6, Muon LR=0.10 mom=0.90
**Naive baseline:** 1.2244 BPB (9L, 512dim, 1024vocab, 8 heads, 4 KV heads)

**SOTA stack (shared by top 5):**
- 11L, 512dim, 8H, 4KVH (GQA), U-Net skip connections
- **MLP 3x** (1536 hidden) with **LeakyReLU(0.5)²** or relu²
- **Tied embeddings**, logit softcap=30
- **XSA (Exclusive Self Attention)** on last 4 layers
- **SmearGate** + **BigramHash(2048, dim=128)**
- **Partial RoPE** (only 16/64 dims rotary)
- **EMA(0.997) + Tight SWA** (not just SWA)
- **Muon lr=0.025, momentum=0.99** + AdamW for embeddings
- **Int6 per-row quant** (MLP+attn) + int8 (rest) + **zstd-22** compression
- **GPTQ-lite**: per-row optimal clip percentile for int6
- **OrthoInit + muP scaling** for projections
- **Test-Time Training (TTT)**: SGD finetune on val chunks before scoring (~-0.0025 BPB)

**Hardware:** 4× RTX A6000 (48GB each). `torch.compile` is BROKEN — never enable it.

**Time budget:** Each experiment runs for **8 minutes** (480 seconds wall clock). Set in `prepare.py`.

## GPU Resource Management

**Before EVERY experiment run**, you MUST:

1. Run `nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader`
2. Determine which GPUs are available (free memory > 20GB AND utilization < 50%)
3. Update `run.sh` with the correct `CUDA_VISIBLE_DEVICES` and `nproc_per_node`
4. Log GPU state in `progress.md`

**Dynamic GPU rules:**
- If 4 GPUs are free → use all 4: `CUDA_VISIBLE_DEVICES=0,1,2,3 nproc_per_node=4`
- If 2 GPUs are free → use 2: `CUDA_VISIBLE_DEVICES=X,Y nproc_per_node=2`
- If 1 GPU is free → use 1: `CUDA_VISIBLE_DEVICES=X nproc_per_node=1`
- If 0 GPUs are free → wait 5 minutes, check again. After 3 retries, report in progress.md and stop.
- A teacher training job may be running on some GPUs. It will eventually finish and free those GPUs. Adapt dynamically.

**IMPORTANT:** When changing GPU count, adjust `TRAIN_BATCH_TOKENS` proportionally to keep tokens-per-step constant, OR accept that fewer GPUs = fewer steps in the time budget.

## Progress Monitoring

**You MUST maintain `progress.md`** — the human monitors this file to track your work.

Update `progress.md` at these events:
- **Before each experiment:** Log experiment number, description, GPU allocation, start time
- **After each experiment:** Log val_bpb result, status (keep/discard/crash), key observations
- **On any issue:** Log errors, GPU contention, unexpected results

Format for `progress.md`:
```markdown
# Autoresearch Progress

## Session Info
- **Started:** YYYY-MM-DD HH:MM
- **Branch:** autoresearch/<tag>
- **Best val_bpb so far:** X.XXXX
- **Total experiments:** N
- **Status:** running / paused / stopped

## Experiment Log

### Experiment N: <short description>
- **Time:** HH:MM - HH:MM
- **GPUs:** 0,1,2,3 (4× A6000)
- **Change:** what was modified
- **Result:** val_bpb=X.XXXX, model_size=X.XMB
- **Decision:** keep / discard / crash
- **Notes:** observations, insights

### Experiment N-1: ...
(newest on top)
```

## Research Directions — Close the Gap to SOTA

**Our gap: 1.3143 → 1.1194 = 0.195 BPB.** Most of this gap is from missing proven SOTA techniques, not novel research.

**PRIORITY ORDER:**
1. Adopt SOTA architecture (biggest impact)
2. Adopt SOTA optimizer settings
3. Adopt SOTA compression (GPTQ-lite, zstd)
4. EMA + TTT (final polish)
5. Novel compression (if architecture gains plateau)

#### PHASE 1: Adopt SOTA Architecture (CRITICAL — do first)

**1.1 MLP 3x + Tied embeddings** (REQUIRED — all top 5 use this)
- Set `TIE_EMBEDDINGS = True, MLP_MULT = 3`
- Tied saves ~0.5MB for lm_head, MLP 3x uses it for more capacity
- Previous attempt: 17.1-18.9MB (over limit). Need better compression to fit.
- Try with int6 + zstd-22 compression instead of zlib-9.
- Expected impact: 0.020–0.040 BPB

**1.2 LeakyReLU(0.5)²** (SOTA #1 uses this, worth ~0.003 BPB)
- Implement: `F.leaky_relu(x, 0.5).square()` instead of relu² or silu
- Preserves negative gradient flow while keeping non-negative outputs
- Need to add to train.py: `ACTIVATION = "leaky_relu_sq"`
- Expected impact: 0.003–0.010 BPB

**1.3 XSA (Exclusive Self Attention)** on last 4 layers
- All top entries use XSA. Need to implement.
- XSA = attention where each token only attends to exclusive (non-overlapping) local windows
- Apply on last 4 of 11 layers only
- Expected impact: 0.005–0.015 BPB

**1.4 Partial RoPE** (only 16/64 dims rotary)
- Modify Rotary class: only apply to first 16 of 64 head dims
- Remaining 48 dims are position-invariant (can learn position-independent features)
- SOTA #3 uses this. Simple change.
- Expected impact: 0.002–0.005 BPB

**1.5 BigramHash(2048)** — retune with SOTA config
- We tried 4096 buckets and it was worse. SOTA uses 1536-2048 buckets.
- Set `BIGRAM_VOCAB_SIZE = 2048, BIGRAM_DIM = 128`
- Expected impact: 0.002–0.005 BPB

**1.6 SmearGate** (all top 5 use it)
- `SMEAR_GATE_ENABLED = True`
- Was neutral alone, but synergizes with MLP 3x + tied embeddings
- Expected impact: 0.000–0.003 BPB

**1.7 Logit softcap=30** (SOTA uses 30, we use 8)
- We optimized to 8 but that was without MLP 3x. SOTA stack uses 30.
- May interact differently with larger MLP.
- Test: `LOGIT_SOFTCAP = 30`

#### PHASE 2: Adopt SOTA Optimizer Settings

**2.1 Muon lr=0.025, momentum=0.99** (HUGE difference from our lr=0.10, mom=0.90)
- SOTA uses much lower LR and higher momentum
- This likely interacts strongly with MLP 3x architecture
- Must test WITH the full SOTA architecture, not in isolation
- Expected impact: 0.010–0.030 BPB

**2.2 EMA(0.997) + Tight SWA**
- Maintain exponential moving average during training (decay=0.997)
- At end: SWA over final few EMA checkpoints
- Different from our previous SWA attempts (which didn't use EMA)
- Expected impact: 0.003–0.010 BPB

**2.3 OrthoInit + muP scaling**
- We already have OrthoInit. Add muP: scale output projections by 1/width
- Expected impact: 0.001–0.005 BPB

#### PHASE 3: SOTA Compression (fit 11L MLP3x in 16MB)

**3.1 zstd-22 compression** (instead of zlib-9)
- Replace `zlib.compress(buf, level=9)` with zstd level 22
- Typically 5-15% better compression ratio
- This alone might make 11L MLP3x fit in 16MB
- Expected size savings: 0.5-1.5MB

**3.2 GPTQ-lite** (per-row optimal clip percentile)
- Instead of fixed quantile, search 5 candidates for min MSE per row
- Zero training cost, improves int6 quality
- Expected impact: 0.000–0.002 BPB

**3.3 Non-uniform quantization** (important layers int6, others int4/5)
- `COMPRESS_METHOD = "nonuniform"`, sweep bit allocations
- Expected size savings: 1-3MB

#### PHASE 4: Test-Time Training (TTT) — worth ~0.0025 BPB

- SOTA #1 uses this for an extra 0.0025 BPB.
- Split val into non-overlapping 32K-token chunks.
- For each chunk: score first (inference mode), then train with SGD (lr=0.002, mom=0.9, 3 epochs).
- Implement AFTER architecture is solid.

#### PHASE 5: Novel Compression (lower priority)

Only pursue if SOTA architecture gains plateau.
- `COMPRESS_METHOD = "fft_int4"`: FFT per row → int4 coefficients → iFFT ≈ int8 (30% smaller than int6, potentially better quality)
- `COMPRESS_METHOD = "factored_int4"`: W ≈ B@A at int4 (WARNING: rank=128 roundtrip showed +1.46 BPB degradation — too lossy)
- LoRA training: extra capacity during training, merged before saving

#### Experimental findings from our testing:
- **factored_int4 rank=128**: 63% size reduction (5.7MB vs 15.3MB) but TERRIBLE reconstruction: +1.46 BPB degradation. Int4 SVD factors too lossy.
- **FFT int4**: 30% smaller than int6 in offline test (10.7MB vs 15.1MB). Reconstruction quality untested.
- **Non-uniform int6/int4**: modest savings (12-14MB vs 15MB).
- **SiLU >> relu²** on our architecture (fewer steps due to slower relu²).
- **SmearGate, BigramHash(4096), SWA**: all neutral or negative on current architecture.
- **12L int6 = 16.6MB**: over limit. Need zstd or mixed quant to fit.

## Experimentation

**What you CAN do:**
- Modify `train.py` — architecture, optimizer, hyperparameters, embedding methods, everything.

**What you CANNOT do:**
- Modify `prepare.py`.
- Install new packages.
- Read/write files outside `/media/Datacenter_storage/winston_001a/openai/`.
- Exceed 16MB artifact size.

## Output Format

The training script prints:
```
---
val_bpb:          X.XXXXXX
val_loss:         X.XXXXXX
training_seconds: 480.X
peak_vram_mb:     XXXXX.X
total_tokens_M:   XXX.X
num_steps:        XXXX
num_params_M:     XX.X
model_bytes:      XXXXXXX
total_bytes:      XXXXXXX
under_16mb:       True
num_layers:       X
```

Extract: `grep "^val_bpb:\|^peak_vram_mb:\|^total_bytes:\|^under_16mb:" run.log`

## Logging Results

Log every experiment to `results.tsv` (tab-separated):

```
commit	val_bpb	memory_gb	total_mb	status	description
```

Example:
```
commit	val_bpb	memory_gb	total_mb	status	description
a1b2c3d	1.350000	7.0	9.7	keep	baseline 9L 512dim
b2c3d4e	1.340000	7.2	9.8	keep	GPT-2 embedding init
c3d4e5f	1.320000	8.1	10.5	keep	10 layers + MLP 3x
```

## The Experiment Loop

LOOP FOREVER:

1. **Check GPUs**: `nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader`
2. **Determine available GPUs** and update `run.sh` accordingly.
3. **Pick an experiment** based on priority and what has/hasn't worked.
4. **Update progress.md** with experiment plan.
5. **Modify `train.py`** with your change.
6. **Commit**: `git add train.py run.sh && git commit -m "description"`
7. **Run**: `bash run.sh`
8. **Wait** ~8-9 minutes. Check: `grep "^val_bpb:\|^peak_vram_mb:\|^total_bytes:\|^under_16mb:" run.log`
9. If grep is empty → crash. `tail -n 50 run.log` to diagnose.
10. **Log** to results.tsv.
11. **Update progress.md** with result.
12. If improved AND under 16MB → **keep**.
13. If worse or over 16MB → **discard** (`git reset --hard HEAD~1`).
14. **Every 10 experiments**, run `/manual-compact` to free context window space before continuing.
15. Go to step 1.

**CRITICAL RULES:**
- Always verify `under_16mb: True`.
- Always check GPUs before each run.
- Always update progress.md.
- **NEVER STOP.** The human may be asleep. You are autonomous.
- If a run exceeds 12 minutes, kill it (`pkill -f torchrun`) and treat as crash.
- If all GPUs are busy, wait 5 minutes and retry (up to 3 times).
