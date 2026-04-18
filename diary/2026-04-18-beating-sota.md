# Parameter Golf Diary — April 18, 2026: Beating SOTA

## Today's Goal

**Beat the SOTA (1.0810 bpb) by implementing techniques from other near-SOTA submissions.**

We're working directly from the SOTA code. Exp 24 already matched it (1.0867 post-quant on 2×H100). Now we improve it.

## Key Decisions Made Today

1. **Switch to 8×H100 10min** — same cost as 2×H100 40min (~$3.50/run), but gives us real competition conditions. No more extrapolating.
2. **$20/day budget cap** — yesterday burned $58, mostly idle pods.
3. **Checkpoint at event boundaries** — save model+optimizer state at each training stage transition, so we can independently tune later stages.
4. **Focus on architecture, not hyperparameter sweeps** — user wants structural improvements.

## The 7-Stage Pipeline

Training has 4 discrete stages (each triggered by a hyperparameter), plus 3 post-training stages:

| Stage | Event | Trigger | SOTA Step | Checkpoint |
|-------|-------|---------|-----------|------------|
| 1. Momentum warmup | 0.92→0.99 | `muon_momentum_warmup_steps=1500` | 0→1500 | ckpt at 1500 |
| 2. Stable training | Full LR, plain arch | — | 1500→5600 | ckpt at 5600 |
| 3. Warmdown | LR starts decaying | `warmdown_iters=14400` | 5600→7000 | ckpt at 7000 |
| 4. Recurrence | 3-layer loop activates | `recur_start_step=7000` | 7000→end | ckpt at end |
| 5. EMA | Smooth weights | `ema_decay=0.9965` | post-train | ckpt pre/post EMA |
| 6. Quantization | INT6 GPTQ + Brotli | `clip_sigmas=12.85` | post-train | testable on ckpt |
| 7. Eval | Sliding window + TTT | `ttt_lr=0.005` | post-train | testable on ckpt |

## Techniques to Implement (from other near-SOTA submissions)

### Training-time (need retraining)

| Technique | Source | Impact | What it does |
|-----------|--------|--------|-------------|
| Progressive recurrence | Apr 6 (1.0835) | reduces loss spike | Fractional activation instead of hard on/off |
| DISABLE_LAYER0_ATTN | Mar 31 (1.1063) | skip useless attn | First-layer attention is mostly noise |
| REPEAT_UNTIE_MLP | Mar 31 (1.1063) | more params | Untied MLP weights in recurring layers |
| BigramHash 3072×112 | Mar 25 | proven +0.005 | Cheap n-gram feature embeddings |
| YaRN RoPE | Mar 24 | better pos encoding | Scaled RoPE for longer contexts |
| Layer-wise LR decay | community | unknown | Different LR per layer depth |
| Differential attention | Microsoft | unknown | Learned attention subtraction |
| SWA + EMA combo | Mar 25 | better averaging | SWA every 50 steps on top of EMA |

### Post-training (testable on checkpoint)

| Technique | Source | Impact | What it does |
|-----------|--------|--------|-------------|
| Hessian-aware SDClip | Apr 6 (1.0835) | better quant | Use Hessian to weight clip importance |
| Per-group clip allocation | Apr 6 | non-uniform quant | Important layers get more bits |
| AR Self-Gen GPTQ | Mar 25 | better calibration | Model generates own quant calibration data |

## Today's Plan

### Step 1: First 8×H100 baseline run with checkpoints (~$5)
- [x] Add checkpoint saving to train_gpt.py (done)
- [x] Create run_8xh100_10m.sh (done)
- [ ] Create 8×H100 SXM pod on Runpod
- [ ] Upload code + verify SP8192 data
- [ ] Run 10min, save checkpoints
- [ ] SCP logs + checkpoints back to VPS
- [ ] Stop pod immediately
- **Expected result:** ~1.08 bpb + 5 checkpoints (steps 1500, 5600, 7000, final pre-EMA, final post-EMA)

### Step 2: Implement easy zero-risk training changes
Stack these in one run:
- [ ] Progressive recurrence (fractional activation)
- [ ] DISABLE_LAYER0_ATTN
- [ ] SWA + EMA combo

### Step 3: Run with all easy changes (~$4)
- [ ] One 8×H100 10min run with all changes from Step 2
- [ ] Compare against baseline at each checkpoint
- **Expected result:** ~0.003-0.008 improvement over SOTA

### Step 4: BigramHash (bigger implementation)
- [ ] Implement BigramHash embedding layer
- [ ] One 8×H100 10min run
- **Expected result:** +0.005 (proven in another submission)

## Code Changes Made

1. **train_gpt_sota.py** — Decoded SOTA submission (1.0810 bpb) from LZMA blob into readable 1400-line Python
   - Added checkpoint saving at event boundaries (momentum warmup, warmdown, recurrence, pre/post EMA)
   - Added temporal checkpoints via `CKPT_STEPS=1000,2000,...` env var
   - Added BigramHash embedding (3072×112, disabled by default with `BIGRAM_VOCAB_SIZE=0`)
   - Enable via `CKPT_DIR=/path`

2. **hotstart.py** — Utility for loading checkpoints and resuming/re-quantizing/re-evaluating
   - `resume` — load checkpoint, resume training from that step
   - `requant` — re-quantize with different clip_sigmas, calibration params
   - `reeval` — re-eval quantized model with different TTT params
   - `reema` — re-blend EMA weights at different ratios

3. **run_8xh100_10m.sh** — Launch script for 8×H100 competition conditions
   - Auto-installs brotli if missing
   - Saves to `/workspace/runs/{date}-{name}/` on network volume
   - Auto-logs + auto-stops pod after 30s grace

4. **train_gpt_baseline.py** — Backup of original baseline code

## Infrastructure

- **Pod template:** `runpod/parameter-golf:latest` (template ID: y5cejece4j)
  - Has: flash_attn3, sentencepiece, PyTorch 2.8
  - Missing: brotli (auto-installed by run script)
  - Requires: `PUBLIC_KEY` env var for SSH access
- **Network volume:** `hvpdph5i3g` (US-NE-1, 150GB)
  - Has SP8192 + SP1024 data, parameter-golf repo
- **US-NE-1** has 1×/2×/4×/8× H100 SXM available
- **AP-JP-1** is backup (9/9 availability checks, supports volumes)
- Pod ID for setup: `39623ko4jkxhz4` (stopped)

## Test Results (1×H100)

### Checkpoint saving ✅
Ran 50 steps with `ITERATIONS=50, MUON_MOMENTUM_WARMUP_STEPS=10`. All 8 checkpoints saved:
- `ckpt_event_step5.pt` (temporal)
- `ckpt_event_step10.pt` (momentum warmup end)
- `ckpt_warmdown_start_step15.pt` (warmdown event)
- `ckpt_pre_recurrence_step18.pt` (recurrence event)
- `ckpt_event_step25.pt` (temporal)
- `ckpt_event_step45.pt` (temporal)
- `ckpt_final_pre_ema_step50.pt`
- `ckpt_final_post_ema_step50.pt`
Each ~404 MB (model + optimizer + EMA state). Plus `final_model.pt` (130MB) and `final_model.int6.ptz` (16MB).

### Resume (hotstart) ✅
Loaded step 25 checkpoint → resumed training to step 50. Loss decreasing (5.44→5.21). Recurrence correctly activated (frac=0.50 > 0.35).

### Requant / Reeval / Reema — not tested
Too slow on 1×H100 (GPTQ Hessian collection takes minutes). Will test on 8×H100.

## Next Steps

- [ ] 8×H100 baseline run (SOTA reproduction) with checkpoints
- [ ] 8×H100 BigramHash run (SOTA + BigramHash)
- [ ] Test requant/reeval/reema on 8×H100 checkpoints
- [ ] If BigramHash wins → prepare submission
