# Parameter Golf — Session Notes (2026-03-21)

## What We Read & Researched

### Challenge Rules
- **Goal:** Train best LM fitting in 16MB artifact, training in ≤10 min on 8×H100 SXM
- **Metric:** bits-per-byte (bpb) on FineWeb val set (tokenizer-agnostic)
- **Submission:** PR to `records/track_10min_16mb/` with train log, README, submission.json
- **Statistical bar:** must beat SOTA by ≥0.005 nats with p < 0.01 (typically 3 seeds)
- **Eval budget:** separate 10-minute cap on top of training

### Current Leaderboard (as of 2026-03-21)
| Rank | Score | Summary |
|------|-------|---------|
| 1 | **1.1428** | 10L Int5-MLP + BigramHash(10240) + SWA(0.4) |
| 2 | 1.1458 | Int6 MLP3x + SmearGate + BigramHash + OrthoInit + Muon WD + SWA |
| ... | 1.1928 | LoRA TTT (on naive baseline only) |

### SOTA Architecture (`2026-03-20_10L_Int5MLP_MuonWD04_SWA50`)
- 10 layers, dim=512, 8 heads, 4 KV heads (GQA)
- MLP 3× expansion (hidden=1536), relu² activation
- SmearGate + BigramHash(10240, dim=128) + U-Net skip connections
- **Mixed quantization:** Int5 MLP + Int6 attn + FP16 embeddings
- **SWA:** start_frac=0.4, every=50 steps (collect most-converged checkpoints)
- Muon optimizer WD=0.04, matrix_lr=0.02, seq_len=2048
- zstd-22 compression → ~15.97MB artifact
- Pre-quant bpb ~1.16, post-quant bpb 1.1428

### Key Insight from Leaderboard Progression
- **Sliding window eval** gave the biggest single jump (-0.034 bpb)
- **LoRA TTT** was applied to the *naive baseline* only — never to the SOTA
- All "free" wins (quantization, SWA, BigramHash) are largely harvested at the top

### Idea Sources
- User's `aram-claude-skills` repo (tacit-skills-for-ai): skills based on McGilchrist, Gadamer, Polanyi
- Notion Zettelkasten (primitive AI conversation, basis for the ideas above)
- Idea document shared by user (dual-stream hemispheres, depth recurrence, gestalt token, TTT)

---

## Ideas & Architecture Directions

### Experiment 1 — LoRA TTT on SOTA ✅ IMPLEMENTED
**Why:** LoRA TTT was applied to the 1.1928 baseline, not the 1.1428 SOTA. The same eval trick on a stronger base model should yield ≥ the same gain (+0.002–0.006 bpb).
**Risk:** Low — pure eval change, training unchanged, same artifact.

### Experiment 2 — McGilchrist Register Token 🔜 NEXT
**Inspiration:** Iain McGilchrist's hemispheric asymmetry — right hemisphere holds holistic/global awareness, left decomposes locally. Current transformers are pure left-hemisphere machines.
**Idea:** Add 1–2 "register tokens" per block that attend globally (full sequence) and FiLM-condition the per-token stream — forcing holistic synthesis at each layer.
**Cost:** ~20K extra params (negligible), quantizes cheaply.

### Experiment 3 — Hermeneutic Depth Recurrence 🔜 LATER
**Inspiration:** Gadamer's hermeneutic circle — understanding parts requires whole, whole requires parts.
**Idea:** 3–4 base layers cycled 3× with tiny per-cycle FiLM modulation. Effective depth 12 from 4 layers of weights.

### Experiment 4 — Trigram Hash Embedding 🔜 QUICK WIN
**Idea:** Extend BigramHash to include a TrigramHash (prev2, prev, curr) → 4096-bucket table. Additive to existing bigram. ~524K extra params.

---

## What Was Implemented

### New File: `records/track_10min_16mb/2026-03-21_LoRA_TTT_SOTA/train_gpt.py`
Copied from SOTA. Additions:

1. **`Hyperparameters` additions** (lines ~92–96):
   ```python
   ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
   ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", "8"))
   ttt_lr = float(os.environ.get("TTT_LR", "0.01"))
   ttt_stride = int(os.environ.get("TTT_STRIDE", "256"))
   ```

2. **`BlockWithLoRA` class** — wraps each `Block`, adds LoRA delta to `c_q` and `c_v`. Used only during TTT eval (base model's compiled version is never touched).

3. **`GPTTTT` class** — wraps `GPT` with 10 `BlockWithLoRA` modules. Has `forward_logits()`, `forward()`, `reset_loras()`, `lora_params()`.

4. **`find_doc_boundaries(tokens, bos_id=1)`** — scans val_tokens for BOS token (id=1) to find document start positions.

5. **`eval_val_ttt_sliding()`** — the TTT eval loop:
   - Freezes `base_model` params (`requires_grad_(False)`)
   - Creates one `ttt_opt = Adam(lora_params)` outside doc loop
   - For each document: `reset_loras()` + `ttt_opt.state.clear()` (no cross-doc leakage)
   - Sliding window with `ttt_stride=256`: score new tokens → backward → Adam step
   - `all_reduce` at end for distributed correctness

6. **`main()` wiring** — after quantization roundtrip and standard sliding eval, runs TTT eval:
   ```
   final_int8_zlib_roundtrip val_bpb=...   ← standard baseline
   ttt_eval_exact val_bpb=...              ← TTT result
   ```

### New File: `records/track_10min_16mb/2026-03-21_LoRA_TTT_SOTA/README.md`
### New File: `records/track_10min_16mb/2026-03-21_LoRA_TTT_SOTA/submission.json`
### New File: `records/track_10min_16mb/2026-03-21_LoRA_TTT_SOTA/runpod_launch.sh`
### New File: `docs/plans/2026-03-21-parameter-golf-experiments.md` — full experiment plan

---

## RunPod Setup & Launch

### One-time RunPod Setup
1. Create account at https://console.runpod.io
2. Add SSH public key in Settings tab
3. Deploy pod using Parameter Golf template:
   **https://console.runpod.io/deploy?template=y5cejece4j**
   - For experiments: 1×H100 (~$3–4/hr)
   - For leaderboard runs: 8×H100 SXM (~$20/hr)
4. Enable SSH terminal access, leave other settings at defaults

### Sync Repo to RunPod
```bash
# Replace <user> and <ip> with your pod's SSH details
rsync -avz --exclude='.git' \
  ~/Documents/YerevaNN/openai-parameter-golf/parameter-golf/ \
  <user>@<pod-ip>:/workspace/parameter-golf/
```

### Run Experiment 1 (8×H100 — full leaderboard run)
```bash
ssh <user>@<pod-ip>
cd /workspace/parameter-golf/records/track_10min_16mb/2026-03-21_LoRA_TTT_SOTA
bash runpod_launch.sh
# Runs 3 seeds (42, 1337, 2024) sequentially. ~35–50 min total.
```

### Run Experiment 1 (1×H100 — quick test, single seed)
```bash
ssh <user>@<pod-ip>
# Make sure data is downloaded first:
cd /workspace/parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

cd records/track_10min_16mb/2026-03-21_LoRA_TTT_SOTA
SEED=42 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee train_1gpu_seed42.log
```

### Ablation Env Vars
```bash
TTT_ENABLED=0      # disable TTT → should reproduce SOTA 1.1428
TTT_STRIDE=128     # more frequent updates (slower, potentially better)
TTT_LORA_RANK=4    # lighter LoRA
TTT_LR=0.02        # higher TTT learning rate
```

### What to Look for in Logs
```
# Standard eval (should match SOTA):
final_int8_zlib_roundtrip_exact val_bpb=1.14280000

# TTT eval (the new score):
ttt_eval_exact val_bpb=1.13...

# Progress during TTT:
  ttt_eval [0/6250 docs] running_bpb=1.14xxx
  ttt_eval [500/6250 docs] running_bpb=1.13xxx
  ...
```

---

## File Listing

```
parameter-golf/
├── docs/
│   └── plans/
│       └── 2026-03-21-parameter-golf-experiments.md  ← full experiment plan
├── records/
│   └── track_10min_16mb/
│       ├── 2026-03-20_10L_Int5MLP_MuonWD04_SWA50/   ← CURRENT SOTA (1.1428)
│       │   └── train_gpt.py                          ← base for all experiments
│       ├── 2026-03-21_LoRA_TTT_SOTA/                 ← EXPERIMENT 1 ✅
│       │   ├── train_gpt.py                          ← SOTA + LoRA TTT eval
│       │   ├── README.md
│       │   ├── submission.json
│       │   └── runpod_launch.sh
│       └── 2026-03-21_McGilchrist_Register/          ← EXPERIMENT 2 🔜
│           └── (pending)
└── SESSION_NOTES.md                                  ← this file
```

---

## Next Steps (after seeing Exp 1 results)

1. If TTT gives ≥ 0.003 bpb gain → run 3 seeds, file PR
2. Implement Experiment 2 (McGilchrist Register Token)
3. Consider combining TTT + Register Token (stack gains)
4. If register token architecture trains better → try full R-L-R dual stream

---

## Architecture Ideas Backlog (from McGilchrist / skills library)

| Idea | Source | Est. gain | Status |
|------|--------|-----------|--------|
| LoRA TTT on SOTA | pragmatism | 0.003–0.008 | ✅ implemented |
| Register token (peripheral attention) | McGilchrist R-hemisphere | 0.005–0.015 | 🔜 next |
| Hermeneutic depth recurrence | Gadamer | 0.010–0.020 | 🔜 after |
| Trigram hash embedding | incremental | 0.001–0.003 | 🔜 quick win |
| R-L-R dual stream | full McGilchrist | 0.010–0.030 | 🔜 ambitious |
| Apophatic eval auxiliary | via negativa | 0.002–0.006 | 🔜 eval-only |
