# Parameter Golf — Working Plan & Progress Tracker

## Environment Setup (run at start of every new instance)
```bash
bash /vol/paraG/parameter-golf/setup.sh
export PATH="$HOME/.local/bin:$PATH"
source /vol/paraG/parameter-golf/.venv/bin/activate
cd /vol/paraG/parameter-golf
```

---

## ⚡ NEXT CLAUDE — START HERE (updated Mar 28 — v3 banking done)

**Current best:** v7_ve seed 2, ttt_bpb=**1.15738** (PR: submission/12L-INT4-bQAT-VE)
**Model banking rewrite:** COMPLETE — `train_gpt_v3.py` ready, `v10_banked` config in `run.sh`

### IMMEDIATE: Run v10_banked proxy smoke test (single GPU, 300 steps)
```bash
bash run.sh v10_banked_proxy
# Expected: step_avg ~110ms (no DDP), train loss ~3.1-3.3 at step 300
# If it crashes → check logs. If slow → investigate.
```

### Then run v10_banked on 8×H100:
```bash
SEED=1 bash run.sh v10_banked
# Expected: ~90-100ms/step → ~6000-6700 steps → projected ~1.132-1.139 BPB
# Step avg at step 100 is the key signal:
#   ~90ms → on track for 6700 steps
#   ~100ms → on track for 6000 steps
#   >110ms → Parallel Muon not working, investigate
```

### What changed in train_gpt_v3.py vs v2:
1. **Model Banking**: 4 contiguous 3D bank params replace per-layer CastedLinear
   - `qo_bank[24,512,512]`, `kv_bank[24,256,512]`, `mlp_up_bank[12,1536,512]`, `mlp_down_bank[12,512,1536]`
2. **Parallel Muon**: async reduce-scatter → sharded NS5 → all-gather (no DDP)
3. **Bank QAT**: `GPT._fq(w, clip)` applies STE to bank slices during QAT
4. **Serialization**: `_unbank_state_dict()` + `_rebank_state_dict()` for quantization roundtrip

### After run completes:
```bash
mkdir -p records/track_10min_16mb/2026-03-28_12L_banked_parallel_muon/
cp train_gpt_v3.py records/track_10min_16mb/2026-03-28_12L_banked_parallel_muon/train_gpt.py
# write submission.json + README.md
git add records/track_10min_16mb/2026-03-28_12L_banked_parallel_muon/ train_gpt_v3.py run.sh
git commit -m "12L banked + Parallel Muon — ttt_bpb=X.XXXX"
# open new PR
```

---

## Git / PR Strategy (follow this every run)

**Rule: every H100 run gets committed and pushed. No exceptions.**

Even if it doesn't beat SOTA. Even if it's a failed experiment. The history matters.

### After every run:
```bash
# 1. Create submission dir
mkdir -p records/track_10min_16mb/YYYY-MM-DD_short_description/

# 2. Copy relevant files in
cp train_gpt_v2.py records/track_10min_16mb/YYYY-MM-DD_.../train_gpt.py
# write submission.json, README.md, copy log if saved

# 3. Commit everything changed (code + records dir)
git add records/track_10min_16mb/YYYY-MM-DD_.../
git add train_gpt_v2.py run.sh  # if changed since last commit
git commit -m "Short description of what changed and why"
git push fork main  # or submission/branch-name

# 4. Open PR on github
# - Record: normal submission title
# - Non-record: prefix title with "Non-record:" and explain what you learned
```

### PR types:
| Type | When | Title format |
|---|---|---|
| Record | New best ttt_bpb | "12L INT4 bQAT + VE — val_bpb 1.1550" |
| Non-record | Novel idea, failed experiment, analysis | "Non-record: VE overhead analysis + Parallel Muon findings" |

**Non-record PRs are valuable.** They show active research, novel ideas, and build your leaderboard presence. Document what you tried and what you learned.

---

## Full Experiment Log (Mar 28)

### Run 1: v4_h100 seed 1 ✅ SUBMITTED
- **Config:** 12L INT4 bQAT, EMA_ENABLED=1, LATE_QAT_FRAC=0.65, XSA4, RoPE16, LN_SCALE, TTT
- **Result:** 5021 steps, post-quant 1.1703, ttt_bpb ~1.165 (TTT interrupted at 58%)
- **Artifact:** 15,899,385 bytes (15.90MB)
- **PR:** Pushed to fork, opened PR (review required)
- **Record dir:** `records/track_10min_16mb/2026-03-28_12L_INT4_bQAT_EMAfix/`

### Run 2: v5_rownorm seed 1 ❌ KILLED (step 100)
- **Config:** v4_h100 + MUON_BACKEND=rownorm (row-wise L2 norm replaces NS5)
- **Result:** train_loss=5.38 at step 100 (vs 3.18 for v4_h100) — catastrophic failure
- **step_avg:** 109ms (same as DDP — NS5 was NOT the bottleneck)
- **Lesson:** Row-norm doesn't produce orthogonal updates. NS5 is not the throughput bottleneck — DDP all_reduce is.
- **Code:** v5_rownorm config deprecated (exits with error in run.sh)
- **TODO:** Package as non-record PR with analysis

### Run 3: v4_h100 seed 3 ✅ SUBMITTED (updated existing PR)
- **Config:** Same as seed 1
- **Result:** post-quant 1.2002 (large QAT degradation), ttt_bpb **1.1594** (full 1893-chunk TTT)
- **Artifact:** 15,967,640 bytes (15.97MB)
- **Note:** Large post-quant degradation but TTT recovered aggressively. New best.
- **PR:** Updated submission.json (val_bpb 1.165→1.1594) + README, pushed to fork

### Run 4: v6_parallel seed 1 ❌ KILLED (step ~800)
- **Config:** v4_h100 + Parallel Muon (one RS per matrix param, no DDP)
- **Result:** step_avg 148ms — WORSE than DDP 110ms
- **Lesson:** 40+ individual NCCL RS ops has higher overhead than DDP's single batched all_reduce
- **TODO:** Package as non-record PR with Parallel Muon analysis

### Run 5: v6_parallel (virtual banking) seed 1 ❌ KILLED (step ~800)
- **Config:** v4_h100 + virtual banking Muon (group params by shape, one RS per shape group)
- **Result:** step_avg 117ms — still slower than DDP 110ms
- **Lesson:** Grad stacking copy overhead cancels NCCL savings. Need model banking (3D weight tensors) to avoid copies.
- **Analysis:** SOTA uses 4 banked 3D params (qo/kv/mlp_up/mlp_down). RS operates directly on banked grad — no copy. Without model banking, can't beat DDP.
- **TODO:** Package as non-record PR with virtual banking analysis

### Run 7: v7_ve seed 2 ✅ NEW BEST
- **Config:** same as seed 1 with SEED=2
- **step_avg:** 148ms final. post-quant 1.1624, ttt_bpb **1.15738**
- **Artifact:** 16,408,223 bytes (16.41MB) — within 16,777,216 limit ✓
- **Key:** Seed 2 beat seed 1 by 0.00137 BPB
- **Submission dir:** Updated `records/track_10min_16mb/2026-03-28_12L_INT4_bQAT_VE/`
- **TODO:** Commit + push to `submission/12L-INT4-bQAT-VE`

### Run 6: v7_ve seed 1 ✅ SUBMITTED
- **Config:** v4_h100 + VALUE_EMBED_LAYERS=2 + VALUE_EMBED_DIM=128 (VE at layers 10,11)
- **step_avg:** 137ms final (154ms last step during QAT). ~4380 total steps.
- **Result:** pre-quant 1.1754, post-quant 1.1643, ttt_bpb **1.15875**
- **Artifact:** 16,290,425 bytes (16.29MB) — within 16,777,216 limit ✓
- **Val checkpoints:** step 1000: 1.2963 (+0.007 vs v4_h100), step 2000: 1.2344 (+0.014 vs v4_h100)
- **Key observation:** VE gave +0.014 BPB quality per step at step 2000. Despite 640 fewer steps, net gain of +0.0006 BPB over seed 3. Post-quant very clean (1.1643 vs seed 3's disastrous 1.2002).
- **Submission dir:** `records/track_10min_16mb/2026-03-28_12L_INT4_bQAT_VE/`
- **TODO:** Commit + push + open new PR

---

## Throughput Analysis — Why Parallel Muon Needs Banking

| Run | Approach | step_avg | Steps/10min | Notes |
|---|---|---|---|---|
| v4_h100 (DDP) | DDP all_reduce | ~110ms | ~5450 | 1 batched NCCL op |
| v6_parallel (per-param RS) | 40+ RS+AG ops | 148ms | ~4050 | NCCL launch overhead |
| v6_parallel (virtual banking) | 4-6 RS+AG ops | 117ms | ~5130 | Grad copy overhead |
| SOTA (model banking) | 4 RS+AG ops | 83ms | ~7230 | Grads already banked — no copy |

**Root cause of gap:** SOTA stores weights as 3D tensors `(num_layers, M, K)`. Grad accumulates directly in this shape → RS operates on it with zero copy. Without model banking, any Parallel Muon implementation has extra grad copy overhead that negates NCCL savings.

**To implement model banking:** Replace per-layer `nn.Linear` with 4 banked params (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`). Forward uses `bank[layer_idx]`. Estimated: 1-2 days, gives ~85-90ms → +1400 steps → ~0.020 BPB.

---

## Architecture — What We Have vs SOTA

| Feature | Our v4_h100 | SOTA (PR#549) | Notes |
|---|---|---|---|
| Layers | 12 | 11 | We use 12L INT4 bQAT to fit more capacity |
| Param banking | ❌ | ✅ | Key throughput difference |
| INT4 MLP QAT | ✅ | ❌ | Our novel contribution |
| INT4 Bigram QAT | ✅ (novel) | ❌ | Saves 370KB vs INT6 |
| EMA + QAT reset | ✅ | ❌ | Our fix for EMA-QAT contamination |
| XSA last N | ✅ (4 layers) | ✅ (4 layers) | Same |
| Value Embeddings | 🔲 Testing | ✅ | Testing in v7_ve |
| Parallel Muon | ❌ (needs banking) | ✅ | 83ms vs our 110ms |
| Legal TTT | ✅ | ✅ | Same |
| LeakyReLU² | ✅ | ✅ | Same |
| U-Net skips | ✅ | ? | Our addition |
| resid_mix | ✅ | ? | Our addition |

---

## Next Experiments Queue

| Priority | Experiment | Config | Expected gain | Effort | Status |
|---|---|---|---|---|---|
| 1 | **v10_banked proxy** | `bash run.sh v10_banked_proxy` | Verify correctness, ~110ms single-GPU | Done | **RUN NOW** |
| 2 | **v10_banked SEED=1** | `SEED=1 bash run.sh v10_banked` | ~90-100ms/step → 6000+ steps → ~1.132-1.139 BPB | Done | **NEXT** |
| 3 | v10_banked SEED=2,3 | additional seeds | Seed variance ~0.005 | 0 | Queued |
| 4 | More v7_ve seeds | SEED=3,4 | Seed variance ~0.005 | 0 | Fallback |
| 5 | KL distillation from SOTA | new config | +0.010-0.020 BPB | 4-6h | Future |

### v8_static result: FAILED (no improvement)
step_avg at step 700: **136ms** — identical to v7_ve (137ms). DDP static_graph did NOT fix VE overhead.
VE overhead is structural (not bucket ordering). Root cause unknown. Discarded.

---

## Leaderboard (Mar 27)
| Rank | BPB | Author | PR | Notes |
|---|---|---|---|---|
| 1 | **1.1194** | abaybektursun | #549 | Bank-arch + ParallelMuon |
| 2 | 1.1228 | signalrush | #374 | |
| 3 | 1.1248 | jfprincz | #287 | |
| 4 | 1.1271 | jfprincz | #198 | |
| 5 | 1.1307 | unnir | | |
| **us** | **1.1594** | SoHarshh | open PR | 12L INT4 bQAT + EMAfix |

Gap to SOTA: 0.040 BPB. Path: quality improvements (VE, distillation) + model banking.

---

## Key Technical Findings

1. **INT4 Bigram QAT** (novel): Quantize bigram hash table to INT4 during QAT. Saves ~370KB vs INT6. Enables 12L in 16MB. No prior submission has done this.

2. **EMA reset at QAT activation**: Reset EMA state when QAT turns on, so EMA only averages QAT-adapted weights. Drops post-quant degradation from +0.193 BPB to +0.002 BPB.

3. **LATE_QAT_FRAC=0.65**: Fire QAT at 65% of wallclock budget (390s). Immune to step_ms noise. Seeds are deterministic.

4. **Parallel Muon dead end (without banking)**: Tried 3 implementations (per-param RS, virtual banking RS, row-norm). All slower than DDP without model banking. Model banking = full rewrite of weight storage from 2D per-layer to 3D banked tensors.

5. **Value Embeddings**: Adding VE at 2 layers gives +0.007-0.014 BPB quality improvement per step but costs 27ms/step overhead (137ms vs 110ms). Net effect TBD from v7_ve final result.

---

## Pending PRs to Open

| Content | Type | Status |
|---|---|---|
| v4_h100 seed 3 (1.1594) | Record | ✅ Updated existing PR |
| Rownorm analysis + results | Non-record | TODO |
| Parallel Muon analysis (virtual banking) | Non-record | TODO |
| v7_ve results | Record or Non-record | Pending run completion |

---

## Key File Locations
| File | Purpose |
|---|---|
| `train_gpt_v2.py` | Working file — all experiments |
| `run.sh` | All run configs |
| `final_model_v4_h100_seed3.int8.ptz` | Best artifact (15.97MB) |
| `records/track_10min_16mb/2026-03-28_12L_INT4_bQAT_EMAfix/` | Current submission |
| `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py` | SOTA reference (1.1194) |

---

## Rules
- No git push — user does this manually
- No GPU switching — user decides when to start/stop instances
- No AI traces in any submission files (README, train_gpt.py, submission.json, blurb)
- Keep submission.json val_bpb honest — use measured values
- Every H100 run gets a commit + PR (record or non-record)
