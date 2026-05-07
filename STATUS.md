# Parameter Golf — Status Report
**2026-03-20 4:15pm EDT**

## 🔥 Critical Findings

### 1. Paid Prefix = Risky Exploit
- **PR #262 (1.0539 bpb)** stores 10% of validation tokens as LZMA blob in artifact
- Discord intel (Larry): "It's not rly TTT, just working around 16mb bottleneck"
- **PR #168 (original)** still OPEN/UNMERGED — might get disqualified
- **Recommendation:** Avoid this technique entirely

### 2. PR #198 is THE SOTA Base
- Everyone (alertcat, machdragon, mattqlf, 0xjaishy) forks from PR #198
- **1.1318 bpb** (11L, Int6+zstd, SmearGate, BigramHash, SWA, WD=0.04, FA3)
- Commit messages confirm: "Rebuild from the proven #1 submission (PR #198)"
- **We're still on vanilla baseline (1.6353) — 0.5 bpb behind**

### 3. Mixed Int5/Int6 Proven
- **PR #264 (1.1455 bpb):** Int5 for MLP, Int6 for attention → saves 1.9MB → funds 11th layer
- **alertcat fork:** Int5 saves ~1.8MB via better zstd compression (1.88x vs 1.51x)
- Full-model SGD TTT (2 epochs) beats LoRA TTT by ~0.005 bpb

## ✅ PC1 Results (4090, ~340 steps)

| Experiment | int8 bpb | sliding bpb | vs baseline | Status |
|---|---|---|---|---|
| **sliding_window_eval** | 1.6144 | **1.5842** | **-0.051** ✅ | done (BEST) |
| muon_weight_decay_002 | 1.7758 | — | +0.140 ❌ | done (WORSE) |
| stack_eval_tricks | — | — | — | queued |
| stack_full_v1 | — | — | — | queued |
| stack_full_v2 | — | — | — | queued |

**Currently running:** muon_weight_decay_002 (step 10/340, ~30 min remaining)

## 📊 Competitive Landscape

| # | bpb | Technique | PR | Status |
|---|-----|-----------|-----|--------|
| 1 | 1.0539 | Paid prefix exploit ⚠️ | #262 | OPEN (risky) |
| 2 | 1.1318 | PR #198 base (REAL SOTA) | #198 | OPEN |
| 3 | 1.1455 | Int5/Int6 mixed + full-model TTT | #264 | OPEN |
| **Our PR** | **1.6031** | QK Gain 1.2 + sliding eval | **#259** | OPEN |

**Gap to SOTA:** 0.47 bpb (we're 5+ months behind in techniques)

## 🎯 Strategic Recommendations

### Immediate (Today)
1. ❌ **STOP testing Muon WD / 10+ layers on 4090** — needs 5k+ steps, doesn't transfer
2. ✅ **Finish current batch** (stack_eval_tricks might combine our wins)
3. ✅ **Download PR #198 train_gpt.py** to PC1 as new base

### Short-term (This Week)
1. **Test PR #198 on 4090** — see if 11L + Int6 even fits in 10 min
2. **Add our proven tricks to PR #198:**
   - QK gain init 1.2
   - FP16 embed export
   - Spectral embed init
   - Sliding window eval (already in #198)
3. **Research Int5/Int6 mixed** — we have Int6 QAT code, add Int5 for MLP

### Medium-term (When H100 Access)
1. **Run PR #198 baseline** — validate 1.1318 bpb
2. **Stack improvements:**
   - Int5 MLP / Int6 attention
   - Full-model SGD TTT (2 epochs, lr=0.002)
   - Sigmoid attention gate (mattqlf, 3 lines)
   - RoPE base 50K
3. **Update PR #259** with results

## 🧠 Key Learnings

### What Works on 4090
- ✅ Sliding window eval (-0.051 bpb) — BEST single trick
- ✅ QK gain init 1.2 (-0.022 bpb)
- ✅ FP16 embed export (-0.017 bpb)
- ✅ Spectral embed init (-0.013 bpb)

### What Fails on 4090
- ❌ Muon WD (+0.140 bpb) — needs 5k+ steps
- ❌ 10+ layers — needs 10k+ steps
- ❌ Seq len 2048/4096 — OOM

### What Transfers to H100
- ✅ Init tricks (QK gain, spectral, ortho)
- ✅ Eval tricks (sliding window)
- ✅ Quantization strategies (Int5/Int6 mixed)
- ❌ Hyperparams (LR, warmdown timing)
- ❌ Layer count (4090 can't validate >9L convergence)

## 🚨 Risks

1. **Paid prefix might get banned** — don't invest time/compute
2. **Our PR #259 is stale** — based on vanilla baseline, not PR #198
3. **4090 experiments have limited value** — can't test 11L, 2048 seq, or long training

## 📁 Research Files Updated
- `research/our_results.md` — full results log
- `research/session_2026-03-20_1610.md` — this session
- `research/discord_intel_2026-03-20.md` — TTT exploit discussion
- `research/leaderboard_analysis.md` — (needs update with PR #198 breakdown)

## ⏭️ Next Actions
1. Wait for current batch to finish (~90 min)
2. Download PR #198 to PC1
3. Design new experiments on PR #198 base
4. Notify Eric with findings

---
**Timeline:** 41 days until April 30 deadline
**Current position:** Way behind SOTA, need to pivot to PR #198 base ASAP
